"""Default vision-planner harness (built-in baseline).

This is the fallback harness used when no external --harness is provided.
It mirrors the baseline at meta-harness-remedy/candidates/h000_baseline/harness.py.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_from_response(
    raw_response: str | dict,
    fallback: dict | None = None,
) -> dict:
    """Extract JSON from a raw LLM response (may be wrapped in markdown fences).

    Args:
        raw_response: Raw string or dict from the model.
        fallback: Dict to return on parse failure.
            If None, returns ``{"parse_error": "..."}``.

    Returns:
        Parsed JSON dict, or *fallback* with ``parse_error`` key appended.
    """
    if isinstance(raw_response, dict):
        return raw_response

    text = raw_response.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        result = dict(fallback) if fallback else {}
        result["parse_error"] = f"Failed to parse: {text[:200]}"
        return result


class VisionPlannerHarness:
    """Baseline vision-planner harness for PDF accessibility remediation."""

    ACTION_DEFINITIONS: dict[str, str] = {
        "set_tag": (
            "Assign a structure tag "
            "(P, H1-H6, L, LI, Table, TR, TH, TD, Figure, Caption, Link)"
        ),
        "set_alt_text": "Add alternative text to a figure",
        "reconstruct_table": (
            "Rebuild table structure with correct headers, rows, "
            "columns, spans"
        ),
        "fix_reading_order": "Reorder structure elements",
        "mark_manual_review": "Flag for human review (when uncertain)",
    }

    allowed_actions: tuple[str, ...] | None = None  # None = all actions
    domain_instructions: str = ""

    # -- Grounder (vision) ------------------------------------------------

    def build_grounder_prompt(self, page_image_b64: str, page_dims: dict) -> list[dict]:
        """Construct the messages array for the Grounder (vision model) call."""
        return [
            {
                "role": "user",
                "content": (
                    "Analyze this PDF page image. Identify all semantic regions. "
                    "For each region, provide:\n"
                    "1. bbox: [ymin, xmin, ymax, xmax] (normalized 0-1000 relative to page dimensions)\n"
                    "2. type: one of (heading_l1, heading_l2, heading_l3, paragraph, table, figure, "
                    "list, form_field, link, decorative, page_header, page_footer, page_number)\n"
                    "3. logic: Brief explanation of why it is meaningful or decorative\n"
                    "4. reading_order: Logical sequence number (1, 2, 3...)\n"
                    "5. If type is 'table': provide rows, cols, header_row_indices, header_col_indices\n"
                    "\n"
                    f"Page dimensions: {page_dims['width_pts']}x{page_dims['height_pts']} points\n"
                    "\n"
                    "Return ONLY a JSON object with key 'regions' containing an array."
                ),
                "images": [page_image_b64],
            }
        ]

    def build_grounder_tools(self) -> list[dict] | None:
        """No tool calling for baseline -- use free-form JSON output."""
        return None

    # -- Planner prompt composition ----------------------------------------

    def _format_system_preamble(self) -> str:
        return (
            "You are a PDF/UA accessibility remediation planner. Your job is to "
            "diagnose accessibility failures and plan specific fixes using ONLY "
            "the allowed operations listed below."
        )

    def _format_action_definitions_section(self) -> str:
        actions = self.allowed_actions or tuple(self.ACTION_DEFINITIONS.keys())
        lines = []
        for name in actions:
            if name in self.ACTION_DEFINITIONS:
                lines.append(f"- {name}: {self.ACTION_DEFINITIONS[name]}")
        return "ALLOWED OPERATIONS:\n" + "\n".join(lines)

    def _format_domain_instructions_section(self) -> str:
        if not self.domain_instructions:
            return ""
        return f"DOMAIN-SPECIFIC RULES:\n{self.domain_instructions}"

    def _format_violations_section(self, violations: list[dict]) -> str:
        filtered = self.filter_violations(violations)
        return f"INPUT A -- veraPDF Violations:\n{json.dumps(filtered, indent=2)}"

    def _format_semantic_map_section(self, semantic_map: dict) -> str:
        return f"INPUT B -- Semantic Map (from visual analysis):\n{json.dumps(semantic_map, indent=2)}"

    def _format_anchor_graph_section(self, anchor_graph: dict) -> str:
        formatted_graph = self.format_anchor_graph(anchor_graph)
        return f"INPUT C -- Anchor Graph (maps regions to PDF objects):\n{json.dumps(formatted_graph, indent=2)}"

    def _format_task_section(self) -> str:
        return (
            "TASK:\n"
            "1. Cross-reference each violation with the Semantic Map to understand "
            "what the violated content IS visually\n"
            "2. For structure violations (7.1-x): determine if content is decorative "
            "or meaningful based on visual analysis\n"
            "3. For table violations (7.2-x, 7.5-x): use the table_structure from "
            "visual analysis to plan correct structure\n"
            "4. For untagged content (7.1-3): assign correct tags based on visual type\n"
            "5. For link violations (7.18.5-x): use set_tag with 'Link' to fix link tagging\n"
            "6. Skip font violations -- those are handled by Ghostscript preprocessing\n"
            "\n"
            "Output a JSON object with keys:\n"
            "- 'confidence': float 0-1\n"
            "- 'operations': array of operation objects with keys: "
            "op_id, page, action, target_region, target_anchors, reason\n"
            "- 'manual_review': array of objects with keys: page, region, reason\n"
            "\n"
            "Only emit operations you are confident about. Use mark_manual_review "
            "for anything uncertain."
        )

    # -- Planner (thinking) -----------------------------------------------

    def build_planner_prompt(
        self,
        semantic_map: dict,
        violations: list[dict],
        anchor_graph: dict,
    ) -> list[dict]:
        """Construct the messages array for the Planner (thinking model) call.

        Composed from overridable section methods. Subclasses should override
        individual ``_format_*`` hooks rather than this method.
        """
        sections = [
            self._format_system_preamble(),
            "",
            self._format_action_definitions_section(),
        ]

        domain_instr = self._format_domain_instructions_section()
        if domain_instr:
            sections.append("")
            sections.append(domain_instr)

        sections.extend([
            "",
            self._format_violations_section(violations),
            "",
            self._format_semantic_map_section(semantic_map),
            "",
            self._format_anchor_graph_section(anchor_graph),
            "",
            self._format_task_section(),
        ])

        return [{"role": "user", "content": "\n".join(sections)}]

    def build_planner_tools(self) -> list[dict] | None:
        """No tool calling for baseline -- use free-form JSON output."""
        return None

    def planner_think(self) -> bool:
        """Enable extended thinking for the Planner call."""
        return True

    # -- Violation filtering ----------------------------------------------

    def filter_violations(self, violations: list[dict]) -> list[dict]:
        """Exclude font violations -- those are handled by Ghostscript.

        If ALL violations are font-related (rule_id starts with "6.") or
        character-encoding issues (rule_id starts with "6." after filtering),
        return an empty list so the planner produces an empty/low-confidence
        plan instead of wasting time on structural changes that can regress.
        """
        filtered = [
            v for v in violations
            if not v.get("rule_id", "").startswith("6.")
        ]
        return filtered

    # -- Anchor graph formatting ------------------------------------------

    def format_anchor_graph(self, anchor_graph: dict) -> dict:
        """Pass through the full anchor graph without compression."""
        return anchor_graph

    # -- Output parsing ---------------------------------------------------

    def parse_grounder_output(self, raw_response: str | dict) -> dict:
        """Parse Grounder response into semantic map."""
        return extract_json_from_response(
            raw_response,
            fallback={"regions": []},
        )

    def parse_planner_output(self, raw_response: str | dict) -> dict:
        """Parse Planner response into remediation plan."""
        return extract_json_from_response(
            raw_response,
            fallback={"confidence": 0.0, "operations": [], "manual_review": []},
        )

    # -- Confidence threshold ---------------------------------------------

    def confidence_threshold(self) -> float:
        """Minimum confidence to auto-execute vs. flag for manual review."""
        return 0.7
