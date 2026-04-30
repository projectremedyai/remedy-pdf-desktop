"""Domain-specific planner harnesses for the specialist coordinator."""

from __future__ import annotations

import json
from typing import Any

from project_remedy.pdf_specialists.contracts import DomainName
from project_remedy.vision_planner.harness import VisionPlannerHarness


class SpecialistPlannerHarness(VisionPlannerHarness):
    """Baseline specialist harness with a constrained action set.

    Reuses shared prompt sections (violations, semantic map, anchor graph)
    from ``VisionPlannerHarness`` while overriding only the specialist-
    specific sections (preamble, domain instructions, task description).
    """

    specialist_name: str = "base"
    domain_name: DomainName = "structure"
    owned_domains: tuple[DomainName, ...] = ("structure",)
    allowed_actions: tuple[str, ...] = ("mark_manual_review",)
    domain_instructions: str = ""

    # -- Section overrides (compose, don't replace) -----------------------

    def _format_system_preamble(self) -> str:
        return (
            f"You are the {self.specialist_name} specialist for PDF/UA remediation.\n"
            f"Focus ONLY on the '{self.domain_name}' domain. Ignore unrelated failures."
        )

    def _format_domain_instructions_section(self) -> str:
        rules = []
        if self.domain_instructions:
            rules.append(self.domain_instructions)
        rules.extend([
            "- Never emit operations outside the allowed list.",
            "- If a target is ambiguous, use mark_manual_review.",
            "- Prefer specific anchor-based actions over document-wide guesses.",
        ])
        return "RULES:\n" + "\n".join(rules)

    def _format_violations_section(self, violations: list[dict]) -> str:
        filtered = self.filter_violations(violations)
        return f"INPUT A -- Routed Violations:\n{json.dumps(filtered, indent=2)}"

    def _format_task_section(self) -> str:
        return (
            "Return a JSON object with keys:\n"
            "- confidence\n"
            "- operations\n"
            "- manual_review"
        )

    # -- Violation routing ------------------------------------------------

    def filter_violations(self, violations: list[dict]) -> list[dict]:
        base = super().filter_violations(violations)
        return [v for v in base if v.get("domain") in self.owned_domains]


class StructurePlannerHarness(SpecialistPlannerHarness):
    specialist_name = "structure_planner"
    domain_name = "structure"
    owned_domains = ("structure",)
    allowed_actions = ("artifactize", "set_tag", "mark_manual_review")
    domain_instructions = (
        "- Own non-table, non-figure semantic structure only.\n"
        "- Do not retag anchors that belong to tables or figures.\n"
        "- Use artifactize for decorative headers, footers, page numbers, and dividers."
    )


class TablePlannerHarness(SpecialistPlannerHarness):
    specialist_name = "table_planner"
    domain_name = "table"
    owned_domains = ("table",)
    allowed_actions = ("reconstruct_table", "set_tag", "mark_manual_review")
    domain_instructions = (
        "- Own table structure only.\n"
        "- Use reconstruct_table when headers, row groups, or cell roles are unclear.\n"
        "- Only use set_tag for table-related roles like Table, TR, TH, and TD."
    )


class ReadingOrderPlannerHarness(SpecialistPlannerHarness):
    specialist_name = "reading_order_planner"
    domain_name = "reading_order"
    owned_domains = ("reading_order",)
    allowed_actions = ("fix_reading_order", "mark_manual_review")
    domain_instructions = (
        "- Own ordering only.\n"
        "- Do not retag content or reconstruct tables.\n"
        "- Use fix_reading_order only when the target anchors clearly form a sequence."
    )


class FigureSemanticsPlannerHarness(SpecialistPlannerHarness):
    specialist_name = "figure_semantics_planner"
    domain_name = "figure_semantics"
    owned_domains = ("figure_semantics",)
    allowed_actions = ("set_alt_text", "artifactize", "set_tag", "mark_manual_review")
    domain_instructions = (
        "- Own figures, captions, and decorative-image decisions only.\n"
        "- Use artifactize only for truly decorative figures.\n"
        "- Use set_tag only for figure/caption semantics."
    )


PLANNER_HARNESSES: dict[str, type[SpecialistPlannerHarness]] = {
    "structure": StructurePlannerHarness,
    "table": TablePlannerHarness,
    "reading_order": ReadingOrderPlannerHarness,
    "figure_semantics": FigureSemanticsPlannerHarness,
}


def create_harness_for_domain(domain: str) -> SpecialistPlannerHarness:
    harness_cls = PLANNER_HARNESSES[domain]
    return harness_cls()
