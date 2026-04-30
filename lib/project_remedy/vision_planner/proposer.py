"""Proposer: generate prompt variants when success rate drops below threshold.

Analyzes failure patterns from the experiment store and produces targeted
modifications to the VisionPlannerHarness configuration. Operates on the
harness.py interface (prompt templates, context assembly, output parsing).
"""

from __future__ import annotations

import copy
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from project_remedy.vision_planner.experiment_store import (
    ExperimentStore,
    HarnessVariant,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proposal strategies
# ---------------------------------------------------------------------------


@dataclass
class ProposalStrategy:
    """A specific modification to try on a harness variant."""

    name: str
    description: str
    target: str  # "grounder_prompt", "planner_prompt", "violation_filter", etc.
    modifications: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------


def analyze_failures(
    store: ExperimentStore, harness_id: str
) -> dict[str, Any]:
    """Analyze why a harness is failing and recommend strategies.

    Returns dict with:
    - success_rate: float
    - failure_count: int
    - top_failing_doc_types: list of (type, count)
    - top_failing_violation_types: list of (type, count)
    - recommended_strategies: list of ProposalStrategy
    """
    patterns = store.get_failure_patterns(harness_id)
    experiments = store.get_experiments_for_harness(harness_id)
    success_rate = store.compute_success_rate(harness_id)

    total = len(experiments)
    failures = total - sum(1 for e in experiments if e.passed)

    # Sort failure causes by frequency
    doc_types = sorted(
        patterns["failing_doc_types"].items(), key=lambda x: x[1], reverse=True
    )
    violation_types = sorted(
        patterns["failing_violation_types"].items(), key=lambda x: x[1], reverse=True
    )

    strategies = _recommend_strategies(
        doc_types=doc_types,
        violation_types=violation_types,
        destructive_docs=patterns["destructive_docs"],
        common_errors=patterns["common_errors"],
        success_rate=success_rate,
        experiments=experiments,
    )

    return {
        "success_rate": success_rate,
        "failure_count": failures,
        "total_count": total,
        "top_failing_doc_types": doc_types[:5],
        "top_failing_violation_types": violation_types[:10],
        "destructive_count": len(patterns["destructive_docs"]),
        "recommended_strategies": strategies,
    }


def _recommend_strategies(
    doc_types: list[tuple[str, int]],
    violation_types: list[tuple[str, int]],
    destructive_docs: list[str],
    common_errors: dict[str, int],
    success_rate: float,
    experiments: list,
) -> list[ProposalStrategy]:
    """Generate recommended modification strategies based on failure analysis."""
    strategies: list[ProposalStrategy] = []

    # Strategy 1: Table structure improvements (if table violations dominate)
    table_violations = [
        (vt, c) for vt, c in violation_types
        if vt.startswith("7.2") or vt.startswith("7.5")
    ]
    if table_violations:
        table_count = sum(c for _, c in table_violations)
        total_violations = sum(c for _, c in violation_types)
        if total_violations > 0 and table_count / total_violations > 0.2:
            strategies.append(ProposalStrategy(
                name="table_structure_focus",
                description=(
                    f"Table violations account for {table_count}/{total_violations} "
                    f"({table_count/total_violations:.0%}) of failures. "
                    "Add explicit table reconstruction guidance to planner prompt."
                ),
                target="planner_prompt",
                modifications={
                    "add_table_examples": True,
                    "emphasize_table_structure": True,
                    "table_violation_count": table_count,
                },
            ))

    # Strategy 2: Untagged content handling
    untagged = [
        (vt, c) for vt, c in violation_types if vt.startswith("7.1")
    ]
    if untagged:
        untagged_count = sum(c for _, c in untagged)
        strategies.append(ProposalStrategy(
            name="untagged_content_tagging",
            description=(
                f"{untagged_count} untagged content violations. "
                "Improve grounder region detection and planner tag assignment."
            ),
            target="grounder_prompt",
            modifications={
                "emphasize_comprehensive_detection": True,
                "untagged_violation_count": untagged_count,
            },
        ))

    # Strategy 3: Reduce destructive edits
    if destructive_docs:
        strategies.append(ProposalStrategy(
            name="reduce_destructive_edits",
            description=(
                f"{len(destructive_docs)} documents had more violations after "
                "remediation than before. Raise confidence threshold and add "
                "conservative operation guards."
            ),
            target="confidence_threshold",
            modifications={
                "raise_threshold": True,
                "destructive_doc_count": len(destructive_docs),
                "add_safety_guards": True,
            },
        ))

    # Strategy 4: Parse error recovery
    parse_errors = {
        k: v for k, v in common_errors.items()
        if "parse" in k.lower() or "json" in k.lower()
    }
    if parse_errors:
        strategies.append(ProposalStrategy(
            name="output_parsing_robustness",
            description=(
                "Model output parsing failures detected. "
                "Add fallback parsing or switch to tool-calling mode."
            ),
            target="output_parsing",
            modifications={
                "use_tool_calling": True,
                "add_fallback_parsers": True,
                "parse_error_count": sum(parse_errors.values()),
            },
        ))

    # Strategy 5: Anchor graph compression (if documents are complex)
    complex_docs = [
        (dt, c) for dt, c in doc_types
        if "complex" in dt.lower() or "mixed" in dt.lower()
    ]
    if complex_docs:
        strategies.append(ProposalStrategy(
            name="anchor_graph_compression",
            description=(
                "Complex/mixed documents are failing. "
                "Compress anchor graph to reduce noise and focus on relevant regions."
            ),
            target="anchor_graph_format",
            modifications={
                "compress_graph": True,
                "filter_by_page": True,
                "limit_text_excerpts": True,
            },
        ))

    # Strategy 6: Violation grouping (always worth trying if success < 50%)
    if success_rate < 0.5:
        strategies.append(ProposalStrategy(
            name="violation_grouping",
            description=(
                f"Success rate is {success_rate:.0%}. "
                "Group violations by page and type so planner sees related "
                "violations together."
            ),
            target="planner_prompt",
            modifications={
                "group_by_page": True,
                "group_by_type": True,
                "prioritize_impactful": True,
            },
        ))

    # Strategy 7: Few-shot examples (only if there are actual failures and
    # no other strong strategy -- do not fire if success_rate is already high)
    if success_rate < 0.3 and (not strategies or len(strategies) < 2):
        strategies.append(ProposalStrategy(
            name="few_shot_examples",
            description=(
                "Add few-shot examples of successful remediation plans "
                "from prior experiments to guide the planner."
            ),
            target="planner_prompt",
            modifications={
                "add_examples": True,
                "example_source": "successful_experiments",
            },
        ))

    return strategies


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------


class HarnessProposer:
    """Generate harness variant proposals based on experiment data.

    Analyzes failure patterns and applies targeted modifications to
    create new harness configurations for evaluation.
    """

    def __init__(
        self,
        store: ExperimentStore,
        success_threshold: float = 0.5,
        max_proposals_per_iteration: int = 3,
    ):
        self._store = store
        self._success_threshold = success_threshold
        self._max_proposals = max_proposals_per_iteration

    @property
    def success_threshold(self) -> float:
        return self._success_threshold

    def should_propose(self, harness_id: str) -> bool:
        """Check if the current harness needs improvement.

        Returns True if success rate is below threshold.
        """
        rate = self._store.compute_success_rate(harness_id)
        return rate < self._success_threshold

    def propose_variants(
        self,
        base_harness_id: str,
        base_config: dict,
    ) -> list[dict]:
        """Generate new harness variant proposals.

        Args:
            base_harness_id: The harness to improve on.
            base_config: The current harness configuration dict.

        Returns:
            List of proposal dicts, each with:
            - harness_id: new unique ID
            - parent_id: base_harness_id
            - description: what changed
            - config: modified harness config dict
            - strategy: the ProposalStrategy that generated it
        """
        analysis = analyze_failures(self._store, base_harness_id)
        strategies = analysis["recommended_strategies"]

        if not strategies:
            logger.info(
                "No improvement strategies found for %s (rate=%.1f%%)",
                base_harness_id,
                analysis["success_rate"] * 100,
            )
            return []

        proposals = []
        for strategy in strategies[: self._max_proposals]:
            new_config = self._apply_strategy(base_config, strategy)
            new_id = _generate_harness_id(strategy.name)

            proposals.append({
                "harness_id": new_id,
                "parent_id": base_harness_id,
                "description": strategy.description,
                "config": new_config,
                "strategy": strategy,
            })

            logger.info(
                "Proposed variant %s from %s: %s",
                new_id, base_harness_id, strategy.name,
            )

        return proposals

    def _apply_strategy(
        self, base_config: dict, strategy: ProposalStrategy
    ) -> dict:
        """Apply a strategy's modifications to a base config.

        Returns a new config dict with the strategy's changes applied.
        """
        config = copy.deepcopy(base_config)

        if strategy.target == "planner_prompt":
            config = self._modify_planner_prompt(config, strategy)
        elif strategy.target == "grounder_prompt":
            config = self._modify_grounder_prompt(config, strategy)
        elif strategy.target == "confidence_threshold":
            config = self._modify_confidence(config, strategy)
        elif strategy.target == "output_parsing":
            config = self._modify_parsing(config, strategy)
        elif strategy.target == "anchor_graph_format":
            config = self._modify_anchor_format(config, strategy)

        # Always record what strategy was applied
        config.setdefault("_meta", {})
        config["_meta"]["strategy"] = strategy.name
        config["_meta"]["strategy_description"] = strategy.description

        return config

    def _modify_planner_prompt(self, config: dict, strategy: ProposalStrategy) -> dict:
        """Apply planner prompt modifications."""
        mods = strategy.modifications

        if mods.get("add_table_examples"):
            config.setdefault("planner_additions", [])
            config["planner_additions"].append(
                "\nTABLE RECONSTRUCTION GUIDANCE:\n"
                "When you see table structure violations (7.2-x, 7.5-x), "
                "use reconstruct_table with explicit rows, cols, header_rows, "
                "and cells specification. Always identify header rows first.\n"
            )

        if mods.get("group_by_page"):
            config["violation_grouping"] = "page_then_type"

        if mods.get("group_by_type"):
            config.setdefault("violation_grouping", "type_then_page")

        if mods.get("prioritize_impactful"):
            config["violation_priority_order"] = [
                "7.1",  # Structure (most common failures)
                "7.2",  # Table
                "7.5",  # Table headers
                "1.",   # Alt text
                "7.3",  # Reading order
            ]

        if mods.get("add_examples"):
            config["include_few_shot"] = True
            config["few_shot_source"] = mods.get("example_source", "curated")

        return config

    def _modify_grounder_prompt(self, config: dict, strategy: ProposalStrategy) -> dict:
        """Apply grounder prompt modifications."""
        mods = strategy.modifications

        if mods.get("emphasize_comprehensive_detection"):
            config.setdefault("grounder_additions", [])
            config["grounder_additions"].append(
                "\nIMPORTANT: Identify ALL content on the page. "
                "Do not miss headers, footers, page numbers, or decorative elements. "
                "Untagged content is the most common violation type.\n"
            )

        return config

    def _modify_confidence(self, config: dict, strategy: ProposalStrategy) -> dict:
        """Adjust confidence threshold and safety guards."""
        mods = strategy.modifications

        if mods.get("raise_threshold"):
            current = config.get("confidence_threshold", 0.7)
            config["confidence_threshold"] = min(current + 0.1, 0.95)

        if mods.get("add_safety_guards"):
            config["pre_check_violations"] = True
            config["abort_on_increase"] = True

        return config

    def _modify_parsing(self, config: dict, strategy: ProposalStrategy) -> dict:
        """Improve output parsing robustness."""
        mods = strategy.modifications

        if mods.get("use_tool_calling"):
            config["use_tool_calling"] = True

        if mods.get("add_fallback_parsers"):
            config["fallback_parsers"] = ["json_extract", "regex_extract", "partial_json"]

        return config

    def _modify_anchor_format(self, config: dict, strategy: ProposalStrategy) -> dict:
        """Modify anchor graph formatting."""
        mods = strategy.modifications

        if mods.get("compress_graph"):
            config["anchor_graph_max_anchors_per_page"] = 50

        if mods.get("filter_by_page"):
            config["anchor_graph_filter_by_violation_pages"] = True

        if mods.get("limit_text_excerpts"):
            config["anchor_graph_text_excerpt_max_chars"] = 40

        return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_harness_id(strategy_name: str) -> str:
    """Generate a unique harness ID based on the strategy name."""
    short_uuid = uuid.uuid4().hex[:8]
    slug = re.sub(r"[^a-z0-9]+", "_", strategy_name.lower()).strip("_")
    return f"auto_{slug}_{short_uuid}"


def get_successful_examples(
    store: ExperimentStore,
    harness_id: str,
    limit: int = 3,
) -> list[dict]:
    """Get successful experiment examples for few-shot prompting.

    Returns simplified dicts with violation_types and fix_sequence
    from experiments that passed.
    """
    experiments = store.get_experiments_for_harness(harness_id)
    passed = [e for e in experiments if e.passed]

    # Prefer diverse examples (different document types)
    seen_types: set[str] = set()
    examples: list[dict] = []

    for exp in passed:
        if exp.document_type not in seen_types and len(examples) < limit:
            examples.append({
                "document_type": exp.document_type,
                "violation_types": exp.violation_types,
                "fix_sequence": exp.fix_sequence,
                "violations_before": exp.violations_before,
                "violations_after": exp.violations_after,
            })
            seen_types.add(exp.document_type)

    # Fill remaining slots with any passed examples
    for exp in passed:
        if len(examples) >= limit:
            break
        already = any(
            e["document_type"] == exp.document_type
            and e["violation_types"] == exp.violation_types
            for e in examples
        )
        if not already:
            examples.append({
                "document_type": exp.document_type,
                "violation_types": exp.violation_types,
                "fix_sequence": exp.fix_sequence,
                "violations_before": exp.violations_before,
                "violations_after": exp.violations_after,
            })

    return examples
