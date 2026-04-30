"""Evolution loop: promote winning variants, retire losers.

Orchestrates the propose -> evaluate -> score -> promote/retire cycle.
Designed to be called from the vision-planner pipeline after each batch
of documents, or as a standalone overnight runner.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from project_remedy.vision_planner.experiment_store import (
    ExperimentRecord,
    ExperimentStore,
    HarnessVariant,
)
from project_remedy.vision_planner.proposer import HarnessProposer, analyze_failures
from project_remedy.vision_planner.scorer import HarnessScorer, ScoringResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    """Configuration for the evolution loop."""

    # When to trigger evolution
    success_threshold: float = 0.5      # propose when rate drops below this
    min_docs_before_evolve: int = 10    # need at least N experiments before proposing

    # Promotion / retirement
    promote_threshold: float = 0.05     # promote if improvement > 5pp over current best
    retire_after_iterations: int = 5    # retire if not on frontier after N iterations
    max_active_variants: int = 10       # retire oldest if too many active

    # Loop budget
    max_iterations: int = 20
    max_proposals_per_iteration: int = 3
    abandon_threshold: float = 0.05     # stop if best < 5% after 10 iterations
    abandon_after_iterations: int = 10

    # Database
    db_path: str = ""                   # empty = in-memory


# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------


class EvolutionLoop:
    """Orchestrate the Meta-Harness evolution loop.

    The loop operates in two modes:
    1. **Inline mode**: Called after each pipeline run to record results
       and check if evolution is needed.
    2. **Batch mode**: Run as a standalone loop that iterates through
       propose -> evaluate -> score -> promote/retire.
    """

    def __init__(
        self,
        store: ExperimentStore,
        config: EvolutionConfig | None = None,
    ):
        self._store = store
        self._config = config or EvolutionConfig()
        self._proposer = HarnessProposer(
            store=store,
            success_threshold=self._config.success_threshold,
            max_proposals_per_iteration=self._config.max_proposals_per_iteration,
        )
        self._scorer = HarnessScorer(
            store=store,
            min_docs_for_scoring=self._config.min_docs_before_evolve,
        )
        self._iteration = 0

    @property
    def store(self) -> ExperimentStore:
        return self._store

    @property
    def iteration(self) -> int:
        return self._iteration

    # -- Inline mode (called per-document) ----------------------------------

    def record_result(
        self,
        harness_id: str,
        document_hash: str,
        document_type: str,
        trace: dict,
    ) -> ExperimentRecord:
        """Record a single experiment result from a pipeline run.

        Args:
            harness_id: Which harness variant was used.
            document_hash: Document identifier.
            document_type: Classification of document (e.g., "table_heavy").
            trace: The full trace dict from run_vision_plan().

        Returns the created ExperimentRecord.
        """
        # Extract violation types from the trace
        violation_types = _extract_violation_types(trace)

        # Extract fix sequence (operations from the plan)
        fix_sequence = trace.get("plan", {}).get("operations", [])

        # Generate experiment ID
        exp_id = f"{harness_id}_{document_hash}_{int(time.time())}"

        record = ExperimentRecord(
            experiment_id=exp_id,
            harness_id=harness_id,
            document_hash=document_hash,
            document_type=document_type,
            violation_types=violation_types,
            fix_sequence=fix_sequence,
            violations_before=trace.get("violations_before", 0),
            violations_after=trace.get("violations_after", 0),
            passed=trace.get("passed", False),
            elapsed_seconds=trace.get("elapsed_seconds", 0.0),
            confidence=trace.get("plan", {}).get("confidence", 0.0),
            error=trace.get("error"),
        )

        self._store.record_experiment(record)
        return record

    def check_evolution_needed(self, harness_id: str) -> bool:
        """Check if the current harness needs evolution.

        Returns True if:
        1. Enough experiments have been recorded (min_docs_before_evolve).
        2. Success rate is below the threshold.
        """
        experiments = self._store.get_experiments_for_harness(harness_id)
        if len(experiments) < self._config.min_docs_before_evolve:
            return False
        return self._proposer.should_propose(harness_id)

    def maybe_evolve(
        self,
        current_harness_id: str,
        current_config: dict,
    ) -> list[dict]:
        """Check if evolution is needed and propose variants if so.

        Returns list of proposal dicts (empty if no evolution needed).
        This does NOT evaluate or score the proposals -- that's done by
        the batch mode or the caller.
        """
        if not self.check_evolution_needed(current_harness_id):
            return []

        proposals = self._proposer.propose_variants(
            base_harness_id=current_harness_id,
            base_config=current_config,
        )

        # Register proposals in the store
        for proposal in proposals:
            self._store.register_variant(
                harness_id=proposal["harness_id"],
                description=proposal["description"],
                parent_id=proposal["parent_id"],
                harness_config=proposal["config"],
            )
            self._store.log_evolution(
                iteration=self._iteration,
                action="propose",
                harness_id=proposal["harness_id"],
                details=proposal["description"],
            )

        return proposals

    # -- Batch mode (standalone evolution loop) -----------------------------

    def run_evolution_step(
        self,
        current_harness_id: str,
        current_config: dict,
        evaluate_fn: Callable[[str, dict], list[ExperimentRecord]] | None = None,
    ) -> dict[str, Any]:
        """Run one full evolution step: propose -> evaluate -> score -> promote/retire.

        Args:
            current_harness_id: The current best harness.
            current_config: The current harness config dict.
            evaluate_fn: Optional callback to evaluate a proposal.
                Signature: evaluate_fn(harness_id, config) -> list[ExperimentRecord]
                If None, proposals are registered but not evaluated.

        Returns dict with step results.
        """
        self._iteration += 1
        step_result: dict[str, Any] = {
            "iteration": self._iteration,
            "proposals": [],
            "scores": [],
            "promotions": [],
            "retirements": [],
            "abandoned": False,
        }

        # 1. Check if we should abandon
        if self._should_abandon():
            step_result["abandoned"] = True
            self._store.log_evolution(
                self._iteration, "abandon", current_harness_id,
                "Best conformance below threshold after sufficient iterations",
            )
            logger.warning("Evolution abandoned: insufficient improvement")
            return step_result

        # 2. Propose new variants
        proposals = self.maybe_evolve(current_harness_id, current_config)
        step_result["proposals"] = [
            {"harness_id": p["harness_id"], "description": p["description"]}
            for p in proposals
        ]

        if not proposals:
            logger.info("No proposals generated at iteration %d", self._iteration)
            return step_result

        # 3. Evaluate proposals (if callback provided)
        if evaluate_fn is not None:
            for proposal in proposals:
                try:
                    records = evaluate_fn(
                        proposal["harness_id"], proposal["config"]
                    )
                    for record in records:
                        self._store.record_experiment(record)
                except Exception as e:
                    logger.error(
                        "Evaluation failed for %s: %s",
                        proposal["harness_id"], e,
                    )
                    self._store.log_evolution(
                        self._iteration, "evaluate_error",
                        proposal["harness_id"], str(e),
                    )

        # 4. Score all proposals
        for proposal in proposals:
            score = self._scorer.score_variant(proposal["harness_id"])
            if score:
                step_result["scores"].append({
                    "harness_id": score.harness_id,
                    "conformance_rate": score.conformance_rate,
                    "on_frontier": score.on_pareto_frontier,
                })

        # 5. Promote and retire
        promotions = self._promote_winners()
        retirements = self._retire_losers()

        step_result["promotions"] = promotions
        step_result["retirements"] = retirements

        return step_result

    def run_loop(
        self,
        initial_harness_id: str,
        initial_config: dict,
        evaluate_fn: Callable[[str, dict], list[ExperimentRecord]] | None = None,
        max_iterations: int | None = None,
    ) -> list[dict]:
        """Run the full evolution loop.

        Returns list of step results.
        """
        max_iter = max_iterations or self._config.max_iterations
        results: list[dict] = []
        current_id = initial_harness_id
        current_config = initial_config

        for i in range(max_iter):
            logger.info("Evolution iteration %d/%d", i + 1, max_iter)

            step = self.run_evolution_step(
                current_harness_id=current_id,
                current_config=current_config,
                evaluate_fn=evaluate_fn,
            )
            results.append(step)

            if step["abandoned"]:
                break

            # Update current best from frontier
            frontier = self._store.get_pareto_frontier()
            if frontier:
                best = max(frontier, key=lambda f: f["conformance_rate"])
                best_variant = self._store.get_variant(best["harness_id"])
                if best_variant and best_variant.harness_id != current_id:
                    current_id = best_variant.harness_id
                    logger.info("Switched to best variant: %s", current_id)

        return results

    # -- Promote / retire logic ---------------------------------------------

    def _promote_winners(self) -> list[str]:
        """Promote variants that significantly outperform the current best."""
        frontier = self._store.get_pareto_frontier()
        if not frontier:
            return []

        # Find the current best conformance on the frontier
        best_rate = max(f["conformance_rate"] for f in frontier)

        # Check active variants that beat the frontier by promote_threshold
        active = self._store.list_variants(status="active")
        promotions: list[str] = []

        for variant in active:
            if variant.total_docs == 0:
                continue
            if variant.conformance_rate > best_rate + self._config.promote_threshold:
                self._store.set_variant_status(variant.harness_id, "promoted")
                self._store.log_evolution(
                    self._iteration, "promote", variant.harness_id,
                    f"Conformance {variant.conformance_rate:.1%} exceeds "
                    f"frontier best {best_rate:.1%} by "
                    f"{(variant.conformance_rate - best_rate):.1%}",
                )
                promotions.append(variant.harness_id)
                logger.info("Promoted %s (%.1f%%)", variant.harness_id, variant.conformance_rate * 100)

        return promotions

    def _retire_losers(self) -> list[str]:
        """Retire underperforming variants."""
        active = self._store.list_variants(status="active")
        retirements: list[str] = []

        # Get frontier harness IDs
        frontier = self._store.get_pareto_frontier()
        frontier_ids = {f["harness_id"] for f in frontier}

        for variant in active:
            if variant.total_docs == 0:
                continue  # Not yet evaluated

            # Retire if: has been evaluated, not on frontier, and has been
            # around for enough iterations
            if variant.harness_id not in frontier_ids:
                # Check age by comparing against iteration count
                experiments = self._store.get_experiments_for_harness(variant.harness_id)
                if len(experiments) >= self._config.min_docs_before_evolve:
                    # Has been fully evaluated and is not on frontier
                    self._store.set_variant_status(variant.harness_id, "retired")
                    self._store.log_evolution(
                        self._iteration, "retire", variant.harness_id,
                        f"Not on Pareto frontier after evaluation "
                        f"(conformance={variant.conformance_rate:.1%})",
                    )
                    retirements.append(variant.harness_id)
                    logger.info("Retired %s (%.1f%%)", variant.harness_id, variant.conformance_rate * 100)

        # Enforce max_active_variants
        remaining_active = self._store.list_variants(status="active")
        scored_active = [v for v in remaining_active if v.total_docs > 0]
        if len(scored_active) > self._config.max_active_variants:
            # Retire lowest-performing active variants
            scored_active.sort(key=lambda v: v.conformance_rate)
            excess = len(scored_active) - self._config.max_active_variants
            for v in scored_active[:excess]:
                if v.harness_id not in frontier_ids:
                    self._store.set_variant_status(v.harness_id, "retired")
                    self._store.log_evolution(
                        self._iteration, "retire", v.harness_id,
                        "Max active variants exceeded",
                    )
                    retirements.append(v.harness_id)

        return retirements

    def _should_abandon(self) -> bool:
        """Check if the evolution should be abandoned."""
        if self._iteration < self._config.abandon_after_iterations:
            return False

        frontier = self._store.get_pareto_frontier()
        if not frontier:
            return True

        best_rate = max(f["conformance_rate"] for f in frontier)
        return best_rate < self._config.abandon_threshold

    # -- Reporting ----------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get current evolution status."""
        active = self._store.list_variants(status="active")
        promoted = self._store.list_variants(status="promoted")
        retired = self._store.list_variants(status="retired")
        frontier = self._store.get_pareto_frontier()
        log = self._store.get_evolution_log(limit=10)

        return {
            "iteration": self._iteration,
            "active_variants": len(active),
            "promoted_variants": len(promoted),
            "retired_variants": len(retired),
            "frontier_size": len(frontier),
            "frontier": frontier,
            "recent_actions": [
                {
                    "action": entry["action"],
                    "harness_id": entry["harness_id"],
                    "details": entry["details"],
                }
                for entry in log
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_violation_types(trace: dict) -> list[str]:
    """Extract unique violation type prefixes from a trace.

    Groups by rule category (e.g., "7.1", "7.2", "7.5") rather than
    individual rules for cleaner analysis.
    """
    violation_types: set[str] = set()

    # Try multiple possible trace formats
    violations = trace.get("violations_list", [])
    if not violations:
        # The pipeline may embed violations differently
        plan = trace.get("plan", {})
        if isinstance(plan, dict):
            for op in plan.get("operations", []):
                reason = op.get("reason", "")
                # Try to extract rule IDs from reason strings
                if reason:
                    import re
                    matches = re.findall(r"(\d+\.\d+)", reason)
                    for m in matches:
                        violation_types.add(m)

    for v in violations:
        rule_id = v.get("rule_id", v.get("id", ""))
        if rule_id:
            # Extract category prefix (e.g., "7.1" from "7.1-3")
            parts = rule_id.split("-")
            if parts:
                violation_types.add(parts[0])

    return sorted(violation_types)


def classify_document_type(violations: list[dict], page_count: int) -> str:
    """Classify a document by its predominant violation pattern.

    Returns one of:
    - "table_heavy": >50% table violations
    - "untagged_content": >50% structure violations (7.1)
    - "mixed_structure": multiple violation types
    - "reading_order": primarily reading order issues
    - "figure_alt": primarily figure/alt text issues
    - "simple": few violations, simple document
    - "complex": many violations, multi-page
    """
    if not violations:
        return "simple"

    type_counts: dict[str, int] = {}
    for v in violations:
        rule_id = v.get("rule_id", v.get("id", ""))
        prefix = rule_id.split("-")[0] if rule_id else "unknown"
        type_counts[prefix] = type_counts.get(prefix, 0) + 1

    total = sum(type_counts.values())
    if total == 0:
        return "simple"

    # Check dominant type
    table_count = type_counts.get("7.2", 0) + type_counts.get("7.5", 0)
    struct_count = type_counts.get("7.1", 0)
    order_count = type_counts.get("7.3", 0)
    figure_count = type_counts.get("1", 0) + type_counts.get("1.1", 0)

    if table_count / total > 0.5:
        return "table_heavy"
    if struct_count / total > 0.5:
        return "untagged_content"
    if order_count / total > 0.3:
        return "reading_order"
    if figure_count / total > 0.3:
        return "figure_alt"

    # Multi-type or complex
    if page_count > 10 or total > 20:
        return "complex"
    if len(type_counts) >= 3:
        return "mixed_structure"

    return "simple"
