"""Scorer: evaluate harness variants against held-out document sets.

Computes aggregate metrics from experiment records and determines
whether a variant should be promoted or retired.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from project_remedy.vision_planner.experiment_store import (
    ExperimentRecord,
    ExperimentStore,
    HarnessVariant,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


@dataclass
class ScoringResult:
    """Aggregate scoring result for a harness variant."""

    harness_id: str
    conformance_rate: float
    manual_review_rate: float
    destructive_edit_count: int
    avg_seconds: float
    total_docs: int
    passed_docs: int
    improvement_over_baseline: float | None = None  # delta vs baseline
    on_pareto_frontier: bool = False


def compute_metrics_from_experiments(
    experiments: list[ExperimentRecord],
) -> dict[str, Any]:
    """Compute aggregate metrics from a list of experiment records.

    Returns dict matching the scorer.py format in meta-harness-remedy:
    conformance_rate, manual_review_rate, destructive_edit_count,
    avg_seconds, total_docs, passed_docs.
    """
    if not experiments:
        return {
            "conformance_rate": 0.0,
            "manual_review_rate": 0.0,
            "destructive_edit_count": 0,
            "avg_seconds": 0.0,
            "total_docs": 0,
            "passed_docs": 0,
        }

    total = len(experiments)
    passed = sum(1 for e in experiments if e.passed)

    # Count manual_review operations vs total operations
    total_ops = 0
    manual_ops = 0
    for exp in experiments:
        for op in exp.fix_sequence:
            total_ops += 1
            if op.get("action") == "mark_manual_review":
                manual_ops += 1

    # Destructive edits: docs where violations increased
    destructive = sum(
        1 for e in experiments if e.violations_after > e.violations_before
    )

    # Average time
    times = [e.elapsed_seconds for e in experiments if e.elapsed_seconds > 0]
    avg_seconds = sum(times) / len(times) if times else 0.0

    return {
        "conformance_rate": passed / total,
        "manual_review_rate": manual_ops / total_ops if total_ops > 0 else 0.0,
        "destructive_edit_count": destructive,
        "avg_seconds": round(avg_seconds, 1),
        "total_docs": total,
        "passed_docs": passed,
    }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class HarnessScorer:
    """Score harness variants and update the experiment store metrics.

    Computes metrics from experiment records, updates the variant's
    stored metrics, and determines Pareto frontier membership.
    """

    def __init__(
        self,
        store: ExperimentStore,
        baseline_harness_id: str = "h000_baseline",
        min_docs_for_scoring: int = 5,
    ):
        self._store = store
        self._baseline_id = baseline_harness_id
        self._min_docs = min_docs_for_scoring

    def score_variant(self, harness_id: str) -> ScoringResult | None:
        """Score a harness variant based on its experiment records.

        Returns None if insufficient data (fewer than min_docs experiments).
        Updates the variant's metrics in the store.
        """
        experiments = self._store.get_experiments_for_harness(harness_id)

        if len(experiments) < self._min_docs:
            logger.info(
                "Skipping scoring for %s: only %d/%d docs evaluated",
                harness_id, len(experiments), self._min_docs,
            )
            return None

        metrics = compute_metrics_from_experiments(experiments)

        # Update the variant's stored metrics
        self._store.update_variant_metrics(
            harness_id=harness_id,
            conformance_rate=metrics["conformance_rate"],
            manual_review_rate=metrics["manual_review_rate"],
            destructive_edit_count=metrics["destructive_edit_count"],
            avg_seconds=metrics["avg_seconds"],
            total_docs=metrics["total_docs"],
            passed_docs=metrics["passed_docs"],
        )

        # Compare against baseline
        improvement = None
        baseline = self._store.get_variant(self._baseline_id)
        if baseline and baseline.total_docs > 0:
            improvement = metrics["conformance_rate"] - baseline.conformance_rate

        # Update Pareto frontier
        frontier = self._store.update_pareto_frontier()
        on_frontier = any(f["harness_id"] == harness_id for f in frontier)

        result = ScoringResult(
            harness_id=harness_id,
            conformance_rate=metrics["conformance_rate"],
            manual_review_rate=metrics["manual_review_rate"],
            destructive_edit_count=metrics["destructive_edit_count"],
            avg_seconds=metrics["avg_seconds"],
            total_docs=metrics["total_docs"],
            passed_docs=metrics["passed_docs"],
            improvement_over_baseline=improvement,
            on_pareto_frontier=on_frontier,
        )

        logger.info(
            "Scored %s: conformance=%.1f%% (%d/%d), improvement=%s, frontier=%s",
            harness_id,
            result.conformance_rate * 100,
            result.passed_docs,
            result.total_docs,
            f"{improvement:+.1f}pp" if improvement is not None else "N/A",
            "YES" if on_frontier else "no",
        )

        return result

    def compare_variants(
        self, harness_a: str, harness_b: str
    ) -> dict[str, Any]:
        """Compare two harness variants head-to-head.

        Returns comparison dict with per-metric deltas and a recommendation.
        """
        variant_a = self._store.get_variant(harness_a)
        variant_b = self._store.get_variant(harness_b)

        if variant_a is None or variant_b is None:
            return {"error": "One or both variants not found"}

        deltas = {
            "conformance_rate": variant_a.conformance_rate - variant_b.conformance_rate,
            "manual_review_rate": variant_a.manual_review_rate - variant_b.manual_review_rate,
            "destructive_edit_count": variant_a.destructive_edit_count - variant_b.destructive_edit_count,
            "avg_seconds": variant_a.avg_seconds - variant_b.avg_seconds,
        }

        # Determine winner: higher conformance is primary metric
        if deltas["conformance_rate"] > 0.01:
            recommendation = f"{harness_a} is better (higher conformance)"
        elif deltas["conformance_rate"] < -0.01:
            recommendation = f"{harness_b} is better (higher conformance)"
        elif deltas["destructive_edit_count"] < 0:
            recommendation = f"{harness_a} is better (fewer destructive edits)"
        elif deltas["destructive_edit_count"] > 0:
            recommendation = f"{harness_b} is better (fewer destructive edits)"
        elif deltas["avg_seconds"] < -5:
            recommendation = f"{harness_a} is better (faster)"
        elif deltas["avg_seconds"] > 5:
            recommendation = f"{harness_b} is better (faster)"
        else:
            recommendation = "No significant difference"

        return {
            "variant_a": harness_a,
            "variant_b": harness_b,
            "deltas": deltas,
            "recommendation": recommendation,
        }

    def rank_variants(
        self,
        status: str | None = None,
        metric: str = "conformance_rate",
        limit: int = 10,
    ) -> list[dict]:
        """Rank variants by a specific metric.

        Args:
            status: Filter by variant status (active/retired/promoted) or None for all.
            metric: Which metric to sort by.
            limit: Max variants to return.

        Returns list of dicts with harness_id and metric values.
        """
        variants = self._store.list_variants(status=status)

        # Only include variants with experiments
        scored = [v for v in variants if v.total_docs > 0]

        # Sort by metric (higher is better for conformance, lower for others)
        reverse = metric in ("conformance_rate",)
        scored.sort(key=lambda v: getattr(v, metric, 0), reverse=reverse)

        return [
            {
                "harness_id": v.harness_id,
                "conformance_rate": v.conformance_rate,
                "manual_review_rate": v.manual_review_rate,
                "destructive_edit_count": v.destructive_edit_count,
                "avg_seconds": v.avg_seconds,
                "total_docs": v.total_docs,
                "passed_docs": v.passed_docs,
                "status": v.status,
            }
            for v in scored[:limit]
        ]
