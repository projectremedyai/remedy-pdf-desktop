"""Mode A telemetry (REMEDY-73 follow-up, Tier 2.7).

Structured per-job telemetry records for Tier 2.7 Mode A faithful-rebuild
runs.  No external dependencies — plain dataclasses + structured JSON logs.
Aggregation happens at batch-end in the pipeline orchestrator.

Mirrors :mod:`project_remedy.faithful_rebuild.mode_b_telemetry` and
:mod:`project_remedy.faithful_rebuild.simple_font_telemetry`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModeARunTelemetry:
    """Per-job telemetry for a single Tier 2.7 Mode A attempt."""

    job_id: str
    attempted: bool
    skip_reason: str | None
    rebuild_qualified: bool
    structure_violations_before: int
    structure_violations_after: int
    visual_diff_score: float | None
    violations_before: int
    violations_after: int
    elapsed_seconds: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_mode_a_attempt(
    job_id: str,
    result: Any,  # ModeARunResult — avoid circular import
    violations_before: int,
    violations_after: int,
) -> ModeARunTelemetry:
    """Emit a structured telemetry record at INFO level and return it.

    Caller is responsible for persisting the record if desired; this function
    only emits the log entry (tagged ``mode_a_telemetry <json>``).
    """

    visual = getattr(result, "visual_diff_score", None)
    if visual is not None:
        try:
            visual = float(visual)
        except (TypeError, ValueError):
            visual = None

    rec = ModeARunTelemetry(
        job_id=job_id,
        attempted=bool(getattr(result, "attempted", False)),
        skip_reason=getattr(result, "skip_reason", None),
        rebuild_qualified=bool(getattr(result, "rebuild_qualified", False)),
        structure_violations_before=int(
            getattr(result, "structure_violations_before", 0) or 0
        ),
        structure_violations_after=int(
            getattr(result, "structure_violations_after", 0) or 0
        ),
        visual_diff_score=visual,
        violations_before=int(violations_before),
        violations_after=int(violations_after),
        elapsed_seconds=float(getattr(result, "elapsed_seconds", 0.0) or 0.0),
        error=getattr(result, "error", None),
    )
    logger.info("mode_a_telemetry %s", json.dumps(rec.as_dict()))
    return rec


def aggregate_mode_a_telemetry(
    records: list[ModeARunTelemetry],
) -> dict[str, Any]:
    """Summarize a list of telemetry records for batch-end reporting."""

    total = len(records)
    if total == 0:
        return {
            "total_jobs": 0,
            "total_attempted": 0,
            "total_qualified": 0,
            "total_structure_violations_resolved": 0,
            "total_violations_resolved": 0,
            "jobs_with_error": 0,
            "jobs_visually_drifted": 0,
            "total_elapsed_seconds": 0.0,
            "mean_elapsed_seconds": 0.0,
        }

    attempted = [r for r in records if r.attempted]
    total_elapsed = sum(r.elapsed_seconds for r in records)
    return {
        "total_jobs": total,
        "total_attempted": len(attempted),
        "total_qualified": sum(1 for r in records if r.rebuild_qualified),
        "total_structure_violations_resolved": sum(
            r.structure_violations_before - r.structure_violations_after
            for r in attempted
        ),
        "total_violations_resolved": sum(
            r.violations_before - r.violations_after
            for r in attempted
        ),
        "jobs_with_error": sum(1 for r in records if r.error),
        "jobs_visually_drifted": sum(
            1 for r in records
            if r.skip_reason and "visual drift" in r.skip_reason
        ),
        "total_elapsed_seconds": total_elapsed,
        "mean_elapsed_seconds": total_elapsed / total if total else 0.0,
    }
