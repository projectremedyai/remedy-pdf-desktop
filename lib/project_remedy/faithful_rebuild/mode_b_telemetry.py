"""Mode B telemetry (REMEDY-78).

Structured per-job telemetry records for Tier 2.5 Mode B runs. No external
dependencies — plain dataclasses + structured JSON logs. Aggregation
happens at batch-end in the pipeline orchestrator.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModeBRunTelemetry:
    """Per-job telemetry for a single Mode B attempt."""
    job_id: str
    attempted: bool
    skip_reason: str | None
    eligibility_qualified: bool
    fonts_total: int
    fonts_replaced: int
    cids_recovered: int
    violations_before: int
    violations_after: int
    elapsed_seconds: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_mode_b_attempt(
    job_id: str,
    result: Any,  # ModeBRunResult from mode_b_production — avoid circular import
    violations_before: int,
    violations_after: int,
) -> ModeBRunTelemetry:
    """Emit a structured telemetry record. Logs at INFO with structured JSON.

    Caller is responsible for persisting the record if desired; this function
    only emits the log entry.
    """
    rec = ModeBRunTelemetry(
        job_id=job_id,
        attempted=getattr(result, "attempted", False),
        skip_reason=getattr(result, "skip_reason", None),
        eligibility_qualified=getattr(result, "eligibility_qualified", False),
        fonts_total=getattr(result, "fonts_total", 0),
        fonts_replaced=getattr(result, "fonts_replaced", 0),
        cids_recovered=getattr(result, "cids_recovered", 0),
        violations_before=violations_before,
        violations_after=violations_after,
        elapsed_seconds=getattr(result, "elapsed_seconds", 0.0),
        error=getattr(result, "error", None),
    )
    logger.info("mode_b_telemetry %s", json.dumps(rec.as_dict()))
    return rec


def aggregate_telemetry(records: list[ModeBRunTelemetry]) -> dict[str, Any]:
    """Summarize a list of telemetry records for batch-end reporting."""
    total = len(records)
    if total == 0:
        return {
            "total_jobs": 0,
            "attempted": 0,
            "eligibility_qualified": 0,
            "fonts_replaced_total": 0,
            "cids_recovered_total": 0,
            "jobs_with_error": 0,
            "total_elapsed_seconds": 0.0,
        }
    return {
        "total_jobs": total,
        "attempted": sum(1 for r in records if r.attempted),
        "eligibility_qualified": sum(1 for r in records if r.eligibility_qualified),
        "fonts_replaced_total": sum(r.fonts_replaced for r in records),
        "cids_recovered_total": sum(r.cids_recovered for r in records),
        "violations_delta_total": sum(
            r.violations_before - r.violations_after for r in records
            if r.attempted
        ),
        "jobs_with_error": sum(1 for r in records if r.error),
        "total_elapsed_seconds": sum(r.elapsed_seconds for r in records),
    }
