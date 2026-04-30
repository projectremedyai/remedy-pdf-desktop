"""Simple-font telemetry (REMEDY-73).

Structured per-job telemetry records for Tier 2.6 simple-font runs.  No
external dependencies — plain dataclasses + structured JSON logs.
Aggregation happens at batch-end in the pipeline orchestrator.

Mirrors :mod:`project_remedy.faithful_rebuild.mode_b_telemetry`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SimpleFontRunTelemetry:
    """Per-job telemetry for a single Tier 2.6 simple-font attempt."""

    job_id: str
    attempted: bool
    skip_reason: str | None
    encoding_repair_attempted: bool
    fonts_encoding_repaired: int
    replacement_qualified: bool
    fonts_total: int
    fonts_replaced: int
    violations_before: int
    violations_after: int
    elapsed_seconds: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_simple_font_attempt(
    job_id: str,
    result: Any,  # SimpleFontRunResult — avoid circular import
    violations_before: int,
    violations_after: int,
) -> SimpleFontRunTelemetry:
    """Emit a structured telemetry record at INFO level and return it.

    Caller is responsible for persisting the record if desired; this function
    only emits the log entry.
    """

    rec = SimpleFontRunTelemetry(
        job_id=job_id,
        attempted=bool(getattr(result, "attempted", False)),
        skip_reason=getattr(result, "skip_reason", None),
        encoding_repair_attempted=bool(
            getattr(result, "encoding_repair_attempted", False)
        ),
        fonts_encoding_repaired=int(
            getattr(result, "fonts_encoding_repaired", 0) or 0
        ),
        replacement_qualified=bool(
            getattr(result, "replacement_qualified", False)
        ),
        fonts_total=int(getattr(result, "fonts_total", 0) or 0),
        fonts_replaced=int(getattr(result, "fonts_replaced", 0) or 0),
        violations_before=int(violations_before),
        violations_after=int(violations_after),
        elapsed_seconds=float(getattr(result, "elapsed_seconds", 0.0) or 0.0),
        error=getattr(result, "error", None),
    )
    logger.info("simple_font_telemetry %s", json.dumps(rec.as_dict()))
    return rec


def aggregate_simple_font_telemetry(
    records: list[SimpleFontRunTelemetry],
) -> dict[str, Any]:
    """Summarize a list of telemetry records for batch-end reporting."""

    total = len(records)
    if total == 0:
        return {
            "total_jobs": 0,
            "total_attempted": 0,
            "total_encoding_repair_attempted": 0,
            "total_encoding_repaired": 0,
            "total_replacement_qualified": 0,
            "total_fonts_total": 0,
            "total_replaced": 0,
            "violations_delta_total": 0,
            "jobs_with_error": 0,
            "total_elapsed_seconds": 0.0,
            "mean_elapsed_seconds": 0.0,
        }

    attempted = [r for r in records if r.attempted]
    total_elapsed = sum(r.elapsed_seconds for r in records)
    return {
        "total_jobs": total,
        "total_attempted": len(attempted),
        "total_encoding_repair_attempted": sum(
            1 for r in records if r.encoding_repair_attempted
        ),
        "total_encoding_repaired": sum(
            r.fonts_encoding_repaired for r in records
        ),
        "total_replacement_qualified": sum(
            1 for r in records if r.replacement_qualified
        ),
        "total_fonts_total": sum(r.fonts_total for r in records),
        "total_replaced": sum(r.fonts_replaced for r in records),
        "violations_delta_total": sum(
            r.violations_before - r.violations_after for r in attempted
        ),
        "jobs_with_error": sum(1 for r in records if r.error),
        "total_elapsed_seconds": total_elapsed,
        "mean_elapsed_seconds": total_elapsed / total if total else 0.0,
    }
