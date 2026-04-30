"""Lightweight export helpers for coordinator benchmark records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from project_remedy.pdf_specialists.contracts import CoordinatorResult


def count_acceptance_warnings(acceptance: Any) -> int:
    """Best-effort warning/violation count from an acceptance-like object."""
    return len(list(getattr(acceptance, "warning_entries", []) or []))


def build_benchmark_record(
    *,
    mode: str,
    document_path: Path | str | None,
    document_hash: str,
    output_path: Path | str | None,
    initial_acceptance: Any,
    final_acceptance: Any,
    trace_path: Path | str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a comparison row for non-coordinator benchmark modes."""
    record: dict[str, Any] = {
        "mode": mode,
        "document_path": str(document_path) if document_path is not None else "",
        "document_hash": document_hash,
        "passed": bool(getattr(final_acceptance, "passed", False)),
        "output_path": str(output_path) if output_path is not None else "",
        "trace_path": str(trace_path) if trace_path is not None else "",
        "acceptance_summary": final_acceptance.summary() if hasattr(final_acceptance, "summary") else "",
        "acceptance_passed": bool(getattr(final_acceptance, "passed", False)),
        "violation_count_before": count_acceptance_warnings(initial_acceptance),
        "violation_count_after": count_acceptance_warnings(final_acceptance),
        "conflict_count": 0,
        "planner_client_available": False,
        "planner_client_type": "",
        "route_hints": [],
        "promoted_phases": [],
        "skipped_phases": [],
        "tool_phases_run": [],
        "rebuild_used": False,
        "planner_phase_executed": False,
        "promoted_phase_count": 0,
        "semantic_operations_executed": 0,
        "semantic_manual_review_count": 0,
        "total_change_entries": 0,
        "phase_results": {},
        "skipped": False,
        "skipped_reason": "",
    }
    if extra:
        record.update(extra)
    return record


def build_coordinator_export_record(
    result: CoordinatorResult,
    *,
    document_path: Path | str | None = None,
    document_hash: str = "",
    mode: str = "specialist_coordinator",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable file-based export record from a coordinator result."""
    summary = dict(result.trace_summary)
    record: dict[str, Any] = {
        "mode": mode,
        "document_path": str(document_path) if document_path is not None else "",
        "document_hash": document_hash,
        "passed": bool(result.passed),
        "output_path": str(result.output_path),
        "trace_path": str(result.trace_path) if result.trace_path else "",
        "acceptance_summary": result.acceptance_summary,
        "acceptance_passed": bool(result.acceptance_passed),
        "violation_count_before": int(result.violation_count_before),
        "violation_count_after": int(result.violation_count_after),
        "conflict_count": len(result.conflicts),
        "planner_client_available": bool(summary.get("planner_client_available", False)),
        "planner_client_type": summary.get("planner_client_type", ""),
        "route_hints": list(summary.get("route_hints", [])),
        "promoted_phases": list(summary.get("promoted_phases", [])),
        "skipped_phases": list(summary.get("skipped_phases", [])),
        "tool_phases_run": list(summary.get("tool_phases_run", [])),
        "rebuild_used": bool(summary.get("rebuild_used", False)),
        "planner_phase_executed": bool(summary.get("planner_phase_executed", False)),
        "promoted_phase_count": int(summary.get("promoted_phase_count", 0)),
        "semantic_operations_executed": int(summary.get("semantic_operations_executed", 0)),
        "semantic_manual_review_count": int(summary.get("semantic_manual_review_count", 0)),
        "total_change_entries": int(summary.get("total_change_entries", 0)),
        "phase_results": dict(summary.get("phase_results", {})),
        "skipped": False,
        "skipped_reason": "",
    }
    if extra:
        record.update(extra)
    return record


def append_export_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSONL export record to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
