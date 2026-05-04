"""Residual-driven post-remediation strategy helpers."""

from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import sys
import time
import types
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlparse

from backend.app.config import settings
from backend.app.ollama_runtime import build_local_vision_provider, local_ollama_model_ready


def _ensure_monkeypatchable_compliance_report() -> None:
    """Install a tiny stub when this interpreter cannot import the real module."""
    try:
        importlib.import_module("project_remedy.compliance_report")
    except Exception as exc:
        stub = types.ModuleType("project_remedy.compliance_report")

        def _missing_generate_document_report(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "project_remedy.compliance_report is unavailable in this interpreter"
            ) from exc

        stub.generate_document_report = _missing_generate_document_report
        sys.modules["project_remedy.compliance_report"] = stub
        try:
            project_remedy_pkg = importlib.import_module("project_remedy")
            setattr(project_remedy_pkg, "compliance_report", stub)
        except Exception:
            pass


_ensure_monkeypatchable_compliance_report()

RepairStrategyId = Literal[
    "rebuild_structure",
    "repair_cid_fonts",
    "repair_embedded_fonts",
    "add_math_formula_tags",
    "ocr_rebuild",
    "manual_review",
]

STRATEGY_LABELS: dict[RepairStrategyId, str] = {
    "rebuild_structure": "Rebuild document structure",
    "repair_cid_fonts": "Repair broken CID fonts",
    "repair_embedded_fonts": "Repair broken embedded fonts",
    "add_math_formula_tags": "Add math formula tags",
    "ocr_rebuild": "OCR rebuild",
    "manual_review": "Flag for manual review",
}

DISPLAY_ORDER: tuple[RepairStrategyId, ...] = (
    "repair_cid_fonts",
    "repair_embedded_fonts",
    "rebuild_structure",
    "manual_review",
    "add_math_formula_tags",
    "ocr_rebuild",
)

_PDF_ONLY_STRATEGIES: frozenset[RepairStrategyId] = frozenset({
    "rebuild_structure",
    "repair_cid_fonts",
    "repair_embedded_fonts",
    "add_math_formula_tags",
    "ocr_rebuild",
})

_LOCKS: dict[str, Lock] = {}


@dataclass
class StrategyOption:
    """One strategy row for the picker UI."""

    id: RepairStrategyId
    label: str
    reason: str
    selected_by_default: bool = False


@dataclass
class StrategyRunRecord:
    """Persisted result of one manual post-remediation strategy run."""

    strategy_id: RepairStrategyId
    label: str
    status: str
    summary: str
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None


@dataclass
class StrategyState:
    """Per-job persisted strategy state."""

    job_id: str
    current_output_path: str | None = None
    runs: list[StrategyRunRecord] = field(default_factory=list)


@lru_cache(maxsize=1)
def _load_manual_strategy_config():
    """Load a config tuned for explicit post-remediation strategy runs."""
    from project_remedy.config import load_config

    base_config = load_config()
    pdf_config = replace(
        base_config.pdf_remediation,
        enabled=True,
        font_mode_b_enabled=True,
        simple_font_replacement_enabled=True,
        simple_font_encoding_repair_enabled=True,
        font_mode_a_enabled=True,
    )
    return replace(base_config, pdf_remediation=pdf_config)


def _strategy_dir(job_id: str) -> Path:
    return settings.output_dir / job_id / "repair_strategies"


def _state_path(job_id: str) -> Path:
    return _strategy_dir(job_id) / "state.json"


def _state_to_dict(state: StrategyState) -> dict[str, Any]:
    return {
        "job_id": state.job_id,
        "current_output_path": state.current_output_path,
        "runs": [asdict(run) for run in state.runs],
    }


def _state_from_dict(data: dict[str, Any]) -> StrategyState:
    return StrategyState(
        job_id=data["job_id"],
        current_output_path=data.get("current_output_path"),
        runs=[
            StrategyRunRecord(
                strategy_id=run["strategy_id"],
                label=run["label"],
                status=run["status"],
                summary=run["summary"],
                error=run.get("error"),
                started_at=run.get("started_at", time.time()),
                completed_at=run.get("completed_at"),
            )
            for run in data.get("runs", [])
        ],
    )


def load_strategy_state(job_id: str) -> StrategyState:
    """Load persisted strategy state for a job if present."""
    path = _state_path(job_id)
    if not path.exists():
        return StrategyState(job_id=job_id)

    try:
        data = json.loads(path.read_text())
    except Exception:
        return StrategyState(job_id=job_id)

    return _state_from_dict(data)


def save_strategy_state(state: StrategyState) -> None:
    """Persist strategy state atomically."""
    path = _state_path(state.job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(_state_to_dict(state), indent=2))
    temp_path.replace(path)


def try_acquire_strategy_lock(job_id: str) -> bool:
    """Try to acquire the per-job strategy lock."""
    lock = _LOCKS.setdefault(job_id, Lock())
    return lock.acquire(blocking=False)


def release_strategy_lock(job_id: str) -> None:
    """Release the per-job strategy lock if held."""
    lock = _LOCKS.setdefault(job_id, Lock())
    if lock.locked():
        lock.release()


def is_strategy_locked(job_id: str) -> bool:
    """Return whether the per-job strategy executor is currently running."""
    return _LOCKS.setdefault(job_id, Lock()).locked()


def get_current_strategy_output_path(job: Any) -> Path | None:
    """Return the latest successful strategy output, if any."""
    state = load_strategy_state(job.id)
    if not state.current_output_path:
        return None

    current_path = Path(state.current_output_path)
    if current_path.exists():
        return current_path

    state.current_output_path = None
    save_strategy_state(state)
    return None


def _resolve_current_output_path(job: Any, state: StrategyState) -> Path | None:
    """Resolve the current working PDF path for strategy analysis."""
    if state.current_output_path:
        current_path = Path(state.current_output_path)
        if current_path.exists():
            return current_path
        state.current_output_path = None
        save_strategy_state(state)

    if job.output_path and job.output_path.exists():
        return job.output_path

    return None


def _evaluate_acceptance(job: Any, pdf_path: Path):
    from project_remedy.pdf_acceptance import evaluate_pdf_acceptance

    return evaluate_pdf_acceptance(
        pdf_path,
        original_path=job.input_path,
        config=_load_manual_strategy_config(),
        full_verification=getattr(job, "verification_mode", "sampled") == "full",
    )


def _refresh_job_report(job: Any, pdf_path: Path) -> None:
    """Regenerate acceptance/report data for the latest strategy output."""
    from project_remedy.compliance_report import generate_document_report

    acceptance = _evaluate_acceptance(job, pdf_path)
    report_dir = _strategy_dir(job.id) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    doc_report = generate_document_report(
        original_path=job.input_path,
        remediated_path=pdf_path,
        output_dir=report_dir,
        config=_load_manual_strategy_config(),
        acceptance=acceptance,
        verification_mode=getattr(job, "verification_mode", "sampled"),
    )

    failed_checks = [
        {
            "rule_id": check.get("rule_id"),
            "description": check.get("description"),
            "fixable": check.get("fixable"),
        }
        for check in doc_report.check_results
        if check.get("status") == "Failed"
    ]
    manual_review_checks = [
        {
            "rule_id": check.get("rule_id"),
            "description": check.get("description"),
            "details": check.get("details", []),
            "decision": check.get("decision", "manual_review"),
            "recommendation": check.get("recommendation", ""),
        }
        for check in getattr(doc_report, "manual_review_checks", [])
    ]
    reviewable_checks = [
        {
            "rule_id": check.get("rule_id"),
            "description": check.get("description"),
            "details": check.get("details", []),
            "decision": check.get("decision", "manual_review"),
            "recommendation": check.get("recommendation", ""),
            "status": check.get("status"),
        }
        for check in getattr(doc_report, "reviewable_checks", [])
    ]
    source_limited_issues = [
        {
            "rule_id": issue.get("rule_id"),
            "description": issue.get("description"),
            "details": issue.get("details", []),
            "recommendation": issue.get("recommendation", ""),
        }
        for issue in getattr(doc_report, "source_limited_issues", [])
    ]
    sr_issues_grouped: dict[str, dict[str, Any]] = {}
    for sr_issue in doc_report.sr_issues:
        rule_id = sr_issue.get("rule_id", "")
        if rule_id not in sr_issues_grouped:
            sr_issues_grouped[rule_id] = {
                "rule_id": rule_id,
                "severity": sr_issue.get("severity"),
                "count": 0,
            }
        sr_issues_grouped[rule_id]["count"] += 1

    wcag_results = [
        {
            "criterion_id": result.criterion_id,
            "criterion_name": result.criterion_name,
            "level": result.level,
            "status": result.status,
            "remarks": result.remarks,
        }
        for result in doc_report.wcag_results
    ]
    wcag_failures = [result for result in wcag_results if result["status"] == "FAIL"]
    wcag_manual_reviews = [result for result in wcag_results if result["status"] == "REVIEW"]
    verification_coverage = getattr(doc_report, "verification_coverage", None) or {
        "mode": getattr(job, "verification_mode", "sampled"),
        "vision_checked": False,
        "total_pages": int(getattr(doc_report, "remediated_pages", 0) or 0),
        "analyzed_page_count": 0,
        "analyzed_pages": [],
        "covers_all_pages": False,
        "unresolved_sampled_checks": [],
    }

    job.output_path = pdf_path
    job.report_path = report_dir / doc_report.report_filename if doc_report.report_filename else None
    job.report_data = {
        "document_name": doc_report.document_name,
        "conformance": doc_report.conformance,
        "tag_count": doc_report.tag_count,
        "pages": doc_report.remediated_pages,
        "failed_checks": failed_checks,
        "reviewable_checks": reviewable_checks,
        "manual_review_checks": manual_review_checks,
        "source_limited_issues": source_limited_issues,
        "source_limited_count": len(source_limited_issues),
        "sr_issues": list(sr_issues_grouped.values()),
        "wcag_results": wcag_results,
        "wcag_failures": wcag_failures,
        "wcag_manual_reviews": wcag_manual_reviews,
        "wcag_pass_count": sum(1 for result in wcag_results if result["status"] == "PASS"),
        "wcag_fail_count": len(wcag_failures),
        "wcag_review_count": len(wcag_manual_reviews),
        "total_issues": len(doc_report.issues),
        "screen_reader_readability": doc_report.screen_reader_readability,
        "verapdf_checked": doc_report.verapdf_checked,
        "verapdf_passed": doc_report.verapdf_passed,
        "verapdf_violation_count": len(doc_report.verapdf_violations),
        "verification_coverage": verification_coverage,
        "reading_order_source": getattr(job, "reading_order_source", "none"),
        "xy_cut": getattr(job, "xy_cut_result", None),
    }


def _has_ocr_signal(job: Any, acceptance: Any) -> bool:
    report = getattr(job, "report_data", None) or {}
    keywords = (
        "ocr",
        "image-only",
        "image only",
        "searchable text layer",
        "character encoding",
    )

    for failure in getattr(acceptance, "checker_failures", []) or []:
        if getattr(failure, "rule_id", "") in {"doc-not-image-only", "page-char-encoding"}:
            return True
        detail_text = " ".join(str(detail) for detail in getattr(failure, "details", []) or [])
        if any(keyword in detail_text.lower() for keyword in keywords):
            return True

    for check in report.get("failed_checks", []) or []:
        rule_id = str(check.get("rule_id", ""))
        if rule_id in {"doc-not-image-only", "page-char-encoding"}:
            return True
        detail_text = " ".join(
            str(fragment)
            for fragment in (
                check.get("rule_id", ""),
                check.get("description", ""),
            )
        )
        if any(keyword in detail_text.lower() for keyword in keywords):
            return True

    for failure in report.get("wcag_failures", []) or []:
        detail_text = " ".join(
            str(fragment)
            for fragment in (
                failure.get("criterion_id", ""),
                failure.get("criterion_name", ""),
                failure.get("remarks", ""),
            )
        )
        if any(keyword in detail_text.lower() for keyword in keywords):
            return True

    return False


def _has_math_signal(job: Any, acceptance: Any) -> bool:
    report = getattr(job, "report_data", None) or {}
    text_fragments: list[str] = []

    for failure in getattr(acceptance, "checker_failures", []) or []:
        text_fragments.append(str(getattr(failure, "rule_id", "")))
        text_fragments.extend(str(detail) for detail in getattr(failure, "details", []) or [])

    for check in report.get("failed_checks", []) or []:
        text_fragments.extend([
            str(check.get("rule_id", "")),
            str(check.get("description", "")),
        ])

    for failure in report.get("wcag_failures", []) or []:
        text_fragments.extend([
            str(failure.get("criterion_id", "")),
            str(failure.get("criterion_name", "")),
            str(failure.get("remarks", "")),
        ])

    haystack = " ".join(text_fragments).lower()
    return any(token in haystack for token in ("formula", "equation", "mathml", "latex"))


def _has_heading_signal(job: Any, acceptance: Any | None = None) -> bool:
    """Return True when residuals indicate missing/invalid heading navigation."""
    report = getattr(job, "report_data", None) or {}
    text_fragments: list[str] = []

    if acceptance is not None:
        tag_tree_result = getattr(acceptance, "tag_tree_result", None)
        for issue in getattr(tag_tree_result, "issues", []) or []:
            text_fragments.extend([
                str(getattr(issue, "rule_id", "")),
                str(getattr(issue, "description", "")),
            ])

    for issue in report.get("sr_issues", []) or []:
        text_fragments.extend([
            str(issue.get("rule_id", "")),
            str(issue.get("description", "")),
        ])

    for failure in report.get("wcag_failures", []) or []:
        text_fragments.extend([
            str(failure.get("criterion_id", "")),
            str(failure.get("criterion_name", "")),
            str(failure.get("remarks", "")),
        ])

    haystack = " ".join(text_fragments).lower()
    return any(
        token in haystack
        for token in (
            "sr-no-headings",
            "sr-heading-start",
            "sr-heading-skip",
            "headings and labels",
            "no headings found",
        )
    )


def _pdf_has_formula_tags(pdf_path: Path) -> bool:
    try:
        import pikepdf
        from project_remedy.pdf_checker import _get_struct_type, walk_structure_tree

        with pikepdf.open(pdf_path) as pdf:
            for node, _depth, _parent in walk_structure_tree(pdf):
                if _get_struct_type(node) == "Formula":
                    return True
    except Exception:
        return False
    return False


def _missing_ocr_dependencies() -> list[str]:
    missing: list[str] = []

    if shutil.which("tesseract") is None:
        missing.append("tesseract")

    try:
        import fitz  # noqa: F401
    except Exception:
        missing.append("pymupdf")

    try:
        import pypdf  # noqa: F401
    except Exception:
        missing.append("pypdf")

    return missing


def _is_loopback_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").strip().lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _describe_math_strategy_runtime() -> tuple[bool, str]:
    if local_ollama_model_ready():
        return (
            True,
            "Math-related residuals were detected. Uses the app's local Ollama model to detect formulas and add /Formula tags with MathML when possible.",
        )

    config = _load_manual_strategy_config()
    backend = str(getattr(config.api, "llm_backend", "") or "ollama").strip().lower()

    if backend == "ollama":
        endpoint = str(config.api.vision_base_url or config.api.base_url or "http://localhost:11434/v1").strip()
        if _is_loopback_url(endpoint):
            return (
                True,
                "Math-related residuals were detected. Uses the configured local vision endpoint to detect formulas and add /Formula tags with MathML when possible.",
            )
        return (
            True,
            "Math-related residuals were detected. Uses the configured vision endpoint to detect formulas; depending on that endpoint, page images may be sent over the network.",
        )

    return (
        False,
        "Math-related residuals were detected, but math formula tagging does not have a usable vision provider in the current configuration.",
    )


def _build_math_vision_provider() -> Any | None:
    provider = build_local_vision_provider()
    if provider is not None:
        return provider

    from project_remedy.pdf_vision import create_provider_from_config

    return create_provider_from_config(_load_manual_strategy_config())


def _close_vision_provider(provider: Any) -> None:
    close = getattr(provider, "close", None)
    if callable(close):
        close()


def _dedupe_options(options: list[StrategyOption]) -> list[StrategyOption]:
    deduped: list[StrategyOption] = []
    seen: set[RepairStrategyId] = set()
    for option in options:
        if option.id in seen:
            continue
        deduped.append(option)
        seen.add(option.id)
    return deduped


def _needs_manual_review(job: Any, acceptance: Any) -> bool:
    fix_summary = getattr(job, "fix_summary", None) or {}
    if bool(fix_summary.get("needs_manual_review")):
        return True
    if getattr(acceptance, "warning_entries", None):
        return bool(acceptance.warning_entries)
    report = getattr(job, "report_data", None) or {}
    return any(
        report.get(key)
        for key in (
            "failed_checks",
            "manual_review_checks",
            "sr_issues",
            "wcag_failures",
            "wcag_manual_reviews",
        )
    )


def _build_supported_options(job: Any, acceptance: Any) -> tuple[list[StrategyOption], list[StrategyOption]]:
    from project_remedy.faithful_rebuild.mode_a_production import should_attempt_mode_a
    from project_remedy.faithful_rebuild.mode_b_production import should_attempt_mode_b
    from project_remedy.faithful_rebuild.simple_font_production import should_attempt_simple_font

    config = _load_manual_strategy_config()

    mode_b_applicable, mode_b_reason = should_attempt_mode_b(acceptance, config)
    simple_applicable, simple_reason = should_attempt_simple_font(acceptance, config)
    mode_a_applicable, mode_a_reason = should_attempt_mode_a(acceptance, config)
    heading_applicable = _has_heading_signal(job, acceptance)
    structure_applicable = mode_a_applicable or heading_applicable
    manual_review_applicable = _needs_manual_review(job, acceptance)

    applicable: list[StrategyOption] = []
    unavailable: list[StrategyOption] = []

    def add_option(strategy_id: RepairStrategyId, is_applicable: bool, reason: str, *, selected: bool = False) -> None:
        option = StrategyOption(
            id=strategy_id,
            label=STRATEGY_LABELS[strategy_id],
            reason=reason,
            selected_by_default=selected,
        )
        if is_applicable:
            applicable.append(option)
        else:
            unavailable.append(option)

    add_option(
        "repair_cid_fonts",
        mode_b_applicable,
        (
            "Residual CID/font-mapping issues were detected."
            if mode_b_applicable
            else _supported_unavailable_reason("repair_cid_fonts", mode_b_reason)
        ),
        selected=mode_b_applicable,
    )
    add_option(
        "repair_embedded_fonts",
        simple_applicable,
        (
            "Residual embedded-font issues were detected."
            if simple_applicable
            else _supported_unavailable_reason("repair_embedded_fonts", simple_reason)
        ),
        selected=simple_applicable,
    )
    add_option(
        "rebuild_structure",
        structure_applicable,
        (
            "Residual heading/navigation structure issues were detected."
            if heading_applicable
            else "Residual structure or reading-order issues were detected."
            if mode_a_applicable
            else _supported_unavailable_reason("rebuild_structure", mode_a_reason)
        ),
        selected=structure_applicable,
    )
    add_option(
        "manual_review",
        manual_review_applicable,
        (
            "Residual issues still need human review."
            if manual_review_applicable
            else "No residual issues were detected to flag."
        ),
        selected=False,
    )

    return applicable, unavailable


def _supported_unavailable_reason(strategy_id: RepairStrategyId, raw_reason: str) -> str:
    if strategy_id == "repair_cid_fonts":
        if raw_reason.startswith("verapdf:") or raw_reason.startswith("checker:"):
            return "CID-font repair did not qualify on the current output."
        return "No residual CID-font issues were detected."

    if strategy_id == "repair_embedded_fonts":
        if raw_reason.startswith("verapdf:"):
            return "Embedded-font repair did not qualify on the current output."
        return "No residual embedded-font issues were detected."

    if strategy_id == "rebuild_structure":
        if raw_reason.startswith("checker:") or raw_reason.startswith("verapdf:"):
            return "Structure rebuild did not qualify on the current output."
        return "No residual structure issues were detected."

    return "Not available."


def _build_extended_options(
    job: Any,
    acceptance: Any,
    pdf_path: Path,
) -> tuple[list[StrategyOption], list[StrategyOption]]:
    ocr_signal = _has_ocr_signal(job, acceptance)
    math_signal = _has_math_signal(job, acceptance)
    ocr_missing = _missing_ocr_dependencies()
    math_runtime_available, math_runtime_reason = _describe_math_strategy_runtime()
    has_formula_tags = _pdf_has_formula_tags(pdf_path)

    applicable: list[StrategyOption] = []
    unavailable: list[StrategyOption] = []

    if ocr_signal and not ocr_missing:
        applicable.append(
            StrategyOption(
                id="ocr_rebuild",
                label=STRATEGY_LABELS["ocr_rebuild"],
                reason="Residual OCR or broken-text-layer issues were detected. Rebuilds the searchable text layer locally with Tesseract OCR.",
            )
        )
    else:
        if not ocr_signal:
            ocr_reason = "No OCR or image-only residuals were detected on the current output."
        else:
            missing_text = ", ".join(ocr_missing)
            ocr_reason = (
                "Residual OCR or broken-text-layer issues were detected, but OCR rebuild "
                f"requires local dependencies that are not available here: {missing_text}."
            )
        unavailable.append(
            StrategyOption(
                id="ocr_rebuild",
                label=STRATEGY_LABELS["ocr_rebuild"],
                reason=ocr_reason,
            )
        )

    if math_signal and not has_formula_tags and math_runtime_available:
        applicable.append(
            StrategyOption(
                id="add_math_formula_tags",
                label=STRATEGY_LABELS["add_math_formula_tags"],
                reason=math_runtime_reason,
            )
        )
    else:
        if not math_signal:
            math_reason = "No math-related residuals were detected on the current output."
        elif has_formula_tags:
            math_reason = "Formula tags already exist on the current output. This pass only adds missing formula tags."
        else:
            math_reason = math_runtime_reason
        unavailable.append(
            StrategyOption(
                id="add_math_formula_tags",
                label=STRATEGY_LABELS["add_math_formula_tags"],
                reason=math_reason,
            )
        )

    return applicable, unavailable


def _build_unsupported_options(_job: Any, _acceptance: Any) -> list[StrategyOption]:
    """Backward-compatible hook for tests that patch legacy unsupported rows."""
    return []


def describe_strategy_picker(job: Any) -> dict[str, Any]:
    """Return picker state for the current remediated PDF."""
    state = load_strategy_state(job.id)
    current_path = _resolve_current_output_path(job, state)
    latest_output_available = bool(state.current_output_path and current_path and current_path.exists())
    latest_output_label = None

    for run in reversed(state.runs):
        if run.status == "success":
            latest_output_label = run.label
            break

    if not current_path:
        unavailable = [
            StrategyOption(
                id=strategy_id,
                label=STRATEGY_LABELS[strategy_id],
                reason="No remediated PDF is available yet.",
            )
            for strategy_id in DISPLAY_ORDER
        ]
        return {
            "running": is_strategy_locked(job.id),
            "analysis_basis": "remediated_pdf",
            "latest_output_available": False,
            "latest_output_label": latest_output_label,
            "applicable_strategies": [],
            "unavailable_strategies": [asdict(option) for option in unavailable],
            "runs": [asdict(run) for run in state.runs],
        }

    if current_path.suffix.lower() != ".pdf":
        applicable: list[StrategyOption] = []
        unavailable = [
            StrategyOption(
                id=strategy_id,
                label=STRATEGY_LABELS[strategy_id],
                reason=(
                    "Only available for PDF output."
                    if strategy_id in _PDF_ONLY_STRATEGIES
                    else "Only available for PDF output in a later post-remediation pass."
                ),
            )
            for strategy_id in DISPLAY_ORDER
            if strategy_id != "manual_review"
        ]

        # The manual-review flag is still useful even when the output is not a PDF.
        manual_review_option = StrategyOption(
            id="manual_review",
            label=STRATEGY_LABELS["manual_review"],
            reason="Residual issues still need human review.",
            selected_by_default=False,
        )
        report = getattr(job, "report_data", None) or {}
        if any(
            report.get(key)
            for key in (
                "failed_checks",
                "manual_review_checks",
                "sr_issues",
                "wcag_failures",
                "wcag_manual_reviews",
            )
        ):
            applicable.append(manual_review_option)
        else:
            unavailable.append(
                StrategyOption(
                    id="manual_review",
                    label=STRATEGY_LABELS["manual_review"],
                    reason="No residual issues were detected to flag.",
                )
            )

        return {
            "running": is_strategy_locked(job.id),
            "analysis_basis": "strategy_output" if latest_output_available else "remediated_pdf",
            "latest_output_available": latest_output_available,
            "latest_output_label": latest_output_label,
            "applicable_strategies": [asdict(option) for option in applicable],
            "unavailable_strategies": [asdict(option) for option in unavailable],
            "runs": [asdict(run) for run in state.runs],
        }

    acceptance = _evaluate_acceptance(job, current_path)
    applicable, unavailable = _build_supported_options(job, acceptance)
    extended_applicable, extended_unavailable = _build_extended_options(job, acceptance, current_path)
    applicable.extend(extended_applicable)
    unavailable.extend(extended_unavailable)
    unavailable.extend(_build_unsupported_options(job, acceptance))
    applicable = _dedupe_options(applicable)
    unavailable = _dedupe_options(unavailable)

    applicable_sorted = sorted(
        applicable,
        key=lambda option: DISPLAY_ORDER.index(option.id),
    )
    unavailable_sorted = sorted(
        unavailable,
        key=lambda option: DISPLAY_ORDER.index(option.id),
    )

    return {
        "running": is_strategy_locked(job.id),
        "analysis_basis": "strategy_output" if latest_output_available else "remediated_pdf",
        "latest_output_available": latest_output_available,
        "latest_output_label": latest_output_label,
        "applicable_strategies": [asdict(option) for option in applicable_sorted],
        "unavailable_strategies": [asdict(option) for option in unavailable_sorted],
        "runs": [asdict(run) for run in state.runs],
    }


def run_strategy_sequence(job: Any, strategy_ids: list[str]) -> dict[str, Any]:
    """Run the selected strategies sequentially against the current PDF."""
    catalog = describe_strategy_picker(job)
    applicable_ids = {
        option["id"] for option in catalog["applicable_strategies"]
    }

    if not strategy_ids:
        raise ValueError("Select at least one applicable strategy.")

    invalid = [strategy_id for strategy_id in strategy_ids if strategy_id not in applicable_ids]
    if invalid:
        raise ValueError("One or more selected strategies are no longer applicable.")

    state = load_strategy_state(job.id)
    current_path = _resolve_current_output_path(job, state)
    if current_path is None:
        raise ValueError("No remediated PDF is available yet.")

    run_dir = _strategy_dir(job.id)
    run_dir.mkdir(parents=True, exist_ok=True)

    for strategy_id in strategy_ids:
        typed_id = strategy_id  # Runtime value already validated above.
        if typed_id == "manual_review":
            run = StrategyRunRecord(
                strategy_id="manual_review",
                label=STRATEGY_LABELS["manual_review"],
                status="flagged",
                summary="Marked for manual review.",
                completed_at=time.time(),
            )
            state.runs.append(run)
            save_strategy_state(state)
            continue

        run_number = len(state.runs) + 1
        output_path = run_dir / f"{run_number:02d}_{typed_id}.pdf"

        if typed_id == "repair_cid_fonts":
            run = _run_mode_b_strategy(job, current_path, output_path)
        elif typed_id == "repair_embedded_fonts":
            run = _run_simple_font_strategy(job, current_path, output_path)
        elif typed_id == "rebuild_structure":
            run = _run_mode_a_strategy(job, current_path, output_path)
        elif typed_id == "add_math_formula_tags":
            run = _run_math_strategy(current_path, output_path)
        elif typed_id == "ocr_rebuild":
            run = _run_ocr_strategy(current_path, output_path)
        else:
            run = StrategyRunRecord(
                strategy_id=typed_id,
                label=STRATEGY_LABELS[typed_id],
                status="failed",
                summary="This strategy is not available in v1.",
                error="Unsupported strategy",
                completed_at=time.time(),
            )

        state.runs.append(run)

        if run.status == "success":
            state.current_output_path = str(output_path)
            current_path = output_path
        else:
            output_path.unlink(missing_ok=True)

        save_strategy_state(state)

        if run.status == "failed":
            break

    if current_path and current_path.exists() and current_path.suffix.lower() == ".pdf":
        _refresh_job_report(job, current_path)

    return describe_strategy_picker(job)


def _run_mode_b_strategy(job: Any, current_path: Path, output_path: Path) -> StrategyRunRecord:
    from project_remedy.faithful_rebuild.mode_b_production import run_mode_b

    started_at = time.time()
    result = run_mode_b(
        pdf_path=current_path,
        output_path=output_path,
        original_path=job.input_path,
        config=_load_manual_strategy_config(),
    )

    if result.output_valid and output_path.exists() and result.fonts_replaced > 0:
        summary = f"Repaired {result.fonts_replaced} CID font(s)."
        if result.cids_recovered > 0:
            summary = f"{summary[:-1]} and recovered {result.cids_recovered} CID mapping(s)."
        return StrategyRunRecord(
            strategy_id="repair_cid_fonts",
            label=STRATEGY_LABELS["repair_cid_fonts"],
            status="success",
            summary=summary,
            error=result.error,
            started_at=started_at,
            completed_at=time.time(),
        )

    status = "failed" if result.error else "skipped"
    summary = result.skip_reason or result.error or "CID-font repair did not produce a new output."
    return StrategyRunRecord(
        strategy_id="repair_cid_fonts",
        label=STRATEGY_LABELS["repair_cid_fonts"],
        status=status,
        summary=summary,
        error=result.error,
        started_at=started_at,
        completed_at=time.time(),
    )


def _run_simple_font_strategy(job: Any, current_path: Path, output_path: Path) -> StrategyRunRecord:
    from project_remedy.faithful_rebuild.simple_font_production import run_simple_font

    started_at = time.time()
    result = run_simple_font(
        pdf_path=current_path,
        output_path=output_path,
        original_path=job.input_path,
        config=_load_manual_strategy_config(),
    )

    mutations = result.fonts_encoding_repaired + result.fonts_replaced
    if result.output_valid and output_path.exists() and mutations > 0:
        parts: list[str] = []
        if result.fonts_encoding_repaired > 0:
            parts.append(f"repaired {result.fonts_encoding_repaired} encoding issue(s)")
        if result.fonts_replaced > 0:
            parts.append(f"repaired {result.fonts_replaced} embedded font(s)")
        summary = " and ".join(parts).capitalize() + "."
        if result.error:
            summary = f"{summary} Warning: {result.error}"
        return StrategyRunRecord(
            strategy_id="repair_embedded_fonts",
            label=STRATEGY_LABELS["repair_embedded_fonts"],
            status="success",
            summary=summary,
            error=result.error,
            started_at=started_at,
            completed_at=time.time(),
        )

    status = "failed" if result.error and mutations == 0 else "skipped"
    summary = result.skip_reason or result.error or "Embedded-font repair did not produce a new output."
    return StrategyRunRecord(
        strategy_id="repair_embedded_fonts",
        label=STRATEGY_LABELS["repair_embedded_fonts"],
        status=status,
        summary=summary,
        error=result.error,
        started_at=started_at,
        completed_at=time.time(),
    )


def _reapply_accessibility_fixes_on_rebuilt(
    rebuilt_path: Path,
    original_path: Path,
    verification_mode: str,
) -> tuple[int, int, str | None]:
    """Re-run fix_and_verify against a rebuilt PDF so the clean structure
    keeps headings / figure alt / language tagging that the structural
    rebuild strips.

    REMEDY-69 #1 + #2: without this, the rebuild produces a PDF with a
    pristine `/Document/Sect → /P /P …` tree but no heading tags and no
    alt text, so the refreshed ACR is worse than the pre-rebuild one —
    the user has to re-upload the rebuilt file to recover them.

    Returns (fixes_applied, fixes_skipped, error_message_or_none).
    The rebuilt PDF is overwritten in-place on success. On any error,
    the rebuilt file is left untouched so the caller still has a valid
    structural rebuild to fall back on.
    """
    from project_remedy.pdf_fixer import fix_and_verify

    vision_provider = None
    try:
        vision_provider = _build_math_vision_provider()
    except Exception as exc:
        logger_msg = f"vision provider unavailable during rebuild re-remediation: {exc}"
        return 0, 0, logger_msg

    try:
        report = fix_and_verify(
            pdf_path=rebuilt_path,
            output_path=rebuilt_path,
            thorough=(verification_mode == "full"),
            max_cycles=3,
            original_path=original_path,
            vision_provider_override=vision_provider,
        )
        return int(report.fixed_count), int(report.skipped_count), None
    except Exception as exc:
        return 0, 0, f"re-remediation after rebuild failed: {exc}"
    finally:
        if vision_provider is not None:
            _close_vision_provider(vision_provider)


def _run_mode_a_strategy(job: Any, current_path: Path, output_path: Path) -> StrategyRunRecord:
    from project_remedy.faithful_rebuild.mode_a_production import run_mode_a

    started_at = time.time()
    if _has_heading_signal(job):
        heading_run = _run_heading_structure_strategy(
            current_path,
            output_path,
            started_at=started_at,
        )
        if heading_run.status == "success":
            return heading_run

    result = run_mode_a(
        pdf_path=current_path,
        output_path=output_path,
        original_path=job.input_path,
        config=_load_manual_strategy_config(),
    )

    if result.output_valid and output_path.exists():
        # Re-apply accessibility fixes on the rebuilt (clean-structure) PDF
        # so the user's download + ACR reflect both the structural rebuild
        # AND heading synthesis / alt text / metadata fixes.
        fixed, _skipped, re_remediation_error = _reapply_accessibility_fixes_on_rebuilt(
            rebuilt_path=output_path,
            original_path=job.input_path,
            verification_mode=getattr(job, "verification_mode", "sampled"),
        )

        summary_parts = ["Rebuilt document structure"]
        if result.visual_diff_score is not None:
            summary_parts.append(
                f"{(result.visual_diff_score * 100):.2f}% visual drift"
            )
        if re_remediation_error is None and fixed > 0:
            summary_parts.append(f"re-applied {fixed} accessibility fix(es)")
        elif re_remediation_error is not None:
            summary_parts.append("accessibility re-remediation skipped (see error)")
        summary = ", ".join(summary_parts) + "."

        return StrategyRunRecord(
            strategy_id="rebuild_structure",
            label=STRATEGY_LABELS["rebuild_structure"],
            status="success",
            summary=summary,
            error=result.error or re_remediation_error,
            started_at=started_at,
            completed_at=time.time(),
        )

    status = "failed" if result.error else "skipped"
    summary = result.skip_reason or result.error or "Structure rebuild did not produce a new output."
    return StrategyRunRecord(
        strategy_id="rebuild_structure",
        label=STRATEGY_LABELS["rebuild_structure"],
        status=status,
        summary=summary,
        error=result.error,
        started_at=started_at,
        completed_at=time.time(),
    )


def _run_heading_structure_strategy(
    current_path: Path,
    output_path: Path,
    *,
    started_at: float | None = None,
) -> StrategyRunRecord:
    """Create heading navigation on PDFs whose only structure residual is headings."""
    import pikepdf

    from project_remedy.pdf_fixer import (
        _save_remediated_pdf,
        fix_heading_nesting,
        fix_heading_synthesis,
    )
    from project_remedy.tag_tree_reader import validate_tag_tree

    started_at = started_at or time.time()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_path, output_path)
        with pikepdf.open(output_path, allow_overwriting_input=True) as pdf:
            changes = fix_heading_nesting(pdf)
            if not changes:
                changes = fix_heading_synthesis(pdf, vision_provider=None)
            if not changes:
                return StrategyRunRecord(
                    strategy_id="rebuild_structure",
                    label=STRATEGY_LABELS["rebuild_structure"],
                    status="skipped",
                    summary="Heading structure repair did not find a title-like text block.",
                    started_at=started_at,
                    completed_at=time.time(),
                )
            _save_remediated_pdf(pdf, output_path)

        heading_issues = [
            issue
            for issue in validate_tag_tree(output_path).issues
            if issue.rule_id in {"sr-no-headings", "sr-heading-start", "sr-heading-skip"}
        ]
        if heading_issues:
            output_path.unlink(missing_ok=True)
            return StrategyRunRecord(
                strategy_id="rebuild_structure",
                label=STRATEGY_LABELS["rebuild_structure"],
                status="failed",
                summary="Heading structure repair did not clear heading validation.",
                error="; ".join(issue.description for issue in heading_issues[:3]),
                started_at=started_at,
                completed_at=time.time(),
            )

        return StrategyRunRecord(
            strategy_id="rebuild_structure",
            label=STRATEGY_LABELS["rebuild_structure"],
            status="success",
            summary=", ".join(changes) + ".",
            started_at=started_at,
            completed_at=time.time(),
        )
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        return StrategyRunRecord(
            strategy_id="rebuild_structure",
            label=STRATEGY_LABELS["rebuild_structure"],
            status="failed",
            summary="Heading structure repair failed.",
            error=str(exc),
            started_at=started_at,
            completed_at=time.time(),
        )


def _run_math_strategy(current_path: Path, output_path: Path) -> StrategyRunRecord:
    from project_remedy.math.extractor import extract_formulas
    from project_remedy.math.tagger import tag_formulas

    started_at = time.time()
    provider = None

    try:
        provider = _build_math_vision_provider()
        if provider is None:
            return StrategyRunRecord(
                strategy_id="add_math_formula_tags",
                label=STRATEGY_LABELS["add_math_formula_tags"],
                status="failed",
                summary="Math formula tagging could not start because no vision provider is available.",
                error="No vision provider available",
                started_at=started_at,
                completed_at=time.time(),
            )

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            extraction = extract_formulas(current_path, provider)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        if not extraction.formulas:
            status = "failed" if extraction.errors else "skipped"
            summary = (
                extraction.errors[0]
                if extraction.errors
                else "No formulas were detected to tag on the current output."
            )
            return StrategyRunRecord(
                strategy_id="add_math_formula_tags",
                label=STRATEGY_LABELS["add_math_formula_tags"],
                status=status,
                summary=summary,
                error=extraction.errors[0] if extraction.errors else None,
                started_at=started_at,
                completed_at=time.time(),
            )

        changes = tag_formulas(current_path, extraction.formulas, output_path=output_path)
        if changes and output_path.exists():
            summary = f"Tagged {len(changes)} math formula(s)."
            if extraction.errors:
                summary = f"{summary} Detection reported {len(extraction.errors)} page warning(s)."
            return StrategyRunRecord(
                strategy_id="add_math_formula_tags",
                label=STRATEGY_LABELS["add_math_formula_tags"],
                status="success",
                summary=summary,
                error=extraction.errors[0] if extraction.errors else None,
                started_at=started_at,
                completed_at=time.time(),
            )

        return StrategyRunRecord(
            strategy_id="add_math_formula_tags",
            label=STRATEGY_LABELS["add_math_formula_tags"],
            status="skipped",
            summary="Math formula tagging did not produce a new output.",
            started_at=started_at,
            completed_at=time.time(),
        )
    except Exception as exc:
        return StrategyRunRecord(
            strategy_id="add_math_formula_tags",
            label=STRATEGY_LABELS["add_math_formula_tags"],
            status="failed",
            summary=str(exc),
            error=str(exc),
            started_at=started_at,
            completed_at=time.time(),
        )
    finally:
        if provider is not None:
            _close_vision_provider(provider)


def _run_ocr_strategy(current_path: Path, output_path: Path) -> StrategyRunRecord:
    from project_remedy.pdf_fixer import fix_all

    started_at = time.time()

    try:
        report = fix_all(
            current_path,
            output_path=output_path,
            only="page-char-encoding",
        )
    except Exception as exc:
        return StrategyRunRecord(
            strategy_id="ocr_rebuild",
            label=STRATEGY_LABELS["ocr_rebuild"],
            status="failed",
            summary=str(exc),
            error=str(exc),
            started_at=started_at,
            completed_at=time.time(),
        )

    rebuild_changes = [
        change
        for change in report.changes
        if change.startswith("Rebuilt searchable text layer with Tesseract OCR")
        or change.startswith("Rebuilt image-only PDF with Tesseract OCR")
    ]
    if rebuild_changes and output_path.exists():
        return StrategyRunRecord(
            strategy_id="ocr_rebuild",
            label=STRATEGY_LABELS["ocr_rebuild"],
            status="success",
            summary=rebuild_changes[0],
            started_at=started_at,
            completed_at=time.time(),
        )

    error = report.skipped[0] if report.skipped else None
    failure_tokens = ("error", "failed", "not found", "unavailable")
    status = "failed" if error and any(token in error.lower() for token in failure_tokens) else "skipped"
    summary = error or (report.changes[0] if report.changes else "OCR rebuild did not produce a new output.")
    return StrategyRunRecord(
        strategy_id="ocr_rebuild",
        label=STRATEGY_LABELS["ocr_rebuild"],
        status=status,
        summary=summary,
        error=error,
        started_at=started_at,
        completed_at=time.time(),
    )
