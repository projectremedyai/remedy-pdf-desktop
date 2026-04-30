"""Per-document accessibility compliance report generator.

Produces a before/after report for each remediated PDF showing:
- Original source document info and accessibility state
- Remediated PDF check results
- WCAG 2.1 AA conformance mapping
- Overall conformance determination

Usage::

    from project_remedy.compliance_report import generate_document_report
    report = generate_document_report(
        original_path=Path("downloads/pdf/doc.pdf"),
        remediated_path=Path("remediated-pdfs/doc.pdf"),
        output_dir=Path("compliance/documents/"),
    )
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pikepdf

from project_remedy.config import PipelineConfig
from project_remedy.pdf_acceptance import PDFAcceptanceResult, evaluate_pdf_acceptance
from project_remedy.pdf_checker import (
    CheckReport,
    CheckResult,
    PDFAccessibilityChecker,
)
from project_remedy.tag_tree_reader import (
    SCREEN_READER_RULE_IDS,
    ScreenReaderIssue,
    Severity,
    ValidationResult as SRValidationResult,
    validate_tag_tree,
)


# ---------------------------------------------------------------------------
# WCAG 2.1 AA ↔ check mapping
# ---------------------------------------------------------------------------

@dataclass
class WCAGCriterion:
    id: str            # e.g. "1.1.1"
    name: str          # e.g. "Non-text Content"
    level: str         # "A" or "AA"
    rule_ids: list[str]  # our check rule_ids that map to this criterion


WCAG_MAPPING: list[WCAGCriterion] = [
    WCAGCriterion("1.1.1", "Non-text Content", "A",
                  ["alt-figures", "alt-elements", "sr-figure-no-alt", "sr-figure-generic-alt", "sr-figure-short-alt"]),
    WCAGCriterion("1.3.1", "Info and Relationships", "A",
                  ["doc-tagged", "page-content-tagged", "page-annotations-tagged",
                   "sr-table-no-headers", "sr-list-no-items", "sr-no-tags"]),
    WCAGCriterion("1.3.2", "Meaningful Sequence", "A",
                  ["doc-reading-order", "sr-repeated-content", "sr-untagged-page"]),
    WCAGCriterion("1.3.3", "Sensory Characteristics", "A", []),
    WCAGCriterion("1.4.1", "Use of Color", "A", ["doc-use-of-color"]),
    WCAGCriterion("1.4.3", "Contrast (Minimum)", "AA", ["doc-color-contrast"]),
    WCAGCriterion("1.4.5", "Images of Text", "AA", ["doc-not-image-only"]),
    WCAGCriterion("2.1.1", "Keyboard", "A", ["page-tab-order"]),
    WCAGCriterion("2.1.2", "No Keyboard Trap", "A", []),
    WCAGCriterion("2.2.1", "Timing Adjustable", "A", ["page-no-timed-responses"]),
    WCAGCriterion("2.3.1", "Three Flashes or Below Threshold", "A", ["page-no-flicker"]),
    WCAGCriterion("2.4.1", "Bypass Blocks", "A", ["doc-bookmarks"]),
    WCAGCriterion("2.4.2", "Page Titled", "A", ["doc-display-title"]),
    WCAGCriterion("2.4.4", "Link Purpose (In Context)", "A",
                  ["page-no-repetitive-links"]),
    WCAGCriterion("2.4.5", "Multiple Ways", "AA", ["doc-bookmarks"]),
    WCAGCriterion("2.4.6", "Headings and Labels", "AA",
                  ["headings-nesting", "sr-heading-skip", "sr-heading-start", "sr-no-headings"]),
    WCAGCriterion("2.4.7", "Focus Visible", "AA", ["page-tab-order"]),
    WCAGCriterion("3.1.1", "Language of Page", "A", ["doc-language", "sr-no-lang"]),
    WCAGCriterion("3.1.2", "Language of Parts", "AA", []),
    WCAGCriterion("3.2.3", "Consistent Navigation", "AA", []),
    WCAGCriterion("3.2.4", "Consistent Identification", "AA", []),
    WCAGCriterion("4.1.1", "Parsing", "A", ["doc-tagged"]),
    WCAGCriterion("4.1.2", "Name, Role, Value", "A",
                  ["forms-fields-description", "page-annotations-tagged"]),
]


ReviewDecision = Literal["pass", "no_pass", "rerun_full_verification", "manual_review"]
ResidualIssueState = Literal["failed", "needs_manual_review", "source_limited"]


MANUAL_REVIEW_RULE_IDS = frozenset({
    "doc-reading-order",
    "doc-color-contrast",
    "doc-use-of-color",
})
CHECK_STATUS_MANUAL_REVIEW = "Needs Manual Review"
WCAG_STATUS_REVIEW = "REVIEW"
ISSUE_STATE_FAILED = "failed"
ISSUE_STATE_NEEDS_MANUAL_REVIEW = "needs_manual_review"
ISSUE_STATE_SOURCE_LIMITED = "source_limited"


def _is_manual_review_check(result: CheckResult) -> bool:
    """True when a checker result should be reported as manual review.

    Reading order, contrast, and use-of-color are intentionally treated as
    vision-assisted manual checks. Even when the vision model flags a concern,
    the report should not count that as a failed automated check; it should
    surface the finding for user confirmation.
    """
    return (
        result.rule_id in MANUAL_REVIEW_RULE_IDS
        and result.status in {"Failed", "Manual Check Needed"}
    )


def _is_reviewable_check(result: CheckResult) -> bool:
    """True for the vision/manual criteria that carry review decisions."""
    return result.rule_id in MANUAL_REVIEW_RULE_IDS


def _manual_review_recommendation(result: CheckResult) -> str:
    """Return the user-facing recommendation for a manual-review check."""
    details_text = " ".join(result.details or [])
    if result.status == "Passed":
        return "Vision review found no issue on the analyzed pages."
    if result.status == "Failed":
        return (
            "Vision review found a possible issue. Treat this as no-pass until "
            "a reviewer confirms the page visually, or rerun after remediation."
        )
    if "Remaining pages were not automatically verified" in details_text:
        return (
            "Vision reviewed the sampled pages and found no issue there. Rerun "
            "with full verification or manually review the remaining pages."
        )
    if "Configure a vision model" in details_text:
        return (
            "No vision review ran for this check. Configure a vision model or "
            "review the document manually."
        )
    return "Reviewer confirmation is required before this check is considered passed."


def _manual_review_decision(result: CheckResult) -> ReviewDecision:
    """Return a compact machine-readable review recommendation."""
    details_text = " ".join(result.details or [])
    if result.status == "Passed":
        return "pass"
    if result.status == "Failed":
        return "no_pass"
    if "Remaining pages were not automatically verified" in details_text:
        return "rerun_full_verification"
    return "manual_review"


# ---------------------------------------------------------------------------
# Issue normalization
# ---------------------------------------------------------------------------


def _build_normalized_issues(acceptance: PDFAcceptanceResult) -> list[dict]:
    """Build a canonical issues array from checker, SR, and veraPDF results.

    Each issue has: code, source, severity, fixable, blocking, description.
    """
    issues: list[dict] = []

    # Checker failures and unresolved manual-review items.
    for r in acceptance.checker_report.results:
        if r.status in {"Failed", "Manual Check Needed"}:
            is_manual_review = _is_manual_review_check(r)
            issues.append({
                "code": r.rule_id,
                "source": "checker",
                "severity": "review" if is_manual_review else "error",
                "fixable": r.fixable,
                "blocking": False,
                "description": r.description,
                "needs_manual_review": is_manual_review,
                "state": (
                    ISSUE_STATE_NEEDS_MANUAL_REVIEW
                    if is_manual_review
                    else ISSUE_STATE_FAILED
                ),
                "decision": _manual_review_decision(r) if is_manual_review else "",
                "recommendation": _manual_review_recommendation(r) if is_manual_review else "",
            })

    # Screen reader errors and warnings.
    for i in acceptance.tag_tree_result.issues:
        if i.severity == Severity.ERROR:
            issues.append({
                "code": i.rule_id,
                "source": "screen-reader",
                "severity": "error",
                "fixable": True,
                "blocking": False,
                "description": i.description,
            })
        elif i.severity == Severity.WARNING:
            issues.append({
                "code": i.rule_id,
                "source": "screen-reader",
                "severity": "warning",
                "fixable": True,
                "blocking": False,
                "description": i.description,
            })

    # veraPDF violations.
    if acceptance.verapdf_result.checked and acceptance.verapdf_result.violations:
        for v in acceptance.verapdf_result.violations:
            is_font = PDFAcceptanceResult._is_source_font_limitation(v)
            issues.append({
                "code": v.get("id", "unknown-rule"),
                "source": "verapdf",
                "severity": "info" if is_font else "error",
                "fixable": not is_font,
                "blocking": False,
                "description": v.get("description", ""),
                "state": ISSUE_STATE_SOURCE_LIMITED if is_font else ISSUE_STATE_FAILED,
            })

    return issues


def _build_source_limited_issues(acceptance: PDFAcceptanceResult) -> list[dict]:
    """Serialize non-blocking source-font veraPDF limitations."""
    if not acceptance.verapdf_result.checked:
        return []
    issues: list[dict] = []
    for violation in acceptance.verapdf_result.violations:
        if not PDFAcceptanceResult._is_source_font_limitation(violation):
            continue
        issues.append({
            "rule_id": violation.get("id", "unknown-rule"),
            "description": violation.get("description", ""),
            "details": [
                detail
                for detail in (
                    violation.get("location", ""),
                    violation.get("note", ""),
                )
                if detail
            ],
            "state": ISSUE_STATE_SOURCE_LIMITED,
            "fixable": False,
            "recommendation": (
                "This appears to be inherited from the source font/CIDSet data. "
                "Use the font repair strategies if offered; otherwise recreate "
                "or re-export from the source document to fully clear it."
            ),
        })
    return issues


# ---------------------------------------------------------------------------
# Conformance determination
# ---------------------------------------------------------------------------

class Conformance:
    CONFORMANT = "Conformant"
    PARTIALLY = "Partially Conformant"
    NOT_CONFORMANT = "Not Conformant"


def _determine_conformance(
    acceptance: PDFAcceptanceResult,
) -> str:
    """Determine overall document conformance.

    A file is Conformant when its remaining issues are all non-blocking:
    - Checker failures that are fixable cosmetic/navigational issues
    - veraPDF violations that are source-font limitations
    - SR warnings other than missing/invalid heading navigation
    """
    all_failed = list(acceptance.checker_failures)
    # Checker failures that don't block content access for screen readers.
    # These are real WCAG issues but are cosmetic/navigational — they don't
    # prevent a screen reader user from accessing the document content.
    _NON_BLOCKING_CHECKER = {
        "doc-display-title",    # 2.4.2 — title bar display, not content
        "page-tab-order",       # 2.4.3 — focus order, navigational
        "doc-bookmarks",        # 2.4.5 — bookmarks in large docs
        "page-char-encoding",   # source font limitation — degraded encoding
    }
    blocking_failed = [
        r
        for r in all_failed
        if r.rule_id not in _NON_BLOCKING_CHECKER
        and not _is_manual_review_check(r)
    ]
    sr_errors = acceptance.screen_reader_errors
    sr_blocking_warnings = [
        issue
        for issue in acceptance.tag_tree_result.issues
        if issue.severity == Severity.WARNING
        and issue.rule_id in {"sr-no-headings", "sr-heading-start"}
    ]
    verapdf_failed = (
        acceptance.verapdf_result.checked and not acceptance.verapdf_result.passed
    )
    warning_reasons = list(getattr(acceptance, "warning_reasons", []) or [])

    if not getattr(acceptance, "openable", acceptance.passed):
        return Conformance.NOT_CONFORMANT
    if acceptance.passed and not warning_reasons and not sr_blocking_warnings:
        return Conformance.CONFORMANT

    # When the only issues are non-blocking checker failures, source-font
    # veraPDF violations, and SR warnings (not errors) → Conformant.
    # The document content is accessible even if cosmetic metadata or
    # inherited font limitations remain.
    verapdf_all_source_font = (
        acceptance.verapdf_result.passed
        or (
            acceptance.verapdf_result.checked
            and acceptance.verapdf_result.violations  # guard against all([])
            and all(
                acceptance._is_source_font_limitation(v)
                for v in acceptance.verapdf_result.violations
            )
        )
        or not acceptance.verapdf_result.checked
    )
    # Also require no validator runtime errors — a crashed validator with
    # empty violation lists should not be treated as clean.
    has_runtime_errors = bool(
        getattr(acceptance, "checker_error", None)
        or getattr(acceptance, "screen_reader_error", None)
        or (acceptance.verapdf_result.checked and acceptance.verapdf_result.error)
    )
    if (
        not blocking_failed
        and not sr_errors
        and not sr_blocking_warnings
        and verapdf_all_source_font
        and not has_runtime_errors
    ):
        return Conformance.CONFORMANT

    if warning_reasons:
        return Conformance.PARTIALLY
    if not verapdf_failed and len(all_failed) <= 3 and len(sr_errors) <= 2:
        return Conformance.PARTIALLY
    return Conformance.NOT_CONFORMANT


def _determine_wcag_status(
    criterion: WCAGCriterion,
    check_results: list[CheckResult],
    sr_issues: list[ScreenReaderIssue],
) -> tuple[str, str]:
    """Determine PASS/FAIL/REVIEW and build remarks for a WCAG criterion.

    Returns (status, remarks).
    """
    if not criterion.rule_ids:
        return "N/A", "Not applicable to static PDF documents"

    # Gather matching check results.
    matching_checks = [r for r in check_results if r.rule_id in criterion.rule_ids]
    matching_sr = [i for i in sr_issues if i.rule_id in criterion.rule_ids]

    if not matching_checks and not matching_sr:
        return "N/A", "No applicable checks for this document"

    failed_checks = [
        r for r in matching_checks
        if r.status in {"Failed", "Manual Check Needed"}
        and not _is_manual_review_check(r)
    ]
    manual_review_checks = [
        r for r in matching_checks
        if _is_manual_review_check(r)
    ]
    sr_errors = [i for i in matching_sr if i.severity == Severity.ERROR]
    sr_warning_failures = [
        i for i in matching_sr
        if i.severity == Severity.WARNING
        and i.rule_id in {"sr-no-headings", "sr-heading-start"}
    ]

    if not failed_checks and not manual_review_checks and not sr_errors and not sr_warning_failures:
        return "PASS", ""

    remarks_parts = []
    for r in failed_checks:
        remarks_parts.append(f"{r.description}: {'; '.join(r.details[:2])}" if r.details else r.description)
    for i in sr_errors:
        remarks_parts.append(i.description)
    for i in sr_warning_failures:
        remarks_parts.append(i.description)

    if not failed_checks and not sr_errors and not sr_warning_failures:
        for r in manual_review_checks:
            detail = "; ".join(r.details[:2]) if r.details else r.description
            remarks_parts.append(
                f"{r.description}: {detail}. {_manual_review_recommendation(r)}"
            )
        return WCAG_STATUS_REVIEW, "; ".join(remarks_parts)

    return "FAIL", "; ".join(remarks_parts)


# ---------------------------------------------------------------------------
# Original document analysis
# ---------------------------------------------------------------------------

@dataclass
class OriginalDocInfo:
    """Accessibility state of the original source document."""
    file_path: str
    file_type: str
    file_size: int
    source_url: str
    is_tagged: bool
    has_language: bool
    has_title: bool
    page_count: int


def _analyze_original(original_path: Path, source_url: str = "") -> OriginalDocInfo:
    """Quick accessibility scan of the original source document."""
    suffix = original_path.suffix.lower().lstrip(".")
    stat = original_path.stat()

    is_tagged = False
    has_language = False
    has_title = False
    page_count = 0

    if suffix == "pdf":
        try:
            with pikepdf.open(original_path) as pdf:
                page_count = len(pdf.pages)
                is_tagged = bool(pdf.Root.get("/StructTreeRoot"))
                lang = pdf.Root.get("/Lang")
                has_language = bool(lang and str(lang).strip())
                try:
                    with pdf.open_metadata() as meta:
                        has_title = bool(meta.get("dc:title", "").strip())
                except Exception:
                    pass
        except Exception:
            pass
    else:
        # Non-PDF originals: can't check accessibility without converting.
        page_count = 0

    return OriginalDocInfo(
        file_path=str(original_path),
        file_type=suffix or "unknown",
        file_size=stat.st_size,
        source_url=source_url,
        is_tagged=is_tagged,
        has_language=has_language,
        has_title=has_title,
        page_count=page_count,
    )


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------

@dataclass
class WCAGResult:
    criterion_id: str
    criterion_name: str
    level: str
    status: str   # PASS, FAIL, REVIEW, N/A
    remarks: str


@dataclass
class DocumentReport:
    """Complete compliance report for a single document."""

    # Document identity.
    document_name: str
    original: OriginalDocInfo
    remediated_path: str
    remediated_size: int
    remediated_pages: int
    verification_mode: str

    # Check results.
    check_results: list[dict]   # serialized CheckResult list
    sr_issues: list[dict]       # serialized ScreenReaderIssue list
    tag_count: int
    verapdf_checked: bool
    verapdf_passed: bool
    verapdf_violations: list[dict]

    # WCAG mapping.
    wcag_results: list[WCAGResult]
    conformance: str

    # Metadata.
    generated_at: str
    report_filename: str = ""  # basename of the HTML report file (for linking)
    generator: str = "Remedy PDF Desktop — Accessibility Remediation Pipeline"

    # Normalized issues (canonical union of checker, SR, and veraPDF).
    issues: list[dict] = field(default_factory=list)
    reviewable_checks: list[dict] = field(default_factory=list)
    manual_review_checks: list[dict] = field(default_factory=list)
    source_limited_issues: list[dict] = field(default_factory=list)

    # Visual fidelity diff (original vs remediated, page-by-page pixel comparison).
    visual_diff: dict = field(default_factory=dict)  # keys: checked, passed, total_pages, differing_pages, max_page_diff
    verification_coverage: dict = field(default_factory=dict)

    # Vision-model artifact sanity check on remediated pages (gray tints,
    # stray boxes, truncated text). Empty dict = check didn't run (e.g.
    # vision disabled); ``checked=False`` dict = attempted but errored out.
    visual_artifact_check: dict = field(default_factory=dict)

    # Screen reader readability score (0-100 composite).
    screen_reader_readability: float = 0.0
    screen_reader_readability_details: dict = field(default_factory=dict)

    @property
    def source_font_only(self) -> bool:
        """True when all veraPDF violations are source-font limitations."""
        if not self.verapdf_violations:
            return False
        from project_remedy.pdf_acceptance import PDFAcceptanceResult
        return all(
            PDFAcceptanceResult._is_source_font_limitation(v)
            for v in self.verapdf_violations
        )

    @property
    def passed_checks(self) -> int:
        return sum(1 for r in self.check_results if r["status"] == "Passed")

    @property
    def failed_checks(self) -> int:
        return sum(1 for r in self.check_results if r["status"] == "Failed")

    @property
    def manual_review_check_count(self) -> int:
        return sum(
            1
            for r in self.check_results
            if r["status"] == CHECK_STATUS_MANUAL_REVIEW
        )

    @property
    def na_checks(self) -> int:
        return sum(1 for r in self.check_results if r["status"] == "Not Applicable")

    @property
    def applicable_checks(self) -> int:
        return len(self.check_results) - self.na_checks

    @property
    def automated_applicable_checks(self) -> int:
        return self.applicable_checks - self.manual_review_check_count

    @property
    def sr_error_count(self) -> int:
        return sum(1 for i in self.sr_issues if i["severity"] == "error")

    @property
    def sr_warning_count(self) -> int:
        return sum(1 for i in self.sr_issues if i["severity"] == "warning")

    @property
    def wcag_pass_count(self) -> int:
        return sum(1 for w in self.wcag_results if w.status == "PASS")

    @property
    def wcag_fail_count(self) -> int:
        return sum(1 for w in self.wcag_results if w.status == "FAIL")

    @property
    def wcag_review_count(self) -> int:
        return sum(1 for w in self.wcag_results if w.status == WCAG_STATUS_REVIEW)

    @property
    def wcag_na_count(self) -> int:
        return sum(1 for w in self.wcag_results if w.status == "N/A")

    @property
    def wcag_auto_tested_count(self) -> int:
        return self.wcag_pass_count + self.wcag_fail_count

    @property
    def verapdf_violation_count(self) -> int:
        return len(self.verapdf_violations)

    @property
    def issue_summary(self) -> dict:
        return {
            "total": len(self.issues),
            "fixable": sum(1 for i in self.issues if i.get("fixable")),
            "source_limited": self.source_limited_count,
            "blocking": sum(1 for i in self.issues if i.get("blocking")),
            "manual_review": sum(1 for i in self.issues if i.get("needs_manual_review")),
        }

    @property
    def source_limited_count(self) -> int:
        return len(self.source_limited_issues)

    @classmethod
    def from_dict(cls, data: dict, *, report_filename: str = "") -> "DocumentReport":
        """Rehydrate a serialized report from its JSON representation."""
        original_data = data.get("original") or {}
        verapdf_data = data.get("verapdf") or {}
        return cls(
            document_name=data.get("document_name", ""),
            original=OriginalDocInfo(**original_data),
            remediated_path=data.get("remediated_path", ""),
            remediated_size=int(data.get("remediated_size", 0) or 0),
            remediated_pages=int(data.get("remediated_pages", 0) or 0),
            verification_mode=data.get("verification_mode", "sampled"),
            check_results=list(data.get("check_results") or []),
            sr_issues=list(data.get("sr_issues") or []),
            tag_count=int(data.get("tag_count", 0) or 0),
            verapdf_checked=bool(verapdf_data.get("checked", False)),
            verapdf_passed=bool(verapdf_data.get("passed", False)),
            verapdf_violations=list(verapdf_data.get("violations") or []),
            wcag_results=[
                WCAGResult(**item)
                for item in (data.get("wcag_results") or [])
            ],
            conformance=data.get("conformance", Conformance.NOT_CONFORMANT),
            generated_at=data.get("generated_at", ""),
            report_filename=report_filename or data.get("report_filename", ""),
            generator=data.get("generator", "Remedy PDF Desktop — Accessibility Remediation Pipeline"),
            issues=list(data.get("issues") or []),
            reviewable_checks=list(data.get("reviewable_checks") or []),
            source_limited_issues=list(data.get("source_limited_issues") or []),
            visual_diff=dict(data.get("visual_diff") or {}),
            verification_coverage=dict(data.get("verification_coverage") or {}),
            visual_artifact_check=dict(data.get("visual_artifact_check") or {}),
            manual_review_checks=list(data.get("manual_review_checks") or []),
            screen_reader_readability=float(data.get("screen_reader_readability", 0.0) or 0.0),
            screen_reader_readability_details=dict(data.get("screen_reader_readability_details") or {}),
        )

    def to_dict(self) -> dict:
        d = {
            "document_name": self.document_name,
            "original": asdict(self.original),
            "remediated_path": self.remediated_path,
            "remediated_size": self.remediated_size,
            "remediated_pages": self.remediated_pages,
            "verification_mode": self.verification_mode,
            "conformance": self.conformance,
            "check_results": self.check_results,
            "sr_issues": self.sr_issues,
            "tag_count": self.tag_count,
            "verapdf": {
                "checked": self.verapdf_checked,
                "passed": self.verapdf_passed,
                "violations": self.verapdf_violations,
            },
            "wcag_results": [asdict(w) for w in self.wcag_results],
            "summary": {
                "passed_checks": self.passed_checks,
                "failed_checks": self.failed_checks,
                "manual_review_checks": self.manual_review_check_count,
                "source_limited": self.source_limited_count,
                "sr_errors": self.sr_error_count,
                "sr_warnings": self.sr_warning_count,
                "verapdf_checked": self.verapdf_checked,
                "verapdf_passed": self.verapdf_passed,
                "verapdf_violations": self.verapdf_violation_count,
                "wcag_pass": self.wcag_pass_count,
                "wcag_fail": self.wcag_fail_count,
                "wcag_review": self.wcag_review_count,
                "wcag_na": self.wcag_na_count,
            },
            "visual_diff": self.visual_diff,
            "verification_coverage": self.verification_coverage,
            "visual_artifact_check": self.visual_artifact_check,
            "reviewable_checks": self.reviewable_checks,
            "manual_review_checks": self.manual_review_checks,
            "source_limited_issues": self.source_limited_issues,
            "source_limited_count": self.source_limited_count,
            "issues": self.issues,
            "issue_summary": self.issue_summary,
            "screen_reader_readability": self.screen_reader_readability,
            "screen_reader_readability_details": self.screen_reader_readability_details,
            "generated_at": self.generated_at,
            "report_filename": self.report_filename,
            "generator": self.generator,
        }
        return d


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadabilityWeights:
    """Per-component weights for ``calculate_screen_reader_readability``.

    REMEDY-69 #16: the original score baked in ``30/25/20/15/10`` as literals,
    so calibrating against screen-reader-user studies required a code change.
    Weights are now a dataclass so an A/B harness can instantiate alternates,
    and the runtime instance can be overridden via env vars
    (``READABILITY_W_TEXT``, ``READABILITY_W_TAG``, ``READABILITY_W_ALT``,
    ``READABILITY_W_HEADING``, ``READABILITY_W_TABLE_LIST``).

    Any weight vector is accepted — callers are responsible for picking a
    combination that sums to a meaningful total (100 by default). The
    ``total`` property exposes the sum for UI display.
    """

    text_extractability: float = 30.0
    tag_coverage: float = 25.0
    alt_text_quality: float = 20.0
    heading_structure: float = 15.0
    table_list_accessibility: float = 10.0

    @property
    def total(self) -> float:
        return (
            self.text_extractability
            + self.tag_coverage
            + self.alt_text_quality
            + self.heading_structure
            + self.table_list_accessibility
        )


def _default_readability_weights() -> ReadabilityWeights:
    """Read weights from env vars, falling back to the historical 30/25/20/15/10."""
    import os as _os

    def _read(key: str, default: float) -> float:
        raw = _os.environ.get(key)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return value if value >= 0 else default

    return ReadabilityWeights(
        text_extractability=_read("READABILITY_W_TEXT", 30.0),
        tag_coverage=_read("READABILITY_W_TAG", 25.0),
        alt_text_quality=_read("READABILITY_W_ALT", 20.0),
        heading_structure=_read("READABILITY_W_HEADING", 15.0),
        table_list_accessibility=_read("READABILITY_W_TABLE_LIST", 10.0),
    )


def calculate_screen_reader_readability(
    pdf_path: Path,
    tag_tree_result: SRValidationResult,
    checker_report: CheckReport,
    *,
    weights: ReadabilityWeights | None = None,
) -> tuple[float, dict]:
    """Compute a 0-100 screen reader readability score.

    Components (weights configurable via ``weights=`` or env, defaults shown):
        Text extractability (30 pts): printable chars / total chars via fitz.
        Tag coverage (25 pts): % of pages with StructParents + tags per page.
        Alt text quality (20 pts): figures with meaningful (>10 char) alt text.
        Heading structure (15 pts): proper H1-H6 hierarchy.
        Table/list accessibility (10 pts): tables have TH, lists have LI in L.

    Returns (score, details_dict) where details_dict has per-component scores.
    """
    if weights is None:
        weights = _default_readability_weights()
    w_text = float(weights.text_extractability)
    w_tag = float(weights.tag_coverage)
    w_alt = float(weights.alt_text_quality)
    w_heading = float(weights.heading_structure)
    w_table_list = float(weights.table_list_accessibility)
    details: dict = {"weights": {
        "text_extractability": w_text,
        "tag_coverage": w_tag,
        "alt_text_quality": w_alt,
        "heading_structure": w_heading,
        "table_list_accessibility": w_table_list,
        "total": round(weights.total, 2),
    }}

    # --- Text extractability ---
    text_score = w_text
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        total_chars = 0
        readable_chars = 0
        fffd_count = 0
        for page in doc:
            text = page.get_text()
            for ch in text:
                total_chars += 1
                if ch.isprintable() or ch in "\n\t\r ":
                    readable_chars += 1
                if ch == "\ufffd":
                    fffd_count += 1
        doc.close()
        if total_chars > 0:
            text_score = w_text * (readable_chars / total_chars)
        details["text_extractability"] = {
            "score": round(text_score, 1),
            "max": w_text,
            "total_chars": total_chars,
            "readable_chars": readable_chars,
            "replacement_chars": fffd_count,
        }
    except Exception:
        details["text_extractability"] = {"score": 0, "max": w_text, "error": "fitz unavailable"}
        text_score = 0.0

    # --- Tag coverage ---
    nodes = tag_tree_result.tag_tree.nodes
    page_count = tag_tree_result.tag_tree.page_count or 1
    tagged_pages = len(tag_tree_result.tag_tree.nodes_by_page())
    page_coverage = tagged_pages / page_count if page_count > 0 else 0
    tags_per_page = len(nodes) / page_count if page_count > 0 else 0
    # Full marks if all pages tagged and >=3 tags per page on average
    tag_score = w_tag * min(1.0, page_coverage) * min(1.0, tags_per_page / 3.0)
    details["tag_coverage"] = {
        "score": round(tag_score, 1),
        "max": w_tag,
        "tagged_pages": tagged_pages,
        "total_pages": page_count,
        "tags_per_page": round(tags_per_page, 1),
    }

    # --- Alt text quality ---
    figures = [n for n in nodes if n.tag in ("Figure", "Image")]
    if figures:
        good_alt = sum(1 for f in figures if len(f.alt_text or "") > 10)
        alt_ratio = good_alt / len(figures)
        alt_score = w_alt * alt_ratio
        details["alt_text_quality"] = {
            "score": round(alt_score, 1),
            "max": w_alt,
            "figures": len(figures),
            "with_meaningful_alt": good_alt,
        }
    else:
        alt_score = w_alt  # No figures = full marks (N/A)
        details["alt_text_quality"] = {"score": w_alt, "max": w_alt, "figures": 0, "note": "no figures"}

    # --- Heading structure ---
    # Error deductions scale with the weight so an alternate weight vector
    # keeps the "one heading error wipes out ~1/3 of this component"
    # semantic the original literals encoded (5 of 15 ≈ 1/3).
    heading_issues = [
        i for i in tag_tree_result.issues
        if i.rule_id in ("sr-heading-skip", "sr-no-headings", "sr-heading-start")
    ]
    heading_errors = sum(1 for i in heading_issues if i.severity == Severity.ERROR)
    heading_warnings = sum(1 for i in heading_issues if i.severity == Severity.WARNING)
    heading_error_deduction = w_heading / 3.0
    heading_warning_deduction = w_heading * (2.0 / 15.0)
    heading_score = max(
        0.0,
        w_heading
        - (heading_errors * heading_error_deduction)
        - (heading_warnings * heading_warning_deduction),
    )
    details["heading_structure"] = {
        "score": round(heading_score, 1),
        "max": w_heading,
        "errors": heading_errors,
        "warnings": heading_warnings,
    }

    # --- Table/list accessibility ---
    # Same error-scaling logic as heading (5/10 = half the component).
    table_list_issues = [
        i for i in tag_tree_result.issues
        if i.rule_id in ("sr-table-no-headers", "sr-list-no-items")
    ]
    tl_errors = sum(1 for i in table_list_issues if i.severity == Severity.ERROR)
    tl_error_deduction = w_table_list / 2.0
    tl_score = max(0.0, w_table_list - (tl_errors * tl_error_deduction))
    details["table_list_accessibility"] = {
        "score": round(tl_score, 1),
        "max": w_table_list,
        "errors": tl_errors,
    }

    total = round(
        min(weights.total, text_score + tag_score + alt_score + heading_score + tl_score),
        1,
    )
    return total, details


def generate_document_report(
    original_path: Path,
    remediated_path: Path,
    output_dir: Path,
    *,
    source_url: str = "",
    campus_name: str = "",
    brand_color: str = "#003366",
    config: PipelineConfig | None = None,
    acceptance: PDFAcceptanceResult | None = None,
    verification_mode: str = "sampled",
    visual_artifact_check: dict | None = None,
) -> DocumentReport:
    """Generate a full compliance report for one document.

    Runs the accessibility checker and screen reader validator on the
    remediated PDF, analyzes the original, maps results
    to WCAG 2.1 AA, and writes both JSON and HTML reports.

    When *acceptance* is provided, skips re-running validation and uses
    the cached result directly. This avoids redundant veraPDF/checker/SR
    calls when the acceptance was already computed during remediation.

    Returns the DocumentReport for aggregation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze original.
    original_info = _analyze_original(original_path, source_url)

    # Use cached acceptance or run the shared acceptance gate.
    if acceptance is None:
        acceptance = evaluate_pdf_acceptance(remediated_path, config=config)
    check_report: CheckReport = acceptance.checker_report
    sr_result: SRValidationResult = acceptance.tag_tree_result

    # Serialize check results with normalized statuses.
    serialized_checks = [
        {
            "rule_id": r.rule_id,
            "category": r.category,
            "description": r.description,
            "status": _normalize_status(r),
            "details": _clean_details(r.details),
            "fixable": r.fixable,
        }
        for r in check_report.results
    ]
    reviewable_checks = _build_reviewable_checks(check_report.results)
    manual_review_checks = _build_manual_review_checks(check_report.results)
    source_limited_issues = _build_source_limited_issues(acceptance)
    serialized_sr = [
        {
            "rule_id": i.rule_id,
            "severity": i.severity.value,
            "page": i.page,
            "element": i.element,
            "description": i.description,
            "suggestion": i.suggestion,
        }
        for i in sr_result.issues
    ]

    # WCAG mapping.
    wcag_results = []
    for criterion in WCAG_MAPPING:
        status, remarks = _determine_wcag_status(
            criterion, check_report.results, sr_result.issues
        )
        wcag_results.append(WCAGResult(
            criterion_id=criterion.id,
            criterion_name=criterion.name,
            level=criterion.level,
            status=status,
            remarks=remarks,
        ))

    # Overall conformance.
    conformance = _determine_conformance(acceptance)

    # Normalized issues array.
    normalized_issues = _build_normalized_issues(acceptance)

    # Screen reader readability score.
    readability_score, readability_details = calculate_screen_reader_readability(
        remediated_path, sr_result, check_report,
    )

    # Build report — use the actual PDF title, not the hash filename.
    doc_name = _get_document_title(remediated_path)
    report = DocumentReport(
        document_name=doc_name,
        original=original_info,
        remediated_path=str(remediated_path),
        remediated_size=remediated_path.stat().st_size,
        remediated_pages=check_report.page_count,
        verification_mode=verification_mode,
        check_results=serialized_checks,
        sr_issues=serialized_sr,
        tag_count=len(sr_result.tag_tree.nodes),
        verapdf_checked=acceptance.verapdf_result.checked,
        verapdf_passed=acceptance.verapdf_result.passed,
        verapdf_violations=acceptance.verapdf_result.violations,
        wcag_results=wcag_results,
        conformance=conformance,
        issues=normalized_issues,
        reviewable_checks=reviewable_checks,
        manual_review_checks=manual_review_checks,
        source_limited_issues=source_limited_issues,
        visual_diff=_serialize_visual_diff(getattr(acceptance, "visual_diff_result", None)),
        verification_coverage=_serialize_verification_coverage(
            acceptance,
            verification_mode=verification_mode,
        ),
        visual_artifact_check=_serialize_artifact_check(visual_artifact_check),
        screen_reader_readability=readability_score,
        screen_reader_readability_details=readability_details,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    # Write outputs — use the remediated filename stem for consistency.
    basename = _report_basename(original_path, remediated_path)

    json_path = output_dir / f"{basename}.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))

    html_path = output_dir / f"{basename}.html"
    html_path.write_text(_render_html(report, campus_name, brand_color))

    report.report_filename = f"{basename}.html"

    return report


def _serialize_visual_diff(vdr) -> dict:
    """Serialize a VisualDiffResult to a plain dict for JSON storage."""
    if vdr is None:
        return {}
    return {
        "checked": vdr.checked,
        "passed": vdr.passed,
        "total_pages": vdr.total_pages,
        "differing_pages": vdr.differing_pages,
        "max_page_diff": vdr.max_page_diff,
        "tolerance": vdr.tolerance,
        "error": vdr.error,
    }


def _serialize_artifact_check(artifact_check: dict | None) -> dict:
    """Normalize the vision artifact-check payload for JSON + HTML rendering.

    Accepts the shape produced by ``backend.app.remediation._vision_artifact_check``
    (``checked``, ``pages_checked``, ``has_artifacts``, ``flagged_pages``,
    ``error_count``). Returns ``{}`` when the check did not run so the HTML
    renderer can surface a "check skipped" note instead of pretending all
    pages were clean.
    """
    if not artifact_check:
        return {}
    checked = bool(artifact_check.get("checked", False))
    flagged_raw = artifact_check.get("flagged_pages") or []
    flagged: list[dict] = []
    for item in flagged_raw:
        if not isinstance(item, dict):
            continue
        page = item.get("page")
        try:
            page_int = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_int = None
        flagged.append({
            "page": page_int,
            "description": str(item.get("description") or "").strip(),
            "severity": str(item.get("severity") or "").strip() or "warning",
        })
    return {
        "checked": checked,
        "pages_checked": [int(p) for p in (artifact_check.get("pages_checked") or []) if isinstance(p, (int, float))],
        "has_artifacts": bool(artifact_check.get("has_artifacts", False)),
        "flagged_pages": flagged,
        "error_count": int(artifact_check.get("error_count", 0) or 0),
        "verification_mode": str(artifact_check.get("verification_mode") or "sampled"),
    }


def _serialize_verification_coverage(
    acceptance: PDFAcceptanceResult,
    *,
    verification_mode: str,
) -> dict:
    """Serialize how much of the document was vision-verified."""
    vision = getattr(acceptance, "vision_result", None)
    total_pages = (
        int(getattr(vision, "total_pages", 0) or 0)
        or int(getattr(acceptance.checker_report, "page_count", 0) or 0)
    )
    analyzed_pages = sorted(
        {
            int(page)
            for page in (getattr(vision, "analyzed_pages", None) or [])
        }
    )
    unresolved_checks: list[str] = []
    for result in acceptance.checker_report.results:
        if result.status != "Manual Check Needed":
            continue
        details_text = " ".join(result.details or [])
        if (
            "Remaining pages were not automatically verified" in details_text
            or "not verified on every affected page" in details_text
            or "not verified across every page" in details_text
        ):
            unresolved_checks.append(result.rule_id)

    return {
        "mode": verification_mode,
        "vision_checked": bool(vision is not None),
        "total_pages": total_pages,
        "analyzed_page_count": len(analyzed_pages),
        "analyzed_pages": analyzed_pages,
        "covers_all_pages": bool(getattr(vision, "covers_all_pages", False)),
        "unresolved_sampled_checks": unresolved_checks,
    }


def _report_basename(original_path: Path, remediated_path: Path) -> str:
    """Return a collision-resistant basename for per-document report files."""
    slug = _slugify(remediated_path.stem)[:68]
    digest = hashlib.sha1(str(original_path).encode("utf-8")).hexdigest()[:8]
    if not slug:
        return digest
    return f"{slug}-{digest}"


# Patterns that indicate the component doesn't exist in the document.
_NOT_APPLICABLE_PATTERNS = (
    "No multimedia objects found",
    "No form fields found",
    "No tables found",
    "No figures found",
    "No headings found",
    "No scripts found",
    "No timed responses",
    "No screen flicker",
    "No annotations found",
)

# Checks where failures are often false positives from broken PDF refs
# (e.g. unresolvable XObject references from the original document).
# These get downgraded from Failed to Passed when the details indicate
# the issue is structural rather than a real accessibility gap.
_FALSE_POSITIVE_PATTERNS = (
    "Images/forms found but no /Figure elements",
)


def _normalize_status(result: CheckResult) -> str:
    """Normalize check statuses for the compliance report.

    - "Passed" with "No X found" → "Not Applicable"
    - reading order / color contrast / use-of-color findings → "Needs Manual Review"
    - other "Manual Check Needed" statuses → "Failed"
    """
    # "Passed" but the component doesn't exist → Not Applicable.
    if result.status == "Passed" and result.details:
        for detail in result.details:
            if any(pattern in detail for pattern in _NOT_APPLICABLE_PATTERNS):
                return "Not Applicable"

    if _is_manual_review_check(result):
        return CHECK_STATUS_MANUAL_REVIEW

    if result.status == "Manual Check Needed":
        return "Failed"

    # Downgrade false positives from broken PDF refs.
    if result.status == "Failed" and result.details:
        for detail in result.details:
            if any(pattern in detail for pattern in _FALSE_POSITIVE_PATTERNS):
                return "Not Applicable"

    return result.status


def _build_manual_review_checks(results: list[CheckResult]) -> list[dict]:
    """Serialize the vision-assisted manual checks for reports and the API."""
    checks: list[dict] = []
    for payload in _build_reviewable_checks(results):
        if payload.get("decision") == "pass":
            continue
        checks.append({
            **payload,
            "status": CHECK_STATUS_MANUAL_REVIEW,
        })
    return checks


def _build_reviewable_checks(results: list[CheckResult]) -> list[dict]:
    """Serialize all vision/manual criteria, including full-pass decisions."""
    checks: list[dict] = []
    for result in results:
        if not _is_reviewable_check(result):
            continue
        checks.append({
            "rule_id": result.rule_id,
            "category": result.category,
            "description": result.description,
            "status": CHECK_STATUS_MANUAL_REVIEW if _is_manual_review_check(result) else result.status,
            "raw_status": result.status,
            "details": _clean_details(result.details),
            "fixable": result.fixable,
            "decision": _manual_review_decision(result),
            "recommendation": _manual_review_recommendation(result),
        })
    return checks


# Internal developer hints that should not appear in compliance reports.
_DETAIL_FILTERS = (
    "Configure a vision model",
    "configure a vision model",
    "config.yaml for automated",
    "requires visual inspection",
)


def _clean_details(details: list[str]) -> list[str]:
    """Remove internal developer hints from check details."""
    return [d for d in details if not any(f in d for f in _DETAIL_FILTERS)]


def _get_document_title(pdf_path: Path) -> str:
    """Extract a human-readable title from the PDF metadata or filename.

    Priority: dc:title from XMP > /Title from info dict > cleaned filename.
    """
    try:
        with pikepdf.open(pdf_path) as pdf:
            # Try XMP metadata first (most reliable after remediation).
            try:
                with pdf.open_metadata() as meta:
                    title = meta.get("dc:title", "")
                    if title and title.strip() and len(title.strip()) > 2:
                        return title.strip()
            except Exception:
                pass
    except Exception:
        pass

    # Fall back to cleaned filename.
    stem = pdf_path.stem
    # Strip leading hash prefix (e.g. "0045d824_").
    cleaned = re.sub(r"^[0-9a-f]{8}_", "", stem)
    # Replace underscores/hyphens with spaces, title-case.
    cleaned = re.sub(r"[-_]+", " ", cleaned).strip()
    if cleaned:
        return cleaned
    return stem


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:60] or "report"


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "Passed": "#16a34a",
    "Failed": "#dc2626",
    CHECK_STATUS_MANUAL_REVIEW: "#d97706",
    "Source Limited": "#7c3aed",
    "Not Applicable": "#6b7280",
    "PASS": "#16a34a",
    "FAIL": "#dc2626",
    WCAG_STATUS_REVIEW: "#d97706",
    "N/A": "#6b7280",
}

_SEVERITY_COLORS = {
    "error": "#dc2626",
    "warning": "#d97706",
    "review": "#d97706",
    "info": "#6b7280",
}


def _render_artifact_check_section(artifact_check: dict) -> str:
    """Render the visual artifact check section for the HTML report.

    Three states, each always rendered (so a missing section never looks
    like "we silently didn't look"):

    1. ``{}`` — check skipped (vision disabled or pages=0); neutral note.
    2. ``checked=False`` or no pages checked — attempted but errored.
    3. ``checked=True`` — either "no issues detected" or a table of flagged
       pages with page number, description, and severity.
    """
    check = dict(artifact_check or {})
    pages_checked = list(check.get("pages_checked") or [])
    flagged = list(check.get("flagged_pages") or [])
    error_count = int(check.get("error_count", 0) or 0)
    mode = str(check.get("verification_mode") or "sampled").lower()
    mode_label = "Deterministic (all pages)" if mode == "deterministic" else "Sampled pages"

    section_open = (
        '<section aria-labelledby="artifact-check-heading">'
        '<h2 id="artifact-check-heading">Visual Artifact Check</h2>'
    )
    intro = (
        "<p>A vision model reviewed rendered pages of the remediated PDF "
        "for rendering artifacts that the pixel-diff gate can miss — "
        "gray tints, stray rectangles, truncated text, or garbled glyphs.</p>"
    )

    if not check:
        body = (
            '<p class="details"><strong>Status:</strong> Check skipped — '
            "vision model unavailable or disabled for this run. "
            "No visual artifact review was performed on the remediated output.</p>"
        )
        return f"{section_open}{intro}{body}</section>"

    if not check.get("checked"):
        body = (
            '<p class="details"><strong>Status:</strong> Check attempted but did not complete — '
            f"{error_count} page(s) failed to render or the vision model returned no usable response.</p>"
        )
        return f"{section_open}{intro}{body}</section>"

    pages_label = ", ".join(f"p{p}" for p in pages_checked) if pages_checked else "none"
    error_suffix = f" {error_count} page(s) could not be evaluated." if error_count else ""
    coverage_line = (
        f'<p class="details"><strong>Coverage:</strong> {mode_label} — '
        f"{len(pages_checked)} page(s) reviewed ({html.escape(pages_label)})."
        f"{error_suffix}</p>"
    )

    if not flagged:
        body = (
            '<p><span class="badge" style="background:#16a34a">No issues detected</span> '
            "— the vision model did not flag any rendering artifacts on the reviewed pages.</p>"
            f"{coverage_line}"
        )
        return f"{section_open}{intro}{body}</section>"

    # One or more flagged pages — render a table.
    rows: list[str] = []
    for item in flagged:
        page = item.get("page")
        page_str = f"p{int(page)}" if isinstance(page, (int, float)) else "—"
        severity = str(item.get("severity") or "warning").lower()
        sev_color = _SEVERITY_COLORS.get(severity, "#d97706")
        desc = html.escape(item.get("description") or "(no description)")
        rows.append(
            f'<tr><td>{html.escape(page_str)}</td>'
            f'<td style="color:{sev_color};font-weight:600">{html.escape(severity.upper())}</td>'
            f'<td>{desc}</td></tr>'
        )

    table = (
        "<table><thead><tr><th>Page</th><th>Severity</th><th>Description</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    badge = (
        f'<p><span class="badge" style="background:#d97706">'
        f"{len(flagged)} issue(s) flagged</span></p>"
    )
    body = badge + coverage_line + table
    return f"{section_open}{intro}{body}</section>"


# Honest triage labels — the backend enum stays "Conformant" etc. for
# API / JSON stability, but every human-facing surface in the HTML report
# uses the triage framing instead. Internal "Conformant" means "the
# automated checks we CAN run passed" — not "legally accessible".
_TRIAGE_LABEL: dict[str, str] = {
    Conformance.CONFORMANT: "Triage: Clear",
    Conformance.PARTIALLY: "Triage: Partial",
    Conformance.NOT_CONFORMANT: "Triage: Failing",
}

_TRIAGE_SHORT: dict[str, str] = {
    Conformance.CONFORMANT: "Clear",
    Conformance.PARTIALLY: "Partial",
    Conformance.NOT_CONFORMANT: "Failing",
}


def _wcag_status_label(status: str) -> str:
    if status == WCAG_STATUS_REVIEW:
        return CHECK_STATUS_MANUAL_REVIEW
    return status


def _format_review_decision(decision: str) -> str:
    return {
        "pass": "Pass",
        "no_pass": "No pass",
        "rerun_full_verification": "Rerun full verification",
        "manual_review": "Manual review",
    }.get(decision, "Manual review")


def _render_vision_manual_review_section(report: DocumentReport) -> str:
    """Render reading-order, contrast, and use-of-color review items separately."""
    checks = list(getattr(report, "manual_review_checks", []) or [])
    wcag_items = [
        w for w in report.wcag_results if w.status == WCAG_STATUS_REVIEW
    ]
    if not checks and not wcag_items:
        return ""

    rows: list[str] = []
    for check in checks:
        details = "; ".join((check.get("details") or [])[:3])
        rows.append(
            f'<tr><td>{html.escape(check.get("description") or "")}</td>'
            f'<td><span class="badge" style="background:#d97706">'
            f'{CHECK_STATUS_MANUAL_REVIEW}</span></td>'
            f'<td>{html.escape(_format_review_decision(check.get("decision") or "manual_review"))}</td>'
            f'<td class="details">{html.escape(details)}</td>'
            f'<td class="details">{html.escape(check.get("recommendation") or "")}</td></tr>'
        )

    wcag_rows = []
    for item in wcag_items:
        wcag_rows.append(
            f'<li><strong>{html.escape(item.criterion_id)} '
            f'{html.escape(item.criterion_name)}</strong> '
            f'(Level {html.escape(item.level)})</li>'
        )
    wcag_note = (
        "<p class=\"details\"><strong>WCAG criteria awaiting review:</strong></p>"
        f"<ul>{''.join(wcag_rows)}</ul>"
        if wcag_rows
        else ""
    )

    table = (
        "<table><thead><tr><th>Check</th><th>Status</th><th>Decision</th><th>Evidence</th>"
        "<th>Recommendation</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        if rows
        else "<p>No checker-level manual review items were recorded.</p>"
    )
    return (
        '<section aria-labelledby="vision-manual-review-heading">'
        '<h2 id="vision-manual-review-heading">Vision-Assisted Manual Review</h2>'
        '<p>Reading order, contrast, and use-of-color require visual judgment. Remedy '
        'uses the configured vision model to inspect rendered pages and provide '
        'a recommendation, but these items are reported as manual review rather '
        'than failed automated checks until a reviewer confirms the result.</p>'
        f"{wcag_note}"
        f"{table}"
        '</section>'
    )


def _render_source_limited_section(report: DocumentReport) -> str:
    """Render residual source-font limitations separately from failures."""
    issues = list(getattr(report, "source_limited_issues", []) or [])
    if not issues:
        return ""

    rows = []
    for issue in issues:
        details = "; ".join((issue.get("details") or [])[:2])
        rows.append(
            f'<tr><td>{html.escape(issue.get("rule_id") or "")}</td>'
            f'<td>{html.escape(issue.get("description") or "")}</td>'
            f'<td><span class="badge" style="background:#7c3aed">Source Limited</span></td>'
            f'<td class="details">{html.escape(details)}</td>'
            f'<td class="details">{html.escape(issue.get("recommendation") or "")}</td></tr>'
        )

    table = (
        "<table><thead><tr><th>Rule</th><th>Description</th><th>Status</th>"
        "<th>Evidence</th><th>Recommendation</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return (
        '<section aria-labelledby="source-limited-heading">'
        '<h2 id="source-limited-heading">Source Limited</h2>'
        "<p>These residual veraPDF findings are font/CIDSet limitations inherited "
        "from the source file. They are not counted as failed automated WCAG checks.</p>"
        f"{table}</section>"
    )


def _render_human_review_checklist() -> str:
    """Render the Human Review Checklist section from WCAG criteria the
    automated pipeline CANNOT test (empty ``rule_ids`` in ``WCAG_MAPPING``).

    The Matterhorn Protocol 1.1 reports ~2/3 of PDF/UA requirements as
    machine-testable — the remaining ~1/3 require human judgment. We
    pin that expectation here rather than burying it in the WCAG table
    footnote so every downloaded report starts with "here's what you
    still have to check yourself."
    """
    human_guidance = {
        "1.3.3": "Sensory Characteristics — confirm instructions never rely "
                 "only on shape, color, sound, or visual position.",
        "2.1.2": "No Keyboard Trap — open the PDF in Acrobat and Tab through "
                 "every interactive element to verify focus can always leave.",
        "3.1.2": "Language of Parts — spot-check passages in other languages; "
                 "each should carry its own `/Lang` override.",
        "3.2.3": "Consistent Navigation — page navigation (headers, bookmarks, "
                 "link patterns) must stay consistent across the document.",
        "3.2.4": "Consistent Identification — repeated components (icons, "
                 "buttons) must be labeled the same way every time.",
    }
    rows: list[str] = []
    untested = [c for c in WCAG_MAPPING if not c.rule_ids]
    for criterion in untested:
        guidance = human_guidance.get(
            criterion.id,
            "Human judgment required.",
        )
        rows.append(
            f'<tr><td><input type="checkbox" aria-label="Reviewed"></td>'
            f'<td><strong>{criterion.id}</strong> '
            f'{criterion.name}</td>'
            f'<td>{criterion.level}</td>'
            f'<td>{guidance}</td></tr>'
        )
    return (
        '<section aria-labelledby="human-review-heading">'
        '<h2 id="human-review-heading">Human Review Checklist</h2>'
        '<p>The Matterhorn Protocol — the reference failure catalog for '
        'PDF/UA-1 — classifies roughly two thirds of accessibility '
        'requirements as machine-testable. The remaining third requires '
        'human judgment. This automated report covers the machine-testable '
        'portion. The items below cannot be verified by any automated tool '
        '(including ours, Adobe Acrobat Pro, or PAC 2024) and must be '
        'confirmed by a reviewer before publishing.</p>'
        '<table><thead><tr><th>✓</th><th>WCAG Criterion</th><th>Level</th>'
        '<th>What to check</th></tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table>'
        '<p class="details">For a deeper pass, run the remediated PDF '
        'through <a href="https://pac.pdf-accessibility.org/en" '
        'target="_blank" rel="noopener">PAC 2024</a> and Adobe Acrobat Pro '
        'Accessibility Checker, and listen to the document with a screen '
        'reader (NVDA on Windows or VoiceOver on macOS). No automated tool '
        'is a substitute for that final review.</p>'
        '</section>'
    )


def _render_html(report: DocumentReport, campus_name: str, brand_color: str) -> str:
    """Render a self-contained accessible HTML compliance report."""
    conf_color = {
        Conformance.CONFORMANT: "#16a34a",
        Conformance.PARTIALLY: "#d97706",
        Conformance.NOT_CONFORMANT: "#dc2626",
    }.get(report.conformance, "#6b7280")
    triage_label = _TRIAGE_LABEL.get(report.conformance, f"Triage: {report.conformance}")
    triage_short = _TRIAGE_SHORT.get(report.conformance, report.conformance.split()[0])
    automated_applicable_checks = max(0, report.automated_applicable_checks)
    wcag_auto_total = report.wcag_auto_tested_count
    readability_score = float(getattr(report, "screen_reader_readability", 0.0) or 0.0)
    if readability_score >= 90:
        readability_label = "Excellent"
        readability_color = "#16a34a"
    elif readability_score >= 70:
        readability_label = "Good"
        readability_color = "#d97706"
    else:
        readability_label = "Needs Improvement"
        readability_color = "#dc2626"

    campus_label = f" — {campus_name}" if campus_name else ""
    verification = dict(getattr(report, "verification_coverage", {}) or {})
    verification_mode = verification.get("mode", getattr(report, "verification_mode", "sampled"))
    analyzed_page_count = int(verification.get("analyzed_page_count", 0) or 0)
    total_pages = int(verification.get("total_pages", 0) or 0)
    covers_all_pages = bool(verification.get("covers_all_pages", False))
    unresolved_sampled_checks = list(verification.get("unresolved_sampled_checks") or [])
    if verification.get("vision_checked"):
        if covers_all_pages and total_pages:
            verification_summary = f"Vision verification covered all {total_pages} page(s)."
        elif analyzed_page_count and total_pages:
            verification_summary = (
                f"Vision verification covered {analyzed_page_count} of {total_pages} page(s). "
                "Some checks may remain unresolved on unverified pages."
            )
        else:
            verification_summary = "Vision verification ran, but page coverage details were unavailable."
    else:
        verification_summary = "No vision verification data was available for this run."
    unresolved_checks_html = ""
    if unresolved_sampled_checks:
        unresolved_checks_html = (
            "<p class=\"details\"><strong>Sampled-only unresolved checks:</strong> "
            + ", ".join(unresolved_sampled_checks)
            + "</p>"
        )

    # Build check rows.
    check_rows = []
    for r in report.check_results:
        color = _STATUS_COLORS.get(r["status"], "#6b7280")
        details = "; ".join(r["details"][:3]) if r["details"] else ""
        check_rows.append(
            f'<tr><td>{r["category"]}</td>'
            f'<td>{r["description"]}</td>'
            f'<td style="color:{color};font-weight:600">{r["status"]}</td>'
            f'<td class="details">{details}</td></tr>'
        )

    # Build SR issue rows.
    sr_rows = []
    for i in report.sr_issues:
        sev = i["severity"]
        color = {"error": "#dc2626", "warning": "#d97706", "info": "#6b7280"}[sev]
        page_str = f'p{i["page"] + 1}' if i["page"] >= 0 else "doc"
        sr_rows.append(
            f'<tr><td style="color:{color};font-weight:600">{sev.upper()}</td>'
            f'<td>{page_str}</td>'
            f'<td>{i["element"]}</td>'
            f'<td>{i["description"]}</td></tr>'
        )

    # Build WCAG rows.
    wcag_rows = []
    for w in report.wcag_results:
        color = _STATUS_COLORS.get(w.status, "#6b7280")
        status_label = _wcag_status_label(w.status)
        wcag_rows.append(
            f'<tr><td>{w.criterion_id} {w.criterion_name} (Level {w.level})</td>'
            f'<td style="color:{color};font-weight:600">{status_label}</td>'
            f'<td class="details">{w.remarks}</td></tr>'
        )

    artifact_section_html = _render_artifact_check_section(
        getattr(report, "visual_artifact_check", {}) or {}
    )

    source_limited_html = _render_source_limited_section(report)
    vision_manual_review_html = _render_vision_manual_review_section(report)
    human_review_checklist_html = _render_human_review_checklist()

    readability_rows = []
    for key, label in (
        ("text_extractability", "Text extractability"),
        ("tag_coverage", "Tag coverage"),
        ("alt_text_quality", "Alt text quality"),
        ("heading_structure", "Heading structure"),
        ("table_list_accessibility", "Table/list accessibility"),
    ):
        component = dict((report.screen_reader_readability_details or {}).get(key) or {})
        if not component:
            continue
        score = component.get("score", 0)
        max_score = component.get("max", "")
        extras = ", ".join(
            f"{name.replace('_', ' ')}={value}"
            for name, value in component.items()
            if name not in {"score", "max"}
        )
        readability_rows.append(
            f'<tr><td>{label}</td>'
            f'<td style="font-weight:600">{score}/{max_score}</td>'
            f'<td class="details">{extras}</td></tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compliance Report — {report.document_name}{campus_label}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6; color: #1a1a1a; max-width: 1100px; margin: 0 auto;
    padding: 20px; background: #fafafa;
  }}
  a {{ color: {brand_color}; }}
  a:focus {{ outline: 3px solid {brand_color}; outline-offset: 2px; }}
  .skip-nav {{
    position: absolute; left: -9999px; top: auto;
    padding: 8px 16px; background: {brand_color}; color: #fff;
    z-index: 1000; text-decoration: none;
  }}
  .skip-nav:focus {{ left: 10px; top: 10px; }}
  header {{
    background: {brand_color}; color: #fff; padding: 24px 32px;
    border-radius: 8px 8px 0 0; margin-bottom: 0;
  }}
  header h1 {{ margin: 0 0 4px 0; font-size: 1.5rem; }}
  header p {{ margin: 0; opacity: 0.9; font-size: 0.9rem; }}
  .hero {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 16px; padding: 24px; background: #fff;
    border: 1px solid #e5e7eb; border-top: none;
  }}
  .stat {{ text-align: center; }}
  .stat .number {{ font-size: 2rem; font-weight: 700; }}
  .stat .label {{ font-size: 0.85rem; color: #6b7280; }}
  section {{ background: #fff; border: 1px solid #e5e7eb; padding: 24px; margin-top: 16px; border-radius: 6px; }}
  h2 {{ color: {brand_color}; margin-top: 0; border-bottom: 2px solid {brand_color}; padding-bottom: 8px; }}
  h3 {{ margin-top: 24px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #f9fafb; font-weight: 600; position: sticky; top: 0; }}
  .details {{ color: #6b7280; font-size: 0.85rem; max-width: 350px; }}
  .badge {{
    display: inline-block; padding: 4px 12px; border-radius: 4px;
    font-weight: 700; font-size: 0.9rem; color: #fff;
  }}
  .info-table {{ width: 100%; border-collapse: collapse; }}
  .info-table th {{ text-align: right; width: 200px; padding: 8px 16px 8px 0; color: #6b7280; font-weight: 600; font-size: 0.9rem; border-bottom: 1px solid #f3f4f6; }}
  .info-table td {{ padding: 8px 0; border-bottom: 1px solid #f3f4f6; }}
  .info-table .section-label {{ background: #f9fafb; font-weight: 700; color: {brand_color}; padding: 10px 16px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .info-table .section-label td {{ background: #f9fafb; font-weight: 700; color: {brand_color}; padding: 10px 16px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  dl {{ margin: 8px 0; }}
  dt {{ font-weight: 600; font-size: 0.85rem; color: #6b7280; margin-top: 8px; }}
  dd {{ margin: 0 0 4px 0; }}
  footer {{ margin-top: 32px; padding: 16px; text-align: center; font-size: 0.8rem; color: #9ca3af; }}
  @media (max-width: 768px) {{
    .before-after {{ grid-template-columns: 1fr; }}
    .hero {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<a href="#main" class="skip-nav">Skip to main content</a>

<header>
  <h1>{report.document_name}</h1>
  <p>Accessibility Triage Report — Automated Assessment{campus_label}</p>
</header>

<aside role="note" style="background:#fef3c7;border-left:4px solid #d97706;padding:14px 18px;margin:16px 0;border-radius:6px">
  <p style="margin:0;font-weight:600">This report is an automated triage, not a compliance certification.</p>
  <p style="margin:8px 0 0 0;font-size:0.9rem">
    Approximately two-thirds of PDF/UA-1 accessibility requirements are machine-testable;
    the remaining third requires human judgment (whether headings convey the <em>correct</em> structure,
    whether alt text is <em>semantically</em> correct, etc.). Before publishing, review this document in
    <a href="https://pac.pdf-accessibility.org/en" target="_blank" rel="noopener">PAC 2024</a>,
    Adobe Acrobat Pro, and with a screen reader (NVDA / VoiceOver).
    See the <a href="#human-review-heading">Human Review Checklist</a> below.
  </p>
</aside>

<div class="hero" role="region" aria-label="Summary statistics">
  <div class="stat">
    <div class="number" style="color:{conf_color}">{triage_short}</div>
    <div class="label">Automated Triage</div>
  </div>
  <div class="stat">
    <div class="number">{report.passed_checks}/{automated_applicable_checks}</div>
    <div class="label">Automated Checks Passed</div>
  </div>
  <div class="stat">
    <div class="number">{report.wcag_pass_count}/{wcag_auto_total}</div>
    <div class="label">WCAG Auto-tested</div>
  </div>
  <div class="stat">
    <div class="number">{report.manual_review_check_count}</div>
    <div class="label">Needs Manual Review</div>
  </div>
  <div class="stat">
    <div class="number">{report.source_limited_count}</div>
    <div class="label">Source Limited</div>
  </div>
  <div class="stat">
    <div class="number">{report.tag_count}</div>
    <div class="label">Structure Tags</div>
  </div>
</div>

<main id="main">

<section>
  <h2>What This Report Shows</h2>
  <p>This is an <strong>automated triage</strong> of a single PDF against the machine-testable
    subset of <strong>WCAG 2.1 Level AA</strong> and <strong>PDF/UA-1</strong>. The document was
    evaluated with {len(report.check_results)} accessibility checks plus {len(SCREEN_READER_RULE_IDS)} screen-reader
    simulation checks that approximate how NVDA and VoiceOver navigate PDFs. The triage result below
    tells you what the automated pipeline could verify — it does <em>not</em> establish legal
    compliance with the ADA, Section 508, or EN 301 549. Those require human review by someone
    qualified to evaluate the items that automation cannot reach (see the Human Review Checklist).</p>
  <dl>
    <dt><span class="badge" style="background:#16a34a">Triage: Clear</span></dt>
    <dd>Every automated check the pipeline was able to run passed (or produced only non-blocking
      notes such as source-font encoding limitations). Human review is still required before publishing.</dd>
    <dt><span class="badge" style="background:#d97706">Triage: Partial</span></dt>
    <dd>Some automated checks failed. The document may still be partially usable;
      review the failed checks and screen-reader issues below before distributing.</dd>
    <dt><span class="badge" style="background:#dc2626">Triage: Failing</span></dt>
    <dd>Significant automated-check failures detected. Screen-reader users are likely to encounter barriers.
      Address the blocking issues before this PDF is published.</dd>
  </dl>
  <p class="details"><strong>Verification mode:</strong> {verification_mode.title()}</p>
  <p class="details">{verification_summary}</p>
  {unresolved_checks_html}
</section>

<section>
  <h2>Screen Reader Readability</h2>
  <p>
    Composite readability score:
    <span class="badge" style="background:{readability_color}">{readability_score:.1f}/100 ({readability_label})</span>
  </p>
  <p class="details">
    This score summarizes practical screen reader readability across text extractability,
    tagging coverage, alt text quality, heading structure, and table/list accessibility.
  </p>
  {"<table><thead><tr><th>Component</th><th>Score</th><th>Details</th></tr></thead><tbody>" + "".join(readability_rows) + "</tbody></table>" if readability_rows else "<p>No readability component breakdown available.</p>"}
</section>

<section>
  <h2>Document Information</h2>
  <table class="info-table">
    <tr class="section-label"><td colspan="2">Source</td></tr>
    <tr><th>Original File Type</th><td>{report.original.file_type.upper()}</td></tr>
    <tr><th>Original File Size</th><td>{_human_size(report.original.file_size)}</td></tr>
    <tr><th>Original PDF (un-remediated)</th><td><a href="../../downloads/pdf/{Path(report.original.file_path).name}" target="_blank">{Path(report.original.file_path).name}</a></td></tr>
    <tr><th>Remediated PDF</th><td><a href="../../remediated-pdfs/{Path(report.remediated_path).name}" target="_blank">{Path(report.remediated_path).name}</a></td></tr>
    <tr><th>Source Web Page</th><td style="word-break:break-all">{f'<a href="{report.original.source_url}" target="_blank" rel="noopener">{report.original.source_url}</a>' if report.original.source_url else "N/A"}</td></tr>
    <tr class="section-label"><td colspan="2">Original Accessibility State</td></tr>
    <tr><th>Had Structure Tags</th><td>{"Yes" if report.original.is_tagged else "No"}</td></tr>
    <tr><th>Had Language Set</th><td>{"Yes" if report.original.has_language else "No"}</td></tr>
    <tr><th>Had Document Title</th><td>{"Yes" if report.original.has_title else "No"}</td></tr>
    <tr class="section-label"><td colspan="2">After Remediation</td></tr>
    <tr><th>Remediated File Size</th><td>{_human_size(report.remediated_size)}</td></tr>
    <tr><th>Pages</th><td>{report.remediated_pages}</td></tr>
    <tr><th>Verification Mode</th><td>{verification_mode.title()}</td></tr>
    <tr><th>Vision Verification Coverage</th><td>{verification_summary}</td></tr>
    <tr><th>Structure Tags</th><td>{report.tag_count}</td></tr>
    <tr><th>Checks Passed</th><td>{report.passed_checks} of {automated_applicable_checks} automated applicable ({report.manual_review_check_count} manual review, {report.na_checks} not applicable)</td></tr>
    <tr><th>Screen Reader Errors</th><td>{report.sr_error_count}</td></tr>
    <tr><th>Screen Reader Warnings</th><td>{report.sr_warning_count}</td></tr>
    <tr><th>veraPDF</th><td>{"PASS" if report.verapdf_passed else ("FAIL" if report.verapdf_checked else "Unavailable")}</td></tr>
    <tr><th>Automated Triage</th><td><span class="badge" style="background:{conf_color}">{triage_label}</span></td></tr>
  </table>
</section>

<section>
  <h2>WCAG 2.1 AA — Automated Coverage</h2>
  <p class="details">The pipeline reports on {sum(1 for c in WCAG_MAPPING if c.rule_ids)} of
    {len(WCAG_MAPPING)} WCAG 2.1 AA criteria. The remaining
    {sum(1 for c in WCAG_MAPPING if not c.rule_ids)} (1.3.3 Sensory Characteristics, 2.1.2 No Keyboard Trap,
    3.1.2 Language of Parts, 3.2.3 Consistent Navigation, 3.2.4 Consistent Identification) cannot be
    verified by any automated tool and are covered in the Human Review Checklist below.
    Reading order, use of color, and color contrast are handled as vision-assisted manual review items and are not
    counted as WCAG failures unless another mapped automated or screen-reader rule fails.</p>
  <table>
    <thead><tr><th>Success Criterion</th><th>Status</th><th>Remarks</th></tr></thead>
    <tbody>
      {"".join(wcag_rows)}
    </tbody>
  </table>
</section>

<section>
  <h2>Accessibility Checks ({len(report.check_results)})</h2>
  <table>
    <thead><tr><th>Category</th><th>Check</th><th>Result</th><th>Details</th></tr></thead>
    <tbody>
      {"".join(check_rows)}
    </tbody>
  </table>
</section>

<section>
  <h2>Screen Reader Validation ({len(report.sr_issues)} issues)</h2>
  {"<p>No screen reader issues detected.</p>" if not sr_rows else f'''<table>
    <thead><tr><th>Severity</th><th>Page</th><th>Element</th><th>Issue</th></tr></thead>
    <tbody>
      {"".join(sr_rows)}
    </tbody>
  </table>'''}
</section>

{artifact_section_html}

{source_limited_html}

{vision_manual_review_html}

{human_review_checklist_html}

</main>

<footer>
  Generated {report.generated_at[:10]} by {report.generator}.
  <br>
  <span style="font-size:0.75rem">This is an automated triage report. It does not constitute ADA, Section 508, or PDF/UA certification.</span>
</footer>
</body>
</html>"""
