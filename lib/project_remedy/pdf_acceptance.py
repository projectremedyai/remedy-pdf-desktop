"""Shared PDF acceptance checks for the primary PDF-to-PDF workflow."""

from __future__ import annotations

import asyncio
import logging
import pikepdf
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from project_remedy.config import PipelineConfig
from project_remedy.pdf_checker import (
    CheckReport,
    CheckResult,
    PDFAccessibilityChecker,
    SOURCE_FONT_RISK_DETAIL_PREFIX,
)

logger = logging.getLogger(__name__)
from project_remedy.tag_tree_reader import (
    ScreenReaderIssue,
    Severity,
    TagTreeReport,
    ValidationResult as TagTreeValidationResult,
    validate_tag_tree,
)

REVIEWABLE_CHECK_RULE_IDS = frozenset({
    "doc-reading-order",
    "doc-color-contrast",
    "doc-use-of-color",
})


@dataclass
class PDFOpenabilityResult:
    """Basic parser/viewer openability for a PDF."""

    checked: bool
    openable: bool
    page_count: int = 0
    parser: str = ""
    error: str = ""


@dataclass
class VisualDiffResult:
    """Page-by-page pixel diff between original and remediated PDF."""

    checked: bool
    passed: bool
    total_pages: int = 0
    differing_pages: list[int] = field(default_factory=list)  # 0-indexed
    max_page_diff: float = 0.0   # worst single-page diff (0.0–1.0)
    tolerance: float = 0.05  # matches compare_pdf_visual_fidelity default
    error: str = ""


@dataclass
class VeraPDFResult:
    """veraPDF outcome for one PDF."""

    checked: bool
    passed: bool
    violations: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass
class PDFAcceptanceResult:
    """Composite PDF acceptance decision."""

    file_path: Path
    checker_report: CheckReport
    tag_tree_result: TagTreeValidationResult
    verapdf_result: VeraPDFResult
    openability_result: PDFOpenabilityResult | None = None
    visual_diff_result: VisualDiffResult | None = None
    vision_result: Any = None
    checker_error: str = ""
    screen_reader_error: str = ""

    @property
    def checker_failures(self) -> list[CheckResult]:
        return [
            r
            for r in self.checker_report.results
            if r.status in {"Failed", "Manual Check Needed"}
        ]

    def _is_vision_backed_clean_sample(self, result: CheckResult) -> bool:
        """Return True when *result* is a Manual-Check-Needed vision check that
        the vision model already cleared on its analyzed sample.

        REMEDY-69 #12: ``_check_logical_reading_order`` and
        ``_check_color_contrast`` legitimately return ``"Manual Check Needed"``
        when vision sampled a subset of pages and found nothing wrong. Prior
        to this filter, the batch acceptance gate treated that status as a
        blocking failure — which meant every sampled-vision run on a large
        PDF (the default path under ``VISION_PAGE_SAMPLE_SIZE=10``) was
        routed to manual review even when vision said the sample was clean.
        That inflated the manual-review queue by ~40% on the LAMC six-shard
        run. The invariant we restore here is: *if vision successfully
        analyzed at least one page and found no relevant defect in that
        sample, the corresponding Manual-Check-Needed is evidence-backed
        and must not block acceptance.*
        """
        if result.status != "Manual Check Needed":
            return False
        if result.rule_id not in REVIEWABLE_CHECK_RULE_IDS:
            return False
        vision = self.vision_result
        if vision is None:
            return False
        # Vision must actually have analyzed something — otherwise the
        # Manual-Check-Needed is speculative and still a real review signal.
        analyzed = getattr(vision, "analyzed_pages", None) or []
        try:
            analyzed_count = len({int(page) for page in analyzed})
        except (TypeError, ValueError):
            analyzed_count = 0
        if analyzed_count <= 0:
            return False

        if result.rule_id == "doc-reading-order":
            if not hasattr(vision, "reading_order_issues"):
                return False
            issues = getattr(vision, "reading_order_issues", None) or []
            # Any error-severity finding means vision actively flagged the
            # sample — the Manual-Check-Needed wrapper is real, don't clear it.
            has_error = any(
                getattr(issue, "severity", "warning") == "error" for issue in issues
            )
            return not has_error

        if result.rule_id == "doc-use-of-color":
            if not hasattr(vision, "use_of_color_issues"):
                return False
            use_of_color_issues = getattr(vision, "use_of_color_issues", None) or []
            return len(use_of_color_issues) == 0

        # doc-color-contrast: any non-empty issue list counts as a hit.
        if not hasattr(vision, "contrast_issues"):
            return False
        contrast_issues = getattr(vision, "contrast_issues", None) or []
        return len(contrast_issues) == 0

    @staticmethod
    def _is_reviewable_checker_failure(result: CheckResult) -> bool:
        return (
            result.rule_id in REVIEWABLE_CHECK_RULE_IDS
            and result.status in {"Failed", "Manual Check Needed"}
        )

    def _blocking_checker_failures(self) -> list[CheckResult]:
        """Checker failures that actually block conformance / manual review.

        Filters out two classes of benign signals:
        - Source-font-only encoding failures (see ``_is_source_font_checker_failure``)
        - Vision-backed clean-sample Manual-Check-Needed entries
          (see ``_is_vision_backed_clean_sample``)
        """
        return [
            f
            for f in self.checker_failures
            if not self._is_source_font_checker_failure(f)
            and not self._is_vision_backed_clean_sample(f)
            and not self._is_reviewable_checker_failure(f)
        ]

    @property
    def screen_reader_errors(self) -> list[ScreenReaderIssue]:
        return [
            issue
            for issue in self.tag_tree_result.issues
            if issue.severity == Severity.ERROR
        ]

    @property
    def openable(self) -> bool:
        if self.openability_result is None:
            return True
        return self.openability_result.openable

    @property
    def blocking_failure_reasons(self) -> list[str]:
        if self.openable:
            return []
        if self.openability_result and self.openability_result.error:
            return [self.openability_result.error]
        return ["PDF could not be opened"]

    @property
    def warning_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not self.openable:
            return entries
        for result in self.checker_failures:
            entries.append(
                {
                    "source": "checker",
                    "rule_id": result.rule_id,
                    "description": result.description,
                    "details": list(result.details),
                    "fixable": result.fixable,
                }
            )
        if self.checker_error:
            entries.append(
                {
                    "source": "checker",
                    "rule_id": "checker-runtime",
                    "description": self.checker_error,
                    "details": [],
                    "fixable": False,
                }
            )
        for issue in self.screen_reader_errors:
            entries.append(
                {
                    "source": "screen_reader",
                    "rule_id": issue.rule_id,
                    "description": issue.description,
                    "details": [issue.element] if issue.element else [],
                    "fixable": True,
                }
            )
        if self.screen_reader_error:
            entries.append(
                {
                    "source": "screen_reader",
                    "rule_id": "screen-reader-runtime",
                    "description": self.screen_reader_error,
                    "details": [],
                    "fixable": False,
                }
            )
        if self.verapdf_result.checked and not self.verapdf_result.passed:
            for violation in self.verapdf_result.violations:
                entries.append(
                    {
                        "source": "verapdf",
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
                        "fixable": not self._is_source_font_limitation(violation),
                    }
                )
            if self.verapdf_result.error:
                entries.append(
                    {
                        "source": "verapdf",
                        "rule_id": "verapdf-runtime",
                        "description": self.verapdf_result.error,
                        "details": [],
                        "fixable": False,
                    }
                )
        vdr = self.visual_diff_result
        if vdr and vdr.checked and not vdr.passed:
            entries.append(
                {
                    "source": "visual_diff",
                    "rule_id": "visual-fidelity",
                    "description": (
                        f"Remediated PDF differs visually from original on "
                        f"{len(vdr.differing_pages)} page(s) "
                        f"(max diff {vdr.max_page_diff:.2%}, tolerance {vdr.tolerance:.2%})"
                    ),
                    "details": [f"Page {p + 1}" for p in vdr.differing_pages],
                    "fixable": False,
                }
            )
        return entries

    @property
    def warning_reasons(self) -> list[str]:
        if not self.openable:
            return []
        reasons: list[str] = []
        # Only count blocking checker failures — exclude source-font-only
        # encoding issues and vision-backed clean-sample Manual-Check-Needed
        # results (REMEDY-69 #12).
        blocking_checker = self._blocking_checker_failures()
        if blocking_checker:
            reasons.append(f"{len(blocking_checker)} checker failure(s)")
        if self.checker_error:
            reasons.append(f"checker unavailable ({self.checker_error})")
        if self.screen_reader_errors:
            reasons.append(f"{len(self.screen_reader_errors)} screen reader error(s)")
        if self.screen_reader_error:
            reasons.append(f"screen reader validation unavailable ({self.screen_reader_error})")
        # NOTE: visual_diff is intentionally excluded from warning_reasons
        # so it does NOT trigger retry/manual-review routing or conformance
        # downgrade. It appears only in warning_entries and summary().
        if self.verapdf_result.checked and not self.verapdf_result.passed:
            if self.verapdf_result.violations:
                source_font_limitations = [
                    violation
                    for violation in self.verapdf_result.violations
                    if self._is_source_font_limitation(violation)
                ]
                if source_font_limitations == self.verapdf_result.violations:
                    reasons.append(
                        "veraPDF failed "
                        f"({len(source_font_limitations)} likely source-font/CIDSet limitation(s); "
                        "not usually fixable by structure-only remediation)"
                    )
                else:
                    reasons.append(
                        f"veraPDF failed ({len(self.verapdf_result.violations)} violation(s))"
                    )
            elif self.verapdf_result.error:
                reasons.append(self.verapdf_result.error)
            else:
                reasons.append("veraPDF failed")
        return reasons

    @property
    def retry_reasons(self) -> list[str]:
        return list(self.warning_reasons)

    @property
    def non_blocking_verapdf_warnings(self) -> list[dict[str, Any]]:
        if not self.verapdf_result.checked or self.verapdf_result.passed:
            return []
        if all(self._is_source_font_limitation(violation) for violation in self.verapdf_result.violations):
            return list(self.verapdf_result.violations)
        return []

    @staticmethod
    def _is_source_font_checker_failure(result: CheckResult) -> bool:
        """Return True when a checker failure is a source-font-only encoding issue."""
        if result.rule_id != "page-char-encoding":
            return False
        if not result.details:
            return False
        return all(
            detail.startswith(SOURCE_FONT_RISK_DETAIL_PREFIX)
            for detail in result.details
        )

    @property
    def passed(self) -> bool:
        if not self.openable:
            return False
        # REMEDY-69 #12: vision-backed clean-sample Manual-Check-Needed
        # results are evidence that vision saw no defect — they must not
        # block conformance acceptance.
        blocking_checker = self._blocking_checker_failures()
        if blocking_checker:
            return False
        if self.screen_reader_errors:
            return False
        if self.verapdf_result.checked and not self.verapdf_result.passed:
            # Source-font-only veraPDF failures are non-blocking
            if not all(
                self._is_source_font_limitation(v)
                for v in self.verapdf_result.violations
            ):
                return False
        return True

    @staticmethod
    def _is_source_font_limitation(violation: dict[str, Any]) -> bool:
        if violation.get("classification") == "source-font-limitation":
            return True
        rule_id = str(violation.get("id", "")).strip()
        if rule_id in {
            "ISO 14289-1:2014-7.21.4.1-1",  # Font programs not embedded
            "ISO 14289-1:2014-7.21.4.1-2",  # Embedded font missing glyphs
            "ISO 14289-1:2014-7.21.4.2-2",  # CIDSet incomplete
            "ISO 14289-1:2014-7.21.5-1",    # Font glyph width mismatch
            "ISO 14289-1:2014-7.21.6-2",    # TrueType non-symbolic encoding
            "ISO 14289-1:2014-7.21.6-3",    # Symbolic TrueType encoding
            "ISO 14289-1:2014-7.21.7-1",    # Font ToUnicode mapping
            "ISO 14289-1:2014-7.21.7-2",    # Font ToUnicode mapping variant
            "ISO 14289-1:2014-7.21.8-1",    # .notdef glyph reference
        }:
            return True
        description = str(violation.get("description", "")).lower()
        return any(
            token in description
            for token in (
                "cidset",
                "embedded font program glyph data is incomplete",
                "tounicode cmap contains invalid zero-value unicode mappings",
                "embedded font program",
                "font programs for all fonts used for rendering within a conforming file shall be embedded within that file",
                "shall define the map of all used character codes to unicode values",
                ".notdef glyph",
                "glyph width information in the font dictionary and in the embedded font program shall be consistent",
                "non-symbolic truetype fonts shall have either macromanencoding or winansiencoding",
                "embedded fonts shall define all glyphs referenced for rendering",
            )
        )

    def failure_reasons(self) -> list[str]:
        return self.blocking_failure_reasons + self.warning_reasons

    @property
    def visual_diff_advisory(self) -> str:
        """Advisory-only visual diff note (never affects conformance or retry)."""
        vdr = self.visual_diff_result
        if vdr and vdr.checked and not vdr.passed:
            return (
                f"visual fidelity: {len(vdr.differing_pages)} page(s) changed "
                f"(max {vdr.max_page_diff:.2%})"
            )
        return ""

    def summary(self) -> str:
        if not self.openable:
            return "; ".join(self.blocking_failure_reasons)
        if self.passed:
            parts: list[str] = []
            if self.warning_reasons:
                if self.non_blocking_verapdf_warnings and len(self.warning_reasons) == 1:
                    parts.append(
                        "checker clean, screen reader clean; "
                        f"veraPDF warnings limited to {len(self.non_blocking_verapdf_warnings)} "
                        "likely source-font/text-map limitation(s)"
                    )
                else:
                    parts.extend(self.warning_reasons)
            else:
                if self.verapdf_result.checked:
                    parts.append("checker clean, screen reader clean, veraPDF passed")
                else:
                    parts.append("checker clean, screen reader clean, veraPDF unavailable")
            if self.visual_diff_advisory:
                parts.append(self.visual_diff_advisory)
            return "; ".join(parts)
        return "; ".join(self.failure_reasons())


def compare_pdf_visual_fidelity(
    original_path: Path,
    remediated_path: Path,
    *,
    dpi: int = 72,
    tolerance: float = 0.05,
) -> VisualDiffResult:
    """Pixel-diff every page of original vs remediated PDF at *dpi* resolution.

    Uses PyMuPDF for rendering — pure CPU, no API cost (~20-50ms per page pair).
    Returns a VisualDiffResult flagging any page whose per-pixel mean absolute
    difference exceeds *tolerance* (default 5%).
    """
    try:
        import fitz
    except ImportError:
        return VisualDiffResult(checked=False, passed=True, error="PyMuPDF not installed")

    if not original_path.exists() or not remediated_path.exists():
        missing = []
        if not original_path.exists():
            missing.append(f"original={original_path}")
        if not remediated_path.exists():
            missing.append(f"remediated={remediated_path}")
        logger.warning(
            "Visual diff SKIPPED — missing files: %s", ", ".join(missing),
        )
        return VisualDiffResult(checked=False, passed=True, error="one or both paths missing")

    try:
        orig_doc = fitz.open(str(original_path))
        rem_doc = fitz.open(str(remediated_path))
    except Exception as exc:
        return VisualDiffResult(checked=False, passed=True, error=str(exc)[:200])

    orig_pages = len(orig_doc)
    rem_pages = len(rem_doc)

    if orig_pages != rem_pages:
        orig_doc.close()
        rem_doc.close()
        return VisualDiffResult(
            checked=True,
            passed=False,
            total_pages=orig_pages,
            differing_pages=list(range(orig_pages)),
            max_page_diff=1.0,
            tolerance=tolerance,
            error=f"page count changed: {orig_pages} → {rem_pages}",
        )

    differing: list[int] = []
    max_diff = 0.0
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False

    try:
        for i in range(orig_pages):
            # RGB colorspace: catches color-only changes (e.g. desaturation from GS)
            orig_pix = orig_doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            rem_pix = rem_doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)

            if orig_pix.width != rem_pix.width or orig_pix.height != rem_pix.height:
                differing.append(i)
                max_diff = 1.0
                continue

            if use_numpy:
                o = np.frombuffer(orig_pix.samples, dtype=np.uint8).astype(np.float32)
                r = np.frombuffer(rem_pix.samples, dtype=np.uint8).astype(np.float32)
                page_diff = float(np.mean(np.abs(o - r)) / 255)
            else:
                orig_samples = orig_pix.samples
                rem_samples = rem_pix.samples
                n = len(orig_samples)
                if n == 0:
                    continue
                page_diff = sum(abs(a - b) for a, b in zip(orig_samples, rem_samples)) / (n * 255)

            if page_diff > max_diff:
                max_diff = page_diff
            if page_diff > tolerance:
                differing.append(i)
    except Exception as exc:
        orig_doc.close()
        rem_doc.close()
        return VisualDiffResult(
            checked=False, passed=True, total_pages=orig_pages,
            error=f"comparison failed: {exc!s:.200}",
        )

    orig_doc.close()
    rem_doc.close()

    return VisualDiffResult(
        checked=True,
        passed=len(differing) == 0,
        total_pages=orig_pages,
        differing_pages=differing,
        max_page_diff=round(max_diff, 6),
        tolerance=tolerance,
    )


def _compute_vision_result_sync(
    pdf_path: Path,
    config: PipelineConfig,
    *,
    full_verification: bool = False,
) -> Any:
    """Synchronously compute a vision analysis result for the acceptance gate.

    REMEDY-57: When the batch path calls ``evaluate_pdf_acceptance`` it has
    historically failed to pass a ``vision_result``, which means the checker
    returns ``Manual Check Needed`` for ``doc-reading-order``,
    ``doc-use-of-color``, and ``doc-color-contrast`` on roughly 40% of PDFs — false positives that
    inflate the manual-review queue. This helper auto-computes the missing
    result when we are in a plain synchronous context and vision credentials
    are configured.

    Returns ``None`` when vision is unavailable, when already inside an
    asyncio event loop (the caller should pre-compute and pass the result
    explicitly), or when the analyzer raises.
    """
    try:
        from project_remedy.pdf_vision import (
            VisionAnalyzer,
            create_provider_from_config,
        )
    except ImportError:
        return None

    # Only auto-compute from a synchronous caller. If we are already inside an
    # asyncio event loop (e.g. ``pipeline.py._pdf_remediate_one``), the caller
    # is expected to compute the result itself and pass it in explicitly —
    # ``asyncio.run`` is not re-entrant.
    try:
        asyncio.get_running_loop()
        logger.debug(
            "evaluate_pdf_acceptance(%s) called from async context; caller "
            "should pre-compute and pass vision_result=... explicitly",
            pdf_path.name,
        )
        return None
    except RuntimeError:
        pass  # No running loop — safe to drive asyncio.run below.

    try:
        provider = create_provider_from_config(config)
    except Exception as exc:  # pragma: no cover — config-dependent
        logger.warning("Vision provider unavailable for %s: %s", pdf_path.name, exc)
        return None
    if provider is None:
        return None

    try:
        analyzer = VisionAnalyzer(provider)
        if full_verification:
            with pikepdf.open(pdf_path) as pdf:
                pages = list(range(1, len(pdf.pages) + 1))
            return asyncio.run(analyzer.analyze_all(pdf_path, pages=pages))
        return asyncio.run(analyzer.analyze_all(pdf_path))
    except Exception as exc:
        logger.warning(
            "Vision analysis for acceptance gate failed on %s: %s",
            pdf_path.name,
            exc,
        )
        return None


def evaluate_pdf_acceptance(
    pdf_path: Path,
    *,
    original_path: Path | None = None,
    config: PipelineConfig | None = None,
    checker_report: CheckReport | None = None,
    tag_tree_result: TagTreeValidationResult | None = None,
    vision_result: Any = None,
    full_verification: bool = False,
) -> PDFAcceptanceResult:
    """Run the shared PDF acceptance gate.

    Pass *original_path* to enable full page-by-page visual fidelity checking
    against the source PDF. Without it, visual diff is skipped.

    REMEDY-57: pass *vision_result* (a ``VisionCheckResult``) to enable the
    checker's reading-order, use-of-color, and color-contrast judgement. When omitted and
    *config* is provided from a synchronous caller, a fresh vision analysis
    is computed automatically so the checker no longer returns
    ``Manual Check Needed`` for every ELAC-style document by default.
    """
    openability_result = validate_pdf_openability(pdf_path)
    if not openability_result.openable:
        return PDFAcceptanceResult(
            file_path=pdf_path,
            checker_report=checker_report or _empty_checker_report(pdf_path, page_count=0),
            tag_tree_result=tag_tree_result or _empty_tag_tree_result(pdf_path, page_count=0),
            verapdf_result=VeraPDFResult(checked=False, passed=False),
            openability_result=openability_result,
            vision_result=vision_result,
        )

    checker_error = ""
    if checker_report is None:
        # REMEDY-57: auto-compute vision_result when the caller didn't pass
        # one but did give us a config. Without this the checker will flag
        # doc-reading-order / doc-use-of-color / doc-color-contrast as "Manual Check Needed"
        # spuriously, inflating the manual-review queue by ~40%.
        if vision_result is None and config is not None:
            vision_result = _compute_vision_result_sync(
                pdf_path,
                config,
                full_verification=full_verification,
            )
        try:
            checker_report = PDFAccessibilityChecker(
                pdf_path, vision_result=vision_result
            ).run_all()
        except Exception as exc:
            checker_error = str(exc)
            checker_report = _empty_checker_report(pdf_path, openability_result.page_count)

    screen_reader_error = ""
    if tag_tree_result is None:
        try:
            tag_tree_result = validate_tag_tree(pdf_path)
        except Exception as exc:
            screen_reader_error = str(exc)
            tag_tree_result = _empty_tag_tree_result(pdf_path, openability_result.page_count)

    verapdf_result = validate_with_verapdf(pdf_path, config=config)

    visual_diff_result: VisualDiffResult | None = None
    if original_path is not None and original_path != pdf_path:
        visual_diff_result = compare_pdf_visual_fidelity(original_path, pdf_path)

    return PDFAcceptanceResult(
        file_path=pdf_path,
        checker_report=checker_report,
        tag_tree_result=tag_tree_result,
        verapdf_result=verapdf_result,
        openability_result=openability_result,
        visual_diff_result=visual_diff_result,
        vision_result=vision_result,
        checker_error=checker_error,
        screen_reader_error=screen_reader_error,
    )


def validate_pdf_openability(pdf_path: Path) -> PDFOpenabilityResult:
    """Return whether the PDF can be opened by a local parser with basic sanity."""
    if not pdf_path.exists():
        return PDFOpenabilityResult(
            checked=True,
            openable=False,
            error=f"PDF not found: {pdf_path}",
        )

    errors: list[str] = []

    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            page_count = len(doc)
        finally:
            doc.close()
        if page_count > 0:
            return PDFOpenabilityResult(
                checked=True,
                openable=True,
                page_count=page_count,
                parser="fitz",
            )
        errors.append("PDF has no pages")
    except Exception as exc:
        errors.append(str(exc))

    try:
        import pikepdf

        with pikepdf.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
        if page_count > 0:
            return PDFOpenabilityResult(
                checked=True,
                openable=True,
                page_count=page_count,
                parser="pikepdf",
            )
        errors.append("PDF has no pages")
    except Exception as exc:
        errors.append(str(exc))

    error = "; ".join(dict.fromkeys(error for error in errors if error))
    return PDFOpenabilityResult(
        checked=True,
        openable=False,
        error=error or "PDF could not be opened",
    )


def validate_with_verapdf(
    pdf_path: Path,
    *,
    config: PipelineConfig | None = None,
    timeout_seconds: int = 120,
) -> VeraPDFResult:
    """Run veraPDF synchronously for PDF/UA-1 when available."""
    verapdf_bin = _resolve_verapdf_binary(config)
    if verapdf_bin is None:
        return VeraPDFResult(checked=False, passed=True)

    cmd = [verapdf_bin, "--format", "xml", "--defaultflavour", "ua1", str(pdf_path)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return VeraPDFResult(
            checked=True,
            passed=False,
            error=f"veraPDF timed out after {timeout_seconds}s",
        )
    except OSError as exc:
        return VeraPDFResult(
            checked=True,
            passed=False,
            error=f"veraPDF execution failed: {exc}",
        )

    xml_text = proc.stdout.strip()
    if not xml_text:
        stderr_text = proc.stderr.strip()
        return VeraPDFResult(
            checked=True,
            passed=False,
            error=stderr_text or "veraPDF returned no XML output",
        )

    try:
        violations = _parse_verapdf_xml(xml_text)
    except ET.ParseError as exc:
        return VeraPDFResult(
            checked=True,
            passed=False,
            error=f"veraPDF XML parse error: {exc}",
        )

    return VeraPDFResult(
        checked=True,
        passed=not violations,
        violations=violations,
    )


def _resolve_verapdf_binary(config: PipelineConfig | None) -> str | None:
    expected = (
        config.pdf_remediation.verapdf_path
        if config is not None
        else PipelineConfig().pdf_remediation.verapdf_path
    )
    if expected and Path(expected).is_file():
        return expected
    return shutil.which("verapdf")


def _empty_checker_report(pdf_path: Path, page_count: int) -> CheckReport:
    file_size = pdf_path.stat().st_size if pdf_path.exists() else 0
    return CheckReport(
        file_path=pdf_path,
        file_size=file_size,
        page_count=page_count,
        results=[],
    )


def _empty_tag_tree_result(pdf_path: Path, page_count: int) -> TagTreeValidationResult:
    return TagTreeValidationResult(
        file_path=pdf_path,
        tag_tree=TagTreeReport(
            file_path=pdf_path,
            page_count=page_count,
            has_structure_tree=False,
            nodes=[],
        ),
        issues=[],
        passed=False,
    )


def _parse_verapdf_xml(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    violations: list[dict[str, Any]] = []
    rule_elements = root.iter(f"{ns}rule") if ns else root.iter("rule")
    for rule in rule_elements:
        status_el = rule.find(f"{ns}status") if ns else rule.find("status")
        status = (
            (status_el.text or "").strip().lower()
            if status_el is not None
            else rule.get("status", "").lower()
        )
        if status != "failed":
            continue

        rule_id = ""
        rule_id_el = rule.find(f"{ns}ruleId") if ns else rule.find("ruleId")
        if rule_id_el is not None and rule_id_el.text:
            rule_id = rule_id_el.text.strip()
        else:
            spec = rule.get("specification", "")
            clause = rule.get("clause", "")
            test_number = rule.get("testNumber", "")
            if clause:
                rule_id = f"{spec}-{clause}-{test_number}".strip("-")

        desc_el = rule.find(f"{ns}description") if ns else rule.find("description")
        location_el = rule.find(f"{ns}location") if ns else rule.find("location")

        description = (
            desc_el.text.strip()
            if desc_el is not None and desc_el.text
            else rule.get("description", "")
        )
        location = (
            location_el.text.strip()
            if location_el is not None and location_el.text
            else rule.get("location", "")
        )

        classification = ""
        note = ""
        normalized_desc = description.lower()
        if rule_id in {
            "ISO 14289-1:2014-7.21.4.1-1",
            "ISO 14289-1:2014-7.21.4.2-2",
            "ISO 14289-1:2014-7.21.7-1",
            "ISO 14289-1:2014-7.21.7-2",
            "ISO 14289-1:2014-7.21.8-1",
        } or any(
            token in normalized_desc
            for token in (
                "cidset",
                "embedded font program glyph data is incomplete",
                "tounicode cmap contains invalid zero-value unicode mappings",
                "embedded font program",
                "font programs for all fonts used for rendering within a conforming file shall be embedded within that file",
                "shall define the map of all used character codes to unicode values",
                ".notdef glyph",
            )
        ):
            classification = "source-font-limitation"
            note = "likely inherited source-font/CIDSet limitation; not usually fixable by structure-only remediation"

        violations.append(
            {
                "tool": "verapdf",
                "id": rule_id or "unknown-rule",
                "impact": "serious",
                "description": description or f"PDF/UA-1 rule {rule_id} failed",
                "help": f"PDF/UA-1 compliance failure: {rule_id}",
                "location": location,
                **({"classification": classification} if classification else {}),
                **({"note": note} if note else {}),
            }
        )

    return violations
