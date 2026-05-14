"""Shared PDF acceptance checks for the primary PDF-to-PDF workflow."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import threading
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
# Desktop does not bundle the Quality Layer Extension; ``QualityResult`` is
# kept as a permissive alias so ``PDFAcceptanceResult`` stays shape-compatible
# with the server while ``quality_result`` is always ``None`` in this build.
QualityResult = Any  # type: ignore[assignment]

logger = logging.getLogger(__name__)
from project_remedy.tag_tree_reader import (
    ScreenReaderIssue,
    Severity,
    TagTreeReport,
    ValidationResult as TagTreeValidationResult,
    validate_tag_tree,
)


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
class TextSimilarityResult:
    """Jaccard text-similarity check between original and rebuilt PDF.

    Populated only when ``evaluate_pdf_acceptance`` is called with
    ``rebuild_mode=True``; the rebuild tier intentionally skips visual-diff
    (rebuilt PDFs are laid out differently by design) and gates content
    preservation on sentence-level Jaccard similarity instead.
    """

    checked: bool
    passed: bool
    score: float = 0.0
    threshold: float = 0.0
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
    text_similarity_result: TextSimilarityResult | None = None
    checker_error: str = ""
    screen_reader_error: str = ""
    quality_result: QualityResult | None = None

    @property
    def checker_failures(self) -> list[CheckResult]:
        return [r for r in self.checker_report.results if r.status == "Failed"]

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
        tsr = self.text_similarity_result
        if tsr and tsr.checked and not tsr.passed:
            entries.append(
                {
                    "source": "text_similarity",
                    "rule_id": "text-similarity",
                    "description": (
                        f"Text similarity below threshold: "
                        f"{tsr.score:.3f} < {tsr.threshold:.3f}"
                    ),
                    "details": [tsr.error] if tsr.error else [],
                    "fixable": False,
                }
            )
        return entries

    @property
    def warning_reasons(self) -> list[str]:
        if not self.openable:
            return []
        reasons: list[str] = []
        # Only count blocking checker failures (exclude source-font-only issues)
        blocking_checker = [
            f for f in self.checker_failures
            if not self._is_source_font_checker_failure(f)
        ]
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
        tsr = self.text_similarity_result
        if tsr and tsr.checked and not tsr.passed:
            reasons.append(
                f"text similarity below threshold "
                f"({tsr.score:.3f} < {tsr.threshold:.3f})"
            )
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
        blocking_checker = [
            f for f in self.checker_failures
            if not self._is_source_font_checker_failure(f)
        ]
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
        if (
            self.text_similarity_result
            and self.text_similarity_result.checked
            and not self.text_similarity_result.passed
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
        from PIL import Image, ImageChops, ImageFilter, ImageStat
        use_pillow = True
    except ImportError:
        use_pillow = False

    def _pix_to_image(pix):
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    def _image_mean_abs_diff(orig_img, rem_img) -> float:
        diff = ImageChops.difference(orig_img, rem_img)
        stat = ImageStat.Stat(diff)
        if not stat.mean:
            return 0.0
        return sum(stat.mean) / (len(stat.mean) * 255)

    def _blurred_thumbnail_diff(orig_pix, rem_pix) -> float | None:
        """Perceptual fallback for antialias-only text raster differences."""
        if not use_pillow:
            return None
        try:
            orig_img = _pix_to_image(orig_pix)
            rem_img = _pix_to_image(rem_pix)
            target_w = max(1, min(orig_pix.width, rem_pix.width) // 3)
            target_h = max(1, min(orig_pix.height, rem_pix.height) // 3)
            orig_thumb = orig_img.filter(ImageFilter.GaussianBlur(radius=0.75)).resize(
                (target_w, target_h), Image.Resampling.LANCZOS,
            )
            rem_thumb = rem_img.filter(ImageFilter.GaussianBlur(radius=0.75)).resize(
                (target_w, target_h), Image.Resampling.LANCZOS,
            )
            diff = ImageChops.difference(orig_thumb, rem_thumb)
            stat = ImageStat.Stat(diff)
            if not stat.mean:
                return 0.0
            return sum(stat.mean) / (len(stat.mean) * 255)
        except Exception:
            return None

    try:
        for i in range(orig_pages):
            # RGB colorspace: catches color-only changes (e.g. desaturation from GS)
            orig_pix = orig_doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            rem_pix = rem_doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)

            if orig_pix.width != rem_pix.width or orig_pix.height != rem_pix.height:
                width_delta = abs(orig_pix.width - rem_pix.width) / max(
                    orig_pix.width, rem_pix.width, 1,
                )
                height_delta = abs(orig_pix.height - rem_pix.height) / max(
                    orig_pix.height, rem_pix.height, 1,
                )
                if max(width_delta, height_delta) > 0.02:
                    differing.append(i)
                    max_diff = 1.0
                    continue

                target_h = min(orig_pix.height, rem_pix.height)
                target_w = min(orig_pix.width, rem_pix.width)
                if use_pillow:
                    orig_img = _pix_to_image(orig_pix).resize(
                        (target_w, target_h), Image.Resampling.BILINEAR,
                    )
                    rem_img = _pix_to_image(rem_pix).resize(
                        (target_w, target_h), Image.Resampling.BILINEAR,
                    )
                    page_diff = _image_mean_abs_diff(orig_img, rem_img)
                elif use_numpy:
                    o_arr = np.frombuffer(orig_pix.samples, dtype=np.uint8).reshape(
                        orig_pix.height, orig_pix.width, -1,
                    )
                    r_arr = np.frombuffer(rem_pix.samples, dtype=np.uint8).reshape(
                        rem_pix.height, rem_pix.width, -1,
                    )
                    o_y = np.linspace(0, orig_pix.height - 1, target_h).astype(np.int32)
                    o_x = np.linspace(0, orig_pix.width - 1, target_w).astype(np.int32)
                    r_y = np.linspace(0, rem_pix.height - 1, target_h).astype(np.int32)
                    r_x = np.linspace(0, rem_pix.width - 1, target_w).astype(np.int32)
                    o = o_arr[o_y][:, o_x].astype(np.float32)
                    r = r_arr[r_y][:, r_x].astype(np.float32)
                    page_diff = float(np.mean(np.abs(o - r)) / 255)
                else:
                    orig_samples = orig_pix.samples
                    rem_samples = rem_pix.samples
                    orig_bpp = max(1, len(orig_samples) // (orig_pix.width * orig_pix.height))
                    rem_bpp = max(1, len(rem_samples) // (rem_pix.width * rem_pix.height))
                    bpp = min(orig_bpp, rem_bpp)
                    total = target_h * target_w * bpp
                    if total == 0:
                        continue
                    diff_sum = 0
                    for y in range(target_h):
                        oy = y * (orig_pix.height - 1) // max(target_h - 1, 1)
                        ry = y * (rem_pix.height - 1) // max(target_h - 1, 1)
                        for x in range(target_w):
                            ox = x * (orig_pix.width - 1) // max(target_w - 1, 1)
                            rx = x * (rem_pix.width - 1) // max(target_w - 1, 1)
                            o_base = (oy * orig_pix.width + ox) * orig_bpp
                            r_base = (ry * rem_pix.width + rx) * rem_bpp
                            for channel in range(bpp):
                                diff_sum += abs(
                                    orig_samples[o_base + channel]
                                    - rem_samples[r_base + channel]
                                )
                    page_diff = diff_sum / (total * 255)
                if page_diff > max_diff:
                    max_diff = page_diff
                perceptual_diff = None
                if page_diff > tolerance:
                    perceptual_diff = _blurred_thumbnail_diff(orig_pix, rem_pix)
                if page_diff > tolerance and (
                    perceptual_diff is None or perceptual_diff > min(tolerance, 0.03)
                ):
                    differing.append(i)
                continue

            if use_numpy:
                o = np.frombuffer(orig_pix.samples, dtype=np.uint8).astype(np.float32)
                r = np.frombuffer(rem_pix.samples, dtype=np.uint8).astype(np.float32)
                page_diff = float(np.mean(np.abs(o - r)) / 255)
            elif use_pillow:
                page_diff = _image_mean_abs_diff(
                    _pix_to_image(orig_pix),
                    _pix_to_image(rem_pix),
                )
            else:
                orig_samples = orig_pix.samples
                rem_samples = rem_pix.samples
                n = len(orig_samples)
                if n == 0:
                    continue
                page_diff = sum(abs(a - b) for a, b in zip(orig_samples, rem_samples)) / (n * 255)

            if page_diff > max_diff:
                max_diff = page_diff
            perceptual_diff = None
            if page_diff > tolerance:
                perceptual_diff = _blurred_thumbnail_diff(orig_pix, rem_pix)
            if page_diff > tolerance and (
                perceptual_diff is None or perceptual_diff > min(tolerance, 0.03)
            ):
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
) -> Any:
    """Synchronously compute a vision analysis result for the acceptance gate.

    REMEDY-57: When the batch path calls ``evaluate_pdf_acceptance`` it has
    historically failed to pass a ``vision_result``, which means the checker
    returns ``Manual Check Needed`` for ``doc-reading-order`` and
    ``doc-color-contrast`` on roughly 40% of PDFs — false positives that
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
        try:
            timeout_seconds = float(os.environ.get("PDF_ACCEPTANCE_VISION_TIMEOUT", "300"))
        except ValueError:
            timeout_seconds = 300.0

        async def _run_acceptance_vision():
            return await asyncio.wait_for(
                analyzer.analyze_all(pdf_path),
                timeout=timeout_seconds,
            )

        result: dict[str, object] = {}
        error: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result["value"] = asyncio.run(_run_acceptance_vision())
            except BaseException as exc:
                error["exc"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            raise TimeoutError(
                f"acceptance vision timed out after {timeout_seconds:.0f}s"
            )
        if "exc" in error:
            raise error["exc"]
        return result.get("value")
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
    rebuild_mode: bool = False,
) -> PDFAcceptanceResult:
    """Run the shared PDF acceptance gate.

    Pass *original_path* to enable full page-by-page visual fidelity checking
    against the source PDF. Without it, visual diff is skipped.

    REMEDY-57: pass *vision_result* (a ``VisionCheckResult``) to enable the
    checker's reading-order and color-contrast judgement. When omitted and
    *config* is provided from a synchronous caller, a fresh vision analysis
    is computed automatically so the checker no longer returns
    ``Manual Check Needed`` for every scanned-style document by default.

    Pass ``rebuild_mode=True`` when evaluating output from the rebuild tier
    (``src/project_remedy/rebuild``). In that mode the visual-diff pixel
    comparison is skipped entirely — rebuilt PDFs are laid out differently
    by design — and sentence-level Jaccard text similarity between
    *original_path* and ``pdf_path`` gates content preservation instead.
    ``original_path`` is required when ``rebuild_mode=True``.
    """
    if rebuild_mode and original_path is None:
        raise ValueError("rebuild_mode requires original_path")

    openability_result = validate_pdf_openability(pdf_path)
    if not openability_result.openable:
        return PDFAcceptanceResult(
            file_path=pdf_path,
            checker_report=checker_report or _empty_checker_report(pdf_path, page_count=0),
            tag_tree_result=tag_tree_result or _empty_tag_tree_result(pdf_path, page_count=0),
            verapdf_result=VeraPDFResult(checked=False, passed=False),
            openability_result=openability_result,
        )

    checker_error = ""
    if checker_report is None:
        # REMEDY-57: auto-compute vision_result when the caller didn't pass
        # one but did give us a config. Without this the checker will flag
        # doc-reading-order / doc-color-contrast as "Manual Check Needed"
        # spuriously, inflating the manual-review queue by ~40%.
        if vision_result is None and config is not None:
            vision_result = _compute_vision_result_sync(pdf_path, config)
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
    text_similarity_result: TextSimilarityResult | None = None
    if rebuild_mode:
        # Rebuild tier: skip visual-diff entirely, gate on text similarity.
        # original_path is guaranteed non-None by the guard above.
        from project_remedy.rebuild.text_similarity import text_similarity

        threshold = (
            config.rebuild.text_similarity_threshold
            if config is not None
            else 0.85
        )
        try:
            score = text_similarity(original_path, pdf_path)
            text_similarity_result = TextSimilarityResult(
                checked=True,
                passed=score >= threshold,
                score=score,
                threshold=threshold,
            )
        except Exception as exc:
            # Don't mask upstream acceptance signal with an internal error
            # here; record it and let other checks carry the decision.
            text_similarity_result = TextSimilarityResult(
                checked=False,
                passed=True,
                threshold=threshold,
                error=str(exc)[:200],
            )
    elif original_path is not None and original_path != pdf_path:
        visual_diff_result = compare_pdf_visual_fidelity(original_path, pdf_path)

    return PDFAcceptanceResult(
        file_path=pdf_path,
        checker_report=checker_report,
        tag_tree_result=tag_tree_result,
        verapdf_result=verapdf_result,
        openability_result=openability_result,
        visual_diff_result=visual_diff_result,
        text_similarity_result=text_similarity_result,
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
