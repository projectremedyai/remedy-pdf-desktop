"""Contrast remediator — the AI-driven detect -> fix -> validate loop.

For each page:
  1. Detect contrast issues via AI
  2. Apply programmatic fixes (text colors, image enhancement, graphic fills)
  3. Re-render and validate via AI
  4. Repeat up to MAX_ITERATIONS if fixes didn't pass
  5. Flag remaining issues for manual review
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Callable

import pikepdf
from pikepdf import Name

from project_remedy.content_stream.parser import GraphicsStateTracker
from project_remedy.content_stream.modifier import ContentStreamModifier, ColorModification
from project_remedy.contrast.color_utils import (
    cmyk_to_rgb,
    gray_to_rgb,
    nearest_passing_color,
    contrast_ratio,
)
from project_remedy.contrast.detector import ContrastDetector, _get_page_count
from project_remedy.contrast.image_enhancer import ImageContrastEnhancer
from project_remedy.contrast.models import (
    ContrastAnalysis,
    ContrastIssue,
    ContrastIssueType,
    PageContrastResult,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]


class ContrastRemediator:
    """Full AI-driven contrast remediation with validation loop.

    Parameters
    ----------
    llm_client:
        A GeminiClient or OllamaClient instance.
    dpi:
        Resolution for page rendering.
    """

    MAX_ITERATIONS = 3

    def __init__(self, llm_client: Any, dpi: int = 150):
        self._detector = ContrastDetector(llm_client, dpi)
        self._tracker = GraphicsStateTracker()
        self._modifier = ContentStreamModifier()
        self._image_enhancer = ImageContrastEnhancer()
        self._dpi = dpi

    async def remediate_document(
        self,
        pdf_path: str,
        output_path: str,
        level: str = "AA",
        progress_callback: ProgressCallback | None = None,
    ) -> ContrastAnalysis:
        """Run full contrast remediation on a document."""
        analysis = ContrastAnalysis()
        page_count = _get_page_count(pdf_path)

        for page_idx in range(page_count):
            if progress_callback:
                pct = page_idx / max(page_count, 1)
                progress_callback(
                    "contrast", pct,
                    f"Contrast: analyzing page {page_idx + 1}/{page_count}...",
                )

            page_result = await self._remediate_page(
                pdf_path, output_path, page_idx, level
            )
            analysis.pages.append(page_result)

        analysis.compute_totals()

        if progress_callback:
            progress_callback(
                "contrast", 1.0,
                f"Contrast: fixed {analysis.issues_fixed}/{analysis.total_issues} issues",
            )

        return analysis

    async def _remediate_page(
        self,
        pdf_path: str,
        output_path: str,
        page_index: int,
        level: str,
    ) -> PageContrastResult:
        """Detect -> fix -> validate loop for a single page."""
        # Step 1: Detect
        issues = await self._detector.detect_page(pdf_path, page_index, level)
        if not issues:
            return PageContrastResult(page_index=page_index)

        total_found = len(issues)
        unfixed = list(issues)

        for iteration in range(self.MAX_ITERATIONS):
            if not unfixed:
                break

            logger.info(
                "Page %d, iteration %d: %d issues to fix",
                page_index, iteration + 1, len(unfixed),
            )

            # Step 2: Apply fixes (blocking pikepdf operations in thread)
            work_path = output_path
            fixed_count = await asyncio.to_thread(
                self._apply_fixes, work_path, page_index, unfixed, iteration
            )

            if fixed_count == 0:
                logger.info("No fixes could be applied on iteration %d", iteration + 1)
                break

            # Step 3: Validate
            unfixed = await self._detector.validate_fixes(
                work_path, page_index, unfixed, level
            )

        fixed = total_found - len(unfixed)
        return PageContrastResult(
            page_index=page_index,
            issues=issues,
            issues_fixed=fixed,
            issues_remaining=len(unfixed),
        )

    def _apply_fixes(
        self,
        pdf_path: str,
        page_index: int,
        issues: list[ContrastIssue],
        iteration: int,
    ) -> int:
        """Apply programmatic fixes for a set of issues. Returns count of fixes applied."""
        pdf = pikepdf.open(pdf_path, allow_overwriting_input=True)
        fixed = 0
        try:
            for issue in issues:
                success = False
                if issue.issue_type == ContrastIssueType.TEXT:
                    success = self._fix_text_issue(pdf, page_index, issue, iteration)
                elif issue.issue_type == ContrastIssueType.IMAGE:
                    success = self._fix_image_issue(pdf, page_index, issue, iteration)
                elif issue.issue_type == ContrastIssueType.GRAPHIC:
                    success = self._fix_graphic_issue(pdf, page_index, issue, iteration)

                if success:
                    fixed += 1

            if fixed > 0:
                pdf.save(pdf_path)
        except Exception:
            logger.exception("Failed to apply contrast fixes on page %d", page_index)
        finally:
            pdf.close()

        return fixed

    def _fix_text_issue(
        self,
        pdf: pikepdf.Pdf,
        page_index: int,
        issue: ContrastIssue,
        iteration: int,
    ) -> bool:
        """Fix a text contrast issue by adjusting the color operator in the content stream."""
        if issue.suggested_fg is None:
            return False

        page = pdf.pages[page_index]
        annotations = self._tracker.track(page)

        # Find text-rendering instructions near the issue bbox
        target_idx = self._find_color_for_text(annotations, issue)
        if target_idx is None:
            # No existing color instruction found — insert one before the text
            text_idx = self._find_text_instruction(annotations, issue)
            if text_idx is not None:
                r, g, b = issue.suggested_fg
                self._modifier.insert_color_before(page, text_idx, (r, g, b), "rg")
                return True
            return False

        # Replace the existing color instruction
        ann = annotations[target_idx]
        r, g, b = issue.suggested_fg

        if ann.state.fill_colorspace == "DeviceGray":
            gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
            self._modifier.replace_color_at(page, target_idx, (gray,), "g")
        elif ann.state.fill_colorspace == "DeviceCMYK":
            c = 1.0 - r
            m = 1.0 - g
            y = 1.0 - b
            k = min(c, m, y)
            if k < 1.0:
                c = (c - k) / (1.0 - k)
                m = (m - k) / (1.0 - k)
                y = (y - k) / (1.0 - k)
            else:
                c = m = y = 0.0
            self._modifier.replace_color_at(page, target_idx, (c, m, y, k), "k")
        else:
            self._modifier.replace_color_at(page, target_idx, (r, g, b), "rg")

        return True

    def _fix_image_issue(
        self,
        pdf: pikepdf.Pdf,
        page_index: int,
        issue: ContrastIssue,
        iteration: int,
    ) -> bool:
        """Fix an image contrast issue by enhancing and re-embedding."""
        page = pdf.pages[page_index]
        resources = page.obj.get("/Resources", pikepdf.Dictionary())
        xobjects = resources.get("/XObject", pikepdf.Dictionary())

        target_name = self._find_image_xobject(page, xobjects, issue)
        if target_name is None:
            return False

        xobj = xobjects[target_name]
        try:
            image_data = xobj.read_bytes()
            width = int(xobj.get("/Width", 0))
            height = int(xobj.get("/Height", 0))

            if width == 0 or height == 0:
                return False

            from PIL import Image

            cs = str(xobj.get("/ColorSpace", "/DeviceRGB"))
            if "Gray" in cs:
                mode = "L"
                expected_len = width * height
            else:
                mode = "RGB"
                expected_len = width * height * 3

            if len(image_data) < expected_len:
                return False

            img = Image.frombytes(mode, (width, height), image_data[:expected_len])
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            factor = 1.3 + (iteration * 0.2)
            enhanced = self._image_enhancer.enhance(png_bytes, target_increase=factor)

            enhanced_img = Image.open(io.BytesIO(enhanced))
            if enhanced_img.mode != mode:
                enhanced_img = enhanced_img.convert(mode)

            new_data = enhanced_img.tobytes()
            xobj.write(new_data)

            return True
        except Exception:
            logger.debug("Failed to enhance image %s", target_name)
            return False

    def _fix_graphic_issue(
        self,
        pdf: pikepdf.Pdf,
        page_index: int,
        issue: ContrastIssue,
        iteration: int,
    ) -> bool:
        """Fix a graphic element contrast issue (fill/stroke color adjustment)."""
        if issue.suggested_fg is None:
            return False

        page = pdf.pages[page_index]
        annotations = self._tracker.track(page)

        target_idx = self._find_graphic_color(annotations, issue)
        if target_idx is None:
            return False

        ann = annotations[target_idx]
        r, g, b = issue.suggested_fg

        op = ann.operator
        if op in ("RG", "G", "K", "SC", "SCN"):
            # Stroke color
            if ann.state.stroke_colorspace == "DeviceGray":
                gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
                self._modifier.replace_color_at(page, target_idx, (gray,), "G")
            elif ann.state.stroke_colorspace == "DeviceCMYK":
                c, m, y, k = 1 - r, 1 - g, 1 - b, 0.0
                k = min(c, m, y)
                if k < 1:
                    c, m, y = (c - k) / (1 - k), (m - k) / (1 - k), (y - k) / (1 - k)
                else:
                    c = m = y = 0.0
                self._modifier.replace_color_at(page, target_idx, (c, m, y, k), "K")
            else:
                self._modifier.replace_color_at(page, target_idx, (r, g, b), "RG")
        else:
            # Fill color
            if ann.state.fill_colorspace == "DeviceGray":
                gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
                self._modifier.replace_color_at(page, target_idx, (gray,), "g")
            elif ann.state.fill_colorspace == "DeviceCMYK":
                c, m, y, k = 1 - r, 1 - g, 1 - b, 0.0
                k = min(c, m, y)
                if k < 1:
                    c, m, y = (c - k) / (1 - k), (m - k) / (1 - k), (y - k) / (1 - k)
                else:
                    c = m = y = 0.0
                self._modifier.replace_color_at(page, target_idx, (c, m, y, k), "k")
            else:
                self._modifier.replace_color_at(page, target_idx, (r, g, b), "rg")

        return True

    # --- Helper methods for locating content stream instructions ---

    def _find_color_for_text(
        self,
        annotations: list,
        issue: ContrastIssue,
    ) -> int | None:
        """Find the fill color instruction that sets the color for text at the issue location."""
        text_ops = {"Tj", "TJ", "'", '"'}
        fill_color_ops = {"rg", "g", "k", "sc", "scn"}

        text_indices = [
            i for i, ann in enumerate(annotations)
            if ann.operator in text_ops and ann.state.in_text_block
        ]

        if not text_indices:
            return None

        best_text_idx = self._match_by_position(annotations, text_indices, issue)
        if best_text_idx is None:
            best_text_idx = text_indices[0]

        for i in range(best_text_idx, -1, -1):
            if annotations[i].operator in fill_color_ops:
                return i

        return None

    def _find_text_instruction(
        self,
        annotations: list,
        issue: ContrastIssue,
    ) -> int | None:
        """Find the text instruction at the issue location for insertion."""
        text_ops = {"Tj", "TJ", "'", '"'}
        indices = [
            i for i, ann in enumerate(annotations)
            if ann.operator in text_ops and ann.state.in_text_block
        ]
        return self._match_by_position(annotations, indices, issue)

    def _find_graphic_color(
        self,
        annotations: list,
        issue: ContrastIssue,
    ) -> int | None:
        """Find fill/stroke color instruction for a graphic element."""
        color_ops = {"rg", "RG", "g", "G", "k", "K", "sc", "SC", "scn", "SCN"}
        path_ops = {"f", "F", "f*", "B", "B*", "b", "b*", "S", "s"}

        path_indices = [
            i for i, ann in enumerate(annotations)
            if ann.operator in path_ops
        ]

        best_path_idx = self._match_by_position(annotations, path_indices, issue)
        if best_path_idx is None:
            return None

        for i in range(best_path_idx, -1, -1):
            if annotations[i].operator in color_ops:
                return i

        return None

    def _find_image_xobject(
        self,
        page: pikepdf.Page,
        xobjects: pikepdf.Dictionary,
        issue: ContrastIssue,
    ) -> str | None:
        """Find the image XObject name closest to the issue bbox."""
        annotations = self._tracker.track(page)
        do_indices = [
            i for i, ann in enumerate(annotations)
            if ann.operator == "Do"
        ]

        if not do_indices:
            return None

        image_names = []
        for idx in do_indices:
            ann = annotations[idx]
            if ann.operands:
                name = str(ann.operands[0]).lstrip("/")
                xobj = xobjects.get(name)
                if xobj is not None:
                    subtype = str(xobj.get("/Subtype", ""))
                    if subtype == "/Image":
                        image_names.append(name)

        if len(image_names) == 1:
            return image_names[0]
        if image_names:
            return image_names[0]  # Best effort: return first
        return None

    def _match_by_position(
        self,
        annotations: list,
        candidate_indices: list[int],
        issue: ContrastIssue,
    ) -> int | None:
        """Match an issue's bbox to the closest instruction by text matrix position."""
        if not candidate_indices:
            return None
        if not issue.bbox or len(issue.bbox) < 4:
            return candidate_indices[0] if candidate_indices else None

        # Issue bbox is in page-percentage coordinates
        target_x = issue.bbox[0] / 100.0 * 612
        target_y = (100 - issue.bbox[1]) / 100.0 * 792  # PDF y is bottom-up

        best_idx = None
        best_dist = float("inf")

        for idx in candidate_indices:
            ann = annotations[idx]
            if ann.state.in_text_block:
                x = ann.state.text_matrix[4]
                y = ann.state.text_matrix[5]
            else:
                x = ann.state.ctm[4]
                y = ann.state.ctm[5]

            dist = (x - target_x) ** 2 + (y - target_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        return best_idx
