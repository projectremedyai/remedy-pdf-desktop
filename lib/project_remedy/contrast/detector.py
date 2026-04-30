"""AI-driven contrast detection — renders pages, sends to vision AI, parses results."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

from project_remedy.contrast.color_utils import (
    contrast_ratio,
    hex_to_rgb,
    is_large_text,
    wcag_threshold,
    nearest_passing_color,
    NON_TEXT_THRESHOLD,
)
from project_remedy.contrast.models import (
    ContrastIssue,
    ContrastIssueType,
    CONTRAST_DETECTION_SCHEMA,
    CONTRAST_VALIDATION_SCHEMA,
)
from project_remedy.contrast.prompts import (
    contrast_detection_prompt,
    contrast_validation_prompt,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]


def _render_page(pdf_path: str | Path, page_index: int, dpi: int = 150) -> bytes:
    """Render a single PDF page to PNG bytes using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    try:
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(f"Page {page_index} out of range (0-{len(doc) - 1})")
        page = doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    finally:
        doc.close()


def _get_page_count(pdf_path: str | Path) -> int:
    """Get the number of pages in a PDF."""
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


class ContrastDetector:
    """Detects contrast issues using AI vision analysis supplemented by programmatic checks.

    Parameters
    ----------
    llm_client:
        A GeminiClient or OllamaClient instance with ``vision()`` and
        ``_generate()`` methods.
    dpi:
        Resolution for page rendering (higher = more accurate, slower).
    """

    def __init__(self, llm_client: Any, dpi: int = 150):
        self._client = llm_client
        self._dpi = dpi

    async def detect_document(
        self,
        pdf_path: str | Path,
        level: str = "AA",
        progress_callback: ProgressCallback | None = None,
    ) -> list[ContrastIssue]:
        """Detect contrast issues across all pages of a document."""
        page_count = _get_page_count(pdf_path)
        all_issues: list[ContrastIssue] = []

        for page_idx in range(page_count):
            if progress_callback:
                pct = page_idx / max(page_count, 1)
                progress_callback(
                    "contrast_detect", pct,
                    f"Scanning page {page_idx + 1}/{page_count} for contrast issues...",
                )

            issues = await self.detect_page(pdf_path, page_idx, level)
            all_issues.extend(issues)

        if progress_callback:
            progress_callback(
                "contrast_detect", 1.0,
                f"Found {len(all_issues)} contrast issues across {page_count} pages",
            )

        return all_issues

    async def detect_page(
        self,
        pdf_path: str | Path,
        page_index: int,
        level: str = "AA",
    ) -> list[ContrastIssue]:
        """Detect contrast issues on a single page using AI vision."""
        image_data = await asyncio.to_thread(
            _render_page, pdf_path, page_index, self._dpi
        )
        prompt = contrast_detection_prompt(level)

        raw = await self._call_vision(image_data, prompt, CONTRAST_DETECTION_SCHEMA)
        if raw is None:
            return []

        issues: list[ContrastIssue] = []
        for item in raw.get("issues", []):
            issue = self._parse_ai_issue(item, page_index, level)
            if issue is not None:
                issues.append(issue)

        return issues

    async def validate_fixes(
        self,
        pdf_path: str | Path,
        page_index: int,
        original_issues: list[ContrastIssue],
        level: str = "AA",
    ) -> list[ContrastIssue]:
        """Re-render page and validate fixes via AI. Returns still-unfixed issues."""
        if not original_issues:
            return []

        image_data = await asyncio.to_thread(
            _render_page, pdf_path, page_index, self._dpi
        )

        # Build issue descriptions for the validation prompt
        descriptions = []
        for issue in original_issues:
            desc = (
                f"- ID: {issue.id}, Type: {issue.issue_type.value}, "
                f"Location: [{', '.join(f'{v:.0f}' for v in issue.bbox)}], "
                f"Description: {issue.description or issue.text_content}"
            )
            descriptions.append(desc)

        prompt = contrast_validation_prompt(level, "\n".join(descriptions))
        raw = await self._call_vision(image_data, prompt, CONTRAST_VALIDATION_SCHEMA)

        if raw is None:
            # If AI call fails, assume none were fixed (conservative)
            return original_issues

        # Parse validation results
        fixed_ids: set[str] = set()
        if raw.get("all_fixed"):
            return []

        for result in raw.get("results", []):
            if result.get("fixed"):
                fixed_ids.add(result.get("issue_id", ""))

        # Return issues that were NOT fixed
        unfixed = []
        for issue in original_issues:
            if issue.id not in fixed_ids:
                issue.fix_attempts += 1
                unfixed.append(issue)
            else:
                issue.fixed = True

        return unfixed

    async def _call_vision(
        self,
        image_data: bytes,
        prompt: str,
        schema: dict,
    ) -> dict | None:
        """Call the AI vision model and parse JSON response.

        Uses GeminiClient's internal _generate() with response_schema
        for structured output. Falls back to vision() + manual JSON
        parsing if _generate() is unavailable.
        """
        try:
            from google.genai import types

            # Write image to a temp file for the vision call
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(image_data)
                tmp_path = Path(tmp.name)

            try:
                # Use GeminiClient's _generate() for structured JSON output
                parts = [
                    types.Part.from_bytes(data=image_data, mime_type="image/png"),
                    prompt,
                ]
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                    max_output_tokens=4096,
                )
                response_text = await self._client._generate(
                    contents=parts,
                    config=config,
                )
                return json.loads(response_text)
            finally:
                tmp_path.unlink(missing_ok=True)

        except AttributeError:
            # Fallback: client doesn't have _generate() (e.g. OllamaClient)
            # Use vision() and parse JSON manually
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(image_data)
                    tmp_path = Path(tmp.name)

                try:
                    response_text = await self._client.vision(
                        image_path=tmp_path,
                        prompt=prompt + "\n\nReturn ONLY valid JSON.",
                    )
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                    if json_match:
                        return json.loads(json_match.group(0))
                    return None
                finally:
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.exception("AI vision call failed for contrast detection")
                return None

        except Exception:
            logger.exception("AI vision call failed for contrast detection")
            return None

    def _parse_ai_issue(
        self,
        item: dict,
        page_index: int,
        level: str,
    ) -> ContrastIssue | None:
        """Parse a single AI-detected issue into a ContrastIssue model."""
        try:
            issue_type = ContrastIssueType(item.get("issue_type", "text"))

            fg_hex = item.get("fg_color_hex", "#000000")
            bg_hex = item.get("bg_color_hex", "#ffffff")
            try:
                fg = hex_to_rgb(fg_hex)
                bg = hex_to_rgb(bg_hex)
            except ValueError:
                fg = (0.0, 0.0, 0.0)
                bg = (1.0, 1.0, 1.0)

            ratio = contrast_ratio(fg, bg)

            is_large = item.get("is_large_text", False)
            bold = item.get("is_bold", False)
            font_size = item.get("estimated_font_size")
            if font_size is not None:
                is_large = is_large_text(font_size, bold)

            # Determine required ratio
            if issue_type == ContrastIssueType.TEXT:
                required = wcag_threshold(level, is_large)
                criterion = "1.4.6" if level.upper() == "AAA" else "1.4.3"
            else:
                required = NON_TEXT_THRESHOLD
                criterion = "1.4.11"

            # Compute suggested fix
            suggested_fg = None
            if ratio < required:
                suggested_fg = nearest_passing_color(fg, bg, required)

            return ContrastIssue(
                id=uuid.uuid4().hex[:12],
                issue_type=issue_type,
                page_index=page_index,
                bbox=item.get("bbox", []),
                fg_color=fg,
                bg_color=bg,
                contrast_ratio=round(ratio, 2),
                required_ratio=required,
                wcag_criterion=criterion,
                is_large_text=is_large,
                font_size=font_size,
                is_bold=bold,
                text_content=item.get("text_content", ""),
                description=item.get("description", ""),
                suggested_fg=suggested_fg,
            )
        except Exception:
            logger.debug("Failed to parse AI contrast issue: %s", item)
            return None
