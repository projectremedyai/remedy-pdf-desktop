"""Detect and extract math formulas from PDF pages using vision."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf

logger = logging.getLogger(__name__)


def _parse_json_response(text: str) -> dict:
    """Parse JSON from model output, handling unescaped LaTeX backslashes."""
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Fix unescaped backslashes in LaTeX (e.g. \frac → \\frac)
    # Match backslash followed by a letter that isn't a valid JSON escape
    text = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', text)

    return json.loads(text)

MATH_DETECTION_PROMPT = """\
Analyze this PDF page image. Identify ALL mathematical formulas, equations, \
and expressions. For each formula found, return:
1. The LaTeX representation
2. A plain-English readable description (for screen readers)
3. The approximate location on the page (top/middle/bottom, left/center/right)

Return JSON only:
{
  "formulas": [
    {
      "latex": "E = mc^2",
      "alt_text": "E equals m c squared",
      "location": "middle-center",
      "inline": false
    }
  ],
  "formula_count": 1
}

If no formulas are found, return {"formulas": [], "formula_count": 0}.
JSON only, no markdown fences."""


@dataclass
class DetectedFormula:
    """A math formula found on a PDF page."""

    latex: str
    alt_text: str
    page: int
    location: str = ""
    inline: bool = False


@dataclass
class ExtractionResult:
    """All formulas found in a document."""

    formulas: list[DetectedFormula] = field(default_factory=list)
    pages_scanned: int = 0
    errors: list[str] = field(default_factory=list)


def render_page(pdf_path: Path, page_num: int, dpi: int = 200) -> Path:
    """Render a single page to a temporary PNG."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    tmp = Path(tempfile.mktemp(suffix=".png"))
    pix.save(str(tmp))
    doc.close()
    return tmp


def extract_formulas(
    pdf_path: Path,
    vision_provider,
    *,
    pages: list[int] | None = None,
) -> ExtractionResult:
    """Scan PDF pages for math formulas using the vision model.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    vision_provider:
        A VisionProvider instance backed by the app's configured local vision runtime.
    pages:
        Specific 0-indexed pages to scan. None = all pages.
    """
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    if pages is None:
        pages = list(range(page_count))

    result = ExtractionResult(pages_scanned=len(pages))

    for page_num in pages:
        tmp_img = None
        try:
            tmp_img = render_page(pdf_path, page_num)

            response = asyncio.get_event_loop().run_until_complete(
                vision_provider.analyze_image(tmp_img, MATH_DETECTION_PROMPT)
            )

            data = _parse_json_response(response)
            for f in data.get("formulas", []):
                result.formulas.append(DetectedFormula(
                    latex=f.get("latex", ""),
                    alt_text=f.get("alt_text", ""),
                    page=page_num,
                    location=f.get("location", ""),
                    inline=f.get("inline", False),
                ))
        except json.JSONDecodeError as exc:
            result.errors.append(f"Page {page_num + 1}: JSON parse error — {exc}")
            logger.warning("Math extraction JSON error on page %d: %s", page_num + 1, exc)
        except Exception as exc:
            result.errors.append(f"Page {page_num + 1}: {exc}")
            logger.warning("Math extraction error on page %d: %s", page_num + 1, exc)
        finally:
            if tmp_img is not None:
                tmp_img.unlink(missing_ok=True)

    logger.info(
        "Extracted %d formulas from %d pages of %s",
        len(result.formulas), len(pages), pdf_path.name,
    )
    return result
