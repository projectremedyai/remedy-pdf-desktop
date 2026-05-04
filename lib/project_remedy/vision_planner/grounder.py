"""Grounder: render PDF pages and call a vision model for semantic analysis.

The caller passes a generate_raw-capable client; image bytes and text are
sent in the generic content format understood by the local runtime client.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def _extract_response_text(response: Any) -> str:
    """Extract text content from a model response or plain string."""
    if isinstance(response, str):
        return response.strip()
    try:
        return (response.text or "").strip()
    except Exception:
        return ""


async def run_grounder(
    pdf_path: Path,
    harness: Any,
    client: Any,
    model: str | None = None,
) -> dict:
    """Run grounder on all pages. Returns combined semantic map.

    Returns dict with "pages", "grounder_prompts", "grounder_responses".
    """
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)

    pages: list[dict] = []
    prompts: list[list[dict]] = []
    responses: list[str] = []
    page_images: list[bytes] = []  # raw PNG bytes for planner vision

    for page_idx in range(page_count):
        page = doc[page_idx]

        # Render at configurable DPI. 72 is sufficient for structure
        # detection (heading/paragraph/table classification). Higher DPI
        # is only needed for OCR or fine text reading.
        render_dpi = getattr(harness, "grounder_dpi", 72)
        pix = page.get_pixmap(dpi=render_dpi)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode()
        page_images.append(img_bytes)

        page_dims = {
            "width_pts": page.rect.width,
            "height_pts": page.rect.height,
        }

        # Get harness messages (for trace logging)
        harness_messages = harness.build_grounder_prompt(b64, page_dims)
        prompts.append(harness_messages)

        text_content = harness_messages[0].get("content", "") if harness_messages else ""
        parts = [img_bytes, text_content]

        # Call the vision model with thinking enabled for deeper reasoning
        try:
            response = await client.generate_raw(
                contents=parts,
                config=None,
                model_override=model,
                think=True,
            )
            raw = _extract_response_text(response)
            responses.append(raw)
        except Exception as e:
            logger.error("Grounder failed on page %d: %s", page_idx, e)
            raw = json.dumps({"regions": [], "error": str(e)})
            responses.append(raw)

        # Parse with harness
        semantic_map = harness.parse_grounder_output(raw)
        semantic_map["page"] = page_idx
        pages.append(semantic_map)

    doc.close()

    return {
        "pages": pages,
        "grounder_prompts": prompts,
        "grounder_responses": responses,
        "page_images": page_images,
    }
