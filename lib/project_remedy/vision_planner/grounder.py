"""Grounder: render PDF pages and call a vision model for semantic analysis.

Supports both GeminiClient and OllamaClient as the backend.  The caller
passes whichever client ``tier3_retry`` constructed; the grounder detects
the client type at runtime so it can build the right content format.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def _is_gemini_client(client: Any) -> bool:
    """Return *True* if *client* is a GeminiClient (duck-type check).

    Uses attribute probing instead of ``isinstance`` so that this module
    does not need to import ``google.genai`` (which may not be installed
    when running with the Ollama backend only).
    """
    return hasattr(client, "_client") and hasattr(
        getattr(client, "_client", None) or object(), "aio"
    )


def _extract_response_text(response: Any) -> str:
    """Extract text content from a Gemini API response or a plain string.

    GeminiClient.generate_raw() returns a plain ``str``, while legacy code
    paths that call genai.Client directly pass through a response object with
    a ``.text`` attribute. Without the isinstance branch, str responses were
    silently dropped: `response.text` on a str raised AttributeError, the
    except clause returned "", and every grounder page parsed as empty JSON
    with confidence 0.0. This broke Tier 3 Vision Planner entirely.
    """
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

        # Build content parts appropriate for the backend
        text_content = harness_messages[0].get("content", "") if harness_messages else ""

        if _is_gemini_client(client):
            # GeminiClient — use google.genai types
            from google.genai import types

            parts: list[Any] = [
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                text_content,
            ]
            gen_config: Any = types.GenerateContentConfig(
                max_output_tokens=16384,
                temperature=0.2,
            )
            # Get tools (if harness defines them)
            tools = harness.build_grounder_tools()
            if tools:
                gen_config.tools = tools
        else:
            # OllamaClient — pass raw bytes + text; generate_raw() handles
            # conversion to the OpenAI content format.
            parts = [img_bytes, text_content]
            gen_config = None  # OllamaClient.generate_raw() uses sane defaults

        # Call the vision model with thinking enabled for deeper reasoning
        think = not _is_gemini_client(client)  # Ollama supports think=True
        try:
            response = await client.generate_raw(
                contents=parts,
                config=gen_config,
                model_override=model,
                **({"think": True} if think else {}),
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
