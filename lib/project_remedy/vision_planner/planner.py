"""Planner: assemble context and call a thinking model for remediation plan.

Supports both GeminiClient and OllamaClient as the backend.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _is_gemini_client(client: Any) -> bool:
    """Return *True* if *client* is a GeminiClient (duck-type check)."""
    return hasattr(client, "_client") and hasattr(
        getattr(client, "_client", None) or object(), "aio"
    )


def _extract_response_text(response: Any) -> str:
    """Extract text content from a Gemini API response or a plain string.

    GeminiClient.generate_raw() returns a plain ``str``, while legacy code
    paths that call genai.Client directly pass through a response object with
    a ``.text`` attribute. Without the isinstance branch, str responses were
    silently dropped (see the matching fix in grounder.py).
    """
    if isinstance(response, str):
        return response.strip()
    try:
        return (response.text or "").strip()
    except Exception:
        return ""


async def run_planner(
    semantic_map: dict,
    violations: list[dict],
    anchor_graph: dict,
    harness: Any,
    client: Any,
    model: str | None = None,
    page_images: list[bytes] | None = None,
) -> dict:
    """Run planner to generate remediation plan.

    Args:
        page_images: Optional list of PNG bytes for each page. When provided,
            the planner can see the actual pages alongside the semantic map,
            enabling better remediation decisions.

    Returns dict with "plan", "planner_prompt", "planner_response".
    """
    # Build prompt via harness
    messages = harness.build_planner_prompt(semantic_map, violations, anchor_graph)

    planner_prompt = messages

    # Extract text content from messages
    text_content = messages[0].get("content", "") if messages else ""

    if _is_gemini_client(client):
        from google.genai import types

        gen_config: Any = types.GenerateContentConfig(
            max_output_tokens=16384,
            temperature=0.2,
        )
        tools = harness.build_planner_tools()
        if tools:
            gen_config.tools = tools
    else:
        # OllamaClient — generate_raw() uses sane defaults; no tools support
        gen_config = None

    # Build contents: text + optional page images for visual context
    contents: list[Any] = []
    if page_images and not _is_gemini_client(client):
        # Send page images so the planner can see what it's planning for
        for img_bytes in page_images:
            contents.append(img_bytes)
    contents.append(text_content)

    think = not _is_gemini_client(client)  # Ollama supports think=True
    try:
        response = await client.generate_raw(
            contents=contents,
            config=gen_config,
            model_override=model,
            **({"think": True} if think else {}),
        )
        raw_str = _extract_response_text(response)
    except Exception as e:
        logger.error("Planner failed: %s", e)
        raw_str = json.dumps({
            "confidence": 0,
            "operations": [],
            "manual_review": [],
            "error": str(e),
        })

    # Parse with harness
    plan = harness.parse_planner_output(raw_str)

    return {
        "plan": plan,
        "planner_prompt": planner_prompt,
        "planner_response": raw_str,
    }
