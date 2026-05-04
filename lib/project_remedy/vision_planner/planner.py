"""Planner: assemble context and call a thinking model for remediation plan."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _extract_response_text(response: Any) -> str:
    """Extract text content from a model response or plain string."""
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

    # Build contents: text + optional page images for visual context
    contents: list[Any] = []
    if page_images:
        # Send page images so the planner can see what it's planning for
        for img_bytes in page_images:
            contents.append(img_bytes)
    contents.append(text_content)

    try:
        response = await client.generate_raw(
            contents=contents,
            config=None,
            model_override=model,
            think=True,
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
