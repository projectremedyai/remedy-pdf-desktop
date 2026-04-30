"""Vision AI prompts for contrast detection and validation."""

from __future__ import annotations

from project_remedy.vision_prompts import (
    contrast_detection_prompt as _shared_contrast_detection_prompt,
    contrast_validation_prompt as _shared_contrast_validation_prompt,
)


def contrast_detection_prompt(level: str = "AA") -> str:
    """Build the prompt for AI-driven contrast issue detection."""
    return _shared_contrast_detection_prompt(level)


def contrast_validation_prompt(level: str, issue_descriptions: str) -> str:
    """Build the prompt for AI-driven fix validation."""
    return _shared_contrast_validation_prompt(level, issue_descriptions)
