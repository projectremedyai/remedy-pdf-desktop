"""Data models for contrast analysis — internal models and AI response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContrastIssueType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    GRAPHIC = "graphic"


class ContrastIssue(BaseModel):
    """A single contrast issue identified on a page."""

    id: str
    issue_type: ContrastIssueType
    page_index: int
    bbox: list[float] = Field(default_factory=list)  # [x0, y0, x1, y1]
    fg_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    contrast_ratio: float = 0.0
    required_ratio: float = 4.5
    wcag_criterion: str = "1.4.3"  # "1.4.3" (AA), "1.4.6" (AAA), "1.4.11" (non-text)
    is_large_text: bool = False
    font_size: float | None = None
    is_bold: bool = False
    text_content: str = ""
    description: str = ""
    suggested_fg: tuple[float, float, float] | None = None
    fixed: bool = False
    fix_attempts: int = 0


class PageContrastResult(BaseModel):
    """Contrast analysis for a single page."""

    page_index: int
    issues: list[ContrastIssue] = Field(default_factory=list)
    issues_fixed: int = 0
    issues_remaining: int = 0


class ContrastAnalysis(BaseModel):
    """Full document contrast analysis."""

    pages: list[PageContrastResult] = Field(default_factory=list)
    total_issues: int = 0
    issues_fixed: int = 0
    issues_remaining: int = 0

    def compute_totals(self) -> None:
        self.total_issues = sum(len(p.issues) for p in self.pages)
        self.issues_fixed = sum(p.issues_fixed for p in self.pages)
        self.issues_remaining = sum(p.issues_remaining for p in self.pages)


# --- AI response schemas ---

CONTRAST_DETECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_type": {
                        "type": "string",
                        "enum": ["text", "image", "graphic", "use_of_color"],
                    },
                    "content_kind": {
                        "type": "string",
                        "enum": ["pdf_text", "vector_text", "image_text", "diagram_artwork", "unknown"],
                    },
                    "description": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Bounding box as percentage [x0, y0, x1, y1] of page dimensions (0-100)",
                    },
                    "fg_color_hex": {"type": "string"},
                    "bg_color_hex": {"type": "string"},
                    "estimated_contrast_ratio": {"type": "number"},
                    "estimated_ratio": {"type": "number"},
                    "required_ratio": {"type": "number"},
                    "auto_fixable": {"type": "boolean"},
                    "wcag_criterion": {"type": "string"},
                    "fix_rgb": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["fail", "borderline"],
                    },
                    "text_content": {"type": "string"},
                    "is_large_text": {"type": "boolean"},
                    "is_bold": {"type": "boolean"},
                    "estimated_font_size": {"type": "number"},
                    "suggestion": {"type": "string"},
                },
                "required": [
                    "issue_type",
                    "description",
                    "bbox",
                    "fg_color_hex",
                    "bg_color_hex",
                    "estimated_contrast_ratio",
                    "severity",
                ],
            },
        },
        "page_has_contrast_issues": {"type": "boolean"},
        "use_of_color_issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "location": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["description"],
            },
        },
        "overall_assessment": {"type": "string"},
    },
    "required": ["issues", "page_has_contrast_issues"],
}


CONTRAST_VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "fixed": {"type": "boolean"},
                    "current_contrast_assessment": {"type": "string"},
                    "remaining_problems": {"type": "string"},
                },
                "required": ["issue_id", "fixed"],
            },
        },
        "all_fixed": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["results", "all_fixed"],
}
