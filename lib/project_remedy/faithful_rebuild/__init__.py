"""Faithful PDF rebuild — operator-preserving tagged PDF reconstruction."""

from __future__ import annotations

from project_remedy.faithful_rebuild.models import (
    FaithfulRebuildResult,
    FontFingerprint,
    FontMatch,
    MCIDEntry,
    MCIDManifest,
    PreparedFont,
)
from project_remedy.faithful_rebuild.pipeline import faithful_rebuild

__all__ = [
    "FaithfulRebuildResult",
    "FontFingerprint",
    "FontMatch",
    "MCIDEntry",
    "MCIDManifest",
    "PreparedFont",
    "faithful_rebuild",
]
