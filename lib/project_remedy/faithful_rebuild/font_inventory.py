"""Production font inventory contract (REMEDY-78).

Declares required Unicode block coverage for Mode B to be production-ready.
Deployments that bundle a font directory can verify coverage at startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class FontInventoryContract:
    """Contract for production font coverage.

    REQUIRED_BLOCKS must all be covered by at least one indexed font for
    Mode B to be considered production-ready. RECOMMENDED_BLOCKS are
    aspirational — missing coverage warns but doesn't block.
    """

    REQUIRED_BLOCKS: frozenset[str] = frozenset({
        "Basic Latin",
        "Latin-1 Supplement",
        "Latin Extended-A",
        "Latin Extended-B",
        "General Punctuation",
    })

    RECOMMENDED_BLOCKS: frozenset[str] = frozenset({
        "Cyrillic",
        "Greek and Coptic",
        "Armenian",
        "Mathematical Operators",
        "Block Elements",
        "Geometric Shapes",
        "Dingbats",
    })


@dataclass
class InventoryVerification:
    """Structured result of verify_production_font_inventory()."""
    meets_required: bool
    meets_recommended: bool
    missing_required_blocks: list[str] = field(default_factory=list)
    missing_recommended_blocks: list[str] = field(default_factory=list)
    total_fonts_indexed: int = 0
    per_block_font_count: dict[str, int] = field(default_factory=dict)


def _codepoints_in_block(block_name: str) -> range:
    """Return the codepoint range for a named Unicode block.

    Imports the module-private _UNICODE_BLOCKS from font_analysis to avoid
    duplicating the table here.
    """
    from project_remedy.faithful_rebuild.font_analysis import _UNICODE_BLOCKS
    for start, end, name in _UNICODE_BLOCKS:
        if name == block_name:
            return range(start, end + 1)
    return range(0, 0)


def _font_covers_block(entry: Any, block_name: str, min_coverage: float = 0.5) -> bool:
    """Return True if a FontIndex entry covers at least half of a block's codepoints.

    A block is considered covered when >=50% of its codepoints are present in
    the font's glyph_coverage. The 50% threshold is a heuristic — many system
    fonts include only a subset of a block (e.g., Cyrillic fonts often skip
    archaic variants).
    """
    cov = getattr(entry, "glyph_coverage", None)
    if not cov:
        return False
    block_range = _codepoints_in_block(block_name)
    if len(block_range) == 0:
        return False
    covered = sum(1 for cp in block_range if cp in cov)
    return (covered / len(block_range)) >= min_coverage


def verify_production_font_inventory(
    index: Any,  # FontIndex from font_matcher — avoid circular import
) -> InventoryVerification:
    """Check whether the current font index meets the production contract.

    NEVER RAISES. Callers decide how to handle missing coverage. Emits a
    WARNING log when required blocks are missing.
    """
    entries = getattr(index, "entries", []) or []
    result = InventoryVerification(
        meets_required=True,
        meets_recommended=True,
        total_fonts_indexed=len(entries),
    )

    all_blocks = FontInventoryContract.REQUIRED_BLOCKS | FontInventoryContract.RECOMMENDED_BLOCKS
    for block in sorted(all_blocks):
        covering_count = sum(1 for e in entries if _font_covers_block(e, block))
        result.per_block_font_count[block] = covering_count
        if covering_count == 0:
            if block in FontInventoryContract.REQUIRED_BLOCKS:
                result.meets_required = False
                result.missing_required_blocks.append(block)
            if block in FontInventoryContract.RECOMMENDED_BLOCKS:
                result.meets_recommended = False
                result.missing_recommended_blocks.append(block)

    if not result.meets_required:
        logger.warning(
            "Font inventory missing required blocks: %s (total fonts indexed: %d)",
            result.missing_required_blocks,
            result.total_fonts_indexed,
        )

    return result
