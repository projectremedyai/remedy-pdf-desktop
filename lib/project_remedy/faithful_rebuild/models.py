"""Shared data types for the faithful_rebuild module.

These dataclasses are used across all sub-modules:
  font_extractor, font_matcher, page_renderer, content_assembler,
  structure_writer, and the top-level orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pikepdf


# ---------------------------------------------------------------------------
# Font types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FontFingerprint:
    """Immutable description of a font extracted from a source PDF.

    Attributes:
        source_id: Resource key as it appears in the source PDF (e.g. "F1").
        family: Human-readable family name (e.g. "Arial").
        postscript_name: PostScript name from the font descriptor.
        weight: Numeric weight (100–900; 400 = Regular, 700 = Bold).
        width_class: OS/2 width class (1–9; 5 = Normal).
        italic: True if the font is italic/oblique.
        serif: True = serif, False = sans-serif, None = unknown.
        mono: True if the font is monospaced.
        panose: 10-element PANOSE tuple from the OS/2 table, or None.
        cap_height: Cap-height in font units, or None.
        x_height: x-height in font units, or None.
        glyph_coverage: Frozenset of Unicode codepoints with glyphs present.
        embedded_program: Raw font program bytes (Type1, CFF, TTF …), or None.
        path: Path to a matching font file on disk (resolved later), or None.
    """

    source_id: str
    family: str
    postscript_name: str
    weight: int
    width_class: int
    italic: bool
    serif: bool | None
    mono: bool
    panose: tuple[int, ...] | None
    cap_height: int | None
    x_height: int | None
    glyph_coverage: frozenset[int]
    embedded_program: bytes | None
    path: Path | None = None


@dataclass
class FontMatch:
    """Result of matching a :class:`FontFingerprint` to a system or embedded font.

    Attributes:
        source: The fingerprint this match was derived from.
        resolved_path: Absolute path to the chosen font file, or None if using
                       the embedded program directly.
        use_embedded: True when the embedded font program should be used instead
                      of a system font.
        confidence: Match confidence in the range [0, 1].
        fallback_reason: Human-readable explanation when confidence < 1, or None.
    """

    source: FontFingerprint
    resolved_path: Path | None
    use_embedded: bool
    confidence: float
    fallback_reason: str | None = None


@dataclass
class PreparedFont:
    """A font ready for embedding into a rebuilt PDF page.

    Attributes:
        resource_key: Name used to reference this font in page resources (e.g. "F1").
        postscript_name: PostScript name for the /BaseFont entry.
        font_bytes: Raw font program bytes to embed.
        gid_for_codepoint: Mapping from Unicode codepoint to glyph ID.
        width_for_gid: Mapping from glyph ID to advance width in 1/1000 em units.
        to_unicode: Mapping from character code (CID) to Unicode codepoint.
        ascent: Ascender in font units.
        descent: Descender in font units (typically negative).
        cap_height: Cap-height in font units.
        flags: PDF font descriptor /Flags bitmask.
        font_bbox: Font bounding box [llx, lly, urx, ury] in font units.
        italic_angle: Italic angle in degrees (0 for upright).
        stem_v: Dominant vertical stem width in font units.
    """

    resource_key: str
    postscript_name: str
    font_bytes: bytes
    gid_for_codepoint: dict[int, int]
    width_for_gid: dict[int, int]
    to_unicode: dict[int, int]
    ascent: int
    descent: int
    cap_height: int
    flags: int
    font_bbox: list[int]
    italic_angle: float
    stem_v: int


# ---------------------------------------------------------------------------
# Content run types (immutable, z-ordered)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextRun:
    """A single span of text on a page, tied to one MCID.

    Coordinates use a top-left origin (y increases downward), matching the
    PDF MediaBox top-to-bottom rendering convention used by the rebuilder.

    Attributes:
        mcid: Marked-content identifier that owns this run.
        z_index: Rendering order (lower values drawn first).
        font_key: Resource key of the font (matches :attr:`PreparedFont.resource_key`).
        font_size: Rendered font size in points.
        baseline_x: X coordinate of the text baseline origin.
        baseline_y_top: Y coordinate measured from the top of the page.
        cids: Ordered list of character IDs (CIDs) to render.
        rgb: Fill colour as (r, g, b) in [0, 1], or None for black.
        cmyk: Fill colour as (c, m, y, k) in [0, 1], or None.
        text_matrix: Full 6-element text matrix [a b c d e f], or None.
        tj_adjustments: Per-glyph kerning adjustments for TJ operator, or None.
    """

    mcid: int
    z_index: int
    font_key: str
    font_size: float
    baseline_x: float
    baseline_y_top: float
    cids: list[int]
    rgb: tuple[float, float, float] | None = None
    cmyk: tuple[float, float, float, float] | None = None
    text_matrix: list[float] | None = None
    tj_adjustments: list[float] | None = None


@dataclass(frozen=True)
class ImageRun:
    """A raster image placed on a page, tied to one MCID.

    Coordinates use a top-left origin.

    Attributes:
        mcid: Marked-content identifier that owns this image.
        z_index: Rendering order.
        xobject_name: XObject resource key in the page dictionary.
        x0: Left edge of the image in points from the left of the page.
        y0_top: Top edge of the image in points from the top of the page.
        width: Rendered width in points.
        height: Rendered height in points.
        tag: PDF structure tag; "Figure" for informative images,
             "Artifact" for decorative ones.
    """

    mcid: int
    z_index: int
    xobject_name: str
    x0: float
    y0_top: float
    width: float
    height: float
    tag: str = "Figure"


@dataclass(frozen=True)
class VectorRun:
    """A vector graphics sequence on a page, tied to one MCID.

    Attributes:
        mcid: Marked-content identifier that owns this run.
        z_index: Rendering order.
        tag: PDF structure tag (e.g. "Figure", "Artifact").
        operators: Raw PDF content-stream bytes for the graphic.
    """

    mcid: int
    z_index: int
    tag: str
    operators: bytes


# ---------------------------------------------------------------------------
# Manifest types
# ---------------------------------------------------------------------------


@dataclass
class MCIDEntry:
    """Metadata for a single marked-content span.

    Attributes:
        mcid: Marked-content identifier (matches run types above).
        tag: PDF structure element tag (e.g. "P", "H1", "Figure", "TH").
        semantic_type: Human-readable semantic category
                       (e.g. "paragraph", "heading", "figure", "table_cell").
        alt_text: Alternate text for figures and other non-text elements, or None.
        element_id: Optional ID attribute for linking (e.g. header cells to data cells).
        table_spec: For table cells, a dict with keys such as
                    ``row``, ``col``, ``scope``, and ``header_ids``.
    """

    mcid: int
    tag: str
    semantic_type: str
    alt_text: str | None = None
    element_id: str | None = None
    table_spec: dict[str, Any] | None = None


@dataclass
class MCIDManifest:
    """Complete mapping of MCIDs for one page.

    Attributes:
        entries: Ordered list of :class:`MCIDEntry` objects, one per MCID.
    """

    entries: list[MCIDEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


@dataclass
class FaithfulRebuildResult:
    """Summary result returned by the faithful rebuild orchestrator.

    Attributes:
        success: True if the rebuilt PDF passed all acceptance gates.
        source_path: Path to the original source PDF.
        output_path: Path to the rebuilt PDF, or None if rebuild failed before
                     a file was written.
        mode: Rebuild mode used (e.g. "full", "font_only", "structure_only").
        visual_diff_pct: Visual similarity difference vs. source as a fraction
                         in [0, 1] (0 = identical).
        verapdf_violations: Number of veraPDF violations remaining.
        text_coverage_pct: Percentage of source text successfully reproduced
                           in the rebuilt PDF.
        pages_rebuilt: Number of pages included in the rebuilt output.
        font_matches: List of :class:`FontMatch` objects, one per font used.
        error: Human-readable error message if ``success`` is False, else None.
    """

    success: bool
    source_path: Path
    output_path: Path | None
    mode: str
    visual_diff_pct: float
    verapdf_violations: int
    text_coverage_pct: float
    pages_rebuilt: int
    font_matches: list[FontMatch]
    error: str | None = None


# ---------------------------------------------------------------------------
# Canary eligibility
# ---------------------------------------------------------------------------


@dataclass
class CanaryEligibility:
    """Result of checking whether a PDF qualifies for the v3 canary.

    A PDF qualifies iff qualifies=True AND disqualifying_reasons is empty.
    The two fields are kept separate so callers can inspect partial results
    during debugging (e.g., multiple reasons accumulated before qualification
    is definitively decided).

    Fields are optional because audit fills them progressively; for a
    qualifying doc, all non-Optional fields will be populated.
    """
    qualifies: bool = False
    font_object: Any = None            # pikepdf.Object (indirect)
    font_key: str | None = None
    page_index: int | None = None
    used_cids: frozenset[int] = field(default_factory=frozenset)
    cid_unicode_map: dict[int, int] | None = None
    trigger_rules: frozenset[str] = field(default_factory=frozenset)
    disqualifying_reasons: list[str] = field(default_factory=list)
    placements: list[tuple[int, str]] = field(default_factory=list)
    recovered_cids_count: int = 0

    def __post_init__(self):
        if not self.qualifies and not self.disqualifying_reasons:
            raise ValueError(
                "CanaryEligibility(qualifies=False) requires "
                "disqualifying_reasons to be non-empty."
            )


# ---------------------------------------------------------------------------
# v4 Measurement: bucket classification
# ---------------------------------------------------------------------------


# Primary bucket labels (see v4 spec: first-match-wins order).
BUCKET_LABELS = frozenset({
    "v3_qualifying",
    "near_miss_partial_unicode_map",
    "near_miss_multi_placement",
    "near_miss_form_xobject",
    "near_miss_multi_font",
    "out_of_scope_simple_font",
    "out_of_scope_other",
    "out_of_scope_no_broken_fonts",
})

# Scope extensions that can appear in also_requires.
SCOPE_EXTENSIONS = frozenset({
    "partial_unicode_map",
    "multi_placement",
    "form_xobject",
    "multi_font",
})


@dataclass(frozen=True)
class BucketClassification:
    """Classification of a PDF for the v4 measurement audit.

    primary_bucket is the first-matching blocker in the v4 spec order.
    also_requires is every scope extension the doc fails, independent of
    primary. For a v3_qualifying doc, also_requires is empty.
    """

    primary_bucket: str
    also_requires: frozenset[str]

    def __post_init__(self):
        if self.primary_bucket not in BUCKET_LABELS:
            raise ValueError(
                f"Invalid primary_bucket {self.primary_bucket!r}; "
                f"must be one of {sorted(BUCKET_LABELS)}"
            )
        unknown = self.also_requires - SCOPE_EXTENSIONS
        if unknown:
            raise ValueError(
                f"Invalid also_requires entries {sorted(unknown)}; "
                f"must be subset of {sorted(SCOPE_EXTENSIONS)}"
            )


# ---------------------------------------------------------------------------
# Multi-font replacement: document-level eligibility
# ---------------------------------------------------------------------------


@dataclass
class MultiCanaryEligibility:
    """Document-level eligibility for multi-font Mode B replacement.

    Holds N per-font CanaryEligibility records. qualifies_document is True
    iff the list is non-empty AND every font qualifies individually.
    """
    font_eligibilities: list[CanaryEligibility]
    disqualifying_reasons: list[str] = field(default_factory=list)

    @property
    def qualifies_document(self) -> bool:
        if not self.font_eligibilities:
            return False
        if self.disqualifying_reasons:
            return False
        return all(e.qualifies for e in self.font_eligibilities)


# ---------------------------------------------------------------------------
# Simple-font replacement (REMEDY-73 Phase 2): per-font + document aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimpleFontEligibility:
    """Eligibility of ONE simple font (Type1 / TrueType-non-Type0) for
    replacement under the REMEDY-73 Phase 2 track.

    Mirrors the shape of :class:`CanaryEligibility` but is scoped to simple
    fonts: the replacement track for veraPDF 7.21.4.1-1 on Type1 / TrueType
    fonts whose ``/FontDescriptor`` lacks a ``/FontFile`` (Type1) or
    ``/FontFile2`` (TrueType) entry.

    Fields
    ------
    qualifies:
        True iff the font can be safely replaced by ``SimpleFontReplacer``
        (Phase 2 Chunk C).
    font_object:
        The indirect :class:`pikepdf.Object` for the font dict, or ``None``
        when the font was rejected before an object handle was captured.
    font_key:
        Resource-dict key that names this font on the page (e.g. ``"/F1"``).
    page_index:
        0-based index of the page where the font was first discovered.
    font_subtype:
        ``"/Type1"`` or ``"/TrueType"`` — simple-font subtypes only.
    base_font:
        Value of ``/BaseFont`` as a Python str, including any 6-letter
        subset prefix (e.g. ``"ABCDEF+Calibri"``).
    used_char_codes:
        Frozenset of 1-byte char codes used by the font in page content
        streams.  Populated by the embedder/replacer in later chunks; may
        be empty when eligibility is checked against the font dict alone
        (Chunk A scope — no content-stream walk yet).
    code_to_glyph:
        Map of used char code → glyph name, derived from ``/Encoding``
        (base encoding + ``/Differences``).  ``None`` iff the helper could
        not produce a complete map for every ``used_char_codes`` entry.
    trigger_rules:
        Subset of ``{"7.21.4.1-1"}`` (veraPDF rules this font currently
        fires); non-empty means the font "needs" replacement.
    placements:
        List of ``(page_index, font_key)`` pairs where the font appears.
        For Chunk A we populate with a single entry; the orchestrator in a
        later chunk may union across placements.
    disqualifying_reasons:
        Stable-string telemetry keys explaining *why* a font did not
        qualify.  See the module-level taxonomy in
        :mod:`project_remedy.faithful_rebuild.simple_font_replacer`.
    """

    qualifies: bool
    font_object: pikepdf.Object | None = None
    font_key: str = ""
    page_index: int = 0
    font_subtype: str = ""
    base_font: str = ""
    used_char_codes: frozenset[int] = field(default_factory=frozenset)
    code_to_glyph: dict[int, str] | None = None
    trigger_rules: frozenset[str] = field(default_factory=frozenset)
    placements: list[tuple[int, str]] = field(default_factory=list)
    disqualifying_reasons: list[str] = field(default_factory=list)


@dataclass
class MultiSimpleFontEligibility:
    """Document-level aggregation across multiple simple fonts.

    ``qualifies_document`` is True iff at least one per-font entry
    qualifies AND no document-level disqualifying reasons were recorded.
    Individual non-qualifying entries are retained for telemetry but do not
    block the document from replacement — the replacer acts per-font.
    """

    font_eligibilities: list[SimpleFontEligibility] = field(default_factory=list)
    disqualifying_reasons: list[str] = field(default_factory=list)

    @property
    def qualifies_document(self) -> bool:
        return (
            any(e.qualifies for e in self.font_eligibilities)
            and not self.disqualifying_reasons
        )
