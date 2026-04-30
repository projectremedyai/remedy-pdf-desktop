"""Subset Type1/CFF and TrueType-simple fonts for embedding in simple-font slots.

This module is the simple-font (non-CID) analogue of :mod:`font_embedder`.
It is consumed by :class:`~project_remedy.faithful_rebuild.simple_font_replacer.SimpleFontReplacer`
(REMEDY-73 Phase 2 Chunk C) which writes replacement fonts into existing
``/Type1`` or ``/TrueType`` (non-CID) font dicts.

Unlike the Mode B Type0 path, subsetting is keyed by **glyph names** coming
from the source font's encoding (``encoding_map`` — ``char_code → glyph_name``)
rather than by Unicode codepoints. This is important for symbol / private-name
fonts where ``glyph_name → Unicode`` is non-standard.

Two preparers:

* :func:`prepare_type1_font` — for OTF/CFF candidates. Output is a CFF
  (Type1C) program, intended for ``/FontFile3`` with ``/Subtype /Type1C``.
* :func:`prepare_truetype_simple_font` — for glyf-backed TrueType candidates
  used in a ``/TrueType`` (non-CID) slot. Output is a full subsetted TrueType
  program, intended for ``/FontFile2`` with ``/Length1``.

Neither function writes PDF objects; that is the responsibility of
``SimpleFontReplacer``. These functions only produce a portable
:class:`PreparedSimpleFont` data bundle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from fontTools.subset import Options, Subsetter
from fontTools.ttLib import TTFont


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedSimpleFont:
    """Output of preparing a simple-font program for embedding.

    Contract:
      - ``font_bytes`` is either a CFF (Type1C) subsetted program (for the
        Type1 slot) or a TrueType subsetted program (for the TrueType-simple
        slot).
      - ``postscript_name`` is the subsetted PS name including the six-letter
        ``ABCDEF+`` subset prefix.
      - ``width_for_char_code`` is in 1/1000 em (standard PDF simple-font
        units). Only used char codes are populated; callers consume these
        values directly when building ``/Widths`` / ``/FirstChar`` /
        ``/LastChar``.
      - ``glyph_name_for_char_code`` is the glyph-name mapping needed to
        write ``/Encoding`` / ``/Differences``.
      - ``font_file_subtype`` is ``/Type1C`` for CFF, or ``None`` for
        TrueType. The TrueType path uses ``/FontFile2`` + ``/Length1`` and
        has no subtype.
      - ``first_char`` / ``last_char`` are ``min(char_codes)`` /
        ``max(char_codes)`` respectively.
    """

    font_bytes: bytes
    postscript_name: str
    width_for_char_code: dict[int, int]
    glyph_name_for_char_code: dict[int, str]
    font_file_subtype: str | None
    first_char: int
    last_char: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scale_1000(value: int | float, units_per_em: int) -> int:
    """Scale a font-unit value to PDF's 1/1000-em coordinate space."""
    if units_per_em == 0:
        return int(value)
    return int(round(value * 1000 / units_per_em))


def _subset_tag(seed: str) -> str:
    """Derive a deterministic six-uppercase-letter subset prefix.

    Follows the PDF convention: ``ABCDEF+`` where ``ABCDEF`` is unique per
    subset. We hash the seed (typically the source PS name + subset glyph
    set) so the tag is stable across runs.
    """
    digest = hashlib.sha1(seed.encode("utf-8")).digest()
    letters = []
    for i in range(6):
        letters.append(chr(ord("A") + (digest[i] % 26)))
    return "".join(letters)


def _load_font(font_source: Path | bytes) -> TTFont:
    """Load *font_source* (path or raw bytes) as a :class:`TTFont`."""
    if isinstance(font_source, (bytes, bytearray)):
        return TTFont(BytesIO(bytes(font_source)))
    return TTFont(str(font_source))


def _read_ps_name(tt: TTFont, fallback: str = "Font") -> str:
    """Read the PostScript name (name ID 6) from *tt*, with a fallback."""
    if "name" in tt:
        for record in tt["name"].names:
            if record.nameID == 6:
                try:
                    candidate = record.toUnicode()
                    if candidate:
                        return candidate
                except Exception:
                    continue
    return fallback


def _validate_required_glyphs(
    tt: TTFont,
    encoding_map: dict[int, str],
    char_codes: frozenset[int],
) -> None:
    """Raise ``ValueError`` if any required glyph name is missing from *tt*.

    A glyph name is required if its char code is in *char_codes*.
    """
    glyph_order = set(tt.getGlyphOrder())
    missing: list[tuple[int, str]] = []
    for code in sorted(char_codes):
        glyph_name = encoding_map.get(code)
        if glyph_name is None:
            missing.append((code, "<unmapped>"))
            continue
        if glyph_name not in glyph_order:
            missing.append((code, glyph_name))
    if missing:
        preview = ", ".join(
            f"code={code:#04x}->{name!r}" for code, name in missing[:5]
        )
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ValueError(
            f"Font does not contain required glyph names: {preview}{more}"
        )


def _collect_required_glyph_names(
    encoding_map: dict[int, str],
    char_codes: frozenset[int],
) -> set[str]:
    """Return the set of glyph names that must survive subsetting."""
    names: set[str] = set()
    for code in char_codes:
        glyph_name = encoding_map.get(code)
        if glyph_name:
            names.add(glyph_name)
    return names


def _widths_and_encoding(
    tt: TTFont,
    encoding_map: dict[int, str],
    char_codes: frozenset[int],
) -> tuple[dict[int, int], dict[int, str]]:
    """Compute per-code widths (1/1000 em) and the forwarded encoding map.

    Widths come from the font's ``hmtx`` table, scaled to ``1000 / upem``.
    Codes whose glyph name cannot be resolved get no width entry (callers
    are expected to emit ``0`` as the default in the final ``/Widths``).
    """
    units_per_em = tt["head"].unitsPerEm if "head" in tt else 1000
    hmtx_metrics = tt["hmtx"].metrics if "hmtx" in tt else {}

    width_for_code: dict[int, int] = {}
    glyph_name_for_code: dict[int, str] = {}
    for code in char_codes:
        glyph_name = encoding_map.get(code)
        if glyph_name is None:
            continue
        glyph_name_for_code[code] = glyph_name
        entry = hmtx_metrics.get(glyph_name)
        if entry is None:
            continue
        advance, _lsb = entry
        width_for_code[code] = _scale_1000(advance, units_per_em)
    return width_for_code, glyph_name_for_code


def _subset(
    tt: TTFont,
    glyph_names: set[str],
) -> None:
    """Run fontTools :class:`Subsetter` keyed by glyph names.

    This is the glyph-name-keyed variant used for simple-font subsetting —
    distinct from the codepoint-keyed approach in :mod:`font_embedder`.
    """
    options = Options()
    options.desubroutinize = False
    options.notdef_outline = True
    options.flavor = None  # keep native sfnt wrapper
    options.drop_tables += ["FFTM"]  # cosmetic FontForge table, silence warnings
    # retain_gids keeps GIDs stable, which keeps hmtx metrics directly
    # usable without a remap.
    options.retain_gids = True

    subsetter = Subsetter(options=options)
    subsetter.populate(glyphs=list(glyph_names))
    subsetter.subset(tt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_type1_font(
    font_source: Path | bytes,
    *,
    char_codes: frozenset[int],
    encoding_map: dict[int, str],
) -> PreparedSimpleFont:
    """Prepare a Type1/CFF OTF font for embedding in a simple-font slot.

    The output :attr:`PreparedSimpleFont.font_bytes` is a **raw CFF** (Type1C)
    program suitable for ``/FontFile3`` with ``/Subtype /Type1C``. We prefer
    raw-CFF over whole-OTF because:

    1. The PDF spec only defines ``/Type1C`` and ``/CIDFontType0C`` as valid
       ``/FontFile3`` subtypes; embedding a full OTF wrapper puts us in
       reader-specific territory.
    2. The raw CFF is significantly smaller (no OS/2, name, cmap, etc.).
    3. fontTools ``CFF`` table has a reliable ``compile(tt)`` method that
       produces a standalone CFF program.

    Args:
        font_source: Path to an ``.otf`` file, or raw font bytes.
        char_codes: 1-byte char codes (0-255) actually used in the content
            stream. Typically these come from ``SimpleFontEligibility``.
        encoding_map: ``char_code → glyph_name`` mapping derived from the
            **source** font's ``/Encoding`` / ``/Differences``. The glyph
            names must exist in the candidate font.

    Returns:
        A :class:`PreparedSimpleFont` with ``font_file_subtype='/Type1C'``.

    Raises:
        ValueError: The font has no ``CFF `` table (not OTF/CFF1), any
            glyph name required by ``encoding_map`` is missing from the
            font, or the font is CFF2-only (PDF ``/FontFile3 /Type1C``
            accepts CFF1 only — CFF2 requires a separate conversion path
            that is not implemented here).
    """
    tt = _load_font(font_source)
    try:
        if "CFF " not in tt:
            if "CFF2" in tt:
                raise ValueError(
                    "prepare_type1_font cannot embed CFF2-only fonts — "
                    "PDF /FontFile3 /Type1C requires CFF1 outlines. "
                    "Convert CFF2 → CFF1 upstream or provide a CFF1 source."
                )
            raise ValueError(
                "prepare_type1_font requires an OTF/CFF1 font (no 'CFF ' "
                "table found)"
            )

        _validate_required_glyphs(tt, encoding_map, char_codes)

        required_names = _collect_required_glyph_names(encoding_map, char_codes)
        # Subsetter requires at least one glyph; .notdef is always kept by
        # notdef_outline=True but we add it defensively in case the font
        # names .notdef differently.
        _subset(tt, required_names)

        # Metrics come from the hmtx on the SUBSETTED font.
        width_for_code, glyph_name_for_code = _widths_and_encoding(
            tt, encoding_map, char_codes
        )

        # Original PS name + deterministic subset tag.
        base_ps_name = _read_ps_name(tt, fallback="Type1Font")
        tag_seed = base_ps_name + "|" + ",".join(sorted(required_names))
        postscript_name = f"{_subset_tag(tag_seed)}+{base_ps_name}"

        # Extract JUST the CFF program (raw Type1C bytes).
        cff_table = tt["CFF "]
        font_bytes = cff_table.compile(tt)

        first_char = min(char_codes) if char_codes else 0
        last_char = max(char_codes) if char_codes else 0

        return PreparedSimpleFont(
            font_bytes=bytes(font_bytes),
            postscript_name=postscript_name,
            width_for_char_code=dict(width_for_code),
            glyph_name_for_char_code=dict(glyph_name_for_code),
            font_file_subtype="/Type1C",
            first_char=first_char,
            last_char=last_char,
        )
    finally:
        tt.close()


def prepare_truetype_simple_font(
    font_source: Path | bytes,
    *,
    char_codes: frozenset[int],
    encoding_map: dict[int, str],
) -> PreparedSimpleFont:
    """Prepare a glyf-backed TrueType font for embedding in a ``/TrueType``
    (non-CID) simple-font slot.

    Differs from :func:`font_embedder.prepare_truetype_font` in two ways:

    * Subsetting is keyed by **glyph names** (from ``encoding_map.values()``)
      instead of Unicode codepoints. This is required because the source
      encoding may be non-Unicode (``WinAnsi`` / ``MacRoman`` / custom
      ``/Differences``).
    * The output is a standalone TrueType program intended for
      ``/FontFile2`` + ``/Length1``. No ``/CIDToGIDMap`` is emitted — that
      is the Type0 path.

    Args:
        font_source: Path to a ``.ttf`` file, or raw font bytes.
        char_codes: 1-byte char codes (0-255) actually used in the content
            stream.
        encoding_map: ``char_code → glyph_name`` mapping derived from the
            **source** font's ``/Encoding`` / ``/Differences``.

    Returns:
        A :class:`PreparedSimpleFont` with ``font_file_subtype=None``.

    Raises:
        ValueError: The font has no ``glyf`` table (not a glyf-backed
            TrueType), or the ``cmap`` table is missing, or any glyph name
            required by ``encoding_map`` is missing from the font.
    """
    tt = _load_font(font_source)
    try:
        if "glyf" not in tt:
            raise ValueError(
                "prepare_truetype_simple_font requires a glyf-backed TrueType "
                "font (no 'glyf' table found)"
            )
        if "cmap" not in tt:
            raise ValueError(
                "prepare_truetype_simple_font requires a font with a 'cmap' "
                "table"
            )

        _validate_required_glyphs(tt, encoding_map, char_codes)

        required_names = _collect_required_glyph_names(encoding_map, char_codes)
        _subset(tt, required_names)

        width_for_code, glyph_name_for_code = _widths_and_encoding(
            tt, encoding_map, char_codes
        )

        base_ps_name = _read_ps_name(tt, fallback="TrueTypeSimple")
        tag_seed = base_ps_name + "|" + ",".join(sorted(required_names))
        postscript_name = f"{_subset_tag(tag_seed)}+{base_ps_name}"

        # Serialize the whole subsetted TrueType program.
        buf = BytesIO()
        tt.save(buf)
        font_bytes = buf.getvalue()

        first_char = min(char_codes) if char_codes else 0
        last_char = max(char_codes) if char_codes else 0

        return PreparedSimpleFont(
            font_bytes=bytes(font_bytes),
            postscript_name=postscript_name,
            width_for_char_code=dict(width_for_code),
            glyph_name_for_char_code=dict(glyph_name_for_code),
            font_file_subtype=None,
            first_char=first_char,
            last_char=last_char,
        )
    finally:
        tt.close()
