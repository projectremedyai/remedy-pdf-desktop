"""Subset TrueType fonts and register as Type0/CIDFontType2 in PDF.

This module is the bridge between font matching (which finds the right font)
and page rendering (which writes content streams referencing that font).  It:

1. Subsets a TrueType font to only the glyphs needed for a given text.
2. Builds the complete Type0 -> CIDFontType2 -> FontDescriptor dictionary
   chain required by the PDF spec for CID-keyed fonts.
3. Attaches a ToUnicode CMap so that text is extractable by screen readers.
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

import pikepdf
from fontTools.subset import Options, Subsetter
from fontTools.ttLib import TTFont

from project_remedy.faithful_rebuild.models import PreparedFont
from project_remedy.pdf_fixer import build_bfchar_cmap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scale_1000(value: int | float, units_per_em: int) -> int:
    """Scale a font-unit value to PDF's 1/1000-em coordinate space."""
    if units_per_em == 0:
        return int(value)
    return int(round(value * 1000 / units_per_em))


def _compute_flags(tt: TTFont) -> int:
    """Compute the PDF FontDescriptor /Flags bitmask from an OpenType font.

    Bit positions (1-indexed per PDF spec):
        1  FixedPitch
        2  Serif
        3  Symbolic
        6  Nonsymbolic
        7  Italic
        19 ForceBold
    """
    flags = 0

    # FixedPitch (bit 1 = value 1)
    if "post" in tt:
        if tt["post"].isFixedPitch:
            flags |= 1

    os2 = tt.get("OS/2")

    # Serif (bit 2 = value 2) — check PANOSE family kind
    if os2 is not None and hasattr(os2, "panose") and os2.panose is not None:
        panose = os2.panose
        # panose.bSerifStyle: 0 = any, 1 = no fit, 2-10 = serif variants,
        # 11-15 = sans-serif variants.  We consider 2-10 as serif.
        serif_style = getattr(panose, "bSerifStyle", 0)
        if 2 <= serif_style <= 10:
            flags |= 2  # Serif

    # Symbolic vs Nonsymbolic — we always set Nonsymbolic (bit 6 = 32)
    # for Latin text fonts with a Unicode cmap; Symbolic (bit 3 = 4) otherwise.
    has_unicode_cmap = False
    if "cmap" in tt:
        for table in tt["cmap"].tables:
            if table.platformID == 3 and table.platEncID == 1:
                has_unicode_cmap = True
                break
            if table.platformID == 0:
                has_unicode_cmap = True
                break
    if has_unicode_cmap:
        flags |= 32  # Nonsymbolic
    else:
        flags |= 4  # Symbolic

    # Italic (bit 7 = 64)
    if os2 is not None:
        fs_selection = getattr(os2, "fsSelection", 0)
        if fs_selection & 1:  # bit 0 of fsSelection = ITALIC
            flags |= 64
    elif "post" in tt:
        if tt["post"].italicAngle != 0:
            flags |= 64

    # ForceBold (bit 19 = 262144)
    if os2 is not None:
        weight = getattr(os2, "usWeightClass", 400)
        if weight >= 700:
            flags |= 262144

    return flags


def _build_cidset(gids: set[int]) -> bytes:
    """Build a CIDSet bitmap of used glyph IDs.

    The CIDSet is a stream of bytes where bit *n* (big-endian, MSB first)
    indicates that glyph ID *n* is present.
    """
    if not gids:
        return b"\x00"
    max_gid = max(gids)
    num_bytes = (max_gid // 8) + 1
    bitmap = bytearray(num_bytes)
    for gid in gids:
        byte_idx = gid // 8
        bit_idx = 7 - (gid % 8)
        bitmap[byte_idx] |= 1 << bit_idx
    return bytes(bitmap)


def _build_w_array(width_for_gid: dict[int, int]) -> pikepdf.Array:
    """Build a sparse /W array in the format ``[start [w0 w1 ...] ...]``.

    Groups consecutive glyph IDs into runs to keep the array compact.
    """
    if not width_for_gid:
        return pikepdf.Array()

    sorted_gids = sorted(width_for_gid.keys())
    result = pikepdf.Array()

    i = 0
    while i < len(sorted_gids):
        run_start = sorted_gids[i]
        widths: list[int] = [width_for_gid[run_start]]
        j = i + 1
        while j < len(sorted_gids) and sorted_gids[j] == sorted_gids[j - 1] + 1:
            widths.append(width_for_gid[sorted_gids[j]])
            j += 1

        result.append(run_start)
        result.append(pikepdf.Array(widths))
        i = j

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_truetype_font(
    source: Path | bytes,
    resource_key: str,
    text: str,
) -> PreparedFont:
    """Subset a TrueType font to the glyphs required by *text*.

    Args:
        source: Path to a ``.ttf`` file, or raw font bytes (for embedded fonts).
        resource_key: PDF resource name (e.g. ``"F0"``).
        text: The text that will be rendered with this font; only the glyphs
              needed for these characters are retained in the subset.

    Returns:
        A :class:`PreparedFont` ready for :func:`register_type0_font`.
    """
    # Load the font
    if isinstance(source, (bytes, bytearray)):
        tt = TTFont(BytesIO(source))
    else:
        tt = TTFont(str(source))

    # Subset to only the characters in `text`
    options = Options()
    options.retain_gids = True
    options.desubroutinize = False
    # Ensure the font program is kept intact for embedding
    options.flavor = None
    # Keep notdef
    options.notdef_outline = True

    subsetter = Subsetter(options=options)
    codepoints = {ord(ch) for ch in text}
    subsetter.populate(unicodes=codepoints)
    subsetter.subset(tt)

    # Extract cmap (Unicode -> glyph ID mapping)
    best_cmap: dict[int, str] | None = None
    if "cmap" in tt:
        for table in tt["cmap"].tables:
            if table.platformID == 3 and table.platEncID == 1:
                best_cmap = table.cmap
                break
            if table.platformID == 0 and best_cmap is None:
                best_cmap = table.cmap

    if best_cmap is None:
        best_cmap = {}

    # Build gid_for_codepoint and to_unicode (gid -> codepoint)
    glyph_order = tt.getGlyphOrder()
    glyph_name_to_gid: dict[str, int] = {
        name: idx for idx, name in enumerate(glyph_order)
    }

    gid_for_codepoint: dict[int, int] = {}
    to_unicode: dict[int, int] = {}
    for codepoint, glyph_name in best_cmap.items():
        if codepoint not in codepoints:
            continue
        gid = glyph_name_to_gid.get(glyph_name)
        if gid is not None and gid > 0:
            gid_for_codepoint[codepoint] = gid
            to_unicode[gid] = codepoint

    # Build width_for_gid from hmtx
    units_per_em = tt["head"].unitsPerEm
    hmtx = tt["hmtx"]
    width_for_gid: dict[int, int] = {}
    for gid in gid_for_codepoint.values():
        glyph_name = glyph_order[gid] if gid < len(glyph_order) else None
        if glyph_name and glyph_name in hmtx.metrics:
            advance, _ = hmtx.metrics[glyph_name]
            width_for_gid[gid] = _scale_1000(advance, units_per_em)

    # Extract metrics
    os2 = tt.get("OS/2")
    head = tt["head"]

    ascent = _scale_1000(os2.sTypoAscender, units_per_em) if os2 else 800
    descent = _scale_1000(os2.sTypoDescender, units_per_em) if os2 else -200

    cap_height = 0
    if os2 and hasattr(os2, "sCapHeight") and os2.sCapHeight:
        cap_height = _scale_1000(os2.sCapHeight, units_per_em)
    if cap_height <= 0:
        # Fallback: approximate from ascent
        cap_height = int(ascent * 0.7) if ascent > 0 else 700

    font_bbox = [
        _scale_1000(head.xMin, units_per_em),
        _scale_1000(head.yMin, units_per_em),
        _scale_1000(head.xMax, units_per_em),
        _scale_1000(head.yMax, units_per_em),
    ]

    italic_angle = 0.0
    if "post" in tt:
        italic_angle = float(tt["post"].italicAngle)

    flags = _compute_flags(tt)

    # PostScript name
    postscript_name = ""
    if "name" in tt:
        for record in tt["name"].names:
            if record.nameID == 6:  # PostScript name
                try:
                    postscript_name = record.toUnicode()
                    break
                except Exception:
                    pass
    if not postscript_name:
        postscript_name = resource_key

    # Serialize subsetted font
    buf = BytesIO()
    tt.save(buf)
    font_bytes = buf.getvalue()

    tt.close()

    return PreparedFont(
        resource_key=resource_key,
        postscript_name=postscript_name,
        font_bytes=font_bytes,
        gid_for_codepoint=gid_for_codepoint,
        width_for_gid=width_for_gid,
        to_unicode=to_unicode,
        ascent=ascent,
        descent=descent,
        cap_height=cap_height,
        flags=flags,
        font_bbox=font_bbox,
        italic_angle=italic_angle,
        stem_v=80,
    )


def register_type0_font(
    pdf: pikepdf.Pdf,
    resources: pikepdf.Dictionary,
    prepared: PreparedFont,
) -> None:
    """Register a :class:`PreparedFont` as a Type0/CIDFontType2 font in *pdf*.

    This builds the full dictionary chain required by the PDF spec:

    - ``/Type0`` top-level font with ``/Encoding /Identity-H``
    - ``/CIDFontType2`` descendant with ``/CIDToGIDMap /Identity``, ``/DW``,
      ``/W`` widths array
    - ``/FontDescriptor`` with ``/FontFile2``, ``/CIDSet``, and metrics
    - ``/ToUnicode`` CMap for text extraction

    After this call the font is usable via the resource key in content streams,
    e.g. ``BT /F0 12 Tf <004800650068> Tj ET``.

    Args:
        pdf: The pikepdf.Pdf to embed into.
        resources: The page (or form XObject) ``/Resources`` dictionary.
        prepared: A :class:`PreparedFont` from :func:`prepare_truetype_font`.
    """
    base_font_name = pikepdf.Name(f"/{prepared.postscript_name}")

    # --- FontFile2 (embedded font program) ---
    font_stream = pikepdf.Stream(pdf, prepared.font_bytes)
    font_stream["/Length1"] = len(prepared.font_bytes)
    font_file = pdf.make_indirect(font_stream)

    # --- CIDSet ---
    used_gids = set(prepared.gid_for_codepoint.values())
    # Always include gid 0 (.notdef)
    used_gids.add(0)
    cidset_bytes = _build_cidset(used_gids)
    cidset_stream = pdf.make_indirect(pikepdf.Stream(pdf, cidset_bytes))

    # --- FontDescriptor ---
    font_descriptor = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/FontDescriptor"),
                "/FontName": base_font_name,
                "/Flags": prepared.flags,
                "/FontBBox": pikepdf.Array(prepared.font_bbox),
                "/ItalicAngle": prepared.italic_angle,
                "/Ascent": prepared.ascent,
                "/Descent": prepared.descent,
                "/CapHeight": prepared.cap_height,
                "/StemV": prepared.stem_v,
                "/FontFile2": font_file,
                "/CIDSet": cidset_stream,
            }
        )
    )

    # --- /W array ---
    w_array = _build_w_array(prepared.width_for_gid)

    # --- CIDFontType2 (descendant) ---
    cid_system_info = pikepdf.Dictionary(
        {
            "/Registry": pikepdf.String("Adobe"),
            "/Ordering": pikepdf.String("Identity"),
            "/Supplement": 0,
        }
    )

    cid_font = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/CIDFontType2"),
                "/BaseFont": base_font_name,
                "/CIDSystemInfo": cid_system_info,
                "/FontDescriptor": font_descriptor,
                "/CIDToGIDMap": pikepdf.Name("/Identity"),
                "/DW": 1000,
                "/W": w_array,
            }
        )
    )

    # --- ToUnicode CMap ---
    cmap_bytes = build_bfchar_cmap(prepared.to_unicode, byte_width=2)
    tounicode_stream = pdf.make_indirect(pikepdf.Stream(pdf, cmap_bytes))

    # --- Type0 top-level font ---
    type0_font = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type0"),
                "/BaseFont": base_font_name,
                "/Encoding": pikepdf.Name("/Identity-H"),
                "/DescendantFonts": pikepdf.Array([cid_font]),
                "/ToUnicode": tounicode_stream,
            }
        )
    )

    # --- Register in resources ---
    if "/Font" not in resources:
        resources["/Font"] = pikepdf.Dictionary()
    resources["/Font"][pikepdf.Name(f"/{prepared.resource_key}")] = type0_font
