"""Shared pure helpers for reading font programs embedded in PDF objects.

Used by font_analysis.py (read-only CID→Unicode recovery) and can be
used by pdf_fixer.py (ToUnicode synthesis) to eliminate duplication.
All functions are pure: they do not mutate any PDF object.
"""

from __future__ import annotations

import pikepdf


def load_embedded_ttfont(
    descriptor: pikepdf.Dictionary | None,
) -> "TTFont | None":
    """Load the embedded font program from a FontDescriptor as a fontTools TTFont.

    Tries /FontFile2 (TrueType), /FontFile3 (CFF/OpenType-CFF), /FontFile (Type1)
    in that order. Handles CFF fonts that need sfntVersion='OTTO'.

    Returns TTFont on success, None on failure. Caller must close() the TTFont.
    """
    if descriptor is None:
        return None

    font_stream = descriptor.get("/FontFile2")
    is_cff = False
    if font_stream is None:
        font_stream = descriptor.get("/FontFile3")
        if font_stream is not None:
            is_cff = True
    if font_stream is None:
        font_stream = descriptor.get("/FontFile")
    if font_stream is None:
        return None

    try:
        from io import BytesIO
        from fontTools.ttLib import TTFont

        font_bytes = bytes(font_stream.read_bytes())
        bio = BytesIO(font_bytes)

        tt = None
        if is_cff:
            for sfnt in ("OTTO", None):
                try:
                    bio.seek(0)
                    tt = TTFont(bio, sfntVersion=sfnt)
                    break
                except Exception:
                    tt = None
        if tt is None:
            bio.seek(0)
            tt = TTFont(bio)
        return tt
    except Exception:
        return None


def parse_cidtogidmap(
    cidtogidmap: pikepdf.Object | None,
    gid_to_unicode: dict[int, int],
) -> dict[int, int]:
    """Apply a /CIDToGIDMap to translate GID→Unicode into CID→Unicode.

    Handles /Identity (CID==GID), None (assume identity), and stream
    (big-endian uint16 array of GID values indexed by CID).
    """
    if cidtogidmap is None or str(cidtogidmap) == "/Identity":
        return dict(gid_to_unicode)

    if not hasattr(cidtogidmap, "read_bytes"):
        return dict(gid_to_unicode)

    try:
        map_bytes = bytes(cidtogidmap.read_bytes())
    except Exception:
        return dict(gid_to_unicode)

    cid_to_unicode: dict[int, int] = {}
    for cid in range(len(map_bytes) // 2):
        gid = (map_bytes[cid * 2] << 8) | map_bytes[cid * 2 + 1]
        if gid in gid_to_unicode:
            cid_to_unicode[cid] = gid_to_unicode[gid]
    return cid_to_unicode


_PRINTABLE_RANGE = range(0x20, 0x110000)
_UNICODE_SPECIALS = frozenset({0xFEFF, 0xFFFE, 0xFFFF})


def _filter_printable(mapping: dict[int, int]) -> dict[int, int]:
    return {
        cid: uni for cid, uni in mapping.items()
        if uni in _PRINTABLE_RANGE and uni not in _UNICODE_SPECIALS
    }


def recover_cid_unicode_via_cmap(
    descriptor: pikepdf.Dictionary | None,
    cidtogidmap: pikepdf.Object | None,
) -> dict[int, int] | None:
    """Layer 2: extract GID→Unicode from embedded font's cmap table,
    apply CIDToGIDMap, return CID→Unicode. Returns None on failure."""
    tt = load_embedded_ttfont(descriptor)
    if tt is None:
        return None

    try:
        best_cmap = tt.getBestCmap()
        if not best_cmap:
            return None

        gid_to_unicode: dict[int, int] = {}
        for unicode_val, glyph_name in best_cmap.items():
            try:
                gid = tt.getGlyphID(glyph_name)
            except KeyError:
                continue
            if gid not in gid_to_unicode:
                gid_to_unicode[gid] = unicode_val

        if not gid_to_unicode:
            return None

        result = parse_cidtogidmap(cidtogidmap, gid_to_unicode)
        result = _filter_printable(result)
        return result if result else None
    except Exception:
        return None
    finally:
        try:
            tt.close()
        except Exception:
            pass


def recover_cid_unicode_via_post(
    descriptor: pikepdf.Dictionary | None,
    cidtogidmap: pikepdf.Object | None,
) -> dict[int, int] | None:
    """Layer 3: read post table glyph names, resolve via Adobe Glyph List,
    apply CIDToGIDMap. Falls back when Layer 2 (cmap) returns empty.
    Skips post format 3.0 (no glyph names)."""
    tt = load_embedded_ttfont(descriptor)
    if tt is None:
        return None

    try:
        from fontTools.agl import toUnicode as agl_to_unicode

        post_table = tt.get("post")
        if post_table is None or getattr(post_table, "formatType", 3.0) == 3.0:
            return None

        glyph_order = tt.getGlyphOrder()
        gid_to_unicode: dict[int, int] = {}
        for gid, name in enumerate(glyph_order):
            if not name or name == ".notdef" or name.startswith("glyph"):
                continue
            unicode_str = agl_to_unicode(name)
            if unicode_str:
                gid_to_unicode[gid] = ord(unicode_str[0])

        if not gid_to_unicode:
            return None

        result = parse_cidtogidmap(cidtogidmap, gid_to_unicode)
        result = _filter_printable(result)
        return result if result else None
    except Exception:
        return None
    finally:
        try:
            tt.close()
        except Exception:
            pass
