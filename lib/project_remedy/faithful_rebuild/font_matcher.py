"""Font fingerprinting and system font matching.

Fingerprints fonts found in source PDFs and matches them against system-installed
fonts (or their embedded programs) for faithful PDF rebuilding.

Key entry points:
  - :func:`fingerprint_pdf_font` — extract a :class:`FontFingerprint` from a
    ``pikepdf.Dictionary`` font resource.
  - :func:`scan_system_fonts` — scan system directories for glyf-backed
    TrueType (.ttf) files and build a :class:`FontIndex`.
  - :func:`match_font` — find the best match for a source fingerprint.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import pikepdf
from pikepdf import Dictionary, Name

from project_remedy.faithful_rebuild.models import FontFingerprint, FontMatch

try:
    from fontTools.ttLib import TTFont
except ImportError:  # pragma: no cover
    TTFont = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE14_FAMILIES: dict[str, str] = {
    "helvetica": "Helvetica",
    "courier": "Courier",
    "timesroman": "Times",
    "times": "Times",
    "symbol": "Symbol",
    "zapfdingbats": "ZapfDingbats",
}

# PostScript name stems used to resolve Base14 variants
_BASE14_PS_STEMS = {
    "helvetica", "courier", "times", "symbol", "zapfdingbats",
}

_SYSTEM_FONT_DIRS: list[str] = [
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
    "~/Library/Fonts",
    "/usr/share/fonts",
    "~/.fonts",
]

# Canary hard-restricts candidates to glyf-backed TrueType. .ttc is ambiguous
# (collection index needs disambiguation) and .otf is often OpenType-CFF which
# register_type0_font() cannot emit. Both are rejected at scan time for the
# Mode B default (``font_class="truetype_glyf"``).
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".ttf"})

# TrueType/OpenType sfnt version bytes that indicate a glyf-backed outline
# format suitable for /CIDFontType2 + /FontFile2. Anything else — OpenType-CFF
# ('OTTO'), Apple TrueType ('true'), TrueType Collection ('ttcf') — is rejected
# for the canary because register_type0_font() only emits /CIDFontType2.
_GLYF_BACKED_SFNT_VERSION: bytes = b"\x00\x01\x00\x00"

# OpenType-CFF sfnt version ('OTTO') — used by the simple-font (REMEDY-73)
# ``type1_cff`` class, where the Type1C CFF program is embedded via
# ``/FontFile3 /Subtype /Type1C``.
_OTTO_SFNT_VERSION: bytes = b"OTTO"

# Valid font_class values for scan_system_fonts().
#   truetype_glyf — glyf-backed .ttf (default; unchanged Mode B behavior)
#   truetype_any  — any .ttf regardless of sfnt version
#   type1_cff     — .otf with a CFF / CFF2 table (OpenType-CFF)
#   any           — union of the above
_VALID_FONT_CLASSES: frozenset[str] = frozenset(
    {"truetype_glyf", "truetype_any", "type1_cff", "any"}
)

# Map each class to the file extensions it considers during directory scan.
_CLASS_TO_EXTENSIONS: dict[str, frozenset[str]] = {
    "truetype_glyf": frozenset({".ttf"}),
    "truetype_any": frozenset({".ttf"}),
    "type1_cff": frozenset({".otf"}),
    "any": frozenset({".ttf", ".otf"}),
}


def _is_glyf_backed_truetype(font_bytes: bytes) -> bool:
    """Return True iff font_bytes starts with an sfnt version that indicates
    a glyf-backed TrueType outline format.

    Does not validate the rest of the font program; a caller must still open
    with fontTools.ttLib.TTFont() to confirm the program is well-formed. This
    is only the first-gate format filter.
    """
    if len(font_bytes) < 4:
        return False
    return font_bytes[:4] == _GLYF_BACKED_SFNT_VERSION


def _is_otto_sfnt(font_bytes: bytes) -> bool:
    """Return True iff font_bytes starts with the 'OTTO' sfnt version,
    which indicates OpenType-CFF.

    First-gate filter for the ``type1_cff`` scan class. A full confirmation
    still requires opening with fontTools to detect a ``CFF `` or ``CFF2``
    table.
    """
    if len(font_bytes) < 4:
        return False
    return font_bytes[:4] == _OTTO_SFNT_VERSION


def _accepts_font_class(
    font_class: str,
    *,
    suffix: str,
    header: bytes,
) -> bool:
    """First-gate filter used by :func:`_scan_system_fonts_impl`.

    Only checks extension + sfnt header. Deeper validation (e.g. the actual
    presence of a CFF table for ``type1_cff``) is performed by
    :func:`_fingerprint_ttf` / :func:`_fingerprint_otf` after the font is
    opened with fontTools.
    """
    if font_class not in _VALID_FONT_CLASSES:
        raise ValueError(
            f"invalid font_class {font_class!r}; "
            f"expected one of {sorted(_VALID_FONT_CLASSES)}"
        )
    if suffix not in _CLASS_TO_EXTENSIONS[font_class]:
        return False

    if font_class == "truetype_glyf":
        return _is_glyf_backed_truetype(header)
    if font_class == "truetype_any":
        return suffix == ".ttf"
    if font_class == "type1_cff":
        # .otf files almost always carry an 'OTTO' header when they hold
        # CFF outlines; we accept the header gate here and let fontTools
        # confirm the CFF / CFF2 table during fingerprinting.
        return _is_otto_sfnt(header)
    if font_class == "any":
        # Any of the above headers is acceptable.
        return (
            _is_glyf_backed_truetype(header)
            or _is_otto_sfnt(header)
            or suffix == ".ttf"
        )
    return False  # pragma: no cover — unreachable given validation above

# PDF font descriptor flag bits
_FLAG_FIXED_PITCH = 1 << 0
_FLAG_SERIF = 1 << 1
_FLAG_ITALIC = 1 << 6
_FLAG_FORCE_BOLD = 1 << 18


# ---------------------------------------------------------------------------
# FontIndex
# ---------------------------------------------------------------------------


@dataclass
class FontIndex:
    """Collection of fingerprinted system fonts with fast lookups.

    After calling :meth:`build_indices`, the index provides O(1) lookup by
    PostScript name and O(1) lookup by normalized family name.
    """

    entries: list[FontFingerprint] = field(default_factory=list)
    _by_ps_name: dict[str, FontFingerprint] = field(default_factory=dict)
    _by_family: dict[str, list[FontFingerprint]] = field(default_factory=dict)

    def build_indices(self) -> None:
        """Populate ``_by_ps_name`` and ``_by_family`` from :attr:`entries`."""
        self._by_ps_name.clear()
        self._by_family.clear()
        for fp in self.entries:
            # PostScript name index — first occurrence wins
            norm_ps = _normalize_name(fp.postscript_name)
            if norm_ps and norm_ps not in self._by_ps_name:
                self._by_ps_name[norm_ps] = fp
            # Also store by exact (un-normalized) PS name for precise lookup
            if fp.postscript_name and fp.postscript_name not in self._by_ps_name:
                self._by_ps_name[fp.postscript_name] = fp

            # Family index
            norm_fam = _normalize_name(fp.family)
            if norm_fam:
                self._by_family.setdefault(norm_fam, []).append(fp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fingerprint_pdf_font(
    resource_key: str,
    font_dict: pikepdf.Dictionary,
) -> FontFingerprint:
    """Extract a :class:`FontFingerprint` from a PDF font dictionary.

    Handles Type1, TrueType, and CIDFont dictionaries.  For Base14 fonts
    (Helvetica, Times-Roman, Courier, etc.) hard-coded classification is used.

    Args:
        resource_key: Resource name as it appears in the page dict (e.g. ``"F1"``).
        font_dict: The ``pikepdf.Dictionary`` for the font resource.

    Returns:
        A fully populated :class:`FontFingerprint`.
    """
    base_font = _str_or(font_dict.get(Name("/BaseFont")), "")
    # Strip leading '/' from pikepdf Name objects
    if base_font.startswith("/"):
        base_font = base_font[1:]

    ps_name = base_font
    family = _family_from_ps_name(ps_name)

    # Defaults
    weight = 400
    italic = False
    serif: bool | None = None
    mono = False
    cap_height: int | None = None
    x_height: int | None = None
    panose: tuple[int, ...] | None = None
    width_class = 5
    glyph_coverage: frozenset[int] = frozenset()
    embedded_program: bytes | None = None

    # ---- Detect bold/italic from PostScript name suffixes ----
    ps_lower = ps_name.lower()
    if "bold" in ps_lower:
        weight = 700
    if "italic" in ps_lower or "oblique" in ps_lower:
        italic = True

    # ---- Base14 classification ----
    base14_serif, base14_mono = _classify_base14(ps_name)
    if base14_serif is not None or base14_mono:
        serif = base14_serif
        mono = base14_mono

    # ---- Choose dict to read FontDescriptor from ----
    # For /Type0 fonts, the real FontDescriptor lives on the descendant
    # CIDFontType2 dict, not on the top-level dict. Resolve it here so every
    # downstream field lookup uses the correct dict.
    subtype = _str_or(font_dict.get(Name("/Subtype")), "")
    font_dict_for_descriptor: pikepdf.Dictionary = font_dict
    if subtype == "/Type0":
        descendants = font_dict.get(Name("/DescendantFonts"))
        if descendants is not None and len(descendants) > 0:
            # pikepdf transparently dereferences indirect objects when
            # indexing an Array — no explicit .get_object() call is needed
            # (and pikepdf.Object does not expose such a method).
            descendant = descendants[0]
            if descendant.get(Name("/FontDescriptor")) is not None:
                font_dict_for_descriptor = descendant

    # ---- FontDescriptor ----
    fd = font_dict_for_descriptor.get(Name("/FontDescriptor"))
    if fd is not None:
        fd = _resolve(fd)

        # Flags
        flags = _int_or(fd.get(Name("/Flags")), 0)
        if flags & _FLAG_ITALIC:
            italic = True
        if flags & _FLAG_SERIF:
            serif = True
        if flags & _FLAG_FIXED_PITCH:
            mono = True
        if flags & _FLAG_FORCE_BOLD:
            weight = max(weight, 700)

        # Explicit weight
        explicit_weight = _int_or(fd.get(Name("/FontWeight")), 0)
        if explicit_weight > 0:
            weight = explicit_weight

        # Metrics
        cap_height = _int_or(fd.get(Name("/CapHeight")), None)
        x_height = _int_or(fd.get(Name("/XHeight")), None)

        # Embedded program (TrueType)
        font_file2 = fd.get(Name("/FontFile2"))
        if font_file2 is not None:
            font_file2 = _resolve(font_file2)
            try:
                embedded_program = bytes(font_file2.read_bytes())
            except Exception:
                embedded_program = None

        # Embedded program (CFF / Type1C)
        if embedded_program is None:
            font_file3 = fd.get(Name("/FontFile3"))
            if font_file3 is not None:
                font_file3 = _resolve(font_file3)
                try:
                    embedded_program = bytes(font_file3.read_bytes())
                except Exception:
                    pass

        # Embedded program (Type1)
        if embedded_program is None:
            font_file1 = fd.get(Name("/FontFile"))
            if font_file1 is not None:
                font_file1 = _resolve(font_file1)
                try:
                    embedded_program = bytes(font_file1.read_bytes())
                except Exception:
                    pass

    # ---- Glyph coverage from embedded program ----
    if embedded_program is not None and TTFont is not None:
        glyph_coverage = _glyph_coverage_from_bytes(embedded_program)
        # Also try to extract PANOSE, cap_height, x_height from embedded
        extra = _metrics_from_bytes(embedded_program)
        if extra:
            if extra.get("panose"):
                panose = extra["panose"]
            if cap_height is None and extra.get("cap_height"):
                cap_height = extra["cap_height"]
            if x_height is None and extra.get("x_height"):
                x_height = extra["x_height"]
            if extra.get("width_class"):
                width_class = extra["width_class"]

    return FontFingerprint(
        source_id=resource_key,
        family=family,
        postscript_name=ps_name,
        weight=weight,
        width_class=width_class,
        italic=italic,
        serif=serif,
        mono=mono,
        panose=panose,
        cap_height=cap_height,
        x_height=x_height,
        glyph_coverage=glyph_coverage,
        embedded_program=embedded_program,
    )


@lru_cache(maxsize=4)
def scan_system_fonts(*, font_class: str = "truetype_glyf") -> FontIndex:
    """Cached system font scan — returns a :class:`FontIndex` of candidates.

    Args:
        font_class: Which font family to collect. One of:

          * ``"truetype_glyf"`` (default) — ``.ttf`` with sfnt version
            ``0x00010000`` (glyf-backed TrueType). Original Mode B behavior;
            this value is preserved by all existing callers.
          * ``"truetype_any"`` — ``.ttf`` regardless of sfnt version.
          * ``"type1_cff"`` — ``.otf`` with an OpenType-CFF outline
            (``OTTO`` sfnt version). Target for REMEDY-73 simple-font
            replacement (``/FontFile3`` + ``/Subtype /Type1C``).
          * ``"any"`` — the union of the above.

    Returns:
        A :class:`FontIndex` with populated lookup dicts.

    Raises:
        ValueError: if ``font_class`` is not one of the above.

    FontIndex is treated as read-only by downstream callers. If tests or
    internals need to rebuild the index (e.g., after changes in
    ``_ALLOWED_EXTENSIONS`` or for a different class), call
    ``scan_system_fonts.cache_clear()``.
    """
    if font_class not in _VALID_FONT_CLASSES:
        raise ValueError(
            f"invalid font_class {font_class!r}; "
            f"expected one of {sorted(_VALID_FONT_CLASSES)}"
        )
    return _scan_system_fonts_impl(font_class=font_class)


def _scan_system_fonts_impl(*, font_class: str = "truetype_glyf") -> FontIndex:
    """Scan system font directories and build a :class:`FontIndex`.

    Scans macOS and Linux standard font directories for fonts matching the
    requested ``font_class`` and fingerprints each with fontTools. For
    ``type1_cff``, accepted candidates must additionally have a ``CFF ``
    or ``CFF2`` table after opening — the ``.otf`` extension + ``OTTO``
    header alone are not sufficient.

    Returns:
        A :class:`FontIndex` with populated lookup dicts.
    """
    index = FontIndex()
    seen_paths: set[Path] = set()
    extensions = _CLASS_TO_EXTENSIONS[font_class]

    for dir_str in _SYSTEM_FONT_DIRS:
        font_dir = Path(dir_str).expanduser()
        if not font_dir.is_dir():
            continue
        for path in font_dir.iterdir():
            suffix = path.suffix.lower()
            if suffix not in extensions:
                continue
            try:
                with path.open("rb") as handle:
                    header = handle.read(4)
            except OSError:
                continue
            if not _accepts_font_class(font_class, suffix=suffix, header=header):
                continue
            real = path.resolve()
            if real in seen_paths:
                continue
            seen_paths.add(real)

            # For type1_cff we require a CFF/CFF2 table to be actually
            # present in the opened font. The 'OTTO' header gate is a
            # necessary-but-not-sufficient first check.
            if font_class == "type1_cff" and not _has_cff_table(path):
                continue

            fp = _fingerprint_ttf(path)
            if fp is not None:
                index.entries.append(fp)

    index.build_indices()
    return index


def _has_cff_table(path: Path) -> bool:
    """Return True iff *path* opens as a font with a ``CFF`` / ``CFF2`` table.

    Used as the final-gate filter for the ``type1_cff`` scan class.
    """
    if TTFont is None:
        return False
    try:
        with TTFont(str(path), fontNumber=0, lazy=True) as font:
            return ("CFF " in font) or ("CFF2" in font)
    except Exception:
        return False


def match_font(
    source: FontFingerprint,
    index: FontIndex,
    required_codepoints: set[int] | None = None,
    *,
    min_confidence: float = 0.60,
    require_codepoints: frozenset[int] | None = None,
) -> FontMatch:
    """Find the best matching font for *source* in *index*.

    Match cascade:
      a. Embedded program is glyf-backed TrueType -> ``use_embedded=True``, confidence 1.0
      b. Exact PostScript name match -> confidence 0.95
      c. Family name match -> confidence 0.85
      d. Scored ranking across all candidates -> variable confidence
      e. No match -> confidence 0.0, ``fallback_reason`` set

    Canary Mode B hardening:
      - Embedded-font fast path now requires glyf-backed TrueType (sfnt
        version ``0x00010000``). OpenType-CFF (``OTTO``) and other sfnt
        variants fall through to the system scan because downstream
        ``register_type0_font()`` only emits ``/CIDFontType2`` + ``/FontFile2``.
      - Any match with confidence below ``min_confidence`` is rejected —
        the function returns ``FontMatch(confidence=0.0, fallback_reason=...)``.
      - If ``require_codepoints`` is given, the selected candidate must cover
        every codepoint; otherwise the match is rejected. If the embedded
        program itself does not cover ``require_codepoints``, ``match_font``
        falls through to the system-scan cascade rather than accepting the
        embedded fast path — giving system candidates a chance to satisfy
        coverage before outright rejection.

    Args:
        source: The font fingerprint to match.
        index: A pre-built :class:`FontIndex`.
        required_codepoints: Legacy soft-coverage hint. When provided, scoring
            is reduced for candidates missing codepoints but no hard rejection
            occurs. Retained for backwards compatibility with existing callers.
        min_confidence: Minimum confidence threshold. Matches below this return
            ``FontMatch(confidence=0.0, fallback_reason=...)``. Default 0.60.
        require_codepoints: Hard-required codepoint coverage. A candidate that
            does not cover every codepoint in this set is rejected. When non-
            empty, the check is performed against the resolved font file on
            disk (via fontTools) for accuracy. For the embedded fast path,
            coverage is verified against ``source.glyph_coverage``; on miss
            the match falls through to the system-scan cascade.

    Returns:
        A :class:`FontMatch`. Never ``None``. On rejection,
        ``confidence == 0.0`` and ``fallback_reason`` is populated.
    """
    # (a) Embedded program — fast path only for glyf-backed TrueType.
    # OpenType-CFF ('OTTO') cannot be emitted by register_type0_font(), so
    # those programs must fall through to the system scan to find a glyf
    # substitute.
    if (
        source.embedded_program is not None
        and _is_glyf_backed_truetype(source.embedded_program[:4])
        and _is_valid_truetype(source.embedded_program)
    ):
        # Enforce require_codepoints against source.glyph_coverage before
        # accepting the fast path. If the embedded program doesn't cover the
        # required codepoints, fall through to the system-scan cascade so a
        # system candidate gets a chance to satisfy coverage (the threshold
        # block at the end handles final reject if no candidate works).
        embedded_covers_required = True
        if require_codepoints:
            missing_in_embedded = require_codepoints - source.glyph_coverage
            if missing_in_embedded:
                embedded_covers_required = False
        if embedded_covers_required:
            return FontMatch(
                source=source,
                resolved_path=None,
                use_embedded=True,
                confidence=1.0,
                fallback_reason=None,
            )

    best_match: FontMatch | None = None

    # Filter-then-rank: when require_codepoints is non-empty, restrict the
    # cascade to candidates that cover all required codepoints. This prevents
    # the matcher from picking a name-similar-but-coverage-incomplete candidate
    # and rejecting outright while other candidates would serve.
    #
    # Example (v3 canary heerf-CARES bug): Calibri,Bold source -> stage (d)
    # picks Trebuchet MS Bold by name similarity -> post-check rejects for
    # missing Latin Extended-B. With the filter, only candidates that cover
    # all required codepoints enter the cascade, so the scorer picks
    # Arial-BoldMT (or similar) instead.
    effective_index = index
    if require_codepoints:
        covering_entries = [
            e for e in index.entries
            if e.glyph_coverage and require_codepoints.issubset(e.glyph_coverage)
        ]
        if covering_entries:
            # Build a filtered FontIndex. FontIndex(entries=...) + build_indices()
            # constructs the _by_ps_name and _by_family lookups on the subset.
            effective_index = FontIndex(entries=covering_entries)
            effective_index.build_indices()
        else:
            # No candidate covers all required codepoints. Return an informative
            # structured failure immediately rather than letting the cascade pick
            # a partial-coverage candidate and then rejecting it with the less
            # precise "no system font found" message.
            sample = sorted(require_codepoints)[:5]
            return FontMatch(
                source=source,
                resolved_path=None,
                use_embedded=False,
                confidence=0.0,
                fallback_reason=(
                    f"No candidate in index covers "
                    f"{len(require_codepoints)} required codepoints: {sample}"
                ),
            )

    # (b) Exact PostScript name match
    norm_ps = _normalize_name(source.postscript_name)
    candidate = (
        effective_index._by_ps_name.get(norm_ps)
        or effective_index._by_ps_name.get(source.postscript_name)
    )
    if candidate is not None:
        conf = 0.95
        if required_codepoints and candidate.glyph_coverage:
            missing = required_codepoints - candidate.glyph_coverage
            if missing:
                conf *= max(0.5, 1.0 - len(missing) / len(required_codepoints))
        best_match = FontMatch(
            source=source,
            resolved_path=candidate.path,
            use_embedded=False,
            confidence=conf,
            fallback_reason=None if conf >= 0.9 else "some required codepoints missing",
        )

    # (c) Family name match
    if best_match is None:
        norm_fam = _normalize_name(source.family)
        family_candidates = effective_index._by_family.get(norm_fam, [])
        if family_candidates:
            best_fp, best_score = _best_from_list(source, family_candidates, required_codepoints)
            conf = max(0.85, min(best_score, 0.94))
            best_match = FontMatch(
                source=source,
                resolved_path=best_fp.path,
                use_embedded=False,
                confidence=conf,
                fallback_reason=None,
            )

    # (d) Scored ranking across all entries
    if best_match is None and effective_index.entries:
        best_fp, best_score = _best_from_list(source, effective_index.entries, required_codepoints)
        if best_score > 0.3:
            best_match = FontMatch(
                source=source,
                resolved_path=best_fp.path,
                use_embedded=False,
                confidence=round(best_score, 4),
                fallback_reason=f"no exact match; best scored candidate: {best_fp.postscript_name}",
            )

    # (e) Nothing found at all
    if best_match is None:
        return FontMatch(
            source=source,
            resolved_path=None,
            use_embedded=False,
            confidence=0.0,
            fallback_reason=f"no system font found for {source.postscript_name!r}",
        )

    # --- Canary Mode B threshold checks -------------------------------------
    # (1) Confidence floor. Reject if below the caller's min_confidence.
    if best_match.confidence < min_confidence:
        return FontMatch(
            source=source,
            resolved_path=None,
            use_embedded=False,
            confidence=0.0,
            fallback_reason=(
                f"No candidate met min_confidence={min_confidence:.2f} "
                f"(best={best_match.confidence:.2f})"
            ),
        )

    # (2) Hard-required codepoint coverage. Open the resolved font file and
    # verify every required codepoint is present.
    # Skip this check when the cascade already operated on a pre-filtered index
    # (effective_index is not index): every candidate in that pool was already
    # verified against require_codepoints via glyph_coverage frozensets, so
    # re-opening with fontTools is redundant and would fail for stub test paths.
    if require_codepoints and best_match.resolved_path is not None and effective_index is index:
        try:
            if TTFont is None:
                raise RuntimeError("fontTools not available")
            tt = TTFont(str(best_match.resolved_path))
            try:
                cmap = tt.getBestCmap() or {}
                missing = set(require_codepoints) - set(cmap.keys())
            finally:
                tt.close()
        except Exception as exc:
            return FontMatch(
                source=source,
                resolved_path=None,
                use_embedded=False,
                confidence=0.0,
                fallback_reason=f"Candidate unreadable for coverage check: {exc}",
            )
        if missing:
            sample = sorted(missing)[:5]
            return FontMatch(
                source=source,
                resolved_path=None,
                use_embedded=False,
                confidence=0.0,
                fallback_reason=(
                    f"Candidate missing {len(missing)} required codepoints: "
                    f"{sample}"
                ),
            )

    return best_match


# ---------------------------------------------------------------------------
# Helper functions (public for testing)
# ---------------------------------------------------------------------------


def _normalize_name(value: str) -> str:
    """Lowercase, keep only alphanumeric characters."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _classify_base14(ps_name: str) -> tuple[bool | None, bool]:
    """Return ``(serif, mono)`` classification for a Base14 font.

    Returns ``(None, False)`` for unknown fonts.
    """
    stem = _normalize_name(ps_name)
    # Strip known suffixes
    for suffix in ("bold", "italic", "oblique", "bolditalic", "boldoblique", "roman"):
        stem = stem.removesuffix(suffix)

    if stem.startswith("times"):
        return (True, False)
    if stem.startswith("helvetica"):
        return (False, False)
    if stem.startswith("courier"):
        return (None, True)  # Courier serif classification varies
    if stem.startswith("symbol") or stem.startswith("zapfdingbats"):
        return (None, False)

    return (None, False)


def _is_valid_truetype(data: bytes) -> bool:
    """Return True if *data* can be parsed as a TrueType or OpenType font."""
    if TTFont is None or not data:
        return False
    try:
        with io.BytesIO(data) as buf:
            font = TTFont(buf)
            font.close()
        return True
    except Exception:
        return False


def _fingerprint_ttf(path: Path) -> FontFingerprint | None:
    """Fingerprint a TrueType/OpenType font file on disk.

    Returns None if the file cannot be parsed.
    """
    if TTFont is None:
        return None
    try:
        font = TTFont(str(path), fontNumber=0)
    except Exception:
        logger.debug("Could not parse font file: %s", path)
        return None

    try:
        return _fingerprint_from_ttfont(font, path)
    except Exception:
        logger.debug("Error fingerprinting %s", path, exc_info=True)
        return None
    finally:
        font.close()


def _fingerprint_from_ttfont(font: "TTFont", path: Path | None = None) -> FontFingerprint:
    """Build a FontFingerprint from an open fontTools TTFont."""
    # PostScript name
    name_table = font.get("name")
    ps_name = ""
    family = ""
    if name_table:
        # Name ID 6 = PostScript name
        rec = name_table.getName(6, 3, 1, 0x0409) or name_table.getName(6, 1, 0, 0)
        if rec:
            ps_name = str(rec)
        # Name ID 1 = Font Family
        rec = name_table.getName(1, 3, 1, 0x0409) or name_table.getName(1, 1, 0, 0)
        if rec:
            family = str(rec)

    # OS/2 table metrics
    weight = 400
    width_class = 5
    italic = False
    serif: bool | None = None
    mono = False
    panose: tuple[int, ...] | None = None
    cap_height: int | None = None
    x_height: int | None = None

    os2 = font.get("OS/2")
    if os2:
        weight = getattr(os2, "usWeightClass", 400) or 400
        width_class = getattr(os2, "usWidthClass", 5) or 5
        fs_selection = getattr(os2, "fsSelection", 0) or 0
        italic = bool(fs_selection & 1)  # bit 0 = italic

        # PANOSE
        panose_obj = getattr(os2, "panose", None)
        if panose_obj is not None:
            try:
                panose_bytes = [
                    panose_obj.bFamilyType,
                    panose_obj.bSerifStyle,
                    panose_obj.bWeight,
                    panose_obj.bProportion,
                    panose_obj.bContrast,
                    panose_obj.bStrokeVariation,
                    panose_obj.bArmStyle,
                    panose_obj.bLetterForm,
                    panose_obj.bMidline,
                    panose_obj.bXHeight,
                ]
                if any(b != 0 for b in panose_bytes):
                    panose = tuple(panose_bytes)
            except (AttributeError, TypeError):
                pass

        cap_height_val = getattr(os2, "sCapHeight", None)
        if cap_height_val and cap_height_val > 0:
            cap_height = int(cap_height_val)

        x_height_val = getattr(os2, "sxHeight", None)
        if x_height_val and x_height_val > 0:
            x_height = int(x_height_val)

        # Serif classification from sFamilyClass
        family_class = getattr(os2, "sFamilyClass", 0) or 0
        major = (family_class >> 8) & 0xFF
        if major in (1, 2, 3, 4, 5, 7):
            serif = True
        elif major in (8,):
            serif = False

        # Monospaced from Panose bProportion == 9
        if panose and panose[3] == 9:
            mono = True

    # Also check post table for monospaced
    post = font.get("post")
    if post:
        if getattr(post, "isFixedPitch", 0):
            mono = True

    # Glyph coverage from cmap
    glyph_coverage = _glyph_coverage_from_ttfont(font)

    return FontFingerprint(
        source_id="",  # filled by caller if needed
        family=family,
        postscript_name=ps_name,
        weight=weight,
        width_class=width_class,
        italic=italic,
        serif=serif,
        mono=mono,
        panose=panose,
        cap_height=cap_height,
        x_height=x_height,
        glyph_coverage=glyph_coverage,
        embedded_program=None,
        path=path,
    )


def _scale_1000(value: int, units_per_em: int) -> int:
    """Scale a font-unit value to 1000 units-per-em."""
    if units_per_em <= 0:
        return value
    return round(value * 1000 / units_per_em)


def _panose_distance(a: tuple[int, ...] | None, b: tuple[int, ...] | None) -> int:
    """Sum of per-byte absolute differences between two PANOSE tuples.

    Returns a large value (1000) when either input is None.
    """
    if a is None or b is None:
        return 1000
    length = min(len(a), len(b))
    return sum(abs(a[i] - b[i]) for i in range(length))


def _score_candidate(
    source: FontFingerprint,
    candidate: FontFingerprint,
    required_codepoints: set[int] | None,
) -> float:
    """Score how well *candidate* matches *source*.

    Returns a value in [0, 1] where 1.0 is a perfect match.

    Scoring components (weights sum to 1.0):
      - Weight match: 0.25
      - Serif/sans match: 0.20
      - Mono match: 0.10
      - Italic match: 0.15
      - Cap height similarity: 0.05
      - x-height similarity: 0.05
      - PANOSE distance: 0.10
      - Codepoint coverage: 0.10
    """
    score = 0.0

    # Weight match (0.25)
    weight_diff = abs(source.weight - candidate.weight)
    if weight_diff == 0:
        score += 0.25
    elif weight_diff <= 100:
        score += 0.20
    elif weight_diff <= 200:
        score += 0.10
    else:
        score += 0.0

    # Serif/sans match (0.20)
    if source.serif is None or candidate.serif is None:
        score += 0.10  # unknown, partial credit
    elif source.serif == candidate.serif:
        score += 0.20
    else:
        score += 0.0

    # Mono match (0.10)
    if source.mono == candidate.mono:
        score += 0.10
    else:
        score += 0.0

    # Italic match (0.15)
    if source.italic == candidate.italic:
        score += 0.15
    else:
        score += 0.0

    # Cap height similarity (0.05)
    if source.cap_height is not None and candidate.cap_height is not None:
        ch_diff = abs(source.cap_height - candidate.cap_height)
        if ch_diff <= 20:
            score += 0.05
        elif ch_diff <= 50:
            score += 0.03
        else:
            score += 0.01
    else:
        score += 0.025  # unknown, partial

    # x-height similarity (0.05)
    if source.x_height is not None and candidate.x_height is not None:
        xh_diff = abs(source.x_height - candidate.x_height)
        if xh_diff <= 20:
            score += 0.05
        elif xh_diff <= 50:
            score += 0.03
        else:
            score += 0.01
    else:
        score += 0.025  # unknown, partial

    # PANOSE distance (0.10)
    pd = _panose_distance(source.panose, candidate.panose)
    if pd == 0:
        score += 0.10
    elif pd <= 5:
        score += 0.07
    elif pd <= 15:
        score += 0.04
    elif pd < 1000:
        score += 0.01
    else:
        score += 0.05  # both None — neutral

    # Codepoint coverage (0.10)
    if required_codepoints and candidate.glyph_coverage:
        covered = required_codepoints & candidate.glyph_coverage
        ratio = len(covered) / len(required_codepoints) if required_codepoints else 1.0
        score += 0.10 * ratio
    elif not required_codepoints:
        score += 0.10  # no requirement, full marks
    else:
        score += 0.05  # unknown coverage, partial

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _str_or(value: object, default: str) -> str:
    """Convert pikepdf Name or String to str, or return *default*."""
    if value is None:
        return default
    return str(value)


def _int_or(value: object, default: int | None) -> int | None:
    """Convert pikepdf numeric to int, or return *default*."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve(obj: object) -> object:
    """Resolve an indirect reference if needed."""
    if isinstance(obj, pikepdf.Object):
        try:
            return pikepdf.Object.parse(obj.unparse())
        except Exception:
            pass
    return obj


def _family_from_ps_name(ps_name: str) -> str:
    """Derive a family name from a PostScript name.

    Examples:
      ``"TimesNewRomanPSMT"`` -> ``"TimesNewRomanPSMT"``
      ``"Helvetica-BoldOblique"`` -> ``"Helvetica"``
      ``"ArialMT"`` -> ``"ArialMT"``
    """
    # For Base14, strip suffixes
    stem = ps_name
    for suffix in ("-Bold", "-Italic", "-BoldItalic", "-Oblique", "-BoldOblique", "-Roman"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _glyph_coverage_from_bytes(data: bytes) -> frozenset[int]:
    """Extract glyph coverage (Unicode codepoints) from font program bytes."""
    if TTFont is None or not data:
        return frozenset()
    try:
        with io.BytesIO(data) as buf:
            font = TTFont(buf)
            result = _glyph_coverage_from_ttfont(font)
            font.close()
            return result
    except Exception:
        return frozenset()


def _glyph_coverage_from_ttfont(font: "TTFont") -> frozenset[int]:
    """Extract glyph coverage from an open TTFont."""
    codepoints: set[int] = set()
    try:
        cmap = font.getBestCmap()
        if cmap:
            codepoints.update(cmap.keys())
    except Exception:
        pass
    return frozenset(codepoints)


def _metrics_from_bytes(data: bytes) -> dict | None:
    """Extract PANOSE, cap_height, x_height, width_class from font bytes."""
    if TTFont is None or not data:
        return None
    try:
        with io.BytesIO(data) as buf:
            font = TTFont(buf)
            result: dict = {}
            os2 = font.get("OS/2")
            if os2:
                panose_obj = getattr(os2, "panose", None)
                if panose_obj is not None:
                    try:
                        panose_bytes = [
                            panose_obj.bFamilyType,
                            panose_obj.bSerifStyle,
                            panose_obj.bWeight,
                            panose_obj.bProportion,
                            panose_obj.bContrast,
                            panose_obj.bStrokeVariation,
                            panose_obj.bArmStyle,
                            panose_obj.bLetterForm,
                            panose_obj.bMidline,
                            panose_obj.bXHeight,
                        ]
                        if any(b != 0 for b in panose_bytes):
                            result["panose"] = tuple(panose_bytes)
                    except (AttributeError, TypeError):
                        pass

                cap_h = getattr(os2, "sCapHeight", None)
                if cap_h and cap_h > 0:
                    result["cap_height"] = int(cap_h)

                x_h = getattr(os2, "sxHeight", None)
                if x_h and x_h > 0:
                    result["x_height"] = int(x_h)

                wc = getattr(os2, "usWidthClass", None)
                if wc and wc > 0:
                    result["width_class"] = int(wc)

            font.close()
            return result if result else None
    except Exception:
        return None


def _best_from_list(
    source: FontFingerprint,
    candidates: Sequence[FontFingerprint],
    required_codepoints: set[int] | None,
) -> tuple[FontFingerprint, float]:
    """Find the best match in *candidates* by score."""
    best_fp = candidates[0]
    best_score = -1.0
    for fp in candidates:
        s = _score_candidate(source, fp, required_codepoints)
        if s > best_score:
            best_score = s
            best_fp = fp
    return best_fp, best_score
