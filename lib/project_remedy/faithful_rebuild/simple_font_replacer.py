"""Simple-font replacement and encoding repair (REMEDY-73).

Phase 1 — **Encoding repair for veraPDF 7.21.6-3.**

    A symbolic TrueType font (``/FontDescriptor /Flags`` bit 3 set, value 4)
    must not carry an ``/Encoding`` entry per PDF/UA-1 7.21.6-3.  The repair
    here is purely metadata: delete the ``/Encoding`` key from qualifying
    symbolic TrueType font dictionaries.  No glyph data is touched, no font
    program is rewritten, and the content stream keeps referring to the same
    character codes.

    Public API:

      - :class:`EncodingRepairEligibility` — dataclass describing whether a
        font qualifies for encoding repair and why / why not.
      - :func:`check_encoding_repair_eligibility` — inspect a font dict.
      - :func:`repair_encoding_on_font` — mutate a single font dict.
      - :class:`EncodingRepairReport` — summary of a whole-PDF sweep.
      - :func:`repair_encoding_on_pdf` — sweep every page-level font dict in
        a :class:`pikepdf.Pdf`.

Phase 2 — **Simple-font replacement for veraPDF 7.21.4.1-1.**

    Chunk A (this file, below the Phase 1 block) adds the eligibility check
    that Chunks B/C will consume:

      - :class:`SimpleFontEligibility`  (in :mod:`.models`)
      - :class:`MultiSimpleFontEligibility`  (in :mod:`.models`)
      - :func:`check_simple_font_eligibility`
      - :func:`check_simple_multifont_eligibility`

    The actual ``SimpleFontReplacer`` (subset + re-embed + emplace) lands in a
    separate chunk and is not part of this module's public surface yet.

Scope note
----------

Only fonts reachable from ``pdf.pages[i].Resources.Font`` are examined by
:func:`repair_encoding_on_pdf` and
:func:`check_simple_multifont_eligibility`.  Fonts living in Form XObjects,
annotation appearance streams, or ``AcroForm /DR`` are *not* visited.  That is
acceptable for the targeted ~300-700 documents in the REMEDY-73 simple-font
bucket; later chunks can widen coverage if needed.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field
from pathlib import Path

import pikepdf
from fontTools import agl
from pikepdf import Name

from project_remedy.faithful_rebuild.models import (
    MultiSimpleFontEligibility,
    SimpleFontEligibility,
)
from project_remedy.faithful_rebuild.pua_handler import should_skip_font_for_pua


# --- Dataclasses -----------------------------------------------------------


@dataclass(frozen=True)
class EncodingRepairEligibility:
    """Whether a font dict is eligible for Phase 1 encoding repair."""

    eligible: bool
    reason: str


@dataclass
class EncodingRepairReport:
    """Summary of a whole-PDF encoding-repair sweep."""

    fonts_examined: int = 0
    fonts_repaired: int = 0
    fonts_tounicode_synthesized: int = 0
    per_font: list[dict] = field(default_factory=list)


# --- Helpers ---------------------------------------------------------------


# PDF spec font-descriptor Flags bits (1-indexed in the spec; 0-indexed here
# as Python shifts).  Bit 3 (value 4) = Symbolic, bit 6 (value 32) = Nonsymbolic.
_FLAG_SYMBOLIC = 1 << 2  # == 4


def _get_subtype(font_dict: pikepdf.Object) -> Name | None:
    """Return the font's ``/Subtype`` as a pikepdf.Name, or None if missing."""
    try:
        value = font_dict.get("/Subtype")
    except AttributeError:
        return None
    return value if isinstance(value, Name) else None


def _get_flags(font_dict: pikepdf.Object) -> int:
    """Return the integer ``/FontDescriptor /Flags`` value, or 0 if missing."""
    try:
        descriptor = font_dict.get("/FontDescriptor")
    except AttributeError:
        return 0
    if descriptor is None:
        return 0
    try:
        raw_flags = descriptor.get("/Flags", 0)
    except AttributeError:
        return 0
    try:
        return int(raw_flags)
    except (TypeError, ValueError):
        return 0


def _has_encoding_key(font_dict: pikepdf.Object) -> bool:
    """Return True if the font dict has an ``/Encoding`` entry."""
    try:
        return font_dict.get("/Encoding") is not None
    except AttributeError:
        return False


# --- Public API: eligibility ----------------------------------------------


def check_encoding_repair_eligibility(
    font_dict: pikepdf.Object,
) -> EncodingRepairEligibility:
    """Decide whether this font dict qualifies for 7.21.6-3 encoding repair.

    Eligible iff all of:

    1. ``/Subtype`` is ``/TrueType``.
    2. ``/FontDescriptor /Flags`` has bit 3 (value 4, Symbolic) set.
    3. An ``/Encoding`` key is present (there is something to remove).

    The returned :class:`EncodingRepairEligibility` always carries a
    human-readable ``reason`` explaining the decision.
    """

    subtype = _get_subtype(font_dict)
    if subtype != Name("/TrueType"):
        subtype_repr = str(subtype) if subtype is not None else "<missing>"
        return EncodingRepairEligibility(
            eligible=False,
            reason=f"Subtype is {subtype_repr}, not /TrueType",
        )

    flags = _get_flags(font_dict)
    if not (flags & _FLAG_SYMBOLIC):
        return EncodingRepairEligibility(
            eligible=False,
            reason=(
                f"FontDescriptor /Flags={flags} does not have Symbolic bit set "
                "(bit 3, value 4)"
            ),
        )

    if not _has_encoding_key(font_dict):
        return EncodingRepairEligibility(
            eligible=False,
            reason="No /Encoding key present; nothing to repair",
        )

    return EncodingRepairEligibility(
        eligible=True,
        reason="Symbolic TrueType font has /Encoding; removing it satisfies 7.21.6-3",
    )


# --- Public API: per-font repair ------------------------------------------


def _preserve_tounicode_before_encoding_strip(
    font_dict: pikepdf.Object,
    pdf: pikepdf.Pdf,
) -> bool:
    """Best-effort synth of ``/ToUnicode`` while ``/Encoding`` is still present.

    A symbolic TrueType font may be relying on its ``/Encoding /Differences``
    glyph names (e.g. ``/parenlefttp``) as the PDF/UA "ToUnicode alternative"
    path allowed by ISO 14289-1 7.21.7.  Once Phase 1 deletes ``/Encoding``
    to satisfy 7.21.6-3, veraPDF can no longer resolve those glyphs and flags
    7.21.7-1.  This helper pre-emptively materializes a ``/ToUnicode`` CMap
    from the existing encoding so the mapping survives ``/Encoding`` removal.

    Returns ``True`` iff a new ``/ToUnicode`` stream was attached.  Returns
    ``False`` if the font already had ``/ToUnicode``, the synthesizer could
    not produce a mapping, or synthesis raised (all failures are swallowed —
    this is a best-effort step that must not break Phase 1).
    """

    # Don't clobber an existing ToUnicode — the document author's mapping
    # (or a prior-stage repair) is authoritative.
    try:
        if font_dict.get("/ToUnicode") is not None:
            return False
    except AttributeError:
        return False

    # Lazy import to avoid a hard pdf_fixer dependency at module import time
    # and to keep the fix scoped to Phase 1's public surface.
    try:
        from fontTools import agl as _agl
        from project_remedy.pdf_fixer import (
            _synth_simple_font_tounicode as _synth,
        )
    except Exception:  # pragma: no cover - defensive import guard
        return False

    def _agl_to_unicode(name: str) -> str | None:
        try:
            value = _agl.toUnicode(name)
        except Exception:
            return None
        return value or None

    try:
        result = _synth(font_dict, pdf, _agl_to_unicode)
    except Exception:
        return False

    return bool(result)


def repair_encoding_on_font(
    font_dict: pikepdf.Object,
    *,
    pdf: pikepdf.Pdf | None = None,
) -> bool:
    """Remove ``/Encoding`` from a symbolic TrueType font dict if eligible.

    Returns ``True`` if the ``/Encoding`` key was removed, ``False`` if the
    font is not eligible (and therefore untouched).  The font dict is mutated
    in place when repair is performed.

    When ``pdf`` is provided and the font has no ``/ToUnicode``, this
    function will attempt to synthesize a ``/ToUnicode`` CMap *before*
    deleting ``/Encoding``.  This preserves the glyph-name-to-Unicode
    mapping path that veraPDF's 7.21.7-1 check otherwise loses when
    ``/Encoding /Differences`` is removed.  Synthesis is best-effort and
    does not affect the return value — repair always proceeds.

    Raises
    ------
    RuntimeError
        If the font is eligible per :func:`check_encoding_repair_eligibility`
        but the ``/Encoding`` key cannot be deleted — this would indicate an
        unexpected object shape (e.g. a read-only proxy) and should be
        investigated rather than silently ignored.
    """

    eligibility = check_encoding_repair_eligibility(font_dict)
    if not eligibility.eligible:
        return False

    # Best-effort ToUnicode synth while /Encoding is still present so the
    # Unicode mapping survives the strip below.  Return value is not load-
    # bearing on the repair itself — Phase 1 still runs if synth is skipped.
    if pdf is not None:
        _preserve_tounicode_before_encoding_strip(font_dict, pdf)

    try:
        del font_dict["/Encoding"]
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Failed to delete /Encoding from eligible font: {exc!r}"
        ) from exc

    # Sanity check: we just deleted it, so it must be gone.  If not, something
    # is deeply wrong with the object model.
    if _has_encoding_key(font_dict):
        raise RuntimeError(
            "repair_encoding_on_font deleted /Encoding but key is still present"
        )

    return True


# --- Public API: whole-PDF sweep ------------------------------------------


def _font_base_name(font_dict: pikepdf.Object) -> str:
    """Return a human-readable identifier for a font dict (/BaseFont or '?')."""
    try:
        base = font_dict.get("/BaseFont")
    except AttributeError:
        return "?"
    if base is None:
        return "?"
    return str(base)


def _font_objgen_key(font_obj: pikepdf.Object) -> tuple:
    """Return a dedup key for a font object.

    Indirect fonts use their ``objgen`` tuple, so the same indirect object
    referenced from multiple pages is only visited once.  Inline font dicts
    (no indirect id) fall back to :func:`id`, which is unique for the duration
    of the sweep.
    """
    try:
        if getattr(font_obj, "is_indirect", False):
            return font_obj.objgen
    except Exception:  # pragma: no cover - defensive
        pass
    return ("inline", id(font_obj))


def repair_encoding_on_pdf(pdf: pikepdf.Pdf) -> EncodingRepairReport:
    """Sweep every page-level font in ``pdf`` and repair 7.21.6-3 violations.

    For each unique font reachable via ``pdf.pages[i].Resources.Font``:

      - If eligible, remove its ``/Encoding`` entry and record the action.
      - Otherwise, record that the font was skipped and why.

    Returns an :class:`EncodingRepairReport` with examined / repaired counts
    and a ``per_font`` list of dicts containing::

        {"font": "<BaseFont>", "action": "repaired" | "skipped", "reason": "..."}

    The sweep does not visit Form XObjects, annotation appearances, or
    ``AcroForm /DR``; see the module docstring for scope notes.
    """

    report = EncodingRepairReport()
    seen: set[tuple] = set()

    for page in pdf.pages:
        try:
            resources = page.obj.get("/Resources")
        except AttributeError:
            continue
        if resources is None:
            continue
        try:
            fonts = resources.get("/Font")
        except AttributeError:
            continue
        if fonts is None:
            continue

        for _key, font_obj in fonts.items():
            if not isinstance(font_obj, pikepdf.Object):
                continue

            dedup_key = _font_objgen_key(font_obj)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            report.fonts_examined += 1
            base_name = _font_base_name(font_obj)

            eligibility = check_encoding_repair_eligibility(font_obj)
            if not eligibility.eligible:
                report.per_font.append({
                    "font": base_name,
                    "action": "skipped",
                    "reason": eligibility.reason,
                })
                continue

            # Snapshot ToUnicode presence before repair so we can tell if the
            # preservation step (inside repair_encoding_on_font) synthesized
            # a new one.
            try:
                had_tounicode = font_obj.get("/ToUnicode") is not None
            except AttributeError:
                had_tounicode = False

            repaired = repair_encoding_on_font(font_obj, pdf=pdf)
            if repaired:
                report.fonts_repaired += 1
                try:
                    now_has_tounicode = font_obj.get("/ToUnicode") is not None
                except AttributeError:
                    now_has_tounicode = False
                synthesized = (not had_tounicode) and now_has_tounicode
                if synthesized:
                    report.fonts_tounicode_synthesized += 1
                entry = {
                    "font": base_name,
                    "action": "repaired",
                    "reason": eligibility.reason,
                }
                if synthesized:
                    entry["tounicode_synthesized"] = True
                report.per_font.append(entry)
            else:  # pragma: no cover - check_ said eligible but repair returned False
                report.per_font.append({
                    "font": base_name,
                    "action": "skipped",
                    "reason": "eligibility check passed but repair returned False",
                })

    return report


# ---------------------------------------------------------------------------
# Phase 2 Chunk A — simple-font replacement eligibility
# ---------------------------------------------------------------------------
#
# The actual replacer (subset + re-embed + emplace) lands in a later chunk.
# This block only supplies the eligibility check Chunks B/C will consume.
#
# Disqualifying-reason taxonomy (stable strings for telemetry).  Every
# ``SimpleFontEligibility.disqualifying_reasons`` entry produced below begins
# with one of these prefixes, followed by ``": "`` and human-readable detail.
# Keep this list in sync with tests and downstream routing.
DISQUALIFYING_REASONS: frozenset[str] = frozenset({
    "wrong_subtype",          # not /Type1 or /TrueType
    "base14_font",            # BaseFont is one of the PDF 14 standard names
    "symbolic_truetype",      # 7.21.6-3 / Phase 1 repair path, not Phase 2
    "font_embedded",          # /FontFile or /FontFile2 already present
    "missing_font_descriptor", # no /FontDescriptor at all
    "encoding_unresolvable",  # could not derive code_to_glyph for used codes
    "pua_or_custom_glyphs",   # PUA-dominated or custom-glyph-set font
    "no_trigger_rules",       # font is healthy — no 7.21.4.1-1 to fix
})


# PDF 1.7 Annex D.1 Base14 font names.  We strip any 6-letter subset prefix
# (e.g. ``ABCDEF+Times-Roman``) before comparing.
_BASE14_NAMES: frozenset[str] = frozenset({
    "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
    "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
    "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique",
    "Symbol", "ZapfDingbats",
})


def _is_base14_font(base_font_name: str) -> bool:
    """True iff ``base_font_name`` is one of the 14 PDF standard fonts.

    The comparison is performed *after* stripping any leading ``/`` that
    pikepdf ``Name`` values carry and any 6-letter subset prefix
    (``ABCDEF+``).  Base14 fonts need no embedding — the reader already has
    them — so they are out of scope for the simple-font replacement track.
    """

    stripped = base_font_name.lstrip("/").split("+", 1)[-1]
    return stripped in _BASE14_NAMES


# Build WinAnsi / MacRoman / Standard / MacExpert base-encoding tables as
# ``{code: glyph_name}``.  These four are the only ``/BaseEncoding`` names the
# PDF 1.7 spec (§9.6.6.4) permits for simple fonts.
#
# WinAnsi and MacRoman are derived from Python's codec tables by round-tripping
# each byte through the codec to Unicode and then back to a glyph name via
# ``fontTools.agl``.  This mirrors what :mod:`pdf_fixer` already does for
# ``_WINANSI_MAP`` / ``_MACROMAN_MAP``, kept local so no import dependency is
# introduced on ``pdf_fixer``.  StandardEncoding and MacRomanEncoding glyph
# lists also ship with fontTools for the PS side; we use those directly for
# StandardEncoding to avoid the AGL round-trip losing un-assigned slots.
_WINANSI_CODE_TO_GLYPH: dict[int, str] = {}
_MACROMAN_CODE_TO_GLYPH: dict[int, str] = {}
_STANDARD_CODE_TO_GLYPH: dict[int, str] = {}
_MACEXPERT_CODE_TO_GLYPH: dict[int, str] = {}


def _ensure_encoding_maps() -> None:
    """Populate ``_WINANSI_CODE_TO_GLYPH`` & friends on first use.

    Idempotent; cheap after the first call.
    """

    if _WINANSI_CODE_TO_GLYPH:
        return

    # WinAnsi: cp1252 → Unicode → AGL glyph name.  Unassigned cp1252 slots
    # raise UnicodeDecodeError; leave those codes unmapped.
    for code in range(256):
        try:
            ch = codecs.decode(bytes([code]), "cp1252")
        except UnicodeDecodeError:
            continue
        if not ch:
            continue
        glyph = agl.UV2AGL.get(ord(ch))
        if glyph:
            _WINANSI_CODE_TO_GLYPH[code] = glyph

    # MacRoman: same pattern via Python's ``mac-roman`` codec.
    for code in range(256):
        try:
            ch = codecs.decode(bytes([code]), "mac-roman")
        except UnicodeDecodeError:
            continue
        if not ch:
            continue
        glyph = agl.UV2AGL.get(ord(ch))
        if glyph:
            _MACROMAN_CODE_TO_GLYPH[code] = glyph

    # Adobe Standard Encoding: fontTools ships the glyph-name table directly.
    try:
        from fontTools.encodings.StandardEncoding import StandardEncoding

        for code, name in enumerate(StandardEncoding):
            if name and name != ".notdef":
                _STANDARD_CODE_TO_GLYPH[code] = name
    except Exception:  # pragma: no cover - defensive
        pass

    # MacExpertEncoding is not shipped as a first-class fontTools table.  The
    # simple-font replacement track does not need to cover MacExpert in
    # practice (it is vanishingly rare in modern PDFs); we leave the map
    # empty and let ``_derive_code_to_glyph`` return ``None`` for any font
    # that actually requires it.


def _derive_code_to_glyph(
    font_dict: pikepdf.Dictionary,
    used_char_codes: frozenset[int],
) -> dict[int, str] | None:
    """Derive ``char_code → glyph_name`` for every code in ``used_char_codes``.

    The PDF 1.7 spec (§9.6.6) allows ``/Encoding`` to be either a ``Name``
    (e.g. ``/WinAnsiEncoding``) or a ``Dictionary`` with ``/BaseEncoding``
    plus an optional ``/Differences`` array that overrides individual codes.

    Strategy:

    1. Resolve the base encoding table:

       * ``Name`` value → the corresponding base table.
       * ``Dictionary`` value → look up ``/BaseEncoding`` (default
         WinAnsiEncoding per spec for Type1/TrueType).
       * Missing ``/Encoding`` key → WinAnsiEncoding (reader convention).

    2. Apply any ``/Differences`` entries on top of the base table.  Glyph
       names resolved through the Adobe Glyph List; ``.notdef`` entries
       overwrite/clear the base mapping.

    3. If every code in ``used_char_codes`` is covered, return the map.
       Otherwise return ``None`` — the font cannot be safely replaced
       because we do not know which glyph the content stream expects for
       those codes.

    If ``used_char_codes`` is empty, return an empty dict — callers that
    care should gate on emptiness before relying on this helper.
    """

    _ensure_encoding_maps()

    encoding = font_dict.get("/Encoding")

    code_to_glyph: dict[int, str] = {}
    if encoding is None:
        code_to_glyph = dict(_WINANSI_CODE_TO_GLYPH)
    elif isinstance(encoding, pikepdf.Name):
        enc_name = str(encoding)
        if enc_name == "/WinAnsiEncoding":
            code_to_glyph = dict(_WINANSI_CODE_TO_GLYPH)
        elif enc_name == "/MacRomanEncoding":
            code_to_glyph = dict(_MACROMAN_CODE_TO_GLYPH)
        elif enc_name == "/StandardEncoding":
            code_to_glyph = dict(_STANDARD_CODE_TO_GLYPH)
        elif enc_name == "/MacExpertEncoding":
            code_to_glyph = dict(_MACEXPERT_CODE_TO_GLYPH)
        else:
            # Unknown named encoding — cannot derive.
            return None
    elif isinstance(encoding, pikepdf.Dictionary):
        base_enc_raw = encoding.get("/BaseEncoding")
        base_enc = str(base_enc_raw) if base_enc_raw is not None else "/WinAnsiEncoding"
        if base_enc == "/WinAnsiEncoding":
            code_to_glyph = dict(_WINANSI_CODE_TO_GLYPH)
        elif base_enc == "/MacRomanEncoding":
            code_to_glyph = dict(_MACROMAN_CODE_TO_GLYPH)
        elif base_enc == "/StandardEncoding":
            code_to_glyph = dict(_STANDARD_CODE_TO_GLYPH)
        elif base_enc == "/MacExpertEncoding":
            code_to_glyph = dict(_MACEXPERT_CODE_TO_GLYPH)
        else:
            return None

        diffs = encoding.get("/Differences")
        if diffs is not None:
            try:
                items = list(diffs)
            except Exception:
                return None
            current_code: int | None = None
            for item in items:
                if isinstance(item, pikepdf.Name):
                    if current_code is None:
                        # Differences array started with a glyph name — invalid
                        # per spec; we cannot recover.
                        return None
                    glyph_name = str(item).lstrip("/")
                    if glyph_name == ".notdef":
                        code_to_glyph.pop(current_code, None)
                    else:
                        code_to_glyph[current_code] = glyph_name
                    current_code += 1
                else:
                    try:
                        current_code = int(item)
                    except (TypeError, ValueError):
                        return None
    else:
        return None

    # Validate coverage of every used code.
    missing = [c for c in used_char_codes if c not in code_to_glyph]
    if missing:
        return None

    # Narrow to only the used codes — callers that want the full table can
    # rebuild it.  Tight mapping keeps telemetry focused.
    return {code: code_to_glyph[code] for code in used_char_codes}


# PDF /FontDescriptor /Flags bit layout (1-indexed in the spec):
#   bit 3 (value 4)  — Symbolic
#   bit 6 (value 32) — Nonsymbolic
_FLAG_SYMBOLIC_BIT = 1 << 2  # 4


def _get_used_char_codes_from_widths(font_dict: pikepdf.Dictionary) -> frozenset[int]:
    """Best-effort list of "codes this font declares widths for".

    Phase 2 Chunk A operates on font dicts in isolation — the real content
    stream walk is a Chunk B/C concern.  For eligibility we still need to
    validate that ``code_to_glyph`` can be derived for *some* non-empty set
    of codes, otherwise the later stages have nothing to replace.

    We synthesise a "declared codes" set from ``/FirstChar`` / ``/LastChar``
    / ``/Widths`` (all three required for non-Base14 simple fonts per PDF
    1.7 §9.6.2.1).  If the trio is missing or malformed we return an empty
    frozenset; callers treat that as a disqualifying condition.
    """

    try:
        first_char = font_dict.get("/FirstChar")
        last_char = font_dict.get("/LastChar")
        widths = font_dict.get("/Widths")
    except Exception:
        return frozenset()

    if first_char is None or last_char is None or widths is None:
        return frozenset()
    try:
        first = int(first_char)
        last = int(last_char)
        width_list = list(widths)
    except Exception:
        return frozenset()

    if first < 0 or last < first or len(width_list) != (last - first + 1):
        return frozenset()

    codes: set[int] = set()
    for offset, width in enumerate(width_list):
        try:
            if int(width) > 0:
                codes.add(first + offset)
        except (TypeError, ValueError):
            continue
    return frozenset(codes)


def check_simple_font_eligibility(
    pdf: pikepdf.Pdf,
    font_obj: pikepdf.Object,
    font_key: str,
    page_index: int,
) -> SimpleFontEligibility:
    """Decide whether a single simple font can be replaced by SimpleFontReplacer.

    Eligibility rules (all must hold for ``qualifies=True``):

    1. ``/Subtype`` is ``/Type1`` or ``/TrueType`` (Type0/Type3/MMType1 etc.
       are out of scope — the Mode B canary handles Type0).
    2. ``/BaseFont`` is **not** one of the 14 PDF standard fonts (no
       embedding is required for those).
    3. For TrueType, the font is **not** symbolic with an ``/Encoding`` key
       — that combination is 7.21.6-3 / Phase 1 territory.
    4. The font currently violates 7.21.4.1-1: no ``/FontFile`` (Type1) or
       ``/FontFile2`` (TrueType) on its ``/FontDescriptor``.
    5. A complete ``code_to_glyph`` map can be derived for the font's
       declared code range (from ``/FirstChar`` / ``/LastChar`` / ``/Widths``).
    6. The font is not dominated by PUA / custom-glyph names (the
       :mod:`.pua_handler` check must not return ``skip``).

    The ``pdf`` handle is accepted for API symmetry with
    :func:`font_analysis.check_canary_eligibility`; Chunk A does not walk
    content streams, but Chunk B will and will want the handle.
    """

    del pdf  # Not used in Chunk A — kept for API symmetry with canary path.

    reasons: list[str] = []

    # 1. Subtype.
    subtype_obj = font_dict_get(font_obj, "/Subtype")
    subtype_str = str(subtype_obj) if subtype_obj is not None else ""
    if subtype_str not in ("/Type1", "/TrueType"):
        return SimpleFontEligibility(
            qualifies=False,
            font_object=font_obj if isinstance(font_obj, pikepdf.Object) else None,
            font_key=font_key,
            page_index=page_index,
            font_subtype=subtype_str,
            base_font=str(font_dict_get(font_obj, "/BaseFont") or ""),
            placements=[(page_index, font_key)],
            disqualifying_reasons=[
                f"wrong_subtype: {subtype_str or '<missing>'} is not /Type1 or /TrueType"
            ],
        )

    base_font = str(font_dict_get(font_obj, "/BaseFont") or "")

    # 2. Base14 (no embedding needed — out of scope by definition).
    if _is_base14_font(base_font):
        return SimpleFontEligibility(
            qualifies=False,
            font_object=font_obj if isinstance(font_obj, pikepdf.Object) else None,
            font_key=font_key,
            page_index=page_index,
            font_subtype=subtype_str,
            base_font=base_font,
            placements=[(page_index, font_key)],
            disqualifying_reasons=[
                f"base14_font: {base_font} is a PDF standard font (no embedding required)"
            ],
        )

    # 3. Symbolic TrueType with /Encoding → Phase 1 territory.
    descriptor = font_dict_get(font_obj, "/FontDescriptor")
    if subtype_str == "/TrueType":
        flags = 0
        if descriptor is not None:
            try:
                flags = int(descriptor.get("/Flags", 0) or 0)
            except (TypeError, ValueError):
                flags = 0
        if (flags & _FLAG_SYMBOLIC_BIT) and font_dict_get(font_obj, "/Encoding") is not None:
            return SimpleFontEligibility(
                qualifies=False,
                font_object=font_obj if isinstance(font_obj, pikepdf.Object) else None,
                font_key=font_key,
                page_index=page_index,
                font_subtype=subtype_str,
                base_font=base_font,
                placements=[(page_index, font_key)],
                disqualifying_reasons=[
                    "symbolic_truetype: symbolic TrueType with /Encoding — "
                    "use Phase 1 encoding repair (7.21.6-3) instead"
                ],
            )

    # 4. 7.21.4.1-1 — missing embedded font program.
    trigger_rules: set[str] = set()
    if descriptor is None:
        reasons.append("missing_font_descriptor: /FontDescriptor is absent")
    else:
        if subtype_str == "/Type1":
            font_program = descriptor.get("/FontFile") or descriptor.get("/FontFile3")
        else:  # /TrueType
            font_program = descriptor.get("/FontFile2")
        if font_program is None:
            trigger_rules.add("7.21.4.1-1")
        else:
            reasons.append(
                "font_embedded: embedded font program is present; "
                "7.21.4.1-1 does not apply"
            )

    if reasons:
        return SimpleFontEligibility(
            qualifies=False,
            font_object=font_obj if isinstance(font_obj, pikepdf.Object) else None,
            font_key=font_key,
            page_index=page_index,
            font_subtype=subtype_str,
            base_font=base_font,
            placements=[(page_index, font_key)],
            disqualifying_reasons=reasons,
        )

    # 5. Derive code_to_glyph for declared codes.
    used_codes = _get_used_char_codes_from_widths(font_obj)
    code_to_glyph = _derive_code_to_glyph(font_obj, used_codes) if used_codes else None
    if not used_codes or code_to_glyph is None:
        return SimpleFontEligibility(
            qualifies=False,
            font_object=font_obj if isinstance(font_obj, pikepdf.Object) else None,
            font_key=font_key,
            page_index=page_index,
            font_subtype=subtype_str,
            base_font=base_font,
            used_char_codes=used_codes,
            code_to_glyph=code_to_glyph,
            trigger_rules=frozenset(trigger_rules),
            placements=[(page_index, font_key)],
            disqualifying_reasons=[
                "encoding_unresolvable: could not derive a complete "
                "code→glyph map for the font's declared codes"
            ],
        )

    # 6. PUA / custom-glyph gate.  The PUA helper wants a code→unicode map;
    # synthesise one from the glyph names via the Adobe Glyph List.  Any
    # glyph name that does not resolve is simply omitted — the helper
    # already copes with small maps (it inspects the font name too).
    code_unicode_map: dict[int, int] = {}
    for code, glyph_name in code_to_glyph.items():
        unicode_str = agl.toUnicode(glyph_name)
        if unicode_str and len(unicode_str) == 1:
            code_unicode_map[code] = ord(unicode_str)
    # Cast font_obj to the Dictionary-shaped object the helper expects.
    font_dict_for_pua = font_obj if isinstance(font_obj, pikepdf.Object) else None
    if font_dict_for_pua is not None:
        skip, pua_reason = should_skip_font_for_pua(code_unicode_map, font_dict_for_pua)
        if skip:
            return SimpleFontEligibility(
                qualifies=False,
                font_object=font_obj,
                font_key=font_key,
                page_index=page_index,
                font_subtype=subtype_str,
                base_font=base_font,
                used_char_codes=used_codes,
                code_to_glyph=code_to_glyph,
                trigger_rules=frozenset(trigger_rules),
                placements=[(page_index, font_key)],
                disqualifying_reasons=[f"pua_or_custom_glyphs: {pua_reason}"],
            )

    # No trigger rules → font does not need replacement.
    if not trigger_rules:
        return SimpleFontEligibility(
            qualifies=False,
            font_object=font_obj if isinstance(font_obj, pikepdf.Object) else None,
            font_key=font_key,
            page_index=page_index,
            font_subtype=subtype_str,
            base_font=base_font,
            used_char_codes=used_codes,
            code_to_glyph=code_to_glyph,
            trigger_rules=frozenset(trigger_rules),
            placements=[(page_index, font_key)],
            disqualifying_reasons=[
                "no_trigger_rules: font has an embedded program; no 7.21.4.1-1 to fix"
            ],
        )

    return SimpleFontEligibility(
        qualifies=True,
        font_object=font_obj,
        font_key=font_key,
        page_index=page_index,
        font_subtype=subtype_str,
        base_font=base_font,
        used_char_codes=used_codes,
        code_to_glyph=code_to_glyph,
        trigger_rules=frozenset(trigger_rules),
        placements=[(page_index, font_key)],
        disqualifying_reasons=[],
    )


def font_dict_get(font_obj: pikepdf.Object, key: str) -> pikepdf.Object | None:
    """Tolerant ``.get`` wrapper that returns ``None`` on non-Dictionary inputs.

    ``pikepdf.Object`` exposes a dict-style ``.get`` only when the underlying
    object is a Dictionary.  Inline/array/malformed values show up as bare
    objects whose ``.get`` call raises ``AttributeError``.  We treat any such
    failure as "key absent" so eligibility callers can continue to the
    ``wrong_subtype`` rejection path without crashing.
    """

    try:
        return font_obj.get(key)
    except AttributeError:
        return None


def check_simple_multifont_eligibility(
    pdf: pikepdf.Pdf,
) -> MultiSimpleFontEligibility:
    """Scan page-level ``/Font`` resources and aggregate per-font eligibility.

    For every unique font indirect-object reachable from
    ``pdf.pages[i].Resources.Font``, run :func:`check_simple_font_eligibility`
    once and append the result.  Fonts referenced from multiple pages are
    visited once (same dedup-by-objgen pattern as
    :func:`repair_encoding_on_pdf`).

    Scope note — **Form XObjects and ``AcroForm /DR`` are out of scope for
    Chunk A.**  The later replacer chunk may need to widen the walk if a
    font is referenced from an appearance stream as well as a page; tests
    in the current chunk pin the narrow scope.
    """

    aggregate = MultiSimpleFontEligibility()
    seen: set[tuple] = set()

    for page_index, page in enumerate(pdf.pages):
        try:
            resources = page.obj.get("/Resources")
        except AttributeError:
            continue
        if resources is None:
            continue
        try:
            fonts = resources.get("/Font")
        except AttributeError:
            continue
        if fonts is None:
            continue

        for key, font_obj in fonts.items():
            if not isinstance(font_obj, pikepdf.Object):
                continue
            dedup_key = _font_objgen_key(font_obj)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            eligibility = check_simple_font_eligibility(
                pdf,
                font_obj,
                font_key=str(key),
                page_index=page_index,
            )
            aggregate.font_eligibilities.append(eligibility)

    return aggregate


# ---------------------------------------------------------------------------
# Phase 2 Chunk C — SimpleFontReplacer (7-step pipeline)
# ---------------------------------------------------------------------------
#
# Mirrors the shape of ``CanaryReplacer`` (Mode B) but targets simple-font
# slots (``/Type1`` / ``/TrueType`` non-CID) whose ``/FontDescriptor`` is
# missing its font program.  The 7 steps are:
#
#   1. Fingerprint source font
#   2. Match candidate (CFF first for Type1, then TrueType-simple fallback)
#   3. Subset + embed via prepare_type1_font / prepare_truetype_simple_font
#   4. Build replacement /Encoding entry covering used char codes
#   5. Build replacement /ToUnicode CMap (byte_width=1)
#   6. Build replacement /Widths array [FirstChar..LastChar]
#   7. Assemble new font dict and emplace() onto source object
#
# Content stream char codes are preserved — no content-stream rewriting.

from project_remedy.faithful_rebuild import font_matcher as _font_matcher
from project_remedy.faithful_rebuild.font_analysis import extract_used_char_codes
from project_remedy.faithful_rebuild.canary_replacer import ReplacementReport
from project_remedy.faithful_rebuild.simple_font_embedder import (
    PreparedSimpleFont,
    prepare_truetype_simple_font,
    prepare_type1_font,
)
from project_remedy.pdf_fixer import build_bfchar_cmap as _build_bfchar_cmap


def _tighten_used_char_codes(
    pdf: pikepdf.Pdf,
    eligibility: SimpleFontEligibility,
) -> frozenset[int]:
    """Walk every page in *pdf* that places this font and return the union of
    content-stream 1-byte codes actually used with it.

    Falls back to ``eligibility.used_char_codes`` (upper bound from Widths) if
    no placements are listed or the walker raises — the eligibility value is
    always a superset of the real usage.
    """

    placements = eligibility.placements or [
        (eligibility.page_index, eligibility.font_key)
    ]
    actual: set[int] = set()
    for page_idx, font_key in placements:
        if page_idx < 0 or page_idx >= len(pdf.pages):
            continue
        try:
            codes = extract_used_char_codes(pdf.pages[page_idx], font_key)
        except Exception:
            # Unparseable content stream — fall back silently; the Widths-based
            # upper bound is still safe to subset against.
            continue
        actual.update(codes)
    if not actual:
        return eligibility.used_char_codes
    # Intersect with the declared set so we never subset a code whose glyph
    # name we don't know.  The eligibility map covers exactly the declared
    # Widths range.
    declared = eligibility.used_char_codes
    if declared:
        actual &= set(declared)
        if not actual:
            return declared
    return frozenset(actual)


def _build_simple_encoding(
    char_codes: frozenset[int],
    code_to_glyph: dict[int, str],
    *,
    base_encoding_name: str = "/WinAnsiEncoding",
) -> pikepdf.Name | pikepdf.Dictionary:
    """Return a value suitable for a simple font's ``/Encoding`` entry.

    If every used char code resolves to the same glyph name that
    ``base_encoding_name`` would produce, return the bare ``Name`` — no
    ``/Differences`` array is needed.  Otherwise return a Dictionary with
    ``/BaseEncoding`` plus a minimal ``/Differences`` array covering only
    the divergent codes.
    """

    _ensure_encoding_maps()
    base_table: dict[int, str]
    if base_encoding_name == "/WinAnsiEncoding":
        base_table = _WINANSI_CODE_TO_GLYPH
    elif base_encoding_name == "/MacRomanEncoding":
        base_table = _MACROMAN_CODE_TO_GLYPH
    elif base_encoding_name == "/StandardEncoding":
        base_table = _STANDARD_CODE_TO_GLYPH
    elif base_encoding_name == "/MacExpertEncoding":
        base_table = _MACEXPERT_CODE_TO_GLYPH
    else:
        # Unknown base — fall back to an explicit Differences covering every code.
        base_table = {}

    divergent: list[tuple[int, str]] = []
    for code in sorted(char_codes):
        expected = base_table.get(code)
        actual = code_to_glyph.get(code)
        if actual is None:
            # Shouldn't happen — the replacer filters code_to_glyph to used
            # codes before calling us — but guard anyway.
            continue
        if expected != actual:
            divergent.append((code, actual))

    if not divergent:
        return Name(base_encoding_name)

    # Build a minimal /Differences array.  Spec §9.6.6.1: the array is
    # sequences of a starting code number followed by one or more glyph names.
    # We emit one ``code [name, name, ...]`` run per contiguous range.
    differences: list = []
    prev_code: int | None = None
    for code, glyph in divergent:
        if prev_code is None or code != prev_code + 1:
            differences.append(code)
        differences.append(Name(f"/{glyph}"))
        prev_code = code

    return pikepdf.Dictionary(
        Type=Name("/Encoding"),
        BaseEncoding=Name(base_encoding_name),
        Differences=pikepdf.Array(differences),
    )


def _build_widths_array(
    first_char: int,
    last_char: int,
    width_for_char_code: dict[int, int],
    *,
    default_width: int = 0,
) -> pikepdf.Array:
    """Return a ``/Widths`` array spanning [first_char, last_char] inclusive.

    Per PDF 1.7 §9.2.4 / §9.6.2.1, codes the font does not cover use the
    descriptor's ``/MissingWidth`` (default 0).  Emitting 0 for unused codes
    is correct and keeps the array a simple one-entry-per-code list.
    """

    widths = pikepdf.Array()
    for code in range(first_char, last_char + 1):
        widths.append(int(width_for_char_code.get(code, default_width)))
    return widths


def _code_to_unicode_from_glyph(
    code_to_glyph: dict[int, str],
) -> dict[int, int | str]:
    """Return ``{code: int_codepoint_or_str}`` via the Adobe Glyph List.

    ``fontTools.agl.toUnicode`` returns a possibly empty / possibly
    multi-char string.  ``build_bfchar_cmap`` accepts both ``int`` (single
    codepoint) and ``str`` (multi-char, e.g. ligatures).  Single-char
    strings are unpacked to ``ord``.
    Codes whose glyph name has no AGL mapping are omitted entirely — the
    resulting bfchar will simply not name them, and Acrobat renders those
    codes as "no unicode" glyphs (acceptable for exotic symbols).
    """

    out: dict[int, int | str] = {}
    for code, glyph in code_to_glyph.items():
        unicode_str = agl.toUnicode(glyph)
        if not unicode_str:
            continue
        if len(unicode_str) == 1:
            out[code] = ord(unicode_str)
        else:
            out[code] = unicode_str
    return out


def _try_prepare_font(
    match: "_font_matcher.FontMatch",
    source_subtype: str,
    char_codes: frozenset[int],
    encoding_map: dict[int, str],
    source_fp: "_font_matcher.FontFingerprint | None" = None,
) -> tuple[PreparedSimpleFont, str] | None:
    """Run either :func:`prepare_type1_font` or
    :func:`prepare_truetype_simple_font` against *match*.

    Returns ``(prepared, kind)`` where ``kind`` is ``"type1c"`` or
    ``"truetype"``, or ``None`` if both preparers reject the candidate.  The
    caller converts a None result into a ReplacementReport failure.
    """

    # Resolve font bytes/path for the preparer.
    font_source: Path | bytes | None
    if match.use_embedded and source_fp is not None and source_fp.embedded_program is not None:
        font_source = source_fp.embedded_program
    elif match.resolved_path is not None:
        font_source = match.resolved_path
    else:
        return None

    # Choose preparer.  Type1 prefers CFF; TrueType prefers glyf.
    if source_subtype == "/Type1":
        # Try CFF first, then fall back to TrueType-simple.
        try:
            prepared = prepare_type1_font(
                font_source,
                char_codes=char_codes,
                encoding_map=encoding_map,
            )
            return prepared, "type1c"
        except ValueError:
            pass
        try:
            prepared = prepare_truetype_simple_font(
                font_source,
                char_codes=char_codes,
                encoding_map=encoding_map,
            )
            return prepared, "truetype"
        except ValueError:
            return None

    # /TrueType source — glyf path first; accept CFF if the candidate happens
    # to be an OTF (rare but possible when matcher returned an OTF for a
    # TrueType slot).  Emitting a Type1C program into a /TrueType slot is
    # invalid, so we only fall back to Type1 when the slot itself is Type1 —
    # therefore this branch does not attempt it.
    try:
        prepared = prepare_truetype_simple_font(
            font_source,
            char_codes=char_codes,
            encoding_map=encoding_map,
        )
        return prepared, "truetype"
    except ValueError:
        return None


class SimpleFontReplacer:
    """7-step simple-font replacement pipeline.

    Mirrors :class:`~project_remedy.faithful_rebuild.canary_replacer.CanaryReplacer`
    and :class:`~project_remedy.faithful_rebuild.multifont_replacer.MultiFontReplacer`
    in shape but targets ``/Type1`` and ``/TrueType`` (non-CID) fonts whose
    ``/FontDescriptor`` is missing ``/FontFile`` / ``/FontFile2`` (veraPDF
    7.21.4.1-1).

    Steps:

      1. Fingerprint source font (PostScript name, serif/mono via
         FontDescriptor /Flags).
      2. Match candidate.  Type1 sources try CFF candidates first, fall back
         to TrueType-simple.  TrueType sources go straight to TrueType-simple.
      3. Subset + embed via :func:`prepare_type1_font` or
         :func:`prepare_truetype_simple_font`.
      4. Build replacement ``/Encoding`` entry covering used char codes.
      5. Build replacement ``/ToUnicode`` CMap (byte_width=1).
      6. Build replacement ``/Widths`` array [FirstChar..LastChar].
      7. Assemble new font dict and :meth:`pikepdf.Object.emplace` it onto
         the source object.

    The page content stream is not touched — char codes stay the same, only
    the font resource's backing dict is swapped.

    The replacer never raises; caught failures return
    ``ReplacementReport(status="failed", reason=...)``.
    """

    #: Confidence floor — intentionally lower than :class:`CanaryReplacer`'s
    #: 0.60 because simple-font serif/mono classification is less precise
    #: than Mode B's glyph-coverage floor.  See design spec §Phase 2.
    MIN_CONFIDENCE: float = 0.55

    def __init__(self) -> None:
        # State is per-call; no configuration needed.
        pass

    def replace(
        self,
        pdf: pikepdf.Pdf,
        eligibility: SimpleFontEligibility,
    ) -> ReplacementReport:
        """Replace a single qualifying simple font.

        Returns a :class:`ReplacementReport`.  Never raises — any caught
        exception surfaces as ``status="failed"`` with the exception kind
        embedded in the reason string.
        """

        # Fast gate — eligibility short-circuit mirrors CanaryReplacer.
        if not eligibility.qualifies:
            return ReplacementReport(
                status="skipped",
                reason=(
                    "eligibility did not qualify: "
                    + "; ".join(eligibility.disqualifying_reasons)
                ),
            )

        font_obj = eligibility.font_object
        font_key = eligibility.font_key
        code_to_glyph = eligibility.code_to_glyph
        if (
            font_obj is None
            or not font_key
            or code_to_glyph is None
            or not eligibility.used_char_codes
        ):
            return ReplacementReport(
                status="failed",
                reason=(
                    "eligibility qualifies=True but required fields missing: "
                    f"font_object={font_obj is not None}, "
                    f"font_key={font_key!r}, "
                    f"code_to_glyph={code_to_glyph is not None}, "
                    f"used_char_codes={bool(eligibility.used_char_codes)}"
                ),
            )

        source_subtype = eligibility.font_subtype

        # --- Step 1: Fingerprint source font ------------------------------
        try:
            fp = _font_matcher.fingerprint_pdf_font(font_key, font_obj)
        except Exception as exc:
            return ReplacementReport(
                status="failed",
                reason=f"fingerprint_pdf_font raised {type(exc).__name__}: {exc}",
            )

        # --- Tighten used codes via content-stream walk -------------------
        # Chunk A eligibility uses /FirstChar../LastChar../Widths as an upper
        # bound; the walker gives the actual byte codes the content stream
        # shows.  Intersect to the set we can safely subset (glyph names
        # only exist for the declared codes).
        tight_codes = _tighten_used_char_codes(pdf, eligibility)
        if not tight_codes:
            return ReplacementReport(
                status="failed",
                reason="no used char codes after content-stream walk",
            )
        # Narrow code_to_glyph to the tight set — the preparer subsets by
        # glyph name, so we want only the glyphs actually shown.
        tight_code_to_glyph = {
            code: glyph
            for code, glyph in code_to_glyph.items()
            if code in tight_codes
        }
        if not tight_code_to_glyph:
            return ReplacementReport(
                status="failed",
                reason="no code_to_glyph coverage after tightening to used codes",
            )

        # --- Step 2: Match candidate --------------------------------------
        # Type1 sources want CFF first, so scan the CFF index first.
        match: "_font_matcher.FontMatch | None" = None
        if source_subtype == "/Type1":
            try:
                idx_cff = _font_matcher.scan_system_fonts(font_class="type1_cff")
            except Exception as exc:
                return ReplacementReport(
                    status="failed",
                    reason=f"scan_system_fonts(type1_cff) raised: {exc}",
                )
            cff_match = _font_matcher.match_font(
                fp, idx_cff, min_confidence=self.MIN_CONFIDENCE
            )
            if cff_match.confidence >= self.MIN_CONFIDENCE:
                match = cff_match
        if match is None:
            try:
                idx_tt = _font_matcher.scan_system_fonts(font_class="truetype_any")
            except Exception as exc:
                return ReplacementReport(
                    status="failed",
                    reason=f"scan_system_fonts(truetype_any) raised: {exc}",
                )
            tt_match = _font_matcher.match_font(
                fp, idx_tt, min_confidence=self.MIN_CONFIDENCE
            )
            if tt_match.confidence >= self.MIN_CONFIDENCE:
                match = tt_match

        if match is None or match.confidence < self.MIN_CONFIDENCE:
            reason = (
                (match.fallback_reason if match else None)
                or f"no match above min_confidence={self.MIN_CONFIDENCE:.2f}"
            )
            return ReplacementReport(
                status="failed",
                reason=f"no match: {reason}",
            )

        # --- Step 3: Subset + embed ---------------------------------------
        try:
            prepare_result = _try_prepare_font(
                match,
                source_subtype,
                frozenset(tight_code_to_glyph.keys()),
                tight_code_to_glyph,
                source_fp=fp,
            )
        except Exception as exc:
            return ReplacementReport(
                status="failed",
                reason=f"prepare_font raised {type(exc).__name__}: {exc}",
                matched_ps_name=None,
            )
        if prepare_result is None:
            return ReplacementReport(
                status="failed",
                reason="prepare_font could not subset candidate (glyph mismatch)",
                matched_ps_name=None,
            )
        prepared, prepared_kind = prepare_result

        # The slot dictates which FontFile slot / subtype we must use.  A
        # /Type1 slot accepts both CFF (/FontFile3 /Type1C) and — per PDF 1.7
        # — a rebuilt Type1 program, but we never produce plain Type1; a
        # /TrueType slot only accepts /FontFile2.  If the candidate can't
        # produce the right program kind, fail gracefully.
        if source_subtype == "/TrueType" and prepared_kind != "truetype":
            return ReplacementReport(
                status="failed",
                reason=(
                    f"candidate prepared as {prepared_kind!r} but source slot is "
                    "/TrueType — only /FontFile2 is valid here"
                ),
                matched_ps_name=prepared.postscript_name,
            )

        # --- Step 4: Build replacement /Encoding --------------------------
        new_encoding = _build_simple_encoding(
            frozenset(tight_code_to_glyph.keys()),
            tight_code_to_glyph,
            base_encoding_name="/WinAnsiEncoding",
        )

        # --- Step 5: Build replacement /ToUnicode CMap (byte_width=1) ----
        code_to_unicode = _code_to_unicode_from_glyph(tight_code_to_glyph)
        try:
            tounicode_bytes = _build_bfchar_cmap(code_to_unicode, byte_width=1)
        except Exception as exc:
            return ReplacementReport(
                status="failed",
                reason=f"build_bfchar_cmap raised {type(exc).__name__}: {exc}",
                matched_ps_name=prepared.postscript_name,
            )
        tounicode_stream = pdf.make_stream(tounicode_bytes)

        # --- Step 6: Build replacement /Widths ---------------------------
        widths_array = _build_widths_array(
            prepared.first_char,
            prepared.last_char,
            prepared.width_for_char_code,
            default_width=0,
        )

        # --- Step 7: Assemble new font dict and emplace -------------------
        # Re-use as much of the old /FontDescriptor as possible so metric
        # fields (Ascent, Descent, ItalicAngle, ...) stay believable.  The
        # preparer's subset prefix drives the new /FontName.
        old_descriptor = font_obj.get("/FontDescriptor") or pikepdf.Dictionary()

        # Emit the font program stream with the right key/subtype.
        if prepared_kind == "type1c":
            fontfile3_stream = pdf.make_stream(prepared.font_bytes)
            fontfile3_stream["/Subtype"] = Name("/Type1C")
            fontfile3_stream["/Length1"] = len(prepared.font_bytes)
            font_program_kwargs = {"FontFile3": fontfile3_stream}
        else:  # truetype
            fontfile2_stream = pdf.make_stream(prepared.font_bytes)
            fontfile2_stream["/Length1"] = len(prepared.font_bytes)
            font_program_kwargs = {"FontFile2": fontfile2_stream}

        new_descriptor = pikepdf.Dictionary(
            Type=Name("/FontDescriptor"),
            FontName=Name(f"/{prepared.postscript_name}"),
            Flags=int(old_descriptor.get("/Flags", 32) or 32),
            FontBBox=old_descriptor.get("/FontBBox", [-100, -200, 1100, 900]),
            ItalicAngle=int(old_descriptor.get("/ItalicAngle", 0) or 0),
            Ascent=int(old_descriptor.get("/Ascent", 900) or 900),
            Descent=int(old_descriptor.get("/Descent", -200) or -200),
            CapHeight=int(old_descriptor.get("/CapHeight", 700) or 700),
            StemV=int(old_descriptor.get("/StemV", 80) or 80),
            **font_program_kwargs,
        )

        new_font_dict = pikepdf.Dictionary(
            Type=Name("/Font"),
            Subtype=Name(source_subtype),
            BaseFont=Name(f"/{prepared.postscript_name}"),
            Encoding=new_encoding,
            FirstChar=int(prepared.first_char),
            LastChar=int(prepared.last_char),
            Widths=widths_array,
            ToUnicode=tounicode_stream,
            FontDescriptor=new_descriptor,
        )

        new_indirect = pdf.make_indirect(new_font_dict)
        try:
            font_obj.emplace(new_indirect)
        except Exception as exc:
            return ReplacementReport(
                status="failed",
                reason=f"emplace raised {type(exc).__name__}: {exc}",
                matched_ps_name=prepared.postscript_name,
            )

        return ReplacementReport(
            status="replaced",
            matched_ps_name=prepared.postscript_name,
            replaced_cids_count=len(tight_code_to_glyph),
        )


__all__ = [
    "EncodingRepairEligibility",
    "EncodingRepairReport",
    "DISQUALIFYING_REASONS",
    "SimpleFontReplacer",
    "check_encoding_repair_eligibility",
    "check_simple_font_eligibility",
    "check_simple_multifont_eligibility",
    "repair_encoding_on_font",
    "repair_encoding_on_pdf",
]
