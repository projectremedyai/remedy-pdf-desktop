"""Font audit detector registry (REMEDY-76).

Provides one pure detector per veraPDF font rule. Each detector is a pure
function over font dict + used CIDs + derived unicode map, returning
bool (violation present). The aggregator composes all detectors.

This module mirrors the existing audit_font_violations in font_analysis.py
but with a more extensible shape. Migration from the existing function is
deferred — this module is additive.

Each detector carries support metadata (routable_to_mode_b, etc.) so
downstream routing can use it directly.

IMPORTANT — 7.21.7-1 correctness note (REMEDY-76 sweep findings):
    The correctness sweep (docs/findings/remedy76-audit-correctness-sweep-20260416.md)
    found that the existing audit_font_violations logic for 7.21.7-1 produces 0/10
    agreement with veraPDF on sampled docs. The detector here is therefore implemented
    with veraPDF-aligned semantics: flag 7.21.7-1 ONLY when /ToUnicode is absent or
    its stream is empty. Multi-char mappings (ligatures), 4-byte bfrange bases (surrogate
    pairs), and array-form bfrange entries are all valid per ISO 14289-1 clause 7.21.7
    and must not trigger this rule.

    The existing audit_font_violations function in font_analysis.py uses
    cid_unicode_map is None as a proxy for 7.21.7-1, which conflates canary
    eligibility requirements (single-char-only mappings) with veraPDF compliance.
    That function is not changed here (out of scope for REMEDY-76); this registry
    provides the corrected detector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Any

import pikepdf


_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")

# Flags bit constants per ISO 32000-1 Table 123.
# /Flags is a non-negative integer on /FontDescriptor. Bit 1 is the
# least-significant bit in PDF-spec numbering, so bit N → value (1 << (N-1)).
_FLAG_SYMBOLIC = 1 << 2      # bit 3 (value 4)
_FLAG_NONSYMBOLIC = 1 << 5   # bit 6 (value 32)

# Names recognised as valid predefined TrueType encodings per
# ISO 14289-1 clause 7.21.6.
_TRUETYPE_VALID_ENCODING_NAMES = frozenset({"/WinAnsiEncoding", "/MacRomanEncoding"})


def _get_flags(font_dict: pikepdf.Dictionary, descendant: Any) -> int | None:
    """Return the /Flags integer from the font's /FontDescriptor.

    For simple fonts the descriptor is on font_dict itself; for Type0 fonts
    it is on the descendant. Returns None if the flags entry is not a readable
    integer.
    """
    descriptor = font_dict.get("/FontDescriptor")
    if descriptor is None and descendant is not None:
        descriptor = descendant.get("/FontDescriptor")
    if descriptor is None:
        return None
    flags = descriptor.get("/Flags")
    if flags is None:
        return None
    try:
        return int(flags)
    except (TypeError, ValueError):
        return None


def _is_truetype(font_dict: pikepdf.Dictionary) -> bool:
    """Return True if this is a simple /TrueType font (not /Type0)."""
    return str(font_dict.get("/Subtype", "")) == "/TrueType"


def _is_valid_encoding_dict(encoding: Any) -> bool:
    """A well-formed Encoding dictionary must be a Dictionary and must either
    have a valid /BaseEncoding name (or no BaseEncoding, defaulting to the
    font's built-in encoding) AND optionally a /Differences array.

    This is the permissive well-formedness check for 7.21.6-1; the stricter
    AGL-glyph-name check for 7.21.6-2 is separate.
    """
    if not isinstance(encoding, pikepdf.Dictionary):
        return False
    base = encoding.get("/BaseEncoding")
    if base is not None and str(base) not in _TRUETYPE_VALID_ENCODING_NAMES:
        return False
    differences = encoding.get("/Differences")
    if differences is not None and not isinstance(differences, pikepdf.Array):
        return False
    return True


def _differences_glyph_names(encoding: pikepdf.Dictionary) -> list[str]:
    """Return the list of glyph names in the /Differences array (skipping
    the integer code operands). Names are returned without the leading '/'.
    """
    differences = encoding.get("/Differences")
    if differences is None:
        return []
    names: list[str] = []
    for entry in differences:
        # Numeric entries are code indices; Name entries are glyph names.
        if isinstance(entry, pikepdf.Name):
            names.append(str(entry).lstrip("/"))
    return names


def _all_names_in_agl(names: list[str]) -> bool:
    """Return True if every glyph name is present in the Adobe Glyph List.

    Names with a ``uniXXXX`` or ``uXXXXXX`` shape are also accepted as
    well-formed (they encode a Unicode codepoint directly per the AGL spec).
    """
    try:
        from fontTools.agl import AGL2UV
    except Exception:
        # fontTools not importable: treat as "can't verify" → don't flag.
        return True
    uni_re = re.compile(r"^(?:uni[0-9A-F]{4}|u[0-9A-F]{4,6})$")
    for name in names:
        if name in AGL2UV:
            continue
        if uni_re.match(name):
            continue
        return False
    return True


@dataclass(frozen=True)
class RuleSupport:
    """Metadata about how a rule can be addressed."""

    rule_id: str
    description: str
    detected: bool               # do we implement detection?
    routable_to_mode_b: bool     # can Mode B replacement fix this?
    routable_to_simple_font: bool  # can a future simple-font fix this?
    manual_only: bool            # requires human intervention


# Detector signature: (font_dict, descendant, used_cids, cid_unicode_map) -> bool
DetectorFn = Callable[[pikepdf.Dictionary, Any, frozenset[int], "dict[int, int] | None"], bool]


@dataclass
class RegisteredDetector:
    rule_id: str
    support: RuleSupport
    detect: DetectorFn


# ---------------------------------------------------------------------------
# Pure detector functions
# ---------------------------------------------------------------------------

def _detect_7_21_4_1_1(
    font_dict: pikepdf.Dictionary,
    descendant: Any,
    used_cids: frozenset[int],
    cid_unicode_map: "dict[int, int] | None",
) -> bool:
    """FontFile2 missing on the descendant FontDescriptor.

    Correctness sweep (REMEDY-76): 8/8 samples agree with veraPDF.
    This detector is safe for routing decisions.

    Note: this matches the existing audit_font_violations logic in
    font_analysis.py — kept in sync.
    """
    if descendant is None:
        return True  # no descendant means no FontFile2
    descriptor = descendant.get("/FontDescriptor")
    if descriptor is None:
        return True
    return descriptor.get("/FontFile2") is None


def _detect_7_21_4_2_2(
    font_dict: pikepdf.Dictionary,
    descendant: Any,
    used_cids: frozenset[int],
    cid_unicode_map: "dict[int, int] | None",
) -> bool:
    """Per veraPDF: only applies to embedded subsetted CID fonts with a CIDSet.

    Matches the REMEDY-70 fix to audit_font_violations. Preconditions:
    - FontFile2 present (font is embedded)
    - CIDSet present on descriptor
    - BaseFont name has 6-uppercase-letter subset prefix (e.g. ABCDEF+FontName)
    - CIDSet bitmask does not cover all used_cids
    """
    if descendant is None:
        return False
    descriptor = descendant.get("/FontDescriptor")
    if descriptor is None:
        return False
    font_file2 = descriptor.get("/FontFile2")
    cidset = descriptor.get("/CIDSet")
    if font_file2 is None or cidset is None:
        return False
    basefont_name = str(descendant.get("/BaseFont", "")).lstrip("/")
    if not _SUBSET_PREFIX_RE.match(basefont_name):
        return False
    # Import _cidset_covers from font_analysis
    from project_remedy.faithful_rebuild.font_analysis import _cidset_covers
    return not _cidset_covers(cidset, used_cids)


def _detect_7_21_7_1(
    font_dict: pikepdf.Dictionary,
    descendant: Any,
    used_cids: frozenset[int],
    cid_unicode_map: "dict[int, int] | None",
) -> bool:
    """ToUnicode CMap absent or empty (veraPDF-aligned semantics).

    ISO 14289-1 clause 7.21.7: a font shall contain a /ToUnicode entry.
    veraPDF flags this rule when the /ToUnicode key is absent or the stream
    is empty. It does NOT flag it for multi-char mappings (ligatures),
    surrogate-pair bfrange entries, or array-form bfrange entries — all of
    which are valid per the PDF specification.

    IMPORTANT: Do NOT gate this on cid_unicode_map is None. That proxy
    conflates canary eligibility (which requires single-char-only mappings for
    font replacement) with veraPDF rule compliance. The correctness sweep
    (REMEDY-76) found 0/10 agreement using the cid_unicode_map proxy; this
    implementation achieves the expected behaviour.

    Correctness sweep (REMEDY-76): divergence analysis confirmed this
    veraPDF-aligned implementation is correct; the prior cid_unicode_map-based
    approach produced 100% false positives on sampled documents.
    """
    tounicode = font_dict.get("/ToUnicode")
    if tounicode is None:
        return True
    try:
        data = bytes(tounicode.read_bytes())
        return not data
    except Exception:
        return True


def _detect_7_21_6_1(
    font_dict: pikepdf.Dictionary,
    descendant: Any,
    used_cids: frozenset[int],
    cid_unicode_map: "dict[int, int] | None",
) -> bool:
    """TrueType Encoding entry (if present) must be well-formed.

    Per ISO 14289-1 clause 7.21.6 (ISO 32000-1 Table 114): a TrueType font's
    /Encoding entry, when present, must either be one of the named predefined
    encodings (/WinAnsiEncoding, /MacRomanEncoding) OR an Encoding dictionary
    whose /BaseEncoding (if given) is one of those names and whose
    /Differences is an Array of ``integer [name...]`` operand pairs.

    Scope:
      - Applies to /Subtype == /TrueType only. Type0 fonts are out of scope.
      - If /Encoding is absent, this rule does not trigger (other rules —
        7.21.6-2 for non-symbolic, 7.21.6-3 for symbolic — handle presence
        requirements).

    Best-effort: ambiguous cases (e.g., Encoding is an indirect reference to
    something we can't classify) are NOT flagged, to avoid false positives.
    """
    if not _is_truetype(font_dict):
        return False
    encoding = font_dict.get("/Encoding")
    if encoding is None:
        return False
    if isinstance(encoding, pikepdf.Name):
        return str(encoding) not in _TRUETYPE_VALID_ENCODING_NAMES
    if isinstance(encoding, pikepdf.Dictionary):
        return not _is_valid_encoding_dict(encoding)
    # Unknown shape — err on the side of not flagging.
    return False


def _detect_7_21_6_2(
    font_dict: pikepdf.Dictionary,
    descendant: Any,
    used_cids: frozenset[int],
    cid_unicode_map: "dict[int, int] | None",
) -> bool:
    """Non-symbolic TrueType font must use WinAnsi/MacRoman or an Encoding
    dictionary whose /Differences contains only Adobe Glyph List names.

    Per ISO 14289-1 clause 7.21.6 (ISO 32000-1 9.6.6.4): non-symbolic TrueType
    fonts must be mappable to Unicode via one of these means.

    Scope:
      - Applies to /Subtype == /TrueType only.
      - Applies only when /FontDescriptor /Flags has the Nonsymbolic bit
        (bit 6, value 32) set. If the Symbolic bit (bit 3, value 4) is set
        this rule does not apply; see 7.21.6-3 instead.
      - If flags cannot be read, err on the side of not flagging (ambiguous
        intent, better to avoid false positives).

    Violation patterns:
      - No /Encoding at all.
      - /Encoding is a name other than /WinAnsiEncoding or /MacRomanEncoding.
      - /Encoding is a dictionary whose /Differences contains any glyph name
        not in the Adobe Glyph List (and not a /uniXXXX or /uXXXXXX
        well-formed Unicode name).
    """
    if not _is_truetype(font_dict):
        return False
    flags = _get_flags(font_dict, descendant)
    if flags is None:
        return False
    if flags & _FLAG_SYMBOLIC:
        # Symbolic — out of scope for this rule.
        return False
    if not (flags & _FLAG_NONSYMBOLIC):
        # Neither symbolic nor nonsymbolic flag set — ambiguous, don't flag.
        return False
    encoding = font_dict.get("/Encoding")
    if encoding is None:
        return True
    if isinstance(encoding, pikepdf.Name):
        return str(encoding) not in _TRUETYPE_VALID_ENCODING_NAMES
    if isinstance(encoding, pikepdf.Dictionary):
        if not _is_valid_encoding_dict(encoding):
            return True
        names = _differences_glyph_names(encoding)
        if not names:
            # A valid dict with no /Differences defers to BaseEncoding, which
            # _is_valid_encoding_dict already validated as WinAnsi/MacRoman.
            return False
        return not _all_names_in_agl(names)
    return False


def _detect_7_21_6_3(
    font_dict: pikepdf.Dictionary,
    descendant: Any,
    used_cids: frozenset[int],
    cid_unicode_map: "dict[int, int] | None",
) -> bool:
    """Symbolic TrueType font shall NOT have an /Encoding entry.

    Per ISO 14289-1 clause 7.21.6 (ISO 32000-1 9.6.6.4): a symbolic TrueType
    font defines its own glyph-to-code mapping via its cmap table and must
    not declare a PDF-level Encoding. Presence of /Encoding on a symbolic
    TrueType font is a violation.

    Scope:
      - Applies to /Subtype == /TrueType only.
      - Applies only when /FontDescriptor /Flags has the Symbolic bit set
        (bit 3, value 4). If Nonsymbolic is set, see 7.21.6-2.
      - If flags cannot be read, err on the side of not flagging.
    """
    if not _is_truetype(font_dict):
        return False
    flags = _get_flags(font_dict, descendant)
    if flags is None:
        return False
    if not (flags & _FLAG_SYMBOLIC):
        return False
    return font_dict.get("/Encoding") is not None


def _detect_7_21_8_1(
    font_dict: pikepdf.Dictionary,
    descendant: Any,
    used_cids: frozenset[int],
    cid_unicode_map: "dict[int, int] | None",
) -> bool:
    """CIDSystemInfo of a CID font must match the associated CMap.

    Per ISO 14289-1 clause 7.21.8 / ISO 32000-1 9.7.4: when a Type0 font's
    /Encoding is /Identity-H or /Identity-V, the descendant CIDFont's
    /CIDSystemInfo dictionary must have Registry == "Adobe" and
    Ordering == "Identity" (the Identity encoding is only valid with the
    Adobe-Identity-0 ROS).

    Best-effort scope:
      - Only checks the /Identity-H and /Identity-V case. Named CMaps (e.g.,
        /UniJIS-UTF16-H) would require parsing the CMap's own CIDSystemInfo,
        which we do not attempt here — those cases are not flagged.
      - Missing descendant or missing /CIDSystemInfo are not flagged by this
        rule (those violate other structural rules).
    """
    encoding = font_dict.get("/Encoding")
    if not isinstance(encoding, pikepdf.Name):
        return False
    if str(encoding) not in ("/Identity-H", "/Identity-V"):
        return False
    if descendant is None:
        return False
    csi = descendant.get("/CIDSystemInfo")
    if csi is None:
        return False
    registry = csi.get("/Registry")
    ordering = csi.get("/Ordering")
    if registry is None or ordering is None:
        return False
    try:
        registry_str = str(registry)
        ordering_str = str(ordering)
    except Exception:
        return False
    return not (registry_str == "Adobe" and ordering_str == "Identity")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: list[RegisteredDetector] = [
    RegisteredDetector(
        rule_id="7.21.4.1-1",
        support=RuleSupport(
            rule_id="7.21.4.1-1",
            description="Font program must be embedded (FontFile2 on descendant FontDescriptor)",
            detected=True,
            routable_to_mode_b=True,
            routable_to_simple_font=True,
            manual_only=False,
        ),
        detect=_detect_7_21_4_1_1,
    ),
    RegisteredDetector(
        rule_id="7.21.4.2-2",
        support=RuleSupport(
            rule_id="7.21.4.2-2",
            description="CIDSet must identify all CIDs present in the embedded subset font program",
            detected=True,
            routable_to_mode_b=True,
            routable_to_simple_font=False,
            manual_only=False,
        ),
        detect=_detect_7_21_4_2_2,
    ),
    RegisteredDetector(
        rule_id="7.21.7-1",
        support=RuleSupport(
            rule_id="7.21.7-1",
            description="ToUnicode CMap entry must be present and non-empty",
            detected=True,
            routable_to_mode_b=True,
            routable_to_simple_font=True,
            manual_only=False,
        ),
        detect=_detect_7_21_7_1,
    ),
    RegisteredDetector(
        rule_id="7.21.6-1",
        support=RuleSupport(
            rule_id="7.21.6-1",
            description=(
                "TrueType Encoding entry (if present) must be a valid "
                "predefined name (WinAnsi/MacRoman) or an Encoding dictionary"
            ),
            detected=True,
            # Mode B is Type0-only (CIDFontType2); simple-font encoding
            # malformations are repaired by SimpleFontReplacer rewriting the
            # /Encoding key.
            routable_to_mode_b=False,
            routable_to_simple_font=True,
            manual_only=False,
        ),
        detect=_detect_7_21_6_1,
    ),
    RegisteredDetector(
        rule_id="7.21.6-2",
        support=RuleSupport(
            rule_id="7.21.6-2",
            description=(
                "Non-symbolic TrueType font must use WinAnsi/MacRoman or "
                "have an Encoding dictionary whose Differences contain only "
                "Adobe Glyph List names"
            ),
            detected=True,
            routable_to_mode_b=False,
            routable_to_simple_font=True,
            manual_only=False,
        ),
        detect=_detect_7_21_6_2,
    ),
    RegisteredDetector(
        rule_id="7.21.6-3",
        support=RuleSupport(
            rule_id="7.21.6-3",
            description=(
                "Symbolic TrueType font shall not have an /Encoding entry "
                "(R73 Phase 1 fix: remove /Encoding key)"
            ),
            detected=True,
            routable_to_mode_b=False,
            routable_to_simple_font=True,
            manual_only=False,
        ),
        detect=_detect_7_21_6_3,
    ),
    RegisteredDetector(
        rule_id="7.21.8-1",
        support=RuleSupport(
            rule_id="7.21.8-1",
            description=(
                "CIDFont's CIDSystemInfo must match the associated CMap "
                "(Identity-H/V requires Adobe-Identity-0)"
            ),
            detected=True,
            routable_to_mode_b=False,
            routable_to_simple_font=False,
            manual_only=True,
        ),
        detect=_detect_7_21_8_1,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_all_detectors(
    font_dict: pikepdf.Dictionary,
    used_cids: frozenset[int],
    cid_unicode_map: "dict[int, int] | None",
) -> frozenset[str]:
    """Run every registered detector and return the set of triggered rule IDs.

    A detector that raises is silently skipped so one buggy detector cannot
    block the others from running.
    """
    from project_remedy.faithful_rebuild.font_analysis import _get_descendant
    descendant = _get_descendant(font_dict)
    violations: set[str] = set()
    for detector in REGISTRY:
        try:
            if detector.detect(font_dict, descendant, used_cids, cid_unicode_map):
                violations.add(detector.rule_id)
        except Exception:
            # Don't let a buggy detector block others; skip silently.
            continue
    return frozenset(violations)


def get_support_matrix() -> list[RuleSupport]:
    """Return the support metadata for all registered rules."""
    return [d.support for d in REGISTRY]


def rule_is_routable_to_mode_b(rule_id: str) -> bool:
    """Return True if the given rule can be addressed by Mode B font replacement."""
    for detector in REGISTRY:
        if detector.rule_id == rule_id:
            return detector.support.routable_to_mode_b
    return False
