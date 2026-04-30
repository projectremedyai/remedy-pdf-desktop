"""PUA / custom-glyph detection helpers (REMEDY-74).

Some fonts map glyphs to Unicode Private Use Area (PUA) codepoints
(U+E000..U+F8FF, U+F0000..U+FFFFD, U+100000..U+10FFFD), or expose custom
glyph names that don't resolve via the Adobe Glyph List or post-table
mapping. In both cases any font-replacement attempt that reads the
"original text" gets nonsense — the codepoints aren't real Unicode letters
and the glyph names aren't standardised.

The safest outcome is **detect and skip**, with a clear reason code, so
routing can decide to leave the font alone (manual review) rather than
emit a broken replacement.

This module is defensive and pure:
  - no I/O
  - no PDF mutation
  - no network

Public surface:
  - ``PUA_RANGES``
  - ``is_pua_codepoint``
  - ``PUAAnalysis``
  - ``analyze_pua_usage``
  - ``should_skip_font_for_pua``

Heuristic tradeoffs
-------------------
The custom-glyph-name detector is intentionally conservative. We strip the
6-letter subset prefix (e.g. ``ABCDEF+``) and then check whether the
remainder begins with a recognisable font-family stem or is composed of
"name-like" characters. The bias is **toward NOT flagging** — a false
positive here blocks font replacement for a perfectly good font, which is
more disruptive than missing one bad font (that will then fail its own
validation downstream).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pikepdf

# ---------------------------------------------------------------------------
# PUA ranges
# ---------------------------------------------------------------------------

#: Unicode Private Use Areas, inclusive on both ends.
#: - BMP PUA:       U+E000   .. U+F8FF
#: - SPUA-A:        U+F0000  .. U+FFFFD
#: - SPUA-B:        U+100000 .. U+10FFFD
PUA_RANGES: tuple[tuple[int, int], ...] = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)


def is_pua_codepoint(codepoint: int) -> bool:
    """Return True iff ``codepoint`` falls in any Unicode Private Use Area.

    Pure, branch-lite; safe to call in tight loops.
    """
    for lo, hi in PUA_RANGES:
        if lo <= codepoint <= hi:
            return True
    return False


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Ratio cutoffs. Kept at module scope so callers can introspect them if
# they want to document the policy in their own logs.
_RATIO_SKIP_THRESHOLD: float = 0.5   # >= this -> skip
_RATIO_REVIEW_THRESHOLD: float = 0.1  # >= this and < skip -> review

# Subset prefix is exactly 6 uppercase letters followed by '+', per the
# PDF 1.7 spec (§9.6.4). Used to strip the prefix before analysing the
# font family stem.
_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")

# A short list of font-family stems we treat as "definitely real". If the
# stripped name *starts with* any of these (case-insensitive) we do NOT
# flag it as custom, no matter what follows (so "Helvetica-Bold",
# "TimesNewRomanPSMT", etc. pass through). This list is a heuristic, not
# exhaustive — bias is toward NOT flagging.
_KNOWN_FAMILY_STEMS: tuple[str, ...] = (
    "arial",
    "avenir",
    "calibri",
    "cambria",
    "cambriamath",
    "century",
    "comic",
    "consolas",
    "courier",
    "dejavu",
    "futura",
    "garamond",
    "georgia",
    "helvetica",
    "impact",
    "lato",
    "liberation",
    "lucida",
    "menlo",
    "merriweather",
    "minion",
    "monaco",
    "montserrat",
    "myriad",
    "noto",
    "opensans",
    "palatino",
    "pt",  # PTSans, PTSerif
    "roboto",
    "segoe",
    "source",
    "symbol",
    "tahoma",
    "times",
    "trebuchet",
    "ubuntu",
    "verdana",
    "zapf",
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PUAAnalysis:
    """Result of analysing a font for PUA / custom-glyph risk.

    Attributes
    ----------
    total_codepoints:
        Number of CIDs observed in the supplied ``cid_unicode_map``.
    pua_codepoints:
        How many of those map to a PUA codepoint.
    pua_ratio:
        ``pua_codepoints / total_codepoints`` when ``total_codepoints > 0``,
        else ``0.0``.
    has_custom_glyph_names:
        Best-effort heuristic: True if the font's BaseFont / FontName
        (after stripping any 6-letter subset prefix) does not look like a
        real font-family stem. Biased toward returning False.
    recommendation:
        ``"skip"`` — do not attempt font replacement / text recovery.
        ``"review"`` — refer to human review or a stronger tier.
        ``"proceed"`` — safe to continue with normal pipeline.
    reason:
        Short human-readable sentence explaining the recommendation.
    """

    total_codepoints: int
    pua_codepoints: int
    pua_ratio: float
    has_custom_glyph_names: bool
    recommendation: Literal["skip", "proceed", "review"]
    reason: str


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------


def _extract_font_name(font_dict: pikepdf.Dictionary) -> str:
    """Return the best available font name for heuristics.

    Tries ``/BaseFont`` first, then ``/FontDescriptor`` → ``/FontName``.
    Strips the leading ``/`` that pikepdf Names carry. Empty string if
    neither is present or any access raises.
    """
    try:
        bf = font_dict.get("/BaseFont")
        if bf is not None:
            name = str(bf).lstrip("/")
            if name:
                return name
    except Exception:
        # pikepdf can raise PdfError on malformed refs; treat as absent.
        pass

    try:
        desc = font_dict.get("/FontDescriptor")
        if desc is not None:
            fn = desc.get("/FontName")
            if fn is not None:
                name = str(fn).lstrip("/")
                if name:
                    return name
    except Exception:
        pass

    return ""


def _looks_like_custom_glyph_name(font_name: str) -> bool:
    """Heuristic: does this look like a non-standard custom-glyph font?

    The rule, after stripping any subset prefix:
      1. Empty → False (we can't tell).
      2. Starts with a known font-family stem → False.
      3. Contains 3+ digits interleaved with letters ("MyWeirdGlyphSet123"
         style) → True.
      4. Otherwise → False (bias toward NOT flagging).
    """
    if not font_name:
        return False

    stripped = _SUBSET_PREFIX_RE.sub("", font_name, count=1)
    if not stripped:
        return False

    lower = stripped.lower()
    for stem in _KNOWN_FAMILY_STEMS:
        if lower.startswith(stem):
            return False

    # Count digits — custom glyph set names in the wild very often carry a
    # numeric suffix (GlyphSet123, Icons42, etc.).
    digit_count = sum(1 for ch in stripped if ch.isdigit())
    if digit_count >= 3:
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_pua_usage(
    cid_unicode_map: dict[int, int] | None,
    font_dict: pikepdf.Dictionary,
) -> PUAAnalysis:
    """Analyse a font's Unicode coverage for PUA / custom-glyph risk.

    Parameters
    ----------
    cid_unicode_map:
        Mapping from CID → Unicode codepoint, as recovered from ToUnicode
        or post-table analysis. May be ``None`` or empty — we interpret
        that as "no signal" and return a ``review`` recommendation.
    font_dict:
        The font's PDF dictionary. Used only to read ``/BaseFont`` and
        ``/FontDescriptor/FontName`` — never mutated.

    Returns
    -------
    PUAAnalysis
    """
    font_name = _extract_font_name(font_dict)
    has_custom = _looks_like_custom_glyph_name(font_name)

    # No signal case: None or empty map.
    if not cid_unicode_map:
        if has_custom:
            reason = (
                f"No CID→Unicode map; font name '{font_name}' looks custom "
                "(non-standard glyph set) — skip."
            )
            return PUAAnalysis(
                total_codepoints=0,
                pua_codepoints=0,
                pua_ratio=0.0,
                has_custom_glyph_names=True,
                recommendation="skip",
                reason=reason,
            )
        return PUAAnalysis(
            total_codepoints=0,
            pua_codepoints=0,
            pua_ratio=0.0,
            has_custom_glyph_names=False,
            recommendation="review",
            reason=(
                "No CID→Unicode map available; cannot assess PUA usage — "
                "refer to review."
            ),
        )

    total = len(cid_unicode_map)
    pua_count = sum(1 for cp in cid_unicode_map.values() if is_pua_codepoint(cp))
    ratio = pua_count / total if total > 0 else 0.0

    # Custom-glyph names override to skip regardless of PUA ratio.
    if has_custom:
        return PUAAnalysis(
            total_codepoints=total,
            pua_codepoints=pua_count,
            pua_ratio=ratio,
            has_custom_glyph_names=True,
            recommendation="skip",
            reason=(
                f"Font name '{font_name}' looks like a custom glyph set "
                f"(PUA ratio={ratio:.2f}) — skip."
            ),
        )

    if ratio >= _RATIO_SKIP_THRESHOLD:
        rec: Literal["skip", "review", "proceed"] = "skip"
        reason = (
            f"{pua_count}/{total} codepoints ({ratio:.0%}) fall in the "
            "Unicode Private Use Area — skip font replacement."
        )
    elif ratio >= _RATIO_REVIEW_THRESHOLD:
        rec = "review"
        reason = (
            f"{pua_count}/{total} codepoints ({ratio:.0%}) in the PUA — "
            "refer to review."
        )
    else:
        rec = "proceed"
        reason = (
            f"Only {pua_count}/{total} codepoints ({ratio:.0%}) in the PUA; "
            "safe to proceed."
        )

    return PUAAnalysis(
        total_codepoints=total,
        pua_codepoints=pua_count,
        pua_ratio=ratio,
        has_custom_glyph_names=False,
        recommendation=rec,
        reason=reason,
    )


def should_skip_font_for_pua(
    cid_unicode_map: dict[int, int] | None,
    font_dict: pikepdf.Dictionary,
) -> tuple[bool, str]:
    """Thin wrapper: skip iff ``analyze_pua_usage`` recommends ``"skip"``.

    Review and proceed both return ``(False, reason)`` — routing layers
    decide separately whether a ``review`` verdict should escalate.
    """
    analysis = analyze_pua_usage(cid_unicode_map, font_dict)
    return (analysis.recommendation == "skip", analysis.reason)
