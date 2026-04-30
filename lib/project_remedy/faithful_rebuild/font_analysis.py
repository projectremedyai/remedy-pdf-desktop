"""Pure analysis helpers for the font canary.

No PDF mutation. Every function is a pure query over existing PDF state.
Mutating repair logic lives in pdf_fixer.py and vision_planner/rule_router.py;
those must not be called from this module.

See docs/superpowers/specs/2026-04-14-font-canary-mode-b-design.md for
design context and the broader canary architecture.
"""

from __future__ import annotations

import logging
import re

import pikepdf
from pikepdf import PdfError

logger = logging.getLogger(__name__)


def _decode_cids_from_string_operand(operand: pikepdf.String | bytes) -> list[int]:
    """Decode a Tj/TJ string operand into a list of 2-byte CIDs.

    Assumes Identity-H encoding (canary invariant). Source bytes are pairs of
    big-endian CID bytes: bytes[0]<<8 | bytes[1].

    An odd-byte-length operand is padded with a 0x00 byte and logged as a
    warning. Upstream audit should never feed such operands in practice;
    canary eligibility requires well-formed Identity-H content.
    """
    raw = bytes(operand) if isinstance(operand, pikepdf.String) else operand
    if len(raw) % 2 != 0:
        logger.debug(
            "odd-length CID string operand (%d bytes); padding final byte",
            len(raw),
        )
        raw = raw + b"\x00"
    return [(raw[i] << 8) | raw[i + 1] for i in range(0, len(raw), 2)]


def extract_used_cids(
    page: pikepdf.Page,
    target_font_key: str,
) -> frozenset[int]:
    """Walk the page content stream and collect every CID shown via the
    target font (Tj and TJ only).

    Raises ValueError if the content stream uses ' or " text-showing
    operators, or contains an inline image while the target font is the
    current font inside a BT..ET frame. Canary eligibility check treats
    these as disqualifying.

    Args:
        page: pikepdf.Page to inspect.
        target_font_key: resource key (e.g. "/F1") whose usage to collect.

    Returns:
        frozenset of CIDs (0..65535) used with the target font.
    """
    used: set[int] = set()
    current_font_key: str | None = None
    in_text_object: bool = False

    try:
        instructions = list(pikepdf.parse_content_stream(page))
    except PdfError as exc:
        raise ValueError(f"Unparseable content stream: {exc}") from exc

    for operands, op in instructions:
        op_str = str(op)

        if op_str == "BT":
            in_text_object = True
            current_font_key = None  # font state resets at BT per PDF spec
        elif op_str == "ET":
            in_text_object = False
            current_font_key = None
        elif op_str == "Tf" and len(operands) >= 1:
            font_name = operands[0]
            if isinstance(font_name, pikepdf.Name):
                current_font_key = str(font_name)
        elif op_str == "Tj":
            if current_font_key == target_font_key and len(operands) >= 1:
                used.update(_decode_cids_from_string_operand(operands[0]))
        elif op_str == "TJ":
            if current_font_key == target_font_key and len(operands) >= 1:
                for item in operands[0]:
                    if isinstance(item, (pikepdf.String, bytes)):
                        used.update(_decode_cids_from_string_operand(item))
                    # numbers are kerning — ignore
        elif op_str == "'":
            if current_font_key == target_font_key:
                raise ValueError(
                    f"Canary disqualifying: quote operator (') used with font "
                    f"{target_font_key!r}. Supported operators are Tj and TJ."
                )
        elif op_str == '"':
            if current_font_key == target_font_key:
                raise ValueError(
                    f"Canary disqualifying: doublequote operator (\") used with "
                    f"font {target_font_key!r}. Supported operators are Tj and TJ."
                )
        elif op_str == "INLINE IMAGE":
            if in_text_object and current_font_key == target_font_key:
                raise ValueError(
                    f"Canary disqualifying: inline image inside BT..ET frame "
                    f"while font {target_font_key!r} is current."
                )

    return frozenset(used)


def extract_used_char_codes(
    page: pikepdf.Page,
    target_font_key: str,
) -> frozenset[int]:
    """Walk the page content stream and collect every 1-byte character code
    shown via the target simple font (Type1 or non-CID TrueType).

    Unlike :func:`extract_used_cids` (which targets Type0 / Identity-H),
    this function:
      - decodes string operands as individual bytes (0..255), not 2-byte CIDs
      - handles ' and " operators (legitimate PDF text operators for simple
        fonts; not disqualifying here)
      - returns frozenset[int] of codes 0..255

    Raises ValueError on unparseable content streams.
    """
    used: set[int] = set()
    current_font_key: str | None = None

    try:
        instructions = list(pikepdf.parse_content_stream(page))
    except PdfError as exc:
        raise ValueError(f"Unparseable content stream: {exc}") from exc

    def _collect_from_string(op: object) -> None:
        if isinstance(op, pikepdf.String):
            used.update(bytes(op))
        elif isinstance(op, (bytes, bytearray)):
            used.update(op)

    for operands, op in instructions:
        op_str = str(op)

        if op_str == "BT":
            current_font_key = None
        elif op_str == "ET":
            current_font_key = None
        elif op_str == "Tf" and len(operands) >= 1:
            font_name = operands[0]
            if isinstance(font_name, pikepdf.Name):
                current_font_key = str(font_name)
        elif op_str == "Tj":
            if current_font_key == target_font_key and len(operands) >= 1:
                _collect_from_string(operands[0])
        elif op_str == "TJ":
            if current_font_key == target_font_key and len(operands) >= 1:
                for item in operands[0]:
                    _collect_from_string(item)  # numbers are ignored as non-string
        elif op_str == "'":
            # "next-line-then-show-text" — shows the string operand with current font
            if current_font_key == target_font_key and len(operands) >= 1:
                _collect_from_string(operands[0])
        elif op_str == '"':
            # "aw ac string"" — operand[2] is the shown string
            if current_font_key == target_font_key and len(operands) >= 3:
                _collect_from_string(operands[2])

    return frozenset(used)


_BFCHAR_ENTRY_RE = re.compile(
    rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>",
)
_BFRANGE_ENTRY_RE = re.compile(
    rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>",
)

# Matches the 6-uppercase-letter subset prefix that PDF font-subsetting tools
# prepend to font names, e.g. "ABCDEF+Arial" or "QQYRIS+Arial-BoldMT".
# Used by audit_font_violations to replicate veraPDF's 7.21.4.2-2 precondition.
_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")


def _parse_tounicode_bfchar(cmap_bytes: bytes) -> dict[int, list[int]]:
    """Parse bfchar/bfrange entries from a ToUnicode CMap.

    Returns {src_cid: [unicode_scalars]}. A list of scalars per src_cid covers
    multi-char mappings; the caller treats any entry with len>1 as a ligature
    and disqualifies the canary.
    """
    out: dict[int, list[int]] = {}

    # bfchar blocks
    for block in re.findall(
        rb"beginbfchar(.*?)endbfchar", cmap_bytes, flags=re.DOTALL
    ):
        for match in _BFCHAR_ENTRY_RE.finditer(block):
            src_hex, dst_hex = match.group(1), match.group(2)
            src_cid = int(src_hex, 16)
            dst_bytes = bytes.fromhex(dst_hex.decode())
            if len(dst_bytes) % 2 != 0:
                continue  # malformed; skip
            scalars = [
                (dst_bytes[i] << 8) | dst_bytes[i + 1]
                for i in range(0, len(dst_bytes), 2)
            ]
            out[src_cid] = scalars

    # bfrange blocks
    for block in re.findall(
        rb"beginbfrange(.*?)endbfrange", cmap_bytes, flags=re.DOTALL
    ):
        for match in _BFRANGE_ENTRY_RE.finditer(block):
            lo = int(match.group(1), 16)
            hi = int(match.group(2), 16)
            base_hex = match.group(3)
            base_bytes = bytes.fromhex(base_hex.decode())
            if len(base_bytes) != 2:
                # Multi-char base — v3 conservatively skips these.
                continue
            base = (base_bytes[0] << 8) | base_bytes[1]
            for i in range(hi - lo + 1):
                out[lo + i] = [base + i]

    return out


def derive_cid_unicode_map(
    font_dict: pikepdf.Dictionary,
    used_cids: frozenset[int],
) -> dict[int, int] | None:
    """Return cid -> unicode_scalar for every used CID, or None if coverage is
    incomplete or any mapping is multi-character.

    Pure: does not mutate font_dict. v3 scope: only uses the source /ToUnicode
    CMap (bfchar + bfrange). Does not fall back to font-program cmap or post
    tables — those paths are deferred to v4.
    """
    tounicode = font_dict.get("/ToUnicode")
    if tounicode is None:
        return None
    try:
        cmap_bytes = bytes(tounicode.read_bytes())
    except Exception:
        return None
    if not cmap_bytes:
        return None

    parsed = _parse_tounicode_bfchar(cmap_bytes)

    result: dict[int, int] = {}
    for cid in used_cids:
        scalars = parsed.get(cid)
        if scalars is None:
            return None  # incomplete coverage
        if len(scalars) != 1:
            return None  # multi-char mapping (ligature) — out of v3 scope
        result[cid] = scalars[0]
    return result


def _get_descendant(font_dict: pikepdf.Dictionary) -> pikepdf.Dictionary | None:
    """Return /DescendantFonts[0] resolved to the actual dict, or None.

    pikepdf transparently dereferences indirect objects — indexing an Array
    that contains an indirect Dictionary returns the dereferenced Dictionary
    directly. No explicit .get_object() call is needed (and pikepdf.Object
    does not expose such a method in any case).
    """
    descendants = font_dict.get("/DescendantFonts")
    if not descendants or len(descendants) == 0:
        return None
    return descendants[0]


def _cidset_covers(
    cidset_stream: pikepdf.Stream,
    used_cids: frozenset[int],
) -> bool:
    """Check CIDSet bitmask: bit (c // 8), bit index (7 - c % 8) must be set
    for every used CID. Returns False on first missing bit.

    The bitmask is a packed byte string: byte 0, bit 7 is CID 0; byte 0, bit 0
    is CID 7; byte 1, bit 7 is CID 8; etc.
    """
    try:
        bits = bytes(cidset_stream.read_bytes())
    except Exception:
        return False
    for cid in used_cids:
        byte_idx = cid // 8
        bit_idx = 7 - (cid % 8)
        if byte_idx >= len(bits):
            return False
        if not (bits[byte_idx] & (1 << bit_idx)):
            return False
    return True


def audit_font_violations(
    font_dict: pikepdf.Dictionary,
    used_cids: frozenset[int],
    cid_unicode_map: dict[int, int] | None,
) -> frozenset[str]:
    """Return the set of veraPDF rule IDs this font currently violates.

    Canary-in-scope rules only:
      - 7.21.7-1 — ToUnicode CMap absent or its stream empty (veraPDF-aligned)
      - 7.21.4.1-1 — FontFile2 missing on the descendant FontDescriptor
      - 7.21.4.2-2 — CIDSet missing or incomplete vs used_cids

    Does NOT check 7.21.6-3 (simple-font rule, out of v3 scope).
    Pure: does not mutate font_dict.

    The ``cid_unicode_map`` parameter is retained for call-site stability but
    is no longer consulted for 7.21.7-1. The correctness sweep
    (docs/findings/remedy76-audit-correctness-sweep-20260416.md) proved that
    ``derive_cid_unicode_map`` returning None also fires on valid constructs
    (ligature bfchar, surrogate-pair bfrange, array-form bfrange) that
    veraPDF accepts. Using it as a 7.21.7-1 proxy produced 0/10 agreement.
    Eligibility callers that still need the stricter invariant should inspect
    ``cid_unicode_map`` directly — not through this audit.
    """
    violations: set[str] = set()

    # 7.21.7-1 — flag only when /ToUnicode is absent or the stream is empty.
    # See docstring for why ``cid_unicode_map`` is no longer consulted here.
    tounicode = font_dict.get("/ToUnicode")
    if tounicode is None:
        violations.add("7.21.7-1")
    else:
        try:
            if not bytes(tounicode.read_bytes()):
                violations.add("7.21.7-1")
        except Exception:
            violations.add("7.21.7-1")

    descendant = _get_descendant(font_dict)
    if descendant is None:
        # Without a descendant, both 7.21.4.1-1 and 7.21.4.2-2 are violated.
        violations.add("7.21.4.1-1")
        violations.add("7.21.4.2-2")
        return frozenset(violations)

    descriptor = descendant.get("/FontDescriptor")
    if descriptor is None:
        violations.add("7.21.4.1-1")
        violations.add("7.21.4.2-2")
        return frozenset(violations)

    # 7.21.4.1-1 — FontFile2 must be present on descendant's descriptor.
    font_file2 = descriptor.get("/FontFile2")
    if font_file2 is None:
        violations.add("7.21.4.1-1")

    # 7.21.4.2-2 — If the font is an embedded subsetted CID font with a
    # CIDSet stream, that CIDSet must identify all CIDs present in the font.
    # Apply the check only when all preconditions hold, matching veraPDF's
    # rule logic:
    #   containsFontFile == false || fontName.search(/[A-Z]{6}\+/) != 0
    #     || containsCIDSet == false || cidSetListsAllGlyphs == true
    # (veraPDF checks against font-program CIDs; we approximate with
    #  used_cids — strictly looser, but used_cids is a subset of font-program
    #  CIDs in practice, so false negatives are rare.)
    cidset = descriptor.get("/CIDSet")
    if font_file2 is not None and cidset is not None:
        basefont_name = str(descendant.get("/BaseFont", "")).lstrip("/")
        is_subset = bool(_SUBSET_PREFIX_RE.match(basefont_name))
        if is_subset and not _cidset_covers(cidset, used_cids):
            violations.add("7.21.4.2-2")

    return frozenset(violations)


def _is_canary_replacement_candidate(
    trigger_rules: frozenset[str],
    cid_unicode_map: dict[int, int] | None,
) -> bool:
    """True iff the font is a Mode B canary replacement candidate.

    After REMEDY-76, ``audit_font_violations`` no longer conflates partial
    CID→Unicode coverage with veraPDF rule 7.21.7-1 (that was a 0/10-agreement
    proxy). Callers that want the combined "font needs replacement" signal
    must now consult both axes:

    - ``trigger_rules`` non-empty — veraPDF flags at least one fixable rule
    - ``cid_unicode_map is None`` — partial coverage (ligature/surrogate/array
      bfrange) that canary replacement can still address

    Either condition makes the font a candidate.
    """
    return bool(trigger_rules) or cid_unicode_map is None


from project_remedy.faithful_rebuild.models import CanaryEligibility


def check_canary_eligibility(pdf: pikepdf.Pdf) -> CanaryEligibility:
    """Check whether pdf qualifies for the v3 canary.

    Enforces every constraint in the spec's 'Canary eligibility' section:
      - Exactly one broken Type0/CIDFontType2/Identity-H font
      - That font's used on exactly one page, in exactly one resource dict
      - Font not in Form XObjects, annotation appearances, or AcroForm /DR
      - Content stream uses only Tj / TJ for the font (no ', ")
      - 100% CID->Unicode coverage derivable via source /ToUnicode
      - Non-empty trigger_rules (font actually needs replacement)

    See spec: docs/superpowers/specs/2026-04-14-font-canary-mode-b-design.md
    """
    reasons: list[str] = []

    # Scan every page for Type0/Identity-H fonts, grouped by indirect object id.
    # Key: font_obj.objgen (tuple). Value: dict with font_obj + placements.
    broken_type0: dict[tuple, dict] = {}

    for page_idx, page in enumerate(pdf.pages):
        resources = page.obj.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get("/Font")
        if fonts is None:
            continue
        for key, font_obj in fonts.items():
            # Resolve indirect to the underlying dict but keep the indirect
            # reference for later emplace (caller-facing).
            if not isinstance(font_obj, pikepdf.Object):
                continue
            if not font_obj.is_indirect:
                # Inline font dict — canary requires indirect for emplace.
                # Treat as a disqualifying presence; collect for later.
                font_dict = font_obj
                objgen = ("inline", id(font_obj))
            else:
                # pikepdf objects transparently deref; just use the object.
                font_dict = font_obj
                objgen = font_obj.objgen

            if font_dict.get("/Subtype") != pikepdf.Name("/Type0"):
                continue
            if font_dict.get("/Encoding") != pikepdf.Name("/Identity-H"):
                continue
            descendant = _get_descendant(font_dict)
            if descendant is None:
                continue
            if descendant.get("/Subtype") != pikepdf.Name("/CIDFontType2"):
                continue

            entry = broken_type0.setdefault(objgen, {
                "font_obj": font_obj,
                "placements": [],  # list of (page_idx, font_key)
            })
            entry["placements"].append((page_idx, str(key)))

    # Now filter to fonts that actually have violations.
    candidate_fonts = []
    for objgen, entry in broken_type0.items():
        font_obj = entry["font_obj"]
        placements = entry["placements"]
        # For simplicity, use the first placement as "the" placement — but
        # we'll reject if placements > 1 below.
        if not placements:
            continue
        page_idx, font_key = placements[0]
        # pikepdf transparently dereferences indirects; the object itself
        # supports dict-style access.
        font_dict = font_obj
        page = pdf.pages[page_idx]
        try:
            used_cids = extract_used_cids(page, font_key)
        except ValueError as exc:
            reasons.append(f"font {font_key} on page {page_idx}: {exc}")
            continue
        if not used_cids:
            # Font present but never used via Tj/TJ; not a candidate
            continue
        cid_unicode_map = derive_cid_unicode_map(font_dict, used_cids)
        trigger_rules = audit_font_violations(font_dict, used_cids, cid_unicode_map)
        if not _is_canary_replacement_candidate(trigger_rules, cid_unicode_map):
            # Font is healthy (veraPDF-clean and full CID coverage); not a candidate
            continue
        candidate_fonts.append({
            "objgen": objgen,
            "font_obj": font_obj,
            "font_key": font_key,
            "page_idx": page_idx,
            "placements": placements,
            "used_cids": used_cids,
            "cid_unicode_map": cid_unicode_map,
            "trigger_rules": trigger_rules,
        })

    if not candidate_fonts:
        reasons.append("No broken Type0/Identity-H fonts found")
        return CanaryEligibility(qualifies=False, disqualifying_reasons=reasons)

    if len(candidate_fonts) > 1:
        reasons.append(
            f"Multiple broken fonts ({len(candidate_fonts)}); canary requires exactly one"
        )
        return CanaryEligibility(qualifies=False, disqualifying_reasons=reasons)

    cand = candidate_fonts[0]

    # Single-placement check — only one (page, key) pair allowed.
    if len(cand["placements"]) > 1:
        page_list = [p for p, _ in cand["placements"]]
        reasons.append(
            f"Font used on multiple pages {page_list}; canary requires single-placement"
        )

    # 100% CID->Unicode coverage must be derivable
    if cand["cid_unicode_map"] is None:
        reasons.append(
            f"Cannot derive complete CID->Unicode map for font {cand['font_key']}; "
            "canary requires 100% coverage via source /ToUnicode"
        )

    # Form XObject / annotation / AcroForm-DR enumeration
    for scope, resources in _iter_resource_dicts(pdf):
        if scope == "page":
            continue  # page resources already counted in placements
        fonts_in_scope = resources.get("/Font") if resources is not None else None
        if fonts_in_scope is None:
            continue
        for key, other_font in fonts_in_scope.items():
            if not isinstance(other_font, pikepdf.Object):
                continue
            if not other_font.is_indirect:
                continue
            if other_font.objgen == cand["objgen"]:
                reasons.append(
                    f"Font {cand['font_key']} also referenced from {scope} "
                    "(canary requires page-resource-only placement)"
                )

    if reasons:
        return CanaryEligibility(qualifies=False, disqualifying_reasons=reasons)

    return CanaryEligibility(
        qualifies=True,
        font_object=cand["font_obj"],
        font_key=cand["font_key"],
        page_index=cand["page_idx"],
        used_cids=cand["used_cids"],
        cid_unicode_map=cand["cid_unicode_map"],
        trigger_rules=cand["trigger_rules"],
        disqualifying_reasons=[],
    )


def check_multifont_eligibility(pdf: pikepdf.Pdf) -> "MultiCanaryEligibility":
    """Check each broken Type0 font for replacement eligibility — STRICT.

    Differs from check_canary_eligibility:
      - Does NOT reject when multiple fonts are broken (returns N eligibilities)
      - Does NOT reject when a font is used on multiple pages
      - Unions used_cids across all placements per font
      - Still requires STRICT ToUnicode coverage per font (replacement-safe)
      - Still rejects fonts referenced from Form XObjects / AcroForm DR
    """
    from project_remedy.faithful_rebuild.models import (
        CanaryEligibility, MultiCanaryEligibility,
    )

    broken_type0: dict[tuple, dict] = {}

    for page_idx, page in enumerate(pdf.pages):
        resources = page.obj.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get("/Font")
        if fonts is None:
            continue
        for key, font_obj in fonts.items():
            if not isinstance(font_obj, pikepdf.Object):
                continue
            if not font_obj.is_indirect:
                continue
            if font_obj.get("/Subtype") != pikepdf.Name("/Type0"):
                continue
            if font_obj.get("/Encoding") != pikepdf.Name("/Identity-H"):
                continue
            descendant = _get_descendant(font_obj)
            if descendant is None:
                continue
            if descendant.get("/Subtype") != pikepdf.Name("/CIDFontType2"):
                continue
            objgen = font_obj.objgen
            entry = broken_type0.setdefault(objgen, {
                "font_obj": font_obj,
                "placements": [],
            })
            entry["placements"].append((page_idx, str(key)))

    eligibilities: list[CanaryEligibility] = []

    for objgen, entry in broken_type0.items():
        font_obj = entry["font_obj"]
        placements = entry["placements"]

        combined_used_cids: set[int] = set()
        for page_idx, font_key in placements:
            try:
                page_cids = extract_used_cids(pdf.pages[page_idx], font_key)
                combined_used_cids.update(page_cids)
            except ValueError:
                continue
        used_cids = frozenset(combined_used_cids)

        if not used_cids:
            continue

        cid_unicode_map = derive_cid_unicode_map(font_obj, used_cids)
        trigger_rules = audit_font_violations(font_obj, used_cids, cid_unicode_map)

        if not _is_canary_replacement_candidate(trigger_rules, cid_unicode_map):
            continue

        first_page, first_key = placements[0]

        if cid_unicode_map is None:
            eligibilities.append(CanaryEligibility(
                qualifies=False,
                font_object=font_obj,
                font_key=first_key,
                page_index=first_page,
                used_cids=used_cids,
                cid_unicode_map=None,
                trigger_rules=trigger_rules,
                placements=placements,
                disqualifying_reasons=[
                    f"Cannot derive complete CID->Unicode coverage for "
                    f"font {first_key} (strict gate). Recovery-enabled "
                    f"replacement deferred."
                ],
            ))
            continue

        form_xobject_conflict = False
        for scope, resources in _iter_resource_dicts(pdf):
            if scope == "page":
                continue
            fonts_in_scope = resources.get("/Font") if resources is not None else None
            if fonts_in_scope is None:
                continue
            for _k, other_font in fonts_in_scope.items():
                if isinstance(other_font, pikepdf.Object) and other_font.is_indirect:
                    if other_font.objgen == objgen:
                        form_xobject_conflict = True
                        break
            if form_xobject_conflict:
                break

        if form_xobject_conflict:
            eligibilities.append(CanaryEligibility(
                qualifies=False,
                font_object=font_obj,
                font_key=first_key,
                page_index=first_page,
                used_cids=used_cids,
                cid_unicode_map=cid_unicode_map,
                trigger_rules=trigger_rules,
                placements=placements,
                disqualifying_reasons=["Font referenced from Form XObject or AcroForm DR"],
            ))
            continue

        eligibilities.append(CanaryEligibility(
            qualifies=True,
            font_object=font_obj,
            font_key=first_key,
            page_index=first_page,
            used_cids=used_cids,
            cid_unicode_map=cid_unicode_map,
            trigger_rules=trigger_rules,
            placements=placements,
            disqualifying_reasons=[],
        ))

    return MultiCanaryEligibility(font_eligibilities=eligibilities)


def check_multifont_eligibility_with_recovery(
    pdf: pikepdf.Pdf,
) -> "MultiCanaryEligibility":
    """Same as check_multifont_eligibility, but uses recovery-enabled derivation.

    Differences:
      - Uses derive_cid_unicode_map_with_fallback instead of strict
        derive_cid_unicode_map
      - Still requires FULL coverage: set(recovered_map.keys()) == set(used_cids).
        Partial coverage after recovery is still a rejection.
      - Tracks recovered_cids_count per eligibility for telemetry.

    This is replacement-safe: the returned cid_unicode_map is complete
    (not partial), so CanaryReplacer.replace() can safely index cid_unicode_map[cid]
    for every used CID.
    """
    from project_remedy.faithful_rebuild.models import (
        CanaryEligibility, MultiCanaryEligibility,
    )

    broken_type0: dict[tuple, dict] = {}

    for page_idx, page in enumerate(pdf.pages):
        resources = page.obj.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get("/Font")
        if fonts is None:
            continue
        for key, font_obj in fonts.items():
            if not isinstance(font_obj, pikepdf.Object):
                continue
            if not font_obj.is_indirect:
                continue
            if font_obj.get("/Subtype") != pikepdf.Name("/Type0"):
                continue
            if font_obj.get("/Encoding") != pikepdf.Name("/Identity-H"):
                continue
            descendant = _get_descendant(font_obj)
            if descendant is None:
                continue
            if descendant.get("/Subtype") != pikepdf.Name("/CIDFontType2"):
                continue
            objgen = font_obj.objgen
            entry = broken_type0.setdefault(objgen, {
                "font_obj": font_obj,
                "placements": [],
            })
            entry["placements"].append((page_idx, str(key)))

    eligibilities: list[CanaryEligibility] = []

    for objgen, entry in broken_type0.items():
        font_obj = entry["font_obj"]
        placements = entry["placements"]

        combined_used_cids: set[int] = set()
        for page_idx, font_key in placements:
            try:
                page_cids = extract_used_cids(pdf.pages[page_idx], font_key)
                combined_used_cids.update(page_cids)
            except ValueError:
                continue
        used_cids = frozenset(combined_used_cids)

        if not used_cids:
            continue

        # Recovery-enabled derivation — returns (map, recovered_count)
        recovered_map, recovered_count = derive_cid_unicode_map_with_fallback(
            font_obj, used_cids
        )

        # Violations must still be present for font to be a replacement candidate.
        # Use STRICT derivation for violation check — recovery shouldn't mask
        # missing-ToUnicode (7.21.7-1) as healthy. 7.21.4.1-1 (FontFile2) and
        # 7.21.4.2-2 (CIDSet) are independent of cid_unicode_map anyway.
        strict_map = derive_cid_unicode_map(font_obj, used_cids)
        trigger_rules = audit_font_violations(font_obj, used_cids, strict_map)

        if not _is_canary_replacement_candidate(trigger_rules, strict_map):
            continue

        first_page, first_key = placements[0]

        # REPLACEMENT-SAFE GATE: recovered_map must cover ALL used CIDs.
        # derive_cid_unicode_map_with_fallback returns (dict, int) — the dict
        # may be empty but never None. We check key-set equality to state the
        # invariant directly (Codex recommendation).
        if not recovered_map or set(recovered_map.keys()) != set(used_cids):
            missing = len(used_cids) - len(recovered_map)
            eligibilities.append(CanaryEligibility(
                qualifies=False,
                font_object=font_obj,
                font_key=first_key,
                page_index=first_page,
                used_cids=used_cids,
                cid_unicode_map=None,
                trigger_rules=trigger_rules,
                placements=placements,
                disqualifying_reasons=[
                    f"Even with recovery, {missing} of {len(used_cids)} used "
                    f"CIDs cannot be mapped to Unicode. Font cannot be "
                    f"replacement-safe."
                ],
                recovered_cids_count=recovered_count,
            ))
            continue

        # Form XObject / AcroForm DR guard (same as strict path)
        form_xobject_conflict = False
        for scope, resources in _iter_resource_dicts(pdf):
            if scope == "page":
                continue
            fonts_in_scope = resources.get("/Font") if resources is not None else None
            if fonts_in_scope is None:
                continue
            for _k, other_font in fonts_in_scope.items():
                if isinstance(other_font, pikepdf.Object) and other_font.is_indirect:
                    if other_font.objgen == objgen:
                        form_xobject_conflict = True
                        break
            if form_xobject_conflict:
                break

        if form_xobject_conflict:
            eligibilities.append(CanaryEligibility(
                qualifies=False,
                font_object=font_obj,
                font_key=first_key,
                page_index=first_page,
                used_cids=used_cids,
                cid_unicode_map=recovered_map,
                trigger_rules=trigger_rules,
                placements=placements,
                disqualifying_reasons=["Font referenced from Form XObject or AcroForm DR"],
                recovered_cids_count=recovered_count,
            ))
            continue

        eligibilities.append(CanaryEligibility(
            qualifies=True,
            font_object=font_obj,
            font_key=first_key,
            page_index=first_page,
            used_cids=used_cids,
            cid_unicode_map=recovered_map,
            trigger_rules=trigger_rules,
            placements=placements,
            disqualifying_reasons=[],
            recovered_cids_count=recovered_count,
        ))

    return MultiCanaryEligibility(font_eligibilities=eligibilities)


def _iter_resource_dicts(pdf: pikepdf.Pdf):
    """Yield (scope, resources_dict) tuples for every non-page resource dict.

    Scopes:
      - "page": page-level /Resources
      - "form_xobject": /Resources on a /Subtype /Form XObject
      - "annotation_appearance": /Resources on an annotation /AP variant
      - "acroform_dr": the AcroForm /DR dictionary
    """
    import pikepdf

    for page in pdf.pages:
        resources = page.obj.get("/Resources")
        if resources is not None:
            yield ("page", resources)
            xobjects = resources.get("/XObject")
            if xobjects is not None:
                for xname, xobj in xobjects.items():
                    if isinstance(xobj, pikepdf.Object) and xobj.is_indirect:
                        if xobj.get("/Subtype") == pikepdf.Name("/Form"):
                            xresources = xobj.get("/Resources")
                            if xresources is not None:
                                yield ("form_xobject", xresources)
        annots = page.obj.get("/Annots")
        if annots is not None:
            for annot in annots:
                if not hasattr(annot, "get"):
                    continue
                ap = annot.get("/AP")
                if ap is None:
                    continue
                for state_key in ("/N", "/R", "/D"):
                    state = ap.get(state_key)
                    if state is None:
                        continue
                    if isinstance(state, pikepdf.Dictionary):
                        for sub in state.values():
                            if isinstance(sub, pikepdf.Object) and sub.is_indirect:
                                sresources = sub.get("/Resources")
                                if sresources is not None:
                                    yield ("annotation_appearance", sresources)

    try:
        root = pdf.Root
    except Exception:
        return
    acroform = root.get("/AcroForm")
    if acroform is not None:
        dr = acroform.get("/DR")
        if dr is not None:
            yield ("acroform_dr", dr)


# ---------------------------------------------------------------------------
# v4 Measurement: Unicode block mapping for codepoint histogram
# ---------------------------------------------------------------------------

# Static table of (range_start, range_end_inclusive, block_name). Ranges cover
# the blocks most relevant to document fonts. Codepoints outside these ranges
# return "Other/Unassigned". Source: Unicode standard Blocks.txt (subset).
_UNICODE_BLOCKS: list[tuple[int, int, str]] = [
    (0x0000, 0x007F, "Basic Latin"),
    (0x0080, 0x00FF, "Latin-1 Supplement"),
    (0x0100, 0x017F, "Latin Extended-A"),
    (0x0180, 0x024F, "Latin Extended-B"),
    (0x0250, 0x02AF, "IPA Extensions"),
    (0x02B0, 0x02FF, "Spacing Modifier Letters"),
    (0x0300, 0x036F, "Combining Diacritical Marks"),
    (0x0370, 0x03FF, "Greek and Coptic"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic Supplement"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul Jamo"),
    (0x1E00, 0x1EFF, "Latin Extended Additional"),
    (0x1F00, 0x1FFF, "Greek Extended"),
    (0x2000, 0x206F, "General Punctuation"),
    (0x2070, 0x209F, "Superscripts and Subscripts"),
    (0x20A0, 0x20CF, "Currency Symbols"),
    (0x2100, 0x214F, "Letterlike Symbols"),
    (0x2150, 0x218F, "Number Forms"),
    (0x2190, 0x21FF, "Arrows"),
    (0x2200, 0x22FF, "Mathematical Operators"),
    (0x2300, 0x23FF, "Miscellaneous Technical"),
    (0x2400, 0x243F, "Control Pictures"),
    (0x2460, 0x24FF, "Enclosed Alphanumerics"),
    (0x2500, 0x257F, "Box Drawing"),
    (0x2580, 0x259F, "Block Elements"),
    (0x25A0, 0x25FF, "Geometric Shapes"),
    (0x2600, 0x26FF, "Miscellaneous Symbols"),
    (0x2700, 0x27BF, "Dingbats"),
    (0x27C0, 0x27EF, "Miscellaneous Mathematical Symbols-A"),
    (0x2900, 0x297F, "Supplemental Arrows-B"),
    (0x2980, 0x29FF, "Miscellaneous Mathematical Symbols-B"),
    (0x2A00, 0x2AFF, "Supplemental Mathematical Operators"),
    (0x2B00, 0x2BFF, "Miscellaneous Symbols and Arrows"),
    (0x3000, 0x303F, "CJK Symbols and Punctuation"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x3100, 0x312F, "Bopomofo"),
    (0x3130, 0x318F, "Hangul Compatibility Jamo"),
    (0x3400, 0x4DBF, "CJK Unified Ideographs Extension A"),
    (0x4E00, 0x9FFF, "CJK Unified Ideographs"),
    (0xA000, 0xA48F, "Yi Syllables"),
    (0xAC00, 0xD7AF, "Hangul Syllables"),
    (0xE000, 0xF8FF, "Private Use Area"),
    (0xF900, 0xFAFF, "CJK Compatibility Ideographs"),
    (0xFB00, 0xFB4F, "Alphabetic Presentation Forms"),
    (0xFE30, 0xFE4F, "CJK Compatibility Forms"),
    (0xFE70, 0xFEFF, "Arabic Presentation Forms-B"),
    (0xFF00, 0xFFEF, "Halfwidth and Fullwidth Forms"),
    (0x1D400, 0x1D7FF, "Mathematical Alphanumeric Symbols"),
    (0x20000, 0x2A6DF, "CJK Unified Ideographs Extension B"),
]


def derive_partial_cid_unicode_map(
    font_dict: pikepdf.Dictionary,
    used_cids: frozenset[int],
) -> tuple[dict[int, int], int]:
    """Return (derived_map, underivable_count) — partial-tolerant version of
    derive_cid_unicode_map.

    For the v4 codepoint histogram we want whatever CIDs DO derive even if
    coverage isn't 100%. Single-char mappings go into derived_map; any CID
    whose ToUnicode is missing, empty, or multi-char is counted as underivable.

    This is distinct from derive_cid_unicode_map, which returns None on any
    incomplete coverage (strict canary gate).
    """
    tounicode = font_dict.get("/ToUnicode")
    if tounicode is None:
        return ({}, len(used_cids))
    try:
        cmap_bytes = bytes(tounicode.read_bytes())
    except Exception:
        return ({}, len(used_cids))
    if not cmap_bytes:
        return ({}, len(used_cids))

    parsed = _parse_tounicode_bfchar(cmap_bytes)
    derived: dict[int, int] = {}
    underivable = 0
    for cid in used_cids:
        scalars = parsed.get(cid)
        if scalars is None or len(scalars) != 1:
            underivable += 1
        else:
            derived[cid] = scalars[0]
    return (derived, underivable)


from project_remedy.faithful_rebuild.font_program_reader import (  # noqa: E402
    recover_cid_unicode_via_cmap,
    recover_cid_unicode_via_post,
)


def derive_cid_unicode_map_with_fallback(
    font_dict: pikepdf.Dictionary,
    used_cids: frozenset[int],
) -> tuple[dict[int, int], int]:
    """Derive CID→Unicode for used CIDs, with font-program fallback.

    Cascade:
      1. Parse existing /ToUnicode CMap (partial-tolerant)
      2. For CIDs NOT covered (excluding multichar entries), try Layer 2
         (cmap table via font_program_reader.recover_cid_unicode_via_cmap)
      3. For CIDs still NOT covered, try Layer 3 (post table via
         font_program_reader.recover_cid_unicode_via_post)

    Multichar ToUnicode entries (ligatures) are preserved as permanently
    underivable — fallback cannot override them with a single scalar.

    Returns (derived_map, recovered_count). derived_map may be incomplete.
    recovered_count is the number of CIDs filled by Layer 2/3 (not present
    in the original ToUnicode).

    Scope: audit-only. NOT used by check_canary_eligibility or
    classify_eligibility_bucket, which stay strict-ToUnicode.
    """
    # Step 1: parse existing ToUnicode (using the raw bfchar parser)
    tounicode = font_dict.get("/ToUnicode")
    parsed_raw: dict[int, list[int]] = {}
    if tounicode is not None:
        try:
            cmap_bytes = bytes(tounicode.read_bytes())
            if cmap_bytes:
                parsed_raw = _parse_tounicode_bfchar(cmap_bytes)
        except Exception:
            pass

    # Separate single-char (usable) from multichar (permanently underivable)
    tounicode_map: dict[int, int] = {}
    multichar_cids: set[int] = set()
    for cid in used_cids:
        scalars = parsed_raw.get(cid)
        if scalars is None:
            continue
        if len(scalars) == 1:
            tounicode_map[cid] = scalars[0]
        else:
            multichar_cids.add(cid)

    # CIDs eligible for fallback: used but not in single-char map AND not multichar
    missing_cids = used_cids - frozenset(tounicode_map.keys()) - multichar_cids

    if not missing_cids:
        return (tounicode_map, 0)

    # Resolve descendant for font-program access
    descendant = _get_descendant(font_dict)
    if descendant is None:
        return (tounicode_map, 0)

    descriptor = descendant.get("/FontDescriptor")
    cidtogidmap = descendant.get("/CIDToGIDMap")

    recovered_count = 0

    # Layer 2: cmap table
    cmap_recovery = recover_cid_unicode_via_cmap(descriptor, cidtogidmap)
    if cmap_recovery:
        for cid in list(missing_cids):
            if cid in cmap_recovery:
                tounicode_map[cid] = cmap_recovery[cid]
                recovered_count += 1
        missing_cids = missing_cids - frozenset(tounicode_map.keys())

    # Layer 3: post table (only if still have gaps)
    if missing_cids:
        post_recovery = recover_cid_unicode_via_post(descriptor, cidtogidmap)
        if post_recovery:
            for cid in list(missing_cids):
                if cid in post_recovery:
                    tounicode_map[cid] = post_recovery[cid]
                    recovered_count += 1

    return (tounicode_map, recovered_count)


def _unicode_block_name(codepoint: int) -> str:
    """Return the Unicode block name for a codepoint, or 'Other/Unassigned'.

    Pure: no I/O, deterministic. Uses a static block table covering ranges
    relevant to document fonts. Unknown ranges (unassigned, surrogates,
    obscure scripts) fall through to 'Other/Unassigned'.
    """
    for start, end, name in _UNICODE_BLOCKS:
        if start <= codepoint <= end:
            return name
    return "Other/Unassigned"


from project_remedy.faithful_rebuild.models import BucketClassification


def _collect_type0_candidates_from_resources(
    resources: pikepdf.Dictionary,
    placement_scope: str,
    page_idx: int,
    type0_candidates: dict[tuple, dict],
    has_simple_broken_font_ref: list[bool],
    has_other_broken_font_ref: list[bool],
    page: "pikepdf.Page | None" = None,
) -> None:
    """Scan a /Resources dictionary and update type0_candidates in-place.

    Args:
        resources: The /Resources dictionary to scan.
        placement_scope: Either "page" (placement tracked) or "form_xobject"
            (font noted as form-xobject-only if not already seen at page scope).
        page_idx: Page index for placement tracking (used when scope is "page").
        type0_candidates: Mutable dict keyed by objgen; updated in-place.
        has_simple_broken_font_ref: Single-element list used as a mutable bool.
        has_other_broken_font_ref: Single-element list used as a mutable bool.
        page: The pikepdf.Page object, required for simple-font CID checking.
    """
    fonts = resources.get("/Font")
    if fonts is None:
        return

    for key, font_obj in fonts.items():
        if not isinstance(font_obj, pikepdf.Object):
            continue
        if not font_obj.is_indirect:
            continue
        objgen = font_obj.objgen
        subtype = font_obj.get("/Subtype")

        if subtype == pikepdf.Name("/Type0"):
            if font_obj.get("/Encoding") != pikepdf.Name("/Identity-H"):
                has_other_broken_font_ref[0] = True
                continue
            descendant = _get_descendant(font_obj)
            if descendant is None or descendant.get("/Subtype") != pikepdf.Name("/CIDFontType2"):
                has_other_broken_font_ref[0] = True
                continue

            entry = type0_candidates.setdefault(objgen, {
                "font_obj": font_obj,
                "placements": [],        # (page_idx, key) pairs from page scope
                "only_in_form_xobject": True,  # flipped to False on first page hit
            })

            if placement_scope == "page":
                entry["only_in_form_xobject"] = False
                entry["placements"].append((page_idx, str(key)))
            # For form_xobject scope we just ensure the font is known; placement
            # tracking happens at page scope.  The only_in_form_xobject flag
            # stays True until the font is seen at page scope.

        elif subtype in (
            pikepdf.Name("/Type1"),
            pikepdf.Name("/TrueType"),
            pikepdf.Name("/MMType1"),
        ):
            if placement_scope == "page" and page is not None:
                # Consider simple fonts only if they actually have violations.
                # Full audit for simple fonts is out of v4.0 scope; this
                # just detects presence for bucket routing.
                try:
                    used_cids = extract_used_cids(page, str(key))
                except ValueError:
                    used_cids = frozenset()
                if used_cids:
                    if font_obj.get("/ToUnicode") is None:
                        has_simple_broken_font_ref[0] = True

        elif subtype == pikepdf.Name("/Type3"):
            has_other_broken_font_ref[0] = True


def classify_eligibility_bucket(pdf: pikepdf.Pdf) -> BucketClassification:
    """Classify a PDF into a v4 measurement bucket.

    Returns a BucketClassification with:
      - primary_bucket: first-matching blocker in spec order (most restrictive
        single blocker wins as primary)
      - also_requires: all scope-extension-level constraints the doc fails

    Bucket order (first match wins as primary, most restrictive first):
      1. v3_qualifying (passes existing check_canary_eligibility)
      2. near_miss_multi_font (most restrictive near-miss — requires concurrent
         font replacement, highest engineering cost to fix)
      3. near_miss_form_xobject
      4. near_miss_multi_placement
      5. near_miss_partial_unicode_map (least restrictive near-miss — only needs
         better ToUnicode derivation)
      6. out_of_scope_simple_font (broken Type1 or simple TrueType)
      7. out_of_scope_other (Type3/Identity-V/pattern/weird)
      8. out_of_scope_no_broken_fonts (no broken fonts anywhere)

    Primary bucket = the most restrictive single blocker (highest engineering
    cost to fix).  multi_font is most restrictive because it requires concurrent
    font replacement; partial_unicode_map is least restrictive because it only
    needs better ToUnicode derivation.

    Bug-fix notes (Task 8 follow-up):
      - Bug 1: Also enumerates Form XObject /Resources/Font so fonts that live
        ONLY in a form xobject are collected (not silently dropped).
      - Bug 2: Unions used_cids across ALL placements of a font; candidate is
        only skipped if the union is empty (was: only first placement checked).
      - Bug 3: On ValueError from extract_used_cids, sets has_other_broken_font
        instead of silently discarding the candidate.
      - Bug 4: also_requires flags are computed as UNION across all broken_type0
        entries (was: checked only against broken_type0[0]).
    """
    # Phase 1: Collect all Type0/Identity-H candidates.
    # We use two passes: first page-level resources, then form-xobject resources.
    # This ensures page-level placements are recorded before form-xobject ones,
    # so only_in_form_xobject is accurate.
    type0_candidates: dict[tuple, dict] = {}
    has_simple_broken_font_ref = [False]
    has_other_broken_font_ref = [False]

    # Pass A: page-level /Resources (and their simple-font detection)
    for page_idx, page in enumerate(pdf.pages):
        resources = page.obj.get("/Resources")
        if resources is None:
            continue
        _collect_type0_candidates_from_resources(
            resources=resources,
            placement_scope="page",
            page_idx=page_idx,
            type0_candidates=type0_candidates,
            has_simple_broken_font_ref=has_simple_broken_font_ref,
            has_other_broken_font_ref=has_other_broken_font_ref,
            page=page,
        )

    # Pass B: Form XObject /Resources (Bug 1 fix — fonts only in form xobjects)
    for page in pdf.pages:
        page_resources = page.obj.get("/Resources")
        if page_resources is None:
            continue
        xobjects = page_resources.get("/XObject")
        if xobjects is None:
            continue
        for _xname, xobj in xobjects.items():
            if not isinstance(xobj, pikepdf.Object) or not xobj.is_indirect:
                continue
            if xobj.get("/Subtype") != pikepdf.Name("/Form"):
                continue
            xresources = xobj.get("/Resources")
            if xresources is None:
                continue
            _collect_type0_candidates_from_resources(
                resources=xresources,
                placement_scope="form_xobject",
                page_idx=-1,  # not used for form_xobject scope
                type0_candidates=type0_candidates,
                has_simple_broken_font_ref=has_simple_broken_font_ref,
                has_other_broken_font_ref=has_other_broken_font_ref,
                page=None,
            )

    has_simple_broken_font = has_simple_broken_font_ref[0]
    has_other_broken_font = has_other_broken_font_ref[0]

    # Phase 2: Filter Type0 candidates to those with active violations.
    # Bug 2 fix: union used_cids across ALL placements (not just first).
    # Bug 3 fix: on ValueError, count as has_other_broken_font instead of drop.
    broken_type0: list[dict] = []

    for objgen, entry in type0_candidates.items():
        font_obj = entry["font_obj"]
        only_in_form_xobject = entry.get("only_in_form_xobject", False)
        placements = entry["placements"]

        # Bug 1 fix: form-xobject-only fonts have no page placements.
        # We need to check the form xobject's own content stream, but since
        # extract_used_cids operates on pikepdf.Page objects, we can't directly
        # walk form-xobject content here.  Instead, we treat form-xobject-only
        # fonts as having non-empty used_cids if the font is structurally broken
        # (audit_font_violations with an assumed non-empty CID set of {0}).
        # The font is definitionally reachable (referenced from an XObject that's
        # invoked from a page), so we use a sentinel CID 0 for violation checking.
        if only_in_form_xobject:
            sentinel_cids = frozenset({0})
            cid_unicode_map = derive_cid_unicode_map(font_obj, sentinel_cids)
            trigger_rules = audit_font_violations(font_obj, sentinel_cids, cid_unicode_map)
            if not trigger_rules:
                continue
            entry["used_cids"] = sentinel_cids
            entry["cid_unicode_map"] = cid_unicode_map
            entry["objgen"] = objgen
            entry["only_in_form_xobject"] = True
            broken_type0.append(entry)
            continue

        # Bug 2 fix: union used_cids across all placements.
        combined_used_cids: set[int] = set()
        value_error_seen = False
        for page_idx, font_key in placements:
            try:
                page_used = extract_used_cids(pdf.pages[page_idx], font_key)
                combined_used_cids.update(page_used)
            except ValueError:
                # Bug 3 fix: ValueError means the font is there but uses
                # unsupported operators — count as has_other_broken_font.
                value_error_seen = True

        if value_error_seen and not combined_used_cids:
            # Font triggered a ValueError on every placement — out_of_scope_other.
            has_other_broken_font = True
            continue

        used_cids = frozenset(combined_used_cids)
        if not used_cids:
            # Font declared but never used via Tj/TJ on any placement.
            continue

        cid_unicode_map = derive_cid_unicode_map(font_obj, used_cids)
        trigger_rules = audit_font_violations(font_obj, used_cids, cid_unicode_map)
        if not _is_canary_replacement_candidate(trigger_rules, cid_unicode_map):
            continue

        entry["used_cids"] = used_cids
        entry["cid_unicode_map"] = cid_unicode_map
        entry["objgen"] = objgen
        broken_type0.append(entry)

    # Out-of-scope shortcuts — no broken Type0 fonts.
    if not broken_type0:
        if has_simple_broken_font:
            return BucketClassification(
                primary_bucket="out_of_scope_simple_font",
                also_requires=frozenset(),
            )
        if has_other_broken_font:
            return BucketClassification(
                primary_bucket="out_of_scope_other",
                also_requires=frozenset(),
            )
        return BucketClassification(
            primary_bucket="out_of_scope_no_broken_fonts",
            also_requires=frozenset(),
        )

    # Phase 3: Build also_requires as a UNION across ALL broken_type0 entries.
    # Bug 4 fix: was checking only broken_type0[0] for multi_placement,
    # partial_unicode_map, and form_xobject flags.
    also_requires: set[str] = set()

    # multi_font: more than one broken Type0 font.
    if len(broken_type0) > 1:
        also_requires.add("multi_font")

    # Collect all broken font objgens for form_xobject lookup.
    broken_objgens: set[tuple] = {e["objgen"] for e in broken_type0}

    # Union-based flags across all broken Type0 fonts.
    for entry in broken_type0:
        # multi_placement: ANY broken font used on more than one page.
        if len(entry["placements"]) > 1:
            also_requires.add("multi_placement")

        # partial_unicode_map: ANY broken font's cid_unicode_map is None.
        if entry["cid_unicode_map"] is None:
            also_requires.add("partial_unicode_map")

        # form_xobject: fonts that are ONLY in form xobjects are already tracked.
        if entry.get("only_in_form_xobject"):
            also_requires.add("form_xobject")

    # form_xobject: also check whether any broken font (that IS at page scope)
    # is ALSO referenced from a non-page resource dict.
    for scope, resources in _iter_resource_dicts(pdf):
        if scope == "page":
            continue
        fonts_in_scope = resources.get("/Font") if resources is not None else None
        if fonts_in_scope is None:
            continue
        for _key, other_font in fonts_in_scope.items():
            if not isinstance(other_font, pikepdf.Object):
                continue
            if not other_font.is_indirect:
                continue
            if other_font.objgen in broken_objgens:
                also_requires.add("form_xobject")

    # If no scope extensions required, it's v3_qualifying.
    if not also_requires:
        return BucketClassification(
            primary_bucket="v3_qualifying",
            also_requires=frozenset(),
        )

    # Primary bucket = first-matching blocker in spec order (most restrictive first).
    if "multi_font" in also_requires:
        primary = "near_miss_multi_font"
    elif "form_xobject" in also_requires:
        primary = "near_miss_form_xobject"
    elif "multi_placement" in also_requires:
        primary = "near_miss_multi_placement"
    elif "partial_unicode_map" in also_requires:
        primary = "near_miss_partial_unicode_map"
    else:
        # Defensive; should not reach here.
        primary = "out_of_scope_other"

    return BucketClassification(
        primary_bucket=primary,
        also_requires=frozenset(also_requires),
    )


def extract_codepoint_histogram(pdf: pikepdf.Pdf) -> dict:
    """Return a per-doc Unicode-block histogram of broken-font codepoint demand.

    Walks every Type0/CIDFontType2/Identity-H font referenced from page
    resources. For each broken font (one with violations per
    audit_font_violations), derives codepoints via derive_cid_unicode_map_with_fallback
    (including font-program recovery) and bins them by Unicode block.

    Returns a dict:
      {
        "total_codepoints_demanded": int,  # unique codepoints derived (including recovery)
        "codepoints_derived_count": int,   # same as total_codepoints_demanded
        "blocks": {block_name: count},     # unique codepoint count per block
      }

    Pure: does not mutate pdf.

    Bug-fix note: a two-pass approach is used to avoid page-order bias.  A
    Type0 font that appears on page 0 with an empty content stream (no Tj/TJ)
    AND on page 1 with real usage must not be silently dropped.  Pass 1
    collects all (font_obj, [(page, key)]) placements without filtering; Pass
    2 unions used_cids across all placements before processing each font once.
    """
    # Pass 1: collect all qualifying Type0 fonts and their placements.
    # Key: objgen tuple.  Value: {font_obj, placements: [(page, key)]}.
    font_placements: dict[tuple, dict] = {}

    for page in pdf.pages:
        resources = page.obj.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get("/Font")
        if fonts is None:
            continue
        for key, font_obj in fonts.items():
            if not isinstance(font_obj, pikepdf.Object):
                continue
            if not font_obj.is_indirect:
                continue
            font_dict = font_obj
            if font_dict.get("/Subtype") != pikepdf.Name("/Type0"):
                continue
            if font_dict.get("/Encoding") != pikepdf.Name("/Identity-H"):
                continue
            descendant = _get_descendant(font_dict)
            if descendant is None:
                continue
            if descendant.get("/Subtype") != pikepdf.Name("/CIDFontType2"):
                continue
            objgen = font_obj.objgen
            entry = font_placements.setdefault(objgen, {
                "font_obj": font_obj,
                "placements": [],
            })
            entry["placements"].append((page, str(key)))

    # Pass 2: for each unique font, union used_cids across all placements.
    all_derived_codepoints: set[int] = set()

    for objgen, entry in font_placements.items():
        font_obj = entry["font_obj"]
        combined_used_cids: set[int] = set()
        for page, key in entry["placements"]:
            try:
                page_cids = extract_used_cids(page, key)
                combined_used_cids.update(page_cids)
            except ValueError:
                continue
        used_cids = frozenset(combined_used_cids)
        if not used_cids:
            continue

        # Only include fonts that are actually broken.
        cid_unicode_map = derive_cid_unicode_map(font_obj, used_cids)
        violations = audit_font_violations(font_obj, used_cids, cid_unicode_map)
        if not _is_canary_replacement_candidate(violations, cid_unicode_map):
            continue

        fallback_map, _recovered = derive_cid_unicode_map_with_fallback(
            font_obj, used_cids
        )
        all_derived_codepoints.update(fallback_map.values() if fallback_map else [])

    blocks: dict[str, int] = {}
    for cp in all_derived_codepoints:
        name = _unicode_block_name(cp)
        blocks[name] = blocks.get(name, 0) + 1

    return {
        "total_codepoints_demanded": len(all_derived_codepoints),
        "codepoints_derived_count": len(all_derived_codepoints),
        "blocks": blocks,
    }
