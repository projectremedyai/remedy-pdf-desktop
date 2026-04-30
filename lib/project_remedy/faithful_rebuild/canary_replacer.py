"""CanaryReplacer — 9-step CIDToGIDMap swap pipeline for the v3 canary.

Single-placement, Type0/CIDFontType2/Identity-H only. Preserves source
content stream bytes exactly by preserving source CID space and swapping
the mapping tables (CIDToGIDMap + ToUnicode + /W + /CIDSet) under a new
font program.

See docs/superpowers/specs/2026-04-14-font-canary-mode-b-design.md for the
full design and rationale.

Steps:
  1. Fingerprint source font
  2. Match candidate with confidence + coverage thresholds
  3. Subset + embed candidate
  4. Build replacement /CIDToGIDMap in source-CID space
  5. Build replacement /ToUnicode CMap over source CIDs
  6. Build replacement /W array in source-CID space
  7. Build replacement /CIDSet bitmask in source-CID space
  8. Assemble new Type0 -> CIDFontType2 -> FontDescriptor chain
  9. emplace() the new Type0 dict onto the source font object
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pikepdf
from pikepdf import Dictionary, Name

from project_remedy.faithful_rebuild import font_embedder, font_matcher
from project_remedy.faithful_rebuild.models import CanaryEligibility
from project_remedy.pdf_fixer import build_bfchar_cmap

logger = logging.getLogger(__name__)


@dataclass
class ReplacementReport:
    """Result of a CanaryReplacer.replace() invocation.

    Attributes:
        status: "replaced" | "skipped" | "failed".
        reason: Human-readable explanation when status != "replaced".
        matched_ps_name: PostScript name of the matched candidate font
                         (from prepare_truetype_font), or None if no
                         candidate was prepared. May fall back to the PDF
                         resource key (e.g. "/F1") if the candidate font's
                         name table is unreadable; that fallback is a known
                         font_embedder limitation, not a matcher issue.
        replaced_cids_count: Number of source CIDs that had mapping tables
                             rewritten (equals len(eligibility.used_cids) on
                             success).
    """

    status: str  # "replaced" | "skipped" | "failed"
    reason: str | None = None
    matched_ps_name: str | None = None
    replaced_cids_count: int = 0


class CanaryReplacer:
    """Executes the CIDToGIDMap-swap replacement for a single qualifying font
    in a qualifying PDF.

    Runs all 9 steps (fingerprint, match, subset, CIDToGIDMap, ToUnicode,
    /W, /CIDSet, Type0 assembly, emplace) end-to-end. The replacement
    mutates the PDF in place via ``emplace()``, preserving the source
    font object's indirect-object identity so the page content stream —
    which still references the source CIDs — remains byte-identical.
    """

    MIN_CONFIDENCE: float = 0.60

    def replace(
        self,
        pdf: pikepdf.Pdf,
        eligibility: CanaryEligibility,
    ) -> ReplacementReport:
        """Execute the CIDToGIDMap-swap replacement pipeline.

        Mutates ``pdf`` in place via ``emplace()`` on the source font
        object. The page content stream is preserved byte-identically —
        only the font resource's backing dict is swapped.

        Args:
            pdf: The source PDF, opened via pikepdf.
            eligibility: Result of :func:`check_canary_eligibility`. Must
                         have ``qualifies=True`` and populated
                         ``cid_unicode_map``; otherwise the call returns
                         ``ReplacementReport(status="skipped")``.

        Returns:
            :class:`ReplacementReport` summarising the outcome.
        """
        if not eligibility.qualifies:
            return ReplacementReport(
                status="skipped",
                reason="eligibility.qualifies is False",
            )

        font_obj = eligibility.font_object
        # pikepdf.Object transparently dereferences — use directly as dict.
        font_dict = font_obj
        font_key = eligibility.font_key
        used_cids = eligibility.used_cids
        cid_unicode_map = eligibility.cid_unicode_map
        # Eligibility with qualifies=True guarantees these are populated,
        # but be defensive for use by future callers.
        if cid_unicode_map is None or not used_cids or font_key is None:
            return ReplacementReport(
                status="failed",
                reason=(
                    "eligibility qualifies=True but required fields missing: "
                    f"cid_unicode_map={cid_unicode_map is not None}, "
                    f"used_cids={bool(used_cids)}, font_key={font_key!r}"
                ),
            )

        # ------------------------------------------------------------------
        # Step 1: Fingerprint source font
        # ------------------------------------------------------------------
        fp = font_matcher.fingerprint_pdf_font(font_key, font_dict)

        # ------------------------------------------------------------------
        # Step 2: Match candidate with confidence + full coverage
        # ------------------------------------------------------------------
        index = font_matcher.scan_system_fonts()
        required_codepoints = frozenset(cid_unicode_map.values())
        match = font_matcher.match_font(
            fp,
            index,
            min_confidence=self.MIN_CONFIDENCE,
            require_codepoints=required_codepoints,
        )
        if match.confidence < self.MIN_CONFIDENCE:
            return ReplacementReport(
                status="failed",
                reason=f"No matching font: {match.fallback_reason}",
                matched_ps_name=None,
            )

        # ------------------------------------------------------------------
        # Step 3: Subset + embed candidate
        # ------------------------------------------------------------------
        # prepare_truetype_font(source: Path | bytes, resource_key, text)
        # When use_embedded=True, resolved_path is None and we pass the raw
        # embedded program bytes instead.
        if match.use_embedded:
            if fp.embedded_program is None:
                return ReplacementReport(
                    status="failed",
                    reason="match.use_embedded=True but source has no embedded_program",
                    matched_ps_name=None,
                )
            font_source: "bytes | object" = fp.embedded_program
        else:
            if match.resolved_path is None:
                return ReplacementReport(
                    status="failed",
                    reason="match.use_embedded=False but resolved_path is None",
                    matched_ps_name=None,
                )
            font_source = match.resolved_path

        subset_text = "".join(chr(cp) for cp in sorted(set(cid_unicode_map.values())))
        try:
            prepared = font_embedder.prepare_truetype_font(
                font_source,
                resource_key=font_key,
                text=subset_text,
            )
        except Exception as exc:
            return ReplacementReport(
                status="failed",
                reason=f"prepare_truetype_font raised {type(exc).__name__}: {exc}",
                matched_ps_name=None,
            )

        # matched_ps_name is derived from the prepared candidate, not the source
        # font. This is the PostScript name of the font that was actually embedded
        # (e.g. "Geneva" or "Geneva-Regular"), which is what the JSONL logs should report.
        matched_ps_name = prepared.postscript_name

        # ------------------------------------------------------------------
        # Step 4: Build replacement /CIDToGIDMap in SOURCE-CID space.
        #
        # 2 bytes per CID, big-endian GID. Allocate sized to max(used_cids)+1;
        # CIDs not in used_cids default to 0 (.notdef).
        # ------------------------------------------------------------------
        max_cid = max(used_cids)
        cid_to_gid_bytes = bytearray(2 * (max_cid + 1))
        # Also build a source-CID -> new-GID dict we use for /W.
        new_gid_for_cid: dict[int, int] = {}
        for cid in used_cids:
            unicode_cp = cid_unicode_map[cid]
            new_gid = prepared.gid_for_codepoint.get(unicode_cp)
            if new_gid is None:
                return ReplacementReport(
                    status="failed",
                    reason=(
                        f"Replacement font has no glyph for U+{unicode_cp:04X} "
                        f"(source CID {cid:04X})"
                    ),
                    matched_ps_name=matched_ps_name,
                )
            cid_to_gid_bytes[2 * cid] = (new_gid >> 8) & 0xFF
            cid_to_gid_bytes[2 * cid + 1] = new_gid & 0xFF
            new_gid_for_cid[cid] = new_gid

        # ------------------------------------------------------------------
        # Step 5: Build replacement /ToUnicode CMap over SOURCE CIDs.
        #
        # CID-keyed Identity-H fonts use 2-byte source codes, so byte_width=2.
        # ------------------------------------------------------------------
        tounicode_bytes = build_bfchar_cmap(cid_unicode_map, byte_width=2)

        # ------------------------------------------------------------------
        # Step 6: Build replacement /W array in SOURCE-CID space.
        #
        # For each source CID, width = prepared.width_for_gid[new_gid_for_cid[cid]].
        # Emitted as [cid_start [w0 w1 w2 ...]] runs over consecutive source CIDs.
        # CORRECTNESS INVARIANT (Codex): /W is keyed by SOURCE CID, not GID.
        # ------------------------------------------------------------------
        sorted_cids = sorted(used_cids)
        w_runs: list[tuple[int, list[int]]] = []
        run_start: int | None = None
        run_widths: list[int] = []
        last_cid: int | None = None
        for cid in sorted_cids:
            new_gid = new_gid_for_cid[cid]
            width = prepared.width_for_gid.get(new_gid)
            if width is None:
                return ReplacementReport(
                    status="failed",
                    reason=(
                        f"No width available for GID {new_gid} "
                        f"(source CID {cid:04X} -> U+{cid_unicode_map[cid]:04X})"
                    ),
                    matched_ps_name=matched_ps_name,
                )
            if run_start is None:
                run_start = cid
                run_widths = [width]
                last_cid = cid
            elif last_cid is not None and cid == last_cid + 1:
                run_widths.append(width)
                last_cid = cid
            else:
                w_runs.append((run_start, run_widths))
                run_start = cid
                run_widths = [width]
                last_cid = cid
        if run_start is not None:
            w_runs.append((run_start, run_widths))

        w_array = pikepdf.Array()
        for start, widths in w_runs:
            w_array.append(start)
            w_array.append(pikepdf.Array(widths))

        # ------------------------------------------------------------------
        # Step 7: Build replacement /CIDSet bitmask in SOURCE-CID space.
        #
        # CORRECTNESS INVARIANT (Codex): bits mark SOURCE CIDs, not GIDs.
        # Bit layout: byte (cid // 8), bit (7 - cid % 8).
        # ------------------------------------------------------------------
        cidset_len = (max_cid // 8) + 1
        cidset_bits = bytearray(cidset_len)
        for cid in used_cids:
            byte_idx = cid // 8
            bit_idx = 7 - (cid % 8)
            cidset_bits[byte_idx] |= (1 << bit_idx)

        # ------------------------------------------------------------------
        # Step 8: Assemble new Type0 -> CIDFontType2 -> FontDescriptor chain.
        #
        # We crib numeric descriptor fields (FontBBox, StemV, Ascent, ...)
        # from the old FontDescriptor when available, but override FontFile2
        # and CIDSet with our new streams. The new BaseFont/FontName is the
        # replacement font's PostScript name (from prepare_truetype_font).
        #
        # CIDSystemInfo is forced to Adobe/Identity/0 (literal) because the
        # source content stream uses /Identity-H and we must preserve CID
        # addressing.
        # ------------------------------------------------------------------
        old_descendants = font_dict.get("/DescendantFonts")
        old_descendant = old_descendants[0]
        old_descriptor = old_descendant.get("/FontDescriptor") or Dictionary()

        subset_bytes = prepared.font_bytes
        new_ps_name = prepared.postscript_name

        fontfile2_stream = pdf.make_stream(subset_bytes)
        # FontFile2 requires /Length1 = uncompressed length of the TrueType
        # program. Without it, some validators (and veraPDF) reject the font.
        fontfile2_stream["/Length1"] = len(subset_bytes)

        cidtogidmap_stream = pdf.make_stream(bytes(cid_to_gid_bytes))
        cidset_stream = pdf.make_stream(bytes(cidset_bits))
        tounicode_stream = pdf.make_stream(tounicode_bytes)

        new_descriptor = Dictionary(
            Type=Name("/FontDescriptor"),
            FontName=Name(f"/{new_ps_name}"),
            Flags=int(old_descriptor.get("/Flags", 4)),
            FontBBox=old_descriptor.get("/FontBBox", [-100, -200, 1100, 900]),
            ItalicAngle=int(old_descriptor.get("/ItalicAngle", 0)),
            Ascent=int(old_descriptor.get("/Ascent", 900)),
            Descent=int(old_descriptor.get("/Descent", -200)),
            CapHeight=int(old_descriptor.get("/CapHeight", 700)),
            StemV=int(old_descriptor.get("/StemV", 100)),
            FontFile2=fontfile2_stream,
            CIDSet=cidset_stream,
        )

        new_descendant = Dictionary(
            Type=Name("/Font"),
            Subtype=Name("/CIDFontType2"),
            BaseFont=Name(f"/{new_ps_name}"),
            CIDSystemInfo=Dictionary(
                Registry="Adobe",
                Ordering="Identity",
                Supplement=0,
            ),
            CIDToGIDMap=cidtogidmap_stream,
            DW=1000,
            W=w_array,
            FontDescriptor=new_descriptor,
        )

        # DescendantFonts is emitted as a direct 1-element array containing
        # a direct dict. PDF allows both indirect and direct descendants;
        # keeping it direct avoids creating a second indirect object and
        # matches the idiom used by minimal test fixtures and many
        # real-world encoders.
        new_type0 = Dictionary(
            Type=Name("/Font"),
            Subtype=Name("/Type0"),
            BaseFont=Name(f"/{new_ps_name}"),
            Encoding=Name("/Identity-H"),
            ToUnicode=tounicode_stream,
            DescendantFonts=pikepdf.Array([new_descendant]),
        )

        # ------------------------------------------------------------------
        # Step 9: emplace() the new Type0 dict onto the source font object.
        #
        # emplace() preserves the indirect-object identity — any existing
        # reference (belt-and-braces for single-placement canary) automatically
        # sees the new contents. emplace requires both objects to share the
        # same owner, so we wrap `new_type0` in an indirect object first.
        # ------------------------------------------------------------------
        new_type0_indirect = pdf.make_indirect(new_type0)
        eligibility.font_object.emplace(new_type0_indirect)

        logger.debug(
            "CanaryReplacer steps 1-9 complete: font_key=%s, matched=%s, "
            "cids=%d, cidtogid_bytes=%d, tounicode_bytes=%d, w_runs=%d, "
            "cidset_bytes=%d",
            font_key,
            matched_ps_name,
            len(used_cids),
            len(cid_to_gid_bytes),
            len(tounicode_bytes),
            len(w_runs),
            len(cidset_bits),
        )

        return ReplacementReport(
            status="replaced",
            matched_ps_name=matched_ps_name,
            replaced_cids_count=len(used_cids),
        )
