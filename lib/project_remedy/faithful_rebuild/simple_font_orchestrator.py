"""SimpleMultiFontReplacer — orchestrate SimpleFontReplacer across N fonts.

Mirrors :class:`~project_remedy.faithful_rebuild.multifont_replacer.MultiFontReplacer`
but dispatches to :class:`~project_remedy.faithful_rebuild.simple_font_replacer.SimpleFontReplacer`
for ``/Type1`` / ``/TrueType`` (non-CID) fonts whose ``/FontDescriptor`` is
missing ``/FontFile`` / ``/FontFile2`` (veraPDF 7.21.4.1-1).

Guards:
  - Skip if ``eligibility.qualifies`` is False.
  - Skip if the font object's objgen was already replaced in this call
    (the same indirect font shared across pages has only one backing dict).
  - Per-font failures do not stop subsequent replacements.

Returns ``list[ReplacementReport]`` in eligibility order.
"""

from __future__ import annotations

import pikepdf

from project_remedy.faithful_rebuild.canary_replacer import ReplacementReport
from project_remedy.faithful_rebuild.models import MultiSimpleFontEligibility
from project_remedy.faithful_rebuild.simple_font_replacer import SimpleFontReplacer


class SimpleMultiFontReplacer:
    """Iterate per-font simple-font eligibilities and dispatch to
    :class:`SimpleFontReplacer`.

    Per-call state (``seen_objgens``) deduplicates shared indirect fonts.  PUA
    filtering is the caller's responsibility — Chunk A's eligibility already
    rejects PUA-dominated fonts, so an upstream filter layer is only needed
    when the orchestrator is called with hand-crafted eligibilities.
    """

    def __init__(self) -> None:
        # Per-call state only; no configuration needed.
        pass

    def replace_all(
        self,
        pdf: pikepdf.Pdf,
        multi: MultiSimpleFontEligibility,
    ) -> list[ReplacementReport]:
        """Run :class:`SimpleFontReplacer` against every qualifying entry.

        Args:
            pdf: The PDF whose font dicts will be mutated in place.
            multi: Aggregate of per-font eligibilities, typically produced
                by :func:`check_simple_multifont_eligibility`.

        Returns:
            One :class:`ReplacementReport` per entry in
            ``multi.font_eligibilities`` in the same order.  Non-qualifying
            entries receive ``status="skipped"`` reports.
        """
        replacer = SimpleFontReplacer()
        reports: list[ReplacementReport] = []
        seen_objgens: set[tuple] = set()

        for eligibility in multi.font_eligibilities:
            if not eligibility.qualifies:
                reports.append(
                    ReplacementReport(
                        status="skipped",
                        reason=(
                            "eligibility did not qualify: "
                            + "; ".join(eligibility.disqualifying_reasons)
                        ),
                        matched_ps_name=None,
                    )
                )
                continue

            objgen: tuple | None = None
            if eligibility.font_object is not None:
                try:
                    if getattr(eligibility.font_object, "is_indirect", False):
                        objgen = eligibility.font_object.objgen
                except Exception:
                    objgen = None

            if objgen is not None and objgen in seen_objgens:
                reports.append(
                    ReplacementReport(
                        status="skipped",
                        reason=f"font object {objgen} already replaced in this call",
                        matched_ps_name=None,
                    )
                )
                continue

            report = replacer.replace(pdf, eligibility)
            reports.append(report)
            if report.status == "replaced" and objgen is not None:
                seen_objgens.add(objgen)

        return reports


__all__ = ["SimpleMultiFontReplacer"]
