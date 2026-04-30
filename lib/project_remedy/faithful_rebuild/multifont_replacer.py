"""MultiFontReplacer — orchestrate Mode B replacement across N fonts per doc."""

from __future__ import annotations

import pikepdf

from project_remedy.faithful_rebuild.canary_replacer import (
    CanaryReplacer, ReplacementReport,
)
from project_remedy.faithful_rebuild.models import MultiCanaryEligibility


class MultiFontReplacer:
    """Iterate per-font eligibilities and call CanaryReplacer.replace() sequentially.

    Guards:
      - Skip if eligibility.qualifies is False
      - Skip if font_object.objgen already replaced in this call
      - Per-font failures don't stop subsequent replacements

    Returns list[ReplacementReport] in eligibility order.
    """

    def replace_all(
        self,
        pdf: pikepdf.Pdf,
        multi_eligibility: MultiCanaryEligibility,
    ) -> list[ReplacementReport]:
        replacer = CanaryReplacer()
        reports: list[ReplacementReport] = []
        seen_objgens: set[tuple] = set()

        for eligibility in multi_eligibility.font_eligibilities:
            if not eligibility.qualifies:
                reports.append(ReplacementReport(
                    status="skipped",
                    reason="eligibility did not qualify: " + "; ".join(
                        eligibility.disqualifying_reasons
                    ),
                    matched_ps_name=None,
                ))
                continue

            objgen = eligibility.font_object.objgen if eligibility.font_object else None
            if objgen is not None and objgen in seen_objgens:
                reports.append(ReplacementReport(
                    status="skipped",
                    reason=f"font object {objgen} already replaced in this call",
                    matched_ps_name=None,
                ))
                continue

            report = replacer.replace(pdf, eligibility)
            reports.append(report)
            if report.status == "replaced" and objgen is not None:
                seen_objgens.add(objgen)

        return reports
