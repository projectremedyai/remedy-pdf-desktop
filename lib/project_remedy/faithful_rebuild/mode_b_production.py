"""Mode B production routing (REMEDY-78).

Provides should_attempt_mode_b() router and run_mode_b() execution entry
point used by pipeline.py::_pdf_remediate_one as a Tier 2.5 stage.

Both functions are pure/deterministic where possible:
  - should_attempt_mode_b() is PURE: never opens a PDF, only inspects the
    acceptance result already in memory
  - run_mode_b() opens + modifies the PDF but catches all exceptions
    internally and returns a structured ModeBRunResult (never raises)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModeBRunResult:
    """Result of a Tier 2.5 Mode B replacement attempt."""
    attempted: bool
    skip_reason: str | None = None
    eligibility_qualified: bool = False
    fonts_total: int = 0
    fonts_replaced: int = 0
    cids_recovered: int = 0
    output_valid: bool = False
    per_font_reports: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0


def should_attempt_mode_b(
    acceptance: Any,  # PDFAcceptanceResult — avoid circular import at runtime
    config: Any,      # PipelineConfig
) -> tuple[bool, str]:
    """Decide whether Mode B is worth attempting on this doc.

    PURE function — never opens a PDF. Uses the existing acceptance result
    to decide whether raw residual evidence suggests Mode B can help.

    Returns (attempt, reason). When attempt is False, reason is a short
    diagnostic like "disabled", "no font violations", "no matching rules".
    """
    # Config gate
    pdf_cfg = getattr(config, "pdf_remediation", None)
    if pdf_cfg is None or not getattr(pdf_cfg, "font_mode_b_enabled", False):
        return (False, "disabled")

    trigger_rules = set(getattr(pdf_cfg, "font_mode_b_trigger_rules", ()) or ())
    use_checker_signals = getattr(pdf_cfg, "font_mode_b_use_checker_signals", True)

    # Check acceptance.verapdf_result.violations for any matching rule
    verapdf = getattr(acceptance, "verapdf_result", None)
    violations = getattr(verapdf, "violations", None) if verapdf else None
    if violations:
        for v in violations:
            rule_id = v.get("id") if isinstance(v, dict) else getattr(v, "id", None)
            rule_str = str(rule_id) if rule_id else ""
            # veraPDF rule IDs may include the "ISO 14289-1:2014-" prefix;
            # match on suffix.
            for trig in trigger_rules:
                if rule_str.endswith(trig) or rule_str == trig:
                    return (True, f"verapdf:{trig}")

    # Optional checker signal: page-char-encoding source-font-risk
    if use_checker_signals:
        checker_failures = getattr(acceptance, "checker_failures", None) or []
        for cf in checker_failures:
            rule = getattr(cf, "rule_id", None) if not isinstance(cf, dict) else cf.get("rule_id")
            details = getattr(cf, "details", None) if not isinstance(cf, dict) else cf.get("details", [])
            if rule == "page-char-encoding" and details:
                for d in details:
                    if isinstance(d, str) and "source-font-risk" in d.lower():
                        return (True, "checker:page-char-encoding-source-font-risk")

    return (False, "no matching trigger rules in acceptance")


def run_mode_b(
    pdf_path: Path,
    output_path: Path,
    original_path: Path,
    config: Any,
) -> ModeBRunResult:
    """Execute Mode B replacement with recovery-enabled eligibility.

    NEVER RAISES. All exceptions caught and reported in ModeBRunResult.error.

    1. Open pdf_path
    2. Call check_multifont_eligibility_with_recovery() (REMEDY-77)
    3. If qualifies_document, call MultiFontReplacer.replace_all()
    4. Save to output_path if at least one font replaced
    5. Verify output opens cleanly
    6. Return telemetry-rich result

    Uses recovery-enabled eligibility by default. Strict path (REMEDY-71)
    is accessible via run_mode_b_strict() if needed for future tuning.
    """
    start = time.monotonic()
    result = ModeBRunResult(attempted=True)

    try:
        import pikepdf
        from project_remedy.faithful_rebuild.font_analysis import (
            check_multifont_eligibility_with_recovery,
        )
        from project_remedy.faithful_rebuild.multifont_replacer import MultiFontReplacer
        from project_remedy.faithful_rebuild.pua_handler import should_skip_font_for_pua

        with pikepdf.open(str(pdf_path)) as pdf:
            multi = check_multifont_eligibility_with_recovery(pdf)
            result.fonts_total = len(multi.font_eligibilities)
            result.cids_recovered = sum(
                e.recovered_cids_count for e in multi.font_eligibilities
            )

            if not multi.qualifies_document:
                result.skip_reason = "eligibility did not qualify"
                result.elapsed_seconds = time.monotonic() - start
                return result

            # Pre-replacement PUA filter (REMEDY-74): drop eligibilities whose
            # CID->Unicode map is dominated by Private Use Area codepoints or
            # whose font name looks like custom glyph naming. Replacing these
            # would yield nonsense text content. Record the skip in
            # per_font_reports so downstream telemetry can count the cause.
            pua_skipped_indices: list[int] = []
            for idx, e in enumerate(multi.font_eligibilities):
                if not e.qualifies:
                    continue
                if e.font_object is None:
                    continue
                # pikepdf: indirect objects need .get_object(); direct dicts
                # already are the dict. Use getattr() to handle both shapes.
                font_dict = (
                    e.font_object.get_object()
                    if hasattr(e.font_object, "get_object")
                    else e.font_object
                )
                skip, reason = should_skip_font_for_pua(e.cid_unicode_map, font_dict)
                if skip:
                    pua_skipped_indices.append(idx)
                    result.per_font_reports.append({
                        "status": "skipped_pua",
                        "reason": reason,
                        "matched_ps_name": None,
                    })

            # Disqualify PUA fonts so qualifies_document reflects reality.
            for idx in pua_skipped_indices:
                e = multi.font_eligibilities[idx]
                e.qualifies = False
                e.disqualifying_reasons = [
                    *e.disqualifying_reasons,
                    "PUA/custom-glyph — replacement would corrupt text",
                ]

            if not multi.qualifies_document:
                result.skip_reason = "all eligible fonts filtered by PUA handler"
                result.elapsed_seconds = time.monotonic() - start
                return result

            result.eligibility_qualified = True
            replacer = MultiFontReplacer()
            reports = replacer.replace_all(pdf, multi)
            result.per_font_reports = [
                {
                    "status": r.status,
                    "reason": r.reason,
                    "matched_ps_name": r.matched_ps_name,
                }
                for r in reports
            ]
            result.fonts_replaced = sum(
                1 for r in reports if r.status == "replaced"
            )

            if result.fonts_replaced == 0:
                result.skip_reason = "no fonts replaced"
                result.elapsed_seconds = time.monotonic() - start
                return result

            pdf.save(str(output_path))

        # Post-save validation — reopen cleanly
        try:
            with pikepdf.open(str(output_path)) as _verify:
                _ = len(_verify.pages)
            result.output_valid = True
        except Exception as reopen_exc:
            result.output_valid = False
            result.error = f"output failed reopen: {type(reopen_exc).__name__}: {reopen_exc}"

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.warning("run_mode_b failed for %s: %s", pdf_path, exc)

    result.elapsed_seconds = time.monotonic() - start
    return result
