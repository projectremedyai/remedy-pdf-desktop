"""Simple-font production routing (REMEDY-73).

Provides :func:`should_attempt_simple_font` router and :func:`run_simple_font`
execution entry point used by ``pipeline.py::_pdf_remediate_one`` as the
Tier 2.6 stage (Phase 1 encoding repair + Phase 2 simple-font replacement).

Shape mirrors ``mode_b_production.py`` exactly:
  - :func:`should_attempt_simple_font` is PURE: never opens a PDF, only
    inspects the acceptance result already in memory.
  - :func:`run_simple_font` opens + modifies the PDF but catches all
    exceptions internally and returns a structured
    :class:`SimpleFontRunResult` (never raises).

The stage runs two sub-phases sequentially:

  Phase 1 — Encoding repair for veraPDF 7.21.6-3.  Calls
  :func:`repair_encoding_on_pdf` from :mod:`simple_font_replacer`.  This is a
  low-risk metadata repair (removes ``/Encoding`` on symbolic TrueType fonts).
  Gated by ``config.pdf_remediation.simple_font_encoding_repair_enabled``.

  Phase 2 — SimpleMultiFontReplacer for veraPDF 7.21.4.1-1.  Runs the
  full 7-step subset + re-embed pipeline for ``/Type1`` and ``/TrueType``
  (non-CID) fonts whose ``/FontDescriptor`` is missing its font program.
  Gated by ``config.pdf_remediation.simple_font_replacement_enabled``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SimpleFontRunResult:
    """Result of a Tier 2.6 simple-font run (Phase 1 + Phase 2)."""

    attempted: bool
    skip_reason: str | None = None
    encoding_repair_attempted: bool = False
    fonts_encoding_repaired: int = 0
    replacement_qualified: bool = False
    fonts_total: int = 0
    fonts_replaced: int = 0
    output_valid: bool = False
    per_font_reports: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0


def should_attempt_simple_font(
    acceptance: Any,  # PDFAcceptanceResult — avoid runtime circular import
    config: Any,      # PipelineConfig
) -> tuple[bool, str]:
    """Decide whether the Tier 2.6 simple-font stage is worth attempting.

    PURE function — never opens a PDF.  Decision is driven entirely by the
    already-in-memory acceptance result.

    Trigger rules are read from
    ``config.pdf_remediation.simple_font_replacement_trigger_rules``
    (default ``('7.21.4.1-1',)``).  In addition, the presence of ``7.21.6-3``
    is always a trigger for the Phase 1 encoding-repair path — :func:`run_simple_font`
    decides what to do with each phase independently based on the matched rule.

    Returns:
        Tuple of ``(attempt, reason)``.  Reason values include:

          - ``"disabled"`` — feature flag off.
          - ``"no simple-font violations"`` — none of the trigger rules present.
          - ``"verapdf:<rule>"`` — matched rule suffix.
    """

    pdf_cfg = getattr(config, "pdf_remediation", None)
    if pdf_cfg is None:
        return (False, "disabled")

    replacement_enabled = bool(
        getattr(pdf_cfg, "simple_font_replacement_enabled", False)
    )
    encoding_repair_enabled = bool(
        getattr(pdf_cfg, "simple_font_encoding_repair_enabled", False)
    )
    if not replacement_enabled and not encoding_repair_enabled:
        return (False, "disabled")

    trigger_rules = set(
        getattr(pdf_cfg, "simple_font_replacement_trigger_rules", ()) or ()
    )
    # Phase 1 encoding repair is always keyed to 7.21.6-3 regardless of
    # the replacement trigger_rules config — they're orthogonal phases.
    encoding_repair_rule = "7.21.6-3"

    verapdf = getattr(acceptance, "verapdf_result", None)
    violations = getattr(verapdf, "violations", None) if verapdf else None
    if violations:
        for v in violations:
            rule_id = v.get("id") if isinstance(v, dict) else getattr(v, "id", None)
            rule_str = str(rule_id) if rule_id else ""

            # Phase 2 replacement triggers
            if replacement_enabled:
                for trig in trigger_rules:
                    if rule_str.endswith(trig) or rule_str == trig:
                        return (True, f"verapdf:{trig}")

            # Phase 1 encoding repair trigger
            if encoding_repair_enabled and (
                rule_str.endswith(encoding_repair_rule)
                or rule_str == encoding_repair_rule
            ):
                return (True, f"verapdf:{encoding_repair_rule}")

    return (False, "no simple-font violations")


def run_simple_font(
    pdf_path: Path,
    output_path: Path,
    original_path: Path,
    config: Any,
) -> SimpleFontRunResult:
    """Execute Tier 2.6 (Phase 1 encoding repair + Phase 2 replacement).

    NEVER RAISES.  All exceptions are caught and recorded in
    :attr:`SimpleFontRunResult.error`.  The caller decides whether to post-
    validate via ``evaluate_pdf_acceptance``.

    Flow:

      1. Open ``pdf_path``.
      2. Phase 1 — If ``config.pdf_remediation.simple_font_encoding_repair_enabled``,
         call :func:`repair_encoding_on_pdf` to strip ``/Encoding`` on symbolic
         TrueType fonts (7.21.6-3).  Always runs first because it is the
         lower-risk metadata edit.
      3. Phase 2 — If ``config.pdf_remediation.simple_font_replacement_enabled``,
         call :func:`check_simple_multifont_eligibility`; if the document
         qualifies, run :meth:`SimpleMultiFontReplacer.replace_all`.
      4. Save to ``output_path`` if at least one font was repaired or replaced.
      5. Verify output opens cleanly.

    The two phases are independent — either can be disabled via config without
    affecting the other.
    """

    start = time.monotonic()
    result = SimpleFontRunResult(attempted=True)

    pdf_cfg = getattr(config, "pdf_remediation", None)
    encoding_repair_enabled = bool(
        getattr(pdf_cfg, "simple_font_encoding_repair_enabled", False)
    )
    replacement_enabled = bool(
        getattr(pdf_cfg, "simple_font_replacement_enabled", False)
    )

    try:
        import pikepdf
        from project_remedy.faithful_rebuild.simple_font_orchestrator import (
            SimpleMultiFontReplacer,
        )
        from project_remedy.faithful_rebuild.simple_font_replacer import (
            check_simple_multifont_eligibility,
            repair_encoding_on_pdf,
        )

        any_mutation = False

        with pikepdf.open(str(pdf_path)) as pdf:
            # --- Phase 1: Encoding repair (7.21.6-3) ----------------------
            if encoding_repair_enabled:
                result.encoding_repair_attempted = True
                try:
                    repair_report = repair_encoding_on_pdf(pdf)
                    result.fonts_encoding_repaired = int(
                        getattr(repair_report, "fonts_repaired", 0) or 0
                    )
                    for entry in getattr(repair_report, "per_font", []) or []:
                        result.per_font_reports.append({
                            "phase": "encoding_repair",
                            **entry,
                        })
                    if result.fonts_encoding_repaired > 0:
                        any_mutation = True
                except Exception as exc:
                    # Record Phase 1 failure but continue into Phase 2 —
                    # they are independent.
                    result.error = (
                        f"phase1 encoding repair failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    logger.warning(
                        "simple_font Phase 1 failed for %s: %s", pdf_path, exc
                    )

            # --- Phase 2: SimpleMultiFontReplacer (7.21.4.1-1) ------------
            if replacement_enabled:
                try:
                    multi = check_simple_multifont_eligibility(pdf)
                    result.fonts_total = len(multi.font_eligibilities)

                    if not multi.qualifies_document:
                        if not result.skip_reason:
                            result.skip_reason = (
                                "replacement eligibility did not qualify"
                            )
                    else:
                        result.replacement_qualified = True
                        replacer = SimpleMultiFontReplacer()
                        reports = replacer.replace_all(pdf, multi)
                        for r in reports:
                            result.per_font_reports.append({
                                "phase": "replacement",
                                "status": r.status,
                                "reason": r.reason,
                                "matched_ps_name": r.matched_ps_name,
                            })
                        result.fonts_replaced = sum(
                            1 for r in reports if r.status == "replaced"
                        )
                        if result.fonts_replaced > 0:
                            any_mutation = True
                        elif not result.skip_reason:
                            result.skip_reason = (
                                "no fonts replaced by Phase 2"
                            )
                except Exception as exc:
                    # Record Phase 2 failure; any Phase 1 mutations are still
                    # valid and should be saved.
                    phase2_err = (
                        f"phase2 replacement failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    result.error = (
                        phase2_err if result.error is None
                        else f"{result.error}; {phase2_err}"
                    )
                    logger.warning(
                        "simple_font Phase 2 failed for %s: %s", pdf_path, exc
                    )

            if not replacement_enabled and not encoding_repair_enabled:
                result.skip_reason = "both phases disabled"
                result.elapsed_seconds = time.monotonic() - start
                return result

            if not any_mutation:
                if not result.skip_reason:
                    result.skip_reason = "no fonts repaired or replaced"
                result.elapsed_seconds = time.monotonic() - start
                return result

            pdf.save(str(output_path))

        # Post-save validation — reopen cleanly.
        try:
            with pikepdf.open(str(output_path)) as _verify:
                _ = len(_verify.pages)
            result.output_valid = True
        except Exception as reopen_exc:
            result.output_valid = False
            reopen_msg = (
                f"output failed reopen: "
                f"{type(reopen_exc).__name__}: {reopen_exc}"
            )
            result.error = (
                reopen_msg if result.error is None
                else f"{result.error}; {reopen_msg}"
            )

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.warning("run_simple_font failed for %s: %s", pdf_path, exc)

    result.elapsed_seconds = time.monotonic() - start
    return result
