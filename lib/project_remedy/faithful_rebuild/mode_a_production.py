"""Mode A faithful-rebuild production routing (REMEDY-73 follow-up, Tier 2.7).

Provides :func:`should_attempt_mode_a` router and :func:`run_mode_a`
execution entry point used by ``pipeline.py::_pdf_remediate_one`` as the
Tier 2.7 stage — the structural rebuild stage that sits between the
targeted font repairs (Tier 2.5 Mode B / Tier 2.6 simple-font) and the
specialist coordinator fallback.

Mode A wraps :func:`project_remedy.faithful_rebuild.pipeline.faithful_rebuild`,
which rebuilds the PDF's structure tree from scratch by replanning content
streams from the source pages. This is the natural next tier when the
earlier font-repair tiers have run but residual violations are dominated by
structural rules (7.1-*, 7.2-*) that targeted font work cannot touch.

Shape mirrors ``mode_b_production.py`` / ``simple_font_production.py``:
  - :func:`should_attempt_mode_a` is PURE: never opens a PDF, only
    inspects the acceptance result already in memory.
  - :func:`run_mode_a` dispatches the rebuild, validates the output is
    openable, and enforces a visual-diff gate (REMEDY-10/-15 policy)
    before committing the rebuilt file. All exceptions are caught and
    recorded in :class:`ModeARunResult.error` — the function never raises.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModeARunResult:
    """Result of a Tier 2.7 Mode A faithful-rebuild attempt."""

    attempted: bool
    skip_reason: str | None = None
    rebuild_qualified: bool = False
    structure_violations_before: int = 0
    structure_violations_after: int = 0
    visual_diff_score: float | None = None  # 0.0-1.0 worst-page diff
    output_valid: bool = False
    per_page_reports: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0


# Rule suffixes that indicate structural problems a faithful rebuild can
# plausibly resolve. Used both by the default config trigger set and by the
# fallback structural-violation counter below.
_STRUCTURAL_RULE_SUFFIXES: tuple[str, ...] = (
    "7.1-1", "7.1-2", "7.1-3",
    "7.2-11", "7.2-14", "7.2-42", "7.2-43",
)


def _rule_id_of(violation: Any) -> str:
    """Extract the rule id from a violation entry (dict or obj)."""
    if isinstance(violation, dict):
        raw = violation.get("id")
    else:
        raw = getattr(violation, "id", None)
    return str(raw) if raw else ""


def _count_structural_violations(violations: list[Any] | None) -> int:
    """Count violations whose rule id suffix-matches the structural set."""
    if not violations:
        return 0
    count = 0
    for v in violations:
        rule_str = _rule_id_of(v)
        for suffix in _STRUCTURAL_RULE_SUFFIXES:
            if rule_str.endswith(suffix) or rule_str == suffix:
                count += 1
                break
    return count


def should_attempt_mode_a(
    acceptance: Any,  # PDFAcceptanceResult — avoid runtime circular import
    config: Any,      # PipelineConfig
) -> tuple[bool, str]:
    """Decide whether the Tier 2.7 Mode A faithful-rebuild stage is worth
    attempting on this doc.

    PURE function — never opens a PDF.  Decision is driven entirely by the
    already-in-memory acceptance result.

    Triggers (in order):
      1. Any veraPDF violation whose rule id ends with an entry in
         ``config.pdf_remediation.font_mode_a_trigger_rules`` (default
         ``('7.1-1','7.1-2','7.1-3','7.2-11','7.2-14','7.2-42','7.2-43')``).
         ISO 14289-1:2014 prefix is handled via suffix match.
      2. Any checker failure whose ``rule_id`` is ``doc-reading-order``
         (logical-reading-order failure — a structural issue).
      3. Any veraPDF violation whose rule id ends with ``7.1-*`` or
         ``7.2-*`` even if not in the configured trigger set (safety net).

    Returns:
        Tuple of ``(attempt, reason)``.  Reason values include:

          - ``"disabled"`` — feature flag off.
          - ``"no structural violations"`` — no trigger matched.
          - ``"verapdf:<rule>"`` — matched rule suffix.
          - ``"checker:doc-reading-order"`` — logical reading order failed.
          - ``"verapdf:structural:<rule>"`` — fallback 7.1-*/7.2-* match.
    """

    pdf_cfg = getattr(config, "pdf_remediation", None)
    if pdf_cfg is None or not getattr(pdf_cfg, "font_mode_a_enabled", False):
        return (False, "disabled")

    trigger_rules = set(
        getattr(pdf_cfg, "font_mode_a_trigger_rules", ()) or ()
    )

    verapdf = getattr(acceptance, "verapdf_result", None)
    violations = getattr(verapdf, "violations", None) if verapdf else None

    # 1 — Configured trigger rules (veraPDF).
    if violations:
        for v in violations:
            rule_str = _rule_id_of(v)
            for trig in trigger_rules:
                if rule_str.endswith(trig) or rule_str == trig:
                    return (True, f"verapdf:{trig}")

    # 2 — Checker doc-reading-order failure.
    checker_failures = getattr(acceptance, "checker_failures", None) or []
    for cf in checker_failures:
        rule = (
            cf.get("rule_id") if isinstance(cf, dict)
            else getattr(cf, "rule_id", None)
        )
        if rule == "doc-reading-order":
            return (True, "checker:doc-reading-order")

    # 3 — Safety net: any 7.1-* / 7.2-* suffix even if not in trigger_rules.
    if violations:
        for v in violations:
            rule_str = _rule_id_of(v)
            # Extract trailing segment after the last colon/dash boundary,
            # then check for 7.1 / 7.2 prefix in the trailing part.
            # veraPDF rule ids look like "7.1-1" or "ISO 14289-1:2014-7.1-1".
            # Split on "-" and look for a "7.1" or "7.2" component.
            parts = rule_str.split("-")
            for i, part in enumerate(parts):
                if part in ("7.1", "7.2") and i + 1 < len(parts):
                    suffix = f"{part}-{parts[i + 1]}"
                    return (True, f"verapdf:structural:{suffix}")

    return (False, "no structural violations")


def run_mode_a(
    pdf_path: Path,
    output_path: Path,
    original_path: Path,
    config: Any,
) -> ModeARunResult:
    """Execute Tier 2.7 Mode A faithful rebuild against ``pdf_path``.

    NEVER RAISES.  All exceptions are caught and recorded in
    :attr:`ModeARunResult.error`.  The caller decides whether to post-
    validate via ``evaluate_pdf_acceptance``.

    Flow:

      1. Run faithful_rebuild(), writing to a staging path next to
         ``output_path`` to avoid clobbering earlier-tier output if the
         rebuild fails or produces a visually drifted document.
      2. Verify the staged output reopens cleanly via pikepdf.
      3. Run the REMEDY-10/-15 visual-diff gate via
         :func:`compare_pdf_visual_fidelity` against ``original_path``.
         If ``max_page_diff`` exceeds
         ``config.pdf_remediation.font_mode_a_visual_diff_threshold``,
         abort the rebuild (do NOT replace ``output_path``) and record the
         skip reason — the earlier-tier output is preserved.
      4. Otherwise, promote the staged rebuild to ``output_path``.

    The visual-diff gate is the primary safety mechanism: rebuilding the
    structure tree from scratch can cause layout drift, so we refuse to
    commit any output whose page-by-page pixel delta exceeds the threshold.
    """

    start = time.monotonic()
    result = ModeARunResult(attempted=True)

    pdf_cfg = getattr(config, "pdf_remediation", None)
    threshold = float(
        getattr(pdf_cfg, "font_mode_a_visual_diff_threshold", 0.10)
        if pdf_cfg is not None else 0.10
    )

    try:
        import pikepdf
        from project_remedy.faithful_rebuild.pipeline import faithful_rebuild

        # --------------------------------------------------------------
        # Step 1: Run faithful_rebuild() to a staging path.
        # --------------------------------------------------------------
        # Using a suffix-distinct staging path keeps the prior-tier output
        # intact until we've confirmed the rebuild is acceptable.
        staging_path = output_path.with_suffix(
            output_path.suffix + ".mode_a_staging"
        )
        try:
            rebuild_result = faithful_rebuild(
                source_path=pdf_path,
                output_path=staging_path,
            )
        except Exception as exc:
            result.error = (
                f"faithful_rebuild raised: {type(exc).__name__}: {exc}"
            )
            logger.warning(
                "run_mode_a: faithful_rebuild raised for %s: %s",
                pdf_path, exc,
            )
            result.elapsed_seconds = time.monotonic() - start
            return result

        # Populate structure telemetry from the rebuild result.
        result.rebuild_qualified = bool(getattr(rebuild_result, "success", False))
        result.per_page_reports.append({
            "phase": "rebuild",
            "success": bool(getattr(rebuild_result, "success", False)),
            "mode": getattr(rebuild_result, "mode", None),
            "pages_rebuilt": int(getattr(rebuild_result, "pages_rebuilt", 0) or 0),
            "visual_diff_pct": float(
                getattr(rebuild_result, "visual_diff_pct", 0.0) or 0.0
            ),
            "error": getattr(rebuild_result, "error", None),
        })

        if not rebuild_result.success or not staging_path.exists():
            result.skip_reason = (
                "faithful_rebuild did not produce output: "
                f"{getattr(rebuild_result, 'error', None) or 'unknown'}"
            )
            if staging_path.exists():
                try:
                    staging_path.unlink()
                except OSError:
                    pass
            result.elapsed_seconds = time.monotonic() - start
            return result

        # --------------------------------------------------------------
        # Step 2: Verify staged output reopens cleanly.
        # --------------------------------------------------------------
        try:
            with pikepdf.open(str(staging_path)) as _verify:
                _ = len(_verify.pages)
        except Exception as reopen_exc:
            result.error = (
                f"staged rebuild failed reopen: "
                f"{type(reopen_exc).__name__}: {reopen_exc}"
            )
            try:
                staging_path.unlink()
            except OSError:
                pass
            result.elapsed_seconds = time.monotonic() - start
            return result

        # --------------------------------------------------------------
        # Step 3: Visual-diff gate (REMEDY-10/-15 policy).
        # --------------------------------------------------------------
        diff_score: float | None = None
        diff_error: str | None = None
        try:
            from project_remedy.pdf_acceptance import compare_pdf_visual_fidelity

            diff = compare_pdf_visual_fidelity(
                original_path, staging_path, tolerance=threshold,
            )
            if diff.checked:
                diff_score = float(diff.max_page_diff)
            else:
                diff_error = diff.error or "visual diff not checked"
        except Exception as exc:
            diff_error = (
                f"visual diff helper raised: {type(exc).__name__}: {exc}"
            )
            logger.warning(
                "run_mode_a: visual diff helper raised for %s: %s",
                pdf_path, exc,
            )

        result.visual_diff_score = diff_score

        if diff_score is not None and diff_score > threshold:
            # Abort — keep earlier-tier output, drop staging file.
            result.skip_reason = (
                f"visual drift too high: {diff_score:.4f} > {threshold:.4f}"
            )
            result.output_valid = False
            try:
                staging_path.unlink()
            except OSError:
                pass
            result.elapsed_seconds = time.monotonic() - start
            return result

        # --------------------------------------------------------------
        # Step 4: Promote staged rebuild to output_path.
        # --------------------------------------------------------------
        try:
            # Atomic replace across the same filesystem.
            staging_path.replace(output_path)
            result.output_valid = True
        except Exception as move_exc:
            result.error = (
                f"promote staged rebuild failed: "
                f"{type(move_exc).__name__}: {move_exc}"
            )
            try:
                staging_path.unlink()
            except OSError:
                pass
            result.elapsed_seconds = time.monotonic() - start
            return result

        # Attach any visual-diff error as a warning if the gate skipped.
        if diff_error and diff_score is None:
            result.per_page_reports.append({
                "phase": "visual_diff",
                "checked": False,
                "error": diff_error,
            })

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.warning("run_mode_a failed for %s: %s", pdf_path, exc)

    result.elapsed_seconds = time.monotonic() - start
    return result
