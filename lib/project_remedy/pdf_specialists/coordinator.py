"""Specialist coordinator runtime for PDF remediation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pikepdf

from project_remedy.pdf_acceptance import PDFAcceptanceResult, evaluate_pdf_acceptance
from project_remedy.pdf_semantics import find_node_page
from project_remedy.pdf_specialists.contracts import (
    ConflictRecord,
    CoordinatorResult,
    DomainName,
    MergeResult,
    ProposalClaim,
    RepairAction,
    RepairContext,
    RepairProposal,
    ViolationRef,
)
from project_remedy.pdf_specialists.harnesses import create_harness_for_domain

logger = logging.getLogger(__name__)

DOMAIN_PRIORITY: dict[str, int] = {
    "table": 30,
    "figure_semantics": 30,
    "structure": 20,
    "reading_order": 10,
}
PLANNER_DOMAINS: tuple[DomainName, ...] = (
    "structure",
    "table",
    "reading_order",
    "figure_semantics",
)
TOOL_DOMAINS: tuple[DomainName, ...] = ("font", "contrast", "rebuild")
_PAGE_RE = re.compile(r"\bpage(?:\s*|:)\s*(\d+)\b", re.IGNORECASE)
_HARD_REBUILD_TOKENS: tuple[str, ...] = (
    "source-font",
    "cidset",
    ".notdef glyph",
    "text-map limitation",
)
ALT_HYGIENE_FALLBACK_RULE_MAP: dict[str, tuple[str, ...]] = {
    "alt-associated": ("alt-associated",),
    "alt-elements": ("alt-elements",),
    "alt-hides-annotation": ("alt-hides-annotation",),
    "alt-redundant": ("alt-redundant",),
}
FORM_ANNOTATION_FALLBACK_RULE_MAP: dict[str, tuple[str, ...]] = {
    "forms-fields-tagged": ("forms-fields-tagged", "page-annotations-tagged"),
    "page-annotations-tagged": ("page-annotations-tagged",),
    "page-link-contents": ("page-link-contents",),
    "page-annotation-contents": ("page-annotation-contents",),
    "ISO 14289-1:2014-7.18.4-2": (
        "forms-fields-tagged",
        "page-annotations-tagged",
        "role-map",
        "page-link-contents",
        "page-annotation-contents",
        "alt-associated",
        "alt-elements",
        "alt-hides-annotation",
    ),
}
STRUCTURE_FALLBACK_RULE_MAP: dict[str, tuple[str, ...]] = {
    "headings-nesting": ("headings-nesting",),
    "doc-language": ("doc-language",),
    "sr-no-lang": ("doc-language",),
    "doc-display-title": ("doc-display-title",),
    "doc-tagged": ("doc-tagged",),
    "page-tab-order": ("page-tab-order",),
    "pdfua-id": ("pdfua-id",),
    "ISO 14289-1:2014-6.2-1": ("doc-tagged",),
    "ISO 14289-1:2014-7.2-33": ("doc-language",),
    "ISO 14289-1:2014-7.2-34": ("doc-language",),
    "ISO 14289-1:2014-7.2-29": ("doc-language",),
    "ISO 14289-1:2014-7.1-10": ("doc-display-title",),
    "ISO 14289-1:2014-5-1": ("pdfua-id",),
    "ISO 14289-1:2014-7.4.2-1": ("headings-nesting",),
    # 7.1-3: "Content shall be marked as Artifact or tagged as real content."
    # Deterministic retag + content-tagging matches Vision Planner post_repair
    # behaviour (fix_page_retag + _tag_unmarked_content_streams).
    "ISO 14289-1:2014-7.1-3": ("verapdf-retag", "page-content-tagged"),
}
TABLE_FALLBACK_RULE_MAP: dict[str, tuple[str, ...]] = {
    "ISO 14289-1:2014-7.2-42": ("tables-regularity", "tables-header-scope"),
    "ISO 14289-1:2014-7.5-1": ("tables-header-scope",),
}

# Maximum number of 7.1-3 violations for the conservative deterministic
# retag fallback.  Documents with many 7.1-3 violations likely need vision-
# guided semantic planning, not blind retagging.
_RETAG_FALLBACK_MAX_VIOLATIONS = 5


def classify_violation_domain(
    source: str,
    rule_id: str,
    description: str,
    details: Iterable[str] | None = None,
) -> DomainName:
    """Map an acceptance warning to one specialist domain."""
    haystack = " ".join([rule_id, description, *list(details or [])]).lower()

    if rule_id == "doc-color-contrast" or "contrast" in haystack:
        return "contrast"
    if rule_id.startswith("alt-") or "alt text" in haystack or "alternate text" in haystack:
        return "figure_semantics"
    if "figure" in haystack and "screen reader" in haystack:
        return "figure_semantics"
    if rule_id.startswith("ISO 14289-1:2014-7.21.7") or rule_id.startswith("ISO 14289-1:2014-7.21.8"):
        return "font"
    if rule_id.startswith("ISO 14289-1:2014-7.21.4"):
        return "font"
    if any(token in haystack for token in ("tounicode", "cid", "cidset", "font", "glyph", "encoding")):
        return "font"
    if rule_id.startswith("7.2") or rule_id.startswith("7.5") or "table" in haystack:
        return "table"
    if any(token in haystack for token in ("reading order", "logical order")):
        return "reading_order"
    if rule_id in {"sr-figure-flow"}:
        return "reading_order"
    return "structure"


def normalize_acceptance_violations(
    acceptance: PDFAcceptanceResult | Any,
) -> list[ViolationRef]:
    """Normalize `PDFAcceptanceResult.warning_entries` into routing inputs."""
    warning_entries = list(getattr(acceptance, "warning_entries", []) or [])
    violations: list[ViolationRef] = []
    for entry in warning_entries:
        source = str(entry.get("source", "unknown"))
        rule_id = str(entry.get("rule_id", "unknown-rule"))
        description = str(entry.get("description", ""))
        details = [str(d) for d in entry.get("details", [])]
        page = _parse_page_number(description, details)
        violations.append(
            ViolationRef(
                source=source,
                rule_id=rule_id,
                description=description,
                details=details,
                page=page,
                fixable=bool(entry.get("fixable", True)),
                domain=classify_violation_domain(source, rule_id, description, details),
            )
        )
    return violations


def route_domains(violations: list[ViolationRef]) -> list[DomainName]:
    """Return domains in coordinator execution order."""
    present = {violation.domain for violation in violations}
    ordered: list[DomainName] = []
    for domain in ("font", "table", "figure_semantics", "structure", "reading_order", "contrast"):
        if domain in present:
            ordered.append(domain)  # type: ignore[arg-type]
    if should_attempt_rebuild(violations):
        ordered.append("rebuild")
    return ordered


def should_attempt_rebuild(violations: list[ViolationRef]) -> bool:
    """Use conservative heuristics for whole-document rebuild fallback."""
    return any(_is_hard_rebuild_violation(violation) for violation in violations)


def should_run_rebuild_phase(violations: list[ViolationRef]) -> bool:
    """Only rebuild when the remaining residual is a hard font-only blocker.

    The coordinator already has stronger evidence on conservative structure and
    figure fallback behavior than on rebuild. Restrict rebuild to the cases most
    likely to benefit: remaining non-fixable/source-font font blockers after the
    other phases have had a chance to improve the document.
    """
    font_blockers = [
        violation for violation in violations
        if _is_hard_rebuild_violation(violation)
    ]
    if not font_blockers:
        return False

    remaining_non_font_domains = {
        violation.domain
        for violation in violations
        if violation.domain != "font"
    }
    return not remaining_non_font_domains


def allow_rebuild_for_initial_domains(initial_domains: list[DomainName]) -> bool:
    """Restrict rebuild to docs that were font-only from the start.

    Mixed residual docs are the current weak area and rebuild is expensive. If
    deterministic cleanup removes the non-font issues later in the run, do not
    immediately promote the doc into a rebuild candidate in the same pass.
    """
    routed = {domain for domain in initial_domains if domain != "rebuild"}
    return routed == {"font"}


_CONFIDENCE_WORD_MAP: dict[str, float] = {
    "very high": 0.95,
    "high": 0.85,
    "medium-high": 0.75,
    "medium": 0.6,
    "moderate": 0.6,
    "medium-low": 0.4,
    "low": 0.3,
    "very low": 0.15,
    "none": 0.0,
}


def _coerce_confidence(value: Any) -> float:
    """Convert a planner's confidence output to a float in [0, 1].

    The planner prompt asks for ``float 0-1`` but LLMs frequently return
    string labels ("high", "medium", "low") or numbers-as-strings ("0.85").
    Without this coercion, ``float(value)`` throws on string labels, which
    causes the whole coordinator semantic_merge phase to error out and the
    document to fall through to ``rescued_by: none``.

    Returns 0.0 for any value that can't be interpreted.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return 0.0
        if f != f:  # NaN
            return 0.0
        return max(0.0, min(1.0, f))
    if isinstance(value, str):
        stripped = value.strip().lower()
        if not stripped:
            return 0.0
        try:
            f = float(stripped)
            return max(0.0, min(1.0, f))
        except ValueError:
            pass
        mapped = _CONFIDENCE_WORD_MAP.get(stripped)
        if mapped is not None:
            return mapped
        # Try stripping trailing '%' for percent strings
        if stripped.endswith("%"):
            try:
                f = float(stripped[:-1]) / 100.0
                return max(0.0, min(1.0, f))
            except ValueError:
                pass
    return 0.0


def _is_hard_rebuild_violation(violation: ViolationRef) -> bool:
    if violation.domain == "font" and not violation.fixable:
        return True
    description = violation.description.lower()
    details = " ".join(violation.details).lower()
    haystack = f"{description} {details}"
    return any(token in haystack for token in _HARD_REBUILD_TOKENS)


def should_promote_candidate(
    current_acceptance: PDFAcceptanceResult,
    candidate_acceptance: PDFAcceptanceResult,
    current_violations: list[ViolationRef],
    candidate_violations: list[ViolationRef],
) -> tuple[bool, str]:
    """Return whether a candidate should replace the current working PDF."""
    if not candidate_acceptance.openable:
        return False, "candidate_not_openable"

    before_count = len(current_violations)
    after_count = len(candidate_violations)

    if candidate_acceptance.passed and not current_acceptance.passed:
        return True, "candidate_passed"
    if not current_acceptance.openable and candidate_acceptance.openable:
        return True, "restored_openability"
    if after_count < before_count:
        return True, "reduced_violation_count"

    return False, "no_material_improvement"


def collect_structure_fallback_fix_ids(violations: list[ViolationRef]) -> list[str]:
    """Map structure-domain residual rules to deterministic fallback fixes.

    For ``7.1-3`` ("content not tagged as artifact or real content"), the
    fallback is conservative: it only fires when the violation count is at
    or below ``_RETAG_FALLBACK_MAX_VIOLATIONS``.  Documents with many 7.1-3
    violations likely need vision-guided semantic planning rather than blind
    deterministic retagging.
    """
    # Count 7.1-3 violations to enforce the low-count guard.
    retag_rule = "ISO 14289-1:2014-7.1-3"
    retag_count = sum(
        1 for v in violations
        if v.domain == "structure" and v.rule_id == retag_rule
    )
    retag_allowed = retag_count <= _RETAG_FALLBACK_MAX_VIOLATIONS

    fix_ids: list[str] = []
    for violation in violations:
        if violation.domain != "structure":
            continue
        if violation.rule_id == retag_rule and not retag_allowed:
            continue
        for fix_id in STRUCTURE_FALLBACK_RULE_MAP.get(violation.rule_id, ()):
            if fix_id not in fix_ids:
                fix_ids.append(fix_id)
    return fix_ids


def collect_alt_hygiene_fix_ids(violations: list[ViolationRef]) -> list[str]:
    """Map non-Figure alt hygiene residuals to deterministic cleanup fixes."""
    fix_ids: list[str] = []
    for violation in violations:
        if violation.domain != "figure_semantics":
            continue
        for fix_id in ALT_HYGIENE_FALLBACK_RULE_MAP.get(violation.rule_id, ()):
            if fix_id not in fix_ids:
                fix_ids.append(fix_id)
    return fix_ids


def collect_table_fallback_fix_ids(violations: list[ViolationRef]) -> list[str]:
    """Map table-domain residuals to deterministic table cleanup fixes."""
    fix_ids: list[str] = []
    for violation in violations:
        if violation.domain != "table":
            continue
        for fix_id in TABLE_FALLBACK_RULE_MAP.get(violation.rule_id, ()):
            if fix_id not in fix_ids:
                fix_ids.append(fix_id)
    return fix_ids


def collect_form_annotation_fix_ids(violations: list[ViolationRef]) -> list[str]:
    """Map form/annotation residuals to deterministic cleanup fixes."""
    fix_ids: list[str] = []
    for violation in violations:
        candidate_rules: tuple[str, ...] = ()
        if violation.rule_id in FORM_ANNOTATION_FALLBACK_RULE_MAP:
            candidate_rules = FORM_ANNOTATION_FALLBACK_RULE_MAP[violation.rule_id]
        elif violation.domain == "figure_semantics":
            detail_haystack = " ".join(violation.details).lower()
            if any(token in detail_haystack for token in ("/form", "widget", "/artifact", "stylespan")):
                candidate_rules = (
                    "role-map",
                    "forms-fields-tagged",
                    "page-annotations-tagged",
                    violation.rule_id,
                )

        for fix_id in candidate_rules:
            if fix_id not in fix_ids:
                fix_ids.append(fix_id)
    return fix_ids


def summarize_coordinator_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Build a stable benchmark-friendly summary from a full coordinator trace."""
    phases = list(trace.get("phases", []))
    phase_results: dict[str, dict[str, Any]] = {}
    promoted_phases: list[str] = []
    skipped_phases: list[str] = []
    tool_phase_names: list[str] = []
    total_change_entries = 0

    for phase in phases:
        phase_name = str(phase.get("phase", "unknown"))
        if phase_name in {"font_repair", "contrast_repair", "rebuild"}:
            tool_phase_names.append(phase_name)
        if phase.get("candidate_promoted"):
            promoted_phases.append(phase_name)
        if phase.get("skipped"):
            skipped_phases.append(phase_name)
        total_change_entries += len(phase.get("changes", []) or [])
        phase_results[phase_name] = {
            "promoted": bool(phase.get("candidate_promoted", False)),
            "promotion_reason": phase.get("promotion_reason", ""),
            "skipped": bool(phase.get("skipped", False)),
            "reason": phase.get("reason", ""),
            "violation_count": phase.get("violation_count"),
        }

    final_acceptance = trace.get("final_acceptance", {})
    initial_acceptance = trace.get("initial_acceptance", {})
    metrics = trace.get("metrics", {})
    semantic_execution = next(
        (phase for phase in phases if phase.get("phase") == "semantic_execution"),
        {},
    )

    return {
        "planner_client_available": bool(trace.get("planner_client_available", False)),
        "planner_client_type": trace.get("planner_client_type", ""),
        "route_hints": list(trace.get("route_hints", [])),
        "initial_violation_count": initial_acceptance.get("violation_count", 0),
        "final_violation_count": final_acceptance.get("violation_count", 0),
        "final_passed": bool(final_acceptance.get("passed", False)),
        "phase_count": len(phases),
        "promoted_phases": promoted_phases,
        "skipped_phases": skipped_phases,
        "tool_phases_run": tool_phase_names,
        "rebuild_used": "rebuild" in [str(phase.get("phase", "")) for phase in phases],
        "conflict_count": int(metrics.get("conflict_count", 0)),
        "planner_phase_executed": bool(metrics.get("planner_phase_executed", False)),
        "promoted_phase_count": int(metrics.get("promoted_phase_count", 0)),
        "semantic_operations_executed": int(semantic_execution.get("operations_executed", 0) or 0),
        "semantic_manual_review_count": int(semantic_execution.get("manual_review_count", 0) or 0),
        "total_change_entries": total_change_entries,
        "phase_results": phase_results,
    }


def merge_specialist_proposals(proposals: list[RepairProposal]) -> MergeResult:
    """Merge planner proposals using domain ownership and conflict rules."""
    result = MergeResult()
    accepted_by_target: dict[str, RepairAction] = {}

    for proposal in proposals:
        if proposal.specialist not in result.kept_specialists:
            result.kept_specialists.append(proposal.specialist)
        result.manual_review.extend(proposal.manual_review)

        for action in proposal.actions:
            targets = _exclusive_targets_for_action(action)
            if not targets or action.domain == "reading_order":
                result.actions.append(action)
                continue

            rejected = False
            for target_id in targets:
                existing = accepted_by_target.get(target_id)
                if existing is None:
                    continue

                existing_priority = DOMAIN_PRIORITY.get(existing.domain, 0)
                new_priority = DOMAIN_PRIORITY.get(action.domain, 0)

                if new_priority > existing_priority:
                    result.dropped_actions.append(existing)
                    result.actions = [candidate for candidate in result.actions if candidate is not existing]
                    accepted_by_target[target_id] = action
                    result.conflicts.append(
                        ConflictRecord(
                            target_id=target_id,
                            kept_specialist=action.specialist,
                            dropped_specialists=[existing.specialist],
                            resolution="kept_higher_priority",
                            reason=f"{action.domain} outranks {existing.domain} on exclusive target",
                        )
                    )
                    continue

                if new_priority == existing_priority:
                    result.dropped_actions.extend([existing, action])
                    result.actions = [candidate for candidate in result.actions if candidate is not existing]
                    accepted_by_target.pop(target_id, None)
                    result.conflicts.append(
                        ConflictRecord(
                            target_id=target_id,
                            kept_specialist=None,
                            dropped_specialists=[existing.specialist, action.specialist],
                            resolution="rejected_both",
                            reason="equal-priority exclusive conflict",
                        )
                    )
                    rejected = True
                    break

                result.dropped_actions.append(action)
                result.conflicts.append(
                    ConflictRecord(
                        target_id=target_id,
                        kept_specialist=existing.specialist,
                        dropped_specialists=[action.specialist],
                        resolution="kept_existing",
                        reason=f"{existing.domain} retains exclusive ownership",
                    )
                )
                rejected = True
                break

            if rejected:
                continue

            result.actions.append(action)
            for target_id in targets:
                accepted_by_target[target_id] = action

    return result


@dataclass
class PlannerSpecialist:
    """Single planner specialist definition."""

    name: str
    domain: DomainName

    async def propose(
        self,
        context: RepairContext,
        client: Any,
        model: str | None = None,
    ) -> RepairProposal:
        if not context.semantic_map or not context.anchor_graph:
            return RepairProposal(specialist=self.name, domain=self.domain)

        from project_remedy.vision_planner.planner import run_planner

        harness = create_harness_for_domain(self.domain)
        planner_result = await run_planner(
            semantic_map=context.semantic_map,
            violations=[violation.to_dict() for violation in context.violations],
            anchor_graph=context.anchor_graph,
            harness=harness,
            client=client,
            model=model,
        )
        plan = planner_result["plan"]
        actions: list[RepairAction] = []
        claims: list[ProposalClaim] = []
        for operation in plan.get("operations", []):
            action = RepairAction(
                specialist=self.name,
                domain=self.domain,
                action=str(operation.get("action", "mark_manual_review")),
                page=operation.get("page"),
                target_region=operation.get("target_region"),
                target_anchors=list(operation.get("target_anchors", []) or []),
                payload={
                    key: value
                    for key, value in operation.items()
                    if key not in {"action", "page", "target_region", "target_anchors", "reason", "op_id"}
                },
                reason=str(operation.get("reason", "")),
                confidence=_coerce_confidence(plan.get("confidence")),
                addresses_rules=[
                    violation.rule_id
                    for violation in context.violations
                    if violation.domain == self.domain
                ],
            )
            actions.append(action)
            claim_targets = _exclusive_targets_for_action(action)
            if claim_targets:
                claims.append(
                    ProposalClaim(
                        specialist=self.name,
                        domain=self.domain,
                        scope="anchor" if action.target_anchors else "region",
                        target_ids=claim_targets,
                        exclusive=self.domain != "reading_order",
                    )
                )
        return RepairProposal(
            specialist=self.name,
            domain=self.domain,
            claims=claims,
            actions=actions,
            manual_review=list(plan.get("manual_review", []) or []),
            confidence=_coerce_confidence(plan.get("confidence")),
            rationale=str(planner_result.get("planner_response", ""))[:1000],
            model_name=str(model or ""),
        )


class FontRepairTool:
    """Deterministic font and encoding repair wrapper."""

    name = "font_repair_tool"
    domain: DomainName = "font"

    def apply(self, input_path: Path, output_path: Path, *, config: Any) -> list[str]:
        from project_remedy.pdf_fixer import fix_all

        shutil.copy2(input_path, output_path)
        changes: list[str] = []
        for rule_id in ("font-tounicode", "page-char-encoding"):
            report = fix_all(output_path, output_path, only=rule_id, config=config)
            changes.extend(report.changes)
        return changes


class ContrastTool:
    """Deterministic contrast repair wrapper."""

    name = "contrast_tool"
    domain: DomainName = "contrast"

    def apply(self, input_path: Path, output_path: Path, *, config: Any) -> list[str]:
        from project_remedy.pdf_fixer import fix_all

        shutil.copy2(input_path, output_path)
        report = fix_all(output_path, output_path, only="doc-color-contrast", config=config)
        return report.changes


class RebuildTool:
    """Whole-document rebuild fallback."""

    name = "rebuild_tool"
    domain: DomainName = "rebuild"

    def apply(
        self,
        input_path: Path,
        output_path: Path,
        *,
        config: Any,
        original_path: Path | None,
    ) -> list[str]:
        from project_remedy.pdf_fixer import fix_and_verify
        from project_remedy.pdf_rebuilder import rebuild_pdf

        result = rebuild_pdf(
            source_path=input_path,
            output_path=output_path,
            visual_tolerance=config.pdf_remediation.redistill_visual_tolerance if config else 0.05,
            config=config,
        )
        if not result.success:
            raise RuntimeError(result.error or "rebuild failed")

        fix_and_verify(
            output_path,
            output_path,
            config=config,
            original_path=original_path or input_path,
            max_cycles=1,
        )
        return [
            f"rebuilt document with visual diff {result.visual_diff:.4f}",
        ]


class PDFRepairCoordinator:
    """Coordinate planner specialists and deterministic repair tools."""

    def __init__(
        self,
        *,
        config: Any,
        planner_client: Any | None = None,
        planner_model: str | None = None,
    ) -> None:
        self._config = config
        self._planner_client = planner_client
        self._planner_model = planner_model
        self._planners = [
            PlannerSpecialist("table_planner", "table"),
            PlannerSpecialist("figure_semantics_planner", "figure_semantics"),
            PlannerSpecialist("structure_planner", "structure"),
            PlannerSpecialist("reading_order_planner", "reading_order"),
        ]
        self._font_tool = FontRepairTool()
        self._contrast_tool = ContrastTool()
        self._rebuild_tool = RebuildTool()

    async def remediate(
        self,
        *,
        pdf_path: Path,
        output_path: Path,
        original_path: Path | None = None,
        trace_path: Path | None = None,
    ) -> CoordinatorResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        same_path = False
        try:
            same_path = pdf_path.resolve() == output_path.resolve()
        except FileNotFoundError:
            same_path = pdf_path == output_path

        if not same_path:
            shutil.copy2(pdf_path, output_path)
        elif not output_path.exists():
            raise FileNotFoundError(f"Coordinator input PDF not found: {output_path}")

        current_acceptance = evaluate_pdf_acceptance(
            output_path,
            original_path=original_path or pdf_path,
            config=self._config,
        )
        current_violations = normalize_acceptance_violations(current_acceptance)
        trace: dict[str, Any] = {
            "input_pdf": str(pdf_path),
            "output_pdf": str(output_path),
            "original_pdf": str(original_path) if original_path else None,
            "planner_client_available": self._planner_client is not None,
            "planner_client_type": type(self._planner_client).__name__ if self._planner_client is not None else "",
            "initial_acceptance": {
                "passed": bool(current_acceptance.passed),
                "summary": current_acceptance.summary(),
                "violation_count": len(current_violations),
            },
            "violations": [violation.to_dict() for violation in current_violations],
            "phases": [],
        }

        if current_acceptance.passed:
            return self._finalize_result(output_path, trace_path, current_acceptance, current_violations, [], trace)

        domains = route_domains(current_violations)
        trace["route_hints"] = list(domains)
        allow_rebuild = allow_rebuild_for_initial_domains(domains)

        if "font" in domains:
            current_acceptance, current_violations = self._run_tool_phase(
                trace=trace,
                phase_name="font_repair",
                tool=self._font_tool,
                current_path=output_path,
                original_path=original_path or pdf_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
            )

        semantic_domains = [domain for domain in domains if domain in PLANNER_DOMAINS]
        merge_result = MergeResult()
        if semantic_domains and self._planner_client is not None:
            current_acceptance, current_violations, merge_result = await self._run_semantic_phase(
                trace=trace,
                current_path=output_path,
                original_path=original_path or pdf_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
                semantic_domains=semantic_domains,
            )
        elif semantic_domains:
            trace["phases"].append(
                {
                    "phase": "semantic_merge",
                    "candidate_promoted": False,
                    "skipped": True,
                    "reason": "planner_client_unavailable",
                    "requested_domains": list(semantic_domains),
                }
            )

        if not current_acceptance.passed:
            current_acceptance, current_violations = self._run_form_annotation_fallback(
                trace=trace,
                current_path=output_path,
                original_path=original_path or pdf_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
            )

        if not current_acceptance.passed and any(
            violation.domain == "structure" for violation in current_violations
        ):
            _promoted, _reason, current_acceptance, current_violations = self._run_structure_fallback(
                trace=trace,
                current_path=output_path,
                original_path=original_path or pdf_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
            )

        if not current_acceptance.passed and any(
            violation.domain == "table" for violation in current_violations
        ):
            _promoted, _reason, current_acceptance, current_violations = self._run_table_fallback(
                trace=trace,
                current_path=output_path,
                original_path=original_path or pdf_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
            )

        if not current_acceptance.passed and any(
            violation.domain == "figure_semantics" for violation in current_violations
        ):
            current_acceptance, current_violations = self._run_alt_hygiene_fallback(
                trace=trace,
                current_path=output_path,
                original_path=original_path or pdf_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
            )

        if any(violation.domain == "contrast" for violation in current_violations):
            current_acceptance, current_violations = self._run_tool_phase(
                trace=trace,
                phase_name="contrast_repair",
                tool=self._contrast_tool,
                current_path=output_path,
                original_path=original_path or pdf_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
            )

        if (
            not current_acceptance.passed
            and allow_rebuild
            and should_run_rebuild_phase(current_violations)
        ):
            current_acceptance, current_violations = self._run_rebuild_phase(
                trace=trace,
                current_path=output_path,
                original_path=original_path or pdf_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
            )

        return self._finalize_result(
            output_path,
            trace_path,
            current_acceptance,
            current_violations,
            merge_result.conflicts,
            trace,
        )

    async def _run_semantic_phase(
        self,
        *,
        trace: dict[str, Any],
        current_path: Path,
        original_path: Path,
        current_acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
        semantic_domains: list[DomainName],
    ) -> tuple[PDFAcceptanceResult, list[ViolationRef], MergeResult]:
        context = await self._build_context(current_path, original_path, current_acceptance, current_violations)
        proposals: list[RepairProposal] = []

        for planner in self._planners:
            if planner.domain not in semantic_domains:
                continue
            if planner.domain == "reading_order":
                continue
            proposals.append(
                await planner.propose(context, self._planner_client, self._planner_model)
            )

        merge_result = merge_specialist_proposals(proposals)

        if "reading_order" in semantic_domains:
            reading_context = await self._build_context(current_path, original_path, current_acceptance, current_violations)
            reading_proposal = await self._planners[-1].propose(
                reading_context,
                self._planner_client,
                self._planner_model,
            )
            merge_result.actions.extend(reading_proposal.actions)
            merge_result.manual_review.extend(reading_proposal.manual_review)
            if reading_proposal.specialist not in merge_result.kept_specialists:
                merge_result.kept_specialists.append(reading_proposal.specialist)

        trace["phases"].append({"phase": "semantic_merge", "merge": merge_result.to_dict()})

        if not merge_result.actions and "figure_semantics" in semantic_domains:
            promoted, promotion_reason, acceptance, violations = await self._run_figure_semantics_fallback(
                trace=trace,
                current_path=current_path,
                original_path=original_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
            )
            if promoted or promotion_reason != "no_figure_fallback_available":
                current_acceptance, current_violations = acceptance, violations

        if not merge_result.actions and "structure" in semantic_domains:
            promoted, promotion_reason, acceptance, violations = self._run_structure_fallback(
                trace=trace,
                current_path=current_path,
                original_path=original_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
            )
            if promoted or promotion_reason != "no_structure_fallback_available":
                current_acceptance, current_violations = acceptance, violations

        if not merge_result.actions and "table" in semantic_domains and not current_acceptance.passed:
            promoted, promotion_reason, acceptance, violations = self._run_table_fallback(
                trace=trace,
                current_path=current_path,
                original_path=original_path,
                current_acceptance=current_acceptance,
                current_violations=current_violations,
            )
            if promoted or promotion_reason != "no_table_fallback_available":
                current_acceptance, current_violations = acceptance, violations

        if not merge_result.actions:
            return current_acceptance, current_violations, merge_result

        candidate_path = current_path.with_suffix(".semantic_candidate.pdf")
        promoted, promotion_reason, acceptance, violations = self._execute_semantic_candidate(
            candidate_path=candidate_path,
            current_path=current_path,
            original_path=original_path,
            current_acceptance=current_acceptance,
            current_violations=current_violations,
            context=context,
            merge_result=merge_result,
        )
        trace["phases"].append(
            {
                "phase": "semantic_execution",
                "candidate_promoted": promoted,
                "promotion_reason": promotion_reason,
                "operations_executed": len(merge_result.actions),
                "manual_review_count": len(merge_result.manual_review),
                "conflict_count": len(merge_result.conflicts),
                "summary": acceptance.summary(),
                "violation_count": len(violations),
            }
        )
        return acceptance, violations, merge_result

    async def _run_figure_semantics_fallback(
        self,
        *,
        trace: dict[str, Any],
        current_path: Path,
        original_path: Path,
        current_acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
    ) -> tuple[bool, str, PDFAcceptanceResult, list[ViolationRef]]:
        candidate_path = current_path.with_suffix(".figure_fallback.pdf")
        try:
            changed = await self._apply_figure_semantics_fallback(
                current_path=current_path,
                candidate_path=candidate_path,
                original_path=original_path,
            )
        except Exception as exc:
            trace["phases"].append(
                {
                    "phase": "figure_semantics_fallback",
                    "candidate_promoted": False,
                    "reason": str(exc),
                    "changed_figures": 0,
                }
            )
            candidate_path.unlink(missing_ok=True)
            return False, str(exc), current_acceptance, current_violations

        if not changed:
            candidate_path.unlink(missing_ok=True)
            trace["phases"].append(
                {
                    "phase": "figure_semantics_fallback",
                    "candidate_promoted": False,
                    "promotion_reason": "no_figure_fallback_available",
                    "changed_figures": 0,
                }
            )
            return False, "no_figure_fallback_available", current_acceptance, current_violations

        promoted, promotion_reason, acceptance, violations = self._promote_if_improved(
            candidate_path=candidate_path,
            current_path=current_path,
            original_path=original_path,
            current_acceptance=current_acceptance,
            current_violations=current_violations,
        )
        trace["phases"].append(
            {
                "phase": "figure_semantics_fallback",
                "candidate_promoted": promoted,
                "promotion_reason": promotion_reason,
                "changed_figures": changed,
                "summary": acceptance.summary(),
                "violation_count": len(violations),
            }
        )
        return promoted, promotion_reason, acceptance, violations

    def _run_structure_fallback(
        self,
        *,
        trace: dict[str, Any],
        current_path: Path,
        original_path: Path,
        current_acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
    ) -> tuple[bool, str, PDFAcceptanceResult, list[ViolationRef]]:
        from project_remedy.pdf_fixer import fix_all

        fix_ids = collect_structure_fallback_fix_ids(current_violations)
        if not fix_ids:
            trace["phases"].append(
                {
                    "phase": "structure_fallback",
                    "candidate_promoted": False,
                    "promotion_reason": "no_structure_fallback_available",
                    "fix_ids": [],
                }
            )
            return False, "no_structure_fallback_available", current_acceptance, current_violations

        candidate_path = current_path.with_suffix(".structure_fallback.pdf")
        shutil.copy2(current_path, candidate_path)
        changes: list[str] = []
        try:
            for fix_id in fix_ids:
                report = fix_all(candidate_path, candidate_path, only=fix_id, config=self._config)
                changes.extend(report.changes)
        except Exception as exc:
            trace["phases"].append(
                {
                    "phase": "structure_fallback",
                    "candidate_promoted": False,
                    "reason": str(exc),
                    "fix_ids": fix_ids,
                }
            )
            candidate_path.unlink(missing_ok=True)
            return False, str(exc), current_acceptance, current_violations

        promoted, promotion_reason, acceptance, violations = self._promote_if_improved(
            candidate_path=candidate_path,
            current_path=current_path,
            original_path=original_path,
            current_acceptance=current_acceptance,
            current_violations=current_violations,
        )
        trace["phases"].append(
            {
                "phase": "structure_fallback",
                "candidate_promoted": promoted,
                "promotion_reason": promotion_reason,
                "fix_ids": fix_ids,
                "changes": changes,
                "summary": acceptance.summary(),
                "violation_count": len(violations),
            }
        )
        return promoted, promotion_reason, acceptance, violations

    def _run_alt_hygiene_fallback(
        self,
        *,
        trace: dict[str, Any],
        current_path: Path,
        original_path: Path,
        current_acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
    ) -> tuple[PDFAcceptanceResult, list[ViolationRef]]:
        from project_remedy.pdf_fixer import fix_all

        fix_ids = collect_alt_hygiene_fix_ids(current_violations)
        if not fix_ids:
            trace["phases"].append(
                {
                    "phase": "alt_hygiene_fallback",
                    "candidate_promoted": False,
                    "promotion_reason": "no_alt_hygiene_fallback_available",
                    "fix_ids": [],
                }
            )
            return current_acceptance, current_violations

        candidate_path = current_path.with_suffix(".alt_hygiene_fallback.pdf")
        shutil.copy2(current_path, candidate_path)
        changes: list[str] = []
        try:
            for fix_id in fix_ids:
                report = fix_all(candidate_path, candidate_path, only=fix_id, config=self._config)
                changes.extend(report.changes)
        except Exception as exc:
            trace["phases"].append(
                {
                    "phase": "alt_hygiene_fallback",
                    "candidate_promoted": False,
                    "reason": str(exc),
                    "fix_ids": fix_ids,
                }
            )
            candidate_path.unlink(missing_ok=True)
            return current_acceptance, current_violations

        promoted, promotion_reason, acceptance, violations = self._promote_if_improved(
            candidate_path=candidate_path,
            current_path=current_path,
            original_path=original_path,
            current_acceptance=current_acceptance,
            current_violations=current_violations,
        )
        trace["phases"].append(
            {
                "phase": "alt_hygiene_fallback",
                "candidate_promoted": promoted,
                "promotion_reason": promotion_reason,
                "fix_ids": fix_ids,
                "changes": changes,
                "summary": acceptance.summary(),
                "violation_count": len(violations),
            }
        )
        return acceptance, violations

    def _run_table_fallback(
        self,
        *,
        trace: dict[str, Any],
        current_path: Path,
        original_path: Path,
        current_acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
    ) -> tuple[bool, str, PDFAcceptanceResult, list[ViolationRef]]:
        from project_remedy.pdf_fixer import fix_all

        fix_ids = collect_table_fallback_fix_ids(current_violations)
        if not fix_ids:
            trace["phases"].append(
                {
                    "phase": "table_fallback",
                    "candidate_promoted": False,
                    "promotion_reason": "no_table_fallback_available",
                    "fix_ids": [],
                }
            )
            return False, "no_table_fallback_available", current_acceptance, current_violations

        candidate_path = current_path.with_suffix(".table_fallback.pdf")
        shutil.copy2(current_path, candidate_path)
        changes: list[str] = []
        try:
            for fix_id in fix_ids:
                report = fix_all(candidate_path, candidate_path, only=fix_id, config=self._config)
                changes.extend(report.changes)
        except Exception as exc:
            trace["phases"].append(
                {
                    "phase": "table_fallback",
                    "candidate_promoted": False,
                    "reason": str(exc),
                    "fix_ids": fix_ids,
                }
            )
            candidate_path.unlink(missing_ok=True)
            return False, str(exc), current_acceptance, current_violations

        promoted, promotion_reason, acceptance, violations = self._promote_if_improved(
            candidate_path=candidate_path,
            current_path=current_path,
            original_path=original_path,
            current_acceptance=current_acceptance,
            current_violations=current_violations,
        )
        trace["phases"].append(
            {
                "phase": "table_fallback",
                "candidate_promoted": promoted,
                "promotion_reason": promotion_reason,
                "fix_ids": fix_ids,
                "changes": changes,
                "summary": acceptance.summary(),
                "violation_count": len(violations),
            }
        )
        return promoted, promotion_reason, acceptance, violations

    def _run_form_annotation_fallback(
        self,
        *,
        trace: dict[str, Any],
        current_path: Path,
        original_path: Path,
        current_acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
    ) -> tuple[PDFAcceptanceResult, list[ViolationRef]]:
        from project_remedy.pdf_fixer import fix_all

        fix_ids = collect_form_annotation_fix_ids(current_violations)
        if not fix_ids:
            trace["phases"].append(
                {
                    "phase": "form_annotation_fallback",
                    "candidate_promoted": False,
                    "promotion_reason": "no_form_annotation_fallback_available",
                    "fix_ids": [],
                }
            )
            return current_acceptance, current_violations

        candidate_path = current_path.with_suffix(".form_annotation_fallback.pdf")
        shutil.copy2(current_path, candidate_path)
        changes: list[str] = []
        try:
            for fix_id in fix_ids:
                report = fix_all(candidate_path, candidate_path, only=fix_id, config=self._config)
                changes.extend(report.changes)
        except Exception as exc:
            trace["phases"].append(
                {
                    "phase": "form_annotation_fallback",
                    "candidate_promoted": False,
                    "reason": str(exc),
                    "fix_ids": fix_ids,
                }
            )
            candidate_path.unlink(missing_ok=True)
            return current_acceptance, current_violations

        promoted, promotion_reason, acceptance, violations = self._promote_if_improved(
            candidate_path=candidate_path,
            current_path=current_path,
            original_path=original_path,
            current_acceptance=current_acceptance,
            current_violations=current_violations,
        )
        trace["phases"].append(
            {
                "phase": "form_annotation_fallback",
                "candidate_promoted": promoted,
                "promotion_reason": promotion_reason,
                "fix_ids": fix_ids,
                "changes": changes,
                "summary": acceptance.summary(),
                "violation_count": len(violations),
            }
        )
        return acceptance, violations

    async def _apply_figure_semantics_fallback(
        self,
        *,
        current_path: Path,
        candidate_path: Path,
        original_path: Path,
    ) -> int:
        from project_remedy.pdf_checker import _is_generic_alt_text, walk_structure_tree, _get_struct_type
        from project_remedy.pdf_vision import create_provider_from_config, render_page_to_image

        provider = create_provider_from_config(self._config)
        if provider is None:
            return 0

        shutil.copy2(current_path, candidate_path)

        changed = 0
        with pikepdf.open(candidate_path, allow_overwriting_input=True) as pdf:
            page_to_nodes: dict[int, list[pikepdf.Dictionary]] = {}
            for node, _depth, _parent in walk_structure_tree(pdf):
                if _get_struct_type(node) != "Figure":
                    continue
                alt = str(node.get("/Alt", "") or "").strip()
                if alt and not _is_generic_alt_text(alt):
                    continue
                page_idx = find_node_page(node, pdf)
                if page_idx is None or page_idx < 0:
                    continue
                page_to_nodes.setdefault(page_idx, []).append(node)

            for page_idx, nodes in page_to_nodes.items():
                if len(nodes) != 1:
                    continue
                image_path = None
                try:
                    image_path = render_page_to_image(current_path, page_idx + 1, dpi=150)
                    prompt = (
                        "This PDF page contains one figure whose current alt text is generic or missing. "
                        "Write a concise, specific alt text string for the primary figure only. "
                        "Do not say 'Figure'. Do not add prefixes like 'Image of'. "
                        "Return ONLY the alt text string, max 150 characters."
                    )
                    response = await provider.analyze_image(image_path, prompt, max_tokens=120)
                    alt_text = str(response or "").strip().strip('"').strip("'").strip()
                    if not alt_text or _is_generic_alt_text(alt_text):
                        continue
                    nodes[0]["/Alt"] = pikepdf.String(alt_text[:250])
                    changed += 1
                except Exception:
                    continue
                finally:
                    if image_path is not None:
                        try:
                            image_path.unlink(missing_ok=True)
                        except Exception:
                            pass

            if changed:
                pdf.save(str(candidate_path))
        return changed

    def _execute_semantic_candidate(
        self,
        *,
        candidate_path: Path,
        current_path: Path,
        original_path: Path,
        current_acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
        context: RepairContext,
        merge_result: MergeResult,
    ) -> tuple[bool, str, PDFAcceptanceResult, list[ViolationRef]]:
        from project_remedy.vision_planner.executor import execute_plan, post_repair

        exec_result = execute_plan(
            current_path,
            candidate_path,
            merge_result.to_executor_plan(),
            context.anchor_graph,
        )
        if exec_result.get("errors"):
            logger.info("semantic execution errors: %s", exec_result["errors"])
        post_repair(candidate_path)
        return self._promote_if_improved(
            candidate_path=candidate_path,
            current_path=current_path,
            original_path=original_path,
            current_acceptance=current_acceptance,
            current_violations=current_violations,
        )

    async def _build_context(
        self,
        current_path: Path,
        original_path: Path,
        acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
    ) -> RepairContext:
        from project_remedy.vision_planner.anchor_graph import build_anchor_graph
        from project_remedy.vision_planner.grounder import run_grounder
        from project_remedy.vision_planner.harness import VisionPlannerHarness

        harness = VisionPlannerHarness()
        grounder_result = await run_grounder(
            current_path,
            harness,
            self._planner_client,
            self._planner_model,
        )
        semantic_map = {"pages": grounder_result["pages"]}
        anchor_graph = build_anchor_graph(current_path)
        return RepairContext(
            document_id=hashlib.sha256(current_path.read_bytes()).hexdigest()[:16],
            source_path=original_path,
            working_path=current_path,
            original_path=original_path,
            page_count=len(semantic_map["pages"]),
            acceptance_summary=acceptance.summary(),
            violations=current_violations,
            route_hints=route_domains(current_violations),
            anchor_graph=anchor_graph,
            semantic_map=semantic_map,
            metadata={
                "grounder_prompts": grounder_result.get("grounder_prompts", []),
                "grounder_responses": grounder_result.get("grounder_responses", []),
            },
        )

    def _run_tool_phase(
        self,
        *,
        trace: dict[str, Any],
        phase_name: str,
        tool: Any,
        current_path: Path,
        original_path: Path,
        current_acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
    ) -> tuple[PDFAcceptanceResult, list[ViolationRef]]:
        candidate_path = current_path.with_suffix(f".{phase_name}.pdf")
        try:
            changes = tool.apply(current_path, candidate_path, config=self._config)
        except Exception as exc:
            trace["phases"].append(
                {"phase": phase_name, "candidate_promoted": False, "error": str(exc)}
            )
            candidate_path.unlink(missing_ok=True)
            return current_acceptance, current_violations

        promoted, promotion_reason, acceptance, violations = self._promote_if_improved(
            candidate_path=candidate_path,
            current_path=current_path,
            original_path=original_path,
            current_acceptance=current_acceptance,
            current_violations=current_violations,
        )
        trace["phases"].append(
            {
                "phase": phase_name,
                "candidate_promoted": promoted,
                "promotion_reason": promotion_reason,
                "changes": changes,
                "summary": acceptance.summary(),
                "violation_count": len(violations),
            }
        )
        return acceptance, violations

    def _run_rebuild_phase(
        self,
        *,
        trace: dict[str, Any],
        current_path: Path,
        original_path: Path,
        current_acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
    ) -> tuple[PDFAcceptanceResult, list[ViolationRef]]:
        candidate_path = current_path.with_suffix(".rebuild.pdf")
        try:
            changes = self._rebuild_tool.apply(
                current_path,
                candidate_path,
                config=self._config,
                original_path=original_path,
            )
        except Exception as exc:
            trace["phases"].append(
                {"phase": "rebuild", "candidate_promoted": False, "error": str(exc)}
            )
            candidate_path.unlink(missing_ok=True)
            return current_acceptance, current_violations

        promoted, promotion_reason, acceptance, violations = self._promote_if_improved(
            candidate_path=candidate_path,
            current_path=current_path,
            original_path=original_path,
            current_acceptance=current_acceptance,
            current_violations=current_violations,
        )
        trace["phases"].append(
            {
                "phase": "rebuild",
                "candidate_promoted": promoted,
                "promotion_reason": promotion_reason,
                "changes": changes,
                "summary": acceptance.summary(),
                "violation_count": len(violations),
            }
        )
        return acceptance, violations

    def _promote_if_improved(
        self,
        *,
        candidate_path: Path,
        current_path: Path,
        original_path: Path,
        current_acceptance: PDFAcceptanceResult,
        current_violations: list[ViolationRef],
    ) -> tuple[bool, str, PDFAcceptanceResult, list[ViolationRef]]:
        candidate_acceptance = evaluate_pdf_acceptance(
            candidate_path,
            original_path=original_path,
            config=self._config,
        )
        candidate_violations = normalize_acceptance_violations(candidate_acceptance)
        should_promote, reason = should_promote_candidate(
            current_acceptance,
            candidate_acceptance,
            current_violations,
            candidate_violations,
        )

        if should_promote:
            shutil.move(str(candidate_path), str(current_path))
            return True, reason, candidate_acceptance, candidate_violations

        candidate_path.unlink(missing_ok=True)
        return False, reason, current_acceptance, current_violations

    def _finalize_result(
        self,
        output_path: Path,
        trace_path: Path | None,
        acceptance: PDFAcceptanceResult,
        violations: list[ViolationRef],
        conflicts: list[ConflictRecord],
        trace: dict[str, Any],
    ) -> CoordinatorResult:
        trace["final_acceptance"] = {
            "passed": bool(acceptance.passed),
            "summary": acceptance.summary(),
            "violation_count": len(violations),
        }
        trace["metrics"] = {
            "phase_count": len(trace.get("phases", [])),
            "planner_phase_executed": any(
                phase.get("phase") == "semantic_execution"
                for phase in trace.get("phases", [])
            ),
            "promoted_phase_count": sum(
                1 for phase in trace.get("phases", [])
                if phase.get("candidate_promoted")
            ),
            "conflict_count": len(conflicts),
        }
        trace_summary = summarize_coordinator_trace(trace)
        trace["summary"] = trace_summary
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        return CoordinatorResult(
            passed=bool(acceptance.passed),
            output_path=output_path,
            trace_path=trace_path,
            acceptance_summary=acceptance.summary(),
            acceptance_passed=bool(acceptance.passed),
            violation_count_before=trace["initial_acceptance"]["violation_count"],
            violation_count_after=len(violations),
            conflicts=conflicts,
            trace=trace,
            trace_summary=trace_summary,
        )


async def run_specialist_pdf_repair(
    *,
    pdf_path: Path,
    output_path: Path,
    original_path: Path | None,
    config: Any,
    planner_client: Any | None = None,
    planner_model: str | None = None,
    trace_path: Path | None = None,
) -> CoordinatorResult:
    """Convenience entrypoint used by the pipeline."""
    coordinator = PDFRepairCoordinator(
        config=config,
        planner_client=planner_client,
        planner_model=planner_model,
    )
    return await coordinator.remediate(
        pdf_path=pdf_path,
        output_path=output_path,
        original_path=original_path,
        trace_path=trace_path,
    )


def _parse_page_number(description: str, details: list[str]) -> int | None:
    for chunk in [description, *details]:
        match = _PAGE_RE.search(chunk)
        if match:
            page_num = int(match.group(1))
            return max(page_num - 1, 0)
    return None


def _exclusive_targets_for_action(action: RepairAction) -> list[str]:
    if action.domain == "reading_order":
        return []
    if action.target_anchors:
        return [f"anchor:{anchor_id}" for anchor_id in action.target_anchors]
    if action.target_region:
        return [f"region:{action.target_region}"]
    if action.page is not None:
        return [f"page:{action.page}:{action.domain}"]
    return [f"action:{action.specialist}:{action.action}"]
