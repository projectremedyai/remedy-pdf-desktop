"""Shared contracts for the PDF specialist coordinator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

DomainName = Literal[
    "structure",
    "table",
    "reading_order",
    "figure_semantics",
    "contrast",
    "font",
    "rebuild",
]
ScopeName = Literal["document", "page", "region", "anchor"]


@dataclass
class ViolationRef:
    """Normalized acceptance warning used for routing."""

    source: str
    rule_id: str
    description: str
    details: list[str] = field(default_factory=list)
    page: int | None = None
    fixable: bool = True
    domain: DomainName = "structure"
    severity: str = "warning"
    target_anchor_ids: list[str] = field(default_factory=list)
    target_region_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairContext:
    """Read-only context shared across specialists."""

    document_id: str
    source_path: Path
    working_path: Path
    original_path: Path | None
    page_count: int = 0
    acceptance_summary: str = ""
    violations: list[ViolationRef] = field(default_factory=list)
    route_hints: list[str] = field(default_factory=list)
    anchor_graph: dict[str, Any] = field(default_factory=dict)
    semantic_map: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["working_path"] = str(self.working_path)
        payload["original_path"] = str(self.original_path) if self.original_path else None
        return payload


@dataclass
class ProposalClaim:
    """Ownership claim for conflict resolution."""

    specialist: str
    domain: DomainName
    scope: ScopeName
    target_ids: list[str] = field(default_factory=list)
    exclusive: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairAction:
    """Bounded repair action emitted by a planner specialist."""

    specialist: str
    domain: DomainName
    action: str
    page: int | None = None
    target_region: str | None = None
    target_anchors: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0
    addresses_rules: list[str] = field(default_factory=list)

    def to_executor_dict(self) -> dict[str, Any]:
        result = dict(self.payload)
        result.update(
            {
                "action": self.action,
                "page": self.page,
                "target_region": self.target_region,
                "target_anchors": list(self.target_anchors),
                "reason": self.reason,
            }
        )
        return {k: v for k, v in result.items() if v is not None}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairProposal:
    """Planner specialist output before merge."""

    specialist: str
    domain: DomainName
    claims: list[ProposalClaim] = field(default_factory=list)
    actions: list[RepairAction] = field(default_factory=list)
    manual_review: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialist": self.specialist,
            "domain": self.domain,
            "claims": [claim.to_dict() for claim in self.claims],
            "actions": [action.to_dict() for action in self.actions],
            "manual_review": list(self.manual_review),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "model_name": self.model_name,
        }


@dataclass
class ConflictRecord:
    """Conflict produced while merging specialist proposals."""

    target_id: str
    kept_specialist: str | None
    dropped_specialists: list[str] = field(default_factory=list)
    resolution: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MergeResult:
    """Merged proposal set and its conflict bookkeeping."""

    actions: list[RepairAction] = field(default_factory=list)
    manual_review: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    dropped_actions: list[RepairAction] = field(default_factory=list)
    kept_specialists: list[str] = field(default_factory=list)

    def to_executor_plan(self) -> dict[str, Any]:
        return {
            "confidence": 1.0 if self.actions else 0.0,
            "operations": [action.to_executor_dict() for action in self.actions],
            "manual_review": list(self.manual_review),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "manual_review": list(self.manual_review),
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "dropped_actions": [action.to_dict() for action in self.dropped_actions],
            "kept_specialists": list(self.kept_specialists),
        }


@dataclass
class CoordinatorResult:
    """Outcome of a full coordinator run."""

    passed: bool
    output_path: Path
    trace_path: Path | None = None
    acceptance_summary: str = ""
    acceptance_passed: bool = False
    violation_count_before: int = 0
    violation_count_after: int = 0
    conflicts: list[ConflictRecord] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    trace_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_path": str(self.output_path),
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "acceptance_summary": self.acceptance_summary,
            "acceptance_passed": self.acceptance_passed,
            "violation_count_before": self.violation_count_before,
            "violation_count_after": self.violation_count_after,
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "trace": self.trace,
            "trace_summary": self.trace_summary,
        }
