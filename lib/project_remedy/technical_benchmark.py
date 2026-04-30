"""Schemas for architecture and automotive benchmark manifests and metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "2026-03-26"
SINGLE_PDF_RELIABLE_MODE = "single_pdf_reliable"


def _stringify_path(value: Path | None) -> str | None:
    return str(value) if value is not None else None


def _load_path(value: str | None) -> Path | None:
    return Path(value) if value else None


@dataclass
class CoverageTarget:
    """Expected content items for one technical-accessibility category."""

    expected_count: int = 0
    required_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CoverageTarget:
        data = data or {}
        return cls(
            expected_count=int(data.get("expected_count", 0)),
            required_labels=list(data.get("required_labels", [])),
        )


@dataclass
class TechnicalCoverageTargets:
    """Per-page or per-sample gold expectations used in manual review and QA."""

    legend: CoverageTarget = field(default_factory=CoverageTarget)
    callouts: CoverageTarget = field(default_factory=CoverageTarget)
    dimensions: CoverageTarget = field(default_factory=CoverageTarget)
    schedules: CoverageTarget = field(default_factory=CoverageTarget)
    components: CoverageTarget = field(default_factory=CoverageTarget)
    connectors: CoverageTarget = field(default_factory=CoverageTarget)
    formulas: CoverageTarget = field(default_factory=CoverageTarget)
    diagnostic_steps: CoverageTarget = field(default_factory=CoverageTarget)

    def to_dict(self) -> dict[str, Any]:
        return {
            "legend": self.legend.to_dict(),
            "callouts": self.callouts.to_dict(),
            "dimensions": self.dimensions.to_dict(),
            "schedules": self.schedules.to_dict(),
            "components": self.components.to_dict(),
            "connectors": self.connectors.to_dict(),
            "formulas": self.formulas.to_dict(),
            "diagnostic_steps": self.diagnostic_steps.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TechnicalCoverageTargets:
        data = data or {}
        return cls(
            legend=CoverageTarget.from_dict(data.get("legend")),
            callouts=CoverageTarget.from_dict(data.get("callouts")),
            dimensions=CoverageTarget.from_dict(data.get("dimensions")),
            schedules=CoverageTarget.from_dict(data.get("schedules")),
            components=CoverageTarget.from_dict(data.get("components")),
            connectors=CoverageTarget.from_dict(data.get("connectors")),
            formulas=CoverageTarget.from_dict(data.get("formulas")),
            diagnostic_steps=CoverageTarget.from_dict(data.get("diagnostic_steps")),
        )


@dataclass
class TechnicalPageExpectation:
    """Gold expectations for one source page in the benchmark set."""

    page_number: int
    page_role: str = "technical_source_page"
    expected_sections: list[str] = field(default_factory=list)
    expected_signals: list[str] = field(default_factory=list)
    coverage_targets: TechnicalCoverageTargets = field(default_factory=TechnicalCoverageTargets)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "page_role": self.page_role,
            "expected_sections": list(self.expected_sections),
            "expected_signals": list(self.expected_signals),
            "coverage_targets": self.coverage_targets.to_dict(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TechnicalPageExpectation:
        return cls(
            page_number=int(data["page_number"]),
            page_role=data.get("page_role", "technical_source_page"),
            expected_sections=list(data.get("expected_sections", [])),
            expected_signals=list(data.get("expected_signals", [])),
            coverage_targets=TechnicalCoverageTargets.from_dict(data.get("coverage_targets")),
            notes=list(data.get("notes", [])),
        )


@dataclass
class TechnicalBenchmarkSample:
    """One architecture/automotive benchmark representative."""

    sample_id: str
    pdf_path: Path
    campus: str
    domain: str
    family: str
    route_target: str
    page_count: int
    benchmark_pages: list[int]
    page_expectations: list[TechnicalPageExpectation] = field(default_factory=list)
    expected_signals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "pdf_path": str(self.pdf_path),
            "campus": self.campus,
            "domain": self.domain,
            "family": self.family,
            "route_target": self.route_target,
            "page_count": self.page_count,
            "benchmark_pages": list(self.benchmark_pages),
            "page_expectations": [page.to_dict() for page in self.page_expectations],
            "expected_signals": list(self.expected_signals),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TechnicalBenchmarkSample:
        return cls(
            sample_id=data["sample_id"],
            pdf_path=Path(data["pdf_path"]),
            campus=data["campus"],
            domain=data["domain"],
            family=data["family"],
            route_target=data["route_target"],
            page_count=int(data["page_count"]),
            benchmark_pages=list(data.get("benchmark_pages", [])),
            page_expectations=[
                TechnicalPageExpectation.from_dict(item)
                for item in data.get("page_expectations", [])
            ],
            expected_signals=list(data.get("expected_signals", [])),
            notes=list(data.get("notes", [])),
        )


@dataclass
class TechnicalBenchmarkManifest:
    """Manifest for architecture/automotive benchmark samples."""

    generated_at: str
    mode: str
    schema_version: str = SCHEMA_VERSION
    manifest_id: str = "architecture_automotive_v1"
    samples: list[TechnicalBenchmarkSample] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "mode": self.mode,
            "manifest_id": self.manifest_id,
            "samples": [sample.to_dict() for sample in self.samples],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TechnicalBenchmarkManifest:
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            generated_at=data["generated_at"],
            mode=data.get("mode", SINGLE_PDF_RELIABLE_MODE),
            manifest_id=data.get("manifest_id", "architecture_automotive_v1"),
            samples=[
                TechnicalBenchmarkSample.from_dict(item)
                for item in data.get("samples", [])
            ],
            notes=list(data.get("notes", [])),
        )


@dataclass
class CoverageMeasurement:
    """Observed extraction/coverage metrics for one category."""

    expected_count: int = 0
    found_count: int = 0
    matched_labels: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)

    @property
    def coverage_ratio(self) -> float:
        if self.expected_count <= 0:
            return 1.0
        return min(1.0, self.found_count / self.expected_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_count": self.expected_count,
            "found_count": self.found_count,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "matched_labels": list(self.matched_labels),
            "missing_labels": list(self.missing_labels),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CoverageMeasurement:
        data = data or {}
        return cls(
            expected_count=int(data.get("expected_count", 0)),
            found_count=int(data.get("found_count", 0)),
            matched_labels=list(data.get("matched_labels", [])),
            missing_labels=list(data.get("missing_labels", [])),
        )


@dataclass
class TechnicalCoverageMetrics:
    """Observed content-preservation metrics by category."""

    legend: CoverageMeasurement = field(default_factory=CoverageMeasurement)
    callouts: CoverageMeasurement = field(default_factory=CoverageMeasurement)
    dimensions: CoverageMeasurement = field(default_factory=CoverageMeasurement)
    schedules: CoverageMeasurement = field(default_factory=CoverageMeasurement)
    components: CoverageMeasurement = field(default_factory=CoverageMeasurement)
    connectors: CoverageMeasurement = field(default_factory=CoverageMeasurement)
    formulas: CoverageMeasurement = field(default_factory=CoverageMeasurement)
    diagnostic_steps: CoverageMeasurement = field(default_factory=CoverageMeasurement)

    def to_dict(self) -> dict[str, Any]:
        return {
            "legend": self.legend.to_dict(),
            "callouts": self.callouts.to_dict(),
            "dimensions": self.dimensions.to_dict(),
            "schedules": self.schedules.to_dict(),
            "components": self.components.to_dict(),
            "connectors": self.connectors.to_dict(),
            "formulas": self.formulas.to_dict(),
            "diagnostic_steps": self.diagnostic_steps.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TechnicalCoverageMetrics:
        data = data or {}
        return cls(
            legend=CoverageMeasurement.from_dict(data.get("legend")),
            callouts=CoverageMeasurement.from_dict(data.get("callouts")),
            dimensions=CoverageMeasurement.from_dict(data.get("dimensions")),
            schedules=CoverageMeasurement.from_dict(data.get("schedules")),
            components=CoverageMeasurement.from_dict(data.get("components")),
            connectors=CoverageMeasurement.from_dict(data.get("connectors")),
            formulas=CoverageMeasurement.from_dict(data.get("formulas")),
            diagnostic_steps=CoverageMeasurement.from_dict(data.get("diagnostic_steps")),
        )


@dataclass
class RoutingMetrics:
    """Observed routing decision vs gold decision."""

    expected_domain: str
    predicted_domain: str
    expected_family: str
    predicted_family: str
    expected_route: str
    predicted_route: str
    confidence: float
    reasons: list[str] = field(default_factory=list)

    @property
    def domain_correct(self) -> bool:
        return self.expected_domain == self.predicted_domain

    @property
    def family_correct(self) -> bool:
        return self.expected_family == self.predicted_family

    @property
    def route_correct(self) -> bool:
        return self.expected_route == self.predicted_route

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_domain": self.expected_domain,
            "predicted_domain": self.predicted_domain,
            "domain_correct": self.domain_correct,
            "expected_family": self.expected_family,
            "predicted_family": self.predicted_family,
            "family_correct": self.family_correct,
            "expected_route": self.expected_route,
            "predicted_route": self.predicted_route,
            "route_correct": self.route_correct,
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingMetrics:
        return cls(
            expected_domain=data["expected_domain"],
            predicted_domain=data["predicted_domain"],
            expected_family=data["expected_family"],
            predicted_family=data["predicted_family"],
            expected_route=data["expected_route"],
            predicted_route=data["predicted_route"],
            confidence=float(data.get("confidence", 0.0)),
            reasons=list(data.get("reasons", [])),
        )


@dataclass
class AcceptanceMetrics:
    """Observed acceptance-gate metrics for one benchmark output."""

    checker_failures: int = 0
    checker_failure_ids: list[str] = field(default_factory=list)
    screen_reader_errors: int = 0
    screen_reader_warnings: int = 0
    verapdf_checked: bool = False
    verapdf_passed: bool = False
    verapdf_count: int = 0
    verapdf_ids: list[str] = field(default_factory=list)

    @property
    def checker_clean(self) -> bool:
        return self.checker_failures == 0

    @property
    def screen_reader_clean(self) -> bool:
        return self.screen_reader_errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "checker_failures": self.checker_failures,
            "checker_failure_ids": list(self.checker_failure_ids),
            "checker_clean": self.checker_clean,
            "screen_reader_errors": self.screen_reader_errors,
            "screen_reader_warnings": self.screen_reader_warnings,
            "screen_reader_clean": self.screen_reader_clean,
            "verapdf_checked": self.verapdf_checked,
            "verapdf_passed": self.verapdf_passed,
            "verapdf_count": self.verapdf_count,
            "verapdf_ids": list(self.verapdf_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AcceptanceMetrics:
        data = data or {}
        return cls(
            checker_failures=int(data.get("checker_failures", 0)),
            checker_failure_ids=list(data.get("checker_failure_ids", [])),
            screen_reader_errors=int(data.get("screen_reader_errors", 0)),
            screen_reader_warnings=int(data.get("screen_reader_warnings", 0)),
            verapdf_checked=bool(data.get("verapdf_checked", False)),
            verapdf_passed=bool(data.get("verapdf_passed", False)),
            verapdf_count=int(data.get("verapdf_count", 0)),
            verapdf_ids=list(data.get("verapdf_ids", [])),
        )


@dataclass
class ReadingViewMetrics:
    """Observed properties of the generated single-PDF reading view."""

    original_page_retained: bool = True
    inserted_pages: int = 0
    section_presence: dict[str, bool] = field(default_factory=dict)
    bookmark_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_page_retained": self.original_page_retained,
            "inserted_pages": self.inserted_pages,
            "section_presence": dict(self.section_presence),
            "bookmark_labels": list(self.bookmark_labels),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ReadingViewMetrics:
        data = data or {}
        return cls(
            original_page_retained=bool(data.get("original_page_retained", True)),
            inserted_pages=int(data.get("inserted_pages", 0)),
            section_presence=dict(data.get("section_presence", {})),
            bookmark_labels=list(data.get("bookmark_labels", [])),
            notes=list(data.get("notes", [])),
        )


@dataclass
class TechnicalBenchmarkSampleMetrics:
    """Metrics for one benchmark sample output."""

    sample_id: str
    input_pdf_path: Path
    output_pdf_path: Path | None
    routing: RoutingMetrics
    acceptance: AcceptanceMetrics = field(default_factory=AcceptanceMetrics)
    reading_view: ReadingViewMetrics = field(default_factory=ReadingViewMetrics)
    coverage: TechnicalCoverageMetrics = field(default_factory=TechnicalCoverageMetrics)
    manual_review_required: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "input_pdf_path": str(self.input_pdf_path),
            "output_pdf_path": _stringify_path(self.output_pdf_path),
            "routing": self.routing.to_dict(),
            "acceptance": self.acceptance.to_dict(),
            "reading_view": self.reading_view.to_dict(),
            "coverage": self.coverage.to_dict(),
            "manual_review_required": self.manual_review_required,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TechnicalBenchmarkSampleMetrics:
        return cls(
            sample_id=data["sample_id"],
            input_pdf_path=Path(data["input_pdf_path"]),
            output_pdf_path=_load_path(data.get("output_pdf_path")),
            routing=RoutingMetrics.from_dict(data["routing"]),
            acceptance=AcceptanceMetrics.from_dict(data.get("acceptance")),
            reading_view=ReadingViewMetrics.from_dict(data.get("reading_view")),
            coverage=TechnicalCoverageMetrics.from_dict(data.get("coverage")),
            manual_review_required=bool(data.get("manual_review_required", False)),
            notes=list(data.get("notes", [])),
        )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _coverage_values(
    samples: list[TechnicalBenchmarkSampleMetrics],
    field_name: str,
) -> list[float]:
    values: list[float] = []
    for sample in samples:
        measurement = getattr(sample.coverage, field_name)
        if measurement.expected_count > 0:
            values.append(measurement.coverage_ratio)
    return values


@dataclass
class TechnicalBenchmarkSummary:
    """Run-level rollup for architecture/automotive benchmark metrics."""

    sample_count: int
    domain_accuracy: float
    family_accuracy: float
    route_accuracy: float
    checker_clean_rate: float
    screen_reader_clean_rate: float
    verapdf_pass_rate: float
    manual_review_rate: float
    legend_coverage_rate: float
    callout_coverage_rate: float
    dimension_coverage_rate: float
    schedule_coverage_rate: float
    component_coverage_rate: float
    connector_coverage_rate: float
    formula_coverage_rate: float
    diagnostic_step_coverage_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "domain_accuracy": round(self.domain_accuracy, 4),
            "family_accuracy": round(self.family_accuracy, 4),
            "route_accuracy": round(self.route_accuracy, 4),
            "checker_clean_rate": round(self.checker_clean_rate, 4),
            "screen_reader_clean_rate": round(self.screen_reader_clean_rate, 4),
            "verapdf_pass_rate": round(self.verapdf_pass_rate, 4),
            "manual_review_rate": round(self.manual_review_rate, 4),
            "legend_coverage_rate": round(self.legend_coverage_rate, 4),
            "callout_coverage_rate": round(self.callout_coverage_rate, 4),
            "dimension_coverage_rate": round(self.dimension_coverage_rate, 4),
            "schedule_coverage_rate": round(self.schedule_coverage_rate, 4),
            "component_coverage_rate": round(self.component_coverage_rate, 4),
            "connector_coverage_rate": round(self.connector_coverage_rate, 4),
            "formula_coverage_rate": round(self.formula_coverage_rate, 4),
            "diagnostic_step_coverage_rate": round(self.diagnostic_step_coverage_rate, 4),
        }

    @classmethod
    def from_samples(
        cls,
        samples: list[TechnicalBenchmarkSampleMetrics],
    ) -> TechnicalBenchmarkSummary:
        count = len(samples)
        if count == 0:
            return cls(
                sample_count=0,
                domain_accuracy=0.0,
                family_accuracy=0.0,
                route_accuracy=0.0,
                checker_clean_rate=0.0,
                screen_reader_clean_rate=0.0,
                verapdf_pass_rate=0.0,
                manual_review_rate=0.0,
                legend_coverage_rate=0.0,
                callout_coverage_rate=0.0,
                dimension_coverage_rate=0.0,
                schedule_coverage_rate=0.0,
                component_coverage_rate=0.0,
                connector_coverage_rate=0.0,
                formula_coverage_rate=0.0,
                diagnostic_step_coverage_rate=0.0,
            )

        verapdf_checked = [sample for sample in samples if sample.acceptance.verapdf_checked]
        return cls(
            sample_count=count,
            domain_accuracy=sum(sample.routing.domain_correct for sample in samples) / count,
            family_accuracy=sum(sample.routing.family_correct for sample in samples) / count,
            route_accuracy=sum(sample.routing.route_correct for sample in samples) / count,
            checker_clean_rate=sum(sample.acceptance.checker_clean for sample in samples) / count,
            screen_reader_clean_rate=sum(sample.acceptance.screen_reader_clean for sample in samples) / count,
            verapdf_pass_rate=(
                sum(sample.acceptance.verapdf_passed for sample in verapdf_checked) / len(verapdf_checked)
                if verapdf_checked
                else 0.0
            ),
            manual_review_rate=sum(sample.manual_review_required for sample in samples) / count,
            legend_coverage_rate=_mean(_coverage_values(samples, "legend")),
            callout_coverage_rate=_mean(_coverage_values(samples, "callouts")),
            dimension_coverage_rate=_mean(_coverage_values(samples, "dimensions")),
            schedule_coverage_rate=_mean(_coverage_values(samples, "schedules")),
            component_coverage_rate=_mean(_coverage_values(samples, "components")),
            connector_coverage_rate=_mean(_coverage_values(samples, "connectors")),
            formula_coverage_rate=_mean(_coverage_values(samples, "formulas")),
            diagnostic_step_coverage_rate=_mean(_coverage_values(samples, "diagnostic_steps")),
        )


@dataclass
class TechnicalBenchmarkRun:
    """Run artifact for architecture/automotive benchmark evaluation."""

    generated_at: str
    manifest_path: Path | None
    mode: str
    samples: list[TechnicalBenchmarkSampleMetrics]
    schema_version: str = SCHEMA_VERSION
    notes: list[str] = field(default_factory=list)

    @property
    def summary(self) -> TechnicalBenchmarkSummary:
        return TechnicalBenchmarkSummary.from_samples(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "manifest_path": _stringify_path(self.manifest_path),
            "mode": self.mode,
            "summary": self.summary.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TechnicalBenchmarkRun:
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            generated_at=data["generated_at"],
            manifest_path=_load_path(data.get("manifest_path")),
            mode=data.get("mode", SINGLE_PDF_RELIABLE_MODE),
            samples=[
                TechnicalBenchmarkSampleMetrics.from_dict(item)
                for item in data.get("samples", [])
            ],
            notes=list(data.get("notes", [])),
        )
