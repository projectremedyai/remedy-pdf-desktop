"""Corpus manifest, OCR benchmark, and acceptance-sweep helpers for PDFs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from project_remedy.config import PipelineConfig
from project_remedy.models import DocumentJob, FileType, JobStatus
from project_remedy.ocr_escalation import (
    OCRAdapter,
    OCRBenchmarkResult,
    OCRBenchmarkSample,
    benchmark_ocr_adapters,
    local_benchmark_adapters,
)
from project_remedy.pdf_acceptance import evaluate_pdf_acceptance
from project_remedy.pipeline import Pipeline
from project_remedy.token_tracker import tracker

DEFAULT_LAYOUT_QUOTAS: dict[str, int] = {
    "brochure_sidebar": 2,
    "catalog_large": 2,
    "form_checklist": 2,
    "math_stem": 2,
    "table_directory": 2,
    "schedule_grid": 2,
    "report_cover": 2,
    "map_infographic": 1,
}

DEFAULT_GEMINI_TIER_LAYOUT_ORDER: tuple[str, ...] = (
    "report_cover",
    "form_checklist",
    "map_infographic",
    "math_stem",
    "catalog_large",
    "brochure_sidebar",
    "schedule_grid",
    "table_directory",
)

HIGH_RISK_LAYOUT_CLASSES: set[str] = {
    "brochure_sidebar",
    "catalog_large",
    "form_checklist",
    "map_infographic",
    "math_stem",
}


@dataclass
class CorpusManifestEntry:
    pdf_path: Path
    campus: str
    layout_class: str
    page_count: int
    benchmark_pages: list[int] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["pdf_path"] = str(self.pdf_path)
        return data


@dataclass
class CorpusManifest:
    exports_root: Path
    generated_at: str
    samples: list[CorpusManifestEntry]

    def to_dict(self) -> dict:
        return {
            "exports_root": str(self.exports_root),
            "generated_at": self.generated_at,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass
class OCRBenchmarkRun:
    manifest_path: Path | None
    sample_count: int
    providers: list[str]
    summaries: list[OCRBenchmarkResult]


@dataclass
class AcceptanceSweepEntry:
    pdf_path: Path
    campus: str
    layout_class: str
    passed: bool
    checker_failures: int
    screen_reader_errors: int
    verapdf_checked: bool
    verapdf_passed: bool
    summary: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["pdf_path"] = str(self.pdf_path)
        return data


@dataclass
class AcceptanceSweepResult:
    manifest_path: Path | None
    document_count: int
    passed_count: int
    entries: list[AcceptanceSweepEntry]

    def to_dict(self) -> dict:
        return {
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "document_count": self.document_count,
            "passed_count": self.passed_count,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass
class GeminiTierBenchmarkSummary:
    model: str
    document_count: int
    passed_count: int
    severe_regressions: int
    escalation_count: int
    empty_output_count: int
    median_latency_seconds: float
    cost_per_document: float
    pdf_passed_count: int = 0

    @property
    def pass_rate(self) -> float:
        if self.document_count <= 0:
            return 0.0
        return self.passed_count / self.document_count


@dataclass
class GeminiTierBenchmarkDecision:
    recommended_model: str
    rationale: str
    flash_pass_rate: float
    flash_lite_pass_rate: float


@dataclass
class GeminiTierBenchmarkEntry:
    sample_key: str
    model: str
    campus: str
    layout_class: str
    source_pdf_path: Path
    manifest_pdf_path: Path
    status: str
    html_passed: bool
    pdf_passed: bool | None
    overall_passed: bool
    extracted_chars: int
    generated_chars: int
    empty_output: bool
    escalation_count: int
    latency_seconds: float
    cost_estimate: float
    thought_tokens: int
    error_message: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["source_pdf_path"] = str(self.source_pdf_path)
        data["manifest_pdf_path"] = str(self.manifest_pdf_path)
        return data


@dataclass
class GeminiTierBenchmarkRun:
    manifest_path: Path | None
    generated_at: str
    output_root: Path
    sample_count: int
    samples: list[CorpusManifestEntry]
    entries: list[GeminiTierBenchmarkEntry]
    summaries: list[GeminiTierBenchmarkSummary]
    decision: GeminiTierBenchmarkDecision

    def to_dict(self) -> dict:
        return {
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "generated_at": self.generated_at,
            "output_root": str(self.output_root),
            "sample_count": self.sample_count,
            "samples": [sample.to_dict() for sample in self.samples],
            "entries": [entry.to_dict() for entry in self.entries],
            "summaries": [asdict(summary) for summary in self.summaries],
            "decision": asdict(self.decision),
        }


def recommend_gemini_tier1_model(
    flash: GeminiTierBenchmarkSummary,
    flash_lite: GeminiTierBenchmarkSummary,
) -> GeminiTierBenchmarkDecision:
    """Choose the Tier 1 Gemini model using the remediation benchmark gate.

    Flash-Lite wins only when it has no severe regressions and its pass rate
    stays within two percentage points of Flash on the same benchmark corpus.
    """
    flash_pass_rate = flash.pass_rate
    flash_lite_pass_rate = flash_lite.pass_rate
    pass_rate_gap = flash_pass_rate - flash_lite_pass_rate

    if flash_lite.severe_regressions == 0 and pass_rate_gap <= 0.02:
        return GeminiTierBenchmarkDecision(
            recommended_model=flash_lite.model,
            rationale=(
                "Flash-Lite met the gate: no severe regressions and pass rate "
                "within two percentage points of Flash."
            ),
            flash_pass_rate=flash_pass_rate,
            flash_lite_pass_rate=flash_lite_pass_rate,
        )

    return GeminiTierBenchmarkDecision(
        recommended_model=flash.model,
        rationale=(
            "Flash remains the default because Flash-Lite either introduced "
            "a severe regression or fell more than two percentage points "
            "behind Flash on pass rate."
        ),
        flash_pass_rate=flash_pass_rate,
        flash_lite_pass_rate=flash_lite_pass_rate,
    )


def resolve_gemini_benchmark_source_path(entry: CorpusManifestEntry) -> Path:
    """Prefer the original downloaded PDF when the manifest points at remediated output."""
    remediated = entry.pdf_path
    if "remediated-pdfs" in remediated.parts:
        parent = remediated.parent.parent
        candidate = parent / "downloads" / "pdf" / remediated.name
        if candidate.exists():
            return candidate
    return remediated


def select_gemini_tier_samples(
    manifest: CorpusManifest,
    *,
    sample_count: int = 5,
    explicit_files: list[Path] | None = None,
) -> list[CorpusManifestEntry]:
    """Select a compact representative sample set for Gemini tier benchmarking."""
    explicit_files = explicit_files or []
    selected: list[CorpusManifestEntry] = []
    seen: set[Path] = set()

    for path in explicit_files:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        selected.append(
            CorpusManifestEntry(
                pdf_path=resolved,
                campus="manual",
                layout_class="explicit",
                page_count=_pdf_page_count(resolved),
                benchmark_pages=[1],
                note="explicit benchmark file",
            )
        )
        if len(selected) >= sample_count:
            return selected

    unique_samples: list[CorpusManifestEntry] = []
    for sample in manifest.samples:
        source_path = resolve_gemini_benchmark_source_path(sample).resolve()
        if source_path in seen or not source_path.exists():
            continue
        seen.add(source_path)
        unique_samples.append(
            CorpusManifestEntry(
                pdf_path=source_path,
                campus=sample.campus,
                layout_class=sample.layout_class,
                page_count=sample.page_count,
                benchmark_pages=sample.benchmark_pages,
                note=sample.note,
            )
        )

    chosen_paths = {sample.pdf_path.resolve() for sample in selected}
    for layout_class in DEFAULT_GEMINI_TIER_LAYOUT_ORDER:
        for sample in unique_samples:
            if sample.layout_class != layout_class:
                continue
            resolved = sample.pdf_path.resolve()
            if resolved in chosen_paths:
                continue
            selected.append(sample)
            chosen_paths.add(resolved)
            break
        if len(selected) >= sample_count:
            return selected

    for sample in unique_samples:
        resolved = sample.pdf_path.resolve()
        if resolved in chosen_paths:
            continue
        selected.append(sample)
        chosen_paths.add(resolved)
        if len(selected) >= sample_count:
            break

    return selected


async def _run_gemini_benchmark_sample(
    sample: CorpusManifestEntry,
    *,
    base_config: PipelineConfig,
    model: str,
    output_root: Path,
) -> GeminiTierBenchmarkEntry:
    tracker.reset()
    source_path = sample.pdf_path.resolve()
    sample_slug = _benchmark_slug(source_path.stem)
    model_slug = _benchmark_slug(model)
    run_root = output_root / model_slug / sample_slug

    api_cfg = replace(
        base_config.api,
        llm_backend="gemini",
        gemini_model=model,
        gemini_primary_thinking_level="minimal",
        gemini_batch_enabled=False,
        escalation_backend="gemini",
        escalation_model=base_config.api.escalation_model or "gemini-3.1-pro-preview",
        gemini_escalation_thinking_level="low",
        max_concurrent_calls=1,
    )
    output_cfg = replace(
        base_config.output,
        output_dir=run_root / "output",
        log_dir=run_root / "logs",
        db_path=run_root / "pipeline.db",
    )
    config = replace(base_config, api=api_cfg, output=output_cfg)

    pipeline = Pipeline(config)
    await pipeline.start()
    try:
        file_type = FileType.PDF if source_path.suffix.lower() == ".pdf" else None
        if file_type is None:
            raise ValueError(f"Unsupported benchmark file type: {source_path.suffix}")

        file_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        job = DocumentJob(
            campus_id=sample.campus,
            url=source_path.as_uri(),
            source_page_url=source_path.as_uri(),
            link_text=source_path.stem,
            link_context=f"{sample.layout_class} benchmark sample",
            file_type=file_type,
            local_path=str(source_path),
            file_hash=file_hash,
            file_size=source_path.stat().st_size,
            status=JobStatus.DOWNLOADED,
        )
        await pipeline._db.create_job(job)

        started = time.monotonic()
        await pipeline._extract_one(job)
        extracted_chars = len(job.ocr_markdown.strip())

        await pipeline._convert_one(job)
        generated_chars = len(job.generated_html.strip())

        if job.generated_html:
            await pipeline._vision_process_job(job)
            if config.pdf_remediation.use_programmatic_fixes:
                await pipeline._apply_strategy_fixes(job)
                await pipeline._apply_accessibility_strategies(job)

        html_passed = False
        if job.get_rendered_pages():
            job = await pipeline._validate_one(job)
            html_passed = job.status == JobStatus.VALIDATED

        pdf_passed: bool | None = None
        if (
            pipeline._html_to_pdf
            and job.generated_html
            and config.pdf_remediation.output_format in ("pdf", "both")
        ):
            job = await pipeline._html_to_pdf_one(job)
            if job.remediated_pdf_path:
                acceptance = evaluate_pdf_acceptance(
                    Path(job.remediated_pdf_path),
                    config=config,
                )
                pdf_passed = acceptance.passed
            else:
                pdf_passed = False

        latency_seconds = time.monotonic() - started
        thought_tokens = getattr(pipeline._llm, "total_thought_tokens", 0) + (
            getattr(pipeline._escalation_llm, "total_thought_tokens", 0)
            if getattr(pipeline, "_escalation_llm", None) is not None
            else 0
        )
        empty_output = extracted_chars == 0 or generated_chars == 0
        overall_passed = html_passed

        return GeminiTierBenchmarkEntry(
            sample_key=str(source_path),
            model=model,
            campus=sample.campus,
            layout_class=sample.layout_class,
            source_pdf_path=source_path,
            manifest_pdf_path=source_path,
            status=job.status.value,
            html_passed=html_passed,
            pdf_passed=pdf_passed,
            overall_passed=overall_passed,
            extracted_chars=extracted_chars,
            generated_chars=generated_chars,
            empty_output=empty_output,
            escalation_count=getattr(pipeline, "_escalation_attempts", 0),
            latency_seconds=latency_seconds,
            cost_estimate=pipeline._estimate_api_cost(),
            thought_tokens=thought_tokens,
            error_message=job.error_message,
        )
    finally:
        await pipeline.close()


async def _run_gemini_tier_benchmark_async(
    manifest: CorpusManifest,
    *,
    config: PipelineConfig,
    manifest_path: Path | None = None,
    output_root: Path,
    sample_count: int = 5,
    explicit_files: list[Path] | None = None,
    flash_model: str = "gemini-3-flash-preview",
    flash_lite_model: str = "gemini-3.1-flash-lite-preview",
) -> GeminiTierBenchmarkRun:
    samples = select_gemini_tier_samples(
        manifest,
        sample_count=sample_count,
        explicit_files=explicit_files,
    )
    entries: list[GeminiTierBenchmarkEntry] = []
    for model in (flash_model, flash_lite_model):
        for sample in samples:
            entries.append(
                await _run_gemini_benchmark_sample(
                    sample,
                    base_config=config,
                    model=model,
                    output_root=output_root,
                )
            )

    flash_entries = [entry for entry in entries if entry.model == flash_model]
    flash_lite_entries = [entry for entry in entries if entry.model == flash_lite_model]
    flash_entry_map = {entry.sample_key: entry for entry in flash_entries}
    flash_lite_severe_regressions = sum(
        1
        for entry in flash_lite_entries
        if entry.layout_class in HIGH_RISK_LAYOUT_CLASSES
        and not entry.overall_passed
        and flash_entry_map.get(entry.sample_key)
        and flash_entry_map[entry.sample_key].overall_passed
    )

    flash_summary = _summarize_gemini_tier_entries(
        flash_entries,
        severe_regressions=0,
    )
    flash_lite_summary = _summarize_gemini_tier_entries(
        flash_lite_entries,
        severe_regressions=flash_lite_severe_regressions,
    )
    decision = recommend_gemini_tier1_model(flash_summary, flash_lite_summary)
    return GeminiTierBenchmarkRun(
        manifest_path=manifest_path,
        generated_at=datetime.now(timezone.utc).isoformat(),
        output_root=output_root,
        sample_count=len(samples),
        samples=samples,
        entries=entries,
        summaries=[flash_summary, flash_lite_summary],
        decision=decision,
    )


def run_gemini_tier_benchmark(
    manifest: CorpusManifest,
    *,
    config: PipelineConfig,
    manifest_path: Path | None = None,
    output_root: Path,
    sample_count: int = 5,
    explicit_files: list[Path] | None = None,
    flash_model: str = "gemini-3-flash-preview",
    flash_lite_model: str = "gemini-3.1-flash-lite-preview",
) -> GeminiTierBenchmarkRun:
    """Run the tiered Gemini benchmark over a small representative corpus."""
    return asyncio.run(
        _run_gemini_tier_benchmark_async(
            manifest,
            config=config,
            manifest_path=manifest_path,
            output_root=output_root,
            sample_count=sample_count,
            explicit_files=explicit_files,
            flash_model=flash_model,
            flash_lite_model=flash_lite_model,
        )
    )


def build_representative_corpus_manifest(
    exports_root: Path,
    *,
    layout_quotas: dict[str, int] | None = None,
) -> CorpusManifest:
    """Build a representative page/document manifest from exported remediated PDFs."""
    quotas = layout_quotas or DEFAULT_LAYOUT_QUOTAS
    candidates = _discover_manifest_candidates(exports_root, tracked_layouts=set(quotas))
    selected: list[CorpusManifestEntry] = []
    for layout_class, wanted in quotas.items():
        pool = candidates.get(layout_class, [])
        selected.extend(_pick_representative_samples(pool, wanted))
    selected.sort(key=lambda sample: (sample.layout_class, sample.campus, sample.pdf_path.name))
    return CorpusManifest(
        exports_root=exports_root,
        generated_at=datetime.now(timezone.utc).isoformat(),
        samples=selected,
    )


def save_corpus_manifest(manifest: CorpusManifest, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return output_path


def load_corpus_manifest(path: Path) -> CorpusManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CorpusManifest(
        exports_root=Path(data["exports_root"]),
        generated_at=data["generated_at"],
        samples=[
            CorpusManifestEntry(
                pdf_path=Path(item["pdf_path"]),
                campus=item["campus"],
                layout_class=item["layout_class"],
                page_count=item["page_count"],
                benchmark_pages=list(item.get("benchmark_pages", [])),
                note=item.get("note", ""),
            )
            for item in data.get("samples", [])
        ],
    )


def run_acceptance_sweep(
    manifest: CorpusManifest,
    *,
    config: PipelineConfig | None = None,
) -> AcceptanceSweepResult:
    """Run the shared composite acceptance gate over unique manifest documents."""
    seen: set[Path] = set()
    entries: list[AcceptanceSweepEntry] = []
    for sample in manifest.samples:
        if sample.pdf_path in seen:
            continue
        seen.add(sample.pdf_path)
        acceptance = evaluate_pdf_acceptance(sample.pdf_path, config=config)
        entries.append(
            AcceptanceSweepEntry(
                pdf_path=sample.pdf_path,
                campus=sample.campus,
                layout_class=sample.layout_class,
                passed=acceptance.passed,
                checker_failures=len(acceptance.checker_failures),
                screen_reader_errors=len(acceptance.screen_reader_errors),
                verapdf_checked=acceptance.verapdf_result.checked,
                verapdf_passed=acceptance.verapdf_result.passed,
                summary=acceptance.summary(),
            )
        )
    passed_count = sum(1 for entry in entries if entry.passed)
    return AcceptanceSweepResult(
        manifest_path=None,
        document_count=len(entries),
        passed_count=passed_count,
        entries=entries,
    )


def run_ocr_benchmark(
    manifest: CorpusManifest,
    *,
    adapters: list[OCRAdapter] | None = None,
) -> OCRBenchmarkRun:
    """Benchmark OCR adapters over the representative manifest pages."""
    samples: list[OCRBenchmarkSample] = []
    for sample in manifest.samples:
        for page_number in sample.benchmark_pages or [1]:
            samples.append(
                OCRBenchmarkSample(
                    pdf_path=sample.pdf_path,
                    page_number=page_number,
                    label=sample.pdf_path.name,
                    layout_class=sample.layout_class,
                )
            )
    benchmark_adapters = adapters or local_benchmark_adapters()
    summaries = asyncio.run(benchmark_ocr_adapters(samples, benchmark_adapters))
    return OCRBenchmarkRun(
        manifest_path=None,
        sample_count=len(samples),
        providers=[adapter.name for adapter in benchmark_adapters],
        summaries=summaries,
    )


def default_manifest_path(root: Path) -> Path:
    return root / "benchmarks" / "representative_pdf_manifest.json"


def _discover_manifest_candidates(
    exports_root: Path,
    *,
    tracked_layouts: set[str],
) -> dict[str, list[CorpusManifestEntry]]:
    buckets: dict[str, list[CorpusManifestEntry]] = {key: [] for key in tracked_layouts}
    for pdf_path in sorted(exports_root.glob("*/remediated-pdfs/*.pdf")):
        campus = pdf_path.parts[-3]
        layout_class = _infer_layout_class(pdf_path)
        if layout_class not in buckets:
            continue
        page_count = _pdf_page_count(pdf_path)
        buckets[layout_class].append(
            CorpusManifestEntry(
                pdf_path=pdf_path,
                campus=campus,
                layout_class=layout_class,
                page_count=page_count,
                benchmark_pages=_benchmark_pages_for_layout(page_count, layout_class),
                note="filename heuristic",
            )
        )
    return buckets


def _infer_layout_class(pdf_path: Path) -> str:
    name = pdf_path.name.lower()
    if any(token in name for token in ("calculus-formulas", "statistics-formula", "formula", "formulas", "equation")):
        return "math_stem"
    if "catalog" in name and "addendum" not in name and "supplement" not in name:
        return "catalog_large"
    if any(token in name for token in ("form", "request", "application", "checklist")):
        return "form_checklist"
    if any(token in name for token in ("catalog", "report", "security", "annual")):
        return "report_cover"
    if any(token in name for token in ("schedule", "calendar", "class-schedule")):
        return "schedule_grid"
    if any(token in name for token in ("directory", "resource list", "table", "roster")):
        return "table_directory"
    if any(token in name for token in ("map", "infographic")):
        return "map_infographic"
    if any(token in name for token in ("brochure", "flyer", "tag", "admission", "scholarship")):
        return "brochure_sidebar"
    return "other"


def _benchmark_pages_for_layout(page_count: int, layout_class: str) -> list[int]:
    pages = [1]
    if page_count > 1 and layout_class in {
        "brochure_sidebar",
        "catalog_large",
        "math_stem",
        "report_cover",
        "schedule_grid",
        "table_directory",
        "map_infographic",
    }:
        pages.append(2)
    if layout_class == "catalog_large" and page_count > 2:
        pages.append((page_count + 1) // 2)
    return pages


def _pick_representative_samples(
    pool: list[CorpusManifestEntry],
    wanted: int,
) -> list[CorpusManifestEntry]:
    by_campus: dict[str, list[CorpusManifestEntry]] = {}
    for entry in pool:
        by_campus.setdefault(entry.campus, []).append(entry)

    selected: list[CorpusManifestEntry] = []
    while len(selected) < wanted:
        progressed = False
        for campus in sorted(by_campus):
            if not by_campus[campus]:
                continue
            selected.append(by_campus[campus].pop(0))
            progressed = True
            if len(selected) >= wanted:
                break
        if not progressed:
            break
    return selected


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import pikepdf
    except Exception:
        return 1
    try:
        with pikepdf.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 1


def _summarize_gemini_tier_entries(
    entries: list[GeminiTierBenchmarkEntry],
    *,
    severe_regressions: int,
) -> GeminiTierBenchmarkSummary:
    if not entries:
        return GeminiTierBenchmarkSummary(
            model="",
            document_count=0,
            passed_count=0,
            pdf_passed_count=0,
            severe_regressions=severe_regressions,
            escalation_count=0,
            empty_output_count=0,
            median_latency_seconds=0.0,
            cost_per_document=0.0,
        )

    return GeminiTierBenchmarkSummary(
        model=entries[0].model,
        document_count=len(entries),
        passed_count=sum(1 for entry in entries if entry.overall_passed),
        pdf_passed_count=sum(1 for entry in entries if entry.pdf_passed is True),
        severe_regressions=severe_regressions,
        escalation_count=sum(entry.escalation_count for entry in entries),
        empty_output_count=sum(1 for entry in entries if entry.empty_output),
        median_latency_seconds=median(entry.latency_seconds for entry in entries),
        cost_per_document=(
            sum(entry.cost_estimate for entry in entries) / len(entries)
            if entries
            else 0.0
        ),
    )


def _benchmark_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")[:80]
