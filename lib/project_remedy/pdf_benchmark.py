"""Corpus manifest, OCR benchmark, and acceptance-sweep helpers for PDFs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from project_remedy.config import PipelineConfig
from project_remedy.ocr_escalation import (
    OCRAdapter,
    OCRBenchmarkResult,
    OCRBenchmarkSample,
    benchmark_ocr_adapters,
    local_benchmark_adapters,
)
from project_remedy.pdf_acceptance import evaluate_pdf_acceptance

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
