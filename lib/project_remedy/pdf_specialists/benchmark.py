"""Residual-failure benchmark scaffolding for coordinator comparisons."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from project_remedy.pdf_acceptance import evaluate_pdf_acceptance
from project_remedy.pdf_fixer import fix_and_verify
from project_remedy.pdf_specialists.export import (
    append_export_record,
    build_benchmark_record,
    build_coordinator_export_record,
    count_acceptance_warnings,
)

BenchmarkMode = Literal["baseline", "vision_planner", "specialist_coordinator"]
DiscoverySeedMode = Literal["full", "light"]


@dataclass
class BenchmarkManifestEntry:
    """One document entry in a frozen residual-failure manifest."""

    document_id: str
    pdf_path: str
    dominant_domain: str
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkManifest:
    """Frozen residual-failure manifest."""

    name: str
    entries: list[BenchmarkManifestEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass
class BaselineSeedResult:
    """Baseline artifact and acceptance used to seed fallback benchmarks."""

    row: dict[str, Any]
    output_path: Path
    final_acceptance: Any


@dataclass(frozen=True)
class DiscoverySeedResult:
    """Artifact path produced by discovery-only baseline seeding."""

    output_path: Path
    seed_mode: DiscoverySeedMode


@dataclass
class ResidualCandidateResult:
    """Residual screening result for one source PDF."""

    pdf_path: str
    baseline_path: str
    seed_mode: DiscoverySeedMode
    elapsed_seconds: float
    passed: bool
    acceptance_summary: str
    violation_count: int
    domain_counts: dict[str, int]
    warning_rule_ids: list[str]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_benchmark_manifest(path: Path) -> BenchmarkManifest:
    """Load a JSON manifest from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [
        BenchmarkManifestEntry(
            document_id=str(entry["document_id"]),
            pdf_path=str(entry["pdf_path"]),
            dominant_domain=str(entry["dominant_domain"]),
            tags=list(entry.get("tags", [])),
            notes=str(entry.get("notes", "")),
        )
        for entry in payload.get("entries", [])
    ]
    return BenchmarkManifest(name=str(payload.get("name", path.stem)), entries=entries)


def save_benchmark_manifest(manifest: BenchmarkManifest, path: Path) -> None:
    """Write a benchmark manifest JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def dominant_domain_from_counts(domain_counts: dict[str, int]) -> str:
    """Return the dominant residual domain for manifest stratification."""
    if not domain_counts:
        return "unknown"
    ordered = sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) == 1:
        return ordered[0][0]
    if ordered[0][1] > ordered[1][1]:
        return ordered[0][0]
    return "mixed"


def build_residual_benchmark_manifest(
    *,
    name: str,
    results: list[ResidualCandidateResult],
    max_per_domain: int | None = None,
    min_violation_count: int = 1,
) -> BenchmarkManifest:
    """Build a stratified benchmark manifest from residual screening results."""
    entries: list[BenchmarkManifestEntry] = []
    seen_ids: set[str] = set()
    seen_pdf_paths: set[str] = set()
    per_domain_counts: Counter[str] = Counter()

    eligible = sorted(
        (
            result for result in results
            if not result.passed
            and not result.error
            and result.violation_count >= min_violation_count
            and result.domain_counts
        ),
        key=lambda result: (
            dominant_domain_from_counts(result.domain_counts),
            -result.violation_count,
            result.elapsed_seconds,
            result.pdf_path,
        ),
    )

    for result in eligible:
        if result.pdf_path in seen_pdf_paths:
            continue
        domain = dominant_domain_from_counts(result.domain_counts)
        if max_per_domain is not None and max_per_domain > 0 and per_domain_counts[domain] >= max_per_domain:
            continue

        candidate_path = Path(result.pdf_path)
        document_id = _stable_manifest_document_id(candidate_path, seen_ids)
        seen_ids.add(document_id)
        seen_pdf_paths.add(result.pdf_path)
        per_domain_counts[domain] += 1
        entries.append(
            BenchmarkManifestEntry(
                document_id=document_id,
                pdf_path=result.pdf_path,
                dominant_domain=domain,
                tags=["residual", domain, f"seed:{result.seed_mode}"],
                notes=(
                    f"{result.acceptance_summary}; domains={result.domain_counts}; "
                    f"elapsed={result.elapsed_seconds:.3f}s"
                ),
            )
        )

    return BenchmarkManifest(name=name, entries=entries)


def aggregate_benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate benchmark result rows by comparison mode."""
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        mode = str(row.get("mode", "unknown"))
        by_mode.setdefault(mode, []).append(row)

    summary_by_mode: dict[str, dict[str, Any]] = {}
    for mode, mode_rows in by_mode.items():
        total = len(mode_rows)
        passed = sum(1 for row in mode_rows if row.get("passed"))
        conflict_count = sum(int(row.get("conflict_count", 0) or 0) for row in mode_rows)
        rebuild_used = sum(1 for row in mode_rows if row.get("rebuild_used"))
        planner_phase_executed = sum(1 for row in mode_rows if row.get("planner_phase_executed"))
        skipped_docs = sum(1 for row in mode_rows if row.get("skipped"))
        before_total = sum(int(row.get("violation_count_before", 0) or 0) for row in mode_rows)
        after_total = sum(int(row.get("violation_count_after", 0) or 0) for row in mode_rows)
        summary_by_mode[mode] = {
            "total_docs": total,
            "passed_docs": passed,
            "pass_rate": passed / total if total else 0.0,
            "skipped_docs": skipped_docs,
            "conflict_count": conflict_count,
            "rebuild_used_count": rebuild_used,
            "planner_phase_executed_count": planner_phase_executed,
            "violation_delta_total": before_total - after_total,
        }

    return {
        "total_rows": len(rows),
        "modes": summary_by_mode,
    }


def _fallback_promoted(
    initial_acceptance: Any,
    final_acceptance: Any,
) -> bool:
    """Return whether a fallback mode materially improved over its seed."""
    before = count_acceptance_warnings(initial_acceptance)
    after = count_acceptance_warnings(final_acceptance)
    if getattr(final_acceptance, "passed", False) and not getattr(initial_acceptance, "passed", False):
        return True
    return after < before


def compute_document_hash(path: Path) -> str:
    """Return a short stable document hash for benchmark rows."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def screen_residual_candidate(
    pdf_path: Path,
    *,
    output_dir: Path,
    config: Any,
    seed_mode: DiscoverySeedMode = "full",
) -> ResidualCandidateResult:
    """Run one PDF through a seed path and summarize the residual acceptance."""
    from project_remedy.pdf_specialists.coordinator import normalize_acceptance_violations

    output_dir.mkdir(parents=True, exist_ok=True)
    seeded_path = output_dir / pdf_path.name
    started = time.perf_counter()
    seed_result = run_discovery_seed(
        pdf_path,
        output_path=seeded_path,
        config=config,
        seed_mode=seed_mode,
    )
    elapsed_seconds = round(time.perf_counter() - started, 3)
    acceptance = evaluate_pdf_acceptance(
        seed_result.output_path,
        original_path=pdf_path,
        config=config,
    )
    violations = normalize_acceptance_violations(acceptance)
    domain_counts = Counter(v.domain for v in violations)
    return ResidualCandidateResult(
        pdf_path=str(pdf_path),
        baseline_path=str(seed_result.output_path),
        seed_mode=seed_result.seed_mode,
        elapsed_seconds=elapsed_seconds,
        passed=bool(acceptance.passed),
        acceptance_summary=acceptance.summary(),
        violation_count=len(violations),
        domain_counts=dict(sorted(domain_counts.items())),
        warning_rule_ids=[v.rule_id for v in violations],
    )


def screen_residual_candidate_with_timeout(
    pdf_path: Path,
    *,
    output_dir: Path,
    config: Any,
    timeout_seconds: float,
    seed_mode: DiscoverySeedMode = "full",
) -> ResidualCandidateResult:
    """Run residual screening in a worker process so slow PDFs can be terminated."""
    import multiprocessing

    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_screen_residual_candidate_worker,
        args=(queue, pdf_path, output_dir, config, seed_mode),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return ResidualCandidateResult(
            pdf_path=str(pdf_path),
            baseline_path="",
            seed_mode=seed_mode,
            elapsed_seconds=round(timeout_seconds, 3),
            passed=False,
            acceptance_summary=f"candidate timed out after {timeout_seconds}s",
            violation_count=0,
            domain_counts={},
            warning_rule_ids=[],
            error=f"candidate timed out after {timeout_seconds}s",
        )
    if not queue.empty():
        payload = queue.get()
        return ResidualCandidateResult(**payload)
    return ResidualCandidateResult(
        pdf_path=str(pdf_path),
        baseline_path="",
        seed_mode=seed_mode,
        elapsed_seconds=0.0,
        passed=False,
        acceptance_summary="candidate process exited without result",
        violation_count=0,
        domain_counts={},
        warning_rule_ids=[],
        error="candidate process exited without result",
    )


def run_discovery_seed(
    source_path: Path,
    *,
    output_path: Path,
    config: Any,
    seed_mode: DiscoverySeedMode = "light",
) -> DiscoverySeedResult:
    """Produce a discovery-only seed artifact for representative screening.

    ``full`` mirrors the official benchmark seed path. ``light`` is intentionally
    cheaper: it skips vision-assisted fixes, limits verification looping, and
    bypasses the visual-diff gate so candidate discovery can screen a broader
    pool without changing the official benchmark semantics.
    """
    if seed_mode == "full":
        fix_and_verify(
            source_path,
            output_path,
            config=config,
            original_path=source_path,
        )
    elif seed_mode == "light":
        fix_and_verify(
            source_path,
            output_path,
            config=None,
            max_cycles=1,
            original_path=None,
        )
    else:
        raise ValueError(f"Unsupported discovery seed mode: {seed_mode}")

    return DiscoverySeedResult(
        output_path=output_path if output_path.exists() else source_path,
        seed_mode=seed_mode,
    )


async def run_benchmark_entry(
    entry: BenchmarkManifestEntry,
    *,
    output_dir: Path,
    config: Any,
    modes: tuple[BenchmarkMode, ...] = ("baseline", "vision_planner", "specialist_coordinator"),
    mode_timeout_seconds: float | None = 300.0,
) -> list[dict[str, Any]]:
    """Run one manifest entry through the selected comparison modes."""
    source_path = Path(entry.pdf_path)
    document_hash = compute_document_hash(source_path)
    initial_acceptance = evaluate_pdf_acceptance(
        source_path,
        original_path=source_path,
        config=config,
    )
    baseline_seed: BaselineSeedResult | None = None
    baseline_seed_error = ""
    if "baseline" in modes or any(mode in modes for mode in ("vision_planner", "specialist_coordinator")):
        try:
            baseline_seed = await _run_with_timeout(
                _run_baseline_seed,
                mode_timeout_seconds,
                entry,
                source_path=source_path,
                output_dir=output_dir,
                config=config,
                document_hash=document_hash,
                initial_acceptance=initial_acceptance,
            )
        except Exception as exc:
            baseline_seed_error = (
                f"baseline seed timed out after {mode_timeout_seconds}s"
                if isinstance(exc, TimeoutError)
                else f"baseline seed failed: {exc}"
            )

    rows: list[dict[str, Any]] = []
    for mode in modes:
        try:
            if mode == "baseline":
                if baseline_seed is None:
                    raise RuntimeError(baseline_seed_error or "baseline seed unavailable")
                rows.append(baseline_seed.row)
            elif mode == "vision_planner":
                if baseline_seed is None:
                    raise RuntimeError(baseline_seed_error or "baseline seed unavailable")
                rows.append(
                    await _run_with_timeout(
                        _run_vision_planner_mode,
                        mode_timeout_seconds,
                        entry,
                        source_path=source_path,
                        seed_path=baseline_seed.output_path if baseline_seed else source_path,
                        output_dir=output_dir,
                        config=config,
                        document_hash=document_hash,
                        source_acceptance=initial_acceptance,
                        initial_acceptance=baseline_seed.final_acceptance if baseline_seed else initial_acceptance,
                    )
                )
            elif mode == "specialist_coordinator":
                if baseline_seed is None:
                    raise RuntimeError(baseline_seed_error or "baseline seed unavailable")
                rows.append(
                    await _run_with_timeout(
                        _run_specialist_coordinator_mode,
                        mode_timeout_seconds,
                        entry,
                        source_path=source_path,
                        seed_path=baseline_seed.output_path if baseline_seed else source_path,
                        output_dir=output_dir,
                        config=config,
                        document_hash=document_hash,
                        source_acceptance=initial_acceptance,
                        initial_acceptance=baseline_seed.final_acceptance if baseline_seed else initial_acceptance,
                    )
                )
            else:
                raise ValueError(f"Unsupported benchmark mode: {mode}")
        except Exception as exc:
            error_text = (
                f"mode timed out after {mode_timeout_seconds}s"
                if isinstance(exc, TimeoutError)
                else str(exc)
            )
            rows.append(
                _build_error_record(
                    mode=mode,
                    entry=entry,
                    source_path=source_path,
                    document_hash=document_hash,
                    initial_acceptance=initial_acceptance,
                    error=error_text,
                )
            )

    return rows


async def run_manifest_benchmark(
    manifest: BenchmarkManifest,
    *,
    output_dir: Path,
    config: Any,
    modes: tuple[BenchmarkMode, ...] = ("baseline", "vision_planner", "specialist_coordinator"),
    rows_path: Path | None = None,
    mode_timeout_seconds: float | None = 300.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the selected modes for every manifest entry and aggregate results."""
    rows: list[dict[str, Any]] = []
    for entry in manifest.entries:
        entry_rows = await run_benchmark_entry(
            entry,
            output_dir=output_dir,
            config=config,
            modes=modes,
            mode_timeout_seconds=mode_timeout_seconds,
        )
        rows.extend(entry_rows)
        if rows_path is not None:
            for row in entry_rows:
                append_export_record(rows_path, row)
    return rows, aggregate_benchmark_rows(rows)


def _run_baseline_mode(
    entry: BenchmarkManifestEntry,
    *,
    source_path: Path,
    output_dir: Path,
    config: Any,
    document_hash: str,
    initial_acceptance: Any,
) -> dict[str, Any]:
    return _run_baseline_seed(
        entry,
        source_path=source_path,
        output_dir=output_dir,
        config=config,
        document_hash=document_hash,
        initial_acceptance=initial_acceptance,
    ).row


def _run_baseline_seed(
    entry: BenchmarkManifestEntry,
    *,
    source_path: Path,
    output_dir: Path,
    config: Any,
    document_hash: str,
    initial_acceptance: Any,
) -> BaselineSeedResult:
    mode_dir = output_dir / "baseline"
    mode_dir.mkdir(parents=True, exist_ok=True)
    output_path = mode_dir / f"{entry.document_id}.pdf"

    fix_and_verify(
        source_path,
        output_path,
        config=config,
        original_path=source_path,
    )
    final_acceptance = evaluate_pdf_acceptance(
        output_path if output_path.exists() else source_path,
        original_path=source_path,
        config=config,
    )
    row = build_benchmark_record(
        mode="baseline",
        document_path=source_path,
        document_hash=document_hash,
        output_path=output_path if output_path.exists() else source_path,
        initial_acceptance=initial_acceptance,
        final_acceptance=final_acceptance,
        extra={
            "document_id": entry.document_id,
            "dominant_domain": entry.dominant_domain,
            "tags": list(entry.tags),
            "notes": entry.notes,
        },
    )
    return BaselineSeedResult(
        row=row,
        output_path=output_path if output_path.exists() else source_path,
        final_acceptance=final_acceptance,
    )


async def _run_with_timeout(
    func,
    timeout_seconds: float | None,
    *args,
    **kwargs,
):
    """Run a benchmark mode with an optional timeout."""
    if asyncio.iscoroutinefunction(func):
        coro = func(*args, **kwargs)
    else:
        coro = asyncio.to_thread(func, *args, **kwargs)

    if timeout_seconds is None or timeout_seconds <= 0:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout_seconds)


async def _run_vision_planner_mode(
    entry: BenchmarkManifestEntry,
    *,
    source_path: Path,
    seed_path: Path,
    output_dir: Path,
    config: Any,
    document_hash: str,
    source_acceptance: Any,
    initial_acceptance: Any,
) -> dict[str, Any]:
    from project_remedy.vision_planner.harness import VisionPlannerHarness
    from project_remedy.vision_planner.pipeline import run_vision_plan

    mode_dir = output_dir / "vision_planner"
    mode_dir.mkdir(parents=True, exist_ok=True)
    output_path = mode_dir / f"{entry.document_id}.pdf"
    trace_path = mode_dir / f"{entry.document_id}.trace.json"
    if getattr(initial_acceptance, "passed", False):
        return build_benchmark_record(
            mode="vision_planner",
            document_path=source_path,
            document_hash=document_hash,
            output_path=seed_path,
            initial_acceptance=initial_acceptance,
            final_acceptance=initial_acceptance,
            trace_path=trace_path,
            extra={
                "document_id": entry.document_id,
                "dominant_domain": entry.dominant_domain,
                "tags": list(entry.tags),
                "notes": entry.notes,
                "seed_output_path": str(seed_path),
                "source_violation_count_before": count_acceptance_warnings(source_acceptance),
                "skipped": True,
                "skipped_reason": "baseline_passed_no_fallback_needed",
            },
        )
    planner_client, owns_client = await _create_planner_client(config)

    try:
        result = await run_vision_plan(
            pdf_path=seed_path,
            output_path=trace_path,
            harness=VisionPlannerHarness(),
            client=planner_client,
            config=config,
            pdf_output_path=output_path,
        )
    finally:
        if owns_client:
            await planner_client.close()

    final_target = output_path if output_path.exists() else seed_path
    final_acceptance = evaluate_pdf_acceptance(
        final_target,
        original_path=source_path,
        config=config,
    )
    promoted = _fallback_promoted(initial_acceptance, final_acceptance)
    return build_benchmark_record(
        mode="vision_planner",
        document_path=source_path,
        document_hash=document_hash,
        output_path=final_target,
        trace_path=trace_path,
        initial_acceptance=initial_acceptance,
        final_acceptance=final_acceptance,
        extra={
            "document_id": entry.document_id,
            "dominant_domain": entry.dominant_domain,
            "tags": list(entry.tags),
            "notes": entry.notes,
            "seed_output_path": str(seed_path),
            "source_violation_count_before": count_acceptance_warnings(source_acceptance),
            "planner_client_available": True,
            "planner_client_type": type(planner_client).__name__,
            "planner_phase_executed": True,
            "promoted_phases": ["vision_planner"] if promoted else [],
            "promoted_phase_count": 1 if promoted else 0,
            "semantic_operations_executed": len(result.get("plan", {}).get("operations", []) or []),
            "semantic_manual_review_count": len(result.get("plan", {}).get("manual_review", []) or []),
        },
    )


async def _run_specialist_coordinator_mode(
    entry: BenchmarkManifestEntry,
    *,
    source_path: Path,
    seed_path: Path,
    output_dir: Path,
    config: Any,
    document_hash: str,
    source_acceptance: Any,
    initial_acceptance: Any,
) -> dict[str, Any]:
    from project_remedy.pdf_specialists.coordinator import run_specialist_pdf_repair

    mode_dir = output_dir / "specialist_coordinator"
    mode_dir.mkdir(parents=True, exist_ok=True)
    output_path = mode_dir / f"{entry.document_id}.pdf"
    trace_path = mode_dir / f"{entry.document_id}.trace.json"
    if getattr(initial_acceptance, "passed", False):
        return build_benchmark_record(
            mode="specialist_coordinator",
            document_path=source_path,
            document_hash=document_hash,
            output_path=seed_path,
            initial_acceptance=initial_acceptance,
            final_acceptance=initial_acceptance,
            trace_path=trace_path,
            extra={
                "document_id": entry.document_id,
                "dominant_domain": entry.dominant_domain,
                "tags": list(entry.tags),
                "notes": entry.notes,
                "seed_output_path": str(seed_path),
                "source_violation_count_before": count_acceptance_warnings(source_acceptance),
                "skipped": True,
                "skipped_reason": "baseline_passed_no_fallback_needed",
            },
        )
    planner_client, owns_client = await _create_planner_client(config)

    try:
        result = await run_specialist_pdf_repair(
            pdf_path=seed_path,
            output_path=output_path,
            original_path=source_path,
            config=config,
            planner_client=planner_client,
            trace_path=trace_path,
        )
    finally:
        if owns_client:
            await planner_client.close()

    return build_coordinator_export_record(
        result,
        document_path=source_path,
        document_hash=document_hash,
        extra={
            "document_id": entry.document_id,
            "dominant_domain": entry.dominant_domain,
            "tags": list(entry.tags),
            "notes": entry.notes,
            "mode": "specialist_coordinator",
            "violation_count_before": count_acceptance_warnings(initial_acceptance),
            "seed_output_path": str(seed_path),
            "source_violation_count_before": count_acceptance_warnings(source_acceptance),
        },
    )


def _screen_residual_candidate_worker(queue, pdf_path: Path, output_dir: Path, config: Any, seed_mode: DiscoverySeedMode) -> None:
    try:
        queue.put(
            screen_residual_candidate(
                pdf_path,
                output_dir=output_dir,
                config=config,
                seed_mode=seed_mode,
            ).to_dict()
        )
    except Exception as exc:  # pragma: no cover - forwarded to parent
        queue.put(
            ResidualCandidateResult(
                pdf_path=str(pdf_path),
                baseline_path="",
                seed_mode=seed_mode,
                elapsed_seconds=0.0,
                passed=False,
                acceptance_summary=str(exc),
                violation_count=0,
                domain_counts={},
                warning_rule_ids=[],
                error=str(exc),
            ).to_dict()
        )


def _stable_manifest_document_id(pdf_path: Path, seen_ids: set[str]) -> str:
    """Build a stable manifest document id with collision protection."""
    base = re.sub(r"[^a-z0-9]+", "-", pdf_path.stem.lower()).strip("-") or "document"
    if base not in seen_ids:
        return base
    suffix = hashlib.sha256(str(pdf_path).encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}-{suffix}"
    if candidate not in seen_ids:
        return candidate
    counter = 2
    while f"{candidate}-{counter}" in seen_ids:
        counter += 1
    return f"{candidate}-{counter}"


async def _create_planner_client(config: Any) -> tuple[Any, bool]:
    """Create a generate_raw-capable planner client for benchmark modes."""
    from project_remedy.gemini_client import GeminiClient

    gemini_key = str(getattr(config.api, "gemini_api_key", "") or "").strip()
    gemini_vertexai = bool(getattr(config.api, "gemini_vertexai", False))
    gemini_project = str(getattr(config.api, "gemini_gcp_project", "") or "").strip()

    if gemini_vertexai and not gemini_project:
        gemini_project = _gcloud_default_project()
        if gemini_project:
            gemini_api = replace(
                config.api,
                gemini_gcp_project=gemini_project,
            )
            config = replace(config, api=gemini_api)

    if getattr(config.api, "llm_backend", "") == "gemini":
        if gemini_vertexai:
            if not gemini_project:
                raise RuntimeError("Vertex AI benchmark mode requires a GCP project")
        elif not gemini_key:
            raise RuntimeError("Gemini API key missing for benchmark planner mode")
        client = GeminiClient(config)
        await client.start()
        return client, True

    if getattr(config.api, "escalation_backend", "") == "gemini":
        if gemini_vertexai:
            if not gemini_project:
                raise RuntimeError("Vertex AI benchmark mode requires a GCP project")
        elif not gemini_key:
            raise RuntimeError("Gemini API key missing for benchmark planner mode")
        escalation_api = replace(
            config.api,
            llm_backend="gemini",
            gemini_model=config.api.escalation_model,
            gemini_batch_enabled=False,
        )
        gemini_config = replace(config, api=escalation_api)
        client = GeminiClient(gemini_config)
        await client.start()
        return client, True

    raise RuntimeError(
        "Benchmark planner modes require a generate_raw-capable Gemini configuration",
    )


def _gcloud_default_project() -> str:
    """Best-effort current gcloud project lookup for local benchmark runs."""
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _build_error_record(
    *,
    mode: BenchmarkMode,
    entry: BenchmarkManifestEntry,
    source_path: Path,
    document_hash: str,
    initial_acceptance: Any,
    error: str,
) -> dict[str, Any]:
    return build_benchmark_record(
        mode=mode,
        document_path=source_path,
        document_hash=document_hash,
        output_path=source_path,
        initial_acceptance=initial_acceptance,
        final_acceptance=initial_acceptance,
        extra={
            "document_id": entry.document_id,
            "dominant_domain": entry.dominant_domain,
            "tags": list(entry.tags),
            "notes": entry.notes,
            "error": error,
        },
    )


# ---------------------------------------------------------------------
# CLI entry point — ``python -m project_remedy.pdf_specialists.benchmark``
# ---------------------------------------------------------------------

_GATED_MODES_MESSAGE = (
    "vision_planner and specialist_coordinator benchmark modes require a "
    "real residual-failure corpus plus a Gemini / generate_raw-capable "
    "planner client — see REMEDY-69 #13/#14/#15 for the sourcing work."
)


def _conformance_band(record: dict[str, Any]) -> str:
    """Derive a user-facing conformance label from a benchmark row.

    The benchmark ``build_*_record`` helpers surface ``acceptance_passed``
    and ``violation_count_after`` but not the human-readable banner the
    UI uses. We reconstruct a rough triage band here so operators get a
    glanceable column without reaching into the trace.
    """
    if record.get("error"):
        return "error"
    if record.get("skipped"):
        return "skipped"
    if record.get("acceptance_passed"):
        return "Conformant"
    residual = int(record.get("violation_count_after", 0) or 0)
    before = int(record.get("violation_count_before", 0) or 0)
    if residual == 0:
        return "Conformant"
    if before > 0 and residual < before:
        return "Partially Conformant"
    return "Not Conformant"


def _format_summary_table(rows: list[dict[str, Any]]) -> str:
    """Render the per-entry benchmark summary as an aligned ASCII table."""
    headers = ("document_id", "mode", "acceptance_warnings", "conformance_band")
    data = [
        (
            str(row.get("document_id", "")),
            str(row.get("mode", "")),
            str(int(row.get("violation_count_after", 0) or 0)),
            _conformance_band(row),
        )
        for row in rows
    ]
    widths = [len(h) for h in headers]
    for record in data:
        for i, cell in enumerate(record):
            widths[i] = max(widths[i], len(cell))

    def _row(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [_row(headers), _row(tuple("-" * w for w in widths))]
    lines.extend(_row(record) for record in data)
    return "\n".join(lines)


def _run_manifest_cli(
    manifest_path: Path,
    *,
    output_dir: Path,
    modes: tuple[BenchmarkMode, ...],
    mode_timeout_seconds: float | None,
) -> int:
    """Execute the benchmark pipeline for the given manifest.

    Returns a process exit code: 0 if every entry produced at least one
    result without a crash (``error`` field empty), 1 otherwise. Gated
    modes (``vision_planner``/``specialist_coordinator``) are allowed to
    bubble their missing-prerequisite errors through — they land as
    per-row ``error`` strings, not process-level crashes, but they do
    tip the exit code to 1 so CI can notice.
    """
    from project_remedy.config import load_config

    manifest = load_benchmark_manifest(manifest_path)
    if not manifest.entries:
        print(f"ERROR: manifest {manifest_path} contains no entries", flush=True)
        return 1

    gated = tuple(mode for mode in modes if mode != "baseline")
    if gated:
        print(f"WARNING: {_GATED_MODES_MESSAGE}", flush=True)
        print(f"         requested gated modes: {', '.join(gated)}", flush=True)

    config = load_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"

    # Strip any stale rows file so this invocation writes a clean record.
    if rows_path.exists():
        rows_path.unlink()

    print(f"Benchmark manifest : {manifest_path}")
    print(f"Manifest name      : {manifest.name}")
    print(f"Modes              : {', '.join(modes)}")
    print(f"Output directory   : {output_dir}")
    print(f"Mode timeout (s)   : {mode_timeout_seconds}")
    print(f"Entries            : {len(manifest.entries)}")
    print()

    try:
        rows, summary = asyncio.run(
            run_manifest_benchmark(
                manifest,
                output_dir=output_dir,
                config=config,
                modes=modes,
                rows_path=rows_path,
                mode_timeout_seconds=mode_timeout_seconds,
            )
        )
    except Exception as exc:  # pragma: no cover - surfaced to CI log
        print(f"ERROR: benchmark run crashed: {exc}", flush=True)
        return 1

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(_format_summary_table(rows))
    print()
    print(f"Full summary written to: {summary_path}")
    print(f"Per-row JSONL at:        {rows_path}")

    # Exit 1 if any entry produced an error row — keeps the gate honest
    # when gated modes were requested without their prerequisites.
    if any(row.get("error") for row in rows):
        return 1
    return 0


def _default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("/tmp") / f"remedy-bench-{stamp}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — ``python -m project_remedy.pdf_specialists.benchmark``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m project_remedy.pdf_specialists.benchmark",
        description=(
            "Run the PDF specialist coordinator benchmark against a frozen "
            "manifest. Baseline mode runs fully offline; vision_planner and "
            "specialist_coordinator modes are gated on an external residual "
            "corpus and a Gemini planner client (REMEDY-69 #13/#14/#15)."
        ),
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to a BenchmarkManifest JSON file (see tests/fixtures/corpus/manifest.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for benchmark artifacts (default: /tmp/remedy-bench-<timestamp>).",
    )
    parser.add_argument(
        "--modes",
        default="baseline",
        help=(
            "Comma-separated list of benchmark modes to run. Choices: "
            "baseline, vision_planner, specialist_coordinator. "
            "Default 'baseline' is the only mode that runs without the "
            "external residual corpus."
        ),
    )
    parser.add_argument(
        "--mode-timeout",
        type=float,
        default=300.0,
        help="Per-mode timeout in seconds (default: 300). Pass 0 to disable.",
    )
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", flush=True)
        return 1

    output_dir = (args.output_dir or _default_output_dir()).resolve()

    requested = tuple(m.strip() for m in str(args.modes).split(",") if m.strip())
    valid_modes = {"baseline", "vision_planner", "specialist_coordinator"}
    invalid = [m for m in requested if m not in valid_modes]
    if invalid:
        print(
            f"ERROR: unknown benchmark mode(s): {', '.join(invalid)}. "
            f"Valid choices: {sorted(valid_modes)}",
            flush=True,
        )
        return 1
    if not requested:
        requested = ("baseline",)
    modes: tuple[BenchmarkMode, ...] = tuple(requested)  # type: ignore[assignment]

    timeout = args.mode_timeout if args.mode_timeout and args.mode_timeout > 0 else None

    return _run_manifest_cli(
        manifest_path,
        output_dir=output_dir,
        modes=modes,
        mode_timeout_seconds=timeout,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
