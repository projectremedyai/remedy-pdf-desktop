"""Serving-stack benchmark helpers for Apple-Silicon vision backends."""

from __future__ import annotations

import asyncio
import base64
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx

from project_remedy.ocr_escalation import _token_overlap_score
from project_remedy.pdf_benchmark import CorpusManifest
from project_remedy.pdf_vision import _parse_json_response, render_page_to_image
from project_remedy.vision_prompts import ocr_markdown_prompt, reading_order_prompt

ServingCandidateKind = Literal["lmstudio", "ollama", "vllm_mlx", "vllm_metal", "custom"]
ServingProfileName = Literal["ocr_page", "vision_layout"]
ServingCandidateRole = Literal["baseline", "candidate", "fallback"]

_TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnS7xgAAAAASUVORK5CYII="
)


@dataclass
class ServingBenchmarkCandidate:
    name: str
    kind: ServingCandidateKind
    base_url: str
    model: str
    role: ServingCandidateRole = "candidate"
    api_key: str = "not-needed"
    expected_model: str = ""
    enabled: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ServingBenchmarkConfig:
    generated_at: str
    manifest_path: Path | None
    candidates: list[ServingBenchmarkCandidate]
    profiles: list[ServingProfileName] = field(default_factory=lambda: ["ocr_page", "vision_layout"])
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    rounds_per_level: int = 2
    quality_sample_limit: int = 6
    timeout_seconds: float = 120.0
    include_quality_stage: bool = True
    include_throughput_stage: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "profiles": list(self.profiles),
            "concurrency_levels": list(self.concurrency_levels),
            "rounds_per_level": self.rounds_per_level,
            "quality_sample_limit": self.quality_sample_limit,
            "timeout_seconds": self.timeout_seconds,
            "include_quality_stage": self.include_quality_stage,
            "include_throughput_stage": self.include_throughput_stage,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass
class ServingCompatibilityResult:
    candidate: str
    kind: str
    models_ok: bool
    expected_model_available: bool
    multimodal_ok: bool
    nonempty_response: bool
    response_chars: int
    models: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.models_ok
            and self.expected_model_available
            and self.multimodal_ok
            and self.nonempty_response
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"passed": self.passed}


@dataclass
class ServingQualityEntry:
    candidate: str
    profile: str
    sample_name: str
    layout_class: str
    page_number: int
    success: bool
    latency_ms: float
    output_chars: int
    parsed_json: bool
    output_path: Path | None = None
    token_overlap_vs_baseline: float | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_path"] = str(self.output_path) if self.output_path else None
        return data


@dataclass
class ServingQualitySummary:
    candidate: str
    profile: str
    sample_count: int
    success_count: int
    mean_latency_ms: float
    mean_output_chars: float
    json_parse_success_count: int
    mean_token_overlap_vs_baseline: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ServingThroughputPoint:
    candidate: str
    profile: str
    scenario: str
    concurrency: int
    total_requests: int
    success_count: int
    wall_time_seconds: float
    requests_per_second: float
    p50_latency_ms: float
    p95_latency_ms: float
    error_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ServingBenchmarkRun:
    config_path: Path | None
    generated_at: str
    compatibility: list[ServingCompatibilityResult]
    quality_entries: list[ServingQualityEntry]
    quality_summaries: list[ServingQualitySummary]
    throughput_points: list[ServingThroughputPoint]
    output_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path) if self.config_path else None,
            "generated_at": self.generated_at,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "compatibility": [entry.to_dict() for entry in self.compatibility],
            "quality_entries": [entry.to_dict() for entry in self.quality_entries],
            "quality_summaries": [entry.to_dict() for entry in self.quality_summaries],
            "throughput_points": [entry.to_dict() for entry in self.throughput_points],
        }


@dataclass(frozen=True)
class _BenchmarkSample:
    pdf_path: Path
    layout_class: str
    page_number: int
    sample_name: str
    scenario: str = ""


def default_serving_benchmark_config() -> ServingBenchmarkConfig:
    """Return the default serving benchmark candidate matrix."""
    return ServingBenchmarkConfig(
        generated_at=datetime.now(timezone.utc).isoformat(),
        manifest_path=None,
        candidates=[
            ServingBenchmarkCandidate(
                name="lmstudio-baseline",
                kind="lmstudio",
                base_url="http://localhost:1234/v1",
                model="glm-4.6v-flash",
                expected_model="glm-4.6v-flash",
                role="baseline",
                notes="Current LM Studio vision baseline.",
            ),
            ServingBenchmarkCandidate(
                name="vllm-mlx-pilot",
                kind="vllm_mlx",
                base_url="http://localhost:8000/v1",
                model="mlx-community/GLM-4.6V-Flash-bf16",
                expected_model="mlx-community/GLM-4.6V-Flash-bf16",
                role="candidate",
                notes="Primary Apple-Silicon pilot using the MLX-community GLM-4.6V-Flash conversion.",
            ),
            ServingBenchmarkCandidate(
                name="vllm-metal-pilot",
                kind="vllm_metal",
                base_url="http://localhost:8001/v1",
                model="default",
                role="fallback",
                notes="Fallback pilot only if vLLM-MLX is blocked.",
            ),
        ],
    )


def default_serving_benchmark_config_path(root: Path) -> Path:
    return root / "benchmarks" / "serving_stack_candidates.json"


def save_serving_benchmark_config(config: ServingBenchmarkConfig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return output_path


def load_serving_benchmark_config(path: Path) -> ServingBenchmarkConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ServingBenchmarkConfig(
        generated_at=data["generated_at"],
        manifest_path=Path(data["manifest_path"]) if data.get("manifest_path") else None,
        candidates=[
            ServingBenchmarkCandidate(**item)
            for item in data.get("candidates", [])
        ],
        profiles=list(data.get("profiles", ["ocr_page", "vision_layout"])),
        concurrency_levels=list(data.get("concurrency_levels", [1, 4, 8, 16, 32])),
        rounds_per_level=int(data.get("rounds_per_level", 2)),
        quality_sample_limit=int(data.get("quality_sample_limit", 6)),
        timeout_seconds=float(data.get("timeout_seconds", 120.0)),
        include_quality_stage=bool(data.get("include_quality_stage", True)),
        include_throughput_stage=bool(data.get("include_throughput_stage", True)),
    )


def save_serving_benchmark_run(run: ServingBenchmarkRun, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return output_path


def run_serving_compatibility_probe(
    config: ServingBenchmarkConfig,
) -> list[ServingCompatibilityResult]:
    """Probe enabled candidates for model listing + multimodal request support."""
    return asyncio.run(_run_serving_compatibility_probe_async(config))


def run_serving_compatibility_probe_for_manifest(
    config: ServingBenchmarkConfig,
    manifest: CorpusManifest,
) -> list[ServingCompatibilityResult]:
    """Probe candidates using a real rendered manifest page instead of a synthetic image."""
    quality_samples = _quality_samples_from_manifest(
        manifest,
        limit=max(1, config.quality_sample_limit),
    )
    return asyncio.run(
        _run_serving_compatibility_probe_for_manifest_async(
            config,
            quality_samples,
        )
    )


async def _run_serving_compatibility_probe_async(
    config: ServingBenchmarkConfig,
    *,
    probe_image_data_url: str = _TINY_PNG_DATA_URL,
    probe_prompt: str | None = None,
) -> list[ServingCompatibilityResult]:
    results: list[ServingCompatibilityResult] = []
    prompt = probe_prompt or _prompt_for_profile("ocr_page", page_number=1)
    for candidate in _enabled_candidates(config):
        models: list[str] = []
        models_ok = False
        multimodal_ok = False
        nonempty = False
        response_chars = 0
        error = ""

        try:
            models = await _fetch_models(candidate, timeout_seconds=config.timeout_seconds)
            models_ok = True
        except Exception as exc:
            error = str(exc)

        try:
            response_text, _latency_ms = await _invoke_multimodal_request(
                candidate,
                prompt=prompt,
                image_data_url=probe_image_data_url,
                timeout_seconds=config.timeout_seconds,
            )
            multimodal_ok = True
            response_chars = len(response_text)
            nonempty = bool(response_text.strip())
        except Exception as exc:
            if not error:
                error = str(exc)

        expected_model = candidate.expected_model or candidate.model
        expected_available = (
            True
            if not expected_model or expected_model == "default"
            else expected_model in models
        )
        results.append(
            ServingCompatibilityResult(
                candidate=candidate.name,
                kind=candidate.kind,
                models_ok=models_ok,
                expected_model_available=expected_available,
                multimodal_ok=multimodal_ok,
                nonempty_response=nonempty,
                response_chars=response_chars,
                models=models,
                error=error,
            )
        )
    return results


def run_serving_bakeoff(
    config: ServingBenchmarkConfig,
    manifest: CorpusManifest,
    *,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> ServingBenchmarkRun:
    """Run compatibility, quality, and concurrency benchmark stages."""
    probe_samples = _quality_samples_from_manifest(
        manifest,
        limit=max(1, config.quality_sample_limit),
    )
    quality_samples = (
        _quality_samples_from_manifest(
            manifest,
            limit=config.quality_sample_limit,
        )
        if config.include_quality_stage and config.quality_sample_limit > 0
        else []
    )
    throughput_samples = _throughput_samples_from_manifest(manifest)
    output_dir = output_dir or (Path.cwd() / "benchmarks" / "serving_benchmark_outputs" / _timestamp_slug())
    output_dir.mkdir(parents=True, exist_ok=True)

    compatibility = asyncio.run(
        _run_serving_compatibility_probe_for_manifest_async(
            config,
            probe_samples,
        )
    )
    compatible_names = {entry.candidate for entry in compatibility if entry.passed}
    benchmark_candidates = [
        candidate
        for candidate in _enabled_candidates(config)
        if candidate.name in compatible_names
    ]

    quality_entries: list[ServingQualityEntry] = []
    quality_summaries: list[ServingQualitySummary] = []
    if config.include_quality_stage and quality_samples:
        quality_entries = asyncio.run(
            _run_quality_stage_async(
                benchmark_candidates,
                config,
                quality_samples,
                output_dir=output_dir,
            )
        )
        quality_summaries = _summarize_quality_entries(
            quality_entries,
            baseline_name=_baseline_candidate_name(config),
        )
    throughput_points: list[ServingThroughputPoint] = []
    if config.include_throughput_stage and throughput_samples:
        throughput_points = asyncio.run(
            _run_throughput_stage_async(
                benchmark_candidates,
                config,
                throughput_samples,
            )
        )
    return ServingBenchmarkRun(
        config_path=config_path,
        generated_at=datetime.now(timezone.utc).isoformat(),
        compatibility=compatibility,
        quality_entries=quality_entries,
        quality_summaries=quality_summaries,
        throughput_points=throughput_points,
        output_dir=output_dir,
    )


async def _run_serving_compatibility_probe_for_manifest_async(
    config: ServingBenchmarkConfig,
    samples: list[_BenchmarkSample],
) -> list[ServingCompatibilityResult]:
    if not samples:
        return await _run_serving_compatibility_probe_async(config)

    sample = samples[0]
    image_path = render_page_to_image(sample.pdf_path, sample.page_number)
    try:
        image_data_url = _image_path_to_data_url(image_path)
        prompt = _prompt_for_profile("ocr_page", page_number=sample.page_number)
        return await _run_serving_compatibility_probe_async(
            config,
            probe_image_data_url=image_data_url,
            probe_prompt=prompt,
        )
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            pass


def _enabled_candidates(config: ServingBenchmarkConfig) -> list[ServingBenchmarkCandidate]:
    return [candidate for candidate in config.candidates if candidate.enabled]


def _baseline_candidate_name(config: ServingBenchmarkConfig) -> str | None:
    for candidate in _enabled_candidates(config):
        if candidate.role == "baseline":
            return candidate.name
    enabled = _enabled_candidates(config)
    return enabled[0].name if enabled else None


def _quality_samples_from_manifest(
    manifest: CorpusManifest,
    *,
    limit: int,
) -> list[_BenchmarkSample]:
    samples: list[_BenchmarkSample] = []
    for sample in manifest.samples:
        for page_number in sample.benchmark_pages or [1]:
            samples.append(
                _BenchmarkSample(
                    pdf_path=sample.pdf_path,
                    layout_class=sample.layout_class,
                    page_number=page_number,
                    sample_name=sample.pdf_path.name,
                )
            )
            if len(samples) >= limit:
                return samples
    return samples


def _throughput_samples_from_manifest(manifest: CorpusManifest) -> list[_BenchmarkSample]:
    small = None
    large = None
    for sample in manifest.samples:
        page_number = (sample.benchmark_pages or [1])[0]
        candidate = _BenchmarkSample(
            pdf_path=sample.pdf_path,
            layout_class=sample.layout_class,
            page_number=page_number,
            sample_name=sample.pdf_path.name,
        )
        if small is None and sample.page_count <= 3:
            small = _BenchmarkSample(
                pdf_path=candidate.pdf_path,
                layout_class=candidate.layout_class,
                page_number=candidate.page_number,
                sample_name=candidate.sample_name,
                scenario="small_document",
            )
        if large is None and sample.page_count >= 100:
            large = _BenchmarkSample(
                pdf_path=candidate.pdf_path,
                layout_class=candidate.layout_class,
                page_number=candidate.page_number,
                sample_name=candidate.sample_name,
                scenario="large_document",
            )
        if small and large:
            break

    samples: list[_BenchmarkSample] = []
    if small:
        samples.append(small)
    if large:
        samples.append(large)
    if not samples and manifest.samples:
        sample = manifest.samples[0]
        samples.append(
            _BenchmarkSample(
                pdf_path=sample.pdf_path,
                layout_class=sample.layout_class,
                page_number=(sample.benchmark_pages or [1])[0],
                sample_name=sample.pdf_path.name,
                scenario="default_document",
            )
        )
    return samples


async def _run_quality_stage_async(
    candidates: list[ServingBenchmarkCandidate],
    config: ServingBenchmarkConfig,
    samples: list[_BenchmarkSample],
    *,
    output_dir: Path,
) -> list[ServingQualityEntry]:
    candidate_results = await asyncio.gather(
        *[
            _run_quality_stage_for_candidate_async(
                candidate,
                config,
                samples,
                output_dir=output_dir,
            )
            for candidate in candidates
        ]
    )
    entries: list[ServingQualityEntry] = []
    response_lookup: dict[tuple[str, str, str, int], str] = {}
    for candidate_entries, candidate_lookup in candidate_results:
        entries.extend(candidate_entries)
        response_lookup.update(candidate_lookup)

    baseline_name = _baseline_candidate_name(config)
    if baseline_name:
        for entry in entries:
            key = (baseline_name, entry.profile, entry.sample_name, entry.page_number)
            if not entry.success or key not in response_lookup:
                continue
            baseline_text = response_lookup[key]
            candidate_key = (entry.candidate, entry.profile, entry.sample_name, entry.page_number)
            candidate_text = response_lookup.get(candidate_key)
            if candidate_text is None:
                continue
            entry.token_overlap_vs_baseline = _token_overlap_score(baseline_text, candidate_text)
    entries.sort(key=lambda entry: (entry.candidate, entry.profile, entry.sample_name, entry.page_number))
    return entries


async def _run_quality_stage_for_candidate_async(
    candidate: ServingBenchmarkCandidate,
    config: ServingBenchmarkConfig,
    samples: list[_BenchmarkSample],
    *,
    output_dir: Path,
) -> tuple[list[ServingQualityEntry], dict[tuple[str, str, str, int], str]]:
    entries: list[ServingQualityEntry] = []
    response_lookup: dict[tuple[str, str, str, int], str] = {}
    for profile in config.profiles:
        for sample in samples:
            image_path = render_page_to_image(sample.pdf_path, sample.page_number)
            try:
                image_data_url = _image_path_to_data_url(image_path)
                prompt = _prompt_for_profile(profile, page_number=sample.page_number)
                response_text, latency_ms = await _invoke_multimodal_request(
                    candidate,
                    prompt=prompt,
                    image_data_url=image_data_url,
                    timeout_seconds=config.timeout_seconds,
                )
                parsed_json = bool(_parse_json_response(response_text)) if profile == "vision_layout" else False
                output_path = _write_quality_output(
                    output_dir,
                    candidate_name=candidate.name,
                    profile=profile,
                    sample=sample,
                    text=response_text,
                )
                entry = ServingQualityEntry(
                    candidate=candidate.name,
                    profile=profile,
                    sample_name=sample.sample_name,
                    layout_class=sample.layout_class,
                    page_number=sample.page_number,
                    success=True,
                    latency_ms=latency_ms,
                    output_chars=len(response_text),
                    parsed_json=parsed_json,
                    output_path=output_path,
                )
                response_lookup[(candidate.name, profile, sample.sample_name, sample.page_number)] = response_text
            except Exception as exc:
                entry = ServingQualityEntry(
                    candidate=candidate.name,
                    profile=profile,
                    sample_name=sample.sample_name,
                    layout_class=sample.layout_class,
                    page_number=sample.page_number,
                    success=False,
                    latency_ms=0.0,
                    output_chars=0,
                    parsed_json=False,
                    error=str(exc),
                )
            entries.append(entry)
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                pass
    return entries, response_lookup


def _summarize_quality_entries(
    entries: list[ServingQualityEntry],
    *,
    baseline_name: str | None,
) -> list[ServingQualitySummary]:
    grouped: dict[tuple[str, str], list[ServingQualityEntry]] = {}
    for entry in entries:
        grouped.setdefault((entry.candidate, entry.profile), []).append(entry)

    summaries: list[ServingQualitySummary] = []
    for (candidate, profile), group in sorted(grouped.items()):
        successes = [entry for entry in group if entry.success]
        overlaps = [
            entry.token_overlap_vs_baseline
            for entry in successes
            if entry.token_overlap_vs_baseline is not None
        ]
        if candidate == baseline_name and successes and not overlaps:
            overlaps = [1.0 for _ in successes]
        summaries.append(
            ServingQualitySummary(
                candidate=candidate,
                profile=profile,
                sample_count=len(group),
                success_count=len(successes),
                mean_latency_ms=(sum(entry.latency_ms for entry in successes) / len(successes)) if successes else 0.0,
                mean_output_chars=(sum(entry.output_chars for entry in successes) / len(successes)) if successes else 0.0,
                json_parse_success_count=sum(1 for entry in successes if entry.parsed_json),
                mean_token_overlap_vs_baseline=(sum(overlaps) / len(overlaps)) if overlaps else 0.0,
            )
        )
    return summaries


async def _run_throughput_stage_async(
    candidates: list[ServingBenchmarkCandidate],
    config: ServingBenchmarkConfig,
    scenarios: list[_BenchmarkSample],
) -> list[ServingThroughputPoint]:
    candidate_points = await asyncio.gather(
        *[
            _run_throughput_stage_for_candidate_async(candidate, config, scenarios)
            for candidate in candidates
        ]
    )
    points: list[ServingThroughputPoint] = []
    for group in candidate_points:
        points.extend(group)
    points.sort(key=lambda point: (point.candidate, point.profile, point.scenario, point.concurrency))
    return points


async def _run_throughput_stage_for_candidate_async(
    candidate: ServingBenchmarkCandidate,
    config: ServingBenchmarkConfig,
    scenarios: list[_BenchmarkSample],
) -> list[ServingThroughputPoint]:
    points: list[ServingThroughputPoint] = []
    for profile in config.profiles:
        for sample in scenarios:
            image_path = render_page_to_image(sample.pdf_path, sample.page_number)
            try:
                image_data_url = _image_path_to_data_url(image_path)
                prompt = _prompt_for_profile(profile, page_number=sample.page_number)
                for concurrency in config.concurrency_levels:
                    point = await _run_concurrency_level(
                        candidate,
                        profile=profile,
                        sample=sample,
                        image_data_url=image_data_url,
                        prompt=prompt,
                        concurrency=concurrency,
                        rounds=config.rounds_per_level,
                        timeout_seconds=config.timeout_seconds,
                    )
                    points.append(point)
            finally:
                try:
                    image_path.unlink(missing_ok=True)
                except Exception:
                    pass
    return points


async def _run_concurrency_level(
    candidate: ServingBenchmarkCandidate,
    *,
    profile: str,
    sample: _BenchmarkSample,
    image_data_url: str,
    prompt: str,
    concurrency: int,
    rounds: int,
    timeout_seconds: float,
) -> ServingThroughputPoint:
    latencies: list[float] = []
    success_count = 0
    error_count = 0
    start = time.perf_counter()

    for _ in range(rounds):
        tasks = [
            _invoke_multimodal_request(
                candidate,
                prompt=prompt,
                image_data_url=image_data_url,
                timeout_seconds=timeout_seconds,
            )
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
                continue
            _response_text, latency_ms = result
            success_count += 1
            latencies.append(latency_ms)

    wall_time_seconds = time.perf_counter() - start
    total_requests = concurrency * rounds
    return ServingThroughputPoint(
        candidate=candidate.name,
        profile=profile,
        scenario=sample.scenario or sample.sample_name,
        concurrency=concurrency,
        total_requests=total_requests,
        success_count=success_count,
        wall_time_seconds=wall_time_seconds,
        requests_per_second=(success_count / wall_time_seconds) if wall_time_seconds > 0 else 0.0,
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        error_count=error_count,
    )


async def _fetch_models(
    candidate: ServingBenchmarkCandidate,
    *,
    timeout_seconds: float,
) -> list[str]:
    async with httpx.AsyncClient(
        base_url=candidate.base_url.rstrip("/"),
        headers={
            "Authorization": f"Bearer {candidate.api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0)),
        trust_env=False,
    ) as client:
        response = await client.get("/models")
        response.raise_for_status()
        payload = response.json()
    return [
        str(item.get("id", "")).strip()
        for item in payload.get("data", [])
        if str(item.get("id", "")).strip()
    ]


async def _invoke_multimodal_request(
    candidate: ServingBenchmarkCandidate,
    *,
    prompt: str,
    image_data_url: str,
    timeout_seconds: float,
) -> tuple[str, float]:
    if candidate.kind == "ollama":
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": image_data_url},
        ]
    else:
        content = [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": prompt},
        ]

    payload = {
        "model": candidate.model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }

    start = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=candidate.base_url.rstrip("/"),
        headers={
            "Authorization": f"Bearer {candidate.api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0)),
        trust_env=False,
    ) as client:
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
    latency_ms = (time.perf_counter() - start) * 1000.0
    message = (
        data.get("choices", [{}])[0]
        .get("message", {})
    )
    text = (
        message.get("content")
        or message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )
    return str(text), latency_ms


def _prompt_for_profile(profile: str, *, page_number: int) -> str:
    if profile == "ocr_page":
        return (
            f"Page {page_number}. Extract this PDF page to structured Markdown. "
            "Preserve reading order, headings, paragraphs, lists, tables, and form labels. "
            "Use a short bracketed placeholder for non-text images. "
            "Do not invent text, URLs, or filenames."
        )
    if profile == "vision_layout":
        return (
            "Return compact JSON with a reading_order array. "
            "Each item should include role, label, and brief notes. "
            "Keep the sequence faithful to the page layout."
        )
    raise ValueError(f"Unknown serving benchmark profile: {profile}")


def _image_path_to_data_url(image_path: Path) -> str:
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    suffix = image_path.suffix.lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        suffix, "image/png"
    )
    return f"data:{mime};base64,{b64}"


def _write_quality_output(
    output_dir: Path,
    *,
    candidate_name: str,
    profile: str,
    sample: _BenchmarkSample,
    text: str,
) -> Path:
    safe_name = sample.sample_name.replace("/", "_")
    dest = output_dir / candidate_name / profile / f"{safe_name}_page{sample.page_number}.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    index = (len(values) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    fraction = index - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
