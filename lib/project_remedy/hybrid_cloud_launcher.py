"""Hybrid LiteParse-based routing and launch planning for cloud PDF reruns."""

from __future__ import annotations

import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from project_remedy.config import PipelineConfig

ALL_CAMPUSES = ("LACC", "ELAC", "LAMC", "LAPC", "LAHC", "LASC", "LATTC", "LAVC", "WLAC")
LANE_MODELS = {
    "kimi": "kimi-k2.5",
    "qwen": "qwen3-vl:235b",
}
_CLOUD_BASE_URL = "https://ollama.com/v1"


@dataclass(frozen=True)
class RoutedPDF:
    campus: str
    pdf_path: Path
    classification: str
    lane: str
    char_count: int = 0
    pages_sampled: int = 0
    parser_error: str = ""


@dataclass(frozen=True)
class LaneShardPlan:
    lane: str
    model: str
    shard_index: int
    file_count: int
    manifest_path: Path
    log_path: Path
    pid_path: Path


@dataclass(frozen=True)
class HybridLaunchPlan:
    project_root: Path
    run_root: Path
    total_files: int
    lane_counts: dict[str, int]
    slot_counts: dict[str, int]
    routes: tuple[RoutedPDF, ...]
    shards: tuple[LaneShardPlan, ...]


def collect_downloaded_source_pdfs(
    project_root: Path,
    *,
    campuses: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[tuple[str, Path]]:
    """Return `(campus, pdf_path)` pairs from exports/*/downloads/pdf."""
    exports_root = project_root / "exports"
    selected = [campus.upper() for campus in (campuses or ALL_CAMPUSES)]
    items: list[tuple[str, Path]] = []
    for campus in selected:
        src_dir = exports_root / campus / "downloads" / "pdf"
        if not src_dir.exists():
            continue
        for pdf_path in sorted(src_dir.glob("*.pdf")):
            items.append((campus, pdf_path.resolve()))
            if limit is not None and len(items) >= limit:
                return items
    return items


def route_classification(classification: str, *, route_unknown: str = "qwen") -> str:
    """Map LiteParse classification to a hybrid cloud lane."""
    normalized = str(classification or "").strip().lower()
    if normalized == "text_rich":
        return "kimi"
    if normalized == "sparse_or_scanned":
        return "qwen"
    return route_unknown


def classify_and_route_pdfs(
    items: Iterable[tuple[str, Path]],
    *,
    config: PipelineConfig,
    classification_workers: int = 6,
    route_unknown: str = "qwen",
) -> list[RoutedPDF]:
    """Classify PDFs with LiteParse and assign each file to one cloud lane."""
    from project_remedy.liteparse_adapter import classify_pdf_for_routing

    work = list(items)
    if not work:
        return []

    max_workers = max(1, min(int(classification_workers), len(work)))

    def _classify_one(item: tuple[str, Path]) -> RoutedPDF:
        campus, pdf_path = item
        try:
            triage = classify_pdf_for_routing(pdf_path, config=config)
            classification = getattr(triage, "classification", "unknown")
            char_count = int(getattr(triage, "char_count", 0) or 0)
            pages_sampled = int(getattr(triage, "pages_sampled", 0) or 0)
            parser_error = str(getattr(triage, "parser_error", "") or "").strip()
        except Exception as exc:
            classification = "unknown"
            char_count = 0
            pages_sampled = 0
            parser_error = str(exc).strip()

        return RoutedPDF(
            campus=campus,
            pdf_path=pdf_path,
            classification=classification or "unknown",
            lane=route_classification(classification, route_unknown=route_unknown),
            char_count=char_count,
            pages_sampled=pages_sampled,
            parser_error=parser_error,
        )

    routed: list[RoutedPDF] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_classify_one, item) for item in work]
        for future in as_completed(futures):
            routed.append(future.result())

    routed.sort(key=lambda entry: (entry.lane, entry.campus, str(entry.pdf_path)))
    return routed


def allocate_lane_slots(
    total_slots: int,
    lane_counts: dict[str, int],
    *,
    kimi_slots: int | None = None,
    qwen_slots: int | None = None,
) -> dict[str, int]:
    """Allocate cloud slot counts across non-empty lanes."""
    if total_slots < 1:
        raise ValueError("total_slots must be >= 1")

    lanes = ("kimi", "qwen")
    counts = {lane: max(0, int(lane_counts.get(lane, 0))) for lane in lanes}
    active = [lane for lane, count in counts.items() if count > 0]
    slots = {lane: 0 for lane in lanes}
    if not active:
        return slots

    explicit = {"kimi": kimi_slots, "qwen": qwen_slots}
    if any(value is not None for value in explicit.values()):
        assigned = 0
        for lane, value in explicit.items():
            if value is None or counts[lane] == 0:
                continue
            slots[lane] = max(0, int(value))
            assigned += slots[lane]
        if assigned > total_slots:
            raise ValueError("Explicit lane slots exceed total_slots")
        remaining_lanes = [lane for lane in active if slots[lane] == 0]
        remaining_slots = total_slots - assigned
        if remaining_lanes and remaining_slots > 0:
            auto_counts = {lane: counts[lane] for lane in remaining_lanes}
            auto_slots = allocate_lane_slots(remaining_slots, auto_counts)
            for lane in remaining_lanes:
                slots[lane] = auto_slots[lane]
        return slots

    if total_slots <= len(active):
        ranked = sorted(active, key=lambda lane: (-counts[lane], lane))
        for lane in ranked[:total_slots]:
            slots[lane] = 1
        return slots

    total_active = sum(counts[lane] for lane in active)
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for lane in active:
        exact = total_slots * (counts[lane] / total_active)
        base = math.floor(exact)
        slots[lane] = base
        assigned += base
        remainders.append((exact - base, lane))

    leftover = total_slots - assigned
    for _fraction, lane in sorted(remainders, key=lambda item: (-item[0], item[1]))[:leftover]:
        slots[lane] += 1

    if total_slots >= len(active):
        zero_lanes = [lane for lane in active if slots[lane] == 0]
        donors = [lane for lane in active if slots[lane] > 1]
        for lane in zero_lanes:
            if not donors:
                break
            donor = sorted(donors, key=lambda item: (-slots[item], -counts[item], item))[0]
            slots[donor] -= 1
            slots[lane] += 1
            donors = [item for item in active if slots[item] > 1]
    return slots


def shard_lane_routes(routes: list[RoutedPDF], shard_count: int) -> list[list[RoutedPDF]]:
    """Split one lane's routed PDFs into deterministic shards."""
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    shards = [[] for _ in range(shard_count)]
    for idx, entry in enumerate(sorted(routes, key=lambda item: (item.campus, str(item.pdf_path)))):
        shards[idx % shard_count].append(entry)
    return shards


def build_hybrid_launch_plan(
    project_root: Path,
    *,
    config: PipelineConfig,
    total_slots: int = 10,
    kimi_slots: int | None = None,
    qwen_slots: int | None = None,
    route_unknown: str = "qwen",
    classification_workers: int = 6,
    campuses: Iterable[str] | None = None,
    limit: int | None = None,
    run_label: str | None = None,
) -> HybridLaunchPlan:
    """Classify the corpus and prepare cloud-lane shard plans."""
    items = collect_downloaded_source_pdfs(project_root, campuses=campuses, limit=limit)
    routes = classify_and_route_pdfs(
        items,
        config=config,
        classification_workers=classification_workers,
        route_unknown=route_unknown,
    )
    lane_counts = {
        "kimi": sum(1 for route in routes if route.lane == "kimi"),
        "qwen": sum(1 for route in routes if route.lane == "qwen"),
    }
    slot_counts = allocate_lane_slots(
        total_slots,
        lane_counts,
        kimi_slots=kimi_slots,
        qwen_slots=qwen_slots,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = run_label or f"hybrid-cloud-{timestamp}"
    run_root = project_root / "logs" / "pdf_hybrid_cloud" / slug
    manifest_dir = run_root / "manifests"
    pid_dir = run_root / "pids"

    shards: list[LaneShardPlan] = []
    for lane in ("kimi", "qwen"):
        lane_routes = [route for route in routes if route.lane == lane]
        for shard_index, shard_routes in enumerate(shard_lane_routes(lane_routes, max(1, slot_counts[lane]))):
            manifest_path = manifest_dir / f"{lane}-shard{shard_index}.json"
            log_path = run_root / f"{lane}-shard{shard_index}.log"
            pid_path = pid_dir / f"{lane}-shard{shard_index}.pid"
            shards.append(
                LaneShardPlan(
                    lane=lane,
                    model=LANE_MODELS[lane],
                    shard_index=shard_index,
                    file_count=len(shard_routes),
                    manifest_path=manifest_path,
                    log_path=log_path,
                    pid_path=pid_path,
                )
            )

    return HybridLaunchPlan(
        project_root=project_root,
        run_root=run_root,
        total_files=len(routes),
        lane_counts=lane_counts,
        slot_counts=slot_counts,
        routes=tuple(routes),
        shards=tuple(shards),
    )


def write_hybrid_plan_artifacts(plan: HybridLaunchPlan) -> None:
    """Write route + shard manifests for a hybrid launch plan."""
    plan.run_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = plan.run_root / "manifests"
    pid_dir = plan.run_root / "pids"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)

    route_map: dict[tuple[str, int], list[RoutedPDF]] = {}
    for lane in ("kimi", "qwen"):
        lane_routes = [route for route in plan.routes if route.lane == lane]
        for shard_index, shard_routes in enumerate(shard_lane_routes(lane_routes, max(1, plan.slot_counts[lane]))):
            route_map[(lane, shard_index)] = shard_routes

    for shard in plan.shards:
        entries = route_map[(shard.lane, shard.shard_index)]
        payload = {
            "lane": shard.lane,
            "model": shard.model,
            "file_count": len(entries),
            "files": [
                {
                    "campus": entry.campus,
                    "pdf_path": str(entry.pdf_path),
                    "classification": entry.classification,
                    "char_count": entry.char_count,
                    "pages_sampled": entry.pages_sampled,
                    "parser_error": entry.parser_error,
                }
                for entry in entries
            ],
        }
        shard.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(plan.run_root),
        "total_files": plan.total_files,
        "lane_counts": dict(plan.lane_counts),
        "slot_counts": dict(plan.slot_counts),
        "shards": [
            {
                "lane": shard.lane,
                "model": shard.model,
                "shard_index": shard.shard_index,
                "file_count": shard.file_count,
                "manifest_path": str(shard.manifest_path),
                "log_path": str(shard.log_path),
                "pid_path": str(shard.pid_path),
            }
            for shard in plan.shards
        ],
    }
    (plan.run_root / "run_plan.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def active_batch_remediation_pids(project_root: Path) -> list[int]:
    """Return active rerun worker PIDs for this repo, if any."""
    pattern = f"{project_root}/scripts/batch_remediate.py"
    proc = subprocess.run(
        ["pgrep", "-f", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        token = line.strip()
        if token.isdigit():
            pids.append(int(token))
    return sorted(set(pids))


def lane_environment(
    base_env: dict[str, str],
    *,
    config: PipelineConfig,
    model: str,
    workers_per_process: int,
) -> dict[str, str]:
    """Return env overrides for one cloud lane."""
    env = dict(base_env)
    env.update(
        {
            "LLM_BACKEND": "ollama",
            "OLLAMA_BASE_URL": _CLOUD_BASE_URL,
            "OLLAMA_CLUSTER_NODES": _CLOUD_BASE_URL,
            "VISION_BASE_URL": _CLOUD_BASE_URL,
            "VISION_CLUSTER_NODES": _CLOUD_BASE_URL,
            "OLLAMA_VISION_MODEL": model,
            "OLLAMA_TEXT_MODEL": model,
            "ESCALATION_BACKEND": "ollama",
            "ESCALATION_BASE_URL": _CLOUD_BASE_URL,
            "ESCALATION_MODEL": model,
            "LITEPARSE_ENABLED": "true",
            "LITEPARSE_BIN": str(config.api.liteparse_bin or "lit"),
            "PDF_REMEDIATION_OUTPUT_FORMAT": "pdf",
            "MAX_CONCURRENT_API_CALLS": str(max(1, workers_per_process)),
            "OLLAMA_VISION_MAX_INFLIGHT": "1",
            "OLLAMA_ESCALATION_MAX_INFLIGHT": "1",
            "PDF_FIGURE_ALT_MAX_INFLIGHT": "1",
        }
    )
    if config.api.api_key and not env.get("OLLAMA_API_KEY"):
        env["OLLAMA_API_KEY"] = str(config.api.api_key)
    return env


def launch_hybrid_plan(
    plan: HybridLaunchPlan,
    *,
    config: PipelineConfig,
    python_bin: str,
    workers_per_process: int = 1,
    resume: bool = False,
    recheck: bool = False,
    thorough: bool = False,
    force: bool = False,
) -> list[subprocess.Popen]:
    """Launch one batch_remediate process per non-empty cloud shard."""
    active = active_batch_remediation_pids(plan.project_root)
    if active and not force:
        raise RuntimeError(
            "Active batch_remediate workers already running. Stop them first or pass force=True."
        )

    write_hybrid_plan_artifacts(plan)

    current_link = plan.project_root / "logs" / "pdf_hybrid_cloud" / "current"
    try:
        if current_link.is_symlink() or current_link.exists():
            current_link.unlink()
        current_link.symlink_to(plan.run_root, target_is_directory=True)
    except Exception:
        pass

    launches: list[subprocess.Popen] = []
    for shard in plan.shards:
        if shard.file_count <= 0:
            continue
        shard.log_path.parent.mkdir(parents=True, exist_ok=True)
        shard.pid_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            python_bin,
            str(plan.project_root / "scripts" / "batch_remediate.py"),
            "--manifest",
            str(shard.manifest_path),
            "--workers",
            str(max(1, workers_per_process)),
        ]
        if resume:
            command.append("--resume")
        if recheck:
            command.append("--recheck")
        if thorough:
            command.append("--thorough")

        env = lane_environment(
            os.environ,
            config=config,
            model=shard.model,
            workers_per_process=workers_per_process,
        )
        log_handle = open(shard.log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            command,
            cwd=plan.project_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        shard.pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
        launches.append(proc)
    return launches
