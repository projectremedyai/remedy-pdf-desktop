"""Pipeline runner and in-memory job store.

Remediation pipeline:
  1. XY-Cut++ pre-pass — deterministic reading order (zero AI, zero API calls)
  2. fix_and_verify() — 240+ accessibility fixes with optional vision escalation
  3. generate_document_report() — ACR generation
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
import logging
import os
import time
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app import vision_settings as vision_settings_store
from backend.app.config import settings
from backend.app.ollama_runtime import build_local_vision_provider

logger = logging.getLogger(__name__)


def _apply_vision_settings_env() -> None:
    """Seed vision timeout env before project_remedy.pdf_fixer is imported."""
    try:
        current = vision_settings_store.load()
        os.environ["PDF_FIXER_VISION_PAGE_TIMEOUT"] = str(
            current.page_timeout_seconds
        )
    except Exception:
        logger.exception("Could not apply vision settings to environment")


_apply_vision_settings_env()


# Override the bundled library's batch-tuned manual-review threshold to match
# the single-user workflow this app serves. The library reads the constant
# at call time (line ~9291 of pdf_fixer.py), so overriding here at import
# time is sufficient — fix_and_verify picks up the new value on every call.
def _override_visual_diff_threshold() -> None:
    try:
        from project_remedy import pdf_fixer
        pdf_fixer.VISUAL_DIFF_MANUAL_REVIEW_THRESHOLD = settings.visual_diff_review_threshold
        logger.info(
            "visual-diff review threshold set to %.0f%%",
            settings.visual_diff_review_threshold * 100,
        )
    except Exception:
        logger.exception("Failed to override VISUAL_DIFF_MANUAL_REVIEW_THRESHOLD")


_override_visual_diff_threshold()


@lru_cache(maxsize=1)
def _load_pipeline_config():
    """Load the shared library config once for acceptance/report generation."""
    from project_remedy.config import load_config

    return load_config()


@dataclass
class XYCutResult:
    """Tracks XY-Cut++ pre-pass results per page."""
    pages_analyzed: int = 0
    pages_reordered: int = 0
    pages_skipped_single_column: int = 0
    layout_classes: dict[str, int] = field(default_factory=dict)


@dataclass
class Job:
    id: str
    filename: str
    input_path: Path
    verification_mode: str = "sampled"  # "sampled" | "full"
    output_path: Path | None = None
    report_path: Path | None = None
    report_data: dict[str, Any] | None = None
    fix_summary: dict[str, Any] | None = None
    xy_cut_result: dict[str, Any] | None = None
    reading_order_source: str = "none"  # "xy_cut" | "vision" | "vision_confirmed_xy_cut" | "none"
    status: str = "queued"  # queued, running, completed, failed
    current_step: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    # Experimental faithful rebuild (manually invoked post-remediation)
    rebuild_status: str = "idle"  # idle | running | success | failed
    rebuild_output_path: Path | None = None
    rebuild_visual_diff: float | None = None
    rebuild_pages: int | None = None
    rebuild_error: str | None = None


# Persistent store: jobs live in memory for speed, mirrored to disk as JSON
# so they survive backend restarts. Progress queues stay memory-only — they
# only matter during an active job.
_jobs: dict[str, Job] = {}
_progress_queues: dict[str, asyncio.Queue] = {}
_tasks: dict[str, asyncio.Task] = {}
_jobs_dir = settings.output_dir / "_jobs"


def _job_to_dict(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "filename": job.filename,
        "input_path": str(job.input_path),
        "verification_mode": job.verification_mode,
        "output_path": str(job.output_path) if job.output_path else None,
        "report_path": str(job.report_path) if job.report_path else None,
        "report_data": job.report_data,
        "fix_summary": job.fix_summary,
        "xy_cut_result": job.xy_cut_result,
        "reading_order_source": job.reading_order_source,
        "status": job.status,
        "current_step": job.current_step,
        "error": job.error,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "rebuild_status": job.rebuild_status,
        "rebuild_output_path": str(job.rebuild_output_path) if job.rebuild_output_path else None,
        "rebuild_visual_diff": job.rebuild_visual_diff,
        "rebuild_pages": job.rebuild_pages,
        "rebuild_error": job.rebuild_error,
    }


def _dict_to_job(data: dict[str, Any]) -> Job:
    return Job(
        id=data["id"],
        filename=data["filename"],
        input_path=Path(data["input_path"]),
        verification_mode=data.get("verification_mode", "sampled"),
        output_path=Path(data["output_path"]) if data.get("output_path") else None,
        report_path=Path(data["report_path"]) if data.get("report_path") else None,
        report_data=data.get("report_data"),
        fix_summary=data.get("fix_summary"),
        xy_cut_result=data.get("xy_cut_result"),
        reading_order_source=data.get("reading_order_source", "none"),
        status=data.get("status", "queued"),
        current_step=data.get("current_step", ""),
        error=data.get("error"),
        created_at=data.get("created_at", time.time()),
        completed_at=data.get("completed_at"),
        rebuild_status=data.get("rebuild_status", "idle"),
        rebuild_output_path=Path(data["rebuild_output_path"]) if data.get("rebuild_output_path") else None,
        rebuild_visual_diff=data.get("rebuild_visual_diff"),
        rebuild_pages=data.get("rebuild_pages"),
        rebuild_error=data.get("rebuild_error"),
    )


def _save_job(job: Job) -> None:
    import json
    try:
        _jobs_dir.mkdir(parents=True, exist_ok=True)
        path = _jobs_dir / f"{job.id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_job_to_dict(job), indent=2))
        tmp.replace(path)
    except Exception:
        logger.exception("Failed to persist job %s", job.id)


def _load_jobs_from_disk() -> None:
    import json
    if not _jobs_dir.exists():
        return
    for path in _jobs_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            job = _dict_to_job(data)
            # Resurrect interrupted jobs as failed — we can't resume mid-pipeline
            if job.status in ("queued", "running"):
                job.status = "failed"
                job.error = "Backend restarted before remediation finished"
            _jobs[job.id] = job
        except Exception:
            logger.exception("Failed to load job from %s", path)
    if _jobs:
        logger.info("Loaded %d jobs from disk", len(_jobs))


_load_jobs_from_disk()

# Concurrency gate — caps how many jobs run their heavy work at once.
# Jobs above the limit wait here (FIFO) so CPU/IO-bound remediation work
# doesn't thrash when multiple docs are processed simultaneously.
_job_slots: asyncio.Semaphore | None = None


def _get_job_slots() -> asyncio.Semaphore:
    """Lazy-init the job concurrency semaphore on the running loop."""
    global _job_slots
    if _job_slots is None:
        _job_slots = asyncio.Semaphore(settings.max_concurrent_jobs)
    return _job_slots


def create_job(filename: str, input_path: Path, verification_mode: str = "sampled") -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(
        id=job_id,
        filename=filename,
        input_path=input_path,
        verification_mode=verification_mode,
    )
    _jobs[job_id] = job
    _progress_queues[job_id] = asyncio.Queue()
    _save_job(job)
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def get_progress_queue(job_id: str) -> asyncio.Queue | None:
    return _progress_queues.get(job_id)


def get_latest_job() -> Job | None:
    if not _jobs:
        return None
    return max(_jobs.values(), key=lambda j: j.created_at)


def track_task(job_id: str, task: asyncio.Task) -> None:
    """Keep a reference to a running remediation task so it isn't GC'd."""
    _tasks[job_id] = task
    task.add_done_callback(lambda t: _tasks.pop(job_id, None))


def purge_expired_jobs(ttl_seconds: int) -> int:
    """Remove jobs older than *ttl_seconds* and delete their files.

    Returns the number of jobs removed.
    """
    import shutil

    now = time.time()
    expired = [
        jid for jid, job in _jobs.items()
        if job.status in ("completed", "failed")
        and (now - job.created_at) > ttl_seconds
    ]
    for jid in expired:
        job = _jobs.pop(jid, None)
        _progress_queues.pop(jid, None)
        if job:
            # Clean up input file
            if job.input_path and job.input_path.exists():
                job.input_path.unlink(missing_ok=True)
            # Clean up output directory
            output_dir = settings.output_dir / jid
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            (_jobs_dir / f"{jid}.json").unlink(missing_ok=True)
    return len(expired)


async def _emit(job_id: str, event: dict[str, Any]) -> None:
    q = _progress_queues.get(job_id)
    if q:
        await q.put(event)


async def _run_office_remediation(job, job_output_dir: Path, ext: str) -> None:
    """Remediate a DOCX/PPTX/XLSX file via the Office remediator."""
    from project_remedy.office_remediator import OfficeRemediator

    format_name = {".docx": "Word", ".pptx": "PowerPoint", ".xlsx": "Excel"}[ext]

    await _emit(job.id, {
        "type": "status", "step": "remediation", "status": "running",
        "message": f"Remediating {format_name} document...",
    })

    output_path = job_output_dir / f"{job.input_path.stem}_remediated{ext}"
    remediator = OfficeRemediator()

    result = await remediator.remediate(
        job.input_path,
        output_path,
        title=job.input_path.stem,
    )

    for change in result.changes:
        await _emit(job.id, {
            "type": "status", "step": "remediation", "status": "running",
            "message": change,
        })

    if not result.success:
        raise RuntimeError(result.error_message or "Office remediation failed")

    job.output_path = output_path
    job.status = "completed"
    job.current_step = ""
    job.fix_summary = {
        "fixed_count": len(result.changes),
        "skipped_count": 0,
        "needs_manual_review": False,
        "manual_review_reason": "",
        "gs_was_used": False,
        "tier_used": "office",
        "escalation_attempted": False,
        "visual_diff_pct": 0.0,
    }
    job.report_data = {
        "document_name": job.filename,
        "conformance": "Partially Conformant",
        "tag_count": 0,
        "pages": 0,
        "failed_checks": [],
        "reviewable_checks": [],
        "manual_review_checks": [],
        "source_limited_issues": [],
        "source_limited_count": 0,
        "sr_issues": [],
        "wcag_results": [],
        "wcag_failures": [],
        "wcag_manual_reviews": [],
        "wcag_pass_count": 0,
        "wcag_fail_count": 0,
        "wcag_review_count": 0,
        "total_issues": 0,
        "screen_reader_readability": 0.0,
        "verapdf_checked": False,
        "verapdf_passed": None,
        "verapdf_violation_count": 0,
        "verification_coverage": {
            "mode": job.verification_mode,
            "vision_checked": False,
            "total_pages": 0,
            "analyzed_page_count": 0,
            "analyzed_pages": [],
            "covers_all_pages": False,
            "unresolved_sampled_checks": [],
        },
    }

    await _emit(job.id, {
        "type": "status", "step": "remediation", "status": "completed",
        "message": f"{format_name} remediation complete — {len(result.changes)} changes applied",
    })
    await _emit(job.id, {"type": "complete"})


def _pick_artifact_check_pages(
    page_count: int,
    *,
    verification_mode: str,
    explicit_cap: int,
    max_pages: int,
) -> list[int]:
    """Return 1-based page numbers to submit for visual-artifact review.

    ``explicit_cap`` > 0 pins the count exactly (legacy behaviour — users
    who set ``VISUAL_ARTIFACT_CHECK_PAGES=N`` get N pages). When it's 0,
    coverage auto-scales:

    * deterministic / full verification → every page up to ``max_pages``
    * sampled verification → ~15% of pages bounded to [3, min(max_pages, 20)]

    Selected indices are spread evenly across the document so mid-document
    artifacts aren't invisible between the first and last page — the old
    "first / middle / last only" coverage is what REMEDY-69 #4 flagged.
    """
    if page_count <= 0:
        return []

    mode = (verification_mode or "sampled").lower()

    if explicit_cap > 0:
        target = min(explicit_cap, page_count)
    elif mode in ("deterministic", "full"):
        target = min(page_count, max(1, max_pages))
    else:
        # ~15% of pages, rounded up, bounded to a sensible sample range.
        auto = max(3, (page_count + 6) // 7)
        upper = max(3, min(max_pages, 20))
        target = min(page_count, min(auto, upper))

    if target <= 0:
        return []
    if target >= page_count:
        return list(range(1, page_count + 1))
    if target == 1:
        return [1]
    if target == 2:
        return [1, page_count]

    # Evenly distributed including first and last. Deduplicate in case
    # rounding collides on very short documents.
    step = (page_count - 1) / (target - 1)
    indices = {max(1, min(page_count, int(round(1 + i * step)))) for i in range(target)}
    return sorted(indices)


async def _vision_artifact_check(
    output_pdf: Path,
    vision_provider,
    page_count: int,
    *,
    verification_mode: str = "sampled",
) -> dict[str, Any] | None:
    """Ask the vision model to spot rendering artifacts on selected pages.

    Coverage scales with document length and verification mode (see
    ``_pick_artifact_check_pages``), so mid-document artifacts aren't
    invisible just because first / middle / last pages looked clean
    (REMEDY-69 #4).

    Best-effort: a failure here (rate limit, JSON parse issue, anything)
    returns a dict with ``checked=False`` so the caller can surface "we
    tried and couldn't tell" instead of either lying about clean output
    or blocking the job.
    """
    if vision_provider is None or page_count <= 0:
        return None

    page_indices = _pick_artifact_check_pages(
        page_count,
        verification_mode=verification_mode,
        explicit_cap=int(settings.visual_artifact_check_pages),
        max_pages=int(settings.visual_artifact_check_max_pages),
    )
    if not page_indices:
        return None

    page_timeout = float(os.getenv("PDF_FIXER_VISION_PAGE_TIMEOUT", "90"))

    try:
        from project_remedy.pdf_vision import render_page_to_image
    except ImportError:
        return None

    prompt = (
        "You are reviewing a remediated PDF page for visual rendering problems. "
        "Look specifically for: gray tints or washes across the whole page, "
        "stray black or gray rectangles that aren't part of the original content, "
        "missing or truncated text, garbled glyphs, or any other rendering "
        "artifact that a human would flag as 'this doesn't look right'.\n\n"
        "Reply with a single JSON object only, no prose:\n"
        '{"has_artifacts": bool, "description": "short description or empty string"}\n'
    )

    async def _check_one(page_idx: int) -> dict[str, Any]:
        try:
            image_path = await asyncio.to_thread(
                render_page_to_image, output_pdf, page_idx, 150,
            )
            raw = await asyncio.wait_for(
                vision_provider.analyze_image(
                    Path(image_path), prompt, max_tokens=200,
                ),
                timeout=page_timeout,
            )
            # Parse JSON (model sometimes wraps in ``` fences)
            import json, re
            text = raw.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Best effort: look for the first {...} block
                m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
                parsed = json.loads(m.group(0)) if m else {}
            return {
                "page": page_idx,
                "has_artifacts": bool(parsed.get("has_artifacts", False)),
                "description": str(parsed.get("description", "") or "").strip(),
            }
        except Exception as exc:
            logger.warning("Artifact check failed on page %d: %s", page_idx, exc)
            return {"page": page_idx, "error": str(exc)}

    page_results = await asyncio.gather(*[_check_one(p) for p in page_indices])
    flagged = [r for r in page_results if r.get("has_artifacts")]
    errors = [r for r in page_results if "error" in r]

    return {
        "checked": len(page_results) > len(errors),
        "pages_checked": [r["page"] for r in page_results if "error" not in r],
        "has_artifacts": bool(flagged),
        "flagged_pages": flagged,
        "error_count": len(errors),
    }


def _build_vision_provider():
    """Return the user-selected vision provider.

    Local Ollama still waits for the bundled runtime to boot. Cloud providers
    are built from locally persisted settings and only receive page images
    when the user selected that provider.
    """
    current = vision_settings_store.load()
    os.environ["PDF_FIXER_VISION_PAGE_TIMEOUT"] = str(current.page_timeout_seconds)

    if current.provider == "openrouter":
        if not current.openrouter_api_key:
            logger.warning("OpenRouter vision selected but no API key is configured")
            return None
        try:
            from project_remedy.pdf_vision import OpenRouterVisionProvider

            logger.info("Vision provider: OpenRouter (%s)", current.openrouter_model)
            return OpenRouterVisionProvider(
                api_key=current.openrouter_api_key,
                model=(
                    current.openrouter_model
                    or vision_settings_store.DEFAULT_OPENROUTER_MODEL
                ),
            )
        except Exception:
            logger.exception("Failed to build OpenRouter vision provider; vision disabled")
            return None

    if current.provider == "ollama_cloud":
        if not current.ollama_cloud_api_key:
            logger.warning("Ollama Cloud selected but no API key is configured")
            return None
        try:
            from project_remedy.pdf_vision import OllamaVisionProvider

            logger.info("Vision provider: Ollama Cloud (%s)", current.ollama_cloud_model)
            return OllamaVisionProvider(
                base_url="https://ollama.com/v1",
                api_key=current.ollama_cloud_api_key,
                model=(
                    current.ollama_cloud_model
                    or vision_settings_store.DEFAULT_OLLAMA_CLOUD_MODEL
                ),
            )
        except Exception:
            logger.exception("Failed to build Ollama Cloud vision provider; vision disabled")
            return None

    try:
        provider = build_local_vision_provider(
            wait_seconds=float(settings.vision_ready_wait_seconds),
            model_tag=current.local_model,
        )
        if provider is None:
            logger.info(
                "Local Ollama model %s is not ready; using OCR/placeholder fallbacks",
                current.local_model,
            )
            return None
        logger.info("Vision provider: local Ollama (%s)", current.local_model)
        return provider
    except Exception:
        logger.exception("Failed to build local Ollama vision provider; vision disabled")
        return None


def _reading_order_needs_targeted_retry(acceptance: Any) -> bool:
    """Return True when vision specifically reported a reading-order no-pass."""
    vision = getattr(acceptance, "vision_result", None)
    if vision is None:
        return False
    issues = getattr(vision, "reading_order_issues", None) or []
    if not any(getattr(issue, "severity", "warning") == "error" for issue in issues):
        return False
    checker_report = getattr(acceptance, "checker_report", None)
    results = getattr(checker_report, "results", []) or []
    return any(
        result.rule_id == "doc-reading-order" and result.status == "Failed"
        for result in results
    )


def _run_xy_cut_prepass(pdf_path: Path) -> tuple[int, dict[str, Any]]:
    """Run XY-Cut++ deterministic reading order as a pre-pass.

    Opens the PDF with pikepdf, analyzes page layouts, applies XY-Cut++
    reordering to multi-column pages, saves the result.

    Returns (pages_reordered, metadata_dict).
    """
    import pikepdf
    from project_remedy.pdf_fixer import (
        _apply_xy_cut_reading_order,
        _analyze_page_layout,
        _build_page_structure_summary,
        LayoutClass,
    )

    meta: dict[str, Any] = {
        "pages_analyzed": 0,
        "pages_reordered": 0,
        "pages_skipped_single_column": 0,
        "layout_classes": {},
    }

    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        num_pages = len(pdf.pages)
        meta["pages_analyzed"] = num_pages

        # Build structure summary and analyze each page
        structure_summary = _build_page_structure_summary(pdf)
        analyses: dict[int, Any] = {}
        for page_idx in range(num_pages):
            try:
                analysis = _analyze_page_layout(
                    pdf,
                    page_idx,
                    structure_summary=structure_summary,
                )
            except Exception as exc:
                logger.warning(
                    "XY-Cut pre-pass skipped layout analysis for %s page %d: %s",
                    pdf_path.name,
                    page_idx + 1,
                    exc,
                )
                analysis = types.SimpleNamespace(layout_class=LayoutClass.SINGLE_COLUMN)
            analyses[page_idx] = analysis

            layout = analysis.layout_class
            meta["layout_classes"][layout] = meta["layout_classes"].get(layout, 0) + 1
            if layout == LayoutClass.SINGLE_COLUMN:
                meta["pages_skipped_single_column"] += 1

        # Apply XY-Cut++ with tuned beta for cross-layout detection
        pages_reordered = _apply_xy_cut_reading_order(pdf, analyses)
        meta["pages_reordered"] = pages_reordered

        if pages_reordered > 0:
            pdf.save(pdf_path)

    return pages_reordered, meta


async def run_remediation(job_id: str) -> None:
    """Run the full remediation pipeline for a job.

    Jobs acquire a slot on the global concurrency semaphore before doing
    any heavy work. Additional jobs wait FIFO and emit a ``queue`` status
    event so the UI can show "Waiting in queue...". The slot is released
    via a try/finally regardless of success or failure.
    """
    job = _jobs.get(job_id)
    if not job:
        return

    slots = _get_job_slots()
    waited = False
    if slots.locked() or slots._value <= 0:  # type: ignore[attr-defined]
        waited = True
        await _emit(job_id, {
            "type": "status", "step": "queue", "status": "running",
            "message": f"Waiting in queue (max {settings.max_concurrent_jobs} simultaneous remediations)...",
        })

    await slots.acquire()
    if waited:
        await _emit(job_id, {
            "type": "status", "step": "queue", "status": "completed",
            "message": "Starting remediation",
        })

    job.status = "running"
    job_output_dir = settings.output_dir / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    # Branch: Office formats (DOCX/PPTX/XLSX) use a different pipeline
    ext = job.input_path.suffix.lower()
    if ext in (".docx", ".pptx", ".xlsx"):
        try:
            await _run_office_remediation(job, job_output_dir, ext)
        except Exception as exc:
            logger.exception("Office remediation failed for %s", job_id)
            job.status = "failed"
            job.error = str(exc)
            await _emit(job_id, {"type": "error", "message": str(exc)})
        finally:
            slots.release()
            _progress_queues.pop(job_id, None)
            _save_job(job)
        return

    # Copy input to working copy so XY-Cut++ can modify in-place
    import shutil
    working_pdf = job_output_dir / f"{job.input_path.stem}_working.pdf"
    shutil.copy2(job.input_path, working_pdf)

    try:
        # ── Step 1: XY-Cut++ pre-pass (deterministic, zero API calls) ──
        job.current_step = "xy_cut"
        await _emit(job_id, {
            "type": "status", "step": "xy_cut", "status": "running",
            "message": "Analyzing page layouts and fixing reading order...",
        })

        pages_reordered, xy_meta = await asyncio.to_thread(
            _run_xy_cut_prepass, working_pdf,
        )

        job.xy_cut_result = xy_meta
        job.reading_order_source = "xy_cut" if pages_reordered > 0 else "none"

        await _emit(job_id, {
            "type": "status", "step": "xy_cut", "status": "completed",
            "pages_reordered": pages_reordered,
            "pages_analyzed": xy_meta["pages_analyzed"],
            "message": (
                f"Reading order fixed on {pages_reordered} page(s)"
                if pages_reordered > 0
                else "All pages already in correct reading order"
            ),
        })

        # ── Step 2: Full remediation (fix_and_verify) ──
        job.current_step = "remediation"
        await _emit(job_id, {
            "type": "status", "step": "remediation", "status": "running",
            "message": "Applying accessibility fixes...",
        })

        output_pdf = job_output_dir / f"{job.input_path.stem}_remediated.pdf"

        from project_remedy.pdf_fixer import fix_and_verify

        pipeline_config = _load_pipeline_config()
        vision_provider = _build_vision_provider()
        if vision_provider is not None:
            logger.info("Vision enabled for job %s", job_id)
            await _emit(job_id, {
                "type": "status", "step": "remediation", "status": "running",
                "message": "Local AI runtime loaded — running vision-assisted analysis",
            })

        import queue as stdlib_queue

        fix_progress_q: stdlib_queue.Queue = stdlib_queue.Queue()

        def _on_fix(rule_id: str, description: str, current: int, total: int):
            fix_progress_q.put(f"[{current}/{total}] {description}")

        async def _drain_fix_progress():
            while True:
                try:
                    msg = fix_progress_q.get_nowait()
                    await _emit(job_id, {
                        "type": "status", "step": "remediation", "status": "running",
                        "message": msg,
                    })
                except stdlib_queue.Empty:
                    pass
                await asyncio.sleep(0.3)

        drain_task = asyncio.create_task(_drain_fix_progress())

        full_verification = job.verification_mode == "full"

        # ── Tier 1: sampled or full-document vision verification ──
        fix_report = await asyncio.to_thread(
            fix_and_verify,
            pdf_path=working_pdf,
            output_path=output_pdf,
            config=pipeline_config,
            thorough=full_verification,
            max_cycles=3,
            conformance_repair=True,
            original_path=job.input_path,
            vision_provider_override=vision_provider,
            progress_callback=_on_fix,
        )

        drain_task.cancel()

        await _emit(job_id, {
            "type": "status", "step": "remediation", "status": "running",
            "message": f"Tier 1 complete — {fix_report.fixed_count} fixes applied, {fix_report.skipped_count} skipped",
        })

        # ── Stage 4 / Tier 2: escalate when Tier 1 leaves stubborn issues ──
        # Only worth the vision-call cost when we actually have a provider.
        # Trigger matches batch pipeline: manual-review flag OR many skips.
        tier_used = "full" if full_verification else "tier1"
        escalated = False
        import pikepdf as _pk
        with _pk.open(working_pdf) as _check_pdf:
            _page_count = len(_check_pdf.pages)
        if (
            not full_verification
            and vision_provider is not None
            and _page_count > 10
            and (
            fix_report.needs_manual_review or fix_report.skipped_count > 5
            )
        ):
            escalated = True
            job.current_step = "escalation"
            await _emit(job_id, {
                "type": "status", "step": "escalation", "status": "running",
                "message": "Tier 1 left issues — escalating with every-page vision analysis...",
            })

            escalated_pdf = job_output_dir / f"{job.input_path.stem}_remediated_t2.pdf"
            try:
                # Re-run from the original input with thorough=True so every
                # page is sent to the vision model (Tier 1 samples).
                tier2_report = await asyncio.to_thread(
                    fix_and_verify,
                    pdf_path=working_pdf,
                    output_path=escalated_pdf,
                    config=pipeline_config,
                    thorough=True,
                    max_cycles=3,
                    conformance_repair=True,
                    original_path=job.input_path,
                    vision_provider_override=vision_provider,
                )
            except Exception as exc:
                logger.warning("Tier 2 escalation failed for job %s: %s", job_id, exc)
                tier2_report = None

            # Keep whichever output wins on (resolved issues, then fixes
            # applied). Tier 1 stays the default on ties or on Tier 2 errors.
            def _score(rep) -> tuple[int, int, int]:
                # Higher is better: (not-needing-review, fixed_count, -skipped)
                return (
                    0 if rep.needs_manual_review else 1,
                    rep.fixed_count,
                    -rep.skipped_count,
                )

            if tier2_report is not None and _score(tier2_report) > _score(fix_report):
                # Replace the Tier 1 output with the Tier 2 output
                output_pdf.unlink(missing_ok=True)
                escalated_pdf.rename(output_pdf)
                fix_report = tier2_report
                tier_used = "tier2"
            else:
                # Discard the escalated attempt
                escalated_pdf.unlink(missing_ok=True)

            await _emit(job_id, {
                "type": "status", "step": "escalation", "status": "completed",
                "tier_used": tier_used,
                "message": (
                    f"Tier 2 escalation improved results (using tier2)"
                    if tier_used == "tier2"
                    else "Tier 2 did not improve on Tier 1 — keeping Tier 1"
                ),
            })

        job.output_path = output_pdf
        job.fix_summary = {
            "fixed_count": fix_report.fixed_count,
            "skipped_count": fix_report.skipped_count,
            "needs_manual_review": fix_report.needs_manual_review,
            "manual_review_reason": fix_report.manual_review_reason or "",
            "gs_was_used": fix_report.gs_was_used,
            "tier_used": tier_used,
            "escalation_attempted": escalated,
            "reading_order_retry_attempted": False,
            "reading_order_retry_used": False,
            # Confidence signal for the UI: a low diff % means the remediation
            # changed almost nothing visually. High values suggest something
            # worth the user eyeballing. Threshold above which we flag
            # needs_manual_review is `settings.visual_diff_review_threshold`.
            "visual_diff_pct": float(getattr(fix_report, "visual_diff_pct", 0.0) or 0.0),
        }

        # Update reading order source if vision was used
        if fix_report.gs_was_used and pages_reordered > 0:
            job.reading_order_source = "vision_confirmed_xy_cut"

        await _emit(job_id, {
            "type": "status", "step": "remediation", "status": "completed",
            "fixes_applied": fix_report.fixed_count,
            "fixes_skipped": fix_report.skipped_count,
            "tier_used": tier_used,
            "message": f"Remediation complete — {fix_report.fixed_count} issues fixed",
        })

        # ── Step 3: Generate ACR ──
        job.current_step = "report"
        await _emit(job_id, {
            "type": "status", "step": "report", "status": "running",
            "message": "Generating accessibility conformance report...",
        })
        await _emit(job_id, {
            "type": "status", "step": "report", "status": "running",
            "message": "Comparing original vs remediated document...",
        })

        report_dir = job_output_dir / "report"
        report_dir.mkdir(exist_ok=True)

        from project_remedy.compliance_report import generate_document_report
        from project_remedy.pdf_acceptance import evaluate_pdf_acceptance

        acceptance = await asyncio.to_thread(
            evaluate_pdf_acceptance,
            output_pdf,
            original_path=job.input_path,
            config=pipeline_config,
        )

        if _reading_order_needs_targeted_retry(acceptance) and vision_provider is not None:
            job.current_step = "reading_order_retry"
            job.fix_summary["reading_order_retry_attempted"] = True
            await _emit(job_id, {
                "type": "status",
                "step": "reading_order_retry",
                "status": "running",
                "message": "Reading order still needs review — running one full-document repair pass...",
            })
            retry_pdf = job_output_dir / f"{job.input_path.stem}_remediated_reading_order_retry.pdf"
            try:
                retry_report = await asyncio.to_thread(
                    fix_and_verify,
                    pdf_path=output_pdf,
                    output_path=retry_pdf,
                    config=pipeline_config,
                    thorough=True,
                    max_cycles=2,
                    conformance_repair=True,
                    original_path=job.input_path,
                    vision_provider_override=vision_provider,
                )
                retry_acceptance = await asyncio.to_thread(
                    evaluate_pdf_acceptance,
                    retry_pdf,
                    original_path=job.input_path,
                    config=pipeline_config,
                )
                if (
                    retry_pdf.exists()
                    and retry_acceptance.openable
                    and not _reading_order_needs_targeted_retry(retry_acceptance)
                ):
                    output_pdf.unlink(missing_ok=True)
                    retry_pdf.rename(output_pdf)
                    acceptance = retry_acceptance
                    fix_report.changes.extend(
                        f"Reading-order retry: {change}"
                        for change in retry_report.changes
                    )
                    fix_report.skipped.extend(
                        f"Reading-order retry: {skip}"
                        for skip in retry_report.skipped
                    )
                    job.fix_summary["reading_order_retry_used"] = True
                    job.fix_summary["fixed_count"] = fix_report.fixed_count
                    job.fix_summary["skipped_count"] = fix_report.skipped_count
                    tier_used = f"{tier_used}+reading-order-retry"
                    job.fix_summary["tier_used"] = tier_used
                    await _emit(job_id, {
                        "type": "status",
                        "step": "reading_order_retry",
                        "status": "completed",
                        "message": "Reading-order retry improved the final validation result.",
                    })
                else:
                    retry_pdf.unlink(missing_ok=True)
                    job.fix_summary["needs_manual_review"] = True
                    job.fix_summary["manual_review_reason"] = (
                        job.fix_summary.get("manual_review_reason")
                        or "Reading order still needs manual review after full-document retry."
                    )
                    await _emit(job_id, {
                        "type": "status",
                        "step": "reading_order_retry",
                        "status": "completed",
                        "message": "Reading-order retry did not clear the vision finding — keeping manual review.",
                    })
            except Exception as exc:
                retry_pdf.unlink(missing_ok=True)
                job.fix_summary["needs_manual_review"] = True
                job.fix_summary["manual_review_reason"] = (
                    job.fix_summary.get("manual_review_reason")
                    or "Reading order retry failed and still needs manual review."
                )
                logger.warning("Reading-order retry failed for job %s: %s", job_id, exc)
                await _emit(job_id, {
                    "type": "status",
                    "step": "reading_order_retry",
                    "status": "completed",
                    "message": "Reading-order retry failed — keeping manual review.",
                })

        # ── Post-remediation vision artifact sanity check ──
        # Run BEFORE report generation so its findings are persisted into the
        # downloadable HTML/JSON — not just surfaced to the UI.
        # Coverage auto-scales with doc length + verification mode unless the
        # user pinned VISUAL_ARTIFACT_CHECK_PAGES to an explicit count.
        # Best-effort — never fails the job.
        artifact_check_payload: dict[str, Any] | None = None
        # Gate: vision reachable AND either auto-scale is allowed (max>0) or
        # the user set an explicit positive cap.
        artifact_check_enabled = vision_provider is not None and (
            settings.visual_artifact_check_pages > 0
            or settings.visual_artifact_check_max_pages > 0
        )
        if artifact_check_enabled:
            job.current_step = "artifact_check"
            verification_mode = (
                "deterministic" if full_verification else "sampled"
            )
            await _emit(job_id, {
                "type": "status", "step": "artifact_check", "status": "running",
                "message": "Vision-checking rendered pages for artifacts...",
            })
            artifact_check_payload = await _vision_artifact_check(
                output_pdf,
                vision_provider,
                int(getattr(acceptance.checker_report, "page_count", 0) or 0),
                verification_mode=verification_mode,
            )
            if artifact_check_payload is not None:
                # Tag coverage mode for the report renderer so users know
                # whether vision saw every page or just a sample.
                artifact_check_payload["verification_mode"] = verification_mode
                job.fix_summary["visual_artifact_check"] = artifact_check_payload
                await _emit(job_id, {
                    "type": "status", "step": "artifact_check", "status": "completed",
                    "has_artifacts": artifact_check_payload.get("has_artifacts", False),
                    "pages_checked": len(artifact_check_payload.get("pages_checked", [])),
                })

        doc_report = await asyncio.to_thread(
            generate_document_report,
            original_path=job.input_path,
            remediated_path=output_pdf,
            output_dir=report_dir,
            config=pipeline_config,
            acceptance=acceptance,
            verification_mode=job.verification_mode,
            visual_artifact_check=artifact_check_payload,
        )

        job.report_path = report_dir / doc_report.report_filename if doc_report.report_filename else None

        # Summarize report for API
        failed_checks = [
            {"rule_id": c.get("rule_id"), "description": c.get("description"), "fixable": c.get("fixable")}
            for c in doc_report.check_results
            if c.get("status") == "Failed"
        ]
        manual_review_checks = [
            {
                "rule_id": c.get("rule_id"),
                "description": c.get("description"),
                "details": c.get("details", []),
                "decision": c.get("decision", "manual_review"),
                "recommendation": c.get("recommendation", ""),
            }
            for c in getattr(doc_report, "manual_review_checks", [])
        ]
        reviewable_checks = [
            {
                "rule_id": c.get("rule_id"),
                "description": c.get("description"),
                "details": c.get("details", []),
                "decision": c.get("decision", "manual_review"),
                "recommendation": c.get("recommendation", ""),
                "status": c.get("status"),
            }
            for c in getattr(doc_report, "reviewable_checks", [])
        ]
        source_limited_issues = [
            {
                "rule_id": issue.get("rule_id"),
                "description": issue.get("description"),
                "details": issue.get("details", []),
                "recommendation": issue.get("recommendation", ""),
            }
            for issue in getattr(doc_report, "source_limited_issues", [])
        ]
        sr_issues_grouped: dict[str, dict] = {}
        for sr in doc_report.sr_issues:
            rid = sr.get("rule_id", "")
            if rid not in sr_issues_grouped:
                sr_issues_grouped[rid] = {"rule_id": rid, "severity": sr.get("severity"), "count": 0}
            sr_issues_grouped[rid]["count"] += 1

        wcag_results = [
            {
                "criterion_id": w.criterion_id,
                "criterion_name": w.criterion_name,
                "level": w.level,
                "status": w.status,
                "remarks": w.remarks,
            }
            for w in doc_report.wcag_results
        ]
        wcag_failures = [w for w in wcag_results if w["status"] == "FAIL"]
        wcag_manual_reviews = [w for w in wcag_results if w["status"] == "REVIEW"]

        job.report_data = {
            "document_name": doc_report.document_name,
            "conformance": doc_report.conformance,
            "tag_count": doc_report.tag_count,
            "pages": doc_report.remediated_pages,
            "failed_checks": failed_checks,
            "reviewable_checks": reviewable_checks,
            "manual_review_checks": manual_review_checks,
            "source_limited_issues": source_limited_issues,
            "source_limited_count": len(source_limited_issues),
            "sr_issues": list(sr_issues_grouped.values()),
            "wcag_results": wcag_results,
            "wcag_failures": wcag_failures,
            "wcag_manual_reviews": wcag_manual_reviews,
            "wcag_pass_count": sum(1 for w in wcag_results if w["status"] == "PASS"),
            "wcag_fail_count": len(wcag_failures),
            "wcag_review_count": len(wcag_manual_reviews),
            "total_issues": len(doc_report.issues),
            "screen_reader_readability": doc_report.screen_reader_readability,
            "verapdf_checked": doc_report.verapdf_checked,
            "verapdf_passed": doc_report.verapdf_passed,
            "verapdf_violation_count": len(doc_report.verapdf_violations),
            "verification_coverage": doc_report.verification_coverage,
            "reading_order_source": job.reading_order_source,
            "xy_cut": job.xy_cut_result,
        }

        await _emit(job_id, {
            "type": "status", "step": "report", "status": "completed",
            "message": (
                "Report generated — "
                f"{job.report_data['wcag_pass_count']}/"
                f"{job.report_data['wcag_pass_count'] + job.report_data['wcag_fail_count']} "
                "auto-tested WCAG criteria passed, "
                f"{job.report_data['wcag_review_count']} need manual review"
            ),
        })

        # ── Done ──
        job.status = "completed"
        job.completed_at = time.time()
        job.current_step = ""
        await _emit(job_id, {
            "type": "complete",
            "conformance": doc_report.conformance,
            "fixes_applied": fix_report.fixed_count,
            "issues_remaining": len(doc_report.issues),
            "wcag_pass_count": job.report_data["wcag_pass_count"],
            "reading_order_source": job.reading_order_source,
        })

        # Clean up working copy
        working_pdf.unlink(missing_ok=True)

    except Exception as exc:
        logger.exception("Remediation failed for job %s", job_id)
        job.status = "failed"
        job.error = str(exc)
        job.current_step = ""
        working_pdf.unlink(missing_ok=True)
        await _emit(job_id, {"type": "error", "message": str(exc)})
    finally:
        # Always release the job slot so a failed/cancelled job doesn't
        # permanently shrink the pool the next user can draw from.
        slots.release()
        _save_job(job)
