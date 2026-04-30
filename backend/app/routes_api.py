"""REST API: upload, job status, download, model management."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.app.config import settings
from backend.app.ollama_runtime import get_local_ollama_status, stream_model_pull_events
from backend.app.repair_strategies import (
    describe_strategy_picker,
    get_current_strategy_output_path,
    release_strategy_lock,
    run_strategy_sequence,
    try_acquire_strategy_lock,
)
from backend.app.remediation import create_job, get_job, get_progress_queue, run_remediation, track_task

router = APIRouter(prefix="/api")

LOCAL_MODEL = {
    "size_gb": 3.4,
    "params": "4.66B",
    "ram_gb": 8,
}


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}
_MAGIC_BYTES = {
    ".pdf": b"%PDF-",
    # DOCX/PPTX/XLSX are ZIP-based
    ".docx": b"PK\x03\x04",
    ".pptx": b"PK\x03\x04",
    ".xlsx": b"PK\x03\x04",
}

_VALID_VERIFICATION_MODES = {"sampled", "full"}


def _basename(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name


def _safe_upload_name(filename: str) -> str:
    """Return a filesystem-safe basename while preserving the extension."""
    name = _basename(filename).replace("\x00", "").strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    return name[:180] or "upload"


@router.post("/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile,
    verification_mode: str = Form("sampled"),
) -> dict:
    """Accept a document upload, validate, create job, start remediation."""
    # Rate limit (per-IP)
    from backend.app.main import _check_rate_limit

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(429, "Upload rate limit exceeded — try again shortly")

    if not file.filename:
        raise HTTPException(400, "No filename provided")

    original_filename = _basename(file.filename)
    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    verification_mode = verification_mode.strip().lower()
    if verification_mode not in _VALID_VERIFICATION_MODES:
        raise HTTPException(400, "verification_mode must be 'sampled' or 'full'")

    # Validate magic bytes before writing to disk
    expected = _MAGIC_BYTES[ext]
    header = await file.read(len(expected))
    if header != expected:
        raise HTTPException(
            400,
            f"File does not appear to be a valid {ext.upper().lstrip('.')} document",
        )
    await file.seek(0)

    # Enforce upload size limit
    content_length = request.headers.get("content-length")
    try:
        declared_length = int(content_length) if content_length else 0
    except ValueError as exc:
        raise HTTPException(400, "Invalid Content-Length header") from exc
    if declared_length > settings.max_upload_bytes:
        raise HTTPException(
            413, f"File too large (max {settings.max_upload_bytes // (1024 * 1024)} MB)",
        )

    job_input_dir = settings.upload_dir
    job_input_dir.mkdir(parents=True, exist_ok=True)

    safe_original = _safe_upload_name(original_filename)
    if not safe_original.lower().endswith(ext):
        safe_original = f"{safe_original}{ext}"
    safe_name = f"{uuid.uuid4().hex[:8]}_{safe_original}"
    input_path = job_input_dir / safe_name

    bytes_written = 0
    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 256):
            bytes_written += len(chunk)
            if bytes_written > settings.max_upload_bytes:
                f.close()
                input_path.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"File too large (max {settings.max_upload_bytes // (1024 * 1024)} MB)",
                )
            f.write(chunk)

    page_count = 0
    if ext == ".pdf":
        try:
            import pikepdf
            with pikepdf.open(input_path) as pdf:
                page_count = len(pdf.pages)
            if page_count > settings.max_pages:
                input_path.unlink(missing_ok=True)
                raise HTTPException(400, f"PDF has {page_count} pages (max {settings.max_pages})")
        except pikepdf.PdfError as exc:
            input_path.unlink(missing_ok=True)
            raise HTTPException(400, f"Invalid PDF: {exc}")

    job = create_job(
        filename=original_filename,
        input_path=input_path,
        verification_mode=verification_mode,
    )
    task = asyncio.create_task(run_remediation(job.id))
    track_task(job.id, task)

    return {
        "job_id": job.id,
        "filename": original_filename,
        "pages": page_count,
        "verification_mode": verification_mode,
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """Get job status and report summary."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    result = {
        "id": job.id,
        "filename": job.filename,
        "status": job.status,
        "current_step": job.current_step,
        "error": job.error,
        "verification_mode": job.verification_mode,
        "reading_order_source": job.reading_order_source,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }
    if job.xy_cut_result:
        result["xy_cut"] = job.xy_cut_result
    if job.fix_summary:
        result["fix_summary"] = job.fix_summary
    if job.report_data:
        result["report"] = job.report_data
    return result


_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class StrategyRunRequest(BaseModel):
    """Payload for sequential post-remediation strategy execution."""

    strategies: list[str]


@router.get("/jobs/{job_id}/pdf")
async def download_pdf(job_id: str) -> FileResponse:
    """Download the remediated document (PDF, DOCX, PPTX, or XLSX)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed" or not job.output_path or not job.output_path.exists():
        raise HTTPException(404, "Remediated document not available yet")

    ext = job.output_path.suffix.lower()
    media = _MEDIA_TYPES.get(ext, "application/octet-stream")

    return FileResponse(
        job.output_path,
        media_type=media,
        filename=f"{job.input_path.stem}_remediated{ext}",
    )


@router.get("/jobs/{job_id}/report")
async def download_report(job_id: str) -> FileResponse:
    """Download the ACR report (HTML)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed" or not job.report_path or not job.report_path.exists():
        raise HTTPException(404, "Report not available yet")

    return FileResponse(
        job.report_path,
        media_type="text/html",
        filename=f"{job.input_path.stem}_acr.html",
    )


# --- Residual-driven post-remediation strategies ---

@router.get("/jobs/{job_id}/repair-strategies")
async def get_repair_strategies(job_id: str) -> dict:
    """List post-remediation strategies for the current residual output."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed" or not job.output_path or not job.output_path.exists():
        raise HTTPException(400, "Remediated document not available yet")

    picker = await asyncio.to_thread(describe_strategy_picker, job)
    if picker["latest_output_available"]:
        picker["download_url"] = f"/api/jobs/{job_id}/repair-output"
    return picker


@router.post("/jobs/{job_id}/repair-strategies/run")
async def run_repair_strategies(job_id: str, payload: StrategyRunRequest) -> dict:
    """Run the selected post-remediation strategies sequentially."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed" or not job.output_path or not job.output_path.exists():
        raise HTTPException(400, "Remediated document not available yet")
    if not try_acquire_strategy_lock(job_id):
        raise HTTPException(409, "A strategy run is already in progress for this job")

    try:
        picker = await asyncio.to_thread(run_strategy_sequence, job, payload.strategies)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        release_strategy_lock(job_id)

    picker["running"] = False
    if picker["latest_output_available"]:
        picker["download_url"] = f"/api/jobs/{job_id}/repair-output"
    return picker


@router.get("/jobs/{job_id}/repair-output")
async def download_repair_output(job_id: str) -> FileResponse:
    """Download the latest successful post-remediation strategy output."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    output_path = get_current_strategy_output_path(job)
    if not output_path or not output_path.exists():
        raise HTTPException(404, "Strategy output not available")

    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=f"{job.input_path.stem}_repair_strategies.pdf",
    )


# --- Progress WebSocket ---

@router.websocket("/ws/progress/{job_id}")
async def progress_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream remediation progress events for a job."""
    await websocket.accept()

    queue = get_progress_queue(job_id)
    if not queue:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    try:
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=settings.progress_ws_idle_timeout)
            await websocket.send_json(event)
            if event.get("type") in ("complete", "error"):
                break
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "Timeout waiting for progress"})
    except WebSocketDisconnect:
        pass


# --- Model management ---


@router.get("/model/status")
async def model_status() -> dict:
    """Check whether the local Ollama runtime and model are ready."""
    status = await asyncio.to_thread(get_local_ollama_status)
    return {
        "reachable": status.reachable,
        "installed": status.installed,
        "endpoint": status.endpoint,
        "models_dir": str(status.models_dir),
        "model_tag": status.model_tag,
        "size_mb": round((status.size_bytes or 0) / 1e6, 1),
        "default_model": {"tag": settings.ollama_model_tag, **LOCAL_MODEL},
        "error": status.error,
    }


@router.post("/model/download")
async def model_download() -> StreamingResponse:
    """Pull the default local model through the local Ollama daemon."""
    async def _stream():
        status = await asyncio.to_thread(get_local_ollama_status)
        if not status.reachable:
            yield f"data: {json.dumps({'error': status.error or 'Local Ollama runtime is not reachable'})}\n\n"
            return

        try:
            async for msg in stream_model_pull_events():
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("done") or msg.get("error"):
                    return
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
