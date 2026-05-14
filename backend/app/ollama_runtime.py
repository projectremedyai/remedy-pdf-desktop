from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

import httpx

from backend.app.config import settings

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class LocalOllamaStatus:
    reachable: bool
    installed: bool
    model_tag: str
    endpoint: str
    models_dir: Path
    size_bytes: int | None = None
    error: str | None = None


def ollama_base_url() -> str:
    return str(settings.ollama_base_url).strip().rstrip("/")


def ollama_native_base_url() -> str:
    base_url = ollama_base_url()
    if base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url


def get_local_ollama_status(
    timeout_seconds: float = _HEALTH_TIMEOUT_SECONDS,
    model_tag: str | None = None,
) -> LocalOllamaStatus:
    endpoint = ollama_base_url()
    target_model = (model_tag or settings.ollama_model_tag).strip()
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(f"{ollama_native_base_url()}/api/tags")
            response.raise_for_status()
    except Exception as exc:
        logger.info("Local Ollama runtime unavailable at %s: %s", endpoint, exc)
        return LocalOllamaStatus(
            reachable=False,
            installed=False,
            model_tag=target_model,
            endpoint=endpoint,
            models_dir=settings.ollama_models_dir,
            error=str(exc),
        )

    try:
        payload = response.json()
    except Exception as exc:
        logger.warning("Failed to parse Ollama /api/tags response: %s", exc)
        return LocalOllamaStatus(
            reachable=True,
            installed=False,
            model_tag=target_model,
            endpoint=endpoint,
            models_dir=settings.ollama_models_dir,
            error=f"Invalid Ollama tags response: {exc}",
        )

    installed = False
    size_bytes: int | None = None
    for model in payload.get("models", []):
        if str(model.get("name", "")).strip() != target_model:
            continue
        installed = True
        raw_size = model.get("size")
        try:
            size_bytes = int(raw_size) if raw_size is not None else None
        except (TypeError, ValueError):
            size_bytes = None
        break

    return LocalOllamaStatus(
        reachable=True,
        installed=installed,
        model_tag=target_model,
        endpoint=endpoint,
        models_dir=settings.ollama_models_dir,
        size_bytes=size_bytes,
    )


def local_ollama_model_ready(timeout_seconds: float = _HEALTH_TIMEOUT_SECONDS) -> bool:
    status = get_local_ollama_status(timeout_seconds=timeout_seconds)
    return status.reachable and status.installed


def wait_for_ollama_ready(
    max_seconds: float = 60.0,
    poll_interval: float = 1.0,
    model_tag: str | None = None,
) -> LocalOllamaStatus:
    """Poll Ollama until reachable + model installed, or until timeout.

    The Rust launcher spawns Ollama and the Python backend in parallel, so when
    a remediation job arrives Ollama may still be booting / loading the model.
    Without this wait, early vision calls hit `connection refused` three times
    in a row and get recorded as real accessibility failures. Poll-until-ready
    eliminates that cold-start race.
    """
    deadline = time.monotonic() + max_seconds
    status = get_local_ollama_status(timeout_seconds=3.0, model_tag=model_tag)
    if status.reachable and status.installed:
        return status
    logger.info(
        "Waiting for Ollama to become ready at %s (model=%s, up to %.0fs)",
        status.endpoint,
        status.model_tag,
        max_seconds,
    )
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        status = get_local_ollama_status(timeout_seconds=3.0, model_tag=model_tag)
        if status.reachable and status.installed:
            logger.info("Ollama ready at %s", status.endpoint)
            return status
    logger.warning(
        "Ollama did not become ready within %.0fs (reachable=%s installed=%s)",
        max_seconds,
        status.reachable,
        status.installed,
    )
    return status


def build_local_vision_provider(
    timeout_seconds: float = _HEALTH_TIMEOUT_SECONDS,
    *,
    wait_seconds: float = 0.0,
    model_tag: str | None = None,
):
    """Build the vision provider, optionally waiting for Ollama to boot.

    Pass ``wait_seconds`` > 0 to poll the runtime until it's ready before
    probing — use this at the start of a remediation job so the pipeline
    doesn't race Ollama's startup.
    """
    if wait_seconds > 0:
        status = wait_for_ollama_ready(max_seconds=wait_seconds, model_tag=model_tag)
    else:
        status = get_local_ollama_status(
            timeout_seconds=timeout_seconds,
            model_tag=model_tag,
        )
    if not (status.reachable and status.installed):
        if status.error:
            logger.info("Local Ollama vision provider unavailable: %s", status.error)
        return None

    from project_remedy.pdf_vision import OllamaVisionProvider

    return OllamaVisionProvider(
        base_url=status.endpoint,
        api_key="ollama",
        model=status.model_tag,
    )


async def stream_model_pull_events(
    model_name: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream Ollama pull progress for a local model."""
    native_base_url = ollama_native_base_url()
    timeout = httpx.Timeout(None, connect=5.0, read=None, write=30.0, pool=5.0)
    success = False
    target_model = (model_name or settings.ollama_model_tag).strip()

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{native_base_url}/api/pull",
            json={"name": target_model, "stream": True},
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                payload = json.loads(line)
                if "error" in payload:
                    yield {"error": str(payload["error"])}
                    return

                event: dict[str, Any] = {}
                status_text = str(payload.get("status", "") or "").strip()
                if status_text:
                    event["status"] = status_text
                    if status_text.lower() == "success":
                        success = True

                total = payload.get("total")
                completed = payload.get("completed")
                try:
                    if total is not None:
                        event["total_mb"] = round(int(total) / 1e6, 1)
                    if completed is not None:
                        event["downloaded_mb"] = round(int(completed) / 1e6, 1)
                except (TypeError, ValueError):
                    pass

                if payload.get("digest"):
                    event["digest"] = str(payload["digest"])

                if success:
                    event["done"] = True
                    yield event
                    return

                if event:
                    yield event

    status = get_local_ollama_status(
        timeout_seconds=_HEALTH_TIMEOUT_SECONDS,
        model_tag=target_model,
    )
    if status.installed:
        yield {"done": True}
    else:
        yield {"error": "Model pull finished without confirming installation"}
