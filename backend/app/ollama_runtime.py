from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any, Callable

import httpx

from backend.app.config import settings

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT_SECONDS = 5.0

_TASK_ALIASES = {
    "alt": "alt_text_quality",
    "alt_text": "alt_text_quality",
    "alt_text_quality": "alt_text_quality",
    "figure_alt": "alt_text_quality",
    "figures": "alt_text_quality",
    "contrast": "contrast",
    "color_contrast": "contrast",
    "heading": "heading_hierarchy",
    "headings": "heading_hierarchy",
    "heading_hierarchy": "heading_hierarchy",
    "reading": "reading_order",
    "reading_order": "reading_order",
    "table": "table_structure",
    "tables": "table_structure",
    "table_structure": "table_structure",
    "core_layout": "core_layout",
}


@dataclass(frozen=True)
class LocalOllamaStatus:
    reachable: bool
    installed: bool
    model_tag: str
    endpoint: str
    models_dir: Path
    size_bytes: int | None = None
    error: str | None = None


def _normalise_task_name(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    return _TASK_ALIASES.get(key, key)


def _parse_task_map(raw: str | None) -> dict[str, str]:
    """Parse comma-separated ``task:value`` pairs used for router aliases."""
    result: dict[str, str] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        task, value = item.split(":", 1)
        task_name = _normalise_task_name(task)
        cleaned_value = value.strip()
        if task_name and cleaned_value:
            result[task_name] = cleaned_value
    return result


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _infer_vision_task(prompt: str) -> str | None:
    """Infer a Remedy vision task from stable prompt markers.

    Existing library calls pass only image + prompt to the provider, so the
    desktop router keeps the integration small by classifying the prompt. The
    checks below deliberately use task-specific phrases, not broad words like
    "table" from the shared layout rules.
    """
    text = str(prompt or "").lower()

    if (
        "verifying image alt text quality" in text
        or "current figure tags" in text
        or "figure_index" in text
        or "write specific, descriptive alt text" in text
        or "your previous description was too generic" in text
        or "describe each distinct image or graphic" in text
        or "wcag 1.1.1" in text
        and ("alt text" in text or "non-text content" in text)
    ):
        return "alt_text_quality"

    if (
        "verifying a data table" in text
        or "current table structure from pdf tags" in text
        or "table structure verification" in text
        or "fix_table_headers" in text
        or "describe this table for a screen reader summary" in text
    ):
        return "table_structure"

    if (
        "verifying heading hierarchy" in text
        or "visual heading hierarchy" in text
        or "heading_corrections" in text
        or "correct_tag" in text
        or "heading levels are appropriate" in text
        or "heading hierarchy mismatches" in text
    ):
        return "heading_hierarchy"

    if (
        "structure-tree order" in text
        or "structure tree order" in text
        or "logical reading order" in text
        or "reading_order" in text
        or "order_changed" in text
        or "visual top-to-bottom" in text
    ):
        return "reading_order"

    if (
        "checking color contrast" in text
        or "color contrast issues" in text
        or "contrast_issues" in text
        or "wcag 1.4.3" in text
        or "estimated_ratio" in text
    ):
        return "contrast"

    return None


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


class TaskRoutedOllamaVisionProvider:
    """Prompt-routed wrapper for Remedy's per-task MiniCPM adapters."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        default_model: str,
        task_models: dict[str, str],
        task_base_urls: dict[str, str] | None = None,
        provider_factory: Callable[..., Any] | None = None,
    ) -> None:
        from project_remedy.pdf_vision import OllamaVisionProvider

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = default_model
        self.task_models = {
            _normalise_task_name(task): model
            for task, model in task_models.items()
            if str(model or "").strip()
        }
        self.task_base_urls = {
            _normalise_task_name(task): url.rstrip("/")
            for task, url in (task_base_urls or {}).items()
            if str(url or "").strip()
        }
        self._provider_factory = provider_factory or OllamaVisionProvider
        self._providers: dict[tuple[str, str], Any] = {}
        self._default_provider = self._provider_for(self.model, self.base_url)
        self.last_base_url = self.base_url

    @property
    def node_urls(self) -> tuple[str, ...]:
        return tuple(sorted({base for _model, base in self._providers}))

    def _provider_for(self, model: str, base_url: str) -> Any:
        key = (model, base_url.rstrip("/"))
        provider = self._providers.get(key)
        if provider is None:
            provider = self._provider_factory(
                base_url=key[1],
                api_key=self.api_key,
                model=key[0],
            )
            self._providers[key] = provider
        return provider

    def model_for_prompt(self, prompt: str) -> str:
        task = _infer_vision_task(prompt)
        if task == "core_layout" and task not in self.task_models:
            task = "reading_order"
        return self.task_models.get(task or "", self.model)

    async def analyze_image(
        self,
        image_path: Path | None,
        prompt: str,
        *,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        task = _infer_vision_task(prompt)
        if task == "core_layout" and task not in self.task_models:
            task = "reading_order"

        model = self.task_models.get(task or "", self.model)
        base_url = self.task_base_urls.get(task or "", self.base_url)
        provider = (
            self._default_provider
            if model == self.model and base_url == self.base_url
            else self._provider_for(model, base_url)
        )
        response = await provider.analyze_image(
            image_path,
            prompt,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        self.last_base_url = str(getattr(provider, "last_base_url", base_url))
        return response


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

    task_models = _parse_task_map(os.getenv("OLLAMA_VISION_TASK_MODELS"))
    if task_models:
        missing: list[str] = []
        for task, task_model in task_models.items():
            task_status = get_local_ollama_status(
                timeout_seconds=timeout_seconds,
                model_tag=task_model,
            )
            if not (task_status.reachable and task_status.installed):
                missing.append(f"{task}:{task_model}")
        if missing and not _env_true("OLLAMA_VISION_ROUTER_ALLOW_FALLBACK"):
            logger.warning(
                "Local routed vision provider unavailable; task aliases missing: %s",
                ", ".join(missing),
            )
            return None
        if missing:
            logger.warning(
                "Local routed vision provider will fall back for missing aliases: %s",
                ", ".join(missing),
            )

        return TaskRoutedOllamaVisionProvider(
            base_url=status.endpoint,
            api_key="ollama",
            default_model=status.model_tag,
            task_models=task_models,
            task_base_urls=_parse_task_map(os.getenv("OLLAMA_VISION_TASK_BASE_URLS")),
        )

    return OllamaVisionProvider(base_url=status.endpoint, api_key="ollama", model=status.model_tag)


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
