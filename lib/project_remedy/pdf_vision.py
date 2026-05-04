"""Vision-model analysis for PDF accessibility checks.

Renders PDF pages to images and sends them to a vision model for
spatial analysis that can't be done with structure-tree inspection
alone — reading order validation and color contrast estimation.

Supports Ollama and OpenAI-compatible providers. Provider selection is done
at call time so users can mix and match.

Usage::

    analyzer = VisionAnalyzer(provider="ollama", model="qwen3.5:4b")
    results = await analyzer.analyze_reading_order(Path("doc.pdf"), pages=[1,2,3])
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pikepdf

from project_remedy.pdf_checker import walk_structure_tree, _get_struct_type
from project_remedy.token_tracker import tracker
from project_remedy.vision_prompts import (
    contrast_detection_prompt as build_contrast_detection_prompt,
    reading_order_prompt as build_reading_order_prompt,
)

logger = logging.getLogger(__name__)

_VISION_ANALYZER_PAGE_TIMEOUT = float(
    os.environ.get(
        "VISION_ANALYZER_PAGE_TIMEOUT",
        os.environ.get("PDF_FIXER_VISION_PAGE_TIMEOUT", "30"),
    )
)
_VISION_ANALYZER_TIMEOUT_ABORTS = int(
    os.environ.get(
        "VISION_ANALYZER_TIMEOUT_ABORTS",
        os.environ.get("PDF_FIXER_VISION_PAGE_TIMEOUT_ABORTS", "2"),
    )
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ReadingOrderIssue:
    """A single reading-order problem identified by the vision model."""

    page: int
    description: str
    severity: str = "warning"  # "error" | "warning" | "info"
    suggestion: str = ""


@dataclass
class ContrastIssue:
    """A color contrast problem identified by the vision model."""

    page: int
    description: str
    location: str = ""
    content_kind: str = "unknown"  # pdf_text | vector_text | image_text | diagram_artwork | unknown
    bbox: list[float] = field(default_factory=list)
    text_rgb: list[float] = field(default_factory=list)
    bg_rgb: list[float] = field(default_factory=list)
    fix_rgb: list[float] = field(default_factory=list)
    estimated_ratio: float | None = None
    required_ratio: float = 4.5
    auto_fixable: bool = False
    wcag_criterion: str = "1.4.3"


@dataclass
class UseOfColorIssue:
    """A WCAG 1.4.1 problem identified by the vision model."""

    page: int
    description: str
    location: str = ""
    suggestion: str = ""


@dataclass
class VisionCheckResult:
    """Result of vision-based analysis for one or more pages."""

    reading_order_issues: list[ReadingOrderIssue] = field(default_factory=list)
    contrast_issues: list[ContrastIssue] = field(default_factory=list)
    use_of_color_issues: list[UseOfColorIssue] = field(default_factory=list)
    raw_responses: dict[int, str] = field(default_factory=dict)
    analyzed_pages: list[int] = field(default_factory=list)  # 1-based page numbers
    total_pages: int = 0

    @property
    def reading_order_passed(self) -> bool:
        return not any(i.severity == "error" for i in self.reading_order_issues)

    @property
    def contrast_passed(self) -> bool:
        return len(self.contrast_issues) == 0

    @property
    def use_of_color_passed(self) -> bool:
        return len(self.use_of_color_issues) == 0

    @property
    def analyzed_page_count(self) -> int:
        return len(set(self.analyzed_pages))

    @property
    def covers_all_pages(self) -> bool:
        return self.total_pages > 0 and self.analyzed_page_count >= self.total_pages


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    floats: list[float] = []
    for item in value:
        try:
            floats.append(float(item))
        except (TypeError, ValueError):
            return []
    return floats


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_auto_fixable(value: Any, content_kind: str, fix_rgb: list[float]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return content_kind in {"pdf_text", "vector_text"} and len(fix_rgb) == 3


# ---------------------------------------------------------------------------
# Vision provider protocol
# ---------------------------------------------------------------------------


def _is_vision_unreachable(exc: BaseException) -> bool:
    """True when *exc* indicates the vision provider was unreachable.

    Used by callers to distinguish "model didn't respond" (a runtime
    availability problem that should NOT be reported as a document defect)
    from "model analyzed the page and found an issue" (a real finding).
    """
    try:
        import httpx

        if isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ),
        ):
            return True
    except Exception:
        pass
    # OllamaVisionProvider wraps exhausted retries in a RuntimeError whose
    # message includes the final cause verbatim. Match on that text as a
    # fallback for when the exception type has been stringified away.
    message = str(exc).lower()
    return (
        "all connection attempts failed" in message
        or "connection refused" in message
        or "connection reset" in message
        or "all vision attempts" in message
        or "name or service not known" in message
        or "temporary failure in name resolution" in message
    )


class VisionProvider(Protocol):
    """Minimal interface any vision provider must implement."""

    async def analyze_image(
        self,
        image_path: Path | None,
        prompt: str,
        *,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        """Send an image + prompt to the vision model and return the response."""
        ...


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


class OllamaVisionProvider:
    """Ollama (or any OpenAI-compatible) vision provider."""

    _gate_lock = threading.Lock()
    _endpoint_gates: dict[tuple[str, int], threading.BoundedSemaphore] = {}

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        base_urls: list[str] | None = None,
        api_key: str = "ollama",
        model: str = "llava",
        timeout_seconds: float = 300.0,
        max_inflight: int | None = None,
        stream: bool = False,
        reasoning_effort: str = "low",
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
    ) -> None:
        primary = base_url.rstrip("/")
        self._node_urls = []
        for url in (base_urls or [primary]):
            cleaned = str(url or "").strip().rstrip("/")
            if cleaned and cleaned not in self._node_urls:
                self._node_urls.append(cleaned)
        if not self._node_urls:
            self._node_urls = [primary]
        self.base_url = self._node_urls[0]
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_inflight = self._resolve_max_inflight(model, max_inflight)
        self.stream = stream
        self.reasoning_effort = reasoning_effort
        # Retry hedge for transient upstream failures (Ollama Cloud 500s seen
        # during the 2026-04-06 canonical run — ~2/hr cluster-wide at
        # escalation load, would scale linearly under primary-tier load).
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.environ.get("OLLAMA_VISION_MAX_RETRIES", "2"))
        )
        self.retry_backoff_seconds = (
            retry_backoff_seconds
            if retry_backoff_seconds is not None
            else float(os.environ.get("OLLAMA_VISION_RETRY_BACKOFF", "2.0"))
        )
        self._node_cycle = itertools.cycle(self._node_urls)
        self.last_base_url = self.base_url

    @property
    def node_urls(self) -> tuple[str, ...]:
        """Configured endpoint list in rotation order."""
        return tuple(self._node_urls)

    @staticmethod
    def _resolve_max_inflight(model: str, explicit: int | None) -> int:
        if explicit is not None:
            return max(1, int(explicit))

        model_name = str(model or "").strip().lower()
        env_name = (
            "OLLAMA_ESCALATION_MAX_INFLIGHT"
            if "32b" in model_name
            else "OLLAMA_VISION_MAX_INFLIGHT"
        )
        raw = os.environ.get(env_name, "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                logger.warning("Invalid %s=%r; falling back to defaults", env_name, raw)

        return 1 if "32b" in model_name else 2

    @classmethod
    def _endpoint_gate(cls, base_url: str, max_inflight: int) -> threading.BoundedSemaphore:
        key = (base_url.rstrip("/"), max(1, int(max_inflight)))
        with cls._gate_lock:
            gate = cls._endpoint_gates.get(key)
            if gate is None:
                gate = threading.BoundedSemaphore(value=key[1])
                cls._endpoint_gates[key] = gate
            return gate

    @staticmethod
    async def _acquire_gate(
        gate: threading.BoundedSemaphore,
        timeout_seconds: float,
    ) -> bool:
        """Acquire a thread semaphore without spawning cancellable worker threads.

        ``asyncio.to_thread(gate.acquire)`` is easy to cancel from the coroutine,
        but the worker thread continues waiting. If it later acquires the
        semaphore after the coroutine is gone, no ``finally`` block releases the
        permit, and subsequent vision requests can deadlock. Polling with a
        non-blocking acquire keeps cancellation local to the coroutine.
        """
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        while True:
            if gate.acquire(blocking=False):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)

    def _is_cloud(self, url: str) -> bool:
        return "ollama.com" in url

    async def analyze_image(
        self,
        image_path: Path | None,
        prompt: str,
        *,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        import httpx

        # Read image once — used by both native and compat formats
        image_b64: str | None = None
        image_mime: str = "image/png"
        if image_path is not None:
            raw = image_path.read_bytes()
            image_b64 = base64.b64encode(raw).decode()
            suffix = image_path.suffix.lstrip(".").lower()
            image_mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
                suffix, "image/png"
            )

        last_exc: Exception | None = None
        # Total attempts = (nodes × (max_retries + 1)). For a single cloud
        # endpoint with max_retries=2 this gives 3 attempts; for a 3-node
        # local cluster with max_retries=2 this gives 9 attempts with
        # node rotation. Backoff only kicks in on transient (5xx / network)
        # failures — 4xx fall through to the next node immediately.
        max_attempts = max(len(self._node_urls), 1) * (self.max_retries + 1)
        consecutive_transient = 0
        for attempt_idx in range(max_attempts):
            base_url = next(self._node_cycle)
            is_cloud = self._is_cloud(base_url)

            if is_cloud:
                # Native Ollama API: /api/chat
                # Strip /v1 suffix from base_url since native API doesn't use it
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]
                endpoint = "/api/chat"
                msg: dict[str, Any] = {"role": "user", "content": prompt}
                if image_b64 is not None:
                    msg["images"] = [image_b64]  # Raw base64, no data URL prefix
                payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": [msg],
                    "stream": False,
                    "think": False,
                    "keep_alive": 0,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": max_tokens,
                    },
                }
                if response_format is not None:
                    payload["format"] = response_format
            else:
                # OpenAI compat: /v1/chat/completions (local Ollama)
                endpoint = "/chat/completions"
                content: list[dict[str, Any]] = []
                if image_b64 is not None:
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}}
                    )
                content.append({"type": "text", "text": prompt})
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "stream": self.stream,
                    "reasoning_effort": self.reasoning_effort,
                }
                if response_format is not None:
                    payload["response_format"] = {"type": "json_schema", "json_schema": response_format}

            gate = self._endpoint_gate(base_url, self.max_inflight)
            acquired = False
            try:
                gate_timeout = min(max(float(self.timeout_seconds), 0.1), 30.0)
                if not await self._acquire_gate(gate, gate_timeout):
                    raise TimeoutError(
                        f"Timed out waiting for Ollama vision gate after {gate_timeout:.0f}s"
                    )
                acquired = True
                async with httpx.AsyncClient(
                    base_url=base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=httpx.Timeout(self.timeout_seconds, connect=30.0),
                    trust_env=False,
                ) as client:
                    resp = await client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                self.last_base_url = base_url

                # Extract response and usage — different formats
                if is_cloud:
                    # Native API response
                    response_text = data.get("message", {}).get("content", "")
                    input_tokens = data.get("prompt_eval_count", 0)
                    output_tokens = data.get("eval_count", 0)
                else:
                    # OpenAI compat response
                    response_text = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)

                if input_tokens or output_tokens:
                    tracker.record(
                        "ollama-vision",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                return response_text
            except Exception as exc:
                last_exc = exc
                error_text = str(exc).strip() or exc.__class__.__name__
                logger.warning(
                    "Vision request failed on %s with model %s: %s",
                    base_url,
                    self.model,
                    error_text,
                )

                # Decide whether to retry with backoff or fall through to the
                # next node/attempt. Transient failures (5xx, timeouts, network
                # errors) get exponential backoff before the next attempt.
                is_transient = False
                try:
                    if isinstance(exc, httpx.HTTPStatusError):
                        status = exc.response.status_code
                        is_transient = status in (429, 500, 502, 503, 504)
                    elif isinstance(
                        exc,
                        (
                            httpx.TimeoutException,
                            httpx.ConnectError,
                            httpx.ReadError,
                            httpx.RemoteProtocolError,
                            httpx.NetworkError,
                        ),
                    ):
                        is_transient = True
                except Exception:
                    pass

                if is_transient and attempt_idx < max_attempts - 1:
                    consecutive_transient += 1
                    backoff = min(
                        self.retry_backoff_seconds
                        * (2 ** min(consecutive_transient - 1, 4)),
                        30.0,
                    )
                    logger.info(
                        "Backing off %.1fs before retry %d/%d",
                        backoff,
                        attempt_idx + 2,
                        max_attempts,
                    )
                    await asyncio.sleep(backoff)
                    continue
                if not is_transient:
                    # Non-transient failures (4xx client errors, bad payload,
                    # etc.) are not going to recover on retry. Break out of
                    # the attempt loop so we fail fast instead of wasting
                    # quota on retries the server will keep rejecting.
                    break
                # is_transient but out of retries — fall through to raise.

            finally:
                if acquired:
                    gate.release()

        raise RuntimeError(
            f"All vision attempts ({max_attempts}) failed for {self.model}: {last_exc}"
        )


class OpenAIVisionProvider:
    """OpenAI (or any OpenAI-compatible API like OpenRouter, Together, etc.).

    Retries on 429 (rate limit) and 5xx (upstream errors) with exponential
    backoff so a single transient failure from OpenRouter doesn't kill the
    vision-eligible fix (REMEDY-69 #3). Respects the ``Retry-After`` header
    when the server provides one.
    """

    # Status codes that benefit from a retry. 408/425 are rare but real —
    # request-timeout / too-early — and a second attempt is usually cheap.
    _RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        *,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        retry_backoff_cap_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.environ.get("OPENAI_VISION_MAX_RETRIES", "4"))
        )
        self.retry_backoff_seconds = (
            retry_backoff_seconds
            if retry_backoff_seconds is not None
            else float(os.environ.get("OPENAI_VISION_RETRY_BACKOFF", "2.0"))
        )
        self.retry_backoff_cap_seconds = (
            retry_backoff_cap_seconds
            if retry_backoff_cap_seconds is not None
            else float(os.environ.get("OPENAI_VISION_RETRY_BACKOFF_CAP", "30.0"))
        )

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        """Parse the HTTP Retry-After header.

        The header may be a delay in seconds (``"5"``) or an absolute date
        (``"Wed, 21 Oct 2015 07:28:00 GMT"``). We only handle the seconds
        form — HTTP-date is rare for rate limits in practice.
        """
        if not value:
            return None
        try:
            seconds = float(str(value).strip())
        except ValueError:
            return None
        if seconds < 0:
            return None
        return seconds

    def _compute_backoff(self, attempt: int, retry_after: float | None) -> float:
        """Return seconds to sleep before the next retry.

        Prefers ``Retry-After`` when the server sent one; otherwise uses
        exponential backoff (base * 2^attempt) with small random jitter so
        concurrent callers don't sync-retry.
        """
        if retry_after is not None:
            return min(retry_after, self.retry_backoff_cap_seconds)
        import random
        base = self.retry_backoff_seconds * (2 ** attempt)
        jitter = random.uniform(0.0, self.retry_backoff_seconds)
        return min(base + jitter, self.retry_backoff_cap_seconds)

    async def analyze_image(
        self,
        image_path: Path | None,
        prompt: str,
        *,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        import httpx

        content = []
        if image_path is not None:
            raw = image_path.read_bytes()
            b64 = base64.b64encode(raw).decode()
            suffix = image_path.suffix.lstrip(".").lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
                suffix, "image/png"
            )
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *content,
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=30.0),
            trust_env=False,
        ) as client:
            last_exc: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post("/chat/completions", json=payload)
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                    # Network-layer hiccup — retryable.
                    last_exc = exc
                    if attempt >= self.max_retries:
                        raise
                    delay = self._compute_backoff(attempt, None)
                    logger.warning(
                        "OpenAI vision transient network error (%s): retrying in %.1fs [attempt %d/%d]",
                        type(exc).__name__, delay, attempt + 1, self.max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code in self._RETRYABLE_STATUS and attempt < self.max_retries:
                    retry_after = self._parse_retry_after(
                        resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                    )
                    delay = self._compute_backoff(attempt, retry_after)
                    logger.warning(
                        "OpenAI vision %s on %s: retrying in %.1fs [attempt %d/%d]%s",
                        resp.status_code, self.model, delay, attempt + 1, self.max_retries,
                        f" (server Retry-After={retry_after:.1f}s)" if retry_after is not None else "",
                    )
                    await asyncio.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            # All retries exhausted — raise the last error.
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(
                f"OpenAI vision: exhausted {self.max_retries} retries without success"
            )


def create_provider(
    provider: str,
    *,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    stream: bool = False,
    reasoning_effort: str = "low",
) -> VisionProvider:
    """Factory: create a vision provider by name.

    Parameters
    ----------
    provider:
        One of "ollama", "openai".
    api_key:
        API key (required for OpenAI-compatible providers, optional for Ollama).
    model:
        Model name override. Defaults depend on provider.
    base_url:
        Base URL override (ollama/openai only).
    stream:
        Whether to enable streaming (Ollama only).
    reasoning_effort:
        Reasoning effort level for Ollama compat endpoint (low/medium/high).
    """
    p = provider.lower().strip()

    if p == "ollama":
        return OllamaVisionProvider(
            base_url=base_url or "http://localhost:11434/v1",
            api_key=api_key or "ollama",
            model=model or "llava",
            stream=stream,
            reasoning_effort=reasoning_effort,
        )
    elif p == "openai":
        return OpenAIVisionProvider(
            api_key=api_key,
            model=model or "gpt-4o",
            base_url=base_url or "https://api.openai.com/v1",
        )
    else:
        raise ValueError(
            f"Unknown vision provider '{provider}'. "
            f"Choose from: ollama, openai"
        )


def create_provider_from_config(config) -> VisionProvider | None:
    """Create a vision provider from a ``PipelineConfig``.

    Uses the configured Ollama API key, model names, and base URLs.
    Returns ``None`` if no usable credentials are found.
    """
    backend = str(getattr(config.api, "llm_backend", "") or "ollama").strip().lower()
    if backend != "ollama":
        return create_provider(
            backend,
            api_key=config.api.api_key,
            model=config.api.vision_model,
            base_url=config.api.vision_base_url or config.api.base_url,
            stream=getattr(config.api, "ollama_stream", False),
            reasoning_effort=getattr(config.api, "ollama_reasoning_effort", "low"),
        )

    vision_urls = []
    primary = config.api.vision_base_url or config.api.base_url or "http://localhost:11434/v1"
    vision_urls.append(primary)
    vision_urls.extend(config.api.vision_cluster_nodes or ())
    return OllamaVisionProvider(
        base_url=primary,
        base_urls=vision_urls,
        api_key=config.api.api_key or "ollama",
        model=config.api.vision_model or "llava",
        stream=getattr(config.api, "ollama_stream", False),
        reasoning_effort=getattr(config.api, "ollama_reasoning_effort", "low"),
    )


def create_escalation_provider(config) -> VisionProvider | None:
    """Create a vision provider for Tier 2 escalation from config."""
    backend = config.api.escalation_backend
    model = config.api.escalation_model
    if not model:
        return None
    base_url = ""
    if backend == "ollama":
        base_url = (
            getattr(config.api, "escalation_base_url", "")
            or config.api.base_url
        )
    return create_provider(
        backend,
        api_key=config.api.api_key,
        model=model,
        base_url=base_url,
        stream=getattr(config.api, "ollama_stream", False),
        reasoning_effort=getattr(config.api, "ollama_reasoning_effort", "low"),
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PDF page renderer
# ---------------------------------------------------------------------------


def render_page_to_image(pdf_path: Path, page_num: int, dpi: int = 150) -> Path:
    """Render a single PDF page to a PNG image.

    Uses pikepdf to extract the page and then pdf2image (poppler) or
    falls back to a simple Playwright-based renderer.

    Returns the path to the temporary PNG file.
    """
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
            fmt="png",
        )
        if images:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            images[0].save(tmp.name, "PNG")
            return Path(tmp.name)
    except ImportError:
        pass

    # Fallback: use PyMuPDF (fitz) if available.
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        page = doc[page_num - 1]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pix.save(tmp.name)
        doc.close()
        return Path(tmp.name)
    except ImportError:
        pass

    raise RuntimeError(
        "No PDF renderer available. Install pdf2image (poppler) or PyMuPDF: "
        "pip install pdf2image  OR  pip install pymupdf"
    )


# ---------------------------------------------------------------------------
# Structure order extractor
# ---------------------------------------------------------------------------


def _get_page_structure_order(pdf_path: Path, page_num: int) -> str:
    """Extract the structure tree reading order for a specific page.

    Returns a numbered list of structure elements on that page.
    """
    lines: list[str] = []

    with pikepdf.open(pdf_path) as pdf:
        if page_num < 1 or page_num > len(pdf.pages):
            return "(invalid page number)"

        target_page = pdf.pages[page_num - 1]
        order = 0

        for node, depth, _parent in walk_structure_tree(pdf):
            # Check if this node is on the target page.
            pg = node.get("/Pg")
            if pg is None:
                # Check MCR children for page ref.
                kids = node.get("/K")
                if kids is None:
                    continue
                items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
                on_page = False
                for item in items:
                    resolved = item
                    if hasattr(item, "resolve"):
                        try:
                            resolved = item.resolve()
                        except Exception:
                            continue
                    if isinstance(resolved, pikepdf.Dictionary):
                        item_pg = resolved.get("/Pg")
                        if item_pg is not None:
                            try:
                                page_obj = item_pg.resolve() if hasattr(item_pg, "resolve") else item_pg
                                if page_obj == target_page.obj:
                                    on_page = True
                                    break
                            except Exception:
                                pass
                if not on_page:
                    continue
            else:
                try:
                    resolved_pg = pg.resolve() if hasattr(pg, "resolve") else pg
                    if resolved_pg != target_page.obj:
                        continue
                except Exception:
                    continue

            stype = _get_struct_type(node)
            if not stype:
                continue

            order += 1
            alt = node.get("/Alt")
            indent = "  " * min(depth, 4)
            line = f"{order:3d}. {indent}/{stype}"
            if alt and str(alt).strip():
                line += f'  (alt: "{str(alt)[:40]}")'
            lines.append(line)

    return "\n".join(lines) if lines else "(no structure elements found on this page)"


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------


class VisionAnalyzer:
    """Analyze PDF accessibility using a vision model.

    Parameters
    ----------
    provider:
        Vision provider instance (or use ``from_config()``).

    REMEDY-57 Phase 2: ``analyze_all()`` consults a process-level cache keyed
    on ``(resolved_path, mtime, size)`` so repeated calls on the same PDF do
    not re-spend vision tokens. The cache is invalidated automatically when
    the file changes (save after fix), and can be cleared via
    :func:`clear_vision_cache`.

    REMEDY-57 Phase 4: ``analyze_all()`` applies a configurable page-sampling
    budget (``VISION_PAGE_SAMPLE_SIZE`` env var, default 10) so that large
    catalogs do not spend vision on every page.
    """

    def __init__(self, provider: VisionProvider) -> None:
        self._provider = provider

    @classmethod
    def from_config(
        cls,
        provider: str,
        *,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ) -> VisionAnalyzer:
        """Create a VisionAnalyzer from provider name and config."""
        return cls(create_provider(provider, api_key=api_key, model=model, base_url=base_url))

    async def analyze_reading_order(
        self,
        pdf_path: Path,
        pages: list[int] | None = None,
        dpi: int = 150,
    ) -> VisionCheckResult:
        """Analyze reading order on specified pages (or all pages).

        Renders each page to an image, builds the structure-tree order for
        that page, and asks the vision model to compare.
        """
        result = VisionCheckResult()

        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        result.total_pages = total_pages

        if pages is None:
            pages = list(range(1, total_pages + 1))
        result.analyzed_pages = list(pages)
        timeout_count = 0
        timeout_abort_at = max(1, _VISION_ANALYZER_TIMEOUT_ABORTS)

        for page_num in pages:
            try:
                image_path = render_page_to_image(pdf_path, page_num, dpi=dpi)
            except RuntimeError as e:
                result.reading_order_issues.append(
                    ReadingOrderIssue(
                        page=page_num,
                        description=f"Could not render page: {e}",
                        severity="warning",
                    )
                )
                continue

            try:
                structure_order = _get_page_structure_order(pdf_path, page_num)
                prompt = build_reading_order_prompt(structure_order=structure_order)

                response = await asyncio.wait_for(
                    self._provider.analyze_image(image_path, prompt),
                    timeout=_VISION_ANALYZER_PAGE_TIMEOUT,
                )
                result.raw_responses[page_num] = response
                timeout_count = 0

                # Parse JSON from response.
                parsed = _parse_json_response(response)
                if parsed and "issues" in parsed:
                    for issue in parsed["issues"]:
                        result.reading_order_issues.append(
                            ReadingOrderIssue(
                                page=page_num,
                                description=issue.get("description", ""),
                                severity=issue.get("severity", "warning"),
                                suggestion=issue.get("suggestion", ""),
                            )
                        )

            except asyncio.TimeoutError:
                timeout_count += 1
                logger.warning(
                    "Vision reading-order acceptance timed out on page %d after %.0f s",
                    page_num,
                    _VISION_ANALYZER_PAGE_TIMEOUT,
                )
                if timeout_count >= timeout_abort_at:
                    logger.warning(
                        "Stopping reading-order acceptance after %d page timeout(s) for %s",
                        timeout_count,
                        pdf_path.name,
                    )
                    break
            except Exception as e:
                if _is_vision_unreachable(e):
                    # Connection-class failures mean we couldn't evaluate the
                    # page — not that the page has a reading-order problem.
                    # Don't fabricate an accessibility finding.
                    logger.warning(
                        "Vision unreachable for page %d — skipping reading-order check: %s",
                        page_num,
                        e,
                    )
                else:
                    logger.warning("Vision analysis failed for page %d: %s", page_num, e)
                    result.reading_order_issues.append(
                        ReadingOrderIssue(
                            page=page_num,
                            description=f"Vision analysis error: {e}",
                            severity="warning",
                        )
                    )
            finally:
                # Clean up temp image.
                try:
                    image_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return result

    async def analyze_contrast(
        self,
        pdf_path: Path,
        pages: list[int] | None = None,
        dpi: int = 150,
    ) -> VisionCheckResult:
        """Analyze color contrast on specified pages."""
        result = VisionCheckResult()

        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        result.total_pages = total_pages

        if pages is None:
            pages = list(range(1, total_pages + 1))
        result.analyzed_pages = list(pages)
        timeout_count = 0
        timeout_abort_at = max(1, _VISION_ANALYZER_TIMEOUT_ABORTS)

        for page_num in pages:
            try:
                image_path = render_page_to_image(pdf_path, page_num, dpi=dpi)
            except RuntimeError as e:
                result.contrast_issues.append(
                    ContrastIssue(
                        page=page_num,
                        description=f"Could not render page: {e}",
                    )
                )
                continue

            try:
                response = await asyncio.wait_for(
                    self._provider.analyze_image(
                        image_path,
                        build_contrast_detection_prompt("AA"),
                    ),
                    timeout=_VISION_ANALYZER_PAGE_TIMEOUT,
                )
                result.raw_responses[page_num] = response
                timeout_count = 0

                parsed = _parse_json_response(response)
                if parsed and "issues" in parsed:
                    for issue in parsed["issues"]:
                        issue_kind = str(issue.get("issue_type") or "").strip().lower()
                        criterion = str(issue.get("wcag_criterion") or "").strip()
                        if issue_kind == "use_of_color" or criterion == "1.4.1":
                            result.use_of_color_issues.append(
                                UseOfColorIssue(
                                    page=page_num,
                                    description=issue.get("description", ""),
                                    location=issue.get("location", ""),
                                    suggestion=issue.get("suggestion", ""),
                                )
                            )
                            continue
                        fix_rgb = _float_list(issue.get("fix_rgb"))
                        content_kind = str(
                            issue.get("content_kind")
                            or (
                                "pdf_text"
                                if fix_rgb and issue.get("text_rgb") is not None
                                else "unknown"
                            )
                        )
                        estimated_ratio = _float_or_none(
                            issue.get("estimated_ratio")
                            if "estimated_ratio" in issue
                            else issue.get("estimated_contrast_ratio")
                        )
                        required_ratio = _float_or_none(issue.get("required_ratio")) or 4.5
                        result.contrast_issues.append(
                            ContrastIssue(
                                page=page_num,
                                description=issue.get("description", ""),
                                location=issue.get("location", ""),
                                content_kind=content_kind,
                                bbox=_float_list(issue.get("bbox")),
                                text_rgb=_float_list(issue.get("text_rgb")),
                                bg_rgb=_float_list(issue.get("bg_rgb")),
                                fix_rgb=fix_rgb,
                                estimated_ratio=estimated_ratio,
                                required_ratio=required_ratio,
                                auto_fixable=_parse_auto_fixable(
                                    issue.get("auto_fixable"),
                                    content_kind,
                                    fix_rgb,
                                ),
                                wcag_criterion=str(issue.get("wcag_criterion") or "1.4.3"),
                            )
                        )
                if parsed and "use_of_color_issues" in parsed:
                    for issue in parsed["use_of_color_issues"]:
                        result.use_of_color_issues.append(
                            UseOfColorIssue(
                                page=page_num,
                                description=issue.get("description", ""),
                                location=issue.get("location", ""),
                                suggestion=issue.get("suggestion", ""),
                            )
                        )

            except asyncio.TimeoutError:
                timeout_count += 1
                logger.warning(
                    "Vision contrast acceptance timed out on page %d after %.0f s",
                    page_num,
                    _VISION_ANALYZER_PAGE_TIMEOUT,
                )
                if timeout_count >= timeout_abort_at:
                    logger.warning(
                        "Stopping contrast acceptance after %d page timeout(s) for %s",
                        timeout_count,
                        pdf_path.name,
                    )
                    break
            except Exception as e:
                if _is_vision_unreachable(e):
                    logger.warning(
                        "Vision unreachable for page %d — skipping contrast check: %s",
                        page_num,
                        e,
                    )
                else:
                    logger.warning("Contrast analysis failed for page %d: %s", page_num, e)
                    result.contrast_issues.append(
                        ContrastIssue(
                            page=page_num,
                            description=f"Vision analysis error: {e}",
                        )
                    )
            finally:
                try:
                    image_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return result

    async def analyze_all(
        self,
        pdf_path: Path,
        pages: list[int] | None = None,
        dpi: int = 150,
    ) -> VisionCheckResult:
        """Run both reading order and contrast analysis.

        REMEDY-57 Phase 2: caches the result per PDF so repeated calls (e.g.
        tier-1 then tier-2 acceptance) don't double-spend tokens.
        REMEDY-57 Phase 4: samples pages via ``VISION_PAGE_SAMPLE_SIZE`` env
        var (default 10) so large catalogs don't OOM the token budget.
        """
        # Phase 2: content-addressed cache. Only consult the cache for the
        # "full document" case (pages=None); callers asking for a specific
        # subset want a fresh computation.
        cache_key = _vision_cache_key(pdf_path) if pages is None else None
        if cache_key is not None:
            with _VISION_CACHE_LOCK:
                cached = _VISION_CACHE.get(cache_key)
            if cached is not None:
                return cached

        # Phase 4: sample pages up to the configured budget.
        if pages is None:
            pages = _sampled_pages(pdf_path)

        ro_result, contrast_result = await asyncio.gather(
            self.analyze_reading_order(pdf_path, pages, dpi),
            self.analyze_contrast(pdf_path, pages, dpi),
        )
        merged = VisionCheckResult(
            reading_order_issues=ro_result.reading_order_issues,
            contrast_issues=contrast_result.contrast_issues,
            use_of_color_issues=contrast_result.use_of_color_issues,
            raw_responses={**ro_result.raw_responses, **contrast_result.raw_responses},
            analyzed_pages=sorted(set(ro_result.analyzed_pages) | set(contrast_result.analyzed_pages)),
            total_pages=max(ro_result.total_pages, contrast_result.total_pages),
        )
        if cache_key is not None:
            _vision_cache_put(cache_key, merged)
        return merged


# ---------------------------------------------------------------------------
# REMEDY-57 Phase 2: process-level vision cache + Phase 4: page sampling
# ---------------------------------------------------------------------------

# Keyed on (resolved_path_str, mtime_ns, size). Invalidated automatically by
# key change when the file is rewritten (e.g. after fix_and_verify). Bounded
# to prevent unbounded growth in long-running processes.
_VISION_CACHE: dict[tuple[str, int, int], VisionCheckResult] = {}
_VISION_CACHE_MAX_ENTRIES = int(os.environ.get("VISION_CACHE_MAX_ENTRIES", "128"))
_VISION_CACHE_LOCK = threading.Lock()


def _vision_cache_key(pdf_path: Path) -> tuple[str, int, int] | None:
    """Build a cache key that is invalidated on file change."""
    try:
        st = pdf_path.stat()
    except OSError:
        return None
    return (str(pdf_path.resolve()), st.st_mtime_ns, st.st_size)


def _vision_cache_put(key: tuple[str, int, int], value: VisionCheckResult) -> None:
    with _VISION_CACHE_LOCK:
        if len(_VISION_CACHE) >= _VISION_CACHE_MAX_ENTRIES:
            # Drop an arbitrary entry — this is a small, per-process cache
            # so FIFO/LRU fidelity isn't worth the overhead.
            try:
                _VISION_CACHE.pop(next(iter(_VISION_CACHE)))
            except StopIteration:
                pass
        _VISION_CACHE[key] = value


def clear_vision_cache() -> None:
    """Drop every cached vision analysis. Useful for tests and --recheck."""
    with _VISION_CACHE_LOCK:
        _VISION_CACHE.clear()


def _sampled_pages(pdf_path: Path) -> list[int] | None:
    """Return a page list bounded by ``VISION_PAGE_SAMPLE_SIZE``.

    Returns ``None`` when sampling is disabled (budget <= 0) or when the PDF
    can't be opened — caller should fall back to "all pages".
    """
    raw = os.environ.get("VISION_PAGE_SAMPLE_SIZE", "10")
    try:
        budget = int(raw)
    except ValueError:
        budget = 10
    if budget <= 0:
        return None

    try:
        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
    except Exception:
        return None

    if total_pages <= budget:
        return list(range(1, total_pages + 1))

    # Even stride — first page, last page, and evenly spaced in between.
    # Guarantees front/back coverage for catalog-style docs.
    if budget == 1:
        return [1]
    step = (total_pages - 1) / (budget - 1)
    sampled = sorted({1 + round(i * step) for i in range(budget)})
    return [p for p in sampled if 1 <= p <= total_pages]


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Extract JSON from a vision model response that may contain markdown fences."""
    # Try direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences.
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first [ ... ] array block (heading lists, reading order arrays, etc.)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None
