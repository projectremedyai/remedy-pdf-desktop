"""Vision-model analysis for PDF accessibility checks.

Renders PDF pages to images and sends them to a vision model for
spatial analysis that can't be done with structure-tree inspection
alone — reading order validation and color contrast estimation.

Supports local Ollama, Ollama Cloud, and OpenRouter (OpenAI-compatible).
Provider selection is done at call time so users can mix and match.

Usage::

    analyzer = VisionAnalyzer(provider="ollama", api_key="...", model="gemma4:31b-cloud")
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pikepdf

from project_remedy.pdf_checker import walk_structure_tree, _get_struct_type
from project_remedy.token_tracker import tracker
from project_remedy.vision_prompts import (
    contrast_detection_prompt as build_contrast_detection_prompt,
    heading_hierarchy_quality_prompt,
    page_alt_text_quality_prompt,
    reading_order_prompt as build_reading_order_prompt,
)

logger = logging.getLogger(__name__)


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


@dataclass
class HeadingIssue:
    """A heading hierarchy problem identified by the vision model."""

    page: int
    description: str
    severity: str = "warning"  # "error" | "warning" | "info"
    suggestion: str = ""
    element_index: int | None = None
    current_tag: str = ""
    correct_tag: str = ""
    text: str = ""


@dataclass
class AltTextIssue:
    """An alt-text quality problem identified by the vision model."""

    page: int
    figure_index: int
    current_alt_text: str = ""
    description: str = ""
    severity: str = "warning"  # "error" | "warning" | "info"
    suggested_alt_text: str = ""
    decorative: bool = False
    issue_type: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class _FigureAltEntry:
    """Page-local Figure metadata sent to the alt-text quality prompt."""

    figure_index: int
    current_alt_text: str
    mcids: tuple[int, ...] = ()
    bbox: tuple[int, int, int, int] | None = None


@dataclass
class VisionCheckResult:
    """Result of vision-based analysis for one or more pages."""

    reading_order_issues: list[ReadingOrderIssue] = field(default_factory=list)
    contrast_issues: list[ContrastIssue] = field(default_factory=list)
    heading_issues: list[HeadingIssue] = field(default_factory=list)
    alt_text_issues: list[AltTextIssue] = field(default_factory=list)
    raw_responses: dict[int | str, str] = field(default_factory=dict)

    @property
    def reading_order_passed(self) -> bool:
        return not any(i.severity == "error" for i in self.reading_order_issues)

    @property
    def contrast_passed(self) -> bool:
        return len(self.contrast_issues) == 0

    @property
    def heading_hierarchy_passed(self) -> bool:
        return not any(i.severity == "error" for i in self.heading_issues)

    @property
    def alt_text_quality_passed(self) -> bool:
        return not any(i.severity == "error" for i in self.alt_text_issues)


# ---------------------------------------------------------------------------
# Vision provider protocol
# ---------------------------------------------------------------------------


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
        # Optional retry hedge for transient upstream failures. Default to no
        # retries so table-heavy remediation can fall back instead of waiting
        # through several full vision timeout windows.
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.environ.get("OLLAMA_VISION_MAX_RETRIES", "0"))
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

    def _is_cloud(self, url: str) -> bool:
        return "ollama.com" in url

    def _uses_native_api(self, url: str) -> bool:
        """Use Ollama's native chat API for cloud or root local Ollama URLs."""
        return self._is_cloud(url) or not url.rstrip("/").endswith("/v1")

    def _request_max_tokens(self, requested: int, *, is_cloud: bool) -> int:
        """Bound vision response length so provider calls stay interactive."""
        env_names = (
            ("OLLAMA_CLOUD_VISION_MAX_TOKENS", "OLLAMA_VISION_MAX_TOKENS")
            if is_cloud
            else ("OLLAMA_VISION_MAX_TOKENS",)
        )
        default_cap = 1024 if is_cloud else requested
        cap = default_cap
        for env_name in env_names:
            raw = os.environ.get(env_name, "").strip()
            if not raw:
                continue
            try:
                cap = max(1, int(raw))
                break
            except ValueError:
                logger.warning("Invalid %s=%r; using max token cap %d", env_name, raw, cap)
        return max(1, min(int(requested), cap))

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
            uses_native_api = self._uses_native_api(base_url)
            request_max_tokens = self._request_max_tokens(max_tokens, is_cloud=is_cloud)

            if uses_native_api:
                # Native Ollama API: /api/chat
                # Strip /v1 suffix from base_url since native API doesn't use it
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]
                endpoint = "/api/chat"
                msg: dict[str, Any] = {"role": "user", "content": prompt}
                if image_b64 is not None:
                    msg["images"] = [image_b64]  # Raw base64, no data URL prefix
                keep_alive: int | str = (
                    0 if is_cloud else os.environ.get("OLLAMA_LOCAL_KEEP_ALIVE", "5m")
                )
                payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": [msg],
                    "stream": False,
                    "think": False,
                    "keep_alive": keep_alive,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": request_max_tokens,
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
                    "max_tokens": request_max_tokens,
                    "temperature": 0.2,
                    "stream": self.stream,
                    "reasoning_effort": self.reasoning_effort,
                }
                if response_format is not None:
                    payload["response_format"] = {"type": "json_schema", "json_schema": response_format}

            gate = self._endpoint_gate(base_url, self.max_inflight)
            acquired = False
            try:
                try:
                    gate_timeout = float(
                        os.environ.get(
                            "OLLAMA_VISION_GATE_TIMEOUT_SECONDS",
                            str(min(self.timeout_seconds, 30.0)),
                        )
                    )
                except ValueError:
                    gate_timeout = min(self.timeout_seconds, 30.0)
                gate_deadline = asyncio.get_running_loop().time() + max(1.0, gate_timeout)
                while not gate.acquire(blocking=False):
                    if asyncio.get_running_loop().time() >= gate_deadline:
                        raise TimeoutError(
                            f"Timed out waiting for vision slot on {base_url} "
                            f"after {gate_timeout:.0f}s"
                        )
                    await asyncio.sleep(0.05)
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
                    resp = await asyncio.wait_for(
                        client.post(endpoint, json=payload),
                        timeout=self.timeout_seconds,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                self.last_base_url = base_url

                # Extract response and usage — different formats
                if uses_native_api:
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
                if not str(response_text).strip():
                    raise RuntimeError("empty vision response")
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


class FallbackVisionProvider:
    """Try multiple vision providers in order for the same image prompt."""

    def __init__(self, providers: list[VisionProvider]) -> None:
        if not providers:
            raise ValueError("FallbackVisionProvider requires at least one provider")
        self.providers = providers
        self.model = " -> ".join(str(getattr(provider, "model", "unknown")) for provider in providers)
        self.base_url = str(getattr(providers[0], "base_url", ""))

    async def analyze_image(
        self,
        image_path: Path | None,
        prompt: str,
        *,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        failures: list[str] = []
        for provider in self.providers:
            try:
                response = await provider.analyze_image(
                    image_path,
                    prompt,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                if str(response).strip():
                    return response
                raise RuntimeError("empty vision response")
            except Exception as exc:
                model = str(getattr(provider, "model", "unknown"))
                base_url = str(getattr(provider, "base_url", ""))
                message = f"{model} at {base_url}: {exc}"
                failures.append(message)
                logger.warning("Vision provider failed; trying fallback if available: %s", message)
        raise RuntimeError("All configured vision providers failed: " + " | ".join(failures))


class OpenRouterVisionProvider:
    """OpenRouter vision provider.

    OpenRouter (https://openrouter.ai) is OpenAI-compatible — same
    ``/chat/completions`` shape and base64 ``image_url`` content blocks — and
    routes the request to whichever underlying model the user picks
    (``openai/gpt-4o-mini``, ``anthropic/claude-3-5-sonnet``,
    ``google/gemini-2.0-flash``, etc.). The optional ``HTTP-Referer`` and
    ``X-Title`` headers identify this app on OpenRouter's dashboards and
    leaderboards.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        *,
        http_referer: str = "https://github.com/projectremedyai/remedy-pdf-desktop",
        x_title: str = "Remedy PDF Desktop",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.http_referer = http_referer
        self.x_title = x_title

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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(120.0, connect=30.0),
            trust_env=False,
        ) as client:
            resp = await asyncio.wait_for(
                client.post("/chat/completions", json=payload),
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]
            if not str(response_text).strip():
                raise RuntimeError("empty vision response")
            return response_text



def create_provider(
    provider: str,
    *,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    stream: bool = False,
    reasoning_effort: str = "low",
    timeout_seconds: float | None = None,
) -> VisionProvider:
    """Factory: create a vision provider by name.

    Parameters
    ----------
    provider:
        One of "ollama", "openrouter".
    api_key:
        API key (required for cloud providers, optional for local Ollama).
    model:
        Model name override. Defaults depend on provider.
    base_url:
        Base URL override (ollama/openrouter only).
    stream:
        Whether to enable streaming (Ollama only).
    reasoning_effort:
        Reasoning effort level for Ollama compat endpoint (low/medium/high).
    """
    p = provider.lower().strip()

    if p == "ollama":
        return OllamaVisionProvider(
            base_url=base_url or "https://ollama.com/v1",
            api_key=api_key or "ollama",
            model=model or "gemma4:31b-cloud",
            timeout_seconds=timeout_seconds or _configured_vision_timeout(),
            stream=stream,
            reasoning_effort=reasoning_effort,
        )
    elif p == "openrouter":
        return OpenRouterVisionProvider(
            api_key=api_key,
            model=model or "openai/gpt-4o-mini",
            base_url=base_url or "https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(
            f"Unknown vision provider '{provider}'. "
            f"Choose from: ollama, openrouter"
        )


def _configured_vision_timeout(default: float = 120.0) -> float:
    for name in ("OLLAMA_VISION_TIMEOUT_SECONDS", "OLLAMA_TIMEOUT_SECONDS"):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return max(10.0, float(raw))
        except ValueError:
            logger.warning("Invalid %s=%r; using %.0fs", name, raw, default)
            break
    return default


def _env_csv(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def create_provider_from_config(config) -> VisionProvider | None:
    """Create a vision provider from a ``PipelineConfig``.

    Uses the configured Ollama API key, model names, and base URLs.
    Returns ``None`` if no usable credentials are found.
    """
    primary = config.api.vision_base_url or config.api.base_url or "https://ollama.com/v1"
    vision_urls = [primary, *(config.api.vision_cluster_nodes or ())]
    timeout_seconds = _configured_vision_timeout()
    api_key = config.api.api_key or "ollama"
    stream = getattr(config.api, "ollama_stream", False)
    reasoning_effort = getattr(config.api, "ollama_reasoning_effort", "low")

    primary_provider = OllamaVisionProvider(
        base_url=primary,
        base_urls=vision_urls,
        api_key=api_key,
        model=config.api.vision_model or "gemma4:31b-cloud",
        timeout_seconds=timeout_seconds,
        stream=stream,
        reasoning_effort=reasoning_effort,
    )

    fallback_models = _env_csv("OLLAMA_VISION_FALLBACK_MODELS")
    if not fallback_models:
        return primary_provider

    fallback_urls = _env_csv("OLLAMA_VISION_FALLBACK_BASE_URLS")
    providers: list[VisionProvider] = [primary_provider]
    for index, fallback_model in enumerate(fallback_models):
        if index < len(fallback_urls):
            fallback_url = fallback_urls[index]
        elif fallback_urls:
            fallback_url = fallback_urls[-1]
        else:
            fallback_url = primary
        providers.append(
            OllamaVisionProvider(
                base_url=fallback_url,
                api_key=api_key,
                model=fallback_model,
                timeout_seconds=timeout_seconds,
                stream=stream,
                reasoning_effort=reasoning_effort,
            )
        )
    return FallbackVisionProvider(providers)


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
        api_key=config.api.api_key if backend == "ollama" else "",
        model=model,
        base_url=base_url,
        timeout_seconds=_configured_vision_timeout(),
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


def _resolve_pdf_object(obj):
    """Best-effort resolve for indirect pikepdf objects.

    Avoid ``hasattr(obj, "resolve")`` here: pikepdf arrays can treat attribute
    lookup as dictionary-key access and raise while checking for the method.
    """
    if isinstance(obj, pikepdf.Array):
        return obj
    if isinstance(obj, pikepdf.Object) and obj.is_indirect:
        try:
            return obj.resolve()
        except Exception:
            return obj
    return obj


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
                    resolved = _resolve_pdf_object(item)
                    if isinstance(resolved, pikepdf.Dictionary):
                        item_pg = resolved.get("/Pg")
                        if item_pg is not None:
                            try:
                                page_obj = _resolve_pdf_object(item_pg)
                                if page_obj == target_page.obj:
                                    on_page = True
                                    break
                            except Exception:
                                pass
                if not on_page:
                    continue
            else:
                try:
                    resolved_pg = _resolve_pdf_object(pg)
                    if resolved_pg != target_page.obj:
                        continue
                except Exception:
                    continue

            stype = _get_struct_type(node)
            if not stype:
                continue

            alt = node.get("/Alt")
            actual = node.get("/ActualText")
            if (
                actual is not None
                and not str(actual).strip()
                and stype in {
                    "P", "Span", "H", "H1", "H2", "H3", "H4", "H5", "H6",
                    "LI", "LBody", "TH", "TD",
                }
            ):
                continue
            if (
                actual is None
                and not (alt and str(alt).strip())
                and (
                    stype in {"P", "Span", "Div"}
                    or re.match(r"^H[1-6]$", stype)
                )
            ):
                continue

            order += 1
            indent = "  " * min(depth, 4)
            line = f"{order:3d}. {indent}/{stype}"
            if actual and str(actual).strip():
                preview = str(actual).strip()
                if len(preview) > 220:
                    preview = preview[:217].rstrip() + "..."
                line += f'  (text: "{preview}")'
            if alt and str(alt).strip():
                alt_preview = str(alt).strip()
                if len(alt_preview) > 120:
                    alt_preview = alt_preview[:117].rstrip() + "..."
                line += f'  (alt: "{alt_preview}")'
            lines.append(line)

    return "\n".join(lines) if lines else "(no structure elements found on this page)"


def _node_mcids(node: pikepdf.Dictionary) -> tuple[int, ...]:
    """Return MCIDs referenced directly by a structure node's /K entry."""
    mcids: list[int] = []

    def visit(value: object) -> None:
        resolved = _resolve_pdf_object(value)

        if isinstance(resolved, int):
            mcids.append(int(resolved))
            return
        if isinstance(resolved, pikepdf.Array):
            for item in resolved:
                visit(item)
            return
        if not isinstance(resolved, pikepdf.Dictionary):
            return

        mcid = resolved.get("/MCID")
        if mcid is not None:
            try:
                mcids.append(int(mcid))
            except (TypeError, ValueError):
                pass

        kids = resolved.get("/K")
        if kids is not None:
            visit(kids)

    visit(node.get("/K"))
    return tuple(dict.fromkeys(mcids))


def _matrix_multiply(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    la, lb, lc, ld, le, lf = left
    ra, rb, rc, rd, re, rf = right
    return (
        la * ra + lc * rb,
        lb * ra + ld * rb,
        la * rc + lc * rd,
        lb * rc + ld * rd,
        la * re + lc * rf + le,
        lb * re + ld * rf + lf,
    )


def _matrix_point(
    matrix: tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def _normalize_page_bbox(
    bbox: tuple[float, float, float, float],
    page: pikepdf.Page,
) -> tuple[int, int, int, int] | None:
    media_box = page.get("/CropBox") or page.get("/MediaBox")
    if media_box is None or len(media_box) < 4:
        return None
    try:
        min_x, min_y, max_x, max_y = [float(v) for v in list(media_box)[:4]]
    except (TypeError, ValueError):
        return None

    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return None

    x0, y0, x1, y1 = bbox
    left = round(max(0, min(1000, ((x0 - min_x) / width) * 1000)))
    right = round(max(0, min(1000, ((x1 - min_x) / width) * 1000)))
    top = round(max(0, min(1000, ((max_y - y1) / height) * 1000)))
    bottom = round(max(0, min(1000, ((max_y - y0) / height) * 1000)))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _page_mcid_visual_bboxes(pdf: pikepdf.Pdf, page_idx: int) -> dict[int, tuple[int, int, int, int]]:
    """Best-effort MCID to visual bbox map for image/form XObject figures."""
    if page_idx < 0 or page_idx >= len(pdf.pages):
        return {}

    page = pdf.pages[page_idx]
    resources = page.get("/Resources")
    xobjects = resources.get("/XObject") if isinstance(resources, pikepdf.Dictionary) else None
    if not isinstance(xobjects, pikepdf.Dictionary):
        return {}

    try:
        instructions = pikepdf.parse_content_stream(page)
    except Exception:
        return {}

    ctm: tuple[float, float, float, float, float, float] = (1, 0, 0, 1, 0, 0)
    ctm_stack: list[tuple[float, float, float, float, float, float]] = []
    mcid_stack: list[int | None] = []
    bboxes: dict[int, tuple[int, int, int, int]] = {}

    for operands, operator in instructions:
        op = str(operator)
        if op == "q":
            ctm_stack.append(ctm)
            continue
        if op == "Q":
            if ctm_stack:
                ctm = ctm_stack.pop()
            continue
        if op == "cm" and len(operands) >= 6:
            try:
                matrix = tuple(float(v) for v in operands[:6])
            except (TypeError, ValueError):
                continue
            ctm = _matrix_multiply(ctm, matrix)  # type: ignore[arg-type]
            continue
        if op in {"BDC", "BMC"}:
            mcid = None
            if op == "BDC" and len(operands) >= 2 and isinstance(operands[1], pikepdf.Dictionary):
                raw_mcid = operands[1].get("/MCID")
                if raw_mcid is not None:
                    try:
                        mcid = int(raw_mcid)
                    except (TypeError, ValueError):
                        mcid = None
            mcid_stack.append(mcid)
            continue
        if op == "EMC":
            if mcid_stack:
                mcid_stack.pop()
            continue
        if op != "Do" or not operands:
            continue

        active_mcid = next((m for m in reversed(mcid_stack) if m is not None), None)
        if active_mcid is None:
            continue
        xobject = xobjects.get(str(operands[0]))
        if xobject is None:
            continue
        try:
            xobject = _resolve_pdf_object(xobject)
        except Exception:
            continue
        if not isinstance(xobject, (pikepdf.Dictionary, pikepdf.Stream)):
            continue

        subtype = str(xobject.get("/Subtype", ""))
        if subtype == "/Image":
            unit_bbox = (0.0, 0.0, 1.0, 1.0)
        elif subtype == "/Form":
            form_bbox = xobject.get("/BBox")
            if form_bbox is None or len(form_bbox) < 4:
                unit_bbox = (0.0, 0.0, 1.0, 1.0)
            else:
                try:
                    unit_bbox = tuple(float(v) for v in list(form_bbox)[:4])
                except (TypeError, ValueError):
                    unit_bbox = (0.0, 0.0, 1.0, 1.0)
        else:
            continue

        x0, y0, x1, y1 = unit_bbox
        points = [
            _matrix_point(ctm, x0, y0),
            _matrix_point(ctm, x1, y0),
            _matrix_point(ctm, x0, y1),
            _matrix_point(ctm, x1, y1),
        ]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        normalized = _normalize_page_bbox((min(xs), min(ys), max(xs), max(ys)), page)
        if normalized is not None:
            bboxes.setdefault(active_mcid, normalized)

    return bboxes


def _get_page_figure_alt_entries(pdf_path: Path, page_num: int) -> list[_FigureAltEntry]:
    """Return page-local Figure elements with alt text and visual anchors."""
    entries: list[_FigureAltEntry] = []

    with pikepdf.open(pdf_path) as pdf:
        if page_num < 1 or page_num > len(pdf.pages):
            return []

        target_page = pdf.pages[page_num - 1]
        mcid_bboxes = _page_mcid_visual_bboxes(pdf, page_num - 1)
        figure_index = 0

        for node, _depth, _parent in walk_structure_tree(pdf):
            if _get_struct_type(node) != "Figure":
                continue

            pg = node.get("/Pg")
            if pg is not None:
                try:
                    resolved_pg = _resolve_pdf_object(pg)
                    if resolved_pg != target_page.obj:
                        continue
                except Exception:
                    continue
            else:
                kids = node.get("/K")
                items = list(kids) if isinstance(kids, pikepdf.Array) else [kids] if kids is not None else []
                on_page = False
                for item in items:
                    resolved = _resolve_pdf_object(item)
                    if not isinstance(resolved, pikepdf.Dictionary):
                        continue
                    item_pg = resolved.get("/Pg")
                    if item_pg is None:
                        continue
                    try:
                        page_obj = _resolve_pdf_object(item_pg)
                    except Exception:
                        continue
                    if page_obj == target_page.obj:
                        on_page = True
                        break
                if not on_page:
                    continue

            figure_index += 1
            alt = str(node.get("/Alt", "") or "").strip()
            mcids = _node_mcids(node)
            bbox = next((mcid_bboxes[mcid] for mcid in mcids if mcid in mcid_bboxes), None)
            entries.append(
                _FigureAltEntry(
                    figure_index=figure_index,
                    current_alt_text=alt,
                    mcids=mcids,
                    bbox=bbox,
                )
            )

    return entries


def _get_page_figure_alt_list(pdf_path: Path, page_num: int) -> str:
    """Return numbered Figure elements and alt text for a page."""
    entries = _get_page_figure_alt_entries(pdf_path, page_num)
    lines: list[str] = []
    for entry in entries:
        bbox = (
            f"bbox={list(entry.bbox)}"
            if entry.bbox is not None
            else "bbox=unknown"
        )
        mcids = f"mcids={list(entry.mcids)}" if entry.mcids else "mcids=[]"
        alt = entry.current_alt_text[:180].replace('"', '\\"')
        lines.append(f'{entry.figure_index}. {bbox}; {mcids}; current_alt_text="{alt}"')

    return "\n".join(lines) if lines else "(no Figure tags found on this page)"


def _normal_severity(value: str | None, *, default: str = "warning") -> str:
    severity = str(value or default).strip().lower()
    if severity in {"error", "fail", "failed", "critical"}:
        return "error"
    if severity in {"warning", "warn"}:
        return "warning"
    return "info"


def _normal_struct_tag(value: str | None) -> str:
    """Normalize a model-returned structure tag, allowing only safe tags."""
    tag = str(value or "").strip().lstrip("/")
    compact = re.sub(r"[\s_-]+", "", tag).upper()
    if compact in {"PARAGRAPH", "BODY", "BODYTEXT", "TEXT"}:
        return "P"
    if compact in {"SPAN", "INLINE"}:
        return "Span"
    if compact in {"P", "L", "LI"}:
        return compact
    if compact == "LBODY":
        return "LBody"
    if compact == "LBL":
        return "Lbl"
    level_match = re.match(r"^(?:H|HEADING|HEADINGLEVEL)([1-6])$", compact)
    if level_match:
        return f"H{level_match.group(1)}"
    title_match = re.match(r"^(?:DOCUMENT)?TITLE$", compact)
    if title_match:
        return "H1"
    if compact in {"NOTHEADING", "NONHEADING"}:
        return "P"
    tag = compact
    if tag in {"P", "L", "LI"}:
        return tag
    if re.match(r"^H[1-6]$", tag):
        return tag
    return ""


def _struct_tag_from_suggestion(value: str | None) -> str:
    suggestion = str(value or "")
    match = re.search(
        r"(?:retag|tag|set|change|mark)\s+(?:it\s+)?(?:as|to)?\s*/?"
        r"(H[1-6]|heading\s*level\s*[1-6]|P|paragraph|body\s+text|Span|L|LI|LBody|Lbl)\b",
        suggestion,
        re.I,
    )
    if match:
        return _normal_struct_tag(match.group(1))
    match = re.search(r"/(H[1-6]|P|Span|L|LI|LBody|Lbl)\b", suggestion, re.I)
    if match:
        return _normal_struct_tag(match.group(1))
    return ""


def _string_field(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def _heading_findings(parsed: Any) -> tuple[str, list[dict[str, Any]]]:
    """Return heading finding dictionaries from supported model schemas."""
    if isinstance(parsed, list):
        return "fail" if parsed else "pass", [item for item in parsed if isinstance(item, dict)]
    if not isinstance(parsed, dict):
        return "pass", []

    status = str(parsed.get("status", "pass")).strip().lower()
    candidates: list[Any] = []
    for key in ("findings", "heading_corrections", "corrections", "heading_issues", "issues"):
        value = parsed.get(key)
        if isinstance(value, list):
            candidates.extend(value)

    if not candidates and ("element_index" in parsed or "correct_tag" in parsed):
        candidates.append(parsed)
    findings = [item for item in candidates if isinstance(item, dict)]
    return status, findings


class ReadingOrderVisionAgent:
    """Vision reviewer for logical reading order."""

    def __init__(self, provider: VisionProvider) -> None:
        self._provider = provider

    async def review_page(
        self,
        pdf_path: Path,
        page_num: int,
        *,
        dpi: int = 150,
    ) -> tuple[list[ReadingOrderIssue], str]:
        image_path = render_page_to_image(pdf_path, page_num, dpi=dpi)
        try:
            structure_order = _get_page_structure_order(pdf_path, page_num)
            response = await self._provider.analyze_image(
                image_path,
                build_reading_order_prompt(structure_order=structure_order),
            )
            parsed = _parse_json_response(response)
            issues: list[ReadingOrderIssue] = []
            if isinstance(parsed, dict):
                for issue in parsed.get("issues", []) or []:
                    issues.append(
                        ReadingOrderIssue(
                            page=page_num,
                            description=str(issue.get("description", "")).strip(),
                            severity=_normal_severity(issue.get("severity")),
                            suggestion=str(issue.get("suggestion", "") or ""),
                        )
                    )
            return issues, response
        finally:
            image_path.unlink(missing_ok=True)


class HeadingHierarchyVisionAgent:
    """Vision reviewer for visual heading hierarchy quality."""

    def __init__(self, provider: VisionProvider) -> None:
        self._provider = provider

    async def review_page(
        self,
        pdf_path: Path,
        page_num: int,
        *,
        dpi: int = 150,
    ) -> tuple[list[HeadingIssue], str]:
        image_path = render_page_to_image(pdf_path, page_num, dpi=dpi)
        try:
            logical_order = _get_page_structure_order(pdf_path, page_num)
            response = await self._provider.analyze_image(
                image_path,
                heading_hierarchy_quality_prompt(logical_order=logical_order),
            )
            parsed = _parse_json_response(response)
            issues: list[HeadingIssue] = []
            status, findings = _heading_findings(parsed)
            for finding in findings:
                suggestion = _string_field(
                    finding,
                    "suggested_fix",
                    "suggestion",
                    "fix",
                    "recommendation",
                )
                correct_tag = (
                    _normal_struct_tag(finding.get("correct_tag"))
                    or _normal_struct_tag(finding.get("target_tag"))
                    or _normal_struct_tag(finding.get("expected_tag"))
                    or _struct_tag_from_suggestion(suggestion)
                )
                if not suggestion and correct_tag:
                    suggestion = f"Retag as {correct_tag}"

                description = _string_field(
                    finding,
                    "message",
                    "reason",
                    "description",
                    "issue",
                )
                if not description and correct_tag:
                    description = f"Element should be tagged as {correct_tag}"

                default_severity = "warning"
                if status in {"fail", "failed", "error"} or correct_tag:
                    default_severity = "error"

                issues.append(
                    HeadingIssue(
                        page=page_num,
                        description=description,
                        severity=_normal_severity(
                            finding.get("severity"),
                            default=default_severity,
                        ),
                        suggestion=suggestion,
                        element_index=(
                            _optional_int(finding.get("element_index"))
                            or _optional_int(finding.get("index"))
                            or _optional_int(finding.get("element"))
                        ),
                        current_tag=(
                            _normal_struct_tag(finding.get("current_tag"))
                            or _normal_struct_tag(finding.get("tag"))
                        ),
                        correct_tag=correct_tag,
                        text=_string_field(
                            finding,
                            "visible_text",
                            "text",
                            "heading_text",
                            "element_text",
                        ),
                    )
                )
            if status in {"fail", "failed", "error"} and not issues:
                issues.append(
                    HeadingIssue(
                        page=page_num,
                        description="Vision model reported heading hierarchy mismatch",
                        severity="error",
                    )
                )
            return issues, response
        finally:
            image_path.unlink(missing_ok=True)


class AltTextQualityVisionAgent:
    """Vision reviewer for figure alt-text accuracy and specificity."""

    def __init__(self, provider: VisionProvider) -> None:
        self._provider = provider

    async def review_page(
        self,
        pdf_path: Path,
        page_num: int,
        *,
        dpi: int = 150,
    ) -> tuple[list[AltTextIssue], str | None]:
        entries = _get_page_figure_alt_entries(pdf_path, page_num)
        if not entries:
            return [], None
        current_alt_by_index = {
            entry.figure_index: entry.current_alt_text
            for entry in entries
        }
        figure_list = _get_page_figure_alt_list(pdf_path, page_num)

        image_path = render_page_to_image(pdf_path, page_num, dpi=dpi)
        try:
            response = await self._provider.analyze_image(
                image_path,
                page_alt_text_quality_prompt(figure_list=figure_list),
            )
            parsed = _parse_json_response(response)
            issues: list[AltTextIssue] = []
            if isinstance(parsed, dict):
                items = parsed.get("figures", parsed.get("issues", [])) or []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    status = str(item.get("status", "pass")).strip().lower()
                    issue_type = str(
                        item.get("issue_type")
                        or item.get("failure_type")
                        or item.get("reason_code")
                        or ""
                    ).strip().lower()
                    severity = _normal_severity(
                        item.get("severity"),
                        default="error"
                        if status in {"fail", "failed", "error"} or issue_type
                        else "info",
                    )
                    if status not in {"fail", "failed", "error"} and severity != "error":
                        continue
                    figure_index = _optional_int(item.get("figure_index")) or 0
                    suggested_alt_text = str(
                        item.get("suggested_alt_text")
                        or item.get("suggested_alt")
                        or item.get("replacement_alt_text")
                        or item.get("replacement_alt")
                        or ""
                    ).strip()
                    decorative = _optional_bool(item.get("decorative", item.get("is_decorative", False)))
                    confidence = _optional_float(item.get("confidence"))
                    issues.append(
                        AltTextIssue(
                            page=page_num,
                            figure_index=figure_index,
                            current_alt_text=str(
                                item.get("current_alt_text")
                                or current_alt_by_index.get(figure_index, "")
                            ),
                            description=str(
                                item.get("message")
                                or item.get("failure_reason")
                                or item.get("reason")
                                or "Alt text quality issue"
                            ).strip(),
                            severity=severity,
                            suggested_alt_text=suggested_alt_text,
                            decorative=decorative,
                            issue_type=issue_type,
                            confidence=confidence,
                        )
                    )
            return issues, response
        finally:
            image_path.unlink(missing_ok=True)


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "decorative"}


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

        if pages is None:
            pages = list(range(1, total_pages + 1))

        agent = ReadingOrderVisionAgent(self._provider)
        for page_num in pages:
            try:
                issues, response = await agent.review_page(pdf_path, page_num, dpi=dpi)
                result.reading_order_issues.extend(issues)
                result.raw_responses[page_num] = response
            except RuntimeError as e:
                result.reading_order_issues.append(
                    ReadingOrderIssue(
                        page=page_num,
                        description=f"Could not render page: {e}",
                        severity="warning",
                    )
                )
                continue
            except Exception as e:
                logger.warning("Vision analysis failed for page %d: %s", page_num, e)
                result.reading_order_issues.append(
                    ReadingOrderIssue(
                        page=page_num,
                        description=f"Vision analysis error: {e}",
                        severity="warning",
                    )
                )

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

        if pages is None:
            pages = list(range(1, total_pages + 1))

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
                response = await self._provider.analyze_image(
                    image_path,
                    build_contrast_detection_prompt("AA"),
                )
                result.raw_responses[page_num] = response

                parsed = _parse_json_response(response)
                if parsed and "issues" in parsed:
                    for issue in parsed["issues"]:
                        result.contrast_issues.append(
                            ContrastIssue(
                                page=page_num,
                                description=issue.get("description", ""),
                                location=issue.get("location", ""),
                            )
                        )

            except Exception as e:
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

    async def analyze_heading_hierarchy(
        self,
        pdf_path: Path,
        pages: list[int] | None = None,
        dpi: int = 150,
    ) -> VisionCheckResult:
        """Analyze visual heading hierarchy quality on specified pages."""
        result = VisionCheckResult()

        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        if pages is None:
            pages = list(range(1, total_pages + 1))

        agent = HeadingHierarchyVisionAgent(self._provider)
        for page_num in pages:
            try:
                issues, response = await agent.review_page(pdf_path, page_num, dpi=dpi)
                result.heading_issues.extend(issues)
                result.raw_responses[page_num] = response
            except RuntimeError as e:
                result.heading_issues.append(
                    HeadingIssue(
                        page=page_num,
                        description=f"Could not render page: {e}",
                        severity="warning",
                    )
                )
            except Exception as e:
                logger.warning("Heading hierarchy analysis failed for page %d: %s", page_num, e)
                result.heading_issues.append(
                    HeadingIssue(
                        page=page_num,
                        description=f"Vision analysis error: {e}",
                        severity="warning",
                    )
                )

        return result

    async def analyze_alt_text_quality(
        self,
        pdf_path: Path,
        pages: list[int] | None = None,
        dpi: int = 150,
    ) -> VisionCheckResult:
        """Analyze figure alt-text accuracy and specificity on specified pages."""
        result = VisionCheckResult()

        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        if pages is None:
            pages = list(range(1, total_pages + 1))

        agent = AltTextQualityVisionAgent(self._provider)
        for page_num in pages:
            try:
                issues, response = await agent.review_page(pdf_path, page_num, dpi=dpi)
                result.alt_text_issues.extend(issues)
                if response is not None:
                    result.raw_responses[page_num] = response
            except RuntimeError as e:
                result.alt_text_issues.append(
                    AltTextIssue(
                        page=page_num,
                        figure_index=0,
                        description=f"Could not render page: {e}",
                        severity="warning",
                    )
                )
            except Exception as e:
                logger.warning("Alt text quality analysis failed for page %d: %s", page_num, e)
                result.alt_text_issues.append(
                    AltTextIssue(
                        page=page_num,
                        figure_index=0,
                        description=f"Vision analysis error: {e}",
                        severity="warning",
                    )
                )

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

        ro_result, contrast_result, heading_result, alt_result = await asyncio.gather(
            self.analyze_reading_order(pdf_path, pages, dpi),
            self.analyze_contrast(pdf_path, pages, dpi),
            self.analyze_heading_hierarchy(pdf_path, pages, dpi),
            self.analyze_alt_text_quality(pdf_path, pages, dpi),
        )
        merged = VisionCheckResult(
            reading_order_issues=ro_result.reading_order_issues,
            contrast_issues=contrast_result.contrast_issues,
            heading_issues=heading_result.heading_issues,
            alt_text_issues=alt_result.alt_text_issues,
            raw_responses={
                **{f"reading_order:{k}": v for k, v in ro_result.raw_responses.items()},
                **{f"contrast:{k}": v for k, v in contrast_result.raw_responses.items()},
                **{f"heading:{k}": v for k, v in heading_result.raw_responses.items()},
                **{f"alt_text:{k}": v for k, v in alt_result.raw_responses.items()},
            },
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

    if total_pages > 50 and "VISION_PAGE_SAMPLE_SIZE" not in os.environ:
        try:
            budget = int(os.environ.get("VISION_LARGE_PAGE_SAMPLE_SIZE", "2"))
        except ValueError:
            budget = 2
        if budget <= 0:
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


def _parse_json_response(text: str) -> Any | None:
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
