"""Helpers for describing and probing the remediation runtime layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import httpx


@dataclass(frozen=True)
class RuntimePool:
    """A logical model pool and the endpoints it may use."""

    name: str
    model: str
    endpoints: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeLayout:
    """Effective runtime routing resolved from config + env overrides."""

    llm_backend: str
    vision_pool: RuntimePool
    text_pool: RuntimePool
    escalation_backend: str
    escalation_model: str
    escalation_endpoint: str


@dataclass
class EndpointStatus:
    """Observed status for one OpenAI-compatible `/models` endpoint."""

    pool: str
    endpoint: str
    model_hint: str
    ok: bool
    models: list[str] = field(default_factory=list)
    expected_model_available: bool = False
    error: str = ""


def _dedupe_urls(urls: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for url in urls:
        cleaned = str(url or "").strip().rstrip("/")
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return tuple(seen)


def build_runtime_layout(config) -> RuntimeLayout:
    """Return the effective runtime routing for PDF remediation."""
    api = config.api
    base_url = str(getattr(api, "base_url", "") or "http://localhost:11434/v1").rstrip("/")
    vision_base_url = str(getattr(api, "vision_base_url", "") or base_url).rstrip("/")

    vision_endpoints = _dedupe_urls([vision_base_url, *getattr(api, "vision_cluster_nodes", ())])
    text_endpoints = _dedupe_urls([base_url, *getattr(api, "cluster_nodes", ())])

    escalation_backend = str(getattr(api, "escalation_backend", "") or "").strip() or "ollama"
    escalation_endpoint = ""
    if escalation_backend == "ollama":
        escalation_endpoint = str(
            getattr(api, "escalation_base_url", "") or base_url
        ).rstrip("/")

    backend = str(getattr(api, "llm_backend", "") or "").strip() or "ollama"

    # Show the actual vision model being used, not just the Ollama config field.
    if backend == "gemini":
        gemini_model = str(getattr(api, "gemini_model", "") or "").strip()
        vision_model = f"{gemini_model} (Vertex AI)" if getattr(api, "gemini_vertexai", False) else gemini_model
        vision_eps = ["Vertex AI"]
    else:
        vision_model = str(getattr(api, "vision_model", "") or "").strip()
        vision_eps = vision_endpoints

    return RuntimeLayout(
        llm_backend=backend,
        vision_pool=RuntimePool(
            name="vision",
            model=vision_model,
            endpoints=vision_eps,
        ),
        text_pool=RuntimePool(
            name="text",
            model=str(getattr(api, "text_model", "") or "").strip(),
            endpoints=text_endpoints,
        ),
        escalation_backend=escalation_backend,
        escalation_model=str(getattr(api, "escalation_model", "") or "").strip(),
        escalation_endpoint=escalation_endpoint,
    )


async def probe_openai_compatible_endpoint(
    endpoint: str,
    *,
    api_key: str = "ollama",
    timeout_seconds: float = 5.0,
) -> tuple[bool, list[str], str]:
    """Probe `/models` and return `(ok, models, error)`."""
    headers = {
        "Authorization": f"Bearer {api_key or 'ollama'}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(
            base_url=endpoint,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 5.0)),
            trust_env=False,
        ) as client:
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        return False, [], str(exc)

    models = sorted(
        str(item.get("id", "")).strip()
        for item in data.get("data", [])
        if str(item.get("id", "")).strip()
    )
    return True, models, ""


async def collect_runtime_statuses(config) -> list[EndpointStatus]:
    """Probe all configured runtime endpoints."""
    layout = build_runtime_layout(config)
    api_key = str(getattr(config.api, "api_key", "") or "ollama")

    statuses: list[EndpointStatus] = []
    for pool in (layout.vision_pool, layout.text_pool):
        for endpoint in pool.endpoints:
            ok, models, error = await probe_openai_compatible_endpoint(
                endpoint,
                api_key=api_key,
            )
            statuses.append(
                EndpointStatus(
                    pool=pool.name,
                    endpoint=endpoint,
                    model_hint=pool.model,
                    ok=ok,
                    models=models,
                    expected_model_available=(pool.model in models) if ok and pool.model else ok,
                    error=error,
                )
            )

    if layout.escalation_backend == "ollama" and layout.escalation_endpoint:
        endpoint = layout.escalation_endpoint.rstrip("/")
        if endpoint not in {status.endpoint for status in statuses}:
            ok, models, error = await probe_openai_compatible_endpoint(
                endpoint,
                api_key=api_key,
            )
            statuses.append(
                EndpointStatus(
                    pool="escalation",
                    endpoint=endpoint,
                    model_hint=layout.escalation_model,
                    ok=ok,
                    models=models,
                    expected_model_available=(layout.escalation_model in models)
                    if ok and layout.escalation_model
                    else ok,
                    error=error,
                )
            )

    return statuses
