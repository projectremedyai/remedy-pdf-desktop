"""Z.ai GLM API client — OpenAI-compatible interface.

Supports GLM-5.1 (text/thinking), GLM-4.6V-Flash (vision, free),
and GLM-4.7-Flash (text, free) via the standard OpenAI chat completions
protocol at https://api.z.ai/api/paas/v4.

Usage::

    from project_remedy.zai_client import ZaiClient

    client = ZaiClient()  # reads from env vars
    response = client.generate("Hello, world!")
    response = client.generate_raw(messages=[...], model="glm-5.1")
    response = client.generate_vision(image_bytes, "Describe this image")
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5.1"
DEFAULT_VISION_MODEL = "glm-4.6v-flash"


class ZaiClient:
    """Lightweight Z.ai GLM client using OpenAI-compatible chat completions."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        vision_base_url: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        timeout: float = 300.0,
    ):
        self.api_key = api_key or os.environ.get("ZAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("ZAI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.vision_base_url = (
            vision_base_url
            or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        ).rstrip("/")
        self.vision_api_key = os.environ.get("OPENROUTER_API_KEY", self.api_key)
        self.model = model or os.environ.get("ZAI_PLANNER_MODEL", DEFAULT_MODEL)
        self.vision_model = vision_model or os.environ.get("ZAI_VISION_MODEL", DEFAULT_VISION_MODEL)
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

        if not self.api_key:
            raise ValueError("ZAI_API_KEY not set")

    def _headers(self, *, use_vision: bool = False) -> dict[str, str]:
        key = self.vision_api_key if use_vision else self.api_key
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _post(
        self,
        payload: dict[str, Any],
        *,
        use_vision_endpoint: bool = False,
        max_retries: int = 10,
    ) -> dict[str, Any]:
        """Post to chat/completions endpoint with retry on 429/5xx."""
        base = self.vision_base_url if use_vision_endpoint else self.base_url
        url = f"{base}/chat/completions"
        t0 = time.time()

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self._client.post(url, headers=self._headers(use_vision=use_vision_endpoint), json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    wait = min(2 ** attempt * 2, 30)  # 2s, 4s, 8s, max 30s
                    logger.warning(
                        "zai_client: %d on attempt %d, retrying in %ds",
                        response.status_code, attempt + 1, wait,
                    )
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                data = response.json()
                elapsed = time.time() - t0

                usage = data.get("usage", {})
                logger.debug(
                    "zai_client: model=%s tokens=%d+%d time=%.1fs",
                    payload.get("model", "?"),
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    elapsed,
                )
                return data

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    wait = min(2 ** attempt * 2, 30)
                    logger.warning("zai_client: HTTP %d, retry %d in %ds", e.response.status_code, attempt + 1, wait)
                    time.sleep(wait)
                    continue
                logger.error("zai_client: HTTP %d: %s", e.response.status_code, e.response.text[:500])
                raise
            except Exception as e:
                last_error = e
                logger.error("zai_client: request failed: %s", e)
                raise

        # All retries exhausted
        if last_error:
            raise last_error
        raise RuntimeError("zai_client: all retries exhausted")

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        think: bool = False,
    ) -> str:
        """Simple text generation."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self.generate_raw(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
        )

    def generate_raw(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        contents: list | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        think: bool = False,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        **kwargs,
    ) -> str:
        """Raw chat completion. Returns the assistant message content.

        Accepts either ``messages`` (OpenAI format) or ``contents`` for
        compatibility with the VP pipeline.
        """
        if messages is None and contents is not None:
            messages = self._convert_contents_to_messages(contents)
        if messages is None:
            messages = []

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if think:
            payload["thinking"] = {"type": "enabled"}

        if tools:
            payload["tools"] = tools

        if tool_choice:
            payload["tool_choice"] = tool_choice

        # Route vision models to OpenRouter (or Z.ai general endpoint for free models)
        use_model = payload.get("model", "")
        is_vision = "/" in use_model or "v-flash" in use_model.lower() or "5v" in use_model.lower()
        data = self._post(payload, use_vision_endpoint=is_vision)

        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})

        # Handle tool calls
        if message.get("tool_calls"):
            return json.dumps(message["tool_calls"])

        return message.get("content", "")

    def generate_vision(
        self,
        image_data: bytes | str,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Vision generation with image input."""
        if isinstance(image_data, bytes):
            b64 = base64.b64encode(image_data).decode("ascii")
        else:
            b64 = image_data  # already base64

        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        })

        payload: dict[str, Any] = {
            "model": model or self.vision_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        vision_model = model or self.vision_model
        data = self._post(payload, use_vision_endpoint=True)
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def _convert_contents_to_messages(self, contents: list) -> list[dict[str, Any]]:
        """Convert a contents list to OpenAI-style messages."""
        messages: list[dict[str, Any]] = []
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, bytes):
                # Image bytes — convert to vision message
                b64 = base64.b64encode(item).decode("ascii")
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                })
            elif isinstance(item, dict):
                # Already a message dict
                if "role" in item:
                    messages.append(item)
                elif "text" in item:
                    messages.append({"role": "user", "content": item["text"]})
                elif "inline_data" in item:
                    b64 = item["inline_data"].get("data", "")
                    mime = item["inline_data"].get("mime_type", "image/png")
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                        ],
                    })
            elif isinstance(item, list):
                # List of mixed text/image parts.
                parts = []
                for part in item:
                    if isinstance(part, str):
                        parts.append({"type": "text", "text": part})
                    elif isinstance(part, bytes):
                        b64 = base64.b64encode(part).decode("ascii")
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        })
                    elif isinstance(part, dict) and "text" in part:
                        parts.append({"type": "text", "text": part["text"]})
                if parts:
                    messages.append({"role": "user", "content": parts})
        return messages

    async def start(self):
        """No-op — ZaiClient is synchronous and ready immediately."""
        pass

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
