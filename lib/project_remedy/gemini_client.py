"""Async client for Google Gemini API (vision and text models).

Drop-in replacement for :class:`OllamaClient` using the ``google-genai``
SDK.  Provides the same public interface — ``chat()``, ``ocr()``,
``vision()``, ``health_check()`` — so pipeline stages can use it without
modification.

Key advantage over OllamaClient: native PDF ingestion.  Instead of
rendering each PDF page to a 200-DPI PNG and sending one vision call per
page, GeminiClient passes the raw PDF bytes in a single API call.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from project_remedy.config import PipelineConfig
from project_remedy.token_tracker import tracker
from project_remedy.vision_prompts import ocr_markdown_prompt

logger = logging.getLogger(__name__)

# HTTP / API status codes that warrant automatic retry.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

# Maximum inline PDF size (bytes).  Files larger than this are uploaded
# via the Files API instead of sent inline as Part.from_bytes().
_INLINE_PDF_LIMIT = 50 * 1024 * 1024  # 50 MB


class GeminiClientError(Exception):
    """Raised when a Gemini API call fails after all retries."""


@dataclass
class _UsageBucket:
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    calls: int = 0


class GeminiClient:
    """Reusable async client for all Gemini API interactions.

    Mirrors the :class:`OllamaClient` interface so the existing pipeline
    stages (extractor, converter, validator, vision) work unchanged.

    Parameters
    ----------
    config:
        Pipeline configuration providing API key, model name, concurrency
        limits, and retry settings.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._api_key = config.api.gemini_api_key
        self._model = config.api.gemini_model
        self._vertexai = config.api.gemini_vertexai
        self._gcp_project = config.api.gemini_gcp_project
        self._gcp_location = config.api.gemini_gcp_location
        self._escalation_model = config.api.escalation_model
        self._primary_thinking_level = config.api.gemini_primary_thinking_level
        self._escalation_thinking_level = config.api.gemini_escalation_thinking_level
        # Gemini is multimodal — same model for vision and text.
        self._vision_model = self._model
        self._text_model = self._model
        self._max_retries = config.api.max_retries
        self._backoff_base = config.api.retry_backoff_base
        self._semaphore = asyncio.Semaphore(config.api.max_concurrent_calls)

        # Batch API settings.
        self._batch_enabled = config.api.gemini_batch_enabled
        self._batch_poll_interval = config.api.gemini_batch_poll_interval
        self._batch_timeout = config.api.gemini_batch_timeout

        # Cumulative token usage counters (same interface as OllamaClient).
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_thought_tokens: int = 0
        self._usage_by_bucket: dict[tuple[str, bool], _UsageBucket] = {}

        self._client: genai.Client | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the underlying Gemini client."""
        if self._vertexai:
            if self._api_key:
                # Vertex Express mode: authenticate with the API key (format
                # "AQ...") directly instead of Application Default Credentials.
                # Without this the SDK falls back to ADC which uses the user's
                # personal gcloud login — that can hit "access_denied: Account
                # Restricted" even when the Express API key works fine.
                # See https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode
                self._client = genai.Client(
                    vertexai=True,
                    api_key=self._api_key,
                )
                logger.info(
                    "GeminiClient started via Vertex AI Express (model=%s)",
                    self._model,
                )
            else:
                self._client = genai.Client(
                    vertexai=True,
                    project=self._gcp_project,
                    location=self._gcp_location,
                )
                logger.info(
                    "GeminiClient started via Vertex AI ADC (project=%s, location=%s, model=%s)",
                    self._gcp_project,
                    self._gcp_location,
                    self._model,
                )
        else:
            self._client = genai.Client(api_key=self._api_key)
            logger.info(
                "GeminiClient started via AI Studio (model=%s, escalation_model=%s)",
                self._model,
                self._escalation_model or "(disabled)",
            )

    async def close(self) -> None:
        """Log cumulative token usage.  The genai client has no close()."""
        logger.info(
            "GeminiClient closed. Cumulative tokens — input: %d, output: %d, thoughts: %d",
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_thought_tokens,
        )
        self._client = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            raise RuntimeError("GeminiClient not started. Call start() first.")
        return self._client

    @property
    def total_billable_output_tokens(self) -> int:
        return self.total_output_tokens + self.total_thought_tokens

    @property
    def usage_by_model(self) -> dict[str, dict[str, int]]:
        combined: dict[str, dict[str, int]] = {}
        for (model, _is_batch), bucket in self._usage_by_bucket.items():
            entry = combined.setdefault(
                model,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thought_tokens": 0,
                    "calls": 0,
                },
            )
            entry["input_tokens"] += bucket.input_tokens
            entry["output_tokens"] += bucket.output_tokens
            entry["thought_tokens"] += bucket.thought_tokens
            entry["calls"] += bucket.calls
        return combined

    @property
    def escalation_count(self) -> int:
        return sum(
            bucket.calls
            for (model, _is_batch), bucket in self._usage_by_bucket.items()
            if model == self._escalation_model
        )

    def estimate_cost(self) -> float:
        total = 0.0
        for (model, is_batch), bucket in self._usage_by_bucket.items():
            in_rate, out_rate = self._pricing_for_model(model, is_batch=is_batch)
            total += (bucket.input_tokens / 1_000_000) * in_rate
            total += (
                (bucket.output_tokens + bucket.thought_tokens) / 1_000_000
            ) * out_rate
        return total

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def ocr(
        self,
        file_path: Path | None = None,
        file_url: str | None = None,
    ) -> str:
        """Extract content from a document using the Gemini model.

        For PDFs, the raw file bytes are sent in a single API call (native
        PDF support) instead of rendering per-page PNGs.

        Parameters
        ----------
        file_path:
            Path to a local file (PDF or image).
        file_url:
            Public URL pointing to an image file.

        Returns
        -------
        str
            Extracted Markdown content.
        """
        if file_path is None and file_url is None:
            raise ValueError("Either file_path or file_url must be provided.")

        if file_url:
            return await self._ocr_single_image(image_url=file_url)

        suffix = file_path.suffix.lstrip(".").lower()
        if suffix == "pdf":
            return await self._ocr_pdf(file_path)
        else:
            return await self._ocr_single_image(file_path=file_path)

    async def _ocr_pdf(self, pdf_path: Path) -> str:
        """Send a PDF to Gemini for content extraction in a single call."""
        pdf_bytes = pdf_path.read_bytes()
        file_size = len(pdf_bytes)

        logger.info(
            "OCR: sending PDF %s (%.1f MB) to Gemini natively",
            pdf_path.name,
            file_size / (1024 * 1024),
        )

        prompt = ocr_markdown_prompt(profile="cloud", native_pdf=True)

        if file_size <= _INLINE_PDF_LIMIT:
            # Inline: send raw bytes directly.
            parts = [
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                ),
                prompt,
            ]
        else:
            # Large file: upload via Files API first.
            logger.info(
                "PDF exceeds inline limit (%.1f MB > 50 MB); uploading via Files API.",
                file_size / (1024 * 1024),
            )
            uploaded = await asyncio.to_thread(
                self.client.files.upload,
                file=pdf_path,
            )
            parts = [uploaded, prompt]

        return await self._generate(
            contents=parts,
            config=types.GenerateContentConfig(
                max_output_tokens=16384,
                temperature=0.1,
            ),
        )

    async def _ocr_single_image(
        self,
        file_path: Path | None = None,
        image_url: str | None = None,
        image_b64: str | None = None,
        mime: str = "image/png",
        page_hint: str = "",
    ) -> str:
        """Send a single image to Gemini for content extraction."""
        parts: list[Any] = []

        if image_url:
            parts.append(
                types.Part.from_uri(file_uri=image_url, mime_type="image/png")
            )
        elif image_b64:
            parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(image_b64),
                    mime_type=mime,
                )
            )
        elif file_path:
            raw = file_path.read_bytes()
            suffix = file_path.suffix.lstrip(".").lower()
            mime = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "bmp": "image/bmp",
                "webp": "image/webp",
                "tiff": "image/tiff",
                "tif": "image/tiff",
            }.get(suffix, "image/png")
            parts.append(
                types.Part.from_bytes(data=raw, mime_type=mime)
            )

        parts.append(
            ocr_markdown_prompt(
                profile="cloud",
                page_hint=page_hint,
                native_pdf=False,
            )
        )

        return await self._generate(
            contents=parts,
            config=types.GenerateContentConfig(
                max_output_tokens=16384,
                temperature=0.1,
            ),
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        thinking: bool = False,
        max_tokens: int = 8192,
        temperature: float = 0.3,
    ) -> str:
        """Send a chat completion request.

        Parameters
        ----------
        messages:
            OpenAI-compatible list of ``{"role": ..., "content": ...}`` dicts.
        model:
            Model identifier override.  Defaults to configured model.
        thinking:
            When *True*, enables chain-of-thought thinking mode.
        max_tokens:
            Maximum tokens for the response.
        temperature:
            Sampling temperature.

        Returns
        -------
        str
            The assistant's reply text.
        """
        contents, system_instruction = self._translate_messages(messages)
        target_model = model or self._model

        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system_instruction,
        )

        if thinking and self._is_gemini_25_model(target_model):
            gen_config.thinking_config = types.ThinkingConfig(
                thinking_budget=1024,
            )

        return await self._generate(
            contents=contents,
            config=gen_config,
            model_override=target_model,
        )

    async def vision(
        self,
        image_path: Path | None = None,
        image_url: str | None = None,
        prompt: str = "",
    ) -> str:
        """Analyse an image using multimodal vision.

        Parameters
        ----------
        image_path:
            Path to a local image file.
        image_url:
            Public URL of the image.
        prompt:
            Text prompt to accompany the image.

        Returns
        -------
        str
            The model's description / analysis.
        """
        if image_path is None and image_url is None:
            raise ValueError("Either image_path or image_url must be provided.")

        parts: list[Any] = []

        if image_url:
            parts.append(
                types.Part.from_uri(file_uri=image_url, mime_type="image/png")
            )
        elif image_path:
            raw = image_path.read_bytes()
            suffix = image_path.suffix.lstrip(".").lower()
            mime = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
                "bmp": "image/bmp",
            }.get(suffix, "image/png")
            parts.append(
                types.Part.from_bytes(data=raw, mime_type=mime)
            )

        if prompt:
            parts.append(prompt)

        return await self._generate(
            contents=parts,
            config=types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=0.3,
            ),
        )

    async def compare_images(
        self,
        images: list[bytes],
        prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Send multiple images with a prompt for visual comparison.

        Parameters
        ----------
        images:
            List of PNG image byte buffers.
        prompt:
            Text prompt describing the comparison task.
        max_tokens:
            Maximum response tokens.

        Returns
        -------
        str
            The model's response text.
        """
        parts: list[Any] = [
            types.Part.from_bytes(data=img, mime_type="image/png")
            for img in images
        ]
        parts.append(prompt)

        return await self._generate(
            contents=parts,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.2,
            ),
        )

    async def health_check(self) -> bool:
        """Validate API connectivity with a minimal chat request.

        Returns
        -------
        bool
            *True* if the API responded successfully, *False* otherwise.
        """
        try:
            result = await self.chat(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=8,
                temperature=0.0,
            )
            logger.info("Gemini health check passed (response: %s)", result[:60])
            return True
        except Exception as exc:
            logger.error("Gemini health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    @property
    def batch_enabled(self) -> bool:
        """Whether the Batch API is enabled for this client."""
        return self._batch_enabled

    async def batch_chat(
        self,
        requests: list[dict[str, Any]],
        display_name: str = "batch-job",
    ) -> list[str | None]:
        """Submit a batch of chat requests and wait for results.

        Each request dict must contain:
            - ``request_id``: unique string key for correlating results
            - ``messages``: OpenAI-format message list
            - ``max_tokens`` (optional): defaults to 8192
            - ``temperature`` (optional): defaults to 0.3
            - ``thinking`` (optional): defaults to False

        Returns a list of response strings (or None for failed items),
        ordered to match the input requests list.
        """
        if not requests:
            return []

        # Build inline request dicts.
        inline_requests = []
        request_ids = []
        batch_model = self._model
        for req in requests:
            rid = req["request_id"]
            request_ids.append(rid)
            req_model = req.get("model", self._model)
            if req_model != batch_model and inline_requests:
                raise GeminiClientError(
                    "Gemini batch requests must use a single model per batch job."
                )
            batch_model = req_model
            inline_requests.append(
                self._build_batch_request(
                    messages=req["messages"],
                    request_id=rid,
                    max_tokens=req.get("max_tokens", 8192),
                    temperature=req.get("temperature", 0.3),
                    thinking=req.get("thinking", False),
                    model=req_model,
                )
            )

        logger.info(
            "Submitting batch '%s' with %d requests.",
            display_name,
            len(inline_requests),
        )

        # Submit the batch.
        batch_job = await asyncio.to_thread(
            self.client.batches.create,
            model=batch_model,
            src=inline_requests,
            config={"display_name": display_name},
        )

        logger.info("Batch job created: %s", batch_job.name)

        # Poll until complete.
        batch_job = await self._poll_batch_job(batch_job.name)

        # Extract results and map back by request_id.
        results_map: dict[str, str | None] = {}

        if hasattr(batch_job, "dest") and batch_job.dest:
            responses = getattr(batch_job.dest, "inlined_responses", None) or []
            for resp in responses:
                rid = None
                if hasattr(resp, "metadata") and resp.metadata:
                    rid = resp.metadata.get("key") or resp.metadata.get("request_id")

                if hasattr(resp, "error") and resp.error:
                    logger.warning("Batch request %s failed: %s", rid, resp.error)
                    if rid:
                        results_map[rid] = None
                    continue

                text = ""
                if hasattr(resp, "response") and resp.response:
                    r = resp.response
                    if hasattr(r, "text"):
                        text = r.text or ""
                    elif hasattr(r, "candidates") and r.candidates:
                        parts = r.candidates[0].content.parts
                        text = "".join(p.text for p in parts if hasattr(p, "text"))

                    # Track token usage.
                    if hasattr(r, "usage_metadata") and r.usage_metadata:
                        self._record_usage(
                            batch_model,
                            r.usage_metadata,
                            is_batch=True,
                        )

                if rid:
                    results_map[rid] = text.strip()

        # Return results in original request order.
        return [results_map.get(rid) for rid in request_ids]

    def _build_batch_request(
        self,
        messages: list[dict[str, Any]],
        request_id: str,
        max_tokens: int = 8192,
        temperature: float = 0.3,
        thinking: bool = False,
        model: str | None = None,
    ) -> types.InlinedRequest:
        """Build a single InlinedRequest for batch submission."""
        contents, system_instruction = self._translate_messages(messages)
        target_model = model or self._model

        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system_instruction,
        )
        if thinking and self._is_gemini_25_model(target_model):
            gen_config.thinking_config = types.ThinkingConfig(
                thinking_budget=1024,
            )
        self._apply_reasoning_defaults(gen_config, target_model)

        return types.InlinedRequest(
            contents=contents,
            config=gen_config,
            metadata={"key": request_id},
        )

    async def _poll_batch_job(self, job_name: str) -> Any:
        """Poll a batch job until it completes or times out."""
        import time as _time

        start = _time.monotonic()
        interval = self._batch_poll_interval
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        while True:
            job = await asyncio.to_thread(
                self.client.batches.get, name=job_name
            )
            state = job.state.name if hasattr(job.state, "name") else str(job.state)

            if state in completed_states:
                logger.info("Batch %s finished: %s", job_name, state)
                if state != "JOB_STATE_SUCCEEDED":
                    error_msg = getattr(job, "error", None) or state
                    raise GeminiClientError(
                        f"Batch job {job_name} ended with state {state}: {error_msg}"
                    )
                return job

            elapsed = _time.monotonic() - start
            if elapsed > self._batch_timeout:
                # Cancel and raise.
                try:
                    await asyncio.to_thread(
                        self.client.batches.cancel, name=job_name
                    )
                except Exception:
                    pass
                raise GeminiClientError(
                    f"Batch job {job_name} timed out after {elapsed:.0f}s"
                )

            logger.info(
                "Batch %s state: %s (%.0fs elapsed). Polling in %.0fs...",
                job_name, state, elapsed, interval,
            )
            await asyncio.sleep(interval)
            # Cap polling interval at 120s.
            interval = min(interval * 1.5, 120.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate(
        self,
        contents: list[Any] | str,
        config: types.GenerateContentConfig,
        model_override: str | None = None,
    ) -> str:
        """Execute a generate_content call with retry and concurrency control.

        Parameters
        ----------
        contents:
            Gemini-native contents (list of Parts / strings, or a string).
        config:
            Generation configuration.
        model_override:
            Optional model name override.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        GeminiClientError
            If the request fails after all retry attempts.
        """
        model = model_override or self._model
        self._apply_reasoning_defaults(config, model)
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            async with self._semaphore:
                try:
                    response = await self.client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )

                    # Track token usage.
                    if response.usage_metadata:
                        self._record_usage(model, response.usage_metadata, is_batch=False)

                    # Extract text from response.
                    text = response.text or ""
                    return text.strip()

                except Exception as exc:
                    last_exc = exc
                    error_str = str(exc)

                    # Check if this is a retryable error.
                    is_retryable = any(
                        indicator in error_str.lower()
                        for indicator in [
                            "429", "500", "502", "503",
                            "resource exhausted", "rate limit",
                            "quota", "overloaded", "unavailable",
                            "deadline exceeded", "timeout",
                        ]
                    )

                    if is_retryable and attempt < self._max_retries:
                        wait = self._backoff_base ** attempt
                        logger.warning(
                            "Gemini error on attempt %d/%d: %s. "
                            "Retrying in %.1fs...",
                            attempt,
                            self._max_retries,
                            exc,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    if not is_retryable:
                        raise GeminiClientError(
                            f"Non-retryable Gemini error: {exc}"
                        ) from exc

        raise GeminiClientError(
            f"Gemini call failed after {self._max_retries} attempts: {last_exc}"
        )

    async def generate_raw(
        self,
        contents: list[Any] | str,
        config: types.GenerateContentConfig,
        model_override: str | None = None,
    ) -> str:
        """Raw generation interface for Vision Planner compatibility.

        This is a thin wrapper around _generate() that provides the interface
        expected by vision_planner/grounder.py and vision_planner/planner.py.

        Parameters
        ----------
        contents:
            Gemini-native contents (list of Parts/strings, or string).
        config:
            Generation configuration (temperature, max_tokens, etc).
        model_override:
            Optional model name override.

        Returns
        -------
        str
            Raw text response from the model.
        """
        return await self._generate(contents, config, model_override)

    def _record_usage(self, model: str, usage_metadata: Any, *, is_batch: bool) -> None:
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        thought_tokens = (
            getattr(usage_metadata, "thoughts_token_count", None)
            or getattr(usage_metadata, "total_thought_tokens", None)
            or 0
        )

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_thought_tokens += thought_tokens

        key = (model, is_batch)
        bucket = self._usage_by_bucket.setdefault(key, _UsageBucket())
        bucket.input_tokens += input_tokens
        bucket.output_tokens += output_tokens
        bucket.thought_tokens += thought_tokens
        bucket.calls += 1

        provider = f"gemini:{model}"
        if is_batch:
            provider = f"{provider}:batch"
        tracker.record(
            provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens,
        )
        logger.debug(
            "Gemini token usage — model: %s, batch=%s, in: %d, out: %d, thoughts: %d",
            model,
            is_batch,
            input_tokens,
            output_tokens,
            thought_tokens,
        )

    def _apply_reasoning_defaults(
        self,
        config: types.GenerateContentConfig,
        model: str,
    ) -> None:
        if getattr(config, "thinking_config", None) is not None:
            return

        if self._is_gemini_3_model(model):
            level = (
                self._escalation_thinking_level
                if self._is_escalation_model(model)
                else self._primary_thinking_level
            )
            config.thinking_config = types.ThinkingConfig(thinking_level=level)

    @staticmethod
    def _is_gemini_25_model(model: str) -> bool:
        return model.startswith("gemini-2.5")

    @staticmethod
    def _is_gemini_3_model(model: str) -> bool:
        return model.startswith("gemini-3")

    def _is_escalation_model(self, model: str) -> bool:
        return bool(self._escalation_model) and model == self._escalation_model

    @staticmethod
    def _pricing_for_model(model: str, *, is_batch: bool) -> tuple[float, float]:
        if model.startswith("gemini-3.1-pro-preview"):
            return (1.00, 6.00) if is_batch else (2.00, 12.00)
        if model.startswith("gemini-3.1-flash-lite-preview"):
            return (0.125, 0.75) if is_batch else (0.25, 1.50)
        if model.startswith("gemini-3-flash-preview"):
            return (0.25, 1.50) if is_batch else (0.50, 3.00)
        if model.startswith("gemini-2.5-flash-lite"):
            return (0.05, 0.20) if is_batch else (0.10, 0.40)
        if model.startswith("gemini-2.5-flash"):
            return (0.15, 1.25) if is_batch else (0.30, 2.50)
        return (0.10, 0.40) if is_batch else (0.10, 0.40)

    @staticmethod
    def _translate_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[list[types.Content], str | None]:
        """Convert OpenAI-format messages to Gemini contents + system instruction.

        Parameters
        ----------
        messages:
            List of ``{"role": "system"|"user"|"assistant", "content": str}``
            dicts.

        Returns
        -------
        tuple
            ``(contents, system_instruction)`` where *contents* is a list of
            ``types.Content`` and *system_instruction* is the merged system
            prompt (or *None*).
        """
        system_parts: list[str] = []
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(str(content))
            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=str(content))],
                    )
                )
            elif role == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=str(content))],
                    )
                )

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return contents, system_instruction
