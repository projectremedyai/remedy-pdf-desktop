"""Async client for the Stirling-PDF REST API (Docker container).

Provides PDF manipulation operations — OCR, PDF/A conversion, split, merge,
and metadata inspection — used by the LACCD ADA Document Remediation Pipeline.
All file operations use multipart form uploads against the Stirling-PDF API.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from project_remedy.config import PDFRemediationConfig

logger = logging.getLogger(__name__)

# HTTP status codes that trigger automatic retry.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


class StirlingClientError(Exception):
    """Raised when a Stirling-PDF API call fails after all retries."""


class StirlingClient:
    """Reusable async client for Stirling-PDF file operations.

    Parameters
    ----------
    config:
        PDF remediation configuration providing the Stirling-PDF base URL,
        API key, and retry / concurrency settings from the broader pipeline
        config.
    max_concurrent_calls:
        Maximum number of concurrent requests to Stirling-PDF.
    max_retries:
        Maximum retry attempts per request.
    retry_backoff_base:
        Base for exponential back-off between retries (seconds).
    """

    def __init__(
        self,
        config: PDFRemediationConfig,
        *,
        max_concurrent_calls: int = 3,
        max_retries: int = 3,
        retry_backoff_base: float = 2.0,
    ) -> None:
        self._base_url = config.stirling_url.rstrip("/")
        self._api_key = config.stirling_api_key
        self._max_retries = max_retries
        self._backoff_base = retry_backoff_base
        self._semaphore = asyncio.Semaphore(max_concurrent_calls)

        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the underlying HTTP client."""
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(300.0, connect=30.0),
        )
        logger.info("StirlingClient started (base_url=%s)", self._base_url)

    async def close(self) -> None:
        """Shut down the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("StirlingClient closed")

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("StirlingClient not started. Call start() first.")
        return self._client

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def ocr(self, pdf_path: Path, languages: str = "eng") -> bytes:
        """Run OCR on a PDF and return the processed PDF bytes.

        Parameters
        ----------
        pdf_path:
            Path to the input PDF file.
        languages:
            Tesseract language code(s) for OCR (e.g. ``"eng"``).

        Returns
        -------
        bytes
            OCR-processed PDF content.
        """
        logger.info("Stirling OCR: %s (languages=%s)", pdf_path.name, languages)
        return await self._post_file(
            "/api/v1/misc/ocr",
            pdf_path,
            data={"languages": languages, "ocrType": "force-ocr"},
        )

    async def convert_to_pdfa(self, pdf_path: Path) -> bytes:
        """Convert a PDF to PDF/A-2 format.

        Parameters
        ----------
        pdf_path:
            Path to the input PDF file.

        Returns
        -------
        bytes
            PDF/A-2 compliant PDF content.
        """
        logger.info("Stirling PDF/A conversion: %s", pdf_path.name)
        return await self._post_file(
            "/api/v1/convert/pdf/pdfa",
            pdf_path,
            data={"outputFormat": "pdfa-2"},
        )

    async def get_info(self, pdf_path: Path) -> dict[str, Any]:
        """Retrieve metadata and structural information about a PDF.

        Parameters
        ----------
        pdf_path:
            Path to the input PDF file.

        Returns
        -------
        dict
            JSON object with PDF metadata (page count, author, etc.).
        """
        logger.info("Stirling PDF info: %s", pdf_path.name)
        response_bytes = await self._post_file(
            "/api/v1/general/pdf-info",
            pdf_path,
            expect_json=True,
        )
        import json

        return json.loads(response_bytes)

    async def split(self, pdf_path: Path, pages: str) -> bytes:
        """Split a PDF, keeping only the specified pages.

        Parameters
        ----------
        pdf_path:
            Path to the input PDF file.
        pages:
            Comma-separated page numbers to extract (e.g. ``"1,2,3"``).

        Returns
        -------
        bytes
            PDF containing only the requested pages.
        """
        logger.info("Stirling split: %s (pages=%s)", pdf_path.name, pages)
        return await self._post_file(
            "/api/v1/general/split-pdf",
            pdf_path,
            data={"pages": pages},
        )

    async def merge(self, pdf_paths: list[Path]) -> bytes:
        """Merge multiple PDFs into a single document.

        Parameters
        ----------
        pdf_paths:
            List of paths to the PDF files to merge, in order.

        Returns
        -------
        bytes
            Merged PDF content.
        """
        if not pdf_paths:
            raise ValueError("At least one PDF path is required for merge.")

        names = [p.name for p in pdf_paths]
        logger.info("Stirling merge: %d files (%s)", len(pdf_paths), ", ".join(names))

        files: list[tuple[str, tuple[str, bytes, str]]] = [
            ("fileInput", (p.name, p.read_bytes(), "application/pdf"))
            for p in pdf_paths
        ]

        return await self._post_files(
            "/api/v1/general/merge-pdfs",
            files=files,
        )

    async def health_check(self) -> bool:
        """Verify Stirling-PDF is reachable and responding.

        Performs a lightweight GET against the root URL.

        Returns
        -------
        bool
            *True* if the service is up, *False* otherwise.
        """
        try:
            response = await self.client.get("/api/v1/info/status")
            ok = response.status_code < 400
            if ok:
                logger.info("Stirling health check passed (HTTP %d)", response.status_code)
            else:
                logger.warning(
                    "Stirling health check returned HTTP %d", response.status_code
                )
            return ok
        except Exception as exc:
            logger.error("Stirling health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_file(
        self,
        endpoint: str,
        pdf_path: Path,
        *,
        data: dict[str, str] | None = None,
        expect_json: bool = False,
    ) -> bytes:
        """Upload a single PDF via multipart form and return the response body.

        Parameters
        ----------
        endpoint:
            API path (e.g. ``"/api/v1/misc/ocr"``).
        pdf_path:
            Path to the PDF file to upload.
        data:
            Additional form fields to include in the request.
        expect_json:
            If *True*, the response is expected to be JSON (still returned
            as raw bytes for the caller to parse).

        Returns
        -------
        bytes
            Raw response body (processed PDF or JSON).

        Raises
        ------
        StirlingClientError
            If the request fails after all retry attempts.
        """
        file_bytes = pdf_path.read_bytes()
        files = [("fileInput", (pdf_path.name, file_bytes, "application/pdf"))]
        return await self._post_files(endpoint, files=files, data=data)

    async def _post_files(
        self,
        endpoint: str,
        *,
        files: list[tuple[str, tuple[str, bytes, str]]],
        data: dict[str, str] | None = None,
    ) -> bytes:
        """Execute a multipart POST with retry logic and concurrency control.

        Parameters
        ----------
        endpoint:
            API path relative to the base URL.
        files:
            List of ``(field_name, (filename, bytes, content_type))`` tuples.
        data:
            Additional form fields.

        Returns
        -------
        bytes
            Raw response body.

        Raises
        ------
        StirlingClientError
            If the request fails after all retry attempts.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            async with self._semaphore:
                try:
                    response = await self.client.post(
                        endpoint,
                        files=files,
                        data=data or {},
                    )

                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        wait = self._backoff_base ** attempt
                        logger.warning(
                            "Stirling %s returned %d (attempt %d/%d). "
                            "Retrying in %.1fs...",
                            endpoint,
                            response.status_code,
                            attempt,
                            self._max_retries,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    response.raise_for_status()

                    logger.debug(
                        "Stirling %s succeeded (attempt %d, %d bytes)",
                        endpoint,
                        attempt,
                        len(response.content),
                    )
                    return response.content

                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    logger.error(
                        "Stirling %s HTTP error %d on attempt %d: %s",
                        endpoint,
                        exc.response.status_code,
                        attempt,
                        exc,
                    )
                    if exc.response.status_code in _RETRYABLE_STATUS_CODES:
                        wait = self._backoff_base ** attempt
                        await asyncio.sleep(wait)
                        continue
                    raise StirlingClientError(
                        f"Non-retryable HTTP {exc.response.status_code} "
                        f"from {endpoint}: {exc}"
                    ) from exc

                except (
                    httpx.ConnectError,
                    httpx.ReadTimeout,
                    httpx.WriteTimeout,
                ) as exc:
                    last_exc = exc
                    wait = self._backoff_base ** attempt
                    logger.warning(
                        "Stirling %s network error on attempt %d/%d: %s. "
                        "Retrying in %.1fs...",
                        endpoint,
                        attempt,
                        self._max_retries,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)

        raise StirlingClientError(
            f"Stirling {endpoint} failed after {self._max_retries} attempts: "
            f"{last_exc}"
        )
