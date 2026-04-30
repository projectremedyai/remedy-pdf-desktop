"""Stage 2: Document Download & Inventory.

Async download manager that fetches discovered documents, validates them,
computes SHA-256 hashes for deduplication, and optionally converts legacy
Office formats (.doc, .ppt, .xls) to their modern counterparts via
headless LibreOffice.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from project_remedy.config import PipelineConfig
from project_remedy.database import DatabaseManager
from project_remedy.models import DocumentJob, FileType, JobStatus

logger = logging.getLogger(__name__)

# Mapping of file extensions to expected MIME type prefixes.  We accept
# multiple variants because servers are often misconfigured.
_MIME_EXPECTATIONS: dict[FileType, list[str]] = {
    FileType.PDF: [
        "application/pdf",
    ],
    FileType.DOC: [
        "application/msword",
        "application/octet-stream",
    ],
    FileType.DOCX: [
        "application/vnd.openxmlformats-officedocument.wordprocessingml",
        "application/octet-stream",
        "application/zip",
    ],
    FileType.PPT: [
        "application/vnd.ms-powerpoint",
        "application/octet-stream",
    ],
    FileType.PPTX: [
        "application/vnd.openxmlformats-officedocument.presentationml",
        "application/octet-stream",
        "application/zip",
    ],
    FileType.XLS: [
        "application/vnd.ms-excel",
        "application/octet-stream",
    ],
    FileType.XLSX: [
        "application/vnd.openxmlformats-officedocument.spreadsheetml",
        "application/octet-stream",
        "application/zip",
    ],
}

# Legacy formats that should be converted to their modern equivalents.
_LEGACY_CONVERSION_MAP: dict[FileType, FileType] = {
    FileType.DOC: FileType.DOCX,
    FileType.PPT: FileType.PPTX,
    FileType.XLS: FileType.XLSX,
}

# Default number of concurrent downloads.
_DEFAULT_MAX_CONCURRENT = 10

# Default retry settings.
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 2.0

# Download chunk size.
_CHUNK_SIZE = 64 * 1024  # 64 KiB


def _safe_filename(url: str) -> str:
    """Extract a filesystem-safe filename from a URL."""
    from urllib.parse import unquote, urlparse

    path = urlparse(url).path
    name = unquote(path.split("/")[-1]) if path else "document"
    # Keep only safe characters.
    safe = "".join(c if (c.isalnum() or c in ".-_") else "_" for c in name)
    return safe or "document"


def _check_libreoffice() -> str | None:
    """Return the path to the LibreOffice binary, or None if unavailable."""
    lo_path = shutil.which("libreoffice") or shutil.which("soffice")
    if lo_path is None:
        logger.warning(
            "LibreOffice not found on PATH. Legacy format conversion "
            "(.doc, .ppt, .xls) will be skipped."
        )
    return lo_path


class DocumentDownloader:
    """Stage 2 -- download discovered documents and build a local inventory.

    Parameters
    ----------
    config:
        Full pipeline configuration.
    db:
        An already-connected :class:`DatabaseManager` instance.
    max_concurrent:
        Maximum number of simultaneous downloads. Defaults to 10.
    """

    def __init__(
        self,
        config: PipelineConfig,
        db: DatabaseManager,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    ) -> None:
        self._config = config
        self._db = db
        self._max_concurrent = max_concurrent

        # Retry settings from processing config (or defaults).
        self._max_retries = config.processing.max_retries or _DEFAULT_MAX_RETRIES
        self._backoff_base = config.processing.retry_backoff_base or _DEFAULT_BACKOFF_BASE

        # Base download directory, organized by file type.
        self._download_dir = config.output.output_dir / "downloads"

        # Cache of SHA-256 hashes we have already downloaded, mapping
        # hash -> local path.  Used for deduplication.
        self._hash_cache: dict[str, str] = {}

        # LibreOffice binary path (or None).
        self._libreoffice_path = _check_libreoffice()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def download_all(self, jobs: list[DocumentJob]) -> list[DocumentJob]:
        """Download all documents in *jobs* and return the updated jobs.

        Downloads run concurrently up to ``max_concurrent``.  Each job is
        updated in the database with its local path, file hash, file size,
        and status.  Jobs that fail after all retries are marked FAILED.

        After downloading, legacy formats are converted if LibreOffice is
        available.

        Parameters
        ----------
        jobs:
            DocumentJob instances in DISCOVERED status.

        Returns
        -------
        list[DocumentJob]
            The same jobs, updated with download metadata.
        """
        if not jobs:
            logger.info("No documents to download.")
            return jobs

        logger.info("Starting download of %d documents.", len(jobs))

        # Pre-populate hash cache from any previously downloaded jobs.
        await self._load_hash_cache()

        semaphore = asyncio.Semaphore(self._max_concurrent)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.fields[status]}"),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            overall_task = progress.add_task(
                "Downloading documents",
                total=len(jobs),
                status="starting...",
            )

            async def _guarded_download(job: DocumentJob) -> DocumentJob:
                async with semaphore:
                    result = await self._download_one(job, progress, overall_task)
                    progress.advance(overall_task)
                    return result

            tasks = [_guarded_download(job) for job in jobs]
            completed_jobs = await asyncio.gather(*tasks, return_exceptions=False)

        # Post-download: attempt legacy conversions.
        converted = 0
        for job in completed_jobs:
            if job.status != JobStatus.DOWNLOADED:
                continue
            if job.file_type in _LEGACY_CONVERSION_MAP:
                try:
                    new_path = await self.convert_legacy(job)
                    if new_path:
                        converted += 1
                except Exception:
                    logger.exception(
                        "Legacy conversion failed for %s", job.local_path
                    )

        success_count = sum(1 for j in completed_jobs if j.status == JobStatus.DOWNLOADED)
        fail_count = sum(1 for j in completed_jobs if j.status == JobStatus.FAILED)
        logger.info(
            "Download complete: %d succeeded, %d failed, %d legacy conversions.",
            success_count,
            fail_count,
            converted,
        )

        return list(completed_jobs)

    async def convert_legacy(self, job: DocumentJob) -> Path | None:
        """Convert a legacy Office file to its modern equivalent.

        Uses headless LibreOffice.  Updates the job's ``local_path`` and
        ``file_type`` to point at the converted file.

        Returns the path to the converted file, or ``None`` if conversion
        was skipped or failed.
        """
        if self._libreoffice_path is None:
            logger.debug(
                "Skipping legacy conversion for %s (LibreOffice not available).",
                job.local_path,
            )
            return None

        target_type = _LEGACY_CONVERSION_MAP.get(job.file_type)  # type: ignore[arg-type]
        if target_type is None:
            return None

        src = Path(job.local_path)
        if not src.exists():
            logger.warning("Source file does not exist for conversion: %s", src)
            return None

        # Determine the output format string for LibreOffice.
        format_map = {
            FileType.DOCX: "docx",
            FileType.PPTX: "pptx",
            FileType.XLSX: "xlsx",
        }
        out_format = format_map[target_type]
        out_dir = src.parent

        logger.info("Converting %s -> %s via LibreOffice.", src.name, out_format)

        try:
            proc = await asyncio.create_subprocess_exec(
                self._libreoffice_path,
                "--headless",
                "--convert-to",
                out_format,
                "--outdir",
                str(out_dir),
                str(src),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(
                    "LibreOffice conversion failed (rc=%d): %s",
                    proc.returncode,
                    stderr.decode(errors="replace"),
                )
                return None

        except FileNotFoundError:
            logger.error("LibreOffice binary disappeared: %s", self._libreoffice_path)
            self._libreoffice_path = None
            return None

        # Find the converted file.
        expected_name = src.stem + "." + out_format
        converted_path = out_dir / expected_name
        if not converted_path.exists():
            logger.warning("Expected converted file not found: %s", converted_path)
            return None

        # Update the job.
        job.local_path = str(converted_path)
        job.file_type = target_type
        job.file_size = converted_path.stat().st_size
        await self._db.update_job(job)

        logger.info("Conversion complete: %s (%d bytes)", converted_path, job.file_size)
        return converted_path

    # ------------------------------------------------------------------
    # Internal download logic
    # ------------------------------------------------------------------

    async def _download_one(
        self,
        job: DocumentJob,
        progress: Progress,
        overall_task: TaskID,
    ) -> DocumentJob:
        """Download a single document with retries and validation."""
        url = job.url
        fname = _safe_filename(url)

        # Mark as downloading.
        job.status = JobStatus.DOWNLOADING
        await self._db.update_job_status(job.id, JobStatus.DOWNLOADING)

        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                progress.update(
                    overall_task,
                    status=f"[attempt {attempt}] {fname[:40]}",
                )
                await self._do_download(job, fname)
                return job

            except _DownloadSkippedDuplicate:
                # Deduplication: file already exists with same hash.
                logger.info("Deduplicated (hash match): %s", url)
                return job

            except Exception as exc:
                last_error = f"Attempt {attempt}: {exc}"
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s",
                    attempt,
                    self._max_retries,
                    url,
                    exc,
                )
                if attempt < self._max_retries:
                    backoff = self._backoff_base ** attempt
                    await asyncio.sleep(backoff)

        # All retries exhausted.
        job.status = JobStatus.FAILED
        job.error_message = last_error
        await self._db.update_job_status(job.id, JobStatus.FAILED, error_message=last_error)
        logger.error("Download failed after %d attempts: %s", self._max_retries, url)
        return job

    async def _do_download(self, job: DocumentJob, fname: str) -> None:
        """Execute the actual HTTP download, validate, hash, and store."""
        url = job.url
        file_type = job.file_type

        # Prepare output directory.
        type_dir = self._download_dir / (file_type.value if file_type else "unknown")
        type_dir.mkdir(parents=True, exist_ok=True)

        timeout = httpx.Timeout(60.0, connect=15.0)

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "application/pdf,application/msword,application/vnd.openxmlformats-officedocument.*,*/*;q=0.8",
            },
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                # --- MIME type validation ---
                content_type = response.headers.get("content-type", "").lower()
                if file_type and not self._validate_mime(file_type, content_type):
                    logger.warning(
                        "MIME mismatch for %s: expected %s-like, got '%s'. "
                        "Proceeding anyway (servers may be misconfigured).",
                        url,
                        file_type.value,
                        content_type,
                    )

                # Stream to a temporary file while computing hash.
                hasher = hashlib.sha256()
                tmp_path = type_dir / f".tmp_{fname}"
                total_size = 0

                try:
                    with open(tmp_path, "wb") as fh:
                        async for chunk in response.aiter_bytes(chunk_size=_CHUNK_SIZE):
                            fh.write(chunk)
                            hasher.update(chunk)
                            total_size += len(chunk)
                except Exception:
                    # Clean up partial download.
                    tmp_path.unlink(missing_ok=True)
                    raise

        # --- Validate non-zero size ---
        if total_size == 0:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"Downloaded file is empty: {url}")

        # --- Firecrawl fallback for cloud viewer pages (OneDrive, Adobe) ---
        # If we expected a PDF but got HTML, the URL is a viewer page.
        # Use Firecrawl to scrape and get the rendered content as a PDF.
        if file_type and file_type.value == "pdf":
            with open(tmp_path, "rb") as fh:
                header = fh.read(20)
            if not header.startswith(b"%PDF") and b"<!DOCTYPE" in header or b"<html" in header.lower():
                logger.info(
                    "Downloaded HTML instead of PDF for %s — trying Firecrawl PDF export.",
                    url,
                )
                firecrawl_key = self._config.crawl.firecrawl_api_key
                if firecrawl_key:
                    pdf_bytes = await self._firecrawl_pdf_download(url, firecrawl_key)
                    if pdf_bytes:
                        with open(tmp_path, "wb") as fh:
                            fh.write(pdf_bytes)
                        total_size = len(pdf_bytes)
                        hasher = hashlib.sha256()
                        hasher.update(pdf_bytes)
                        logger.info(
                            "Firecrawl PDF export succeeded: %d bytes", total_size
                        )
                    else:
                        logger.warning("Firecrawl PDF export failed for %s", url)

        file_hash = hasher.hexdigest()

        # --- Deduplication by hash ---
        if file_hash in self._hash_cache:
            existing_path = self._hash_cache[file_hash]
            tmp_path.unlink(missing_ok=True)

            job.local_path = existing_path
            job.file_hash = file_hash
            job.file_size = total_size
            job.status = JobStatus.DOWNLOADED
            await self._db.update_job(job)
            raise _DownloadSkippedDuplicate(file_hash)

        # --- Move to final location ---
        final_name = f"{file_hash[:8]}_{fname}"
        final_path = type_dir / final_name

        # Handle unlikely collision.
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)

        # Update job and caches.
        job.local_path = str(final_path)
        job.file_hash = file_hash
        job.file_size = total_size
        job.status = JobStatus.DOWNLOADED
        await self._db.update_job(job)

        self._hash_cache[file_hash] = str(final_path)

        logger.debug(
            "Downloaded %s -> %s (%d bytes, hash=%s)",
            url,
            final_path,
            total_size,
            file_hash[:12],
        )

    # ------------------------------------------------------------------
    # Firecrawl fallback
    # ------------------------------------------------------------------

    @staticmethod
    async def _firecrawl_pdf_download(url: str, api_key: str) -> bytes | None:
        """Use Firecrawl scrape to render a page and export as raw PDF.

        Falls back to returning None if Firecrawl can't produce a PDF.
        """
        try:
            async with httpx.AsyncClient() as client:
                # First try: ask Firecrawl for rawHtml and extract PDF if embedded
                response = await client.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "url": url,
                        "formats": ["rawHtml"],
                        "waitFor": 5000,  # wait for JS to render
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

                raw_html = data.get("data", {}).get("rawHtml", "")

                # Look for an embedded PDF URL (blob URL or direct link)
                import re
                pdf_urls = re.findall(
                    r'(?:src|href|data-url|content)\s*=\s*["\']([^"\']*\.pdf[^"\']*)["\']',
                    raw_html,
                    re.IGNORECASE,
                )
                if not pdf_urls:
                    # Look for OneDrive download URL pattern in the page
                    pdf_urls = re.findall(
                        r'"(https://[^"]*(?:download|content)[^"]*)"',
                        raw_html,
                    )

                # Try downloading any discovered PDF URLs
                for pdf_url in pdf_urls[:3]:
                    try:
                        pdf_resp = await client.get(
                            pdf_url,
                            follow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0"},
                            timeout=30.0,
                        )
                        if pdf_resp.content[:5] == b"%PDF-":
                            return pdf_resp.content
                    except Exception:
                        continue

                return None
        except Exception as exc:
            logger.warning("Firecrawl PDF download failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_mime(file_type: FileType, content_type: str) -> bool:
        """Check whether *content_type* is acceptable for *file_type*."""
        expected = _MIME_EXPECTATIONS.get(file_type, [])
        if not expected:
            return True  # Unknown type, skip validation.
        return any(content_type.startswith(m) for m in expected)

    async def _load_hash_cache(self) -> None:
        """Pre-populate the hash cache from already-downloaded jobs."""
        downloaded = await self._db.get_jobs_by_status(JobStatus.DOWNLOADED)
        for job in downloaded:
            if job.file_hash and job.local_path:
                self._hash_cache[job.file_hash] = job.local_path
        if self._hash_cache:
            logger.info(
                "Loaded %d file hashes for deduplication.", len(self._hash_cache)
            )


class _DownloadSkippedDuplicate(Exception):
    """Sentinel exception used internally to signal hash deduplication."""

    def __init__(self, file_hash: str) -> None:
        self.file_hash = file_hash
        super().__init__(f"Duplicate file (hash={file_hash[:12]})")
