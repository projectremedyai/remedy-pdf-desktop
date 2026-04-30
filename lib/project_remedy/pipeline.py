"""Main Pipeline Orchestrator for the LACCD ADA Document Remediation Pipeline.

Wires all eight stages together into a coherent, concurrent, fault-tolerant
pipeline that can be run end-to-end or stage-by-stage.

Stages:
    1. Crawl     -- discover document links on campus websites
    2. Download  -- fetch documents to local storage
    3. Extract   -- OCR / native-parse documents to Markdown
    4-5. Convert -- plan and generate WCAG 2.1 AA compliant HTML
    6. Vision    -- process images with alt text / SVG recreation
    7. Validate  -- triple-layer validation with auto-remediation
    8. Deploy    -- organise output, generate manifests
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from project_remedy.config import CampusConfig, PipelineConfig
from project_remedy.converter import HTMLConverter
from project_remedy.crawler import DocumentCrawler
from project_remedy.database import DatabaseManager
from project_remedy.downloader import DocumentDownloader
from project_remedy.extractor import ContentExtractor
from project_remedy.models import DocumentJob, FileType, JobStatus, RenderedPage
from project_remedy.validator import AccessibilityValidator
from project_remedy.vision import VisionProcessor
from project_remedy.ollama_client import OllamaClient
from project_remedy.html_to_pdf import HTMLToPDFConverter
from project_remedy.pdf_acceptance import evaluate_pdf_acceptance, validate_pdf_openability
from project_remedy.pdf_remediator import PDFRemediator
from project_remedy.pdf_fixer import FixReport, fix_and_verify
from project_remedy.pdf_specialists import run_specialist_pdf_repair
from project_remedy.pdf_vision import create_escalation_provider
from project_remedy.office_acceptance import evaluate_office_acceptance, validate_office_package
from project_remedy.office_remediator import OfficeRemediationResult, OfficeRemediator
# StirlingClient no longer required — OCR via Gemini, PDF metadata via pikepdf
from project_remedy.accessibility_strategies import RemediationStrategyRunner
from project_remedy.html_strategy_remediator import HTMLStrategyRemediator
from project_remedy.vision_planner.pipeline import run_vision_plan

logger = logging.getLogger(__name__)

# Type alias for the LLM client (OllamaClient or GeminiClient — same interface).
LLMClient = Any


# ---------------------------------------------------------------------------
# Pipeline report
# ---------------------------------------------------------------------------


@dataclass
class PipelineReport:
    """Summary statistics from a pipeline run."""

    total_documents: int = 0
    discovered: int = 0
    downloaded: int = 0
    extracted: int = 0
    converted: int = 0
    validated: int = 0
    pdf_remediated: int = 0
    failed: int = 0
    flagged: int = 0
    average_lighthouse_score: float = 0.0
    primary_model: str = ""
    escalation_model: str = ""
    escalation_count: int = 0
    thought_tokens: int = 0
    warning_documents: int = 0
    warning_breakdown: dict[str, int] = field(default_factory=dict)
    total_api_cost_estimate: float = 0.0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        """Return a list of human-readable summary lines."""
        mins, secs = divmod(self.duration_seconds, 60)
        lines = [
            f"Total documents:          {self.total_documents}",
            f"Discovered:               {self.discovered}",
            f"Downloaded:               {self.downloaded}",
            f"Extracted:                {self.extracted}",
            f"Converted (HTML):         {self.converted}",
            f"Documents remediated:     {self.pdf_remediated}",
            f"Validated (passed):       {self.validated}",
            f"Flagged (needs review):   {self.flagged}",
            f"Failed:                   {self.failed}",
            f"Avg Lighthouse score:     {self.average_lighthouse_score:.1f}/100",
        ]
        if self.primary_model:
            lines.append(f"Primary model:            {self.primary_model}")
        if self.escalation_model:
            lines.append(f"Escalation model:         {self.escalation_model}")
        if self.escalation_count:
            lines.append(f"Escalation retries:       {self.escalation_count}")
        if self.thought_tokens:
            lines.append(f"Thought tokens:           {self.thought_tokens:,}")
        if self.warning_documents:
            lines.append(f"Remediated w/ warnings:   {self.warning_documents}")
        lines.extend(
            [
                f"Est. API cost:            ${self.total_api_cost_estimate:.2f}",
                f"Duration:                 {int(mins)}m {secs:.1f}s",
                f"Errors:                   {len(self.errors)}",
            ]
        )
        return lines


# ---------------------------------------------------------------------------
# Image placeholder regex for Stage 6
# ---------------------------------------------------------------------------

_IMAGE_PLACEHOLDER_RE = re.compile(
    r'!\[([^\]]*)\]\(([^)]+)\)|'                        # Markdown: ![alt](path)
    r'<img[^>]+alt="([^"]*)"[^>]+src="([^"]*)"[^>]*/?>|'  # HTML <img> with src
    r'alt="\[Image description pending[^"]*\]"',          # Pending vision markers
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_")


def strip_hash_prefix(name: str) -> str:
    """Remove 8-hex-char hash prefix (e.g. '00087451_') from a filename."""
    return _HASH_PREFIX_RE.sub("", name)


def _pdf_filename_from_job(job: DocumentJob) -> str:
    """Derive a human-readable PDF filename from the original document URL."""
    from urllib.parse import unquote

    raw = unquote(job.url.rsplit("/", 1)[-1])
    # Strip the original extension (.pdf, .docx, etc.)
    name = raw.rsplit(".", 1)[0] if "." in raw else raw
    name = strip_hash_prefix(name)
    # Sanitise: keep alphanumerics, spaces, hyphens, underscores
    name = re.sub(r"[^\w\s\-]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    if not name:
        name = job.id[:12]
    return f"{name}.pdf"


def _document_filename_from_job(job: DocumentJob) -> str:
    """Derive a human-readable output filename preserving the source extension."""
    from urllib.parse import unquote

    raw = unquote(job.url.rsplit("/", 1)[-1])
    suffix = Path(job.local_path).suffix or Path(raw).suffix or ""
    name = raw.rsplit(".", 1)[0] if "." in raw else raw
    name = strip_hash_prefix(name)
    name = re.sub(r"[^\w\s\-]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    if not name:
        name = job.id[:12]
    return f"{name}{suffix}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Orchestrates the full 8-stage document remediation pipeline.

    Parameters
    ----------
    config:
        Fully resolved pipeline configuration.
    """

    def __init__(
        self,
        config: PipelineConfig,
        campus: CampusConfig | None = None,
    ) -> None:
        self._config = config
        self._campus = campus
        self._campus_id = campus.code if campus else ""
        self._html_workflow_enabled = config.processing.html_workflow_enabled

        # Components -- initialised lazily by start().
        self._db = DatabaseManager(config.output.db_path)

        # Select LLM client based on config.
        if config.api.llm_backend == "gemini":
            from project_remedy.gemini_client import GeminiClient
            self._llm: LLMClient = GeminiClient(config)
        else:
            self._llm = OllamaClient(config)
        self._escalation_llm: LLMClient | None = None
        self._escalation_attempts = 0
        if config.api.escalation_model:
            if config.api.escalation_backend == "gemini":
                from project_remedy.gemini_client import GeminiClient

                escalation_api = replace(
                    config.api,
                    gemini_model=config.api.escalation_model,
                    gemini_batch_enabled=False,
                )
                escalation_config = replace(config, api=escalation_api)
                self._escalation_llm = GeminiClient(escalation_config)
            elif config.api.escalation_backend == "ollama":
                escalation_concurrency = config.api.max_concurrent_calls
                raw_limit = os.environ.get("OLLAMA_ESCALATION_MAX_INFLIGHT", "").strip()
                if raw_limit:
                    try:
                        escalation_concurrency = max(1, int(raw_limit))
                    except ValueError:
                        logger.warning(
                            "Invalid OLLAMA_ESCALATION_MAX_INFLIGHT=%r; using %d",
                            raw_limit,
                            escalation_concurrency,
                        )
                escalation_api = replace(
                    config.api,
                    llm_backend="ollama",
                    base_url=(config.api.escalation_base_url or config.api.base_url),
                    vision_model=config.api.escalation_model,
                    text_model=config.api.escalation_model,
                    max_concurrent_calls=escalation_concurrency,
                )
                escalation_config = replace(config, api=escalation_api)
                self._escalation_llm = OllamaClient(escalation_config)

        self._crawler = DocumentCrawler(config, self._db, campus=campus)
        self._downloader = DocumentDownloader(
            config, self._db, max_concurrent=config.processing.max_concurrent_calls
        )
        self._extractor = ContentExtractor(config, self._llm, self._db)
        self._converter = HTMLConverter(config, self._llm, self._db, campus=campus)
        self._vision = VisionProcessor(self._llm)
        self._validator = AccessibilityValidator(config, self._llm, self._db)
        self._escalation_extractor = (
            ContentExtractor(config, self._escalation_llm, self._db)
            if self._escalation_llm
            else None
        )
        self._escalation_converter = (
            HTMLConverter(config, self._escalation_llm, self._db, campus=campus)
            if self._escalation_llm
            else None
        )
        self._escalation_validator = (
            AccessibilityValidator(config, self._escalation_llm, self._db)
            if self._escalation_llm
            else None
        )
        self._html_strategy = HTMLStrategyRemediator()
        self._accessibility_strategies = RemediationStrategyRunner()
        self._office_remediator = OfficeRemediator(self._llm)
        self._escalation_office_remediator = (
            OfficeRemediator(self._escalation_llm) if self._escalation_llm else None
        )
        self._office_remediator = OfficeRemediator(self._llm)
        self._escalation_office_remediator = (
            OfficeRemediator(self._escalation_llm) if self._escalation_llm else None
        )

        # HTML-to-Tagged-PDF converter (Playwright + pikepdf).
        self._pdf_remediation_enabled = config.pdf_remediation.enabled
        self._html_to_pdf: HTMLToPDFConverter | None = None
        if self._pdf_remediation_enabled and self._html_workflow_enabled:
            self._html_to_pdf = HTMLToPDFConverter(
                max_concurrent=config.processing.max_concurrent_calls,
            )

        # Legacy PDF remediator — kept for batch-mode helpers / triage.
        self._pdf_remediator: PDFRemediator | None = None
        self._escalation_pdf_remediator: PDFRemediator | None = None
        if self._pdf_remediation_enabled:
            self._pdf_remediator = PDFRemediator(
                config, self._llm,
            )
            if self._escalation_llm:
                self._escalation_pdf_remediator = PDFRemediator(
                    config, self._escalation_llm,
                )

        # Concurrency limiter for CPU/API-bound stages.
        self._semaphore = asyncio.Semaphore(config.processing.max_concurrent_calls)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open database connection and initialise the API client."""
        await self._db.connect()
        await self._llm.start()
        if self._escalation_llm:
            await self._escalation_llm.start()
        if self._html_to_pdf:
            await self._html_to_pdf.start()
        logger.info("Pipeline started (backend=%s).", self._config.api.llm_backend)

    async def close(self) -> None:
        """Release all resources."""
        if self._html_to_pdf:
            await self._html_to_pdf.close()
        await self._llm.close()
        if self._escalation_llm:
            await self._escalation_llm.close()
        await self._db.close()
        logger.info("Pipeline closed.")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def run(self) -> PipelineReport:
        """Execute the full 8-stage pipeline.

        Returns
        -------
        PipelineReport
            Summary of the entire run.
        """
        t0 = time.monotonic()
        report = PipelineReport()

        # Reset any jobs left mid-flight from a previous crash.
        reset = await self._db.reset_incomplete_jobs()
        if reset:
            logger.info("Reset %d incomplete jobs from previous run.", reset)

        # Stage 1: Crawl
        logger.info("=== Stage 1: Crawl ===")
        try:
            discovered = await self._crawler.crawl()
            report.discovered = len(discovered)
            logger.info("Crawl discovered %d documents.", len(discovered))
        except Exception as exc:
            msg = f"Crawl failed: {exc}"
            logger.exception(msg)
            report.errors.append(msg)

        # Stage 2: Download
        logger.info("=== Stage 2: Download ===")
        try:
            to_download = await self._db.get_jobs_by_status(
                JobStatus.DISCOVERED, campus_id=self._campus_id or None
            )
            if to_download:
                downloaded = await self._downloader.download_all(to_download)
                report.downloaded = sum(
                    1 for j in downloaded if j.status == JobStatus.DOWNLOADED
                )
            logger.info("Downloaded %d documents.", report.downloaded)
        except Exception as exc:
            msg = f"Download stage failed: {exc}"
            logger.exception(msg)
            report.errors.append(msg)

        if not self._html_workflow_enabled:
            logger.info("=== Stage 4: Direct PDF Remediation (HTML workflow disabled) ===")
            to_remediate = await self._db.get_jobs_by_status(
                JobStatus.DOWNLOADED,
                JobStatus.CONVERTED,
                JobStatus.PDF_REMEDIATING,
                JobStatus.DOCUMENT_REMEDIATING,
                campus_id=self._campus_id or None,
            )
            await self._run_document_remediation(to_remediate, report)

            all_jobs = await self._db.get_all_jobs(
                campus_id=self._campus_id or None
            )
            report.total_documents = len(all_jobs)
            report.failed = sum(1 for j in all_jobs if j.status == JobStatus.FAILED)
            report.average_lighthouse_score = 0.0
            report.warning_documents = sum(1 for j in all_jobs if j.warning_message)
            report.warning_breakdown = self._compute_warning_breakdown(all_jobs)
            if self._config.api.llm_backend == "gemini":
                report.primary_model = self._config.api.gemini_model
                report.escalation_model = (
                    self._config.api.escalation_model if self._escalation_llm else ""
                )
                report.escalation_count = self._escalation_attempts
                report.thought_tokens = getattr(self._llm, "total_thought_tokens", 0) + (
                    getattr(self._escalation_llm, "total_thought_tokens", 0)
                    if self._escalation_llm
                    else 0
                )
            report.total_api_cost_estimate = self._estimate_api_cost()
            report.duration_seconds = time.monotonic() - t0
            logger.info("Document-only run complete in %.1fs.", report.duration_seconds)
            return report

        # Stage 3: Extract
        logger.info("=== Stage 3: Extract ===")
        to_extract = await self._db.get_jobs_by_status(
            JobStatus.DOWNLOADED, campus_id=self._campus_id or None
        )
        extracted_jobs = await self._run_stage(
            to_extract, self._extract_one, "extraction", report
        )
        report.extracted = len(extracted_jobs)

        # Stage 4-5: Convert (plan + generate) — batch mode if available
        logger.info("=== Stage 4-5: Convert ===")
        to_convert = await self._db.get_jobs_by_status(
            JobStatus.EXTRACTED, campus_id=self._campus_id or None
        )
        if hasattr(self._llm, "batch_enabled") and self._llm.batch_enabled and to_convert:
            converted_jobs = await self._run_conversion_batched(to_convert, report)
        else:
            converted_jobs = await self._run_stage(
                to_convert, self._convert_one, "conversion", report
            )
        report.converted = len(converted_jobs)

        # Stage 6: Vision
        logger.info("=== Stage 6: Vision ===")
        to_vision = await self._db.get_jobs_by_status(
            JobStatus.CONVERTED, campus_id=self._campus_id or None
        )
        for job in to_vision:
            try:
                await self._vision_process_job(job)
            except Exception as exc:
                msg = f"Vision processing failed for {job.id}: {exc}"
                logger.warning(msg)
                report.errors.append(msg)

        # Stage 7: Validate (with optional pre-LLM programmatic fixes)
        logger.info("=== Stage 7: Validate ===")
        to_validate = await self._db.get_jobs_by_status(
            JobStatus.CONVERTED, campus_id=self._campus_id or None
        )

        # Pre-LLM programmatic fixes if enabled
        if self._config.pdf_remediation.use_programmatic_fixes:
            for job in to_validate:
                await self._apply_strategy_fixes(job)
                await self._apply_accessibility_strategies(job)

        validated_jobs = await self._run_stage(
            to_validate, self._validate_one, "validation", report
        )
        report.validated = sum(
            1 for j in validated_jobs if j.status == JobStatus.VALIDATED
        )
        report.flagged = sum(
            1 for j in validated_jobs if j.status == JobStatus.FLAGGED
        )

        # Stage 7b: HTML-to-Tagged-PDF (if enabled)
        if self._pdf_remediation_enabled and self._html_to_pdf:
            logger.info("=== Stage 7b: HTML-to-Tagged-PDF ===")
            to_pdf_remediate = await self._db.get_jobs_by_status(
                JobStatus.VALIDATED, JobStatus.FLAGGED,
                campus_id=self._campus_id or None,
            )
            pdf_fmt = self._config.pdf_remediation.output_format
            if pdf_fmt in ("pdf", "both"):
                # Convert ALL jobs with generated HTML — not just PDFs.
                html_jobs = [
                    j for j in to_pdf_remediate if j.generated_html
                ]
                pdf_remediated = await self._run_stage(
                    html_jobs,
                    self._html_to_pdf_one,
                    "pdf_remediation",
                    report,
                )
                logger.info(
                    "HTML-to-PDF converted %d documents.", len(pdf_remediated),
                )

        # Compute final stats.
        all_jobs = await self._db.get_all_jobs(
            campus_id=self._campus_id or None
        )
        report.total_documents = len(all_jobs)
        report.failed = sum(1 for j in all_jobs if j.status == JobStatus.FAILED)
        report.average_lighthouse_score = self._compute_avg_lighthouse(all_jobs)
        report.warning_documents = sum(1 for j in all_jobs if j.warning_message)
        report.warning_breakdown = self._compute_warning_breakdown(all_jobs)
        if self._config.api.llm_backend == "gemini":
            report.primary_model = self._config.api.gemini_model
            report.escalation_model = (
                self._config.api.escalation_model if self._escalation_llm else ""
            )
            report.escalation_count = self._escalation_attempts
            report.thought_tokens = getattr(self._llm, "total_thought_tokens", 0) + (
                getattr(self._escalation_llm, "total_thought_tokens", 0)
                if self._escalation_llm
                else 0
            )
        report.total_api_cost_estimate = self._estimate_api_cost()
        report.duration_seconds = time.monotonic() - t0

        logger.info("Pipeline run complete in %.1fs.", report.duration_seconds)
        return report

    # ------------------------------------------------------------------
    # Individual stage entry points (for CLI commands)
    # ------------------------------------------------------------------

    async def crawl(self) -> list[DocumentJob]:
        """Run Stages 1-2 only: crawl and download.

        Returns
        -------
        list[DocumentJob]
            All jobs after crawl + download.
        """
        await self._db.reset_incomplete_jobs()

        # Stage 1
        discovered = await self._crawler.crawl()
        logger.info("Crawl discovered %d documents.", len(discovered))

        # Stage 2
        to_download = await self._db.get_jobs_by_status(
            JobStatus.DISCOVERED, campus_id=self._campus_id or None
        )
        if to_download:
            await self._downloader.download_all(to_download)

        return await self._db.get_all_jobs(campus_id=self._campus_id or None)

    async def process(self) -> list[DocumentJob]:
        """Run Stages 3-6 only: extract, convert, vision.

        Picks up DOWNLOADED jobs and moves them through extraction,
        conversion, and vision processing.

        Returns
        -------
        list[DocumentJob]
            Jobs processed in this run.
        """
        await self._db.reset_incomplete_jobs()
        report = PipelineReport()
        processed: list[DocumentJob] = []

        if not self._html_workflow_enabled:
            to_remediate = await self._db.get_jobs_by_status(
                JobStatus.DOWNLOADED,
                JobStatus.CONVERTED,
                JobStatus.PDF_REMEDIATING,
                JobStatus.DOCUMENT_REMEDIATING,
                campus_id=self._campus_id or None,
            )
            return await self._run_document_remediation(to_remediate, report)

        # Stage 3: Extract
        to_extract = await self._db.get_jobs_by_status(
            JobStatus.DOWNLOADED, campus_id=self._campus_id or None
        )
        extracted = await self._run_stage(
            to_extract, self._extract_one, "extraction", report
        )
        processed.extend(extracted)

        # Stage 4-5: Convert (batch mode if available)
        to_convert = await self._db.get_jobs_by_status(
            JobStatus.EXTRACTED, campus_id=self._campus_id or None
        )
        if hasattr(self._llm, "batch_enabled") and self._llm.batch_enabled and to_convert:
            converted = await self._run_conversion_batched(to_convert, report)
        else:
            converted = await self._run_stage(
                to_convert, self._convert_one, "conversion", report
            )
        processed.extend(converted)

        # Stage 6: Vision
        to_vision = await self._db.get_jobs_by_status(
            JobStatus.CONVERTED, campus_id=self._campus_id or None
        )
        for job in to_vision:
            try:
                await self._vision_process_job(job)
            except Exception as exc:
                logger.warning("Vision processing failed for %s: %s", job.id, exc)

        return processed

    async def validate_all(self) -> list[DocumentJob]:
        """Run Stage 7 only: validate converted jobs.

        Returns
        -------
        list[DocumentJob]
            Jobs that were validated in this run.
        """
        await self._db.reset_incomplete_jobs()
        report = PipelineReport()

        if not self._html_workflow_enabled:
            logger.info(
                "HTML workflow disabled; PDF acceptance is enforced during pdf remediation."
            )
            return await self._db.get_jobs_by_status(
                JobStatus.PDF_REMEDIATED,
                JobStatus.DOCUMENT_REMEDIATED,
                JobStatus.FLAGGED,
                campus_id=self._campus_id or None,
            )

        to_validate = await self._db.get_jobs_by_status(
            JobStatus.CONVERTED, campus_id=self._campus_id or None
        )
        return await self._run_stage(
            to_validate, self._validate_one, "validation", report
        )

    async def retry_failed(self) -> list[DocumentJob]:
        """Reset FAILED jobs back to DISCOVERED and reprocess them.

        Returns
        -------
        list[DocumentJob]
            The reset jobs.
        """
        failed = await self._db.get_jobs_by_status(
            JobStatus.FAILED, campus_id=self._campus_id or None
        )
        if not failed:
            logger.info("No failed jobs to retry.")
            return []

        for job in failed:
            job.status = JobStatus.DISCOVERED
            job.error_message = ""
            await self._db.update_job(job)

        logger.info("Reset %d failed jobs to DISCOVERED.", len(failed))
        return failed

    # ------------------------------------------------------------------
    # Internal: per-job stage executors
    # ------------------------------------------------------------------

    def _gemini_escalation_enabled(self) -> bool:
        return self._escalation_llm is not None

    def _note_escalation(self, stage: str, job: DocumentJob, reason: str) -> None:
        self._escalation_attempts += 1
        logger.warning(
            "Retrying %s for job %s with escalation model %s: %s",
            stage,
            job.id,
            self._config.api.escalation_model,
            reason,
        )

    async def _extract_with(self, extractor: ContentExtractor, job: DocumentJob) -> DocumentJob:
        await extractor.extract(job)
        return job

    async def _convert_with(self, converter: HTMLConverter, job: DocumentJob) -> DocumentJob:
        await converter.convert(job)
        return job

    async def _run_document_remediation(
        self,
        jobs: list[DocumentJob],
        report: PipelineReport,
    ) -> list[DocumentJob]:
        supported_jobs = [
            job
            for job in jobs
            if job.file_type in {FileType.PDF, FileType.DOCX, FileType.PPTX, FileType.XLSX}
        ]
        unsupported_jobs = [job for job in jobs if job not in supported_jobs]
        for job in unsupported_jobs:
            logger.info(
                "Skipping unsupported document job %s in document-only mode (%s).",
                job.id,
                job.file_type.value if job.file_type else "unknown",
            )
        if not supported_jobs:
            return []
        remediated = await self._run_stage(
            supported_jobs,
            self._document_remediate_one,
            "document_remediation",
            report,
        )
        report.pdf_remediated += sum(
            1
            for job in remediated
            if job.status in {JobStatus.PDF_REMEDIATED, JobStatus.DOCUMENT_REMEDIATED}
        )
        report.flagged += sum(
            1 for job in remediated if job.status == JobStatus.FLAGGED
        )
        return remediated

    async def _document_remediate_one(self, job: DocumentJob) -> DocumentJob:
        if job.file_type == FileType.PDF:
            return await self._pdf_remediate_one(job)
        if job.file_type in {FileType.DOCX, FileType.PPTX, FileType.XLSX}:
            return await self._office_remediate_one(job)
        return job

    async def _extract_one(self, job: DocumentJob) -> DocumentJob:
        """Extract content from a single downloaded document."""
        try:
            return await self._extract_with(self._extractor, job)
        except Exception as exc:
            if not self._escalation_extractor:
                raise
            self._note_escalation("extraction", job, str(exc))
            job.error_message = ""
            return await self._extract_with(self._escalation_extractor, job)

    async def _convert_one(self, job: DocumentJob) -> DocumentJob:
        """Plan and generate HTML for a single extracted document."""
        try:
            return await self._convert_with(self._converter, job)
        except Exception as exc:
            if not self._escalation_converter:
                raise
            self._note_escalation("conversion", job, str(exc))
            job.error_message = ""
            return await self._convert_with(self._escalation_converter, job)

    async def _run_conversion_batched(
        self,
        jobs: list[DocumentJob],
        report: PipelineReport,
    ) -> list[DocumentJob]:
        """Run conversion using Gemini Batch API (50% cost savings).

        Splits conversion into two batch rounds:
          1. Planning (all jobs in one batch)
          2. HTML generation (all fragments in one batch)
        """
        if not jobs:
            return []

        logger.info(
            "Running batched conversion for %d job(s).", len(jobs)
        )

        # Round 1: Batch all planning calls.
        plan_requests = []
        valid_jobs = []
        for job in jobs:
            if not job.ocr_markdown:
                logger.warning("Job %s has no OCR markdown, skipping.", job.id)
                job.status = JobStatus.FAILED
                job.error_message = "No OCR content for conversion."
                await self._db.update_job(job)
                report.errors.append(f"No OCR content for {job.id}")
                continue

            job.status = JobStatus.PLANNING
            await self._db.update_job(job)
            valid_jobs.append(job)
            plan_requests.append({
                "request_id": f"plan-{job.id}",
                "messages": self._converter.prepare_plan_messages(job),
                "max_tokens": 8192,
                "temperature": 0.3,
                "thinking": True,
            })

        if not plan_requests:
            return []

        logger.info("Batch Round 1: Submitting %d planning requests.", len(plan_requests))
        try:
            plan_results = await self._llm.batch_chat(
                plan_requests,
                display_name=f"plan-{self._campus_id or 'all'}-{len(plan_requests)}jobs",
            )
        except Exception as exc:
            logger.error("Batch planning failed: %s. Falling back to inline.", exc)
            return await self._run_stage(jobs, self._convert_one, "conversion", report)

        # Apply planning results.
        planned_jobs = []
        converted_jobs: list[DocumentJob] = []
        for job, plan_text in zip(valid_jobs, plan_results):
            if not plan_text or not plan_text.strip():
                if self._escalation_converter:
                    self._note_escalation(
                        "conversion",
                        job,
                        "batch planning returned empty result",
                    )
                    try:
                        job.error_message = ""
                        converted_jobs.append(
                            await self._convert_with(self._escalation_converter, job)
                        )
                    except Exception as exc:
                        job.status = JobStatus.FAILED
                        job.error_message = str(exc)
                        await self._db.update_job(job)
                        report.errors.append(f"conversion failed for {job.id}: {exc}")
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = "Batch planning returned empty result."
                    await self._db.update_job(job)
                    report.errors.append(f"Empty plan for {job.id}")
                continue

            await self._converter.apply_plan_result(job, plan_text)
            planned_jobs.append(job)

        if not planned_jobs:
            return []

        # Round 2: Batch all generation calls.
        gen_requests = []
        job_gen_map: dict[str, tuple[DocumentJob, Any]] = {}

        for job in planned_jobs:
            job.status = JobStatus.CONVERTING
            await self._db.update_job(job)
            structured, prompts = await self._converter.prepare_generation_prompts(job)
            job_gen_map[job.id] = (job, structured)
            gen_requests.extend(prompts)

        logger.info(
            "Batch Round 2: Submitting %d generation requests for %d jobs.",
            len(gen_requests), len(planned_jobs),
        )

        try:
            gen_results = await self._llm.batch_chat(
                gen_requests,
                display_name=f"gen-{self._campus_id or 'all'}-{len(gen_requests)}frags",
            )
        except Exception as exc:
            logger.error("Batch generation failed: %s. Falling back to inline.", exc)
            # Inline fallback for remaining planned jobs.
            fallback_results = []
            for job in planned_jobs:
                try:
                    await self._converter.generate(job)
                    fallback_results.append(job)
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    await self._db.update_job(job)
                    report.errors.append(f"conversion failed for {job.id}: {e}")
            return fallback_results

        # Map generation results back to jobs.
        result_map: dict[str, str] = {}
        for prompt, result_text in zip(gen_requests, gen_results):
            rid = prompt["request_id"]
            result_map[rid] = result_text or ""

        for job_id, (job, structured) in job_gen_map.items():
            try:
                front_key = f"gen-{job_id}-front"
                front_html = result_map.get(front_key, "")
                missing_section_output = any(
                    not result_map.get(f"gen-{job_id}-{section.page_key}", "").strip()
                    for section in structured.sections
                )
                missing_front_output = bool(
                    structured.front_matter_markdown.strip() and not front_html.strip()
                )
                if missing_section_output or missing_front_output:
                    if self._escalation_converter:
                        self._note_escalation(
                            "conversion",
                            job,
                            "batch generation returned empty fragment output",
                        )
                        job.error_message = ""
                        converted_jobs.append(
                            await self._convert_with(self._escalation_converter, job)
                        )
                        continue
                    if not front_html:
                        front_html = self._converter._generate_front_matter_fallback(
                            structured.title
                        )
                elif not front_html:
                    front_html = self._converter._generate_front_matter_fallback(
                        structured.title
                    )
                # Ensure H1
                if "<h1" not in front_html.lower():
                    from project_remedy.converter import HTMLConverter
                    front_html = (
                        f"<h1>{structured.title}</h1>\n{front_html}"
                    ).strip()

                section_bodies = {}
                for section in structured.sections:
                    sec_key = f"gen-{job_id}-{section.page_key}"
                    body = result_map.get(sec_key, "")
                    section_bodies[section.page_key] = (
                        self._converter._strip_duplicate_heading(body, section.title)
                        if body else ""
                    )

                await self._converter.assemble_from_fragments(
                    job, structured, front_html, section_bodies
                )
                converted_jobs.append(job)
                logger.info("Batch conversion complete for job %s", job.id)

            except Exception as exc:
                if self._escalation_converter:
                    self._note_escalation("conversion", job, f"batch assembly failed: {exc}")
                    try:
                        job.error_message = ""
                        converted_jobs.append(
                            await self._convert_with(self._escalation_converter, job)
                        )
                        continue
                    except Exception as esc_exc:
                        exc = esc_exc
                job.status = JobStatus.FAILED
                job.error_message = f"Batch assembly failed: {exc}"
                await self._db.update_job(job)
                report.errors.append(f"conversion failed for {job.id}: {exc}")

        return converted_jobs

    async def _run_pdf_remediation_batched(
        self,
        jobs: list[DocumentJob],
        report: PipelineReport,
    ) -> list[DocumentJob]:
        """Run the production PDF remediation path.

        The older batched legacy remediator is intentionally bypassed here.
        Production PDF-to-PDF remediation should go through fix_and_verify()
        so pipeline-tracked reruns behave the same way as the batch tool and
        CLI fix/escalate flows.
        """
        if not jobs:
            return []
        logger.info(
            "PDF remediation batch requested for %d job(s); using fix_and_verify() path instead.",
            len(jobs),
        )
        return await self._run_stage(
            jobs,
            self._pdf_remediate_one,
            "pdf_remediation",
            report,
        )

    async def _validate_one(self, job: DocumentJob) -> DocumentJob:
        """Validate a single converted document with auto-remediation."""
        job.status = JobStatus.VALIDATING
        await self._db.update_job(job)

        rendered_pages = job.get_rendered_pages()
        if not rendered_pages:
            raise ValueError(f"Job {job.id} has no rendered pages to validate.")

        outcomes = []
        updated_pages: list[RenderedPage] = []
        all_results = []
        max_cycles = 0

        for page in rendered_pages:
            html_path = self._rendered_page_staging_path(job, page)
            try:
                outcome = await self._validator.validate_rendered_page(job, page, html_path)
            except Exception as exc:
                if not self._escalation_validator:
                    raise
                self._note_escalation("validation", job, str(exc))
                retry_page = RenderedPage(
                    page_key=page.page_key,
                    kind=page.kind,
                    title=page.title,
                    relative_path=page.relative_path,
                    html=page.html,
                    source_page_range=page.source_page_range,
                    section_slug=page.section_slug,
                )
                outcome = await self._escalation_validator.validate_rendered_page(
                    job,
                    retry_page,
                    html_path,
                )
            outcomes.append(outcome)
            all_results.extend(outcome.results)
            max_cycles = max(max_cycles, outcome.remediation_count)
            updated_pages.append(
                RenderedPage(
                    page_key=page.page_key,
                    kind=page.kind,
                    title=page.title,
                    relative_path=page.relative_path,
                    html=outcome.html,
                    source_page_range=page.source_page_range,
                    section_slug=page.section_slug,
                )
            )

        job.set_rendered_pages(updated_pages)
        job.validation_results = all_results
        job.remediation_count = max_cycles

        canonical_path = next(
            (
                self._rendered_page_staging_path(job, page)
                for page in updated_pages
                if page.kind == "canonical"
            ),
            None,
        )
        if canonical_path is not None:
            job.final_html_path = str(canonical_path)

        failed_pages = [outcome for outcome in outcomes if not outcome.passed]
        if failed_pages and self._escalation_validator:
            self._note_escalation(
                "validation",
                job,
                f"{len(failed_pages)} page(s) remained flagged after tier-1 validation",
            )
            refreshed_outcomes: list[Any] = []
            for outcome in outcomes:
                if outcome.passed:
                    refreshed_outcomes.append(outcome)
                    continue
                html_path = self._rendered_page_staging_path(job, outcome.page)
                retry_page = RenderedPage(
                    page_key=outcome.page.page_key,
                    kind=outcome.page.kind,
                    title=outcome.page.title,
                    relative_path=outcome.page.relative_path,
                    html=outcome.html,
                    source_page_range=outcome.page.source_page_range,
                    section_slug=outcome.page.section_slug,
                )
                refreshed_outcomes.append(
                    await self._escalation_validator.validate_rendered_page(
                        job,
                        retry_page,
                        html_path,
                    )
                )
            outcomes = refreshed_outcomes
            all_results = []
            updated_pages = []
            max_cycles = 0
            for outcome in outcomes:
                all_results.extend(outcome.results)
                max_cycles = max(max_cycles, outcome.remediation_count)
                updated_pages.append(
                    RenderedPage(
                        page_key=outcome.page.page_key,
                        kind=outcome.page.kind,
                        title=outcome.page.title,
                        relative_path=outcome.page.relative_path,
                        html=outcome.html,
                        source_page_range=outcome.page.source_page_range,
                        section_slug=outcome.page.section_slug,
                    )
                )
            job.set_rendered_pages(updated_pages)
            job.validation_results = all_results
            job.remediation_count = max_cycles
            if canonical_path is not None:
                job.final_html_path = str(canonical_path)
            failed_pages = [outcome for outcome in outcomes if not outcome.passed]

        if failed_pages:
            job.status = JobStatus.FLAGGED
            job.error_message = " | ".join(
                outcome.error_message or f"{outcome.page.title} failed validation."
                for outcome in failed_pages
            )
        else:
            job.status = JobStatus.VALIDATED
            job.error_message = ""

        await self._db.update_job(job)

        return job

    async def _run_pdf_fix_and_verify(
        self,
        input_path: Path,
        output_path: Path,
        *,
        thorough: bool,
        vision_provider_override=None,
        original_path: Path | None = None,
        gs_was_used: bool = False,
    ) -> FixReport:
        return await asyncio.to_thread(
            fix_and_verify,
            input_path,
            output_path,
            config=self._config,
            thorough=thorough,
            vision_provider_override=vision_provider_override,
            max_cycles=3,
            conformance_repair=True,
            original_path=original_path,
            gs_was_used=gs_was_used,
        )

    async def _run_specialist_coordinator_fallback(
        self,
        *,
        pdf_path: Path,
        output_path: Path,
        original_path: Path,
        trace_path: Path,
    ):
        planner_client = None
        owns_client = False

        if self._escalation_llm is not None and hasattr(self._escalation_llm, "generate_raw"):
            planner_client = self._escalation_llm
        elif hasattr(self._llm, "generate_raw"):
            planner_client = self._llm
        elif self._config.api.llm_backend == "gemini":
            from project_remedy.gemini_client import GeminiClient

            planner_client = GeminiClient(self._config)
            await planner_client.start()
            owns_client = True

        try:
            return await run_specialist_pdf_repair(
                pdf_path=pdf_path,
                output_path=output_path,
                original_path=original_path,
                config=self._config,
                planner_client=planner_client,
                trace_path=trace_path,
            )
        finally:
            if owns_client and planner_client is not None:
                await planner_client.close()

    def _apply_pdf_fix_report(
        self,
        job: DocumentJob,
        report: FixReport,
        acceptance,
    ) -> tuple[bool, Any]:
        output_path = report.output_path
        if not output_path.exists():
            self._clear_job_acceptance_warnings(job)
            job.status = JobStatus.FAILED
            job.error_message = "Remediated PDF output missing"
            return False, acceptance
        job.remediated_document_path = str(output_path)
        job.remediated_pdf_path = str(output_path)

        # Visual diff gate: flag for manual review if visual degradation is severe.
        if report.needs_manual_review:
            job.status = JobStatus.FLAGGED
            job.error_message = report.manual_review_reason
            self._set_job_acceptance_warnings(job, acceptance.warning_entries, acceptance.summary())
            return False, acceptance

        if acceptance.passed:
            job.status = JobStatus.PDF_REMEDIATED
            job.error_message = ""
            self._set_job_acceptance_warnings(job, acceptance.warning_entries, acceptance.summary())
            return True, acceptance

        if acceptance.openable:
            # PDF opens but has quality failures (checker, SR, or veraPDF)
            job.status = JobStatus.FLAGGED
            job.error_message = "; ".join(acceptance.warning_reasons) or acceptance.summary()
            self._set_job_acceptance_warnings(job, acceptance.warning_entries, acceptance.summary())
            return False, acceptance

        job.status = JobStatus.FLAGGED
        job.error_message = "; ".join(acceptance.blocking_failure_reasons) or acceptance.summary()
        self._clear_job_acceptance_warnings(job)
        return False, acceptance

    def _apply_office_remediation_result(
        self,
        job: DocumentJob,
        result: OfficeRemediationResult,
    ) -> tuple[bool, Any | None]:
        if result.success and result.output_path and result.output_path.exists():
            acceptance = evaluate_office_acceptance(
                result.output_path,
                file_type=job.file_type,
            )
            job.remediated_document_path = str(result.output_path)
            if acceptance.openable:
                job.status = JobStatus.DOCUMENT_REMEDIATED
                job.error_message = ""
                self._set_job_acceptance_warnings(job, acceptance.warning_entries, acceptance.summary())
                return True, acceptance
            job.status = JobStatus.FLAGGED
            job.error_message = "; ".join(acceptance.blocking_failure_reasons) or acceptance.summary()
            self._clear_job_acceptance_warnings(job)
            return False, acceptance

        input_openability = validate_office_package(result.input_path, job.file_type)
        if not input_openability.passed:
            job.status = JobStatus.FLAGGED
            job.error_message = input_openability.error or result.error_message or "Office document could not be opened"
            self._clear_job_acceptance_warnings(job)
            return False, None

        if result.output_path and result.output_path.exists():
            output_openability = validate_office_package(result.output_path, job.file_type)
            if not output_openability.passed:
                job.remediated_document_path = str(result.output_path)
                job.status = JobStatus.FLAGGED
                job.error_message = output_openability.error or "Remediated document could not be opened"
                self._clear_job_acceptance_warnings(job)
                return False, None

        self._clear_job_acceptance_warnings(job)
        job.status = JobStatus.FAILED
        job.error_message = result.error_message or "Office remediation failed"
        return False, None

    async def _pdf_remediate_one(self, job: DocumentJob) -> DocumentJob:
        """Run PDF-to-PDF remediation on a single document."""
        job.status = JobStatus.PDF_REMEDIATING
        await self._db.update_job(job)

        output_dir = self._config.output.output_dir / "pdf_remediated" / job.id[:12]
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = Path(job.local_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Source PDF not found: {pdf_path}")

        output_path = output_dir / _document_filename_from_job(job)
        try:
            tier1_report = await self._run_pdf_fix_and_verify(
                pdf_path,
                output_path,
                thorough=False,
                original_path=pdf_path,
            )
        except Exception as exc:
            source_openability = validate_pdf_openability(pdf_path)
            if not source_openability.openable:
                job.status = JobStatus.FLAGGED
                job.error_message = source_openability.error or str(exc)
                self._clear_job_acceptance_warnings(job)
                await self._db.update_job(job)
                return job
            raise

        tier1_acceptance = evaluate_pdf_acceptance(
            output_path if output_path.exists() else pdf_path,
            original_path=pdf_path,
            config=self._config,
        )
        passed, tier1_acceptance = self._apply_pdf_fix_report(job, tier1_report, tier1_acceptance)

        if (not passed or tier1_acceptance.retry_reasons):
            esc_provider = create_escalation_provider(self._config)
            if esc_provider is not None:
                self._note_escalation(
                    "pdf_remediation",
                    job,
                    "; ".join(tier1_acceptance.retry_reasons or tier1_acceptance.blocking_failure_reasons) or tier1_acceptance.summary(),
                )
                try:
                    tier2_report = await self._run_pdf_fix_and_verify(
                        output_path if output_path.exists() else pdf_path,
                        output_path,
                        thorough=True,
                        vision_provider_override=esc_provider,
                        original_path=pdf_path,
                    )
                except Exception as exc:
                    output_openability = validate_pdf_openability(output_path) if output_path.exists() else validate_pdf_openability(pdf_path)
                    if not output_openability.openable:
                        job.status = JobStatus.FLAGGED
                        job.error_message = output_openability.error or str(exc)
                        self._clear_job_acceptance_warnings(job)
                        await self._db.update_job(job)
                        return job
                    raise
                final_acceptance = evaluate_pdf_acceptance(output_path, original_path=pdf_path, config=self._config)
                self._apply_pdf_fix_report(job, tier2_report, final_acceptance)

        # Experimental specialist coordinator fallback (disabled by default).
        if (
            self._config.pdf_remediation.specialist_coordinator_as_fallback
            and job.status != JobStatus.PDF_REMEDIATED
        ):
            self._note_escalation(
                "pdf_remediation",
                job,
                f"Tier 3: Specialist coordinator fallback for {job.id[:12]}",
            )
            trace_path = output_path.with_suffix(".specialist_trace.json")
            try:
                coordinator_result = await self._run_specialist_coordinator_fallback(
                    pdf_path=output_path if output_path.exists() else pdf_path,
                    output_path=output_path,
                    original_path=pdf_path,
                    trace_path=trace_path,
                )
                if coordinator_result.passed:
                    coordinator_acceptance = evaluate_pdf_acceptance(
                        output_path,
                        original_path=pdf_path,
                        config=self._config,
                    )
                    if coordinator_acceptance.passed:
                        job.status = JobStatus.PDF_REMEDIATED
                        job.error_message = ""
                        self._clear_job_acceptance_warnings(job)
                        logger.info(
                            "Specialist coordinator succeeded for %s: violations %d -> %d",
                            job.id[:12],
                            coordinator_result.violation_count_before,
                            coordinator_result.violation_count_after,
                        )
                    else:
                        self._apply_pdf_fix_report(
                            job,
                            FixReport(input_path=pdf_path, output_path=output_path),
                            coordinator_acceptance,
                        )
                else:
                    logger.info(
                        "Specialist coordinator did not resolve issues for %s: %s",
                        job.id[:12],
                        coordinator_result.acceptance_summary,
                    )
            except Exception as exc:
                logger.warning(
                    "Specialist coordinator fallback failed for %s: %s",
                    job.id[:12],
                    exc,
                )

        # Vision Planner fallback (if enabled and prior fallbacks failed)
        if (
            self._config.pdf_remediation.vision_planner_as_fallback
            and job.status != JobStatus.PDF_REMEDIATED
        ):
            from project_remedy.vision_planner.harness import VisionPlannerHarness
            from project_remedy.gemini_client import GeminiClient

            self._note_escalation(
                "pdf_remediation",
                job,
                f"Tier 3: Vision Planner fallback for {job.id[:12]}",
            )

            trace_path = output_path.with_suffix(".vision_trace.json")
            vp_pdf_output = output_path.with_suffix(".vp.pdf")
            harness = VisionPlannerHarness()

            # Use default Gemini client (or Ollama if configured)
            if self._config.api.llm_backend == "gemini":
                vp_client: Any = GeminiClient(self._config)
                await vp_client.start()
            else:
                vp_client = self._llm

            try:
                vision_result = await run_vision_plan(
                    pdf_path=output_path if output_path.exists() else pdf_path,
                    output_path=trace_path,
                    harness=harness,
                    client=vp_client,
                    config=self._config,
                    pdf_output_path=vp_pdf_output,
                )

                if vision_result.get("passed", False) and vp_pdf_output.exists():
                    # Replace output with VP-remediated version
                    import shutil
                    shutil.move(str(vp_pdf_output), str(output_path))

                    # Re-evaluate acceptance on the VP-remediated file
                    vp_acceptance = evaluate_pdf_acceptance(
                        output_path,
                        original_path=pdf_path,
                        config=self._config,
                    )
                    if vp_acceptance.passed:
                        job.status = JobStatus.PDF_REMEDIATED
                        job.error_message = ""
                        self._clear_job_acceptance_warnings(job)
                        logger.info(
                            "Vision Planner succeeded for %s: violations %d -> %d",
                            job.id[:12],
                            vision_result.get("violations_before", 0),
                            vision_result.get("violations_after", 0),
                        )
                    else:
                        self._apply_pdf_fix_report(job, FixReport(input_path=pdf_path, output_path=output_path), vp_acceptance)
                else:
                    logger.info(
                        "Vision Planner did not resolve issues for %s: %s",
                        job.id[:12],
                        vision_result.get("failure_reasons", ["unknown"]),
                    )
                    # Clean up VP output if it didn't pass
                    if vp_pdf_output.exists():
                        vp_pdf_output.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Vision Planner fallback failed for %s: %s", job.id[:12], exc)
                if vp_pdf_output.exists():
                    vp_pdf_output.unlink(missing_ok=True)

        await self._db.update_job(job)
        return job

    async def _office_remediate_one(self, job: DocumentJob) -> DocumentJob:
        """Run same-format Office remediation on a single document."""
        if job.file_type not in {FileType.DOCX, FileType.PPTX, FileType.XLSX}:
            return job

        job.status = JobStatus.DOCUMENT_REMEDIATING
        await self._db.update_job(job)

        input_path = Path(job.local_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Source document not found: {input_path}")

        output_dir = self._config.output.output_dir / "document_remediated" / job.id[:12]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / _document_filename_from_job(job)
        title = job.link_text or job.url.rsplit("/", 1)[-1]

        result = await self._office_remediator.remediate(
            input_path,
            output_path,
            title=title,
        )
        passed, acceptance = self._apply_office_remediation_result(job, result)
        if (not passed or (acceptance is not None and acceptance.retry_reasons)) and self._escalation_office_remediator:
            self._note_escalation(
                "office_remediation",
                job,
                "; ".join(acceptance.retry_reasons) if acceptance is not None and acceptance.retry_reasons else (job.error_message or "office remediation failed"),
            )
            result = await self._escalation_office_remediator.remediate(
                input_path,
                output_path,
                title=title,
            )
            self._apply_office_remediation_result(job, result)

        await self._db.update_job(job)
        return job

    async def _html_to_pdf_one(self, job: DocumentJob) -> DocumentJob:
        """Convert a job's accessible HTML to a tagged PDF via Playwright."""
        if not self._html_to_pdf:
            return job

        job.status = JobStatus.PDF_REMEDIATING
        await self._db.update_job(job)

        output_dir = self._config.output.output_dir / "pdf_remediated" / job.id[:12]
        filename = _pdf_filename_from_job(job)
        output_path = output_dir / filename

        title = job.link_text or job.url.rsplit("/", 1)[-1]
        result = await self._html_to_pdf.convert(
            job.generated_html,
            output_path,
            title=title,
            language="en",
        )

        if result.success and result.output_path:
            job.remediated_pdf_path = str(result.output_path)
            job.status = JobStatus.PDF_REMEDIATED
        else:
            job.error_message = result.error_message or "HTML-to-PDF conversion failed"
            logger.warning(
                "HTML-to-PDF failed for %s: %s",
                job.id, result.error_message,
            )

        await self._db.update_job(job)
        return job

    async def _apply_strategy_fixes(self, job: DocumentJob) -> None:
        """Apply programmatic HTML fixes before LLM remediation."""
        rendered_pages = job.get_rendered_pages()
        if not rendered_pages:
            return

        updated = False
        new_pages: list[RenderedPage] = []
        for page in rendered_pages:
            fixed_html, fixes = self._html_strategy.remediate(page.html)
            if fixes:
                logger.info(
                    "Strategy fixes for %s/%s: %d fixes applied",
                    job.id[:8], page.page_key, len(fixes),
                )
                updated = True
            new_pages.append(
                RenderedPage(
                    page_key=page.page_key,
                    kind=page.kind,
                    title=page.title,
                    relative_path=page.relative_path,
                    html=fixed_html,
                    source_page_range=page.source_page_range,
                    section_slug=page.section_slug,
                )
            )

        if updated:
            job.set_rendered_pages(new_pages)
            await self._db.update_job(job)

    async def _apply_accessibility_strategies(self, job: DocumentJob) -> None:
        """Apply ASU CIC-style accessibility remediation strategies."""
        rendered_pages = job.get_rendered_pages()
        if not rendered_pages:
            return

        # Determine image directory for this job (for vision-based alt text).
        image_dir = self._config.output.output_dir / "images" / job.id[:12]

        updated = False
        new_pages: list[RenderedPage] = []
        for page in rendered_pages:
            try:
                fixed_html = await self._accessibility_strategies.run(
                    page.html,
                    self._llm,
                    image_dir=str(image_dir) if image_dir.exists() else None,
                    title=job.link_text or "",
                )
                if not isinstance(fixed_html, str):
                    fixed_html = str(fixed_html)
                if fixed_html != page.html:
                    updated = True
                new_pages.append(
                    RenderedPage(
                        page_key=page.page_key,
                        kind=page.kind,
                        title=page.title,
                        relative_path=page.relative_path,
                        html=fixed_html,
                        source_page_range=page.source_page_range,
                        section_slug=page.section_slug,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Accessibility strategies failed for %s/%s: %s",
                    job.id[:8], page.page_key, exc,
                )
                new_pages.append(page)

        if updated:
            job.set_rendered_pages(new_pages)
            await self._db.update_job(job)

    async def _vision_process_job(self, job: DocumentJob) -> None:
        """Process image placeholders in a converted document's HTML.

        Scans the generated HTML for image placeholder patterns and
        replaces them with vision-generated alt text or HTML.
        """
        rendered_pages = job.get_rendered_pages()
        if not rendered_pages:
            return

        updated = False
        new_pages: list[RenderedPage] = []
        for page in rendered_pages:
            html = await self._process_html_with_vision(job, page.html)
            updated = updated or html != page.html
            new_pages.append(
                RenderedPage(
                    page_key=page.page_key,
                    kind=page.kind,
                    title=page.title,
                    relative_path=page.relative_path,
                    html=html,
                    source_page_range=page.source_page_range,
                    section_slug=page.section_slug,
                )
            )

        if updated:
            job.set_rendered_pages(new_pages)
            await self._db.update_job(job)

        # Copy extracted images to the HTML staging directory so they're
        # co-located for file:// validation and preview.
        images_src = self._config.output.output_dir / "images" / job.id[:12]
        if images_src.is_dir():
            html_dest = self._config.output.output_dir / "html" / job.id[:12]
            html_dest.mkdir(parents=True, exist_ok=True)
            for img_file in images_src.iterdir():
                if img_file.is_file():
                    dest_file = html_dest / img_file.name
                    if not dest_file.exists():
                        shutil.copy2(img_file, dest_file)

    async def _process_html_with_vision(self, job: DocumentJob, html: str) -> str:
        """Apply vision-based replacements to a single HTML artifact."""
        config = self._config
        extracted_images_dir = config.output.output_dir / "images" / job.id[:12]
        download_dir = config.output.output_dir / "downloads"

        # Find image references that need vision processing.
        matches = list(_IMAGE_PLACEHOLDER_RE.finditer(html))
        if not matches:
            return html

        logger.info(
            "Vision: processing %d image placeholder(s) for job %s",
            len(matches),
            job.id,
        )

        for match in matches:
            # Try to determine the image path from the match groups.
            # Groups: (md_alt, md_path, html_alt, html_src)
            img_filename = match.group(2) or match.group(4)
            if not img_filename:
                continue

            # Search path: extracted images -> downloads -> job parent dir.
            img_path = extracted_images_dir / img_filename
            if not img_path.exists():
                img_path = download_dir / img_filename
            if not img_path.exists() and job.local_path:
                img_path = Path(job.local_path).parent / img_filename
            if not img_path.exists():
                logger.debug(
                    "Image file not found: %s (skipping vision)", img_filename
                )
                continue

            context = job.link_text or job.link_context or ""
            try:
                result = await self._vision.process_image(img_path, context)

                if result.html_replacement:
                    html = html.replace(match.group(0), result.html_replacement)
                elif result.alt_text:
                    # Replace just the alt text in the existing tag.
                    old_fragment = match.group(0)
                    if result.is_decorative:
                        new_fragment = (
                            f'<img src="{img_filename}" alt="" '
                            f'role="presentation">'
                        )
                    else:
                        new_fragment = (
                            f'<img src="{img_filename}" '
                            f'alt="{result.alt_text}">'
                        )
                    html = html.replace(old_fragment, new_fragment)
            except Exception as exc:
                logger.warning(
                    "Vision processing failed for image %s in job %s: %s",
                    img_filename,
                    job.id,
                    exc,
                )

        return html

    # ------------------------------------------------------------------
    # Internal: concurrent stage runner
    # ------------------------------------------------------------------

    async def _run_stage(
        self,
        jobs: list[DocumentJob],
        handler,
        stage_name: str,
        report: PipelineReport,
    ) -> list[DocumentJob]:
        """Run *handler* concurrently on all *jobs* with semaphore control.

        Individual failures are caught and logged; they do not stop the
        pipeline for other documents.

        Returns
        -------
        list[DocumentJob]
            Successfully processed jobs.
        """
        if not jobs:
            logger.info("No jobs for %s stage.", stage_name)
            return []

        logger.info(
            "Running %s for %d job(s) (concurrency=%d).",
            stage_name,
            len(jobs),
            self._config.processing.max_concurrent_calls,
        )

        async def _guarded(job: DocumentJob) -> DocumentJob | None:
            async with self._semaphore:
                try:
                    return await handler(job)
                except Exception as exc:
                    msg = f"{stage_name} failed for job {job.id}: {exc}"
                    logger.error(msg)
                    report.errors.append(msg)
                    # Mark the job as failed if it is not already.
                    if job.status != JobStatus.FAILED:
                        job.status = JobStatus.FAILED
                        job.error_message = str(exc)
                        await self._db.update_job(job)
                    return None

        results = await asyncio.gather(
            *(_guarded(job) for job in jobs), return_exceptions=False
        )

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _rendered_page_staging_path(
        self,
        job: DocumentJob,
        page: RenderedPage,
    ) -> Path:
        """Determine a local staging path for validating a rendered page."""
        staging_root = self._config.output.output_dir / "html" / job.id[:12]
        staging_root.mkdir(parents=True, exist_ok=True)

        if page.kind == "canonical":
            return staging_root / "canonical.html"

        safe_slug = page.section_slug or page.page_key
        safe_slug = re.sub(r"[^\w-]+", "-", safe_slug).strip("-") or page.page_key
        section_dir = staging_root / "sections"
        section_dir.mkdir(parents=True, exist_ok=True)
        return section_dir / f"{safe_slug}.html"

    def _compute_avg_lighthouse(self, jobs: list[DocumentJob]) -> float:
        """Compute average Lighthouse accessibility score across all jobs."""
        scores: list[float] = []
        for job in jobs:
            for r in job.validation_results:
                if r.tool == "lighthouse" and r.score is not None:
                    scores.append(r.score)
        return sum(scores) / len(scores) if scores else 0.0

    def _estimate_api_cost(self) -> float:
        """Rough cost estimate based on cumulative token usage.

        Pricing varies by backend:
        - Ollama Cloud: ~$0.01/1K input, ~$0.03/1K output (conservative)
        - Gemini: model-aware pricing via GeminiClient usage buckets
        """
        if self._config.api.llm_backend == "gemini":
            total = 0.0
            if hasattr(self._llm, "estimate_cost"):
                total += self._llm.estimate_cost()
            if self._escalation_llm and hasattr(self._escalation_llm, "estimate_cost"):
                total += self._escalation_llm.estimate_cost()
            return total

        inp = self._llm.total_input_tokens
        out = self._llm.total_output_tokens
        input_cost = (inp / 1000) * 0.01
        output_cost = (out / 1000) * 0.03
        return input_cost + output_cost

    def _clear_job_acceptance_warnings(self, job: DocumentJob) -> None:
        job.warning_message = ""
        job.set_acceptance_warnings([])

    def _set_job_acceptance_warnings(
        self,
        job: DocumentJob,
        warnings: list[dict[str, Any]],
        summary: str,
    ) -> None:
        job.set_acceptance_warnings(warnings)
        if warnings:
            job.warning_message = summary
        else:
            job.warning_message = ""

    def _compute_warning_breakdown(self, jobs: list[DocumentJob]) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for job in jobs:
            for entry in job.get_acceptance_warnings():
                source = str(entry.get("source", "unknown") or "unknown")
                breakdown[source] = breakdown.get(source, 0) + 1
        return dict(sorted(breakdown.items()))
