"""Generates LAMC-style export folders for each campus.

Creates the full export directory structure with downloaded documents,
preview HTML files, and CSV manifests matching the LAMC reference format.
"""

from __future__ import annotations

import csv
import logging
import re
import shutil
from pathlib import Path

from project_remedy.config import CampusConfig, PipelineConfig
from project_remedy.database import DatabaseManager
from project_remedy.models import DocumentJob, FileType, JobStatus

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:80] or "untitled"


class ExportsBuilder:
    """Builds LAMC-style export folders for any campus.

    Parameters
    ----------
    config:
        Pipeline configuration.
    db:
        Connected database manager.
    exports_root:
        Root directory for exports (e.g. ``exports/``).
    """

    def __init__(
        self,
        config: PipelineConfig,
        db: DatabaseManager,
        exports_root: Path,
    ) -> None:
        self._config = config
        self._db = db
        self._exports_root = exports_root.resolve()

    async def build_campus_export(self, campus: CampusConfig) -> Path:
        """Build the full export folder for one campus.

        Returns the path to the campus export directory.
        """
        code = campus.code
        campus_dir = self._exports_root / code
        campus_dir.mkdir(parents=True, exist_ok=True)

        # Fetch all jobs for this campus
        all_jobs = await self._db.get_all_jobs(campus_id=code)
        if not all_jobs:
            logger.warning("No jobs found for campus %s.", code)
            return campus_dir

        # Separate excluded jobs (schedules, catalogs) from active jobs
        active_jobs = [j for j in all_jobs if j.status != JobStatus.EXCLUDED]
        excluded_jobs = [j for j in all_jobs if j.status == JobStatus.EXCLUDED]

        logger.info(
            "Building export for %s: %d jobs total (%d active, %d excluded).",
            code, len(all_jobs), len(active_jobs), len(excluded_jobs),
        )

        # 1. Copy active downloads into exports/{CODE}/downloads/{ext}/
        self._copy_downloads(active_jobs, campus_dir)

        # 1b. Copy excluded downloads into exports/{CODE}/excluded/{ext}/
        if excluded_jobs:
            excluded_dir = campus_dir / "excluded"
            self._copy_downloads(excluded_jobs, excluded_dir)

        html_jobs: list[tuple[DocumentJob, str]] = []
        if self._config.processing.html_workflow_enabled:
            # 2. Generate preview HTML files (active jobs only)
            html_jobs = self._generate_preview_html(active_jobs, campus_dir)

        # 2b. Copy remediated documents into export folders.
        self._copy_remediated_documents(active_jobs, campus_dir)

        # 2c. Copy tagged PDFs into exports/{CODE}/tagged-pdfs/
        self._copy_tagged_pdfs(active_jobs, campus_dir)

        # 3. Write existing-document-links.csv (all jobs, status in notes)
        self._write_existing_document_links(all_jobs, campus_dir, code)

        # 4. Write html-replacement-links.csv (active jobs only)
        if self._config.processing.html_workflow_enabled:
            self._write_html_replacement_links(html_jobs, campus_dir, code, campus)

        # 5. Generate accessible 404 error page
        self._generate_404_page(campus_dir, campus)

        logger.info("Export complete for %s -> %s", code, campus_dir)
        return campus_dir

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------

    def _copy_downloads(
        self, jobs: list[DocumentJob], campus_dir: Path
    ) -> None:
        """Copy downloaded files into exports/{CODE}/downloads/{ext}/."""
        downloads_dir = campus_dir / "downloads"

        # Create subdirectories for each file type
        for ext in ("doc", "docx", "pdf", "pptx", "xls", "xlsx"):
            (downloads_dir / ext).mkdir(parents=True, exist_ok=True)

        copied = 0
        for job in jobs:
            if not job.local_path:
                continue
            src = Path(job.local_path)
            if not src.exists():
                continue

            ext = src.suffix.lstrip(".").lower()
            if ext not in ("doc", "docx", "pdf", "pptx", "xls", "xlsx"):
                continue

            dest_dir = downloads_dir / ext
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            copied += 1

        logger.info("Copied %d downloads for %s.", copied, campus_dir.name)

    # ------------------------------------------------------------------
    # Remediated documents
    # ------------------------------------------------------------------

    def _copy_remediated_documents(
        self, jobs: list[DocumentJob], campus_dir: Path
    ) -> None:
        """Copy remediated same-format documents into exports/{CODE}/remediated-documents/{ext}/."""
        root = campus_dir / "remediated-documents"
        copied = 0
        for job in jobs:
            if not job.remediated_document_path:
                continue
            src = Path(job.remediated_document_path)
            if not src.exists():
                continue
            ext = src.suffix.lstrip(".").lower() or "other"
            dest_dir = root / ext
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            copied += 1
        if copied:
            logger.info(
                "Copied %d remediated documents for %s.",
                copied,
                campus_dir.name,
            )

    # ------------------------------------------------------------------
    # Tagged PDFs
    # ------------------------------------------------------------------

    def _copy_tagged_pdfs(
        self, jobs: list[DocumentJob], campus_dir: Path
    ) -> None:
        """Copy Playwright-generated tagged PDFs into exports/{CODE}/tagged-pdfs/."""
        pdf_dir = campus_dir / "tagged-pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for job in jobs:
            if not job.remediated_pdf_path:
                continue
            src = Path(job.remediated_pdf_path)
            if not src.exists():
                continue

            dest = pdf_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            copied += 1

        if copied:
            logger.info("Copied %d tagged PDFs for %s.", copied, campus_dir.name)

    # ------------------------------------------------------------------
    # Preview HTML
    # ------------------------------------------------------------------

    def _generate_preview_html(
        self, jobs: list[DocumentJob], campus_dir: Path
    ) -> list[tuple[DocumentJob, str]]:
        """Generate preview HTML files and return (job, relative_path) pairs.

        Naming: {seq:04d}-{slugify(title)}-{file_hash[:8]}.html
        """
        preview_dir = campus_dir / "preview-html"
        preview_dir.mkdir(parents=True, exist_ok=True)

        # Filter to jobs that have generated HTML
        html_jobs = [
            j for j in jobs
            if j.generated_html and j.status in (
                JobStatus.CONVERTED, JobStatus.VALIDATED, JobStatus.FLAGGED,
                JobStatus.PDF_REMEDIATED,
            )
        ]

        # Sort by created_at for sequential numbering
        html_jobs.sort(key=lambda j: j.created_at)

        result: list[tuple[DocumentJob, str]] = []

        for seq, job in enumerate(html_jobs, start=1):
            title = job.link_text or self._title_from_url(job.url)
            slug = _slugify(title)
            hash_prefix = job.file_hash[:8] if job.file_hash else "00000000"
            filename = f"{seq:04d}-{slug}-{hash_prefix}.html"
            filepath = preview_dir / filename

            html_content = job.generated_html

            # Copy extracted images alongside the preview HTML.
            extracted_images = job.get_extracted_images()
            if extracted_images:
                images_subdir = f"images/{hash_prefix}"
                images_dest = preview_dir / "images" / hash_prefix
                images_dest.mkdir(parents=True, exist_ok=True)

                images_src = (
                    self._config.output.output_dir / "images" / job.id[:12]
                )
                for img in extracted_images:
                    src_file = images_src / img.filename
                    if src_file.exists():
                        dest_file = images_dest / img.filename
                        if not dest_file.exists():
                            shutil.copy2(src_file, dest_file)
                        # Rewrite src references in the HTML to the images subdir.
                        html_content = html_content.replace(
                            f'src="{img.filename}"',
                            f'src="{images_subdir}/{img.filename}"',
                        )
                        html_content = html_content.replace(
                            f"]({img.filename})",
                            f"]({images_subdir}/{img.filename})",
                        )

            filepath.write_text(html_content, encoding="utf-8")
            relative_path = f"preview-html/{filename}"
            result.append((job, relative_path))

        logger.info(
            "Generated %d preview HTML files for %s.",
            len(result),
            campus_dir.name,
        )
        return result

    # ------------------------------------------------------------------
    # CSVs
    # ------------------------------------------------------------------

    def _write_existing_document_links(
        self,
        jobs: list[DocumentJob],
        campus_dir: Path,
        code: str,
    ) -> None:
        """Write existing-document-links.csv with ALL jobs."""
        csv_path = campus_dir / "existing-document-links.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "campus",
                "document_title",
                "source_page_url",
                "original_document_url",
                "notes",
            ])
            for job in jobs:
                title = job.link_text or self._title_from_url(job.url)
                writer.writerow([
                    code,
                    title,
                    job.source_page_url,
                    job.url,
                    f"Pipeline status: {job.status.value}",
                ])

        logger.info(
            "Wrote %d rows to %s.", len(jobs), csv_path.name
        )

    def _write_html_replacement_links(
        self,
        html_jobs: list[tuple[DocumentJob, str]],
        campus_dir: Path,
        code: str,
        campus: CampusConfig,
    ) -> None:
        """Write html-replacement-links.csv for jobs with HTML output."""
        csv_path = campus_dir / "html-replacement-links.csv"

        # Derive base URL from campus start_url
        base_url = campus.start_url.rstrip("/")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "campus",
                "document_title",
                "original_document_url",
                "replacement_html_relative_path",
                "replacement_html_url",
            ])
            for job, relative_path in html_jobs:
                title = job.link_text or self._title_from_url(job.url)
                replacement_url = f"{base_url}/{relative_path}"
                writer.writerow([
                    code,
                    title,
                    job.url,
                    relative_path,
                    replacement_url,
                ])

        logger.info(
            "Wrote %d rows to %s.", len(html_jobs), csv_path.name
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_404_page(campus_dir: Path, campus) -> None:
        """Generate an accessible 404 error page for the campus."""
        primary = getattr(campus, "brand_primary", "#004590")
        name = getattr(campus, "name", "LACCD Campus")
        start_url = getattr(campus, "start_url", "")

        html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex">
  <title>Page Not Found — {name}</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='{primary}'/><text x='16' y='23' font-size='20' text-anchor='middle' fill='white' font-family='system-ui'>A</text></svg>">
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; color: #1A1A2E; background: #fff; }}
    header {{ background: {primary}; color: #fff; padding: 1rem 2rem; }}
    main {{ max-width: 72ch; margin: 2rem auto; padding: 0 1.5rem; }}
    a {{ color: {primary}; text-decoration: underline; }}
    a:focus {{ outline: 3px solid {primary}; outline-offset: 2px; }}
    footer {{ background: #F5F7FA; border-top: 3px solid {primary}; padding: 1.5rem 2rem; margin-top: 3rem; font-size: 0.9rem; color: #5A6670; }}
  </style>
</head>
<body>
  <a href="#main-content" class="skip-nav" style="position:absolute;top:-100%;left:0;background:{primary};color:#fff;padding:.75rem 1.5rem;z-index:10000;font-weight:bold;text-decoration:none;">Skip to main content</a>
  <header>
    <strong>{name}</strong> | Los Angeles Community College District
  </header>
  <main id="main-content">
    <h1>Page Not Found</h1>
    <p>The document you requested could not be found. It may have been moved, renamed, or removed.</p>
    <h2>What You Can Do</h2>
    <ul>
      <li><a href="{start_url}">Return to the {name} homepage</a></li>
      <li>Check the URL for typos</li>
      <li>Use the search feature on the campus website to find the document</li>
      <li><a href="mailto:accessibility@laccd.edu">Contact us</a> if you need help finding a specific document</li>
    </ul>
  </main>
  <footer>
    <p>{name} &mdash; Los Angeles Community College District</p>
  </footer>
</body>
</html>"""

        error_path = campus_dir / "404.html"
        error_path.write_text(html, encoding="utf-8")
        logger.info("Generated 404 page: %s", error_path)

    @staticmethod
    def _title_from_url(url: str) -> str:
        """Extract a human-readable title from a document URL."""
        from urllib.parse import unquote

        path = unquote(url.rsplit("/", 1)[-1])
        # Strip extension
        name = path.rsplit(".", 1)[0] if "." in path else path
        # Replace separators with spaces
        name = re.sub(r"[-_]+", " ", name)
        return name.strip() or "Untitled"
