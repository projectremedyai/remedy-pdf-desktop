"""Core data models for the LAMC ADA Document Remediation Pipeline."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


class JobStatus(enum.Enum):
    """Lifecycle status of a document remediation job."""

    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    PLANNING = "planning"
    PLANNED = "planned"
    CONVERTING = "converting"
    CONVERTED = "converted"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DOCUMENT_REMEDIATING = "document_remediating"
    DOCUMENT_REMEDIATED = "document_remediated"
    PDF_REMEDIATING = "pdf_remediating"
    PDF_REMEDIATED = "pdf_remediated"
    FAILED = "failed"
    FLAGGED = "flagged"
    EXCLUDED = "excluded"


class FileType(enum.Enum):
    """Supported document file types."""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    PPTX = "pptx"
    PPT = "ppt"
    XLSX = "xlsx"
    XLS = "xls"


@dataclass
class ExtractedImage:
    """Metadata for an image extracted from a PDF."""

    filename: str       # e.g. "img_p1_x42.png"
    page_number: int    # 1-based
    xref: int           # PyMuPDF xref for dedup
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "page_number": self.page_number,
            "xref": self.xref,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractedImage:
        return cls(
            filename=data["filename"],
            page_number=data["page_number"],
            xref=data["xref"],
            width=data["width"],
            height=data["height"],
        )


@dataclass
class ValidationResult:
    """Result from a single accessibility validation tool."""

    tool: str  # axe, pa11y, or lighthouse
    score: float | None = None  # Lighthouse accessibility score (0-100)
    violations: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False
    page_key: str = ""
    page_title: str = ""
    page_path: str = ""


@dataclass
class RenderedPage:
    """A deployable HTML artifact derived from a document."""

    page_key: str
    kind: str
    title: str
    relative_path: str
    html: str
    source_page_range: str = ""
    section_slug: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the rendered page for storage."""
        return {
            "page_key": self.page_key,
            "kind": self.kind,
            "title": self.title,
            "relative_path": self.relative_path,
            "html": self.html,
            "source_page_range": self.source_page_range,
            "section_slug": self.section_slug,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderedPage:
        """Deserialize a rendered page from persisted JSON."""
        return cls(
            page_key=data.get("page_key", ""),
            kind=data.get("kind", "canonical"),
            title=data.get("title", ""),
            relative_path=data.get("relative_path", ""),
            html=data.get("html", ""),
            source_page_range=data.get("source_page_range", ""),
            section_slug=data.get("section_slug", ""),
        )


@dataclass
class DocumentJob:
    """Represents a single document remediation job through the pipeline."""

    id: str = field(default_factory=lambda: uuid4().hex)
    campus_id: str = ""
    url: str = ""
    source_page_url: str = ""
    link_text: str = ""
    link_context: str = ""
    file_type: FileType | None = None
    local_path: str = ""
    file_hash: str = ""
    file_size: int = 0
    status: JobStatus = JobStatus.DISCOVERED
    ocr_markdown: str = ""
    html_plan: str = ""
    generated_html: str = ""
    generated_pages_json: str = "[]"
    final_html_path: str = ""
    remediated_document_path: str = ""
    remediated_pdf_path: str = ""
    validation_results: list[ValidationResult] = field(default_factory=list)
    extracted_images_json: str = "[]"
    remediation_count: int = 0
    error_message: str = ""
    warning_message: str = ""
    acceptance_warnings_json: str = "[]"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the job to a dictionary for database storage."""
        return {
            "id": self.id,
            "campus_id": self.campus_id,
            "url": self.url,
            "source_page_url": self.source_page_url,
            "link_text": self.link_text,
            "link_context": self.link_context,
            "file_type": self.file_type.value if self.file_type else None,
            "local_path": self.local_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "status": self.status.value,
            "ocr_markdown": self.ocr_markdown,
            "html_plan": self.html_plan,
            "generated_html": self.generated_html,
            "generated_pages_json": self.generated_pages_json,
            "extracted_images_json": self.extracted_images_json,
            "final_html_path": self.final_html_path,
            "remediated_document_path": self.remediated_document_path,
            "remediated_pdf_path": self.remediated_pdf_path,
            "validation_results": _serialize_validation_results(
                self.validation_results
            ),
            "remediation_count": self.remediation_count,
            "error_message": self.error_message,
            "warning_message": self.warning_message,
            "acceptance_warnings_json": self.acceptance_warnings_json,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentJob:
        """Deserialize a job from a database row dictionary."""
        import json

        validation_raw = data.get("validation_results", "[]")
        if isinstance(validation_raw, str):
            validation_raw = json.loads(validation_raw)

        validation_results = [
            ValidationResult(
                tool=v["tool"],
                score=v.get("score"),
                violations=v.get("violations", []),
                passed=v.get("passed", False),
                page_key=v.get("page_key", ""),
                page_title=v.get("page_title", ""),
                page_path=v.get("page_path", ""),
            )
            for v in validation_raw
        ]

        file_type_raw = data.get("file_type")
        file_type = FileType(file_type_raw) if file_type_raw else None

        return cls(
            id=data["id"],
            campus_id=data.get("campus_id", ""),
            url=data.get("url", ""),
            source_page_url=data.get("source_page_url", ""),
            link_text=data.get("link_text", ""),
            link_context=data.get("link_context", ""),
            file_type=file_type,
            local_path=data.get("local_path", ""),
            file_hash=data.get("file_hash", ""),
            file_size=data.get("file_size", 0),
            status=JobStatus(data.get("status", "discovered")),
            ocr_markdown=data.get("ocr_markdown", ""),
            html_plan=data.get("html_plan", ""),
            generated_html=data.get("generated_html", ""),
            generated_pages_json=data.get("generated_pages_json", "[]"),
            extracted_images_json=data.get("extracted_images_json", "[]"),
            final_html_path=data.get("final_html_path", ""),
            remediated_document_path=data.get("remediated_document_path", ""),
            remediated_pdf_path=data.get("remediated_pdf_path", ""),
            validation_results=validation_results,
            remediation_count=data.get("remediation_count", 0),
            error_message=data.get("error_message", ""),
            warning_message=data.get("warning_message", ""),
            acceptance_warnings_json=data.get("acceptance_warnings_json", "[]"),
            created_at=_parse_datetime(data.get("created_at", "")),
            updated_at=_parse_datetime(data.get("updated_at", "")),
        )

    def get_rendered_pages(self) -> list[RenderedPage]:
        """Return persisted rendered pages, with a canonical fallback."""
        try:
            raw_pages = json.loads(self.generated_pages_json or "[]")
        except json.JSONDecodeError:
            raw_pages = []

        pages = [
            RenderedPage.from_dict(page)
            for page in raw_pages
            if isinstance(page, dict)
        ]
        if pages:
            return pages

        if self.generated_html:
            return [
                RenderedPage(
                    page_key="canonical",
                    kind="canonical",
                    title=self.link_text.strip() or "Document",
                    relative_path="",
                    html=self.generated_html,
                )
            ]

        return []

    def set_rendered_pages(self, pages: list[RenderedPage]) -> None:
        """Persist rendered pages to JSON and sync the canonical payload."""
        self.generated_pages_json = json.dumps(
            [page.to_dict() for page in pages],
            ensure_ascii=False,
        )

        canonical = next(
            (page for page in pages if page.kind == "canonical"),
            None,
        )
        if canonical:
            self.generated_html = canonical.html

    def get_extracted_images(self) -> list[ExtractedImage]:
        """Return deserialized extracted images."""
        try:
            raw = json.loads(self.extracted_images_json or "[]")
        except json.JSONDecodeError:
            return []
        return [ExtractedImage.from_dict(item) for item in raw if isinstance(item, dict)]

    def set_extracted_images(self, images: list[ExtractedImage]) -> None:
        """Serialize extracted images to JSON."""
        self.extracted_images_json = json.dumps(
            [img.to_dict() for img in images],
            ensure_ascii=False,
        )

    def get_acceptance_warnings(self) -> list[dict[str, Any]]:
        """Return structured non-blocking acceptance warnings."""
        try:
            raw = json.loads(self.acceptance_warnings_json or "[]")
        except json.JSONDecodeError:
            return []
        return [item for item in raw if isinstance(item, dict)]

    def set_acceptance_warnings(self, warnings: list[dict[str, Any]]) -> None:
        """Persist structured non-blocking acceptance warnings."""
        self.acceptance_warnings_json = json.dumps(warnings, ensure_ascii=False)


@dataclass
class DeploymentRecord:
    """Record of a single document deployment to a Drupal site."""

    job_id: str = ""
    campus_code: str = ""
    drupal_uuid: str = ""
    drupal_nid: int = 0
    source_url: str = ""
    content_type: str = "page"
    status: str = "created"  # "created" | "updated" | "failed"
    error_message: str = ""
    deployed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DeploymentSummary:
    """Aggregate deployment statistics for a campus."""

    campus_code: str = ""
    total_deployed: int = 0
    created: int = 0
    updated: int = 0
    failed: int = 0
    last_deployed_at: datetime | None = None


def _serialize_validation_results(results: list[ValidationResult]) -> str:
    """Serialize validation results to a JSON string for storage."""
    import json

    return json.dumps(
        [
            {
                "tool": r.tool,
                "score": r.score,
                "violations": r.violations,
                "passed": r.passed,
                "page_key": r.page_key,
                "page_title": r.page_title,
                "page_path": r.page_path,
            }
            for r in results
        ]
    )


def _parse_datetime(value: str | datetime) -> datetime:
    """Parse an ISO datetime string or return the datetime as-is."""
    if isinstance(value, datetime):
        return value
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value)
