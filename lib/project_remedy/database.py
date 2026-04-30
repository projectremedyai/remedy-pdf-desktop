"""Async SQLite database manager for pipeline state persistence.

Provides CRUD operations for document jobs, crawl state tracking, and
validation logging.  All public methods are async and safe to call from
the main asyncio event loop.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import aiosqlite

from project_remedy.models import DeploymentRecord, DeploymentSummary, DocumentJob, JobStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS document_jobs (
    id              TEXT PRIMARY KEY,
    campus_id       TEXT NOT NULL DEFAULT '',
    url             TEXT NOT NULL DEFAULT '',
    source_page_url TEXT NOT NULL DEFAULT '',
    link_text       TEXT NOT NULL DEFAULT '',
    link_context    TEXT NOT NULL DEFAULT '',
    file_type       TEXT,
    local_path      TEXT NOT NULL DEFAULT '',
    file_hash       TEXT NOT NULL DEFAULT '',
    file_size       INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'discovered',
    ocr_markdown    TEXT NOT NULL DEFAULT '',
    html_plan       TEXT NOT NULL DEFAULT '',
    generated_html  TEXT NOT NULL DEFAULT '',
    generated_pages_json TEXT NOT NULL DEFAULT '[]',
    final_html_path TEXT NOT NULL DEFAULT '',
    remediated_document_path TEXT NOT NULL DEFAULT '',
    remediated_pdf_path TEXT NOT NULL DEFAULT '',
    validation_results TEXT NOT NULL DEFAULT '[]',
    remediation_count  INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT NOT NULL DEFAULT '',
    warning_message TEXT NOT NULL DEFAULT '',
    acceptance_warnings_json TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON document_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_campus ON document_jobs(campus_id);
CREATE INDEX IF NOT EXISTS idx_jobs_file_type ON document_jobs(file_type);
CREATE INDEX IF NOT EXISTS idx_jobs_url ON document_jobs(url);

CREATE TABLE IF NOT EXISTS crawl_state (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT NOT NULL UNIQUE,
    depth       INTEGER NOT NULL DEFAULT 0,
    visited     INTEGER NOT NULL DEFAULT 0,
    discovered_at TEXT NOT NULL,
    visited_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_crawl_url ON crawl_state(url);

CREATE TABLE IF NOT EXISTS validation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES document_jobs(id),
    cycle       INTEGER NOT NULL DEFAULT 1,
    page_key    TEXT NOT NULL DEFAULT '',
    page_title  TEXT NOT NULL DEFAULT '',
    page_path   TEXT NOT NULL DEFAULT '',
    tool        TEXT NOT NULL,
    score       REAL,
    violations  TEXT NOT NULL DEFAULT '[]',
    passed      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vlog_job ON validation_log(job_id);

CREATE TABLE IF NOT EXISTS deployment_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT NOT NULL REFERENCES document_jobs(id),
    campus_code   TEXT NOT NULL,
    drupal_uuid   TEXT NOT NULL DEFAULT '',
    drupal_nid    INTEGER NOT NULL DEFAULT 0,
    source_url    TEXT NOT NULL DEFAULT '',
    content_type  TEXT NOT NULL DEFAULT 'page',
    status        TEXT NOT NULL DEFAULT 'created',
    error_message TEXT NOT NULL DEFAULT '',
    deployed_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_deploy_job ON deployment_log(job_id);
CREATE INDEX IF NOT EXISTS idx_deploy_campus ON deployment_log(campus_code);
"""


class DatabaseManager:
    """Async wrapper around an aiosqlite connection with domain helpers."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the database and ensure the schema exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=30000")
        await self._conn.executescript(_SCHEMA_SQL)
        await self._ensure_schema_migrations()
        await self._conn.commit()
        logger.info("Database connected at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed.")

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    # ------------------------------------------------------------------
    # DocumentJob CRUD
    # ------------------------------------------------------------------

    async def create_job(self, job: DocumentJob) -> None:
        """Insert a new document job into the database."""
        data = job.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        sql = f"INSERT OR IGNORE INTO document_jobs ({columns}) VALUES ({placeholders})"
        await self.conn.execute(sql, data)
        await self.conn.commit()

    async def get_job(self, job_id: str) -> DocumentJob | None:
        """Fetch a single job by its ID."""
        cursor = await self.conn.execute(
            "SELECT * FROM document_jobs WHERE id = ?", (job_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return DocumentJob.from_dict(dict(row))

    async def update_job(self, job: DocumentJob) -> None:
        """Persist all mutable fields of *job* back to the database."""
        job.updated_at = datetime.now(timezone.utc)
        data = job.to_dict()
        set_clause = ", ".join(f"{k} = :{k}" for k in data if k != "id")
        sql = f"UPDATE document_jobs SET {set_clause} WHERE id = :id"
        await self.conn.execute(sql, data)
        await self.conn.commit()

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_message: str = "",
    ) -> None:
        """Quick-update only the status (and optionally error) of a job."""
        now = datetime.now(timezone.utc).isoformat()
        await self.conn.execute(
            "UPDATE document_jobs SET status = ?, error_message = ?, updated_at = ? "
            "WHERE id = ?",
            (status.value, error_message, now, job_id),
        )
        await self.conn.commit()

    async def get_jobs_by_status(
        self,
        *statuses: JobStatus,
        campus_id: str | None = None,
    ) -> list[DocumentJob]:
        """Return all jobs whose status matches any of the given statuses.

        If *campus_id* is provided, results are further filtered to that
        campus.
        """
        placeholders = ", ".join("?" for _ in statuses)
        sql = f"SELECT * FROM document_jobs WHERE status IN ({placeholders})"
        params: list[str] = [s.value for s in statuses]
        if campus_id is not None:
            sql += " AND campus_id = ?"
            params.append(campus_id)
        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [DocumentJob.from_dict(dict(r)) for r in rows]

    async def get_jobs_by_campus(self, campus_id: str) -> list[DocumentJob]:
        """Return all jobs for the given *campus_id*."""
        cursor = await self.conn.execute(
            "SELECT * FROM document_jobs WHERE campus_id = ? ORDER BY created_at",
            (campus_id,),
        )
        rows = await cursor.fetchall()
        return [DocumentJob.from_dict(dict(r)) for r in rows]

    async def get_all_jobs(
        self, *, campus_id: str | None = None
    ) -> list[DocumentJob]:
        """Return every job in the database.

        If *campus_id* is provided, only jobs for that campus are returned.
        """
        if campus_id is not None:
            sql = (
                "SELECT * FROM document_jobs WHERE campus_id = ? "
                "ORDER BY created_at"
            )
            cursor = await self.conn.execute(sql, (campus_id,))
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM document_jobs ORDER BY created_at"
            )
        rows = await cursor.fetchall()
        return [DocumentJob.from_dict(dict(r)) for r in rows]

    async def get_stats(
        self, *, campus_id: str | None = None
    ) -> dict[str, int]:
        """Return a {status_value: count} mapping for all jobs.

        If *campus_id* is provided, counts are scoped to that campus.
        """
        if campus_id is not None:
            cursor = await self.conn.execute(
                "SELECT status, COUNT(*) as cnt FROM document_jobs "
                "WHERE campus_id = ? GROUP BY status",
                (campus_id,),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT status, COUNT(*) as cnt FROM document_jobs "
                "GROUP BY status"
            )
        rows = await cursor.fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    async def job_exists_for_url(self, url: str) -> bool:
        """Check whether a job already exists for the given document URL."""
        cursor = await self.conn.execute(
            "SELECT 1 FROM document_jobs WHERE url = ? LIMIT 1", (url,)
        )
        return (await cursor.fetchone()) is not None

    async def job_exists_for_hash(self, file_hash: str) -> bool:
        """Check whether a job already exists for the given content hash."""
        if not file_hash:
            return False
        cursor = await self.conn.execute(
            "SELECT 1 FROM document_jobs WHERE file_hash = ? LIMIT 1", (file_hash,)
        )
        return (await cursor.fetchone()) is not None

    # ------------------------------------------------------------------
    # Crawl state helpers (resumability)
    # ------------------------------------------------------------------

    async def mark_url_discovered(self, url: str, depth: int) -> None:
        """Record a URL the crawler has discovered but not yet visited."""
        now = datetime.now(timezone.utc).isoformat()
        await self.conn.execute(
            "INSERT OR IGNORE INTO crawl_state (url, depth, visited, discovered_at) "
            "VALUES (?, ?, 0, ?)",
            (url, depth, now),
        )
        await self.conn.commit()

    async def mark_url_visited(self, url: str) -> None:
        """Flag a crawl URL as visited."""
        now = datetime.now(timezone.utc).isoformat()
        await self.conn.execute(
            "UPDATE crawl_state SET visited = 1, visited_at = ? WHERE url = ?",
            (now, url),
        )
        await self.conn.commit()

    async def get_unvisited_urls(self) -> list[tuple[str, int]]:
        """Return ``(url, depth)`` pairs for all unvisited crawl targets."""
        cursor = await self.conn.execute(
            "SELECT url, depth FROM crawl_state WHERE visited = 0 ORDER BY depth, rowid"
        )
        return [(row["url"], row["depth"]) for row in await cursor.fetchall()]

    # ------------------------------------------------------------------
    # Validation log
    # ------------------------------------------------------------------

    async def log_validation(
        self,
        job_id: str,
        cycle: int,
        page_key: str,
        page_title: str,
        page_path: str,
        tool: str,
        score: float | None,
        violations: list[dict[str, Any]],
        passed: bool,
    ) -> None:
        """Append an entry to the validation audit log."""
        now = datetime.now(timezone.utc).isoformat()
        await self.conn.execute(
            "INSERT INTO validation_log "
            "("
            "job_id, cycle, page_key, page_title, page_path, tool, score, "
            "violations, passed, created_at"
            ") "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                cycle,
                page_key,
                page_title,
                page_path,
                tool,
                score,
                json.dumps(violations),
                int(passed),
                now,
            ),
        )
        await self.conn.commit()

    # ------------------------------------------------------------------
    # Deployment log
    # ------------------------------------------------------------------

    async def log_deployment(self, record: DeploymentRecord) -> None:
        """Insert a deployment record into the deployment_log table."""
        await self.conn.execute(
            "INSERT INTO deployment_log "
            "(job_id, campus_code, drupal_uuid, drupal_nid, source_url, "
            "content_type, status, error_message, deployed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.job_id,
                record.campus_code,
                record.drupal_uuid,
                record.drupal_nid,
                record.source_url,
                record.content_type,
                record.status,
                record.error_message,
                record.deployed_at.isoformat(),
            ),
        )
        await self.conn.commit()

    async def get_deployments(
        self, campus_code: str | None = None
    ) -> list[DeploymentRecord]:
        """Return deployment records, optionally filtered by campus."""
        if campus_code:
            cursor = await self.conn.execute(
                "SELECT * FROM deployment_log WHERE campus_code = ? "
                "ORDER BY deployed_at DESC",
                (campus_code,),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM deployment_log ORDER BY deployed_at DESC"
            )
        rows = await cursor.fetchall()
        return [self._row_to_deployment(dict(r)) for r in rows]

    async def get_deployment_for_job(self, job_id: str) -> DeploymentRecord | None:
        """Return the most recent deployment record for a given job."""
        cursor = await self.conn.execute(
            "SELECT * FROM deployment_log WHERE job_id = ? "
            "ORDER BY deployed_at DESC LIMIT 1",
            (job_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_deployment(dict(row))

    async def get_deployment_stats(
        self, campus_code: str | None = None
    ) -> DeploymentSummary:
        """Return aggregate deployment statistics."""
        if campus_code:
            cursor = await self.conn.execute(
                "SELECT status, COUNT(*) as cnt FROM deployment_log "
                "WHERE campus_code = ? GROUP BY status",
                (campus_code,),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT status, COUNT(*) as cnt FROM deployment_log "
                "GROUP BY status"
            )
        rows = await cursor.fetchall()
        counts = {row["status"]: row["cnt"] for row in rows}

        # Get last deployed timestamp
        if campus_code:
            cursor2 = await self.conn.execute(
                "SELECT MAX(deployed_at) as last_at FROM deployment_log "
                "WHERE campus_code = ?",
                (campus_code,),
            )
        else:
            cursor2 = await self.conn.execute(
                "SELECT MAX(deployed_at) as last_at FROM deployment_log"
            )
        last_row = await cursor2.fetchone()
        last_at_raw = dict(last_row).get("last_at") if last_row else None
        last_at = None
        if last_at_raw:
            from project_remedy.models import _parse_datetime
            last_at = _parse_datetime(last_at_raw)

        created = counts.get("created", 0)
        updated = counts.get("updated", 0)
        failed = counts.get("failed", 0)

        return DeploymentSummary(
            campus_code=campus_code or "",
            total_deployed=created + updated + failed,
            created=created,
            updated=updated,
            failed=failed,
            last_deployed_at=last_at,
        )

    @staticmethod
    def _row_to_deployment(data: dict) -> DeploymentRecord:
        """Convert a database row to a DeploymentRecord."""
        from project_remedy.models import _parse_datetime
        return DeploymentRecord(
            job_id=data.get("job_id", ""),
            campus_code=data.get("campus_code", ""),
            drupal_uuid=data.get("drupal_uuid", ""),
            drupal_nid=data.get("drupal_nid", 0),
            source_url=data.get("source_url", ""),
            content_type=data.get("content_type", "page"),
            status=data.get("status", "created"),
            error_message=data.get("error_message", ""),
            deployed_at=_parse_datetime(data.get("deployed_at", "")),
        )

    async def _ensure_schema_migrations(self) -> None:
        """Add newly introduced columns when opening an existing database."""
        await self._ensure_column(
            "document_jobs",
            "campus_id",
            "TEXT NOT NULL DEFAULT ''",
        )
        await self._ensure_column(
            "document_jobs",
            "generated_pages_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        await self._ensure_column(
            "document_jobs",
            "extracted_images_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        await self._ensure_column(
            "document_jobs",
            "remediated_document_path",
            "TEXT NOT NULL DEFAULT ''",
        )
        await self._ensure_column(
            "document_jobs",
            "remediated_pdf_path",
            "TEXT NOT NULL DEFAULT ''",
        )
        await self._ensure_column(
            "document_jobs",
            "warning_message",
            "TEXT NOT NULL DEFAULT ''",
        )
        await self._ensure_column(
            "document_jobs",
            "acceptance_warnings_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        await self._ensure_column(
            "validation_log",
            "page_key",
            "TEXT NOT NULL DEFAULT ''",
        )
        await self._ensure_column(
            "validation_log",
            "page_title",
            "TEXT NOT NULL DEFAULT ''",
        )
        await self._ensure_column(
            "validation_log",
            "page_path",
            "TEXT NOT NULL DEFAULT ''",
        )

    async def _ensure_column(
        self,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        """Add a table column if it does not already exist."""
        cursor = await self.conn.execute(f"PRAGMA table_info({table_name})")
        rows = await cursor.fetchall()
        columns = {row["name"] for row in rows}
        if column_name in columns:
            return

        await self.conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
        )

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    async def get_incomplete_jobs(self) -> list[DocumentJob]:
        """Return jobs that were mid-flight when the pipeline last stopped.

        These are jobs in a transitional status (downloading, extracting,
        planning, converting, validating) that should be retried on restart.
        """
        transitional = [
            JobStatus.DOWNLOADING.value,
            JobStatus.EXTRACTING.value,
            JobStatus.PLANNING.value,
            JobStatus.CONVERTING.value,
            JobStatus.VALIDATING.value,
            JobStatus.PDF_REMEDIATING.value,
        ]
        placeholders = ", ".join("?" for _ in transitional)
        sql = (
            f"SELECT * FROM document_jobs WHERE status IN ({placeholders}) "
            "ORDER BY created_at"
        )
        cursor = await self.conn.execute(sql, transitional)
        rows = await cursor.fetchall()
        return [DocumentJob.from_dict(dict(r)) for r in rows]

    async def reset_incomplete_jobs(self) -> int:
        """Roll incomplete jobs back to their previous stable status.

        Returns the number of jobs that were reset.
        """
        rollback_map: dict[str, str] = {
            JobStatus.DOWNLOADING.value: JobStatus.DISCOVERED.value,
            JobStatus.EXTRACTING.value: JobStatus.DOWNLOADED.value,
            JobStatus.PLANNING.value: JobStatus.EXTRACTED.value,
            JobStatus.CONVERTING.value: JobStatus.PLANNED.value,
            JobStatus.VALIDATING.value: JobStatus.CONVERTED.value,
            JobStatus.DOCUMENT_REMEDIATING.value: JobStatus.DOWNLOADED.value,
            JobStatus.PDF_REMEDIATING.value: JobStatus.CONVERTED.value,
        }
        now = datetime.now(timezone.utc).isoformat()
        total = 0
        for from_status, to_status in rollback_map.items():
            cursor = await self.conn.execute(
                "UPDATE document_jobs SET status = ?, updated_at = ? WHERE status = ?",
                (to_status, now, from_status),
            )
            total += cursor.rowcount
        await self.conn.commit()
        if total:
            logger.info("Reset %d incomplete jobs for resumption.", total)
        return total
