"""SQLite-backed experiment tracking for Meta-Harness auto-prompt evolution.

Records (document_type, fix_sequence, outcome) tuples and harness variant
metadata. Provides queries for the proposer and scorer to understand what
has been tried and what works.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRecord:
    """A single experiment run: one harness variant applied to one document."""

    experiment_id: str = ""
    harness_id: str = ""
    document_hash: str = ""
    document_type: str = ""           # e.g. "table_heavy", "mixed_structure"
    violation_types: list[str] = field(default_factory=list)
    fix_sequence: list[dict] = field(default_factory=list)  # operations applied
    violations_before: int = 0
    violations_after: int = 0
    passed: bool = False
    elapsed_seconds: float = 0.0
    confidence: float = 0.0
    error: str | None = None
    created_at: str = ""


@dataclass
class HarnessVariant:
    """Metadata for a harness variant (prompt configuration)."""

    harness_id: str = ""
    parent_id: str | None = None      # which harness it was derived from
    description: str = ""
    status: str = "active"            # active, retired, promoted
    conformance_rate: float = 0.0
    manual_review_rate: float = 0.0
    destructive_edit_count: int = 0
    avg_seconds: float = 0.0
    total_docs: int = 0
    passed_docs: int = 0
    created_at: str = ""
    retired_at: str | None = None
    promoted_at: str | None = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS harness_variants (
    harness_id              TEXT PRIMARY KEY,
    parent_id               TEXT,
    description             TEXT NOT NULL DEFAULT '',
    status                  TEXT NOT NULL DEFAULT 'active',
    conformance_rate        REAL NOT NULL DEFAULT 0.0,
    manual_review_rate      REAL NOT NULL DEFAULT 0.0,
    destructive_edit_count  INTEGER NOT NULL DEFAULT 0,
    avg_seconds             REAL NOT NULL DEFAULT 0.0,
    total_docs              INTEGER NOT NULL DEFAULT 0,
    passed_docs             INTEGER NOT NULL DEFAULT 0,
    created_at              TEXT NOT NULL,
    retired_at              TEXT,
    promoted_at             TEXT,
    harness_config_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS experiment_records (
    experiment_id       TEXT PRIMARY KEY,
    harness_id          TEXT NOT NULL,
    document_hash       TEXT NOT NULL,
    document_type       TEXT NOT NULL DEFAULT '',
    violation_types_json TEXT NOT NULL DEFAULT '[]',
    fix_sequence_json   TEXT NOT NULL DEFAULT '[]',
    violations_before   INTEGER NOT NULL DEFAULT 0,
    violations_after    INTEGER NOT NULL DEFAULT 0,
    passed              INTEGER NOT NULL DEFAULT 0,
    elapsed_seconds     REAL NOT NULL DEFAULT 0.0,
    confidence          REAL NOT NULL DEFAULT 0.0,
    error               TEXT,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (harness_id) REFERENCES harness_variants(harness_id)
);

CREATE INDEX IF NOT EXISTS idx_exp_harness ON experiment_records(harness_id);
CREATE INDEX IF NOT EXISTS idx_exp_document ON experiment_records(document_hash);
CREATE INDEX IF NOT EXISTS idx_exp_passed ON experiment_records(passed);
CREATE INDEX IF NOT EXISTS idx_exp_doc_type ON experiment_records(document_type);
CREATE INDEX IF NOT EXISTS idx_variant_status ON harness_variants(status);

CREATE TABLE IF NOT EXISTS pareto_frontier (
    harness_id          TEXT PRIMARY KEY,
    conformance_rate    REAL NOT NULL DEFAULT 0.0,
    manual_review_rate  REAL NOT NULL DEFAULT 0.0,
    destructive_edit_count INTEGER NOT NULL DEFAULT 0,
    avg_seconds         REAL NOT NULL DEFAULT 0.0,
    updated_at          TEXT NOT NULL,
    FOREIGN KEY (harness_id) REFERENCES harness_variants(harness_id)
);

CREATE TABLE IF NOT EXISTS evolution_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration   INTEGER NOT NULL,
    action      TEXT NOT NULL,
    harness_id  TEXT NOT NULL,
    details     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# ExperimentStore
# ---------------------------------------------------------------------------


class ExperimentStore:
    """SQLite-backed store for experiment tracking.

    For file-backed databases, uses connection-per-call for thread safety.
    For in-memory databases, reuses a single connection (since each new
    connection to :memory: creates a separate database).
    """

    def __init__(self, db_path: Path | str = ":memory:"):
        self._db_path = str(db_path)
        self._is_memory = self._db_path == ":memory:"
        # For in-memory DBs, keep a persistent connection
        if self._is_memory:
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
            self._persistent_conn.execute("PRAGMA foreign_keys=ON")
        else:
            self._persistent_conn = None
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, yield it, and commit/rollback on exit.

        For in-memory databases, reuses a single persistent connection.
        For file-backed databases, opens and closes per call.
        """
        if self._persistent_conn is not None:
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # -- Harness variants ---------------------------------------------------

    def register_variant(
        self,
        harness_id: str,
        description: str = "",
        parent_id: str | None = None,
        harness_config: dict | None = None,
    ) -> HarnessVariant:
        """Register a new harness variant."""
        now = datetime.now(timezone.utc).isoformat()
        config_json = json.dumps(harness_config or {})

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO harness_variants
                   (harness_id, parent_id, description, status, created_at, harness_config_json)
                   VALUES (?, ?, ?, 'active', ?, ?)""",
                (harness_id, parent_id, description, now, config_json),
            )

        return HarnessVariant(
            harness_id=harness_id,
            parent_id=parent_id,
            description=description,
            status="active",
            created_at=now,
        )

    def get_variant(self, harness_id: str) -> HarnessVariant | None:
        """Get a harness variant by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM harness_variants WHERE harness_id = ?",
                (harness_id,),
            ).fetchone()

        if row is None:
            return None

        return HarnessVariant(
            harness_id=row["harness_id"],
            parent_id=row["parent_id"],
            description=row["description"],
            status=row["status"],
            conformance_rate=row["conformance_rate"],
            manual_review_rate=row["manual_review_rate"],
            destructive_edit_count=row["destructive_edit_count"],
            avg_seconds=row["avg_seconds"],
            total_docs=row["total_docs"],
            passed_docs=row["passed_docs"],
            created_at=row["created_at"],
            retired_at=row["retired_at"],
            promoted_at=row["promoted_at"],
        )

    def list_variants(self, status: str | None = None) -> list[HarnessVariant]:
        """List all variants, optionally filtered by status."""
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM harness_variants WHERE status = ? ORDER BY created_at",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM harness_variants ORDER BY created_at",
                ).fetchall()

        return [
            HarnessVariant(
                harness_id=r["harness_id"],
                parent_id=r["parent_id"],
                description=r["description"],
                status=r["status"],
                conformance_rate=r["conformance_rate"],
                manual_review_rate=r["manual_review_rate"],
                destructive_edit_count=r["destructive_edit_count"],
                avg_seconds=r["avg_seconds"],
                total_docs=r["total_docs"],
                passed_docs=r["passed_docs"],
                created_at=r["created_at"],
                retired_at=r["retired_at"],
                promoted_at=r["promoted_at"],
            )
            for r in rows
        ]

    def update_variant_metrics(
        self,
        harness_id: str,
        conformance_rate: float,
        manual_review_rate: float,
        destructive_edit_count: int,
        avg_seconds: float,
        total_docs: int,
        passed_docs: int,
    ) -> None:
        """Update computed metrics on a variant after scoring."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE harness_variants
                   SET conformance_rate = ?,
                       manual_review_rate = ?,
                       destructive_edit_count = ?,
                       avg_seconds = ?,
                       total_docs = ?,
                       passed_docs = ?
                   WHERE harness_id = ?""",
                (
                    conformance_rate,
                    manual_review_rate,
                    destructive_edit_count,
                    avg_seconds,
                    total_docs,
                    passed_docs,
                    harness_id,
                ),
            )

    def set_variant_status(
        self, harness_id: str, status: str
    ) -> None:
        """Change a variant's status (active, retired, promoted)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            updates = {"status": status}
            if status == "retired":
                conn.execute(
                    "UPDATE harness_variants SET status = ?, retired_at = ? WHERE harness_id = ?",
                    (status, now, harness_id),
                )
            elif status == "promoted":
                conn.execute(
                    "UPDATE harness_variants SET status = ?, promoted_at = ? WHERE harness_id = ?",
                    (status, now, harness_id),
                )
            else:
                conn.execute(
                    "UPDATE harness_variants SET status = ? WHERE harness_id = ?",
                    (status, harness_id),
                )

    # -- Experiment records --------------------------------------------------

    def record_experiment(self, record: ExperimentRecord) -> None:
        """Insert an experiment record."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiment_records
                   (experiment_id, harness_id, document_hash, document_type,
                    violation_types_json, fix_sequence_json,
                    violations_before, violations_after, passed,
                    elapsed_seconds, confidence, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.experiment_id,
                    record.harness_id,
                    record.document_hash,
                    record.document_type,
                    json.dumps(record.violation_types),
                    json.dumps(record.fix_sequence),
                    record.violations_before,
                    record.violations_after,
                    1 if record.passed else 0,
                    record.elapsed_seconds,
                    record.confidence,
                    record.error,
                    record.created_at or datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_experiments_for_harness(self, harness_id: str) -> list[ExperimentRecord]:
        """Get all experiment records for a specific harness variant."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiment_records WHERE harness_id = ? ORDER BY created_at",
                (harness_id,),
            ).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def get_experiments_for_document(self, document_hash: str) -> list[ExperimentRecord]:
        """Get all experiment records for a specific document across harnesses."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiment_records WHERE document_hash = ? ORDER BY created_at",
                (document_hash,),
            ).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def get_failure_patterns(self, harness_id: str) -> dict[str, Any]:
        """Analyze failure patterns for a harness variant.

        Returns dict with:
        - failing_doc_types: {doc_type: count}
        - failing_violation_types: {violation_type: count}
        - common_errors: {error: count}
        - destructive_docs: list of document hashes where violations increased
        """
        experiments = self.get_experiments_for_harness(harness_id)
        failing = [e for e in experiments if not e.passed]

        doc_types: dict[str, int] = {}
        violation_types: dict[str, int] = {}
        errors: dict[str, int] = {}
        destructive: list[str] = []

        for exp in failing:
            doc_types[exp.document_type] = doc_types.get(exp.document_type, 0) + 1
            for vt in exp.violation_types:
                violation_types[vt] = violation_types.get(vt, 0) + 1
            if exp.error:
                errors[exp.error] = errors.get(exp.error, 0) + 1

        for exp in experiments:
            if exp.violations_after > exp.violations_before:
                destructive.append(exp.document_hash)

        return {
            "failing_doc_types": doc_types,
            "failing_violation_types": violation_types,
            "common_errors": errors,
            "destructive_docs": destructive,
        }

    def compute_success_rate(self, harness_id: str) -> float:
        """Compute the conformance rate for a harness variant."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed_count
                   FROM experiment_records WHERE harness_id = ?""",
                (harness_id,),
            ).fetchone()

        if row is None or row["total"] == 0:
            return 0.0
        return row["passed_count"] / row["total"]

    # -- Pareto frontier ----------------------------------------------------

    def update_pareto_frontier(self) -> list[dict]:
        """Recompute and persist the Pareto frontier from all active variants."""
        variants = self.list_variants(status="active")
        # Also include promoted variants
        promoted = self.list_variants(status="promoted")
        all_candidates = variants + promoted

        if not all_candidates:
            return []

        # Filter to variants with at least one experiment
        scored = [v for v in all_candidates if v.total_docs > 0]
        if not scored:
            return []

        frontier = []
        for candidate in scored:
            dominated = False
            for other in scored:
                if other.harness_id == candidate.harness_id:
                    continue
                if _dominates(other, candidate):
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM pareto_frontier")
            for v in frontier:
                conn.execute(
                    """INSERT INTO pareto_frontier
                       (harness_id, conformance_rate, manual_review_rate,
                        destructive_edit_count, avg_seconds, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        v.harness_id,
                        v.conformance_rate,
                        v.manual_review_rate,
                        v.destructive_edit_count,
                        v.avg_seconds,
                        now,
                    ),
                )

        return [
            {
                "harness_id": v.harness_id,
                "conformance_rate": v.conformance_rate,
                "manual_review_rate": v.manual_review_rate,
                "destructive_edit_count": v.destructive_edit_count,
                "avg_seconds": v.avg_seconds,
            }
            for v in frontier
        ]

    def get_pareto_frontier(self) -> list[dict]:
        """Get the current Pareto frontier."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pareto_frontier ORDER BY conformance_rate DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    # -- Evolution log ------------------------------------------------------

    def log_evolution(
        self, iteration: int, action: str, harness_id: str, details: str = ""
    ) -> None:
        """Log an evolution action (propose, promote, retire, etc.)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO evolution_log (iteration, action, harness_id, details, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (iteration, action, harness_id, details, now),
            )

    def get_evolution_log(self, limit: int = 50) -> list[dict]:
        """Get recent evolution log entries."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evolution_log ORDER BY log_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            harness_id=row["harness_id"],
            document_hash=row["document_hash"],
            document_type=row["document_type"],
            violation_types=json.loads(row["violation_types_json"]),
            fix_sequence=json.loads(row["fix_sequence_json"]),
            violations_before=row["violations_before"],
            violations_after=row["violations_after"],
            passed=bool(row["passed"]),
            elapsed_seconds=row["elapsed_seconds"],
            confidence=row["confidence"],
            error=row["error"],
            created_at=row["created_at"],
        )


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------


def _dominates(a: HarnessVariant, b: HarnessVariant) -> bool:
    """Return True if variant a Pareto-dominates variant b.

    Maximize: conformance_rate
    Minimize: manual_review_rate, destructive_edit_count, avg_seconds
    """
    checks = [
        a.conformance_rate >= b.conformance_rate,
        a.manual_review_rate <= b.manual_review_rate,
        a.destructive_edit_count <= b.destructive_edit_count,
        a.avg_seconds <= b.avg_seconds,
    ]
    strict = [
        a.conformance_rate > b.conformance_rate,
        a.manual_review_rate < b.manual_review_rate,
        a.destructive_edit_count < b.destructive_edit_count,
        a.avg_seconds < b.avg_seconds,
    ]
    return all(checks) and any(strict)
