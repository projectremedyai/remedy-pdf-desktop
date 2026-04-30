"""Structured logging setup with Rich console and rotating JSON file handlers."""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "job_id"):
            entry["job_id"] = record.job_id  # type: ignore[attr-defined]
        return json.dumps(entry, default=str)


def setup_logging(
    log_dir: Path,
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure the root logger with Rich console and rotating JSON file output.

    Parameters
    ----------
    log_dir:
        Directory where ``pipeline.log.jsonl`` will be written.
    console_level:
        Minimum level for Rich console output.
    file_level:
        Minimum level for the JSON file output.
    max_bytes:
        Maximum size of a single log file before rotation.
    backup_count:
        Number of rotated log files to retain.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log.jsonl"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on repeated calls.
    if root.handlers:
        return

    # -- Rich console handler -----------------------------------------------
    console = Console(stderr=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(console_level)
    root.addHandler(console_handler)

    # -- Rotating JSON file handler -----------------------------------------
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(_JSONFormatter())
    root.addHandler(file_handler)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "aiohttp", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
