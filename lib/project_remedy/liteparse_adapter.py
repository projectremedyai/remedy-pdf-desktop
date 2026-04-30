"""Thin local adapter for LiteParse-based text snapshots and triage.

This module intentionally keeps LiteParse on the read/triage side only.
It does not mutate PDFs and should never be treated as the source of truth
for final remediation decisions.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class LiteParseSnapshot:
    pdf_path: Path
    page_spec: str
    text: str
    char_count: int
    line_count: int
    pages_sampled: int
    used: bool
    timed_out: bool = False
    parser_error: str = ""


@dataclass(frozen=True)
class LiteParseTriage:
    classification: str
    char_count: int
    pages_sampled: int
    used: bool
    timed_out: bool = False
    parser_error: str = ""


def _env_bool(name: str, default: bool) -> bool:
    raw = __import__("os").environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _settings(config=None) -> dict[str, object]:
    api = getattr(config, "api", None)
    return {
        "enabled": bool(
            getattr(api, "liteparse_enabled", _env_bool("LITEPARSE_ENABLED", False))
        ),
        "bin": str(
            getattr(api, "liteparse_bin", __import__("os").environ.get("LITEPARSE_BIN", "lit"))
        ).strip()
        or "lit",
        "timeout_seconds": float(
            getattr(
                api,
                "liteparse_timeout_seconds",
                float(__import__("os").environ.get("LITEPARSE_TIMEOUT_SECONDS", "30")),
            )
        ),
        "sample_pages": int(
            getattr(
                api,
                "liteparse_sample_pages",
                int(__import__("os").environ.get("LITEPARSE_SAMPLE_PAGES", "3")),
            )
        ),
        "text_rich_min_chars": int(
            getattr(
                api,
                "liteparse_text_rich_min_chars",
                int(__import__("os").environ.get("LITEPARSE_TEXT_RICH_MIN_CHARS", "800")),
            )
        ),
        "sparse_max_chars": int(
            getattr(
                api,
                "liteparse_sparse_max_chars",
                int(__import__("os").environ.get("LITEPARSE_SPARSE_MAX_CHARS", "200")),
            )
        ),
    }


def _resolve_binary(binary: str) -> str | None:
    cleaned = str(binary or "").strip()
    if not cleaned:
        return None
    if "/" in cleaned:
        path = Path(cleaned).expanduser()
        return str(path) if path.exists() else None
    return shutil.which(cleaned)


def _page_spec_from_limit(page_limit: int) -> str:
    limit = max(1, int(page_limit))
    return "1" if limit == 1 else f"1-{limit}"


def _count_pages_in_spec(page_spec: str) -> int:
    total = 0
    for chunk in str(page_spec).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            try:
                start_num = int(start)
                end_num = int(end)
                total += max(0, end_num - start_num + 1)
                continue
            except ValueError:
                pass
        total += 1
    return max(total, 1)


@lru_cache(maxsize=256)
def _run_liteparse_cached(
    pdf_path_str: str,
    page_spec: str,
    no_ocr: bool,
    binary_path: str,
    timeout_seconds: float,
) -> LiteParseSnapshot:
    pdf_path = Path(pdf_path_str)
    command = [binary_path, "parse", pdf_path_str, "--target-pages", page_spec]
    if no_ocr:
        command.append("--no-ocr")

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return LiteParseSnapshot(
            pdf_path=pdf_path,
            page_spec=page_spec,
            text="",
            char_count=0,
            line_count=0,
            pages_sampled=_count_pages_in_spec(page_spec),
            used=True,
            timed_out=True,
            parser_error="liteparse timed out",
        )

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        return LiteParseSnapshot(
            pdf_path=pdf_path,
            page_spec=page_spec,
            text="",
            char_count=0,
            line_count=0,
            pages_sampled=_count_pages_in_spec(page_spec),
            used=True,
            parser_error=stderr or stdout or f"liteparse exited with {proc.returncode}",
        )

    return LiteParseSnapshot(
        pdf_path=pdf_path,
        page_spec=page_spec,
        text=stdout,
        char_count=len(stdout),
        line_count=len([line for line in stdout.splitlines() if line.strip()]),
        pages_sampled=_count_pages_in_spec(page_spec),
        used=True,
    )


def clear_liteparse_cache() -> None:
    _run_liteparse_cached.cache_clear()


def liteparse_available(config=None) -> bool:
    settings = _settings(config)
    if not settings["enabled"]:
        return False
    return _resolve_binary(str(settings["bin"])) is not None


def liteparse_text_snapshot(
    pdf_path: Path,
    *,
    config=None,
    page_limit: int | None = None,
    page_spec: str | None = None,
    no_ocr: bool = True,
) -> LiteParseSnapshot:
    settings = _settings(config)
    resolved = pdf_path.expanduser().resolve()
    binary_path = _resolve_binary(str(settings["bin"]))
    chosen_page_spec = page_spec or _page_spec_from_limit(page_limit or int(settings["sample_pages"]))
    if not settings["enabled"] or binary_path is None:
        return LiteParseSnapshot(
            pdf_path=resolved,
            page_spec=chosen_page_spec,
            text="",
            char_count=0,
            line_count=0,
            pages_sampled=_count_pages_in_spec(chosen_page_spec),
            used=False,
        )
    return _run_liteparse_cached(
        str(resolved),
        chosen_page_spec,
        no_ocr,
        binary_path,
        float(settings["timeout_seconds"]),
    )


def classify_pdf_for_routing(pdf_path: Path, *, config=None) -> LiteParseTriage:
    settings = _settings(config)
    snapshot = liteparse_text_snapshot(
        pdf_path,
        config=config,
        page_limit=int(settings["sample_pages"]),
        no_ocr=True,
    )
    classification = "unknown"
    if snapshot.used and not snapshot.timed_out and not snapshot.parser_error:
        if snapshot.char_count >= int(settings["text_rich_min_chars"]):
            classification = "text_rich"
        elif snapshot.char_count <= int(settings["sparse_max_chars"]):
            classification = "sparse_or_scanned"
    return LiteParseTriage(
        classification=classification,
        char_count=snapshot.char_count,
        pages_sampled=snapshot.pages_sampled,
        used=snapshot.used,
        timed_out=snapshot.timed_out,
        parser_error=snapshot.parser_error,
    )

