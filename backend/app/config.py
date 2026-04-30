from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
_env_path = _project_root / ".env"
load_dotenv(_env_path)

_lib_dir = str(_project_root / "lib")
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

_data_dir = _project_root / "data"


@dataclass
class Settings:
    app_env: str = field(
        default_factory=lambda: os.getenv("APP_ENV", "development").strip().lower()
    )

    max_concurrent_jobs: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_JOBS", "5"))
    )

    visual_diff_review_threshold: float = field(
        default_factory=lambda: float(os.getenv("VISUAL_DIFF_REVIEW_THRESHOLD", "0.10"))
    )

    # Explicit cap on pages reviewed by the vision artifact check. 0 means
    # "auto-scale" — ~15% of pages bounded [3, 20] for sampled verification,
    # every page up to ``visual_artifact_check_max_pages`` for deterministic.
    # A positive value pins the count exactly (legacy behaviour).
    visual_artifact_check_pages: int = field(
        default_factory=lambda: int(os.getenv("VISUAL_ARTIFACT_CHECK_PAGES", "0"))
    )

    # Hard upper bound used by auto-scaling and by deterministic-mode coverage
    # so a 1000-page document doesn't burn 1000 vision calls on this check.
    visual_artifact_check_max_pages: int = field(
        default_factory=lambda: int(os.getenv("VISUAL_ARTIFACT_CHECK_MAX_PAGES", "20"))
    )

    upload_dir: Path = field(
        default_factory=lambda: Path(os.getenv("UPLOAD_DIR", str(_data_dir / "uploads")))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", str(_data_dir / "output")))
    )

    max_pages: int = field(default_factory=lambda: int(os.getenv("MAX_PAGES", "50")))

    # Max upload size in bytes (default 50 MB)
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))))

    # Optional API key — if set, all HTTP/WS requests must include it
    # in the X-API-Key header. Empty string = auth disabled (local dev).
    app_api_key: str = field(default_factory=lambda: os.getenv("APP_API_KEY", ""))

    # Rate limiting (requests per minute per IP for upload endpoint)
    upload_rate_limit: int = field(default_factory=lambda: int(os.getenv("UPLOAD_RATE_LIMIT", "10")))

    # CORS allowed origins (comma-separated). Defaults to localhost for dev.
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:1420,http://localhost:5173,http://localhost:3000,tauri://localhost",
        ).split(",") if o.strip()
    ])

    # Job TTL in seconds (default 2 hours). Jobs older than this are
    # purged by the cleanup background task.
    job_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("JOB_TTL_SECONDS", str(2 * 3600))))

    # WebSocket progress idle timeout (seconds). If the remediation pipeline
    # doesn't emit an event for this long, the WS pushes an error to the
    # frontend. Slow vision models can go minutes between events, so default
    # is generous. The backend job continues regardless of WS status.
    progress_ws_idle_timeout: int = field(
        default_factory=lambda: int(os.getenv("PROGRESS_WS_IDLE_TIMEOUT", "900"))
    )

    # Seconds to wait for Ollama to become reachable before starting a job.
    # Covers the cold-start race where the pipeline would otherwise make
    # vision calls before the model has finished loading — those connection
    # refusals used to cascade into false WCAG FAILs in the compliance report.
    vision_ready_wait_seconds: int = field(
        default_factory=lambda: int(os.getenv("VISION_READY_WAIT_SECONDS", "60"))
    )

    # Local Ollama runtime configuration.
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11500/v1")
    )
    ollama_model_tag: str = field(
        default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL", "qwen3.5:4b")
    )
    ollama_models_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("OLLAMA_MODELS", str(_data_dir / "ollama" / "models"))
        )
    )

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
