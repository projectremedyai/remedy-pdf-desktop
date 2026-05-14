"""User-editable vision provider settings for the desktop app."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from backend.app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "gemma4:e4b"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_OLLAMA_CLOUD_MODEL = "qwen3-vl:235b-cloud"
DEFAULT_VISION_PAGE_TIMEOUT_SECONDS = 90

VALID_PROVIDERS = ("local", "ollama_cloud", "openrouter")

_lock = threading.Lock()


@dataclass
class VisionSettings:
    provider: str = "local"
    local_model: str = DEFAULT_LOCAL_MODEL
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    ollama_cloud_model: str = DEFAULT_OLLAMA_CLOUD_MODEL
    openrouter_api_key: str = ""
    ollama_cloud_api_key: str = ""
    page_timeout_seconds: int = DEFAULT_VISION_PAGE_TIMEOUT_SECONDS

    def normalised(self) -> "VisionSettings":
        """Return a copy with invalid fields coerced back to defaults."""
        provider = self.provider if self.provider in VALID_PROVIDERS else "local"
        raw_timeout = self.page_timeout_seconds or DEFAULT_VISION_PAGE_TIMEOUT_SECONDS
        timeout = max(15, min(600, int(raw_timeout)))
        return replace(
            self,
            provider=provider,
            local_model=(self.local_model or DEFAULT_LOCAL_MODEL).strip(),
            openrouter_model=(
                self.openrouter_model or DEFAULT_OPENROUTER_MODEL
            ).strip(),
            ollama_cloud_model=(
                self.ollama_cloud_model or DEFAULT_OLLAMA_CLOUD_MODEL
            ).strip(),
            openrouter_api_key=(self.openrouter_api_key or "").strip(),
            ollama_cloud_api_key=(self.ollama_cloud_api_key or "").strip(),
            page_timeout_seconds=timeout,
        )


def settings_path() -> Path:
    return Path(settings.output_dir).parent / "vision_settings.json"


def initial_from_env() -> VisionSettings:
    """Seed defaults from env vars so existing installs keep working."""
    return VisionSettings(
        provider=os.getenv("VISION_PROVIDER", "local").strip().lower(),
        local_model=os.getenv("OLLAMA_VISION_MODEL", settings.ollama_model_tag).strip(),
        openrouter_model=os.getenv(
            "OPENROUTER_VISION_MODEL", DEFAULT_OPENROUTER_MODEL
        ).strip(),
        ollama_cloud_model=os.getenv(
            "OLLAMA_CLOUD_VISION_MODEL", DEFAULT_OLLAMA_CLOUD_MODEL
        ).strip(),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        ollama_cloud_api_key=os.getenv("OLLAMA_API_KEY", "").strip(),
        page_timeout_seconds=int(
            os.getenv(
                "PDF_FIXER_VISION_PAGE_TIMEOUT",
                str(DEFAULT_VISION_PAGE_TIMEOUT_SECONDS),
            )
        ),
    ).normalised()


def load() -> VisionSettings:
    """Read settings from disk, falling back to env-derived defaults."""
    path = settings_path()
    if not path.exists():
        return initial_from_env()
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not read vision_settings.json (%s); using defaults", exc)
        return initial_from_env()
    if not isinstance(data, dict):
        return initial_from_env()

    base = initial_from_env()
    try:
        merged = VisionSettings(
            provider=str(data.get("provider", base.provider)),
            local_model=str(data.get("local_model", base.local_model)),
            openrouter_model=str(
                data.get("openrouter_model", base.openrouter_model)
            ),
            ollama_cloud_model=str(
                data.get("ollama_cloud_model", base.ollama_cloud_model)
            ),
            openrouter_api_key=str(
                data.get("openrouter_api_key", base.openrouter_api_key)
            ),
            ollama_cloud_api_key=str(
                data.get("ollama_cloud_api_key", base.ollama_cloud_api_key)
            ),
            page_timeout_seconds=int(
                data.get("page_timeout_seconds", base.page_timeout_seconds)
            ),
        )
    except Exception as exc:
        logger.warning("Could not parse vision_settings.json (%s); using defaults", exc)
        return initial_from_env()
    return merged.normalised()


def save(new_settings: VisionSettings) -> VisionSettings:
    """Atomically persist settings to disk, returning the normalised result."""
    cleaned = new_settings.normalised()
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = asdict(cleaned)
    with _lock:
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
    return cleaned


def public_view(settings_obj: VisionSettings) -> dict:
    """Serialise settings for the API, masking secrets."""
    data = asdict(settings_obj)
    for key in ("openrouter_api_key", "ollama_cloud_api_key"):
        value = data.get(key) or ""
        data[f"{key}_set"] = bool(value)
        data[key] = "*" * 8 + value[-4:] if len(value) >= 8 else ""
    return data
