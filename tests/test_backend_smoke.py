from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

import backend.app.routes_api as routes_api
from backend.app.config import settings
from backend.app.main import app
from backend.app.ollama_runtime import (
    LocalOllamaStatus,
    _infer_vision_task,
    _parse_task_map,
)
from backend.app.routes_api import _safe_upload_name
from project_remedy.vision_prompts import (
    contrast_detection_prompt,
    heading_hierarchy_quality_prompt,
    page_alt_text_quality_prompt,
    page_region_analysis_prompt,
    reading_order_prompt,
    semantic_reading_order_prompt,
    wcag_table_verify_prompt,
)


def test_health_endpoint_returns_security_headers(monkeypatch) -> None:
    monkeypatch.setattr(settings, "app_api_key", "")

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_api_key_auth_blocks_protected_api_routes(monkeypatch) -> None:
    monkeypatch.setattr(settings, "app_api_key", "test-secret")

    with TestClient(app) as client:
        protected_response = client.get("/api/jobs/missing-job")
        health_response = client.get("/api/health")

    assert protected_response.status_code == 401
    assert protected_response.json() == {"detail": "Invalid or missing API key"}
    assert health_response.status_code == 200


def test_model_status_reports_local_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "app_api_key", "")
    monkeypatch.setattr(settings, "ollama_model_tag", "qwen3.5:4b")

    def fake_status() -> LocalOllamaStatus:
        return LocalOllamaStatus(
            reachable=True,
            installed=True,
            model_tag="qwen3.5:4b",
            endpoint="http://127.0.0.1:11500/v1",
            models_dir=tmp_path / "models",
            size_bytes=3_400_000_000,
        )

    monkeypatch.setattr(routes_api, "get_local_ollama_status", fake_status)

    with TestClient(app) as client:
        response = client.get("/api/model/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["reachable"] is True
    assert payload["installed"] is True
    assert payload["model_tag"] == "qwen3.5:4b"
    assert payload["endpoint"] == "http://127.0.0.1:11500/v1"
    assert payload["models_dir"] == str(tmp_path / "models")
    assert payload["size_mb"] == 3400.0
    assert payload["default_model"]["tag"] == "qwen3.5:4b"


def test_vision_settings_round_trip_masks_keys(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "app_api_key", "")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    with TestClient(app) as client:
        update = client.put(
            "/api/settings/vision",
            json={
                "provider": "openrouter",
                "openrouter_model": "openai/gpt-4o-mini",
                "openrouter_api_key": "sk-or-v1-test-123456",
                "page_timeout_seconds": 120,
            },
        )
        read_back = client.get("/api/settings/vision")

    assert update.status_code == 200
    payload = update.json()
    assert payload["provider"] == "openrouter"
    assert payload["openrouter_model"] == "openai/gpt-4o-mini"
    assert payload["openrouter_api_key_set"] is True
    assert payload["openrouter_api_key"] == "********3456"
    assert payload["page_timeout_seconds"] == 120
    assert read_back.json()["openrouter_api_key"] == "********3456"


def test_upload_name_is_confined_to_safe_basename() -> None:
    assert _safe_upload_name(r"..\..\weird report?.pdf") == "weird_report_.pdf"
    assert _safe_upload_name("\x00../../") == "upload"


def test_ollama_task_model_env_map_normalises_aliases() -> None:
    parsed = _parse_task_map(
        "alt:minicpm-alt,"
        "heading:minicpm-heading,"
        "color-contrast:minicpm-contrast,"
        "table:minicpm-table,"
        "reading_order:minicpm-reading,"
        "bad-entry"
    )

    assert parsed == {
        "alt_text_quality": "minicpm-alt",
        "heading_hierarchy": "minicpm-heading",
        "contrast": "minicpm-contrast",
        "table_structure": "minicpm-table",
        "reading_order": "minicpm-reading",
    }


def test_ollama_prompt_task_inference_matches_remedy_prompt_families() -> None:
    assert (
        _infer_vision_task(page_alt_text_quality_prompt(figure_list="1. Figure"))
        == "alt_text_quality"
    )
    assert _infer_vision_task(contrast_detection_prompt("AA")) == "contrast"
    assert (
        _infer_vision_task(
            heading_hierarchy_quality_prompt(logical_order="1. /H2 Title")
        )
        == "heading_hierarchy"
    )
    assert (
        _infer_vision_task(
            semantic_reading_order_prompt(element_list="1. /H2 Title")
        )
        == "heading_hierarchy"
    )
    assert (
        _infer_vision_task(
            reading_order_prompt(structure_order="1. /P Intro")
        )
        == "reading_order"
    )
    assert (
        _infer_vision_task(
            page_region_analysis_prompt(element_list="1. /P Intro", profile="local")
        )
        == "reading_order"
    )
    assert (
        _infer_vision_task(wcag_table_verify_prompt("Table > TR > TH"))
        == "table_structure"
    )


def test_removed_provider_classes_stay_absent() -> None:
    """Guard against accidental re-introduction of the OpenAI and Anthropic
    vision provider classes that were removed in favour of OpenRouter.

    Incidental mentions of "openai/..." or "anthropic/..." model slugs are
    allowed because those are valid OpenRouter routing identifiers (e.g.
    ``openai/gpt-4o-mini``, ``anthropic/claude-3-5-sonnet``). The earlier
    Gemini scrub was deliberately retired when the project switched to
    OpenRouter — OpenRouter can route to Gemini models too, so the name now
    appears in docs as one of several routing examples.
    """
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )

    text_suffixes = {
        ".css",
        ".example",
        ".html",
        ".js",
        ".json",
        ".md",
        ".py",
        ".rs",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
    text_names = {"Dockerfile", ".gitignore"}
    self_path = Path(__file__).resolve()
    # Match class identifiers as whole words so substrings (e.g. inside
    # docstrings describing OpenRouter slugs) don't trip the scrub.
    forbidden_classes = ("OpenAIVisionProvider", "AnthropicVisionProvider")

    matches: list[str] = []
    for raw_name in result.stdout.decode("utf-8").split("\0"):
        if not raw_name:
            continue
        path = root / raw_name
        if path.suffix.lower() not in text_suffixes and path.name not in text_names:
            continue
        if path.resolve() == self_path:
            continue  # don't scan this test file itself
        content = path.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden_classes:
            if term in content:
                matches.append(f"{raw_name}: {term}")

    assert matches == []
