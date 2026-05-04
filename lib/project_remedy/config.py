"""Configuration module for the LACCD ADA Document Remediation Pipeline.

Loads settings from .env files and config.yaml, exposing them as typed
dataclasses for use throughout the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampusConfig:
    """Identity and branding for a single LACCD campus."""

    code: str = ""
    name: str = ""
    start_url: str = ""
    brand_primary: str = ""
    brand_accent: str = ""
    brand_neutral: str = ""


@dataclass(frozen=True)
class CrawlConfig:
    """Settings that control the web crawler stage."""

    start_url: str = ""
    max_depth: int = 10          # used by crawl4ai; unused by Firecrawl map
    max_pages: int = 10_000
    rate_limit: float = 2.0      # delay between requests for crawl4ai
    firecrawl_api_key: str = ""  # FIRECRAWL_API_KEY
    crawl_engine: str = "firecrawl"  # "firecrawl" or "crawl4ai"


@dataclass(frozen=True)
class APIConfig:
    """Settings for local LLM providers and optional non-desktop adapters."""

    api_key: str = ""                        # Optional; empty for the desktop local runtime
    base_url: str = "http://127.0.0.1:11500/v1"  # Remedy PDF Desktop local Ollama runtime
    cluster_nodes: tuple[str, ...] = ()      # Additional Ollama node URLs
    vision_base_url: str = ""                # Dedicated LM Studio vision endpoint
    vision_cluster_nodes: tuple[str, ...] = ()  # Additional LM Studio node URLs
    vision_model: str = "qwen3.5:4b"         # configurable vision model
    text_model: str = "qwen3.5:4b"           # configurable text model
    max_concurrent_calls: int = 5
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    llm_backend: str = "ollama"              # Primary LLM backend
    liteparse_enabled: bool = False          # Use LiteParse for local text snapshots / triage
    liteparse_bin: str = "lit"               # LiteParse CLI binary
    liteparse_timeout_seconds: float = 30.0  # Hard timeout for LiteParse snapshot calls
    liteparse_sample_pages: int = 3          # Number of leading pages to sample for routing
    liteparse_text_rich_min_chars: int = 800 # >= means native-text-rich
    liteparse_sparse_max_chars: int = 200    # <= means sparse/scanned
    escalation_backend: str = "ollama"       # Backend for Tier 2 escalation
    escalation_base_url: str = ""            # Dedicated endpoint for Tier 2 escalation
    escalation_model: str = "qwen3-vl:235b-instruct-cloud"  # Model for Tier 2 escalation
    ollama_stream: bool = False              # Disable streaming for Ollama API calls
    ollama_reasoning_effort: str = "none"    # Reasoning effort for Ollama compat endpoint (none/low/medium/high)


@dataclass(frozen=True)
class ProcessingConfig:
    """Settings that govern document processing behaviour."""

    html_workflow_enabled: bool = False
    max_concurrent_calls: int = 5
    max_retries: int = 3
    retry_backoff_base: float = 2.0


@dataclass(frozen=True)
class OutputConfig:
    """Paths for pipeline output artefacts."""

    output_dir: Path = Path("./output")
    log_dir: Path = Path("./logs")
    db_path: Path = Path("./pipeline.db")


@dataclass(frozen=True)
class PDFRemediationConfig:
    """Settings for PDF-to-PDF remediation path."""

    enabled: bool = False
    output_format: str = "pdf"  # "pdf", "html", or "both"
    stirling_url: str = "http://localhost:8080"
    stirling_api_key: str = ""
    verapdf_path: str = "/usr/local/bin/verapdf"
    use_programmatic_fixes: bool = True  # pre-LLM strategy remediation
    ghostscript_enabled: bool = False
    ghostscript_path: str = ""  # auto-detect via shutil.which("gs") if empty
    redistill_visual_tolerance: float = 0.05
    specialist_coordinator_as_fallback: bool = False  # Experimental coordinator fallback after Tier 2
    vision_planner_as_fallback: bool = False  # Tier 3 fallback after Tier 2 escalation fails
    # Residual font-repair stages.
    font_mode_b_enabled: bool = False
    font_mode_b_trigger_rules: tuple[str, ...] = (
        "7.21.4.1-1", "7.21.4.2-2", "7.21.7-1",
    )
    font_mode_b_use_checker_signals: bool = True
    simple_font_replacement_enabled: bool = False
    simple_font_replacement_trigger_rules: tuple[str, ...] = ("7.21.4.1-1",)
    simple_font_encoding_repair_enabled: bool = False
    # Structural rebuild fallback stage.
    font_mode_a_enabled: bool = False
    font_mode_a_trigger_rules: tuple[str, ...] = (
        "7.1-1", "7.1-2", "7.1-3",
        "7.2-11", "7.2-14", "7.2-42", "7.2-43",
    )
    font_mode_a_visual_diff_threshold: float = 0.10


@dataclass(frozen=True)
class ContrastConfig:
    """Settings for PDF color contrast remediation."""

    enabled: bool = True
    level: str = "AA"              # "AA" or "AAA"
    max_iterations: int = 3        # Max fix-validate loops per page
    dpi: int = 150                 # Page rendering resolution
    auto_fix: bool = True          # Apply programmatic fixes (not just detect)


@dataclass(frozen=True)
class ValidationConfig:
    """Validation / remediation loop settings."""

    max_remediation_cycles: int = 3
    fail_on_serious: bool = True
    wave_api_key: str = ""
    wave_report_type: int = 3


@dataclass
class PipelineConfig:
    """Top-level configuration container aggregating all sub-configs."""

    crawl: CrawlConfig = field(default_factory=CrawlConfig)
    api: APIConfig = field(default_factory=APIConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    pdf_remediation: PDFRemediationConfig = field(default_factory=PDFRemediationConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    campuses: list[CampusConfig] = field(default_factory=list)
    drupal_sites: dict = field(default_factory=dict)  # code -> DrupalSiteConfig

    def get_campus(self, code: str) -> CampusConfig:
        """Return the ``CampusConfig`` matching *code* (case-insensitive).

        Raises
        ------
        ValueError
            If no campus with the given code exists.
        """
        code_upper = code.upper()
        for campus in self.campuses:
            if campus.code.upper() == code_upper:
                return campus
        raise ValueError(f"No campus configured with code {code!r}")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and return a YAML file as a dictionary, or empty dict on failure."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _env(key: str, default: str = "") -> str:
    """Return an environment variable value or *default*."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return int(raw)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return float(raw)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


def load_config(
    env_path: Path | None = None,
    yaml_path: Path | None = None,
) -> PipelineConfig:
    """Build a ``PipelineConfig`` by merging .env and YAML sources.

    Resolution order (last wins):
        1. Compiled defaults in the dataclasses above
        2. Values from *config.yaml*
        3. Environment variables (loaded from *.env* if present)

    Parameters
    ----------
    env_path:
        Explicit path to a ``.env`` file.  Falls back to ``.env`` in cwd.
    yaml_path:
        Explicit path to a YAML config file.  Falls back to ``config.yaml``
        in cwd.
    """
    # --- Load .env ---------------------------------------------------------
    dotenv_file = env_path or Path(".env")
    # Keep explicit process env vars authoritative; .env should only fill gaps.
    load_dotenv(dotenv_file, override=False)

    # --- Load YAML ---------------------------------------------------------
    yaml_file = yaml_path or Path("config.yaml")
    yml: dict[str, Any] = _load_yaml(yaml_file)

    crawl_yml = yml.get("crawl", {})
    api_yml = yml.get("api", {})
    processing_yml = yml.get("processing", {})
    output_yml = yml.get("output", {})
    validation_yml = yml.get("validation", {})
    pdf_rem_yml = yml.get("pdf_remediation", {})

    # --- Build sub-configs (env overrides yaml overrides defaults) ----------

    crawl = CrawlConfig(
        start_url=_env("CRAWL_START_URL", crawl_yml.get("start_url", "")),
        max_depth=_env_int("CRAWL_MAX_DEPTH", crawl_yml.get("max_depth", 10)),
        max_pages=_env_int("CRAWL_MAX_PAGES", crawl_yml.get("max_pages", 10_000)),
        rate_limit=_env_float("CRAWL_RATE_LIMIT", crawl_yml.get("rate_limit", 2.0)),
        firecrawl_api_key=_env(
            "FIRECRAWL_API_KEY", crawl_yml.get("firecrawl_api_key", "")
        ),
        crawl_engine=_env(
            "CRAWL_ENGINE", crawl_yml.get("crawl_engine", "firecrawl")
        ),
    )

    api = APIConfig(
        api_key=_env("OLLAMA_API_KEY", api_yml.get("api_key", "")),
        base_url=_env("OLLAMA_BASE_URL", api_yml.get("base_url", "http://127.0.0.1:11500/v1")),
        cluster_nodes=tuple(
            _env("OLLAMA_CLUSTER_NODES", "").split(",")
            if _env("OLLAMA_CLUSTER_NODES")
            else api_yml.get("cluster_nodes", [])
        ),
        vision_base_url=_env(
            "VISION_BASE_URL",
            api_yml.get("vision_base_url", ""),
        ),
        vision_cluster_nodes=tuple(
            _env("VISION_CLUSTER_NODES", "").split(",")
            if _env("VISION_CLUSTER_NODES")
            else api_yml.get("vision_cluster_nodes", [])
        ),
        vision_model=_env(
            "OLLAMA_VISION_MODEL",
            api_yml.get("vision_model", "qwen3.5:4b"),
        ),
        text_model=_env(
            "OLLAMA_TEXT_MODEL",
            api_yml.get("text_model", "qwen3.5:4b"),
        ),
        max_concurrent_calls=_env_int(
            "MAX_CONCURRENT_API_CALLS",
            api_yml.get("max_concurrent_calls", 5),
        ),
        max_retries=_env_int(
            "MAX_RETRIES",
            api_yml.get("max_retries", 3),
        ),
        retry_backoff_base=_env_float(
            "RETRY_BACKOFF_BASE",
            api_yml.get("retry_backoff_base", 2.0),
        ),
        llm_backend=_env(
            "LLM_BACKEND",
            api_yml.get("llm_backend", "ollama"),
        ),
        liteparse_enabled=_env_bool(
            "LITEPARSE_ENABLED",
            api_yml.get("liteparse_enabled", False),
        ),
        liteparse_bin=_env(
            "LITEPARSE_BIN",
            api_yml.get("liteparse_bin", "lit"),
        ),
        liteparse_timeout_seconds=_env_float(
            "LITEPARSE_TIMEOUT_SECONDS",
            api_yml.get("liteparse_timeout_seconds", 30.0),
        ),
        liteparse_sample_pages=_env_int(
            "LITEPARSE_SAMPLE_PAGES",
            api_yml.get("liteparse_sample_pages", 3),
        ),
        liteparse_text_rich_min_chars=_env_int(
            "LITEPARSE_TEXT_RICH_MIN_CHARS",
            api_yml.get("liteparse_text_rich_min_chars", 800),
        ),
        liteparse_sparse_max_chars=_env_int(
            "LITEPARSE_SPARSE_MAX_CHARS",
            api_yml.get("liteparse_sparse_max_chars", 200),
        ),
        escalation_backend=_env(
            "ESCALATION_BACKEND",
            api_yml.get("escalation_backend", "ollama"),
        ),
        escalation_base_url=_env(
            "ESCALATION_BASE_URL",
            api_yml.get("escalation_base_url", ""),
        ),
        escalation_model=_env(
            "ESCALATION_MODEL",
            api_yml.get("escalation_model", "qwen3-vl:235b-instruct-cloud"),
        ),
        ollama_stream=_env_bool(
            "OLLAMA_STREAM",
            api_yml.get("ollama_stream", False),
        ),
        ollama_reasoning_effort=_env(
            "OLLAMA_REASONING_EFFORT",
            api_yml.get("ollama_reasoning_effort", "low"),
        ),
    )

    processing = ProcessingConfig(
        html_workflow_enabled=_env_bool(
            "HTML_WORKFLOW_ENABLED",
            processing_yml.get("html_workflow_enabled", False),
        ),
        max_concurrent_calls=_env_int(
            "MAX_CONCURRENT_API_CALLS",
            processing_yml.get("max_concurrent_calls", 5),
        ),
        max_retries=_env_int(
            "MAX_RETRIES",
            processing_yml.get("max_retries", 3),
        ),
        retry_backoff_base=_env_float(
            "RETRY_BACKOFF_BASE",
            processing_yml.get("retry_backoff_base", 2.0),
        ),
    )

    output = OutputConfig(
        output_dir=Path(
            _env("OUTPUT_DIR", str(output_yml.get("output_dir", "./output")))
        ),
        log_dir=Path(_env("LOG_DIR", str(output_yml.get("log_dir", "./logs")))),
        db_path=Path(
            _env("DB_PATH", str(output_yml.get("db_path", "./pipeline.db")))
        ),
    )

    validation = ValidationConfig(
        max_remediation_cycles=_env_int(
            "VALIDATION_MAX_REMEDIATION_CYCLES",
            validation_yml.get("max_remediation_cycles", 3),
        ),
        fail_on_serious=_env_bool(
            "VALIDATION_FAIL_ON_SERIOUS",
            validation_yml.get("fail_on_serious", True),
        ),
        wave_api_key=_env(
            "WAVE_API_KEY",
            validation_yml.get("wave_api_key", ""),
        ),
        wave_report_type=_env_int(
            "WAVE_REPORT_TYPE",
            validation_yml.get("wave_report_type", 3),
        ),
    )

    pdf_remediation = PDFRemediationConfig(
        enabled=_env_bool(
            "PDF_REMEDIATION_ENABLED",
            pdf_rem_yml.get("enabled", False),
        ),
        output_format=_env(
            "PDF_REMEDIATION_OUTPUT_FORMAT",
            pdf_rem_yml.get("output_format", "both"),
        ),
        stirling_url=_env(
            "STIRLING_URL",
            pdf_rem_yml.get("stirling_url", "http://localhost:8080"),
        ),
        stirling_api_key=_env(
            "STIRLING_API_KEY",
            pdf_rem_yml.get("stirling_api_key", ""),
        ),
        verapdf_path=_env(
            "VERAPDF_PATH",
            pdf_rem_yml.get("verapdf_path", "/usr/local/bin/verapdf"),
        ),
        use_programmatic_fixes=_env_bool(
            "PDF_REMEDIATION_USE_PROGRAMMATIC_FIXES",
            pdf_rem_yml.get("use_programmatic_fixes", True),
        ),
        ghostscript_enabled=_env_bool(
            "GHOSTSCRIPT_ENABLED",
            pdf_rem_yml.get("ghostscript_enabled", False),
        ),
        ghostscript_path=_env(
            "GHOSTSCRIPT_PATH",
            pdf_rem_yml.get("ghostscript_path", ""),
        ),
        redistill_visual_tolerance=_env_float(
            "REDISTILL_VISUAL_TOLERANCE",
            pdf_rem_yml.get("redistill_visual_tolerance", 0.05),
        ),
        specialist_coordinator_as_fallback=_env_bool(
            "SPECIALIST_COORDINATOR_AS_FALLBACK",
            pdf_rem_yml.get("specialist_coordinator_as_fallback", False),
        ),
        vision_planner_as_fallback=_env_bool(
            "VISION_PLANNER_AS_FALLBACK",
            pdf_rem_yml.get("vision_planner_as_fallback", False),
        ),
        font_mode_b_enabled=_env_bool(
            "FONT_MODE_B_ENABLED",
            pdf_rem_yml.get("font_mode_b_enabled", False),
        ),
        font_mode_b_trigger_rules=tuple(
            _env("FONT_MODE_B_TRIGGER_RULES", "").split(",")
            if _env("FONT_MODE_B_TRIGGER_RULES")
            else pdf_rem_yml.get(
                "font_mode_b_trigger_rules",
                ("7.21.4.1-1", "7.21.4.2-2", "7.21.7-1"),
            )
        ),
        font_mode_b_use_checker_signals=_env_bool(
            "FONT_MODE_B_USE_CHECKER_SIGNALS",
            pdf_rem_yml.get("font_mode_b_use_checker_signals", True),
        ),
        simple_font_replacement_enabled=_env_bool(
            "SIMPLE_FONT_REPLACEMENT_ENABLED",
            pdf_rem_yml.get("simple_font_replacement_enabled", False),
        ),
        simple_font_replacement_trigger_rules=tuple(
            _env("SIMPLE_FONT_REPLACEMENT_TRIGGER_RULES", "").split(",")
            if _env("SIMPLE_FONT_REPLACEMENT_TRIGGER_RULES")
            else pdf_rem_yml.get(
                "simple_font_replacement_trigger_rules",
                ("7.21.4.1-1",),
            )
        ),
        simple_font_encoding_repair_enabled=_env_bool(
            "SIMPLE_FONT_ENCODING_REPAIR_ENABLED",
            pdf_rem_yml.get("simple_font_encoding_repair_enabled", False),
        ),
        font_mode_a_enabled=_env_bool(
            "FONT_MODE_A_ENABLED",
            pdf_rem_yml.get("font_mode_a_enabled", False),
        ),
        font_mode_a_trigger_rules=tuple(
            _env("FONT_MODE_A_TRIGGER_RULES", "").split(",")
            if _env("FONT_MODE_A_TRIGGER_RULES")
            else pdf_rem_yml.get(
                "font_mode_a_trigger_rules",
                (
                    "7.1-1", "7.1-2", "7.1-3",
                    "7.2-11", "7.2-14", "7.2-42", "7.2-43",
                ),
            )
        ),
        font_mode_a_visual_diff_threshold=_env_float(
            "FONT_MODE_A_VISUAL_DIFF_THRESHOLD",
            pdf_rem_yml.get("font_mode_a_visual_diff_threshold", 0.10),
        ),
    )

    contrast_yml = yml.get("contrast", {})
    contrast = ContrastConfig(
        enabled=_env_bool(
            "CONTRAST_ENABLED",
            contrast_yml.get("enabled", True),
        ),
        level=_env(
            "CONTRAST_LEVEL",
            contrast_yml.get("level", "AA"),
        ),
        max_iterations=_env_int(
            "CONTRAST_MAX_ITERATIONS",
            contrast_yml.get("max_iterations", 3),
        ),
        dpi=_env_int(
            "CONTRAST_DPI",
            contrast_yml.get("dpi", 150),
        ),
        auto_fix=_env_bool(
            "CONTRAST_AUTO_FIX",
            contrast_yml.get("auto_fix", True),
        ),
    )

    # --- Build campus list ---------------------------------------------------
    campuses: list[CampusConfig] = []
    drupal_sites: dict = {}

    for entry in yml.get("campuses", []):
        if isinstance(entry, dict):
            campuses.append(
                CampusConfig(
                    code=str(entry.get("code", "")),
                    name=str(entry.get("name", "")),
                    start_url=str(entry.get("start_url", "")),
                    brand_primary=str(entry.get("brand_primary", "")),
                    brand_accent=str(entry.get("brand_accent", "")),
                    brand_neutral=str(entry.get("brand_neutral", "")),
                )
            )

            # Parse optional drupal block for this campus
            drupal_yml = entry.get("drupal")
            if isinstance(drupal_yml, dict):
                code = str(entry.get("code", "")).upper()
                try:
                    from project_remedy.mcp_server.config import build_site_config
                except ModuleNotFoundError:
                    # The Drupal MCP package is optional in the current repo state.
                    # Keep the raw config so non-Drupal workflows can still load.
                    drupal_sites[code] = dict(drupal_yml)
                else:
                    drupal_sites[code] = build_site_config(code, drupal_yml)

    return PipelineConfig(
        crawl=crawl,
        api=api,
        processing=processing,
        output=output,
        validation=validation,
        pdf_remediation=pdf_remediation,
        contrast=contrast,
        campuses=campuses,
        drupal_sites=drupal_sites,
    )
