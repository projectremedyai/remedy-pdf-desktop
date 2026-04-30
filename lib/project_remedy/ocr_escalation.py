"""Specialized OCR escalation policy and benchmark scaffolding."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from project_remedy.liteparse_adapter import liteparse_available, liteparse_text_snapshot
from project_remedy.vision_prompts import ocr_markdown_prompt

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass
class OCRBlock:
    text: str
    bbox: tuple[float, float, float, float] | None = None
    role: str = ""
    confidence: float | None = None


@dataclass
class OCRPageResult:
    provider: str
    page_number: int
    markdown: str
    blocks: list[OCRBlock] = field(default_factory=list)
    confidence: float | None = None


@dataclass
class OCRBenchmarkSample:
    pdf_path: Path
    page_number: int
    label: str = ""
    layout_class: str = ""


@dataclass
class OCRBenchmarkResult:
    provider: str
    sample_count: int
    mean_block_count: float
    mean_markdown_length: float
    mean_token_overlap: float = 0.0
    mean_heading_count: float = 0.0
    mean_table_markers: float = 0.0
    empty_output_count: int = 0
    failures: list[str] = field(default_factory=list)


@dataclass
class OCREscalationSignal:
    layout_class: str
    visual_block_count: int
    structured_text_nodes: int
    image_coverage: float = 0.0
    has_small_text: bool = False
    requires_rebuild: bool = False
    structure_warning: bool = False


class OCRAdapter(Protocol):
    name: str

    async def extract_page(self, pdf_path: Path, page_number: int) -> OCRPageResult:
        ...


class CommandLineOCRAdapter:
    """Thin wrapper for optional external OCR tools.

    Commands are intentionally opt-in via environment variables so the default
    remediation path remains stable and dependency-light.
    """

    name = "external-ocr"

    def __init__(self, command: str) -> None:
        self._command = command

    async def extract_page(self, pdf_path: Path, page_number: int) -> OCRPageResult:
        args = shlex.split(self._command) + [str(pdf_path), str(page_number)]
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip() or self.name
            raise RuntimeError(message)
        markdown = stdout.decode("utf-8", errors="replace").strip()
        return OCRPageResult(provider=self.name, page_number=page_number, markdown=markdown)


class GLMOCRAdapter(CommandLineOCRAdapter):
    name = "glm-ocr"


class DeepSeekOCRAdapter(CommandLineOCRAdapter):
    name = "deepseek-ocr"


class OllamaModelOCRAdapter:
    """OCR adapter backed by a local Ollama multimodal model."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = _DEFAULT_OLLAMA_BASE_URL,
        api_key: str = "ollama",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.model_name = model_name
        self.name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    async def extract_page(self, pdf_path: Path, page_number: int) -> OCRPageResult:
        image_b64 = _render_pdf_page_to_base64_png(pdf_path, page_number)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": ocr_markdown_prompt(
                                profile="local",
                                page_hint=f"Page {page_number}",
                                native_pdf=False,
                            ),
                        },
                    ],
                }
            ],
            "max_tokens": 8192,
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._timeout_seconds, connect=30.0),
        ) as client:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

        markdown = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return OCRPageResult(
            provider=self.name,
            page_number=page_number,
            markdown=markdown,
            blocks=_markdown_to_blocks(markdown),
        )


class LiteParseNativeTextAdapter:
    """Fast local native-text snapshot adapter backed by LiteParse."""

    name = "liteparse"

    async def extract_page(self, pdf_path: Path, page_number: int) -> OCRPageResult:
        snapshot = await asyncio.to_thread(
            liteparse_text_snapshot,
            pdf_path,
            page_spec=str(page_number),
            no_ocr=True,
        )
        if not snapshot.used:
            raise RuntimeError("liteparse unavailable")
        if snapshot.timed_out:
            raise RuntimeError(snapshot.parser_error or "liteparse timed out")
        if snapshot.parser_error:
            raise RuntimeError(snapshot.parser_error)
        markdown = snapshot.text.strip()
        if len(markdown) < 10:
            raise RuntimeError("liteparse sparse output")
        return OCRPageResult(
            provider=self.name,
            page_number=page_number,
            markdown=markdown,
            blocks=_markdown_to_blocks(markdown),
        )


class TesseractOCRAdapter:
    """Baseline OCR adapter using local Tesseract if available."""

    name = "tesseract"

    def __init__(self, *, language: str = "eng") -> None:
        self._language = language

    async def extract_page(self, pdf_path: Path, page_number: int) -> OCRPageResult:
        tesseract = shutil.which("tesseract")
        if tesseract is None:
            raise RuntimeError("tesseract not found")

        image_bytes = _render_pdf_page_to_png_bytes(pdf_path, page_number)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
            temp_path = Path(temp_image.name)
            temp_image.write(image_bytes)

        try:
            proc = await asyncio.create_subprocess_exec(
                tesseract,
                str(temp_path),
                "stdout",
                "-l",
                self._language,
                "--psm",
                "6",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                message = stderr.decode("utf-8", errors="replace").strip() or self.name
                raise RuntimeError(message)
        finally:
            temp_path.unlink(missing_ok=True)

        markdown = stdout.decode("utf-8", errors="replace").strip()
        return OCRPageResult(
            provider=self.name,
            page_number=page_number,
            markdown=markdown,
            blocks=_markdown_to_blocks(markdown),
        )


def installed_ollama_models() -> set[str]:
    """Return model names reported by `ollama list`."""
    ollama = shutil.which("ollama")
    if ollama is None:
        return set()
    proc = subprocess.run(
        [ollama, "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return set()

    models: set[str] = set()
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.add(parts[0].strip())
    return models


def local_benchmark_adapters() -> list[OCRAdapter]:
    """Return runnable local OCR adapters in benchmark priority order."""
    adapters: list[OCRAdapter] = []
    if liteparse_available():
        adapters.append(LiteParseNativeTextAdapter())
    models = installed_ollama_models()
    for model in ("glm-ocr:latest", "deepseek-ocr:latest", "qwen3-vl:latest"):
        if model in models:
            adapters.append(OllamaModelOCRAdapter(model))
    if shutil.which("tesseract"):
        adapters.append(TesseractOCRAdapter())
    return adapters


def available_specialized_ocr_adapters() -> list[OCRAdapter]:
    adapters: list[OCRAdapter] = []
    glm_cmd = os.getenv("PROJECT_REMEDY_GLM_OCR_CMD", "").strip()
    deepseek_cmd = os.getenv("PROJECT_REMEDY_DEEPSEEK_OCR_CMD", "").strip()
    if glm_cmd:
        adapters.append(GLMOCRAdapter(glm_cmd))
    if deepseek_cmd:
        adapters.append(DeepSeekOCRAdapter(deepseek_cmd))
    if not adapters:
        for adapter in local_benchmark_adapters():
            if adapter.name in {"glm-ocr:latest", "deepseek-ocr:latest"}:
                adapters.append(adapter)
    return adapters


def should_escalate_specialized_ocr(signal: OCREscalationSignal) -> bool:
    complex_layout = signal.layout_class in {
        "brochure_sidebar",
        "schedule_grid",
        "mixed_graphic_flyer",
        "map_infographic",
        "unknown_complex",
        "hero_cover",
        "report_cover",
    }
    if signal.requires_rebuild or signal.structure_warning:
        return True
    if signal.layout_class in {"single_column", "form_checklist", "table_directory"}:
        return False
    if signal.visual_block_count >= 8 and signal.structured_text_nodes <= 2:
        return True
    if complex_layout and signal.visual_block_count >= 6 and signal.structured_text_nodes <= 3:
        return True
    if signal.image_coverage >= 0.35 and signal.has_small_text:
        return True
    return False


async def benchmark_ocr_adapters(
    samples: list[OCRBenchmarkSample],
    adapters: list[OCRAdapter],
) -> list[OCRBenchmarkResult]:
    results: list[OCRBenchmarkResult] = []
    for adapter in adapters:
        page_results: list[tuple[OCRBenchmarkSample, OCRPageResult]] = []
        failures: list[str] = []
        empty_outputs = 0
        for sample in samples:
            try:
                page_result = await adapter.extract_page(sample.pdf_path, sample.page_number)
                if not page_result.markdown.strip():
                    empty_outputs += 1
                    failures.append(
                        f"{sample.pdf_path.name} p{sample.page_number}: empty output"
                    )
                    continue
                page_results.append((sample, page_result))
            except Exception as exc:
                failures.append(f"{sample.pdf_path.name} p{sample.page_number}: {exc}")
        if page_results:
            mean_blocks = sum(len(r.blocks) for _s, r in page_results) / len(page_results)
            mean_length = sum(len(r.markdown) for _s, r in page_results) / len(page_results)
            overlaps = [
                _token_overlap_score(
                    _extract_reference_text(sample.pdf_path, sample.page_number),
                    result.markdown,
                )
                for sample, result in page_results
            ]
            heading_counts = [_markdown_heading_count(r.markdown) for _s, r in page_results]
            table_markers = [_markdown_table_marker_count(r.markdown) for _s, r in page_results]
        else:
            mean_blocks = 0.0
            mean_length = 0.0
            overlaps = []
            heading_counts = []
            table_markers = []
        results.append(
            OCRBenchmarkResult(
                provider=adapter.name,
                sample_count=len(page_results),
                mean_block_count=mean_blocks,
                mean_markdown_length=mean_length,
                mean_token_overlap=(sum(overlaps) / len(overlaps)) if overlaps else 0.0,
                mean_heading_count=(sum(heading_counts) / len(heading_counts)) if heading_counts else 0.0,
                mean_table_markers=(sum(table_markers) / len(table_markers)) if table_markers else 0.0,
                empty_output_count=empty_outputs,
                failures=failures,
            )
        )
    return results


def _render_pdf_page_to_png_bytes(pdf_path: Path, page_number: int, *, dpi: int = 200) -> bytes:
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def _render_pdf_page_to_base64_png(pdf_path: Path, page_number: int, *, dpi: int = 200) -> str:
    return base64.b64encode(_render_pdf_page_to_png_bytes(pdf_path, page_number, dpi=dpi)).decode()


def _markdown_to_blocks(markdown: str) -> list[OCRBlock]:
    blocks: list[OCRBlock] = []
    for chunk in re.split(r"\n\s*\n", markdown.strip()):
        text = chunk.strip()
        if not text:
            continue
        role = "heading" if text.startswith("#") else "paragraph"
        if "|" in text and "\n" in text:
            role = "table"
        elif re.match(r"^[-*]\s+", text):
            role = "list"
        blocks.append(OCRBlock(text=text, role=role))
    return blocks


def _normalize_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) >= 2
    }


def _token_overlap_score(reference_text: str, candidate_text: str) -> float:
    reference = _normalize_tokens(reference_text)
    if not reference:
        return 0.0
    candidate = _normalize_tokens(candidate_text)
    if not candidate:
        return 0.0
    return len(reference & candidate) / len(reference)


def _markdown_heading_count(markdown: str) -> int:
    return sum(1 for line in markdown.splitlines() if line.lstrip().startswith("#"))


def _markdown_table_marker_count(markdown: str) -> int:
    return sum(1 for line in markdown.splitlines() if "|" in line)


def _extract_reference_text(pdf_path: Path, page_number: int) -> str:
    try:
        import fitz
    except Exception:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return ""
    try:
        return doc[page_number - 1].get_text("text")
    except Exception:
        return ""
    finally:
        doc.close()
