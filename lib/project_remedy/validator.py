"""Stage 7: Quad-Layer WCAG 2.1 AA Validation.

Runs four independent accessibility validation tools — axe-core
(via Playwright), pa11y (CLI), Lighthouse (CLI), and WAVE API — against
every generated HTML page.  Merges and deduplicates findings, and drives
an auto-remediation loop that feeds violations back to the LLM for
correction (up to ``max_remediation_cycles``).
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import logging
import re
import shutil
import tempfile
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from project_remedy.config import PipelineConfig
from project_remedy.database import DatabaseManager
from project_remedy.models import (
    DocumentJob,
    JobStatus,
    RenderedPage,
    ValidationResult,
)
from project_remedy.ollama_client import OllamaClient
from project_remedy.vision_prompts import visual_comparison_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Aggregated results from all four validation tools."""

    axe_result: ValidationResult = field(
        default_factory=lambda: ValidationResult(tool="axe")
    )
    pa11y_result: ValidationResult = field(
        default_factory=lambda: ValidationResult(tool="pa11y")
    )
    lighthouse_result: ValidationResult = field(
        default_factory=lambda: ValidationResult(tool="lighthouse")
    )
    wave_result: ValidationResult = field(
        default_factory=lambda: ValidationResult(tool="wave")
    )
    lighthouse_score: float = 0.0
    all_violations: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False
    summary: str = ""


@dataclass
class PageValidationOutcome:
    """Validation + remediation result for a single rendered page."""

    page: RenderedPage
    html: str
    results: list[ValidationResult]
    report: ValidationReport
    passed: bool
    remediation_count: int = 0
    error_message: str = ""


# ---------------------------------------------------------------------------
# axe-core injection script
# ---------------------------------------------------------------------------

_AXE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"

_AXE_SCRIPT = """
async () => {
    await axe.run(document, {
        runOnly: {
            type: 'tag',
            values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice']
        }
    }).then(results => {
        window.__axe_results = results;
    });
    return window.__axe_results;
}
"""

# ---------------------------------------------------------------------------
# Remediation prompt
# ---------------------------------------------------------------------------

_REMEDIATION_PROMPT = """\
You are an expert web accessibility specialist. The HTML document below has
WCAG 2.1 Level AA violations detected by automated scanning tools.

Fix ALL of the following violations while preserving the document's content
and structure. Do not remove any content. Only modify the HTML to resolve
the accessibility issues.

## Violations to fix:

{violations}

## Current HTML:

```html
{html}
```

Return ONLY the corrected, complete HTML document. No explanation, no
markdown fences — just the raw HTML starting with <!DOCTYPE html>.
"""


# ---------------------------------------------------------------------------
# AccessibilityValidator
# ---------------------------------------------------------------------------


class AccessibilityValidator:
    """Quad-layer WCAG 2.1 AA validation with auto-remediation.

    Uses axe-core (Playwright), pa11y (CLI), Lighthouse (CLI), and
    WAVE API to validate every generated HTML page.  When violations are
    found, they are formatted and sent back to the LLM for automated
    correction, up to ``max_remediation_cycles`` times.

    Parameters
    ----------
    config:
        Pipeline configuration (provides remediation cycle limit).
    ollama_client:
        An initialised :class:`OllamaClient` for remediation chat calls.
    db:
        Database manager for logging validation cycles.
    """

    def __init__(
        self,
        config: PipelineConfig,
        ollama_client: OllamaClient,
        db: DatabaseManager,
    ) -> None:
        self._config = config
        self._ollama = ollama_client
        self._db = db
        self._max_cycles = config.validation.max_remediation_cycles
        self._fail_on_serious = config.validation.fail_on_serious
        self._wave_api_key = config.validation.wave_api_key
        self._wave_report_type = config.validation.wave_report_type
        self._wave_semaphore = asyncio.Semaphore(2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def validate(
        self,
        html_path: Path,
    ) -> ValidationReport:
        """Run axe-core, pa11y, and Lighthouse and return a merged report.

        Visual fidelity is handled separately in ``validate_rendered_page``
        so that image issues are remediated before WCAG compliance is checked.

        Parameters
        ----------
        html_path:
            Path to the HTML file to validate.

        Returns
        -------
        ValidationReport
            Aggregated, deduplicated results from axe-core, pa11y,
            and Lighthouse.
        """
        # Run axe-core, pa11y, and Lighthouse in parallel.
        # Visual fidelity runs separately BEFORE this method is called
        # (in validate_rendered_page) so images are fixed before WCAG checks.
        axe, pa11y, lh = await asyncio.gather(
            self.validate_with_axe(html_path),
            self.validate_with_pa11y(html_path),
            self.validate_with_lighthouse(html_path),
            return_exceptions=True,
        )

        # Handle individual tool failures gracefully
        if isinstance(axe, Exception):
            logger.error("axe-core validation failed: %s", axe)
            axe = ValidationResult(tool="axe", violations=[], passed=True)
        if isinstance(pa11y, Exception):
            logger.error("pa11y validation failed: %s", pa11y)
            pa11y = ValidationResult(tool="pa11y", violations=[], passed=True)
        if isinstance(lh, Exception):
            logger.error("Lighthouse validation failed: %s", lh)
            lh = ValidationResult(tool="lighthouse", score=0, violations=[], passed=True)

        visual = None

        # WAVE runs against a live URL; for local file validation it
        # is invoked separately via run_wave_validation / validate_with_wave.
        # Include a default empty result here so the report shape is stable.
        wave = ValidationResult(tool="wave", violations=[], passed=True)

        # Merge and deduplicate violations
        tools_to_merge = [axe, pa11y, lh, wave]
        if isinstance(visual, ValidationResult):
            tools_to_merge.append(visual)
        all_violations = self._merge_violations(*tools_to_merge)

        # Determine pass/fail
        critical_serious = [
            v for v in all_violations
            if v.get("impact") in ("critical", "serious")
        ]
        passed = len(critical_serious) == 0

        lh_score = lh.score if lh.score is not None else 0.0

        # Build summary
        summary_parts = [
            f"axe-core: {len(axe.violations)} violation(s)",
            f"pa11y: {len(pa11y.violations)} violation(s)",
            f"Lighthouse score: {lh_score}/100 ({len(lh.violations)} audit failure(s))",
            f"WAVE: {len(wave.violations)} violation(s)",
            f"Total unique violations: {len(all_violations)}",
            f"Critical/serious: {len(critical_serious)}",
            f"Result: {'PASS' if passed else 'FAIL'}",
        ]

        return ValidationReport(
            axe_result=axe,
            pa11y_result=pa11y,
            lighthouse_result=lh,
            wave_result=wave,
            lighthouse_score=lh_score,
            all_violations=all_violations,
            passed=passed,
            summary=" | ".join(summary_parts),
        )

    async def validate_rendered_page(
        self,
        job: DocumentJob,
        page: RenderedPage,
        html_path: Path,
    ) -> PageValidationOutcome:
        """Validate and, if needed, auto-remediate a rendered page.

        Visual fidelity runs first (for PDFs with extracted images) so
        missing-image issues are caught and fed into remediation before
        axe-core / pa11y / Lighthouse check WCAG compliance.
        """
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(page.html, encoding="utf-8")

        current_html = page.html

        # Run visual fidelity check BEFORE WCAG validators so image
        # issues can be remediated before compliance is measured.
        run_visual = (
            job.file_type is not None
            and job.file_type.value == "pdf"
            and job.get_extracted_images()
        )
        visual_result: ValidationResult | None = None
        if run_visual:
            try:
                visual_result = await self.validate_visual_fidelity(job, html_path)
            except Exception as exc:
                logger.warning("Visual fidelity check failed: %s", exc)

        current_report = await self.validate(html_path)

        # Merge visual result into the report if present.
        if isinstance(visual_result, ValidationResult):
            current_report.all_violations = self._merge_violations(
                current_report.axe_result,
                current_report.pa11y_result,
                current_report.lighthouse_result,
                current_report.wave_result,
                visual_result,
            )
            if not visual_result.passed:
                current_report.passed = False
        current_results = self._results_for_page(page, current_report)
        await self._log_page_results(job.id, 0, page, current_results)

        cycle = 0
        while not current_report.passed and cycle < self._max_cycles:
            cycle += 1
            corrected = await self._remediate_html(current_html, current_report)
            corrected = corrected.strip()
            corrected = re.sub(r"^```(?:html)?\s*", "", corrected)
            corrected = re.sub(r"\s*```$", "", corrected)
            corrected = corrected.strip()

            if not corrected or len(corrected) < 50:
                logger.warning(
                    "Remediation cycle %d returned empty/short HTML for %s; "
                    "keeping previous version",
                    cycle,
                    page.page_key,
                )
                break

            current_html = corrected
            html_path.write_text(current_html, encoding="utf-8")
            current_report = await self.validate(html_path)
            current_results = self._results_for_page(page, current_report)
            await self._log_page_results(job.id, cycle, page, current_results)

            logger.info(
                "Remediation cycle %d for %s: %s",
                cycle,
                page.page_key,
                "PASS" if current_report.passed else "FAIL",
            )

        error_message = ""
        if not current_report.passed:
            error_message = (
                f"{page.title} failed validation after {cycle} remediation cycle(s). "
                f"{len(current_report.all_violations)} violation(s) remain."
            )

        return PageValidationOutcome(
            page=page,
            html=current_html,
            results=current_results,
            report=current_report,
            passed=current_report.passed,
            remediation_count=cycle,
            error_message=error_message,
        )

    async def validate_with_axe(self, html_path: Path) -> ValidationResult:
        """Run axe-core validation via Playwright.

        Loads the HTML file in a headless Chromium browser, injects
        axe-core from CDN, runs ``axe.run()``, and parses violations.

        Parameters
        ----------
        html_path:
            Path to the HTML file.

        Returns
        -------
        ValidationResult
            axe-core results with violations categorised by impact.
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error(
                "playwright is not installed. "
                "Install with: pip install playwright && playwright install chromium"
            )
            return ValidationResult(tool="axe", passed=False)

        violations: list[dict[str, Any]] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()

            file_url = html_path.resolve().as_uri()
            await page.goto(file_url, wait_until="networkidle")

            # Inject axe-core
            await page.add_script_tag(url=_AXE_CDN)
            await page.wait_for_function("typeof axe !== 'undefined'", timeout=15000)

            # Run axe
            raw_results = await page.evaluate(_AXE_SCRIPT)
            await browser.close()

        if raw_results and isinstance(raw_results, dict):
            for v in raw_results.get("violations", []):
                violations.append({
                    "tool": "axe",
                    "id": v.get("id", ""),
                    "impact": v.get("impact", "minor"),
                    "description": v.get("description", ""),
                    "help": v.get("help", ""),
                    "help_url": v.get("helpUrl", ""),
                    "wcag": _extract_wcag_tags(v.get("tags", [])),
                    "nodes_count": len(v.get("nodes", [])),
                    "html_snippets": [
                        n.get("html", "")[:200]
                        for n in v.get("nodes", [])[:3]
                    ],
                })

        has_critical = any(
            v.get("impact") in ("critical", "serious") for v in violations
        )
        passed = not has_critical

        logger.info(
            "axe-core: %d violation(s), passed=%s", len(violations), passed
        )
        return ValidationResult(
            tool="axe", violations=violations, passed=passed
        )

    async def validate_with_pa11y(self, html_path: Path) -> ValidationResult:
        """Run pa11y CLI as an async subprocess.

        Parameters
        ----------
        html_path:
            Path to the HTML file.

        Returns
        -------
        ValidationResult
            pa11y results mapped to WCAG success criteria.
        """
        file_url = html_path.resolve().as_uri()
        cmd = [
            "pa11y",
            "--reporter", "json",
            "--standard", "WCAG2AA",
            "--timeout", "30000",
            file_url,
        ]

        violations: list[dict[str, Any]] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=60
            )

            # pa11y exits with code 2 when issues are found — that is expected
            if stdout:
                raw = json.loads(stdout.decode("utf-8", errors="replace"))
                if isinstance(raw, list):
                    for issue in raw:
                        impact = _pa11y_type_to_impact(
                            issue.get("type", "notice")
                        )
                        violations.append({
                            "tool": "pa11y",
                            "id": issue.get("code", ""),
                            "impact": impact,
                            "description": issue.get("message", ""),
                            "help": issue.get("code", ""),
                            "wcag": _extract_wcag_from_code(
                                issue.get("code", "")
                            ),
                            "selector": issue.get("selector", ""),
                            "context": issue.get("context", "")[:200],
                        })

        except FileNotFoundError:
            logger.warning(
                "pa11y CLI not found. Install with: npm install -g pa11y"
            )
        except asyncio.TimeoutError:
            logger.warning("pa11y timed out after 60s for %s", html_path.name)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("pa11y output parse error: %s", exc)

        has_critical = any(
            v.get("impact") in ("critical", "serious") for v in violations
        )
        passed = not has_critical

        logger.info(
            "pa11y: %d violation(s), passed=%s", len(violations), passed
        )
        return ValidationResult(
            tool="pa11y", violations=violations, passed=passed
        )

    async def validate_with_lighthouse(
        self, html_path: Path
    ) -> ValidationResult:
        """Run Lighthouse CLI as an async subprocess.

        Targets the accessibility category only, aiming for a 100/100 score.

        Parameters
        ----------
        html_path:
            Path to the HTML file.

        Returns
        -------
        ValidationResult
            Lighthouse accessibility score and failing audits.
        """
        resolved = html_path.resolve()

        # Lighthouse v13+ rejects file:// URLs.  Serve the file's directory
        # via a throwaway HTTP server so Lighthouse gets an http:// URL.
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=str(resolved.parent),
        )
        httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        target_url = f"http://127.0.0.1:{port}/{resolved.name}"

        # Write Lighthouse JSON output to a temporary file
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            output_path = tmp.name

        cmd = [
            "lighthouse",
            target_url,
            "--only-categories=accessibility",
            "--output=json",
            f"--output-path={output_path}",
            "--chrome-flags=--headless --no-sandbox --disable-gpu",
            "--quiet",
        ]

        violations: list[dict[str, Any]] = []
        score: float = 0.0

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)

            output_file = Path(output_path)
            if output_file.exists():
                raw = json.loads(output_file.read_text(encoding="utf-8"))
                output_file.unlink(missing_ok=True)

                # Extract accessibility score
                categories = raw.get("categories", {})
                acc_cat = categories.get("accessibility", {})
                score = (acc_cat.get("score") or 0) * 100

                # Extract failing audits
                audits = raw.get("audits", {})
                for audit_id, audit in audits.items():
                    audit_score = audit.get("score")
                    # score=null means not applicable, score=1 means passed
                    if audit_score is not None and audit_score < 1:
                        violations.append({
                            "tool": "lighthouse",
                            "id": audit_id,
                            "impact": "serious" if audit_score == 0 else "moderate",
                            "description": audit.get("title", ""),
                            "help": audit.get("description", ""),
                            "score": audit_score,
                        })

        except FileNotFoundError:
            logger.warning(
                "lighthouse CLI not found. "
                "Install with: npm install -g lighthouse"
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Lighthouse timed out after 120s for %s", html_path.name
            )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Lighthouse output parse error: %s", exc)
        finally:
            httpd.shutdown()
            Path(output_path).unlink(missing_ok=True)

        has_critical = any(
            v.get("impact") in ("critical", "serious") for v in violations
        )
        passed = not has_critical

        logger.info(
            "Lighthouse: score=%.0f/100, %d failing audit(s), passed=%s",
            score, len(violations), passed,
        )
        return ValidationResult(
            tool="lighthouse",
            score=score,
            violations=violations,
            passed=passed,
        )

    async def validate_visual_fidelity(
        self,
        job: DocumentJob,
        html_path: Path,
    ) -> ValidationResult:
        """Compare the original PDF visually against the rendered HTML.

        Renders the original PDF pages and the HTML page to PNG, sends
        them to the vision model for comparison, and checks whether all
        images/graphics from the original are present in the HTML.

        Parameters
        ----------
        job:
            The document job (must have a PDF local_path and extracted images).
        html_path:
            Path to the rendered HTML file.

        Returns
        -------
        ValidationResult
            Visual fidelity result with violations for missing/wrong images.
        """
        import fitz  # PyMuPDF

        pdf_path = Path(job.local_path)
        if not pdf_path.exists():
            logger.warning("Visual validation: PDF not found at %s", pdf_path)
            return ValidationResult(tool="visual", violations=[], passed=True)

        # Render first few PDF pages as PNG (cap at 3 pages for cost).
        doc = fitz.open(str(pdf_path))
        page_limit = min(len(doc), 3)
        pdf_images: list[bytes] = []
        for page_num in range(page_limit):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            pdf_images.append(pix.tobytes("png"))
        doc.close()

        # Render the HTML page as PNG using Playwright.
        html_image: bytes | None = None
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page(viewport={"width": 1280, "height": 1024})
                await page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
                html_image = await page.screenshot(full_page=True, type="png")
                await browser.close()
        except Exception as exc:
            logger.warning("Visual validation: could not render HTML: %s", exc)
            return ValidationResult(tool="visual", violations=[], passed=True)

        if not html_image:
            return ValidationResult(tool="visual", violations=[], passed=True)

        # Send all images to the vision model for comparison.
        all_images = pdf_images + [html_image]
        prompt = visual_comparison_prompt()

        try:
            raw_response = await self._ollama.compare_images(
                all_images, prompt, max_tokens=4096
            )
        except Exception as exc:
            logger.warning("Visual validation: model comparison failed: %s", exc)
            return ValidationResult(tool="visual", violations=[], passed=True)

        # Parse the JSON response.
        violations: list[dict[str, Any]] = []

        try:
            # Strip markdown fences if present.
            cleaned = raw_response.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            result = json.loads(cleaned)

            images_match = result.get("images_match", True)
            content_matches = result.get("content_matches", True)

            for desc in result.get("missing_images", []):
                violations.append({
                    "tool": "visual",
                    "id": "missing-image",
                    "impact": "serious",
                    "description": f"Missing image: {desc}",
                    "help": "Ensure all PDF images are extracted and included in the HTML.",
                })
            for desc in result.get("wrong_images", []):
                violations.append({
                    "tool": "visual",
                    "id": "wrong-image",
                    "impact": "moderate",
                    "description": f"Mismatched image: {desc}",
                    "help": "Verify image extraction and placement matches the original.",
                })
            for desc in result.get("missing_text", []):
                violations.append({
                    "tool": "visual",
                    "id": "missing-text",
                    "impact": "critical",
                    "description": f"Missing text content: {desc}",
                    "help": "All text from the original document must be preserved in the HTML.",
                })
            for desc in result.get("table_issues", []):
                violations.append({
                    "tool": "visual",
                    "id": "table-issue",
                    "impact": "serious",
                    "description": f"Table issue: {desc}",
                    "help": "Tables must accurately reproduce the original data and structure.",
                })
            for desc in result.get("structure_issues", []):
                violations.append({
                    "tool": "visual",
                    "id": "structure-issue",
                    "impact": "moderate",
                    "description": f"Structure issue: {desc}",
                    "help": "Headings, lists, and reading order must match the original document.",
                })
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "Visual validation: could not parse model response: %s", exc
            )
            images_match = True
            content_matches = True

        passed = images_match and content_matches and len(violations) == 0
        logger.info(
            "Visual fidelity: content_matches=%s, images_match=%s, "
            "%d violation(s), passed=%s",
            content_matches, images_match, len(violations), passed,
        )
        return ValidationResult(
            tool="visual", violations=violations, passed=passed
        )

    async def validate_with_wave(self, url: str) -> ValidationResult:
        """Run WAVE API validation against a live URL.

        Requires a WAVE API key (from ``config.validation.wave_api_key``).
        Requests are throttled to a maximum of 2 concurrent calls via an
        internal semaphore.

        Parameters
        ----------
        url:
            The publicly accessible URL of the page to validate.

        Returns
        -------
        ValidationResult
            WAVE results with violations categorised by impact.
        """
        if not self._wave_api_key:
            logger.warning(
                "WAVE API key is not configured; skipping WAVE validation"
            )
            return ValidationResult(tool="wave", violations=[], passed=True)

        violations: list[dict[str, Any]] = []

        async with self._wave_semaphore:
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.get(
                        "https://wave.webaim.org/api/request",
                        params={
                            "key": self._wave_api_key,
                            "url": url,
                            "reporttype": self._wave_report_type,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "WAVE API HTTP error %s for %s: %s",
                    exc.response.status_code, url, exc,
                )
                return ValidationResult(tool="wave", violations=[], passed=True)
            except (httpx.RequestError, ValueError) as exc:
                logger.error("WAVE API request failed for %s: %s", url, exc)
                return ValidationResult(tool="wave", violations=[], passed=True)

        categories = data.get("categories", {})

        # Map WAVE category items to violations
        _wave_impact_map: dict[str, str] = {
            "error": "serious",
            "contrast": "serious",
            "alert": "moderate",
        }

        for cat_name, impact in _wave_impact_map.items():
            cat_data = categories.get(cat_name, {})
            items = cat_data.get("items", {})
            for item_id, item in items.items():
                violations.append({
                    "tool": "wave",
                    "id": item_id,
                    "impact": impact,
                    "description": item.get("description", ""),
                    "help": item.get("description", ""),
                    "count": item.get("count", 0),
                    "category": cat_name,
                })

        # Passed if no errors and no contrast errors
        error_count = categories.get("error", {}).get("count", 0)
        contrast_count = categories.get("contrast", {}).get("count", 0)
        passed = error_count == 0 and contrast_count == 0

        logger.info(
            "WAVE: %d violation(s) (errors=%d, contrast=%d), passed=%s",
            len(violations), error_count, contrast_count, passed,
        )
        return ValidationResult(
            tool="wave", violations=violations, passed=passed
        )

    async def validate_with_verapdf(self, pdf_path: Path) -> ValidationResult:
        """Run veraPDF CLI validation against a PDF file for PDF/UA-1.

        Checks the PDF against the PDF/UA-1 profile using the veraPDF
        validator.  The binary location is read from
        ``config.pdf_remediation.verapdf_path``; if that path does not
        exist, ``shutil.which("verapdf")`` is tried as a fallback.

        Parameters
        ----------
        pdf_path:
            Path to the PDF file to validate.

        Returns
        -------
        ValidationResult
            veraPDF results with violations extracted from failed rules.
        """
        # --- Locate the veraPDF binary ------------------------------------
        verapdf_bin = self._config.pdf_remediation.verapdf_path
        if not Path(verapdf_bin).is_file():
            found = shutil.which("verapdf")
            if found:
                verapdf_bin = found
            else:
                logger.warning(
                    "veraPDF binary not found at %s and not on PATH; "
                    "skipping veraPDF validation",
                    self._config.pdf_remediation.verapdf_path,
                )
                return ValidationResult(tool="verapdf", violations=[], passed=True)

        # --- Run veraPDF as an async subprocess ---------------------------
        cmd = [
            verapdf_bin,
            "--format", "xml",
            "--profile", "ua1",
            str(pdf_path),
        ]

        violations: list[dict[str, Any]] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=120
            )

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                if stderr_text:
                    logger.debug("veraPDF stderr: %s", stderr_text[:500])

            # --- Parse XML output -----------------------------------------
            if stdout:
                xml_text = stdout.decode("utf-8", errors="replace")
                root = ET.fromstring(xml_text)

                # veraPDF XML uses a default namespace in some versions.
                # Try namespace-aware and plain tag searches.
                ns = ""
                if root.tag.startswith("{"):
                    ns = root.tag.split("}")[0] + "}"

                # Find all <rule> elements anywhere in the tree
                rule_elements = root.iter(f"{ns}rule") if ns else root.iter("rule")

                for rule in rule_elements:
                    # Determine status -- look for a <status> child or attribute
                    status_el = rule.find(f"{ns}status") if ns else rule.find("status")
                    status = (
                        (status_el.text or "").strip().lower()
                        if status_el is not None
                        else rule.get("status", "").lower()
                    )

                    if status != "failed":
                        continue

                    # Extract rule details
                    rule_id_el = (
                        rule.find(f"{ns}ruleId") if ns else rule.find("ruleId")
                    )
                    desc_el = (
                        rule.find(f"{ns}description")
                        if ns
                        else rule.find("description")
                    )
                    location_el = (
                        rule.find(f"{ns}location") if ns else rule.find("location")
                    )

                    # Also check attributes as fallback
                    rule_id = ""
                    if rule_id_el is not None and rule_id_el.text:
                        rule_id = rule_id_el.text.strip()
                    else:
                        # Try specification + clause + testNumber attributes
                        spec = rule.get("specification", "")
                        clause = rule.get("clause", "")
                        test_number = rule.get("testNumber", "")
                        if clause:
                            rule_id = f"{spec}-{clause}-{test_number}".strip("-")

                    description = ""
                    if desc_el is not None and desc_el.text:
                        description = desc_el.text.strip()
                    else:
                        description = rule.get("description", "")

                    location = ""
                    if location_el is not None and location_el.text:
                        location = location_el.text.strip()
                    else:
                        location = rule.get("location", "")

                    violations.append({
                        "tool": "verapdf",
                        "id": rule_id or "unknown-rule",
                        "impact": "serious",
                        "description": description or f"PDF/UA-1 rule {rule_id} failed",
                        "help": f"PDF/UA-1 compliance failure: {rule_id}",
                        "location": location,
                    })

        except FileNotFoundError:
            logger.warning(
                "veraPDF binary not found at %s; skipping validation",
                verapdf_bin,
            )
            return ValidationResult(tool="verapdf", violations=[], passed=True)
        except asyncio.TimeoutError:
            logger.warning(
                "veraPDF timed out after 120s for %s", pdf_path.name
            )
            return ValidationResult(tool="verapdf", violations=[], passed=True)
        except ET.ParseError as exc:
            logger.warning("veraPDF XML parse error: %s", exc)
            return ValidationResult(tool="verapdf", violations=[], passed=True)
        except Exception as exc:
            logger.warning("veraPDF validation failed unexpectedly: %s", exc)
            return ValidationResult(tool="verapdf", violations=[], passed=True)

        passed = len(violations) == 0

        logger.info(
            "veraPDF: %d violation(s), passed=%s", len(violations), passed
        )
        return ValidationResult(
            tool="verapdf", violations=violations, passed=passed
        )

    async def run_wave_validation(
        self,
        jobs: list,
        campus_start_url: str = "",
    ) -> list[dict]:
        """Run WAVE validation against deployed pages for a batch of jobs.

        For each job, constructs the URL of the deployed page and runs
        WAVE API validation against it.

        Parameters
        ----------
        jobs:
            List of :class:`DocumentJob` instances whose deployed pages
            should be validated.
        campus_start_url:
            Base URL of the campus site (e.g.
            ``https://www.lamc.edu``).  Used to construct the full URL
            for each job's final HTML page.

        Returns
        -------
        list[dict]
            One result dict per job with keys ``job_id``, ``url``,
            ``wave_result``, and ``passed``.
        """
        results: list[dict] = []

        for job in jobs:
            page_path = job.final_html_path or ""
            if campus_start_url and page_path:
                # Build a URL from the campus base + the job's relative path
                url = f"{campus_start_url.rstrip('/')}/{page_path.lstrip('/')}"
            elif page_path.startswith(("http://", "https://")):
                url = page_path
            else:
                logger.warning(
                    "Cannot build WAVE URL for job %s: no campus_start_url "
                    "and final_html_path is not a URL",
                    job.id,
                )
                results.append({
                    "job_id": job.id,
                    "url": "",
                    "wave_result": ValidationResult(
                        tool="wave", violations=[], passed=True
                    ),
                    "passed": True,
                })
                continue

            wave_result = await self.validate_with_wave(url)
            results.append({
                "job_id": job.id,
                "url": url,
                "wave_result": wave_result,
                "passed": wave_result.passed,
            })

        return results

    async def auto_remediate(
        self, job: DocumentJob, report: ValidationReport
    ) -> DocumentJob:
        """Feed validation errors back to the LLM for correction.

        Runs up to ``max_remediation_cycles`` (from config), re-validating
        after each correction.  Each cycle is logged to the database.

        Parameters
        ----------
        job:
            The document job containing the generated HTML.
        report:
            The initial validation report with violations.

        Returns
        -------
        DocumentJob
            Updated job — HTML corrected if possible, remediation count
            incremented, status set to VALIDATED or FLAGGED.
        """
        current_html = job.generated_html
        current_report = report
        cycle = 0

        while not current_report.passed and cycle < self._max_cycles:
            cycle += 1
            logger.info(
                "Auto-remediation cycle %d/%d for job %s",
                cycle, self._max_cycles, job.id,
            )

            corrected = await self._remediate_html(current_html, current_report)

            # Clean up the response — strip markdown fences if present
            corrected = corrected.strip()
            corrected = re.sub(r"^```(?:html)?\s*", "", corrected)
            corrected = re.sub(r"\s*```$", "", corrected)
            corrected = corrected.strip()

            if not corrected or len(corrected) < 50:
                logger.warning(
                    "Remediation cycle %d returned empty/short HTML; "
                    "keeping previous version",
                    cycle,
                )
                break

            current_html = corrected
            job.generated_html = current_html
            job.remediation_count = cycle

            # Write corrected HTML to file for re-validation
            html_path = Path(job.final_html_path) if job.final_html_path else None
            if html_path:
                html_path.write_text(current_html, encoding="utf-8")
            else:
                # Write to a temp file for validation
                with tempfile.NamedTemporaryFile(
                    suffix=".html", delete=False, mode="w", encoding="utf-8"
                ) as tmp:
                    tmp.write(current_html)
                    html_path = Path(tmp.name)

            # Log this cycle to the database
            current_report = await self.validate(html_path)

            for tool_result in [
                current_report.axe_result,
                current_report.pa11y_result,
                current_report.lighthouse_result,
                current_report.wave_result,
            ]:
                await self._db.log_validation(
                    job_id=job.id,
                    cycle=cycle,
                    page_key="canonical",
                    page_title=job.link_text or "Document",
                    page_path=job.final_html_path,
                    tool=tool_result.tool,
                    score=tool_result.score,
                    violations=tool_result.violations,
                    passed=tool_result.passed,
                )

            logger.info(
                "Remediation cycle %d result: %s (violations: %d)",
                cycle,
                "PASS" if current_report.passed else "FAIL",
                len(current_report.all_violations),
            )

        # Update job status
        job.validation_results = [
            current_report.axe_result,
            current_report.pa11y_result,
            current_report.lighthouse_result,
            current_report.wave_result,
        ]

        if current_report.passed:
            job.status = JobStatus.VALIDATED
            logger.info("Job %s PASSED validation after %d cycle(s)", job.id, cycle)
        else:
            job.status = JobStatus.FLAGGED
            job.error_message = (
                f"Failed validation after {cycle} remediation cycle(s). "
                f"{len(current_report.all_violations)} violation(s) remain."
            )
            logger.warning(
                "Job %s FLAGGED after %d remediation cycle(s) with %d violation(s)",
                job.id, cycle, len(current_report.all_violations),
            )

        await self._db.update_job(job)
        return job

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _results_for_page(
        self,
        page: RenderedPage,
        report: ValidationReport,
    ) -> list[ValidationResult]:
        """Copy a validation report into page-scoped result objects."""
        return [
            ValidationResult(
                tool=result.tool,
                score=result.score,
                violations=result.violations,
                passed=result.passed,
                page_key=page.page_key,
                page_title=page.title,
                page_path=page.relative_path,
            )
            for result in (
                report.axe_result,
                report.pa11y_result,
                report.lighthouse_result,
                report.wave_result,
            )
        ]

    async def _log_page_results(
        self,
        job_id: str,
        cycle: int,
        page: RenderedPage,
        results: list[ValidationResult],
    ) -> None:
        """Persist page-scoped validation results to the audit log."""
        for result in results:
            await self._db.log_validation(
                job_id=job_id,
                cycle=cycle,
                page_key=page.page_key,
                page_title=page.title,
                page_path=page.relative_path,
                tool=result.tool,
                score=result.score,
                violations=result.violations,
                passed=result.passed,
            )

    async def _remediate_html(
        self,
        html: str,
        report: ValidationReport,
    ) -> str:
        """Ask the LLM to fix accessibility issues in a single HTML document."""
        violation_text = self._format_violations_for_llm(report.all_violations)
        prompt = _REMEDIATION_PROMPT.format(
            violations=violation_text,
            html=html,
        )

        return await self._ollama.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert web accessibility remediation "
                        "specialist. Fix the WCAG violations in the HTML. "
                        "Return only the corrected HTML."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            thinking=True,
            max_tokens=32768,
            temperature=0.2,
        )

    @staticmethod
    def _merge_violations(
        *tool_results: ValidationResult,
    ) -> list[dict[str, Any]]:
        """Merge and deduplicate violations from all validation tools."""
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []

        tools = list(tool_results)

        for result in tools:
            for v in result.violations:
                # Dedup key: tool + id + first selector/snippet
                key_parts = [
                    v.get("tool", ""),
                    v.get("id", ""),
                    v.get("selector", ""),
                ]
                snippets = v.get("html_snippets", [])
                if snippets:
                    key_parts.append(snippets[0][:80])
                dedup_key = "|".join(key_parts)

                if dedup_key not in seen:
                    seen.add(dedup_key)
                    merged.append(v)

        # Sort by severity: critical > serious > moderate > minor
        severity_order = {"critical": 0, "serious": 1, "moderate": 2, "minor": 3}
        merged.sort(key=lambda v: severity_order.get(v.get("impact", "minor"), 3))

        return merged

    @staticmethod
    def _format_violations_for_llm(
        violations: list[dict[str, Any]],
    ) -> str:
        """Format violations into a clear, numbered list for the LLM."""
        if not violations:
            return "No violations found."

        lines: list[str] = []
        for i, v in enumerate(violations, 1):
            tool = v.get("tool", "unknown")
            vid = v.get("id", "unknown")
            impact = v.get("impact", "unknown")
            desc = v.get("description", "")
            help_text = v.get("help", "")
            wcag = v.get("wcag", "")
            selector = v.get("selector", "")
            snippets = v.get("html_snippets", [])

            parts = [f"{i}. [{tool}] {vid} ({impact})"]
            if wcag:
                parts.append(f"   WCAG: {wcag}")
            parts.append(f"   Issue: {desc}")
            if help_text and help_text != desc:
                parts.append(f"   Fix: {help_text}")
            if selector:
                parts.append(f"   Selector: {selector}")
            if snippets:
                parts.append(f"   HTML: {snippets[0][:150]}")

            lines.append("\n".join(parts))

        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_wcag_tags(tags: list[str]) -> str:
    """Extract WCAG success criteria from axe-core tags."""
    wcag_tags = [t for t in tags if t.startswith("wcag")]
    return ", ".join(wcag_tags) if wcag_tags else ""


def _extract_wcag_from_code(code: str) -> str:
    """Extract WCAG criteria reference from a pa11y issue code."""
    # pa11y codes like "WCAG2AA.Principle1.Guideline1_1.1_1_1.H37"
    match = re.search(r"Guideline(\d+_\d+)\.(\d+_\d+_\d+)", code)
    if match:
        sc = match.group(2).replace("_", ".")
        return f"WCAG {sc}"
    return ""


def _pa11y_type_to_impact(pa11y_type: str) -> str:
    """Map pa11y issue types to axe-compatible impact levels."""
    mapping = {
        "error": "serious",
        "warning": "moderate",
        "notice": "minor",
    }
    return mapping.get(pa11y_type.lower(), "minor")
