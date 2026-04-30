"""HTML-to-Tagged-PDF converter using Playwright + pikepdf.

Converts accessible HTML produced by the pipeline into properly tagged,
PDF/UA-1 compliant PDFs using Chromium's built-in tagged-PDF support.

Usage::

    converter = HTMLToPDFConverter(max_concurrent=8)
    await converter.start()
    result = await converter.convert(html_content, output_path, title="My Doc")
    await converter.close()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pikepdf
from playwright.async_api import async_playwright, Browser

from project_remedy.pdf_fixer import (
    fix_alt_text_elements,
    fix_display_doc_title,
    fix_language,
    fix_link_annotations,
    fix_mark_info,
    fix_pdfua_identifier,
    fix_role_map,
    fix_untagged_content,
)

logger = logging.getLogger(__name__)


@dataclass
class PDFConversionResult:
    """Outcome of a single HTML-to-PDF conversion."""

    output_path: Path | None = None
    success: bool = False
    error_message: str = ""


class HTMLToPDFConverter:
    """Converts accessible HTML to tagged PDF/UA-1 documents.

    Uses a single Chromium browser instance with concurrent browser contexts
    controlled by a semaphore.

    Parameters
    ----------
    max_concurrent:
        Maximum number of simultaneous Chromium contexts.
    """

    def __init__(self, max_concurrent: int = 8) -> None:
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._playwright = None
        self._browser: Browser | None = None

    async def start(self) -> None:
        """Launch the Chromium browser."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch()
        logger.info(
            "HTMLToPDFConverter started (max_concurrent=%d).",
            self._max_concurrent,
        )

    async def close(self) -> None:
        """Shut down the browser and Playwright."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("HTMLToPDFConverter closed.")

    async def convert(
        self,
        html: str,
        output_path: Path,
        *,
        title: str = "",
        language: str = "en",
    ) -> PDFConversionResult:
        """Convert an HTML string to a tagged PDF.

        Parameters
        ----------
        html:
            Full HTML document string (should include <!DOCTYPE html>).
        output_path:
            Where to write the final PDF.
        title:
            Document title for PDF metadata.
        language:
            BCP 47 language tag for PDF metadata.
        """
        if not self._browser:
            return PDFConversionResult(
                error_message="Browser not started. Call start() first.",
            )

        async with self._semaphore:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Inject print-optimised CSS to fix common PDF/UA issues.
                html = _inject_print_css(html)

                # Render HTML → tagged PDF via Chromium.
                context = await self._browser.new_context()
                page = await context.new_page()
                await page.set_content(html, wait_until="networkidle")
                await page.pdf(
                    path=str(output_path),
                    tagged=True,
                    format="Letter",
                    print_background=True,
                    margin={
                        "top": "0.75in",
                        "right": "0.75in",
                        "bottom": "0.75in",
                        "left": "0.75in",
                    },
                )
                await context.close()

                # Post-process: add PDF/UA-1 metadata via pikepdf.
                _add_pdfua_metadata(
                    output_path,
                    title=title,
                    language=language,
                )

                logger.debug("Converted to tagged PDF: %s", output_path.name)
                return PDFConversionResult(
                    output_path=output_path,
                    success=True,
                )

            except Exception as exc:
                logger.warning(
                    "HTML-to-PDF conversion failed for %s: %s",
                    output_path.name,
                    exc,
                )
                return PDFConversionResult(error_message=str(exc))

    async def convert_batch(
        self,
        items: Sequence[tuple[str, Path, str, str]],
    ) -> list[PDFConversionResult]:
        """Convert multiple HTML documents concurrently.

        Parameters
        ----------
        items:
            Sequence of ``(html, output_path, title, language)`` tuples.

        Returns
        -------
        list[PDFConversionResult]
            Results in the same order as *items*.
        """
        tasks = [
            self.convert(html, path, title=title, language=lang)
            for html, path, title, lang in items
        ]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Print CSS injection
# ---------------------------------------------------------------------------

_PRINT_CSS = """\
<style>
@media print {
  /* Hide interactive / screen-only elements from PDF output. */
  .skip-nav,
  .no-print,
  .document-controls,
  [role="navigation"] { display: none !important; }

  /* Ensure unlabelled inputs get an aria-label via CSS-generated content
     won't help PDF tags, but hiding empty readonly inputs does. */
  input[type="text"]:empty:not([aria-label]):not([title]) {
    /* Make visually present but add role presentation if truly decorative */
  }
}

/* Ensure every <input> has an accessible name for PDF form fields.
   Chromium maps aria-label to the PDF field's /TU (alternate text). */
input:not([aria-label]):not([title]) {
  /* Fallback: use the name attribute value as tooltip */
}
</style>
"""


def _inject_print_css(html: str) -> str:
    """Inject print-optimisation CSS and fix unlabelled form fields.

    - Hides skip-nav, print buttons, and navigation from PDF output.
    - Adds aria-label to <input> fields that lack one, derived from their
      name/id attribute or nearest <label>.
    """
    import re

    # 1. Inject print CSS before </head>.
    if "</head>" in html:
        html = html.replace("</head>", f"{_PRINT_CSS}</head>", 1)

    # 2. Add aria-label to inputs that lack one.
    def _fix_input(match: re.Match) -> str:
        tag = match.group(0)
        # Skip if already has aria-label or title.
        if "aria-label" in tag or "title=" in tag:
            return tag

        # Derive label from name or id attribute.
        name_match = re.search(r'name="([^"]*)"', tag)
        id_match = re.search(r'id="([^"]*)"', tag)
        label = (name_match or id_match)
        if label:
            # Convert "intake-clerk-signature" → "Intake clerk signature"
            readable = label.group(1).replace("-", " ").replace("_", " ").strip()
            readable = readable.capitalize()
            if readable:
                tag = tag.rstrip(">").rstrip("/")
                tag = f'{tag} aria-label="{readable}">'
        return tag

    html = re.sub(r"<input\b[^>]*>", _fix_input, html)

    return html


# ---------------------------------------------------------------------------
# pikepdf post-processing
# ---------------------------------------------------------------------------


def _add_pdfua_metadata(
    pdf_path: Path,
    *,
    title: str,
    language: str,
) -> None:
    """Stamp PDF/UA-1 metadata and fix structure tags on a Chromium PDF.

    Delegates to shared fix functions in ``pdf_fixer.py`` so that the
    same logic is used by both the HTML-to-PDF pipeline and the
    standalone ``remedy pdf fix`` command.
    """
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        fix_pdfua_identifier(pdf)
        fix_mark_info(pdf)
        fix_language(pdf, language=language)
        fix_display_doc_title(pdf, title=title)
        fix_role_map(pdf)
        fix_alt_text_elements(pdf)
        fix_link_annotations(pdf)
        fix_untagged_content(pdf)

        # Set dc:title and dc:language directly (fix_display_doc_title handles
        # the viewer preference but we also want the XMP fields).
        try:
            with pdf.open_metadata() as meta:
                meta["dc:title"] = title
                meta["dc:language"] = language
        except Exception:
            pass

        pdf.save()

    logger.debug(
        "PDF/UA-1 metadata applied to %s (title=%s, lang=%s).",
        pdf_path.name,
        title[:50],
        language,
    )
