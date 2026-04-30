"""Legacy PDF-to-PDF remediation orchestrator.

This module is retained for experimental and triage workflows only.
The production PDF-to-PDF path should use ``pdf_fixer.fix_and_verify()``
plus the shared composite acceptance gate.

Produces tagged, accessible PDFs from source documents by:
  1. Triaging PDF complexity (scanned / tagged / image-heavy)
  2. Running OCR via Stirling-PDF for scanned documents
  3. Extracting structure + images via PyMuPDF
  4. Invoking the LLM for alt text, structure tags, reading order, language
  5. Injecting the PDF structure tree + metadata via pikepdf / pypdf
  6. Converting to PDF/A via Stirling-PDF
  7. Validating with veraPDF for PDF/UA-1 compliance
  8. Looping if validation fails (up to max_remediation_cycles)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pikepdf
from pypdf import PdfReader, PdfWriter

from project_remedy.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF classification
# ---------------------------------------------------------------------------


class PDFComplexity(Enum):
    """Broad classification of PDF content type."""

    SCANNED = "scanned"       # Image-only, needs OCR
    TAGGED = "tagged"         # Already has structure tree
    TEXT_ONLY = "text_only"   # Has text layer but no tags
    IMAGE_HEAVY = "image_heavy"  # Mixed content, many images


@dataclass
class PDFTriageResult:
    """Outcome of the triage step."""

    complexity: PDFComplexity
    page_count: int = 0
    has_text: bool = False
    has_tags: bool = False
    has_images: bool = False
    image_count: int = 0
    needs_ocr: bool = False


@dataclass
class StructureTag:
    """A single PDF structure tag inferred by the LLM."""

    tag_type: str       # H1, H2, P, Table, Figure, L, LI, etc.
    page: int           # 0-based page number
    content: str = ""   # Text content (for headings, paragraphs)
    alt_text: str = ""  # Alt text (for figures/images)
    bbox: list[float] = field(default_factory=list)  # [x0, y0, x1, y1]


@dataclass
class PDFRemediationResult:
    """Outcome of the PDF remediation pipeline."""

    output_path: Path | None = None
    success: bool = False
    triage: PDFTriageResult | None = None
    tags_injected: int = 0
    validation_passed: bool = False
    remediation_cycles: int = 0
    error_message: str = ""


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_STRUCTURE_PROMPT = """\
You are an expert in PDF accessibility and WCAG 2.1 AA compliance.

Analyze this document text extracted from a PDF and determine the logical \
structure. For each structural element, specify:
- tag_type: one of H1, H2, H3, H4, H5, H6, P, Table, Figure, L (list), \
LI (list item), BlockQuote, Code, Link
- content: the text of the element
- page: 0-based page number

Return a JSON array of objects with these fields. Example:
```json
[
  {{"tag_type": "H1", "page": 0, "content": "Document Title"}},
  {{"tag_type": "P", "page": 0, "content": "First paragraph..."}}
]
```

Document text:

{text}
"""

_ALT_TEXT_PROMPT = """\
You are an accessibility specialist. Generate concise, descriptive alt text \
for this image from a PDF document. The alt text should:
- Be under 125 characters
- Describe the content and function of the image
- Not start with "Image of" or "Picture of"
- Return "decorative" if the image is purely decorative

Context: this image appears in a document titled "{title}" on page {page}.

Return ONLY the alt text string, nothing else.
"""

_LANGUAGE_PROMPT = """\
Detect the primary language of this document text. Return ONLY the BCP 47 \
language tag (e.g., "en", "es", "fr", "zh", "ko"). Nothing else.

Text sample:
{text}
"""


# ---------------------------------------------------------------------------
# PDFRemediator
# ---------------------------------------------------------------------------


class PDFRemediator:
    """Legacy orchestrator for experimental PDF-to-PDF remediation.

    Parameters
    ----------
    config:
        Pipeline configuration (provides PDF remediation settings).
    llm_client:
        An initialised LLM client (OllamaClient or GeminiClient) for
        inference tasks (alt text, structure, language detection, OCR).
    """

    def __init__(
        self,
        config: PipelineConfig,
        llm_client: Any,
        stirling: Any = None,  # kept for backwards compat, unused
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._stirling = stirling  # unused — OCR + PDF/A handled locally
        self._max_cycles = config.validation.max_remediation_cycles
        self._legacy_warning_emitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def remediate(
        self,
        pdf_path: Path,
        output_dir: Path,
        *,
        title: str = "",
    ) -> PDFRemediationResult:
        """Run the full PDF-to-PDF remediation pipeline.

        Parameters
        ----------
        pdf_path:
            Path to the source PDF file.
        output_dir:
            Directory to write the remediated PDF to.
        title:
            Document title (used for metadata and context).

        Returns
        -------
        PDFRemediationResult
            Outcome with the path to the remediated PDF.
        """
        if not self._legacy_warning_emitted:
            logger.warning(
                "PDFRemediator is legacy and not release-safe; prefer pdf_fixer.fix_and_verify() for production PDF-to-PDF remediation."
            )
            self._legacy_warning_emitted = True

        result = PDFRemediationResult()

        try:
            # Step 1: Triage
            triage = await self._triage(pdf_path)
            result.triage = triage
            logger.info(
                "PDF triage: %s (pages=%d, images=%d, needs_ocr=%s)",
                triage.complexity.value,
                triage.page_count,
                triage.image_count,
                triage.needs_ocr,
            )

            working_pdf = pdf_path

            # Step 2: Extract text — use Gemini OCR for scanned docs
            if triage.needs_ocr:
                logger.info("Running Gemini OCR on scanned PDF: %s", pdf_path.name)
                text = await self._llm.ocr(file_path=pdf_path)
            else:
                text = self._extract_text(pdf_path)

            # Step 4: LLM inference — structure, alt text, language
            structure_tags, language = await asyncio.gather(
                self._infer_structure(text),
                self._detect_language(text),
            )

            # Generate alt text for images
            image_alt_texts = await self._generate_image_alt_texts(
                working_pdf, title,
            )

            # Step 5: Inject tags + metadata
            tagged_path = output_dir / f"{pdf_path.stem}_tagged.pdf"
            tags_injected = self._inject_tags(
                working_pdf,
                tagged_path,
                structure_tags=structure_tags,
                image_alt_texts=image_alt_texts,
                language=language,
                title=title or pdf_path.stem,
            )
            result.tags_injected = tags_injected
            logger.info("Injected %d structure tags.", tags_injected)

            # Step 6: Finalize — add PDF/A-like metadata to tagged PDF
            final_path = output_dir / f"{pdf_path.stem}_accessible.pdf"
            try:
                self._add_accessibility_metadata(tagged_path, final_path, title or pdf_path.stem, language)
            except Exception as exc:
                logger.warning(
                    "PDF metadata finalization failed, using tagged PDF: %s", exc,
                )
                final_path = tagged_path

            # Step 6a: Contrast remediation (if enabled)
            if (
                hasattr(self._config, 'contrast')
                and self._config.contrast.enabled
                and self._config.contrast.auto_fix
            ):
                try:
                    from project_remedy.contrast import ContrastRemediator

                    contrast_rem = ContrastRemediator(
                        self._llm,
                        dpi=self._config.contrast.dpi,
                    )
                    contrast_result = await contrast_rem.remediate_document(
                        str(final_path),
                        str(final_path),
                        level=self._config.contrast.level,
                    )
                    logger.info(
                        "Contrast remediation: %d/%d issues fixed",
                        contrast_result.issues_fixed,
                        contrast_result.total_issues,
                    )
                except Exception as exc:
                    logger.warning("Contrast remediation failed: %s", exc)

            result.output_path = final_path
            result.success = True

        except Exception as exc:
            result.error_message = str(exc)
            logger.error("PDF remediation failed for %s: %s", pdf_path, exc)

        return result

    # ------------------------------------------------------------------
    # Step 1: Triage
    # ------------------------------------------------------------------

    async def _triage(self, pdf_path: Path) -> PDFTriageResult:
        """Classify a PDF by complexity and content type."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            # Fallback triage without PyMuPDF
            return await self._triage_basic(pdf_path)

        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        total_text_len = 0
        image_count = 0

        for page in doc:
            total_text_len += len(page.get_text())
            image_count += len(page.get_images(full=True))

        # Check for existing structure tree
        has_tags = False
        catalog = doc.pdf_catalog()
        if catalog:
            try:
                xref = doc.xref_get_key(catalog, "MarkInfo")
                has_tags = xref[0] != "null"
            except Exception:
                pass

        doc.close()

        has_text = total_text_len > 100
        has_images = image_count > 0
        needs_ocr = not has_text and page_count > 0

        if needs_ocr:
            complexity = PDFComplexity.SCANNED
        elif has_tags:
            complexity = PDFComplexity.TAGGED
        elif has_images and image_count > page_count * 2:
            complexity = PDFComplexity.IMAGE_HEAVY
        else:
            complexity = PDFComplexity.TEXT_ONLY

        return PDFTriageResult(
            complexity=complexity,
            page_count=page_count,
            has_text=has_text,
            has_tags=has_tags,
            has_images=has_images,
            image_count=image_count,
            needs_ocr=needs_ocr,
        )

    async def _triage_basic(self, pdf_path: Path) -> PDFTriageResult:
        """Basic triage using pypdf when PyMuPDF is not available."""
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        total_text = ""
        for page in reader.pages:
            total_text += page.extract_text() or ""

        has_text = len(total_text) > 100
        return PDFTriageResult(
            complexity=PDFComplexity.SCANNED if not has_text else PDFComplexity.TEXT_ONLY,
            page_count=page_count,
            has_text=has_text,
            needs_ocr=not has_text,
        )

    # ------------------------------------------------------------------
    # Step 3: Extract text
    # ------------------------------------------------------------------

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract all text from a PDF for LLM analysis."""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            text_parts = []
            for i, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
            doc.close()
            return "\n\n".join(text_parts)
        except ImportError:
            reader = PdfReader(str(pdf_path))
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
            return "\n\n".join(text_parts)

    # ------------------------------------------------------------------
    # Step 6: Finalize
    # ------------------------------------------------------------------

    def _add_accessibility_metadata(
        self, tagged_path: Path, output_path: Path, title: str, language: str
    ) -> None:
        """Add accessibility metadata to the tagged PDF using pikepdf.

        Sets title, language, marked-content flag, and display-doc-title.
        This is the local replacement for Stirling-PDF's PDF/A conversion.
        """
        with pikepdf.open(tagged_path) as pdf:
            # Set document metadata.
            with pdf.open_metadata() as meta:
                meta["dc:title"] = title
                meta["dc:language"] = language
                meta["pdf:Producer"] = "Remedy PDF Desktop Accessibility Remediation Pipeline"
                meta["xmp:CreatorTool"] = "Remedy PDF Desktop"

            # Set /MarkInfo to indicate tagged PDF.
            if "/MarkInfo" not in pdf.Root:
                pdf.Root["/MarkInfo"] = pikepdf.Dictionary({"/Marked": True})
            else:
                pdf.Root["/MarkInfo"]["/Marked"] = True

            # Set /Lang on the document catalog.
            pdf.Root["/Lang"] = language

            # Set /ViewerPreferences to display document title.
            if "/ViewerPreferences" not in pdf.Root:
                pdf.Root["/ViewerPreferences"] = pikepdf.Dictionary()
            pdf.Root["/ViewerPreferences"]["/DisplayDocTitle"] = True

            pdf.save(output_path)

        logger.info(
            "Accessibility metadata written to %s (title=%s, lang=%s)",
            output_path.name, title, language,
        )

    # ------------------------------------------------------------------
    # Step 4: LLM inference
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Batch-mode helpers
    # ------------------------------------------------------------------

    def prepare_structure_messages(self, text: str) -> list[dict[str, str]]:
        """Build messages for a structure inference call (no LLM call)."""
        truncated = text[:30_000]
        prompt = _STRUCTURE_PROMPT.format(text=truncated)
        return [{"role": "user", "content": prompt}]

    def prepare_language_messages(self, text: str) -> list[dict[str, str]]:
        """Build messages for a language detection call (no LLM call)."""
        sample = text[:2000]
        prompt = _LANGUAGE_PROMPT.format(text=sample)
        return [{"role": "user", "content": prompt}]

    def parse_structure_result(self, response: str | None) -> list[StructureTag]:
        """Parse structure inference result from batch response."""
        if not response:
            return []
        try:
            json_text = self._extract_json_array(response)
            raw_tags = json.loads(json_text)
            return [
                StructureTag(
                    tag_type=t.get("tag_type", "P"),
                    page=t.get("page", 0),
                    content=t.get("content", ""),
                    alt_text=t.get("alt_text", ""),
                )
                for t in raw_tags
                if isinstance(t, dict)
            ]
        except Exception as exc:
            logger.warning("Structure parse failed: %s", exc)
            return []

    def parse_language_result(self, response: str | None) -> str:
        """Parse language detection result from batch response."""
        if not response:
            return "en"
        lang = response.strip().lower()
        if len(lang) <= 10 and lang.replace("-", "").isalpha():
            return lang
        return "en"

    async def _infer_structure(self, text: str) -> list[StructureTag]:
        """Ask the LLM to identify document structure from extracted text."""
        if not text.strip():
            return []

        # Truncate to avoid token limits
        truncated = text[:30_000]
        prompt = _STRUCTURE_PROMPT.format(text=truncated)

        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                thinking=False,
                max_tokens=8192,
            )
            # Parse JSON from response
            json_text = self._extract_json_array(response)
            raw_tags = json.loads(json_text)
            return [
                StructureTag(
                    tag_type=t.get("tag_type", "P"),
                    page=t.get("page", 0),
                    content=t.get("content", ""),
                    alt_text=t.get("alt_text", ""),
                )
                for t in raw_tags
                if isinstance(t, dict)
            ]
        except Exception as exc:
            logger.warning("Structure inference failed: %s", exc)
            return []

    async def _detect_language(self, text: str) -> str:
        """Detect the primary language of the document."""
        if not text.strip():
            return "en"

        sample = text[:2000]
        prompt = _LANGUAGE_PROMPT.format(text=sample)

        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                thinking=False,
                max_tokens=16,
            )
            lang = response.strip().lower()
            # Validate it looks like a language tag
            if len(lang) <= 10 and lang.replace("-", "").isalpha():
                return lang
            return "en"
        except Exception:
            return "en"

    async def _generate_image_alt_texts(
        self,
        pdf_path: Path,
        title: str,
    ) -> dict[int, str]:
        """Generate alt text for each image in the PDF.

        Returns a mapping of image xref -> alt text.
        """
        try:
            import fitz
        except ImportError:
            return {}

        doc = fitz.open(str(pdf_path))
        alt_texts: dict[int, str] = {}
        tasks = []

        for page_num, page in enumerate(doc):
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in alt_texts:
                    continue

                async def _gen_alt(xref=xref, page_num=page_num):
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        img_bytes = pix.tobytes("png")
                        import tempfile
                        with tempfile.NamedTemporaryFile(
                            suffix=".png", delete=False
                        ) as tmp:
                            tmp.write(img_bytes)
                            tmp_path = Path(tmp.name)

                        prompt = _ALT_TEXT_PROMPT.format(
                            title=title or "Unknown",
                            page=page_num + 1,
                        )
                        alt = await self._llm.vision(
                            image_path=tmp_path,
                            prompt=prompt,
                        )
                        tmp_path.unlink(missing_ok=True)

                        alt = alt.strip().strip('"').strip("'")
                        if alt.lower() == "decorative":
                            alt = ""
                        alt_texts[xref] = alt
                    except Exception as exc:
                        logger.debug(
                            "Alt text generation failed for xref %d: %s",
                            xref, exc,
                        )
                        alt_texts[xref] = ""

                tasks.append(_gen_alt())

        if tasks:
            await asyncio.gather(*tasks)

        doc.close()
        return alt_texts

    # ------------------------------------------------------------------
    # Step 5: Tag injection via pikepdf / pypdf
    # ------------------------------------------------------------------

    def _inject_tags(
        self,
        input_path: Path,
        output_path: Path,
        *,
        structure_tags: list[StructureTag],
        image_alt_texts: dict[int, str],
        language: str,
        title: str,
    ) -> int:
        """Inject PDF structure tree and metadata into the document.

        Uses pikepdf for the structure tree and pypdf for metadata.

        Returns the number of tags injected.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        # --- Phase 1: pikepdf — structure tree + MarkInfo ---
        with pikepdf.open(input_path) as pdf:
            # Mark the document as tagged
            pdf.Root["/MarkInfo"] = pikepdf.Dictionary({"/Marked": True})

            # Set language
            pdf.Root["/Lang"] = pikepdf.String(language)

            # Build structure tree
            struct_elems = []
            for tag in structure_tags:
                elem = pikepdf.Dictionary(
                    {
                        "/Type": pikepdf.Name("/StructElem"),
                        "/S": pikepdf.Name(f"/{tag.tag_type}"),
                    }
                )
                if tag.alt_text:
                    elem["/Alt"] = pikepdf.String(tag.alt_text)
                struct_elems.append(elem)
                count += 1

            # Add alt text for images
            for xref, alt in image_alt_texts.items():
                if alt:
                    elem = pikepdf.Dictionary(
                        {
                            "/Type": pikepdf.Name("/StructElem"),
                            "/S": pikepdf.Name("/Figure"),
                            "/Alt": pikepdf.String(alt),
                        }
                    )
                    struct_elems.append(elem)
                    count += 1

            if struct_elems:
                struct_tree_root = pikepdf.Dictionary(
                    {
                        "/Type": pikepdf.Name("/StructTreeRoot"),
                        "/K": pikepdf.Array(struct_elems),
                        "/ParentTree": pikepdf.Dictionary({}),
                    }
                )
                pdf.Root["/StructTreeRoot"] = pdf.make_indirect(
                    struct_tree_root
                )

            # Set viewer preferences
            pdf.Root["/ViewerPreferences"] = pikepdf.Dictionary(
                {"/DisplayDocTitle": True}
            )

            pdf.save(str(output_path))

        # --- Phase 2: pypdf — metadata (title, producer) ---
        reader = PdfReader(str(output_path))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        writer.add_metadata(
            {
                "/Title": title,
                "/Producer": "Remedy PDF Desktop — Accessibility Remediation Pipeline",
                "/Creator": "Remedy PDF Desktop",
            }
        )
        with open(output_path, "wb") as f:
            writer.write(f)

        logger.info(
            "Injected %d tags + metadata (lang=%s, title=%s) into %s",
            count,
            language,
            title[:50],
            output_path.name,
        )
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_array(text: str) -> str:
        """Extract a JSON array from LLM response text."""
        # Look for ```json ... ``` blocks first
        import re
        match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if match:
            return match.group(1)
        # Try to find a bare JSON array
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            return match.group(0)
        return "[]"
