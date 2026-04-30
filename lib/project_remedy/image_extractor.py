"""Extract embedded images from PDF files using PyMuPDF.

Single-purpose module that pulls raster images from PDF pages, deduplicates
by xref, converts non-web formats to PNG, and writes them to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

from project_remedy.models import ExtractedImage

logger = logging.getLogger(__name__)

# Images smaller than this are likely bullets, icons, or decorative fragments.
_MIN_DIMENSION = 20


def extract_pdf_images(pdf_path: Path, output_dir: Path) -> list[ExtractedImage]:
    """Extract all meaningful embedded images from a PDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    output_dir:
        Directory where extracted images will be saved.

    Returns
    -------
    list[ExtractedImage]
        Manifest of extracted images with metadata.
    """
    import fitz  # PyMuPDF

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))

    seen_xrefs: set[int] = set()
    results: list[ExtractedImage] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]

            # Deduplicate: PDFs often reuse images across pages.
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                extracted = doc.extract_image(xref)
            except Exception as exc:
                logger.debug(
                    "Could not extract image xref=%d from page %d: %s",
                    xref, page_num + 1, exc,
                )
                continue

            if not extracted or not extracted.get("image"):
                continue

            width = extracted.get("width", 0)
            height = extracted.get("height", 0)

            # Skip tiny images (bullets, spacers, decorative).
            if width < _MIN_DIMENSION or height < _MIN_DIMENSION:
                logger.debug(
                    "Skipping tiny image xref=%d (%dx%d) on page %d",
                    xref, width, height, page_num + 1,
                )
                continue

            img_bytes = extracted["image"]
            ext = extracted.get("ext", "png")

            # Convert non-web formats to PNG.
            if ext in ("jb2", "jbig2", "ccitt", "jpx"):
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # CMYK or other colorspace
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                    ext = "png"
                except Exception as exc:
                    logger.debug(
                        "Pixmap conversion failed for xref=%d: %s", xref, exc
                    )
                    continue

            filename = f"img_p{page_num + 1}_x{xref}.{ext}"
            out_path = output_dir / filename
            out_path.write_bytes(img_bytes)

            results.append(
                ExtractedImage(
                    filename=filename,
                    page_number=page_num + 1,
                    xref=xref,
                    width=width,
                    height=height,
                )
            )

            logger.debug(
                "Extracted image %s (%dx%d) from page %d",
                filename, width, height, page_num + 1,
            )

    doc.close()

    logger.info(
        "Extracted %d images from %s into %s",
        len(results), pdf_path.name, output_dir,
    )
    return results
