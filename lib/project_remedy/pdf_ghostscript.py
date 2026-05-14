"""Ghostscript re-distill preprocessing for font/CIDSet issues.

Detects PDFs with font problems (missing embeddings, broken CIDSets,
missing ToUnicode maps) and re-distills them through Ghostscript PDF/A-2b
to normalize font structures before the tagging pipeline runs.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pikepdf

logger = logging.getLogger(__name__)

# Base14 fonts that don't need embedding or ToUnicode
_BASE14_FONTS = frozenset({
    "/Courier", "/Courier-Bold", "/Courier-Oblique", "/Courier-BoldOblique",
    "/Helvetica", "/Helvetica-Bold", "/Helvetica-Oblique", "/Helvetica-BoldOblique",
    "/Times-Roman", "/Times-Bold", "/Times-Italic", "/Times-BoldItalic",
    "/Symbol", "/ZapfDingbats",
})


def detect_font_issues(pdf_path: Path) -> bool:
    """Check if a PDF has font issues that Ghostscript re-distilling could fix.

    Returns False if:
    - The file doesn't exist or can't be opened
    - The PDF already has good tags (StructTreeRoot with children) — GS would destroy them
    - No font issues detected

    Returns True if any font has:
    - Missing embedding (no FontFile/FontFile2/FontFile3)
    - Incomplete CIDSet (decompressed < 8 bytes)
    - Missing ToUnicode on CID fonts (Type0, excluding Base14)
    - Type1/TrueType fonts missing encoding vectors
    """
    if not pdf_path.exists():
        return False

    try:
        pdf = pikepdf.open(pdf_path)
    except Exception:
        return False

    try:
        # Skip if PDF already has substantial tagging (>50 child elements)
        root = pdf.Root
        struct_tree = root.get("/StructTreeRoot")
        if struct_tree is not None:
            kids = struct_tree.get("/K")
            if kids is not None:
                try:
                    kid_count = len(kids) if isinstance(kids, pikepdf.Array) else 1
                except Exception:
                    kid_count = 1
                if kid_count > 50:
                    # Substantial structure tree — don't destroy it
                    return False

        # Scan all pages + Form XObjects for font issues
        for page in pdf.pages:
            if _check_resources_for_font_issues(page.get("/Resources"), pdf):
                return True

        return False
    except Exception:
        return False
    finally:
        pdf.close()


def _check_resources_for_font_issues(
    resources: pikepdf.Object | None,
    pdf: pikepdf.Pdf,
    _visited: set | None = None,
) -> bool:
    """Check font dictionaries in resources, recursing into Form XObjects."""
    if resources is None:
        return False

    if _visited is None:
        _visited = set()

    # Check fonts in this resource dict
    fonts = resources.get("/Font")
    if fonts is not None:
        try:
            for font_name in fonts.keys():
                font = fonts[font_name]
                if _font_has_issues(font):
                    return True
        except Exception:
            pass

    # Recurse into Form XObjects
    xobjects = resources.get("/XObject")
    if xobjects is not None:
        try:
            for xobj_name in xobjects.keys():
                xobj = xobjects[xobj_name]
                subtype = str(xobj.get("/Subtype", ""))
                if subtype == "/Form":
                    xobj_id = id(xobj)
                    if xobj_id in _visited:
                        continue
                    _visited.add(xobj_id)
                    nested_resources = xobj.get("/Resources")
                    if _check_resources_for_font_issues(nested_resources, pdf, _visited):
                        return True
        except Exception:
            pass

    return False


def _font_has_issues(font: pikepdf.Object) -> bool:
    """Check if a single font dictionary has issues GS could fix."""
    try:
        subtype = str(font.get("/Subtype", ""))
        base_font = str(font.get("/BaseFont", ""))

        # Skip Base14 fonts — they don't need embedding
        if base_font in _BASE14_FONTS:
            return False

        if subtype == "/Type0":
            # CID font — check descendants
            descendants = font.get("/DescendantFonts")
            if descendants is None:
                return True  # Broken Type0

            # Check for missing ToUnicode
            if font.get("/ToUnicode") is None:
                return True

            # Check for zero-value ToUnicode entries
            if _has_zero_tounicode_entries(font):
                return True

            # Check descendant font for embedding issues
            for desc in descendants:
                descriptor = desc.get("/FontDescriptor")
                if descriptor is None:
                    return True
                if not _has_font_file(descriptor):
                    return True
                if _has_incomplete_cidset(descriptor):
                    return True
                # Check for glyph width mismatches (zeros in /Widths)
                if _has_zero_width_glyphs(desc):
                    return True

        elif subtype in ("/Type1", "/TrueType"):
            descriptor = font.get("/FontDescriptor")
            if descriptor is None:
                return False  # Simple font without descriptor — likely standard
            if not _has_font_file(descriptor):
                return True
            # Check for missing encoding on non-standard fonts
            if font.get("/Encoding") is None and base_font not in _BASE14_FONTS:
                return True
            # Check for .notdef glyph patterns in /Differences
            if _has_notdef_differences(font):
                return True
            # Check for glyph width mismatches
            if _has_zero_width_glyphs(font):
                return True

        elif subtype in ("/Type3", "/CIDFontType0", "/CIDFontType2"):
            # Additional font subtypes that can have issues
            descriptor = font.get("/FontDescriptor")
            if descriptor is not None:
                if not _has_font_file(descriptor):
                    return True
            if _has_notdef_differences(font):
                return True
            if _has_zero_width_glyphs(font):
                return True

    except Exception:
        pass

    return False


def _has_font_file(descriptor: pikepdf.Object) -> bool:
    """Check if a font descriptor has an embedded font file."""
    return (
        descriptor.get("/FontFile") is not None
        or descriptor.get("/FontFile2") is not None
        or descriptor.get("/FontFile3") is not None
    )


def _has_incomplete_cidset(descriptor: pikepdf.Object) -> bool:
    """Check if CIDSet stream is suspiciously short (< 8 bytes after decompression)."""
    cidset = descriptor.get("/CIDSet")
    if cidset is None:
        return False
    try:
        # Decompress the stream before checking length
        raw_bytes = bytes(cidset.read_bytes())
        return len(raw_bytes) < 8
    except Exception:
        return False


def _has_notdef_differences(font: pikepdf.Object) -> bool:
    """Check if font encoding /Differences contains .notdef glyph names."""
    encoding = font.get("/Encoding")
    if encoding is None:
        return False
    try:
        if isinstance(encoding, pikepdf.Dictionary):
            diffs = encoding.get("/Differences")
        else:
            return False
        if diffs is None:
            return False
        for item in diffs:
            if isinstance(item, pikepdf.Name) and str(item) == "/.notdef":
                return True
    except Exception:
        pass
    return False


def _has_zero_tounicode_entries(font: pikepdf.Object) -> bool:
    """Check if ToUnicode CMap has zero-value (U+0000) mappings."""
    tounicode = font.get("/ToUnicode")
    if tounicode is None:
        return False
    try:
        raw = bytes(tounicode.read_bytes()).decode("latin-1", errors="replace")
        # Look for <0000> mappings in beginbfchar/beginbfrange sections
        if "<0000>" in raw:
            return True
    except Exception:
        pass
    return False


def _has_zero_width_glyphs(font: pikepdf.Object) -> bool:
    """Check if /Widths array contains zeros for used glyphs."""
    widths = font.get("/Widths")
    if widths is None:
        return False
    try:
        width_list = list(widths)
        if not width_list:
            return False
        # If more than 25% of widths are zero, likely a problem
        zero_count = sum(1 for w in width_list if float(w) == 0.0)
        return zero_count > 0 and zero_count > len(width_list) * 0.25
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# PDF needs classifier
# ---------------------------------------------------------------------------


def classify_pdf_needs(pdf_path: Path) -> str:
    """Classify what remediation approach a PDF needs.

    Returns one of:
    - ``"font_needs_gs"``: has font issues that require GS redistill
    - ``"structure_only"``: only needs tagging/structure fixes, skip GS
    - ``"image_only"``: scanned/image PDF needing OCR
    """
    if not pdf_path.exists():
        return "structure_only"

    try:
        pdf = pikepdf.open(pdf_path)
    except Exception:
        return "structure_only"

    try:
        # Check if image-only (no extractable text)
        has_text = False
        for page in pdf.pages[:3]:  # Sample first 3 pages
            contents = page.get("/Contents")
            if contents is not None:
                try:
                    if isinstance(contents, pikepdf.Array):
                        raw = b"".join(bytes(c.read_bytes()) for c in contents)
                    else:
                        raw = bytes(contents.read_bytes())
                    # Look for text operators (Tj, TJ, ', ")
                    if b"Tj" in raw or b"TJ" in raw:
                        has_text = True
                        break
                except Exception:
                    pass

        if not has_text:
            return "image_only"

        # Check for font issues
        if detect_font_issues(pdf_path):
            return "font_needs_gs"

        return "structure_only"
    except Exception:
        return "structure_only"
    finally:
        pdf.close()


# ---------------------------------------------------------------------------
# Re-distill
# ---------------------------------------------------------------------------

@dataclass
class RedistillResult:
    """Result of Ghostscript re-distill preprocessing."""
    success: bool = False
    visual_diff: float = 1.0
    error: str = ""
    duration_s: float = 0.0
    output_size: int = 0
    mode: str = ""  # which GS mode succeeded: "default", "pdfa", "ocr"


# GS command templates — each is a list of flags (no binary, no output, no input).
# Tried in order by redistill_with_recovery().

_GS_MODE_DEFAULT = {
    "name": "default",
    "flags": [
        "-dBATCH", "-dNOPAUSE", "-dQUIET",
        "-dEmbedAllFonts=true", "-dSubsetFonts=false",
        "-dProvideUnicode=true",
        "-dPDFNOCIDFALLBACK",
        "-sColorConversionStrategy=LeaveColorUnchanged",
        "-sDEVICE=pdfwrite",
    ],
    # PostScript to clear NeverEmbed so Base14 fonts are embedded (pdfwrite
    # default excludes them via /NeverEmbed .standardfonts).
    # Must come after -sDEVICE but before the input file via -c ... -f.
    "ps_prefix": "<</NeverEmbed []>> setdistillerparams",
    "timeout": 300,
}

_GS_MODE_PDFA = {
    "name": "pdfa",
    "flags": [
        "-dPDFA=2", "-dBATCH", "-dNOPAUSE", "-dQUIET",
        "-dPDFACompatibilityPolicy=2",
        "-dEmbedAllFonts=true", "-dSubsetFonts=false",
        "-dProvideUnicode=true",
        "-dPDFNOCIDFALLBACK",
        "-dPDFSTOPONERROR",
        "-sColorConversionStrategy=LeaveColorUnchanged",
        "-sDEVICE=pdfwrite",
    ],
    "timeout": 300,
}

_GS_MODE_OCR = {
    "name": "ocr",
    "flags": [
        "-dBATCH", "-dNOPAUSE", "-dQUIET",
        "-dEmbedAllFonts=true", "-dSubsetFonts=false",
        "-dProvideUnicode=true",
        "-dPDFNOCIDFALLBACK",
        "-sColorConversionStrategy=LeaveColorUnchanged",
        "-sUseOCR=AsNeeded",
        "-sDEVICE=pdfwrite",
    ],
    "ps_prefix": "<</NeverEmbed []>> setdistillerparams",
    "timeout": 900,
}


def _run_gs(
    gs_bin: str,
    mode: dict,
    source_path: Path,
    output_path: Path,
    visual_tolerance: float,
) -> RedistillResult:
    """Run a single GS attempt with the given mode flags."""
    cmd = [gs_bin] + mode["flags"] + [f"-sOutputFile={output_path}"]
    # If mode has a PostScript prefix (e.g. NeverEmbed override), inject
    # it via -c ... -f just before the input file.
    ps_prefix = mode.get("ps_prefix")
    if ps_prefix:
        cmd.extend(["-c", ps_prefix, "-f"])
    cmd.append(str(source_path))
    timeout = mode["timeout"]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        duration = round(time.monotonic() - start, 2)

        if proc.returncode != 0 or not output_path.exists():
            error = (proc.stderr or "").strip()[:500] or f"exit code {proc.returncode}"
            return RedistillResult(error=error, duration_s=duration, mode=mode["name"])

        try:
            visual_diff = _sampled_visual_diff(source_path, output_path)
        except Exception:
            visual_diff = 1.0

        if visual_diff > visual_tolerance:
            output_path.unlink(missing_ok=True)
            return RedistillResult(
                error=f"Visual diff {visual_diff:.4f} exceeds tolerance {visual_tolerance}",
                visual_diff=visual_diff,
                duration_s=duration,
                mode=mode["name"],
            )

        # Text integrity check: detect GS font corruption (garbled text)
        # that visual diff alone cannot catch.
        if not _check_text_integrity(source_path, output_path):
            output_path.unlink(missing_ok=True)
            return RedistillResult(
                error="GS output failed text integrity check (font corruption detected)",
                visual_diff=visual_diff,
                duration_s=duration,
                mode=mode["name"],
            )

        return RedistillResult(
            success=True,
            visual_diff=visual_diff,
            duration_s=duration,
            output_size=output_path.stat().st_size,
            mode=mode["name"],
        )

    except subprocess.TimeoutExpired:
        output_path.unlink(missing_ok=True)
        return RedistillResult(
            error=f"Ghostscript timed out after {timeout}s",
            duration_s=float(timeout),
            mode=mode["name"],
        )
    except Exception as exc:
        return RedistillResult(error=str(exc)[:500], mode=mode["name"])


def redistill_pdf(
    source_path: Path,
    output_path: Path,
    *,
    config=None,
    visual_tolerance: float = 0.03,
    use_ocr: bool = False,
) -> RedistillResult:
    """Re-distill a PDF through Ghostscript to fix font embedding issues.

    Tries a tiered approach:
    1. **Default mode** — embeds all fonts (including Base14) by clearing
       NeverEmbed, without PDF/A enforcement. Handles overprint, annotations,
       interpolated images that PDF/A-2 would reject.
    2. **PDF/A-2 fallback** — if default mode fails, retries with ``-dPDFA=2``
       for its stricter font-enforcement side effects.
    3. **OCR salvage** (if *use_ocr* is True) — adds ``-sUseOCR=AsNeeded``
       to recover text from stripped CID fonts (~29x slower).

    Colors are preserved unchanged via LeaveColorUnchanged.
    """
    if not source_path.exists():
        return RedistillResult(error=f"Source does not exist: {source_path}")

    if config is not None:
        visual_tolerance = config.pdf_remediation.redistill_visual_tolerance

    gs_bin = _find_gs_binary(config)
    if gs_bin is None:
        return RedistillResult(error="Ghostscript binary not found")

    # Tier 1: default mode (no PDF/A enforcement)
    result = _run_gs(gs_bin, _GS_MODE_DEFAULT, source_path, output_path, visual_tolerance)
    if result.success:
        return result

    logger.info("GS default mode failed for %s (%s), trying PDF/A-2", source_path.name, result.error[:80])

    # Tier 2: PDF/A-2 mode (stricter, but handles some edge cases better)
    result_pdfa = _run_gs(gs_bin, _GS_MODE_PDFA, source_path, output_path, visual_tolerance)
    if result_pdfa.success:
        return result_pdfa

    # Tier 3: OCR salvage (only if requested — very slow)
    if use_ocr:
        logger.info("GS PDF/A mode also failed for %s, trying OCR salvage", source_path.name)
        result_ocr = _run_gs(gs_bin, _GS_MODE_OCR, source_path, output_path, visual_tolerance)
        if result_ocr.success:
            return result_ocr
        return result_ocr

    # Return the first (most informative) error
    return result


def _find_gs_binary(config=None) -> str | None:
    """Locate Ghostscript binary from config or PATH."""
    if config is not None:
        cfg_path = config.pdf_remediation.ghostscript_path
        if cfg_path and Path(cfg_path).is_file():
            return cfg_path
    return shutil.which("gs")


def _check_text_integrity(original_path: Path, gs_output_path: Path) -> bool:
    """Compare extracted text to detect GS font corruption.

    GS redistill with ``-dSubsetFonts=true`` can corrupt CID font subsetting
    on some fonts (notably CID-subset heading fonts), producing garbled/unreadable
    text.  The visual diff threshold is too coarse to catch this because the
    affected area is small relative to the full page.

    This check extracts text from both the original and GS output using
    PyMuPDF and flags corruption when:
    - Significant text loss (GS text < 80 % of original length), or
    - High character-level divergence (> 10 % of compared characters differ).

    Returns True if text is intact, False if corruption is detected.
    """
    import fitz

    try:
        src_doc = fitz.open(str(original_path))
    except Exception:
        return True  # Can't verify — assume OK

    try:
        gs_doc = fitz.open(str(gs_output_path))
    except Exception:
        src_doc.close()
        return True

    try:
        for page_idx in range(min(len(src_doc), len(gs_doc))):
            src_text = src_doc[page_idx].get_text("text").strip()
            gs_text = gs_doc[page_idx].get_text("text").strip()

            if not src_text:
                continue

            # Check for significant text loss or corruption.
            # Garbled text often has high Unicode codepoint values or control chars.
            if len(gs_text) < len(src_text) * 0.8:
                logger.warning(
                    "GS text integrity: page %d text length dropped from %d to %d",
                    page_idx, len(src_text), len(gs_text),
                )
                return False

            # Check for character corruption: count chars with very different codepoints.
            common_len = min(len(src_text), len(gs_text))
            if common_len > 0:
                diff_count = sum(
                    1 for a, b in zip(src_text[:common_len], gs_text[:common_len])
                    if a != b
                )
                if diff_count / common_len > 0.1:  # >10% of characters changed
                    logger.warning(
                        "GS text integrity: page %d has %.1f%% character divergence",
                        page_idx, diff_count / common_len * 100,
                    )
                    return False

            # Check for semantic corruption: decimal points missing in percentages
            # Pattern like "343%" instead of "34.3%" indicates GS corrupted the text
            src_percentages = re.findall(r'\b\d{1,3}\.\d+%', src_text)
            gs_percentages = re.findall(r'\b\d{1,3}\.\d+%', gs_text)
            if len(src_percentages) >= 5 and len(gs_percentages) < len(src_percentages) * 0.5:
                # GS lost most decimal points - corruption detected
                logger.warning(
                    "GS text integrity: page %d lost decimal points (%d%% with decimals vs %d%% original)",
                    page_idx, len(gs_percentages), len(src_percentages)
                )
                return False

            # Check for high numeric token corruption (e.g., "343%" vs "34.3%")
            src_numeric = re.findall(r'\b\d+\.?\d*%', src_text)
            gs_numeric = re.findall(r'\b\d+\.?\d*%', gs_text)
            if src_numeric and gs_numeric:
                # Compare numeric values - if many are wildly different, corruption
                mismatched = 0
                for s, g in zip(src_numeric[:50], gs_numeric[:50]):
                    try:
                        s_val = float(s.rstrip('%'))
                        g_val = float(g.rstrip('%'))
                        if abs(s_val - g_val) > 50:  # More than 50 points difference
                            mismatched += 1
                    except ValueError:
                        pass
                if mismatched >= 3:
                    logger.warning(
                        "GS text integrity: page %d has %d wildly mismatched percentages",
                        page_idx, mismatched
                    )
                    return False

        return True
    except Exception as exc:
        logger.debug("GS text integrity check error: %s", exc)
        return True  # On error, don't block — assume OK
    finally:
        src_doc.close()
        gs_doc.close()


def _sampled_visual_diff(source_path: Path, output_path: Path) -> float:
    """Compare visual fidelity on sampled pages (first, middle, last)."""
    import fitz

    try:
        src_doc = fitz.open(str(source_path))
        dst_doc = fitz.open(str(output_path))
    except Exception:
        return 1.0

    src_pages = len(src_doc)
    dst_pages = len(dst_doc)

    if src_pages != dst_pages or src_pages == 0:
        src_doc.close()
        dst_doc.close()
        return 1.0

    # Sample: first, middle, last (deduplicated)
    indices = sorted(set([0, src_pages // 2, src_pages - 1]))

    total_diff = 0.0
    for i in indices:
        src_pix = src_doc[i].get_pixmap(dpi=72)
        dst_pix = dst_doc[i].get_pixmap(dpi=72)

        src_bytes = src_pix.samples
        dst_bytes = dst_pix.samples

        if len(src_bytes) != len(dst_bytes):
            total_diff += 1.0
            continue

        pixel_diffs = sum(abs(a - b) for a, b in zip(src_bytes, dst_bytes))
        max_diff = len(src_bytes) * 255
        total_diff += pixel_diffs / max_diff if max_diff > 0 else 0.0

    src_doc.close()
    dst_doc.close()

    return total_diff / len(indices)
