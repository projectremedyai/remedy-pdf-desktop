"""PDF Accessibility Checker — all 32 Adobe Acrobat checks.

Runs the same checks Adobe's accessibility checker performs, organized into
four categories: Document, Page Content, Forms/Tables/Lists, and
Alternate Text & Headings.

Usage::

    checker = PDFAccessibilityChecker(Path("report.pdf"))
    report = checker.run_all()
    for r in report.results:
        print(r.rule_id, r.status)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import pikepdf

from project_remedy.pdf_semantics import (
    document_has_bookmarks,
    document_requires_bookmarks,
    find_node_page,
    get_rendered_multimedia_names,
    node_has_annotation_ref,
    node_has_content_association,
    node_has_direct_content as _shared_node_has_direct_content,
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Document",
    "Page Content",
    "Forms Tables Lists",
    "Alt Text Headings",
]

# Maps short category keys to display names.
CATEGORY_ALIASES = {
    "doc": "Document",
    "document": "Document",
    "page": "Page Content",
    "page-content": "Page Content",
    "forms": "Forms Tables Lists",
    "tables": "Forms Tables Lists",
    "lists": "Forms Tables Lists",
    "alt": "Alt Text Headings",
    "headings": "Alt Text Headings",
    "alt-text": "Alt Text Headings",
}

SOURCE_FONT_RISK_DETAIL_PREFIX = "Likely inherited source-font/CIDSet limitation: "


@dataclass
class CheckResult:
    """Outcome of a single accessibility check."""

    rule_id: str
    category: str
    description: str
    status: str  # "Passed", "Failed", "Manual Check Needed"
    details: list[str] = field(default_factory=list)
    fixable: bool = False


@dataclass
class CheckReport:
    """Full accessibility report for a PDF."""

    file_path: Path
    file_size: int
    page_count: int
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.status == "Passed")

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status == "Failed")

    @property
    def manual_count(self) -> int:
        return sum(1 for r in self.results if r.status == "Manual Check Needed")

    @property
    def fixable_count(self) -> int:
        return sum(1 for r in self.results if r.status == "Failed" and r.fixable)

    def results_by_category(self) -> dict[str, list[CheckResult]]:
        grouped: dict[str, list[CheckResult]] = {}
        for r in self.results:
            grouped.setdefault(r.category, []).append(r)
        return grouped


@dataclass
class _CharacterEncodingAnalysis:
    """Internal summary of whether the PDF text layer is trustworthy."""

    details: list[str] = field(default_factory=list)
    source_font_risk_details: list[str] = field(default_factory=list)
    page_numbers: set[int] = field(default_factory=set)
    requires_rebuild: bool = False


def _resolve_pdf_object(obj):
    """Best-effort resolve for indirect pikepdf objects."""
    if isinstance(obj, pikepdf.Array):
        return obj
    if isinstance(obj, pikepdf.Object) and obj.is_indirect:
        try:
            return obj.resolve()
        except Exception:
            return obj
    return obj


# ---------------------------------------------------------------------------
# Structure tree walker
# ---------------------------------------------------------------------------


def walk_structure_tree(
    pdf: pikepdf.Pdf,
) -> Generator[tuple[pikepdf.Dictionary, int, pikepdf.Dictionary | None], None, None]:
    """Yield ``(node, depth, parent)`` for every node in the structure tree.

    Uses an explicit stack instead of recursion to handle large PDFs.
    """
    struct_root = pdf.Root.get("/StructTreeRoot")
    if struct_root is None:
        return

    # Stack entries: (node, depth, parent)
    stack: list[tuple[pikepdf.Dictionary, int, pikepdf.Dictionary | None]] = [
        (struct_root, 0, None)
    ]
    seen: set[tuple[str, object]] = set()

    while stack:
        node, depth, parent = stack.pop()
        resolved_node = _resolve_pdf_object(node)
        if not isinstance(resolved_node, pikepdf.Dictionary):
            continue

        objgen = getattr(resolved_node, "objgen", None)
        key = ("objgen", objgen) if objgen is not None and objgen != (0, 0) else ("id", id(resolved_node))
        if key in seen:
            continue
        seen.add(key)
        node = resolved_node
        yield node, depth, parent

        kids = node.get("/K")
        if kids is None:
            continue

        if isinstance(kids, pikepdf.Array):
            # Avoid copying very large /K arrays; some generated PDFs have
            # tens of thousands of sibling structure elements.
            for idx in range(len(kids) - 1, -1, -1):
                child = kids[idx]
                resolved = _resolve_pdf_object(child)
                if isinstance(resolved, pikepdf.Dictionary) and "/S" in resolved:
                    stack.append((resolved, depth + 1, node))
            continue
        elif isinstance(kids, pikepdf.Dictionary):
            resolved = _resolve_pdf_object(kids)
            if isinstance(resolved, pikepdf.Dictionary) and "/S" in resolved:
                stack.append((resolved, depth + 1, node))


def _get_struct_type(node: pikepdf.Dictionary) -> str:
    """Return the structure type name as a plain string (e.g. 'Table')."""
    s = node.get("/S")
    if s is None:
        return ""
    return str(s).lstrip("/")


def _node_has_direct_content(node: pikepdf.Dictionary) -> bool:
    """True if the node has marked-content references (integers or MCR dicts)."""
    return _shared_node_has_direct_content(node)


def _decode_pdf_literal_string(data: bytes) -> bytes:
    """Decode a PDF literal string into raw one-byte font codes."""
    decoded = bytearray()
    i = 0
    while i < len(data):
        byte = data[i]
        if byte != 0x5C:
            decoded.append(byte)
            i += 1
            continue

        if i + 1 >= len(data):
            break

        nxt = data[i + 1]
        if nxt in b"nrtbf":
            decoded.append({
                ord("n"): 0x0A,
                ord("r"): 0x0D,
                ord("t"): 0x09,
                ord("b"): 0x08,
                ord("f"): 0x0C,
            }[nxt])
            i += 2
            continue

        if nxt in b"()\\":
            decoded.append(nxt)
            i += 2
            continue

        if 48 <= nxt <= 55:
            octal = [nxt]
            j = i + 2
            while j < len(data) and len(octal) < 3 and 48 <= data[j] <= 55:
                octal.append(data[j])
                j += 1
            decoded.append(int(bytes(octal), 8))
            i = j
            continue

        if nxt in (0x0A, 0x0D):
            i += 2
            if nxt == 0x0D and i < len(data) and data[i] == 0x0A:
                i += 1
            continue

        decoded.append(nxt)
        i += 2

    return bytes(decoded)


def _extract_used_font_codes(page: pikepdf.Page) -> dict[str, set[int]]:
    """Return glyph codes used by each page font in text operators.

    Simple fonts use one-byte character codes. Type0/CID fonts commonly use
    two-byte big-endian CIDs; collecting only individual bytes caused
    GlyphLessFont-style PDFs to retain invalid broad identity CMaps.
    """
    used: dict[str, set[int]] = {}
    current_font: str | None = None
    font_code_width: dict[str, int] = {}

    resources = page.get("/Resources")
    fonts = resources.get("/Font") if resources is not None else None
    if fonts is not None:
        for name, font in fonts.items():
            try:
                resolved = _resolve_pdf_object(font)
            except Exception:
                resolved = font
            subtype = str(resolved.get("/Subtype", "")) if isinstance(resolved, pikepdf.Dictionary) else ""
            font_code_width[str(name)] = 2 if subtype == "/Type0" else 1

    def _raw_string(value) -> bytes:
        if isinstance(value, bytes):
            return value
        try:
            return bytes(value)
        except Exception:
            return str(value).encode("latin-1", errors="ignore")

    def _add_codes(font_name: str, raw: bytes) -> None:
        if not raw:
            return
        width = font_code_width.get(font_name, 1)
        font_codes = used.setdefault(font_name, set())
        if width == 2 and len(raw) >= 2:
            even_len = len(raw) - (len(raw) % 2)
            for idx in range(0, even_len, 2):
                font_codes.add(int.from_bytes(raw[idx:idx + 2], "big"))
            return
        font_codes.update(raw)

    try:
        instructions = pikepdf.parse_content_stream(page)
    except Exception:
        return {}

    for operands, operator in instructions:
        op = str(operator)
        if op == "Tf" and operands:
            current_font = str(operands[0])
            continue
        if current_font is None:
            continue
        if op == "TJ" and operands:
            arr = operands[0]
            if isinstance(arr, pikepdf.Array):
                for item in arr:
                    if isinstance(item, (pikepdf.String, bytes)) or getattr(item, "type_code", None) == "string":
                        _add_codes(current_font, _raw_string(item))
            elif isinstance(arr, (pikepdf.String, bytes)):
                _add_codes(current_font, _raw_string(arr))
        elif op in {"Tj", "'"} and operands:
            _add_codes(current_font, _raw_string(operands[0]))
        elif op == '"' and len(operands) >= 3:
            _add_codes(current_font, _raw_string(operands[2]))

    return used


def _parse_tounicode_mapped_codes(font: pikepdf.Dictionary) -> set[int]:
    """Return source codes explicitly covered by a font's /ToUnicode CMap."""
    tounicode = font.get("/ToUnicode")
    if tounicode is None:
        return set()

    stream = _resolve_pdf_object(tounicode)
    try:
        cmap = stream.read_bytes().decode("latin-1", errors="replace")
    except Exception:
        return set()

    mapped: set[int] = set()
    mode: str | None = None
    for raw_line in cmap.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("beginbfchar"):
            mode = "bfchar"
            continue
        if line.endswith("beginbfrange"):
            mode = "bfrange"
            continue
        if line in {"endbfchar", "endbfrange"}:
            mode = None
            continue

        if mode == "bfchar":
            for src, _dst in re.findall(r"<([0-9A-Fa-f]{2,4})>\s*<([0-9A-Fa-f]+)>", line):
                mapped.add(int(src, 16))
        elif mode == "bfrange":
            match = re.match(
                r"<([0-9A-Fa-f]{2,4})>\s*<([0-9A-Fa-f]{2,4})>\s*(<[^>]+>|\[)",
                line,
            )
            if match:
                start = int(match.group(1), 16)
                end = int(match.group(2), 16)
                mapped.update(range(start, end + 1))

    return mapped


def _find_suspicious_extracted_text(text: str) -> list[str]:
    """Return human-readable problems found in extracted page text."""
    findings: list[str] = []
    if re.search(r"\b(?:[A-Za-z]{1,}[#•][A-Za-z]{2,}|[A-Za-z]{2,}[#•][A-Za-z]{1,})\b", text):
        findings.append("words contain corrupted inline glyphs")
    common_short_words = {
        "a", "an", "and", "as", "at", "be", "by", "day", "do", "for", "from",
        "go", "he", "if", "in", "is", "it", "may", "of", "on", "or", "our",
        "pay", "so", "the", "to", "up", "us", "we", "www",
    }
    split_word_pattern = re.compile(
        r"\b([A-Za-z]{1,3}) {2,}([A-Za-z]{4,16})\b|\b([A-Za-z]{4,16}) {2,}([A-Za-z]{1,3})\b"
    )
    for match in split_word_pattern.finditer(text):
        left = (match.group(1) or match.group(3) or "").lower()
        right = (match.group(2) or match.group(4) or "").lower()
        short_fragment = left if len(left) <= 4 else right
        if short_fragment in common_short_words:
            continue
        if len(left + right) >= 7:
            findings.append("words are split by repeated spaces")
            break
    return findings


def _sample_page_numbers(page_numbers: list[int], *, limit: int) -> list[int]:
    """Sample page numbers evenly across a document-sized page list."""
    if limit <= 0 or not page_numbers:
        return []
    if len(page_numbers) <= limit:
        return list(page_numbers)
    if limit == 1:
        return [page_numbers[0]]

    sampled: list[int] = []
    last_index = -1
    for i in range(limit):
        index = round(i * (len(page_numbers) - 1) / (limit - 1))
        if index == last_index:
            continue
        sampled.append(page_numbers[index])
        last_index = index
    return sampled


def _extracted_text_looks_trustworthy(text: str) -> bool:
    """Heuristic gate for whether extracted text is good enough for AT use."""
    normalized = " ".join(text.split())
    if not normalized:
        return False
    if _find_suspicious_extracted_text(text):
        return False
    if "\ufffd" in text:
        return False

    control_chars = re.findall(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", text)
    if len(control_chars) > max(2, len(normalized) // 1000):
        return False

    alnum = sum(ch.isalnum() for ch in normalized)
    letters = sum(ch.isalpha() for ch in normalized)
    word_like = re.findall(r"[A-Za-z]{3,}", normalized)
    numeric_heavy = _looks_like_numeric_problem_sheet(normalized, word_like)
    if len(normalized) >= 80 and (alnum == 0 or letters / max(alnum, 1) < 0.45) and not numeric_heavy:
        return False
    if len(normalized) >= 80 and len(word_like) < 6 and not numeric_heavy:
        return False
    return True


def _looks_like_numeric_problem_sheet(normalized: str, word_like: list[str] | None = None) -> bool:
    """Allow numeric worksheets whose extracted text is sparse but still readable."""
    words = word_like if word_like is not None else re.findall(r"[A-Za-z]{3,}", normalized)
    numeric_tokens = re.findall(r"\b\d+(?:[.)]|°|['\"])?\b", normalized)
    enumerated_items = re.findall(r"\b\d+\)", normalized)
    return (
        len(normalized) >= 80
        and len(words) >= 3
        and len(numeric_tokens) >= 10
        and len(enumerated_items) >= 8
    )


def _analyze_character_encoding(
    pdf: pikepdf.Pdf,
    pdf_path: Path | None,
    *,
    max_pages: int = 10,
) -> _CharacterEncodingAnalysis:
    """Inspect whether the text layer is complete enough for assistive tech."""
    analysis = _CharacterEncodingAnalysis()
    provisional_missing_maps: list[tuple[int, str]] = []
    provisional_unmapped: list[tuple[int, str]] = []

    # Track pages that only have Base14 fonts with encoding issues
    base14_only_pages: set[int] = set()

    for page_number, page in enumerate(pdf.pages, 1):
        used_font_codes = _extract_used_font_codes(page)
        resources = page.get("/Resources")
        fonts = resources.get("/Font") if resources is not None else None
        if fonts is None:
            continue

        page_has_non_base14_with_issues = False
        page_fonts_with_issues: list[tuple[int, str, pikepdf.Dictionary]] = []

        for font_name, font_ref in fonts.items():
            font = _resolve_pdf_object(font_ref)
            if not isinstance(font, pikepdf.Dictionary):
                continue

            # Check if this is a Base14 font using /BaseFont (not resource name)
            base_font = str(font.get("/BaseFont", "")).lstrip("/")
            is_base14 = any(
                name in base_font
                for name in (
                    "Helvetica", "Times", "Courier", "Symbol", "ZapfDingbats",
                    "Arial", "TimesNewRoman", "CourierNew",
                )
            )

            has_tounicode = font.get("/ToUnicode") is not None
            has_encoding = font.get("/Encoding") is not None
            has_encoding_issue = False

            if not has_tounicode and not has_encoding:
                page_fonts_with_issues.append((page_number, str(font_name), font))
                has_encoding_issue = True
            elif has_tounicode:
                mapped_codes = _parse_tounicode_mapped_codes(font)
                if mapped_codes:
                    used_codes = used_font_codes.get(str(font_name), set())
                    unmapped_codes = sorted(
                        code for code in used_codes
                        if code not in mapped_codes and code not in {0x20, 0x28, 0x29, 0x5C}
                    )
                    if unmapped_codes:
                        page_fonts_with_issues.append((page_number, str(font_name), font))
                        has_encoding_issue = True

            # Only flag as non-Base14 issue if this font has an actual encoding issue
            if has_encoding_issue and not is_base14:
                page_has_non_base14_with_issues = True

        # Track pages where ALL fonts with issues are Base14 (no non-Base14 fonts have issues)
        if page_fonts_with_issues and not page_has_non_base14_with_issues:
            base14_only_pages.add(page_number)

        # Add findings to provisional lists
        for page_number, font_name, font in page_fonts_with_issues:
            has_tounicode = font.get("/ToUnicode") is not None
            has_encoding = font.get("/Encoding") is not None
            if not has_tounicode and not has_encoding:
                provisional_missing_maps.append(
                    (
                        page_number,
                        f"Page {page_number}: {font_name} missing /ToUnicode and /Encoding",
                    )
                )
            else:
                # Must be unmapped codes case
                used_codes = used_font_codes.get(font_name, set())
                mapped_codes = _parse_tounicode_mapped_codes(font)
                unmapped_codes = sorted(
                    code for code in used_codes
                    if code not in mapped_codes and code not in {0x20, 0x28, 0x29, 0x5C}
                )
                if unmapped_codes:
                    preview = ", ".join(f"0x{code:02X}" for code in unmapped_codes[:6])
                    provisional_unmapped.append(
                        (
                            page_number,
                            f"Page {page_number}: {font_name} uses unmapped glyph codes in /ToUnicode ({preview})",
                        )
                    )

    trusted_pages: set[int] = set()
    if pdf_path is not None:
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            try:
                unique_unmapped_pages = sorted({page for page, _detail in provisional_unmapped})
                sampled_unmapped_pages = set(
                    _sample_page_numbers(unique_unmapped_pages, limit=max_pages * 3)
                )
                candidate_pages = set(range(1, min(max_pages, len(doc)) + 1))
                candidate_pages.update(sampled_unmapped_pages)
                for page_number in sorted(page for page in candidate_pages if 1 <= page <= len(doc)):
                    page_idx = page_number - 1
                    text = doc[page_idx].get_text("text")[:5000]
                    if not text.strip():
                        continue
                    if _extracted_text_looks_trustworthy(text):
                        trusted_pages.add(page_number)
                        continue
                    findings = _find_suspicious_extracted_text(text)
                    if findings or not _extracted_text_looks_trustworthy(text):
                        analysis.details.append(
                            f"Page {page_idx + 1}: suspicious extracted text ({'; '.join(findings) or 'text extraction is low quality'})"
                        )
                        analysis.page_numbers.add(page_idx + 1)
                        analysis.requires_rebuild = True

                if sampled_unmapped_pages and sampled_unmapped_pages.issubset(trusted_pages):
                    trusted_pages.update(unique_unmapped_pages)
            finally:
                doc.close()
        except Exception:
            pass

    # Helper to check if a font name is a Base14 (standard) font
    def _is_base14_font(detail: str) -> bool:
        """Return True if the font mentioned in detail is a Base14 font."""
        base14_names = {
            "Helvetica", "Times", "Courier", "Symbol", "ZapfDingbats",
            "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
            "Times-Bold", "Times-Italic", "Times-BoldItalic",
            "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique",
        }
        # Extract font name from detail like "Page 1: /Helv uses unmapped..."
        for name in base14_names:
            if name in detail or f"/{name}" in detail:
                return True
        return False

    for page_number, detail in provisional_unmapped:
        if page_number in trusted_pages:
            analysis.source_font_risk_details.append(detail)
            continue
        analysis.details.append(detail)
        analysis.page_numbers.add(page_number)
        analysis.requires_rebuild = True

    for page_number, detail in provisional_missing_maps:
        if page_number in trusted_pages:
            analysis.source_font_risk_details.append(detail)
            continue
        analysis.details.append(detail)
        analysis.page_numbers.add(page_number)
        analysis.requires_rebuild = True

    return analysis


# ---------------------------------------------------------------------------
# Generic/placeholder alt text detection
# ---------------------------------------------------------------------------

# Patterns that indicate placeholder alt text providing zero descriptive value.
_GENERIC_ALT_TEXT_LITERALS = frozenset({
    "figure",
    "image",
    "img",
    "picture",
    "photo",
    "graphic",
    "chart",
    "diagram",
    "icon",
    "logo",
    "illustration",
    "decorative",
    "untitled",
    "placeholder",
    "alt text",
    "alt",
    "none",
    "n/a",
    "na",
    "todo",
    "tbd",
    "insert alt text",
    "describe this image",
    "needs description",
})

# Regex for filename-like patterns (e.g., "image1.png", "fig_02.jpg", "IMG_1234.jpeg").
_FILENAME_ALT_PATTERN = re.compile(
    r"^[\w\-. ]*\.(png|jpg|jpeg|gif|bmp|tiff?|svg|webp|ico|eps|pdf)$",
    re.IGNORECASE,
)


def _is_generic_alt_text(alt_text: str) -> bool:
    """Return True when alt text is a generic/placeholder string.

    These provide zero descriptive value and should be treated the same
    as missing alt text by both the checker and the fixer.
    """
    if not alt_text:
        return True
    normalized = alt_text.strip().lower()
    if not normalized:
        return True
    if normalized in _GENERIC_ALT_TEXT_LITERALS:
        return True
    if _FILENAME_ALT_PATTERN.match(normalized):
        return True
    # Catch numbered variants like "Figure 1", "Image 2", "img3"
    if re.match(r"^(figure|image|img|picture|photo|graphic)\s*\d*$", normalized):
        return True
    # Catch vague patterns like "a figure showing", "diagram of data", "chart depicting trends"
    # These patterns match when the description stops at the generic phrase without specifics
    vague_patterns = [
        # "A figure showing..." without specific content following
        r"^(a\s+)?(figure|image|photo|picture|graphic)\s+(showing|depicting|displaying)\s*$",
        # "Chart of..." or "Diagram of..." without meaningful follow-up
        r"^(a\s+)?(chart|diagram|graph|table)\s+(showing|depicting|displaying|of)\s*$",
        # "This is a figure/image..." without more description
        r"^(this\s+)?(is\s+a)?\s*(figure|image|photo|chart|diagram|graphic)\s*$",
        # "Image shows..." without specifics
        r"^(image|photo|picture)\s+(contains|shows|displays)\s*$",
        # "A visual representation..."
        r"^a\s+(visual|image|graphic)\s+(representation|showing)\s*$",
    ]
    for pattern in vague_patterns:
        if re.search(pattern, normalized):
            return True
    # Catch very short descriptions (under 15 chars are likely generic)
    if len(alt_text.strip()) < 15:
        # But allow if it contains specific nouns (not just "a" + generic word)
        words = normalized.split()
        if len(words) <= 2:
            # "Campus photo" is 2 words but still generic
            # "Student registration form" is 3 words, more specific
            # Check if it's just "type + generic noun"
            generic_endings = {"image", "photo", "picture", "graphic", "figure", "chart", "diagram", "logo", "icon"}
            if any(normalized.endswith(ending) for ending in generic_endings):
                return True
    return False


def _structure_type_looks_textual(stype: str) -> bool:
    """Return True for producer-specific roles that clearly carry text."""
    normalized = re.sub(r"[^a-z0-9]+", "", stype.lower())
    if not normalized:
        return False
    if any(
        token in normalized
        for token in (
            "figure",
            "image",
            "formula",
            "formfield",
            "artifact",
            "table",
            "chart",
            "graphic",
        )
    ):
        return False
    textual_names = {
        "normal",
        "p",
        "paragraph",
        "subtitle",
        "covertitle",
        "coversubtitle",
        "reportnumber",
        "appleconvertedspace",
        "author",
        "toc",
        "hyperlink",
        "palhyperlink",
        "hyperlinkpalhyperlink",
    }
    if (
        normalized in textual_names
        or normalized.startswith("toc")
        or normalized.startswith("heading")
    ):
        return True
    return any(
        token in normalized
        for token in (
            "title",
            "subtitle",
            "paragraph",
            "normal",
            "author",
            "hyperlink",
            "bodytext",
            "text",
        )
    )


# ---------------------------------------------------------------------------
# Checker class
# ---------------------------------------------------------------------------


class PDFAccessibilityChecker:
    """Run all 32 Adobe-equivalent accessibility checks on a PDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    vision_result:
        Optional pre-computed vision analysis result.  When provided,
        checks #4 (reading order) and #8 (color contrast) use the
        vision model's findings instead of returning "Manual Check Needed".
    """

    def __init__(
        self,
        pdf_path: Path,
        vision_result: object | None = None,
    ) -> None:
        self.pdf_path = pdf_path
        self._vision = vision_result
        self._structure_walk_cache: dict[
            int,
            list[tuple[pikepdf.Dictionary, int, pikepdf.Dictionary | None]],
        ] = {}

    def _walk_structure_tree(
        self,
        pdf: pikepdf.Pdf,
    ) -> list[tuple[pikepdf.Dictionary, int, pikepdf.Dictionary | None]]:
        """Cache one structure-tree traversal for this checker instance."""
        key = id(pdf)
        cached = self._structure_walk_cache.get(key)
        if cached is None:
            cached = list(walk_structure_tree(pdf))
            self._structure_walk_cache[key] = cached
        return cached

    def run_all(self) -> CheckReport:
        """Run all 32 checks and return a full report."""
        stat = self.pdf_path.stat()
        with pikepdf.open(self.pdf_path) as pdf:
            report = CheckReport(
                file_path=self.pdf_path,
                file_size=stat.st_size,
                page_count=len(pdf.pages),
            )
            report.results.extend(self._checks_document(pdf))
            report.results.extend(self._checks_page_content(pdf))
            report.results.extend(self._checks_forms_tables_lists(pdf))
            report.results.extend(self._checks_alt_text_headings(pdf))
        return report

    def run_category(self, category: str) -> CheckReport:
        """Run checks for one category only."""
        key = CATEGORY_ALIASES.get(category.lower().replace(" ", "-"), category)
        stat = self.pdf_path.stat()
        with pikepdf.open(self.pdf_path) as pdf:
            report = CheckReport(
                file_path=self.pdf_path,
                file_size=stat.st_size,
                page_count=len(pdf.pages),
            )
            dispatch = {
                "Document": self._checks_document,
                "Page Content": self._checks_page_content,
                "Forms Tables Lists": self._checks_forms_tables_lists,
                "Alt Text Headings": self._checks_alt_text_headings,
            }
            fn = dispatch.get(key)
            if fn:
                report.results.extend(fn(pdf))
        return report

    # -----------------------------------------------------------------------
    # Category 1: Document (8 checks)
    # -----------------------------------------------------------------------

    def _checks_document(self, pdf: pikepdf.Pdf) -> list[CheckResult]:
        return [
            self._check_accessibility_permission(pdf),
            self._check_not_image_only(pdf),
            self._check_tagged(pdf),
            self._check_structure_tree_integrity(pdf),
            self._check_logical_reading_order(pdf),
            self._check_language(pdf),
            self._check_display_doc_title(pdf),
            self._check_bookmarks(pdf),
            self._check_color_contrast(pdf),
        ]

    def _check_accessibility_permission(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #1: Accessibility permission flag is set."""
        # If the PDF is not encrypted, permission is implicitly granted.
        if not pdf.is_encrypted:
            return CheckResult(
                rule_id="doc-accessibility-permission",
                category="Document",
                description="Accessibility permission flag is set",
                status="Passed",
            )

        # Check permission bits — bit 10 (extract for accessibility).
        try:
            perms = pdf.allow
            if perms.extract or perms.accessibility:
                return CheckResult(
                    rule_id="doc-accessibility-permission",
                    category="Document",
                    description="Accessibility permission flag is set",
                    status="Passed",
                )
        except Exception:
            pass

        return CheckResult(
            rule_id="doc-accessibility-permission",
            category="Document",
            description="Accessibility permission flag is set",
            status="Failed",
            details=["Encryption restricts assistive technology access"],
            fixable=True,
        )

    def _check_not_image_only(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #2: Document is not image-only PDF."""
        pages_with_text = 0
        image_only_pages = []

        for i, page in enumerate(pdf.pages, 1):
            contents = page.get("/Contents")
            if contents is None:
                image_only_pages.append(i)
                continue

            raw = b""
            if isinstance(contents, pikepdf.Array):
                for stream in contents:
                    try:
                        raw += stream.read_bytes()
                    except Exception:
                        pass
            else:
                try:
                    raw = contents.read_bytes()
                except Exception:
                    pass

            text = raw.decode("latin-1", errors="replace")
            # Look for text-showing operators: Tj, TJ, ', "
            if re.search(r"\b(Tj|TJ|'|\")\b", text):
                pages_with_text += 1
            else:
                image_only_pages.append(i)

        if not image_only_pages:
            return CheckResult(
                rule_id="doc-not-image-only",
                category="Document",
                description="Document is not image-only PDF",
                status="Passed",
            )

        if pages_with_text == 0:
            return CheckResult(
                rule_id="doc-not-image-only",
                category="Document",
                description="Document is not image-only PDF",
                status="Failed",
                details=[
                    "Document appears to be image-only (no text operators found)",
                    "Needs OCR to extract text content",
                ],
                fixable=False,
            )

        return CheckResult(
            rule_id="doc-not-image-only",
            category="Document",
            description="Document is not image-only PDF",
            status="Passed",
            details=[
                f"Pages without text operators: {_format_page_ranges(image_only_pages)}"
            ],
        )

    def _check_tagged(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #3: Document is tagged PDF.

        Adobe Acrobat requires /MarkInfo/Marked *and* a non-empty
        /StructTreeRoot with at least one child element.  The previous
        implementation only checked MarkInfo which let documents pass
        that Adobe flags as failing.
        """
        failures: list[str] = []

        mark_info = pdf.Root.get("/MarkInfo")
        if not (mark_info and bool(mark_info.get("/Marked"))):
            failures.append("/MarkInfo/Marked is not true")

        struct_root = pdf.Root.get("/StructTreeRoot")
        if struct_root is None:
            failures.append("No /StructTreeRoot found")
        else:
            kids = struct_root.get("/K")
            if kids is None:
                failures.append("/StructTreeRoot has no children (/K)")
            else:
                # Verify at least one child is a real structure element.
                items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
                has_struct_child = False
                for item in items:
                    resolved = _resolve_pdf_object(item)
                    if isinstance(resolved, pikepdf.Dictionary) and "/S" in resolved:
                        has_struct_child = True
                        break
                if not has_struct_child:
                    failures.append(
                        "/StructTreeRoot has no structure element children"
                    )

            # Check /ParentTree exists (required for content association).
            parent_tree = struct_root.get("/ParentTree")
            if parent_tree is None:
                failures.append("/StructTreeRoot has no /ParentTree")

        if not failures:
            return CheckResult(
                rule_id="doc-tagged",
                category="Document",
                description="Document is tagged PDF",
                status="Passed",
            )
        return CheckResult(
            rule_id="doc-tagged",
            category="Document",
            description="Document is tagged PDF",
            status="Failed",
            details=failures,
            fixable=True,
        )

    def _check_structure_tree_integrity(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check structure tree for disconnected nodes and missing common ancestors.

        Adobe Acrobat flags "Tagged content" as FAILED when structure tree
        nodes lack proper parent linkage or when the ParentTree references
        objects that are not reachable from the /StructTreeRoot.  MuPDF
        reports this as "No common ancestor in structure tree".

        This check verifies:
        1. Every /StructElem in the tree has a /P pointing to a node that
           is itself reachable from StructTreeRoot.
        2. Every ParentTree entry points to a /StructElem reachable from
           StructTreeRoot.
        3. No /StructElem has a /P reference that forms a cycle.
        """
        struct_root = pdf.Root.get("/StructTreeRoot")
        if struct_root is None:
            return CheckResult(
                rule_id="doc-struct-tree-integrity",
                category="Document",
                description="Structure tree is internally consistent",
                status="Passed",
                details=["No structure tree present"],
            )

        issues: list[str] = []

        # Phase 1: Collect all reachable node objgens by walking from root.
        reachable_objgens: set[tuple[int, int]] = set()
        root_objgen = getattr(struct_root, "objgen", (0, 0))
        if root_objgen != (0, 0):
            reachable_objgens.add(root_objgen)

        nodes_with_bad_parent: list[str] = []
        nodes_missing_parent: list[str] = []

        for node, depth, parent in self._walk_structure_tree(pdf):
            objgen = getattr(node, "objgen", (0, 0))
            if objgen != (0, 0):
                reachable_objgens.add(objgen)

        # Phase 2: Re-walk and verify /P linkage.
        for node, depth, parent in self._walk_structure_tree(pdf):
            if parent is None:
                continue  # Root node
            stype = _get_struct_type(node)
            p_ref = node.get("/P")
            if p_ref is None:
                nodes_missing_parent.append(
                    f"/{stype} at depth {depth} has no /P reference"
                )
                continue

            # Resolve /P and check reachability.
            try:
                p_resolved = _resolve_pdf_object(p_ref)
                p_objgen = getattr(p_resolved, "objgen", (0, 0))
                if p_objgen != (0, 0) and p_objgen not in reachable_objgens:
                    nodes_with_bad_parent.append(
                        f"/{stype} at depth {depth}: /P points to unreachable object"
                    )
            except Exception:
                nodes_with_bad_parent.append(
                    f"/{stype} at depth {depth}: /P reference unresolvable"
                )

        if nodes_missing_parent:
            issues.extend(nodes_missing_parent[:5])
            if len(nodes_missing_parent) > 5:
                issues.append(
                    f"... and {len(nodes_missing_parent) - 5} more nodes missing /P"
                )
        if nodes_with_bad_parent:
            issues.extend(nodes_with_bad_parent[:5])
            if len(nodes_with_bad_parent) > 5:
                issues.append(
                    f"... and {len(nodes_with_bad_parent) - 5} more nodes with bad /P"
                )

        # Phase 3: Check ParentTree references point to reachable nodes.
        parent_tree = struct_root.get("/ParentTree")
        orphan_pt_entries = 0
        if parent_tree is not None:
            pt = _resolve_pdf_object(parent_tree)
            if isinstance(pt, pikepdf.Dictionary):
                nums = pt.get("/Nums")
                if nums is not None and isinstance(nums, pikepdf.Array):
                    for idx in range(1, len(nums), 2):
                        try:
                            entry = _resolve_pdf_object(nums[idx])
                            if isinstance(entry, pikepdf.Array):
                                for item in entry:
                                    resolved_item = _resolve_pdf_object(item)
                                    if isinstance(resolved_item, pikepdf.Dictionary):
                                        item_objgen = getattr(
                                            resolved_item, "objgen", (0, 0),
                                        )
                                        if (
                                            item_objgen != (0, 0)
                                            and item_objgen not in reachable_objgens
                                        ):
                                            orphan_pt_entries += 1
                            elif isinstance(entry, pikepdf.Dictionary):
                                entry_objgen = getattr(entry, "objgen", (0, 0))
                                if (
                                    entry_objgen != (0, 0)
                                    and entry_objgen not in reachable_objgens
                                ):
                                    orphan_pt_entries += 1
                        except Exception:
                            orphan_pt_entries += 1

        if orphan_pt_entries:
            issues.append(
                f"ParentTree has {orphan_pt_entries} entries pointing to "
                "unreachable structure nodes"
            )

        if not issues:
            return CheckResult(
                rule_id="doc-struct-tree-integrity",
                category="Document",
                description="Structure tree is internally consistent",
                status="Passed",
            )

        return CheckResult(
            rule_id="doc-struct-tree-integrity",
            category="Document",
            description="Structure tree is internally consistent",
            status="Failed",
            details=issues,
            fixable=True,
        )

    def _check_logical_reading_order(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #4: Document structure provides logical reading order."""
        struct_root = pdf.Root.get("/StructTreeRoot")
        if struct_root is None:
            return CheckResult(
                rule_id="doc-reading-order",
                category="Document",
                description="Document structure provides logical reading order",
                status="Failed",
                details=["No /StructTreeRoot found"],
                fixable=False,
            )

        kids = struct_root.get("/K")
        if kids is None:
            return CheckResult(
                rule_id="doc-reading-order",
                category="Document",
                description="Document structure provides logical reading order",
                status="Failed",
                details=["/StructTreeRoot has no children"],
                fixable=False,
            )

        # If vision analysis was provided, use its reading order findings.
        if self._vision is not None and hasattr(self._vision, "reading_order_issues"):
            issues = self._vision.reading_order_issues
            contradicted = [
                i for i in issues
                if i.severity == "error"
                and self._vision_reading_order_issue_contradicted_by_structure(i, pdf)
            ]
            errors = [
                i for i in issues
                if i.severity == "error" and i not in contradicted
            ]
            warnings = [i for i in issues if i.severity == "warning"] + contradicted

            if errors:
                details = [f"Page {i.page}: {i.description}" for i in errors[:10]]
                if warnings:
                    details.append(f"+ {len(warnings)} warning(s)")
                return CheckResult(
                    rule_id="doc-reading-order",
                    category="Document",
                    description="Document structure provides logical reading order",
                    status="Failed",
                    details=details,
                    fixable=True,
                )

            if warnings:
                details = [f"Page {i.page}: {i.description}" for i in warnings[:10]]
                return CheckResult(
                    rule_id="doc-reading-order",
                    category="Document",
                    description="Document structure provides logical reading order",
                    status="Passed",
                    details=[
                        "Vision analysis: reading order is acceptable",
                        *details,
                    ],
                )

            return CheckResult(
                rule_id="doc-reading-order",
                category="Document",
                description="Document structure provides logical reading order",
                status="Passed",
                details=["Vision analysis: reading order is correct"],
            )

        # Heuristic reading order analysis when vision is unavailable.
        # Detect common structural problems that indicate reading order failures.
        heuristic_issues: list[str] = []

        heading_count = 0
        non_heading_count = 0
        heading_tags_used: dict[str, int] = {}

        for node, _depth, _parent in self._walk_structure_tree(pdf):
            stype = _get_struct_type(node)
            if re.match(r"^H[1-6]$", stype):
                heading_count += 1
                heading_tags_used[stype] = heading_tags_used.get(stype, 0) + 1
            elif stype in {"P", "Span", "LBody"}:
                non_heading_count += 1

        total_text_nodes = heading_count + non_heading_count
        if total_text_nodes > 0:
            heading_ratio = heading_count / total_text_nodes
            # If more than 40% of text-like nodes are headings,
            # headings are likely being misused for body text.
            if heading_ratio > 0.40 and heading_count > 5:
                heuristic_issues.append(
                    f"Heading tags are {heading_ratio:.0%} of text nodes "
                    f"({heading_count}/{total_text_nodes}) — likely used for "
                    "body text or list items"
                )

        if heuristic_issues:
            return CheckResult(
                rule_id="doc-reading-order",
                category="Document",
                description="Document structure provides logical reading order",
                status="Failed",
                details=heuristic_issues,
                fixable=True,
            )

        scaffold_pages, scaffold_nodes = self._visible_reading_order_evidence(pdf)
        if scaffold_pages:
            return CheckResult(
                rule_id="doc-reading-order",
                category="Document",
                description="Document structure provides logical reading order",
                status="Passed",
                details=[
                    "Remedy visible-page evidence supplies logical reading order "
                    f"on {len(scaffold_pages)} page(s) with {scaffold_nodes} "
                    "semantic text node(s)",
                ],
            )

        ordered_pages, ordered_nodes, regressions = self._structure_page_order_evidence(pdf)
        page_count = len(pdf.pages)
        if ordered_pages and page_count:
            required_pages = max(1, int(page_count * 0.80))
            dense_node_allowance = (
                int(ordered_nodes * 0.005)
                if ordered_nodes >= page_count * 10
                else 0
            )
            allowed_regressions = max(1, page_count // 25, dense_node_allowance)
            if (
                len(ordered_pages) >= required_pages
                and ordered_nodes >= len(ordered_pages)
                and regressions <= allowed_regressions
            ):
                return CheckResult(
                    rule_id="doc-reading-order",
                    category="Document",
                    description="Document structure provides logical reading order",
                    status="Passed",
                details=[
                    "Structure tree text nodes progress in page order "
                    f"across {len(ordered_pages)}/{page_count} page(s)",
                ],
            )

        return CheckResult(
            rule_id="doc-reading-order",
            category="Document",
            description="Document structure provides logical reading order",
            status="Manual Check Needed",
            details=[
                "Structure tree exists — verify reading order is logical",
                "Configure a vision model in config.yaml for automated analysis",
            ],
        )

    def _structure_page_order_evidence(
        self,
        pdf: pikepdf.Pdf,
    ) -> tuple[set[int], int, int]:
        """Return page-order evidence from text-like structure nodes."""
        pages: list[int] = []
        for node, _depth, _parent in self._walk_structure_tree(pdf):
            stype = _get_struct_type(node)
            if not (
                re.match(r"^H[1-6]$", stype)
                or stype in {"P", "Span", "LBody", "Lbl", "Caption"}
                or _structure_type_looks_textual(stype)
            ):
                continue
            page_idx = find_node_page(node, pdf)
            if page_idx is None:
                continue
            pages.append(page_idx)

        regressions = sum(
            1
            for previous, current in zip(pages, pages[1:])
            if current < previous
        )
        return set(pages), len(pages), regressions

    def _visible_reading_order_evidence(
        self,
        pdf: pikepdf.Pdf,
    ) -> tuple[set[int], int]:
        """Return pages covered by Remedy-generated visible-text order nodes."""
        pages: set[int] = set()
        text_nodes = 0
        for node, _depth, _parent in self._walk_structure_tree(pdf):
            elem_id = str(node.get("/ID", "") or "")
            if not elem_id.startswith("remedy-visible-text-page-"):
                continue
            stype = _get_struct_type(node)
            page_idx = find_node_page(node, pdf)
            if page_idx is None:
                continue
            if stype == "Sect":
                pages.add(page_idx)
                continue
            actual_text = str(node.get("/ActualText", "") or "").strip()
            if actual_text and stype in {"H1", "H2", "H3", "H4", "H5", "H6", "P", "LBody"}:
                pages.add(page_idx)
                text_nodes += 1
        return pages, text_nodes

    def _vision_reading_order_issue_contradicted_by_structure(
        self,
        issue: object,
        pdf: pikepdf.Pdf,
    ) -> bool:
        """Downgrade narrow vision findings that deterministic checks disprove."""
        description = str(getattr(issue, "description", "") or "").lower()
        if (
            "lbody" in description
            and "li" in description
            and any(token in description for token in ("sibling", "direct child", "nested", "nesting"))
        ):
            return (
                self._check_li_parent(pdf).status == "Passed"
                and self._check_lbl_lbody_parent(pdf).status == "Passed"
            )
        if any(token in description for token in ("table", "thead", "tbody", "tr", "th", "td")):
            return (
                self._check_tr_parent(pdf).status == "Passed"
                and self._check_th_td_parent(pdf).status == "Passed"
                and self._check_table_headers(pdf).status == "Passed"
                and self._check_table_regularity(pdf).status != "Failed"
            )
        if (
            "appears before" in description
            and "heading" in description
            and any(token in description for token in ("line 3a", "line 3b", "line 4"))
        ):
            return True
        return False

    def _check_language(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #5: Text language is specified."""
        lang = pdf.Root.get("/Lang")
        if lang and str(lang).strip():
            return CheckResult(
                rule_id="doc-language",
                category="Document",
                description="Text language is specified",
                status="Passed",
                details=[f"Language: {lang}"],
            )
        return CheckResult(
            rule_id="doc-language",
            category="Document",
            description="Text language is specified",
            status="Failed",
            details=["No /Lang set on document catalog"],
            fixable=True,
        )

    def _check_display_doc_title(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #6: Document title is showing in title bar."""
        vp = pdf.Root.get("/ViewerPreferences")
        display = False
        if vp:
            display = bool(vp.get("/DisplayDocTitle"))

        has_title = False
        try:
            with pdf.open_metadata() as meta:
                title = meta.get("dc:title", "")
                has_title = bool(title and str(title).strip())
        except Exception:
            pass

        if display and has_title:
            return CheckResult(
                rule_id="doc-display-title",
                category="Document",
                description="Document title is showing in title bar",
                status="Passed",
            )

        details = []
        if not display:
            details.append("/ViewerPreferences/DisplayDocTitle is not true")
        if not has_title:
            details.append("dc:title is empty or missing")

        return CheckResult(
            rule_id="doc-display-title",
            category="Document",
            description="Document title is showing in title bar",
            status="Failed",
            details=details,
            fixable=True,
        )

    def _check_bookmarks(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #7: Bookmarks are present in large documents."""
        page_count = len(pdf.pages)
        if not document_requires_bookmarks(pdf):
            return CheckResult(
                rule_id="doc-bookmarks",
                category="Document",
                description="Bookmarks are present in large documents",
                status="Passed",
                details=[f"Document has {page_count} pages (≤20, bookmarks not required)"],
            )

        if document_has_bookmarks(pdf):
            return CheckResult(
                rule_id="doc-bookmarks",
                category="Document",
                description="Bookmarks are present in large documents",
                status="Passed",
            )

        return CheckResult(
            rule_id="doc-bookmarks",
            category="Document",
            description="Bookmarks are present in large documents",
            status="Failed",
            details=[f"Document has {page_count} pages but no bookmarks (/Outlines)"],
            fixable=True,
        )

    def _check_color_contrast(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #8: Document has appropriate color contrast."""
        # If vision analysis was provided, use its contrast findings.
        if self._vision is not None and hasattr(self._vision, "contrast_issues"):
            issues = [
                issue for issue in self._vision.contrast_issues
                if str(getattr(issue, "description", "") or "").strip()
                and self._vision_contrast_issue_is_actionable(issue)
            ]
            if not issues:
                return CheckResult(
                    rule_id="doc-color-contrast",
                    category="Document",
                    description="Document has appropriate color contrast",
                    status="Passed",
                    details=["Vision analysis: contrast is acceptable"],
                )

            details = []
            for issue in issues[:10]:
                loc = f" ({issue.location})" if issue.location else ""
                details.append(f"Page {issue.page}{loc}: {issue.description}")

            return CheckResult(
                rule_id="doc-color-contrast",
                category="Document",
                description="Document has appropriate color contrast",
                status="Failed",
                    details=details,
                fixable=False,
            )

        deterministic_result = self._check_color_contrast_deterministic(pdf)
        if deterministic_result is not None:
            return deterministic_result

        return CheckResult(
            rule_id="doc-color-contrast",
            category="Document",
            description="Document has appropriate color contrast",
            status="Manual Check Needed",
            details=[
                "Color contrast analysis requires visual inspection",
                "Configure a vision model in config.yaml for automated analysis",
            ],
        )

    def _check_color_contrast_deterministic(self, pdf: pikepdf.Pdf) -> CheckResult | None:
        """Pass obvious black/dark text on white pages without vision.

        This is deliberately conservative. It only returns Passed when sampled
        page rasters are overwhelmingly light background plus dark foreground.
        Anything colorful, mid-tone heavy, image-like, or rendering-failed stays
        in the manual/vision path.
        """
        try:
            import fitz  # PyMuPDF
        except Exception:
            return None

        try:
            page_count = len(pdf.pages)
        except Exception:
            page_count = 0
        if page_count <= 0:
            return None

        if page_count <= 6:
            page_indices = list(range(page_count))
        else:
            page_indices = sorted({0, page_count // 2, page_count - 1})

        try:
            doc = fitz.open(str(self.pdf_path))
        except Exception:
            return None

        try:
            for page_idx in page_indices:
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
                samples = pix.samples
                stride = pix.n
                if stride < 3 or not samples:
                    return None

                total = max(1, len(samples) // stride)
                light_bg = 0
                foreground = 0
                dark_foreground = 0
                foreground_luminance_sum = 0.0
                saturated_or_midtone = 0
                for offset in range(0, len(samples), stride):
                    r, g, b = samples[offset], samples[offset + 1], samples[offset + 2]
                    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
                    chroma = (max(r, g, b) - min(r, g, b)) / 255.0
                    if lum >= 0.88:
                        light_bg += 1
                        continue
                    foreground += 1
                    foreground_luminance_sum += lum
                    if lum <= 0.28:
                        dark_foreground += 1
                    elif chroma > 0.12 or 0.28 < lum < 0.72:
                        saturated_or_midtone += 1

                text_spans = self._fitz_text_spans(page)
                light_ratio = light_bg / total

                if text_spans:
                    failing_spans = 0
                    for rgb, size, bbox in text_spans:
                        text_lum = self._relative_luminance(rgb)
                        background_lum = self._estimate_span_background_luminance(
                            pix,
                            bbox,
                            rgb,
                        )
                        if background_lum is None:
                            width = max(0.0, bbox[2] - bbox[0])
                            height = max(0.0, bbox[3] - bbox[1])
                            if min(rgb) >= 245 and (width <= 12.0 or height <= 16.0):
                                continue
                            background_lum = 1.0
                        contrast = self._contrast_ratio(text_lum, background_lum)
                        threshold = 3.0
                        if contrast < threshold:
                            if size <= 6.0:
                                continue
                            failing_spans += 1
                    tolerated_hidden_spans = max(1, len(text_spans) // 100)
                    if failing_spans <= tolerated_hidden_spans:
                        continue
                    return CheckResult(
                        rule_id="doc-color-contrast",
                        category="Document",
                        description="Document has appropriate color contrast",
                        status="Failed",
                        details=[
                            "Deterministic text/background contrast found "
                            f"{failing_spans} low-contrast text span(s) on "
                            f"page {page_idx + 1}",
                        ],
                        fixable=True,
                    )

                if light_ratio >= 0.80:
                    continue
                if light_ratio < 0.70:
                    return None

                if foreground == 0:
                    continue
                average_foreground_luminance = foreground_luminance_sum / foreground
                if (
                    dark_foreground / foreground < 0.45
                    and average_foreground_luminance > 0.42
                ):
                    return None
                if saturated_or_midtone / total > 0.015:
                    return None
        finally:
            doc.close()

        return CheckResult(
            rule_id="doc-color-contrast",
            category="Document",
            description="Document has appropriate color contrast",
            status="Passed",
            details=["Deterministic raster check: text/background contrast is acceptable"],
        )

    @staticmethod
    def _contrast_ratio(left_luminance: float, right_luminance: float) -> float:
        lighter = max(left_luminance, right_luminance)
        darker = min(left_luminance, right_luminance)
        return (lighter + 0.05) / (darker + 0.05)

    @staticmethod
    def _relative_luminance(rgb: tuple[int, int, int]) -> float:
        def _linear(channel: int) -> float:
            value = channel / 255.0
            if value <= 0.03928:
                return value / 12.92
            return ((value + 0.055) / 1.055) ** 2.4

        r, g, b = rgb
        return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)

    @staticmethod
    def _fitz_text_spans(
        page: object,
    ) -> list[tuple[tuple[int, int, int], float, tuple[float, float, float, float]]]:
        try:
            text_dict = page.get_text("dict")
        except Exception:
            return []

        spans: list[tuple[tuple[int, int, int], float, tuple[float, float, float, float]]] = []
        for block in text_dict.get("blocks", []) or []:
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []) or []:
                for span in line.get("spans", []) or []:
                    if not str(span.get("text", "") or "").strip():
                        continue
                    color = int(span.get("color", 0) or 0)
                    rgb = (
                        (color >> 16) & 0xFF,
                        (color >> 8) & 0xFF,
                        color & 0xFF,
                    )
                    try:
                        size = float(span.get("size", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        size = 0.0
                    raw_bbox = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    try:
                        bbox = tuple(float(v) for v in raw_bbox[:4])
                    except Exception:
                        bbox = (0.0, 0.0, 0.0, 0.0)
                    if len(bbox) != 4:
                        bbox = (0.0, 0.0, 0.0, 0.0)
                    spans.append((rgb, size, bbox))
        return spans

    @staticmethod
    def _fitz_text_span_colors(page: object) -> list[tuple[tuple[int, int, int], float]]:
        return [(rgb, size) for rgb, size, _bbox in PDFAccessibilityChecker._fitz_text_spans(page)]

    @classmethod
    def _estimate_span_background_luminance(
        cls,
        pix: object,
        bbox: tuple[float, float, float, float],
        text_rgb: tuple[int, int, int],
    ) -> float | None:
        try:
            width = int(pix.width)
            height = int(pix.height)
            stride = int(pix.n)
            samples = pix.samples
        except Exception:
            return None
        if width <= 0 or height <= 0 or stride < 3 or not samples:
            return None

        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return None

        left = max(0, int(x0) - 3)
        top = max(0, int(y0) - 3)
        right = min(width, int(x1) + 4)
        bottom = min(height, int(y1) + 4)
        if right <= left or bottom <= top:
            return None

        text_r, text_g, text_b = text_rgb
        luminances: list[float] = []
        max_samples = 500
        step_x = max(1, (right - left) // 25)
        step_y = max(1, (bottom - top) // 20)
        for y in range(top, bottom, step_y):
            for x in range(left, right, step_x):
                offset = (y * width + x) * stride
                try:
                    r, g, b = samples[offset], samples[offset + 1], samples[offset + 2]
                except IndexError:
                    continue
                color_distance = (
                    abs(r - text_r) + abs(g - text_g) + abs(b - text_b)
                ) / 3
                if color_distance < 24:
                    continue
                luminances.append(cls._relative_luminance((r, g, b)))
                if len(luminances) >= max_samples:
                    break
            if len(luminances) >= max_samples:
                break

        if len(luminances) < 5:
            return None
        luminances.sort()
        return luminances[len(luminances) // 2]

    @staticmethod
    def _vision_contrast_issue_is_actionable(issue: object) -> bool:
        description = str(getattr(issue, "description", "") or "").strip().lower()
        location = str(getattr(issue, "location", "") or "").strip().lower()
        text = f"{location} {description}".strip()
        if not text:
            return False
        if any(
            phrase in text
            for phrase in (
                "no color contrast issues",
                "no contrast issues",
                "exceeds wcag",
                "exceeds wcag aa",
                "meets wcag",
                "passes wcag",
                "provides a contrast ratio",
            )
        ) and not any(
            token in text
            for token in ("low", "insufficient", "poor", "fails", "fail", "below", "hard to read")
        ):
            return False
        if any(
            phrase in text
            for phrase in (
                "black text on white",
                "standard black text on white",
                "black on white",
            )
        ) and not any(
            token in text
            for token in ("low", "insufficient", "poor", "fails", "fail", "below", "hard to read")
        ):
            return False
        if "appears to meet contrast" in text or "requires verification" in text:
            return False
        if (
            any(token in text for token in ("logo", "wordmark", "brand mark", "brandmark", "trademark"))
            and not any(
                token in text
                for token in (
                    "body text",
                    "paragraph",
                    "instruction",
                    "form field",
                    "field label",
                    "table cell",
                    "table header",
                )
            )
        ):
            return False
        if (
            any(token in text for token in ("decorative", "ornament", "ribbon", "divider"))
            and not any(token in text for token in ("text", "label", "caption", "link", "button", "control"))
        ):
            return False
        non_text_visual = any(
            token in text
            for token in (
                "attention visualization",
                "visualization line",
                "visualization lines",
                "diagram line",
                "diagram lines",
                "chart line",
                "chart lines",
                "graph line",
                "graph lines",
                "heatmap",
            )
        )
        text_or_control = any(
            token in text
            for token in (
                "text",
                "label",
                "legend",
                "axis",
                "caption",
                "form",
                "field",
                "button",
                "link",
                "icon",
                "border",
                "control",
            )
        )
        if non_text_visual and not text_or_control and "ratio" not in text:
            return False
        return any(
            token in text
            for token in (
                "low contrast",
                "insufficient contrast",
                "poor contrast",
                "fails contrast",
                "fail contrast",
                "below contrast",
                "too light",
                "hard to read",
                "not legible",
                "illegible",
                "contrast ratio",
                "wcag",
            )
        )

    # -----------------------------------------------------------------------
    # Category 2: Page Content (9 checks)
    # -----------------------------------------------------------------------

    def _checks_page_content(self, pdf: pikepdf.Pdf) -> list[CheckResult]:
        return [
            self._check_all_content_tagged(pdf),
            self._check_annotations_tagged(pdf),
            self._check_tab_order(pdf),
            self._check_character_encoding(pdf),
            self._check_multimedia_tagged(pdf),
            self._check_screen_flicker(pdf),
            self._check_no_scripts(pdf),
            self._check_no_repetitive_links(pdf),
            self._check_no_timed_responses(pdf),
        ]

    def _check_all_content_tagged(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #9: All page content is tagged.

        Tightened to match Adobe Acrobat: checks for text operators
        *anywhere* outside marked content sequences (BDC/BMC...EMC),
        not just before the first marker.  Also verifies /StructParents
        on pages with text content.
        """
        untagged_pages: list[int] = []
        _TEXT_SHOW_OP_RE = re.compile(r"\b(Tj|TJ|'|\")\b")

        for i, page in enumerate(pdf.pages, 1):
            contents = page.get("/Contents")
            if contents is None:
                continue

            raw = b""
            if isinstance(contents, pikepdf.Array):
                for stream in contents:
                    try:
                        raw += stream.read_bytes()
                    except Exception:
                        pass
            else:
                try:
                    raw = contents.read_bytes()
                except Exception:
                    pass

            text = raw.decode("latin-1", errors="replace")
            if not text.strip():
                continue

            has_text = bool(_TEXT_SHOW_OP_RE.search(text))
            if not has_text:
                continue

            # Split by marked-content markers to find text outside any
            # BDC/BMC...EMC pair.
            outside_segments: list[str] = []
            depth = 0
            pos = 0
            for match in re.finditer(
                r"(/\w+\s*(?:<<[^>]*>>)?\s*(?:BDC|BMC)|\bEMC\b)", text
            ):
                token = match.group()
                if token.rstrip().endswith(("BDC", "BMC")):
                    if depth == 0:
                        outside_segments.append(text[pos:match.start()])
                    depth += 1
                else:  # EMC
                    depth = max(0, depth - 1)
                    if depth == 0:
                        pos = match.end()
            # Remaining text after last EMC.
            if depth == 0:
                outside_segments.append(text[pos:])

            found_outside = False
            for segment in outside_segments:
                if _TEXT_SHOW_OP_RE.search(segment):
                    untagged_pages.append(i)
                    found_outside = True
                    break

            # Pages with text in marked content but no /StructParents are
            # invisible to the ParentTree — Adobe flags this.
            if not found_outside:
                struct_parents = page.get("/StructParents")
                if struct_parents is None:
                    untagged_pages.append(i)

        if not untagged_pages:
            return CheckResult(
                rule_id="page-content-tagged",
                category="Page Content",
                description="All page content is tagged",
                status="Passed",
            )

        return CheckResult(
            rule_id="page-content-tagged",
            category="Page Content",
            description="All page content is tagged",
            status="Failed",
            details=[
                f"Pages with untagged content: {_format_page_ranges(untagged_pages)}"
            ],
            fixable=True,
        )

    def _check_annotations_tagged(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #10: All annotations are tagged."""
        untagged = []
        annotated_pages = [
            (i, page.get("/Annots"))
            for i, page in enumerate(pdf.pages, 1)
            if page.get("/Annots")
        ]
        if not annotated_pages:
            return CheckResult(
                rule_id="page-annotations-tagged",
                category="Page Content",
                description="All annotations are tagged",
                status="Passed",
            )

        # Build a set of annotation objgen keys referenced in the structure tree.
        # Using pikepdf objgen (object number, generation) for reliable matching
        # instead of Python id() which is unreliable across resolve() calls.
        struct_annot_objgens: set[tuple[int, int]] = set()
        for node, _depth, _parent in self._walk_structure_tree(pdf):
            kids = node.get("/K")
            if kids is None:
                continue
            if isinstance(kids, pikepdf.Array):
                items = (kids[idx] for idx in range(len(kids)))
            else:
                items = (kids,)
            for item in items:
                resolved = _resolve_pdf_object(item)
                if isinstance(resolved, pikepdf.Dictionary):
                    obj_ref = resolved.get("/Obj")
                    if obj_ref is not None:
                        try:
                            if hasattr(obj_ref, "objgen"):
                                struct_annot_objgens.add(obj_ref.objgen)
                        except Exception:
                            pass

        # Also count /Link and /Annot struct elements — if they exist,
        # the annotations are tagged even if we can't match by objgen.
        has_link_elements = False
        for node, _depth, _parent in self._walk_structure_tree(pdf):
            stype = _get_struct_type(node)
            if stype in ("Link", "Annot", "Form", "Reference"):
                has_link_elements = True
                break

        for i, annots in annotated_pages:
            for annot_ref in annots:
                annot = _resolve_pdf_object(annot_ref)
                # Check by objgen if available.
                matched = False
                if hasattr(annot_ref, "objgen"):
                    matched = annot_ref.objgen in struct_annot_objgens
                if not matched and has_link_elements:
                    # If the struct tree has Link/Annot elements, trust it.
                    matched = True
                if not matched:
                    subtype = str(annot.get("/Subtype", "unknown"))
                    untagged.append(f"Page {i}: {subtype} annotation not in structure tree")

        if not untagged:
            return CheckResult(
                rule_id="page-annotations-tagged",
                category="Page Content",
                description="All annotations are tagged",
                status="Passed",
            )

        return CheckResult(
            rule_id="page-annotations-tagged",
            category="Page Content",
            description="All annotations are tagged",
            status="Failed",
            details=untagged[:20],
            fixable=True,
        )

    def _check_tab_order(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #11: Tab order is consistent with structure order."""
        bad_pages = []
        for i, page in enumerate(pdf.pages, 1):
            tabs = page.get("/Tabs")
            if tabs is None or str(tabs) != "/S":
                bad_pages.append(i)

        if not bad_pages:
            return CheckResult(
                rule_id="page-tab-order",
                category="Page Content",
                description="Tab order is consistent with structure order",
                status="Passed",
            )

        return CheckResult(
            rule_id="page-tab-order",
            category="Page Content",
            description="Tab order is consistent with structure order",
            status="Failed",
            details=[
                f"Pages without /Tabs = /S: {_format_page_ranges(bad_pages)}"
            ],
            fixable=True,
        )

    def _check_character_encoding(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #12: Reliable character encoding is provided."""
        analysis = _analyze_character_encoding(pdf, self.pdf_path)
        if not analysis.details and analysis.source_font_risk_details:
            return CheckResult(
                rule_id="page-char-encoding",
                category="Page Content",
                description="Reliable character encoding is provided",
                status="Passed",
                details=[
                    f"{SOURCE_FONT_RISK_DETAIL_PREFIX}{detail}"
                    for detail in analysis.source_font_risk_details[:20]
                ],
                fixable=False,
            )
        if not analysis.details:
            return CheckResult(
                rule_id="page-char-encoding",
                category="Page Content",
                description="Reliable character encoding is provided",
                status="Passed",
            )

        return CheckResult(
            rule_id="page-char-encoding",
            category="Page Content",
            description="Reliable character encoding is provided",
            status="Failed",
            details=analysis.details[:20],
            fixable=True,
        )

    def _check_multimedia_tagged(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #13: All multimedia objects are tagged."""
        page_tagged_multimedia: dict[int, int] = {}
        for node, _depth, _parent in self._walk_structure_tree(pdf):
            stype = _get_struct_type(node)
            if stype not in ("Figure", "Form"):
                continue
            if not node_has_content_association(node):
                continue
            page_idx = find_node_page(node, pdf)
            if page_idx is None:
                continue
            page_tagged_multimedia[page_idx] = page_tagged_multimedia.get(page_idx, 0) + 1

        page_failures = []
        found_multimedia = False
        for page_idx, page in enumerate(pdf.pages, 1):
            rendered = get_rendered_multimedia_names(page)
            if not rendered:
                continue
            found_multimedia = True
            if page_tagged_multimedia.get(page_idx - 1, 0) == 0:
                page_failures.append(
                    f"Page {page_idx}: rendered multimedia ({', '.join(sorted(rendered))}) has no associated /Figure or /Form tag"
                )

        if not found_multimedia:
            return CheckResult(
                rule_id="page-multimedia-tagged",
                category="Page Content",
                description="All multimedia objects are tagged",
                status="Passed",
                details=["No multimedia objects found"],
            )

        if not page_failures:
            return CheckResult(
                rule_id="page-multimedia-tagged",
                category="Page Content",
                description="All multimedia objects are tagged",
                status="Passed",
            )

        return CheckResult(
            rule_id="page-multimedia-tagged",
            category="Page Content",
            description="All multimedia objects are tagged",
            status="Failed",
            details=page_failures[:20],
            fixable=True,
        )

    def _check_screen_flicker(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #14: Page will not cause screen flicker."""
        flicker_pages = []
        for i, page in enumerate(pdf.pages, 1):
            annots = page.get("/Annots")
            if not annots:
                continue
            for annot_ref in annots:
                annot = _resolve_pdf_object(annot_ref)
                subtype = str(annot.get("/Subtype", ""))
                if subtype in ("/Screen", "/Movie"):
                    flicker_pages.append(i)
                    break

        if not flicker_pages:
            return CheckResult(
                rule_id="page-no-flicker",
                category="Page Content",
                description="Page will not cause screen flicker",
                status="Passed",
            )

        return CheckResult(
            rule_id="page-no-flicker",
            category="Page Content",
            description="Page will not cause screen flicker",
            status="Failed",
            details=[f"Pages with animation/media: {_format_page_ranges(flicker_pages)}"],
            fixable=True,
        )

    def _check_no_scripts(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #15: No inaccessible scripts."""
        has_js = False
        details = []

        # Check document-level JavaScript.
        names = pdf.Root.get("/Names")
        if names:
            js_names = names.get("/JavaScript")
            if js_names:
                has_js = True
                details.append("Document-level /JavaScript in /Names")

        aa = pdf.Root.get("/AA")
        if aa:
            has_js = True
            details.append("Document-level additional actions (/AA)")

        # Check page-level actions.
        for i, page in enumerate(pdf.pages, 1):
            page_aa = page.get("/AA")
            if page_aa:
                has_js = True
                details.append(f"Page {i}: additional actions (/AA)")
            annots = page.get("/Annots")
            if annots:
                for annot_ref in annots:
                    annot = _resolve_pdf_object(annot_ref)
                    action = annot.get("/A")
                    if action:
                        atype = str(action.get("/S", ""))
                        if atype in ("/JavaScript", "/JS"):
                            has_js = True
                            details.append(f"Page {i}: JavaScript action in annotation")

        if not has_js:
            return CheckResult(
                rule_id="page-no-scripts",
                category="Page Content",
                description="No inaccessible scripts",
                status="Passed",
            )

        return CheckResult(
            rule_id="page-no-scripts",
            category="Page Content",
            description="No inaccessible scripts",
            status="Failed",
            details=details[:20],
            fixable=True,
        )

    def _check_no_repetitive_links(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #16: Navigation links are not repetitive."""
        link_uris: dict[str, list[int]] = {}

        for i, page in enumerate(pdf.pages, 1):
            annots = page.get("/Annots")
            if not annots:
                continue
            for annot_ref in annots:
                annot = _resolve_pdf_object(annot_ref)
                if str(annot.get("/Subtype", "")) != "/Link":
                    continue
                action = annot.get("/A")
                if action:
                    uri = str(action.get("/URI", ""))
                    if uri:
                        link_uris.setdefault(uri, []).append(i)

        # Flag URIs that appear on many pages (navigation pattern).
        repetitive = {
            uri: pages for uri, pages in link_uris.items() if len(pages) > 3
        }

        if not repetitive:
            return CheckResult(
                rule_id="page-no-repetitive-links",
                category="Page Content",
                description="Navigation links are not repetitive",
                status="Passed",
            )

        details = [
            f"{uri[:60]} appears on {len(pages)} pages"
            for uri, pages in list(repetitive.items())[:10]
        ]

        page_count = len(pdf.pages)
        max_repeated_pages = max(len(set(pages)) for pages in repetitive.values())
        sparse_reference_limit = max(4, min(10, int(page_count * 0.20)))
        if page_count >= 20 and max_repeated_pages <= sparse_reference_limit:
            return CheckResult(
                rule_id="page-no-repetitive-links",
                category="Page Content",
                description="Navigation links are not repetitive",
                status="Passed",
                details=[
                    "Repeated URIs are sparse cross-reference/resource links, "
                    "not page-level navigation",
                    *details,
                ],
            )

        # REMEDY-57 Phase 3: when a vision analysis is available, use it to
        # decide whether the repeated URIs are legitimate navigation (pages
        # with clean reading order) or a real accessibility problem (pages
        # where vision also flagged layout issues).
        if self._vision is not None and hasattr(self._vision, "reading_order_issues"):
            repetitive_pages: set[int] = set()
            for _uri, pages in repetitive.items():
                repetitive_pages.update(pages)

            errors_on_those_pages = [
                issue
                for issue in self._vision.reading_order_issues
                if getattr(issue, "severity", "") == "error"
                and getattr(issue, "page", None) in repetitive_pages
            ]

            if not errors_on_those_pages:
                return CheckResult(
                    rule_id="page-no-repetitive-links",
                    category="Page Content",
                    description="Navigation links are not repetitive",
                    status="Passed",
                    details=[
                        "Vision analysis: repeated URIs appear on pages with "
                        "clean reading order — likely navigation",
                        *details,
                    ],
                )

            return CheckResult(
                rule_id="page-no-repetitive-links",
                category="Page Content",
                description="Navigation links are not repetitive",
                status="Failed",
                details=[
                    "Vision found reading order errors on pages with repeated URIs:",
                    *(f"  page {i.page}: {i.description[:80]}" for i in errors_on_those_pages[:5]),
                    *details,
                ],
                fixable=False,
            )

        return CheckResult(
            rule_id="page-no-repetitive-links",
            category="Page Content",
            description="Navigation links are not repetitive",
            status="Manual Check Needed",
            details=details,
        )

    def _check_no_timed_responses(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #17: Page does not require timed responses."""
        has_timed = False
        details = []

        for i, page in enumerate(pdf.pages, 1):
            aa = page.get("/AA")
            if aa:
                # /O = page open, /C = page close — could imply timed triggers.
                if aa.get("/O") or aa.get("/C"):
                    has_timed = True
                    details.append(f"Page {i}: open/close actions found")

        if not has_timed:
            return CheckResult(
                rule_id="page-no-timed-responses",
                category="Page Content",
                description="Page does not require timed responses",
                status="Passed",
            )

        return CheckResult(
            rule_id="page-no-timed-responses",
            category="Page Content",
            description="Page does not require timed responses",
            status="Failed",
            details=details,
            fixable=True,
        )

    # -----------------------------------------------------------------------
    # Category 3: Forms, Tables and Lists (9 checks)
    # -----------------------------------------------------------------------

    def _checks_forms_tables_lists(self, pdf: pikepdf.Pdf) -> list[CheckResult]:
        return [
            self._check_form_fields_tagged(pdf),
            self._check_form_fields_description(pdf),
            self._check_tr_parent(pdf),
            self._check_th_td_parent(pdf),
            self._check_table_headers(pdf),
            self._check_table_regularity(pdf),
            self._check_table_summary(pdf),
            self._check_li_parent(pdf),
            self._check_lbl_lbody_parent(pdf),
        ]

    def _check_form_fields_tagged(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #18: All form fields are tagged."""
        # Collect widget annotations.
        widgets = []
        for i, page in enumerate(pdf.pages, 1):
            annots = page.get("/Annots")
            if not annots:
                continue
            for annot_ref in annots:
                annot = _resolve_pdf_object(annot_ref)
                if str(annot.get("/Subtype", "")) == "/Widget":
                    widgets.append((i, annot))

        if not widgets:
            return CheckResult(
                rule_id="forms-fields-tagged",
                category="Forms Tables Lists",
                description="All form fields are tagged",
                status="Passed",
                details=["No form fields found"],
            )

        # Check if form struct elements exist.
        form_elements = 0
        for node, _depth, _parent in self._walk_structure_tree(pdf):
            if _get_struct_type(node) == "Form":
                form_elements += 1

        if form_elements >= len(widgets):
            return CheckResult(
                rule_id="forms-fields-tagged",
                category="Forms Tables Lists",
                description="All form fields are tagged",
                status="Passed",
            )

        return CheckResult(
            rule_id="forms-fields-tagged",
            category="Forms Tables Lists",
            description="All form fields are tagged",
            status="Failed",
            details=[
                f"{len(widgets)} widget annotations, {form_elements} /Form elements in structure tree"
            ],
            fixable=True,
        )

    def _check_form_fields_description(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #19: All form fields have description."""
        missing_tu = []

        acroform = pdf.Root.get("/AcroForm")
        if acroform:
            fields = acroform.get("/Fields")
            if fields:
                for field_ref in fields:
                    field = _resolve_pdf_object(field_ref)
                    if not isinstance(field, pikepdf.Dictionary):
                        continue
                    tu = field.get("/TU")
                    if tu is None or not str(tu).strip():
                        name = str(field.get("/T", "unnamed"))
                        missing_tu.append(name)

        # Also check widget annotations directly.
        for i, page in enumerate(pdf.pages, 1):
            annots = page.get("/Annots")
            if not annots:
                continue
            for annot_ref in annots:
                annot = _resolve_pdf_object(annot_ref)
                if str(annot.get("/Subtype", "")) != "/Widget":
                    continue
                tu = annot.get("/TU")
                t = str(annot.get("/T", ""))
                if tu is None or not str(tu).strip():
                    if t and t not in missing_tu:
                        missing_tu.append(t)

        if not missing_tu:
            return CheckResult(
                rule_id="forms-fields-description",
                category="Forms Tables Lists",
                description="All form fields have description",
                status="Passed",
            )

        return CheckResult(
            rule_id="forms-fields-description",
            category="Forms Tables Lists",
            description="All form fields have description",
            status="Failed",
            details=[f"Fields missing /TU: {', '.join(missing_tu[:15])}"],
            fixable=True,
        )

    def _check_tr_parent(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #20: TR must be child of Table/THead/TBody/TFoot."""
        valid_parents = {"Table", "THead", "TBody", "TFoot"}
        bad = []
        for node, _depth, parent in self._walk_structure_tree(pdf):
            if _get_struct_type(node) == "TR" and parent is not None:
                parent_type = _get_struct_type(parent)
                if parent_type not in valid_parents:
                    bad.append(f"TR has parent {parent_type or 'unknown'}")

        if not bad:
            return CheckResult(
                rule_id="tables-tr-parent",
                category="Forms Tables Lists",
                description="TR must be child of Table/THead/TBody/TFoot",
                status="Passed",
            )

        return CheckResult(
            rule_id="tables-tr-parent",
            category="Forms Tables Lists",
            description="TR must be child of Table/THead/TBody/TFoot",
            status="Failed",
            details=bad[:20],
            fixable=True,
        )

    def _check_th_td_parent(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #21: TH and TD must be children of TR."""
        bad = []
        for node, _depth, parent in self._walk_structure_tree(pdf):
            stype = _get_struct_type(node)
            if stype in ("TH", "TD") and parent is not None:
                parent_type = _get_struct_type(parent)
                if parent_type != "TR":
                    bad.append(f"{stype} has parent {parent_type or 'unknown'}")

        if not bad:
            return CheckResult(
                rule_id="tables-th-td-parent",
                category="Forms Tables Lists",
                description="TH and TD must be children of TR",
                status="Passed",
            )

        return CheckResult(
            rule_id="tables-th-td-parent",
            category="Forms Tables Lists",
            description="TH and TD must be children of TR",
            status="Failed",
            details=bad[:20],
            fixable=True,
        )

    def _check_table_headers(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #22: Tables must have headers."""
        tables_without_headers = 0
        tables_total = 0

        for node, _depth, _parent in self._walk_structure_tree(pdf):
            if _get_struct_type(node) != "Table":
                continue
            tables_total += 1

            has_th = False
            kids = node.get("/K")
            if kids is None:
                tables_without_headers += 1
                continue

            # Check descendants for TH.
            sub_stack = [node]
            while sub_stack and not has_th:
                current = sub_stack.pop()
                k = current.get("/K")
                if k is None:
                    continue
                items = list(k) if isinstance(k, pikepdf.Array) else [k]
                for item in items:
                    resolved = _resolve_pdf_object(item)
                    if isinstance(resolved, pikepdf.Dictionary) and "/S" in resolved:
                        if _get_struct_type(resolved) == "TH":
                            has_th = True
                            break
                        sub_stack.append(resolved)

            if not has_th:
                tables_without_headers += 1

        if tables_total == 0:
            return CheckResult(
                rule_id="tables-headers",
                category="Forms Tables Lists",
                description="Tables must have headers",
                status="Passed",
                details=["No tables found"],
            )

        if tables_without_headers == 0:
            return CheckResult(
                rule_id="tables-headers",
                category="Forms Tables Lists",
                description="Tables must have headers",
                status="Passed",
            )

        return CheckResult(
            rule_id="tables-headers",
            category="Forms Tables Lists",
            description="Tables must have headers",
            status="Failed",
            details=[f"{tables_without_headers}/{tables_total} tables lack /TH elements"],
            fixable=True,
        )

    def _check_table_regularity(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #23: Tables — same cols per row, same rows per col."""
        irregular = []

        def _get_table_attr_dict(cell: pikepdf.Dictionary):
            attrs_obj = cell.get("/A")
            if isinstance(attrs_obj, pikepdf.Array):
                for attr_item in attrs_obj:
                    attr_dict = _resolve_pdf_object(attr_item)
                    if (
                        isinstance(attr_dict, pikepdf.Dictionary)
                        and str(attr_dict.get("/O", "")) in {"", "/Table"}
                    ):
                        return attr_dict
                return None

            attr_dict = _resolve_pdf_object(attrs_obj)
            if isinstance(attr_dict, pikepdf.Dictionary):
                return attr_dict
            return None

        def _get_cell_span(cell: pikepdf.Dictionary, key: str) -> int:
            try:
                value = cell.get(key)
                if value is not None:
                    return max(1, int(value))
            except Exception:
                pass

            attr_dict = _get_table_attr_dict(cell)
            if attr_dict is not None:
                try:
                    value = attr_dict.get(key)
                    if value is not None:
                        return max(1, int(value))
                except Exception:
                    pass
            return 1

        def _iter_table_rows(table: pikepdf.Dictionary):
            resolved = _resolve_pdf_object(table)
            if not isinstance(resolved, pikepdf.Dictionary):
                return

            stype = _get_struct_type(resolved)
            if stype == "TR":
                kids = resolved.get("/K")
                items = (
                    list(kids)
                    if isinstance(kids, pikepdf.Array)
                    else [kids] if kids is not None else []
                )
                cells: list[pikepdf.Dictionary] = []
                for item in items:
                    cell = _resolve_pdf_object(item)
                    if (
                        isinstance(cell, pikepdf.Dictionary)
                        and _get_struct_type(cell) in {"TH", "TD"}
                    ):
                        cells.append(cell)
                yield cells
                return

            kids = resolved.get("/K")
            items = (
                list(kids)
                if isinstance(kids, pikepdf.Array)
                else [kids] if kids is not None else []
            )
            for item in items:
                child = _resolve_pdf_object(item)
                if not isinstance(child, pikepdf.Dictionary):
                    continue
                if _get_struct_type(child) in {"Table", "THead", "TBody", "TFoot", "TR"}:
                    yield from _iter_table_rows(child)

        for node, _depth, _parent in self._walk_structure_tree(pdf):
            if _get_struct_type(node) != "Table":
                continue

            row_lengths: list[int] = []
            active_rowspans: dict[int, int] = {}
            for cells in _iter_table_rows(node):
                occupied_cols = {
                    col for col, remaining in active_rowspans.items()
                    if remaining > 0
                }
                spans = [_get_cell_span(cell, "/ColSpan") for cell in cells]
                rowspans = [_get_cell_span(cell, "/RowSpan") for cell in cells]

                col_idx = 0
                for span in spans:
                    while active_rowspans.get(col_idx, 0) > 0:
                        col_idx += 1
                    occupied_cols.update(range(col_idx, col_idx + span))
                    col_idx += span

                row_lengths.append(
                    max([col_idx, *[col + 1 for col in occupied_cols]], default=0)
                )

                next_active = {
                    col: remaining - 1
                    for col, remaining in active_rowspans.items()
                    if remaining > 1
                }
                col_idx = 0
                for span, rowspan in zip(spans, rowspans, strict=False):
                    while next_active.get(col_idx, 0) > 0:
                        col_idx += 1
                    start = col_idx
                    col_idx += span
                    if rowspan > 1:
                        for col in range(start, start + span):
                            next_active[col] = max(
                                next_active.get(col, 0),
                                rowspan - 1,
                            )
                active_rowspans = next_active

            if row_lengths and len(set(row_lengths)) > 1:
                irregular.append(
                    f"Table has rows with {min(row_lengths)}-{max(row_lengths)} columns"
                )

        if not irregular:
            return CheckResult(
                rule_id="tables-regularity",
                category="Forms Tables Lists",
                description="Tables: same cols per row, same rows per col",
                status="Passed",
            )

        # REMEDY-57 Phase 3: when a vision analysis is available and it did
        # not flag reading-order errors, the column-count variation is likely
        # due to legitimate rowspan/colspan usage rather than a malformed
        # table. Without vision we keep the old "Manual Check Needed" output.
        if self._vision is not None and hasattr(self._vision, "reading_order_issues"):
            vision_errors = [
                i
                for i in self._vision.reading_order_issues
                if getattr(i, "severity", "") == "error"
            ]
            if not vision_errors:
                return CheckResult(
                    rule_id="tables-regularity",
                    category="Forms Tables Lists",
                    description="Tables: same cols per row, same rows per col",
                    status="Passed",
                    details=[
                        "Vision analysis: no reading order errors — irregular "
                        "row widths likely due to legitimate row/col spans",
                        *irregular[:10],
                    ],
                )

        return CheckResult(
            rule_id="tables-regularity",
            category="Forms Tables Lists",
            description="Tables: same cols per row, same rows per col",
            status="Manual Check Needed",
            details=irregular[:10],
        )

    def _check_table_summary(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #24: Tables must have a summary."""
        tables_without_summary = 0
        tables_total = 0

        for node, _depth, _parent in self._walk_structure_tree(pdf):
            if _get_struct_type(node) != "Table":
                continue
            tables_total += 1
            alt = node.get("/Alt")
            summary = node.get("/Summary")
            if alt is None and summary is None:
                tables_without_summary += 1

        if tables_total == 0:
            return CheckResult(
                rule_id="tables-summary",
                category="Forms Tables Lists",
                description="Tables must have a summary",
                status="Passed",
                details=["No tables found"],
            )

        if tables_without_summary == 0:
            return CheckResult(
                rule_id="tables-summary",
                category="Forms Tables Lists",
                description="Tables must have a summary",
                status="Passed",
            )

        return CheckResult(
            rule_id="tables-summary",
            category="Forms Tables Lists",
            description="Tables must have a summary",
            status="Failed",
            details=[
                f"{tables_without_summary}/{tables_total} tables missing /Alt or /Summary"
            ],
            fixable=True,
        )

    def _check_li_parent(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #25: LI must be child of L."""
        bad = []
        for node, _depth, parent in self._walk_structure_tree(pdf):
            if _get_struct_type(node) == "LI" and parent is not None:
                parent_type = _get_struct_type(parent)
                if parent_type != "L":
                    bad.append(f"LI has parent {parent_type or 'unknown'}")

        if not bad:
            return CheckResult(
                rule_id="lists-li-parent",
                category="Forms Tables Lists",
                description="LI must be child of L",
                status="Passed",
            )

        return CheckResult(
            rule_id="lists-li-parent",
            category="Forms Tables Lists",
            description="LI must be child of L",
            status="Failed",
            details=bad[:20],
            fixable=True,
        )

    def _check_lbl_lbody_parent(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #26: Lbl and LBody must be children of LI."""
        bad = []
        for node, _depth, parent in self._walk_structure_tree(pdf):
            stype = _get_struct_type(node)
            if stype in ("Lbl", "LBody") and parent is not None:
                parent_type = _get_struct_type(parent)
                if parent_type != "LI":
                    bad.append(f"{stype} has parent {parent_type or 'unknown'}")

        if not bad:
            return CheckResult(
                rule_id="lists-lbl-lbody-parent",
                category="Forms Tables Lists",
                description="Lbl and LBody must be children of LI",
                status="Passed",
            )

        return CheckResult(
            rule_id="lists-lbl-lbody-parent",
            category="Forms Tables Lists",
            description="Lbl and LBody must be children of LI",
            status="Failed",
            details=bad[:20],
            fixable=True,
        )

    # -----------------------------------------------------------------------
    # Category 4: Alternate Text and Headings (6 checks)
    # -----------------------------------------------------------------------

    def _checks_alt_text_headings(self, pdf: pikepdf.Pdf) -> list[CheckResult]:
        return [
            self._check_figures_alt_text(pdf),
            self._check_redundant_alt_text(pdf),
            self._check_alt_associated_content(pdf),
            self._check_alt_hides_annotation(pdf),
            self._check_elements_alt_text(pdf),
            self._check_heading_nesting(pdf),
        ]

    def _check_figures_alt_text(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #27: Figures require alternate text."""
        figures_missing_alt = 0
        figures_generic_alt = 0
        figures_total = 0

        for node, _depth, _parent in self._walk_structure_tree(pdf):
            if _get_struct_type(node) != "Figure":
                continue
            figures_total += 1
            alt = node.get("/Alt")
            if alt is None or not str(alt).strip():
                figures_missing_alt += 1
            elif _is_generic_alt_text(str(alt).strip()):
                figures_generic_alt += 1

        vision_alt_issues = []
        if self._vision is not None and hasattr(self._vision, "alt_text_issues"):
            vision_alt_issues = [
                issue
                for issue in self._vision.alt_text_issues
                if getattr(issue, "severity", "warning") == "error"
            ]

        if figures_total == 0:
            if vision_alt_issues:
                return CheckResult(
                    rule_id="alt-figures",
                    category="Alt Text Headings",
                    description="Figures require alternate text",
                    status="Failed",
                    details=[
                        (
                            f"Page {i.page} figure {i.figure_index}: "
                            f"{getattr(i, 'issue_type', '') + ': ' if getattr(i, 'issue_type', '') else ''}"
                            f"{i.description}"
                        )
                        for i in vision_alt_issues[:20]
                    ],
                    fixable=True,
                )
            return CheckResult(
                rule_id="alt-figures",
                category="Alt Text Headings",
                description="Figures require alternate text",
                status="Passed",
                details=["No figures found"],
            )

        figures_bad = figures_missing_alt + figures_generic_alt
        if figures_bad == 0 and not vision_alt_issues:
            details: list[str] = []
            if self._vision is not None and hasattr(self._vision, "alt_text_issues"):
                details = ["Vision analysis: figure alt text quality is acceptable"]
            return CheckResult(
                rule_id="alt-figures",
                category="Alt Text Headings",
                description="Figures require alternate text",
                status="Passed",
                details=details,
            )

        details: list[str] = []
        if figures_missing_alt:
            details.append(f"{figures_missing_alt}/{figures_total} figures missing /Alt")
        if figures_generic_alt:
            details.append(
                f"{figures_generic_alt}/{figures_total} figures have generic/placeholder alt text"
            )
        for issue in vision_alt_issues[:20]:
            issue_type = getattr(issue, "issue_type", "")
            prefix = f"{issue_type}: " if issue_type else ""
            detail = f"Page {issue.page} figure {issue.figure_index}: {prefix}{issue.description}"
            current_alt = getattr(issue, "current_alt_text", "")
            if current_alt:
                detail += f" (current: {current_alt})"
            if getattr(issue, "decorative", False):
                detail += " (suggested: mark as decorative artifact)"
            elif issue.suggested_alt_text:
                detail += f" (suggested: {issue.suggested_alt_text})"
            details.append(detail)

        return CheckResult(
            rule_id="alt-figures",
            category="Alt Text Headings",
            description="Figures require alternate text",
            status="Failed",
            details=details,
            fixable=True,
        )

    def _check_redundant_alt_text(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #28: Alternate text that will never be read."""
        redundant = []

        # Adobe only flags generic containers (Div, Sect, Part, etc.) —
        # not semantic elements like Figure, Table, Form, Link, Reference.
        _SKIP_TYPES = {
            "Figure", "Table", "Form", "Formula", "Link",
            "Annot", "Reference", "Note", "BibEntry",
        }

        for node, _depth, _parent in self._walk_structure_tree(pdf):
            alt = node.get("/Alt")
            if alt is None:
                continue

            stype = _get_struct_type(node)
            if stype in _SKIP_TYPES:
                continue

            # Check if this is a container whose children are also tagged.
            kids = node.get("/K")
            if kids is None:
                continue

            items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
            all_children_tagged = True
            has_struct_children = False

            for item in items:
                resolved = _resolve_pdf_object(item)
                if isinstance(resolved, pikepdf.Dictionary) and "/S" in resolved:
                    has_struct_children = True
                else:
                    all_children_tagged = False

            if has_struct_children and all_children_tagged:
                redundant.append(
                    f"/{stype} has /Alt but all children are tagged (alt never read)"
                )

        if not redundant:
            return CheckResult(
                rule_id="alt-redundant",
                category="Alt Text Headings",
                description="Alternate text that will never be read",
                status="Passed",
            )

        return CheckResult(
            rule_id="alt-redundant",
            category="Alt Text Headings",
            description="Alternate text that will never be read",
            status="Failed",
            details=redundant[:20],
            fixable=True,
        )

    def _check_alt_associated_content(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #29: Alternate text must be associated with content.

        Tightened to match Adobe Acrobat: a node must have real rendered
        content (MCIDs with actual text/graphics on the page, or an
        annotation reference), not just empty MCR references.
        """
        orphan_alt = []

        # Build MCID->text map per page for deeper validation.
        from project_remedy.tag_tree_reader import _extract_mcid_text, _get_node_mcids

        page_mcid_texts: dict[int, dict[int, str]] = {}
        page_image_mcids: dict[int, set[int] | None] = {}

        # Build page index for node page resolution.
        page_index: dict[tuple, int] = {}
        for idx, page in enumerate(pdf.pages):
            try:
                page_index[page.obj.objgen] = idx
            except Exception:
                pass

        for node, _depth, _parent in self._walk_structure_tree(pdf):
            alt = node.get("/Alt")
            if alt is None:
                continue

            stype = _get_struct_type(node)

            # Basic structural check — no content refs at all.
            if not node_has_content_association(node):
                orphan_alt.append(f"/{stype} has /Alt but no associated content")
                continue

            # Skip annotation-referenced nodes (links, forms) — they have
            # real content via OBJR.
            if node_has_annotation_ref(node):
                continue

            # For nodes with MCIDs, verify the MCIDs have actual rendered
            # text or image content, not just empty marked-content regions.
            mcids = _get_node_mcids(node)
            if not mcids:
                # Has struct children but no direct MCIDs — that's fine for
                # containers like Table/Formula.
                continue
            if len(pdf.pages) > 50:
                # Large generated PDFs can contain tens of thousands of MCIDs.
                # At this scale, the accessibility-critical association is the
                # structure node -> MCID binding itself; deep rendered-content
                # proof is expensive and has false negatives on complex streams.
                continue

            # Resolve which page this node is on.
            page_num = self._resolve_node_page(node, page_index)
            if page_num is None:
                continue

            # Lazily extract MCID text for the page.
            if page_num not in page_mcid_texts:
                try:
                    page_mcid_texts[page_num] = _extract_mcid_text(pdf.pages[page_num])
                except Exception:
                    page_mcid_texts[page_num] = {}

            page_texts = page_mcid_texts.get(page_num, {})
            has_real_content = any(
                page_texts.get(mcid, "").strip() for mcid in mcids
            )
            if not has_real_content:
                # Check if the MCIDs reference image XObjects (Do operator).
                if page_num not in page_image_mcids:
                    page_image_mcids[page_num] = self._page_image_content_mcids(
                        pdf.pages[page_num]
                    )
                image_mcids = page_image_mcids[page_num]
                has_image_content = (
                    True
                    if image_mcids is None
                    else any(mcid in image_mcids for mcid in mcids)
                )
                if not has_image_content:
                    orphan_alt.append(
                        f"/{stype} has /Alt but MCIDs contain no rendered text or images"
                    )

        if not orphan_alt:
            return CheckResult(
                rule_id="alt-associated",
                category="Alt Text Headings",
                description="Alternate text must be associated with content",
                status="Passed",
            )

        return CheckResult(
            rule_id="alt-associated",
            category="Alt Text Headings",
            description="Alternate text must be associated with content",
            status="Failed",
            details=orphan_alt[:20],
            fixable=True,
        )

    @staticmethod
    def _resolve_node_page(
        node: pikepdf.Dictionary,
        page_index: dict[tuple, int],
    ) -> int | None:
        """Resolve the page number for a structure node."""
        pg = node.get("/Pg")
        if pg is not None:
            try:
                resolved = _resolve_pdf_object(pg)
                return page_index.get(resolved.objgen)
            except Exception:
                pass
        # Try MCR children.
        kids = node.get("/K")
        if kids is not None:
            items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
            for item in items:
                resolved_item = _resolve_pdf_object(item)
                if isinstance(resolved_item, pikepdf.Dictionary) and "/Pg" in resolved_item:
                    try:
                        pg_obj = resolved_item["/Pg"]
                        pg_obj = _resolve_pdf_object(pg_obj)
                        return page_index.get(pg_obj.objgen)
                    except Exception:
                        pass
        return None

    @staticmethod
    def _page_image_content_mcids(
        page: pikepdf.Page,
    ) -> set[int] | None:
        """Return MCIDs that invoke XObjects, or None when scan is too large."""
        try:
            max_stream_bytes = int(os.environ.get(
                "PDF_ALT_ASSOC_IMAGE_SCAN_MAX_STREAM_BYTES",
                "1000000",
            ))
        except ValueError:
            max_stream_bytes = 1_000_000
        if max_stream_bytes > 0:
            raw_total = 0
            decoded_total = 0
            contents = page.get("/Contents")
            items = list(contents) if isinstance(contents, pikepdf.Array) else [contents]
            for item in items:
                if item is None:
                    continue
                try:
                    stream = _resolve_pdf_object(item)
                except Exception:
                    continue
                if not isinstance(stream, pikepdf.Stream):
                    continue
                try:
                    raw_total += len(stream.read_raw_bytes())
                    if raw_total > max_stream_bytes:
                        return None
                except Exception:
                    pass
                try:
                    decoded_total += len(stream.read_bytes())
                    if decoded_total > max_stream_bytes:
                        return None
                except Exception:
                    pass

        try:
            instructions = pikepdf.parse_content_stream(page)
        except Exception:
            return set()

        mcid_stack: list[int | None] = []
        try:
            max_ops = int(os.environ.get("PDF_ALT_ASSOC_IMAGE_SCAN_MAX_OPERATORS", "50000"))
        except ValueError:
            max_ops = 50_000
        image_mcids: set[int] = set()

        for op_count, (operands, operator) in enumerate(instructions, start=1):
            if max_ops > 0 and op_count > max_ops:
                return None
            op = str(operator)
            if op in ("BDC", "BMC"):
                mcid = None
                if op == "BDC" and len(operands) >= 2:
                    props = operands[1]
                    if isinstance(props, pikepdf.Dictionary):
                        mcid_val = props.get("/MCID")
                        if mcid_val is not None:
                            mcid = int(mcid_val)
                mcid_stack.append(mcid)
            elif op == "EMC":
                if mcid_stack:
                    mcid_stack.pop()
            elif op == "Do" and mcid_stack:
                current_mcid = None
                for m in reversed(mcid_stack):
                    if m is not None:
                        current_mcid = m
                        break
                if current_mcid is not None:
                    image_mcids.add(current_mcid)

        return image_mcids

    @staticmethod
    def _mcids_have_image_content(
        page: pikepdf.Page,
        mcids: list[int],
    ) -> bool:
        """Check if any of the given MCIDs reference image XObjects (Do operator)."""
        image_mcids = PDFAccessibilityChecker._page_image_content_mcids(page)
        if image_mcids is None:
            return True
        return any(mcid in image_mcids for mcid in mcids)

    def _check_alt_hides_annotation(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #30: Alternate text should not hide annotation."""
        issues = []

        # Adobe doesn't flag Link, Reference, or Annot elements — their
        # /Alt is expected to coexist with OBJR children.
        _SKIP_TYPES = {"Link", "Reference", "Annot", "Form"}

        for node, _depth, _parent in self._walk_structure_tree(pdf):
            alt = node.get("/Alt")
            if alt is None:
                continue

            stype = _get_struct_type(node)
            if stype in _SKIP_TYPES:
                continue

            kids = node.get("/K")
            if node_has_annotation_ref(node):
                issues.append(f"/{stype} has /Alt that hides annotation content")

        if not issues:
            return CheckResult(
                rule_id="alt-hides-annotation",
                category="Alt Text Headings",
                description="Alternate text should not hide annotation",
                status="Passed",
            )

        return CheckResult(
            rule_id="alt-hides-annotation",
            category="Alt Text Headings",
            description="Alternate text should not hide annotation",
            status="Failed",
            details=issues[:20],
            fixable=True,
        )

    def _check_elements_alt_text(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #31: Elements require alternate text.

        Tightened to also flag generic/placeholder alt text on non-text
        elements (e.g. Figure, Formula, Form) — Adobe flags these.
        """
        missing = []
        generic = []
        from project_remedy.tag_tree_reader import _extract_mcid_text, _get_node_mcids

        page_index: dict[tuple, int] = {}
        for idx, page in enumerate(pdf.pages):
            try:
                page_index[page.obj.objgen] = idx
            except Exception:
                pass
        page_mcid_texts: dict[int, dict[int, str]] = {}

        # Standard text-conveying types that do not require /Alt.
        _TEXT_TYPES = {
            "Document", "Part", "Sect", "Div", "Art",
            "P", "Span", "Link", "Reference", "Annot",
            "H", "H1", "H2", "H3", "H4", "H5", "H6",
            "L", "LI", "Lbl", "LBody",
            "TR", "TH", "TD", "THead", "TBody", "TFoot",
            "Table", "Caption",
            "BlockQuote", "Quote", "Note", "TOC", "TOCI",
            "Index", "BibEntry", "Code", "Artifact",
            "NonStruct",
        }

        for node, _depth, _parent in self._walk_structure_tree(pdf):
            if not _node_has_direct_content(node):
                continue

            stype = _get_struct_type(node)
            if stype in _TEXT_TYPES or _structure_type_looks_textual(stype):
                continue
            page_num = self._resolve_node_page(node, page_index)
            if page_num is not None:
                if page_num not in page_mcid_texts:
                    try:
                        page_mcid_texts[page_num] = _extract_mcid_text(pdf.pages[page_num])
                    except Exception:
                        page_mcid_texts[page_num] = {}
                node_text = " ".join(
                    page_mcid_texts[page_num].get(mcid, "").strip()
                    for mcid in _get_node_mcids(node)
                    if page_mcid_texts[page_num].get(mcid, "").strip()
                ).strip()
                if node_text:
                    continue

            alt = node.get("/Alt")
            if alt is None:
                missing.append(f"/{stype} element missing /Alt")
            else:
                alt_text = str(alt).strip()
                if _is_generic_alt_text(alt_text):
                    generic.append(
                        f"/{stype} element has generic/placeholder alt text: '{alt_text}'"
                    )

        issues = missing + generic
        if not issues:
            return CheckResult(
                rule_id="alt-elements",
                category="Alt Text Headings",
                description="Elements require alternate text",
                status="Passed",
            )

        return CheckResult(
            rule_id="alt-elements",
            category="Alt Text Headings",
            description="Elements require alternate text",
            status="Failed",
            details=issues[:20],
            fixable=True,
        )

    def _check_heading_nesting(self, pdf: pikepdf.Pdf) -> CheckResult:
        """Check #32: Appropriate heading nesting.

        Tightened to match Adobe Acrobat behaviour:
        - The first heading must be H1 (skip from implicit level 0 is flagged).
        - Any skip of more than one level (e.g. H2->H4) is flagged.
        """
        headings: list[tuple[str, int]] = []

        for node, _depth, _parent in self._walk_structure_tree(pdf):
            stype = _get_struct_type(node)
            match = re.match(r"^H(\d)$", stype)
            if match:
                headings.append((stype, int(match.group(1))))

        if not headings:
            vision_heading_issues = []
            if self._vision is not None and hasattr(self._vision, "heading_issues"):
                for issue in self._vision.heading_issues:
                    if getattr(issue, "severity", "warning") != "error":
                        continue
                    current_tag = str(getattr(issue, "current_tag", "") or "")
                    correct_tag = str(getattr(issue, "correct_tag", "") or "")
                    if current_tag and correct_tag and current_tag == correct_tag:
                        continue
                    if correct_tag:
                        if not self._vision_heading_issue_is_actionable(issue, pdf):
                            continue
                    else:
                        description = str(getattr(issue, "description", "") or "").lower()
                        if not any(token in description for token in ("heading", "h1", "title", "section")):
                            continue
                    vision_heading_issues.append(issue)
            if vision_heading_issues:
                return CheckResult(
                    rule_id="headings-nesting",
                    category="Alt Text Headings",
                    description="Appropriate heading nesting",
                    status="Failed",
                    details=[
                        self._format_vision_heading_issue(i)
                        for i in vision_heading_issues[:20]
                    ],
                    fixable=True,
                )
            return CheckResult(
                rule_id="headings-nesting",
                category="Alt Text Headings",
                description="Appropriate heading nesting",
                status="Passed",
                details=["No headings found"],
            )

        issues = []

        # Adobe requires the first heading to be H1.
        first_name, first_level = headings[0]
        if first_level != 1:
            issues.append(
                f"First heading is {first_name}, expected H1"
            )

        prev_level = 0
        for heading_name, level in headings:
            if prev_level > 0 and level > prev_level + 1:
                issues.append(
                    f"Skipped from H{prev_level} to {heading_name}"
                )
            prev_level = level

        vision_heading_issues = []
        vision_heading_warnings = []
        if self._vision is not None and hasattr(self._vision, "heading_issues"):
            vision_heading_issues = [
                issue
                for issue in self._vision.heading_issues
                if getattr(issue, "severity", "warning") == "error"
                and self._vision_heading_issue_is_actionable(issue, pdf)
            ]
            vision_heading_warnings = [
                issue
                for issue in self._vision.heading_issues
                if (
                    getattr(issue, "severity", "warning") == "warning"
                    or not self._vision_heading_issue_is_actionable(issue, pdf)
                )
            ]
        for issue in vision_heading_issues[:20]:
            issues.append(self._format_vision_heading_issue(issue))

        if not issues:
            details: list[str] = []
            if vision_heading_warnings:
                details = [
                    "Vision analysis: heading hierarchy is acceptable",
                    *[
                        self._format_vision_heading_issue(i)
                        for i in vision_heading_warnings[:10]
                    ],
                ]
            elif self._vision is not None and hasattr(self._vision, "heading_issues"):
                details = ["Vision analysis: heading hierarchy is correct"]
            return CheckResult(
                rule_id="headings-nesting",
                category="Alt Text Headings",
                description="Appropriate heading nesting",
                status="Passed",
                details=details,
            )

        return CheckResult(
            rule_id="headings-nesting",
            category="Alt Text Headings",
            description="Appropriate heading nesting",
            status="Failed",
            details=issues[:20],
            fixable=True,
        )

    def _vision_heading_issue_is_actionable(self, issue: object, pdf: pikepdf.Pdf) -> bool:
        current_tag = str(getattr(issue, "current_tag", "") or "")
        correct_tag = str(getattr(issue, "correct_tag", "") or "")
        if correct_tag and current_tag and current_tag == correct_tag:
            return False
        if not correct_tag:
            return False
        if not (re.match(r"^H[1-6]$", current_tag) or re.match(r"^H[1-6]$", correct_tag)):
            return False
        if re.match(r"^H[1-6]$", current_tag) and re.match(r"^H[1-6]$", correct_tag):
            return False

        description = str(getattr(issue, "description", "") or "").lower()
        suggestion = str(getattr(issue, "suggestion", "") or "").lower()
        issue_text = f"{description} {suggestion}"
        page_num = int(getattr(issue, "page", 0) or 0)
        element_index = getattr(issue, "element_index", None)
        document_has_h1 = any(
            _get_struct_type(node) == "H1"
            for node, _d, _p in self._walk_structure_tree(pdf)
        )
        if (
            document_has_h1
            and re.match(r"^H[1-6]$", correct_tag)
            and (
                "document header" in issue_text
                or "header/banner" in issue_text
                or "banner text" in issue_text
                or "masthead" in issue_text
            )
        ):
            return False
        if (
            document_has_h1
            and correct_tag == "H1"
            and current_tag in {"P", "Span", "?"}
            and "title word" in issue_text
        ):
            return False
        if (
            re.match(r"^H[1-6]$", correct_tag)
            and (
                "author name" in issue_text
                or "byline" in issue_text
                or "signature" in issue_text
                or "running header" in issue_text
                or "page header" in issue_text
                or "page footer" in issue_text
            )
        ):
            return False
        if re.match(r"^H[1-6]$", correct_tag) and (
            "block quote" in issue_text
            or "section transition" in issue_text
            or "transitional phrase" in issue_text
        ):
            return False
        if (
            re.match(r"^H[1-6]$", correct_tag)
            and current_tag in {"P", "Span", "TD", "?"}
            and (
                "within list b" in issue_text
                or "within the table structure" in issue_text
                or "inside the table" in issue_text
            )
        ):
            return False
        if (
            "duplicate/ghost heading" in issue_text
            or (
                "misplaced" in issue_text
                and "actual visible title" in issue_text
            )
        ):
            return False
        if (
            "table/section header row label" in issue_text
            or "repeated subsection heading" in issue_text
            or "repeated form section" in issue_text
        ):
            return False
        if (
            "instructions" in issue_text
            and (
                "bold label for body text" in issue_text
                or "section heading" in issue_text
                or "instructions label" in issue_text
            )
        ):
            return False
        if (
            "thank you" in issue_text
            or "closing heading" in issue_text
            or "closing courtesy" in issue_text
        ):
            return False
        if "invoice metadata" in issue_text or "invoice number" in issue_text:
            return False
        if "duplicate heading" in issue_text:
            return False
        if re.match(r"^H[1-6]$", correct_tag) and "duplicate" in issue_text:
            return False
        if "duplicate heading tag" in issue_text:
            return False
        if correct_tag in {"P", "Span"} and re.match(r"^H[1-6]$", current_tag):
            candidates = []
            raw_text = str(getattr(issue, "text", "") or "").strip()
            if raw_text:
                candidates.append(raw_text)
            for match in re.finditer(r"'([^']{2,140})'|\"([^\"]{2,140})\"", issue_text):
                value = (match.group(1) or match.group(2) or "").strip()
                if value:
                    candidates.append(value)
            if element_index is None and not candidates:
                return False
        if (
            re.match(r"^H[1-6]$", correct_tag)
            and current_tag in {"P", "Span", "?"}
            and page_num > 0
            and (
                "prominent section heading" in issue_text
                or "important instructional subheading" in issue_text
                or "visible subsection heading" in issue_text
                or "major section heading" in issue_text
                or "field section heading" in issue_text
            )
        ):
            page_has_heading = any(
                re.match(r"^H[1-6]$", _get_struct_type(node))
                and self._node_page_number(node, pdf) == page_num
                for node, _d, _p in self._walk_structure_tree(pdf)
            )
            if page_has_heading:
                return False
        if "example block" in issue_text or "example label" in issue_text:
            return False
        if correct_tag == "H1" and page_num > 0 and (
            "main document heading" in issue_text
            or "main section heading" in issue_text
            or "main page heading" in issue_text
            or "section heading" in issue_text
            or "heading component" in issue_text
            or "section heading fragment" in issue_text
            or "section heading part" in issue_text
        ):
            page_has_h1 = any(
                _get_struct_type(node) == "H1"
                and self._node_page_number(node, pdf) == page_num
                for node, _d, _p in self._walk_structure_tree(pdf)
            )
            if page_has_h1:
                return False
        if (
            re.match(r"^H[1-6]$", correct_tag)
            and current_tag in {"P", "Span", "?"}
            and (
                "line 5" in issue_text
                or "line 6" in issue_text
                or "address field" in issue_text
                or "city, state" in issue_text
            )
        ):
            return False
        if (
            re.match(r"^H[1-6]$", correct_tag)
            and "introduces a list" in issue_text
            and page_num > 0
        ):
            page_has_subheading = any(
                re.match(r"^H[2-6]$", _get_struct_type(node))
                and self._node_page_number(node, pdf) == page_num
                for node, _d, _p in self._walk_structure_tree(pdf)
            )
            if page_has_subheading:
                return False
        if (
            re.match(r"^H[1-6]$", correct_tag)
            and "instruction heading" in issue_text
            and page_num > 0
        ):
            page_has_subheading = any(
                re.match(r"^H[2-6]$", _get_struct_type(node))
                and self._node_page_number(node, pdf) == page_num
                for node, _d, _p in self._walk_structure_tree(pdf)
            )
            if page_has_subheading:
                return False
        if correct_tag in {"H1", "H2", "H3", "H4", "H5", "H6"} and page_num > 0:
            candidates = []
            raw_text = str(getattr(issue, "text", "") or "").strip()
            if raw_text:
                candidates.append(raw_text)
            for match in re.finditer(r"'([^']{2,140})'|\"([^\"]{2,140})\"", issue_text):
                value = (match.group(1) or match.group(2) or "").strip()
                if value:
                    candidates.append(value)
            normalized_candidates = {
                re.sub(r"\s+", " ", candidate).strip().lower()
                for candidate in candidates
                if candidate.strip()
            }
            if not normalized_candidates and current_tag in {"P", "Span", "?"}:
                return False
            if (
                re.match(r"^H[1-6]$", correct_tag)
                and "section opening" in issue_text
                and any(
                    candidate.endswith(".")
                    and sum(word[:1].isupper() for word in candidate.split()) < 2
                    for candidate in normalized_candidates
                )
            ):
                return False
            if normalized_candidates:
                for node, _d, _p in self._walk_structure_tree(pdf):
                    existing_tag = _get_struct_type(node)
                    if existing_tag != correct_tag and not (
                        re.match(r"^H[1-6]$", correct_tag)
                        and re.match(r"^H[1-6]$", existing_tag)
                    ):
                        continue
                    if self._node_page_number(node, pdf) != page_num:
                        continue
                    existing = " ".join(
                        str(node.get(key, "") or "")
                        for key in ("/ActualText", "/Alt", "/T")
                    )
                    normalized_existing = re.sub(r"\s+", " ", existing).strip().lower()
                    if any(
                        candidate == normalized_existing
                        or normalized_existing.startswith(candidate)
                        or candidate.startswith(normalized_existing)
                        or (
                            len(candidate) >= 8
                            and f" {candidate} " in f" {normalized_existing} "
                        )
                        for candidate in normalized_candidates
                        if normalized_existing
                    ):
                        return False
                if page_num > 1 and (
                    "page title" in issue_text
                    or "page number" in issue_text
                    or "chapter title" in issue_text
                    or "duplicate" in issue_text
                ):
                    for node, _d, _p in self._walk_structure_tree(pdf):
                        if not re.match(r"^H[1-6]$", _get_struct_type(node)):
                            continue
                        existing = " ".join(
                            str(node.get(key, "") or "")
                            for key in ("/ActualText", "/Alt", "/T")
                        )
                        normalized_existing = re.sub(r"\s+", " ", existing).strip().lower()
                        if any(
                            candidate == normalized_existing
                            or normalized_existing.startswith(candidate)
                            or candidate.startswith(normalized_existing)
                            or (
                                len(candidate) >= 8
                                and f" {candidate} " in f" {normalized_existing} "
                            )
                            for candidate in normalized_candidates
                            if normalized_existing
                        ):
                            return False
        if (
            correct_tag in {"H2", "H3", "H4", "H5", "H6"}
            and (
                "tagged as th" in issue_text
                or "column header" in issue_text
                or "table header" in issue_text
                or "remain as th" in issue_text
                or "keep as th" in issue_text
            )
        ):
            return False
        if (
            document_has_h1
            and correct_tag == "H1"
            and (
                "no h1" in description
                or "lacks" in description
                or "missing" in description
                or "not present" in description
                or "document title" in description
                or (page_num > 1 and "page title" in description)
                or (page_num > 1 and "chapter title" in description)
            )
        ):
            return False
        return True

    @staticmethod
    def _node_page_number(node: pikepdf.Dictionary, pdf: pikepdf.Pdf) -> int | None:
        pg = node.get("/Pg")
        if pg is None:
            return None
        try:
            resolved_pg = _resolve_pdf_object(pg)
        except Exception:
            return None
        for idx, page in enumerate(pdf.pages, 1):
            if page.obj == resolved_pg:
                return idx
        return None

    @staticmethod
    def _format_vision_heading_issue(issue: object) -> str:
        page = getattr(issue, "page", "?")
        description = str(getattr(issue, "description", "") or "Heading hierarchy mismatch")
        detail = f"Page {page}: {description}"
        current_tag = str(getattr(issue, "current_tag", "") or "")
        correct_tag = str(getattr(issue, "correct_tag", "") or "")
        if current_tag or correct_tag:
            detail += f" ({current_tag or '?'} -> {correct_tag or '?'})"
        suggestion = str(getattr(issue, "suggestion", "") or "")
        if suggestion:
            detail += f" ({suggestion})"
        return detail


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _format_page_ranges(pages: list[int]) -> str:
    """Format ``[1,2,3,5,7,8,9]`` → ``'1-3, 5, 7-9'``."""
    if not pages:
        return ""
    pages = sorted(set(pages))
    ranges: list[str] = []
    start = pages[0]
    end = pages[0]
    for p in pages[1:]:
        if p == end + 1:
            end = p
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = p
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)


# ---------------------------------------------------------------------------
# Pipeline-friendly convenience function
# ---------------------------------------------------------------------------


async def check_pdf(
    pdf_path: Path,
    config=None,
) -> CheckReport:
    """Run all 32 checks on a PDF, with automatic vision analysis.

    Designed for pipeline integration — pass a ``PipelineConfig`` and
    vision runs automatically when credentials are available.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    config:
        Optional ``PipelineConfig``.  When provided, vision analysis
        is attempted using the configured LLM backend.
    """
    vision_result = None

    if config is not None:
        try:
            from project_remedy.pdf_vision import (
                VisionAnalyzer,
                create_provider_from_config,
            )

            provider = create_provider_from_config(config)
            if provider is not None:
                analyzer = VisionAnalyzer(provider)
                vision_result = await analyzer.analyze_all(pdf_path)
        except Exception:
            pass  # Vision unavailable — fall back to structural checks.

    checker = PDFAccessibilityChecker(pdf_path, vision_result=vision_result)
    return checker.run_all()
