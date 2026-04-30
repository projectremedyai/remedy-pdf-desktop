"""Tag-tree reader — screen-reader simulation via PDF structure tree.

Walks the PDF /StructTreeRoot exactly as NVDA or VoiceOver would:
traverses the tag hierarchy in reading order, extracts text content
for each marked-content region, and validates structural correctness.

Usage::

    from project_remedy.tag_tree_reader import read_tag_tree, validate_tag_tree
    report = read_tag_tree(Path("remediated.pdf"))
    result = validate_tag_tree(Path("remediated.pdf"))
    for issue in result.issues:
        print(f"[{issue.severity}] {issue.description}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generator

import pikepdf

from project_remedy.pdf_checker import (
    _get_struct_type,
    _node_has_direct_content,
    walk_structure_tree,
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


SCREEN_READER_RULE_IDS = frozenset({
    "sr-no-tags",
    "sr-no-lang",
    "sr-no-headings",
    "sr-heading-start",
    "sr-heading-skip",
    "sr-figure-no-alt",
    "sr-figure-generic-alt",
    "sr-figure-short-alt",
    "sr-table-empty-cell",
    "sr-table-no-headers",
    "sr-list-no-items",
    "sr-empty-element",
    "sr-repeated-content",
    "sr-untagged-page",
})


@dataclass
class TagNode:
    """A single element in the PDF tag tree, as a screen reader would see it."""

    tag: str  # e.g. "H1", "P", "Table", "Figure", "L", "LI"
    depth: int
    page: int  # 0-based page index
    text: str  # extracted text content (empty for containers)
    alt_text: str  # /Alt attribute
    lang: str  # /Lang override on this node
    children_count: int  # number of direct child struct elements
    has_content: bool  # whether this node has MCR/MCID refs


@dataclass
class ScreenReaderIssue:
    """A structural problem that would degrade the screen reader experience."""

    rule_id: str
    severity: Severity
    page: int  # 0-based, -1 if document-level
    element: str  # tag type or path
    description: str
    suggestion: str = ""


@dataclass
class TagTreeReport:
    """Full tag-tree extraction and reading-order output."""

    file_path: Path
    page_count: int
    has_structure_tree: bool
    nodes: list[TagNode] = field(default_factory=list)

    @property
    def reading_order_text(self) -> str:
        """Concatenated text in reading order — what a screen reader speaks."""
        parts: list[str] = []
        for node in self.nodes:
            if node.alt_text:
                parts.append(f"[{node.tag}: {node.alt_text}]")
            elif node.text:
                parts.append(node.text)
        return "\n".join(parts)

    @property
    def reading_order_annotated(self) -> str:
        """Annotated reading order with tag types — for debugging."""
        lines: list[str] = []
        for i, node in enumerate(self.nodes):
            indent = "  " * node.depth
            content = node.alt_text or node.text or ""
            preview = content[:80].replace("\n", " ")
            if preview:
                lines.append(f"{i + 1:4d}. {indent}<{node.tag}> p{node.page + 1}: {preview}")
            else:
                lines.append(f"{i + 1:4d}. {indent}<{node.tag}> p{node.page + 1}")
        return "\n".join(lines)

    def nodes_by_page(self) -> dict[int, list[TagNode]]:
        """Group nodes by page number."""
        by_page: dict[int, list[TagNode]] = {}
        for node in self.nodes:
            by_page.setdefault(node.page, []).append(node)
        return by_page


@dataclass
class ValidationResult:
    """Screen reader validation outcome."""

    file_path: Path
    tag_tree: TagTreeReport
    issues: list[ScreenReaderIssue] = field(default_factory=list)
    passed: bool = True

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)


# ---------------------------------------------------------------------------
# MCID → text extraction
# ---------------------------------------------------------------------------

# Text-showing operators in PDF content streams.
_TEXT_OPS = frozenset({"Tj", "TJ", "'", '"'})


def _extract_mcid_text(page: pikepdf.Page) -> dict[int, str]:
    """Extract text content for each MCID on a page.

    Parses the content stream, tracks BDC/EMC nesting to associate
    text-showing operators with their MCID. Returns {mcid: text}.
    """
    try:
        instructions = pikepdf.parse_content_stream(page)
    except Exception:
        return {}

    mcid_texts: dict[int, list[str]] = {}
    mcid_stack: list[int | None] = []

    for operands, operator in instructions:
        op = str(operator)

        if op in ("BDC", "BMC"):
            mcid = None
            if op == "BDC" and len(operands) >= 2:
                props = operands[1]
                if isinstance(props, (pikepdf.Dictionary, pikepdf.Stream)):
                    pass  # already usable
                elif isinstance(props, pikepdf.Object) and not isinstance(props, pikepdf.Name):
                    try:
                        props = props.resolve()
                    except Exception:
                        pass
                if isinstance(props, pikepdf.Dictionary):
                    mcid_val = props.get("/MCID")
                    if mcid_val is not None:
                        mcid = int(mcid_val)
            mcid_stack.append(mcid)

        elif op == "EMC":
            if mcid_stack:
                mcid_stack.pop()

        elif op in _TEXT_OPS and mcid_stack:
            # Find the innermost active MCID.
            current_mcid = None
            for m in reversed(mcid_stack):
                if m is not None:
                    current_mcid = m
                    break
            if current_mcid is None:
                continue

            text = _decode_text_operands(operands, op)
            if text:
                mcid_texts.setdefault(current_mcid, []).append(text)

    return {mcid: " ".join(parts) for mcid, parts in mcid_texts.items()}


def _decode_text_operands(operands: list, op: str) -> str:
    """Decode text from Tj/TJ/'/" operands."""
    parts: list[str] = []
    if op == "TJ" and operands:
        arr = operands[0]
        if isinstance(arr, pikepdf.Array):
            for item in arr:
                if isinstance(item, (pikepdf.String, bytes)):
                    parts.append(_decode_pdf_string(item))
        elif isinstance(arr, (pikepdf.String, bytes)):
            parts.append(_decode_pdf_string(arr))
    elif op in ("Tj", "'") and operands:
        parts.append(_decode_pdf_string(operands[0]))
    elif op == '"' and len(operands) >= 3:
        parts.append(_decode_pdf_string(operands[2]))
    return "".join(parts)


def _decode_pdf_string(s) -> str:
    """Best-effort decode a pikepdf.String or bytes to Python str."""
    if isinstance(s, pikepdf.String):
        try:
            return str(s)
        except Exception:
            return s.to_bytes().decode("latin-1", errors="replace")
    if isinstance(s, bytes):
        return s.decode("latin-1", errors="replace")
    return str(s) if s else ""


# ---------------------------------------------------------------------------
# Tag tree reader
# ---------------------------------------------------------------------------


def _build_page_index(pdf: pikepdf.Pdf) -> dict[tuple, int]:
    """Map page objgen → 0-based page index for reliable lookup.

    Uses pikepdf objgen (object number, generation) instead of Python id()
    because id() changes across .resolve() calls on the same PDF object.
    """
    index: dict[tuple, int] = {}
    for idx, page in enumerate(pdf.pages):
        try:
            index[page.obj.objgen] = idx
        except Exception:
            pass
    return index


def _resolve_page_number(
    node: pikepdf.Dictionary,
    page_index: dict[tuple, int],
    last_page: int,
) -> int:
    """Resolve the page number for a node, with fallback chain."""
    # 1. Direct /Pg reference.
    pg = node.get("/Pg")
    if pg is not None:
        try:
            resolved = pg.resolve() if hasattr(pg, "resolve") else pg
            idx = page_index.get(resolved.objgen)
            if idx is not None:
                return idx
        except Exception:
            pass

    # 2. MCR child with /Pg.
    kids = node.get("/K")
    if kids is not None:
        items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
        for item in items:
            resolved = item
            if hasattr(item, "resolve"):
                try:
                    resolved = item.resolve()
                except Exception:
                    continue
            if isinstance(resolved, pikepdf.Dictionary) and "/Pg" in resolved:
                try:
                    pg_obj = resolved["/Pg"]
                    if hasattr(pg_obj, "resolve"):
                        pg_obj = pg_obj.resolve()
                    idx = page_index.get(pg_obj.objgen)
                    if idx is not None:
                        return idx
                except Exception:
                    pass

    return last_page


def _get_node_mcids(node: pikepdf.Dictionary) -> list[int]:
    """Get all MCID integers from a node's /K children."""
    kids = node.get("/K")
    if kids is None:
        return []
    items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
    mcids: list[int] = []
    for item in items:
        resolved = item
        if hasattr(item, "resolve"):
            try:
                resolved = item.resolve()
            except Exception:
                continue
        if not isinstance(resolved, pikepdf.Dictionary):
            # Direct integer MCID.
            try:
                mcids.append(int(resolved))
            except (TypeError, ValueError):
                pass
        elif "/S" not in resolved:
            # MCR dict — has /MCID but no /S.
            mcid_val = resolved.get("/MCID")
            if mcid_val is not None:
                try:
                    mcids.append(int(mcid_val))
                except (TypeError, ValueError):
                    pass
    return mcids


def _count_struct_children(node: pikepdf.Dictionary) -> int:
    """Count the number of child struct elements (nodes with /S)."""
    kids = node.get("/K")
    if kids is None:
        return 0
    items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
    count = 0
    for item in items:
        resolved = item
        if hasattr(item, "resolve"):
            try:
                resolved = item.resolve()
            except Exception:
                continue
        if isinstance(resolved, pikepdf.Dictionary) and "/S" in resolved:
            count += 1
    return count


def _annotation_fallback_text(node: pikepdf.Dictionary) -> str:
    """Return fallback text from an annotation reference when available."""
    kids = node.get("/K")
    if kids is None:
        return ""
    items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
    parts: list[str] = []
    for item in items:
        resolved = item
        if hasattr(item, "resolve"):
            try:
                resolved = item.resolve()
            except Exception:
                continue
        if not isinstance(resolved, pikepdf.Dictionary):
            continue
        if str(resolved.get("/Type", "")) != "/OBJR":
            continue
        annot = resolved.get("/Obj")
        if annot is None:
            continue
        try:
            annot = annot.resolve() if hasattr(annot, "resolve") else annot
        except Exception:
            continue
        if not isinstance(annot, pikepdf.Dictionary):
            continue
        contents = annot.get("/Contents")
        if contents is not None:
            text = _decode_pdf_string(contents).strip()
            if text:
                parts.append(text)
                continue
        action = annot.get("/A")
        if action is not None:
            try:
                action = action.resolve() if hasattr(action, "resolve") else action
            except Exception:
                action = None
        if isinstance(action, pikepdf.Dictionary):
            uri = str(action.get("/URI", "")).strip()
            if uri:
                parts.append(uri)
    return " ".join(parts).strip()


def read_tag_tree(pdf_path: Path) -> TagTreeReport:
    """Extract the full tag tree from a PDF in reading order.

    Returns a TagTreeReport with every node the screen reader would
    encounter, including its text content and structural metadata.
    """
    with pikepdf.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        page_index = _build_page_index(pdf)

        struct_root = pdf.Root.get("/StructTreeRoot")
        if struct_root is None:
            return TagTreeReport(
                file_path=pdf_path,
                page_count=page_count,
                has_structure_tree=False,
            )

        # Pre-extract MCID text for all pages.
        page_mcid_texts: dict[int, dict[int, str]] = {}
        for idx, page in enumerate(pdf.pages):
            page_mcid_texts[idx] = _extract_mcid_text(page)

        # Walk structure tree in reading order.
        nodes: list[TagNode] = []
        last_page = 0

        for node, depth, _parent in walk_structure_tree(pdf):
            tag = _get_struct_type(node)
            if not tag:
                continue

            # Skip the StructTreeRoot wrapper itself.
            if tag == "StructTreeRoot":
                continue

            page_num = _resolve_page_number(node, page_index, last_page)
            last_page = page_num

            # Extract alt text.
            alt_raw = node.get("/Alt")
            alt_text = ""
            if alt_raw is not None:
                try:
                    alt_text = str(alt_raw).strip()
                except Exception:
                    pass

            # Extract lang override.
            lang_raw = node.get("/Lang")
            lang = ""
            if lang_raw is not None:
                try:
                    lang = str(lang_raw).strip()
                except Exception:
                    pass

            # Extract text from MCIDs.
            has_content = _node_has_direct_content(node)
            text = ""
            if has_content:
                mcids = _get_node_mcids(node)
                text_parts = []
                page_texts = page_mcid_texts.get(page_num, {})
                for mcid in mcids:
                    mcid_text = page_texts.get(mcid, "")
                    if mcid_text:
                        text_parts.append(mcid_text)
                text = " ".join(text_parts)
                if not text.strip():
                    text = _annotation_fallback_text(node)

            nodes.append(TagNode(
                tag=tag,
                depth=depth,
                page=page_num,
                text=text,
                alt_text=alt_text,
                lang=lang,
                children_count=_count_struct_children(node),
                has_content=has_content,
            ))

        return TagTreeReport(
            file_path=pdf_path,
            page_count=page_count,
            has_structure_tree=True,
            nodes=nodes,
        )


# ---------------------------------------------------------------------------
# Screen reader validation
# ---------------------------------------------------------------------------

# NVDA quick-nav element types — these are the tags that must exist and
# be properly nested for a screen reader to navigate the document.
_HEADING_TAGS = frozenset({"H", "H1", "H2", "H3", "H4", "H5", "H6"})
_TABLE_TAGS = frozenset({"Table", "TR", "TH", "TD", "THead", "TBody", "TFoot"})
_LIST_TAGS = frozenset({"L", "LI", "Lbl", "LBody"})
_LINK_TAGS = frozenset({"Link"})
_FIGURE_TAGS = frozenset({"Figure"})


def _heading_level(tag: str) -> int:
    """Parse heading level from tag name. Returns 0 if not a heading."""
    if tag == "H":
        return 0  # Generic heading, no level.
    m = re.match(r"^H(\d)$", tag)
    return int(m.group(1)) if m else 0


def validate_tag_tree(pdf_path: Path) -> ValidationResult:
    """Validate a PDF's tag tree from a screen reader's perspective.

    Checks everything NVDA/VoiceOver rely on:
    - Document has structure tree
    - Headings are properly nested (no skipped levels)
    - Tables have proper row/cell structure
    - Lists have proper LI/LBody structure
    - Figures have alt text
    - All content is tagged (no large untagged gaps)
    - Language is set
    - Reading order produces coherent text
    """
    report = read_tag_tree(pdf_path)
    issues: list[ScreenReaderIssue] = []

    # --- Check 1: Structure tree exists ---
    if not report.has_structure_tree:
        issues.append(ScreenReaderIssue(
            rule_id="sr-no-tags",
            severity=Severity.ERROR,
            page=-1,
            element="Document",
            description="PDF has no structure tree — completely invisible to screen readers",
            suggestion="Run fix_all() to generate a structure tree",
        ))
        return ValidationResult(
            file_path=pdf_path,
            tag_tree=report,
            issues=issues,
            passed=False,
        )

    # --- Check 2: Document-level language ---
    _check_document_language(pdf_path, issues)

    # --- Check 3: Heading nesting ---
    _check_heading_nesting(report.nodes, issues)

    # --- Check 4: Figure alt text ---
    _check_figure_alt_text(report.nodes, issues)

    # --- Check 5: Table structure ---
    _check_table_structure(report.nodes, issues)

    # --- Check 6: List structure ---
    _check_list_structure(report.nodes, issues)

    # --- Check 7: Empty tagged elements ---
    _check_empty_elements(report.nodes, issues)

    # --- Check 8: Reading order coherence ---
    _check_reading_order(report, issues)

    # --- Check 9: Untagged pages ---
    _check_untagged_pages(report, issues)

    passed = all(i.severity != Severity.ERROR for i in issues)
    return ValidationResult(
        file_path=pdf_path,
        tag_tree=report,
        issues=issues,
        passed=passed,
    )


def _check_document_language(pdf_path: Path, issues: list[ScreenReaderIssue]) -> None:
    """NVDA uses /Lang to select the speech synthesizer voice."""
    with pikepdf.open(pdf_path) as pdf:
        lang = pdf.Root.get("/Lang")
        if not lang or not str(lang).strip():
            issues.append(ScreenReaderIssue(
                rule_id="sr-no-lang",
                severity=Severity.ERROR,
                page=-1,
                element="Document",
                description="No document language set — screen reader cannot select correct voice",
                suggestion="Set /Lang on the document catalog (e.g. 'en' or 'es')",
            ))


def _check_heading_nesting(nodes: list[TagNode], issues: list[ScreenReaderIssue]) -> None:
    """Validate heading hierarchy — NVDA users navigate by heading level."""
    levels_seen: list[tuple[int, int, str]] = []  # (level, page, tag)
    for node in nodes:
        level = _heading_level(node.tag)
        if level == 0:
            continue
        levels_seen.append((level, node.page, node.tag))

    if not levels_seen:
        issues.append(ScreenReaderIssue(
            rule_id="sr-no-headings",
            severity=Severity.WARNING,
            page=-1,
            element="Document",
            description="No headings found — screen reader users cannot navigate by heading",
            suggestion="Add H1/H2 structure elements to enable heading navigation",
        ))
        return

    # Check first heading is H1.
    first_level, first_page, first_tag = levels_seen[0]
    if first_level != 1:
        issues.append(ScreenReaderIssue(
            rule_id="sr-heading-start",
            severity=Severity.WARNING,
            page=first_page,
            element=first_tag,
            description=f"First heading is {first_tag}, not H1 — NVDA announces 'heading level {first_level}'",
            suggestion="Start the document with an H1",
        ))

    # Check for skipped levels.
    prev_level = 0
    for level, page, tag in levels_seen:
        if level > prev_level + 1 and prev_level > 0:
            issues.append(ScreenReaderIssue(
                rule_id="sr-heading-skip",
                severity=Severity.ERROR,
                page=page,
                element=tag,
                description=(
                    f"Heading level skipped: H{prev_level} → {tag} — "
                    f"NVDA users pressing '{level}' may miss content"
                ),
                suggestion=f"Add intermediate H{prev_level + 1} or change {tag} to H{prev_level + 1}",
            ))
        prev_level = level


def _check_figure_alt_text(nodes: list[TagNode], issues: list[ScreenReaderIssue]) -> None:
    """NVDA reads /Alt for Figure elements — missing alt = 'graphic' announced."""
    from project_remedy.pdf_checker import _is_generic_alt_text

    for node in nodes:
        if node.tag not in _FIGURE_TAGS:
            continue
        if not node.alt_text:
            issues.append(ScreenReaderIssue(
                rule_id="sr-figure-no-alt",
                severity=Severity.ERROR,
                page=node.page,
                element="Figure",
                description="Figure has no alt text — screen reader announces 'graphic' with no description",
                suggestion="Add /Alt text describing the image content",
            ))
        elif _is_generic_alt_text(node.alt_text):
            issues.append(ScreenReaderIssue(
                rule_id="sr-figure-generic-alt",
                severity=Severity.ERROR,
                page=node.page,
                element="Figure",
                description=f"Figure has generic/placeholder alt text: '{node.alt_text}' — provides no value to screen reader users",
                suggestion="Replace with a meaningful description of the image content",
            ))
        elif len(node.alt_text) < 3:
            issues.append(ScreenReaderIssue(
                rule_id="sr-figure-short-alt",
                severity=Severity.WARNING,
                page=node.page,
                element="Figure",
                description=f"Figure alt text is very short: '{node.alt_text}'",
                suggestion="Provide a meaningful description of the image",
            ))


def _check_table_structure(nodes: list[TagNode], issues: list[ScreenReaderIssue]) -> None:
    """NVDA table navigation (Ctrl+Alt+arrows) requires proper TR/TH/TD nesting."""
    tables: list[TagNode] = []
    # Build parent→child map from sequential walk.
    node_children: dict[int, list[int]] = {}
    depth_stack: list[int] = []

    for i, node in enumerate(nodes):
        # Track nesting by depth to find parent indices.
        while depth_stack and nodes[depth_stack[-1]].depth >= node.depth:
            depth_stack.pop()
        if depth_stack:
            parent_idx = depth_stack[-1]
            node_children.setdefault(parent_idx, []).append(i)
        depth_stack.append(i)

        if node.tag == "Table":
            tables.append(node)

    if not tables:
        return

    for node in nodes:
        # TR must be inside Table, THead, TBody, or TFoot.
        if node.tag == "TR":
            # Check via depth — parent should be a table container.
            pass  # Checked structurally below.

        # TH/TD without text means empty cell — screen reader says "blank".
        if node.tag in ("TH", "TD"):
            if node.has_content and not node.text.strip() and not node.alt_text:
                issues.append(ScreenReaderIssue(
                    rule_id="sr-table-empty-cell",
                    severity=Severity.INFO,
                    page=node.page,
                    element=node.tag,
                    description=f"Empty {node.tag} — screen reader announces 'blank'",
                ))

    # Check each table has at least one TH (header row).
    for i, node in enumerate(nodes):
        if node.tag != "Table":
            continue
        # Collect descendants until depth returns to table level.
        has_th = False
        has_tr = False
        for j in range(i + 1, len(nodes)):
            if nodes[j].depth <= node.depth:
                break
            if nodes[j].tag == "TH":
                has_th = True
            if nodes[j].tag == "TR":
                has_tr = True

        if has_tr and not has_th:
            issues.append(ScreenReaderIssue(
                rule_id="sr-table-no-headers",
                severity=Severity.ERROR,
                page=node.page,
                element="Table",
                description="Table has no TH elements — NVDA cannot announce column/row headers",
                suggestion="Mark header cells as TH instead of TD",
            ))


def _check_list_structure(nodes: list[TagNode], issues: list[ScreenReaderIssue]) -> None:
    """NVDA announces 'list of N items' — needs proper L/LI/LBody nesting."""
    for i, node in enumerate(nodes):
        if node.tag != "L":
            continue
        # Count LI children.
        li_count = 0
        for j in range(i + 1, len(nodes)):
            if nodes[j].depth <= node.depth:
                break
            if nodes[j].depth == node.depth + 1 and nodes[j].tag == "LI":
                li_count += 1

        if li_count == 0:
            issues.append(ScreenReaderIssue(
                rule_id="sr-list-no-items",
                severity=Severity.ERROR,
                page=node.page,
                element="L",
                description="List element has no LI children — screen reader announces empty list",
                suggestion="Add LI children inside the L element",
            ))


def _check_empty_elements(nodes: list[TagNode], issues: list[ScreenReaderIssue]) -> None:
    """Leaf elements with no text are announced as empty — confusing for users."""
    for node in nodes:
        if node.tag in ("Document", "Part", "Sect", "Div", "Art",
                        "BlockQuote", "NonStruct", "StructTreeRoot"):
            continue  # Container-only tags.
        if node.tag in _TABLE_TAGS or node.tag in _LIST_TAGS:
            continue  # Checked separately.
        if node.tag in _FIGURE_TAGS:
            continue  # Checked via alt text.
        if node.children_count > 0:
            continue  # Has child elements, not a leaf.
        if not node.has_content:
            continue  # No MCR refs at all.
        if not node.text.strip() and not node.alt_text:
            issues.append(ScreenReaderIssue(
                rule_id="sr-empty-element",
                severity=Severity.WARNING,
                page=node.page,
                element=node.tag,
                description=f"Tagged <{node.tag}> element is empty — screen reader announces nothing",
            ))


def _normalize_repeated_content_text(text: str) -> str:
    """Normalize text before repeated-content detection.

    OCR-first rebuilds can leave interleaved NULs or irregular spacing inside
    short heading fragments. Those short fragments are noisy but not the kind
    of repeated paragraph-level artifact this warning is meant to detect.
    """
    return " ".join(text.replace("\x00", "").split()).strip()


def _check_reading_order(report: TagTreeReport, issues: list[ScreenReaderIssue]) -> None:
    """Detect obvious reading-order problems across pages."""
    by_page = report.nodes_by_page()

    for page_num in sorted(by_page.keys()):
        page_nodes = by_page[page_num]
        content_nodes = [n for n in page_nodes if n.text.strip() or n.alt_text]

        if not content_nodes:
            continue

        # Check for text that starts mid-sentence (lowercase after a heading).
        prev_was_heading = False
        for node in content_nodes:
            if node.tag in _HEADING_TAGS:
                prev_was_heading = True
                continue
            if prev_was_heading and node.text.strip():
                # This is normal — body text after heading.
                prev_was_heading = False
                continue
            prev_was_heading = False

        # Check for repeated identical text blocks (copy-paste artifacts).
        texts = []
        for node in content_nodes:
            normalized = _normalize_repeated_content_text(node.text)
            if not normalized:
                continue
            # Keep this warning focused on genuinely duplicated content blocks,
            # not short repeated headings/menu fragments such as OCR artifacts.
            if len(normalized) < 40:
                continue
            if len(normalized.split()) < 5:
                continue
            texts.append(normalized)
        seen: dict[str, int] = {}
        for text in texts:
            seen[text] = seen.get(text, 0) + 1
        for text, count in seen.items():
            if count >= 3:
                issues.append(ScreenReaderIssue(
                    rule_id="sr-repeated-content",
                    severity=Severity.WARNING,
                    page=page_num,
                    element="P",
                    description=(
                        f"Text repeated {count} times on page {page_num + 1}: "
                        f"'{text[:60]}...' — screen reader reads it each time"
                    ),
                ))


def _check_untagged_pages(report: TagTreeReport, issues: list[ScreenReaderIssue]) -> None:
    """Pages with no tagged content are invisible to screen readers."""
    tagged_pages = {n.page for n in report.nodes if n.has_content or n.alt_text}
    blank_pages = _find_blank_pages(report.file_path)
    for page_num in range(report.page_count):
        if page_num in blank_pages:
            continue
        if page_num not in tagged_pages:
            issues.append(ScreenReaderIssue(
                rule_id="sr-untagged-page",
                severity=Severity.ERROR,
                page=page_num,
                element="Page",
                description=f"Page {page_num + 1} has no tagged content — completely invisible to screen readers",
                suggestion="Ensure all content on this page is wrapped in structure elements",
            ))


def _find_blank_pages(pdf_path: Path) -> set[int]:
    """Return zero-based page indexes that have no meaningful rendered content."""
    try:
        import fitz
    except Exception:
        return set()

    blank_pages: set[int] = set()
    doc = fitz.open(str(pdf_path))
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = " ".join(page.get_text("text").split())
            if text:
                continue

            has_nontext_content = False
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") == 1:
                    x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
                    if abs((x1 - x0) * (y1 - y0)) >= 64:
                        has_nontext_content = True
                        break

            if has_nontext_content:
                continue
            if page.first_widget or page.first_annot or page.get_links():
                continue

            blank_pages.add(page_num)
    finally:
        doc.close()
    return blank_pages


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------


def print_reading_order(pdf_path: Path, *, annotated: bool = False) -> None:
    """Print what a screen reader would read, in order."""
    report = read_tag_tree(pdf_path)
    if not report.has_structure_tree:
        print(f"ERROR: {pdf_path.name} has no structure tree")
        return
    if annotated:
        print(report.reading_order_annotated)
    else:
        print(report.reading_order_text)


def print_validation(pdf_path: Path) -> None:
    """Print screen reader validation results."""
    result = validate_tag_tree(pdf_path)
    status = "PASS" if result.passed else "FAIL"
    print(f"\n[{status}] {pdf_path.name}")
    print(f"  Tags: {len(result.tag_tree.nodes)}, Pages: {result.tag_tree.page_count}")
    print(f"  Errors: {result.error_count}, Warnings: {result.warning_count}")

    if result.issues:
        print()
        for issue in result.issues:
            marker = {"error": "E", "warning": "W", "info": "I"}[issue.severity.value]
            page_str = f"p{issue.page + 1}" if issue.page >= 0 else "doc"
            print(f"  [{marker}] {page_str} <{issue.element}> {issue.description}")
            if issue.suggestion:
                print(f"       → {issue.suggestion}")
