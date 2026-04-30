"""Operator-preserving content stream rebuild with BDC/EMC injection.

Parses the source PDF content stream using the existing
:class:`GraphicsStateTracker`, classifies operators into spans, and
re-emits them with proper BDC/EMC marked-content wrappers.  The original
operators are preserved verbatim — only tagging wrappers are added or
adjusted — so visual fidelity is maintained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pikepdf
from pikepdf import Dictionary, Name, Operator

from project_remedy.content_stream.parser import GraphicsStateTracker
from project_remedy.faithful_rebuild.models import (
    MCIDEntry,
    MCIDManifest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class OperatorSpan:
    """A contiguous range of content-stream instructions with a shared role.

    Attributes:
        span_type: One of ``"text"``, ``"image"``, ``"vector"``,
                   ``"state"``, ``"marked"``.
        start_index: Index of the first instruction in the span (inclusive).
        end_index: Index of the last instruction in the span (inclusive).
        existing_mcid: If the span is an existing marked-content sequence
                       that already carries an MCID, the value is stored here.
        existing_tag: The tag name of an existing BDC/BMC wrapper (e.g. ``"P"``).
        xobject_name: For ``"image"`` spans, the XObject resource key
                      (e.g. ``"Im0"``).
    """

    span_type: str  # "text", "image", "vector", "state", "marked"
    start_index: int
    end_index: int  # inclusive
    existing_mcid: int | None = None
    existing_tag: str | None = None
    xobject_name: str | None = None


# ---------------------------------------------------------------------------
# Operator sets
# ---------------------------------------------------------------------------

# Path construction operators
_PATH_CONSTRUCTION_OPS = frozenset({"m", "l", "c", "v", "y", "re", "h"})

# Path painting operators
_PATH_PAINT_OPS = frozenset({"S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n"})

# Clipping operators
_CLIP_OPS = frozenset({"W", "W*"})

# All path-related operators
_PATH_OPS = _PATH_CONSTRUCTION_OPS | _PATH_PAINT_OPS | _CLIP_OPS


# ---------------------------------------------------------------------------
# classify_operator_spans
# ---------------------------------------------------------------------------


def classify_operator_spans(page: pikepdf.Page) -> list[OperatorSpan]:
    """Parse a page's content stream and classify instructions into spans.

    Returns a list of :class:`OperatorSpan` objects covering the interesting
    regions of the content stream.  "State" operators (``q``, ``Q``, ``cm``,
    ``gs``, colour ops outside spans, etc.) are **not** returned as spans —
    they are emitted as-is during rebuild.

    Span types:

    * ``"text"`` — a ``BT`` … ``ET`` block.
    * ``"image"`` — a ``Do`` operator that references an Image XObject.
    * ``"vector"`` — path construction operators followed by a paint or clip op.
    * ``"marked"`` — an existing ``BDC``/``BMC`` … ``EMC`` sequence.
    """
    tracker = GraphicsStateTracker()
    instructions = tracker.track(page)
    if not instructions:
        return []

    spans: list[OperatorSpan] = []
    n = len(instructions)

    # Pre-resolve which XObject names are images
    image_xobject_names: set[str] = set()
    resources = page.obj.get("/Resources")
    if resources is not None:
        xobjects = resources.get("/XObject")
        if xobjects is not None:
            for key in xobjects.keys():
                key_str = str(key).lstrip("/")
                try:
                    subtype = str(xobjects[key].get("/Subtype", ""))
                    if subtype == "/Image":
                        image_xobject_names.add(key_str)
                except Exception:
                    pass

    # Track which indices have already been consumed by a span
    consumed: set[int] = set()

    # --- Pass 1: detect existing marked-content sequences (BDC/BMC … EMC) ---
    # We need to handle nesting, so use a stack approach.
    mark_stack: list[tuple[int, str | None, int | None]] = []  # (start_idx, tag, mcid)

    for idx, ann in enumerate(instructions):
        op = ann.operator
        if op in ("BDC", "BMC"):
            tag: str | None = None
            mcid: int | None = None
            if ann.operands:
                tag = str(ann.operands[0]).lstrip("/")
            if op == "BDC" and len(ann.operands) >= 2:
                props = ann.operands[1]
                if isinstance(props, pikepdf.Dictionary):
                    mcid_val = props.get("/MCID")
                    if mcid_val is not None:
                        try:
                            mcid = int(mcid_val)
                        except (TypeError, ValueError):
                            pass
            mark_stack.append((idx, tag, mcid))
        elif op == "EMC" and mark_stack:
            start_idx, tag, mcid = mark_stack.pop()
            spans.append(OperatorSpan(
                span_type="marked",
                start_index=start_idx,
                end_index=idx,
                existing_mcid=mcid,
                existing_tag=tag,
            ))
            for j in range(start_idx, idx + 1):
                consumed.add(j)

    # --- Pass 2: detect text blocks (BT … ET) not inside marked spans ---
    i = 0
    while i < n:
        if i in consumed:
            i += 1
            continue
        ann = instructions[i]
        if ann.operator == "BT":
            bt_start = i
            j = i + 1
            while j < n and instructions[j].operator != "ET":
                j += 1
            if j < n:
                # j is the ET
                et_end = j
                spans.append(OperatorSpan(
                    span_type="text",
                    start_index=bt_start,
                    end_index=et_end,
                ))
                for k in range(bt_start, et_end + 1):
                    consumed.add(k)
                i = et_end + 1
                continue
        i += 1

    # --- Pass 3: detect image Do operators not inside marked/text spans ---
    for idx, ann in enumerate(instructions):
        if idx in consumed:
            continue
        if ann.operator == "Do" and ann.operands:
            xobj_name = str(ann.operands[0]).lstrip("/")
            if xobj_name in image_xobject_names:
                spans.append(OperatorSpan(
                    span_type="image",
                    start_index=idx,
                    end_index=idx,
                    xobject_name=xobj_name,
                ))
                consumed.add(idx)

    # --- Pass 4: detect vector paths not inside marked/text/image spans ---
    # A vector span starts at the first path-construction op and ends at
    # the last paint/clip op in a contiguous run.
    i = 0
    while i < n:
        if i in consumed:
            i += 1
            continue
        ann = instructions[i]
        if ann.operator in _PATH_CONSTRUCTION_OPS:
            vec_start = i
            j = i
            # Extend through contiguous path construction + paint + clip ops
            while j < n and not (j in consumed):
                op_j = instructions[j].operator
                if op_j in _PATH_OPS:
                    j += 1
                else:
                    break
            # j is one past the last path op.  The span is valid only if
            # it contains at least one paint or clip op.
            vec_end = j - 1
            has_paint = any(
                instructions[k].operator in (_PATH_PAINT_OPS | _CLIP_OPS)
                for k in range(vec_start, vec_end + 1)
                if k not in consumed
            )
            if has_paint and vec_end >= vec_start:
                spans.append(OperatorSpan(
                    span_type="vector",
                    start_index=vec_start,
                    end_index=vec_end,
                ))
                for k in range(vec_start, vec_end + 1):
                    consumed.add(k)
                i = vec_end + 1
                continue
        i += 1

    # Sort spans by start_index for deterministic ordering
    spans.sort(key=lambda s: s.start_index)
    return spans


# ---------------------------------------------------------------------------
# rebuild_page_preserving
# ---------------------------------------------------------------------------


# Sentinel MCID used for Table-parent MCIDEntry records. Structure builder
# skips entry.mcid lookups when the tag is "Table" with a table_spec, but the
# dataclass still requires an integer — use a value the content stream will
# never emit.
_TABLE_PARENT_SENTINEL_MCID = -1


def _resolve(obj):
    """Follow an indirect reference to its underlying object, if any."""
    try:
        return obj.resolve() if hasattr(obj, "resolve") else obj
    except Exception:
        return obj


def _iter_struct_elem_children(node: pikepdf.Dictionary):
    """Yield resolved struct-elem children of *node* (not MCRs or int MCIDs)."""
    kids = node.get("/K")
    if kids is None:
        return
    items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
    for item in items:
        resolved = _resolve(item)
        if isinstance(resolved, pikepdf.Dictionary) and resolved.get("/S") is not None:
            yield resolved


def _collect_descendant_mcids(node: pikepdf.Dictionary, out: list[int]) -> None:
    """Collect every MCID under *node*, traversing MCR dicts and int kids."""
    kids = node.get("/K")
    if kids is None:
        return
    items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
    for item in items:
        resolved = _resolve(item)
        if isinstance(resolved, int):
            out.append(int(resolved))
            continue
        if isinstance(resolved, pikepdf.Dictionary):
            node_type = resolved.get("/Type")
            if node_type == Name("/MCR"):
                mcid = resolved.get("/MCID")
                if mcid is not None:
                    try:
                        out.append(int(mcid))
                    except (TypeError, ValueError):
                        pass
                continue
            if resolved.get("/S") is not None:
                _collect_descendant_mcids(resolved, out)


def _extract_scope_attribute(cell: pikepdf.Dictionary) -> str | None:
    """Return the ``/Scope`` attribute of a table-header cell, if any.

    Scope lives under ``/A`` (single Attribute dict or Array of them) with
    ``/O`` set to ``/Table``.
    """
    attrs = cell.get("/A")
    if attrs is None:
        return None
    candidates = list(attrs) if isinstance(attrs, pikepdf.Array) else [attrs]
    for candidate in candidates:
        resolved = _resolve(candidate)
        if not isinstance(resolved, pikepdf.Dictionary):
            continue
        if resolved.get("/O") != Name("/Table"):
            continue
        scope = resolved.get("/Scope")
        if scope is not None:
            return str(scope).lstrip("/")
    return None


def _extract_td_headers(cell: pikepdf.Dictionary) -> list[str] | None:
    """Return the ``/Headers`` id list attached to a data cell, if any."""
    attrs = cell.get("/A")
    if attrs is None:
        return None
    candidates = list(attrs) if isinstance(attrs, pikepdf.Array) else [attrs]
    for candidate in candidates:
        resolved = _resolve(candidate)
        if not isinstance(resolved, pikepdf.Dictionary):
            continue
        if resolved.get("/O") != Name("/Table"):
            continue
        headers = resolved.get("/Headers")
        if headers is None:
            continue
        entries = list(headers) if isinstance(headers, pikepdf.Array) else [headers]
        out: list[str] = []
        for h in entries:
            try:
                out.append(str(h))
            except Exception:
                continue
        if out:
            return out
    return None


def _same_page(obj, target_objgen: tuple[int, int] | None) -> bool:
    """pikepdf Objects compare ``==`` permissively across references, so use
    the underlying object/generation pair to tell two page references apart.
    """
    if obj is None or target_objgen is None:
        return False
    resolved = _resolve(obj)
    actual = getattr(resolved, "objgen", None)
    return actual == target_objgen


def _build_table_spec_for_page(
    table_node: pikepdf.Dictionary,
    target_page_obj: pikepdf.Object,
) -> dict | None:
    """Build the ``table_spec`` dict expected by ``_build_table_structure``.

    Returns None when this Table has no MCIDs on *target_page_obj* or when
    the tree has no recognizable TR/TH/TD structure.
    """
    # Confirm at least one descendant MCID is rendered on the target page.
    all_mcids: list[int] = []
    _collect_descendant_mcids(table_node, all_mcids)
    if not all_mcids:
        return None

    target_objgen = getattr(target_page_obj, "objgen", None)

    # Any MCR under the node must reference this page; checking /Pg on the
    # node itself isn't reliable (not all writers set it on /Table).
    page_matches = False
    kids = table_node.get("/K")
    stack = list(kids) if isinstance(kids, pikepdf.Array) else ([kids] if kids is not None else [])
    while stack and not page_matches:
        item = stack.pop()
        resolved = _resolve(item)
        if isinstance(resolved, pikepdf.Dictionary):
            if resolved.get("/Type") == Name("/MCR"):
                if _same_page(resolved.get("/Pg"), target_objgen):
                    page_matches = True
                continue
            if _same_page(resolved.get("/Pg"), target_objgen):
                page_matches = True
                break
            inner_kids = resolved.get("/K")
            if inner_kids is not None:
                items = list(inner_kids) if isinstance(inner_kids, pikepdf.Array) else [inner_kids]
                stack.extend(items)
    if not page_matches:
        return None

    def _walk_rows(node: pikepdf.Dictionary, is_header: bool):
        for child in _iter_struct_elem_children(node):
            child_stype = str(child.get("/S", ""))
            if child_stype == "/TR":
                yield child, is_header
            elif child_stype == "/THead":
                yield from _walk_rows(child, True)
            elif child_stype in ("/TBody", "/TFoot"):
                yield from _walk_rows(child, False)

    rows: list[dict] = []
    for tr, forced_header in _walk_rows(table_node, False):
        cells: list[dict] = []
        for cell in _iter_struct_elem_children(tr):
            cell_stype = str(cell.get("/S", ""))
            if cell_stype not in ("/TH", "/TD"):
                continue
            cell_mcids: list[int] = []
            _collect_descendant_mcids(cell, cell_mcids)
            if not cell_mcids:
                continue
            cell_dict: dict = {
                "mcid": cell_mcids[0],
                "tag": cell_stype.lstrip("/"),
            }
            element_id = cell.get("/ID")
            if element_id is not None:
                cell_dict["element_id"] = str(element_id)
            if cell_stype == "/TH":
                scope = _extract_scope_attribute(cell)
                if scope:
                    cell_dict["scope"] = scope
            else:
                header_ids = _extract_td_headers(cell)
                if header_ids:
                    cell_dict["headers"] = header_ids
            cells.append(cell_dict)
        if not cells:
            continue
        is_header_row = forced_header or all(c["tag"] == "TH" for c in cells)
        rows.append({"header": is_header_row, "cells": cells})

    if not rows:
        return None
    return {"rows": rows}


def extract_source_table_specs(
    source_pdf: pikepdf.Pdf,
    source_page: pikepdf.Page,
) -> list[dict]:
    """Return ``table_spec`` dicts for every /Table on *source_page*.

    Walks the source PDF's structure tree, finds /Table elements whose
    descendants include at least one MCR pointing at ``source_page``, and
    captures their TR/TH/TD hierarchy so the rebuild pipeline can reproduce
    it. Returns ``[]`` when the source PDF has no structure tree or no
    tables on this page — the caller should treat this as "no tables to
    preserve", not an error.
    """
    root = source_pdf.Root.get("/StructTreeRoot")
    if root is None:
        return []
    target_page_obj = source_page.obj

    specs: list[dict] = []
    seen_objgen: set[tuple[int, int]] = set()

    def _visit(node: pikepdf.Dictionary) -> None:
        objgen = getattr(node, "objgen", None)
        if objgen is not None and objgen != (0, 0):
            if objgen in seen_objgen:
                return
            seen_objgen.add(objgen)
        stype = str(node.get("/S", ""))
        if stype == "/Table":
            spec = _build_table_spec_for_page(node, target_page_obj)
            if spec is not None:
                specs.append(spec)
            # Nested tables are uncommon; skipping the subtree avoids
            # double-counting cells as a row of the outer table.
            return
        for child in _iter_struct_elem_children(node):
            _visit(child)

    _visit(root)
    return specs


def rebuild_page_preserving(
    source_pdf: pikepdf.Pdf,
    source_page: pikepdf.Page,
    target_pdf: pikepdf.Pdf,
    target_page: pikepdf.Page,
) -> MCIDManifest:
    """Rebuild a page's content stream with proper BDC/EMC tagging.

    Copies all original operators from *source_page* into *target_page*,
    wrapping untagged spans with marked-content sequences and preserving
    existing valid tagged sequences. When the source PDF has /Table
    struct elements on this page, their TR/TH/TD hierarchy is captured as
    a ``table_spec`` on a dedicated Table MCIDEntry so
    ``build_structure_tree`` can rebuild the full Table/THead/TBody/TR/TH/TD
    subtree in the output (REMEDY-69 #9).

    Args:
        source_pdf: The open source PDF.
        source_page: The page to copy operators from.
        target_pdf: The target PDF being built.
        target_page: The target page to write into.

    Returns:
        An :class:`MCIDManifest` describing all MCIDs emitted on the page.
    """
    tracker = GraphicsStateTracker()
    instructions = tracker.track(source_page)

    spans = classify_operator_spans(source_page)

    # Build a lookup: instruction index -> span (if any)
    index_to_span: dict[int, OperatorSpan] = {}
    for span in spans:
        for idx in range(span.start_index, span.end_index + 1):
            index_to_span[idx] = span

    manifest = MCIDManifest()
    next_mcid = 0
    # Track which spans have already been opened (by start_index)
    opened_spans: set[int] = set()

    # Build the new instruction list
    new_instructions: list[tuple[list, Operator]] = []

    for ann in instructions:
        idx = ann.index
        span = index_to_span.get(idx)

        if span is None:
            # Uncovered instruction (state op) — emit as-is, but skip stray
            # BDC/BMC/EMC that aren't part of any span
            if ann.operator in ("BDC", "BMC", "EMC"):
                continue
            new_instructions.append((list(ann.operands), Operator(ann.operator)))
            continue

        # This instruction belongs to a span
        if span.span_type == "marked" and span.existing_mcid is not None:
            # --- Preserve existing marked-content sequence as-is ---
            if span.start_index not in opened_spans:
                opened_spans.add(span.start_index)
                # Record in manifest
                manifest.entries.append(MCIDEntry(
                    mcid=span.existing_mcid,
                    tag=span.existing_tag or "Span",
                    semantic_type=_tag_to_semantic(span.existing_tag),
                ))
            # Emit the original operator unchanged
            new_instructions.append((list(ann.operands), Operator(ann.operator)))

        elif span.span_type == "marked" and span.existing_mcid is None:
            # --- Existing marked content without MCID: re-tag ---
            if span.start_index not in opened_spans:
                opened_spans.add(span.start_index)
                mcid = next_mcid
                next_mcid += 1
                tag = span.existing_tag or "Span"
                manifest.entries.append(MCIDEntry(
                    mcid=mcid,
                    tag=tag,
                    semantic_type=_tag_to_semantic(tag),
                ))
                # Emit new BDC with MCID (replacing original BDC/BMC)
                new_instructions.append((
                    [Name("/" + tag), Dictionary({"/MCID": mcid})],
                    Operator("BDC"),
                ))
            elif idx == span.end_index:
                # Close with EMC (replacing original EMC)
                new_instructions.append(([], Operator("EMC")))
            else:
                # Interior: skip original BDC/BMC/EMC, emit other ops
                if ann.operator not in ("BDC", "BMC", "EMC"):
                    new_instructions.append(
                        (list(ann.operands), Operator(ann.operator))
                    )
            # For the start instruction (BDC/BMC) we already emitted the
            # replacement above. For end (EMC) we emitted above. For
            # interior ops that are BDC/BMC/EMC we skip.
            # Interior non-marker ops are emitted.
            if idx != span.start_index and idx != span.end_index:
                # Already handled above in the else branch
                pass

        elif span.span_type == "text":
            # --- Wrap text block with /P BDC ... EMC ---
            if span.start_index not in opened_spans:
                opened_spans.add(span.start_index)
                mcid = next_mcid
                next_mcid += 1
                manifest.entries.append(MCIDEntry(
                    mcid=mcid,
                    tag="P",
                    semantic_type="paragraph",
                ))
                # Inject BDC before the BT
                new_instructions.append((
                    [Name("/P"), Dictionary({"/MCID": mcid})],
                    Operator("BDC"),
                ))
            # Emit original instruction
            new_instructions.append((list(ann.operands), Operator(ann.operator)))
            if idx == span.end_index:
                # After ET, close with EMC
                new_instructions.append(([], Operator("EMC")))

        elif span.span_type == "image":
            # --- Wrap image Do with /Figure BDC ... EMC ---
            if span.start_index not in opened_spans:
                opened_spans.add(span.start_index)
                mcid = next_mcid
                next_mcid += 1
                manifest.entries.append(MCIDEntry(
                    mcid=mcid,
                    tag="Figure",
                    semantic_type="figure",
                ))
                new_instructions.append((
                    [Name("/Figure"), Dictionary({"/MCID": mcid})],
                    Operator("BDC"),
                ))
            new_instructions.append((list(ann.operands), Operator(ann.operator)))
            if idx == span.end_index:
                new_instructions.append(([], Operator("EMC")))

        elif span.span_type == "vector":
            # --- Wrap vector path with /Artifact BDC ... EMC ---
            if span.start_index not in opened_spans:
                opened_spans.add(span.start_index)
                mcid = next_mcid
                next_mcid += 1
                manifest.entries.append(MCIDEntry(
                    mcid=mcid,
                    tag="Artifact",
                    semantic_type="artifact",
                ))
                new_instructions.append((
                    [Name("/Artifact"), Dictionary({"/MCID": mcid})],
                    Operator("BDC"),
                ))
            new_instructions.append((list(ann.operands), Operator(ann.operator)))
            if idx == span.end_index:
                new_instructions.append(([], Operator("EMC")))

        else:
            # Fallback: emit as-is
            new_instructions.append((list(ann.operands), Operator(ann.operator)))

    # Write rebuilt stream to target page
    new_stream = pikepdf.unparse_content_stream(new_instructions)
    target_page.contents_coalesce()

    # Ensure target page has a /Contents stream
    if "/Contents" not in target_page.obj:
        target_page.obj["/Contents"] = target_pdf.make_stream(new_stream)
    else:
        target_page.obj["/Contents"].write(new_stream)

    # Copy resources from source to target
    source_resources = source_page.obj.get("/Resources")
    if source_resources is not None:
        try:
            # Ensure the resource dict is an indirect object before copying
            if not source_resources.is_indirect:
                source_resources = source_pdf.make_indirect(source_resources)
            target_page.obj["/Resources"] = target_pdf.copy_foreign(source_resources)
        except (pikepdf.ForeignObjectError, RuntimeError):
            # Source and target are the same PDF — just assign directly
            target_page.obj["/Resources"] = source_resources

    # Capture table structure from source so structure_builder can emit a
    # real Table/THead/TBody/TR/TH/TD subtree instead of flat leaf cells.
    # Existing TH/TD MCIDEntries from the preservation path above stay in
    # the manifest; structure_builder deduplicates via the cell MCIDs
    # listed in table_spec (table_consumed_mcids).
    try:
        table_specs = extract_source_table_specs(source_pdf, source_page)
    except Exception as exc:  # pragma: no cover — defensive, never fatal
        logger.warning("table-spec extraction failed, falling back to flat cells: %s", exc)
        table_specs = []
    for spec in table_specs:
        manifest.entries.append(MCIDEntry(
            mcid=_TABLE_PARENT_SENTINEL_MCID,
            tag="Table",
            semantic_type="table",
            table_spec=spec,
        ))

    return manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tag_to_semantic(tag: str | None) -> str:
    """Map a PDF structure tag to a human-readable semantic type."""
    if tag is None:
        return "span"
    tag_lower = tag.lower()
    mapping = {
        "p": "paragraph",
        "h1": "heading",
        "h2": "heading",
        "h3": "heading",
        "h4": "heading",
        "h5": "heading",
        "h6": "heading",
        "h": "heading",
        "figure": "figure",
        "artifact": "artifact",
        "span": "span",
        "table": "table",
        "tr": "table_row",
        "th": "table_header",
        "td": "table_cell",
        "l": "list",
        "li": "list_item",
        "lbl": "label",
        "lbody": "list_body",
        "link": "link",
    }
    return mapping.get(tag_lower, "span")
