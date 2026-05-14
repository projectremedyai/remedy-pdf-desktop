"""Shared PDF semantics for checker and fixer logic."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pikepdf

MULTIMEDIA_ANNOT_TYPES = frozenset({"/RichMedia", "/Screen", "/Movie", "/Sound"})


def resolve_pdf_object(obj):
    """Best-effort resolve for pikepdf objects."""
    if isinstance(obj, pikepdf.Object) and obj.is_indirect:
        try:
            return obj.resolve()
        except Exception:
            return obj
    return obj


def _page_objgen_index_map(pdf: pikepdf.Pdf) -> dict[tuple[int, int], int]:
    """Return a cached page objgen -> page index map for the document."""
    cached = getattr(pdf, "_page_objgen_index_map", None)
    if isinstance(cached, dict):
        return cached

    mapping: dict[tuple[int, int], int] = {}
    for idx, page in enumerate(pdf.pages):
        try:
            mapping[page.obj.objgen] = idx
        except Exception:
            continue

    pdf._page_objgen_index_map = mapping
    return mapping


def _node_page_cache(pdf: pikepdf.Pdf) -> dict[tuple[str, object], int | None]:
    """Return a cached structure-node -> page index map for the document."""
    cached = getattr(pdf, "_node_page_cache", None)
    if isinstance(cached, dict):
        return cached
    cache: dict[tuple[str, object], int | None] = {}
    pdf._node_page_cache = cache
    return cache


def _node_page_cache_key(node: pikepdf.Dictionary) -> tuple[str, object]:
    """Return a stable cache key for a structure node."""
    resolved = resolve_pdf_object(node)
    try:
        objgen = resolved.objgen
    except Exception:
        objgen = None
    if objgen is not None and objgen != (0, 0):
        return ("objgen", objgen)
    return ("id", id(resolved))


def iter_resolved_kids(node: pikepdf.Dictionary) -> Iterator[object]:
    """Yield a node's resolved /K children without materializing large arrays."""
    kids = node.get("/K")
    if kids is None:
        return
    if isinstance(kids, pikepdf.Array):
        for idx in range(len(kids)):
            yield resolve_pdf_object(kids[idx])
    else:
        yield resolve_pdf_object(kids)


def node_has_struct_children(node: pikepdf.Dictionary) -> bool:
    """True when the node has at least one child structure element."""
    kids = node.get("/K")
    if kids is None:
        return False
    items = kids if isinstance(kids, pikepdf.Array) else [kids]
    for item in items:
        child = resolve_pdf_object(item)
        if isinstance(child, pikepdf.Dictionary) and "/S" in child:
            return True
    return False


def node_has_annotation_ref(node: pikepdf.Dictionary) -> bool:
    """True when the node references an annotation through OBJR or /Obj."""
    for child in iter_resolved_kids(node):
        if not isinstance(child, pikepdf.Dictionary):
            continue
        obj_type = str(child.get("/Type", ""))
        if obj_type == "/OBJR" or child.get("/Obj") is not None:
            return True
    return False


def node_has_direct_content(node: pikepdf.Dictionary) -> bool:
    """True if the node has MCR/MCID/OBJR-like direct content children."""
    for child in iter_resolved_kids(node):
        if not isinstance(child, pikepdf.Dictionary):
            return True
        if "/S" not in child:
            return True
    return False


def node_has_content_association(node: pikepdf.Dictionary) -> bool:
    """True when an alt-bearing node is tied to actual rendered content."""
    stype = str(node.get("/S", "")).lstrip("/")
    if stype in {"Table", "Formula"}:
        return node_has_direct_content(node) or node_has_struct_children(node)
    return node_has_direct_content(node) or node_has_annotation_ref(node)


def document_requires_bookmarks(pdf: pikepdf.Pdf) -> bool:
    """Adobe-style rule: bookmarks are required for documents over 20 pages."""
    return len(pdf.pages) > 20


def document_has_bookmarks(pdf: pikepdf.Pdf) -> bool:
    """True when the document has a non-empty /Outlines tree."""
    outlines = pdf.Root.get("/Outlines")
    return bool(outlines and outlines.get("/Count", 0) != 0)


def get_page_index_from_ref(pdf: pikepdf.Pdf, page_ref) -> int | None:
    """Resolve a page reference to a 0-based page index."""
    page_obj = resolve_pdf_object(page_ref)
    try:
        target = page_obj.objgen
    except Exception:
        return None
    return _page_objgen_index_map(pdf).get(target)


def find_node_page(node: pikepdf.Dictionary, pdf: pikepdf.Pdf) -> int | None:
    """Find a structure node's page via /Pg or child MCR references."""
    cache = _node_page_cache(pdf)
    cache_key = _node_page_cache_key(node)
    if cache_key in cache:
        return cache[cache_key]

    pg = node.get("/Pg")
    if pg is not None:
        idx = get_page_index_from_ref(pdf, pg)
        if idx is not None:
            cache[cache_key] = idx
            return idx

    for child in iter_resolved_kids(node):
        if isinstance(child, pikepdf.Dictionary) and "/Pg" in child:
            idx = get_page_index_from_ref(pdf, child["/Pg"])
            if idx is not None:
                cache[cache_key] = idx
                return idx
    cache[cache_key] = None
    return None


def get_rendered_multimedia_names(page: pikepdf.Page) -> set[str]:
    """Return multimedia annotation subtype names present on the page.

    PDF multimedia is represented through rich-media/screen/movie/sound
    annotations, not ordinary /Form XObjects used for layout reuse.
    """
    annots = page.get("/Annots")
    if not annots:
        return set()

    names: set[str] = set()
    for annot_ref in annots:
        annot = resolve_pdf_object(annot_ref)
        if not isinstance(annot, pikepdf.Dictionary):
            continue
        subtype = str(annot.get("/Subtype", ""))
        if subtype in MULTIMEDIA_ANNOT_TYPES:
            names.add(subtype.lstrip("/"))
    return names


def get_rendered_image_names(page: pikepdf.Page) -> list[str]:
    """Return rendered image XObject names in content-stream order."""
    resources = page.get("/Resources")
    if not resources:
        return []

    xobjects = resources.get("/XObject")
    if not xobjects:
        return []

    try:
        instructions = pikepdf.parse_content_stream(page)
    except Exception:
        return []

    names: list[str] = []
    for operands, operator in instructions:
        if str(operator) != "Do" or not operands:
            continue
        raw_name = str(operands[0]).lstrip("/")
        try:
            xobj_ref = xobjects.get(f"/{raw_name}") or xobjects.get(raw_name)
        except Exception:
            xobj_ref = xobjects.get(raw_name)
        if xobj_ref is None:
            continue
        xobj = resolve_pdf_object(xobj_ref)
        if isinstance(xobj, pikepdf.Stream) and str(xobj.get("/Subtype", "")) == "/Image":
            names.append(raw_name)
    return names
