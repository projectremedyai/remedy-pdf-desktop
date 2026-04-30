"""Build PDF/UA-1 structure tree and ParentTree from MCID manifests.

Takes the :class:`MCIDManifest` produced by
:mod:`project_remedy.faithful_rebuild.content_builder` and wires up the
full structure tree required for PDF/UA-1 conformance:

    StructTreeRoot -> Document -> Sect (per page) -> leaf elements

Each non-Artifact MCID in the content stream receives a struct element
and a corresponding entry in the ParentTree.
"""

from __future__ import annotations

import logging

import pikepdf
from pikepdf import Array, Dictionary, Name, String

from project_remedy.faithful_rebuild.models import MCIDEntry, MCIDManifest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard PDF/UA RoleMap
# ---------------------------------------------------------------------------

_STANDARD_ROLE_MAP: dict[str, str] = {
    "/DocumentFragment": "/Sect",
    "/Textbody": "/P",
    "/Footnote": "/Note",
    "/Endnote": "/Note",
    "/Title": "/H1",
    "/Subtitle": "/H2",
}


# ---------------------------------------------------------------------------
# ParentTreeBuilder
# ---------------------------------------------------------------------------


class ParentTreeBuilder:
    """Incrementally build the ``/ParentTree`` number tree.

    The ParentTree maps ``/StructParents`` page keys to arrays of struct
    elements, where each array index corresponds to an MCID on that page.

    Usage::

        builder = ParentTreeBuilder(pdf)
        key = builder.register_page(page)
        builder.attach_leaf(page, mcid=0, leaf_elem=elem)
        ...
        num_tree = builder.build_number_tree()
    """

    def __init__(self, pdf: pikepdf.Pdf) -> None:
        self._pdf = pdf
        self.next_key: int = 0
        # page objgen -> (key, list[elem | pikepdf.Null])
        self._page_map: dict[tuple[int, int], tuple[int, list]] = {}

    def register_page(self, page: pikepdf.Page) -> int:
        """Assign ``/StructParents`` to *page* and return the key."""
        key = self.next_key
        self.next_key += 1
        page.obj["/StructParents"] = key
        self._page_map[page.obj.objgen] = (key, [])
        return key

    def attach_leaf(
        self,
        page: pikepdf.Page,
        mcid: int,
        leaf_elem: pikepdf.Object,
    ) -> None:
        """Map *mcid* on *page* to *leaf_elem* in the parent array.

        Pads with ``pikepdf.Null`` if there are gaps in the MCID sequence.
        """
        objgen = page.obj.objgen
        if objgen not in self._page_map:
            raise ValueError(
                f"Page {objgen} not registered; call register_page() first"
            )
        _, arr = self._page_map[objgen]
        while len(arr) <= mcid:
            arr.append(None)
        arr[mcid] = leaf_elem

    def build_number_tree(self) -> Dictionary:
        """Return a ``/ParentTree`` dictionary with ``/Nums`` array."""
        nums = Array()
        for key, arr in sorted(self._page_map.values(), key=lambda x: x[0]):
            nums.append(key)
            nums.append(self._pdf.make_indirect(Array(arr)))
        return Dictionary({"/Nums": nums})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _append_kid(
    parent: pikepdf.Object,
    child: pikepdf.Object,
) -> None:
    """Append *child* to the ``/K`` field of *parent*.

    Handles three cases:
    - ``/K`` absent or null: set ``/K`` to *child* directly.
    - ``/K`` is a single object: convert to an ``Array`` of two.
    - ``/K`` is already an ``Array``: append.
    """
    existing = parent.get("/K")
    if existing is None:
        parent["/K"] = child
    elif isinstance(existing, Array):
        existing.append(child)
    else:
        parent["/K"] = Array([existing, child])


def _make_struct_elem(
    pdf: pikepdf.Pdf,
    tag: str,
    parent: pikepdf.Object,
    page_obj: pikepdf.Object | None = None,
) -> pikepdf.Object:
    """Create an indirect ``/StructElem`` and attach it as a child of *parent*.

    Args:
        pdf: The target PDF.
        tag: Structure tag (e.g. ``"Document"``, ``"Sect"``, ``"P"``).
        parent: Parent struct element or StructTreeRoot.
        page_obj: Optional page dictionary to set ``/Pg``.

    Returns:
        The new indirect struct element.
    """
    d: dict = {
        "/S": Name("/" + tag),
        "/Type": Name("/StructElem"),
        "/P": parent,
    }
    if page_obj is not None:
        d["/Pg"] = page_obj
    elem = pdf.make_indirect(Dictionary(d))
    _append_kid(parent, elem)
    return elem


def _make_leaf(
    pdf: pikepdf.Pdf,
    tag: str,
    parent: pikepdf.Object,
    page: pikepdf.Page,
    mcid: int,
    parent_tree: ParentTreeBuilder,
    *,
    alt_text: str | None = None,
    element_id: str | None = None,
    attributes: dict | None = None,
    headers: list[str] | None = None,
) -> pikepdf.Object:
    """Create a leaf struct element with an ``/MCR`` child.

    The element is registered in the ParentTree and attached to *parent*.

    Args:
        pdf: The target PDF.
        tag: Structure tag (e.g. ``"P"``, ``"H1"``, ``"Figure"``).
        parent: Parent struct element.
        page: The page this element appears on.
        mcid: Marked-content identifier.
        parent_tree: Builder for the ParentTree.
        alt_text: Optional alternate text (set as ``/Alt``).
        element_id: Optional ID attribute (set as ``/ID``).
        attributes: Optional dict of PDF attributes to set on the element.
        headers: Optional list of header IDs for TD elements.

    Returns:
        The new indirect struct element.
    """
    mcr = Dictionary({
        "/Type": Name("/MCR"),
        "/Pg": page.obj,
        "/MCID": mcid,
    })

    d: dict = {
        "/S": Name("/" + tag),
        "/Type": Name("/StructElem"),
        "/P": parent,
        "/Pg": page.obj,
        "/K": mcr,
    }

    if alt_text is not None:
        d["/Alt"] = String(alt_text)

    if element_id is not None:
        d["/ID"] = String(element_id)

    if headers:
        d["/Headers"] = Array([String(h) for h in headers])

    elem = pdf.make_indirect(Dictionary(d))

    if attributes:
        attr_dict: dict = {"/O": Name("/Table")}
        if "scope" in attributes:
            attr_dict["/Scope"] = Name("/" + attributes["scope"])
        if attributes:
            elem["/A"] = pdf.make_indirect(Dictionary(attr_dict))

    _append_kid(parent, elem)
    parent_tree.attach_leaf(page, mcid, elem)
    return elem


def _build_table_structure(
    pdf: pikepdf.Pdf,
    page: pikepdf.Page,
    parent: pikepdf.Object,
    table_spec: dict,
    parent_tree: ParentTreeBuilder,
) -> pikepdf.Object:
    """Build Table/THead/TBody/TR/TH/TD subtree from *table_spec*.

    Expected *table_spec* format::

        {
            "rows": [
                {
                    "header": True,
                    "cells": [
                        {"mcid": 1, "tag": "TH", "scope": "Column",
                         "element_id": "h1", "content": "..."},
                        ...
                    ]
                },
                {
                    "header": False,
                    "cells": [
                        {"mcid": 3, "tag": "TD", "headers": ["h1"],
                         "content": "..."},
                        ...
                    ]
                },
            ]
        }

    Returns:
        The Table struct element.
    """
    table_elem = _make_struct_elem(pdf, "Table", parent, page.obj)

    rows = table_spec.get("rows", [])
    header_rows = [r for r in rows if r.get("header")]
    body_rows = [r for r in rows if not r.get("header")]

    # THead
    if header_rows:
        thead = _make_struct_elem(pdf, "THead", table_elem, page.obj)
        for row in header_rows:
            tr = _make_struct_elem(pdf, "TR", thead, page.obj)
            for cell in row.get("cells", []):
                cell_mcid = cell.get("mcid")
                cell_tag = cell.get("tag", "TH")
                scope = cell.get("scope")
                eid = cell.get("element_id")
                attrs = {}
                if scope:
                    attrs["scope"] = scope
                _make_leaf(
                    pdf, cell_tag, tr, page, cell_mcid, parent_tree,
                    element_id=eid,
                    attributes=attrs if attrs else None,
                )

    # TBody
    if body_rows:
        tbody = _make_struct_elem(pdf, "TBody", table_elem, page.obj)
        for row in body_rows:
            tr = _make_struct_elem(pdf, "TR", tbody, page.obj)
            for cell in row.get("cells", []):
                cell_mcid = cell.get("mcid")
                cell_tag = cell.get("tag", "TD")
                cell_headers = cell.get("headers")
                _make_leaf(
                    pdf, cell_tag, tr, page, cell_mcid, parent_tree,
                    headers=cell_headers,
                )

    return table_elem


def _build_xmp_packet(title: str) -> bytes:
    """Build an XMP metadata packet with ``dc:title`` and ``pdfuaid:part=1``.

    Args:
        title: Document title for ``dc:title``.

    Returns:
        UTF-8 encoded XMP packet bytes.
    """
    # Escape XML special characters in the title
    safe_title = (
        title
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

    xmp = (
        '<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '    <rdf:Description rdf:about=""\n'
        '        xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '        xmlns:pdfuaid="http://www.aiim.org/pdfua/ns/id/">\n'
        '      <dc:title>\n'
        '        <rdf:Alt>\n'
        f'          <rdf:li xml:lang="x-default">{safe_title}</rdf:li>\n'
        '        </rdf:Alt>\n'
        '      </dc:title>\n'
        '      <pdfuaid:part>1</pdfuaid:part>\n'
        '    </rdf:Description>\n'
        '  </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )
    return xmp.encode("utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_structure_tree(
    pdf: pikepdf.Pdf,
    manifests: dict[int, MCIDManifest],
    *,
    title: str = "Untitled",
    lang: str = "en-US",
) -> None:
    """Create a complete PDF/UA-1 structure tree from MCID manifests.

    Builds the following tree shape::

        StructTreeRoot
          -> Document
            -> Sect (per page)
              -> leaf elements (H1-H6, P, Figure, Table, L, LI, Link, Form, Span)

    Artifacts are skipped (no struct element). Table entries with a
    ``table_spec`` are expanded into Table/THead/TBody/TR/TH/TD subtrees.

    Also sets:
    - ``/MarkInfo << /Marked true >>``
    - ``/Lang``
    - ``/ViewerPreferences << /DisplayDocTitle true >>``
    - XMP metadata with ``pdfuaid:part=1``
    - ``/docinfo /Title``

    Args:
        pdf: The target PDF (modified in place).
        manifests: Mapping of page index to :class:`MCIDManifest`.
        title: Document title.
        lang: BCP-47 language tag.
    """
    pt_builder = ParentTreeBuilder(pdf)

    # --- StructTreeRoot ---
    struct_root = pdf.make_indirect(Dictionary({
        "/Type": Name("/StructTreeRoot"),
    }))

    # --- RoleMap ---
    role_map = Dictionary()
    for custom, standard in _STANDARD_ROLE_MAP.items():
        role_map[Name(custom)] = Name(standard)
    struct_root["/RoleMap"] = role_map

    # --- Document element ---
    doc_elem = _make_struct_elem(pdf, "Document", struct_root)

    # --- Process pages in order ---
    # Track MCIDs that have been consumed by table_spec expansion so we don't
    # create duplicate leaf elements for them.
    for page_idx in sorted(manifests.keys()):
        if page_idx >= len(pdf.pages):
            logger.warning(
                "Manifest references page %d but PDF only has %d pages; skipping",
                page_idx,
                len(pdf.pages),
            )
            continue

        page = pdf.pages[page_idx]
        pt_builder.register_page(page)
        manifest = manifests[page_idx]

        sect = _make_struct_elem(pdf, "Sect", doc_elem, page.obj)

        # Collect MCIDs consumed by table structures so we skip them as
        # standalone leaves.
        table_consumed_mcids: set[int] = set()
        for entry in manifest.entries:
            if entry.tag == "Table" and entry.table_spec:
                # Collect MCIDs from the table_spec rows
                for row in entry.table_spec.get("rows", []):
                    for cell in row.get("cells", []):
                        cell_mcid = cell.get("mcid")
                        if cell_mcid is not None:
                            table_consumed_mcids.add(cell_mcid)

        for entry in manifest.entries:
            # Skip artifacts entirely
            if entry.tag == "Artifact":
                continue

            # Skip MCIDs already consumed by table structures
            if entry.mcid in table_consumed_mcids:
                continue

            # Table with table_spec: build full subtree
            if entry.tag == "Table" and entry.table_spec:
                _build_table_structure(
                    pdf, page, sect, entry.table_spec, pt_builder,
                )
                continue

            # All other tags: create leaf with MCR
            _make_leaf(
                pdf,
                entry.tag,
                sect,
                page,
                entry.mcid,
                pt_builder,
                alt_text=entry.alt_text,
                element_id=entry.element_id,
            )

    # --- Wire ParentTree ---
    parent_tree_dict = pt_builder.build_number_tree()
    struct_root["/ParentTree"] = pdf.make_indirect(parent_tree_dict)
    struct_root["/ParentTreeNextKey"] = pt_builder.next_key

    # --- Attach to document catalog ---
    pdf.Root["/StructTreeRoot"] = struct_root

    # --- MarkInfo ---
    pdf.Root["/MarkInfo"] = Dictionary({"/Marked": True})

    # --- Lang ---
    pdf.Root["/Lang"] = String(lang)

    # --- ViewerPreferences ---
    pdf.Root["/ViewerPreferences"] = Dictionary({"/DisplayDocTitle": True})

    # --- XMP metadata ---
    xmp_bytes = _build_xmp_packet(title)
    metadata_stream = pdf.make_stream(xmp_bytes)
    metadata_stream["/Type"] = Name("/Metadata")
    metadata_stream["/Subtype"] = Name("/XML")
    pdf.Root["/Metadata"] = metadata_stream

    # --- DocInfo /Title ---
    pdf.docinfo["/Title"] = String(title)

    logger.info(
        "Built structure tree: %d pages, %d total entries",
        len(manifests),
        sum(len(m.entries) for m in manifests.values()),
    )
