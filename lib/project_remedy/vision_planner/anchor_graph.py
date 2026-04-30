"""Build anchor graph mapping visual content to PDF structure elements."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pikepdf

from project_remedy.pdf_checker import walk_structure_tree
from project_remedy.pdf_semantics import find_node_page

logger = logging.getLogger(__name__)


def _get_mcids_from_node(node: pikepdf.Dictionary) -> list[int]:
    """Extract all MCIDs directly referenced by a structure element."""
    kids = node.get("/K")
    if kids is None:
        return []

    mcids: list[int] = []
    items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]

    for item in items:
        if isinstance(item, (int, pikepdf.Object)) and not isinstance(
            item, pikepdf.Dictionary
        ):
            try:
                mcids.append(int(item))
            except (TypeError, ValueError):
                pass
        elif isinstance(item, pikepdf.Dictionary):
            mcid = item.get("/MCID")
            if mcid is not None:
                try:
                    mcids.append(int(mcid))
                except (TypeError, ValueError):
                    pass

    return mcids


def _get_struct_type(node: pikepdf.Dictionary) -> str:
    """Return the structure type name as a plain string."""
    s = node.get("/S")
    if s is None:
        return ""
    return str(s).lstrip("/")


def build_anchor_graph(pdf_path: Path) -> dict:
    """Build anchor graph mapping visual content to PDF structure.

    Returns dict with "anchors" list and "page_count" int.
    """
    anchors: list[dict[str, Any]] = []

    # -- Extract text blocks and images per page using fitz --
    fitz_doc = fitz.open(str(pdf_path))
    page_count = len(fitz_doc)

    page_blocks: dict[int, list[dict]] = {}
    for page_idx in range(page_count):
        page = fitz_doc[page_idx]
        text_dict = page.get_text("dict")
        blocks = []
        for block in text_dict.get("blocks", []):
            bbox = block.get("bbox", (0, 0, 0, 0))
            if block.get("type") == 0:  # text
                text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span.get("text", "")
                    text += " "
                text = text.strip()[:80]
                if text:
                    blocks.append({
                        "bbox_pdf": list(bbox),
                        "text_excerpt": text,
                        "block_type": "text",
                    })
            elif block.get("type") == 1:  # image
                blocks.append({
                    "bbox_pdf": list(bbox),
                    "text_excerpt": "[image]",
                    "block_type": "image",
                })
        page_blocks[page_idx] = blocks
    fitz_doc.close()

    # -- Walk structure tree to collect MCID → struct info per page --
    # Maps (page_idx, mcid) → struct info
    mcid_struct_map: dict[tuple[int, int], dict] = {}
    try:
        pdf = pikepdf.open(str(pdf_path))
        for node, _depth, _parent in walk_structure_tree(pdf):
            stype = _get_struct_type(node)
            if not stype or stype == "StructTreeRoot":
                continue

            page_idx = find_node_page(node, pdf)
            if page_idx is None:
                continue

            struct_elem_id = None
            try:
                if hasattr(node, "objgen"):
                    struct_elem_id = f"obj_{node.objgen[0]}_{node.objgen[1]}"
            except Exception:
                pass

            for mcid in _get_mcids_from_node(node):
                mcid_struct_map[(page_idx, mcid)] = {
                    "struct_type": stype,
                    "struct_elem_id": struct_elem_id,
                }
        pdf.close()
    except Exception as e:
        logger.warning("Failed to walk structure tree: %s", e)

    # -- Build anchors: one per text/image block per page --
    # Track which MCIDs have been assigned to avoid double-mapping
    assigned_mcids: set[tuple[int, int]] = set()
    anchor_idx = 0

    for page_idx in range(page_count):
        for block in page_blocks.get(page_idx, []):
            anchor_id = f"p{page_idx}_a{anchor_idx}"
            anchor: dict[str, Any] = {
                "anchor_id": anchor_id,
                "page": page_idx,
                "bbox_pdf": block["bbox_pdf"],
                "text_excerpt": block["text_excerpt"],
                "block_type": block["block_type"],
                "mcids": [],
                "struct_elem_id": None,
                "struct_type": None,
                "tagged": False,
            }

            # Assign first unmatched MCID on this page
            for key, info in mcid_struct_map.items():
                if key[0] == page_idx and key not in assigned_mcids:
                    anchor["mcids"].append(key[1])
                    anchor["struct_elem_id"] = info["struct_elem_id"]
                    anchor["struct_type"] = info["struct_type"]
                    anchor["tagged"] = True
                    assigned_mcids.add(key)
                    break

            anchors.append(anchor)
            anchor_idx += 1

    return {
        "anchors": anchors,
        "page_count": page_count,
    }
