"""Execute remediation plan operations via pikepdf."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pikepdf

logger = logging.getLogger(__name__)


@dataclass
class OpResult:
    """Structured result from a single remediation operation."""

    status: str  # "applied", "skipped", "error"
    action: str
    op_id: str
    detail: str


def _read_page_content(page) -> bytes:
    """Read raw content stream bytes from a page."""
    contents = page.get("/Contents")
    if contents is None:
        return b""
    if isinstance(contents, pikepdf.Array):
        parts: list[bytes] = []
        for stream in contents:
            try:
                parts.append(stream.read_bytes())
            except Exception:
                pass
        return b"\n".join(parts)
    try:
        return contents.read_bytes()
    except Exception:
        return b""


def _find_tagged_mcid_match(raw: str, mcid: int) -> re.Match[str] | None:
    """Locate any tagged marked-content block for a specific MCID."""
    pattern = rf"/\w+\s*<<[^>]*?/MCID\s+{mcid}\b[^>]*>>\s*BDC(.*?)EMC"
    return re.search(pattern, raw, re.S)


def _get_struct_type(node: pikepdf.Dictionary) -> str:
    s = node.get("/S")
    return str(s).lstrip("/") if s else ""


def _remove_node_from_parent(parent: pikepdf.Dictionary, node: pikepdf.Dictionary) -> bool:
    """Remove node from its parent's /K entry."""
    kids = parent.get("/K")
    if kids is None:
        return False

    items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
    new_items = []
    removed = False

    for kid in items:
        try:
            if hasattr(kid, "objgen") and hasattr(node, "objgen") and kid.objgen == node.objgen:
                removed = True
                continue
        except Exception:
            pass
        new_items.append(kid)

    if not removed:
        return False

    if not new_items:
        del parent["/K"]
    elif len(new_items) == 1:
        parent["/K"] = new_items[0]
    else:
        parent["/K"] = pikepdf.Array(new_items)
    return True


def _resolve_anchor_mcids(
    operation: dict, anchor_graph: dict
) -> tuple[int | None, list[int], str | None]:
    """Resolve target_anchors to (page_idx, mcids, struct_elem_id) from the anchor graph.

    Returns a 3-tuple so callers that only destructure two values still work
    (Python will raise, but all call sites are updated in this module).
    The third element is the ``struct_elem_id`` of the *first* resolved anchor,
    or ``None`` when unavailable.
    """
    target_anchors = operation.get("target_anchors", [])
    if not target_anchors:
        return None, [], None

    anchors_by_id = {a["anchor_id"]: a for a in anchor_graph.get("anchors", [])}

    page_idx = None
    mcids: list[int] = []
    struct_elem_id: str | None = None
    for anchor_id in target_anchors:
        anchor = anchors_by_id.get(anchor_id)
        if anchor is None:
            continue
        if page_idx is None:
            page_idx = anchor["page"]
        mcids.extend(anchor.get("mcids", []))
        if struct_elem_id is None:
            struct_elem_id = anchor.get("struct_elem_id")

    # Fallback: use operation's page field
    if page_idx is None:
        page_idx = operation.get("page")

    return page_idx, mcids, struct_elem_id


def _has_child_struct_elems(node: pikepdf.Dictionary) -> bool:
    """Return True if *node* has any child StructElems (descendants with /S)."""
    kids = node.get("/K")
    if kids is None:
        return False
    items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
    for item in items:
        if isinstance(item, pikepdf.Dictionary) and "/S" in item:
            return True
    return False


def _do_artifactize(
    pdf: pikepdf.Pdf,
    page_idx: int,
    mcids: list[int],
    reason: str,
    *,
    allow_artifactize: bool = False,
    op_id: str = "",
) -> list[OpResult]:
    """Rewrite tagged content as /Artifact and remove from structure tree.

    When *allow_artifactize* is ``False`` (the safe default), the operation is
    skipped to prevent orphan MCIDs in the structure tree.  When ``True``, the
    function additionally removes the target node from its parent in the
    structure tree (the previously dead-code path via ``_remove_node_from_parent``).
    Nodes with child StructElems are always skipped to avoid wholesale subtree
    destruction.
    """
    if not allow_artifactize:
        detail = "artifactize disabled (causes orphan MCIDs)"
        logger.info("artifactize skipped for %s: %s", op_id, detail)
        return [OpResult(status="skipped", action="artifactize", op_id=op_id, detail=detail)]

    if page_idx is None or page_idx < 0 or page_idx >= len(pdf.pages):
        detail = f"invalid page {page_idx}"
        return [OpResult(status="error", action="artifactize", op_id=op_id, detail=detail)]

    # --- safety: find the matching struct node and check for children --------
    from project_remedy.pdf_checker import walk_structure_tree

    target_node = None
    target_parent = None
    mcid_set = set(mcids)
    for node, _depth, parent in walk_structure_tree(pdf):
        node_mcids = set()
        kids = node.get("/K")
        if kids is None:
            continue
        items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
        for item in items:
            if isinstance(item, (int, pikepdf.Object)) and not isinstance(item, pikepdf.Dictionary):
                try:
                    node_mcids.add(int(item))
                except (TypeError, ValueError):
                    pass
            elif isinstance(item, pikepdf.Dictionary):
                m = item.get("/MCID")
                if m is not None:
                    try:
                        node_mcids.add(int(m))
                    except (TypeError, ValueError):
                        pass
        if node_mcids and node_mcids.intersection(mcid_set):
            target_node = node
            target_parent = parent
            break

    if target_node is not None and _has_child_struct_elems(target_node):
        detail = "target has child StructElems; refusing to artifactize subtree"
        logger.warning("artifactize skipped for %s: %s", op_id, detail)
        return [OpResult(status="skipped", action="artifactize", op_id=op_id, detail=detail)]

    # --- rewrite content stream ---------------------------------------------
    page = pdf.pages[page_idx]
    raw = _read_page_content(page).decode("latin-1", errors="replace")
    updated = raw
    results: list[OpResult] = []

    for mcid in mcids:
        match = _find_tagged_mcid_match(updated, mcid)
        if match is None:
            continue
        body = match.group(1).rstrip()
        replacement = f"/Artifact BMC\n{body}\nEMC"
        updated = updated[: match.start()] + replacement + updated[match.end() :]
        results.append(OpResult(
            status="applied",
            action="artifactize",
            op_id=op_id,
            detail=f"artifactized MCID {mcid} on page {page_idx}",
        ))

    if results:
        page["/Contents"] = pdf.make_stream(updated.encode("latin-1"))

        # Remove node from structure tree (previously dead code)
        if target_node is not None and target_parent is not None:
            removed = _remove_node_from_parent(target_parent, target_node)
            if removed:
                logger.debug("Removed artifactized node from structure tree for %s", op_id)

    return results


def _find_node_by_objgen(pdf: pikepdf.Pdf, struct_elem_id: str) -> pikepdf.Dictionary | None:
    """Resolve a ``struct_elem_id`` like ``obj_42_0`` to the matching indirect object."""
    parts = struct_elem_id.split("_")
    if len(parts) != 3 or parts[0] != "obj":
        return None
    try:
        obj_num, gen_num = int(parts[1]), int(parts[2])
    except (ValueError, TypeError):
        return None

    from project_remedy.pdf_checker import walk_structure_tree

    for node, _depth, _parent in walk_structure_tree(pdf):
        try:
            if hasattr(node, "objgen") and node.objgen == (obj_num, gen_num):
                return node
        except Exception:
            pass
    return None


def _do_set_tag(
    pdf: pikepdf.Pdf,
    page_idx: int,
    mcids: list[int],
    operation: dict,
    *,
    struct_elem_id: str | None = None,
    op_id: str = "",
) -> list[OpResult]:
    """Change the structure tag on existing tagged content.

    If *struct_elem_id* is provided (e.g. from the anchor graph), the node is
    resolved directly by its indirect-object number.  Otherwise falls back to
    the original MCID-intersection scan.
    """
    new_tag = operation.get("tag", "P")

    if page_idx is None or page_idx < 0 or page_idx >= len(pdf.pages):
        return [OpResult(status="error", action="set_tag", op_id=op_id,
                         detail=f"invalid page {page_idx}")]

    # --- fast path: resolve by struct_elem_id --------------------------------
    if struct_elem_id:
        node = _find_node_by_objgen(pdf, struct_elem_id)
        if node is not None:
            old_type = _get_struct_type(node)
            if old_type == new_tag:
                detail = f"set_tag is a no-op ({old_type} -> {new_tag}), skipping"
                logger.info("set_tag skipped for %s: %s", op_id, detail)
                return [OpResult(status="skipped", action="set_tag", op_id=op_id, detail=detail)]
            node["/S"] = pikepdf.Name(f"/{new_tag}")
            detail = (
                f"changed {old_type} -> {new_tag} via struct_elem_id "
                f"{struct_elem_id} on page {page_idx}"
            )
            logger.info("set_tag applied for %s: %s", op_id, detail)
            return [OpResult(status="applied", action="set_tag", op_id=op_id, detail=detail)]
        # If direct lookup failed, fall through to MCID scan
        logger.debug("set_tag: struct_elem_id %s not found, falling back to MCID scan", struct_elem_id)

    # --- fallback: MCID intersection scan ------------------------------------
    from project_remedy.pdf_checker import walk_structure_tree

    for node, _depth, _parent in walk_structure_tree(pdf):
        kids = node.get("/K")
        if kids is None:
            continue
        items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]

        node_mcids = set()
        for item in items:
            if isinstance(item, (int, pikepdf.Object)) and not isinstance(
                item, pikepdf.Dictionary
            ):
                try:
                    node_mcids.add(int(item))
                except (TypeError, ValueError):
                    pass
            elif isinstance(item, pikepdf.Dictionary):
                m = item.get("/MCID")
                if m is not None:
                    try:
                        node_mcids.add(int(m))
                    except (TypeError, ValueError):
                        pass

        if node_mcids and node_mcids.intersection(mcids):
            old_type = _get_struct_type(node)
            node["/S"] = pikepdf.Name(f"/{new_tag}")
            detail = (
                f"changed {old_type} -> {new_tag} for MCIDs "
                f"{sorted(node_mcids & set(mcids))} on page {page_idx}"
            )
            logger.info("set_tag applied for %s: %s", op_id, detail)
            return [OpResult(status="applied", action="set_tag", op_id=op_id, detail=detail)]

    detail = f"no matching element for MCIDs {mcids} on page {page_idx}"
    logger.warning("set_tag skipped for %s: %s", op_id, detail)
    return [OpResult(status="skipped", action="set_tag", op_id=op_id, detail=detail)]


def _do_set_alt_text(
    pdf: pikepdf.Pdf,
    page_idx: int,
    mcids: list[int],
    operation: dict,
    *,
    struct_elem_id: str | None = None,
    op_id: str = "",
) -> list[OpResult]:
    """Set /Alt text on a structure element, promoting it to Figure if needed.

    If *struct_elem_id* is provided the node is resolved directly; otherwise
    falls back to MCID-intersection scanning.  Unlike the previous version,
    this no longer requires the target node to *already* be a Figure -- if the
    planner says "set alt text", the element is first re-tagged as Figure.
    """
    alt_text = operation.get("alt_text", operation.get("reason", ""))
    if not alt_text:
        return [OpResult(status="error", action="set_alt_text", op_id=op_id,
                         detail="no alt_text provided")]

    from project_remedy.pdf_checker import walk_structure_tree

    def _apply_alt(node: pikepdf.Dictionary) -> OpResult:
        old_type = _get_struct_type(node)
        retagged = ""
        if old_type != "Figure":
            node["/S"] = pikepdf.Name("/Figure")
            retagged = f" (retagged {old_type} -> Figure)"
            logger.info("set_alt_text: retagged %s -> Figure for %s", old_type, op_id)
        node["/Alt"] = pikepdf.String(alt_text[:125])
        detail = f"set alt text on page {page_idx}: '{alt_text[:50]}...'{retagged}"
        return OpResult(status="applied", action="set_alt_text", op_id=op_id, detail=detail)

    # --- fast path: resolve by struct_elem_id --------------------------------
    if struct_elem_id:
        node = _find_node_by_objgen(pdf, struct_elem_id)
        if node is not None:
            result = _apply_alt(node)
            logger.info("set_alt_text applied for %s via struct_elem_id", op_id)
            return [result]
        logger.debug("set_alt_text: struct_elem_id %s not found, falling back to MCID scan", struct_elem_id)

    # --- fallback: MCID intersection scan (no longer Figure-only) ------------
    for node, _depth, _parent in walk_structure_tree(pdf):
        kids = node.get("/K")
        if kids is None:
            continue
        items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]

        node_mcids = set()
        for item in items:
            if isinstance(item, (int, pikepdf.Object)) and not isinstance(
                item, pikepdf.Dictionary
            ):
                try:
                    node_mcids.add(int(item))
                except (TypeError, ValueError):
                    pass
            elif isinstance(item, pikepdf.Dictionary):
                m = item.get("/MCID")
                if m is not None:
                    try:
                        node_mcids.add(int(m))
                    except (TypeError, ValueError):
                        pass

        if node_mcids and node_mcids.intersection(mcids):
            result = _apply_alt(node)
            logger.info("set_alt_text applied for %s via MCID scan", op_id)
            return [result]

    detail = f"no matching element for MCIDs {mcids} on page {page_idx}"
    logger.warning("set_alt_text skipped for %s: %s", op_id, detail)
    return [OpResult(status="skipped", action="set_alt_text", op_id=op_id, detail=detail)]


def _do_reconstruct_table(
    pdf: pikepdf.Pdf,
    page_idx: int,
    mcids: list[int],
    operation: dict,
    anchor_graph: dict,
    *,
    op_id: str = "",
) -> list[OpResult]:
    """Rebuild table structure with proper Table/THead/TBody/TR/TH/TD hierarchy.

    Creates a new Table StructElem tree based on the planner's structure spec,
    linking existing MCIDs from the content stream. Replaces any existing Table
    StructElem that contains the target MCIDs.
    """
    structure = operation.get("structure", {})
    if not structure:
        return [OpResult(status="error", action="reconstruct_table", op_id=op_id,
                         detail="no structure spec provided")]

    num_rows = structure.get("rows", 0)
    num_cols = structure.get("cols", 0)
    header_rows = set(structure.get("header_rows", []))
    cells_spec = structure.get("cells", [])

    if num_rows == 0 or num_cols == 0:
        return [OpResult(status="error", action="reconstruct_table", op_id=op_id,
                         detail="rows or cols is 0")]

    expected_cells = num_rows * num_cols
    if len(cells_spec) < expected_cells * 0.5:
        detail = (
            f"incomplete cells_spec (only {len(cells_spec)}/{expected_cells} "
            f"cells specified)"
        )
        logger.warning("reconstruct_table skipped for %s: %s", op_id, detail)
        return [OpResult(status="skipped", action="reconstruct_table", op_id=op_id,
                         detail=detail)]

    if page_idx is None or page_idx < 0 or page_idx >= len(pdf.pages):
        return [OpResult(status="error", action="reconstruct_table", op_id=op_id,
                         detail=f"invalid page {page_idx}")]

    page_obj = pdf.pages[page_idx]

    # Build a map of (row, col) -> cell spec
    cell_map: dict[tuple[int, int], dict] = {}
    for cell in cells_spec:
        cell_map[(cell.get("row", 0), cell.get("col", 0))] = cell

    # Collect all MCIDs from target anchors for this table
    anchors_by_id = {a["anchor_id"]: a for a in anchor_graph.get("anchors", [])}
    target_anchors = operation.get("target_anchors", [])
    table_mcids: list[int] = []
    for aid in target_anchors:
        anchor = anchors_by_id.get(aid)
        if anchor and anchor.get("mcids"):
            table_mcids.extend(anchor["mcids"])

    if not table_mcids and not mcids:
        return [OpResult(status="error", action="reconstruct_table", op_id=op_id,
                         detail="no MCIDs to assign to cells")]

    all_mcids = table_mcids or mcids

    # Find and remove existing Table StructElem containing these MCIDs
    from project_remedy.pdf_checker import walk_structure_tree

    existing_table_parent = None
    existing_table_node = None
    existing_table_index = None
    target_mcid_set = set(all_mcids)

    for node, _depth, parent in walk_structure_tree(pdf):
        if _get_struct_type(node) != "Table":
            continue
        # Check if this table contains any of our target MCIDs
        node_mcids = _collect_descendant_mcids(node)
        if node_mcids & target_mcid_set:
            existing_table_parent = parent
            existing_table_node = node
            # Find index in parent's /K
            if parent is not None:
                parent_kids = parent.get("/K")
                if parent_kids is not None:
                    items = list(parent_kids) if isinstance(parent_kids, pikepdf.Array) else [parent_kids]
                    for idx, kid in enumerate(items):
                        try:
                            if hasattr(kid, "objgen") and hasattr(node, "objgen") and kid.objgen == node.objgen:
                                existing_table_index = idx
                                break
                        except Exception:
                            pass
            break

    # Build new Table structure
    table_node = pdf.make_indirect(pikepdf.Dictionary({
        "/Type": pikepdf.Name("/StructElem"),
        "/S": pikepdf.Name("/Table"),
        "/K": pikepdf.Array(),
    }))

    # Distribute MCIDs across cells (round-robin if we have more MCIDs than cells)
    mcid_queue = list(all_mcids)

    # Build rows
    row_nodes: list[pikepdf.Dictionary] = []
    for r in range(num_rows):
        tr = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/StructElem"),
            "/S": pikepdf.Name("/TR"),
            "/K": pikepdf.Array(),
        }))
        row_nodes.append(tr)

        for c in range(num_cols):
            spec = cell_map.get((r, c), {})
            tag = spec.get("tag", "TH" if r in header_rows else "TD")
            scope = spec.get("scope")

            cell_node = pdf.make_indirect(pikepdf.Dictionary({
                "/Type": pikepdf.Name("/StructElem"),
                "/S": pikepdf.Name(f"/{tag}"),
                "/P": tr,
            }))

            # Set /Scope for TH cells
            if tag == "TH" and scope:
                scope_name = scope.capitalize()
                if scope_name in ("Col", "Column"):
                    scope_name = "Column"
                elif scope_name in ("Row",):
                    scope_name = "Row"
                cell_node["/A"] = pikepdf.Array([
                    pikepdf.Dictionary({
                        "/O": pikepdf.Name("/Table"),
                        "/Scope": pikepdf.Name(f"/{scope_name}"),
                    })
                ])

            # Assign next MCID to this cell (if available)
            if mcid_queue:
                mcid_val = mcid_queue.pop(0)
                mcr = pikepdf.Dictionary({
                    "/Type": pikepdf.Name("/MCR"),
                    "/Pg": page_obj,
                    "/MCID": mcid_val,
                })
                cell_node["/K"] = mcr
            else:
                cell_node["/K"] = pikepdf.Array()

            tr["/K"] = pikepdf.Array(
                list(tr["/K"]) + [cell_node]
            ) if "/K" in tr and isinstance(tr["/K"], pikepdf.Array) else pikepdf.Array([cell_node])

    # Group rows into THead and TBody
    thead_rows = [row_nodes[r] for r in sorted(header_rows) if r < len(row_nodes)]
    tbody_rows = [row_nodes[r] for r in range(num_rows) if r not in header_rows and r < len(row_nodes)]

    table_kids: list[pikepdf.Object] = []

    if thead_rows:
        thead = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/StructElem"),
            "/S": pikepdf.Name("/THead"),
            "/P": table_node,
            "/K": pikepdf.Array(thead_rows),
        }))
        for tr in thead_rows:
            tr["/P"] = thead
        table_kids.append(thead)

    if tbody_rows:
        tbody = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/StructElem"),
            "/S": pikepdf.Name("/TBody"),
            "/P": table_node,
            "/K": pikepdf.Array(tbody_rows),
        }))
        for tr in tbody_rows:
            tr["/P"] = tbody
        table_kids.append(tbody)

    table_node["/K"] = pikepdf.Array(table_kids)

    # Insert new table into the structure tree
    results: list[OpResult] = []
    if existing_table_parent is not None and existing_table_index is not None:
        # Replace existing table
        table_node["/P"] = existing_table_parent
        parent_kids = existing_table_parent.get("/K")
        items = list(parent_kids) if isinstance(parent_kids, pikepdf.Array) else [parent_kids]
        items[existing_table_index] = table_node
        existing_table_parent["/K"] = pikepdf.Array(items) if len(items) > 1 else items[0]
        detail = (
            f"replaced existing table on page {page_idx} "
            f"({num_rows}x{num_cols}, {len(thead_rows)} header rows)"
        )
        results.append(OpResult(status="applied", action="reconstruct_table",
                                op_id=op_id, detail=detail))
    else:
        # No existing table found -- attach to StructTreeRoot or page's parent
        struct_root = pdf.Root.get("/StructTreeRoot")
        if struct_root is not None:
            table_node["/P"] = struct_root
            root_kids = struct_root.get("/K")
            if root_kids is None:
                struct_root["/K"] = table_node
            elif isinstance(root_kids, pikepdf.Array):
                root_kids.append(table_node)
            else:
                struct_root["/K"] = pikepdf.Array([root_kids, table_node])
            detail = (
                f"created new table on page {page_idx} "
                f"({num_rows}x{num_cols}, {len(thead_rows)} header rows)"
            )
            results.append(OpResult(status="applied", action="reconstruct_table",
                                    op_id=op_id, detail=detail))

    return results


def _do_fix_reading_order(
    pdf: pikepdf.Pdf,
    page_idx: int,
    mcids: list[int],
    operation: dict,
    anchor_graph: dict,
    *,
    op_id: str = "",
) -> list[OpResult]:
    """Reorder structure elements to match the desired reading order.

    The target_anchors list specifies the desired order. We map each anchor
    to its StructElem, find their common parent, and reorder the parent's /K array.
    """
    target_anchors = operation.get("target_anchors", [])
    if not target_anchors:
        return [OpResult(status="error", action="fix_reading_order", op_id=op_id,
                         detail="no target_anchors provided")]

    if page_idx is None or page_idx < 0 or page_idx >= len(pdf.pages):
        return [OpResult(status="error", action="fix_reading_order", op_id=op_id,
                         detail=f"invalid page {page_idx}")]

    # Map anchor IDs to struct_elem_ids
    anchors_by_id = {a["anchor_id"]: a for a in anchor_graph.get("anchors", [])}
    desired_struct_ids: list[str | None] = []
    for aid in target_anchors:
        anchor = anchors_by_id.get(aid)
        if anchor:
            desired_struct_ids.append(anchor.get("struct_elem_id"))
        else:
            desired_struct_ids.append(None)

    # Find StructElems by their objgen-based IDs
    from project_remedy.pdf_checker import walk_structure_tree

    struct_id_to_node: dict[str, pikepdf.Dictionary] = {}
    node_to_parent: dict[str, pikepdf.Dictionary] = {}

    for node, _depth, parent in walk_structure_tree(pdf):
        try:
            if hasattr(node, "objgen"):
                sid = f"obj_{node.objgen[0]}_{node.objgen[1]}"
                struct_id_to_node[sid] = node
                if parent is not None:
                    node_to_parent[sid] = parent
        except Exception:
            pass

    # Find nodes for each desired anchor
    ordered_nodes: list[pikepdf.Dictionary] = []
    for sid in desired_struct_ids:
        if sid and sid in struct_id_to_node:
            ordered_nodes.append(struct_id_to_node[sid])

    if len(ordered_nodes) < 2:
        return [OpResult(status="skipped", action="fix_reading_order", op_id=op_id,
                         detail="fewer than 2 elements resolved, nothing to reorder")]

    # Find common parent -- use the parent of the first node
    first_sid = desired_struct_ids[0]
    if first_sid is None or first_sid not in node_to_parent:
        return [OpResult(status="error", action="fix_reading_order", op_id=op_id,
                         detail="cannot find parent of first element")]

    parent = node_to_parent[first_sid]
    parent_kids = parent.get("/K")
    if parent_kids is None:
        return [OpResult(status="error", action="fix_reading_order", op_id=op_id,
                         detail="parent has no /K array")]

    current_items = list(parent_kids) if isinstance(parent_kids, pikepdf.Array) else [parent_kids]

    # Build set of objgens we're reordering
    reorder_objgens = set()
    for node in ordered_nodes:
        try:
            if hasattr(node, "objgen"):
                reorder_objgens.add(node.objgen)
        except Exception:
            pass

    # Split items into: items we're reordering vs items we're not touching
    other_items: list[pikepdf.Object] = []
    reorder_positions: list[int] = []

    for idx, item in enumerate(current_items):
        try:
            if hasattr(item, "objgen") and item.objgen in reorder_objgens:
                reorder_positions.append(idx)
            else:
                other_items.append((idx, item))
        except Exception:
            other_items.append((idx, item))

    if len(reorder_positions) < 2:
        return [OpResult(status="skipped", action="fix_reading_order", op_id=op_id,
                         detail="fewer than 2 matching elements in parent /K")]

    # Rebuild /K array: place reordered nodes at the positions where
    # reorderable nodes originally were, keep everything else in place
    new_items = list(current_items)  # copy
    for new_pos_idx, orig_pos in enumerate(sorted(reorder_positions)):
        if new_pos_idx < len(ordered_nodes):
            new_items[orig_pos] = ordered_nodes[new_pos_idx]

    parent["/K"] = pikepdf.Array(new_items) if len(new_items) > 1 else new_items[0]

    detail = f"reordered {len(ordered_nodes)} elements on page {page_idx}"
    logger.info("fix_reading_order applied for %s: %s", op_id, detail)
    return [OpResult(status="applied", action="fix_reading_order", op_id=op_id,
                     detail=detail)]


def _collect_descendant_mcids(node: pikepdf.Dictionary) -> set[int]:
    """Collect all MCIDs from a node and its descendants."""
    mcids: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        kids = current.get("/K")
        if kids is None:
            continue
        items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
        for item in items:
            if isinstance(item, (int, pikepdf.Object)) and not isinstance(item, pikepdf.Dictionary):
                try:
                    mcids.add(int(item))
                except (TypeError, ValueError):
                    pass
            elif isinstance(item, pikepdf.Dictionary):
                mcid = item.get("/MCID")
                if mcid is not None:
                    try:
                        mcids.add(int(mcid))
                    except (TypeError, ValueError):
                        pass
                if "/S" in item:
                    stack.append(item)
    return mcids


def execute_plan(
    pdf_path: Path,
    output_path: Path,
    plan: dict,
    anchor_graph: dict,
    *,
    allow_artifactize: bool = False,
) -> dict:
    """Execute remediation plan on a PDF.

    Parameters
    ----------
    allow_artifactize:
        When ``False`` (default), artifactize operations are skipped because
        the current implementation rewrites content streams without properly
        cleaning the structure tree, which causes ``post_repair()`` to create
        new 7.1-3 violations.

    Returns
    -------
    dict with keys ``"applied"``, ``"skipped"``, ``"errors"`` -- each a list
    of :class:`OpResult` dicts (serialised via ``dataclasses.asdict``).
    """
    from dataclasses import asdict

    all_results: list[OpResult] = []

    operations = plan.get("operations", [])
    if not operations:
        all_results.append(OpResult(
            status="skipped", action="plan", op_id="plan",
            detail="no operations in plan"))
        import shutil
        shutil.copy2(str(pdf_path), str(output_path))
        return _partition_results(all_results)

    try:
        pdf = pikepdf.open(str(pdf_path))
    except Exception as e:
        all_results.append(OpResult(
            status="error", action="open", op_id="open",
            detail=f"failed to open PDF: {e}"))
        import shutil
        shutil.copy2(str(pdf_path), str(output_path))
        return _partition_results(all_results)

    for op in operations:
        action = op.get("action", "")
        op_id = op.get("op_id", "?")
        page_idx, mcids, struct_elem_id = _resolve_anchor_mcids(op, anchor_graph)

        try:
            if action == "artifactize":
                results = _do_artifactize(
                    pdf, page_idx, mcids, op.get("reason", ""),
                    allow_artifactize=allow_artifactize, op_id=op_id,
                )
                if not results:
                    results = [OpResult(status="skipped", action="artifactize",
                                        op_id=op_id,
                                        detail=f"no matching content on page {page_idx}")]
                all_results.extend(results)

            elif action == "set_tag":
                results = _do_set_tag(
                    pdf, page_idx, mcids, op,
                    struct_elem_id=struct_elem_id, op_id=op_id,
                )
                all_results.extend(results)

            elif action == "set_alt_text":
                results = _do_set_alt_text(
                    pdf, page_idx, mcids, op,
                    struct_elem_id=struct_elem_id, op_id=op_id,
                )
                all_results.extend(results)

            elif action == "mark_manual_review":
                all_results.append(OpResult(
                    status="skipped", action="mark_manual_review", op_id=op_id,
                    detail=op.get("reason", "flagged for review")))

            elif action == "reconstruct_table":
                results = _do_reconstruct_table(
                    pdf, page_idx, mcids, op, anchor_graph, op_id=op_id,
                )
                if not results:
                    results = [OpResult(status="skipped", action="reconstruct_table",
                                        op_id=op_id,
                                        detail=f"no changes on page {page_idx}")]
                all_results.extend(results)

            elif action == "fix_reading_order":
                results = _do_fix_reading_order(
                    pdf, page_idx, mcids, op, anchor_graph, op_id=op_id,
                )
                all_results.extend(results)

            else:
                all_results.append(OpResult(
                    status="skipped", action=action, op_id=op_id,
                    detail=f"unknown action '{action}'"))

        except Exception as e:
            logger.exception("Unhandled error executing %s for %s", action, op_id)
            all_results.append(OpResult(
                status="error", action=action, op_id=op_id,
                detail=f"{action} failed: {e}"))

    # XY-Cut++ fallback: apply deterministic reading order on complex pages
    # when the plan did not include any reading-order operations.
    # Guard: only run when executing a full plan (>= 2 ops), not in greedy
    # single-op evaluation mode where it would be misattributed.
    plan_actions = {op.get("action") for op in operations}
    if "fix_reading_order" not in plan_actions and len(operations) >= 2:
        try:
            from project_remedy.pdf_fixer import (
                _apply_xy_cut_reading_order,
                _analyze_page_layout,
                _build_page_structure_summary,
            )
            structure_summary = _build_page_structure_summary(pdf)
            analyses = {}
            for pidx in range(len(pdf.pages)):
                analyses[pidx] = _analyze_page_layout(
                    pdf, pidx, structure_summary=structure_summary,
                )
            xy_pages = _apply_xy_cut_reading_order(pdf, analyses)
            if xy_pages:
                all_results.append(OpResult(
                    status="applied", action="xy_cut_reading_order",
                    op_id="xy_cut_fallback",
                    detail=f"XY-Cut++ reordered {xy_pages} pages (planner omitted reading order)",
                ))
        except Exception as e:
            logger.debug("XY-Cut++ fallback skipped: %s", e)

    # Save the modified PDF
    try:
        from project_remedy.pdf_fixer import _normalize_structure_tree_indirect_objects
        _normalize_structure_tree_indirect_objects(pdf)
        pdf.save(str(output_path), object_stream_mode=pikepdf.ObjectStreamMode.disable)
    except Exception as e:
        all_results.append(OpResult(
            status="error", action="save", op_id="save",
            detail=f"failed to save PDF: {e}"))
        import shutil
        shutil.copy2(str(pdf_path), str(output_path))
    finally:
        pdf.close()

    result = _partition_results(all_results)
    logger.info(
        "execute_plan finished: %d applied, %d skipped, %d errors",
        len(result["applied"]), len(result["skipped"]), len(result["errors"]),
    )
    return result


def _partition_results(results: list[OpResult]) -> dict:
    """Partition a flat list of :class:`OpResult` into applied/skipped/errors.

    Each bucket contains serialised dicts for JSON compatibility while still
    carrying the full structured information.
    """
    from dataclasses import asdict

    applied: list[dict] = []
    skipped: list[dict] = []
    errors: list[dict] = []
    for r in results:
        d = asdict(r)
        if r.status == "applied":
            applied.append(d)
        elif r.status == "skipped":
            skipped.append(d)
        else:
            errors.append(d)
    return {"applied": applied, "skipped": skipped, "errors": errors}


def post_repair(pdf_path: Path) -> list[str]:
    """Run bounded structural repairs using proven pdf_fixer functions.

    These are non-semantic repairs — they fix structural invariants
    (ParentTree, MCIDs, orphan elements, BDC/EMC balance) without
    changing what content IS, only ensuring the structure tree is valid.
    """
    from project_remedy.pdf_fixer import (
        fix_bdc_emc_balance,
        fix_page_retag,
        fix_pdfua_identifier,
        fix_table_header_scope,
        fix_table_headers,
        fix_table_parent_structure,
        _tag_unmarked_content_streams,
        _normalize_structure_tree_indirect_objects,
    )

    changes: list[str] = []

    try:
        pdf = pikepdf.open(str(pdf_path), allow_overwriting_input=True)
    except Exception as e:
        return [f"post_repair: failed to open PDF: {e}"]

    try:
        # 1. Reconcile MCIDs, ParentTree, remove orphan nodes
        changes.extend(fix_page_retag(pdf))

        # 2. Wrap remaining unmarked text in BDC/EMC
        tagged = _tag_unmarked_content_streams(pdf)
        if tagged:
            changes.append(f"Tagged {tagged} pages with unmarked content streams")

        # 3. Fix orphan TR/TH/TD, promote header rows to THead
        changes.extend(fix_table_parent_structure(pdf))

        # 4. Promote first-row TD to TH if table lacks headers
        changes.extend(fix_table_headers(pdf))

        # 5. Set /Scope on TH cells
        changes.extend(fix_table_header_scope(pdf))

        # 6. Fix BDC/EMC pairing
        changes.extend(fix_bdc_emc_balance(pdf))

        # 7. Set PDF/UA-1 identifier (rule 5-1)
        changes.extend(fix_pdfua_identifier(pdf))

        # Normalize and save
        _normalize_structure_tree_indirect_objects(pdf)
        pdf.save(str(pdf_path))
    except Exception as e:
        changes.append(f"post_repair error: {e}")
    finally:
        pdf.close()

    return changes