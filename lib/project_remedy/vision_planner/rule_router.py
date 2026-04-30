"""Deterministic rule router for common veraPDF violations.

Fixes rules that have known deterministic solutions via pikepdf,
without needing the vision grounder or AI planner. Only violations
that require semantic understanding (alt text, reading order, content
classification) are passed through to the AI planner.

Architecture: deterministic-first, LLM-second.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import pikepdf

logger = logging.getLogger(__name__)


@dataclass
class _ContentOp:
    operands: list
    operator: pikepdf.Operator


@dataclass
class _MarkedContentNode:
    operands: list
    operator: pikepdf.Operator
    tag: str
    children: list["_ContentNode"]


@dataclass
class _FormXObjectInfo:
    has_marked_content: bool = False
    has_text: bool = False
    has_image: bool = False
    has_painting: bool = False


_ContentNode = _ContentOp | _MarkedContentNode

_TEXT_SHOWING_OPS = frozenset({"Tj", "TJ", "'", '"'})
_VISIBLE_CONTENT_OPS = frozenset(
    {
        *tuple(_TEXT_SHOWING_OPS),
        "Do", "EI",
        "S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n", "sh",
    }
)
_PAINTING_OPS = frozenset({"S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n", "sh"})
_SAFE_GRAPHICS_SCAFFOLDING_OPS = frozenset(
    {
        "q", "Q", "cm", "w", "J", "j", "M", "d", "ri", "i", "gs",
        "CS", "cs", "SC", "SCN", "sc", "scn", "G", "g", "RG", "rg", "K", "k",
        "m", "l", "c", "v", "y", "h", "re", "W", "W*", "BT", "ET", "Tf", "Td",
        "TD", "Tm", "T*", "Tc", "Tw", "Tz", "TL", "Tr", "Ts",
    }
)


def _resolve_pdf_object(obj):
    """Best-effort resolve of an indirect pikepdf object."""
    try:
        return obj.get_object()
    except Exception:
        try:
            return obj.resolve()
        except Exception:
            return obj


def _named_resource_lookup(mapping, name: str):
    """Resolve ``/Name`` or ``Name`` keys from a PDF resource dictionary."""
    if mapping is None:
        return None
    try:
        return mapping.get(f"/{name}") or mapping.get(name)
    except Exception:
        return mapping.get(name)


def _build_marked_content_tree(instructions: list[tuple]) -> list[_ContentNode]:
    """Convert a flat content stream into nested marked-content nodes."""
    root: list[_ContentNode] = []
    stack: list[list[_ContentNode]] = [root]

    for operands, operator in instructions:
        op = str(operator)
        if op in {"BDC", "BMC"}:
            tag = str(operands[0]).lstrip("/") if operands else ""
            node = _MarkedContentNode(list(operands), operator, tag, [])
            stack[-1].append(node)
            stack.append(node.children)
            continue
        if op == "EMC":
            if len(stack) > 1:
                stack.pop()
            else:
                stack[-1].append(_ContentOp(list(operands), operator))
            continue
        stack[-1].append(_ContentOp(list(operands), operator))

    return root


def _flatten_marked_content_tree(nodes: list[_ContentNode]) -> list[tuple]:
    """Flatten nested marked-content nodes back to content-stream tuples."""
    flattened: list[tuple] = []
    for node in nodes:
        if isinstance(node, _ContentOp):
            flattened.append((node.operands, node.operator))
            continue
        flattened.append((node.operands, node.operator))
        flattened.extend(_flatten_marked_content_tree(node.children))
        flattened.append(([], pikepdf.Operator("EMC")))
    return flattened


def _block_mcid(node: _MarkedContentNode) -> int | None:
    """Return the MCID for a ``BDC`` block, if present."""
    if str(node.operator) != "BDC" or len(node.operands) < 2:
        return None
    props = node.operands[1]
    props = _resolve_pdf_object(props)
    if not isinstance(props, pikepdf.Dictionary):
        return None
    try:
        mcid = props.get("/MCID")
        return None if mcid is None else int(mcid)
    except Exception:
        return None


def _clone_marked_content_node(
    node: _MarkedContentNode,
    children: list[_ContentNode],
    *,
    mcid: int | None = None,
) -> _MarkedContentNode:
    """Clone a marked-content node, optionally replacing its MCID."""
    operands = list(node.operands)
    if mcid is not None and str(node.operator) == "BDC" and len(operands) >= 2:
        props = _resolve_pdf_object(operands[1])
        if isinstance(props, pikepdf.Dictionary):
            new_props = pikepdf.Dictionary(props)
            new_props["/MCID"] = mcid
            operands[1] = new_props
    return _MarkedContentNode(operands, node.operator, node.tag, children)


def _nodes_have_visible_content(nodes: list[_ContentNode]) -> bool:
    """True when a segment contains rendered content, not just state changes."""
    for node in nodes:
        if isinstance(node, _MarkedContentNode):
            if _nodes_have_visible_content(node.children):
                return True
            continue
        if str(node.operator) in _VISIBLE_CONTENT_OPS:
            return True
    return False


def _is_safe_graphics_scaffolding(node: _ContentNode) -> bool:
    """True when *node* only adjusts state around nearby rendered content."""
    return isinstance(node, _ContentOp) and str(node.operator) in _SAFE_GRAPHICS_SCAFFOLDING_OPS


def _rewrite_nested_marked_content(nodes: list[_ContentNode]) -> tuple[list[_ContentNode], int]:
    """Strip illegal artifact/tagged wrappers while preserving inner content."""
    rewritten: list[_ContentNode] = []
    changed = 0

    for node in nodes:
        if isinstance(node, _ContentOp):
            rewritten.append(node)
            continue

        children, child_changes = _rewrite_nested_marked_content(node.children)
        changed += child_changes

        normalized_children: list[_ContentNode] = []
        for child in children:
            if isinstance(child, _MarkedContentNode):
                if node.tag == "Artifact" and child.tag != "Artifact":
                    normalized_children.extend(child.children)
                    changed += 1
                    continue
                if node.tag != "Artifact" and child.tag == "Artifact":
                    normalized_children.extend(child.children)
                    changed += 1
                    continue
            normalized_children.append(child)

        node.children = normalized_children
        rewritten.append(node)

    return rewritten, changed


def _analyze_form_xobject(
    xobj: pikepdf.Stream,
    cache: dict[object, _FormXObjectInfo],
    seen: set[object] | None = None,
) -> _FormXObjectInfo:
    """Classify a Form XObject as decorative vector content vs. real content."""
    key = getattr(xobj, "objgen", None)
    if key in (None, (0, 0)):
        key = ("direct", id(xobj))
    if key in cache:
        return cache[key]

    if seen is None:
        seen = set()
    if key in seen:
        # Treat recursion as unsafe for artifactization.
        return _FormXObjectInfo(has_image=True)
    seen.add(key)

    info = _FormXObjectInfo()
    try:
        instructions = list(pikepdf.parse_content_stream(xobj))
    except Exception:
        cache[key] = info
        seen.remove(key)
        return info

    resources = _resolve_pdf_object(xobj.get("/Resources"))
    xobjects = resources.get("/XObject") if isinstance(resources, pikepdf.Dictionary) else None

    for operands, operator in instructions:
        op = str(operator)
        if op in {"BDC", "BMC"}:
            info.has_marked_content = True
            continue
        if op in _TEXT_SHOWING_OPS:
            info.has_text = True
            continue
        if op in _PAINTING_OPS:
            info.has_painting = True
            continue
        if op != "Do" or not operands:
            continue

        xobj_name = str(operands[0]).lstrip("/")
        child_ref = _named_resource_lookup(xobjects, xobj_name)
        if child_ref is None:
            info.has_image = True
            continue
        child = _resolve_pdf_object(child_ref)
        if not isinstance(child, pikepdf.Stream):
            info.has_image = True
            continue

        subtype = str(child.get("/Subtype", ""))
        if subtype == "/Image":
            info.has_image = True
        elif subtype == "/Form":
            child_info = _analyze_form_xobject(child, cache, seen)
            info.has_marked_content |= child_info.has_marked_content
            info.has_text |= child_info.has_text
            info.has_image |= child_info.has_image
            info.has_painting |= child_info.has_painting
        else:
            info.has_image = True

    cache[key] = info
    seen.remove(key)
    return info


def _is_decorative_form_do(
    node: _ContentNode,
    xobjects,
    cache: dict[object, _FormXObjectInfo],
) -> bool:
    """True when a ``Do`` invocation targets a vector-only decorative Form XObject."""
    if not isinstance(node, _ContentOp) or str(node.operator) != "Do" or not node.operands:
        return False

    xobj_name = str(node.operands[0]).lstrip("/")
    xobj_ref = _named_resource_lookup(xobjects, xobj_name)
    if xobj_ref is None:
        return False
    xobj = _resolve_pdf_object(xobj_ref)
    if not isinstance(xobj, pikepdf.Stream) or str(xobj.get("/Subtype", "")) != "/Form":
        return False

    info = _analyze_form_xobject(xobj, cache)
    return (
        not info.has_marked_content
        and not info.has_text
        and not info.has_image
        and info.has_painting
    )


def _rewrite_decorative_form_invocations(
    nodes: list[_ContentNode],
    *,
    xobjects,
    form_cache: dict[object, _FormXObjectInfo],
    next_mcid: int,
) -> tuple[list[_ContentNode], int, int]:
    """Split tagged MCID blocks around decorative Form XObject invocations."""
    rewritten: list[_ContentNode] = []
    changes = 0

    for node in nodes:
        if isinstance(node, _ContentOp):
            rewritten.append(node)
            continue

        children, child_changes, next_mcid = _rewrite_decorative_form_invocations(
            node.children,
            xobjects=xobjects,
            form_cache=form_cache,
            next_mcid=next_mcid,
        )
        node.children = children
        changes += child_changes

        if node.tag == "Artifact":
            rewritten.append(node)
            continue

        block_mcid = _block_mcid(node)
        emitted: list[_ContentNode] = []
        current_segment: list[_ContentNode] = []
        idx = 0
        used_original_mcid = False
        block_changed = 0

        def emit_segment(segment: list[_ContentNode]) -> None:
            nonlocal used_original_mcid, next_mcid
            if not segment:
                return
            mcid = None
            if block_mcid is not None:
                if not used_original_mcid:
                    mcid = block_mcid
                    used_original_mcid = True
                else:
                    mcid = next_mcid
                    next_mcid += 1
            emitted.append(_clone_marked_content_node(node, list(segment), mcid=mcid))

        while idx < len(children):
            child = children[idx]
            if not _is_decorative_form_do(child, xobjects, form_cache):
                current_segment.append(child)
                idx += 1
                continue

            leading_artifact: list[_ContentNode] = []
            while current_segment and _is_safe_graphics_scaffolding(current_segment[-1]):
                leading_artifact.insert(0, current_segment.pop())

            if current_segment:
                emit_segment(current_segment)
                current_segment = []

            artifact_chunk = leading_artifact + [child]
            idx += 1
            while idx < len(children) and _is_safe_graphics_scaffolding(children[idx]):
                artifact_chunk.append(children[idx])
                idx += 1

            emitted.append(
                _MarkedContentNode(
                    [pikepdf.Name("/Artifact")],
                    pikepdf.Operator("BMC"),
                    "Artifact",
                    artifact_chunk,
                )
            )
            block_changed += 1

        if current_segment:
            emit_segment(current_segment)

        if block_changed:
            rewritten.extend(emitted)
            changes += block_changed
        else:
            rewritten.append(node)

    return rewritten, changes, next_mcid

# ---------------------------------------------------------------------------
# Rule classification
# ---------------------------------------------------------------------------

# Rules that have deterministic fixes -- no AI needed.
DETERMINISTIC_RULES: frozenset[str] = frozenset(
    {
        "5-1",  # PDF/UA identifier
        "7.1-1",  # Artifact inside tagged content (content stream nesting)
        "7.1-2",  # Tagged content inside Artifact (content stream nesting)
        "7.1-3",  # Untagged content (mark as Artifact or tag)
        "7.1-10",  # ViewerPreferences DisplayDocTitle
        "7.2-1",  # Heading nesting
        "7.2-2",  # Natural language in Outline entries
        "7.2-3",  # Table may contain only TR/THead/TBody/TFoot/Caption
        "7.2-5",  # THead/TBody/TFoot must be in Table
        "7.2-6",  # TBody must be in Table
        "7.2-10",  # TR may contain only TH and TD
        "7.2-14",  # Table with THead must have TBody
        "7.2-17",  # LI must be in L element
        "7.2-19",  # L may contain only L/LI/Caption
        "7.2-20",  # LI may contain only Lbl and LBody
        "7.2-21",  # Natural language in ActualText
        "7.2-34",  # Natural language for text in page content
        "7.2-42",  # Table header cell scope (best-effort)
        "7.2-43",  # Table rows same column count (best-effort)
        "7.3-1",  # Table row structure
        "7.4-1",  # Table header structure
        "7.4.2-1",  # Header scope
        "7.5-1",  # Table headers and IDs (best-effort)
        "7.10-1",  # Optional content configuration
        "7.18.1-1",  # Annotation lacking /Contents or /Alt
        "7.18.1-2",  # Non-widget annotation lacking alternate description
        "7.18.5-1",  # Link tagging
        "7.18.5-2",  # Link alternate description
        "7.21.4.1-1",  # Font programs not embedded (best-effort; source-font limitation)
        "7.21.4.1-2",  # Embedded font missing glyphs (best-effort; source-font limitation)
        "7.21.4.2-2",  # CIDSet incomplete (best-effort; source-font limitation)
        "7.21.5-1",  # Font glyph width mismatch (best-effort; source-font limitation)
        "7.21.6-2",  # TrueType non-symbolic encoding (best-effort; source-font limitation)
        "7.21.6-3",  # Symbolic TrueType encoding (best-effort; source-font limitation)
        "7.21.7-1",  # Font ToUnicode mapping (best-effort; source-font limitation)
        "7.21.7-2",  # Font ToUnicode mapping variant (best-effort; source-font limitation)
        "7.21.8-1",  # .notdef glyph reference (best-effort; source-font limitation)
    }
)

# Rules that need AI (semantic understanding required).
# Listed for documentation; anything not in DETERMINISTIC_RULES is implicitly
# routed to the AI planner.
AI_REQUIRED_RULES: frozenset[str] = frozenset(
    {
        "7.1-11",  # Reading order (needs vision to determine correct order)
        "7.1-8",  # Figure alt text (needs vision to describe image)
    }
)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _normalize_rule_id(raw: str) -> str:
    """Normalize a veraPDF rule ID to its short form.

    Handles:
    - Path-style: ``clause-7.1/7.1-3`` -> ``7.1-3``
    - ISO-style:  ``ISO 14289-1:2014-7.1-3`` -> ``7.1-3``
    """
    import re

    # Path-style prefix
    if "/" in raw:
        raw = raw.split("/")[-1]
    # ISO prefix: "ISO 14289-1:2014-X.Y-Z" -> extract the clause part
    iso_match = re.match(r"ISO\s+\d+-\d+:\d+-(.+)", raw)
    if iso_match:
        return iso_match.group(1)
    return raw


def _rule_matches(rule: str, prefix: str) -> bool:
    """True when *rule* is exactly *prefix* or a dot-separated sub-clause.

    ``_rule_matches("7.1-1", "7.1-1")``   -> True
    ``_rule_matches("7.1-1.2", "7.1-1")`` -> True
    ``_rule_matches("7.1-11", "7.1-1")``  -> **False**
    """
    if rule == prefix:
        return True
    return rule.startswith(prefix) and len(rule) > len(prefix) and rule[len(prefix)] == "."


def _any_rule_matches(rules: set[str], *prefixes: str) -> bool:
    """True if any rule in *rules* matches any of the given *prefixes*."""
    return any(_rule_matches(r, p) for r in rules for p in prefixes)


def _is_deterministic(rule_clean: str) -> bool:
    """Return True if *rule_clean* matches a deterministic rule.

    Matching is exact or sub-clause: ``7.1-1`` matches ``7.1-1`` but
    **not** ``7.1-11``.  A trailing dot (``7.1-1.2``) is accepted as a
    sub-clause of ``7.1-1``.
    """
    return any(_rule_matches(rule_clean, dr) for dr in DETERMINISTIC_RULES)


def route_violations(
    violations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split violations into deterministic-fixable and AI-required.

    Each violation dict is expected to carry a ``"rule"`` or ``"ruleId"`` key
    whose value is the veraPDF rule identifier (possibly path-prefixed).

    Returns:
        ``(deterministic_violations, ai_violations)``
    """
    deterministic: list[dict[str, Any]] = []
    ai_needed: list[dict[str, Any]] = []

    for v in violations:
        rule = v.get("id", v.get("rule_id", v.get("rule", v.get("ruleId", ""))))
        rule_clean = _normalize_rule_id(rule)

        if _is_deterministic(rule_clean):
            deterministic.append(v)
        else:
            ai_needed.append(v)

    logger.info(
        "rule_router: %d deterministic, %d AI-required out of %d total violations",
        len(deterministic),
        len(ai_needed),
        len(violations),
    )
    return deterministic, ai_needed


# ---------------------------------------------------------------------------
# Deterministic fix application
# ---------------------------------------------------------------------------


def apply_deterministic_fixes(
    pdf_path: Path,
    violations: list[dict[str, Any]],
) -> tuple[list[str], int]:
    """Apply deterministic fixes for known rules.

    Opens the PDF at *pdf_path*, applies fixes **in dependency order**, saves
    back to the same path, and returns a list of human-readable change
    descriptions plus the count of violations addressed.

    Returns:
        ``(changes_applied, violations_fixed_count)``
    """
    if not violations:
        return [], 0

    changes: list[str] = []

    try:
        pdf = pikepdf.open(str(pdf_path), allow_overwriting_input=True)
    except Exception as e:
        logger.error("rule_router: failed to open PDF %s: %s", pdf_path, e)
        return [f"deterministic_fix: failed to open PDF: {e}"], 0

    try:
        # Late imports -- keeps this module lightweight when only routing.
        from project_remedy.pdf_fixer import (
            _normalize_structure_tree_indirect_objects,
            _tag_unmarked_content_streams,
            fix_bdc_emc_balance,
            fix_create_structure_tree,
            fix_heading_nesting,
            fix_page_retag,
            fix_pdfua_identifier,
            fix_table_header_scope,
            fix_table_headers,
            fix_table_parent_structure,
            fix_untagged_content,
            fix_unmarked_operators_as_artifacts,
        )

        # Determine which rule families are present.
        rules_present: set[str] = {
            _normalize_rule_id(v.get("id", v.get("rule_id", v.get("rule", v.get("ruleId", "")))))
            for v in violations
        }

        # ---------------------------------------------------------------
        # Apply fixes in dependency order (structure before content).
        # ---------------------------------------------------------------

        # 1. Artifact/tagged nesting (7.1-1: artifact inside tagged,
        #    7.1-2: tagged inside artifact) — content stream cleanup
        if _any_rule_matches(rules_present, "7.1-1", "7.1-2"):
            result = fix_bdc_emc_balance(pdf)
            changes.extend(result)
            result = _fix_artifact_tagged_nesting(pdf)
            changes.extend(result)

        # 2. Untagged content (7.1-3) — mark as Artifact or tag
        if _any_rule_matches(rules_present, "7.1-3"):
            for _fix_fn, _fix_name in [
                (_fix_orphan_mcids, "_fix_orphan_mcids"),
                (fix_untagged_content, "fix_untagged_content"),
                (fix_page_retag, "fix_page_retag"),
                (fix_unmarked_operators_as_artifacts, "fix_unmarked_operators_as_artifacts"),
                (_fix_untagged_xobject_content, "_fix_untagged_xobject_content"),
            ]:
                try:
                    result = _fix_fn(pdf)
                    changes.extend(result)
                except Exception as e:
                    logger.debug("rule_router: %s error: %s", _fix_name, e)

            try:
                tagged = _tag_unmarked_content_streams(pdf)
                if tagged:
                    changes.append(f"rule_router: tagged {tagged} pages with unmarked content")
            except Exception as e:
                logger.debug("rule_router: _tag_unmarked_content_streams error: %s", e)

            # 2b. Re-run nesting fix — orphan MCID repair can introduce
            # new 7.1-1/7.1-2 by wrapping content inside artifact scopes
            try:
                result = _fix_artifact_tagged_nesting(pdf)
                if result:
                    changes.extend(result)
                    changes.append("rule_router: post-pass nesting cleanup after orphan MCID repair")
            except Exception as e:
                logger.debug("rule_router: post-pass nesting cleanup error: %s", e)

        # 3. Heading nesting (7.2-1)
        if _any_rule_matches(rules_present, "7.2-1"):
            result = fix_heading_nesting(pdf)
            changes.extend(result)

        # 4. Table structure (7.3-1, 7.4-1, 7.4.2-1, 7.2-3, 7.2-6, 7.2-10, 7.2-14)
        if _any_rule_matches(
            rules_present,
            "7.3-1", "7.4-1", "7.4.2-1",
            "7.2-3", "7.2-6", "7.2-10", "7.2-14",
        ):
            result = fix_table_parent_structure(pdf)
            changes.extend(result)
            result = fix_table_headers(pdf)
            changes.extend(result)
            result = fix_table_header_scope(pdf)
            changes.extend(result)
            result = _fix_table_structure_rules(pdf)
            changes.extend(result)

        # 4b. Table regularity (7.2-42: header scope, 7.2-43: column count)
        if _any_rule_matches(rules_present, "7.2-42", "7.2-43"):
            result = _fix_table_regularity(pdf)
            changes.extend(result)

        # 4c. List structure (7.2-17: LI in L, 7.2-19: L may only contain L/LI/Caption)
        if _any_rule_matches(rules_present, "7.2-17", "7.2-19", "7.2-20"):
            result = _fix_list_structure(pdf)
            changes.extend(result)
            if _any_rule_matches(rules_present, "7.2-17"):
                result = _fix_li_parent_containment(pdf)
                changes.extend(result)

        # 5. Link tagging (7.18.5-1)
        if _any_rule_matches(rules_present, "7.18.5-1"):
            result = _fix_link_tags(pdf)
            changes.extend(result)

        # 6. Annotation descriptions (7.18.1-1: missing /Contents or /Alt)
        if _any_rule_matches(rules_present, "7.18.1-1"):
            from project_remedy.pdf_fixer import (
                fix_annotation_descriptions,
                fix_annotations_tagged,
            )
            result = fix_annotations_tagged(pdf)
            changes.extend(result)
            result = fix_annotation_descriptions(pdf)
            changes.extend(result)

        # 7. Optional content config (7.10-1)
        if _any_rule_matches(rules_present, "7.10-1"):
            from project_remedy.pdf_fixer import fix_optional_content_config_names
            result = fix_optional_content_config_names(pdf)
            changes.extend(result)

        # 8. Annotation descriptions/content (7.21.4.1-1, 7.21.4.1-2, 7.21.4.2-2)
        if _any_rule_matches(
            rules_present,
            "7.21.4.1-1", "7.21.4.1-2", "7.21.4.2-2",
        ):
            from project_remedy.pdf_fixer import (
                fix_annotation_descriptions,
                fix_annotations_tagged,
                fix_form_field_descriptions,
                fix_form_fields_tagged,
            )
            result = fix_annotations_tagged(pdf)
            changes.extend(result)
            result = fix_form_fields_tagged(pdf)
            changes.extend(result)
            result = fix_form_field_descriptions(pdf)
            changes.extend(result)
            result = fix_annotation_descriptions(pdf)
            changes.extend(result)
            # CIDSet is related to 7.21.4.2-2 and stays here.
            if _any_rule_matches(rules_present, "7.21.4.2-2"):
                result = _fix_cidset(pdf)
                changes.extend(result)

        # 9. Font/encoding (7.21.5-1, 7.21.6-2, 7.21.6-3, 7.21.7-1, 7.21.8-1)
        if _any_rule_matches(
            rules_present,
            "7.21.5-1", "7.21.6-2", "7.21.6-3", "7.21.7-1", "7.21.8-1",
        ):
            from project_remedy.pdf_fixer import fix_tounicode, fix_char_encoding

            # 9a. Fix Ghostscript-corrupted encoding Differences BEFORE
            # running ToUnicode synthesis (removes ~gs~gName~ artifacts)
            result = _fix_gs_corrupted_encoding(pdf)
            changes.extend(result)

            result = fix_tounicode(pdf)
            changes.extend(result)
            result = fix_char_encoding(pdf)
            changes.extend(result)

            # 9b. Layer 4: fitz-based ToUnicode recovery for CID fonts
            # where Layers 1-3 failed (font still lacks ToUnicode)
            if _any_rule_matches(rules_present, "7.21.7-1"):
                # Save current state, apply fitz mapping, verify no regression
                pdf.save(str(pdf_path))  # flush Layers 1-3
                result = _fix_tounicode_fitz_layer4(pdf, pdf_path)
                changes.extend(result)

        # 10. Tab order (7.21.7-1, 7.21.7-2) depends on annotation structure.
        if _any_rule_matches(rules_present, "7.21.7-1", "7.21.7-2"):
            from project_remedy.pdf_fixer import (
                fix_annotations_tagged,
                fix_form_fields_tagged,
                fix_tab_order,
            )
            result = fix_annotations_tagged(pdf)
            changes.extend(result)
            result = fix_form_fields_tagged(pdf)
            changes.extend(result)
            result = fix_tab_order(pdf)
            changes.extend(result)

        # 11. ViewerPreferences (7.1-10)
        if _any_rule_matches(rules_present, "7.1-10"):
            result = _fix_viewer_preferences(pdf)
            changes.extend(result)

        # 12. Natural language (7.2-2, 7.2-21, 7.2-34)
        if _any_rule_matches(rules_present, "7.2-2", "7.2-21", "7.2-34"):
            result = _fix_natural_language(pdf)
            changes.extend(result)

        # 13. Annotation descriptions (7.18.1-2, 7.18.5-2)
        if _any_rule_matches(rules_present, "7.18.1-2", "7.18.5-2"):
            from project_remedy.pdf_fixer import (
                fix_annotation_descriptions,
                fix_annotations_tagged,
            )
            result = fix_annotations_tagged(pdf)
            changes.extend(result)
            result = fix_annotation_descriptions(pdf)
            changes.extend(result)

        # 14. List item structure (7.2-20: LI may only contain Lbl/LBody)
        if _any_rule_matches(rules_present, "7.2-20"):
            result = _fix_list_item_structure(pdf)
            changes.extend(result)

        # 15. Table headers/IDs (7.5-1)
        if _any_rule_matches(rules_present, "7.5-1"):
            result = _fix_table_headers_ids(pdf)
            changes.extend(result)

        # 16. THead/TBody/TFoot containment (7.2-5)
        if _any_rule_matches(rules_present, "7.2-5"):
            result = fix_table_parent_structure(pdf)
            changes.extend(result)

        # 17. BDC/EMC balance (cleanup -- always safe to run after edits)
        result = fix_bdc_emc_balance(pdf)
        changes.extend(result)

        # 18. PDF/UA identifier (5-1) -- must be last so all structure is in place
        if _any_rule_matches(rules_present, "5-1"):
            result = fix_pdfua_identifier(pdf)
            changes.extend(result)

        # ---------------------------------------------------------------
        # Normalize indirect objects and persist.
        # ---------------------------------------------------------------
        _normalize_structure_tree_indirect_objects(pdf)
        pdf.save(str(pdf_path))
        logger.info(
            "rule_router: saved %s with %d changes",
            pdf_path.name,
            len(changes),
        )

    except Exception as e:
        logger.exception("rule_router: error applying deterministic fixes")
        changes.append(f"deterministic_fix error: {e}")
    finally:
        pdf.close()

    fixed_count = sum(1 for c in changes if not c.startswith("deterministic_fix error"))
    return changes, fixed_count


# ---------------------------------------------------------------------------
# Link tag fix (7.18.5-1) -- only new fix function; everything else delegates
# to existing pdf_fixer helpers.
# ---------------------------------------------------------------------------


def _fix_link_tags(pdf: pikepdf.Pdf) -> list[str]:
    """Fix 7.18.5-1: annotations with ``/Subtype /Link`` should live in ``/Link`` StructElems.

    Walks the structure tree looking for OBJR (object reference) entries that
    point at a link annotation.  When the parent StructElem is not already
    typed ``/Link``, its ``/S`` name is rewritten.
    """
    changes: list[str] = []

    try:
        from project_remedy.pdf_checker import walk_structure_tree

        for node, _depth, _parent in walk_structure_tree(pdf):
            kids = node.get("/K")
            if kids is None:
                continue
            items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]

            has_link_annot = False
            for item in items:
                if not isinstance(item, pikepdf.Dictionary):
                    continue
                # OBJR = object reference to an annotation
                if item.get("/Type") != pikepdf.Name("/OBJR"):
                    continue
                obj = item.get("/Obj")
                if obj is None:
                    continue
                try:
                    if obj.get("/Subtype") == pikepdf.Name("/Link"):
                        has_link_annot = True
                        break
                except Exception:
                    pass

            if not has_link_annot:
                continue

            current_type = str(node.get("/S", "")).lstrip("/")
            if current_type == "Link":
                continue

            node["/S"] = pikepdf.Name("/Link")
            msg = f"rule_router: retagged {current_type or '(none)'} -> Link for annotation link"
            logger.debug(msg)
            changes.append(msg)

    except Exception as e:
        logger.exception("rule_router: _fix_link_tags error")
        changes.append(f"rule_router: _fix_link_tags error: {e}")

    return changes


# ---------------------------------------------------------------------------
# Fix orphan / missing MCIDs in page content (7.1-3)
# ---------------------------------------------------------------------------


def _coerce_int(value) -> int | None:
    """Return ``value`` as ``int`` when possible."""
    try:
        return int(value)
    except Exception:
        return None


def _ensure_root_struct_container(
    pdf: pikepdf.Pdf,
    struct_root: pikepdf.Dictionary,
) -> pikepdf.Dictionary:
    """Return the root Document-like container used for new fallback nodes."""
    kids = struct_root.get("/K")
    items = list(kids) if isinstance(kids, pikepdf.Array) else ([kids] if kids is not None else [])

    for item in items:
        resolved = _resolve_pdf_object(item)
        if isinstance(resolved, pikepdf.Dictionary) and str(resolved.get("/S", "")).lstrip("/") == "Document":
            return resolved

    for item in items:
        resolved = _resolve_pdf_object(item)
        if isinstance(resolved, pikepdf.Dictionary) and "/S" in resolved:
            return resolved

    doc_elem = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/StructElem"),
                "/S": pikepdf.Name("/Document"),
                "/P": struct_root,
                "/K": pikepdf.Array(),
            }
        )
    )
    if kids is None:
        struct_root["/K"] = doc_elem
    elif isinstance(kids, pikepdf.Array):
        kids.append(doc_elem)
    else:
        struct_root["/K"] = pikepdf.Array([kids, doc_elem])
    return doc_elem


def _mcid_from_props_operand(props_operand, properties) -> int | None:
    """Resolve an MCID from a BDC property operand or named property reference."""
    if isinstance(props_operand, pikepdf.Name):
        prop_ref = _named_resource_lookup(properties, str(props_operand).lstrip("/"))
        resolved_props = _resolve_pdf_object(prop_ref)
    else:
        resolved_props = _resolve_pdf_object(props_operand)

    if not isinstance(resolved_props, pikepdf.Dictionary):
        return None
    return _coerce_int(resolved_props.get("/MCID"))


def _next_available_page_mcid(
    instructions: list[tuple],
    properties,
) -> int:
    """Return the next unused MCID referenced by page content."""
    highest = -1
    for operands, operator in instructions:
        if str(operator) != "BDC" or len(operands) < 2:
            continue
        mcid = _mcid_from_props_operand(operands[1], properties)
        if mcid is not None:
            highest = max(highest, mcid)
    return highest + 1


def _page_parent_tree_entry(struct_root: pikepdf.Dictionary, page, mcid: int):
    """Return the current page parent-tree entry for ``mcid`` if present."""
    from project_remedy.pdf_fixer import _parent_tree_num_arrays

    struct_parents = _coerce_int(page.get("/StructParents"))
    if struct_parents is None:
        return None

    for nums, _leaf in _parent_tree_num_arrays(struct_root):
        for i in range(0, len(nums) - 1, 2):
            if _coerce_int(nums[i]) != struct_parents:
                continue
            arr = _resolve_pdf_object(nums[i + 1])
            if not isinstance(arr, pikepdf.Array):
                return None
            if mcid < 0 or mcid >= len(arr):
                return None
            return arr[mcid]
    return None


def _parent_tree_entry_matches(
    entry,
    *,
    expected_node,
    page_idx: int,
    mcid: int,
    pdf: pikepdf.Pdf,
) -> bool:
    """True when a parent-tree entry already resolves to the requested page/MCID."""
    from project_remedy.pdf_fixer import _find_node_page, _get_node_mcids, _same_pdf_object

    if entry is None:
        return False
    if expected_node is not None and _same_pdf_object(entry, expected_node):
        return True

    resolved = _resolve_pdf_object(entry)
    if not isinstance(resolved, pikepdf.Dictionary):
        return False
    if expected_node is not None and _same_pdf_object(resolved, expected_node):
        return True

    if "/S" in resolved:
        return _find_node_page(resolved, pdf) == page_idx and mcid in _get_node_mcids(resolved)

    if _coerce_int(resolved.get("/MCID")) != mcid:
        return False
    return _same_pdf_object(resolved.get("/Pg"), pdf.pages[page_idx].obj)


def _fix_orphan_mcids(pdf: pikepdf.Pdf) -> list[str]:
    """Repair MCID references that are missing from the page parent tree.

    Handles three high-volume 7.1-3 patterns:
    - named ``/Properties`` BDC references whose property dict lacks ``/MCID``
    - inline ``/Tag << >> BDC`` dictionaries with no ``/MCID``
    - existing MCID references whose page parent-tree array is too short or null

    New fallback structure nodes are created as ``/NonStruct`` under the
    document's root structure container.
    """
    changes: list[str] = []
    struct_root = pdf.Root.get("/StructTreeRoot")
    if struct_root is None:
        return changes

    from project_remedy.pdf_checker import walk_structure_tree
    from project_remedy.pdf_fixer import (
        _append_struct_child,
        _find_node_page,
        _get_node_mcids,
        _make_mcr_struct_elem,
        _set_parent_tree_entry,
    )

    container = _ensure_root_struct_container(pdf, struct_root)
    existing_nodes: dict[tuple[int, int], pikepdf.Dictionary] = {}
    for node, _depth, _parent in walk_structure_tree(pdf):
        page_idx = _find_node_page(node, pdf)
        if page_idx < 0 or page_idx >= len(pdf.pages):
            continue
        for mcid in _get_node_mcids(node):
            existing_nodes.setdefault((page_idx, mcid), node)

    assigned_missing_mcids = 0
    linked_orphan_mcids = 0
    fixed_pages = 0

    for page_idx, page in enumerate(pdf.pages):
        try:
            instructions = list(pikepdf.parse_content_stream(page))
        except Exception as e:
            logger.debug("rule_router: _fix_orphan_mcids page %d parse error: %s", page_idx, e)
            continue

        if not instructions:
            continue

        resources = _resolve_pdf_object(page.get("/Resources"))
        properties = None
        if isinstance(resources, pikepdf.Dictionary):
            properties = _resolve_pdf_object(resources.get("/Properties"))
            if not isinstance(properties, pikepdf.Dictionary):
                properties = None

        next_mcid = _next_available_page_mcid(instructions, properties)
        rewritten: list[tuple] = []
        referenced_mcids: dict[int, str] = {}
        page_assigned = 0
        page_linked = 0
        stream_changed = False

        for operands, operator in instructions:
            new_operands = list(operands)
            if str(operator) != "BDC" or len(new_operands) < 2:
                rewritten.append((new_operands, operator))
                continue

            tag = str(new_operands[0]).lstrip("/")
            if tag == "Artifact":
                rewritten.append((new_operands, operator))
                continue

            props_operand = new_operands[1]
            props_dict = None
            mcid = None

            if isinstance(props_operand, pikepdf.Name):
                prop_ref = _named_resource_lookup(properties, str(props_operand).lstrip("/"))
                props_dict = _resolve_pdf_object(prop_ref)
                mcid = _coerce_int(props_dict.get("/MCID")) if isinstance(props_dict, pikepdf.Dictionary) else None
                if mcid is None and isinstance(props_dict, pikepdf.Dictionary):
                    mcid = next_mcid
                    next_mcid += 1
                    page_assigned += 1
                    stream_changed = True
                    inline_props = pikepdf.Dictionary(props_dict)
                    inline_props["/MCID"] = mcid
                    new_operands[1] = inline_props
            else:
                props_dict = _resolve_pdf_object(props_operand)
                if isinstance(props_dict, pikepdf.Dictionary):
                    mcid = _coerce_int(props_dict.get("/MCID"))
                    if mcid is None:
                        mcid = next_mcid
                        next_mcid += 1
                        page_assigned += 1
                        stream_changed = True
                        inline_props = pikepdf.Dictionary(props_dict)
                        inline_props["/MCID"] = mcid
                        new_operands[1] = inline_props

            if mcid is not None:
                referenced_mcids.setdefault(mcid, tag)

            rewritten.append((new_operands, operator))

        if stream_changed:
            page.contents_coalesce()
            page["/Contents"] = pdf.make_stream(pikepdf.unparse_content_stream(rewritten))

        for mcid in sorted(referenced_mcids):
            existing_node = existing_nodes.get((page_idx, mcid))
            entry = _page_parent_tree_entry(struct_root, page, mcid)
            if _parent_tree_entry_matches(
                entry,
                expected_node=existing_node,
                page_idx=page_idx,
                mcid=mcid,
                pdf=pdf,
            ):
                continue

            if existing_node is None:
                existing_node = _make_mcr_struct_elem(pdf, page, container, tag="NonStruct", mcid=mcid)
                _append_struct_child(container, existing_node)
                existing_nodes[(page_idx, mcid)] = existing_node
            else:
                _set_parent_tree_entry(pdf, page, mcid, existing_node)
            page_linked += 1

        if page_assigned or page_linked:
            fixed_pages += 1
            assigned_missing_mcids += page_assigned
            linked_orphan_mcids += page_linked

    if fixed_pages:
        changes.append(
            "rule_router: repaired orphan MCIDs on "
            f"{fixed_pages} page(s) "
            f"(assigned {assigned_missing_mcids} missing MCID(s), "
            f"linked {linked_orphan_mcids} parent-tree entry/entries)"
        )
    return changes


def _fix_pagination_to_artifact(pdf: pikepdf.Pdf) -> list[str]:
    """Backward-compatible wrapper for the generalized orphan-MCID repair."""
    return _fix_orphan_mcids(pdf)


# ---------------------------------------------------------------------------
# Fix untagged Form XObject content (7.1-3)
# ---------------------------------------------------------------------------


def _fix_untagged_xobject_content(pdf: pikepdf.Pdf) -> list[str]:
    """Artifactize decorative Form ``Do`` invocations inside tagged MCID blocks.

    Wrapping the Form XObject stream itself is not enough for many 7.1-3 cases:
    veraPDF still sees the page-level ``Do`` content item inside a tagged MCID
    and treats the invoked untagged Form content as real untagged content.

    The safe deterministic fix is to split the page's marked-content sequence
    around decorative vector-only Form invocations and move those invocations
    into their own ``/Artifact BMC .. EMC`` blocks. After that rewrite, rerun
    the page retagger so orphaned/renumbered MCIDs are reconciled.
    """
    changes: list[str] = []
    fixed_invocations = 0

    from project_remedy.pdf_fixer import _find_existing_mcids, _read_page_content, fix_page_retag

    form_cache: dict[object, _FormXObjectInfo] = {}

    for page_idx, page in enumerate(pdf.pages):
        resources = page.get("/Resources")
        if resources is None:
            continue
        xobjects = resources.get("/XObject")
        if xobjects is None:
            continue

        try:
            instructions = list(pikepdf.parse_content_stream(page))
            if not instructions:
                continue

            tree = _build_marked_content_tree(instructions)
            raw = _read_page_content(page).decode("latin-1", errors="replace")
            next_mcid = max(_find_existing_mcids(raw, page=page), default=-1) + 1
            rewritten, page_fixed, _next_mcid = _rewrite_decorative_form_invocations(
                tree,
                xobjects=xobjects,
                form_cache=form_cache,
                next_mcid=next_mcid,
            )
            if not page_fixed:
                continue

            page.contents_coalesce()
            page["/Contents"] = pdf.make_stream(
                pikepdf.unparse_content_stream(_flatten_marked_content_tree(rewritten))
            )
            fixed_invocations += page_fixed
        except Exception as e:
            logger.debug(
                "rule_router: _fix_untagged_xobject_content page %d error: %s",
                page_idx,
                e,
            )

    if fixed_invocations:
        changes.append(
            "rule_router: artifactized "
            f"{fixed_invocations} decorative Form XObject invocation(s) inside tagged content"
        )
        changes.extend(fix_page_retag(pdf))
    return changes


# ---------------------------------------------------------------------------
# Artifact/tagged nesting fix (7.1-1, 7.1-2)
# ---------------------------------------------------------------------------


def _fix_artifact_tagged_nesting(pdf: pikepdf.Pdf) -> list[str]:
    """Fix 7.1-1 (Artifact inside tagged) and 7.1-2 (tagged inside Artifact).

    Walks each page's content stream and removes /Artifact BDC..EMC blocks
    that appear inside tagged BDC..EMC blocks (7.1-1), and removes tagged
    BDC..EMC blocks that appear inside /Artifact BDC..EMC blocks (7.1-2).

    The fix strips the inner marker pair, promoting the content to the
    enclosing scope (i.e., if Artifact is inside tagged, we remove the
    Artifact markers so the content becomes tagged; if tagged is inside
    Artifact, we remove the tagged markers so the content becomes Artifact).
    """
    changes: list[str] = []
    fixed_pages = 0

    from project_remedy.pdf_fixer import fix_page_retag

    for page_idx, page in enumerate(pdf.pages):
        try:
            instructions = list(pikepdf.parse_content_stream(page))
            if not instructions:
                continue

            tree = _build_marked_content_tree(instructions)
            rewritten, removed_blocks = _rewrite_nested_marked_content(tree)
            if removed_blocks:
                page.contents_coalesce()
                page["/Contents"] = pdf.make_stream(
                    pikepdf.unparse_content_stream(_flatten_marked_content_tree(rewritten))
                )
                fixed_pages += 1

        except Exception as e:
            logger.debug("rule_router: _fix_artifact_tagged_nesting page %d error: %s", page_idx, e)

    if fixed_pages:
        changes.append(f"rule_router: fixed artifact/tagged nesting on {fixed_pages} page(s)")
        changes.extend(fix_page_retag(pdf))
    return changes


# ---------------------------------------------------------------------------
# Table regularity fix (7.2-42, 7.2-43)
# ---------------------------------------------------------------------------


def _fix_table_regularity(pdf: pikepdf.Pdf) -> list[str]:
    """Delegate 7.2-42/7.2-43 repairs to the stronger shared table fixers."""
    changes: list[str] = []

    try:
        from project_remedy.pdf_fixer import fix_table_header_scope, fix_table_regularity

        changes.extend(fix_table_header_scope(pdf))
        changes.extend(fix_table_regularity(pdf))

    except Exception as e:
        logger.exception("rule_router: _fix_table_regularity error")
        changes.append(f"rule_router: _fix_table_regularity error: {e}")

    return changes


# ---------------------------------------------------------------------------
# CIDSet fix (7.21.4.2-2)
# ---------------------------------------------------------------------------


def _fix_cidset(pdf: pikepdf.Pdf) -> list[str]:
    """Fix 7.21.4.2-2: CIDSet must list ALL CIDs present in the font program.

    Uses fontTools to count actual glyphs in the font program and builds a
    precise CIDSet bitstring. Falls back to blanket 0xFF for unparseable fonts.
    """
    changes: list[str] = []
    fixed = 0

    for page in pdf.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get("/Font")
        if fonts is None:
            continue

        try:
            for font_name in fonts.keys():
                font = fonts[font_name]
                subtype = str(font.get("/Subtype", ""))
                if subtype != "/Type0":
                    continue

                desc_fonts = font.get("/DescendantFonts")
                if desc_fonts is None:
                    continue

                for df_ref in (list(desc_fonts) if isinstance(desc_fonts, pikepdf.Array) else [desc_fonts]):
                    try:
                        df = df_ref if isinstance(df_ref, pikepdf.Dictionary) else df_ref.get_object()
                        fd = df.get("/FontDescriptor")
                        if fd is None:
                            continue
                        fd = fd if isinstance(fd, pikepdf.Dictionary) else fd.get_object()

                        # Find font program
                        font_stream = None
                        for key in ("/FontFile2", "/FontFile3", "/FontFile"):
                            font_stream = fd.get(key)
                            if font_stream is not None:
                                break

                        if font_stream is not None:
                            try:
                                font_bytes = bytes(font_stream.read_bytes())
                                num_glyphs = _count_font_glyphs(font_bytes)
                                num_bytes = (num_glyphs + 7) // 8
                                cidset = bytearray(num_bytes)
                                for cid in range(num_glyphs):
                                    cidset[cid // 8] |= (1 << (7 - (cid % 8)))
                                fd["/CIDSet"] = pdf.make_stream(bytes(cidset))
                                fixed += 1
                                continue
                            except Exception:
                                pass

                        # Fallback: blanket CIDSet
                        fd["/CIDSet"] = pdf.make_stream(b"\xff" * 8192)
                        fixed += 1
                    except Exception:
                        pass
        except Exception:
            pass

    if fixed:
        changes.append(f"rule_router: regenerated CIDSet for {fixed} CID font(s)")
    return changes


def _count_font_glyphs(font_bytes: bytes) -> int:
    """Count glyphs in a font program (TrueType or CFF)."""
    from io import BytesIO

    magic = font_bytes[:4]
    if magic[:2] == b"\x01\x00":
        # CFF font
        try:
            from fontTools.cffLib import CFFFontSet
            cff = CFFFontSet()
            cff.decompile(BytesIO(font_bytes), None)
            top_dict = cff.topDictIndex[0]
            if hasattr(top_dict, "charset") and top_dict.charset:
                return len(top_dict.charset)
            if hasattr(top_dict, "CharStrings"):
                return len(top_dict.CharStrings)
        except Exception:
            pass
        raise ValueError("Cannot parse CFF font")

    # TrueType / OpenType-TT
    from fontTools.ttLib import TTFont
    tt = TTFont(BytesIO(font_bytes))
    count = len(tt.getGlyphOrder())
    tt.close()
    return count


# ---------------------------------------------------------------------------
# Font embedding fix (7.21.4.1-1) — best effort
# ---------------------------------------------------------------------------


def _fix_tounicode_fitz_layer4(pdf: pikepdf.Pdf, pdf_path: Path) -> list[str]:
    """Layer 4: Use PyMuPDF (fitz) to recover ToUnicode mappings for CID fonts.

    fitz often resolves CID font Unicode internally even when the PDF's
    ToUnicode CMap is missing, because it uses its own font resolution.
    This pass finds Type0 fonts still lacking ToUnicode after Layers 1-3,
    extracts character mappings from fitz's rawdict, and writes ToUnicode CMaps.
    """
    changes: list[str] = []
    fixed_fonts = 0

    try:
        import fitz
        from project_remedy.pdf_fixer import _build_bfchar_cmap, _is_tounicode_empty_or_invalid

        doc = fitz.open(str(pdf_path))

        for page_idx, page in enumerate(pdf.pages):
            resources = page.get("/Resources")
            if resources is None:
                continue
            fonts_dict = resources.get("/Font")
            if fonts_dict is None:
                continue

            fitz_page = doc[page_idx]
            try:
                rawdict = fitz_page.get_text("rawdict")
            except Exception:
                continue

            for font_name in fonts_dict.keys():
                font = fonts_dict[font_name]
                subtype = str(font.get("/Subtype", ""))
                if subtype != "/Type0":
                    continue

                # Skip fonts that already have a complete ToUnicode.
                # But DO process fonts with incomplete ToUnicode (7.21.7-1)
                tounicode = font.get("/ToUnicode")
                has_tounicode = tounicode is not None and not _is_tounicode_empty_or_invalid(tounicode)
                # We'll check completeness AFTER extracting used codes below

                base_font = str(font.get("/BaseFont", "")).lstrip("/")
                match_name = base_font.split("+", 1)[1] if "+" in base_font else base_font

                # Extract CID codes from content stream
                instructions = pikepdf.parse_content_stream(page)
                current_font = None
                codes_in_order: list[int] = []

                for operands, operator in instructions:
                    op = str(operator)
                    if op == "Tf" and len(operands) >= 1:
                        current_font = str(operands[0])
                    elif op in ("Tj", "'", '"') and current_font == str(font_name):
                        raw = operands[-1] if op == '"' else operands[0]
                        try:
                            raw_bytes = bytes(raw)
                        except Exception:
                            raw_bytes = b""
                        for i in range(0, len(raw_bytes) - 1, 2):
                            codes_in_order.append((raw_bytes[i] << 8) | raw_bytes[i + 1])
                    elif op == "TJ" and current_font == str(font_name):
                        arr = operands[0]
                        if isinstance(arr, pikepdf.Array):
                            for item in arr:
                                if isinstance(item, (pikepdf.String, bytes)):
                                    try:
                                        raw_bytes = bytes(item)
                                    except Exception:
                                        raw_bytes = b""
                                    for i in range(0, len(raw_bytes) - 1, 2):
                                        codes_in_order.append((raw_bytes[i] << 8) | raw_bytes[i + 1])

                if not codes_in_order:
                    continue

                # Get fitz Unicode chars for matching font
                fitz_chars: list[str] = []
                for block in rawdict.get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if match_name not in span.get("font", ""):
                                continue
                            for char_info in span.get("chars", []):
                                c = char_info.get("c", "")
                                if c and c != "\ufffd":
                                    fitz_chars.append(c)

                # Build CID→Unicode mapping
                mapping: dict[int, int] = {}
                if len(codes_in_order) == len(fitz_chars):
                    for code, char in zip(codes_in_order, fitz_chars):
                        mapping[code] = ord(char)
                elif fitz_chars:
                    min_len = min(len(codes_in_order), len(fitz_chars))
                    for code, char in zip(codes_in_order[:min_len], fitz_chars[:min_len]):
                        mapping[code] = ord(char)

                if not mapping:
                    continue

                # If font already has ToUnicode, merge fitz mappings with existing
                if has_tounicode:
                    from project_remedy.pdf_checker import _parse_tounicode_mapped_codes
                    try:
                        existing_codes = _parse_tounicode_mapped_codes(tounicode)
                        new_codes = {code: uni for code, uni in mapping.items()
                                    if code not in existing_codes}
                        if not new_codes:
                            continue
                        # Parse existing CMap data to preserve, then merge
                        existing_bytes = bytes(tounicode.read_bytes())
                        existing_text = existing_bytes.decode("latin-1", errors="replace")
                        # Extract existing bfchar mappings
                        import re
                        existing_mapping: dict[int, int] = {}
                        for match in re.finditer(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", existing_text):
                            src_hex, dst_hex = match.group(1), match.group(2)
                            try:
                                src = int(src_hex, 16)
                                dst = int(dst_hex[:4], 16)  # First BMP char
                                existing_mapping[src] = dst
                            except ValueError:
                                pass
                        # Merge: existing + new (existing takes precedence)
                        merged = {**mapping, **existing_mapping}
                        mapping = merged
                    except Exception:
                        pass

                # Write ToUnicode CMap
                cmap_bytes = _build_bfchar_cmap(mapping, byte_width=2)
                font["/ToUnicode"] = pdf.make_indirect(pikepdf.Stream(pdf, cmap_bytes))
                fixed_fonts += 1

        doc.close()

    except Exception as e:
        logger.debug("rule_router: _fix_tounicode_fitz_layer4 error: %s", e)

    if fixed_fonts:
        changes.append(f"rule_router: Layer 4 fitz-based ToUnicode for {fixed_fonts} CID font(s)")
    return changes


def _fix_gs_corrupted_encoding(pdf: pikepdf.Pdf) -> list[str]:
    """Fix 7.21.6-2: Remove Ghostscript-corrupted /Differences entries.

    Ghostscript redistillation can produce non-AGL glyph names like
    /~gs~gName~0001 in the /Differences array. veraPDF rejects these as
    non-Unicode-compliant. Fix: remove the Differences, keep BaseEncoding.
    """
    changes: list[str] = []
    fixed = 0

    for page in pdf.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get("/Font")
        if fonts is None:
            continue

        try:
            for font_name in fonts.keys():
                font = fonts[font_name]
                subtype = str(font.get("/Subtype", ""))
                if subtype != "/TrueType":
                    continue

                encoding = font.get("/Encoding")
                if not isinstance(encoding, pikepdf.Dictionary):
                    continue

                diffs = encoding.get("/Differences")
                if diffs is None:
                    continue

                has_bad_names = False
                for item in diffs:
                    if isinstance(item, pikepdf.Name):
                        name = str(item).lstrip("/")
                        if name.startswith("~gs~") or name.startswith(".notdef"):
                            has_bad_names = True
                            break

                if not has_bad_names:
                    continue

                base_enc = encoding.get("/BaseEncoding")
                if base_enc:
                    font["/Encoding"] = base_enc
                else:
                    font["/Encoding"] = pikepdf.Name("/WinAnsiEncoding")
                fixed += 1
        except Exception:
            pass

    if fixed:
        changes.append(
            f"rule_router: removed GS-corrupted Differences from {fixed} TrueType font(s)"
        )
    return changes


def _fix_font_embedding(pdf: pikepdf.Pdf) -> list[str]:
    """Best-effort fix for 7.21.4.1-1: font programs should be embedded.

    For fonts that have a FontDescriptor but no embedded font file,
    this is generally not fixable without the original font file.
    However, we can check if there's a font file reference that's broken
    and try to repair it, or mark the font as a known limitation.

    For subset-embedded fonts that are missing some glyphs (7.21.4.1-2),
    there's not much we can do without the original font, but we log it.
    """
    changes: list[str] = []
    not_embedded = 0

    for page in pdf.pages:
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get("/Font")
        if fonts is None:
            continue

        try:
            for font_name in fonts.keys():
                font = fonts[font_name]
                fd = font.get("/FontDescriptor")
                if fd is None:
                    continue
                fd = fd if isinstance(fd, pikepdf.Dictionary) else fd.get_object()

                has_font_file = (
                    fd.get("/FontFile") is not None
                    or fd.get("/FontFile2") is not None
                    or fd.get("/FontFile3") is not None
                )
                if not has_font_file:
                    not_embedded += 1
        except Exception:
            pass

    if not_embedded:
        changes.append(
            f"rule_router: {not_embedded} font(s) not embedded (source-font limitation, cannot fix without original font file)"
        )
    return changes


# ---------------------------------------------------------------------------
# Table structure rules fix (7.2-3, 7.2-6, 7.2-10, 7.2-14)
# ---------------------------------------------------------------------------


def _fix_table_structure_rules(pdf: pikepdf.Pdf) -> list[str]:
    """Delegate structure repair to the shared table fixers.

    The local router version previously retagged malformed children in place,
    which preserves the wrong tree shape on irregular real-world tables.
    """
    changes: list[str] = []

    try:
        from project_remedy.pdf_fixer import (
            fix_table_headers,
            fix_table_parent_structure,
            fix_table_td_headers,
        )

        changes.extend(fix_table_parent_structure(pdf))
        changes.extend(fix_table_headers(pdf))
        changes.extend(fix_table_td_headers(pdf))

    except Exception as e:
        logger.exception("rule_router: _fix_table_structure_rules error")
        changes.append(f"rule_router: _fix_table_structure_rules error: {e}")

    return changes


# ---------------------------------------------------------------------------
# List structure fix (7.2-19)
# ---------------------------------------------------------------------------


def _fix_list_structure(pdf: pikepdf.Pdf) -> list[str]:
    """Fix 7.2-19: L element may contain only L, LI, and Caption elements.

    Non-list children inside L elements are wrapped in LI>LBody.
    """
    changes: list[str] = []

    try:
        from project_remedy.pdf_checker import walk_structure_tree

        L_CHILD_TYPES = {"L", "LI", "Caption"}

        for node, _depth, _parent in walk_structure_tree(pdf):
            s_type = str(node.get("/S", "")).lstrip("/")
            if s_type != "L":
                continue

            kids = node.get("/K")
            if kids is None:
                continue
            items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]

            new_kids = pikepdf.Array()
            fixed = False

            for item in items:
                try:
                    child = item if isinstance(item, pikepdf.Dictionary) else item.get_object()
                    child_type = str(child.get("/S", "")).lstrip("/")
                    if child_type in L_CHILD_TYPES:
                        new_kids.append(item)
                    else:
                        # Wrap in LI > LBody
                        lbody = pikepdf.Dictionary({
                            "/Type": pikepdf.Name("/StructElem"),
                            "/S": pikepdf.Name("/LBody"),
                            "/K": pikepdf.Array([item]),
                        })
                        lbody = pdf.make_indirect(lbody)
                        child["/P"] = lbody

                        li = pikepdf.Dictionary({
                            "/Type": pikepdf.Name("/StructElem"),
                            "/S": pikepdf.Name("/LI"),
                            "/P": node,
                            "/K": pikepdf.Array([lbody]),
                        })
                        li = pdf.make_indirect(li)
                        lbody["/P"] = li
                        new_kids.append(li)
                        fixed = True
                except Exception:
                    new_kids.append(item)

            if fixed:
                node["/K"] = new_kids
                changes.append("rule_router: wrapped non-LI children in L element into LI>LBody")

    except Exception as e:
        logger.exception("rule_router: _fix_list_structure error")
        changes.append(f"rule_router: _fix_list_structure error: {e}")

    return changes


# ---------------------------------------------------------------------------
# LI parent containment fix (7.2-17)
# ---------------------------------------------------------------------------


def _fix_li_parent_containment(pdf: pikepdf.Pdf) -> list[str]:
    """Fix 7.2-17: LI elements must be contained in L elements.

    Wraps orphan LI nodes (whose parent is not L) in a synthetic L wrapper.
    """
    changes: list[str] = []
    fixed = 0

    try:
        from project_remedy.pdf_checker import walk_structure_tree

        for node, _depth, parent in walk_structure_tree(pdf):
            if parent is None:
                continue
            s_type = str(node.get("/S", "")).lstrip("/")
            if s_type != "LI":
                continue
            parent_type = str(parent.get("/S", "")).lstrip("/")
            if parent_type == "L":
                continue

            # Wrap LI in a synthetic L
            l_wrapper = pdf.make_indirect(pikepdf.Dictionary({
                "/Type": pikepdf.Name("/StructElem"),
                "/S": pikepdf.Name("/L"),
                "/P": parent,
                "/K": pikepdf.Array([node]),
            }))
            node["/P"] = l_wrapper

            # Replace the LI in parent's /K with the wrapper
            kids = parent.get("/K")
            if isinstance(kids, pikepdf.Array):
                new_kids = pikepdf.Array()
                for kid in kids:
                    try:
                        kid_obj = kid if isinstance(kid, pikepdf.Dictionary) else kid.get_object()
                        if getattr(kid_obj, "objgen", None) == getattr(node, "objgen", None):
                            new_kids.append(l_wrapper)
                        else:
                            new_kids.append(kid)
                    except Exception:
                        new_kids.append(kid)
                parent["/K"] = new_kids
            elif kids is not None:
                parent["/K"] = l_wrapper
            fixed += 1

    except Exception as e:
        logger.exception("rule_router: _fix_li_parent_containment error")

    if fixed:
        changes.append(f"rule_router: wrapped {fixed} orphan LI element(s) in L wrapper")
    return changes


# ---------------------------------------------------------------------------
# ViewerPreferences fix (7.1-10)
# ---------------------------------------------------------------------------


def _fix_viewer_preferences(pdf: pikepdf.Pdf) -> list[str]:
    """Fix 7.1-10: ViewerPreferences must have DisplayDocTitle = true."""
    changes: list[str] = []
    vp = pdf.Root.get("/ViewerPreferences")
    if vp is None:
        pdf.Root["/ViewerPreferences"] = pikepdf.Dictionary({
            "/DisplayDocTitle": True,
        })
        changes.append("rule_router: added /ViewerPreferences with DisplayDocTitle=true")
    elif not isinstance(vp, pikepdf.Dictionary):
        pass
    else:
        if vp.get("/DisplayDocTitle") != True:
            vp["/DisplayDocTitle"] = True
            changes.append("rule_router: set DisplayDocTitle=true")
    return changes


# ---------------------------------------------------------------------------
# Natural language fixes (7.2-2, 7.2-21, 7.2-34)
# ---------------------------------------------------------------------------


def _fix_natural_language(pdf: pikepdf.Pdf) -> list[str]:
    """Fix natural language rules by ensuring /Lang is set.

    7.2-2:  Natural language in Outline entries
    7.2-21: Natural language in ActualText
    7.2-34: Natural language for text in page content

    Sets /Lang on the document catalog if missing, and propagates to
    structure elements and outline entries.
    """
    changes: list[str] = []

    # Ensure document-level /Lang
    if pdf.Root.get("/Lang") is None:
        pdf.Root["/Lang"] = pikepdf.String("en")
        changes.append("rule_router: set document /Lang=en")

    lang_val = pdf.Root["/Lang"]

    # Set /Lang on StructTreeRoot if missing
    struct_root = pdf.Root.get("/StructTreeRoot")
    if struct_root is not None:
        doc_elem = struct_root.get("/K")
        if doc_elem is not None:
            if isinstance(doc_elem, pikepdf.Array) and len(doc_elem) > 0:
                doc_elem = doc_elem[0]
            if hasattr(doc_elem, "get_object"):
                try:
                    doc_elem = doc_elem.get_object()
                except Exception:
                    pass
            if isinstance(doc_elem, pikepdf.Dictionary) and doc_elem.get("/Lang") is None:
                doc_elem["/Lang"] = lang_val
                changes.append("rule_router: set /Lang on document StructElem")

    # Set /Lang on Outline entries (7.2-2)
    outlines = pdf.Root.get("/Outlines")
    if outlines is not None:
        fixed_outlines = 0

        def _walk_outlines(node):
            nonlocal fixed_outlines
            if not isinstance(node, pikepdf.Dictionary):
                return
            if node.get("/Title") is not None and node.get("/Lang") is None:
                node["/Lang"] = lang_val
                fixed_outlines += 1
            first = node.get("/First")
            if first is not None:
                try:
                    _walk_outlines(first if isinstance(first, pikepdf.Dictionary) else first.get_object())
                except Exception:
                    pass
            nxt = node.get("/Next")
            if nxt is not None:
                try:
                    _walk_outlines(nxt if isinstance(nxt, pikepdf.Dictionary) else nxt.get_object())
                except Exception:
                    pass

        try:
            _walk_outlines(outlines)
        except Exception:
            pass
        if fixed_outlines:
            changes.append(f"rule_router: set /Lang on {fixed_outlines} outline entries")

    return changes


# ---------------------------------------------------------------------------
# List item structure fix (7.2-20)
# ---------------------------------------------------------------------------


def _fix_list_item_structure(pdf: pikepdf.Pdf) -> list[str]:
    """Fix 7.2-20: LI may only contain Lbl and LBody.

    Non-Lbl/LBody children inside LI are wrapped in LBody.
    """
    changes: list[str] = []

    try:
        from project_remedy.pdf_checker import walk_structure_tree

        LI_CHILD_TYPES = {"Lbl", "LBody"}

        for node, _depth, _parent in walk_structure_tree(pdf):
            s_type = str(node.get("/S", "")).lstrip("/")
            if s_type != "LI":
                continue

            kids = node.get("/K")
            if kids is None:
                continue
            items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]

            new_kids = pikepdf.Array()
            fixed = False

            for item in items:
                try:
                    child = item if isinstance(item, pikepdf.Dictionary) else item.get_object()
                    child_type = str(child.get("/S", "")).lstrip("/")
                    if child_type in LI_CHILD_TYPES:
                        new_kids.append(item)
                    else:
                        # Wrap in LBody
                        lbody = pikepdf.Dictionary({
                            "/Type": pikepdf.Name("/StructElem"),
                            "/S": pikepdf.Name("/LBody"),
                            "/P": node,
                            "/K": pikepdf.Array([item]),
                        })
                        lbody = pdf.make_indirect(lbody)
                        try:
                            child["/P"] = lbody
                        except Exception:
                            pass
                        new_kids.append(lbody)
                        fixed = True
                except Exception:
                    new_kids.append(item)

            if fixed:
                node["/K"] = new_kids
                changes.append("rule_router: wrapped non-Lbl/LBody children in LI into LBody")

    except Exception as e:
        logger.debug("rule_router: _fix_list_item_structure error: %s", e)

    return changes


# ---------------------------------------------------------------------------
# Table headers/IDs fix (7.5-1)
# ---------------------------------------------------------------------------


def _fix_table_headers_ids(pdf: pikepdf.Pdf) -> list[str]:
    """Fix 7.5-1: table structure must be determinable via Headers and IDs.

    For tables that don't have /Headers or /ID attributes on their cells,
    add /Scope to TH cells and /ID attributes to enable header association.
    """
    changes: list[str] = []

    try:
        from project_remedy.pdf_checker import walk_structure_tree

        table_count = 0
        for node, _depth, _parent in walk_structure_tree(pdf):
            s_type = str(node.get("/S", "")).lstrip("/")
            if s_type != "Table":
                continue

            # Walk table rows and add /Scope to TH cells
            kids = node.get("/K")
            if kids is None:
                continue

            th_fixed = 0
            items = list(kids) if isinstance(kids, pikepdf.Array) else [kids]

            for row_ref in items:
                try:
                    row = row_ref if isinstance(row_ref, pikepdf.Dictionary) else row_ref.get_object()
                    row_type = str(row.get("/S", "")).lstrip("/")

                    # Handle THead/TBody/TFoot containing TRs
                    if row_type in ("THead", "TBody", "TFoot"):
                        sub_kids = row.get("/K")
                        if sub_kids is None:
                            continue
                        sub_items = list(sub_kids) if isinstance(sub_kids, pikepdf.Array) else [sub_kids]
                    elif row_type == "TR":
                        sub_items = [row_ref]
                    else:
                        continue

                    for tr_ref in sub_items:
                        try:
                            tr = tr_ref if isinstance(tr_ref, pikepdf.Dictionary) else tr_ref.get_object()
                            if str(tr.get("/S", "")).lstrip("/") != "TR":
                                continue
                            cells = tr.get("/K")
                            if cells is None:
                                continue
                            cell_list = list(cells) if isinstance(cells, pikepdf.Array) else [cells]

                            for cell_ref in cell_list:
                                try:
                                    cell = cell_ref if isinstance(cell_ref, pikepdf.Dictionary) else cell_ref.get_object()
                                    if str(cell.get("/S", "")).lstrip("/") == "TH":
                                        # Add /Scope if missing
                                        attrs = cell.get("/A")
                                        has_scope = False
                                        if isinstance(attrs, pikepdf.Dictionary):
                                            has_scope = attrs.get("/Scope") is not None
                                        elif isinstance(attrs, pikepdf.Array):
                                            for a in attrs:
                                                try:
                                                    a_obj = a if isinstance(a, pikepdf.Dictionary) else a.get_object()
                                                    if isinstance(a_obj, pikepdf.Dictionary) and a_obj.get("/Scope") is not None:
                                                        has_scope = True
                                                        break
                                                except Exception:
                                                    pass

                                        if not has_scope:
                                            # Determine scope: Column for first row, Row otherwise
                                            scope = "Column" if row_type == "THead" else "Row"
                                            scope_dict = pikepdf.Dictionary({
                                                "/O": pikepdf.Name("/Table"),
                                                "/Scope": pikepdf.Name(f"/{scope}"),
                                            })
                                            if attrs is None:
                                                cell["/A"] = scope_dict
                                            elif isinstance(attrs, pikepdf.Array):
                                                attrs.append(scope_dict)
                                            else:
                                                cell["/A"] = pikepdf.Array([attrs, scope_dict])
                                            th_fixed += 1
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

            if th_fixed:
                table_count += 1
                changes.append(f"rule_router: added /Scope to {th_fixed} TH cells in table")

    except Exception as e:
        logger.debug("rule_router: _fix_table_headers_ids error: %s", e)

    return changes
