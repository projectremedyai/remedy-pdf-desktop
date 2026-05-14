# Copyright 2025-2026 Hancom Inc. (original Java implementation)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Python port for Project Remedy.
# Original: XYCutPlusPlusSorter.java from opendataloader-pdf
# Reference: arXiv:2504.10258 — XY-Cut++ reading order algorithm
"""XY-Cut++ reading order algorithm for PDF layout analysis.

A deterministic, geometry-only algorithm that sorts page elements into
reading order by recursively splitting regions along the largest gaps.

Four phases:
  1. Pre-mask cross-layout elements (headers/footers spanning columns)
  2. Compute density ratio → choose axis preference
  3. Recursive segmentation with adaptive XY/YX-Cut
  4. Merge cross-layout elements back at correct Y positions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants (preserved from Java source)
# ---------------------------------------------------------------------------
DEFAULT_BETA: float = 2.0
DEFAULT_DENSITY_THRESHOLD: float = 0.9
OVERLAP_THRESHOLD: float = 0.1
MIN_OVERLAP_COUNT: int = 2
MIN_GAP_THRESHOLD: float = 5.0  # PDF points
NARROW_ELEMENT_WIDTH_RATIO: float = 0.1


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BBox:
    """Bounding box in PDF coordinates (origin bottom-left, Y increases up)."""

    left: float
    bottom: float
    right: float
    top: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.top - self.bottom)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bottom + self.top) / 2.0

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.left, other.left),
            min(self.bottom, other.bottom),
            max(self.right, other.right),
            max(self.top, other.top),
        )


# ---------------------------------------------------------------------------
# Element type alias
# ---------------------------------------------------------------------------
Element = tuple[BBox, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def xy_cut_sort(
    elements: list[Element],
    beta: float = DEFAULT_BETA,
    density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
) -> list[Element]:
    """Sort elements into reading order using XY-Cut++.

    Parameters
    ----------
    elements:
        List of ``(BBox, payload)`` tuples.  Payload is opaque (struct
        element ref, MCID, SemanticRegion, etc.).
    beta:
        Cross-layout width threshold multiplier.
    density_threshold:
        Density ratio above which horizontal cuts are preferred.

    Returns
    -------
    The same tuples, reordered into reading order.
    """
    if not elements:
        return elements if elements is not None else []

    valid = [(b, p) for b, p in elements if b is not None]
    if len(valid) <= 1:
        return valid

    # Phase 1 — pre-mask cross-layout elements
    cross = _identify_cross_layout(valid, beta)
    cross_set = set(id(e) for e in cross)
    remaining = [e for e in valid if id(e) not in cross_set]

    if not remaining:
        return _sort_y_then_x(valid)

    # Phase 2 — density → axis preference
    density = _compute_density_ratio(remaining)
    prefer_horizontal = density > density_threshold

    # Phase 3 — recursive segmentation
    sorted_main = _recursive_segment(remaining, prefer_horizontal)

    # Phase 4 — merge cross-layout elements
    return _merge_cross_layout(sorted_main, cross)


# ---------------------------------------------------------------------------
# Phase 1 — cross-layout detection
# ---------------------------------------------------------------------------
def _identify_cross_layout(elements: list[Element], beta: float) -> list[Element]:
    if len(elements) < 3:
        return []

    max_width = max(b.width for b, _ in elements)
    threshold = beta * max_width

    return [
        e
        for e in elements
        if e[0].width >= threshold
        and _has_minimum_overlaps(e, elements, MIN_OVERLAP_COUNT)
    ]


def _has_minimum_overlaps(
    element: Element, all_elements: list[Element], min_count: int
) -> bool:
    count = 0
    for other in all_elements:
        if other is element:
            continue
        if _horizontal_overlap_ratio(element[0], other[0]) >= OVERLAP_THRESHOLD:
            count += 1
            if count >= min_count:
                return True
    return False


def _horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap_left = max(a.left, b.left)
    overlap_right = min(a.right, b.right)
    overlap_w = max(0.0, overlap_right - overlap_left)
    if overlap_w <= 0:
        return 0.0
    smaller = min(a.width, b.width)
    return overlap_w / smaller if smaller > 0 else 0.0


# ---------------------------------------------------------------------------
# Phase 2 — density ratio
# ---------------------------------------------------------------------------
def _compute_density_ratio(elements: list[Element]) -> float:
    if not elements:
        return 1.0
    region = _bounding_region(elements)
    if region is None or region.area <= 0:
        return 1.0
    content_area = sum(b.area for b, _ in elements)
    return min(1.0, content_area / region.area)


def _bounding_region(elements: list[Element]) -> BBox | None:
    if not elements:
        return None
    result = elements[0][0]
    for b, _ in elements[1:]:
        result = result.union(b)
    return result if result.area > 0 else None


# ---------------------------------------------------------------------------
# Phase 3 — recursive segmentation
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _CutInfo:
    position: float
    gap: float


def _recursive_segment(
    elements: list[Element], prefer_horizontal: bool
) -> list[Element]:
    if len(elements) <= 1:
        return list(elements)

    h_cut = _best_horizontal_cut(elements)
    v_cut = _best_vertical_cut(elements)

    valid_h = h_cut.gap >= MIN_GAP_THRESHOLD
    valid_v = v_cut.gap >= MIN_GAP_THRESHOLD

    if valid_h and valid_v:
        if h_cut.gap == v_cut.gap:
            use_h = prefer_horizontal
        else:
            use_h = h_cut.gap > v_cut.gap
    elif valid_h:
        use_h = True
    elif valid_v:
        use_h = False
    else:
        return _sort_y_then_x(elements)

    if use_h:
        groups = _split_horizontal(elements, h_cut.position)
    else:
        groups = _split_vertical(elements, v_cut.position)

    if len(groups) <= 1:
        return _sort_y_then_x(elements)

    result: list[Element] = []
    for g in groups:
        result.extend(_recursive_segment(g, prefer_horizontal))
    return result


# ---------------------------------------------------------------------------
# Cut detection
# ---------------------------------------------------------------------------
def _best_horizontal_cut(elements: list[Element]) -> _CutInfo:
    if len(elements) < 2:
        return _CutInfo(0.0, 0.0)

    # Sort top-to-bottom (descending top Y)
    by_top = sorted(elements, key=lambda e: -e[0].top)

    largest_gap = 0.0
    cut_pos = 0.0
    prev_bottom: float | None = None

    for b, _ in by_top:
        if prev_bottom is not None and prev_bottom > b.top:
            gap = prev_bottom - b.top
            if gap > largest_gap:
                largest_gap = gap
                cut_pos = (prev_bottom + b.top) / 2.0
        prev_bottom = b.bottom if prev_bottom is None else min(prev_bottom, b.bottom)

    return _CutInfo(cut_pos, largest_gap)


def _best_vertical_cut(elements: list[Element]) -> _CutInfo:
    if len(elements) < 2:
        return _CutInfo(0.0, 0.0)

    edge_cut = _vertical_cut_by_edges(elements)
    if edge_cut.gap >= MIN_GAP_THRESHOLD:
        return edge_cut

    # Retry without narrow outliers (page numbers, footnote markers)
    if len(elements) >= 3:
        region = _bounding_region(elements)
        if region is not None:
            narrow_threshold = region.width * NARROW_ELEMENT_WIDTH_RATIO
            filtered = [e for e in elements if e[0].width >= narrow_threshold]
            if 2 <= len(filtered) < len(elements):
                filtered_cut = _vertical_cut_by_edges(filtered)
                if filtered_cut.gap > edge_cut.gap and filtered_cut.gap >= MIN_GAP_THRESHOLD:
                    return filtered_cut

    return edge_cut


def _vertical_cut_by_edges(elements: list[Element]) -> _CutInfo:
    by_left = sorted(elements, key=lambda e: (e[0].left, e[0].right))

    largest_gap = 0.0
    cut_pos = 0.0
    prev_right: float | None = None

    for b, _ in by_left:
        if prev_right is not None and b.left > prev_right:
            gap = b.left - prev_right
            if gap > largest_gap:
                largest_gap = gap
                cut_pos = (prev_right + b.left) / 2.0
        prev_right = b.right if prev_right is None else max(prev_right, b.right)

    return _CutInfo(cut_pos, largest_gap)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def _split_horizontal(elements: list[Element], cut_y: float) -> list[list[Element]]:
    above = [e for e in elements if e[0].center_y > cut_y]
    below = [e for e in elements if e[0].center_y <= cut_y]
    groups: list[list[Element]] = []
    if above:
        groups.append(above)
    if below:
        groups.append(below)
    return groups


def _split_vertical(elements: list[Element], cut_x: float) -> list[list[Element]]:
    left = [e for e in elements if e[0].center_x < cut_x]
    right = [e for e in elements if e[0].center_x >= cut_x]
    groups: list[list[Element]] = []
    if left:
        groups.append(left)
    if right:
        groups.append(right)
    return groups


# ---------------------------------------------------------------------------
# Phase 4 — merge cross-layout
# ---------------------------------------------------------------------------
def _merge_cross_layout(
    sorted_main: list[Element], cross: list[Element]
) -> list[Element]:
    if not cross:
        return sorted_main
    if not sorted_main:
        return _sort_y_then_x(cross)

    sorted_cross = _sort_y_then_x(cross)
    result: list[Element] = []
    mi, ci = 0, 0

    while mi < len(sorted_main) or ci < len(sorted_cross):
        if ci >= len(sorted_cross):
            result.append(sorted_main[mi])
            mi += 1
        elif mi >= len(sorted_main):
            result.append(sorted_cross[ci])
            ci += 1
        else:
            main_top = sorted_main[mi][0].top
            cross_top = sorted_cross[ci][0].top
            if cross_top >= main_top:
                result.append(sorted_cross[ci])
                ci += 1
            else:
                result.append(sorted_main[mi])
                mi += 1

    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _sort_y_then_x(elements: list[Element]) -> list[Element]:
    return sorted(elements, key=lambda e: (-e[0].top, e[0].left))
