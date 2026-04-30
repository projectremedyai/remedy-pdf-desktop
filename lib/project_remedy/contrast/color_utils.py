"""WCAG color math — luminance, contrast ratio, color conversion.

All colors are represented as (r, g, b) tuples with values normalized to 0-1.
"""

from __future__ import annotations

import colorsys
import math


def relative_luminance(r: float, g: float, b: float) -> float:
    """Compute WCAG 2.x relative luminance from sRGB values (0-1).

    Uses the sRGB linearization formula from WCAG 2.x.
    """
    def linearize(c: float) -> float:
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def contrast_ratio(fg_rgb: tuple[float, float, float],
                   bg_rgb: tuple[float, float, float]) -> float:
    """Compute WCAG contrast ratio between two sRGB colors.

    Returns a value >= 1.0 (lighter on top).
    """
    l1 = relative_luminance(*fg_rgb)
    l2 = relative_luminance(*bg_rgb)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> tuple[float, float, float]:
    """Convert CMYK (0-1) to RGB (0-1) using simple subtractive model."""
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return (r, g, b)


def gray_to_rgb(g: float) -> tuple[float, float, float]:
    """Convert grayscale (0-1) to RGB (0-1)."""
    return (g, g, g)


def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    """Convert hex color string (#RRGGBB or RRGGBB) to RGB (0-1)."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b)


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert RGB (0-1) to hex string (#RRGGBB)."""
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
    )


def int_color_to_rgb(color_int: int) -> tuple[float, float, float]:
    """Convert pymupdf integer color (0xRRGGBB) to RGB (0-1)."""
    r = ((color_int >> 16) & 0xFF) / 255.0
    g = ((color_int >> 8) & 0xFF) / 255.0
    b = (color_int & 0xFF) / 255.0
    return (r, g, b)


def is_large_text(font_size_pt: float, is_bold: bool = False) -> bool:
    """Determine if text qualifies as 'large' under WCAG 1.4.3.

    Large text: >= 18pt, or >= 14pt if bold.
    """
    if font_size_pt >= 18.0:
        return True
    if is_bold and font_size_pt >= 14.0:
        return True
    return False


def wcag_threshold(level: str = "AA", large: bool = False) -> float:
    """Return the minimum contrast ratio for the given WCAG level and text size.

    AA: 4.5:1 normal, 3.0:1 large
    AAA: 7.0:1 normal, 4.5:1 large
    """
    if level.upper() == "AAA":
        return 4.5 if large else 7.0
    # Default AA
    return 3.0 if large else 4.5


# WCAG 1.4.11 non-text contrast threshold
NON_TEXT_THRESHOLD = 3.0


def nearest_passing_color(
    fg: tuple[float, float, float],
    bg: tuple[float, float, float],
    threshold: float,
) -> tuple[float, float, float]:
    """Find the nearest color to `fg` that passes contrast against `bg`.

    Adjusts lightness in HSL space via binary search while preserving
    hue and saturation. Returns the original color if it already passes.
    """
    if contrast_ratio(fg, bg) >= threshold:
        return fg

    h, l, s = colorsys.rgb_to_hls(*fg)
    bg_lum = relative_luminance(*bg)

    # Determine direction: if bg is light, darken fg; if bg is dark, lighten fg
    darken = bg_lum > 0.5

    lo, hi = 0.0, 1.0
    best_l = 0.0 if darken else 1.0

    for _ in range(64):
        mid = (lo + hi) / 2.0
        candidate = colorsys.hls_to_rgb(h, mid, s)
        ratio = contrast_ratio(candidate, bg)

        if ratio >= threshold:
            best_l = mid
            if darken:
                hi = mid  # try less dark
            else:
                lo = mid  # try less light
        else:
            if darken:
                lo = mid  # need darker
            else:
                hi = mid  # need lighter

    return colorsys.hls_to_rgb(h, best_l, s)
