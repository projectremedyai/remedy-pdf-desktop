"""LLM-powered accessibility remediation strategies for HTML documents.

Adapted from the ASU CIC PDF Accessibility repository's remediation
strategies.  Each strategy class targets a specific WCAG 2.1 AA
violation category that benefits from LLM reasoning (semantic analysis,
alt-text generation, context-aware fixes).

These strategies complement the deterministic fixes in
:mod:`html_strategy_remediator` — they are intended to run *after* the
mechanical pre-LLM pass, handling the cases that require contextual
understanding or vision capabilities.

All strategies share a common async interface::

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client,       # Ollama-compatible client
        **context,
    ) -> BeautifulSoup

The ``llm_client`` must expose at minimum:
    - ``async chat(messages=[...]) -> str``
    - ``async vision(image_path=..., prompt=...) -> str``

Usage::

    from project_remedy.accessibility_strategies import RemediationStrategyRunner

    runner = RemediationStrategyRunner()
    soup, report = await runner.run(html_string, llm_client)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional, Union

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Type alias matching pipeline.py convention.
LLMClient = Any


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_dark_color(color: str) -> bool:
    """Determine if a CSS color value is dark (perceived brightness < 0.5).

    Handles hex (#RGB, #RRGGBB), rgb()/rgba(), and a small set of named
    colours.  Returns False for values it cannot parse.
    """
    color = color.strip().lower()

    r = g = b = 0

    if color.startswith("#"):
        if len(color) == 4:
            r = int(color[1] * 2, 16)
            g = int(color[2] * 2, 16)
            b = int(color[3] * 2, 16)
        elif len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
        else:
            return False
    elif color.startswith("rgb"):
        m = re.search(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", color)
        if m:
            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            return False
    elif color in {
        "black", "darkblue", "darkgreen", "darkred",
        "navy", "purple", "brown", "maroon",
    }:
        return True
    else:
        return False

    brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return brightness < 0.5


def _relative_luminance(r: int, g: int, b: int) -> float:
    """WCAG relative luminance from sRGB values."""
    components = []
    for c in (r, g, b):
        s = c / 255.0
        components.append(
            s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * components[0] + 0.7152 * components[1] + 0.0722 * components[2]


def _contrast_ratio(
    rgb1: tuple[int, int, int],
    rgb2: tuple[int, int, int],
) -> float:
    """WCAG contrast ratio between two RGB colours."""
    l1 = _relative_luminance(*rgb1)
    l2 = _relative_luminance(*rgb2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _parse_css_color(value: str) -> tuple[int, int, int] | None:
    """Parse hex or rgb() to (r, g, b).  Returns None on failure."""
    value = value.strip().lower()

    hex_match = re.match(r"^#([0-9a-f]{3,8})$", value)
    if hex_match:
        h = hex_match.group(1)
        if len(h) == 3:
            return int(h[0] * 2, 16), int(h[1] * 2, 16), int(h[2] * 2, 16)
        if len(h) >= 6:
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    rgb_match = re.match(
        r"^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})", value
    )
    if rgb_match:
        return (
            int(rgb_match.group(1)),
            int(rgb_match.group(2)),
            int(rgb_match.group(3)),
        )

    return None


def _is_decorative_image(img: Tag) -> bool:
    """Heuristic check for whether an image is decorative."""
    if img.get("role") in ("presentation", "none"):
        return True

    css_classes = img.get("class", [])
    if css_classes:
        decorative_patterns = {
            "decorative", "icon", "bullet", "separator", "spacer", "bg",
        }
        joined = " ".join(css_classes).lower()
        if any(p in joined for p in decorative_patterns):
            return True

    width = img.get("width")
    height = img.get("height")
    if width and height:
        try:
            if int(width) <= 16 and int(height) <= 16:
                return True
        except (ValueError, TypeError):
            pass

    parent = img.parent
    if parent and isinstance(parent, Tag) and parent.name == "div":
        parent_classes = " ".join(parent.get("class", []))
        if any(p in parent_classes.lower() for p in ("banner", "header", "logo", "icon")):
            return True

    return False


def _normalize_header_text(text: str) -> str:
    """Normalize header text for reliable matching."""
    if not text:
        return ""
    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9 ]", "", normalized)
    return normalized.strip()


def _fuzzy_match_header(
    ai_headers: dict[str, str],
    actual_header: str,
    threshold: float = 0.6,
) -> tuple[str | None, str | None]:
    """Find the best fuzzy match for a header in the AI response."""
    normalized_actual = _normalize_header_text(actual_header)

    # Try exact normalized match first.
    for ai_text, scope in ai_headers.items():
        if _normalize_header_text(ai_text) == normalized_actual:
            return ai_text, scope

    # Fuzzy matching.
    best_match = None
    best_ratio = 0.0
    best_scope = None

    for ai_text, scope in ai_headers.items():
        normalized_ai = _normalize_header_text(ai_text)
        if not normalized_actual or not normalized_ai:
            continue
        ratio = SequenceMatcher(None, normalized_actual, normalized_ai).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = ai_text
            best_scope = scope

    if best_ratio >= threshold:
        return best_match, best_scope

    return None, None


def _infer_scope_from_position(table: Tag, th: Tag) -> str:
    """Determine scope='col' or scope='row' from the header's position."""
    parent_row = th.parent
    if not parent_row or parent_row.name != "tr":
        return "col"

    # Headers inside <thead> are column headers.
    if th.parent.parent and th.parent.parent.name == "thead":
        return "col"

    all_rows = table.find_all("tr")
    row_index = list(all_rows).index(parent_row) if parent_row in all_rows else -1
    cell_index = (
        list(parent_row.find_all(["th", "td"])).index(th)
        if th in parent_row.find_all(["th", "td"])
        else -1
    )

    if row_index == 0:
        return "col"
    elif cell_index == 0:
        return "row"

    if (
        row_index == 1
        and table.find("thead")
        and th.parent in table.find("thead").find_all("tr")
    ):
        return "col"

    return "col"


def _find_common_prefix(strings: list[str]) -> str:
    """Find common text prefix among a list of strings."""
    if not strings:
        return ""
    strings = [s.lower() for s in strings]
    prefix = strings[0]
    for s in strings[1:]:
        while s[:len(prefix)] != prefix and prefix:
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix if len(prefix) > 3 else ""


# ---------------------------------------------------------------------------
# Strategy report dataclass
# ---------------------------------------------------------------------------

@dataclass
class StrategyReport:
    """Summary of all remediation strategies that were applied."""

    strategy_name: str
    fixes_applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.fixes_applied) > 0 and len(self.errors) == 0


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------

class BaseRemediationStrategy:
    """Abstract base for all remediation strategies."""

    name: str = "base"

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        **context: Any,
    ) -> BeautifulSoup:
        """Apply the remediation strategy to the soup.

        Parameters
        ----------
        soup:
            The parsed HTML document.
        llm_client:
            An Ollama-compatible client instance with ``chat()``
            and ``vision()`` methods.
        **context:
            Extra context (e.g. ``image_dir``, ``document_title``).

        Returns
        -------
        BeautifulSoup
            The modified soup (same object, mutated in place).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ColorContrastRemediation
# ---------------------------------------------------------------------------

class ColorContrastRemediation(BaseRemediationStrategy):
    """Fix insufficient colour contrast ratios (WCAG 2.1 SC 1.4.3).

    Scans elements with inline ``color`` / ``background-color`` styles.
    When the contrast ratio is below 4.5:1 (AA for normal text), it
    adjusts the foreground or background colour to meet the threshold.

    Adapted from ASU CIC ``color_contrast_remediation.py``.
    """

    name = "color_contrast"

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        **context: Any,
    ) -> BeautifulSoup:
        fixes: list[str] = []

        for tag in soup.find_all(style=True):
            if not isinstance(tag, Tag):
                continue

            style = tag.get("style", "")
            if not style:
                continue

            bg_match = re.search(r"background-color:\s*([^;]+)", style)
            fg_match = re.search(r"(?:^|;)\s*color:\s*([^;]+)", style)

            if bg_match and fg_match:
                bg_rgb = _parse_css_color(bg_match.group(1))
                fg_rgb = _parse_css_color(fg_match.group(1))

                if bg_rgb and fg_rgb:
                    ratio = _contrast_ratio(fg_rgb, bg_rgb)
                    if ratio < 4.5:
                        # Determine whether to change fg or bg.
                        is_dark_bg = _is_dark_color(bg_match.group(1))
                        new_color = "#FFFFFF" if is_dark_bg else "#000000"
                        style = re.sub(
                            r"(?:^|;)\s*color:\s*[^;]+",
                            f"; color: {new_color}",
                            style,
                        )
                        tag["style"] = style.lstrip("; ")
                        fixes.append(
                            f"Adjusted text color to {new_color} "
                            f"(was ratio {ratio:.1f}:1, requires 4.5:1)"
                        )
                        continue

            # Element with only background-color — skip (no fg to compare).
            if bg_match and not fg_match:
                bg_color = bg_match.group(1).strip()
                is_dark_bg = _is_dark_color(bg_color)
                # Only add fg if the bg is actually set and there is no fg.
                new_color = "#FFFFFF" if is_dark_bg else "#000000"
                if "color:" not in style:
                    tag["style"] = style.rstrip(";") + f"; color: {new_color}"
                    fixes.append(
                        f"Added text color {new_color} to element with "
                        f"background-color: {bg_color}"
                    )
                continue

            # Element with only color — check against white background assumption.
            if fg_match and not bg_match:
                fg_rgb = _parse_css_color(fg_match.group(1))
                if fg_rgb:
                    ratio = _contrast_ratio(fg_rgb, (255, 255, 255))
                    if ratio < 4.5:
                        tag["style"] = re.sub(
                            r"(?:^|;)\s*color:\s*[^;]+",
                            "; color: #000000",
                            style,
                        ).lstrip("; ")
                        fixes.append(
                            f"Changed low-contrast text color to #000000 "
                            f"(was ratio {ratio:.1f}:1)"
                        )

        if fixes:
            logger.info(
                "ColorContrastRemediation: %d fix(es) applied", len(fixes)
            )
        return soup


# ---------------------------------------------------------------------------
# DocumentStructureRemediation
# ---------------------------------------------------------------------------

class DocumentStructureRemediation(BaseRemediationStrategy):
    """Add missing document-level structure: lang, title, skip-nav.

    Adapted from ASU CIC ``document_structure_remediation.py``.
    WCAG 2.1 SC 2.4.1 (skip nav), SC 3.1.1 (lang), SC 2.4.2 (title).
    """

    name = "document_structure"

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        **context: Any,
    ) -> BeautifulSoup:
        self._fix_missing_language(soup)
        self._fix_missing_title(soup, llm_client)
        self._fix_missing_skip_link(soup)
        return soup

    # -- lang attribute --

    @staticmethod
    def _fix_missing_language(soup: BeautifulSoup) -> None:
        html_tag = soup.find("html")
        if html_tag and isinstance(html_tag, Tag) and not html_tag.get("lang"):
            html_tag["lang"] = "en"
            logger.debug("DocumentStructure: added lang='en'")

    # -- title --

    @staticmethod
    def _fix_missing_title(soup: BeautifulSoup, llm_client: LLMClient) -> None:
        head = soup.find("head")
        if not head or not isinstance(head, Tag):
            return

        existing = head.find("title")
        if existing and isinstance(existing, Tag) and existing.get_text(strip=True):
            return

        title_text = "Document Title"
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title_text = h1.get_text(strip=True)
        else:
            meta = soup.find("meta", attrs={"name": "description"})
            if meta and meta.get("content"):
                title_text = meta["content"].strip()[:60]

        if existing and isinstance(existing, Tag):
            existing.string = title_text
        else:
            title_tag = soup.new_tag("title")
            title_tag.string = title_text
            head.append(title_tag)

        logger.debug("DocumentStructure: set title to '%s'", title_text)

    # -- skip link --

    @staticmethod
    def _fix_missing_skip_link(soup: BeautifulSoup) -> None:
        # Check for existing skip links.
        for a_tag in soup.find_all("a"):
            href = (a_tag.get("href") or "").lower()
            text = a_tag.get_text(strip=True).lower()
            if href.startswith("#main") or "skip" in text:
                return

        main = soup.find("main") or soup.find(attrs={"role": "main"})
        if main and isinstance(main, Tag):
            if not main.get("id"):
                main["id"] = "main-content"
            target_id = main["id"]
        else:
            target_id = "main-content"

        body = soup.find("body")
        if not body or not isinstance(body, Tag):
            return

        skip_link = soup.new_tag("a")
        skip_link["href"] = f"#{target_id}"
        skip_link["class"] = "skip-link"
        skip_link["style"] = (
            "position:absolute;top:-40px;left:0;background:#000;"
            "color:#fff;padding:8px;z-index:100;transition:top 0.3s;"
        )
        skip_link.string = "Skip to main content"
        body.insert(0, skip_link)

        # Add focus styles.
        head = soup.find("head")
        if head and isinstance(head, Tag):
            skip_style_exists = False
            for style_tag in head.find_all("style"):
                if ".skip-link" in (style_tag.string or ""):
                    skip_style_exists = True
                    break
            if not skip_style_exists:
                style_tag = soup.new_tag("style")
                style_tag.string = (
                    ".skip-link:focus{top:0;}"
                )
                head.append(style_tag)

        logger.debug("DocumentStructure: added skip link")


# ---------------------------------------------------------------------------
# FigureRemediation
# ---------------------------------------------------------------------------

class FigureRemediation(BaseRemediationStrategy):
    """Wrap standalone images in ``<figure>`` / ``<figcaption>``.

    Adapted from ASU CIC ``figure_remediation.py``.
    WCAG 2.1 SC 1.1.1, SC 1.3.1.
    """

    name = "figure"

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        **context: Any,
    ) -> BeautifulSoup:
        fixes: list[str] = []

        for img in list(soup.find_all("img")):
            if not isinstance(img, Tag):
                continue
            if img.find_parent("figure"):
                continue

            caption_text = self._find_caption(img, soup)
            figure = soup.new_tag("figure")
            figure["role"] = "group"
            figure["class"] = figure.get("class", []) + ["content-figure"]

            if img.get("id"):
                figure["id"] = img["id"]
                del img["id"]

            img_parent = img.parent
            img.extract()
            figure.append(img)

            if caption_text:
                figcaption = soup.new_tag("figcaption")
                figcaption.string = caption_text
                figure["aria-label"] = caption_text

                if re.match(r"^(Figure|Fig\.)\s+\d+", caption_text, re.IGNORECASE):
                    figure.insert(0, figcaption)
                else:
                    figure.append(figcaption)

            if img_parent and isinstance(img_parent, Tag):
                img_parent.append(figure)
            else:
                body = soup.find("body")
                if body:
                    body.append(figure)

            fixes.append(
                "Wrapped image in <figure>" +
                (f" with caption: {caption_text[:50]}" if caption_text else "")
            )

        # Add figure CSS if figures were created.
        if fixes:
            self._inject_figure_css(soup)
            logger.info("FigureRemediation: %d fix(es)", len(fixes))

        return soup

    @staticmethod
    def _find_caption(img: Tag, soup: BeautifulSoup) -> str:
        """Attempt to derive a caption from context around the image."""
        # From substantive alt text.
        if img.get("alt") and len(img["alt"]) > 10:
            return img["alt"]

        # From adjacent "Figure N" paragraph.
        next_p = img.find_next("p")
        if next_p:
            p_text = next_p.get_text(strip=True)
            if p_text.startswith(("Figure", "Fig.")):
                next_p.decompose()
                return p_text

        # From title attribute.
        if img.get("title"):
            return img["title"]

        # From descriptive filename.
        if img.get("src"):
            filename = img["src"].split("/")[-1].split(".")[0]
            if len(filename) > 5 and not filename.isdigit():
                words = re.findall(r"[A-Za-z]+", filename)
                if words:
                    return " ".join(w.capitalize() for w in words)

        # From preceding heading.
        prev_heading = img.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
        if prev_heading and prev_heading.get_text(strip=True):
            return prev_heading.get_text(strip=True)

        return ""

    @staticmethod
    def _inject_figure_css(soup: BeautifulSoup) -> None:
        head = soup.find("head")
        if not head or not isinstance(head, Tag):
            return

        for style_tag in head.find_all("style"):
            if ".content-figure" in (style_tag.string or ""):
                return

        css = (
            ".content-figure{margin:1em 0;padding:0.5em;border:1px solid #ddd;}"
            ".content-figure img{max-width:100%;height:auto;}"
            ".content-figure figcaption{margin-top:0.5em;font-style:italic;color:#666;}"
        )
        style_tag = soup.new_tag("style")
        style_tag.string = css
        head.append(style_tag)


# ---------------------------------------------------------------------------
# FormRemediation
# ---------------------------------------------------------------------------

class FormRemediation(BaseRemediationStrategy):
    """Add labels, fieldsets, and ARIA attributes to form controls.

    Adapted from ASU CIC ``form_remediation.py``.
    WCAG 2.1 SC 1.3.1, SC 3.3.2, SC 4.1.2.
    """

    name = "form"

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        **context: Any,
    ) -> BeautifulSoup:
        self._fix_missing_labels(soup)
        self._fix_missing_fieldsets(soup)
        self._fix_required_indicators(soup)
        return soup

    @staticmethod
    def _fix_missing_labels(soup: BeautifulSoup) -> None:
        """Add <label> elements for unlabelled form controls."""
        fixes_count = 0
        for control in soup.find_all(["input", "select", "textarea"]):
            if not isinstance(control, Tag):
                continue

            input_type = (control.get("type") or "").lower()
            if input_type in ("hidden", "submit", "button", "reset", "image"):
                continue

            # Already labelled?
            if control.get("aria-label") or control.get("aria-labelledby"):
                continue

            control_id = control.get("id")
            if control_id:
                existing_label = soup.find("label", attrs={"for": control_id})
                if existing_label:
                    continue

            if control.find_parent("label"):
                continue

            # Generate an ID if needed.
            if not control_id:
                control_type = control.name
                input_type_str = (
                    control.get("type", "text")
                    if control_type == "input"
                    else control_type
                )
                siblings = len(list(control.find_previous_siblings(control_type)))
                control_id = f"{input_type_str}-{siblings + 1}"
                control["id"] = control_id

            # Derive label text.
            label_text = "Label"
            if control.get("placeholder"):
                label_text = control["placeholder"]
            elif control.get("name"):
                name = control["name"]
                label_text = " ".join(
                    w.capitalize() for w in re.split(r"[_\-]", name)
                )
            elif control.get("type"):
                label_text = control["type"].capitalize()

            label_tag = soup.new_tag("label")
            label_tag["for"] = control_id
            label_tag.string = label_text
            control.insert_before(label_tag)
            fixes_count += 1

        if fixes_count:
            logger.info("FormRemediation: added %d label(s)", fixes_count)

    @staticmethod
    def _fix_missing_fieldsets(soup: BeautifulSoup) -> None:
        """Wrap related form controls in <fieldset>/<legend>."""
        for form in soup.find_all("form"):
            if not isinstance(form, Tag):
                continue
            if form.find("fieldset"):
                continue

            # Group inputs by common name prefix.
            all_inputs = form.find_all(["input", "select", "textarea"])
            name_groups: dict[str, list[Tag]] = {}
            for inp in all_inputs:
                name = inp.get("name", "")
                if name and "_" in name:
                    prefix = name.split("_")[0]
                    name_groups.setdefault(prefix, []).append(inp)

            added = 0
            for prefix, inputs in name_groups.items():
                if len(inputs) < 2:
                    continue

                parent = inputs[0].parent
                if not parent or not isinstance(parent, Tag):
                    continue

                fieldset = soup.new_tag("fieldset")
                legend = soup.new_tag("legend")
                legend.string = prefix.capitalize()
                fieldset.append(legend)

                content = parent.decode_contents()
                parent.clear()
                fieldset.append(BeautifulSoup(content, "html.parser"))
                parent.append(fieldset)
                added += 1

            if added:
                logger.debug("FormRemediation: added %d fieldset(s)", added)

    @staticmethod
    def _fix_required_indicators(soup: BeautifulSoup) -> None:
        """Ensure required fields have aria-required and visual indicators."""
        for control in soup.find_all(["input", "select", "textarea"]):
            if not isinstance(control, Tag):
                continue
            if control.get("required") or control.get("aria-required") == "true":
                if not control.get("aria-required"):
                    control["aria-required"] = "true"

                if control.get("id"):
                    label = soup.find("label", attrs={"for": control["id"]})
                    if label and isinstance(label, Tag):
                        if "*" not in label.get_text():
                            if label.string:
                                label.string = f"{label.string} *"
                            else:
                                label.append(" *")


# ---------------------------------------------------------------------------
# HeadingRemediation
# ---------------------------------------------------------------------------

class HeadingRemediation(BaseRemediationStrategy):
    """Fix heading hierarchy: missing h1, skipped levels, empty headings.

    Adapted from ASU CIC ``heading_remediation.py``.
    WCAG 2.1 SC 1.3.1, SC 2.4.6.

    Uses the LLM to generate contextually appropriate heading text
    when headings are empty or when intermediate levels need insertion.
    """

    name = "heading"

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        **context: Any,
    ) -> BeautifulSoup:
        self._fix_missing_h1(soup)
        await self._fix_empty_headings(soup, llm_client)
        self._fix_skipped_levels(soup)
        return soup

    @staticmethod
    def _fix_missing_h1(soup: BeautifulSoup) -> None:
        """Add an <h1> if none exists, deriving text from <title>."""
        if soup.find("h1"):
            return

        title_tag = soup.find("title")
        title_text = (
            title_tag.get_text(strip=True) if title_tag else "Document Title"
        )

        insertion_point = (
            soup.find("header")
            or soup.find("main")
            or soup.find("body")
            or soup
        )

        h1 = soup.new_tag("h1")
        h1.string = title_text

        first_child = next(insertion_point.children, None)
        if first_child:
            first_child.insert_before(h1)
        else:
            insertion_point.append(h1)

        logger.debug("HeadingRemediation: added h1 '%s'", title_text)

    async def _fix_empty_headings(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
    ) -> None:
        """Use the LLM to generate text for empty headings."""
        heading_tags = ("h1", "h2", "h3", "h4", "h5", "h6")
        generic_re = re.compile(
            r"^(heading|title|section|\s*\d+\s*)$", re.IGNORECASE
        )

        for heading in soup.find_all(heading_tags):
            if not isinstance(heading, Tag):
                continue
            text = heading.get_text(strip=True)
            if text and not generic_re.match(text):
                continue

            # Gather surrounding context for the LLM.
            next_p = heading.find_next("p")
            context_text = (
                next_p.get_text(strip=True)[:200] if next_p else ""
            )
            level = heading.name

            if context_text:
                prompt = (
                    f"Generate a short, descriptive heading (<{level}>) for "
                    f"a section whose content begins: \"{context_text}\". "
                    f"Respond with ONLY the heading text, no HTML tags."
                )
                try:
                    new_text = await llm_client.chat(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=60,
                        temperature=0.3,
                    )
                    new_text = new_text.strip().strip('"').strip("'")
                    if new_text:
                        heading.string = new_text
                        logger.debug(
                            "HeadingRemediation: filled empty %s with '%s'",
                            level, new_text,
                        )
                        continue
                except Exception as exc:
                    logger.warning(
                        "LLM heading generation failed: %s", exc
                    )

            # Fallback without LLM.
            level_num = int(heading.name[1])
            if level_num == 1:
                heading.string = "Document Title"
            else:
                siblings = len(list(heading.find_previous_siblings(heading.name)))
                heading.string = f"Section {siblings + 1}"

    @staticmethod
    def _fix_skipped_levels(soup: BeautifulSoup) -> None:
        """Re-level headings so no levels are skipped going deeper.

        Walks all headings in document order and adjusts any that skip
        a level so the hierarchy is contiguous.
        """
        heading_tags = ("h1", "h2", "h3", "h4", "h5", "h6")
        all_headings = soup.find_all(heading_tags)
        if not all_headings:
            return

        max_allowed = 1
        for heading in all_headings:
            if not isinstance(heading, Tag):
                continue
            current_level = int(heading.name[1])
            if current_level > max_allowed:
                old_name = heading.name
                heading.name = f"h{max_allowed}"
                logger.debug(
                    "HeadingRemediation: changed <%s> to <h%d>",
                    old_name, max_allowed,
                )
                max_allowed = max_allowed + 1
            else:
                max_allowed = current_level + 1


# ---------------------------------------------------------------------------
# ImageRemediation
# ---------------------------------------------------------------------------

class ImageRemediation(BaseRemediationStrategy):
    """Generate alt text for images missing it, using LLM vision.

    Adapted from ASU CIC ``image_remediation.py``.
    WCAG 2.1 SC 1.1.1 (non-text content).

    If ``image_dir`` is provided in context, the strategy resolves
    relative ``src`` paths against it and uses the LLM's vision
    capability.  Otherwise it falls back to context-based alt text.
    """

    name = "image"

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        **context: Any,
    ) -> BeautifulSoup:
        image_dir = context.get("image_dir", "")
        fixes: list[str] = []

        for img in soup.find_all("img"):
            if not isinstance(img, Tag):
                continue

            alt = img.get("alt")

            # Case 1: no alt attribute at all.
            if alt is None:
                new_alt = await self._generate_alt(
                    img, soup, llm_client, image_dir
                )
                img["alt"] = new_alt
                fixes.append(f"Added alt text: {new_alt[:60]}")
                continue

            # Case 2: empty alt — decide decorative vs informative.
            if alt == "":
                if _is_decorative_image(img):
                    img["role"] = "presentation"
                    fixes.append("Confirmed decorative image with role=presentation")
                else:
                    new_alt = await self._generate_alt(
                        img, soup, llm_client, image_dir
                    )
                    img["alt"] = new_alt
                    fixes.append(f"Replaced empty alt with: {new_alt[:60]}")
                continue

            # Case 3: generic alt text.
            generic_patterns = {
                "image", "picture", "photo", "graphic", "diagram",
                "chart", "graph", "icon", "img", "pic", "photograph",
                "untitled", "no description",
            }
            if alt.strip().lower() in generic_patterns:
                new_alt = await self._generate_alt(
                    img, soup, llm_client, image_dir
                )
                img["alt"] = new_alt
                fixes.append(
                    f"Replaced generic alt '{alt}' with: {new_alt[:60]}"
                )
                continue

            # Case 4: excessively long alt text (> 125 chars).
            if len(alt) > 125:
                prompt = (
                    f"Condense this alt text to under 125 characters while "
                    f"preserving essential information:\n\n\"{alt}\"\n\n"
                    f"Respond with ONLY the condensed alt text."
                )
                try:
                    condensed = await llm_client.chat(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100,
                        temperature=0.2,
                    )
                    condensed = condensed.strip().strip('"')
                    if condensed and len(condensed) <= 125:
                        img["alt"] = condensed
                        fixes.append(f"Shortened alt from {len(alt)} to {len(condensed)} chars")
                except Exception:
                    logger.warning("Failed to condense long alt text via LLM")

        if fixes:
            logger.info("ImageRemediation: %d fix(es)", len(fixes))
        return soup

    async def _generate_alt(
        self,
        img: Tag,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        image_dir: str,
    ) -> str:
        """Generate alt text using LLM vision or context fallback."""
        src = img.get("src", "")

        # Try vision-based generation if we can resolve the image path.
        if image_dir and src:
            image_path = Path(image_dir) / src
            if not image_path.exists():
                # Try just the filename.
                image_path = Path(image_dir) / Path(src).name
            if image_path.exists():
                try:
                    prompt = (
                        "Generate concise, descriptive alt text for this "
                        "image. The alt text should:\n"
                        "1. Describe the essential content and function\n"
                        "2. Be under 125 characters\n"
                        "3. Not begin with 'image of' or 'picture of'\n"
                        "4. Be suitable for a screen reader\n\n"
                        "Respond with ONLY the alt text, no quotes."
                    )
                    alt_text = await llm_client.vision(
                        image_path=image_path,
                        prompt=prompt,
                    )
                    alt_text = alt_text.strip().strip('"').strip("'")
                    if alt_text:
                        return alt_text[:125]
                except Exception as exc:
                    logger.warning("Vision alt-text generation failed: %s", exc)

        # Fallback: use the LLM chat with context clues.
        context_parts: list[str] = []

        # Nearby text.
        parent = img.parent
        if parent and isinstance(parent, Tag):
            parent_text = parent.get_text(strip=True)
            if parent_text and len(parent_text) < 200:
                context_parts.append(f"Surrounding text: {parent_text}")

        # Preceding heading.
        heading = img.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
        if heading:
            context_parts.append(f"Section heading: {heading.get_text(strip=True)}")

        # Image filename.
        if src:
            filename = Path(src).stem
            context_parts.append(f"Filename: {filename}")

        if context_parts:
            prompt = (
                "Generate concise alt text (under 125 characters) for an "
                "image based on this context:\n" +
                "\n".join(context_parts) +
                "\n\nRespond with ONLY the alt text, no quotes."
            )
            try:
                alt_text = await llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=80,
                    temperature=0.3,
                )
                alt_text = alt_text.strip().strip('"').strip("'")
                if alt_text:
                    return alt_text[:125]
            except Exception:
                logger.warning("LLM context-based alt-text generation failed")

        return "Image description unavailable"


# ---------------------------------------------------------------------------
# LandmarkRemediation
# ---------------------------------------------------------------------------

class LandmarkRemediation(BaseRemediationStrategy):
    """Add ARIA landmarks: main, nav, banner (header), contentinfo (footer).

    Adapted from ASU CIC ``landmark_remediation.py``.
    WCAG 2.1 SC 1.3.1, SC 2.4.1.
    """

    name = "landmark"

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        **context: Any,
    ) -> BeautifulSoup:
        self._fix_missing_main(soup)
        self._fix_missing_header(soup)
        self._fix_missing_footer(soup)
        self._fix_missing_nav(soup)
        return soup

    @staticmethod
    def _fix_missing_main(soup: BeautifulSoup) -> None:
        if soup.find("main") or soup.find(attrs={"role": "main"}):
            return

        body = soup.find("body")
        if not body or not isinstance(body, Tag):
            return

        main = soup.new_tag("main")
        main["id"] = "main-content"
        main["role"] = "main"

        header = soup.find("header") or soup.find(attrs={"role": "banner"})
        nav = soup.find("nav") or soup.find(attrs={"role": "navigation"})
        footer = soup.find("footer") or soup.find(attrs={"role": "contentinfo"})

        outside_tags = {header, nav, footer}
        for child in list(body.children):
            if not isinstance(child, Tag):
                continue
            if child in outside_tags:
                continue
            if child.name in ("script", "style", "noscript"):
                continue
            main.append(child.extract())

        if nav:
            nav.insert_after(main)
        elif header:
            header.insert_after(main)
        else:
            body.insert(0, main)

        logger.debug("LandmarkRemediation: added <main>")

    @staticmethod
    def _fix_missing_header(soup: BeautifulSoup) -> None:
        if soup.find("header") or soup.find(attrs={"role": "banner"}):
            return

        body = soup.find("body")
        if not body or not isinstance(body, Tag):
            return

        header = soup.new_tag("header")
        header["role"] = "banner"

        title_text = "Document Title"
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title_text = title_tag.string

        h1 = soup.find("h1")
        if h1:
            h1.extract()
            header.append(h1)
        else:
            h1 = soup.new_tag("h1")
            h1.string = title_text
            header.append(h1)

        body.insert(0, header)
        logger.debug("LandmarkRemediation: added <header>")

    @staticmethod
    def _fix_missing_footer(soup: BeautifulSoup) -> None:
        if soup.find("footer") or soup.find(attrs={"role": "contentinfo"}):
            return

        body = soup.find("body")
        if not body or not isinstance(body, Tag):
            return

        footer = soup.new_tag("footer")
        footer["role"] = "contentinfo"
        p = soup.new_tag("p")
        p.string = "Document footer"
        footer.append(p)
        body.append(footer)
        logger.debug("LandmarkRemediation: added <footer>")

    @staticmethod
    def _fix_missing_nav(soup: BeautifulSoup) -> None:
        if soup.find("nav") or soup.find(attrs={"role": "navigation"}):
            return

        # Look for existing navigation-like lists.
        nav_list = soup.find(
            "ul",
            class_=lambda c: c and ("nav" in c or "menu" in c),
        )

        if not nav_list:
            # Look for a series of links in the same container.
            links = soup.find_all("a")
            for i in range(len(links) - 1):
                if (
                    links[i].parent == links[i + 1].parent
                    and links[i].parent.name in ("div", "p", "header")
                ):
                    nav_container = links[i].parent
                    nav = soup.new_tag("nav")
                    nav["role"] = "navigation"
                    nav["aria-label"] = "Main navigation"

                    ul = soup.new_tag("ul")
                    for link in nav_container.find_all("a"):
                        li = soup.new_tag("li")
                        li.append(link.extract())
                        ul.append(li)
                    nav.append(ul)

                    header = soup.find("header") or soup.find(attrs={"role": "banner"})
                    if header:
                        header.insert_after(nav)
                    else:
                        body = soup.find("body")
                        if body:
                            body.insert(0, nav)
                    logger.debug("LandmarkRemediation: created <nav> from links")
                    return
            return

        nav = soup.new_tag("nav")
        nav["role"] = "navigation"
        nav["aria-label"] = "Main navigation"
        nav_list.insert_before(nav)
        nav.append(nav_list.extract())
        logger.debug("LandmarkRemediation: wrapped nav list in <nav>")


# ---------------------------------------------------------------------------
# LinkRemediation
# ---------------------------------------------------------------------------

class LinkRemediation(BaseRemediationStrategy):
    """Add descriptive text to ambiguous or empty links.

    Adapted from ASU CIC ``link_remediation.py``.
    WCAG 2.1 SC 2.4.4 (link purpose in context), SC 2.4.9.

    Uses the LLM to generate context-aware link text when the URL
    and surrounding content are available.
    """

    name = "link"

    _GENERIC_TEXTS = frozenset({
        "click here", "here", "read more", "more", "learn more",
        "details", "link", "this link", "go", "more details",
        "more info", "more information", "view", "view more",
        "see more", "continue", "continue reading",
    })

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        **context: Any,
    ) -> BeautifulSoup:
        await self._fix_empty_links(soup, llm_client)
        await self._fix_generic_links(soup, llm_client)
        self._fix_url_as_text(soup)
        self._fix_new_window_no_warning(soup)
        return soup

    async def _fix_empty_links(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
    ) -> None:
        """Add text to links with no visible content."""
        for link in soup.find_all("a"):
            if not isinstance(link, Tag):
                continue
            if link.get_text(strip=True):
                continue
            if link.get("aria-label"):
                continue

            # Check for image with alt inside the link.
            img = link.find("img")
            if img and isinstance(img, Tag) and img.get("alt"):
                continue

            href = link.get("href", "")
            if not href:
                continue

            # Derive text from URL.
            if href.startswith("http"):
                domain_match = re.search(r"https?://(?:www\.)?([^/]+)", href)
                if domain_match:
                    link.string = f"Link to {domain_match.group(1)}"
                    continue

            link.string = f"Link to {href}"

    async def _fix_generic_links(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
    ) -> None:
        """Replace generic link text with context-aware descriptions."""
        for link in soup.find_all("a"):
            if not isinstance(link, Tag):
                continue
            if link.get("aria-label"):
                continue

            text = link.get_text(strip=True).lower()
            if text not in self._GENERIC_TEXTS:
                continue

            href = link.get("href", "")

            # Try context from parent.
            parent = link.parent
            context_text = ""
            if parent and isinstance(parent, Tag) and parent.name != "body":
                full = parent.get_text(strip=True)
                context_text = full.replace(link.get_text(strip=True), "").strip()
                if len(context_text) > 80:
                    context_text = context_text[:77] + "..."

            if context_text:
                # Ask the LLM for a better link label.
                prompt = (
                    f"A link currently says \"{text}\" and is surrounded by "
                    f"this context: \"{context_text}\". The link points to: "
                    f"{href}\n\nGenerate a short, descriptive link text "
                    f"(under 60 characters) that conveys the link's purpose. "
                    f"Respond with ONLY the link text."
                )
                try:
                    new_text = await llm_client.chat(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=40,
                        temperature=0.3,
                    )
                    new_text = new_text.strip().strip('"')
                    if new_text and len(new_text) < 80:
                        link.string = new_text
                        continue
                except Exception:
                    pass

            # Fallback: domain-based text.
            if href.startswith("http"):
                domain_match = re.search(r"https?://(?:www\.)?([^/]+)", href)
                if domain_match:
                    link.string = f"Visit {domain_match.group(1)} website"
                    continue

            # Simple replacements.
            replacements = {
                "click here": "View details",
                "read more": "Read more about this topic",
                "learn more": "Learn more about this topic",
            }
            link.string = replacements.get(text, "View related information")

    @staticmethod
    def _fix_url_as_text(soup: BeautifulSoup) -> None:
        """Replace raw URLs used as link text with domain names."""
        for link in soup.find_all("a"):
            if not isinstance(link, Tag):
                continue
            text = link.get_text(strip=True)
            if not text.startswith(("http://", "https://", "www.")):
                continue

            domain_match = re.search(r"https?://(?:www\.)?([^/]+)", text)
            if domain_match:
                link.string = f"Visit {domain_match.group(1)}"
            else:
                link.string = "Visit website"

    @staticmethod
    def _fix_new_window_no_warning(soup: BeautifulSoup) -> None:
        """Add screen-reader text to links that open in new windows."""
        for link in soup.find_all("a", attrs={"target": "_blank"}):
            if not isinstance(link, Tag):
                continue

            text = link.get_text(strip=True)
            if "new window" in text.lower() or "new tab" in text.lower():
                continue

            sr_span = soup.new_tag("span")
            sr_span["class"] = "sr-only"
            sr_span["style"] = (
                "position:absolute;width:1px;height:1px;"
                "padding:0;margin:-1px;overflow:hidden;"
                "clip:rect(0,0,0,0);border:0;"
            )
            sr_span.string = " (opens in new window)"
            link.append(sr_span)

            if not link.get("title"):
                link["title"] = f"{text} (opens in new window)"

            # Security best practice.
            existing_rel = link.get("rel", [])
            if isinstance(existing_rel, str):
                existing_rel = existing_rel.split()
            rel_set = set(existing_rel)
            rel_set.update(["noopener", "noreferrer"])
            link["rel"] = list(rel_set)


# ---------------------------------------------------------------------------
# TableRemediation
# ---------------------------------------------------------------------------

class TableRemediation(BaseRemediationStrategy):
    """Add scope, headers, captions, thead/tbody to data tables.

    Adapted from ASU CIC ``table_remediation.py``, ``table_detection.py``,
    and ``table_remediation_direct.py``.
    WCAG 2.1 SC 1.3.1 (info and relationships).

    Uses the LLM to determine header scope (col vs row) and identify
    header rows in complex tables.  Falls back to positional inference
    when the LLM is unavailable or returns invalid responses.
    """

    name = "table"

    async def apply(
        self,
        soup: BeautifulSoup,
        llm_client: LLMClient,
        **context: Any,
    ) -> BeautifulSoup:
        tables = soup.find_all("table")
        if not tables:
            return soup

        for table in tables:
            if not isinstance(table, Tag):
                continue

            self._preprocess_table(table, soup)
            self._fix_missing_headers(table)
            await self._fix_missing_scope(table, soup, llm_client)
            self._fix_missing_thead(table, soup)
            self._fix_missing_tbody(table, soup)
            self._fix_missing_caption(table, soup)
            self._fix_header_ids(table, soup)

        logger.info("TableRemediation: processed %d table(s)", len(tables))
        return soup

    # -- Preprocessing --

    @staticmethod
    def _preprocess_table(table: Tag, soup: BeautifulSoup) -> None:
        """Detect header-like first rows and promote td -> th."""
        if table.find("th"):
            return

        rows = table.find_all("tr")
        if not rows:
            return

        first_row = rows[0]
        first_cells = first_row.find_all("td")
        if not first_cells:
            return

        header_indicators = 0
        for cell in first_cells:
            if cell.find(["b", "strong"]):
                header_indicators += 1
            style = cell.get("style", "")
            if "bold" in style.lower() or "font-weight" in style.lower():
                header_indicators += 1
            classes = cell.get("class", [])
            if any("header" in c.lower() for c in classes):
                header_indicators += 1

        if header_indicators > len(first_cells) / 2:
            for td in first_row.find_all("td"):
                td.name = "th"
                td["scope"] = "col"
            logger.debug("TableRemediation: promoted first row cells to <th>")

    # -- Missing headers --

    @staticmethod
    def _fix_missing_headers(table: Tag) -> None:
        """Convert first-row <td> to <th> if table has no headers."""
        if table.find("th"):
            return

        first_row = table.find("tr")
        if not first_row or not isinstance(first_row, Tag):
            return

        for td in first_row.find_all("td"):
            td.name = "th"

    # -- Scope via LLM --

    async def _fix_missing_scope(
        self,
        table: Tag,
        soup: BeautifulSoup,
        llm_client: LLMClient,
    ) -> None:
        """Add scope attributes to <th> elements using LLM analysis."""
        headers = [
            th for th in table.find_all("th")
            if not th.get("scope")
        ]
        if not headers:
            return

        actual_texts = [h.get_text(strip=True) for h in headers]
        header_list = ", ".join(f'"{t}"' for t in actual_texts if t)

        # Truncate table HTML for the prompt if it is very large.
        table_html = str(table)
        if len(table_html) > 4000:
            table_html = table_html[:4000] + "\n... [truncated]"

        prompt = (
            "Analyze this HTML table and determine scope='col' or scope='row' "
            "for each header cell (<th>).\n\n"
            f"Headers: {header_list}\n\n"
            f"Table HTML:\n{table_html}\n\n"
            "Respond ONLY with a JSON object mapping header text to 'col' or 'row'. "
            "Example: {\"Name\": \"col\", \"Category\": \"row\"}"
        )

        table_analysis: dict[str, str] = {}
        try:
            response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1,
            )

            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                table_analysis = json.loads(response[json_start:json_end])
        except Exception as exc:
            logger.warning("LLM table scope analysis failed: %s", exc)

        # Apply scope with AI matches + positional fallback.
        for th in headers:
            header_text = th.get_text(strip=True)

            # Try exact match.
            if header_text and header_text in table_analysis:
                scope_val = table_analysis[header_text]
                if scope_val in ("col", "row"):
                    th["scope"] = scope_val
                    continue

            # Try fuzzy match.
            if header_text and table_analysis:
                matched, scope_val = _fuzzy_match_header(
                    table_analysis, header_text
                )
                if matched and scope_val in ("col", "row"):
                    th["scope"] = scope_val
                    continue

            # Positional fallback.
            th["scope"] = _infer_scope_from_position(table, th)

    # -- thead --

    @staticmethod
    def _fix_missing_thead(table: Tag, soup: BeautifulSoup) -> None:
        """Move header rows into a <thead> element."""
        if table.find("thead"):
            return

        rows = table.find_all("tr")
        if not rows:
            return

        first_row = rows[0]
        if first_row.find("th"):
            thead = soup.new_tag("thead")
            first_row.extract()
            thead.append(first_row)
            table.insert(0, thead)

    # -- tbody --

    @staticmethod
    def _fix_missing_tbody(table: Tag, soup: BeautifulSoup) -> None:
        """Wrap non-header/footer rows in <tbody>."""
        if table.find("tbody"):
            return

        body_rows = [
            row for row in table.find_all("tr")
            if not row.parent or row.parent.name not in ("thead", "tfoot")
        ]
        if not body_rows:
            return

        tbody = soup.new_tag("tbody")
        for row in body_rows:
            row.extract()
            tbody.append(row)

        thead = table.find("thead")
        if thead:
            thead.insert_after(tbody)
        else:
            table.append(tbody)

    # -- caption --

    @staticmethod
    def _fix_missing_caption(table: Tag, soup: BeautifulSoup) -> None:
        """Add a <caption> based on the preceding heading."""
        if table.find("caption"):
            return
        if table.get("aria-label") or table.get("aria-labelledby"):
            return
        if (table.get("role") or "").lower() in ("presentation", "none"):
            return

        caption_text = "Data table"
        prev_heading = table.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
        if prev_heading:
            caption_text = prev_heading.get_text(strip=True)

        caption = soup.new_tag("caption")
        caption.string = caption_text
        table.insert(0, caption)

    # -- header IDs --

    @staticmethod
    def _fix_header_ids(table: Tag, soup: BeautifulSoup) -> None:
        """Add id attributes to <th> and headers attributes to <td>."""
        headers = table.find_all("th")
        if not headers:
            return

        # Assign IDs to headers that lack them.
        for i, th in enumerate(headers):
            if th.get("id"):
                continue
            text = th.get_text(strip=True).lower()
            if text:
                id_text = re.sub(r"[^a-z0-9]+", "-", text)[:20]
                if not id_text or not id_text[0].isalpha():
                    id_text = f"header-{i + 1}"
                th["id"] = f"th-{id_text}"
            else:
                th["id"] = f"th-{i + 1}"

        # Build a position map.
        header_map: dict[tuple[int, int], str] = {}
        for th in headers:
            if not th.get("id"):
                continue
            parent_row = th.parent
            if not parent_row or parent_row.name != "tr":
                continue
            all_rows = table.find_all("tr")
            row_idx = list(all_rows).index(parent_row) if parent_row in all_rows else -1
            cells = list(parent_row.find_all(["th", "td"]))
            col_idx = cells.index(th) if th in cells else -1
            if row_idx >= 0 and col_idx >= 0:
                header_map[(row_idx, col_idx)] = th["id"]

        # Link data cells.
        for row_idx, tr in enumerate(table.find_all("tr")):
            for col_idx, td in enumerate(tr.find_all("td")):
                headers_for_cell = []
                if (0, col_idx) in header_map:
                    headers_for_cell.append(header_map[(0, col_idx)])
                if (row_idx, 0) in header_map:
                    headers_for_cell.append(header_map[(row_idx, 0)])
                if headers_for_cell:
                    td["headers"] = " ".join(headers_for_cell)


# ---------------------------------------------------------------------------
# RemediationStrategyRunner
# ---------------------------------------------------------------------------

class RemediationStrategyRunner:
    """Apply all LLM-powered remediation strategies in sequence.

    Usage::

        runner = RemediationStrategyRunner()
        soup, reports = await runner.run(html_string, llm_client)
        fixed_html = str(soup)

    The strategies are applied in a deliberate order: document
    structure and landmarks first (so later strategies can rely on
    them), then headings, images, links, tables, forms, figures, and
    finally colour contrast.
    """

    def __init__(
        self,
        strategies: list[BaseRemediationStrategy] | None = None,
    ) -> None:
        self.strategies: list[BaseRemediationStrategy] = strategies or [
            DocumentStructureRemediation(),
            LandmarkRemediation(),
            HeadingRemediation(),
            ImageRemediation(),
            LinkRemediation(),
            TableRemediation(),
            FormRemediation(),
            FigureRemediation(),
            ColorContrastRemediation(),
        ]

    async def run(
        self,
        html: str,
        llm_client: LLMClient,
        **context: Any,
    ) -> tuple[BeautifulSoup, list[StrategyReport]]:
        """Run all strategies on the HTML document.

        Parameters
        ----------
        html:
            Raw HTML string.
        llm_client:
            An Ollama-compatible client instance.
        **context:
            Extra context forwarded to each strategy (e.g. ``image_dir``).

        Returns
        -------
        tuple[BeautifulSoup, list[StrategyReport]]
            The modified soup and a list of reports for each strategy.
        """
        soup = BeautifulSoup(html, "html.parser")
        reports: list[StrategyReport] = []

        for strategy in self.strategies:
            report = StrategyReport(strategy_name=strategy.name)
            try:
                logger.info("Running strategy: %s", strategy.name)
                soup = await strategy.apply(soup, llm_client, **context)
                report.fixes_applied.append(f"{strategy.name}: completed")
            except Exception as exc:
                report.errors.append(f"{strategy.name}: {exc}")
                logger.exception(
                    "Strategy %s failed; continuing with remaining strategies",
                    strategy.name,
                )
            reports.append(report)

        return soup, reports
