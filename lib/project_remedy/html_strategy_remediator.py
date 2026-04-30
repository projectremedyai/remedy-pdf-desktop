"""Pre-LLM surgical HTML accessibility fixes using BeautifulSoup.

Applies deterministic, programmatic fixes for common WCAG 2.1 AA
violations BEFORE the HTML is sent to the LLM for complex remediation.
This reduces LLM cycles, improves consistency, and handles the
mechanical fixes that do not require semantic understanding.

Each strategy method operates on a BeautifulSoup tree in place and
returns a list of human-readable descriptions of fixes applied.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup, Comment, Doctype, NavigableString, Tag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GENERIC_LINK_TEXTS = frozenset({
    "click here",
    "here",
    "read more",
    "more",
    "learn more",
    "link",
    "this link",
    "go",
    "details",
    "more details",
    "more info",
    "more information",
    "info",
    "download",
    "download here",
    "view",
    "view more",
    "see more",
    "continue",
    "continue reading",
    "full article",
    "full story",
})

_VIEWPORT_CONTENT = "width=device-width, initial-scale=1"

_SKIP_LINK_STYLE = (
    "position:absolute;left:-9999px;top:auto;width:1px;height:1px;"
    "overflow:hidden;z-index:10000;"
)

_SKIP_LINK_FOCUS_STYLE = (
    "position:fixed;left:0;top:0;width:auto;height:auto;"
    "overflow:visible;background:#fff;color:#000;"
    "padding:8px 16px;font-size:1rem;z-index:10000;"
    "outline:2px solid #005fcc;"
)

# Heading tag names in order
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


# ---------------------------------------------------------------------------
# Contrast helpers (simplified)
# ---------------------------------------------------------------------------

def _parse_css_color(value: str) -> tuple[int, int, int] | None:
    """Parse a hex or rgb() color string to an (r, g, b) tuple.

    Returns None for values that cannot be parsed (named colours,
    hsl, currentColor, etc.) — those are left for the LLM.
    """
    value = value.strip().lower()

    # #rrggbb or #rgb
    hex_match = re.match(r"^#([0-9a-f]{3,8})$", value)
    if hex_match:
        h = hex_match.group(1)
        if len(h) == 3:
            return int(h[0] * 2, 16), int(h[1] * 2, 16), int(h[2] * 2, 16)
        if len(h) >= 6:
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    # rgb(r, g, b) or rgba(r, g, b, a)
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


def _relative_luminance(r: int, g: int, b: int) -> float:
    """Calculate WCAG relative luminance from sRGB values."""
    components = []
    for c in (r, g, b):
        s = c / 255.0
        components.append(s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4)
    return 0.2126 * components[0] + 0.7152 * components[1] + 0.0722 * components[2]


def _contrast_ratio(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    """Return the WCAG contrast ratio between two RGB colours."""
    l1 = _relative_luminance(*rgb1)
    l2 = _relative_luminance(*rgb2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# HTMLStrategyRemediator
# ---------------------------------------------------------------------------


class HTMLStrategyRemediator:
    """Applies deterministic, pre-LLM HTML accessibility fixes.

    Each strategy targets a specific class of WCAG 2.1 AA violation
    that can be resolved mechanically without semantic understanding.
    The LLM is reserved for complex remediation that requires context.

    Usage::

        remediator = HTMLStrategyRemediator()
        fixed_html, fixes = remediator.remediate(raw_html)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remediate(self, html: str) -> tuple[str, list[str]]:
        """Apply all programmatic accessibility fixes to *html*.

        Parameters
        ----------
        html:
            Raw HTML string (may be a fragment or full document).

        Returns
        -------
        tuple[str, list[str]]
            A 2-tuple of (fixed_html, list_of_fix_descriptions).
        """
        soup = BeautifulSoup(html, "html.parser")
        all_fixes: list[str] = []

        # Run strategies in logical order: document structure first,
        # then landmarks, headings, links, images, tables, forms,
        # colour, and finally figures.
        strategies = [
            # Document structure
            self._fix_missing_lang,
            self._fix_missing_title,
            self._fix_missing_viewport,
            # Landmarks
            self._fix_missing_main_landmark,
            self._fix_missing_skip_link,
            self._fix_missing_nav_landmark,
            # Headings
            self._fix_empty_headings,
            self._fix_skipped_heading_levels,
            # Links
            self._fix_empty_links,
            self._fix_generic_link_text,
            self._fix_new_window_links,
            # Images
            self._fix_missing_alt_text,
            self._fix_empty_alt_on_linked_images,
            # Tables
            self._fix_missing_table_headers,
            self._fix_missing_table_scope,
            self._fix_missing_table_caption,
            # Forms
            self._fix_missing_form_labels,
            # Colour / contrast
            self._fix_insufficient_contrast,
            # Figures
            self._fix_improper_figures,
        ]

        for strategy in strategies:
            try:
                fixes = strategy(soup)
                if fixes:
                    all_fixes.extend(fixes)
            except Exception:
                logger.exception(
                    "Strategy %s raised an exception; skipping",
                    strategy.__name__,
                )

        # Doctype must be handled on the string level because
        # BeautifulSoup's html.parser does not round-trip it reliably.
        fixed_html = str(soup)
        doctype_fixes = self._fix_missing_doctype_str(fixed_html)
        if doctype_fixes:
            fixed_html = doctype_fixes[0]
            all_fixes.extend(doctype_fixes[1])

        if all_fixes:
            logger.info(
                "HTMLStrategyRemediator applied %d fix(es): %s",
                len(all_fixes),
                "; ".join(all_fixes),
            )
        else:
            logger.debug("HTMLStrategyRemediator: no fixes needed")

        return fixed_html, all_fixes

    # ------------------------------------------------------------------
    # Document structure strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_missing_doctype_str(html: str) -> tuple[str, list[str]] | None:
        """Ensure the HTML string starts with ``<!DOCTYPE html>``.

        Operates on the raw string because BeautifulSoup's html.parser
        does not reliably preserve or inject the doctype declaration.
        """
        stripped = html.lstrip()
        if stripped.lower().startswith("<!doctype"):
            return None
        return (f"<!DOCTYPE html>\n{html}", ["Added missing <!DOCTYPE html> declaration"])

    @staticmethod
    def _fix_missing_lang(soup: BeautifulSoup) -> list[str]:
        """Add ``lang="en"`` to the ``<html>`` element if missing."""
        fixes: list[str] = []
        html_tag = soup.find("html")
        if html_tag is None:
            # No <html> wrapper — nothing we can safely do here;
            # the fragment may be intentional.
            return fixes
        if isinstance(html_tag, Tag) and not html_tag.get("lang"):
            html_tag["lang"] = "en"
            fixes.append('Added lang="en" to <html> element')
            logger.debug("Added lang attribute to <html>")
        return fixes

    @staticmethod
    def _fix_missing_title(soup: BeautifulSoup) -> list[str]:
        """Add a ``<title>`` element inside ``<head>`` if missing.

        Uses the text of the first ``<h1>`` as the title, falling
        back to ``"Document"`` when no heading is available.
        """
        fixes: list[str] = []
        head = soup.find("head")
        if head is None:
            return fixes  # Fragment or headless document; skip.

        if not isinstance(head, Tag):
            return fixes

        existing_title = head.find("title")
        if existing_title is not None:
            # Title exists — check if it is empty.
            if isinstance(existing_title, Tag) and not existing_title.get_text(strip=True):
                h1 = soup.find("h1")
                title_text = h1.get_text(strip=True) if h1 else "Document"
                existing_title.string = title_text
                fixes.append(f"Populated empty <title> with \"{title_text}\"")
            return fixes

        # No <title> at all — create one.
        h1 = soup.find("h1")
        title_text = h1.get_text(strip=True) if h1 else "Document"
        title_tag = soup.new_tag("title")
        title_tag.string = title_text
        head.append(title_tag)
        fixes.append(f"Added <title> element: \"{title_text}\"")
        logger.debug("Injected <title>: %s", title_text)
        return fixes

    @staticmethod
    def _fix_missing_viewport(soup: BeautifulSoup) -> list[str]:
        """Add a viewport ``<meta>`` tag if missing."""
        fixes: list[str] = []
        head = soup.find("head")
        if head is None or not isinstance(head, Tag):
            return fixes

        viewport = head.find("meta", attrs={"name": "viewport"})
        if viewport is not None:
            return fixes

        meta = soup.new_tag("meta", attrs={
            "name": "viewport",
            "content": _VIEWPORT_CONTENT,
        })
        head.append(meta)
        fixes.append("Added viewport <meta> tag")
        logger.debug("Injected viewport meta")
        return fixes

    # ------------------------------------------------------------------
    # Landmark strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_missing_main_landmark(soup: BeautifulSoup) -> list[str]:
        """Wrap body content in ``<main>`` if no ``<main>`` exists."""
        fixes: list[str] = []
        if soup.find("main") is not None:
            return fixes

        body = soup.find("body")
        if body is None or not isinstance(body, Tag):
            return fixes

        # Collect direct children of <body> (skip <header>, <footer>,
        # <nav>, <script>, <style> — they belong outside <main>).
        outside_main_tags = {"header", "footer", "nav", "script", "style", "link"}
        children_to_wrap: list[Any] = []
        for child in list(body.children):
            if isinstance(child, Tag) and child.name in outside_main_tags:
                continue
            children_to_wrap.append(child)

        if not children_to_wrap:
            return fixes

        main_tag = soup.new_tag("main", id="main-content")

        # Insert <main> at the position of the first wrapped child.
        first_child = children_to_wrap[0]
        first_child.insert_before(main_tag)

        for child in children_to_wrap:
            main_tag.append(child.extract())

        fixes.append("Wrapped body content in <main> landmark")
        logger.debug("Injected <main> landmark")
        return fixes

    @staticmethod
    def _fix_missing_skip_link(soup: BeautifulSoup) -> list[str]:
        """Add a skip-to-main-content link if one is not present."""
        fixes: list[str] = []

        # Check for an existing skip link (href="#main*" or class*="skip").
        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href", "")
            classes = " ".join(a_tag.get("class", []))
            text = a_tag.get_text(strip=True).lower()
            if (
                href.startswith("#main")
                or "skip" in classes.lower()
                or "skip" in text
            ):
                return fixes

        # Determine the target id.
        main_tag = soup.find("main")
        target_id = "main-content"
        if main_tag is not None and isinstance(main_tag, Tag):
            existing_id = main_tag.get("id")
            if existing_id:
                target_id = existing_id
            else:
                main_tag["id"] = target_id

        body = soup.find("body")
        if body is None or not isinstance(body, Tag):
            return fixes

        skip_link = soup.new_tag(
            "a",
            href=f"#{target_id}",
            attrs={
                "class": "skip-link",
                "style": _SKIP_LINK_STYLE,
                "onfocus": f"this.style.cssText='{_SKIP_LINK_FOCUS_STYLE}'",
                "onblur": f"this.style.cssText='{_SKIP_LINK_STYLE}'",
            },
        )
        skip_link.string = "Skip to main content"
        body.insert(0, skip_link)

        fixes.append("Added skip-to-main-content link")
        logger.debug("Injected skip link targeting #%s", target_id)
        return fixes

    @staticmethod
    def _fix_missing_nav_landmark(soup: BeautifulSoup) -> list[str]:
        """Wrap navigation-like ``<ul>``/``<ol>`` lists in ``<nav>``.

        Heuristic: a list whose every ``<li>`` child contains only an
        ``<a>`` tag (and optional whitespace) is likely navigation.
        Only wraps lists that are direct children of ``<body>``,
        ``<header>``, or ``<footer>`` to avoid false positives.
        """
        fixes: list[str] = []
        nav_parents = {"body", "header", "footer", "div", "[document]"}

        for list_tag in soup.find_all(["ul", "ol"]):
            if not isinstance(list_tag, Tag):
                continue

            parent = list_tag.parent
            if parent is None or not isinstance(parent, Tag):
                continue

            # Skip if already inside a <nav>.
            if parent.name == "nav" or list_tag.find_parent("nav"):
                continue

            # Only consider lists in plausible navigation containers.
            if parent.name not in nav_parents:
                continue

            # Check if every <li> contains exactly one <a>.
            li_items = list_tag.find_all("li", recursive=False)
            if len(li_items) < 2:
                continue  # Too few items to be navigation.

            is_nav = True
            for li in li_items:
                if not isinstance(li, Tag):
                    is_nav = False
                    break
                # Get non-whitespace direct children.
                significant_children = [
                    c for c in li.children
                    if not (isinstance(c, NavigableString) and not c.strip())
                ]
                if len(significant_children) != 1:
                    is_nav = False
                    break
                child = significant_children[0]
                if not isinstance(child, Tag) or child.name != "a":
                    is_nav = False
                    break

            if not is_nav:
                continue

            nav_tag = soup.new_tag("nav", attrs={"aria-label": "Navigation"})
            list_tag.insert_before(nav_tag)
            nav_tag.append(list_tag.extract())
            fixes.append("Wrapped navigation list in <nav> landmark")
            logger.debug("Wrapped <ul>/<ol> in <nav>")

        return fixes

    # ------------------------------------------------------------------
    # Heading strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_empty_headings(soup: BeautifulSoup) -> list[str]:
        """Remove heading elements that have no text content."""
        fixes: list[str] = []
        for heading in soup.find_all(_HEADING_TAGS):
            if not isinstance(heading, Tag):
                continue
            if not heading.get_text(strip=True):
                tag_name = heading.name
                heading.decompose()
                fixes.append(f"Removed empty <{tag_name}> element")
                logger.debug("Removed empty <%s>", tag_name)
        return fixes

    @staticmethod
    def _fix_skipped_heading_levels(soup: BeautifulSoup) -> list[str]:
        """Adjust heading levels so they do not skip (e.g. h1 -> h3).

        Walks all headings in document order and demotes any that skip
        a level so the hierarchy is contiguous.  Only adjusts downward
        (deeper) skips; it does not change headings that move back up
        to a shallower level.
        """
        fixes: list[str] = []
        all_headings = soup.find_all(_HEADING_TAGS)
        if not all_headings:
            return fixes

        # Build a level map: current tag -> desired level.
        max_allowed = 1  # Next heading must be <= this level.

        for heading in all_headings:
            if not isinstance(heading, Tag):
                continue
            current_level = int(heading.name[1])

            if current_level > max_allowed:
                new_level = max_allowed
                old_name = heading.name
                heading.name = f"h{new_level}"
                fixes.append(
                    f"Changed <{old_name}> to <h{new_level}> "
                    f"(fixed skipped heading level)"
                )
                logger.debug("Adjusted <%s> -> <h%d>", old_name, new_level)
                max_allowed = new_level + 1
            else:
                max_allowed = current_level + 1

        return fixes

    # ------------------------------------------------------------------
    # Link strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_empty_links(soup: BeautifulSoup) -> list[str]:
        """Add ``aria-label`` to links that have no visible text."""
        fixes: list[str] = []
        for a_tag in soup.find_all("a"):
            if not isinstance(a_tag, Tag):
                continue

            # Skip if there is already an aria-label or title.
            if a_tag.get("aria-label") or a_tag.get("title"):
                continue

            visible_text = a_tag.get_text(strip=True)
            if visible_text:
                continue

            # Check for image with alt text inside the link.
            img = a_tag.find("img")
            if img and isinstance(img, Tag) and img.get("alt"):
                continue

            # Derive a label from the href.
            href = a_tag.get("href", "")
            if href and href != "#":
                # Use the last meaningful path segment.
                segments = [s for s in href.rstrip("/").split("/") if s]
                label = segments[-1] if segments else "Link"
                # Clean up file extensions and hyphens/underscores.
                label = re.sub(r"\.[a-zA-Z0-9]+$", "", label)
                label = label.replace("-", " ").replace("_", " ").strip()
                if not label:
                    label = "Link"
            else:
                label = "Link"

            a_tag["aria-label"] = label
            fixes.append(f"Added aria-label=\"{label}\" to empty link")
            logger.debug("Added aria-label to empty <a>")

        return fixes

    @staticmethod
    def _fix_generic_link_text(soup: BeautifulSoup) -> list[str]:
        """Enhance links whose text is generic (e.g. "click here").

        Attempts to pull context from the surrounding paragraph or
        sentence to build a more descriptive ``aria-label``.
        """
        fixes: list[str] = []
        for a_tag in soup.find_all("a"):
            if not isinstance(a_tag, Tag):
                continue
            if a_tag.get("aria-label"):
                continue

            text = a_tag.get_text(strip=True).lower()
            if text not in _GENERIC_LINK_TEXTS:
                continue

            # Try to build context from the parent element's text.
            parent = a_tag.parent
            context_text = ""
            if parent and isinstance(parent, Tag):
                full_text = parent.get_text(separator=" ", strip=True)
                # Remove the generic text itself.
                context_text = full_text.replace(a_tag.get_text(strip=True), "").strip()
                # Truncate to something reasonable.
                if len(context_text) > 80:
                    context_text = context_text[:77].rsplit(" ", 1)[0] + "..."

            if context_text:
                label = context_text
            else:
                # Fall back to href-derived label.
                href = a_tag.get("href", "")
                segments = [s for s in href.rstrip("/").split("/") if s]
                label = segments[-1] if segments else ""
                label = re.sub(r"\.[a-zA-Z0-9]+$", "", label)
                label = label.replace("-", " ").replace("_", " ").strip()

            if label:
                a_tag["aria-label"] = label
                fixes.append(
                    f"Enhanced generic link text \"{text}\" with "
                    f"aria-label=\"{label}\""
                )
                logger.debug("Enhanced generic link: %s -> %s", text, label)

        return fixes

    @staticmethod
    def _fix_new_window_links(soup: BeautifulSoup) -> list[str]:
        """Append ``(opens in new window)`` to ``target="_blank"`` links.

        Also adds ``rel="noopener noreferrer"`` for security.
        """
        fixes: list[str] = []
        for a_tag in soup.find_all("a", attrs={"target": "_blank"}):
            if not isinstance(a_tag, Tag):
                continue

            # Ensure rel="noopener noreferrer".
            existing_rel = a_tag.get("rel", [])
            if isinstance(existing_rel, str):
                existing_rel = existing_rel.split()
            rel_set = set(existing_rel)
            rel_set.update(["noopener", "noreferrer"])
            a_tag["rel"] = list(rel_set)

            # Check if the warning text already exists.
            full_text = a_tag.get_text(strip=True)
            if "opens in new window" in full_text.lower():
                continue
            # Also check aria-label.
            aria = a_tag.get("aria-label", "")
            if "opens in new window" in aria.lower():
                continue

            # Append a visually-hidden indicator.
            sr_span = soup.new_tag(
                "span",
                attrs={
                    "class": "sr-only",
                    "style": (
                        "position:absolute;width:1px;height:1px;"
                        "padding:0;margin:-1px;overflow:hidden;"
                        "clip:rect(0,0,0,0);border:0;"
                    ),
                },
            )
            sr_span.string = " (opens in new window)"
            a_tag.append(sr_span)
            fixes.append(
                f"Added '(opens in new window)' indicator to "
                f"target=\"_blank\" link"
            )
            logger.debug("Added new-window indicator to <a>")

        return fixes

    # ------------------------------------------------------------------
    # Image strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_missing_alt_text(soup: BeautifulSoup) -> list[str]:
        """Add ``alt=""`` to images missing the ``alt`` attribute.

        Marks them as decorative; the LLM can later provide real alt
        text if the image is meaningful.
        """
        fixes: list[str] = []
        for img in soup.find_all("img"):
            if not isinstance(img, Tag):
                continue
            if img.has_attr("alt"):
                continue
            img["alt"] = ""
            src = img.get("src", "unknown")
            fixes.append(f"Added alt=\"\" (decorative) to <img src=\"{src}\">")
            logger.debug("Added empty alt to <img src=%s>", src)
        return fixes

    @staticmethod
    def _fix_empty_alt_on_linked_images(soup: BeautifulSoup) -> list[str]:
        """Use link text for images inside ``<a>`` that have empty alt.

        When an ``<img>`` with ``alt=""`` is the only content of an
        ``<a>`` tag, screen readers skip it entirely.  If the link has
        meaningful surrounding text we use that; otherwise we derive
        a label from the href.
        """
        fixes: list[str] = []
        for a_tag in soup.find_all("a"):
            if not isinstance(a_tag, Tag):
                continue

            img = a_tag.find("img")
            if img is None or not isinstance(img, Tag):
                continue

            alt = img.get("alt")
            if alt is None or alt.strip():
                continue  # alt is missing (handled elsewhere) or non-empty.

            # Check if the link has other text content.
            link_text = a_tag.get_text(strip=True)
            if link_text:
                img["alt"] = link_text
                fixes.append(
                    f"Set linked image alt text to \"{link_text}\""
                )
                logger.debug("Set linked image alt from link text: %s", link_text)
                continue

            # Derive from href.
            href = a_tag.get("href", "")
            segments = [s for s in href.rstrip("/").split("/") if s]
            label = segments[-1] if segments else ""
            label = re.sub(r"\.[a-zA-Z0-9]+$", "", label)
            label = label.replace("-", " ").replace("_", " ").strip()
            if label:
                img["alt"] = label
                fixes.append(f"Set linked image alt text to \"{label}\"")
                logger.debug("Set linked image alt from href: %s", label)

        return fixes

    # ------------------------------------------------------------------
    # Table strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_missing_table_headers(soup: BeautifulSoup) -> list[str]:
        """Promote the first row to ``<th>`` if no ``<th>`` exists."""
        fixes: list[str] = []
        for table in soup.find_all("table"):
            if not isinstance(table, Tag):
                continue

            if table.find("th"):
                continue  # Already has header cells.

            # Find the first row (in <thead> or first <tr>).
            first_row = table.find("tr")
            if first_row is None or not isinstance(first_row, Tag):
                continue

            cells = first_row.find_all("td", recursive=False)
            if not cells:
                continue

            # Check that the first row looks like a header: it should
            # not contain block-level elements or be obviously data.
            promoted = False
            for td in cells:
                if not isinstance(td, Tag):
                    continue
                td.name = "th"
                td["scope"] = "col"
                promoted = True

            if promoted:
                # Wrap in <thead> if not already.
                if not table.find("thead"):
                    thead = soup.new_tag("thead")
                    first_row.insert_before(thead)
                    thead.append(first_row.extract())

                    # Wrap remaining rows in <tbody> if not already.
                    remaining_rows = table.find_all("tr", recursive=False)
                    if remaining_rows and not table.find("tbody"):
                        tbody = soup.new_tag("tbody")
                        remaining_rows[0].insert_before(tbody)
                        for row in remaining_rows:
                            tbody.append(row.extract())

                fixes.append("Promoted first table row to <th> header cells")
                logger.debug("Promoted first <tr> to header row")

        return fixes

    @staticmethod
    def _fix_missing_table_scope(soup: BeautifulSoup) -> list[str]:
        """Add ``scope="col"`` or ``scope="row"`` to ``<th>`` elements."""
        fixes: list[str] = []
        for th in soup.find_all("th"):
            if not isinstance(th, Tag):
                continue
            if th.get("scope"):
                continue

            # Determine scope based on position.
            parent_row = th.parent
            if parent_row is None or not isinstance(parent_row, Tag):
                th["scope"] = "col"
                fixes.append("Added scope=\"col\" to <th>")
                continue

            # If the <th> is inside <thead>, it is a column header.
            thead = th.find_parent("thead")
            if thead:
                th["scope"] = "col"
                fixes.append("Added scope=\"col\" to <th>")
                continue

            # If the <th> is the first cell in its row, it is a row header.
            first_cell = parent_row.find(["th", "td"])
            if first_cell is th:
                th["scope"] = "row"
                fixes.append("Added scope=\"row\" to <th>")
            else:
                th["scope"] = "col"
                fixes.append("Added scope=\"col\" to <th>")

            logger.debug("Added scope to <th>")

        return fixes

    @staticmethod
    def _fix_missing_table_caption(soup: BeautifulSoup) -> list[str]:
        """Add an empty ``<caption>`` placeholder if missing."""
        fixes: list[str] = []
        for table in soup.find_all("table"):
            if not isinstance(table, Tag):
                continue

            # Skip tables that already have a caption or aria-label.
            if table.find("caption"):
                continue
            if table.get("aria-label") or table.get("aria-labelledby"):
                continue

            # Skip layout tables (role="presentation" or role="none").
            role = table.get("role", "").lower()
            if role in ("presentation", "none"):
                continue

            caption = soup.new_tag("caption")
            caption.string = "Data table"
            table.insert(0, caption)
            fixes.append("Added <caption> placeholder to table")
            logger.debug("Injected <caption> placeholder")

        return fixes

    # ------------------------------------------------------------------
    # Form strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_missing_form_labels(soup: BeautifulSoup) -> list[str]:
        """Associate ``<label>`` elements with inputs or add ``aria-label``.

        Targets ``<input>``, ``<select>``, and ``<textarea>`` elements
        that have neither an associated ``<label>`` nor an ``aria-label``.
        """
        fixes: list[str] = []
        labelable = ["input", "select", "textarea"]

        for element in soup.find_all(labelable):
            if not isinstance(element, Tag):
                continue

            # Skip hidden or submit/button inputs.
            input_type = (element.get("type") or "").lower()
            if input_type in ("hidden", "submit", "button", "reset", "image"):
                continue

            # Check for existing labelling.
            if element.get("aria-label") or element.get("aria-labelledby"):
                continue
            if element.get("title"):
                continue

            element_id = element.get("id")
            if element_id:
                # Look for a <label for="..."> pointing at this element.
                label = soup.find("label", attrs={"for": element_id})
                if label:
                    continue

            # Check if the element is wrapped in a <label>.
            parent_label = element.find_parent("label")
            if parent_label:
                continue

            # Derive a label from name, placeholder, or id.
            name = element.get("name", "")
            placeholder = element.get("placeholder", "")
            label_text = placeholder or name or element_id or element.name

            if label_text:
                # Clean up the label text.
                label_text = label_text.replace("-", " ").replace("_", " ").strip()
                label_text = label_text.title()

            element["aria-label"] = label_text or "Input"
            fixes.append(
                f"Added aria-label=\"{label_text or 'Input'}\" to "
                f"<{element.name}> element"
            )
            logger.debug("Added aria-label to <%s>", element.name)

        return fixes

    # ------------------------------------------------------------------
    # Colour / contrast strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_insufficient_contrast(soup: BeautifulSoup) -> list[str]:
        """Remove inline colour styles that create low contrast.

        This is a simplified approach: we parse inline ``color`` and
        ``background-color`` on the same element.  If the contrast
        ratio is below 4.5:1 (WCAG AA for normal text), we strip both
        colour declarations so the element inherits accessible defaults.

        Named CSS colours, external stylesheets, and inherited colours
        are left for the LLM or manual review.
        """
        fixes: list[str] = []

        for tag in soup.find_all(style=True):
            if not isinstance(tag, Tag):
                continue

            style = tag.get("style", "")
            if not style:
                continue

            # Parse inline color and background-color.
            fg_match = re.search(r"(?:^|;)\s*color\s*:\s*([^;]+)", style, re.IGNORECASE)
            bg_match = re.search(
                r"background(?:-color)?\s*:\s*([^;]+)", style, re.IGNORECASE
            )

            if not fg_match or not bg_match:
                continue

            fg_rgb = _parse_css_color(fg_match.group(1))
            bg_rgb = _parse_css_color(bg_match.group(1))

            if fg_rgb is None or bg_rgb is None:
                continue  # Cannot parse — leave for LLM.

            ratio = _contrast_ratio(fg_rgb, bg_rgb)
            if ratio >= 4.5:
                continue  # Passes WCAG AA.

            # Strip the problematic colour declarations.
            cleaned = re.sub(
                r"(?:^|;)\s*color\s*:\s*[^;]+;?", "", style, flags=re.IGNORECASE
            )
            cleaned = re.sub(
                r"background(?:-color)?\s*:\s*[^;]+;?", "", cleaned, flags=re.IGNORECASE
            )
            cleaned = cleaned.strip().strip(";").strip()

            if cleaned:
                tag["style"] = cleaned
            else:
                del tag["style"]

            fixes.append(
                f"Removed low-contrast inline styles "
                f"(ratio {ratio:.1f}:1, requires 4.5:1)"
            )
            logger.debug(
                "Stripped low-contrast inline styles (ratio %.1f:1)",
                ratio,
            )

        return fixes

    # ------------------------------------------------------------------
    # Figure strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_improper_figures(soup: BeautifulSoup) -> list[str]:
        """Wrap lone ``<img>`` + adjacent caption text in ``<figure>``.

        Looks for patterns where an ``<img>`` is followed by a sibling
        element that looks like a caption (short ``<p>``, ``<span>``,
        or ``<em>`` with text containing "figure" or "image" keywords,
        or any element with a ``caption`` class).
        """
        fixes: list[str] = []
        caption_keywords = {"figure", "fig.", "fig ", "image", "photo", "illustration"}
        caption_classes = {"caption", "wp-caption-text", "figure-caption"}

        for img in soup.find_all("img"):
            if not isinstance(img, Tag):
                continue

            # Skip if already inside a <figure>.
            if img.find_parent("figure"):
                continue

            # Look at the next sibling.
            next_sib = img.next_sibling
            # Skip whitespace-only text nodes.
            while isinstance(next_sib, NavigableString) and not next_sib.strip():
                next_sib = next_sib.next_sibling

            if next_sib is None or not isinstance(next_sib, Tag):
                continue

            # Check if the sibling looks like a caption.
            sib_text = next_sib.get_text(strip=True).lower()
            sib_classes = set(c.lower() for c in next_sib.get("class", []))
            is_caption = False

            if sib_classes & caption_classes:
                is_caption = True
            elif next_sib.name in ("p", "span", "em", "small", "div"):
                if any(kw in sib_text for kw in caption_keywords):
                    is_caption = True
                elif len(sib_text) < 120 and sib_text:
                    # Short text immediately after an image is likely a caption.
                    # Only wrap if the text is reasonably short.
                    pass  # Conservative: only keyword/class matches.

            if not is_caption:
                continue

            figure = soup.new_tag("figure")
            img.insert_before(figure)
            figure.append(img.extract())

            figcaption = soup.new_tag("figcaption")
            figcaption.append(next_sib.extract())
            figure.append(figcaption)

            fixes.append("Wrapped <img> and caption in <figure>/<figcaption>")
            logger.debug("Created <figure> with <figcaption>")

        return fixes
