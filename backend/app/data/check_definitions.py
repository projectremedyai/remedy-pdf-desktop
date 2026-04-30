"""Static definitions for all PDF accessibility check rules."""

from __future__ import annotations

from typing import Any

# 33 checker rules + 14 screen reader rules
CHECK_DEFINITIONS: dict[str, dict[str, Any]] = {
    # === Document checks ===
    "doc-accessibility-permission": {
        "category": "Document",
        "description": "Encryption allows assistive technology access",
        "detail": "Verifies that PDF security settings don't block screen readers and other assistive technologies from accessing document content.",
        "wcag": ["4.1.2"],
        "fixable": True,
    },
    "doc-not-image-only": {
        "category": "Document",
        "description": "Document has selectable text (not image-only)",
        "detail": "Checks that the PDF contains actual text content, not just scanned images. Image-only PDFs require OCR to become accessible.",
        "wcag": ["1.4.5"],
        "fixable": False,
    },
    "doc-tagged": {
        "category": "Document",
        "description": "Document has PDF structure tree (tagged PDF)",
        "detail": "Verifies the PDF has a structure tree that defines the logical reading order, headings, paragraphs, tables, etc. Untagged PDFs are not accessible.",
        "wcag": ["1.3.1", "1.3.2"],
        "fixable": True,
    },
    "doc-reading-order": {
        "category": "Document",
        "description": "Structure provides logical reading order",
        "detail": "Checks that the tag tree ordering matches the expected visual reading order. Common source of failures — often a source-document limitation.",
        "wcag": ["1.3.2"],
        "fixable": False,
    },
    "doc-struct-tree-integrity": {
        "category": "Document",
        "description": "Structure tree is internally consistent",
        "detail": "Verifies that structure elements and ParentTree entries are reachable and correctly linked. This is a supplemental integrity check beyond the core Adobe-equivalent set.",
        "wcag": ["1.3.1", "4.1.1"],
        "fixable": True,
    },
    "doc-language": {
        "category": "Document",
        "description": "Document language is specified",
        "detail": "Verifies the /Lang entry is set on the document (e.g., 'en-US'). Screen readers need this to select the correct pronunciation engine.",
        "wcag": ["3.1.1"],
        "fixable": True,
    },
    "doc-display-title": {
        "category": "Document",
        "description": "PDF viewer displays document title (not filename)",
        "detail": "Checks that ViewerPreferences/DisplayDocTitle is true, so the title bar shows a meaningful title instead of the filename.",
        "wcag": ["2.4.2"],
        "fixable": True,
    },
    "doc-bookmarks": {
        "category": "Document",
        "description": "Document has bookmarks (required for 21+ pages)",
        "detail": "For documents with 21 or more pages, bookmarks (outlines) are required to provide navigation. Shorter docs get N/A.",
        "wcag": ["2.4.5"],
        "fixable": True,
    },
    "doc-color-contrast": {
        "category": "Document",
        "description": "Color contrast meets WCAG AA (4.5:1 for text)",
        "detail": "Checks that text has sufficient color contrast against its background. Minimum 4.5:1 for normal text, 3:1 for large text.",
        "wcag": ["1.4.3"],
        "fixable": False,
    },
    "doc-use-of-color": {
        "category": "Document",
        "description": "Color is not the only means of conveying information",
        "detail": "Checks whether information conveyed with color also has a non-color cue such as text, shape, iconography, pattern, or labels.",
        "wcag": ["1.4.1"],
        "fixable": False,
    },
    # === Page Content checks ===
    "page-content-tagged": {
        "category": "Page Content",
        "description": "All page content is tagged",
        "detail": "Every content element on every page must be inside a structure tag or marked as an artifact. Untagged content is invisible to screen readers.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "page-annotations-tagged": {
        "category": "Page Content",
        "description": "All annotations (links, forms) are tagged",
        "detail": "Link annotations, form fields, and other interactive elements must be wrapped in structure tags with appropriate types (Link, Form, Widget).",
        "wcag": ["4.1.2"],
        "fixable": True,
    },
    "page-tab-order": {
        "category": "Page Content",
        "description": "Tab order follows structure order",
        "detail": "Checks that each page has /Tabs set to /S, ensuring keyboard tab navigation follows the logical structure tree order.",
        "wcag": ["2.1.1", "2.4.3"],
        "fixable": True,
    },
    "page-char-encoding": {
        "category": "Page Content",
        "description": "Font character encoding is reliable",
        "detail": "Verifies that fonts have proper ToUnicode mappings. Source-font limitation — if the original fonts lack proper encoding, this cannot be fixed without re-creating the document.",
        "wcag": ["4.1.1"],
        "fixable": False,
    },
    "page-multimedia-tagged": {
        "category": "Page Content",
        "description": "Multimedia objects are tagged",
        "detail": "Any embedded video or audio elements must be properly tagged.",
        "wcag": ["1.2.1"],
        "fixable": True,
    },
    "page-no-flicker": {
        "category": "Page Content",
        "description": "No screen flicker effects",
        "detail": "Checks that the page doesn't contain elements that flash or flicker at rates that could cause seizures.",
        "wcag": ["2.3.1"],
        "fixable": True,
    },
    "page-no-scripts": {
        "category": "Page Content",
        "description": "No inaccessible scripts",
        "detail": "JavaScript actions in the PDF must not create accessibility barriers.",
        "wcag": ["4.1.2"],
        "fixable": False,
    },
    "page-no-repetitive-links": {
        "category": "Page Content",
        "description": "No repetitive navigation links",
        "detail": "Navigation links should not be unnecessarily duplicated across pages.",
        "wcag": ["2.4.1"],
        "fixable": True,
    },
    "page-no-timed-responses": {
        "category": "Page Content",
        "description": "No timed interactions",
        "detail": "The document must not require timed responses from the user.",
        "wcag": ["2.2.1"],
        "fixable": True,
    },
    # === Forms, Tables, Lists ===
    "forms-fields-tagged": {
        "category": "Forms Tables Lists",
        "description": "Form fields are tagged",
        "detail": "All form fields (text inputs, checkboxes, dropdowns) must be inside Form or Widget structure tags.",
        "wcag": ["4.1.2"],
        "fixable": True,
    },
    "forms-fields-description": {
        "category": "Forms Tables Lists",
        "description": "Form fields have descriptions",
        "detail": "Each form field must have a tooltip (/TU) or alternate description so screen readers can announce what the field is for.",
        "wcag": ["4.1.2", "1.3.1"],
        "fixable": True,
    },
    "tables-tr-parent": {
        "category": "Forms Tables Lists",
        "description": "TR must be child of Table/THead/TBody/TFoot",
        "detail": "Table row elements must be properly nested inside table container elements, not floating as direct children of other elements.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "tables-th-td-parent": {
        "category": "Forms Tables Lists",
        "description": "TH and TD must be children of TR",
        "detail": "Table header and data cells must be inside table rows. Cells directly under Table (skipping TR) break table semantics.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "tables-headers": {
        "category": "Forms Tables Lists",
        "description": "Tables must have header rows",
        "detail": "Data tables need at least one row of TH (header) cells so screen readers can announce column names when reading data cells.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "tables-regularity": {
        "category": "Forms Tables Lists",
        "description": "Tables have consistent column/row counts",
        "detail": "Each row should have the same number of columns (accounting for colspan/rowspan). Irregular tables confuse screen readers.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "tables-summary": {
        "category": "Forms Tables Lists",
        "description": "Tables have summaries",
        "detail": "Complex data tables should have a /Summary attribute or /Alt text describing the table's purpose and structure.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "lists-li-parent": {
        "category": "Forms Tables Lists",
        "description": "LI must be child of L (list)",
        "detail": "List item elements must be nested inside a List container. Orphaned LI elements break list navigation in screen readers.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "lists-lbl-lbody-parent": {
        "category": "Forms Tables Lists",
        "description": "Lbl and LBody must be children of LI",
        "detail": "List label (bullet/number) and body content must be inside a list item. Improperly nested list parts break reading order.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    # === Alt Text & Headings ===
    "alt-figures": {
        "category": "Alt Text Headings",
        "description": "Figures require alternative text",
        "detail": "All Figure elements must have /Alt text describing the image. This is the single most impactful accessibility requirement for visual content.",
        "wcag": ["1.1.1"],
        "fixable": True,
    },
    "alt-redundant": {
        "category": "Alt Text Headings",
        "description": "No redundant alternative text",
        "detail": "Alt text should not duplicate adjacent text content. Screen reader users would hear the same information twice.",
        "wcag": ["1.1.1"],
        "fixable": True,
    },
    "alt-associated-content": {
        "category": "Alt Text Headings",
        "description": "Alt text is associated with correct content",
        "detail": "Alternative text must be linked to the correct image or element, not orphaned or misattached.",
        "wcag": ["1.1.1"],
        "fixable": True,
    },
    "alt-hides-annotation": {
        "category": "Alt Text Headings",
        "description": "Alt text doesn't hide annotations",
        "detail": "Alternative text should not obscure or replace interactive annotations (links, form fields).",
        "wcag": ["1.1.1"],
        "fixable": True,
    },
    "alt-elements": {
        "category": "Alt Text Headings",
        "description": "Other elements require alternative text",
        "detail": "Non-figure elements that serve as images or decorative content must also have appropriate alt text or be marked as artifacts.",
        "wcag": ["1.1.1"],
        "fixable": True,
    },
    "headings-nesting": {
        "category": "Alt Text Headings",
        "description": "Proper heading nesting (H1 -> H2 -> H3)",
        "detail": "Heading levels must not skip (e.g., H1 -> H3 without H2). Screen reader users rely on heading hierarchy for document navigation.",
        "wcag": ["1.3.1", "2.4.6"],
        "fixable": True,
    },
    # === Screen Reader validation checks ===
    "sr-no-tags": {
        "category": "Screen Reader",
        "description": "Document has no structure tree",
        "detail": "The PDF has zero structure tags — it is completely inaccessible to screen readers. Needs full re-tagging.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "sr-no-lang": {
        "category": "Screen Reader",
        "description": "Document language not specified",
        "detail": "The /Lang entry is missing. Screen readers cannot determine which pronunciation rules to use.",
        "wcag": ["3.1.1"],
        "fixable": True,
    },
    "sr-no-headings": {
        "category": "Screen Reader",
        "description": "No heading elements in document",
        "detail": "The document has no H1-H6 heading tags. Screen reader users cannot navigate by headings.",
        "wcag": ["2.4.6"],
        "fixable": True,
    },
    "sr-heading-start": {
        "category": "Screen Reader",
        "description": "First heading is not H1",
        "detail": "The first heading in the document should be H1. Starting with H2 or lower breaks the expected hierarchy.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "sr-heading-skip": {
        "category": "Screen Reader",
        "description": "Skipped heading levels",
        "detail": "Heading levels jump (e.g., H1 to H3 without H2). Screen reader users may think content is missing.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "sr-figure-no-alt": {
        "category": "Screen Reader",
        "description": "Figure has no alt text",
        "detail": "A Figure tag exists but has no /Alt text. Screen readers will announce 'image' with no description.",
        "wcag": ["1.1.1"],
        "fixable": True,
    },
    "sr-figure-generic-alt": {
        "category": "Screen Reader",
        "description": "Figure alt text is generic or placeholder",
        "detail": "A Figure tag has placeholder alt text such as 'image' or 'figure'. Screen readers will announce it, but it provides no usable description.",
        "wcag": ["1.1.1"],
        "fixable": True,
    },
    "sr-figure-short-alt": {
        "category": "Screen Reader",
        "description": "Figure alt text is very short (< 5 chars)",
        "detail": "The alt text exists but is likely too short to be meaningful (e.g., 'img' or 'pic').",
        "wcag": ["1.1.1"],
        "fixable": True,
    },
    "sr-table-no-headers": {
        "category": "Screen Reader",
        "description": "Table has no header cells",
        "detail": "A Table element has no TH cells. Screen readers cannot announce column names when reading table data.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "sr-table-empty-cell": {
        "category": "Screen Reader",
        "description": "Table has empty cells",
        "detail": "One or more table cells have no text content. May indicate a table structure issue or missing data.",
        "wcag": ["1.3.1"],
        "fixable": False,
    },
    "sr-list-no-items": {
        "category": "Screen Reader",
        "description": "List has no LI children",
        "detail": "An L (List) element exists but contains no LI children. The list is announced but has no items to read.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "sr-empty-element": {
        "category": "Screen Reader",
        "description": "Tagged element is empty",
        "detail": "A structure element (heading, paragraph, etc.) has no text content. Screen readers may announce the tag type with no content.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
    "sr-repeated-content": {
        "category": "Screen Reader",
        "description": "Content appears multiple times in reading order",
        "detail": "The same text appears more than once in the structure tree, causing screen readers to read duplicated content.",
        "wcag": ["1.3.2"],
        "fixable": True,
    },
    "sr-untagged-page": {
        "category": "Screen Reader",
        "description": "Page has no tagged content",
        "detail": "An entire page has no structure tags. All content on this page is invisible to screen readers.",
        "wcag": ["1.3.1"],
        "fixable": True,
    },
}


def get_check_details(rule_id: str) -> dict[str, Any] | None:
    """Look up a check rule by ID."""
    defn = CHECK_DEFINITIONS.get(rule_id)
    if defn is None:
        return None
    return {"rule_id": rule_id, **defn}


def list_all_checks() -> list[dict[str, str]]:
    """Return all check IDs grouped by category."""
    result = []
    for rule_id, defn in CHECK_DEFINITIONS.items():
        result.append({
            "rule_id": rule_id,
            "category": defn["category"],
            "description": defn["description"],
        })
    return result
