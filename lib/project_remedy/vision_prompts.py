"""Shared prompt builders for vision-backed document remediation."""

from __future__ import annotations

from typing import Literal

PromptProfile = Literal["cloud", "local"]

_LAYOUT_RULES = """\
Document remediation rules:
- Preserve a meaningful reading sequence. Do not merge unrelated columns, sidebars, callouts, or footer content.
- Keep headings separate from body text and preserve heading hierarchy.
- Keep list structures explicit instead of flattening them into paragraphs.
- Keep tables and directories explicit; do not linearize cell content into prose.
- Keep form prompts, labels, values, and widgets grouped in reading order.
- Treat purely decorative backgrounds, banners, borders, spacers, and watermarks as artifacts, not content.
- If layout intent is ambiguous, say so explicitly instead of guessing.
"""


def ocr_markdown_prompt(
    *,
    profile: PromptProfile,
    page_hint: str = "",
    native_pdf: bool = False,
) -> str:
    hint = f" ({page_hint})" if page_hint else ""
    profile_line = (
        "Use the full PDF context to preserve cross-page structure and page boundaries."
        if native_pdf
        else "Work from the rendered page image and preserve only visible content."
    )
    detail = (
        "Return detailed, region-aware Markdown. Separate pages with <!-- Page N --> comments when applicable."
        if profile == "cloud"
        else "Return compact, faithful Markdown with strict structure and no commentary."
    )
    return (
        f"Extract ALL content from this document{hint} as structured Markdown.\n"
        f"{profile_line}\n"
        f"{_LAYOUT_RULES}\n"
        "Output requirements:\n"
        "- Preserve headings, paragraphs, lists, tables, captions, and form prompts in reading order.\n"
        "- For images, logos, screenshots, charts, or diagrams, output ![brief visual description](IMAGE_PLACEHOLDER).\n"
        "- Do NOT invent URLs, filenames, or missing text.\n"
        "- Keep multi-column pages column-aware.\n"
        "- Keep sidebars and callouts separate from the main article flow.\n"
        f"{detail}"
    )


def language_detection_prompt() -> str:
    return (
        "What language is this document primarily written in? "
        "Return ONLY the ISO 639-1 language code, for example en, es, fr, zh, ko, vi, or tl. "
        "If the document is bilingual, return the primary language."
    )


def title_from_image_prompt() -> str:
    return (
        "Look at this document page and determine the main document title. "
        "Prefer the visually dominant title, not a logo or small section label. "
        "Return ONLY the title text. If there is no clear title, return NONE."
    )


def title_from_text_prompt(text: str) -> str:
    return (
        "Given the beginning of a document, determine the best document title. "
        "Prefer the main title rather than a section heading. "
        "Return ONLY the title text.\n\n"
        f"Document text:\n{text}"
    )


def figure_alt_prompt_retry(*, context: str = "", image_type: str = "") -> str:
    """Stronger prompt for retrying when first attempt returned generic text."""
    context_line = f"Context: {context}\n" if context else ""
    type_examples = {
        "chart": "Bar chart titled 'Annual Budget' showing 15% increase in IT spending",
        "diagram": "Org chart showing Chancellor at top with 9 college presidents reporting",
        "infographic": "Timeline of campus development from 1950-2020 highlighting 5 major construction phases",
        "photograph": "Campus quad with 4 students studying under oak trees, Student Services building visible",
        "": "Screenshot of login page with organization logo and username/password fields",
    }
    example = type_examples.get(image_type, type_examples[""])

    return (
        "Your previous description was too generic. Write SPECIFIC alt text for this image.\n"
        "CRITICAL:\n"
        "- Name exact elements visible: people, objects, text, numbers, locations\n"
        "- Include specific titles, names, or labels visible in the image\n"
        "- Mention quantities, percentages, or key data points if present\n"
        "- BAD: 'A chart showing data' or 'A diagram of a process'\n"
        "- GOOD: " + example + "\n"
        "- Maximum 150 characters\n"
        "- NO 'image of', 'picture of', 'photo of', 'figure showing' prefixes\n"
        f"{context_line}"
        "Return ONLY the specific description, nothing else."
    )


def figure_alt_prompt(*, context: str = "", image_type: str = "") -> str:
    context_line = f"Context: {context}\n" if context else ""
    type_guidance = ""
    if image_type == "chart":
        type_guidance = (
            "This is a DATA CHART or GRAPH.\n"
            "- State the chart type (bar chart, line graph, pie chart, etc.)\n"
            "- Include the title and what is being measured\n"
            "- Mention key trends or the main takeaway\n"
            "Example: 'Bar chart showing enrollment trends 2019-2023, with STEM majors increasing 45%'\n"
        )
    elif image_type == "diagram":
        type_guidance = (
            "This is a DIAGRAM or FLOWCHART.\n"
            "- Describe the process, structure, or system shown\n"
            "- Mention key steps, components, or relationships\n"
            "- Include the diagram title or purpose\n"
            "Example: 'Flowchart of student registration process from application to enrollment'\n"
        )
    elif image_type == "infographic":
        type_guidance = (
            "This is an INFOGRAPHIC.\n"
            "- Summarize the main topic and key points presented\n"
            "- Include statistics or key data if prominent\n"
            "- Describe the visual organization (timeline, comparison, etc.)\n"
        )
    elif image_type == "photograph":
        type_guidance = (
            "This is a PHOTOGRAPH.\n"
            "- Describe the scene, people, or objects visible\n"
            "- Include setting/location context if relevant\n"
            "- Mention any text signs or labels in the image\n"
        )

    return (
        "Write specific, descriptive alt text for this image.\n"
        "CRITICAL RULES:\n"
        "- Maximum 150 characters.\n"
        "- Be SPECIFIC: name what is shown, not generic categories\n"
        "- Do NOT start with 'image of', 'picture of', 'photo of', 'figure showing'\n"
        "- Do NOT use generic phrases like 'a diagram' or 'a chart' without details\n"
        "- Include: what it is, its purpose, and any essential visible text\n"
        "- If decorative (border, spacer, background pattern), return exactly 'Decorative image'\n"
        f"{type_guidance}"
        f"{context_line}"
        "Return ONLY the alt text string, nothing else."
    )


def image_classification_prompt() -> str:
    return (
        "Analyze this image and classify it into exactly one category. "
        "Return ONLY valid JSON with a single key 'category'.\n"
        "Allowed values: photograph, chart, diagram, infographic, decorative.\n"
        "Definitions:\n"
        "- photograph: a real-world photo or simple illustrative graphic\n"
        "- chart: a quantitative data visualization\n"
        "- diagram: a structural or process diagram\n"
        "- infographic: a composite visual mixing text, graphics, and data\n"
        "- decorative: a non-informational ornament, border, divider, or spacer"
    )


def chart_prompt() -> str:
    return (
        "This image contains a chart. Extract the data and return ONLY valid JSON with:\n"
        "- chart_type\n- title\n- x_label\n- y_label\n- legend\n- data\n- summary\n"
        "Be faithful to labels, series names, and values. If data is unreadable, say so in summary."
    )


def diagram_prompt() -> str:
    return (
        "This image contains a diagram. Return ONLY valid JSON with:\n"
        "- diagram_type\n- title\n- description\n- nodes\n- connections\n- summary\n"
        "Capture structure, hierarchy, and directional flow in reading order."
    )


def infographic_prompt() -> str:
    return (
        "This image is a complex infographic. Break it into logical sections and return ONLY valid JSON with:\n"
        "- title\n- sections [{heading, content, data_points?}]\n- summary\n"
        "Keep sections distinct and preserve meaningful visual grouping."
    )


def reading_order_prompt(*, structure_order: str, layout_hint: str = "") -> str:
    hint = f"Layout hint from local analysis: {layout_hint}\n" if layout_hint else ""
    return (
        "You are a PDF accessibility expert. Compare the structure-tree order below against the visual page layout.\n"
        f"{hint}"
        f"Structure tree order:\n{structure_order}\n\n"
        f"{_LAYOUT_RULES}\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "page_layout": "single_column" | "hero_cover" | "brochure_sidebar" | "form_checklist" | "table_directory" | "schedule_grid" | "mixed_graphic_flyer" | "map_infographic" | "report_cover" | "unknown_complex",\n'
        '  "issues": [{"severity": "error" | "warning", "description": "...", "suggestion": "..."}],\n'
        '  "summary": "..."\n'
        "}\n"
        "If the reading order is acceptable, return an empty issues array."
    )


def page_region_analysis_prompt(
    *,
    element_list: str,
    profile: PromptProfile,
) -> str:
    detail = (
        "Return detailed roles, confidence, and whether content should be split into additional regions."
        if profile == "cloud"
        else "Keep the response compact and JSON-only."
    )
    return (
        "Analyze this rendered PDF page for reading order and contrast.\n"
        f"{_LAYOUT_RULES}\n"
        "Current tagged elements on the page:\n"
        f"{element_list}\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "layout_class": "single_column" | "hero_cover" | "brochure_sidebar" | "form_checklist" | "table_directory" | "schedule_grid" | "mixed_graphic_flyer" | "map_infographic" | "report_cover" | "unknown_complex",\n'
        '  "reading_order": [1, 2, 3],\n'
        '  "order_changed": true,\n'
        '  "requires_resegmentation": false,\n'
        '  "contrast_issues": [{"description": "...", "text_rgb": [0,0,0], "bg_rgb": [1,1,1], "fix_rgb": [0,0,0]}],\n'
        '  "notes": "..."\n'
        "}\n"
        f"{detail}"
    )


def semantic_reading_order_prompt(*, element_list: str) -> str:
    """Vision prompt that identifies visual heading hierarchy, sidebar/main
    layout, footer/fine-print, and fragmented list structures."""
    return (
        "You are a PDF accessibility expert analyzing this rendered page.\n"
        f"{_LAYOUT_RULES}\n"
        "The current structure tree tags on this page are:\n"
        f"{element_list}\n\n"
        "Analyze the VISUAL layout and answer:\n"
        "1. **Heading hierarchy**: For every text element that VISUALLY appears to be a heading or title "
        "(larger font, bold, prominent position, section divider), identify what H-level (H1-H6) it "
        "should be based on visual size/weight/prominence. If the current tag is wrong (e.g. body text "
        "tagged as H2, or a footer tagged as H2), flag it.\n"
        "2. **Sidebar vs main**: If the page has a sidebar or secondary column, identify which elements "
        "belong to the sidebar and which to the main content. Main content should read first.\n"
        "3. **Footer/fine-print**: Identify any text that is clearly footer content, fine print, "
        "disclaimers, or page numbers. These should NOT be tagged as headings.\n"
        "4. **Fragmented lists**: Identify consecutive elements that visually form a list (bulleted, "
        "numbered, or consistently formatted items) but are tagged as separate P elements.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "heading_corrections": [\n'
        '    {"element_index": 1, "current_tag": "H2", "correct_tag": "P", '
        '"reason": "visually body text, not a heading"},\n'
        '    {"element_index": 3, "current_tag": "P", "correct_tag": "H2", '
        '"reason": "visually a section heading based on font size"}\n'
        "  ],\n"
        '  "sidebar_elements": [5, 6, 7],\n'
        '  "main_content_elements": [1, 2, 3, 4, 8, 9],\n'
        '  "footer_elements": [10, 11],\n'
        '  "list_groups": [\n'
        '    {"start_index": 3, "end_index": 6, "list_type": "bulleted"}\n'
        "  ],\n"
        '  "reading_order": [1, 2, 3, 4, 8, 9, 5, 6, 7, 10, 11],\n'
        '  "order_changed": true\n'
        "}\n"
        "element_index values are 1-based, matching the element list above.\n"
        "If no corrections are needed, return empty arrays. Be conservative — only flag "
        "clear mismatches between visual appearance and semantic tags."
    )


def contrast_detection_prompt(level: str = "AA") -> str:
    normal = "4.5:1" if level == "AA" else "7.0:1"
    large = "3.0:1" if level == "AA" else "4.5:1"
    return (
        f"Analyze this PDF page image for color contrast issues under WCAG {level}.\n"
        f"{_LAYOUT_RULES}\n"
        "Examine text, image-of-text, form affordances, icons, lines, fills, and borders.\n"
        "Return ONLY valid JSON matching the provided schema.\n"
        f"Thresholds: normal text {normal}, large text {large}, non-text graphics 3.0:1."
    )


def contrast_validation_prompt(level: str, issue_descriptions: str) -> str:
    return (
        f"Verify whether the following contrast issues on this PDF page now pass WCAG {level}.\n"
        f"{issue_descriptions}\n\n"
        "Return ONLY valid JSON matching the provided schema. "
        "Judge text, image-of-text, and non-text graphics against the correct threshold for each case."
    )


def heading_detection_prompt(*, is_first_page: bool = False) -> str:
    first_page_note = (
        "This is the FIRST page — the document title (H1) is likely here.\n"
        if is_first_page else ""
    )
    return (
        "Analyze this document page and identify ALL text that serves as a heading or section title.\n"
        f"{first_page_note}"
        "Rules:\n"
        "- H1: Document title (one per document, usually largest/most prominent text)\n"
        "- H2: Major section breaks (e.g., 'Personal Information', 'Financial Summary', 'Introduction')\n"
        "- H3-H6: Subsection headings nested under their parent sections\n"
        "- Form section dividers and labeled fieldset groups count as headings\n"
        "- Do NOT tag body text, form field labels, page numbers, or repeated header/footer text as headings\n"
        "- Short documents (flyers, notices, calendars, letters) may only have H1 — that is fine\n"
        "- IMPORTANT: If this is the first page and you see no obvious section headings, still identify\n"
        "  the most prominent or largest text as H1. Every document needs at least one heading.\n"
        "  Do NOT return [] for the first page — find the title or main subject.\n"
        "Return ONLY valid JSON array:\n"
        '[{"text": "exact heading text", "level": 1, "y_position": 0.25}]\n'
        "y_position is the vertical position as fraction of page height (0=top, 1=bottom).\n"
        "For pages after the first: if no headings exist on that specific page, return []."
    )


def visual_comparison_prompt() -> str:
    return (
        "Compare the original PDF page images against the remediated HTML image.\n"
        f"{_LAYOUT_RULES}\n"
        "Check image presence, text completeness, table fidelity, and structural preservation. "
        "Return ONLY valid JSON with content_matches, images_match, missing_images, wrong_images, "
        "missing_text, table_issues, structure_issues, and details."
    )


# ---------------------------------------------------------------------------
# WCAG 2.1 AA Vision Verification Prompts (two-tier system)
# ---------------------------------------------------------------------------


def wcag_page_triage_prompt(structural_hints: str) -> str:
    """Tier A: Cheap triage prompt — classify page and decide which checks apply."""
    return (
        "You are verifying one rendered PDF page for WCAG 2.1 AA accessibility compliance.\n\n"
        f"Structural hints from the PDF metadata:\n{structural_hints}\n\n"
        "Tasks:\n"
        "1. Classify the page type (blank, text, mixed, table, form, figure_heavy, scan, cover, unknown)\n"
        "2. Decide which accessibility checks are applicable on this page\n"
        "3. Decide whether the page can be auto-passed, needs focused review, or needs manual review\n"
        "4. List only obvious accessibility issues — do not guess\n\n"
        "Rules:\n"
        "- A continuation page in a long document may legitimately have no new heading\n"
        "- Do not fail contrast unless a region is clearly suspect\n"
        "- Ignore decorative borders, repeated headers/footers, and isolated page numbers\n"
        "- A page with only a page number or boilerplate header/footer can be auto-passed\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "page_type": "text",\n'
        '  "skip": false,\n'
        '  "skip_reason": "",\n'
        '  "applicable_checks": {\n'
        '    "headings": true,\n'
        '    "reading_order": true,\n'
        '    "alt_text_accuracy": false,\n'
        '    "color_contrast": false,\n'
        '    "table_structure": false,\n'
        '    "form_labels": false\n'
        "  },\n"
        '  "focus_queue": ["core_layout"],\n'
        '  "confidence": 0.9\n'
        "}"
    )


def wcag_core_layout_verify_prompt(
    logical_order: str,
    heading_context: str,
) -> str:
    """Tier B: Verify headings, structure, and reading order on a page."""
    return (
        "You are verifying headings and reading order on one PDF page for WCAG 2.1 AA.\n\n"
        f"Current logical reading order from the PDF structure tree:\n{logical_order}\n\n"
        f"Heading context (previous/next page headings):\n{heading_context}\n\n"
        "Tasks:\n"
        "1. Does each visible section that should have a heading actually have one?\n"
        "2. Does the logical reading order match the visual top-to-bottom, left-to-right flow?\n"
        "   Pay special attention to multi-column layouts and sidebars.\n"
        "3. Are there artifacts (page numbers, decorative elements) incorrectly in the reading order?\n"
        "4. Are heading levels appropriate for the visual hierarchy?\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "headings": {\n'
        '    "status": "pass",\n'
        '    "confidence": 0.9,\n'
        '    "findings": []\n'
        "  },\n"
        '  "reading_order": {\n'
        '    "status": "pass",\n'
        '    "confidence": 0.85,\n'
        '    "corrected_order": null,\n'
        '    "findings": []\n'
        "  }\n"
        "}\n"
        "Each finding: {\"issue_id\": \"...\", \"severity\": \"error|warning\", "
        "\"message\": \"...\", \"suggested_fix\": \"...\", \"fixer\": \"fix_function_name\"}"
    )


def heading_hierarchy_quality_prompt(*, logical_order: str) -> str:
    """Verify visual heading hierarchy against current PDF tags."""
    return (
        "You are a PDF accessibility expert verifying heading hierarchy.\n\n"
        "Current tagged reading order. Element numbers are 1-based and must be used for corrections:\n"
        f"{logical_order}\n\n"
        "Use the rendered page image, not just the existing tag sequence. A structurally valid H1/H2/H3 "
        "sequence can still be wrong when visual hierarchy disagrees with the tags.\n\n"
        "Flag clear problems including:\n"
        "- The document/page title or title-like prominent text is tagged as P/Span instead of H1/H2.\n"
        "- A visible section/subsection heading has the wrong level for its visual prominence or nesting.\n"
        "- Heading levels are semantically out of order for the visual page structure, even if they do not skip numerically.\n"
        "- Body text, schedule rows, table rows, labels, page numbers, headers/footers, or fine print are tagged as H1-H6.\n"
        "- A subtitle/byline/field label is over-promoted as a heading.\n\n"
        "Be conservative: only use severity=error when the rendered page and current tag make the correction clear. "
        "Use warning for ambiguous visual hierarchy or multi-page context uncertainty.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "status": "pass" | "fail",\n'
        '  "findings": [\n'
        '    {"severity": "error", "element_index": 4, "current_tag": "H1", '
        '"visible_text": "Feb. 10 Review Syllabus", "message": "Schedule row is tagged as H1", '
        '"correct_tag": "P", "suggested_fix": "Retag as P"}\n'
        "  ]\n"
        "}\n"
        "When a specific tagged element can be safely corrected, include element_index and correct_tag as one of "
        "H1, H2, H3, H4, H5, H6, P, or Span. For a missing heading where no tagged element maps cleanly, "
        "omit element_index and explain the missing title/section heading."
    )


def page_alt_text_quality_prompt(*, figure_list: str) -> str:
    """Verify figure alt text quality on a rendered PDF page."""
    return (
        "You are verifying image alt text quality for one PDF page under WCAG 1.1.1.\n\n"
        "Current Figure tags, approximate page locations, and alt text:\n"
        f"{figure_list}\n\n"
        "Bboxes are normalized [left, top, right, bottom] coordinates on the rendered page. "
        "Use them to match each tagged Figure to the visible image, chart, icon, logo, or decorative mark. "
        "Evaluate every listed Figure. Do not pass alt text just because it exists.\n\n"
        "Fail a Figure when the current alt text is missing, generic, placeholder-like, too vague, "
        "swapped with another Figure, visually inaccurate, hallucinated, misleading, or too verbose to be useful. "
        "For composite cover figures, fail when alt text only repeats logo/title text and omits a substantive "
        "visible photo, map, chart, diagram, or other informative visual in the same figure. "
        "Do not merge nearby real page text, running headers, or title text into a Figure's alt text unless that text "
        "is clearly inside the listed Figure itself; if the figure bbox is unknown, prefer the non-text visual content. "
        "For informative figures, suggested_alt_text must be accurate, specific, concise, and under 180 characters. "
        "For purely decorative figures such as borders, spacers, flourishes, repeated watermarks, or background texture, "
        "return status=fail, decorative=true, issue_type=\"decorative\", and suggested_alt_text=\"\" so the fixer can mark it as an artifact. "
        "Do not fail real text-only content that is not one of the listed Figure tags.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "figures": [\n'
        '    {"figure_index": 1, "status": "pass", "severity": "info", "decorative": false, '
        '"issue_type": "", "message": "", "suggested_alt_text": "", "confidence": 0.93}\n'
        "  ]\n"
        "}\n"
        "Allowed issue_type values: missing, generic, vague, swapped, inaccurate, hallucinated, verbose, decorative, other. "
        "Use status=fail and severity=error only when the visual evidence is clear. "
        "If uncertain, pass the figure and explain nothing."
    )


def wcag_figure_alt_verify_prompt(
    current_alt_text: str,
    nearby_text: str,
) -> str:
    """Tier B: Verify alt text accuracy for a specific figure."""
    return (
        "You are verifying alt text for one figure in a PDF under WCAG 1.1.1 Non-text Content.\n\n"
        "You will see two images:\n"
        "1. The full page (for context)\n"
        "2. The cropped figure\n\n"
        f"Current alt text: \"{current_alt_text}\"\n"
        f"Nearby text context: \"{nearby_text}\"\n\n"
        "Tasks:\n"
        "1. Is this figure informative or decorative?\n"
        "2. If informative, is the current alt text accurate, concise, and non-hallucinatory?\n"
        "3. If it fails, provide a corrected alt text (max 150 chars)\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "status": "pass",\n'
        '  "confidence": 0.9,\n'
        '  "decorative": false,\n'
        '  "failure_reason": "",\n'
        '  "suggested_alt_text": ""\n'
        "}"
    )


def wcag_table_verify_prompt(table_structure: str) -> str:
    """Tier B: Verify table headers and structure."""
    return (
        "You are verifying a data table on a PDF page for WCAG 1.3.1.\n\n"
        f"Current table structure from PDF tags:\n{table_structure}\n\n"
        "Tasks:\n"
        "1. Does the visual table have proper header cells (TH) for every column/row header?\n"
        "2. Is the table structure regular (consistent row/column counts)?\n"
        "3. Do header associations make sense for the data?\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "status": "pass",\n'
        '  "confidence": 0.85,\n'
        '  "findings": []\n'
        "}\n"
        "Each finding: {\"issue_id\": \"...\", \"severity\": \"error|warning\", "
        "\"message\": \"...\", \"fixer\": \"fix_table_headers\"}"
    )


def wcag_contrast_verify_prompt() -> str:
    """Tier B: Verify color contrast on suspect regions."""
    return (
        "You are checking color contrast on a PDF page region for WCAG 1.4.3.\n\n"
        "Minimum contrast ratios:\n"
        "- Normal text (< 18pt or < 14pt bold): 4.5:1\n"
        "- Large text (≥ 18pt or ≥ 14pt bold): 3:1\n\n"
        "Tasks:\n"
        "1. Does the text in this region have sufficient contrast against its background?\n"
        "2. Is any information conveyed by color alone (WCAG 1.4.1)?\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "status": "pass",\n'
        '  "confidence": 0.8,\n'
        '  "findings": []\n'
        "}\n"
        "Each finding: {\"issue_id\": \"...\", \"severity\": \"error\", "
        "\"message\": \"...\", \"estimated_ratio\": \"3.2:1\", \"fixer\": \"fix_color_contrast\"}"
    )
