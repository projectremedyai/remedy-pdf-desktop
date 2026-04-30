"""Stage 6: Image & Visual Processing using multimodal vision.

Analyzes images extracted from documents and produces accessible
alternatives — concise alt text, SVG chart recreations with HTML data
tables, structured diagram descriptions, infographic decompositions,
and decorative-image detection.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from project_remedy.ollama_client import OllamaClient
from project_remedy.vision_prompts import (
    chart_prompt,
    diagram_prompt,
    figure_alt_prompt,
    image_classification_prompt,
    infographic_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImageResult:
    """Result of processing a single image through the vision pipeline."""

    alt_text: str = ""
    html_replacement: str = ""  # Optional SVG/table/description HTML
    is_decorative: bool = False
    processing_mode: str = ""  # alt_text | chart | diagram | infographic | decorative

    def to_dict(self) -> dict[str, Any]:
        return {
            "alt_text": self.alt_text,
            "html_replacement": self.html_replacement,
            "is_decorative": self.is_decorative,
            "processing_mode": self.processing_mode,
        }


# ---------------------------------------------------------------------------
# Processing-mode constants
# ---------------------------------------------------------------------------

MODE_ALT_TEXT = "alt_text"
MODE_CHART = "chart"
MODE_DIAGRAM = "diagram"
MODE_INFOGRAPHIC = "infographic"
MODE_DECORATIVE = "decorative"

# ---------------------------------------------------------------------------
# VisionProcessor
# ---------------------------------------------------------------------------


class VisionProcessor:
    """Processes images extracted from documents using multimodal vision.

    Automatically classifies each image and applies the appropriate
    processing strategy — alt text generation, chart data extraction
    and SVG recreation, diagram description, infographic decomposition,
    or decorative detection.

    Parameters
    ----------
    ollama_client:
        An initialised :class:`OllamaClient` instance.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        self._ollama = ollama_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_image(
        self, image_path: Path, context: str = ""
    ) -> ImageResult:
        """Classify an image and process it with the appropriate strategy.

        Parameters
        ----------
        image_path:
            Path to a local image file.
        context:
            Optional surrounding text or link context from the document
            to help the model understand the image's purpose.

        Returns
        -------
        ImageResult
            Contains alt text, optional HTML replacement, decorative flag,
            and the processing mode that was applied.
        """
        mode = await self._classify_image(image_path)
        logger.info("Image %s classified as: %s", image_path.name, mode)

        if mode == MODE_DECORATIVE:
            return ImageResult(
                alt_text="",
                html_replacement='<img src="" alt="" role="presentation">',
                is_decorative=True,
                processing_mode=MODE_DECORATIVE,
            )

        if mode == MODE_CHART:
            html = await self.recreate_chart_as_svg(image_path)
            # Extract a summary for alt text from the chart data
            alt = await self.generate_alt_text(image_path, context)
            return ImageResult(
                alt_text=alt,
                html_replacement=html,
                is_decorative=False,
                processing_mode=MODE_CHART,
            )

        if mode == MODE_DIAGRAM:
            html = await self.describe_diagram(image_path)
            alt = await self.generate_alt_text(image_path, context)
            return ImageResult(
                alt_text=alt,
                html_replacement=html,
                is_decorative=False,
                processing_mode=MODE_DIAGRAM,
            )

        if mode == MODE_INFOGRAPHIC:
            html = await self._decompose_infographic(image_path)
            alt = await self.generate_alt_text(image_path, context)
            return ImageResult(
                alt_text=alt,
                html_replacement=html,
                is_decorative=False,
                processing_mode=MODE_INFOGRAPHIC,
            )

        # Default: photograph / simple graphic — alt text only
        alt = await self.generate_alt_text(image_path, context)
        return ImageResult(
            alt_text=alt,
            html_replacement="",
            is_decorative=False,
            processing_mode=MODE_ALT_TEXT,
        )

    async def generate_alt_text(
        self, image_path: Path, context: str = ""
    ) -> str:
        """Generate concise descriptive alt text (max 125 chars).

        Parameters
        ----------
        image_path:
            Path to a local image file.
        context:
            Optional surrounding text to improve description accuracy.

        Returns
        -------
        str
            Alt text, guaranteed to be at most 125 characters.
        """
        prompt = figure_alt_prompt(context=context.strip())

        raw = await self._ollama.vision(image_path=image_path, prompt=prompt)
        # Strip quotes and whitespace the model may have added
        alt = raw.strip().strip('"').strip("'")
        # Enforce 125 char limit
        if len(alt) > 125:
            alt = alt[:122] + "..."
        logger.debug("Alt text for %s: %s", image_path.name, alt)
        return alt

    async def recreate_chart_as_svg(self, image_path: Path) -> str:
        """Extract chart data and return accessible SVG + HTML data table.

        Parameters
        ----------
        image_path:
            Path to a chart image.

        Returns
        -------
        str
            HTML string containing an accessible SVG visualisation with
            ARIA labels, an HTML ``<table>`` with the extracted data, and
            a text summary paragraph.
        """
        raw = await self._ollama.vision(image_path=image_path, prompt=chart_prompt())
        data = self._parse_json(raw)

        if not data:
            logger.warning(
                "Could not parse chart data for %s; falling back to alt text",
                image_path.name,
            )
            alt = await self.generate_alt_text(image_path)
            return f'<p>{_escape(alt)}</p>'

        chart_type = data.get("chart_type", "chart")
        title = data.get("title") or "Chart"
        x_label = data.get("x_label", "")
        y_label = data.get("y_label", "")
        legend = data.get("legend") or []
        rows = data.get("data") or []
        summary = data.get("summary", "")

        # Build accessible SVG
        svg = self._build_chart_svg(chart_type, title, rows, x_label, y_label, legend)

        # Build HTML data table
        table = self._build_data_table(title, rows)

        # Compose full HTML replacement
        parts = [
            f'<figure role="group" aria-label="{_escape(title)}">',
            svg,
            f"<figcaption>{_escape(title)}</figcaption>",
            table,
        ]
        if summary:
            parts.append(f'<p class="chart-summary">{_escape(summary)}</p>')
        parts.append("</figure>")

        html = "\n".join(parts)
        logger.info("Recreated chart SVG+table for %s", image_path.name)
        return html

    async def describe_diagram(self, image_path: Path) -> str:
        """Describe a diagram as structured, accessible HTML.

        Parameters
        ----------
        image_path:
            Path to a diagram image.

        Returns
        -------
        str
            HTML containing a description list or nested list that
            conveys the diagram structure to screen reader users.
        """
        raw = await self._ollama.vision(image_path=image_path, prompt=diagram_prompt())
        data = self._parse_json(raw)

        if not data:
            logger.warning(
                "Could not parse diagram data for %s; falling back to alt text",
                image_path.name,
            )
            alt = await self.generate_alt_text(image_path)
            return f'<p>{_escape(alt)}</p>'

        diagram_type = data.get("diagram_type", "diagram")
        title = data.get("title") or "Diagram"
        description = data.get("description", "")
        nodes = data.get("nodes") or []
        connections = data.get("connections") or []
        summary_text = data.get("summary", "")

        parts = [
            f'<div role="img" aria-label="{_escape(title)}" '
            f'class="diagram-description">',
            f"<h3>{_escape(title)}</h3>",
        ]

        if description:
            parts.append(f"<p>{_escape(description)}</p>")

        # Build a description list of nodes
        if nodes:
            parts.append(f'<h4>{diagram_type.replace("_", " ").title()} Components</h4>')
            parts.append("<dl>")
            for node in nodes:
                label = node.get("label", "")
                details = node.get("details", "")
                parts.append(f"  <dt>{_escape(label)}</dt>")
                if details:
                    parts.append(f"  <dd>{_escape(details)}</dd>")
                else:
                    parts.append(f"  <dd>{_escape(label)} node</dd>")
            parts.append("</dl>")

        # Describe connections as a nested list
        if connections:
            parts.append("<h4>Connections</h4>")
            parts.append("<ul>")
            # Build a lookup for node labels
            node_map = {
                n.get("id", ""): n.get("label", n.get("id", ""))
                for n in nodes
            }
            for conn in connections:
                from_label = node_map.get(
                    conn.get("from_id", ""), conn.get("from_id", "?")
                )
                to_label = node_map.get(
                    conn.get("to_id", ""), conn.get("to_id", "?")
                )
                conn_label = conn.get("label", "")
                arrow = f" ({conn_label})" if conn_label else ""
                parts.append(
                    f"  <li>{_escape(from_label)} &rarr; "
                    f"{_escape(to_label)}{_escape(arrow)}</li>"
                )
            parts.append("</ul>")

        if summary_text:
            parts.append(f'<p class="diagram-summary">{_escape(summary_text)}</p>')

        parts.append("</div>")

        html = "\n".join(parts)
        logger.info("Described diagram for %s", image_path.name)
        return html

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _classify_image(self, image_path: Path) -> str:
        """Ask the vision model to classify an image into a processing mode."""
        raw = await self._ollama.vision(
            image_path=image_path, prompt=image_classification_prompt()
        )
        data = self._parse_json(raw)

        if not data:
            logger.warning(
                "Classification failed for %s; defaulting to alt_text",
                image_path.name,
            )
            return MODE_ALT_TEXT

        category = data.get("category", "").lower().strip()

        category_map = {
            "photograph": MODE_ALT_TEXT,
            "chart": MODE_CHART,
            "diagram": MODE_DIAGRAM,
            "infographic": MODE_INFOGRAPHIC,
            "decorative": MODE_DECORATIVE,
        }

        mode = category_map.get(category, MODE_ALT_TEXT)
        return mode

    async def _decompose_infographic(self, image_path: Path) -> str:
        """Break an infographic into logical sections as accessible HTML."""
        raw = await self._ollama.vision(
            image_path=image_path, prompt=infographic_prompt()
        )
        data = self._parse_json(raw)

        if not data:
            alt = await self.generate_alt_text(image_path)
            return f'<p>{_escape(alt)}</p>'

        title = data.get("title") or "Infographic"
        sections = data.get("sections") or []
        summary = data.get("summary", "")

        parts = [
            f'<div role="img" aria-label="{_escape(title)}" '
            f'class="infographic-description">',
            f"<h3>{_escape(title)}</h3>",
        ]

        if summary:
            parts.append(f"<p>{_escape(summary)}</p>")

        for i, section in enumerate(sections, 1):
            heading = section.get("heading", f"Section {i}")
            content = section.get("content", "")
            data_points = section.get("data_points") or []

            parts.append(f"<h4>{_escape(heading)}</h4>")
            if content:
                parts.append(f"<p>{_escape(content)}</p>")
            if data_points:
                parts.append("<ul>")
                for dp in data_points:
                    parts.append(f"  <li>{_escape(str(dp))}</li>")
                parts.append("</ul>")

        parts.append("</div>")

        html = "\n".join(parts)
        logger.info("Decomposed infographic for %s", image_path.name)
        return html

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        """Attempt to parse JSON from model output, stripping fences."""
        text = raw.strip()
        # Remove markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
            return None
        except (json.JSONDecodeError, ValueError):
            logger.debug("JSON parse failed for: %.200s", text)
            return None

    @staticmethod
    def _build_chart_svg(
        chart_type: str,
        title: str,
        rows: list[dict[str, Any]],
        x_label: str,
        y_label: str,
        legend: list[str],
    ) -> str:
        """Generate an accessible SVG bar/line/pie chart from extracted data."""
        if not rows:
            return f'<p role="img" aria-label="{_escape(title)}">Chart data unavailable.</p>'

        # Determine data keys (exclude 'label')
        sample = rows[0]
        value_keys = [k for k in sample if k != "label"]

        # Colour palette with sufficient contrast
        colours = [
            "#004590", "#D94E0F", "#2E7D32", "#6A1B9A",
            "#C62828", "#00695C", "#E65100", "#1565C0",
        ]

        if chart_type == "pie":
            return _build_pie_svg(title, rows, value_keys, colours)

        # Default: bar chart SVG
        return _build_bar_svg(title, rows, value_keys, colours, x_label, y_label, legend)

    @staticmethod
    def _build_data_table(title: str, rows: list[dict[str, Any]]) -> str:
        """Build an accessible HTML data table from chart/diagram data."""
        if not rows:
            return ""

        keys = list(rows[0].keys())

        lines = [
            '<table class="chart-data-table">',
            f"  <caption>Data table: {_escape(title)}</caption>",
            "  <thead>",
            "    <tr>",
        ]
        for key in keys:
            header = key.replace("_", " ").title()
            lines.append(f'      <th scope="col">{_escape(header)}</th>')
        lines.extend(["    </tr>", "  </thead>", "  <tbody>"])

        for row in rows:
            lines.append("    <tr>")
            for i, key in enumerate(keys):
                val = row.get(key, "")
                if i == 0:
                    lines.append(f'      <th scope="row">{_escape(str(val))}</th>')
                else:
                    lines.append(f"      <td>{_escape(str(val))}</td>")
            lines.append("    </tr>")

        lines.extend(["  </tbody>", "</table>"])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SVG builders (module-level helpers)
# ---------------------------------------------------------------------------


def _build_bar_svg(
    title: str,
    rows: list[dict[str, Any]],
    value_keys: list[str],
    colours: list[str],
    x_label: str,
    y_label: str,
    legend: list[str],
) -> str:
    """Generate an accessible SVG bar chart."""
    width = 600
    height = 400
    margin = {"top": 40, "right": 20, "bottom": 60, "left": 60}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    # Find max value across all series
    all_values: list[float] = []
    for row in rows:
        for k in value_keys:
            try:
                all_values.append(float(row.get(k, 0)))
            except (ValueError, TypeError):
                pass
    max_val = max(all_values) if all_values else 1

    n_groups = len(rows)
    n_series = len(value_keys)
    group_width = chart_w / max(n_groups, 1)
    bar_width = (group_width * 0.8) / max(n_series, 1)
    gap = group_width * 0.1

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_escape(title)}" '
        f'style="max-width:100%;height:auto;">',
        f'  <title>{_escape(title)}</title>',
    ]

    # Bars
    for gi, row in enumerate(rows):
        label = str(row.get("label", f"Item {gi + 1}"))
        gx = margin["left"] + gi * group_width + gap

        for si, key in enumerate(value_keys):
            try:
                val = float(row.get(key, 0))
            except (ValueError, TypeError):
                val = 0
            bar_h = (val / max_val) * chart_h if max_val else 0
            bx = gx + si * bar_width
            by = margin["top"] + chart_h - bar_h
            colour = colours[si % len(colours)]
            series_name = legend[si] if si < len(legend) else key

            lines.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" '
                f'width="{bar_width:.1f}" height="{bar_h:.1f}" '
                f'fill="{colour}" '
                f'aria-label="{_escape(label)}, {_escape(series_name)}: {val}">'
                f"<title>{_escape(label)} - {_escape(series_name)}: {val}</title>"
                f"</rect>"
            )

        # X-axis label
        tx = gx + group_width * 0.4
        ty = margin["top"] + chart_h + 20
        lines.append(
            f'  <text x="{tx:.1f}" y="{ty}" '
            f'text-anchor="middle" font-size="11" fill="#1A1A2E">'
            f"{_escape(label)}</text>"
        )

    # Axes
    ax_y = margin["top"] + chart_h
    lines.append(
        f'  <line x1="{margin["left"]}" y1="{margin["top"]}" '
        f'x2="{margin["left"]}" y2="{ax_y}" '
        f'stroke="#1A1A2E" stroke-width="1"/>'
    )
    lines.append(
        f'  <line x1="{margin["left"]}" y1="{ax_y}" '
        f'x2="{margin["left"] + chart_w}" y2="{ax_y}" '
        f'stroke="#1A1A2E" stroke-width="1"/>'
    )

    # Axis labels
    if y_label:
        lines.append(
            f'  <text x="15" y="{margin["top"] + chart_h // 2}" '
            f'text-anchor="middle" font-size="12" fill="#1A1A2E" '
            f'transform="rotate(-90, 15, {margin["top"] + chart_h // 2})">'
            f"{_escape(y_label)}</text>"
        )
    if x_label:
        lines.append(
            f'  <text x="{margin["left"] + chart_w // 2}" y="{height - 5}" '
            f'text-anchor="middle" font-size="12" fill="#1A1A2E">'
            f"{_escape(x_label)}</text>"
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _build_pie_svg(
    title: str,
    rows: list[dict[str, Any]],
    value_keys: list[str],
    colours: list[str],
) -> str:
    """Generate an accessible SVG pie chart."""
    import math

    width = 400
    height = 400
    cx, cy, r = 200, 200, 150

    # Use first value key
    val_key = value_keys[0] if value_keys else "value"
    slices: list[tuple[str, float]] = []
    for row in rows:
        label = str(row.get("label", ""))
        try:
            val = float(row.get(val_key, 0))
        except (ValueError, TypeError):
            val = 0
        slices.append((label, val))

    total = sum(v for _, v in slices) or 1

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_escape(title)}" '
        f'style="max-width:100%;height:auto;">',
        f'  <title>{_escape(title)}</title>',
    ]

    angle = -90  # Start at top
    for i, (label, val) in enumerate(slices):
        pct = val / total
        sweep = pct * 360
        start_rad = math.radians(angle)
        end_rad = math.radians(angle + sweep)

        x1 = cx + r * math.cos(start_rad)
        y1 = cy + r * math.sin(start_rad)
        x2 = cx + r * math.cos(end_rad)
        y2 = cy + r * math.sin(end_rad)

        large = 1 if sweep > 180 else 0
        colour = colours[i % len(colours)]

        d = (
            f"M {cx} {cy} "
            f"L {x1:.2f} {y1:.2f} "
            f"A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f} Z"
        )

        lines.append(
            f'  <path d="{d}" fill="{colour}" '
            f'aria-label="{_escape(label)}: {val} ({pct:.0%})">'
            f"<title>{_escape(label)}: {val} ({pct:.0%})</title>"
            f"</path>"
        )
        angle += sweep

    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _escape(text: str) -> str:
    """Minimal HTML entity escaping for attribute/content safety."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
