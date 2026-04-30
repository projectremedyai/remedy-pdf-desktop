"""Convert between LaTeX, MathML, and plain-text math representations."""

from __future__ import annotations

import logging
import re

import latex2mathml.converter

logger = logging.getLogger(__name__)


def latex_to_mathml(latex: str) -> str:
    """Convert a LaTeX math string to MathML.

    Strips surrounding $...$ or \\[...\\] delimiters if present.
    Returns empty string on failure.
    """
    cleaned = _strip_delimiters(latex)
    if not cleaned:
        return ""

    try:
        return latex2mathml.converter.convert(cleaned)
    except Exception as exc:
        logger.warning("LaTeX→MathML conversion failed for %r: %s", cleaned[:60], exc)
        return ""


def latex_to_alt_text(latex: str) -> str:
    """Generate a screen-reader-friendly description from LaTeX.

    Uses simple pattern replacements for common constructs.
    Falls back to the raw LaTeX if nothing better can be produced.
    """
    cleaned = _strip_delimiters(latex)
    if not cleaned:
        return ""

    text = cleaned

    # Fractions
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1 over \2", text)

    # Powers / superscripts
    text = re.sub(r"\^(\{[^}]+\}|[a-zA-Z0-9])", _expand_superscript, text)

    # Subscripts
    text = re.sub(r"_(\{[^}]+\}|[a-zA-Z0-9])", _expand_subscript, text)

    # Square roots
    text = re.sub(r"\\sqrt\{([^}]+)\}", r"square root of \1", text)
    text = re.sub(r"\\sqrt\[(\d+)\]\{([^}]+)\}", r"\1th root of \2", text)

    # Common symbols
    replacements = {
        "\\pm": "plus or minus",
        "\\times": "times",
        "\\div": "divided by",
        "\\cdot": "times",
        "\\leq": "less than or equal to",
        "\\geq": "greater than or equal to",
        "\\neq": "not equal to",
        "\\approx": "approximately equal to",
        "\\infty": "infinity",
        "\\pi": "pi",
        "\\alpha": "alpha",
        "\\beta": "beta",
        "\\gamma": "gamma",
        "\\delta": "delta",
        "\\theta": "theta",
        "\\lambda": "lambda",
        "\\sigma": "sigma",
        "\\sum": "sum of",
        "\\int": "integral of",
        "\\partial": "partial",
        "\\nabla": "nabla",
        "\\rightarrow": "arrow",
        "\\leftarrow": "left arrow",
        "\\Rightarrow": "implies",
    }
    for cmd, word in replacements.items():
        text = text.replace(cmd, f" {word} ")

    # Clean up remaining backslash commands
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    # Clean up braces and extra whitespace
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text or cleaned


def _strip_delimiters(latex: str) -> str:
    """Remove surrounding math delimiters."""
    s = latex.strip()
    for start, end in [("$$", "$$"), ("$", "$"), ("\\[", "\\]"), ("\\(", "\\)")]:
        if s.startswith(start) and s.endswith(end):
            s = s[len(start):-len(end)].strip()
    return s


def _expand_superscript(match: re.Match) -> str:
    exp = match.group(1).strip("{}")
    if exp == "2":
        return " squared"
    if exp == "3":
        return " cubed"
    return f" to the power of {exp}"


def _expand_subscript(match: re.Match) -> str:
    sub = match.group(1).strip("{}")
    return f" sub {sub}"
