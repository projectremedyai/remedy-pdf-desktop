"""Shared STEM content classification for math and chemistry remediation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

STEM_ROLE_TEXT = "text"
STEM_ROLE_FORMULA = "formula"
STEM_ROLE_DIAGRAM = "diagram"

STEM_KIND_TEXT = "text"
STEM_KIND_FORMULA = "formula"
STEM_KIND_CHEMISTRY_REACTION = "chemistry_reaction"
STEM_KIND_CHEMISTRY_STRUCTURE_DIAGRAM = "chemistry_structure_diagram"

STEM_DOMAIN_TEXT = "text"
STEM_DOMAIN_MATH = "math"
STEM_DOMAIN_CHEMISTRY = "chemistry"

STEM_SOURCE_FORMAT_PLAIN_TEXT = "plain_text"
STEM_SOURCE_FORMAT_INLINE_TEX = "inline_tex"
STEM_SOURCE_FORMAT_DISPLAY_TEX = "display_tex"
STEM_SOURCE_FORMAT_MIXED_TEXT_TEX = "mixed_text_tex"
STEM_SOURCE_FORMAT_LATEX_LIKE = "latex_like"

_BOLD_ONLY_RE = re.compile(r"^\*\*(.+?)\*\*$", re.DOTALL)
_LEADING_ENUM_LABEL_RE = re.compile(
    r"^(?P<label>\([A-Za-z0-9]+\)|[A-Za-z0-9]+[.)])\s+(?P<body>.+)$",
    re.DOTALL,
)
_CHEM_REACTION_ARROW_RE = re.compile(
    r"(?:\\(?:rightleftharpoons|leftharpoons|leftrightarrow|rightarrow|leftarrow|to)|[⇌↔⇄→←])"
)
_CHEM_STATE_RE = re.compile(r"\((?:aq|g|l|s)\)", re.IGNORECASE)
_CHEM_SPECIES_RE = re.compile(
    r"(?:\\text\{[A-Za-z0-9+\-]+\}|[A-Z][a-z]?)(?:_\{?\d+\}?|[A-Z][a-z]?|\([A-Za-z0-9+\-]+\))*"
)
_CHEM_STRUCTURE_STRONG_HINTS = (
    "bond-line",
    "bond line",
    "dash-wedge",
    "dash wedge",
    "fischer projection",
    "lewis structure",
    "newman projection",
    "orbital diagram",
    "reaction mechanism",
    "resonance structure",
    "skeletal formula",
    "structural formula",
)
_CHEM_STRUCTURE_SOFT_HINTS = (
    "bond",
    "diagram",
    "lewis",
    "mechanism",
    "orbital",
    "projection",
    "resonance",
    "ring",
    "skeletal",
    "structure",
)
_CHEM_BOND_CHAIN_RE = re.compile(
    r"(?:[A-Z][A-Za-z0-9()]*)(?:\s*(?:-|=|≡|–|—)\s*(?:[A-Z][A-Za-z0-9()]*)){2,}"
)
_CHEM_FORMULA_TOKEN_RE = re.compile(
    r"(?:[A-Z][a-z]?)(?:\d+|[A-Z][a-z]?|\([A-Za-z0-9+\-]+\))*"
)
_CHEM_STRUCTURE_OPERATOR_RE = re.compile(r"(?:-|=|≡|–|—)")
_MATH_LATEX_HINT_RE = re.compile(
    r"(\\frac|\\sqrt|\\sum|\\int|\\lim|\\alpha|\\beta|\\theta|\\pi|\\sigma|\\mu|\\Delta|\\partial|\$[^$]+\$)"
)
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_DISPLAY_TEX_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_TEX_RE = re.compile(r"(?<!\$)\$[^$]+\$(?!\$)")
_MATH_OPERATOR_CHAR_RE = re.compile(r"[=+<>÷×^→←↔⇌⇄≈≠±]")
_MATH_WORD_HINTS = frozenset(
    {"sin", "cos", "tan", "cot", "sec", "csc", "lim", "log", "ln", "dx", "dy"}
)


@dataclass(frozen=True)
class STEMSyntaxProfile:
    """Surface syntax facts used by near-term math and chemistry routing."""

    source_format: str = STEM_SOURCE_FORMAT_PLAIN_TEXT
    line_count: int = 0
    operator_count: int = 0
    has_tex_commands: bool = False
    has_inline_tex: bool = False
    has_display_tex: bool = False
    has_reaction_arrow: bool = False
    has_chemistry_state: bool = False
    has_charge_annotation: bool = False


@dataclass(frozen=True)
class STEMTargetProfile:
    """Declared target policy for the current STEM object."""

    near_term_target: str | None = None
    near_term_renderer: str | None = None
    future_planning_targets: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class STEMTextObject:
    """Coarse STEM classification for extracted text blocks."""

    raw_text: str
    normalized_text: str
    kind: str
    role: str
    domain: str = STEM_DOMAIN_TEXT
    syntax_profile: STEMSyntaxProfile = field(default_factory=STEMSyntaxProfile)
    target_profile: STEMTargetProfile = field(default_factory=STEMTargetProfile)
    primary_target: str | None = None
    label: str | None = None
    body_text: str = ""
    normalized_body_text: str = ""
    reasons: tuple[str, ...] = ()

    @property
    def is_formula(self) -> bool:
        return self.role == STEM_ROLE_FORMULA

    @property
    def is_diagram(self) -> bool:
        return self.role == STEM_ROLE_DIAGRAM

    @property
    def source_text(self) -> str:
        return self.body_text or self.raw_text.strip()

    @property
    def source_format(self) -> str:
        return self.syntax_profile.source_format

    @property
    def future_planning_targets(self) -> tuple[str, ...]:
        return self.target_profile.future_planning_targets

    @property
    def benchmark_tags(self) -> tuple[str, ...]:
        tags = [self.kind]
        if self.is_formula:
            tags.extend(["mathml-path", "mathjax-near-term-target"])
            if self.source_format != STEM_SOURCE_FORMAT_PLAIN_TEXT:
                tags.append("source-annotation-preserved")
            if "nemeth" in self.future_planning_targets:
                tags.append("future-nemeth-planning")
        if self.kind == STEM_KIND_CHEMISTRY_REACTION:
            tags.append("chemistry-reaction-equation")
        if self.kind == STEM_KIND_CHEMISTRY_STRUCTURE_DIAGRAM:
            tags.append("chemistry-structure-separation")
        if self.label:
            tags.append("labeled-object")
        return tuple(tags)

    def to_dict(self) -> dict[str, object]:
        """Serialize a stable benchmark/debug view of the STEM object."""
        syntax_profile = asdict(self.syntax_profile)
        target_profile = asdict(self.target_profile)
        target_profile["future_planning_targets"] = list(self.target_profile.future_planning_targets)
        target_profile["blockers"] = list(self.target_profile.blockers)
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "kind": self.kind,
            "role": self.role,
            "domain": self.domain,
            "primary_target": self.primary_target,
            "label": self.label,
            "body_text": self.body_text,
            "normalized_body_text": self.normalized_body_text,
            "reasons": list(self.reasons),
            "syntax_profile": syntax_profile,
            "target_profile": target_profile,
            "benchmark_tags": list(self.benchmark_tags),
        }


def normalize_stem_text(text: str) -> str:
    """Return whitespace-normalized STEM text."""
    return " ".join(text.strip().split())


def _domain_for_kind(kind: str) -> str:
    """Map a coarse STEM kind into a broader content domain."""
    if kind == STEM_KIND_FORMULA:
        return STEM_DOMAIN_MATH
    if kind in {STEM_KIND_CHEMISTRY_REACTION, STEM_KIND_CHEMISTRY_STRUCTURE_DIAGRAM}:
        return STEM_DOMAIN_CHEMISTRY
    return STEM_DOMAIN_TEXT


def _infer_stem_source_format(text: str) -> str:
    """Infer the dominant source syntax for the preserved STEM payload."""
    stripped = text.strip()
    if not stripped:
        return STEM_SOURCE_FORMAT_PLAIN_TEXT

    has_display_tex = bool(_DISPLAY_TEX_RE.search(stripped))
    has_inline_tex = bool(_INLINE_TEX_RE.search(stripped))
    if has_display_tex or has_inline_tex:
        without_tex = _DISPLAY_TEX_RE.sub(" ", stripped)
        without_tex = _INLINE_TEX_RE.sub(" ", without_tex)
        if normalize_stem_text(without_tex):
            return STEM_SOURCE_FORMAT_MIXED_TEXT_TEX
        if has_display_tex:
            return STEM_SOURCE_FORMAT_DISPLAY_TEX
        return STEM_SOURCE_FORMAT_INLINE_TEX

    if _LATEX_COMMAND_RE.search(stripped):
        return STEM_SOURCE_FORMAT_LATEX_LIKE
    return STEM_SOURCE_FORMAT_PLAIN_TEXT


def _build_syntax_profile(text: str) -> STEMSyntaxProfile:
    """Capture low-level syntax facts that benchmark artifacts can inspect."""
    stripped = text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    normalized = normalize_stem_text(stripped)
    return STEMSyntaxProfile(
        source_format=_infer_stem_source_format(stripped),
        line_count=len(lines),
        operator_count=len(_MATH_OPERATOR_CHAR_RE.findall(normalized)),
        has_tex_commands=bool(_LATEX_COMMAND_RE.search(normalized)),
        has_inline_tex=bool(_INLINE_TEX_RE.search(stripped)),
        has_display_tex=bool(_DISPLAY_TEX_RE.search(stripped)),
        has_reaction_arrow=bool(_CHEM_REACTION_ARROW_RE.search(normalized)),
        has_chemistry_state=bool(_CHEM_STATE_RE.search(normalized)),
        has_charge_annotation=bool(re.search(r"\^\{?[0-9]*[+-]\}?", normalized)),
    )


def _build_target_profile(kind: str, primary_target: str | None) -> STEMTargetProfile:
    """Declare current near-term routing plus explicit future-planning placeholders."""
    if primary_target == "mathml":
        return STEMTargetProfile(
            near_term_target="mathml",
            near_term_renderer="mathjax",
            future_planning_targets=("nemeth",),
        )
    if kind == STEM_KIND_CHEMISTRY_STRUCTURE_DIAGRAM:
        return STEMTargetProfile(
            near_term_target="diagram-description",
            future_planning_targets=("accessible-chemistry-diagram",),
            blockers=("topology-not-modeled", "not-mathml-eligible"),
        )
    return STEMTargetProfile(near_term_target=primary_target)


def looks_like_chemical_reaction_equation_text(text: str) -> bool:
    """Detect compact chemistry reaction equations without promoting prose."""
    stripped = normalize_stem_text(text)
    if not stripped or not _CHEM_REACTION_ARROW_RE.search(stripped):
        return False

    state_hits = len(_CHEM_STATE_RE.findall(stripped))
    species_hits = len(_CHEM_SPECIES_RE.findall(stripped))
    coefficient_hits = len(re.findall(r"\b\d+(?=(?:\\text\{)?[A-Z])", stripped))
    charge_hits = len(re.findall(r"\^\{?[0-9]*[+-]\}?", stripped))
    return (
        state_hits >= 2
        or species_hits >= 3
        or coefficient_hits >= 1
        or charge_hits >= 1
    )


def looks_like_chemistry_structure_diagram_text(text: str) -> bool:
    """Detect chemistry structure-diagram candidates that should stay off the math path."""
    stripped = normalize_stem_text(text)
    if not stripped or looks_like_chemical_reaction_equation_text(stripped):
        return False

    lowered = stripped.lower()
    if any(hint in lowered for hint in _CHEM_STRUCTURE_STRONG_HINTS):
        return True

    formula_token_hits = len(_CHEM_FORMULA_TOKEN_RE.findall(stripped))
    operator_hits = len(_CHEM_STRUCTURE_OPERATOR_RE.findall(stripped))
    soft_hint_hits = sum(1 for hint in _CHEM_STRUCTURE_SOFT_HINTS if hint in lowered)

    if _CHEM_BOND_CHAIN_RE.search(stripped) and formula_token_hits >= 2:
        return True
    if soft_hint_hits >= 2 and formula_token_hits >= 1 and operator_hits >= 1:
        return True
    return False


def looks_like_formula_expression_text(text: str) -> bool:
    """Detect compact formula-like text that should stay on the math path."""
    stripped = normalize_stem_text(text)
    if not stripped:
        return False
    if looks_like_chemical_reaction_equation_text(stripped):
        return True
    if looks_like_chemistry_structure_diagram_text(stripped):
        return False

    operator_hits = len(re.findall(r"[=+<>]", stripped))
    paren_hits = len(re.findall(r"[()\[\]]", stripped))
    math_word_hits = sum(
        1
        for word in re.findall(r"[A-Za-z]+", stripped.lower())
        if word in _MATH_WORD_HINTS
    )
    alpha_tokens = re.findall(r"[A-Za-z]+", stripped)
    sentence_like = len(alpha_tokens) >= 7 and operator_hits == 0 and paren_hits <= 2

    if sentence_like:
        return False
    if _MATH_LATEX_HINT_RE.search(stripped):
        return True
    if re.search(r"[A-Za-z0-9][_^][A-Za-z0-9]", stripped):
        return True
    if re.search(r"[A-Za-z0-9)\]]\s*[=+<>]\s*[A-Za-z0-9(\[]", stripped):
        return True
    if paren_hits >= 4 and operator_hits >= 1:
        return True
    if math_word_hits >= 1 and operator_hits >= 2:
        return True
    return False


def _split_stem_label_candidate(text: str) -> tuple[str | None, str]:
    """Extract a leading STEM label when the remaining body still looks STEM-like."""
    stripped = text.strip()
    if not stripped:
        return None, ""

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines:
        bold_match = _BOLD_ONLY_RE.fullmatch(lines[0])
        if bold_match and len(lines) > 1:
            body = "\n".join(lines[1:]).strip()
            if body:
                return bold_match.group(1).strip(), body

    normalized = normalize_stem_text(stripped)
    enum_match = _LEADING_ENUM_LABEL_RE.match(normalized)
    if enum_match:
        return enum_match.group("label").strip(), enum_match.group("body").strip()
    return None, stripped


def _classify_normalized_stem_text(
    normalized: str,
) -> tuple[str, str, str | None, tuple[str, ...]]:
    """Classify already-normalized text into the current coarse STEM model."""
    if not normalized:
        return STEM_KIND_TEXT, STEM_ROLE_TEXT, None, ()

    if looks_like_chemical_reaction_equation_text(normalized):
        return (
            STEM_KIND_CHEMISTRY_REACTION,
            STEM_ROLE_FORMULA,
            "mathml",
            ("reaction-arrow", "chemistry-equation-shape"),
        )

    if looks_like_chemistry_structure_diagram_text(normalized):
        return (
            STEM_KIND_CHEMISTRY_STRUCTURE_DIAGRAM,
            STEM_ROLE_DIAGRAM,
            "diagram-description",
            ("chemistry-structure-shape",),
        )

    if looks_like_formula_expression_text(normalized):
        return (
            STEM_KIND_FORMULA,
            STEM_ROLE_FORMULA,
            "mathml",
            ("formula-expression-shape",),
        )

    return STEM_KIND_TEXT, STEM_ROLE_TEXT, None, ()


def classify_stem_text_object(text: str) -> STEMTextObject:
    """Classify a text block into a coarse STEM object with preserved block payload."""
    raw = text
    stripped = text.strip()
    normalized = normalize_stem_text(stripped)
    if not normalized:
        return STEMTextObject(
            raw_text=raw,
            normalized_text="",
            kind=STEM_KIND_TEXT,
            role=STEM_ROLE_TEXT,
            domain=STEM_DOMAIN_TEXT,
            body_text="",
            normalized_body_text="",
        )

    label_candidate, body_candidate = _split_stem_label_candidate(stripped)
    normalized_body = normalize_stem_text(body_candidate)
    kind, role, primary_target, reasons = _classify_normalized_stem_text(normalized)
    label = None
    body_text = stripped
    normalized_body_text = normalized

    if label_candidate and normalized_body:
        candidate_kind, candidate_role, candidate_target, candidate_reasons = (
            _classify_normalized_stem_text(normalized_body)
        )
        if candidate_kind != STEM_KIND_TEXT:
            kind = candidate_kind
            role = candidate_role
            primary_target = candidate_target
            reasons = candidate_reasons + ("leading-label",)
            label = label_candidate
            body_text = body_candidate
            normalized_body_text = normalized_body

    syntax_profile = _build_syntax_profile(body_text or stripped)
    domain = _domain_for_kind(kind)
    target_profile = _build_target_profile(kind, primary_target)

    return STEMTextObject(
        raw_text=raw,
        normalized_text=normalized,
        kind=kind,
        role=role,
        domain=domain,
        syntax_profile=syntax_profile,
        target_profile=target_profile,
        primary_target=primary_target,
        label=label,
        body_text=body_text,
        normalized_body_text=normalized_body_text,
        reasons=reasons,
    )
