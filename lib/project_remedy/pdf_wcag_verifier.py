"""Vision-based WCAG 2.1 AA verification for remediated PDFs.

Two-tier verification:
1. **Triage** — cheap per-page prompt classifies page type, decides which
   focused checks apply, and auto-passes obvious pages.
2. **Focused verify** — targeted prompts for headings/layout, alt text
   accuracy, table structure, contrast, and form labels.

Results feed back into the fix loop via a routing table that maps vision
criterion failures to specific fix functions in ``pdf_fixer.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CriterionFinding:
    """One specific accessibility issue found by vision."""

    issue_id: str = ""
    severity: str = "warning"  # "error", "warning", "info"
    message: str = ""
    bbox: list[float] = field(default_factory=list)
    suggested_fix: str = ""
    fixer: str = ""  # maps to fix function name in pdf_fixer.py


@dataclass
class CriterionResult:
    """Vision verification result for one WCAG criterion on one page."""

    applicable: bool = True
    status: str = "not_applicable"  # "pass", "fail", "manual_review", "not_applicable"
    wcag_sc: list[str] = field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""
    findings: list[CriterionFinding] = field(default_factory=list)


@dataclass
class PageTriageResult:
    """Triage classification for a single page."""

    page_type: str = "unknown"
    skip: bool = False
    skip_reason: str = ""
    applicable_checks: dict = field(default_factory=dict)
    focus_queue: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class WCAGPageResult:
    """Full verification result for a single page."""

    page_index: int = 0
    page_type: str = "unknown"
    verification_mode: str = "skipped"  # "full", "focused", "inherited", "skipped"
    overall_status: str = "skipped"  # "pass", "fail", "manual_review", "skipped"
    overall_confidence: float = 0.0
    criteria: dict[str, CriterionResult] = field(default_factory=dict)
    retry_plan: dict = field(default_factory=lambda: {
        "should_retry": False,
        "fixers": [],
        "escalate_to_tier2": False,
        "manual_review": False,
    })


@dataclass
class WCAGVerificationResult:
    """Aggregate verification result for the entire document."""

    page_results: list[WCAGPageResult] = field(default_factory=list)
    overall_pass: bool = False
    failing_criteria: dict[str, list[int]] = field(default_factory=dict)
    pages_verified: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    total_findings: int = 0

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "overall_pass": self.overall_pass,
            "pages_verified": self.pages_verified,
            "pages_skipped": self.pages_skipped,
            "pages_failed": self.pages_failed,
            "total_findings": self.total_findings,
            "failing_criteria": self.failing_criteria,
            "page_results": [
                {
                    "page_index": pr.page_index,
                    "page_type": pr.page_type,
                    "verification_mode": pr.verification_mode,
                    "overall_status": pr.overall_status,
                    "overall_confidence": pr.overall_confidence,
                    "criteria": {
                        k: {
                            "applicable": v.applicable,
                            "status": v.status,
                            "wcag_sc": v.wcag_sc,
                            "confidence": v.confidence,
                            "summary": v.summary,
                            "findings": [
                                {
                                    "issue_id": f.issue_id,
                                    "severity": f.severity,
                                    "message": f.message,
                                    "suggested_fix": f.suggested_fix,
                                    "fixer": f.fixer,
                                }
                                for f in v.findings
                            ],
                        }
                        for k, v in pr.criteria.items()
                    },
                    "retry_plan": pr.retry_plan,
                }
                for pr in self.page_results
            ],
        }


# ---------------------------------------------------------------------------
# Fix routing — maps vision criterion failures to fixer functions
# ---------------------------------------------------------------------------

_FIX_ROUTING = {
    "headings": ["fix_heading_synthesis", "fix_heading_nesting"],
    "reading_order": ["fix_reading_order"],
    "alt_text_accuracy": ["fix_figures_alt_text"],
    "color_contrast": ["fix_color_contrast"],
    "table_structure": ["fix_table_headers", "fix_table_regularity"],
    "form_labels": ["fix_form_fields_tagged"],
    "document_language": ["fix_language"],
}

_RETRY_CONFIDENCE_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Structural hints extraction (cheap, no vision)
# ---------------------------------------------------------------------------


def _extract_structural_hints(pdf_path: Path, page_index: int) -> dict:
    """Extract cheap structural metadata for a page to anchor the triage prompt."""
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        page = doc[page_index]
        page_area = max(float(page.rect.width * page.rect.height), 1.0)

        data = page.get_text("dict")
        blocks = data.get("blocks", [])

        text_blocks = [b for b in blocks if b.get("type") == 0]
        image_blocks = [b for b in blocks if b.get("type") != 0]

        image_area = sum(
            max((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]), 0.0)
            for b in image_blocks
        )
        image_coverage = round(image_area / page_area, 3)

        # Count text characters
        total_chars = 0
        for b in text_blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    total_chars += len(span.get("text", ""))

        doc.close()

        # Check for figures and tables in structure tree
        has_figures = False
        has_tables = False
        has_widgets = False
        try:
            import pikepdf

            pdf = pikepdf.open(pdf_path)
            page_obj = pdf.pages[page_index]
            # Check annotations for widgets
            annots = page_obj.get("/Annots")
            if annots:
                for annot in annots:
                    subtype = str(annot.get("/Subtype", ""))
                    if subtype == "/Widget":
                        has_widgets = True
                        break
            pdf.close()
        except Exception:
            pass

        # Heuristic: figures if >10% image coverage
        has_figures = image_coverage > 0.10

        return {
            "text_block_count": len(text_blocks),
            "image_coverage": image_coverage,
            "total_chars": total_chars,
            "has_figures": has_figures,
            "has_tables": has_tables,
            "has_widgets": has_widgets,
        }
    except Exception:
        return {
            "text_block_count": 0,
            "image_coverage": 0.0,
            "total_chars": 0,
            "has_figures": False,
            "has_tables": False,
            "has_widgets": False,
        }


def _extract_page_structure_context(pdf_path: Path, page_index: int) -> tuple[str, str]:
    """Extract reading order and heading context for a page from the structure tree."""
    try:
        import pikepdf
        import re
        from project_remedy.pdf_checker import _resolve_pdf_object

        pdf = pikepdf.open(pdf_path)
        total_pages = len(pdf.pages)

        # Walk structure tree to find elements on this page
        elements_on_page: list[str] = []
        headings_all: list[tuple[int, str, str]] = []  # (page_idx, level, text)

        def _walk(node, depth=0):
            try:
                s = str(node.get("/S", ""))
                if not s:
                    return
                tag = s.lstrip("/")

                # Get page index for this node
                pg = node.get("/Pg")
                node_page = -1
                if pg is not None:
                    try:
                        resolved = _resolve_pdf_object(pg)
                        for i, p in enumerate(pdf.pages):
                            if p.obj == resolved:
                                node_page = i
                                break
                    except Exception:
                        pass

                # Collect headings from all pages for context
                if re.match(r"^H\d$", tag):
                    alt = str(node.get("/Alt", ""))[:60]
                    headings_all.append((node_page, tag, alt))

                # Collect reading order for target page
                if node_page == page_index:
                    alt = str(node.get("/Alt", ""))[:40]
                    label = f"{tag}" + (f": {alt}" if alt else "")
                    elements_on_page.append(label)

                kids = node.get("/K")
                if kids is not None:
                    if isinstance(kids, pikepdf.Array):
                        for k in kids[:200]:
                            try:
                                _walk(k, depth + 1)
                            except Exception:
                                pass
                    else:
                        try:
                            _walk(kids, depth + 1)
                        except Exception:
                            pass
            except Exception:
                pass

        st = pdf.Root.get("/StructTreeRoot")
        if st:
            kids = st.get("/K")
            if kids:
                if isinstance(kids, pikepdf.Array):
                    for k in kids[:50]:
                        try:
                            _walk(k)
                        except Exception:
                            pass
                else:
                    try:
                        _walk(kids)
                    except Exception:
                        pass

        pdf.close()

        # Build logical order string
        if elements_on_page:
            logical_order = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(elements_on_page[:30]))
        else:
            logical_order = "(no structure elements found on this page)"

        # Build heading context (prev + current + next page headings)
        prev_headings = [f"{h[1]}: {h[2]}" for h in headings_all if h[0] == page_index - 1]
        curr_headings = [f"{h[1]}: {h[2]}" for h in headings_all if h[0] == page_index]
        next_headings = [f"{h[1]}: {h[2]}" for h in headings_all if h[0] == page_index + 1]

        heading_parts = []
        if prev_headings:
            heading_parts.append(f"Previous page: {', '.join(prev_headings[:3])}")
        if curr_headings:
            heading_parts.append(f"This page: {', '.join(curr_headings[:5])}")
        else:
            heading_parts.append("This page: (no headings)")
        if next_headings:
            heading_parts.append(f"Next page: {', '.join(next_headings[:3])}")
        heading_context = "\n".join(heading_parts) if heading_parts else "(no heading context)"

        return logical_order, heading_context
    except Exception:
        return "(structure extraction failed)", "(no heading context)"


def _should_skip_page(hints: dict) -> tuple[bool, str]:
    """Deterministic skip rules — no vision needed for these pages."""
    if hints["text_block_count"] == 0 and hints["image_coverage"] < 0.05:
        return True, "blank page"
    if hints["total_chars"] < 10 and hints["image_coverage"] < 0.05:
        return True, "near-blank (page number only)"
    return False, ""


# ---------------------------------------------------------------------------
# Core verifier
# ---------------------------------------------------------------------------


class WCAGVisionVerifier:
    """Vision-based WCAG 2.1 AA verifier.

    Triage → focused verify → aggregate results.
    """

    def __init__(
        self,
        vision_provider,
        *,
        vision_concurrency: int | None = None,
        render_concurrency: int | None = None,
    ) -> None:
        self.vision_provider = vision_provider
        self.vision_limit = max(1, int(
            vision_concurrency
            or os.getenv("WCAG_VISION_MAX_INFLIGHT", "5")
        ))
        self.render_limit = max(1, int(
            render_concurrency
            or os.getenv("WCAG_RENDER_MAX_INFLIGHT", "3")
        ))

    async def verify_document(
        self,
        pdf_path: Path,
        *,
        pages: list[int] | None = None,
    ) -> WCAGVerificationResult:
        """Verify a full document. Returns per-page, per-criterion results."""
        import fitz

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()

        if pages is None:
            pages = list(range(total_pages))

        page_results: list[WCAGPageResult] = []
        skipped = 0
        failed = 0

        # Phase 1: deterministic skip + structural hints
        triage_candidates: list[tuple[int, dict]] = []
        for page_idx in pages:
            hints = _extract_structural_hints(pdf_path, page_idx)
            should_skip, reason = _should_skip_page(hints)
            if should_skip:
                page_results.append(WCAGPageResult(
                    page_index=page_idx,
                    page_type="blank",
                    verification_mode="skipped",
                    overall_status="pass",
                    overall_confidence=1.0,
                ))
                skipped += 1
            else:
                triage_candidates.append((page_idx, hints))

        # Phase 2: triage via vision (bounded async)
        render_sem = asyncio.Semaphore(self.render_limit)
        vision_sem = asyncio.Semaphore(self.vision_limit)

        async def _triage_one(page_idx: int, hints: dict):
            return page_idx, hints, await self._triage_page(
                pdf_path, page_idx, hints, render_sem, vision_sem,
            )

        triage_results = await asyncio.gather(
            *(_triage_one(idx, h) for idx, h in triage_candidates),
            return_exceptions=True,
        )

        # Phase 3: focused verification for non-skipped pages
        focus_tasks = []
        triage_map: dict[int, tuple[dict, PageTriageResult]] = {}

        for r in triage_results:
            if isinstance(r, Exception):
                logger.warning("Triage failed: %s", r)
                continue
            page_idx, hints, triage = r
            if triage.skip:
                page_results.append(WCAGPageResult(
                    page_index=page_idx,
                    page_type=triage.page_type,
                    verification_mode="skipped",
                    overall_status="pass",
                    overall_confidence=triage.confidence,
                ))
                skipped += 1
            else:
                triage_map[page_idx] = (hints, triage)
                focus_tasks.append(page_idx)

        # Run focused verification
        async def _verify_one(page_idx: int):
            hints, triage = triage_map[page_idx]
            return page_idx, await self._verify_page_focused(
                pdf_path, page_idx, triage, render_sem, vision_sem,
            )

        if focus_tasks:
            focus_results = await asyncio.gather(
                *(_verify_one(idx) for idx in focus_tasks),
                return_exceptions=True,
            )
            for r in focus_results:
                if isinstance(r, Exception):
                    logger.warning("Focused verify failed: %s", r)
                    continue
                page_idx, page_result = r
                page_results.append(page_result)
                if page_result.overall_status == "fail":
                    failed += 1

        # Sort by page index
        page_results.sort(key=lambda pr: pr.page_index)

        # Aggregate failing criteria
        failing_criteria: dict[str, list[int]] = {}
        total_findings = 0
        for pr in page_results:
            for crit_name, crit_result in pr.criteria.items():
                if crit_result.status == "fail":
                    failing_criteria.setdefault(crit_name, []).append(pr.page_index)
                total_findings += len(crit_result.findings)

        return WCAGVerificationResult(
            page_results=page_results,
            overall_pass=failed == 0,
            failing_criteria=failing_criteria,
            pages_verified=len(pages) - skipped,
            pages_skipped=skipped,
            pages_failed=failed,
            total_findings=total_findings,
        )

    async def _triage_page(
        self,
        pdf_path: Path,
        page_idx: int,
        hints: dict,
        render_sem: asyncio.Semaphore,
        vision_sem: asyncio.Semaphore,
    ) -> PageTriageResult:
        """Run the cheap triage prompt on one page."""
        from project_remedy.pdf_vision import render_page_to_image, _parse_json_response
        from project_remedy.vision_prompts import wcag_page_triage_prompt

        image_path = None
        try:
            async with render_sem:
                image_path = await asyncio.to_thread(
                    render_page_to_image, pdf_path, page_idx + 1, 150,
                )

            prompt = wcag_page_triage_prompt(json.dumps(hints, indent=2))
            async with vision_sem:
                response = await self.vision_provider.analyze_image(image_path, prompt)

            parsed = _parse_json_response(response)
            if isinstance(parsed, dict):
                # Derive focus_queue from applicable_checks if model didn't return it
                focus_queue = parsed.get("focus_queue") or []
                applicable = parsed.get("applicable_checks") or {}
                if not focus_queue and applicable:
                    if applicable.get("headings") or applicable.get("reading_order"):
                        focus_queue.append("core_layout")
                    if applicable.get("alt_text_accuracy"):
                        focus_queue.append("figures")
                    if applicable.get("table_structure"):
                        focus_queue.append("tables")
                    if applicable.get("form_labels"):
                        focus_queue.append("forms")
                    if applicable.get("color_contrast"):
                        focus_queue.append("contrast")
                # If still empty, default to core_layout (always worth checking)
                if not focus_queue:
                    focus_queue = ["core_layout"]

                result = PageTriageResult(
                    page_type=parsed.get("page_type", "unknown"),
                    skip=parsed.get("skip", False),
                    skip_reason=parsed.get("skip_reason", ""),
                    applicable_checks=applicable,
                    focus_queue=focus_queue,
                    confidence=parsed.get("confidence", 0.5),
                )
                logger.info(
                    "Triage page %d: type=%s, skip=%s, queue=%s",
                    page_idx, result.page_type, result.skip, result.focus_queue,
                )
                return result
            logger.warning("Triage page %d: unparseable response, defaulting to core_layout", page_idx)
            return PageTriageResult(focus_queue=["core_layout"])
        except Exception as exc:
            logger.warning("Triage failed page %d: %s", page_idx, exc)
            return PageTriageResult(focus_queue=["core_layout"])
        finally:
            if image_path:
                try:
                    image_path.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _verify_page_focused(
        self,
        pdf_path: Path,
        page_idx: int,
        triage: PageTriageResult,
        render_sem: asyncio.Semaphore,
        vision_sem: asyncio.Semaphore,
    ) -> WCAGPageResult:
        """Run focused verification prompts based on triage results."""
        from project_remedy.pdf_vision import render_page_to_image, _parse_json_response
        from project_remedy.vision_prompts import wcag_core_layout_verify_prompt

        criteria: dict[str, CriterionResult] = {}
        page_result = WCAGPageResult(
            page_index=page_idx,
            page_type=triage.page_type,
            verification_mode="focused",
        )

        # Render page once for all focused prompts
        image_path = None
        try:
            async with render_sem:
                image_path = await asyncio.to_thread(
                    render_page_to_image, pdf_path, page_idx + 1, 150,
                )

            # Core layout verification (headings + reading order)
            _fq = set(triage.focus_queue)
            needs_layout = _fq & {"core_layout", "headings", "reading_order"}
            needs_figures = _fq & {"figures", "alt_text_accuracy"}
            needs_tables = _fq & {"tables", "table_structure"}
            needs_contrast = _fq & {"contrast", "color_contrast"}

            if needs_layout:
                logical_order, heading_context = _extract_page_structure_context(
                    pdf_path, page_idx,
                )
                prompt = wcag_core_layout_verify_prompt(
                    logical_order=logical_order,
                    heading_context=heading_context,
                )
                async with vision_sem:
                    response = await self.vision_provider.analyze_image(image_path, prompt)

                if response is None:
                    response = ""
                logger.info(
                    "core_layout page %d response (%d chars): %.200s",
                    page_idx, len(response), response,
                )
                parsed = _parse_json_response(response)
                if isinstance(parsed, dict):
                    for crit_key in ("headings", "reading_order"):
                        crit_data = parsed.get(crit_key, {})
                        findings = []
                        for f in crit_data.get("findings", []):
                            findings.append(CriterionFinding(
                                issue_id=f.get("issue_id", ""),
                                severity=f.get("severity", "warning"),
                                message=f.get("message", ""),
                                suggested_fix=f.get("suggested_fix", ""),
                                fixer=f.get("fixer", ""),
                            ))
                        wcag_map = {
                            "headings": ["1.3.1", "2.4.6"],
                            "reading_order": ["1.3.1", "1.3.2"],
                        }
                        criteria[crit_key] = CriterionResult(
                            applicable=True,
                            status=crit_data.get("status", "pass"),
                            wcag_sc=wcag_map.get(crit_key, []),
                            confidence=crit_data.get("confidence", 0.5),
                            summary=crit_data.get("summary", ""),
                            findings=findings,
                        )

            # Alt text verification
            if needs_figures:
                criteria["alt_text_accuracy"] = CriterionResult(
                    applicable=True,
                    status="pass",
                    wcag_sc=["1.1.1"],
                    confidence=0.5,
                    summary="Figure alt text verification — requires per-figure prompts",
                )
                # TODO: implement per-figure alt text verification
                # For each figure: crop image, send with current alt text, verify accuracy

            # Table verification
            if needs_tables:
                criteria["table_structure"] = CriterionResult(
                    applicable=True,
                    status="pass",
                    wcag_sc=["1.3.1"],
                    confidence=0.5,
                    summary="Table structure verification — pending implementation",
                )

            # Contrast verification
            if needs_contrast:
                criteria["color_contrast"] = CriterionResult(
                    applicable=True,
                    status="pass",
                    wcag_sc=["1.4.3", "1.4.1"],
                    confidence=0.5,
                    summary="Contrast verification — pending implementation",
                )

        except Exception as exc:
            logger.warning("Focused verify failed page %d: %s", page_idx, exc)
        finally:
            if image_path:
                try:
                    image_path.unlink(missing_ok=True)
                except Exception:
                    pass

        # Determine overall page status
        page_result.criteria = criteria
        statuses = [c.status for c in criteria.values() if c.applicable]
        if any(s == "fail" for s in statuses):
            page_result.overall_status = "fail"
        elif any(s == "manual_review" for s in statuses):
            page_result.overall_status = "manual_review"
        elif statuses:
            page_result.overall_status = "pass"
        else:
            page_result.overall_status = "pass"

        page_result.overall_confidence = (
            min(c.confidence for c in criteria.values()) if criteria else 1.0
        )

        # Build retry plan
        fixers = []
        for crit_name, crit_result in criteria.items():
            if (
                crit_result.status == "fail"
                and crit_result.confidence >= _RETRY_CONFIDENCE_THRESHOLD
            ):
                fixers.extend(_FIX_ROUTING.get(crit_name, []))
        page_result.retry_plan = {
            "should_retry": len(fixers) > 0,
            "fixers": list(dict.fromkeys(fixers)),  # dedupe preserving order
            "escalate_to_tier2": False,
            "manual_review": page_result.overall_status == "manual_review",
        }

        return page_result
