#!/usr/bin/env python3
"""Curate a heldout PDF corpus that exercises every MiniCPM task adapter.

The WCAG verifier only issues a ``table_structure`` or ``contrast`` vision call
when its **triage** step puts ``tables`` / ``contrast`` into the page's
``focus_queue`` (``pdf_wcag_verifier.py:685-691``). Triage is itself a vision
call over the rendered page, keyed on ``applicable_checks.table_structure`` and
``applicable_checks.color_contrast`` (``:610-622``) — i.e. it fires on what the
page *looks like*, not on a deterministic PDF property.

So a corpus that only contains plain text forms can never exercise the table or
contrast adapters. This script scores each candidate for the two visual traits
and selects a corpus with guaranteed coverage of both.

Detectors (both cheap, CPU-only, no rendering):

* ``tables``   — ``page.find_tables()`` (PyMuPDF's ruling-line/alignment finder).
                 This approximates what the triage model sees. Tagged ``/Table``
                 structure elements are *not* required and are not used.
* ``contrast`` — every text span's colour is compared against the fill rect it
                 sits inside (white when none) using the WCAG 2.x contrast-ratio
                 formula. A span below ``--contrast-threshold`` (default 4.5, the
                 AA threshold for normal text) marks the document.

Reuses ``run_e2e_heldout_gate.py``'s contamination guard, so a document whose
de-hashed stem appears in any ``data/tasks/*/{train,val}.jsonl`` split is never
selected. Documents are additionally deduped by content SHA-256, so the same PDF
appearing under two names in two pools is only ever emitted once.

Usage::

    ./build_gate_corpus.py --out eval_runs/e2e_gate_prep/gate_corpus_v1.txt

Writes a newline-delimited path list (consumable by
``run_e2e_heldout_gate.py --corpus-list``) plus a ``.json`` sidecar summarising
counts by trigger type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minicpm_edge.constants import DEFAULT_DATA_ROOT, DEFAULT_EVAL_ROOT
from run_e2e_heldout_gate import _normalise_stem, is_real_pdf, trained_doc_ids

DEFAULT_POOLS: tuple[Path, ...] = (
    Path("~/code/lamc_district_forms/data/visual_match/downloads/district"),
    Path("~/code/lamc_district_forms/data/visual_match/downloads/lamc"),
    Path("~/code/lamc_district_forms/data/visual_match/downloads/all_campuses"),
)

CONTRAST_THRESHOLD = 4.5  # WCAG 2.x AA, normal text
MIN_SPAN_CHARS = 3


@dataclass
class Candidate:
    doc_id: str
    path: str
    pool: str
    sha256: str
    pages_scanned: int = 0
    tables: int = 0
    low_contrast_spans: int = 0
    error: str = ""

    @property
    def has_tables(self) -> bool:
        return self.tables > 0

    @property
    def has_contrast(self) -> bool:
        return self.low_contrast_spans > 0

    @property
    def trigger(self) -> str:
        if self.has_tables and self.has_contrast:
            return "both"
        if self.has_tables:
            return "tables"
        if self.has_contrast:
            return "contrast"
        return "plain"


# ----------------------------------------------------------------------
# WCAG contrast maths
# ----------------------------------------------------------------------


def _channel(value: float) -> float:
    value /= 255.0
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: int) -> float:
    return (
        0.2126 * _channel((rgb >> 16) & 0xFF)
        + 0.7152 * _channel((rgb >> 8) & 0xFF)
        + 0.0722 * _channel(rgb & 0xFF)
    )


def contrast_ratio(foreground: int, background: int) -> float:
    high, low = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def _to_rgb_int(colour: Any) -> int | None:
    """Normalise PyMuPDF colours (float grey, or an (r,g,b) 0..1 tuple)."""
    if colour is None:
        return None
    if isinstance(colour, (int, float)):
        grey = int(max(0.0, min(1.0, float(colour))) * 255)
        return (grey << 16) | (grey << 8) | grey
    try:
        r, g, b = (int(max(0.0, min(1.0, float(c))) * 255) for c in tuple(colour)[:3])
    except (TypeError, ValueError):
        return None
    return (r << 16) | (g << 8) | b


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------


def inspect_pdf(path: Path, *, max_pages: int, threshold: float) -> tuple[int, int, int, str]:
    """Return ``(pages_scanned, tables, low_contrast_spans, error)``."""
    import fitz

    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt PDF must not kill the scan
        return 0, 0, 0, f"{type(exc).__name__}: {exc}"

    tables = 0
    low_contrast = 0
    scanned = 0
    try:
        for index in range(min(max_pages, doc.page_count)):
            page = doc[index]
            scanned += 1
            try:
                tables += len(page.find_tables().tables)
            except Exception:  # noqa: BLE001 - find_tables is best-effort
                pass

            fills = []
            try:
                for drawing in page.get_drawings():
                    rect = drawing.get("rect")
                    fill = _to_rgb_int(drawing.get("fill"))
                    if rect is not None and fill is not None:
                        fills.append((rect, fill))
            except Exception:  # noqa: BLE001
                pass

            try:
                blocks = page.get_text("dict").get("blocks", [])
            except Exception:  # noqa: BLE001
                continue
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if len(str(span.get("text") or "").strip()) < MIN_SPAN_CHARS:
                            continue
                        bbox = fitz.Rect(span["bbox"])
                        background = 0xFFFFFF
                        for rect, fill in fills:
                            if rect.contains(bbox):
                                background = fill
                        foreground = span.get("color")
                        if not isinstance(foreground, int):
                            continue
                        if contrast_ratio(foreground, background) < threshold:
                            low_contrast += 1
    except Exception as exc:  # noqa: BLE001
        return scanned, tables, low_contrast, f"{type(exc).__name__}: {exc}"
    finally:
        doc.close()
    return scanned, tables, low_contrast, ""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def _stable_order(
    candidates: list[Candidate], pool_rank: dict[str, int] | None = None
) -> list[Candidate]:
    """Deterministic pseudo-random order (no RNG seed to thread around).

    When ``pool_rank`` is supplied, earlier pools are exhausted before later
    ones; within a pool the order is still content-derived and stable.
    """

    def key(cand: Candidate) -> tuple[int, str]:
        rank = pool_rank.get(cand.pool, len(pool_rank)) if pool_rank else 0
        return rank, hashlib.sha256(cand.doc_id.encode()).hexdigest()

    return sorted(candidates, key=key)


def select(
    candidates: list[Candidate],
    *,
    target: int,
    min_tables: int,
    min_contrast: int,
    pool_rank: dict[str, int] | None = None,
) -> tuple[list[Candidate], list[str]]:
    """Meet the table/contrast quotas, then pad with a representative sample.

    Padding deliberately does **not** drain the richest buckets first: doing so
    yields a corpus made entirely of table+contrast documents, which is not
    representative of the pool and would bias every dimension's pass rate. Once
    the quotas are met, the remainder is drawn in a deterministic pseudo-random
    order across all leftover candidates, preserving the pool's natural mix.
    """
    warnings: list[str] = []
    by_trigger: dict[str, list[Candidate]] = {"both": [], "tables": [], "contrast": [], "plain": []}
    for cand in candidates:
        by_trigger[cand.trigger].append(cand)

    chosen: list[Candidate] = []
    seen: set[str] = set()

    def take(pool: list[Candidate], count: int) -> int:
        taken = 0
        for cand in pool:
            if taken >= count:
                break
            if cand.sha256 in seen:
                continue
            seen.add(cand.sha256)
            chosen.append(cand)
            taken += 1
        return taken

    # Quota phase. Spend pure buckets first so `both` docs stay available to
    # cover whichever quota is short, and so padding stays representative.
    tables_have = take(_stable_order(by_trigger["tables"], pool_rank), min_tables)
    contrast_have = take(_stable_order(by_trigger["contrast"], pool_rank), min_contrast)
    if tables_have < min_tables or contrast_have < min_contrast:
        need = max(min_tables - tables_have, min_contrast - contrast_have)
        filled = take(_stable_order(by_trigger["both"], pool_rank), need)
        tables_have += filled
        contrast_have += filled

    if tables_have < min_tables:
        warnings.append(f"only {tables_have} table-trigger docs available (wanted {min_tables})")
    if contrast_have < min_contrast:
        warnings.append(f"only {contrast_have} contrast-trigger docs available (wanted {min_contrast})")

    # Representative padding across everything still unselected.
    remaining = [c for c in candidates if c.sha256 not in seen]
    take(_stable_order(remaining, pool_rank), max(0, target - len(chosen)))

    if len(chosen) < target:
        warnings.append(f"selected {len(chosen)} docs; target was {target}")
    return chosen[:target], warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pool", type=Path, action="append", default=None, help="Repeatable; defaults to the three clean LAMC pools in order")
    parser.add_argument("--trained-data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--target", type=int, default=180, help="Corpus size to aim for")
    parser.add_argument("--min-tables", type=int, default=20)
    parser.add_argument("--min-contrast", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=3, help="Pages inspected per PDF")
    parser.add_argument("--contrast-threshold", type=float, default=CONTRAST_THRESHOLD)
    parser.add_argument("--scan-limit", type=int, default=0, help="Cap PDFs inspected per pool (0 = all)")
    parser.add_argument(
        "--scan-cache",
        type=Path,
        default=DEFAULT_EVAL_ROOT / "e2e_gate_prep" / "gate_corpus_scan.json",
        help="Where the full scan is cached; reused by --from-cache",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Re-select from --scan-cache without re-inspecting any PDF",
    )
    parser.add_argument(
        "--pad",
        choices=("pool-order", "representative"),
        default="pool-order",
        help="pool-order exhausts --pool entries in order (default); representative samples "
        "across every pool at once, which widens contrast coverage",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_EVAL_ROOT / "e2e_gate_prep" / "gate_corpus_v1.txt")
    args = parser.parse_args()

    pools = [p.expanduser() for p in (args.pool or list(DEFAULT_POOLS))]
    skipped: Counter[str] = Counter()

    if args.from_cache:
        cached = json.loads(args.scan_cache.read_text(encoding="utf-8"))
        candidates = [Candidate(**row) for row in cached["candidates"]]
        skipped = Counter(cached.get("skipped") or {})
        print(f"loaded {len(candidates)} candidates from {args.scan_cache}")
        return _emit(args, pools, candidates, skipped)

    try:
        import fitz  # noqa: F401
    except ImportError:
        print("ERROR: PyMuPDF (fitz) is required: pip install pymupdf", file=sys.stderr)
        return 1
    import fitz

    fitz.TOOLS.mupdf_display_errors(False)  # corrupt-PDF noise is expected and handled

    seen_trained = trained_doc_ids(args.trained_data_root)

    candidates: list[Candidate] = []
    hashes: set[str] = set()

    for pool in pools:
        if not pool.is_dir():
            skipped["missing_pool"] += 1
            continue
        for path in sorted(pool.glob("*.pdf")):
            if args.scan_limit and sum(1 for c in candidates if c.pool == pool.name) >= args.scan_limit:
                break
            doc_id = _normalise_stem(path.stem)
            if not is_real_pdf(path):
                skipped["not_a_pdf"] += 1
                continue
            if doc_id in seen_trained:
                skipped["in_training_split"] += 1
                continue
            digest = sha256_of(path)
            if digest in hashes:
                skipped["duplicate_content"] += 1
                continue
            hashes.add(digest)

            scanned, tables, low, error = inspect_pdf(
                path, max_pages=args.max_pages, threshold=args.contrast_threshold
            )
            if error:
                skipped["inspect_error"] += 1
                continue
            candidates.append(
                Candidate(
                    doc_id=doc_id,
                    path=str(path),
                    pool=pool.name,
                    sha256=digest,
                    pages_scanned=scanned,
                    tables=tables,
                    low_contrast_spans=low,
                )
            )
        print(
            f"scanned pool={pool.name} candidates={sum(1 for c in candidates if c.pool == pool.name)}",
            flush=True,
        )

    args.scan_cache.parent.mkdir(parents=True, exist_ok=True)
    args.scan_cache.write_text(
        json.dumps({"skipped": dict(skipped), "candidates": [asdict(c) for c in candidates]}, indent=2),
        encoding="utf-8",
    )
    print(f"scan cached -> {args.scan_cache}")
    return _emit(args, pools, candidates, skipped)


def _emit(
    args: argparse.Namespace,
    pools: list[Path],
    candidates: list[Candidate],
    skipped: Counter[str],
) -> int:
    pool_rank = (
        {p.name: i for i, p in enumerate(pools)} if args.pad == "pool-order" else None
    )
    chosen, warnings = select(
        candidates,
        target=args.target,
        min_tables=args.min_tables,
        min_contrast=args.min_contrast,
        pool_rank=pool_rank,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(c.path for c in chosen) + "\n", encoding="utf-8")

    trigger_counts = Counter(c.trigger for c in chosen)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pools": [str(p) for p in pools],
        "scanned": len(candidates),
        "selected": len(chosen),
        "skipped": dict(skipped),
        "scanned_by_trigger": dict(Counter(c.trigger for c in candidates)),
        "by_trigger": dict(trigger_counts),
        "coverage": {
            "docs_with_tables": sum(1 for c in chosen if c.has_tables),
            "docs_with_contrast": sum(1 for c in chosen if c.has_contrast),
        },
        "by_pool": dict(Counter(c.pool for c in chosen)),
        "warnings": warnings,
        "settings": {
            "pad": args.pad,
            "target": args.target,
            "min_tables": args.min_tables,
            "min_contrast": args.min_contrast,
            "max_pages": args.max_pages,
            "contrast_threshold": args.contrast_threshold,
        },
        "documents": [asdict(c) for c in chosen],
    }
    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nselected={len(chosen)} -> {args.out}")
    print(f"  pool distribution (scanned): {summary['scanned_by_trigger']}")
    print(f"  corpus by trigger: {dict(trigger_counts)}")
    print(f"  with tables: {summary['coverage']['docs_with_tables']}")
    print(f"  with contrast: {summary['coverage']['docs_with_contrast']}")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    return 0 if not warnings else 2


if __name__ == "__main__":
    raise SystemExit(main())
