#!/usr/bin/env python3
"""Build deterministic v3 MiniCPM training JSONL files from train splits only.

v3 addresses the two v2 failures diagnosed in eval_runs/v2_miss_analysis_h100:

- reading_order: v2 collapsed to pass-all because the v2 builder found zero v1
  train misses and retrained on the same 200 rows. Every existing fail example
  is a single corruption family (whole-list rotation), so v3 synthesizes a
  diverse corruption family (adjacent swaps, section reversal, interleave,
  window shuffle, rotation) from clean pass rows.
- heading: v2 missed the 0.90 exact-correction gate on level-pair exactness
  (H2/H3/H4 confusions), not detection. v3 weights duplication by miss type
  (pair misses heavier than status misses) and is intended to be trained at
  4x/36 high resolution so font-size hierarchy is visible (see
  HIGHRES_TRAINING.md; the training forward path fix is verified).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Any

from build_v2_training_data import (
    add_heading_hint,
    add_task_tag,
    portable_row,
    score_rows,
)
from minicpm_edge.constants import DEFAULT_DATA_ROOT, task_by_key
from minicpm_edge.datasets import load_jsonl, write_jsonl
from minicpm_edge.metrics import record_key


LISTING_HEADER = "Structure tree order:"
LISTING_FOOTER = "Document remediation rules:"
LINE_RE = re.compile(r"^(\s*)(\d+)\.(\s+)(.*)$")

FAIL_SUMMARY = "Reading order is corrupted; restore the delivered structure-tree order."
PASS_SUMMARY_MARK = "matches the delivered gold"

CORRUPTION_DESCRIPTIONS = {
    "rotation": (
        "The tagged reading order starts in a later visual region before earlier "
        "body content, which can make columns, sidebars, or tables read out of sequence."
    ),
    "adjacent_swap": (
        "Adjacent elements are swapped relative to the delivered order, which can "
        "silently reorder neighboring content for screen readers."
    ),
    "section_reverse": (
        "A contiguous section of the tagged order is reversed relative to the "
        "delivered order, so that block reads backwards."
    ),
    "interleave": (
        "Two regions of the tagged order are interleaved, mixing unrelated content "
        "into an alternating sequence."
    ),
    "window_shuffle": (
        "A span of the tagged order is shuffled relative to the delivered order, "
        "scrambling the reading sequence within that region."
    ),
}

TRAIN_CORRUPTIONS = ("adjacent_swap", "section_reverse", "interleave", "window_shuffle", "rotation")
MIN_LISTING_LEN = 6


def assistant_part(row: dict[str, Any]) -> dict[str, Any] | None:
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
            content = message["content"]
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                return part
    return None


def user_text_part(row: dict[str, Any]) -> dict[str, Any] | None:
    for message in row.get("messages", []):
        if message.get("role", "user") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
            content = message["content"]
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                return part
    return None


def is_pass_row(row: dict[str, Any]) -> bool:
    part = assistant_part(row)
    return bool(part) and PASS_SUMMARY_MARK in str(part.get("text") or "")


def parse_listing(text: str) -> tuple[str, list[str], str] | None:
    """Split prompt text into (head, listing lines in display order, tail).

    Returns None when the listing is elided, non-contiguous, or too short to
    corrupt safely.
    """
    if LISTING_HEADER not in text or LISTING_FOOTER not in text:
        return None
    head, rest = text.split(LISTING_HEADER, 1)
    block, tail = rest.split(LISTING_FOOTER, 1)
    if "more elements" in block:
        return None
    lines = [line for line in block.splitlines() if line.strip()]
    parsed: list[tuple[int, str]] = []
    for line in lines:
        match = LINE_RE.match(line)
        if not match:
            return None
        parsed.append((int(match.group(2)), line))
    numbers = [number for number, _ in parsed]
    if len(numbers) < MIN_LISTING_LEN:
        return None
    if numbers != list(range(1, len(numbers) + 1)):
        return None
    return head, [line for _, line in parsed], tail


def rebuild_text(head: str, lines: list[str], tail: str) -> str:
    return head + LISTING_HEADER + "\n" + "\n".join(lines) + "\n\n" + LISTING_FOOTER + tail


def corrupt(lines: list[str], kind: str, rng: random.Random) -> list[str] | None:
    n = len(lines)
    out = list(lines)
    if kind == "rotation":
        k = rng.randint(max(1, n // 4), max(2, (3 * n) // 4))
        out = out[k:] + out[:k]
    elif kind == "adjacent_swap":
        swaps = max(1, n // 10)
        positions = rng.sample(range(n - 1), min(swaps, n - 1))
        for pos in positions:
            out[pos], out[pos + 1] = out[pos + 1], out[pos]
    elif kind == "section_reverse":
        span = max(3, n // 4)
        start = rng.randint(0, n - span)
        out[start : start + span] = reversed(out[start : start + span])
    elif kind == "interleave":
        half = n // 2
        first, second = out[:half], out[half:]
        merged: list[str] = []
        for index in range(max(len(first), len(second))):
            if index < len(second):
                merged.append(second[index])
            if index < len(first):
                merged.append(first[index])
        out = merged
    elif kind == "window_shuffle":
        span = max(4, n // 3)
        start = rng.randint(0, n - span)
        window = out[start : start + span]
        for _ in range(10):
            rng.shuffle(window)
            if window != out[start : start + span]:
                break
        out[start : start + span] = window
    else:
        raise ValueError(f"unknown corruption kind: {kind}")
    if out == list(lines):
        return None
    return out


def fail_target(page_layout: str, kind: str, count: int) -> str:
    order = ", ".join(str(number) for number in range(1, count + 1))
    payload = {
        "page_layout": page_layout,
        "issues": [
            {
                "severity": "error",
                "description": CORRUPTION_DESCRIPTIONS[kind],
                "suggestion": f"Restore the delivered gold reading order: {order}",
            }
        ],
        "summary": FAIL_SUMMARY,
    }
    return json.dumps(payload)


def page_layout_of(row: dict[str, Any]) -> str:
    part = assistant_part(row)
    try:
        parsed = json.loads(str(part.get("text") or "")) if part else {}
    except json.JSONDecodeError:
        parsed = {}
    return str(parsed.get("page_layout") or "unknown_complex")


def synthesize_fail(
    pass_row: dict[str, Any],
    *,
    kind: str,
    variant_index: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    text_part = user_text_part(pass_row)
    if text_part is None:
        return None
    parsed = parse_listing(str(text_part.get("text") or ""))
    if parsed is None:
        return None
    head, lines, tail = parsed
    corrupted = corrupt(lines, kind, rng)
    if corrupted is None:
        return None
    row = copy.deepcopy(pass_row)
    new_text = user_text_part(row)
    assert new_text is not None
    new_text["text"] = rebuild_text(head, corrupted, tail)
    target = assistant_part(row)
    assert target is not None
    target["text"] = fail_target(page_layout_of(pass_row), kind, len(lines))
    meta = row.setdefault("meta", {})
    meta["variant"] = f"synth_{kind}_{variant_index}"
    meta["v3_synthetic"] = True
    return row


def build_reading_order(args: argparse.Namespace) -> int:
    spec = task_by_key("reading_order")
    src_jsonl = args.data_root / spec.local_dir / (args.split + ".jsonl")
    out_dir = args.out.parent
    rows = load_jsonl(src_jsonl)
    rng = random.Random(args.seed)

    out: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    for row in rows:
        built = portable_row(row, src_jsonl=src_jsonl, out_dir=out_dir)
        if args.add_task_tags:
            add_task_tag(built, "reading_order")
        if not args.synthetic_only:
            out.append(built)
        if is_pass_row(built):
            pass_rows.append(built)

    synth_count = 0
    skipped = 0
    kind_counts: dict[str, int] = {}
    for row_index, pass_row in enumerate(pass_rows):
        for variant_index in range(args.variants_per_pass):
            kind = TRAIN_CORRUPTIONS[(row_index + variant_index) % len(TRAIN_CORRUPTIONS)]
            synth = synthesize_fail(
                pass_row,
                kind=kind,
                variant_index=variant_index,
                rng=random.Random((args.seed, row_index, variant_index).__hash__()),
            )
            if synth is None:
                skipped += 1
                continue
            out.append(synth)
            synth_count += 1
            kind_counts[kind] = kind_counts.get(kind, 0) + 1

    # Re-balance pass/fail: duplicate pass rows until pass count matches fail count.
    if args.balance and not args.synthetic_only:
        fails = sum(1 for row in out if not is_pass_row(row))
        passes = [row for row in out if is_pass_row(row)]
        deficit = fails - len(passes)
        index = 0
        while deficit > 0 and passes:
            duplicate = copy.deepcopy(passes[index % len(passes)])
            duplicate.setdefault("meta", {})["v3_duplicate_reason"] = "pass_balance"
            out.append(duplicate)
            deficit -= 1
            index += 1

    rng.shuffle(out)
    write_jsonl(args.out, out)
    print(
        json.dumps(
            {
                "task": "reading_order",
                "split": args.split,
                "source_rows": len(rows),
                "synthetic_fail_rows": synth_count,
                "synthetic_skipped": skipped,
                "corruption_mix": kind_counts,
                "output_rows": len(out),
            },
            sort_keys=True,
        )
    )
    return 0


def heading_miss_kind(score: dict[str, Any] | None) -> str | None:
    if not score:
        return None
    if not score.get("valid_json") or not score.get("status_match") or score.get("pass_false_positive"):
        return "status"
    if not score.get("exact_corrections", True):
        return "pair"
    return None


def build_heading(args: argparse.Namespace) -> int:
    spec = task_by_key("heading")
    src_jsonl = args.data_root / spec.local_dir / "train.jsonl"
    out_dir = args.out.parent
    rows = load_jsonl(src_jsonl)
    scores = score_rows(args.scores)

    out: list[dict[str, Any]] = []
    duplicated = {"status": 0, "pair": 0}
    for index, row in enumerate(rows):
        key = record_key(row, index)
        built = portable_row(row, src_jsonl=src_jsonl, out_dir=out_dir)
        if args.add_task_tags:
            add_task_tag(built, "heading")
        add_heading_hint(built)
        out.append(built)
        kind = heading_miss_kind(scores.get(key))
        if kind is None:
            continue
        copies = args.dup_pair_miss if kind == "pair" else args.dup_status_miss
        for copy_index in range(copies):
            duplicate = copy.deepcopy(built)
            duplicate.setdefault("meta", {})["v3_duplicate_reason"] = f"train_split_v1_{kind}_miss_{copy_index}"
            out.append(duplicate)
            duplicated[kind] += 1

    write_jsonl(args.out, out)
    print(
        json.dumps(
            {
                "task": "heading",
                "source_rows": len(rows),
                "status_miss_duplicates": duplicated["status"],
                "pair_miss_duplicates": duplicated["pair"],
                "output_rows": len(out),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    ro = sub.add_parser("reading-order", help="Build reading-order v3 rows with synthetic corruptions")
    ro.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ro.add_argument("--split", choices=("train", "val"), default="train")
    ro.add_argument("--out", type=Path, required=True)
    ro.add_argument("--variants-per-pass", type=int, default=2)
    ro.add_argument("--seed", type=int, default=3407)
    ro.add_argument("--add-task-tags", action="store_true")
    ro.add_argument("--balance", action="store_true", default=True)
    ro.add_argument("--no-balance", dest="balance", action="store_false")
    ro.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Emit only synthetic fail rows (for building a diagnostic val_hard split)",
    )
    ro.set_defaults(func=build_reading_order)

    heading = sub.add_parser("heading", help="Build heading v3 rows with miss-type-weighted duplication")
    heading.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    heading.add_argument("--scores", type=Path, required=True)
    heading.add_argument("--out", type=Path, required=True)
    heading.add_argument("--dup-status-miss", type=int, default=2)
    heading.add_argument("--dup-pair-miss", type=int, default=3)
    heading.add_argument("--add-task-tags", action="store_true")
    heading.set_defaults(func=build_heading)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
