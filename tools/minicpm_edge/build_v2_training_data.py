#!/usr/bin/env python3
"""Build deterministic v2 MiniCPM training JSONL files from train splits only."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

from minicpm_edge.constants import DEFAULT_DATA_ROOT, TASKS, TaskSpec, task_by_key
from minicpm_edge.datasets import load_jsonl, resolve_image_path, write_jsonl
from minicpm_edge.metrics import record_key
from minicpm_edge.model_io import HEADING_DEPTH_HINT


TASK_TAGS = {
    "alt": "Task: alt_text_quality",
    "contrast": "Task: contrast",
    "heading": "Task: heading_hierarchy",
    "reading_order": "Task: reading_order",
    "table": "Task: table_structure",
}

MULTITASK_MIX = {
    "heading": 1400,
    "reading_order": 1000,
    "table": 600,
    "alt": 600,
    "contrast": 400,
}


def score_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    return {str(row.get("example_id")): row for row in rows}


def is_hard_miss(task_key: str, score: dict[str, Any] | None) -> bool:
    if not score:
        return False
    if not score.get("valid_json"):
        return True
    if not score.get("status_match"):
        return True
    if score.get("pass_false_positive"):
        return True
    if task_key == "heading":
        return not bool(score.get("exact_corrections", True))
    if task_key == "reading_order":
        corrected = score.get("corrected_order_match")
        return corrected is False
    return False


def user_text_parts(row: dict[str, Any]) -> list[dict[str, Any]]:
    for message in row.get("messages", []):
        if message.get("role", "user") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
            content = message["content"]
        return [
            part
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
        ]
    return []


def add_task_tag(row: dict[str, Any], task_key: str) -> None:
    parts = user_text_parts(row)
    if not parts:
        return
    tag = TASK_TAGS[task_key]
    text = str(parts[0].get("text") or "")
    if not text.startswith(tag):
        parts[0]["text"] = f"{tag}\n\n{text}"


def add_heading_hint(row: dict[str, Any]) -> None:
    parts = user_text_parts(row)
    if not parts:
        return
    text = str(parts[-1].get("text") or "")
    if HEADING_DEPTH_HINT not in text:
        parts[-1]["text"] = text + HEADING_DEPTH_HINT


def portable_row(row: dict[str, Any], *, src_jsonl: Path, out_dir: Path) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    src_dir = src_jsonl.resolve().parent
    for message in copied.get("messages", []):
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            image = part.get("image")
            if not isinstance(image, str):
                continue
            absolute = resolve_image_path(Path(image), src_dir)
            part["image"] = os.path.relpath(absolute, out_dir.resolve())
    return copied


def build_task_rows(
    spec: TaskSpec,
    *,
    data_root: Path,
    out_dir: Path,
    scores_path: Path | None,
    add_tags: bool,
    duplicate_hard_misses: bool,
) -> list[dict[str, Any]]:
    src_jsonl = data_root / spec.local_dir / "train.jsonl"
    rows = load_jsonl(src_jsonl)
    scores = score_rows(scores_path)
    out: list[dict[str, Any]] = []
    hard_count = 0
    for index, row in enumerate(rows):
        key = record_key(row, index)
        built = portable_row(row, src_jsonl=src_jsonl, out_dir=out_dir)
        if add_tags:
            add_task_tag(built, spec.key)
        if spec.key == "heading":
            add_heading_hint(built)
        out.append(built)
        if duplicate_hard_misses and is_hard_miss(spec.key, scores.get(key)):
            duplicate = copy.deepcopy(built)
            duplicate.setdefault("meta", {})["v2_duplicate_reason"] = "train_split_v1_hard_miss"
            out.append(duplicate)
            hard_count += 1
    print(
        json.dumps(
            {
                "task": spec.key,
                "source_rows": len(rows),
                "hard_miss_duplicates": hard_count,
                "output_rows": len(out),
            },
            sort_keys=True,
        )
    )
    return out


def build_task(args: argparse.Namespace) -> int:
    spec = task_by_key(args.task)
    out_dir = args.out.parent
    rows = build_task_rows(
        spec,
        data_root=args.data_root,
        out_dir=out_dir,
        scores_path=args.scores,
        add_tags=args.add_task_tags,
        duplicate_hard_misses=True,
    )
    write_jsonl(args.out, rows)
    return 0


def sample_with_replacement(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("cannot sample from an empty row set")
    return [copy.deepcopy(rng.choice(rows)) for _ in range(count)]


def load_rows_for_mix(
    key: str,
    *,
    data_root: Path,
    out_dir: Path,
    heading_v2: Path | None,
    reading_order_v2: Path | None,
) -> list[dict[str, Any]]:
    override = {"heading": heading_v2, "reading_order": reading_order_v2}.get(key)
    if override:
        return load_jsonl(override)
    spec = task_by_key(key)
    src_jsonl = data_root / spec.local_dir / "train.jsonl"
    rows = [
        portable_row(row, src_jsonl=src_jsonl, out_dir=out_dir)
        for row in load_jsonl(src_jsonl)
    ]
    for row in rows:
        add_task_tag(row, key)
    return rows


def build_multitask(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    out_dir = args.out.parent
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for key, count in MULTITASK_MIX.items():
        task_rows = load_rows_for_mix(
            key,
            data_root=args.data_root,
            out_dir=out_dir,
            heading_v2=args.heading_v2,
            reading_order_v2=args.reading_order_v2,
        )
        for row in task_rows:
            add_task_tag(row, key)
            if key == "heading":
                add_heading_hint(row)
        sampled = sample_with_replacement(task_rows, count, rng)
        rows.extend(sampled)
        counts[key] = len(sampled)
    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    print(json.dumps({"mix": counts, "output_rows": len(rows), "seed": args.seed}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    task = sub.add_parser("task", help="Build heading or reading-order v2 rows")
    task.add_argument("--task", choices=("heading", "reading_order"), required=True)
    task.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    task.add_argument("--scores", type=Path, default=None)
    task.add_argument("--out", type=Path, required=True)
    task.add_argument("--add-task-tags", action="store_true")
    task.set_defaults(func=build_task)

    multi = sub.add_parser("multitask", help="Build the fixed 4000-row multitask v2 mix")
    multi.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    multi.add_argument("--heading-v2", type=Path, default=None)
    multi.add_argument("--reading-order-v2", type=Path, default=None)
    multi.add_argument("--out", type=Path, required=True)
    multi.add_argument("--seed", type=int, default=3407)
    multi.set_defaults(func=build_multitask)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
