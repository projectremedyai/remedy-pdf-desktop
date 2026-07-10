#!/usr/bin/env python3
"""Compare v1 and v2 MiniCPM eval runs and summarize misses."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from minicpm_edge.constants import DEFAULT_DATA_ROOT, DEFAULT_EVAL_ROOT
from minicpm_edge.datasets import load_jsonl
from minicpm_edge.metrics import (
    heading_pairs,
    normalized_status,
    parse_jsonish,
    prediction_text,
    record_key,
    target_text,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    for index, row in enumerate(rows):
        row["_index"] = index
        row["_text"] = prediction_text(row)
        row["_parsed"] = parse_jsonish(row["_text"])
    return rows


def _score_rows(path: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    for index, row in enumerate(rows):
        row["_index"] = index
    return rows


def _safe_excerpt(text: str, limit: int = 420) -> str:
    clean = " ".join(str(text or "").split())
    return clean[:limit] + ("..." if len(clean) > limit else "")


def _pairs_json(pairs: set[tuple[int, str]]) -> list[dict[str, Any]]:
    return [{"element_index": index, "correct_tag": tag} for index, tag in sorted(pairs)]


def _class_balance(rows: list[dict[str, Any]], task: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        parsed = parse_jsonish(target_text(row))
        counts[str(normalized_status(parsed, task))] += 1
    return dict(sorted(counts.items()))


def _heading_analysis(
    *,
    val_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    v1_scores: list[dict[str, Any]],
    v2_scores: list[dict[str, Any]],
    v1_preds: list[dict[str, Any]],
    v2_preds: list[dict[str, Any]],
) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    tag_confusions: Counter[str] = Counter()

    for index, row in enumerate(val_rows):
        gold = parse_jsonish(target_text(row))
        gold_pairs = heading_pairs(gold)
        old_pairs = heading_pairs(v1_preds[index]["_parsed"])
        new_pairs = heading_pairs(v2_preds[index]["_parsed"])
        old_exact = bool(v1_scores[index].get("exact_corrections"))
        new_exact = bool(v2_scores[index].get("exact_corrections"))
        old_status = bool(v1_scores[index].get("status_match"))
        new_status = bool(v2_scores[index].get("status_match"))

        if old_exact and new_exact:
            category = "both_exact"
        elif old_exact and not new_exact:
            category = "v2_regression"
        elif not old_exact and new_exact:
            category = "v2_improvement"
        else:
            category = "both_miss"
        if old_status and not new_status:
            category += "_status_regression"
        categories[category] += 1

        missing = gold_pairs - new_pairs
        extra = new_pairs - gold_pairs
        for gold_index, gold_tag in missing:
            guessed = [tag for idx, tag in new_pairs if idx == gold_index]
            tag_confusions[f"{gold_tag}->{guessed[0] if guessed else 'missing'}"] += 1

        if category != "both_exact":
            user_text = next(
                (
                    str(part.get("text") or "")
                    for part in row["messages"][0].get("content", [])
                    if isinstance(part, dict) and part.get("type") == "text"
                ),
                "",
            )
            examples.append(
                {
                    "index": index,
                    "example_id": record_key(row, index),
                    "category": category,
                    "gold_status": v2_scores[index].get("gold_status"),
                    "v1_status": v1_scores[index].get("pred_status"),
                    "v2_status": v2_scores[index].get("pred_status"),
                    "gold_pairs": _pairs_json(gold_pairs),
                    "v1_pairs": _pairs_json(old_pairs),
                    "v2_pairs": _pairs_json(new_pairs),
                    "v2_missing_pairs": _pairs_json(missing),
                    "v2_extra_pairs": _pairs_json(extra),
                    "prompt_excerpt": _safe_excerpt(user_text),
                    "v1_prediction_excerpt": _safe_excerpt(v1_preds[index]["_text"]),
                    "v2_prediction_excerpt": _safe_excerpt(v2_preds[index]["_text"]),
                }
            )

    return {
        "task": "heading_hierarchy",
        "train_balance": _class_balance(train_rows, "heading_hierarchy"),
        "val_balance": _class_balance(val_rows, "heading_hierarchy"),
        "categories": dict(sorted(categories.items())),
        "tag_confusions": dict(tag_confusions.most_common()),
        "examples": examples,
        "diagnosis": [
            "Heading v2 did not fail because of JSON validity or broad status classification.",
            "The remaining gap is correction-pair exactness: wrong or missing element_index/correct_tag pairs.",
            "The v2 data duplication improved some misses but introduced regressions, so the next heading run should target correction-pair supervision rather than more generic examples.",
        ],
    }


def _reading_order_analysis(
    *,
    val_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    v1_scores: list[dict[str, Any]],
    v2_scores: list[dict[str, Any]],
    v1_preds: list[dict[str, Any]],
    v2_preds: list[dict[str, Any]],
) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for index, row in enumerate(val_rows):
        old_status = v1_scores[index].get("pred_status")
        new_status = v2_scores[index].get("pred_status")
        gold_status = v2_scores[index].get("gold_status")
        category = f"{gold_status}:v1_{old_status}:v2_{new_status}"
        categories[category] += 1
        if bool(v1_scores[index].get("status_match")) != bool(v2_scores[index].get("status_match")):
            user_text = next(
                (
                    str(part.get("text") or "")
                    for part in row["messages"][0].get("content", [])
                    if isinstance(part, dict) and part.get("type") == "text"
                ),
                "",
            )
            examples.append(
                {
                    "index": index,
                    "example_id": record_key(row, index),
                    "category": category,
                    "prompt_excerpt": _safe_excerpt(user_text),
                    "v1_prediction_excerpt": _safe_excerpt(v1_preds[index]["_text"]),
                    "v2_prediction_excerpt": _safe_excerpt(v2_preds[index]["_text"]),
                }
            )

    return {
        "task": "reading_order",
        "train_balance": _class_balance(train_rows, "reading_order"),
        "val_balance": _class_balance(val_rows, "reading_order"),
        "categories": dict(sorted(categories.items())),
        "examples": examples,
        "diagnosis": [
            "Reading-order v2 is a pass-all regression on validation: every fail example became pass.",
            "The v2 builder found zero v1 train misses to duplicate, so it produced the same 200-row split with a fresh rank-16 adapter.",
            "The next reading-order run should not be another blind rank increase; it needs newly mined train-only hard negatives or synthetic corruptions that force fail detection.",
        ],
    }


def _write_markdown(summary: dict[str, Any], out: Path) -> None:
    lines = [
        "# MiniCPM V2 Miss Analysis",
        "",
        "Generated from local validation rows, score JSONL, and prediction JSONL.",
        "",
    ]
    for task in summary["tasks"]:
        lines.extend(
            [
                f"## {task['task']}",
                "",
                f"- Train balance: `{task['train_balance']}`",
                f"- Val balance: `{task['val_balance']}`",
                f"- Categories: `{task['categories']}`",
                "",
                "Diagnosis:",
            ]
        )
        for item in task["diagnosis"]:
            lines.append(f"- {item}")
        if task.get("tag_confusions"):
            top = dict(list(task["tag_confusions"].items())[:10])
            lines.extend(["", f"Top heading tag confusions: `{top}`"])
        lines.extend(["", "Representative regressions/misses:"])
        for example in task["examples"][:10]:
            lines.extend(
                [
                    "",
                    f"- `{example['example_id']}` ({example['category']})",
                    f"  - Prompt: {example['prompt_excerpt']}",
                    f"  - v1: {example['v1_prediction_excerpt']}",
                    f"  - v2: {example['v2_prediction_excerpt']}",
                ]
            )
            if "v2_missing_pairs" in example:
                lines.append(f"  - v2 missing pairs: `{example['v2_missing_pairs']}`")
                lines.append(f"  - v2 extra pairs: `{example['v2_extra_pairs']}`")
        lines.append("")
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EVAL_ROOT / "v2_miss_analysis_h100")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    heading = _heading_analysis(
        val_rows=load_jsonl(args.data_root / "heading_hierarchy" / "val.jsonl"),
        train_rows=load_jsonl(args.data_root / "heading_hierarchy" / "train.jsonl"),
        v1_scores=_score_rows(args.eval_root / "heading_v1_patched" / "heading.scores.jsonl"),
        v2_scores=_score_rows(args.eval_root / "heading_v2_notag_16x1_h100" / "heading.scores.jsonl"),
        v1_preds=_prediction_rows(args.eval_root / "heading_v1_patched" / "heading.predictions.jsonl"),
        v2_preds=_prediction_rows(args.eval_root / "heading_v2_notag_16x1_h100" / "heading.predictions.jsonl"),
    )
    reading_order = _reading_order_analysis(
        val_rows=load_jsonl(args.data_root / "reading_order" / "val.jsonl"),
        train_rows=load_jsonl(args.data_root / "reading_order" / "train.jsonl"),
        v1_scores=_score_rows(args.eval_root / "reading_order_v1" / "reading_order.scores.jsonl"),
        v2_scores=_score_rows(args.eval_root / "reading_order_v2_16x1_h100" / "reading_order.scores.jsonl"),
        v1_preds=_prediction_rows(args.eval_root / "reading_order_v1" / "reading_order.predictions.jsonl"),
        v2_preds=_prediction_rows(args.eval_root / "reading_order_v2_16x1_h100" / "reading_order.predictions.jsonl"),
    )
    summary = {
        "tasks": [heading, reading_order],
        "recommendations": [
            "Keep v1 aliases as desktop defaults.",
            "Do not build multitask v2 from these candidates.",
            "For heading v3, train against correction-pair exactness and inspect H2/H3/H4 misses before changing rank again.",
            "For reading-order v3, add train-only hard negatives/corruptions because v2 had no mined misses and regressed to pass-all.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, out_dir / "report.md")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
