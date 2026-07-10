#!/usr/bin/env python3
"""Score MiniCPM Remedy task predictions against conversation JSONL targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicpm_edge.datasets import load_jsonl
from minicpm_edge.metrics import load_predictions, score_predictions, target_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--gold-as-predictions", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scores-out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    gold_rows = load_jsonl(args.val)
    if args.limit > 0:
        gold_rows = gold_rows[: args.limit]
    if args.gold_as_predictions:
        preds = {str(i): target_text(row) for i, row in enumerate(gold_rows)}
    elif args.predictions is not None:
        preds = load_predictions(args.predictions, gold_rows)
    else:
        raise SystemExit("pass --predictions or --gold-as-predictions")

    scores, summary = score_predictions(gold_rows, preds)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.scores_out:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        args.scores_out.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scores),
            encoding="utf-8",
        )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
