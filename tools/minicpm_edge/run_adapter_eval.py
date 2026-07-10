#!/usr/bin/env python3
"""Evaluate one MiniCPM PEFT adapter on a Remedy task split."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from minicpm_edge.constants import BASE_MODEL, DEFAULT_DATA_ROOT, DEFAULT_EVAL_ROOT, TASKS, TaskSpec


def spec_by_key(key: str) -> TaskSpec:
    for spec in TASKS:
        if spec.key == key:
            return spec
    raise argparse.ArgumentTypeError(f"unknown task key {key!r}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(cmd: list[str]) -> None:
    print("[adapter-eval] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def summarize_predictions(path: Path) -> dict:
    rows = load_jsonl(path)
    latencies = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    peak = [float(row["peak_memory_mb"]) for row in rows if row.get("peak_memory_mb") is not None]
    return {
        "latency_seconds_avg": round(statistics.mean(latencies), 4) if latencies else None,
        "latency_seconds_p95": round(statistics.quantiles(latencies, n=20)[18], 4)
        if len(latencies) >= 20 else None,
        "peak_memory_mb_max": round(max(peak), 1) if peak else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--task", required=True, type=spec_by_key)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT / "adapter_eval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--downsample-mode", choices=("16x", "4x"), default="16x")
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    spec = args.task
    script_dir = Path(__file__).resolve().parent
    args.eval_root.mkdir(parents=True, exist_ok=True)
    data = args.data_root / spec.local_dir / f"{args.split}.jsonl"
    pred = args.eval_root / f"{spec.key}.predictions.jsonl"
    metrics = args.eval_root / f"{spec.key}.metrics.json"
    scores = args.eval_root / f"{spec.key}.scores.jsonl"

    run(
        [
            sys.executable,
            str(script_dir / "generate_predictions_minicpm.py"),
            "--model",
            args.model,
            "--adapter",
            str(args.adapter),
            "--val",
            str(data),
            "--out",
            str(pred),
            "--profile",
            spec.key,
            "--limit",
            str(args.limit),
            "--downsample-mode",
            args.downsample_mode,
            "--max-slice-nums",
            str(args.max_slice_nums),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
    )
    run(
        [
            sys.executable,
            str(script_dir / "eval_task_metrics.py"),
            "--val",
            str(data),
            "--predictions",
            str(pred),
            "--out",
            str(metrics),
            "--scores-out",
            str(scores),
            "--limit",
            str(args.limit),
        ]
    )
    summary = {
        "model": args.model,
        "adapter": str(args.adapter),
        "task": spec.task_name,
        "split": args.split,
        "downsample_mode": args.downsample_mode,
        "max_slice_nums": args.max_slice_nums,
        "limit": args.limit,
        "metrics": load_json(metrics),
        **summarize_predictions(pred),
    }
    (args.eval_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
