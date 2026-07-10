#!/usr/bin/env python3
"""Train the MiniCPM Remedy adapter family in the planned promotion order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from minicpm_edge.constants import BASE_MODEL, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, MULTITASK, TASKS, TaskSpec


DEFAULT_ORDER = ("alt", "contrast", "heading", "table", "reading_order")


def spec_by_key(key: str) -> TaskSpec:
    for spec in (*TASKS, MULTITASK):
        if spec.key == key:
            return spec
    raise argparse.ArgumentTypeError(f"unknown task key {key!r}")


def run(cmd: list[str]) -> None:
    print("[train-family] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="*", default=list(DEFAULT_ORDER), type=spec_by_key)
    parser.add_argument("--include-multitask", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--downsample-mode", choices=("16x", "4x"), default="16x")
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    specs = list(args.tasks)
    if args.include_multitask and MULTITASK not in specs:
        specs.append(MULTITASK)

    script = Path(__file__).resolve().parent / "train_lora_minicpm.py"
    for spec in specs:
        train = args.data_root / spec.local_dir / "train.jsonl"
        val = args.data_root / spec.local_dir / "val.jsonl"
        if not train.exists() or not val.exists():
            raise SystemExit(f"missing dataset for {spec.key}: run sync_task_data.py first")
        cmd = [
            sys.executable,
            str(script),
            "--model",
            args.model,
            "--train",
            str(train),
            "--val",
            str(val),
            "--out",
            str(args.output_root / spec.output_dir),
            "--epochs",
            str(args.epochs),
            "--max-steps",
            str(args.max_steps),
            "--rank",
            str(args.rank),
            "--alpha",
            str(args.alpha),
            "--lr",
            str(args.lr),
            "--batch",
            str(args.batch),
            "--grad-accum",
            str(args.grad_accum),
            "--downsample-mode",
            args.downsample_mode,
            "--max-slice-nums",
            str(args.max_slice_nums),
        ]
        if args.qlora:
            cmd.append("--qlora")
        if args.push_to_hub:
            cmd.extend(["--push-to-hub", "--hub-model-id", spec.hub_repo])
        run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
