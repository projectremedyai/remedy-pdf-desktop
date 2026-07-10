#!/usr/bin/env python3
"""Run a one-batch MiniCPM training forward probe.

This is a fast GPU-side diagnostic for the high-resolution training path. It
uses the same collator as train_lora_minicpm.py, then performs one model forward
without running an optimizer step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minicpm_edge.constants import BASE_MODEL
from minicpm_edge.datasets import load_conversations
from train_lora_minicpm import MiniCPMCollator


def _shape(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, list):
        return {"type": "list", "len": len(value)}
    return {"type": type(value).__name__, "value": value}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--downsample-mode", choices=("16x", "4x"), default="4x")
    parser.add_argument("--max-slice-nums", type=int, default=36)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--omit-forward-downsample",
        action="store_true",
        help="Drop downsample_mode before forward to reproduce the old failure path.",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this probe; run it on the GPU workbench.")

    processor = AutoProcessor.from_pretrained(args.model)
    rows = load_conversations(args.train)
    if args.index < 0 or args.index >= len(rows):
        raise SystemExit(f"--index must be in [0, {len(rows) - 1}]")

    collator = MiniCPMCollator(
        processor,
        downsample_mode=args.downsample_mode,
        max_slice_nums=args.max_slice_nums,
    )
    batch = collator([rows[args.index]])
    if args.omit_forward_downsample:
        batch.pop("downsample_mode", None)

    print(
        json.dumps(
            {
                "mode": args.downsample_mode,
                "max_slice_nums": args.max_slice_nums,
                "omit_forward_downsample": args.omit_forward_downsample,
                "batch": {key: _shape(value) for key, value in batch.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.train()

    device = model.device
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)

    with torch.no_grad():
        outputs = model(**batch)
    loss = getattr(outputs, "loss", None)
    print(
        json.dumps(
            {
                "forward_ok": True,
                "loss": float(loss.detach().cpu()) if loss is not None else None,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
