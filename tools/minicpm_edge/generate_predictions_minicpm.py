#!/usr/bin/env python3
"""Generate Remedy task predictions with base or PEFT-adapted MiniCPM-V-4.6."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from minicpm_edge.constants import BASE_MODEL
from minicpm_edge.datasets import load_conversations
from minicpm_edge.metrics import record_key, task_name
from minicpm_edge.model_io import augment_messages_for_profile, decode_generated, prepare_inputs


def prediction_row(
    rec: dict[str, Any],
    index: int,
    prediction: str,
    *,
    latency_seconds: float,
    peak_memory_mb: float | None,
    profile: str,
) -> dict[str, Any]:
    return {
        "example_id": record_key(rec, index),
        "task": task_name(rec),
        "prediction": prediction,
        "latency_seconds": round(latency_seconds, 4),
        "peak_memory_mb": round(peak_memory_mb, 1) if peak_memory_mb is not None else None,
        "profile": profile,
        "meta": rec.get("meta") or {},
    }


def generate_one(
    model,
    processor,
    rec: dict[str, Any],
    *,
    max_new_tokens: int,
    downsample_mode: str,
    max_slice_nums: int,
    profile: str,
) -> str:
    user_messages = augment_messages_for_profile(rec["messages"][:-1], profile=profile)
    inputs = prepare_inputs(
        processor,
        user_messages,
        downsample_mode=downsample_mode,
        max_slice_nums=max_slice_nums,
    ).to(model.device)
    out = model.generate(
        **inputs,
        downsample_mode=downsample_mode,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return decode_generated(processor, inputs, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--downsample-mode", choices=("16x", "4x"), default="16x")
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--profile", default="base")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = load_conversations(args.val, args.limit)
    print(f"[minicpm-predict] records={len(rows)} model={args.model}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
    )
    if args.adapter is not None:
        from peft import PeftModel

        print(f"[minicpm-predict] attaching adapter={args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.config.use_cache = True
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for index, rec in enumerate(rows):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            prediction = generate_one(
                model,
                processor,
                rec,
                max_new_tokens=args.max_new_tokens,
                downsample_mode=args.downsample_mode,
                max_slice_nums=args.max_slice_nums,
                profile=args.profile,
            )
            elapsed = time.perf_counter() - started
            peak_mb = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else None
            )
            handle.write(
                json.dumps(
                    prediction_row(
                        rec,
                        index,
                        prediction,
                        latency_seconds=elapsed,
                        peak_memory_mb=peak_mb,
                        profile=args.profile,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            print(f"[minicpm-predict] {index + 1}/{len(rows)} done", flush=True)
    print(f"[minicpm-predict] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
