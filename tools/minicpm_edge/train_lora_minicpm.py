#!/usr/bin/env python3
"""Train a PEFT LoRA adapter for MiniCPM-V-4.6 on Remedy conversation JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from minicpm_edge.constants import BASE_MODEL
from minicpm_edge.datasets import load_conversations
from minicpm_edge.model_io import (
    bnb_config,
    lora_target_modules,
    materialize_images,
)


class MiniCPMCollator:
    """Tokenize full conversations and mask the prompt portion of labels."""

    def __init__(self, processor, *, downsample_mode: str, max_slice_nums: int) -> None:
        self.processor = processor
        self.downsample_mode = downsample_mode
        self.max_slice_nums = max_slice_nums

    def _encode(self, messages, *, add_generation_prompt: bool):
        return self.processor.apply_chat_template(
            materialize_images(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "downsample_mode": self.downsample_mode,
                "max_slice_nums": self.max_slice_nums,
            },
        )

    def __call__(self, batch):
        import torch

        encoded_rows = []
        input_ids_list = []
        labels_list = []
        for rec in batch:
            messages = rec["messages"]
            full = self._encode(messages, add_generation_prompt=False)
            prompt = self._encode(messages[:-1], add_generation_prompt=True)
            ids = full["input_ids"][0]
            labels = ids.clone()
            labels[: prompt["input_ids"].shape[1]] = -100
            encoded_rows.append(full)
            input_ids_list.append(ids)
            labels_list.append(labels)

        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        max_len = max(ids.shape[0] for ids in input_ids_list)
        input_ids = []
        attention_mask = []
        labels = []
        for ids, row_labels in zip(input_ids_list, labels_list, strict=False):
            pad_n = max_len - ids.shape[0]
            input_ids.append(torch.cat([ids, torch.full((pad_n,), pad_id, dtype=ids.dtype)]))
            attention_mask.append(
                torch.cat(
                    [
                        torch.ones(ids.shape[0], dtype=torch.long),
                        torch.zeros(pad_n, dtype=torch.long),
                    ]
                )
            )
            labels.append(torch.cat([row_labels, torch.full((pad_n,), -100, dtype=row_labels.dtype)]))

        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            # MiniCPM-V 4.6 needs the same visual token downsampling mode in
            # the model forward call that the processor used for placeholders.
            "downsample_mode": self.downsample_mode,
        }
        for key in encoded_rows[0].keys():
            if key in {"input_ids", "attention_mask"}:
                continue
            values = [row[key] for row in encoded_rows]
            if torch.is_tensor(values[0]):
                try:
                    out[key] = torch.cat(values, dim=0)
                except Exception:
                    out[key] = torch.stack(values)
            else:
                flattened = []
                for value in values:
                    if isinstance(value, list):
                        flattened.extend(value)
                    else:
                        flattened.append(value)
                out[key] = flattened
        return out


def attach_or_create_lora(
    model,
    *,
    init_adapter: Path | None,
    rank: int,
    alpha: int,
    dropout: float,
    tune_vision: bool,
):
    if init_adapter is not None:
        from peft import PeftModel

        print(f"[minicpm-train] continuing from adapter={init_adapter}", flush=True)
        return PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)

    from peft import LoraConfig, get_peft_model

    targets = lora_target_modules(model, tune_vision=tune_vision)
    if not targets:
        raise RuntimeError("No LoRA target modules found; inspect MiniCPM module names")
    print(
        f"[minicpm-train] LoRA targets={len(targets)} "
        f"({'including' if tune_vision else 'excluding'} vision tower)",
        flush=True,
    )
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    return get_peft_model(model, config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--downsample-mode", choices=("16x", "4x"), default="16x")
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--tune-vision-layers", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--init-adapter", type=Path, default=None)
    parser.add_argument("--hub-model-id", default="")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    import os
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(
        f"[minicpm-train] model={args.model} rank={args.rank} "
        f"qlora={args.qlora} device={device}",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(args.model)
    quant = bnb_config(args.qlora)
    if quant is None:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=args.attn_implementation,
        )
        model.config.use_cache = False
    else:
        from peft import prepare_model_for_kbit_training

        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            quantization_config=quant,
            device_map="auto",
            attn_implementation=args.attn_implementation,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    model = attach_or_create_lora(
        model,
        init_adapter=args.init_adapter,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        tune_vision=args.tune_vision_layers,
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_rows = load_conversations(args.train)
    val_rows = load_conversations(args.val) if args.val else None
    print(
        f"[minicpm-train] train={len(train_rows)}"
        + (f" val={len(val_rows)}" if val_rows is not None else ""),
        flush=True,
    )

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        remove_unused_columns=False,
        save_strategy="no",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id or None,
        hub_private_repo=True if args.push_to_hub else None,
        hub_token=hf_token or None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_rows,
        eval_dataset=val_rows,
        data_collator=MiniCPMCollator(
            processor,
            downsample_mode=args.downsample_mode,
            max_slice_nums=args.max_slice_nums,
        ),
    )
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    processor.save_pretrained(str(args.out))
    if args.push_to_hub:
        trainer.push_to_hub()
    print(f"[minicpm-train] saved adapter -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
