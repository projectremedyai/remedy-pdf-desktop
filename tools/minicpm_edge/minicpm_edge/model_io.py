"""MiniCPM-V-4.6 model IO helpers shared by train/eval/router scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


LORA_PROJ_KEYS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

HEADING_DEPTH_HINT = (
    "\n\nHeading-depth rule: when a visible heading begins with a dotted section "
    "number, infer nesting depth from the number of components. For example, "
    "3 is higher than 3.3, 3.3 is higher than 3.3.7, and 3.3.7.1 is deeper "
    "than all of them. Preserve that depth in correct_tag."
)

HEADING_PROFILES = {"heading", "heading_hierarchy"}


def _open_image(path: str | Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def materialize_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of messages where image path parts hold PIL images."""
    copied: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image":
                image_value = part.get("image") or part.get("url")
                if image_value is None:
                    parts.append(dict(part))
                elif isinstance(image_value, (str, Path)):
                    parts.append({"type": "image", "image": _open_image(image_value)})
                else:
                    parts.append({"type": "image", "image": image_value})
            else:
                parts.append(dict(part))
        copied.append({"role": message.get("role", "user"), "content": parts})
    return copied


def augment_messages_for_profile(
    messages: list[dict[str, Any]],
    *,
    profile: str,
) -> list[dict[str, Any]]:
    """Apply task-specific inference hints while preserving message shape."""
    if profile not in HEADING_PROFILES:
        return messages
    copied = _copy_messages(messages)
    _append_hint_to_last_user(copied, HEADING_DEPTH_HINT)
    return copied


def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            copied_content: str | list[dict[str, Any]] = content
        else:
            copied_content = [
                dict(part) if isinstance(part, dict) else {"type": "text", "text": str(part)}
                for part in content or []
            ]
        copied.append({"role": message.get("role", "user"), "content": copied_content})
    return copied


def _append_hint_to_last_user(messages: list[dict[str, Any]], hint: str) -> None:
    for message in reversed(messages):
        if message.get("role", "user") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            if hint not in content:
                message["content"] = content + hint
            return
        parts = content if isinstance(content, list) else []
        for part in reversed(parts):
            if not isinstance(part, dict):
                continue
            if part.get("type") not in {"text", "input_text"}:
                continue
            text = str(part.get("text") or "")
            if hint not in text:
                part["text"] = text + hint
            return
        parts.append({"type": "text", "text": hint.strip()})
        message["content"] = parts
        return


def prepare_inputs(
    processor,
    messages: list[dict[str, Any]],
    *,
    downsample_mode: str,
    max_slice_nums: int,
):
    """Build MiniCPM inputs from multimodal chat messages."""
    return processor.apply_chat_template(
        materialize_images(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "downsample_mode": downsample_mode,
            "max_slice_nums": max_slice_nums,
        },
    )


def decode_generated(processor, inputs, generated_ids) -> str:
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return strip_empty_think(text)


def strip_empty_think(text: str) -> str:
    """Remove a leading MiniCPM/Qwen-style think block if it leaks into output."""
    raw = str(text or "").strip()
    raw = re.sub(r"^<think>\s*</think>\s*", "", raw, flags=re.DOTALL)
    return raw.strip()


def lora_target_modules(model, *, tune_vision: bool) -> list[str]:
    """Return exact Linear module names for LoRA, text tower by default."""
    import torch.nn as nn

    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(key in name for key in LORA_PROJ_KEYS):
            continue
        is_vision = "vision" in name.lower() or "visual" in name.lower()
        if is_vision and not tune_vision:
            continue
        targets.append(name)
    return targets


def bnb_config(qlora: bool):
    if not qlora:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
