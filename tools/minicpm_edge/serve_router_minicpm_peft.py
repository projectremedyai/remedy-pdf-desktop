#!/usr/bin/env python3
"""Serve MiniCPM-V-4.6 PEFT adapters behind an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from minicpm_edge.constants import (
    ALIASES,
    BASE_MODEL,
    DEFAULT_OUTPUT_ROOT,
    LEGACY_ALIASES,
    MULTITASK,
    STABLE_ALIAS,
    TASK_MODEL_MAP,
    TASKS,
)
from minicpm_edge.model_io import (
    augment_messages_for_profile,
    decode_generated,
    strip_empty_think,
)


def _decode_image_url(url: str):
    from PIL import Image

    if not url.startswith("data:") or ";base64," not in url:
        raise ValueError("only data:...;base64 image URLs are supported")
    _prefix, encoded = url.split(";base64,", 1)
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, str):
            converted.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        parts: list[dict[str, Any]] = []
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append({"type": "text", "text": str(part.get("text") or "")})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url") or {}
                parts.append(
                    {
                        "type": "image",
                        "image": _decode_image_url(str(image_url.get("url") or "")),
                    }
                )
            elif part.get("type") == "image":
                parts.append(dict(part))
        converted.append({"role": role, "content": parts})
    return converted


def router_env(base_url: str) -> str:
    return "\n".join(
        [
            "OLLAMA_API_KEY=dummy",
            f"OLLAMA_BASE_URL={base_url.rstrip('/')}",
            f"VISION_BASE_URL={base_url.rstrip('/')}",
            f"OLLAMA_VISION_MODEL={STABLE_ALIAS}",
            "OLLAMA_VISION_TASK_MODELS="
            + ",".join(f"{task}:{model}" for task, model in TASK_MODEL_MAP.items()),
            "OLLAMA_VISION_TASK_BASE_URLS=",
            "OLLAMA_VISION_ROUTER_ALLOW_FALLBACK=0",
            "OLLAMA_VISION_MAX_INFLIGHT=1",
            "OLLAMA_VISION_GATE_TIMEOUT_SECONDS=600",
            "OLLAMA_VISION_MAX_TOKENS=1024",
        ]
    )


class RouterState:
    def __init__(
        self,
        *,
        base_model: str,
        adapters: dict[str, Path],
        aliases: dict[str, str],
        downsample_mode: str,
        max_slice_nums: int,
        adapter_image_settings: dict[str, tuple[str, int]],
        attn_implementation: str,
        device: str = "cuda",
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.lock = threading.Lock()
        self.aliases = aliases
        self.model_names = tuple(aliases)
        self.downsample_mode = downsample_mode
        self.max_slice_nums = max_slice_nums
        self.adapter_image_settings = adapter_image_settings
        self.processor = AutoProcessor.from_pretrained(base_model)
        # CUDA keeps the original accelerate device_map + bf16 path so GPU
        # workbenches are unaffected. mps/cpu load unmapped, then move.
        if device == "cuda":
            base = AutoModelForImageTextToText.from_pretrained(
                base_model,
                dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation=attn_implementation,
            )
        else:
            dtype = torch.float16 if device == "mps" else torch.float32
            print(f"[minicpm-router] loading base on device={device} dtype={dtype}", flush=True)
            base = AutoModelForImageTextToText.from_pretrained(
                base_model,
                dtype=dtype,
                device_map=None,
                attn_implementation=attn_implementation,
            ).to(device)
        self.has_adapters = bool(adapters)
        if adapters:
            first_alias, first_adapter = next(iter(adapters.items()))
            print(
                f"[minicpm-router] attaching adapter={first_alias} path={first_adapter}",
                flush=True,
            )
            self.model = PeftModel.from_pretrained(
                base,
                str(first_adapter),
                adapter_name=first_alias,
            )
            for adapter_name, path in list(adapters.items())[1:]:
                print(
                    f"[minicpm-router] loading adapter={adapter_name} path={path}",
                    flush=True,
                )
                self.model.load_adapter(str(path), adapter_name=adapter_name)
        else:
            self.model = base
        self.model.config.use_cache = True
        self.model.eval()

    def generate(self, payload: dict[str, Any]) -> tuple[str, str]:
        import torch

        requested_model = str(payload.get("model") or STABLE_ALIAS)
        adapter_name = self.aliases.get(requested_model)
        if adapter_name is None:
            known = ", ".join(self.model_names)
            raise ValueError(f"unknown model {requested_model!r}; known models: {known}")

        messages = convert_messages(payload.get("messages") or [])
        if not messages:
            raise ValueError("messages must include at least one message")
        messages = augment_messages_for_profile(messages, profile=adapter_name)
        default_max_tokens = 1024 if adapter_name == "heading" else 768
        max_tokens = int(
            payload.get("max_tokens") or payload.get("max_new_tokens") or default_max_tokens
        )
        downsample_mode, max_slice_nums = self.adapter_image_settings.get(
            adapter_name,
            (self.downsample_mode, self.max_slice_nums),
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "downsample_mode": downsample_mode,
                "max_slice_nums": max_slice_nums,
            },
        ).to(self.model.device)
        with self.lock, torch.inference_mode():
            if self.has_adapters:
                self.model.set_adapter(adapter_name)
            out = self.model.generate(
                **inputs,
                downsample_mode=downsample_mode,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        return requested_model, strip_empty_think(decode_generated(self.processor, inputs, out))


class OpenAIHandler(BaseHTTPRequestHandler):
    server_version = "RemedyMiniCPMRouter/0.1"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        state: RouterState = self.server.router_state  # type: ignore[attr-defined]
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok", "models": list(state.model_names)})
            return
        if self.path in {"/v1/models", "/models"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "project-remedy",
                        }
                        for name in state.model_names
                    ],
                },
            )
            return
        if self.path == "/api/tags":
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._send_json(
                HTTPStatus.OK,
                {
                    "models": [
                        {
                            "name": name,
                            "model": name,
                            "modified_at": now,
                            "size": 0,
                            "digest": "",
                            "details": {
                                "format": "peft",
                                "family": "minicpm-v",
                                "parameter_size": "MiniCPM-V-4.6+LoRA",
                            },
                        }
                        for name in state.model_names
                    ]
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        state: RouterState = self.server.router_state  # type: ignore[attr-defined]
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            started = time.perf_counter()
            model_name, text = state.generate(payload)
            elapsed = time.perf_counter() - started
            now = int(time.time())
            self._send_json(
                HTTPStatus.OK,
                {
                    "id": f"chatcmpl-{now}",
                    "object": "chat.completion",
                    "created": now,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
            print(
                f"[minicpm-router] model={model_name} status=200 "
                f"elapsed={elapsed:.2f}s chars={len(text)}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[minicpm-router] request failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"message": f"{type(exc).__name__}: {exc}"}},
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[minicpm-router] {self.address_string()} {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alt-adapter", type=Path, default=None)
    parser.add_argument("--table-adapter", type=Path, default=None)
    parser.add_argument("--contrast-adapter", type=Path, default=None)
    parser.add_argument("--reading-order-adapter", type=Path, default=None)
    parser.add_argument("--heading-adapter", type=Path, default=None)
    parser.add_argument("--multitask-adapter", type=Path, default=None)
    parser.add_argument("--include-multitask", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--downsample-mode", choices=("16x", "4x"), default="16x")
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument(
        "--adapter-image-settings",
        default="",
        help="Optional comma list like heading=4x:36,reading_order=4x:36; unspecified adapters use the global image settings.",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        default="cuda",
        help="Serving device. Default cuda (GPU workbenches); mps/cpu for local Apple Silicon runs.",
    )
    parser.add_argument("--print-env", action="store_true")
    return parser.parse_args()


def adapter_path(root: Path, override: Path | None, dirname: str) -> Path:
    return (override if override is not None else root / dirname).expanduser()


def build_adapter_maps(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, str]]:
    if args.base_only:
        return {}, {name: "base" for name in ALIASES if name != MULTITASK.alias}
    root = args.adapter_root.expanduser()
    override_by_key = {
        "alt": args.alt_adapter,
        "table": args.table_adapter,
        "contrast": args.contrast_adapter,
        "reading_order": args.reading_order_adapter,
        "heading": args.heading_adapter,
        "multitask": args.multitask_adapter,
    }
    adapters: dict[str, Path] = {}
    for spec in TASKS:
        adapters[spec.key] = adapter_path(root, override_by_key[spec.key], spec.output_dir)
    if args.include_multitask:
        adapters[MULTITASK.key] = adapter_path(root, args.multitask_adapter, MULTITASK.output_dir)
    aliases = {
        STABLE_ALIAS: "alt",
        "minicpm-v46-remedy-alt-v1": "alt",
        TASK_MODEL_MAP["table_structure"]: "table",
        TASK_MODEL_MAP["contrast"]: "contrast",
        TASK_MODEL_MAP["reading_order"]: "reading_order",
        TASK_MODEL_MAP["heading_hierarchy"]: "heading",
    }
    # Retired aliases stay resolvable and serve the task's promoted adapter.
    for legacy_alias, task_key in LEGACY_ALIASES.items():
        aliases.setdefault(legacy_alias, task_key)
    if args.include_multitask:
        aliases[MULTITASK.alias] = MULTITASK.key
    return adapters, aliases


def parse_adapter_image_settings(raw: str) -> dict[str, tuple[str, int]]:
    settings: dict[str, tuple[str, int]] = {}
    aliases = {
        "alt_text_quality": "alt",
        "table_structure": "table",
        "heading_hierarchy": "heading",
    }
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise argparse.ArgumentTypeError(
                f"invalid adapter image setting {item!r}; expected adapter=16x:1"
            )
        name, value = item.split("=", 1)
        downsample, slices = value.split(":", 1)
        name = aliases.get(name.strip(), name.strip())
        if name not in {"alt", "table", "contrast", "reading_order", "heading", "multitask"}:
            raise argparse.ArgumentTypeError(f"unknown adapter image setting key {name!r}")
        if downsample not in {"16x", "4x"}:
            raise argparse.ArgumentTypeError(f"invalid downsample mode {downsample!r}")
        try:
            max_slice_nums = int(slices)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid max slice count {slices!r}") from exc
        if max_slice_nums < 1:
            raise argparse.ArgumentTypeError("max slice count must be >= 1")
        settings[name] = (downsample, max_slice_nums)
    return settings


def main() -> int:
    args = parse_args()
    adapters, aliases = build_adapter_maps(args)
    adapter_image_settings = parse_adapter_image_settings(args.adapter_image_settings)
    missing = {
        name: path
        for name, path in adapters.items()
        if not (path / "adapter_config.json").exists()
        or not (path / "adapter_model.safetensors").exists()
    }
    if missing and not args.allow_missing:
        lines = [f"{name}: {path}" for name, path in missing.items()]
        raise SystemExit("missing adapter(s):\n" + "\n".join(lines))
    if missing:
        for name in list(missing):
            adapters.pop(name, None)
        aliases = {alias: key for alias, key in aliases.items() if key in adapters}
        if not adapters:
            raise SystemExit("all adapters are missing; pass --base-only for base-model serving")
    if args.print_env:
        print(router_env(f"http://127.0.0.1:{args.port}/v1"))

    state = RouterState(
        base_model=args.model,
        adapters=adapters,
        aliases=aliases,
        downsample_mode=args.downsample_mode,
        max_slice_nums=args.max_slice_nums,
        adapter_image_settings=adapter_image_settings,
        attn_implementation=args.attn_implementation,
        device=args.device,
    )
    server = ThreadingHTTPServer((args.host, args.port), OpenAIHandler)
    server.router_state = state  # type: ignore[attr-defined]
    print(
        f"[minicpm-router] listening on http://{args.host}:{args.port} "
        f"with {len(state.model_names)} aliases",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
