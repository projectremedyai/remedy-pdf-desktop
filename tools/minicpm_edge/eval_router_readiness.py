#!/usr/bin/env python3
"""Check whether MiniCPM Remedy adapters and metrics are ready for router smoke."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any
import urllib.request

from minicpm_edge.constants import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EVAL_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MULTITASK,
    PROMOTION_GATES,
    STABLE_ALIAS,
    TASKS,
)
from minicpm_edge.datasets import load_jsonl
from minicpm_edge.metrics import parse_jsonish


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gate(name: str, observed: Any, *, min_value: float | None = None, max_value: float | None = None, source: str = "") -> dict[str, Any]:
    passed = observed is not None
    if passed and min_value is not None:
        passed = float(observed) >= min_value
    if passed and max_value is not None:
        passed = float(observed) <= max_value
    expected: dict[str, Any] = {}
    if min_value is not None:
        expected["min"] = min_value
    if max_value is not None:
        expected["max"] = max_value
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "source": source,
    }


def adapter_check(alias: str, path: Path) -> dict[str, Any]:
    return {
        "alias": alias,
        "path": str(path),
        "has_config": (path / "adapter_config.json").exists(),
        "has_weights": (path / "adapter_model.safetensors").exists(),
        "passed": (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists(),
    }


def task_metrics(metrics_dir: Path, key: str, task_name: str) -> tuple[Path, dict[str, Any]]:
    path = metrics_dir / f"{key}.metrics.json"
    data = load_json(path)
    return path, dict(((data.get("by_task") or {}).get(task_name) or {}))


def gates_for_task(metrics_dir: Path, spec) -> list[dict[str, Any]]:
    path, metrics = task_metrics(metrics_dir, spec.key, spec.task_name)
    gates = [
        gate(
            f"{spec.task_name} valid_json_rate >= {PROMOTION_GATES['valid_json_rate']}",
            metrics.get("valid_json_rate"),
            min_value=PROMOTION_GATES["valid_json_rate"],
            source=str(path),
        ),
        gate(
            f"{spec.task_name} pass_false_positive_rate <= {PROMOTION_GATES['pass_false_positive_rate']}",
            metrics.get("pass_false_positive_rate"),
            max_value=PROMOTION_GATES["pass_false_positive_rate"],
            source=str(path),
        ),
    ]
    if spec.min_status_accuracy is not None:
        gates.append(
            gate(
                f"{spec.task_name} status_accuracy >= {spec.min_status_accuracy}",
                metrics.get("status_accuracy"),
                min_value=spec.min_status_accuracy,
                source=str(path),
            )
        )
    if spec.min_exact_correction_accuracy is not None:
        gates.append(
            gate(
                f"{spec.task_name} exact_correction_accuracy >= {spec.min_exact_correction_accuracy}",
                metrics.get("exact_correction_accuracy"),
                min_value=spec.min_exact_correction_accuracy,
                source=str(path),
            )
        )
    if spec.min_near_threshold_accuracy is not None:
        gates.append(
            gate(
                f"{spec.task_name} near_threshold_status_accuracy >= {spec.min_near_threshold_accuracy}",
                metrics.get("near_threshold_status_accuracy"),
                min_value=spec.min_near_threshold_accuracy,
                source=str(path),
            )
        )
    return gates


def all_passed(items: list[dict[str, Any]]) -> bool:
    return all(bool(item.get("passed")) for item in items)


def parse_adapter_image_settings(raw: str) -> dict[str, dict[str, int | str]]:
    aliases = {
        "alt_text_quality": "alt",
        "table_structure": "table",
        "heading_hierarchy": "heading",
    }
    settings: dict[str, dict[str, int | str]] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise argparse.ArgumentTypeError(
                f"invalid adapter image setting {item!r}; expected adapter=16x:1"
            )
        name, value = item.split("=", 1)
        downsample_mode, max_slice_nums = value.split(":", 1)
        name = aliases.get(name.strip(), name.strip())
        if downsample_mode not in {"16x", "4x"}:
            raise argparse.ArgumentTypeError(f"invalid downsample mode {downsample_mode!r}")
        try:
            slice_count = int(max_slice_nums)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid max slice count {max_slice_nums!r}"
            ) from exc
        if slice_count < 1:
            raise argparse.ArgumentTypeError("max slice count must be >= 1")
        settings[name] = {
            "downsample_mode": downsample_mode,
            "max_slice_nums": slice_count,
        }
    return settings


def router_image_policy(
    *,
    specs: list[Any],
    downsample_mode: str,
    max_slice_nums: int,
    adapter_image_settings: str,
) -> dict[str, Any]:
    overrides = parse_adapter_image_settings(adapter_image_settings)
    by_task = {}
    for spec in specs:
        setting = overrides.get(
            spec.key,
            {"downsample_mode": downsample_mode, "max_slice_nums": max_slice_nums},
        )
        by_task[spec.task_name] = setting
    return {
        "default": {"downsample_mode": downsample_mode, "max_slice_nums": max_slice_nums},
        "overrides": overrides,
        "by_task": by_task,
    }


def endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        return base + path[3:]
    if not base.endswith("/v1") and path in {"/models", "/chat/completions"}:
        return base + "/v1" + path
    return base + path


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if payload is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local/dev router URL
        return json.loads(response.read().decode("utf-8"))


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def live_payload(row: dict[str, Any], row_dir: Path, model: str, max_tokens: int) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for part in row["messages"][0].get("content", []):
        if part.get("type") == "image":
            image = Path(str(part.get("image") or ""))
            image_path = image if image.is_absolute() else row_dir / image
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
        elif part.get("type") == "text":
            content.append({"type": "text", "text": str(part.get("text") or "")})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }


def live_router_check(
    *,
    base_url: str,
    data_root: Path,
    specs: list[Any],
    limit: int,
    max_tokens: int,
) -> dict[str, Any]:
    model_listing = request_json(endpoint(base_url, "/models"))
    listed = {
        str(item.get("id") or item.get("name") or item.get("model"))
        for item in model_listing.get("data", [])
        if isinstance(item, dict)
    }
    aliases = [STABLE_ALIAS, *[spec.alias for spec in specs]]
    model_gates = [
        gate(f"router lists {alias}", alias in listed, min_value=1, source=endpoint(base_url, "/models"))
        for alias in aliases
    ]

    router_errors = 0
    task_live: dict[str, Any] = {}
    live_gates: list[dict[str, Any]] = []
    for spec in specs:
        if spec.key == MULTITASK.key:
            continue
        model = STABLE_ALIAS if spec.key == "alt" else spec.alias
        val = data_root / spec.local_dir / "val.jsonl"
        rows = load_jsonl(val)[:limit]
        valid = 0
        errors: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            try:
                response = request_json(
                    endpoint(base_url, "/chat/completions"),
                    live_payload(row, val.parent, model, max_tokens),
                )
                content = response["choices"][0]["message"]["content"]
                if parse_jsonish(content) is not None:
                    valid += 1
            except Exception as exc:  # noqa: BLE001
                router_errors += 1
                errors.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
        valid_rate = round(valid / len(rows), 4) if rows else None
        task_live[spec.task_name] = {
            "model": model,
            "count": len(rows),
            "valid_json_rate": valid_rate,
            "errors": errors,
        }
        live_gates.append(
            gate(
                f"{spec.task_name} live valid_json_rate >= {PROMOTION_GATES['valid_json_rate']}",
                valid_rate,
                min_value=PROMOTION_GATES["valid_json_rate"],
                source=endpoint(base_url, "/chat/completions"),
            )
        )
    live_gates.append(gate("live router errors == 0", router_errors, max_value=0, source=base_url))
    return {
        "base_url": base_url,
        "listed_models": sorted(listed),
        "model_gates": model_gates,
        "task_live": task_live,
        "router_errors": router_errors,
        "live_gates": live_gates,
        "passed": all_passed(model_gates) and all_passed(live_gates),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_EVAL_ROOT / "tuned")
    parser.add_argument("--include-multitask", action="store_true")
    parser.add_argument("--router-base-url", default="", help="Optional live router URL, with or without /v1")
    parser.add_argument("--live-limit", type=int, default=3)
    parser.add_argument("--live-max-tokens", type=int, default=384)
    parser.add_argument("--downsample-mode", choices=("16x", "4x"), default="16x")
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--adapter-image-settings", default="")
    parser.add_argument("--out", type=Path, default=DEFAULT_EVAL_ROOT / "router_readiness.json")
    args = parser.parse_args()

    specs = [*TASKS, MULTITASK] if args.include_multitask else list(TASKS)
    image_policy = router_image_policy(
        specs=specs,
        downsample_mode=args.downsample_mode,
        max_slice_nums=args.max_slice_nums,
        adapter_image_settings=args.adapter_image_settings,
    )
    adapters = [
        adapter_check(spec.alias, args.adapter_root / spec.output_dir)
        for spec in specs
    ]
    metrics_gates: list[dict[str, Any]] = []
    for spec in TASKS:
        metrics_gates.extend(gates_for_task(args.metrics_dir, spec))

    live_router = None
    if args.router_base_url:
        live_router = live_router_check(
            base_url=args.router_base_url,
            data_root=args.data_root,
            specs=specs,
            limit=args.live_limit,
            max_tokens=args.live_max_tokens,
        )

    static_ready = all_passed(adapters) and all_passed(metrics_gates)
    summary = {
        "adapters": adapters,
        "metric_gates": metrics_gates,
        "router_image_policy": image_policy,
        "live_router": live_router,
        "ready_for_router_smoke": static_ready and (live_router is None or live_router["passed"]),
        "multitask_included": args.include_multitask,
        "multitask_policy": "Do not promote unless it matches or beats every per-task adapter gate.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ready_for_router_smoke"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
