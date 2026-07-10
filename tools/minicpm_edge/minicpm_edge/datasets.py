"""Dataset helpers for Remedy MiniCPM training and evaluation JSONL."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .constants import EXPECTED_COUNTS, TaskSpec


SOURCE_DIR_TO_LOCAL_DIR = {
    "data_v2": "alt_text_quality",
    "data_table": "table_structure",
    "data_contrast": "contrast",
    "data_reading_order": "reading_order",
    "heading_hierarchy": "heading_hierarchy",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def resolve_images(rows: list[dict[str, Any]], base_dir: Path) -> list[dict[str, Any]]:
    """Resolve conversation image paths relative to the JSONL file directory."""
    resolved: list[dict[str, Any]] = []
    for rec in rows:
        rec = json.loads(json.dumps(rec))
        for message in rec.get("messages", []):
            for part in message.get("content", []):
                if part.get("type") != "image" or not isinstance(part.get("image"), str):
                    continue
                image = Path(part["image"])
                absolute = resolve_image_path(image, base_dir)
                part["image"] = str(absolute)
        resolved.append(rec)
    return resolved


def resolve_image_path(image: str | Path, base_dir: Path) -> Path:
    """Resolve a JSONL image path, including sibling task references.

    The contrast-weighted multitask corpus can include paths such as
    ``../data/heading_hierarchy/renders/...`` from the source finetune tree.
    After sync, sibling task folders live directly under ``tasks/`` rather
    than under ``tasks/data/``. Try that normalized location before returning
    the ordinary relative path.
    """
    image_path = Path(image)
    if image_path.is_absolute():
        return image_path

    candidates = [base_dir / image_path]
    parts = image_path.parts
    if len(parts) >= 3 and parts[0] == ".." and parts[1] == "data":
        mapped = SOURCE_DIR_TO_LOCAL_DIR.get(parts[2], parts[2])
        candidates.append(base_dir.parent / mapped / Path(*parts[3:]))
    if len(parts) >= 2 and parts[0] == "..":
        mapped = SOURCE_DIR_TO_LOCAL_DIR.get(parts[1])
        if mapped:
            candidates.append(base_dir.parent / mapped / Path(*parts[2:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_conversations(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = resolve_images(load_jsonl(path), path.resolve().parent)
    return rows[:limit] if limit > 0 else rows


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def split_counts(data_dir: Path) -> dict[str, int]:
    return {
        "train": count_jsonl(data_dir / "train.jsonl"),
        "val": count_jsonl(data_dir / "val.jsonl"),
    }


def copy_task_dataset(spec: TaskSpec, source_root: Path, data_root: Path) -> dict[str, Any]:
    """Copy a task dataset directory into the local ignored MiniCPM data root."""
    src = (source_root / spec.source_dir).resolve()
    dst = data_root / spec.local_dir
    if not (src / "train.jsonl").exists() or not (src / "val.jsonl").exists():
        raise FileNotFoundError(f"{src} must contain train.jsonl and val.jsonl")
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        ".DS_Store",
        "outputs*",
        "checkpoints*",
    )
    shutil.copytree(src, dst, ignore=ignore)
    for split in ("train", "val"):
        rewrite_copied_image_paths(dst / f"{split}.jsonl", src, dst)
    counts = split_counts(dst)
    expected = EXPECTED_COUNTS.get(spec.local_dir)
    return {
        "task": spec.local_dir,
        "source": str(src),
        "destination": str(dst),
        "counts": counts,
        "expected": expected,
        "matches_expected": expected is None or counts == expected,
    }


def rewrite_copied_image_paths(jsonl: Path, source_dir: Path, copied_dir: Path) -> None:
    """Rewrite copied JSONL image references so the folder is portable."""
    rows = load_jsonl(jsonl)
    changed = False
    for row in rows:
        for message in row.get("messages", []):
            for part in message.get("content", []):
                if part.get("type") != "image" or not isinstance(part.get("image"), str):
                    continue
                raw = Path(part["image"])
                if raw.is_absolute():
                    try:
                        rel = raw.resolve().relative_to(source_dir)
                        absolute = copied_dir / rel
                    except ValueError:
                        absolute = raw
                else:
                    absolute = resolve_image_path(raw, copied_dir)
                if absolute.exists():
                    part["image"] = os.path.relpath(absolute, copied_dir.resolve())
                    changed = True
    if changed:
        write_jsonl(jsonl, rows)


def relativize_images(jsonl: Path, out_dir: Path) -> None:
    """Rewrite image paths in a copied JSONL so they stay portable from out_dir."""
    rows = load_jsonl(jsonl)
    base = jsonl.resolve().parent
    for row in rows:
        for message in row.get("messages", []):
            for part in message.get("content", []):
                if part.get("type") != "image" or not isinstance(part.get("image"), str):
                    continue
                image = Path(part["image"])
                absolute = image if image.is_absolute() else base / image
                part["image"] = os.path.relpath(absolute, out_dir.resolve())
    write_jsonl(jsonl, rows)
