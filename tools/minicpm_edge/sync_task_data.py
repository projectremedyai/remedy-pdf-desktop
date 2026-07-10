#!/usr/bin/env python3
"""Copy Remedy multitask corpora into the local MiniCPM edge workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minicpm_edge.constants import DEFAULT_DATA_ROOT, DEFAULT_SOURCE_ROOT, MULTITASK, TASKS
from minicpm_edge.datasets import copy_task_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--include-multitask", action="store_true", default=True)
    args = parser.parse_args()

    specs = [*TASKS, MULTITASK] if args.include_multitask else list(TASKS)
    results = [
        copy_task_dataset(spec, args.source_root.expanduser(), args.data_root.expanduser())
        for spec in specs
    ]
    manifest = {
        "source_root": str(args.source_root.expanduser()),
        "data_root": str(args.data_root.expanduser()),
        "tasks": results,
        "all_counts_match": all(result["matches_expected"] for result in results),
    }
    args.data_root.mkdir(parents=True, exist_ok=True)
    (args.data_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not manifest["all_counts_match"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

