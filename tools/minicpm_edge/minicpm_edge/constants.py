"""Shared constants for the MiniCPM-V-4.6 Remedy adapter workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_MODEL = "openbmb/MiniCPM-V-4.6"
STABLE_ALIAS = "minicpm-v46-remedy"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "tasks"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"
DEFAULT_EVAL_ROOT = ROOT / "eval_runs"
DEFAULT_SOURCE_ROOT = Path(
    "/Users/laccd/code/lamc_district_forms/remedy-server-multitask-next/tools/finetune"
)


@dataclass(frozen=True)
class TaskSpec:
    key: str
    task_name: str
    alias: str
    source_dir: str
    local_dir: str
    hub_repo: str
    min_status_accuracy: float | None = None
    min_exact_correction_accuracy: float | None = None
    min_near_threshold_accuracy: float | None = None

    @property
    def output_dir(self) -> str:
        return self.hub_repo.replace("johnnyrobotai/", "")


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        key="alt",
        task_name="alt_text_quality",
        alias="minicpm-v46-remedy-alt-v1",
        source_dir="data_v2",
        local_dir="alt_text_quality",
        hub_repo="johnnyrobotai/remedy-minicpm-v46-alt-v1-lora",
        min_status_accuracy=0.90,
    ),
    TaskSpec(
        key="table",
        task_name="table_structure",
        alias="minicpm-v46-remedy-table-v1",
        source_dir="data_table",
        local_dir="table_structure",
        hub_repo="johnnyrobotai/remedy-minicpm-v46-table-v1-lora",
        min_status_accuracy=1.00,
    ),
    TaskSpec(
        key="contrast",
        task_name="contrast",
        alias="minicpm-v46-remedy-contrast-v1",
        source_dir="data_contrast",
        local_dir="contrast",
        hub_repo="johnnyrobotai/remedy-minicpm-v46-contrast-v1-lora",
        min_status_accuracy=0.90,
        min_near_threshold_accuracy=0.85,
    ),
    TaskSpec(
        key="reading_order",
        task_name="reading_order",
        # v4 promoted 2026-07-10: trained on the synthetic corruption family,
        # val 1.00 / val_hard 1.00 at 16x/1. Serve at 16x:1 (trained profile);
        # the v1-era reading_order=4x:36 override is no longer required.
        alias="minicpm-v46-remedy-reading-order-v4",
        source_dir="data_reading_order",
        local_dir="reading_order",
        hub_repo="johnnyrobotai/remedy-minicpm-v46-reading-order-v4-lora",
        min_status_accuracy=0.80,
    ),
    TaskSpec(
        key="heading",
        task_name="heading_hierarchy",
        alias="minicpm-v46-remedy-heading-v1",
        source_dir="data/heading_hierarchy",
        local_dir="heading_hierarchy",
        hub_repo="johnnyrobotai/remedy-minicpm-v46-heading-v1-lora",
        min_status_accuracy=0.95,
        min_exact_correction_accuracy=0.85,
    ),
)

MULTITASK = TaskSpec(
    key="multitask",
    task_name="multitask",
    alias="minicpm-v46-remedy-multitask-v1",
    source_dir="data_multitask_contrast_weighted",
    local_dir="multitask_contrast_weighted",
    hub_repo="johnnyrobotai/remedy-minicpm-v46-multitask-v1-lora",
)

TASK_MODEL_MAP = {
    "contrast": "minicpm-v46-remedy-contrast-v1",
    "reading_order": "minicpm-v46-remedy-reading-order-v4",
    "heading_hierarchy": "minicpm-v46-remedy-heading-v1",
    "table_structure": "minicpm-v46-remedy-table-v1",
}

# Retired aliases kept resolvable so saved desktop configs keep working; they
# serve the currently promoted adapter for the task.
LEGACY_ALIASES = {
    "minicpm-v46-remedy-reading-order-v1": "reading_order",
}

ALIASES = {
    STABLE_ALIAS: "alt",
    "minicpm-v46-remedy-alt-v1": "alt",
    **{task.alias: task.key for task in TASKS if task.key != "alt"},
    MULTITASK.alias: MULTITASK.key,
    **LEGACY_ALIASES,
}

EXPECTED_COUNTS = {
    "alt_text_quality": {"train": 266, "val": 50},
    "table_structure": {"train": 208, "val": 20},
    "reading_order": {"train": 200, "val": 44},
    "contrast": {"train": 143, "val": 17},
    "heading_hierarchy": {"train": 1070, "val": 210},
    "multitask_contrast_weighted": {"train": 2602, "val": 341},
}

PROMOTION_GATES = {
    "valid_json_rate": 0.90,
    "pass_false_positive_rate": 0.10,
}


def all_specs(include_multitask: bool = False) -> tuple[TaskSpec, ...]:
    return (*TASKS, MULTITASK) if include_multitask else TASKS


def task_by_key(key: str) -> TaskSpec:
    for spec in all_specs(include_multitask=True):
        if spec.key == key:
            return spec
    raise KeyError(f"unknown task key: {key}")


def task_by_name(name: str) -> TaskSpec:
    for spec in TASKS:
        if spec.task_name == name:
            return spec
    raise KeyError(f"unknown task name: {name}")

