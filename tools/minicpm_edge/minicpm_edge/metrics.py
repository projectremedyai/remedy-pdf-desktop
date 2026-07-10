"""Task-aware JSON metrics for Remedy MiniCPM adapter predictions."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_jsonish(text: str | dict | list | None) -> Any:
    if isinstance(text, (dict, list)):
        return text
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def target_text(rec: dict[str, Any]) -> str:
    return str(rec["messages"][-1]["content"][0]["text"])


def task_name(rec: dict[str, Any]) -> str:
    meta = rec.get("meta") or {}
    return str(meta.get("task") or rec.get("task") or "")


def record_key(rec: dict[str, Any], index: int) -> str:
    meta = rec.get("meta") or {}
    if meta.get("example_id"):
        return str(meta["example_id"])
    parts = [
        str(meta.get("doc_id") or rec.get("doc_id") or ""),
        str(meta.get("page") or rec.get("page") or rec.get("page_index") or ""),
        str(meta.get("task") or rec.get("task") or ""),
        str(meta.get("variant") or rec.get("variant") or ""),
    ]
    if any(parts):
        return "|".join(parts)
    return str(index)


def prediction_text(row: dict[str, Any]) -> str:
    for key in ("response", "prediction", "generated", "output", "text"):
        if key in row:
            return str(row[key])
    if "messages" in row:
        return target_text(row)
    return json.dumps(row, ensure_ascii=False)


def normalized_status(parsed: Any, task: str) -> str | None:
    if not isinstance(parsed, dict):
        return None
    status = str(parsed.get("status", "")).strip().lower()
    if status in {"pass", "fail"}:
        return status
    if task == "alt_text_quality":
        figures = parsed.get("figures", parsed.get("issues"))
        if isinstance(figures, list):
            return "fail" if any(
                isinstance(item, dict)
                and str(item.get("status", "pass")).strip().lower()
                in {"fail", "failed", "error"}
                for item in figures
            ) else "pass"
    issues = parsed.get("issues")
    if isinstance(issues, list):
        return "fail" if issues else "pass"
    findings = parsed.get("findings")
    if isinstance(findings, list):
        return "fail" if findings else "pass"
    return None


def heading_pairs(parsed: Any) -> set[tuple[int, str]]:
    if not isinstance(parsed, dict):
        return set()
    items: list[Any] = []
    for key in ("findings", "heading_corrections", "corrections", "heading_issues", "issues"):
        value = parsed.get(key)
        if isinstance(value, list):
            items.extend(value)
    pairs: set[tuple[int, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            element_index = int(item.get("element_index") or item.get("index") or item.get("element"))
        except Exception:
            continue
        tag = str(item.get("correct_tag") or item.get("target_tag") or item.get("expected_tag") or "")
        tag = tag.strip().lstrip("/").upper()
        if re.fullmatch(r"H[1-6]|P|SPAN", tag):
            pairs.add((element_index, "Span" if tag == "SPAN" else tag))
    return pairs


def reading_order(parsed: Any) -> tuple[int, ...] | None:
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("corrected_order", parsed.get("reading_order"))
    if value in (None, False) or not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            return None
    return tuple(out)


def contrast_ratios(parsed: Any) -> list[float]:
    if not isinstance(parsed, dict):
        return []
    issues = parsed.get("issues", parsed.get("contrast_issues"))
    if not isinstance(issues, list):
        return []
    ratios: list[float] = []
    for item in issues:
        if not isinstance(item, dict):
            continue
        try:
            ratios.append(float(item.get("ratio")))
        except Exception:
            pass
    return ratios


def score_one(task: str, gold: Any, pred: Any) -> dict[str, Any]:
    gold_status = normalized_status(gold, task)
    pred_status = normalized_status(pred, task)
    out: dict[str, Any] = {
        "task": task,
        "valid_json": pred is not None,
        "gold_status": gold_status,
        "pred_status": pred_status,
        "status_match": gold_status is not None and pred_status == gold_status,
    }
    if task == "heading_hierarchy":
        gold_pairs = heading_pairs(gold)
        pred_pairs = heading_pairs(pred)
        out.update(
            exact_corrections=gold_pairs == pred_pairs,
            correction_recall=(len(gold_pairs & pred_pairs) / len(gold_pairs) if gold_pairs else 1.0),
            correction_precision=(len(gold_pairs & pred_pairs) / len(pred_pairs) if pred_pairs else 1.0),
            pass_false_positive=(gold_status == "pass" and pred_status == "fail"),
        )
    elif task == "reading_order":
        gold_order = reading_order(gold)
        pred_order = reading_order(pred)
        out.update(
            corrected_order_match=(gold_order == pred_order if gold_order is not None else None),
            pass_false_positive=(gold_status == "pass" and pred_status == "fail"),
        )
    elif task == "contrast":
        gold_ratios = contrast_ratios(gold)
        pred_ratios = contrast_ratios(pred)
        out.update(
            gold_ratios=gold_ratios,
            pred_ratios=pred_ratios,
            near_threshold=any(4.2 <= ratio <= 4.8 for ratio in gold_ratios + pred_ratios),
            pass_false_positive=(gold_status == "pass" and pred_status == "fail"),
        )
    elif task in {"table_structure", "alt_text_quality"}:
        out["pass_false_positive"] = gold_status == "pass" and pred_status == "fail"
    return out


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        by_task[score["task"]].append(score)
    summary: dict[str, Any] = {"total": len(scores), "by_task": {}}
    for task, rows in by_task.items():
        confusion = Counter((row.get("gold_status"), row.get("pred_status")) for row in rows)
        task_summary: dict[str, Any] = {
            "count": len(rows),
            "valid_json_rate": round(sum(row["valid_json"] for row in rows) / len(rows), 4),
            "status_accuracy": round(sum(row["status_match"] for row in rows) / len(rows), 4),
            "confusion": {
                f"{gold}->{pred}": n
                for (gold, pred), n in sorted(confusion.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))
            },
            "pass_false_positive_rate": round(
                sum(bool(row.get("pass_false_positive")) for row in rows) / len(rows),
                4,
            ),
        }
        if task == "heading_hierarchy":
            fail_rows = [row for row in rows if row.get("gold_status") == "fail"]
            task_summary.update(
                exact_correction_accuracy=round(
                    sum(bool(row.get("exact_corrections")) for row in fail_rows) / len(fail_rows),
                    4,
                ) if fail_rows else None,
                correction_recall=round(
                    sum(float(row.get("correction_recall", 0.0)) for row in fail_rows) / len(fail_rows),
                    4,
                ) if fail_rows else None,
                correction_precision=round(
                    sum(float(row.get("correction_precision", 0.0)) for row in fail_rows) / len(fail_rows),
                    4,
                ) if fail_rows else None,
            )
        if task == "reading_order":
            order_rows = [row for row in rows if row.get("corrected_order_match") is not None]
            task_summary["corrected_order_accuracy"] = round(
                sum(bool(row.get("corrected_order_match")) for row in order_rows) / len(order_rows),
                4,
            ) if order_rows else None
        if task == "contrast":
            near = [row for row in rows if row.get("near_threshold")]
            task_summary["near_threshold_status_accuracy"] = round(
                sum(bool(row["status_match"]) for row in near) / len(near),
                4,
            ) if near else None
        summary["by_task"][task] = task_summary
    return summary


def load_predictions(path: Path, gold_rows: list[dict[str, Any]]) -> dict[str, str]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keyed: dict[str, str] = {}
    for i, row in enumerate(rows):
        key = str(row.get("example_id") or row.get("id") or "")
        if not key:
            key = record_key(row, i)
        text = prediction_text(row)
        keyed[key] = text
        keyed[str(i)] = text
    return keyed


def score_predictions(
    val_rows: list[dict[str, Any]],
    predictions: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for i, rec in enumerate(val_rows):
        key = record_key(rec, i)
        task = task_name(rec)
        # Generated prediction files are in validation-row order. Prefer the
        # row index because some corpora intentionally pair pass/fail variants
        # with the same document/page key.
        pred_text = predictions.get(str(i), predictions.get(key, ""))
        score = score_one(task, parse_jsonish(target_text(rec)), parse_jsonish(pred_text))
        score["example_id"] = key
        scores.append(score)
    return scores, summarize(scores)
