"""Closed-loop agentic PDF remediation for Tier 3 rescue.

The previous Vision Planner pipeline was open-loop:

    grounder -> planner -> executor

The planner never saw execution results, so prompt evolution mostly learned
how to produce plausible plans rather than effective ones. This module flips
that contract. The model is treated as a tool-using agent that can inspect the
current PDF, apply one fix, immediately see the updated veraPDF result, and
iterate until the document passes or the retry budget is exhausted.

The implementation is intentionally conservative:

* one mutating tool call per assistant turn
* automatic veraPDF verification after every mutation
* unsafe raw artifactization is disabled by default
* all tool schemas use OpenAI-compatible function calling
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
import inspect
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import pikepdf

from project_remedy.pdf_acceptance import VeraPDFResult, validate_with_verapdf
from project_remedy.pdf_checker import _get_struct_type, walk_structure_tree
from project_remedy.pdf_fixer import (
    _find_node_page,
    _get_node_mcids,
    _normalize_structure_tree_indirect_objects,
    _read_page_content,
    fix_annotation_descriptions,
    fix_annotations_tagged,
    fix_bdc_emc_balance,
    fix_form_field_descriptions,
    fix_form_fields_tagged,
    fix_heading_nesting,
    fix_language,
    fix_link_annotations,
    fix_list_structure,
    fix_tab_order,
    fix_table_header_scope,
    fix_table_headers,
    fix_table_parent_structure,
    fix_table_td_headers,
    fix_untagged_content,
)
from project_remedy.pdf_vision import render_page_to_image
from project_remedy.tag_tree_reader import _extract_mcid_text
from project_remedy.vision_planner.executor import _do_artifactize, _do_set_tag
from project_remedy.vision_planner.rule_router import (
    _fix_pagination_to_artifact,
    _fix_viewer_preferences,
    _normalize_rule_id,
)

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are a PDF/UA-1 accessibility remediation agent working on one PDF at a time.

Your objective is to reduce veraPDF violations to zero without introducing regressions.

Workflow:
1. Start by inspecting the current veraPDF violations and the relevant PDF structure.
2. Choose the top remaining violation family.
3. Apply exactly one mutating fix tool for that family.
4. Read the post-fix verification returned by the tool.
5. If the tool result says `progress_state="improved"`, continue from the updated verification.
6. If the tool result says `progress_state="no_progress"`, `progress_state="regressed"`, or `progress_state="tool_error"`, do not repeat that exact fix blindly. Inspect again and switch approaches.
7. When a generic high-level fix tool stalls on a family, escalate quickly: inspect the affected page/object, then use `run_pikepdf_code` for a targeted edit.

Rules:
- Fix one violation family at a time.
- Prefer structural and tagging issues before metadata or annotation hygiene.
- Do not call more than one mutating fix tool in a single turn.
- Read the compact remediation-state summary each turn. It replaces most older history.
- Use inspection tools when the target page, MCID, or tag is ambiguous.
- Source-font violations in the 7.21.x family are often source-document limitations. Do not spend the full budget on them unless another tool result shows clear progress.
- Raw artifactization is dangerous. Only use mark_as_artifact when no safer deterministic fix applies.
- If a tool result includes `retry_guidance`, follow it.
- If a tool result says an exact mutation is blocked, choose a different approach immediately.
- Call finish with status="fixed" only after veraPDF reports zero violations.
- Call finish with status="manual_review" when you are stuck, the budget is exhausted, or the remaining issues are likely source-font limitations.

Advanced tool — run_pikepdf_code:
- Use run_pikepdf_code aggressively after a high-level tool fails to reduce violations for the same family.
- The variable `pdf` is already an open pikepdf.Pdf. You can modify it directly.
- Inspect first, then write the smallest targeted change that matches the current violation.
- Common patterns:
  - Rewrite content stream BDC/EMC markers: `page["/Contents"] = pdf.make_stream(new_bytes)`
  - Change structure element type: `node["/S"] = pikepdf.Name("/Artifact")`
  - Add /Lang: `pdf.Root["/Lang"] = pikepdf.String("en")`
  - Read content stream: `raw = page["/Contents"].read_bytes()`
- Always inspect first (`read_structure_tree`, `read_content_stream`, `read_page_image`) to understand the exact problem before writing code.
- Assign to `result` to return diagnostic info, object ids, or counts: `result = f"Fixed {count} markers on page {page_no}"`
- If custom code fails or has no effect, inspect again and change the code. Do not rerun the same code unchanged.
"""


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_verapdf",
            "description": "Run veraPDF on the current working PDF and return the remaining violations.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_structure_tree",
            "description": "Read the current structure tree, optionally filtered to one page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "1-based page number. Omit to inspect the whole document.",
                    },
                    "max_nodes": {
                        "type": "integer",
                        "description": "Maximum number of structure nodes to return.",
                    },
                    "include_text": {
                        "type": "boolean",
                        "description": "Whether to include MCID text excerpts for nodes with direct content.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_content_stream",
            "description": "Read the decoded content stream for a page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "1-based page number.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum number of characters to return.",
                    },
                },
                "required": ["page"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page_image",
            "description": "Render one page and ask the vision model a question about it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "1-based page number.",
                    },
                    "question": {
                        "type": "string",
                        "description": "What to ask the vision model about the page image.",
                    },
                },
                "required": ["page"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_untagged_content",
            "description": "Tag untagged content gaps and link existing MCIDs into the structure tree.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_heading_nesting",
            "description": "Repair heading level gaps such as H1 -> H3.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_table_structure",
            "description": "Repair parent-child table structure such as Table/TR/TH/TD containment.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_table_headers",
            "description": "Repair table header semantics, including TH promotion, scope, and TD header references.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_list_structure",
            "description": "Repair list containment such as L/LI/Lbl/LBody nesting.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_annotations",
            "description": "Repair annotation, widget, and link tagging and description fields.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_tab_order",
            "description": "Set /Tabs = /S on pages so tab order follows structure order.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_language",
            "description": "Set or normalize the document language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Language code such as en or es.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_viewer_preferences",
            "description": "Ensure ViewerPreferences/DisplayDocTitle is true.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_pagination_markers",
            "description": "Repair pagination and other non-MCID BDC markers created by preprocessing.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_bdc_emc_balance",
            "description": "Repair simple marked-content BDC/EMC imbalance.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_as_artifact",
            "description": "Mark specific MCIDs as Artifact. Unsafe by default and usually disabled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "1-based page number.",
                    },
                    "mcids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "MCIDs to artifactize on the page.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why these MCIDs are decorative or pagination-only.",
                    },
                },
                "required": ["page", "mcids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_tag",
            "description": "Change the structure tag for one MCID-backed element on a page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "1-based page number.",
                    },
                    "mcid": {
                        "type": "integer",
                        "description": "MCID to retag.",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Target tag such as P, H1, H2, Table, TR, TH, TD, Figure, Caption, Link, L, LI, Lbl, LBody.",
                    },
                },
                "required": ["page", "mcid", "tag"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pikepdf_code",
            "description": (
                "Execute arbitrary Python/pikepdf code against the current working PDF. "
                "The PDF is opened as `pdf` (pikepdf.Pdf). Your code can modify it freely. "
                "After execution the PDF is saved automatically. Use this for fine-grained "
                "edits that the high-level fix tools cannot handle — e.g. rewriting specific "
                "BDC/EMC markers, changing individual structure elements, modifying content "
                "streams. The code runs in a restricted namespace with pikepdf, re, and "
                "pathlib available. Return value is captured as the tool result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code to execute. The variable `pdf` is a pikepdf.Pdf "
                            "already opened on the working copy. Assign to `result` to "
                            "return a value. Example: "
                            "`page = pdf.pages[0]; result = str(page.get('/Tabs'))`"
                        ),
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Stop the remediation loop and report the outcome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["fixed", "manual_review", "error"],
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short explanation of why the loop is ending.",
                    },
                },
                "required": ["status", "reason"],
                "additionalProperties": False,
            },
        },
    },
]


MUTATING_TOOL_NAMES = {
    "run_pikepdf_code",
    "fix_untagged_content",
    "fix_heading_nesting",
    "fix_table_structure",
    "fix_table_headers",
    "fix_list_structure",
    "fix_annotations",
    "fix_tab_order",
    "fix_language",
    "fix_viewer_preferences",
    "fix_pagination_markers",
    "fix_bdc_emc_balance",
    "mark_as_artifact",
    "set_tag",
}

TOOL_FAMILIES = {
    "fix_untagged_content": "structure",
    "fix_heading_nesting": "structure",
    "fix_table_structure": "table",
    "fix_table_headers": "table",
    "fix_list_structure": "list",
    "fix_annotations": "annotation",
    "fix_tab_order": "annotation",
    "fix_language": "metadata",
    "fix_viewer_preferences": "metadata",
    "fix_pagination_markers": "structure",
    "fix_bdc_emc_balance": "structure",
    "mark_as_artifact": "structure",
    "set_tag": "structure",
}


@dataclass
class AgentLoopConfig:
    """Runtime knobs for the closed-loop remediation agent."""

    max_fix_attempts: int = 5
    max_fix_attempts_per_family: int | None = 2
    max_tool_rounds: int = 30
    max_structure_nodes: int = 200
    max_content_chars: int = 4000
    max_recent_messages: int = 8
    max_attempt_summaries: int = 6
    pikepdf_escalation_threshold: int = 1
    block_repeated_no_progress: bool = True
    temperature: float = 0.0
    max_tokens: int = 4096
    allow_unsafe_artifactize: bool = False


@dataclass
class ToolDispatchResult:
    """Result from a single tool call."""

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    mutated: bool = False
    finish: bool = False


@dataclass
class MutationOutcome:
    """Normalized result from one mutating tool execution."""

    ok: bool
    changes: list[str]
    error: str = ""


def build_initial_user_message(
    pdf_path: Path,
    verification: dict[str, Any],
    config: AgentLoopConfig,
) -> str:
    """Construct the first user message for the agent."""

    payload = {
        "pdf_path": str(pdf_path),
        "attempt_budget": config.max_fix_attempts,
        "per_family_attempt_budget": config.max_fix_attempts_per_family,
        "initial_verapdf": verification,
        "instructions": [
            "Inspect before the first fix.",
            "Use exactly one mutating fix tool per turn.",
            "Prefer the highest-impact remaining violation family.",
            "Use finish when the document is conformant or clearly stuck.",
        ],
    }
    return json.dumps(payload, indent=2)


def summarize_verapdf_result(result: VeraPDFResult, *, max_items: int = 25) -> dict[str, Any]:
    """Serialize veraPDF output into a compact, model-friendly shape."""

    violations = result.violations if result.checked else []
    normalized: list[dict[str, Any]] = []
    rule_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()

    for violation in violations:
        rule_id = _normalize_rule_id(
            violation.get("id", violation.get("rule_id", violation.get("rule", ""))),
        )
        family = rule_id.rsplit("-", 1)[0] if "-" in rule_id else rule_id
        rule_counts[rule_id] += 1
        family_counts[family] += 1
        if len(normalized) < max_items:
            normalized.append(
                {
                    "rule_id": rule_id,
                    "page": violation.get("page", 0),
                    "description": violation.get("description", violation.get("help", "")),
                    "location": violation.get("location", ""),
                },
            )

    return {
        "checked": result.checked,
        "passed": result.passed if result.checked else True,
        "error": result.error,
        "violation_count": len(violations),
        "by_rule": dict(rule_counts),
        "by_family": dict(family_counts),
        "violations": normalized,
        "truncated": len(violations) > len(normalized),
    }


def parse_tool_calls(raw: Any) -> list[dict[str, Any]]:
    """Parse a model response into OpenAI-style tool-call records."""

    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        tool_calls = raw.get("tool_calls")
        if isinstance(tool_calls, list):
            return tool_calls
        return []
    if not isinstance(raw, str):
        return []

    text = raw.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
        return parsed["tool_calls"]
    return []


async def _maybe_await(value: Any) -> Any:
    """Await a result when it is awaitable, otherwise return it directly."""

    if inspect.isawaitable(value):
        return await value
    return value


def _message_content_to_text(content: Any) -> str:
    """Normalize OpenAI message content to plain text for logging/messages."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


async def _call_openai_compatible(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str | None,
    config: AgentLoopConfig,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat backend and return the assistant message."""

    payload: dict[str, Any] = {
        "model": model or getattr(client, "model", None) or getattr(client, "_model", None),
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    if payload["model"] is None:
        payload.pop("model")

    if hasattr(client, "_post"):
        data = await asyncio.to_thread(client._post, payload)
        choices = data.get("choices", [])
        if not choices:
            return {"role": "assistant", "content": ""}
        message = dict(choices[0].get("message", {}))
        message.setdefault("role", "assistant")
        return message

    chat = getattr(getattr(client, "chat", None), "completions", None)
    create = getattr(chat, "create", None)
    if create is not None:
        if inspect.iscoroutinefunction(create):
            response = await create(**payload)
        else:
            response = await asyncio.to_thread(create, **payload)
        choices = getattr(response, "choices", None) or response.get("choices", [])
        if not choices:
            return {"role": "assistant", "content": ""}
        first = choices[0]
        message = getattr(first, "message", None) or first.get("message", {})
        result = {
            "role": getattr(message, "role", None) or message.get("role", "assistant"),
            "content": getattr(message, "content", None) or message.get("content", ""),
        }
        tool_calls = getattr(message, "tool_calls", None) or message.get("tool_calls")
        if tool_calls:
            serialized = []
            for tool_call in tool_calls:
                serialized.append(
                    {
                        "id": getattr(tool_call, "id", None) or tool_call.get("id"),
                        "type": getattr(tool_call, "type", None) or tool_call.get("type", "function"),
                        "function": {
                            "name": getattr(getattr(tool_call, "function", None), "name", None)
                            or tool_call.get("function", {}).get("name"),
                            "arguments": getattr(getattr(tool_call, "function", None), "arguments", None)
                            or tool_call.get("function", {}).get("arguments", "{}"),
                        },
                    },
                )
            result["tool_calls"] = serialized
        return result

    generate_raw = getattr(client, "generate_raw", None)
    if generate_raw is None:
        raise TypeError("Client does not expose an OpenAI-compatible chat interface")

    kwargs: dict[str, Any] = {"messages": messages, "tools": tools}
    parameters = inspect.signature(generate_raw).parameters
    if "model" in parameters:
        kwargs["model"] = model
    elif "model_override" in parameters:
        kwargs["model_override"] = model
    if "temperature" in parameters:
        kwargs["temperature"] = config.temperature
    if "max_tokens" in parameters:
        kwargs["max_tokens"] = config.max_tokens
    if "think" in parameters:
        kwargs["think"] = True

    raw_response = await _maybe_await(generate_raw(**kwargs))
    tool_calls = parse_tool_calls(raw_response)
    if tool_calls:
        return {"role": "assistant", "content": "", "tool_calls": tool_calls}
    return {"role": "assistant", "content": _message_content_to_text(raw_response)}


def _safe_page_number(page: int, total_pages: int) -> int:
    """Validate a 1-based page number and convert to 0-based index."""

    if page < 1 or page > total_pages:
        raise ValueError(f"page {page} out of range 1..{total_pages}")
    return page - 1


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    """Return a truncated string plus a truncation flag."""

    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _family_for_tool(name: str) -> str | None:
    """Map a mutating tool name to its violation family bucket."""

    return TOOL_FAMILIES.get(name)


def _load_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Decode the JSON argument payload for one tool call."""

    function = tool_call.get("function", {})
    raw = function.get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_argument_parse_error": raw}


def _tool_response_message(tool_call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build a standard tool response message for the next model turn."""

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload, ensure_ascii=True, default=str),
    }


def _assistant_message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize an assistant message before adding it to history."""

    record = {"role": "assistant"}
    if message.get("content") is not None:
        record["content"] = message.get("content", "")
    if message.get("tool_calls"):
        record["tool_calls"] = message["tool_calls"]
    return record


def _verification_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Describe the change in violation count after a fix."""

    return {
        "before": before.get("violation_count", 0),
        "after": after.get("violation_count", 0),
        "delta": after.get("violation_count", 0) - before.get("violation_count", 0),
        "improved": after.get("violation_count", 0) < before.get("violation_count", 0),
    }


def _top_count_entries(
    counts: dict[str, Any],
    *,
    key_name: str,
    max_items: int = 6,
) -> list[dict[str, Any]]:
    """Return the highest-count items from a rule/family counter mapping."""

    normalized = []
    for key, value in counts.items():
        try:
            normalized.append((str(key), int(value)))
        except (TypeError, ValueError):
            continue
    normalized.sort(key=lambda item: (-item[1], item[0]))
    return [{key_name: key, "count": count} for key, count in normalized[:max_items]]


def _rule_delta(before: dict[str, Any], after: dict[str, Any], *, max_items: int = 6) -> dict[str, Any]:
    """Describe how per-rule counts changed after a mutation."""

    before_counts = {str(key): int(value) for key, value in (before.get("by_rule") or {}).items()}
    after_counts = {str(key): int(value) for key, value in (after.get("by_rule") or {}).items()}

    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []

    for rule_id in sorted(set(before_counts) | set(after_counts)):
        before_count = before_counts.get(rule_id, 0)
        after_count = after_counts.get(rule_id, 0)
        delta = after_count - before_count
        entry = {
            "rule_id": rule_id,
            "before": before_count,
            "after": after_count,
            "delta": delta,
        }
        if delta < 0:
            improved.append(entry)
        elif delta > 0:
            regressed.append(entry)
        elif after_count:
            unchanged.append(entry)

    unchanged.sort(key=lambda item: (-item["after"], item["rule_id"]))
    return {
        "improved_rules": improved[:max_items],
        "regressed_rules": regressed[:max_items],
        "persistent_rules": unchanged[:max_items],
        "top_remaining_rules": _top_count_entries(after_counts, key_name="rule_id", max_items=max_items),
        "top_remaining_families": _top_count_entries(
            {str(key): int(value) for key, value in (after.get("by_family") or {}).items()},
            key_name="family",
            max_items=max_items,
        ),
    }


def _progress_state(*, mutation_ok: bool, delta: dict[str, Any]) -> str:
    """Classify the outcome of a mutating tool call."""

    if not mutation_ok:
        return "tool_error"
    if delta.get("improved"):
        return "improved"
    if delta.get("delta", 0) > 0:
        return "regressed"
    return "no_progress"


def _coerce_mutation_outcome(value: Any) -> MutationOutcome:
    """Normalize mutating tool outputs to a structured result."""

    if isinstance(value, MutationOutcome):
        return value
    if isinstance(value, dict):
        raw_changes = value.get("changes", [])
        if isinstance(raw_changes, list):
            changes = [str(item) for item in raw_changes]
        elif raw_changes in (None, ""):
            changes = []
        else:
            changes = [str(raw_changes)]
        return MutationOutcome(
            ok=bool(value.get("ok", True)),
            changes=changes,
            error=str(value.get("error", "")).strip(),
        )
    if value is None:
        return MutationOutcome(ok=True, changes=[])
    if isinstance(value, list):
        return MutationOutcome(ok=True, changes=[str(item) for item in value])
    return MutationOutcome(ok=True, changes=[str(value)])


class AgentLoop:
    """Closed-loop PDF fixer that exposes repair primitives as model tools."""

    def __init__(
        self,
        *,
        pdf_path: Path,
        client: Any,
        model: str | None = None,
        config: Any | None = None,
        loop_config: AgentLoopConfig | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        vision_client: Any | None = None,
    ):
        self.input_path = Path(pdf_path)
        self.client = client
        self.model = model
        self.pipeline_config = config
        self.loop_config = loop_config or AgentLoopConfig()
        self.system_prompt = system_prompt
        self.vision_client = vision_client or client
        self._family_attempts: Counter[str] = Counter()
        self._family_stagnation: Counter[str] = Counter()
        self._fix_attempts = 0
        self._working_path: Path | None = None
        self._recent_messages: list[dict[str, Any]] = []
        self._attempt_summaries: list[dict[str, Any]] = []
        self._blocked_mutation_signatures: dict[str, dict[str, Any]] = {}

    async def run(self, output_path: Path | None = None) -> dict[str, Any]:
        """Run the remediation loop and return a trace."""

        with tempfile.NamedTemporaryFile(suffix=self.input_path.suffix, delete=False) as tmp:
            working_path = Path(tmp.name)
        shutil.copy2(self.input_path, working_path)
        self._working_path = working_path
        self._recent_messages = []
        self._attempt_summaries = []
        self._blocked_mutation_signatures = {}
        self._family_stagnation = Counter()
        self._family_attempts = Counter()
        self._fix_attempts = 0

        baseline = self._tool_check_verapdf()
        current_verification = baseline
        loop_start = time.time()
        trace: dict[str, Any] = {
            "input_pdf": str(self.input_path),
            "working_pdf": str(working_path),
            "passed": baseline.get("passed", False),
            "status": "fixed" if baseline.get("passed", False) else "in_progress",
            "reason": "already conformant" if baseline.get("passed", False) else "",
            "initial_verapdf": baseline,
            "final_verapdf": baseline,
            "violations_before": baseline.get("violation_count", 0),
            "violations_after": baseline.get("violation_count", 0),
            "elapsed_seconds": 0.0,
            "attempts_used": 0,
            "family_attempts": {},
            "messages": [],
            "tool_history": [],
        }

        if baseline.get("passed", False):
            trace["elapsed_seconds"] = round(time.time() - loop_start, 1)
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(working_path, output_path)
                trace["output_pdf"] = str(output_path)
            working_path.unlink(missing_ok=True)
            return trace

        finish_payload: dict[str, Any] | None = None

        try:
            for _round in range(self.loop_config.max_tool_rounds):
                if self._fix_attempts >= self.loop_config.max_fix_attempts:
                    finish_payload = {
                        "status": "manual_review",
                        "reason": f"Reached fix budget ({self.loop_config.max_fix_attempts} mutating attempts)",
                    }
                    break

                assistant_message = await _call_openai_compatible(
                    client=self.client,
                    messages=self._build_model_messages(
                        baseline=baseline,
                        current_verification=current_verification,
                    ),
                    tools=TOOL_DEFINITIONS,
                    model=self.model,
                    config=self.loop_config,
                )
                assistant_record = _assistant_message_for_history(assistant_message)
                self._append_recent_message(assistant_record)
                trace["messages"].append(assistant_record)

                tool_calls = assistant_message.get("tool_calls") or []
                if not tool_calls:
                    content = _message_content_to_text(assistant_message.get("content", ""))
                    reminder = {
                        "role": "user",
                        "content": (
                            "Use the provided tools. If the document is conformant, call finish. "
                            "If not, inspect and then call exactly one mutating fix tool."
                        ),
                    }
                    if content:
                        reminder["content"] += f"\n\nYour last text response was:\n{content}"
                    self._append_recent_message(reminder)
                    continue

                mutating_seen = False
                for index, tool_call in enumerate(tool_calls):
                    name = tool_call.get("function", {}).get("name", "")
                    arguments = _load_tool_arguments(tool_call)

                    if name in MUTATING_TOOL_NAMES:
                        if mutating_seen:
                            payload = {
                                "ok": False,
                                "error": "Only one mutating tool call is allowed per assistant turn",
                                "tool": name,
                            }
                            self._append_recent_message(
                                _tool_response_message(tool_call.get("id", f"call_{index}"), payload),
                            )
                            trace["tool_history"].append(
                                {
                                    "name": name,
                                    "arguments": arguments,
                                    "output": payload,
                                    "mutated": False,
                                },
                            )
                            continue
                        mutating_seen = True

                    dispatch = await self.dispatch_tool(name, arguments)
                    tool_message = _tool_response_message(tool_call.get("id", f"call_{index}"), dispatch.output)
                    self._append_recent_message(tool_message)
                    trace["tool_history"].append(
                        {
                            "name": dispatch.name,
                            "arguments": dispatch.arguments,
                            "output": dispatch.output,
                            "mutated": dispatch.mutated,
                        },
                    )

                    if dispatch.finish:
                        finish_payload = dispatch.output
                        break

                    if dispatch.name == "check_verapdf" and dispatch.output.get("checked"):
                        current_verification = dispatch.output
                    elif dispatch.mutated and dispatch.output.get("verification"):
                        current_verification = dispatch.output["verification"]

                    if dispatch.mutated and dispatch.output.get("verification", {}).get("passed"):
                        finish_payload = {
                            "status": "fixed",
                            "reason": f"{name} reduced remaining violations to zero",
                        }
                        break

                if finish_payload is not None:
                    break

            if finish_payload is None:
                finish_payload = {
                    "status": "manual_review",
                    "reason": f"Reached tool-round limit ({self.loop_config.max_tool_rounds})",
                }

            final_verification = self._tool_check_verapdf()
            trace["final_verapdf"] = final_verification
            trace["attempts_used"] = self._fix_attempts
            trace["family_attempts"] = dict(self._family_attempts)

            status = finish_payload["status"]
            reason = finish_payload["reason"]
            if status == "fixed" and not final_verification.get("passed"):
                status = "manual_review"
                reason = (
                    "Agent requested fixed but veraPDF still reports "
                    f"{final_verification.get('violation_count', 0)} violations: {reason}"
                )

            trace["status"] = status
            trace["reason"] = reason
            trace["passed"] = final_verification.get("passed", False)
            trace["violations_after"] = final_verification.get("violation_count", 0)
            trace["elapsed_seconds"] = round(time.time() - loop_start, 1)

            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(working_path, output_path)
                trace["output_pdf"] = str(output_path)
            return trace
        finally:
            working_path.unlink(missing_ok=True)

    async def dispatch_tool(self, name: str, arguments: dict[str, Any]) -> ToolDispatchResult:
        """Dispatch one tool call against the current working PDF."""

        if self._working_path is None:
            raise RuntimeError("AgentLoop working path is not initialized")

        if arguments.get("_argument_parse_error") is not None:
            return ToolDispatchResult(
                name=name,
                arguments=arguments,
                output={
                    "ok": False,
                    "error": "Invalid JSON tool arguments",
                    "raw_arguments": arguments["_argument_parse_error"],
                },
            )

        if name == "finish":
            payload = {
                "status": arguments.get("status", "manual_review"),
                "reason": str(arguments.get("reason", "")).strip() or "No reason provided",
            }
            return ToolDispatchResult(
                name=name,
                arguments=arguments,
                output=payload,
                finish=True,
            )

        if name == "check_verapdf":
            return ToolDispatchResult(name=name, arguments=arguments, output=self._tool_check_verapdf())
        if name == "read_structure_tree":
            return ToolDispatchResult(
                name=name,
                arguments=arguments,
                output=self._tool_read_structure_tree(
                    page=arguments.get("page"),
                    max_nodes=arguments.get("max_nodes"),
                    include_text=bool(arguments.get("include_text", True)),
                ),
            )
        if name == "read_content_stream":
            return ToolDispatchResult(
                name=name,
                arguments=arguments,
                output=self._tool_read_content_stream(
                    page=int(arguments.get("page", 0)),
                    max_chars=int(arguments.get("max_chars") or self.loop_config.max_content_chars),
                ),
            )
        if name == "read_page_image":
            output = await self._tool_read_page_image(
                page=int(arguments.get("page", 0)),
                question=str(arguments.get("question", "")).strip(),
            )
            return ToolDispatchResult(name=name, arguments=arguments, output=output)

        if name in MUTATING_TOOL_NAMES:
            family = _family_for_tool(name)
            if (
                family
                and self.loop_config.max_fix_attempts_per_family is not None
                and self._family_attempts[family] >= self.loop_config.max_fix_attempts_per_family
            ):
                return ToolDispatchResult(
                    name=name,
                    arguments=arguments,
                    output={
                        "ok": False,
                        "error": (
                            f"Per-family budget exhausted for {family} "
                            f"({self.loop_config.max_fix_attempts_per_family})"
                        ),
                    },
                )

            before = self._tool_check_verapdf()
            signature = self._mutation_signature(name, arguments, before)
            if self.loop_config.block_repeated_no_progress and signature in self._blocked_mutation_signatures:
                blocked = self._blocked_mutation_signatures[signature]
                return ToolDispatchResult(
                    name=name,
                    arguments=arguments,
                    output={
                        "ok": False,
                        "blocked": True,
                        "error": (
                            "This exact mutating tool call already failed to improve "
                            "the same verification state"
                        ),
                        "last_attempt": blocked,
                        "current_verification": before,
                        "retry_guidance": blocked.get("retry_guidance", ""),
                    },
                )

            try:
                mutation = _coerce_mutation_outcome(self._apply_mutating_tool(name, arguments))
            except Exception as exc:  # pragma: no cover - defensive path
                logger.exception("Mutating tool %s raised unexpectedly", name)
                mutation = MutationOutcome(
                    ok=False,
                    changes=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            after = self._tool_check_verapdf()
            self._fix_attempts += 1
            if family:
                self._family_attempts[family] += 1

            delta = _verification_delta(before, after)
            progress_state = _progress_state(mutation_ok=mutation.ok, delta=delta)
            rule_delta = _rule_delta(before, after)
            retry_guidance = self._build_retry_guidance(
                name=name,
                family=family,
                arguments=arguments,
                before=before,
                after=after,
                progress_state=progress_state,
            )
            attempt_summary = {
                "attempt": self._fix_attempts,
                "tool": name,
                "family": family,
                "arguments": arguments,
                "progress_state": progress_state,
                "delta": delta,
                "top_remaining_rules": rule_delta["top_remaining_rules"],
                "retry_guidance": retry_guidance,
            }
            self._record_attempt_summary(attempt_summary)

            if family:
                if progress_state == "improved":
                    self._family_stagnation[family] = 0
                else:
                    self._family_stagnation[family] += 1

            if progress_state == "improved":
                self._blocked_mutation_signatures.pop(signature, None)
            else:
                self._blocked_mutation_signatures[signature] = attempt_summary

            payload = {
                "ok": mutation.ok,
                "tool": name,
                "changes": mutation.changes,
                "tool_error": mutation.error,
                "verification": after,
                "delta": delta,
                "rule_delta": rule_delta,
                "progress_state": progress_state,
                "tool_effective": bool(delta.get("improved")),
                "retry_guidance": retry_guidance,
                "attempts_used": self._fix_attempts,
                "attempts_remaining": max(0, self.loop_config.max_fix_attempts - self._fix_attempts),
                "family_attempts": dict(self._family_attempts),
                "family_stagnation": dict(self._family_stagnation),
            }
            return ToolDispatchResult(
                name=name,
                arguments=arguments,
                output=payload,
                mutated=True,
            )

        return ToolDispatchResult(
            name=name,
            arguments=arguments,
            output={"ok": False, "error": f"Unknown tool '{name}'"},
        )

    def _append_recent_message(self, message: dict[str, Any]) -> None:
        """Keep only the most recent assistant/tool/user messages in raw form."""

        self._recent_messages.append(message)
        max_messages = max(0, int(self.loop_config.max_recent_messages))
        if max_messages and len(self._recent_messages) > max_messages:
            self._recent_messages = self._recent_messages[-max_messages:]

    def _record_attempt_summary(self, summary: dict[str, Any]) -> None:
        """Store a compact ledger of prior mutating attempts for future turns."""

        self._attempt_summaries.append(summary)
        max_items = max(0, int(self.loop_config.max_attempt_summaries))
        if max_items and len(self._attempt_summaries) > max_items:
            self._attempt_summaries = self._attempt_summaries[-max_items:]

    def _mutation_signature(
        self,
        name: str,
        arguments: dict[str, Any],
        verification: dict[str, Any],
    ) -> str:
        """Fingerprint an exact mutation against the current verification state."""

        payload = {
            "tool": name,
            "arguments": arguments,
            "violation_count": verification.get("violation_count", 0),
            "by_rule": verification.get("by_rule", {}),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)

    def _only_source_font_violations(self, verification: dict[str, Any]) -> bool:
        """Return True when only 7.21.x source-font violations remain."""

        by_rule = verification.get("by_rule") or {}
        if not by_rule:
            return False
        return all(str(rule_id).startswith("7.21") for rule_id in by_rule)

    def _build_strategy_hints(self, current_verification: dict[str, Any]) -> list[str]:
        """Generate dynamic, verification-driven guidance for the next turn."""

        hints: list[str] = []
        if self._attempt_summaries:
            last_attempt = self._attempt_summaries[-1]
            if last_attempt.get("progress_state") != "improved":
                hints.append(
                    "The most recent mutation did not help. Do not repeat it verbatim; inspect again and switch approaches."
                )

        for family, stall_count in sorted(self._family_stagnation.items()):
            if stall_count >= self.loop_config.pikepdf_escalation_threshold:
                hints.append(
                    f"{family} has stalled {stall_count} time(s). After inspection, prefer run_pikepdf_code over another generic fix tool for that family."
                )

        top_violation = next(
            (violation for violation in current_verification.get("violations", []) if violation.get("rule_id")),
            None,
        )
        if top_violation is not None:
            rule_id = top_violation.get("rule_id", "")
            page = top_violation.get("page")
            if page:
                hints.append(
                    f"Highest-priority remaining issue appears to be {rule_id} on page {page}. Inspect that page before the next mutation."
                )
            else:
                hints.append(f"Highest-priority remaining issue appears to be {rule_id}.")

        if self._only_source_font_violations(current_verification):
            hints.append(
                "Only 7.21.x source-font violations remain. Avoid burning the full budget unless a tool result shows real progress."
            )

        return hints

    def _build_runtime_state_message(self, current_verification: dict[str, Any]) -> str:
        """Summarize prior attempts and current state for the next model turn."""

        payload: dict[str, Any] = {
            "current_verapdf": current_verification,
            "attempt_budget": {
                "used": self._fix_attempts,
                "remaining": max(0, self.loop_config.max_fix_attempts - self._fix_attempts),
            },
            "family_attempts": dict(self._family_attempts),
            "family_stagnation": dict(self._family_stagnation),
        }
        if self._attempt_summaries:
            payload["recent_attempts_summary"] = self._attempt_summaries[-self.loop_config.max_attempt_summaries :]
        hints = self._build_strategy_hints(current_verification)
        if hints:
            payload["strategy_hints"] = hints
        return (
            "Current remediation state. Use this compact state instead of relying on the full older transcript.\n"
            + json.dumps(payload, indent=2, ensure_ascii=True, default=str)
        )

    def _build_model_messages(
        self,
        *,
        baseline: dict[str, Any],
        current_verification: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Construct the next model prompt with compact state plus recent raw turns."""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_initial_user_message(self.input_path, baseline, self.loop_config)},
            {"role": "user", "content": self._build_runtime_state_message(current_verification)},
        ]
        messages.extend(self._recent_messages)
        return messages

    def _build_retry_guidance(
        self,
        *,
        name: str,
        family: str | None,
        arguments: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
        progress_state: str,
    ) -> str:
        """Explain what the model should do next after a mutation result."""

        del arguments
        parts: list[str] = []
        if progress_state == "improved":
            parts.append("This mutation helped. Re-read the updated verification before choosing the next fix.")
        else:
            parts.append("Do not repeat this exact mutation against the same verification state.")
            if progress_state == "regressed":
                parts.append("It made things worse.")
            elif progress_state == "tool_error":
                parts.append("The tool itself failed.")
            else:
                parts.append("It did not reduce the remaining violations.")
            if name != "run_pikepdf_code":
                parts.append(
                    "Inspect the affected page/object first, then prefer run_pikepdf_code or a different mutating tool."
                )
            else:
                parts.append(
                    "Inspect again and change the code strategy instead of rerunning the same snippet."
                )

        top_violation = next(
            (violation for violation in (after.get("violations") or before.get("violations") or []) if violation.get("rule_id")),
            None,
        )
        if top_violation is not None and top_violation.get("page"):
            parts.append(f"Focus next inspection on page {top_violation['page']}.")
        if family:
            parts.append(f"Current family: {family}.")
        return " ".join(parts)

    def _tool_check_verapdf(self) -> dict[str, Any]:
        """Run veraPDF on the current working PDF."""

        assert self._working_path is not None
        result = validate_with_verapdf(self._working_path, config=self.pipeline_config)
        return summarize_verapdf_result(result)

    def _tool_read_structure_tree(
        self,
        *,
        page: int | None,
        max_nodes: int | None,
        include_text: bool,
    ) -> dict[str, Any]:
        """Inspect the structure tree in a compact, page-aware format."""

        assert self._working_path is not None
        with pikepdf.open(self._working_path) as pdf:
            requested_page_idx = None
            if page is not None:
                requested_page_idx = _safe_page_number(int(page), len(pdf.pages))

            page_text: dict[int, dict[int, str]] = {}
            if include_text:
                for page_idx, pdf_page in enumerate(pdf.pages):
                    if requested_page_idx is None or requested_page_idx == page_idx:
                        page_text[page_idx] = _extract_mcid_text(pdf_page)

            limit = int(max_nodes or self.loop_config.max_structure_nodes)
            nodes: list[dict[str, Any]] = []

            for node, depth, _parent in walk_structure_tree(pdf):
                tag = _get_struct_type(node)
                if not tag or tag == "StructTreeRoot":
                    continue
                node_page = _find_node_page(node, pdf)
                if requested_page_idx is not None and node_page != requested_page_idx:
                    continue

                mcids = _get_node_mcids(node)
                entry = {
                    "page": node_page + 1 if node_page is not None else None,
                    "depth": depth,
                    "tag": tag,
                    "mcids": mcids[:12],
                    "child_count": 0 if node.get("/K") is None else len(list(node.get("/K"))) if isinstance(node.get("/K"), pikepdf.Array) else 1,
                    "alt_text": str(node.get("/Alt", "")).strip() if node.get("/Alt") is not None else "",
                    "obj_id": f"obj_{node.objgen[0]}_{node.objgen[1]}" if getattr(node, "objgen", None) else "",
                }
                if include_text and node_page is not None and mcids:
                    excerpts = []
                    for mcid in mcids[:4]:
                        text = page_text.get(node_page, {}).get(mcid, "").strip()
                        if text:
                            excerpts.append(text)
                    if excerpts:
                        text_value, truncated = _truncate_text(" ".join(excerpts), 300)
                        entry["text"] = text_value
                        entry["text_truncated"] = truncated
                nodes.append(entry)
                if len(nodes) >= limit:
                    break

            return {
                "ok": True,
                "page": page,
                "nodes": nodes,
                "truncated": len(nodes) >= limit,
            }

    def _tool_read_content_stream(self, *, page: int, max_chars: int) -> dict[str, Any]:
        """Inspect one page's decoded content stream."""

        assert self._working_path is not None
        with pikepdf.open(self._working_path) as pdf:
            page_idx = _safe_page_number(page, len(pdf.pages))
            raw = _read_page_content(pdf.pages[page_idx]).decode("latin-1", errors="replace")
            text, truncated = _truncate_text(raw, max_chars)
            return {
                "ok": True,
                "page": page,
                "content": text,
                "truncated": truncated,
                "length": len(raw),
                "bdc_count": raw.count("BDC"),
                "emc_count": raw.count("EMC"),
            }

    async def _tool_read_page_image(self, *, page: int, question: str) -> dict[str, Any]:
        """Render a page and ask the configured vision backend a question."""

        assert self._working_path is not None
        with pikepdf.open(self._working_path) as pdf:
            _safe_page_number(page, len(pdf.pages))
        prompt = question or (
            "Describe the page layout, semantic regions, and any likely PDF/UA "
            "issues such as headings, tables, lists, links, decorative pagination, or artifacts."
        )
        image_path = render_page_to_image(self._working_path, page_num=page, dpi=144)
        try:
            vision_client = self.vision_client
            if hasattr(vision_client, "generate_vision"):
                image_bytes = image_path.read_bytes()
                generate_vision = vision_client.generate_vision
                if inspect.iscoroutinefunction(generate_vision):
                    response = await generate_vision(image_bytes, prompt)
                else:
                    response = await asyncio.to_thread(generate_vision, image_bytes, prompt)
            elif hasattr(vision_client, "analyze_image"):
                response = await _maybe_await(vision_client.analyze_image(image_path, prompt))
            else:
                return {
                    "ok": False,
                    "error": "No vision-capable client available for read_page_image",
                }
            return {
                "ok": True,
                "page": page,
                "question": prompt,
                "response": str(response).strip(),
            }
        finally:
            image_path.unlink(missing_ok=True)

    def _apply_mutating_tool(self, name: str, arguments: dict[str, Any]) -> MutationOutcome | list[str]:
        """Apply one mutating tool to the working PDF in place."""

        assert self._working_path is not None
        with pikepdf.open(self._working_path, allow_overwriting_input=True) as pdf:
            if name == "fix_untagged_content":
                changes = fix_untagged_content(pdf)
            elif name == "fix_heading_nesting":
                changes = fix_heading_nesting(pdf)
            elif name == "fix_table_structure":
                changes = []
                changes.extend(fix_table_parent_structure(pdf))
            elif name == "fix_table_headers":
                changes = []
                changes.extend(fix_table_headers(pdf))
                changes.extend(fix_table_header_scope(pdf))
                changes.extend(fix_table_td_headers(pdf))
            elif name == "fix_list_structure":
                changes = fix_list_structure(pdf)
            elif name == "fix_annotations":
                changes = []
                changes.extend(fix_annotations_tagged(pdf))
                changes.extend(fix_form_fields_tagged(pdf))
                changes.extend(fix_link_annotations(pdf))
                changes.extend(fix_annotation_descriptions(pdf))
                changes.extend(fix_form_field_descriptions(pdf))
            elif name == "fix_tab_order":
                changes = fix_tab_order(pdf)
            elif name == "fix_language":
                changes = fix_language(pdf, language=str(arguments.get("language", "en")))
            elif name == "fix_viewer_preferences":
                changes = _fix_viewer_preferences(pdf)
            elif name == "fix_pagination_markers":
                changes = _fix_pagination_to_artifact(pdf)
            elif name == "fix_bdc_emc_balance":
                changes = fix_bdc_emc_balance(pdf)
            elif name == "mark_as_artifact":
                page_idx = _safe_page_number(int(arguments["page"]), len(pdf.pages))
                mcids = [int(mcid) for mcid in arguments.get("mcids", [])]
                reason = str(arguments.get("reason", "")).strip() or "Agent requested artifactization"
                changes = [
                    result.detail
                    for result in _do_artifactize(
                        pdf,
                        page_idx,
                        mcids,
                        reason,
                        allow_artifactize=self.loop_config.allow_unsafe_artifactize,
                        op_id="agent_loop.mark_as_artifact",
                    )
                ]
            elif name == "set_tag":
                page_idx = _safe_page_number(int(arguments["page"]), len(pdf.pages))
                mcid = int(arguments["mcid"])
                tag = str(arguments["tag"]).strip()
                changes = [
                    result.detail
                    for result in _do_set_tag(
                        pdf,
                        page_idx,
                        [mcid],
                        {"tag": tag},
                        op_id="agent_loop.set_tag",
                    )
                ]
            elif name == "run_pikepdf_code":
                code = str(arguments.get("code", ""))
                changes = self._execute_pikepdf_code(pdf, code)
            else:
                raise ValueError(f"Unsupported mutating tool: {name}")

            _normalize_structure_tree_indirect_objects(pdf)
            pdf.save(
                str(self._working_path),
                object_stream_mode=pikepdf.ObjectStreamMode.disable,
            )
        return changes

    def _execute_pikepdf_code(self, pdf: pikepdf.Pdf, code: str) -> MutationOutcome:
        """Execute agent-written pikepdf code in a restricted namespace."""
        import re as _re

        if not code.strip():
            return MutationOutcome(ok=False, changes=[], error="No code provided")

        namespace: dict[str, Any] = {
            "pdf": pdf,
            "pikepdf": pikepdf,
            "re": _re,
            "Path": Path,
            "result": None,
        }

        try:
            exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
        except Exception as e:
            return MutationOutcome(
                ok=False,
                changes=[],
                error=f"{type(e).__name__}: {e}",
            )

        result_val = namespace.get("result")
        changes: list[str] = []
        if result_val is not None:
            changes.append(f"pikepdf code result: {result_val}")
        if not changes:
            changes.append("pikepdf code executed successfully")
        return MutationOutcome(ok=True, changes=changes)


async def run_agent_loop(
    pdf_path: Path,
    *,
    client: Any,
    output_path: Path | None = None,
    model: str | None = None,
    config: Any | None = None,
    loop_config: AgentLoopConfig | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    vision_client: Any | None = None,
) -> dict[str, Any]:
    """Run the Tier 3 agentic fixer against one PDF."""

    loop = AgentLoop(
        pdf_path=pdf_path,
        client=client,
        model=model,
        config=config,
        loop_config=loop_config,
        system_prompt=system_prompt,
        vision_client=vision_client,
    )
    return await loop.run(output_path=output_path)
