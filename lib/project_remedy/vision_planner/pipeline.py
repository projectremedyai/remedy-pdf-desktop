"""Orchestrate the full vision-plan pipeline: grounder -> planner -> executor.

Includes optional experiment tracking for Meta-Harness auto-prompt evolution.
When an EvolutionLoop is provided, each pipeline run records its results
and can trigger harness evolution when success rate drops.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from project_remedy.pdf_acceptance import validate_with_verapdf
from project_remedy.vision_planner.anchor_graph import build_anchor_graph
from project_remedy.vision_planner.executor import execute_plan, post_repair
from project_remedy.vision_planner.grounder import run_grounder
from project_remedy.vision_planner.planner import run_planner

logger = logging.getLogger(__name__)


def _count_violations(verapdf_result) -> int:
    """Extract violation count from a VeraPDFResult."""
    if not verapdf_result.checked:
        return 0
    return len(verapdf_result.violations)


async def _greedy_execute(
    pdf_path: Path,
    plan: dict,
    anchor_graph: dict,
    config: Any,
    original_path: Path | None = None,
) -> dict:
    """Apply operations greedily: test each on a temp copy, keep only improvements.

    For each operation in the plan, we:
    1. Copy the current-best PDF to a temp file
    2. Apply the single operation
    3. Run veraPDF to count violations
    4. Accept if violations decreased or stayed the same; reject otherwise

    Returns a dict with ``applied``, ``skipped``, ``errors``, ``greedy``,
    and ``final_violations`` keys.
    """
    operations = plan.get("operations", [])
    if not operations:
        return {"applied": [], "skipped": [], "errors": [], "greedy": True}

    # Baseline violation count on current PDF
    baseline_result = validate_with_verapdf(pdf_path, config=config)
    baseline_violations = _count_violations(baseline_result)

    accepted_ops: list[dict] = []
    rejected_ops: list[dict] = []
    current_best = pdf_path  # path to the evolving best version

    for op in operations:
        op_id = op.get("op_id", "?")
        action = op.get("action", "?")

        # Create a temp INPUT copy and a separate OUTPUT path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
            tmp_input = Path(tmp_in.name)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_out:
            tmp_output_op = Path(tmp_out.name)
        shutil.copy2(str(current_best), str(tmp_input))

        # Apply just this ONE operation (separate input/output paths)
        single_plan = {"operations": [op], "confidence": plan.get("confidence", 0)}
        try:
            execute_plan(tmp_input, tmp_output_op, single_plan, anchor_graph)
        except Exception as e:
            logger.warning(
                "greedy: op %s (%s) execute failed: %s — skipping", op_id, action, e
            )
            rejected_ops.append({"op": op, "reason": f"execution failed: {e}"})
            tmp_input.unlink(missing_ok=True)
            tmp_output_op.unlink(missing_ok=True)
            continue

        # Check if it helped
        try:
            post_result = validate_with_verapdf(tmp_output_op, config=config)
            post_violations = _count_violations(post_result)

            if post_violations <= baseline_violations:
                # Improvement or no change — accept this op
                verb = "improved" if post_violations < baseline_violations else "neutral"
                logger.info(
                    "greedy: ACCEPT op %s (%s) — %s (%d -> %d violations)",
                    op_id, action, verb, baseline_violations, post_violations,
                )
                accepted_ops.append(op)
                shutil.copy2(str(tmp_output_op), str(current_best))
                baseline_violations = post_violations
            else:
                # Regression — reject
                logger.info(
                    "greedy: REJECT op %s (%s) — violations increased %d -> %d",
                    op_id, action, baseline_violations, post_violations,
                )
                rejected_ops.append({
                    "op": op,
                    "reason": f"violations increased {baseline_violations} -> {post_violations}",
                })
        except Exception as e:
            logger.warning(
                "greedy: op %s (%s) evaluation failed: %s — skipping", op_id, action, e
            )
            rejected_ops.append({"op": op, "reason": f"evaluation failed: {e}"})
        finally:
            tmp_input.unlink(missing_ok=True)
            tmp_output_op.unlink(missing_ok=True)

    return {
        "applied": [
            {"op_id": op.get("op_id"), "action": op.get("action")} for op in accepted_ops
        ],
        "skipped": rejected_ops,
        "errors": [],
        "greedy": True,
        "final_violations": baseline_violations,
    }


async def run_vision_plan(
    pdf_path: Path,
    output_path: Path | None,
    harness: Any,
    client: Any,
    model: str | None = None,
    config: Any | None = None,
    evolution_loop: Any | None = None,
    harness_id: str | None = None,
    pdf_output_path: Path | None = None,
    greedy_validate: bool = True,
    replan_passes: int = 1,
) -> dict:
    """Run the full vision-plan pipeline. Returns trace dict.

    Args:
        pdf_path: Input PDF file path.
        output_path: Where to write the trace JSON (optional).
        harness: VisionPlannerHarness instance.
        client: LLM client with generate_raw().
        model: Override model name (optional).
        config: PipelineConfig for veraPDF etc.
        evolution_loop: Optional EvolutionLoop for experiment tracking.
        harness_id: Identifier for the harness variant being used.
        pdf_output_path: Where to write the remediated PDF (optional).
            If not provided, the remediated PDF is discarded after validation.
        greedy_validate: When True (default), apply operations one at a time
            and keep only those that reduce (or don't increase) violations.
            When False, apply all operations at once (legacy behavior).
        replan_passes: Number of replan attempts when greedy execution rejects
            operations and violations remain (default 1). Set to 0 to disable.
    """
    t0 = time.time()
    trace: dict[str, Any] = {
        "pdf": str(pdf_path),
        "passed": False,
        "violations_before": 0,
        "violations_after": 0,
        "violations_after_repair": 0,
        "elapsed_seconds": 0.0,
        "plan": {"confidence": 0, "operations": [], "manual_review": []},
        "grounder_prompt": [],
        "grounder_response": "",
        "planner_prompt": [],
        "planner_response": "",
        "failure_reasons": [],
        "post_repair_changes": [],
        "escalated": False,
        "error": None,
    }

    try:
        # 1. Get violations before remediation
        verapdf_before = validate_with_verapdf(pdf_path, config=config)
        violations_list = verapdf_before.violations if verapdf_before.checked else []
        trace["violations_before"] = len(violations_list)

        # Normalize violations for the harness
        normalized_violations: list[dict] = []
        for v in violations_list:
            normalized_violations.append({
                "rule_id": v.get("id", v.get("rule_id", "")),
                "description": v.get("description", v.get("help", "")),
                "page": v.get("page", 0),
                "location": v.get("location", ""),
            })
        trace["violations_list"] = normalized_violations

        # 1b. Deterministic rule router — fix known rules without AI
        from project_remedy.vision_planner.rule_router import (
            route_violations,
            apply_deterministic_fixes,
        )

        det_violations, ai_violations = route_violations(normalized_violations)
        if det_violations:
            det_changes, det_fixed = apply_deterministic_fixes(
                pdf_path, det_violations
            )
            trace["deterministic_fixes"] = det_changes
            trace["deterministic_violations_routed"] = len(det_violations)
            logger.info(
                "rule_router: %d deterministic violations routed, %d fixes applied",
                len(det_violations),
                det_fixed,
            )

            # Re-check violations after deterministic fixes
            verapdf_post_det = validate_with_verapdf(pdf_path, config=config)
            post_det_violations = verapdf_post_det.violations if verapdf_post_det.checked else []
            trace["violations_after_deterministic"] = len(post_det_violations)

            # Update normalized_violations to only include remaining issues
            normalized_violations = []
            for v in post_det_violations:
                normalized_violations.append({
                    "rule_id": v.get("id", v.get("rule_id", "")),
                    "description": v.get("description", v.get("help", "")),
                    "page": v.get("page", 0),
                    "location": v.get("location", ""),
                })

            if not normalized_violations:
                trace["passed"] = True
                trace["violations_after"] = 0
                trace["failure_reasons"] = []
                logger.info("All violations fixed by deterministic router — skipping AI planner")
                return trace
        else:
            trace["deterministic_fixes"] = []
            trace["deterministic_violations_routed"] = 0

        # 2. Build anchor graph
        anchor_graph = build_anchor_graph(pdf_path)

        # 3-6. Run grounder + planner (errors here should not prevent post-repair)
        plan = {"confidence": 0, "operations": [], "manual_review": []}
        semantic_map: dict = {"pages": []}  # default; overwritten on grounder success
        try:
            grounder_result = await run_grounder(pdf_path, harness, client, model)
            trace["grounder_prompt"] = grounder_result["grounder_prompts"]
            trace["grounder_response"] = grounder_result["grounder_responses"]

            semantic_map = {"pages": grounder_result["pages"]}
            filtered_violations = harness.filter_violations(normalized_violations)

            # Pass page images to planner so it can see what it's planning for
            page_images = grounder_result.get("page_images")

            planner_result = await run_planner(
                semantic_map, filtered_violations, anchor_graph, harness, client, model,
                page_images=page_images,
            )
            plan = planner_result["plan"]
            trace["plan"] = plan
            trace["planner_prompt"] = planner_result["planner_prompt"]
            trace["planner_response"] = planner_result["planner_response"]

            confidence = plan.get("confidence", 0)
            threshold = harness.confidence_threshold()
            if confidence < threshold:
                for op in plan.get("operations", []):
                    op["action"] = "mark_manual_review"
                    op["reason"] = f"confidence {confidence:.2f} < threshold {threshold:.2f}: {op.get('reason', '')}"
                trace["failure_reasons"].append(
                    f"confidence {confidence:.2f} below threshold {threshold:.2f}"
                )
        except Exception as e:
            logger.warning("Grounder/planner failed: %s — proceeding with empty plan + post-repair", e)
            trace["failure_reasons"].append(f"grounder/planner error: {e}")

        # 7. Execute plan (targeted operations)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_output = Path(tmp.name)

        if greedy_validate and plan.get("operations"):
            # --- Greedy validated execution: one op at a time ---
            logger.info(
                "greedy execute: %d operations for %s",
                len(plan.get("operations", [])), pdf_path.name,
            )
            # Copy source to tmp_output so greedy executor modifies in-place
            shutil.copy2(str(pdf_path), str(tmp_output))

            greedy_result = await _greedy_execute(
                tmp_output, plan, anchor_graph, config,
                original_path=pdf_path,
            )

            trace["exec_applied"] = greedy_result.get("applied", [])
            trace["exec_skipped"] = greedy_result.get("skipped", [])
            trace["exec_errors"] = greedy_result.get("errors", [])
            trace["greedy"] = True

            # --- Replan passes: if ops were rejected and violations remain ---
            rejected_ops = greedy_result.get("skipped", [])
            remaining_violations = greedy_result.get("final_violations", 0)
            replan_traces: list[dict] = []

            for pass_num in range(replan_passes):
                if remaining_violations == 0:
                    logger.info("greedy replan: 0 violations remain, skipping replan pass %d", pass_num + 1)
                    break
                if not rejected_ops:
                    logger.info("greedy replan: no rejected ops, skipping replan pass %d", pass_num + 1)
                    break

                logger.info(
                    "greedy replan pass %d/%d: %d violations remain, %d ops were rejected",
                    pass_num + 1, replan_passes, remaining_violations, len(rejected_ops),
                )

                # Get fresh violations from the current tmp_output
                replan_verapdf = validate_with_verapdf(tmp_output, config=config)
                replan_violations_list = replan_verapdf.violations if replan_verapdf.checked else []
                replan_normalized: list[dict] = []
                for v in replan_violations_list:
                    replan_normalized.append({
                        "rule_id": v.get("id", v.get("rule_id", "")),
                        "description": v.get("description", v.get("help", "")),
                        "page": v.get("page", 0),
                        "location": v.get("location", ""),
                    })

                # Build rejection context for the planner
                rejection_context = []
                for rej in rejected_ops:
                    rej_op = rej.get("op", {})
                    rejection_context.append(
                        f"- {rej_op.get('action', '?')} on {rej_op.get('target_anchors', '?')}: "
                        f"{rej.get('reason', 'unknown')}"
                    )
                rejection_summary = "\n".join(rejection_context)

                # Re-run planner with remaining violations + rejection context
                try:
                    replan_filtered = harness.filter_violations(replan_normalized)
                    # Inject rejection context into violations so planner sees it
                    augmented_violations = list(replan_filtered) + [{
                        "rule_id": "replan-context",
                        "description": (
                            "These operations were tried and rejected "
                            "(they increased violations or failed):\n"
                            + rejection_summary
                            + "\nTry different approaches."
                        ),
                        "page": 0,
                        "location": "",
                    }]

                    replan_result = await run_planner(
                        semantic_map, augmented_violations, anchor_graph,
                        harness, client, model,
                    )
                    replan_plan = replan_result["plan"]

                    if not replan_plan.get("operations"):
                        logger.info("greedy replan pass %d: planner returned no operations", pass_num + 1)
                        break

                    logger.info(
                        "greedy replan pass %d: planner proposed %d new operations",
                        pass_num + 1, len(replan_plan.get("operations", [])),
                    )

                    replan_greedy = await _greedy_execute(
                        tmp_output, replan_plan, anchor_graph, config,
                        original_path=pdf_path,
                    )

                    replan_trace = {
                        "pass": pass_num + 1,
                        "applied": replan_greedy.get("applied", []),
                        "skipped": replan_greedy.get("skipped", []),
                        "final_violations": replan_greedy.get("final_violations", 0),
                    }
                    replan_traces.append(replan_trace)

                    # Update for next iteration
                    trace["exec_applied"].extend(replan_greedy.get("applied", []))
                    trace["exec_skipped"].extend(replan_greedy.get("skipped", []))
                    rejected_ops = replan_greedy.get("skipped", [])
                    remaining_violations = replan_greedy.get("final_violations", 0)

                except Exception as e:
                    logger.warning("greedy replan pass %d failed: %s", pass_num + 1, e)
                    replan_traces.append({"pass": pass_num + 1, "error": str(e)})
                    break

            if replan_traces:
                trace["replan_passes"] = replan_traces

        else:
            # --- Legacy bulk execution ---
            exec_result = execute_plan(pdf_path, tmp_output, plan, anchor_graph)

            if isinstance(exec_result, dict):
                trace["exec_applied"] = exec_result.get("applied", [])
                trace["exec_skipped"] = exec_result.get("skipped", [])
                trace["exec_errors"] = exec_result.get("errors", [])
            else:
                trace["exec_applied"] = getattr(exec_result, "applied", [])
                trace["exec_skipped"] = getattr(exec_result, "skipped", [])
                trace["exec_errors"] = getattr(exec_result, "errors", [])
            trace["greedy"] = False

        exec_errors = trace.get("exec_errors", [])
        if exec_errors:
            trace["failure_reasons"].extend(
                e if isinstance(e, str) else e.get("detail", str(e)) for e in exec_errors
            )

        # 8. Bounded post-repair (structural integrity fixes)
        repair_changes = post_repair(tmp_output)
        trace["post_repair_changes"] = repair_changes

        # 9. Check conformance after post-repair
        verapdf_mid = validate_with_verapdf(tmp_output, config=config)
        violations_mid = verapdf_mid.violations if verapdf_mid.checked else []
        trace["violations_after_repair"] = len(violations_mid)

        # 10. Escalate to fix_and_verify if still failing
        escalated = False
        if verapdf_mid.checked and not verapdf_mid.passed:
            try:
                from project_remedy.pdf_fixer import fix_and_verify
                fix_and_verify(tmp_output, tmp_output, config=config, max_cycles=2)
                escalated = True
            except Exception as e:
                trace["failure_reasons"].append(f"escalation failed: {e}")

        trace["escalated"] = escalated

        # 11. Final veraPDF check
        verapdf_after = validate_with_verapdf(tmp_output, config=config)
        violations_after = verapdf_after.violations if verapdf_after.checked else []
        trace["violations_after"] = len(violations_after)
        trace["passed"] = verapdf_after.passed if verapdf_after.checked else False

        # 12. "Accept only if better" gate -- revert if VP made things
        # the same or worse (prevents common regression pattern e.g. 2->4).
        violations_before_count = trace["violations_before"]
        violations_after_count = len(violations_after)
        if violations_before_count > 0 and violations_after_count >= violations_before_count:
            logger.warning(
                "VP plan did not reduce violations (before=%d, after=%d) "
                "for %s — reverting to original PDF",
                violations_before_count,
                violations_after_count,
                pdf_path.name,
            )
            # Revert: copy original PDF back over the temp output
            shutil.copy2(str(pdf_path), str(tmp_output))
            trace["passed"] = False
            trace["failure_reasons"].append(
                f"VP plan did not reduce violations "
                f"(before={violations_before_count}, after={violations_after_count})"
            )
            trace["vp_reverted"] = True
        else:
            trace["vp_reverted"] = False

        # Persist remediated PDF if caller wants it and it passed
        if pdf_output_path and trace["passed"] and tmp_output.exists():
            pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_output, pdf_output_path)
            trace["pdf_output"] = str(pdf_output_path)

        # Clean up temp file
        try:
            tmp_output.unlink()
        except OSError:
            pass

    except Exception as e:
        logger.error("Vision-plan pipeline failed: %s", e)
        trace["error"] = str(e)
        trace["failure_reasons"].append(str(e))

    trace["elapsed_seconds"] = round(time.time() - t0, 1)

    # Record experiment for Meta-Harness evolution tracking
    if evolution_loop is not None and harness_id:
        try:
            from project_remedy.vision_planner.evolution import classify_document_type

            doc_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
            doc_type = classify_document_type(
                trace.get("violations_list", []),
                trace.get("page_count", 1),
            )
            evolution_loop.record_result(
                harness_id=harness_id,
                document_hash=doc_hash,
                document_type=doc_type,
                trace=trace,
            )
        except Exception as evo_err:
            logger.debug("Experiment recording failed (non-fatal): %s", evo_err)

    # Write trace JSON if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(trace, indent=2, default=str) + "\n")

    return trace
