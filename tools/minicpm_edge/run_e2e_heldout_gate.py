#!/usr/bin/env python3
"""Run the end-to-end heldout PDF remediation gate against a live Remedy backend.

This is the final promotion gate from ``plans/edge-minicpm-v46-lora-router.txt``:
adapter-level metrics are necessary but not sufficient — the router must survive
a real ``/v1/remediate`` round trip over heldout PDFs.

For each heldout PDF the harness:

1. ``POST /v1/remediate?quality=true`` (multipart upload) -> job id
2. polls ``GET /v1/jobs/{id}`` until ``done`` / ``failed``
3. resolves the quality-layer block, trying in order:
   a. ``<job-dir>/<job_id>/report/*.json`` -> ``quality_result`` (``--job-dir``,
      local backends only; free, no extra judge pass)
   b. ``GET /v1/jobs/{id}/report`` when it happens to serve JSON
   c. ``POST /v1/quality/audit/pdf`` on the remediated output from
      ``GET /v1/jobs/{id}/result`` (always works; re-runs the judges)
4. scores per-dimension outcomes, optionally against corpus annotations

Note: on the current backend ``/v1/jobs/{id}/report`` serves **HTML** (the ACR),
so step (b) is normally skipped. ``?quality=true`` writes ``quality_result``
into the report's JSON sibling, which only step (a) can reach.

Two corpus modes:

* ``--corpus DIR``      — every ``*.pdf`` under DIR (recursive), scored by the
  backend quality layer alone.
* ``--manifest PATH``   — a ``tools/corpus_annotations/v1/manifest.jsonl``; the
  ``known_bad`` artifact is remediated and scored against the annotation's
  per-dimension gold scores.

Contamination guard (on by default): any PDF whose stem matches a ``meta.doc_id``
seen in ``data/tasks/*/{train,val}.jsonl`` is skipped. Stems are normalised by
stripping a leading 12-hex-char content-hash prefix (``0e29771b214b_Foo.pdf`` ->
``Foo``) because the LAMC pools are hash-prefixed while the training doc_ids are
not — matching raw stems silently leaks hundreds of training documents.

Usage::

    ./run_e2e_heldout_gate.py \
        --corpus /path/to/heldout_pdfs \
        --backend-url http://127.0.0.1:8000 \
        --out eval_runs/e2e_gate_v1

Exit codes: 0 gate passed, 2 gate failed, 1 harness error.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minicpm_edge.constants import DEFAULT_DATA_ROOT, DEFAULT_EVAL_ROOT

# Dimensions the backend quality layer reports for PDFs.
PDF_DIMENSIONS: tuple[str, ...] = (
    "alt_text",
    "reading_order",
    "heading_semantics",
    "table_structure",
    "link_text",
    "decorative",
    "complex_content",
)

# ``0e29771b214b_2025-26 HELPING HAND_7FINAL.pdf`` -> ``2025-26 HELPING HAND_7FINAL``
HASH_PREFIX = re.compile(r"^[0-9a-f]{12}_")

TERMINAL_STATUSES = frozenset({"done", "failed"})
PDF_MAGIC = b"%PDF"


@dataclass
class DocRecord:
    doc_id: str
    source: str
    status: str
    job_id: str = ""
    elapsed_seconds: float = 0.0
    overall_pass: bool = False
    dimensions: dict[str, Any] = field(default_factory=dict)
    failing_dimensions: list[str] = field(default_factory=list)
    not_applicable_dimensions: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    scored_via: str = ""
    error: str = ""
    completed_at: str = ""
    # --wcag-verify additions (absent/empty when the flag is off; existing
    # consumers of records.jsonl keep working unchanged).
    wcag_overall_pass: bool | None = None
    wcag_failing_criteria: list[str] = field(default_factory=list)
    wcag_pages_verified: int = 0
    wcag_total_findings: int = 0
    wcag_seconds: float = 0.0
    wcag_error: str = ""
    contrast_audit_seconds: float = 0.0
    contrast_audit_error: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_stem(stem: str) -> str:
    return HASH_PREFIX.sub("", stem)


# ----------------------------------------------------------------------
# HTTP (stdlib only, mirroring eval_router_readiness.py)
# ----------------------------------------------------------------------


def _headers(api_key: str) -> dict[str, str]:
    return {"x-api-key": api_key} if api_key else {}


def _request(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local backend URL
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _get_json(url: str, api_key: str, timeout: float = 60.0) -> dict[str, Any]:
    code, body = _request(url, headers=_headers(api_key), timeout=timeout)
    if code >= 400:
        raise RuntimeError(f"GET {url} -> {code}: {body[:300].decode('utf-8', 'replace')}")
    return json.loads(body)


def _encode_multipart(filename: str, payload: bytes) -> tuple[bytes, str]:
    boundary = f"----remedy{uuid.uuid4().hex}"
    safe = filename.replace('"', "")
    body = b"".join(
        [
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{safe}"\r\n'.encode(),
            b"Content-Type: application/pdf\r\n\r\n",
            payload,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _post_file(url: str, path: Path, api_key: str, timeout: float = 300.0) -> tuple[int, bytes]:
    body, content_type = _encode_multipart(path.name, path.read_bytes())
    headers = {"Content-Type": content_type, **_headers(api_key)}
    return _request(url, method="POST", data=body, headers=headers, timeout=timeout)


# ----------------------------------------------------------------------
# Corpus discovery + contamination guard
# ----------------------------------------------------------------------


def trained_doc_ids(data_root: Path) -> set[str]:
    """Collect every ``meta.doc_id`` present in any task's train/val split."""
    seen: set[str] = set()
    for split in sorted(data_root.glob("*/train.jsonl")) + sorted(data_root.glob("*/val.jsonl")):
        for line in split.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                meta = json.loads(line).get("meta") or {}
            except json.JSONDecodeError:
                continue
            doc_id = meta.get("doc_id")
            if doc_id:
                seen.add(str(doc_id))
    return seen


def is_real_pdf(path: Path) -> bool:
    """Guard against placeholder artifacts (the v1 corpus ships 40-byte stubs)."""
    try:
        with path.open("rb") as handle:
            return handle.read(4) == PDF_MAGIC
    except OSError:
        return False


@dataclass
class Candidate:
    doc_id: str
    path: Path
    annotation: dict[str, Any] | None = None


def candidates_from_corpus(corpus: Path) -> list[Candidate]:
    return [
        Candidate(doc_id=_normalise_stem(p.stem), path=p)
        for p in sorted(corpus.rglob("*.pdf"))
    ]


def candidates_from_list(list_path: Path) -> list[Candidate]:
    """Read a newline-delimited PDF path list (see build_gate_corpus.py).

    Blank lines and ``#`` comments are ignored; ``~`` is expanded. Order is
    preserved so a curated corpus keeps its intended coverage under ``--limit``.
    """
    out: list[Candidate] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line).expanduser()
        out.append(Candidate(doc_id=_normalise_stem(path.stem), path=path))
    return out


def candidates_from_manifest(manifest: Path, corpus_root: Path) -> list[Candidate]:
    """Read a corpus_annotations manifest.jsonl; remediate the known-bad artifact."""
    out: list[Candidate] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("format") != "pdf":
            continue
        bad_paths = entry.get("known_bad_artifact_paths") or []
        if not bad_paths:
            continue
        annotation_path = corpus_root / entry["annotation_path"]
        annotation = (
            json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation_path.exists()
            else None
        )
        out.append(
            Candidate(
                doc_id=entry["doc_id"],
                path=corpus_root / bad_paths[0],
                annotation=annotation,
            )
        )
    return out


def partition(
    items: list[Candidate],
    *,
    seen: set[str],
    allow_contaminated: bool,
) -> tuple[list[Candidate], list[dict[str, str]]]:
    keep: list[Candidate] = []
    skipped: list[dict[str, str]] = []
    for cand in items:
        if not cand.path.exists():
            skipped.append({"doc_id": cand.doc_id, "reason": "missing_file", "path": str(cand.path)})
            continue
        if not is_real_pdf(cand.path):
            skipped.append(
                {"doc_id": cand.doc_id, "reason": "not_a_pdf_placeholder", "path": str(cand.path)}
            )
            continue
        if not allow_contaminated and cand.doc_id in seen:
            skipped.append({"doc_id": cand.doc_id, "reason": "in_training_split", "path": str(cand.path)})
            continue
        keep.append(cand)
    return keep, skipped


# ----------------------------------------------------------------------
# Remediation round trip
# ----------------------------------------------------------------------


def submit(backend: str, path: Path, api_key: str, *, allow_semantic_rebuild: bool) -> str:
    url = f"{backend}/v1/remediate?quality=true"
    if allow_semantic_rebuild:
        url += "&allow_semantic_rebuild=true"
    code, body = _post_file(url, path, api_key)
    if code not in (200, 202):
        raise RuntimeError(f"submit -> {code}: {body[:300].decode('utf-8', 'replace')}")
    job = json.loads(body)
    job_id = job.get("id") or job.get("job_id")
    if not job_id:
        raise RuntimeError(f"submit response missing job id: {body[:200]!r}")
    return str(job_id)


def poll(backend: str, job_id: str, api_key: str, *, interval: float, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        job = _get_json(f"{backend}/v1/jobs/{job_id}", api_key)
        if str(job.get("status")) in TERMINAL_STATUSES:
            return job
        time.sleep(interval)
    raise TimeoutError(f"job {job_id} did not finish within {timeout:.0f}s (last stage={job.get('stage')})")


def quality_from_job_dir(job_dir: Path, job_id: str) -> dict[str, Any] | None:
    """Read ``quality_result`` from the job's JSON report sibling on disk.

    ``?quality=true`` embeds the quality-layer audit here, but the HTTP report
    endpoint only serves the HTML ACR — so this is the only way to reuse the
    backend's own audit instead of paying for a second judge pass.
    """
    report_dir = job_dir / job_id / "report"
    if not report_dir.is_dir():
        return None
    for candidate in sorted(report_dir.glob("*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        block = payload.get("quality_result")
        if isinstance(block, dict) and block.get("dimensions"):
            return block
    return None


def quality_from_report_endpoint(backend: str, job_id: str, api_key: str) -> dict[str, Any] | None:
    """Best-effort: the report endpoint serves HTML today, so tolerate non-JSON."""
    code, body = _request(
        f"{backend}/v1/jobs/{job_id}/report", headers=_headers(api_key), timeout=120.0
    )
    if code >= 400:
        return None
    try:
        report = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    for key in ("quality_result", "quality", "quality_audit", "quality_layer"):
        block = report.get(key)
        if isinstance(block, dict) and block.get("dimensions"):
            return block
    return None


def quality_via_audit(backend: str, job_id: str, api_key: str, workdir: Path) -> dict[str, Any]:
    """Fallback: download the remediated PDF and audit it directly."""
    code, body = _request(f"{backend}/v1/jobs/{job_id}/result", headers=_headers(api_key), timeout=300.0)
    if code >= 400:
        raise RuntimeError(f"result -> {code}: {body[:200].decode('utf-8', 'replace')}")
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / f"{job_id}.pdf"
    out.write_bytes(body)
    code, audit = _post_file(f"{backend}/v1/quality/audit/pdf", out, api_key)
    if code >= 400:
        raise RuntimeError(f"audit -> {code}: {audit[:200].decode('utf-8', 'replace')}")
    return json.loads(audit)


def remediated_pdf(
    backend: str,
    job_id: str,
    api_key: str,
    workdir: Path,
    job_dir: Path | None,
) -> Path:
    """Locate the remediated PDF, preferring the on-disk copy over a download."""
    if job_dir:
        local = job_dir / job_id / "remediated.pdf"
        if local.is_file():
            return local
    code, body = _request(
        f"{backend}/v1/jobs/{job_id}/result", headers=_headers(api_key), timeout=300.0
    )
    if code >= 400:
        raise RuntimeError(f"result -> {code}: {body[:200].decode('utf-8', 'replace')}")
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / f"{job_id}.pdf"
    out.write_bytes(body)
    return out


def wcag_verify(backend: str, pdf: Path, api_key: str, *, timeout: float) -> dict[str, Any]:
    """POST the remediated PDF to the 2-tier WCAG verifier.

    This is the only entrypoint that reaches the ``table_structure`` and
    ``contrast`` ``task=`` call sites (``pdf_wcag_verifier.py:756/771``), so it is
    the only way the gate exercises those two adapters. ``/v1/remediate`` alone
    never touches them.
    """
    code, body = _post_file(f"{backend}/v1/validate/pdf/wcag", pdf, api_key, timeout=timeout)
    if code >= 400:
        raise RuntimeError(f"wcag -> {code}: {body[:300].decode('utf-8', 'replace')}")
    return json.loads(body)


def contrast_audit(backend: str, pdf: Path, api_key: str, *, timeout: float) -> dict[str, Any]:
    code, body = _post_file(f"{backend}/v1/pdf/contrast/audit", pdf, api_key, timeout=timeout)
    if code >= 400:
        raise RuntimeError(f"contrast -> {code}: {body[:300].decode('utf-8', 'replace')}")
    return json.loads(body)


def _dimension_score(value: Any) -> float | None:
    if isinstance(value, dict):
        raw = value.get("score")
    else:
        raw = value
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def find_regressions(
    observed: dict[str, Any],
    annotation: dict[str, Any] | None,
    tolerance: float,
) -> list[str]:
    """Dimensions scoring materially below the annotated gold score."""
    if not annotation:
        return []
    gold = annotation.get("dimensions") or {}
    applicable = set(annotation.get("applicable_dimensions") or gold.keys())
    regressions: list[str] = []
    for dim in sorted(applicable):
        want = _dimension_score(gold.get(dim))
        got = _dimension_score(observed.get(dim))
        if want is None or got is None:
            continue
        if got < want - tolerance:
            regressions.append(f"{dim}: {got:.2f} < {want:.2f}-{tolerance:.2f}")
    return regressions


def run_one(
    cand: Candidate,
    *,
    backend: str,
    api_key: str,
    interval: float,
    timeout: float,
    tolerance: float,
    allow_semantic_rebuild: bool,
    workdir: Path,
    job_dir: Path | None,
    do_wcag: bool = False,
    do_contrast: bool = False,
    verify_timeout: float = 1800.0,
) -> DocRecord:
    rec = DocRecord(doc_id=cand.doc_id, source=str(cand.path), status="error")
    started = time.perf_counter()
    try:
        rec.job_id = submit(backend, cand.path, api_key, allow_semantic_rebuild=allow_semantic_rebuild)
        job = poll(backend, rec.job_id, api_key, interval=interval, timeout=timeout)
        if str(job.get("status")) != "done":
            rec.status = "failed"
            rec.error = str(job.get("error") or "job failed")
            return rec

        quality = quality_from_job_dir(job_dir, rec.job_id) if job_dir else None
        rec.scored_via = "job_dir.quality_result" if quality else ""
        if not quality:
            quality = quality_from_report_endpoint(backend, rec.job_id, api_key)
            rec.scored_via = "report.quality_result" if quality else ""
        if not quality:
            quality = quality_via_audit(backend, rec.job_id, api_key, workdir)
            rec.scored_via = "quality/audit/pdf"

        rec.dimensions = quality.get("dimensions") or {}
        rec.overall_pass = bool(quality.get("overall_pass"))
        rec.failing_dimensions = list(quality.get("failing_dimensions") or [])
        rec.not_applicable_dimensions = list(quality.get("not_applicable_dimensions") or [])
        rec.regressions = find_regressions(rec.dimensions, cand.annotation, tolerance)
        if rec.regressions:
            rec.overall_pass = False
        rec.status = "scored"

        if do_wcag or do_contrast:
            pdf = remediated_pdf(backend, rec.job_id, api_key, workdir, job_dir)
            if do_wcag:
                started_wcag = time.perf_counter()
                try:
                    result = wcag_verify(backend, pdf, api_key, timeout=verify_timeout)
                    rec.wcag_overall_pass = bool(result.get("overall_pass"))
                    rec.wcag_failing_criteria = list(result.get("failing_criteria") or [])
                    rec.wcag_pages_verified = int(result.get("pages_verified") or 0)
                    rec.wcag_total_findings = int(result.get("total_findings") or 0)
                except Exception as exc:  # noqa: BLE001 - verification is additive
                    rec.wcag_error = f"{type(exc).__name__}: {exc}"
                finally:
                    rec.wcag_seconds = round(time.perf_counter() - started_wcag, 2)
            if do_contrast:
                started_contrast = time.perf_counter()
                try:
                    contrast_audit(backend, pdf, api_key, timeout=verify_timeout)
                except Exception as exc:  # noqa: BLE001
                    rec.contrast_audit_error = f"{type(exc).__name__}: {exc}"
                finally:
                    rec.contrast_audit_seconds = round(time.perf_counter() - started_contrast, 2)
    except Exception as exc:  # noqa: BLE001 - one bad doc must not kill the run
        rec.status = "error"
        rec.error = f"{type(exc).__name__}: {exc}"
    finally:
        rec.elapsed_seconds = round(time.perf_counter() - started, 2)
        rec.completed_at = _now()
    return rec


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def _done_doc_ids(records_path: Path) -> set[str]:
    if not records_path.exists():
        return set()
    done: set[str] = set()
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "scored":
            done.add(str(rec.get("doc_id")))
    return done


def latest_per_doc(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the last record for each doc_id.

    ``records.jsonl`` is append-only and ``--resume`` re-runs any doc that is not
    already ``scored``. Without this, a transient failure (backend restart, dropped
    tunnel) leaves an ``error`` row that permanently fails the ``no_errors`` gate
    even after the retry succeeds.
    """
    by_doc: dict[str, dict[str, Any]] = {}
    for record in records:
        by_doc[str(record.get("doc_id"))] = record
    return list(by_doc.values())


def summarise(
    records: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    *,
    min_overall: float,
    min_dimension: float,
) -> dict[str, Any]:
    records = latest_per_doc(records)
    scored = [r for r in records if r.get("status") == "scored"]
    errored = [r for r in records if r.get("status") != "scored"]
    total = len(scored)
    passed = sum(1 for r in scored if r.get("overall_pass"))
    overall_rate = (passed / total) if total else 0.0

    per_dim: dict[str, Any] = {}
    for dim in PDF_DIMENSIONS:
        applicable = [
            r for r in scored if dim not in (r.get("not_applicable_dimensions") or [])
        ]
        if not applicable:
            per_dim[dim] = {"applicable": 0, "passed": 0, "pass_rate": None, "gate_passed": True}
            continue
        ok = sum(1 for r in applicable if dim not in (r.get("failing_dimensions") or []))
        rate = ok / len(applicable)
        per_dim[dim] = {
            "applicable": len(applicable),
            "passed": ok,
            "pass_rate": round(rate, 4),
            "gate_passed": rate >= min_dimension,
        }

    # WCAG-verify stats are recorded, not gated: they measure a different
    # endpoint than overall_pass and must not silently change gate semantics.
    wcag_ran = [r for r in scored if r.get("wcag_overall_pass") is not None or r.get("wcag_error")]
    wcag_ok = [r for r in wcag_ran if r.get("wcag_overall_pass")]
    wcag_errors = [r for r in wcag_ran if r.get("wcag_error")]
    wcag_criteria: Counter[str] = Counter()
    for record in wcag_ran:
        wcag_criteria.update(record.get("wcag_failing_criteria") or [])
    wcag_block = {
        "documents": len(wcag_ran),
        "passed": len(wcag_ok),
        "errored": len(wcag_errors),
        "pass_rate": round(len(wcag_ok) / len(wcag_ran), 4) if wcag_ran else None,
        "failing_criteria_counts": dict(wcag_criteria.most_common()),
        "mean_seconds": (
            round(sum(float(r.get("wcag_seconds") or 0) for r in wcag_ran) / len(wcag_ran), 2)
            if wcag_ran
            else None
        ),
    }

    regressed = [r["doc_id"] for r in scored if r.get("regressions")]
    gates = {
        "overall_pass_rate": {
            "observed": round(overall_rate, 4),
            "expected_min": min_overall,
            "passed": overall_rate >= min_overall and total > 0,
        },
        "no_errors": {"observed": len(errored), "expected_max": 0, "passed": not errored},
        "no_regressions": {"observed": len(regressed), "expected_max": 0, "passed": not regressed},
    }
    dim_gate_ok = all(v["gate_passed"] for v in per_dim.values())
    gate_passed = all(g["passed"] for g in gates.values()) and dim_gate_ok

    return {
        "generated_at": _now(),
        "counts": {
            "scored": total,
            "passed": passed,
            "errored": len(errored),
            "skipped": len(skipped),
        },
        "skipped_reasons": {
            reason: sum(1 for s in skipped if s["reason"] == reason)
            for reason in sorted({s["reason"] for s in skipped})
        },
        "per_dimension": per_dim,
        "wcag_verify": wcag_block,
        "gates": gates,
        "regressed_docs": regressed,
        "errored_docs": [{"doc_id": r["doc_id"], "error": r.get("error")} for r in errored],
        "gate_passed": gate_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path, help="Directory of heldout PDFs (recursive)")
    source.add_argument(
        "--corpus-list",
        type=Path,
        help="Newline-delimited PDF path list, e.g. gate_corpus_v1.txt from build_gate_corpus.py",
    )
    source.add_argument("--manifest", type=Path, help="corpus_annotations manifest.jsonl")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        help="Repo root the manifest's relative paths resolve against (manifest mode)",
    )
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="", help="x-api-key header (APP_API_KEY)")
    parser.add_argument(
        "--job-dir",
        type=Path,
        help="Backend JOB_DIR (e.g. ~/.local/share/remedy-server/job_data). Reuses the "
        "backend's own quality_result instead of re-running the judges.",
    )
    parser.add_argument("--trained-data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--allow-contaminated",
        action="store_true",
        help="Do not skip PDFs whose doc_id appears in a train/val split",
    )
    parser.add_argument("--allow-semantic-rebuild", action="store_true")
    parser.add_argument(
        "--wcag-verify",
        action="store_true",
        help="POST each remediated PDF to /v1/validate/pdf/wcag. This is the ONLY way the "
        "table_structure and contrast adapters are exercised; /v1/remediate never reaches them.",
    )
    parser.add_argument(
        "--contrast-audit",
        action="store_true",
        help="Additionally POST each remediated PDF to /v1/pdf/contrast/audit",
    )
    parser.add_argument("--verify-timeout", type=float, default=1800.0)
    parser.add_argument("--limit", type=int, default=0, help="Cap documents (0 = all)")
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument("--score-tolerance", type=float, default=0.05)
    parser.add_argument("--min-overall-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-dimension-pass-rate", type=float, default=0.75)
    parser.add_argument("--resume", action="store_true", help="Skip doc_ids already scored in records.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="List the corpus and exit without calling the backend")
    parser.add_argument("--out", type=Path, default=DEFAULT_EVAL_ROOT / "e2e_heldout_gate")
    args = parser.parse_args()

    backend = str(args.backend_url).rstrip("/")

    if args.manifest:
        corpus_root = args.corpus_root or args.manifest.resolve().parents[3]
        items = candidates_from_manifest(args.manifest, corpus_root)
    elif args.corpus_list:
        items = candidates_from_list(args.corpus_list)
    else:
        items = candidates_from_corpus(args.corpus)

    seen = trained_doc_ids(args.trained_data_root) if not args.allow_contaminated else set()
    keep, skipped = partition(items, seen=seen, allow_contaminated=args.allow_contaminated)

    args.out.mkdir(parents=True, exist_ok=True)
    records_path = args.out / "records.jsonl"
    summary_path = args.out / "summary.json"

    if args.resume:
        already = _done_doc_ids(records_path)
        keep = [c for c in keep if c.doc_id not in already]
    if args.limit:
        keep = keep[: args.limit]

    print(f"candidates={len(items)} eligible={len(keep)} skipped={len(skipped)}")
    for reason in sorted({s["reason"] for s in skipped}):
        print(f"  skipped[{reason}]={sum(1 for s in skipped if s['reason'] == reason)}")

    if args.dry_run:
        (args.out / "dry_run.json").write_text(
            json.dumps(
                {"eligible": [{"doc_id": c.doc_id, "path": str(c.path)} for c in keep], "skipped": skipped},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"dry run -> {args.out / 'dry_run.json'}")
        return 0

    if not keep:
        print("ERROR: no eligible heldout PDFs; refusing to report a gate result.")
        return 1

    workdir = args.out / "outputs"
    with records_path.open("a", encoding="utf-8") as handle:
        for index, cand in enumerate(keep, start=1):
            rec = run_one(
                cand,
                backend=backend,
                api_key=args.api_key,
                interval=args.poll_interval,
                timeout=args.job_timeout,
                tolerance=args.score_tolerance,
                allow_semantic_rebuild=args.allow_semantic_rebuild,
                workdir=workdir,
                job_dir=args.job_dir,
                do_wcag=args.wcag_verify,
                do_contrast=args.contrast_audit,
                verify_timeout=args.verify_timeout,
            )
            handle.write(json.dumps(asdict(rec)) + "\n")
            handle.flush()
            flag = "PASS" if rec.overall_pass else rec.status.upper()
            extra = ""
            if args.wcag_verify:
                wcag = rec.wcag_error or (
                    f"wcag_pass={rec.wcag_overall_pass} findings={rec.wcag_total_findings} "
                    f"({rec.wcag_seconds}s)"
                )
                extra = f" | {wcag}"
            print(
                f"[{index}/{len(keep)}] {cand.doc_id}: {flag} ({rec.elapsed_seconds}s)"
                f"{extra} {rec.error}".rstrip()
            )

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarise(
        records,
        skipped,
        min_overall=args.min_overall_pass_rate,
        min_dimension=args.min_dimension_pass_rate,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["gates"], indent=2))
    print(f"gate_passed={summary['gate_passed']} -> {summary_path}")
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
