# E2E Heldout PDF Remediation Gate — Runbook

The final promotion gate from `plans/edge-minicpm-v46-lora-router.txt`:

> End-to-end heldout PDF remediation must remain the final gate.

Harness: `tools/minicpm_edge/run_e2e_heldout_gate.py` (stdlib only, no new deps).

---

## 0. Blockers you must clear first

| # | Blocker | Status |
|---|---------|--------|
| 1 | `tools/corpus_annotations/v1` is **not a usable corpus** — all 90 PDF artifacts are 40–47 byte placeholder text files (`Synthetic pdf source artifact for pdf_001`). Annotations are `automated_bootstrap` with a uniform `0.9` on every dimension. | **Must build a real corpus** |
| 2 | The backend on `:8000` is **not this repo** and **not pointed at MiniCPM** — see §1. | Must repoint |
| 3 | ~~The MiniCPM router cannot run on this Mac as-is~~ — **RESOLVED**, see §7. `--device mps` + a `transformers>=5.7` venv. | Resolved |
| 4 | ~~`remedy-server` ignores `OLLAMA_VISION_TASK_MODELS`~~ — true only of the **deployed** tree (61 commits stale). Source `main` has routing since `a512960`. See §8. | Deploy `origin/main` |

The harness **fails closed** on all three: placeholder PDFs are rejected by a
`%PDF` magic check, and it refuses to emit a gate verdict when zero eligible
documents remain.

---

## 1. Backend: what is actually running

`launchctl` job `ai.projectremedy.remedy-server.local` runs
`~/.local/share/remedy-server` (repo `projectremedyai/remedy-server`) — **not**
`remedy-pdf-desktop`. Its plist declares **no `EnvironmentVariables`**, so config
comes from `~/.local/share/remedy-server/.env`:

```
OLLAMA_BASE_URL=https://ollama.com/v1      # Ollama Cloud
OLLAMA_VISION_MODEL=kimi-k2.7-code:cloud   # NOT minicpm-v46-remedy
```

So today the gate would score **Ollama Cloud / Kimi**, not the MiniCPM router.
To point it at the router, add to that `.env` and restart the launchd job:

```
OLLAMA_BASE_URL=http://127.0.0.1:8000/v1
OLLAMA_VISION_MODEL=minicpm-v46-remedy
OLLAMA_VISION_TASK_MODELS=contrast:minicpm-v46-remedy-contrast-v1,reading_order:minicpm-v46-remedy-reading-order-v1,heading_hierarchy:minicpm-v46-remedy-heading-v1,table_structure:minicpm-v46-remedy-table-v1
```

> Port collision: the router **also** defaults to `:8000`. Serve it on another
> port (e.g. `--port 8010`) or the backend will talk to itself.

The desktop repo's own backend (`backend/app/`) exposes `/api/upload`, not
`/v1/remediate`. Its task routing lives in the **uncommitted** `+223` line diff
to `backend/app/ollama_runtime.py`: `build_local_vision_provider()` reads
`OLLAMA_VISION_TASK_MODELS`, verifies each alias is installed in Ollama, and
returns a `TaskRoutedOllamaVisionProvider` that picks a per-task model by
classifying the prompt with `_infer_vision_task()` (string-marker matching).
If any alias is missing it returns `None` (falls back to no vision) unless
`OLLAMA_VISION_ROUTER_ALLOW_FALLBACK=1`.

---

## 2. Build a real heldout corpus

Real PDFs exist, but the training splits drew from them. **194** real `doc_id`s
appear in `data/tasks/*/{train,val}.jsonl`.

Naive stem matching is **unsafe**: the LAMC pools are content-hash prefixed
(`0e29771b214b_Foo.pdf`) while training `doc_id`s are not (`Foo`). Matching raw
stems reports *zero* contamination and silently leaks 257 training documents.
The harness strips a leading `^[0-9a-f]{12}_` before comparing.

Clean pools (contamination measured **after** de-hashing):

| Pool | Total | Contaminated | Heldout |
|------|------:|-------------:|--------:|
| `lamc_district_forms/data/visual_match/downloads/all_campuses` | 1700 | 75 | **1625** |
| `lamc_district_forms/data/visual_match/downloads/lamc` | 464 | 182 | **282** |
| `lamc_district_forms/data/visual_match/downloads/district` | 187 | 0 | **187** |
| `tools/minicpm_edge/data/tasks/heading_hierarchy/real_sources` | 17 | 12 | 5 |

Avoid `lamc_district_forms/lamc_remediated/**` and
`_cleanup_archive_*/remediation_backups/**` — those hold the 194 training sources.

Always confirm the guard is doing work before a real run:

```bash
cd tools/minicpm_edge
./run_e2e_heldout_gate.py \
  --corpus ~/code/lamc_district_forms/data/visual_match/downloads/district \
  --dry-run --out eval_runs/e2e_gate_v1
# candidates=187 eligible=187 skipped=0
```

> Caveat: the guard is name-based. Same document under a different filename
> would still leak. For a publishable gate, dedupe by content hash or title.

---

## 3. Run the gate

```bash
cd tools/minicpm_edge

./run_e2e_heldout_gate.py \
  --corpus ~/code/lamc_district_forms/data/visual_match/downloads/district \
  --backend-url http://127.0.0.1:8000 \
  --job-dir ~/.local/share/remedy-server/job_data \
  --out eval_runs/e2e_gate_v1 \
  --min-overall-pass-rate 0.80 \
  --min-dimension-pass-rate 0.75
```

Exit codes: `0` gate passed · `2` gate failed · `1` harness error.

Useful flags:

- `--dry-run` — list the corpus + skip reasons, never touch the backend.
- `--limit N` — smoke a slice.
- `--resume` — skip `doc_id`s already `scored` in `records.jsonl`.
- `--manifest <manifest.jsonl> [--corpus-root DIR]` — annotation mode: remediate
  each `known_bad` artifact and flag any dimension scoring more than
  `--score-tolerance` (default `0.05`) below its annotated gold score.
- `--allow-contaminated` — disable the training-overlap guard (**don't**).
- `--api-key` — sets `x-api-key` (needed only if `APP_API_KEY` is set; it is empty today).

### How scoring resolves

`?quality=true` makes the backend attach a `quality_result` block. The harness
tries three sources, in order:

1. `--job-dir <JOB_DIR>/<job_id>/report/*.json` → `quality_result`. **Free.**
2. `GET /v1/jobs/{id}/report` if it returns JSON.
3. `POST /v1/quality/audit/pdf` on `GET /v1/jobs/{id}/result`. Always works,
   but **re-runs the judges** (extra latency + LLM spend).

> `/v1/jobs/{id}/report` serves the **HTML** ACR (`FileResponse(..., media_type="text/html")`),
> so path 2 is normally skipped. Pass `--job-dir` for local backends — otherwise
> every document pays for a second judge pass.

### Gates

- `overall_pass_rate ≥ --min-overall-pass-rate`
- every applicable dimension's `pass_rate ≥ --min-dimension-pass-rate`
- zero errored documents
- zero regressions vs annotations (annotation mode only)

Dimensions scored (PDF): `alt_text`, `reading_order`, `heading_semantics`,
`table_structure`, `link_text`, `decorative`, `complex_content`.

Outputs: `<out>/records.jsonl` (one row per doc) and `<out>/summary.json`.

---

## 4. Router: MiniCPM on this Mac

> **Superseded by §7** — all three blockers below are now cleared and the router
> has been run on MPS. Kept as the original diagnosis.

`serve_router_minicpm_peft.py` *was* **CUDA-only**. Three blockers:

1. **Code** — `serve_router_minicpm_peft.py:113-118`:
   ```python
   base = AutoModelForImageTextToText.from_pretrained(
       base_model,
       dtype=torch.bfloat16,   # :115
       device_map="cuda",      # :116  <-- hard blocker
       attn_implementation=attn_implementation,  # :117 default "sdpa" — MPS-safe
   )
   ```
   Minimal fix: `device_map={"": "mps"}` (or load with `device_map=None` then
   `base.to("mps")`), and prefer `dtype=torch.float16` on MPS. Nothing else
   needs to change — inputs already follow `self.model.device` (`:173`).
   There is **no** `--device` flag and no autodetect. *(Not modified — reported only.)*
2. **Deps** — `peft` is **not installed** for system `python3` (required at `:103`).
   `torch 2.10.0` (MPS available ✅) and `accelerate 1.13.0` are present.
   `transformers 4.57.6` is installed but `requirements-l4.txt` pins `>=5.7.0`.
   Do **not** `pip install -r requirements-l4.txt` on Apple Silicon — it pulls
   `bitsandbytes` (CUDA-only). Install `peft` alone.
3. **Weights** — `openbmb/MiniCPM-V-4.6` is a metadata-only stub in the HF cache
   (32 KB; `config.json` + `README.md`, no `*.safetensors`). First run triggers a
   multi-GB download.

Adapters are already on disk under `tools/minicpm_edge/outputs/` (all v1 aliases
plus `heading-v2` / `reading-order-v2`), each with `adapter_config.json` +
`adapter_model.safetensors`.

Router API: OpenAI-compatible — `POST /v1/chat/completions`, `GET /v1/models`,
`GET /health`, plus an Ollama `GET /api/tags` shim. Images must be base64
`image_url` data-URIs. Default `--port 8000`.

**Because of blockers 1–3, no local router smoke was run.**

---

## 5. Re-establish the Heidi L4 tunnel

The tunnel is dead: `eval_runs/heidi_router_tunnel.pid` holds PID `91850` (not
running) and `heidi_router_tunnel.log` is empty.

Target cluster: `johnny-cluster` / `la-mission-college-cluster`, id
`6a3c4156b27a919f67b48c42`, provider `ibm`, public IP `169.63.101.217`.
There is **no** `heidi node` subcommand; the real surface is
`cluster | credential | job | login | logout | org | stack | storage | training | user | version | whoami`.
`heidi cluster ssh` fetches the key and head-node address for you.

```bash
# 1. Start the router ON the L4 box (pick a port; 8000 collides with the backend)
#    python serve_router_minicpm_peft.py --port 8010

# 2. Forward it locally (preferred — heidi supplies the key)
heidi cluster ssh 6a3c4156b27a919f67b48c42 --user johnnyadmin -- -N -L 8010:localhost:8010

# 2b. Direct ssh, if you still have the pem
ssh -i johnnyadmin-SSH-Key.pem -N -L 8010:localhost:8010 johnnyadmin@169.63.101.217

# 3. Point the backend at it (see §1), then restart the launchd job
```

Caveats:
- The exact local↔remote port pair is **not recorded** (log empty); `8000:localhost:8000`
  is the inference from `README.txt`, and it collides with the backend. Choose deliberately.
- Username `johnnyadmin` is inferred from shell history; heidi's default is `ubuntu`.
- `johnnyadmin-SSH-Key.pem` is not on disk — use the `heidi cluster ssh` form.
- **All clusters currently report `NODES 0`.** A node must be started and the
  router launched on it before the tunnel resolves.

---

## 6. Smoke result (2026-07-09)

One genuinely-heldout PDF (`Petition-for-Repeated-Coursework`, from
`visual_match/downloads/lamc`, confirmed absent from all train/val splits) was
pushed through the **live backend as currently configured** — i.e. against
**Ollama Cloud / `kimi-k2.7-code:cloud`**, *not* the MiniCPM router.

```
candidates=1 eligible=1 skipped=0
[1/1] Petition-for-Repeated-Coursework: SCORED (55.13s)
gate_passed=False
```

| Dimension | Score |
|---|---|
| alt_text | 1.0 |
| heading_semantics | 1.0 |
| link_text | 1.0 |
| decorative | 1.0 |
| complex_content | 1.0 |
| **reading_order** | **0.5** ❌ |
| **table_structure** | **0.5** ❌ |

`scored_via: job_dir.quality_result` (fast path — no second judge pass).
Artifacts: `eval_runs/e2e_gate_prep/smoke/{records.jsonl,summary.json}`.

This validates the harness plumbing (submit → poll → score → gate) end to end.
It is **not** a MiniCPM router result and carries no signal about the adapters.
The two failures are a baseline for the cloud model on this document.

---

## 7. MiniCPM smoke on local MPS (2026-07-09)

**Result: 2/2 documents remediated end-to-end through the real MiniCPM router on
Apple Silicon MPS. Gate `False` (reading_order / table_structure).**

### 7.1 What changed

`serve_router_minicpm_peft.py` gained a `--device {cuda,mps,cpu}` flag
(**default `cuda`** — GPU workbenches are unaffected). On `mps`/`cpu` it loads
with `device_map=None` + `.to(device)`, `float16` on mps / `float32` on cpu,
keeping `sdpa` attention. The CUDA branch is byte-for-byte the original path.

### 7.2 The real blocker was `transformers`, not the device

`transformers 4.57.6` (system) rejects the checkpoint:

```
ValueError: The checkpoint you are trying to load has model type `minicpmv4_6`
but Transformers does not recognize this architecture.
```

`requirements-l4.txt` already pins `transformers[torch]>=5.7.0`. Rather than
upgrade system-wide, serve from an isolated venv that reuses system torch:

```bash
cd tools/minicpm_edge
python3 -m venv --system-site-packages .venv-mps
./.venv-mps/bin/pip install --upgrade "transformers>=5.7.0" peft accelerate pillow
# do NOT install requirements-l4.txt — it pulls CUDA-only bitsandbytes
```

Verified isolation: system `python3` stays on `transformers 4.57.6`; the live
server's venv is untouched.

### 7.3 Serve the router (MPS, port 8010)

```bash
cd tools/minicpm_edge
./.venv-mps/bin/python serve_router_minicpm_peft.py \
  --device mps --port 8010 \
  --downsample-mode 16x --max-slice-nums 1 \
  --adapter-image-settings "table=4x:36,reading_order=4x:36"
```

Base weights downloaded once (~2.4 GB into the HF cache). All 6 aliases load;
`GET /v1/models` and `GET /health` return them. Resident set: **5.2 GB** — no OOM,
no unsupported-op fallback. A 32-token completion on a 346×448 render took
**10.4 s** and returned coherent text.

### 7.4 Second backend instance (port 8020, fully isolated)

The live launchd service on `:8000` and its `.env` were **not** touched. Env and
isolated `job_data`/`jobs.db` live in `eval_runs/e2e_gate_prep/instance8020/`:

```bash
cd ~/.local/share/remedy-server
set -a; source <repo>/tools/minicpm_edge/eval_runs/e2e_gate_prep/instance8020/env.sh; set +a
./.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8020
```

`load_dotenv(..., override=False)` means exported vars beat the live `.env`.

### 7.5 ⚠ remedy-server ignores `OLLAMA_VISION_TASK_MODELS`

**All 60 router calls across both documents used `model=minicpm-v46-remedy`
(the base/alt alias). The four task adapters were never invoked.**

`project_remedy/config.py:233` reads a single `OLLAMA_VISION_MODEL`, plus
optional `OLLAMA_VISION_FALLBACK_MODELS` (`pdf_vision.py:678`). There is **no**
per-task routing anywhere in `remedy-server` / `project_remedy`. Task routing
exists *only* in this repo's `backend/app/ollama_runtime.py`
(`TaskRoutedOllamaVisionProvider`, the uncommitted +223-line diff).

Consequence: **a gate run against `remedy-server` cannot exercise the adapter
router.** To gate the adapters you must either (a) run the gate against the
*desktop* backend, or (b) port `TaskRoutedOllamaVisionProvider` into
`project_remedy`. The `§1` env recipe is necessary but not sufficient.

### 7.6 Smoke results

```bash
./run_e2e_heldout_gate.py \
  --corpus ~/code/lamc_district_forms/data/visual_match/downloads/district \
  --backend-url http://127.0.0.1:8020 \
  --job-dir <repo>/tools/minicpm_edge/eval_runs/e2e_gate_prep/instance8020/job_data \
  --limit 2 --poll-interval 10 --job-timeout 3600 \
  --out eval_runs/e2e_gate_prep/smoke_minicpm
```

```
candidates=187 eligible=2 skipped=1   # skipped: 2c3621afe371_CB-College.pdf (not a PDF)
[1/2] Load Banking Form (AI):                             SCORED (130.15s)
[2/2] Request for Transfer Administrator HR-C307 ... (AI): SCORED (240.25s)
gate_passed=False
```

| Dimension | Load Banking Form | Request for Transfer Admin |
|---|---:|---:|
| alt_text | 1.00 | 1.00 |
| heading_semantics | 1.00 | 1.00 |
| link_text | 1.00 | 1.00 |
| decorative | 1.00 | 1.00 |
| complex_content | 1.00 | 1.00 |
| **reading_order** | **0.40** ❌ | **0.30** ❌ |
| **table_structure** | 1.00 | **0.00** ❌ |

Both scored via the `job_dir.quality_result` fast path (no second judge pass).
Zero backend errors; zero router 5xx.

### 7.7 Timings and whether a full 187-doc run is practical

- Router: 60 calls, **mean 5.5 s**, median 2.7 s, max 41.1 s (total 328 s of GPU time).
- Wall clock per doc: **130 s** and **240 s** (mean ≈ 185 s); ~30 router calls/doc.
- Projection: **187 docs × 185 s ≈ 9.6 h serial** on this Mac.

**Verdict: technically feasible overnight, but not worth doing here.** Two reasons:

1. It would gate the **base/alt adapter only** (§7.5) — the run produces no signal
   about contrast / reading-order / heading / table adapters, which is the whole
   point of the gate.
2. `--max-slice-nums 1` at `16x` is the low-resolution profile. The README's
   per-adapter high-res settings (`table=4x:36`, `reading_order=4x:36`) only bind
   when those adapters are actually selected — which never happens here — so the
   two failing dimensions ran at the *coarse* image profile. `reading_order`
   scoring 0.30–0.40 is therefore expected and **not** evidence the adapter is bad.

**Recommendation: run the full gate on the L4** (see §5), where 187 docs at
GPU speed is ~30–60 min, *after* per-task routing reaches whichever backend the
gate targets. Use this Mac only for plumbing smoke tests.

### 7.8 Teardown

```bash
pkill -f serve_router_minicpm_peft          # router :8010
pkill -f "uvicorn backend.app.main:app --host 127.0.0.1 --port 8020"
# the launchd service on :8000 is a separate process — leave it alone
```

---

## 8. Per-task routing in remedy-server — already upstream (2026-07-09)

**No port was needed. `remedy-server` `main` already implements per-task vision
routing. The deployed copy is simply 61 commits stale.**

### 8.1 What §7.5 actually found

§7.5 concluded "remedy-server ignores `OLLAMA_VISION_TASK_MODELS`". That was true
of the copy it inspected — `~/.local/share/remedy-server` (the **deployed** tree,
pinned at `b222d51`). It is **not** true of the source repo.

| Tree | Path | Commit | Has routing? |
|---|---|---|---|
| Deployed | `~/.local/share/remedy-server` | `b222d51` | ❌ |
| **Source** | `~/code/lamc_district_forms/remedy-server` | `3e6e3d3` (main) | ✅ |

Routing landed in **`a512960` "feat: add routed vision adapters and EPUB export"**
(`.env.example +11`, `pdf_vision.py +182`, `tests/unit/test_pdf_vision.py +225`).
`git rev-list --count b222d51..main` = **61**. `a512960` is absent from the
deployed history.

> Note: `~/code/lamc_district_forms/remedy-server` and `-multitask-next` are both
> clones of `projectremedyai/remedy-server`. The source repo's working tree is
> **clean** — there was nothing to write and nothing to commit, so no feature
> branch was created.

### 8.2 The upstream design (differs from the desktop repo's)

`TaskRoutedVisionProvider` (`src/project_remedy/pdf_vision.py:750`) routes on an
**explicit `task=` kwarg**, not on `_infer_vision_task()` prompt-marker sniffing.
This is strictly more robust — no prompt-template drift to track. All five call
sites already pass it:

| Task | Call site |
|---|---|
| `reading_order` | `pdf_wcag_verifier.py:709`, `pdf_vision.py:1550` |
| `heading_hierarchy` | `pdf_wcag_verifier.py:721`, `pdf_vision.py:1588`, `pdf_fixer.py:7962` |
| `table_structure` | `pdf_wcag_verifier.py:756` |
| `contrast` | `pdf_wcag_verifier.py:771`, `contrast/detector.py:206` |
| `alt_text_quality` | `pdf_vision.py:1692` |

`build_vision_provider` (`pdf_vision.py:961`) short-circuits with
`if not task_models: return provider`, so single-model behavior is **byte-identical**
when `OLLAMA_VISION_TASK_MODELS` is unset. Unmatched tasks fall through to the
primary provider. `OLLAMA_VISION_ROUTER_ALLOW_FALLBACK` (default off) controls
whether a failing task provider falls back instead of raising.

Tests already exist — 8/8 pass:

```bash
cd ~/code/lamc_district_forms/remedy-server
./.venv/bin/python -m pytest tests/unit/test_pdf_vision.py -q -k "task or rout"
# 8 passed, 7 deselected
```

### 8.3 E2E verification (router :8010 MPS, backend :8020 from SOURCE)

> Gotcha: the source repo's venv console scripts carry a stale shebang
> (`#!/Users/laccd/Desktop/...`, from before the repo moved). Use
> `./.venv/bin/python -m uvicorn ...`, not `./.venv/bin/uvicorn`.

Router alias selection — **the success criterion, met**:

| Alias | Calls (routed run) | Calls (§7 unrouted) |
|---|---:|---:|
| `minicpm-v46-remedy` (base/alt) | 35 | **60** |
| `minicpm-v46-remedy-heading-v1` | **19** | 0 |
| `minicpm-v46-remedy-reading-order-v1` | **6** | 0 |
| `minicpm-v46-remedy-table-v1` | 0 | 0 |
| `minicpm-v46-remedy-contrast-v1` | 0 | 0 |

`table` and `contrast` never fired because their call sites are **conditional**,
not because routing is broken:
- table: `pdf_wcag_verifier.py:748` guards on `if needs_tables:` — neither form
  PDF carries tagged tables.
- contrast: `contrast/detector.py:206` only invokes vision when the detector
  surfaces low-contrast candidates.

To exercise those two, pick corpus docs with tagged tables / low-contrast regions.

### 8.4 Smoke scores — routing changed the output, not the score

`eval_runs/e2e_gate_prep/smoke_minicpm_routed/`

| Dimension | Load Banking Form | Request for Transfer Admin |
|---|---:|---:|
| alt_text · heading_semantics · link_text · decorative · complex_content | 1.00 | 1.00 |
| reading_order | 0.40 | 0.30 |
| table_structure | 1.00 | 0.00 |

**Identical to the unrouted §7.6 scores, to 2 dp.** But the remediated PDFs
**differ by SHA-256**, so routing genuinely changed the remediation. The scores
didn't move because the two failing dimensions are graded by *deterministic
structural proxies*, not by the vision model:

- `reading_order` → `transcript_comprehension_proxy` (`reading_order_judge_v1`)
- `table_structure` → `cell_lookup_structure` (`table_structure_judge_v1`)

> **This is the important consequence.** The quality layer's `reading_order` and
> `table_structure` scores measure the *tag tree of the output PDF*, not the
> vision model's answers. Swapping adapters can therefore leave the gate score
> untouched. The E2E gate validates the **pipeline**, and is a weak instrument for
> ranking adapters — per-task adapter metrics (`eval_task_metrics.py`) remain the
> sharp tool. Treat the E2E gate as a regression guard, not a leaderboard.

Wall clock (MPS): 150 s and 180 s per doc (vs 130 s / 240 s unrouted) — noise at n=2.

### 8.5 Leftover step: deploy

Nothing to merge. The gap is purely deployment — the live service runs a tree 61
commits behind. To make the live `:8000` service route per task:

```bash
cd ~/.local/share/remedy-server
git fetch origin && git merge --ff-only origin/main   # picks up a512960
./.venv/bin/pip install -e .                          # re-sync deps if needed
# add to .env: OLLAMA_VISION_TASK_MODELS=... (see §1)
launchctl kickstart -k gui/$(id -u)/ai.projectremedy.remedy-server.local
```

For the L4: the box needs the same `origin/main` (or newer) checkout before the
187-doc gate will exercise the adapters. Verify with the alias-count table in
§8.3 — if the router log shows only the base alias, the checkout is stale.

---

## 9. Adapter reachability: `/v1/remediate` can only exercise 3 of 5 adapters

§8.3 showed `table-v1` and `contrast-v1` never firing. I first assumed the 2-doc
slice just lacked tagged tables. **That was wrong.** A third smoke on a PDF with
a real tagged table (`/StructTreeRoot` + `/Table`,
`0a04a3025ff2_SignatureAuthorization Form.pdf`, one of 20 such docs in the
district pool) still produced **zero** `table-v1` calls:

```
[1/1] SignatureAuthorization Form: SCORED (120.2s)
alias counts: base=4  heading-v1=3  reading-order-v1=1  table-v1=0  contrast-v1=0
scores: heading_semantics 0.82 ❌ · table_structure 0.00 ❌ · rest 1.00
```

### Root cause — the `task=` call sites live behind different HTTP entrypoints

| Task | `task=` call site | Reachable from |
|---|---|---|
| `alt_text_quality` | `pdf_vision.py:1692` | **`/v1/remediate`** |
| `reading_order` | `pdf_vision.py:1550` | **`/v1/remediate`** |
| `heading_hierarchy` | `pdf_vision.py:1588`, `pdf_fixer.py:7962` | **`/v1/remediate`** |
| `reading_order`, `heading_hierarchy`, **`table_structure`** | `pdf_wcag_verifier.py:709/721/756` | `/v1/validate/pdf/wcag` only |
| **`contrast`** | `pdf_wcag_verifier.py:771` | `/v1/validate/pdf/wcag` only |
| **`contrast`** | `contrast/detector.py:206` | `/v1/pdf/contrast/audit`, `/v1/pdf/contrast/fix` only |

`pdf_wcag_verifier` is imported by exactly one module — `backend/app/validate_routes.py:190`.
`ContrastDetector` is constructed only in `backend/app/pdf_fix_routes.py:228`.
Neither is on the `/v1/remediate` job path.

### Consequence for the gate

**The "end-to-end heldout PDF remediation" gate, as specced, structurally cannot
exercise the `table_structure` or `contrast` adapters.** It covers `alt`,
`reading_order`, and `heading` only. Combined with §8.4 (the `reading_order` and
`table_structure` *scores* are deterministic tag-tree proxies, not vision output),
the gate's coverage of the router is thinner than the plan assumes.

To gate all five adapters, the harness must additionally hit — per document —
`POST /v1/validate/pdf/wcag` (drives `table_structure` + `contrast` + a second
pass on `reading_order`/`heading`) and optionally `POST /v1/pdf/contrast/audit`.
That is a harness change (`run_e2e_heldout_gate.py`), not a `project_remedy` change.

Recommended follow-up, in order:
1. Extend the harness with a `--wcag-verify` pass that calls `/v1/validate/pdf/wcag`
   on each remediated output and records its per-task criteria.
2. Keep `eval_task_metrics.py` as the authoritative adapter leaderboard.
3. Treat the E2E gate as a pipeline regression guard (§8.4).

---

## 10. Gate corpus v1 — composition and selection method

> Numbered §10, not §9: §9 is the adapter-reachability finding. Read §9 first —
> it constrains what this corpus can and cannot achieve.

Artifacts:

| File | Contents |
|---|---|
| `eval_runs/e2e_gate_prep/gate_corpus_v1.txt` | 180 absolute PDF paths, one per line |
| `eval_runs/e2e_gate_prep/gate_corpus_v1.json` | Per-doc triggers, counts, settings, warnings |
| `eval_runs/e2e_gate_prep/gate_corpus_scan.json` | Full 1,884-doc scan cache (re-select without rescanning) |
| `tools/minicpm_edge/build_gate_corpus.py` | The builder (CPU-only, ~7 min cold, instant `--from-cache`) |

### 10.1 What actually triggers the table / contrast adapters

§8.3 said table/contrast "never fire on plain form PDFs because their call sites
are conditional." Tracing it properly:

`needs_tables` / `needs_contrast` (`pdf_wcag_verifier.py:690-691`) read
`triage.focus_queue`. That queue is populated (`:610-622`) from the **triage
vision model's** `applicable_checks.table_structure` / `.color_contrast` on the
**rendered page**. So the trigger is *what the page looks like* — not a tagged
`/Table` element, and not any deterministic PDF property.

Corollary: my earlier tagged-table probe (`/StructTreeRoot` + `/Table`) was the
wrong proxy. The builder uses visual signals instead.

### 10.2 Detectors (CPU-only, no rendering, PyMuPDF)

- **tables** — `page.find_tables()`, PyMuPDF's ruling-line/alignment finder.
  Approximates what the triage model sees. Tagged structure elements unused.
- **contrast** — each text span's colour vs the fill rect it sits inside (white
  when none), scored with the WCAG 2.x contrast-ratio formula. A span below
  `--contrast-threshold` (default **4.5**, AA for normal text) marks the doc.
  Background-awareness matters: white-on-pale-blue headers are invisible to a
  naive "compare against white" check.

First 3 pages per PDF (`--max-pages`). Corrupt PDFs are counted and skipped, not fatal.

### 10.3 Guards

- Contamination: reuses `run_e2e_heldout_gate.trained_doc_ids()` + `_normalise_stem()`,
  so the `^[0-9a-f]{12}_` hash prefix is stripped before comparing against every
  `data/tasks/*/{train,val}.jsonl` `meta.doc_id`. **Verified: 0 contaminated.**
- Deduplication: content **SHA-256**, so the same PDF under two names in two
  pools is emitted once. **Verified: 180 paths → 180 distinct hashes.**
- Determinism: no RNG. Ordering is `sha256(doc_id)`, so reruns are byte-identical.

### 10.4 Scanned pool (1,884 clean, deduped, uncontaminated docs)

| Pool | n | with tables | with contrast | both | tables only | contrast only | plain |
|---|--:|--:|--:|--:|--:|--:|--:|
| `district` | 182 | 89 | 24 | 16 | 73 | 8 | 85 |
| `lamc` | 279 | 129 | 102 | 52 | 77 | 50 | 100 |
| `all_campuses` | 1423 | 700 | 742 | 415 | 285 | 327 | 396 |
| **total** | **1884** | 918 | 868 | 483 | 435 | 385 | 581 |

### 10.5 Selected corpus (`--pad pool-order`, the default)

Pools are consumed in the order given (`district → lamc → all_campuses`), so the
gate stays comparable with the §7/§8 district runs and only spills when a quota
demands it. District carries just 24 contrast docs, so 12 docs spill into `lamc`
to satisfy the contrast quota.

| Metric | Value |
|---|---|
| Selected | **180** |
| By pool | district **168**, lamc **12** |
| By trigger | plain 74 · tables 70 · contrast 20 · both 16 |
| **Docs with tables** | **86** (quota ≥20 ✅) |
| **Docs with contrast** | **36** (quota ≥20 ✅) |
| Warnings | none |

`--pad representative` samples across all three pools at once and widens contrast
coverage substantially (86 tables / **86** contrast, 46 both) at the cost of
comparability with prior district runs. Use it if the contrast adapter needs a
stronger signal:

```bash
./build_gate_corpus.py --from-cache --pad representative --target 180 --out <path>
```

### 10.6 Running the gate against it

`run_e2e_heldout_gate.py` gained `--corpus-list` (mutually exclusive with
`--corpus` / `--manifest`; ignores blank lines and `#` comments; preserves order,
so `--limit N` keeps the quota-satisfying head):

```bash
cd tools/minicpm_edge
./run_e2e_heldout_gate.py \
  --corpus-list eval_runs/e2e_gate_prep/gate_corpus_v1.txt \
  --backend-url http://127.0.0.1:8020 \
  --job-dir <instance>/job_data \
  --out eval_runs/e2e_gate_v1
# dry-run verified: candidates=180 eligible=180 skipped=0
```

### 10.7 ⚠ This corpus is necessary but not sufficient

Per §9, `/v1/remediate` **never reaches** the `table_structure` or `contrast`
`task=` call sites — they live behind `/v1/validate/pdf/wcag` and
`/v1/pdf/contrast/*`. Feeding this corpus to the current harness will still show
**zero** `table-v1` / `contrast-v1` router calls.

The corpus makes those adapters *reachable* once the harness adds a
`--wcag-verify` pass (`POST /v1/validate/pdf/wcag` per remediated output). Until
then it buys: a representative 180-doc gate for `alt` / `reading_order` /
`heading`, plus a pre-verified pool of 86 table-triggering and 36
contrast-triggering documents ready for that pass.
