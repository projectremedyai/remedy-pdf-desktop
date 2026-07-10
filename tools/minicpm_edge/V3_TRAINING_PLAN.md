# V3 Adapter Training Plan (heading, reading_order)

Prepared 2026-07-09 from the v2 miss analysis in
`eval_runs/v2_miss_analysis_h100/`. Data is already built locally under
`data/v3/`; this file is the runbook for the next GPU workbench.

## Diagnosis recap

- **heading v2** (0.8476 exact correction vs 0.90 gate): detection and JSON are
  fine (status 0.9952). Misses are level-pair exactness — H3↔H4, H2↔H3
  confusions. At `16x/1` the page render is too downsampled for the model to
  see font-size hierarchy, and the v1 high-res *generation* sweep regressed
  because the adapter was trained at `16x/1` (train/serve resolution mismatch).
  The training forward-path fix for high-res is verified
  (`HIGHRES_TRAINING.md`, probe artifact
  `eval_runs/highres_training_probe_l4/summary.json`).
- **reading_order v2** (status 0.50, pass-all collapse): the v2 builder found
  zero v1 train misses, so v2 retrained on the identical 200 rows with a fresh
  rank-16 adapter. Every existing fail example is one corruption family
  (whole-list rotation), so there was nothing new to learn and the fresh
  adapter regressed to always-pass.

## v3 changes

1. **heading**: duplication weighted by miss type from the train-split v1
   scores (`eval_runs/train_hard_v1_h100/heading/heading.scores.jsonl`):
   pair-exactness misses ×3, status misses ×2 → 1398 rows at
   `data/v3/heading_hierarchy/train.jsonl`. Train and evaluate at **4x/36**.
2. **reading_order**: synthetic corruption family generated from clean pass
   rows — `adjacent_swap`, `section_reverse`, `interleave`, `window_shuffle`,
   `rotation` — plus pass-duplication to keep the split balanced → 464 rows at
   `data/v3/reading_order/train.jsonl`. A held-out diagnostic split
   `data/v3/reading_order/val_hard.jsonl` (51 synthetic fails built from val
   pass rows) checks generalization beyond rotation. Keep rank 8 (the v2
   rank-16 bump is not the fix).

Rebuild commands (deterministic, already run):

```bash
python3 build_v3_training_data.py reading-order \
  --out data/v3/reading_order/train.jsonl --add-task-tags
python3 build_v3_training_data.py reading-order \
  --split val --synthetic-only --variants-per-pass 3 \
  --out data/v3/reading_order/val_hard.jsonl --add-task-tags
python3 build_v3_training_data.py heading \
  --scores eval_runs/train_hard_v1_h100/heading/heading.scores.jsonl \
  --out data/v3/heading_hierarchy/train.jsonl --add-task-tags
```

## GPU workbench runbook

All commands run from `tools/minicpm_edge/` with the workspace venv active
(`README.txt` Initial L4 Setup).

### 0. Prove the high-res training forward path (mandatory before spend)

```bash
PYTHONPATH=. python3 probe_training_forward.py \
  --train data/v3/heading_hierarchy/train.jsonl \
  --downsample-mode 4x --max-slice-nums 36
```

Must pass. The `--omit-forward-downsample` variant must still fail.

### 1. heading v3 — train at 4x/36

```bash
python3 train_lora_minicpm.py \
  --train data/v3/heading_hierarchy/train.jsonl \
  --out outputs/remedy-minicpm-v46-heading-v3-lora \
  --rank 8 --alpha 16 --batch 1 --grad-accum 8 \
  --downsample-mode 4x --max-slice-nums 36
```

Memory note: 4x/36 multiplies visual tokens; if the box OOMs, add `--qlora`
before shrinking slices.

### 2. reading_order v3 — train at 4x/36 (matches its serving policy)

```bash
python3 train_lora_minicpm.py \
  --train data/v3/reading_order/train.jsonl \
  --out outputs/remedy-minicpm-v46-reading-order-v3-lora \
  --rank 8 --alpha 16 --batch 1 --grad-accum 8 \
  --downsample-mode 4x --max-slice-nums 36
```

### 3. Evaluate — resolution must match training

```bash
# heading v3 against the standard val split, at 4x/36
python3 run_adapter_eval.py \
  --adapter outputs/remedy-minicpm-v46-heading-v3-lora \
  --task heading --downsample-mode 4x --max-slice-nums 36 \
  --eval-root eval_runs/heading_v3_4x36

# reading_order v3 against the standard val split, at 4x/36
python3 run_adapter_eval.py \
  --adapter outputs/remedy-minicpm-v46-reading-order-v3-lora \
  --task reading_order --downsample-mode 4x --max-slice-nums 36 \
  --eval-root eval_runs/reading_order_v3_4x36

# reading_order v3 against the held-out corruption diagnostic:
# run_adapter_eval reads <data-root>/reading_order/val.jsonl, so stage val_hard.
# Image relpaths are ../../tasks/reading_order/renders/... so any dir two
# levels under data/ resolves them — a plain copy is enough.
mkdir -p data/v3_hard_stage/reading_order
cp data/v3/reading_order/val_hard.jsonl data/v3_hard_stage/reading_order/val.jsonl
python3 run_adapter_eval.py \
  --adapter outputs/remedy-minicpm-v46-reading-order-v3-lora \
  --task reading_order --data-root data/v3_hard_stage \
  --downsample-mode 4x --max-slice-nums 36 \
  --eval-root eval_runs/reading_order_v3_valhard_4x36
```

### 4. Gates (v2 gate levels still apply)

- heading: valid JSON ≥ 0.90, status ≥ 0.95, **exact correction ≥ 0.90**,
  pass-FP ≤ 0.10.
- reading_order: valid JSON ≥ 0.90, status ≥ 0.80, pass-FP ≤ 0.10, and the
  new diagnostic: **val_hard status ≥ 0.80** (all rows are fails; a pass-all
  model scores 0 here — this is the collapse detector).

### 5. Promotion policy

- v1 aliases stay the desktop defaults until v3 beats gates.
- If heading v3 passes: switch the router `--adapter-image-settings` to add
  `heading_hierarchy=4x:36` (trained resolution must be served).
- Do not build multitask v3 until both per-task v3 candidates pass.
- Update the Hub adapter cards with v3 eval results before pushing
  (`--push-to-hub` on a rerun, or `huggingface-cli upload`).

---

# V4 Addendum — results (2026-07-10, Runpod H100)

The v3 runs above FAILED their gates. Root-cause chain, in order of discovery:

1. **Undertraining (the real killer).** This plan's training commands omitted
   `--max-steps`; the trainer defaults to 1 epoch. v1 used `--max-steps 300`
   (~12 epochs, per the adapter card). One-epoch runs (25–58 steps) collapse
   into the "always pass" basin (train loss stuck ~4.3). Ruled out first:
   task-tag eval mismatch (tag-injected re-evals: no effect), resolution
   (worse at 16x/1), trainer label corruption (empirically verified clean).
2. **Eval truncation.** `run_adapter_eval.py` defaults `--max-new-tokens 384`;
   heading fail cases with multiple findings need more (v1's promotion eval
   used 1024). All fail->None "invalid JSON" records in v3/v4 heading evals
   were truncation artifacts. Always pass `--max-new-tokens 1024` for heading.

## v4 results (v3 data unchanged, proper budgets)

| Adapter | Profile | Steps | val | val_hard |
|---|---|---|---|---|
| reading-order v4 | 16x/1 | 300 | status **1.00**, JSON 1.00, FP 0 | **1.00** (51/51 corruptions caught) |
| heading v4 | 4x/36 | 500 | status **1.00**, JSON 1.00, exact **0.829** | n/a |

- **reading-order v4: PROMOTE.** First adapter proven robust across all five
  corruption families (v1 was only ever tested on rotation).
- **heading: keep v1 promoted.** The high-res hypothesis is refuted cleanly:
  exact correction v1 0.857 > v2 0.848 > v4 0.829 — resolution does not fix
  H2/H3/H4 level picks. Detection is perfect in v4 (status 1.00). The ~0.85
  exact-correction ceiling across every variant points at **gold-label
  ambiguity / missing document-level context**, not pixels. To beat the 0.90
  gate, next candidates: multi-page context in the prompt, or a gold-label
  audit of the confused pairs.
- 2026-07-10 E2E gate (175 heldout docs): vision-driven dimensions healthy
  (alt 0.93); structural dims failed (reading_order 0.47, table 0.51,
  heading_semantics 0.65) — same failures under the cloud-model baseline, so
  the bottleneck is `project_remedy`'s tag-tree writing, not the adapters.
  Artifact: `eval_runs/e2e_gate_v1_full/`.
