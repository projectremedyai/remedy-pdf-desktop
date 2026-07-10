MiniCPM-V-4.6 Edge Adapter Workspace
====================================

Purpose
-------
This folder implements the MiniCPM-V-4.6 path for Remedy PDF Desktop:

- sync the existing Remedy task corpora from the sibling multitask repo,
- run base MiniCPM evals over all task validation sets,
- train PEFT LoRA adapters for alt text, contrast, heading, table, reading order, and multitask,
- serve the resulting adapter family behind an OpenAI-compatible router with an Ollama /api/tags shim.

Generated folders are ignored by git:

- tools/minicpm_edge/data/
- tools/minicpm_edge/eval_runs/
- tools/minicpm_edge/outputs/
- tools/minicpm_edge/outputs*/
- tools/minicpm_edge/checkpoints/
- tools/minicpm_edge/remote_bundle/

Initial L4 Setup
----------------
Use Python 3.12 on the Heidi L4 box. Install PyTorch separately from the CUDA 12.9 wheel index, then install the remaining dependencies.

    uv python install 3.12
    uv venv --python 3.12 .venv
    . .venv/bin/activate
    uv pip install --index-url https://download.pytorch.org/whl/cu129 torch torchvision
    uv pip install -r tools/minicpm_edge/requirements-l4.txt

Data Sync
---------

    python tools/minicpm_edge/sync_task_data.py

Expected local counts:

- alt_text_quality: train 266, val 50
- table_structure: train 208, val 20
- reading_order: train 200, val 44
- contrast: train 143, val 17
- heading_hierarchy: train 1070, val 210
- multitask_contrast_weighted: train 2602, val 341

Base Eval
---------
Fast smoke over one example per task:

    python tools/minicpm_edge/run_base_eval.py --limit 1

Full base eval:

    python tools/minicpm_edge/run_base_eval.py

High-resolution generation is compatible with the current MiniCPM/Transformers
stack. The H100 v1 sweep used `downsample_mode=4x` and `max_slice_nums=36`
without shape errors. It should not be enabled globally, though: table and
reading-order passed cleanly at high resolution, while heading exact correction
and contrast gates regressed.

Current router policy:

- table: `4x/36`
- reading_order: `4x/36`
- alt_text_quality: `16x/1`
- contrast: `16x/1`
- heading_hierarchy: `16x/1`

High-resolution sweep:

    python tools/minicpm_edge/run_base_eval.py --limit 10 --downsample-mode 4x --max-slice-nums 36 --eval-root tools/minicpm_edge/eval_runs/base_4x36

High-resolution training note:

MiniCPM-V 4.6 requires `downsample_mode` in both `processor.apply_chat_template`
and the model forward/generation call. The generation scripts already pass it to
`model.generate`; the training collator now includes it in the Trainer batch for
`model.forward`. Before spending another H100 run, verify the training forward
path with:

    python tools/minicpm_edge/probe_training_forward.py \
      --train tools/minicpm_edge/data/tasks/heading_hierarchy/train.jsonl \
      --downsample-mode 4x \
      --max-slice-nums 36

To reproduce the old failure path on a GPU workbench, omit the forward argument:

    python tools/minicpm_edge/probe_training_forward.py \
      --train tools/minicpm_edge/data/tasks/heading_hierarchy/train.jsonl \
      --downsample-mode 4x \
      --max-slice-nums 36 \
      --omit-forward-downsample

LoRA Smoke
----------

    python tools/minicpm_edge/train_adapter_family.py --tasks alt --max-steps 30 --rank 8 --alpha 16 --batch 1 --grad-accum 8

If the L4 OOMs on bf16, rerun the smoke with:

    python tools/minicpm_edge/train_adapter_family.py --tasks alt --max-steps 30 --rank 8 --alpha 16 --batch 1 --grad-accum 8 --qlora

Full Adapter Family
-------------------

    python tools/minicpm_edge/train_adapter_family.py --include-multitask --rank 8 --alpha 16 --batch 1 --grad-accum 8 --push-to-hub

This writes private PEFT adapters to the johnnyrobotai repos named in minicpm_edge/constants.py.

Tuned Eval
----------

Alt-only eval before the rest of the adapter family exists:

    python tools/minicpm_edge/run_tuned_eval.py --tasks alt --eval-root tools/minicpm_edge/eval_runs/alt_v1

Full per-task eval after all task adapters exist:

    python tools/minicpm_edge/run_tuned_eval.py
    python tools/minicpm_edge/eval_router_readiness.py

Multitask comparison:

    python tools/minicpm_edge/run_tuned_eval.py --multitask --eval-root tools/minicpm_edge/eval_runs/multitask

Router
------

    python tools/minicpm_edge/serve_router_minicpm_peft.py \
      --host 0.0.0.0 \
      --port 8000 \
      --include-multitask \
      --downsample-mode 16x \
      --max-slice-nums 1 \
      --adapter-image-settings "table=4x:36,reading_order=4x:36" \
      --print-env

Adapter aliases:

- minicpm-v46-remedy
- minicpm-v46-remedy-alt-v1
- minicpm-v46-remedy-table-v1
- minicpm-v46-remedy-contrast-v1
- minicpm-v46-remedy-reading-order-v1
- minicpm-v46-remedy-heading-v1
- minicpm-v46-remedy-multitask-v1

Desktop prototype env:

    OLLAMA_BASE_URL=http://127.0.0.1:<tunnel-port>/v1
    OLLAMA_VISION_MODEL=minicpm-v46-remedy
    OLLAMA_VISION_TASK_MODELS=contrast:minicpm-v46-remedy-contrast-v1,reading_order:minicpm-v46-remedy-reading-order-v1,heading_hierarchy:minicpm-v46-remedy-heading-v1,table_structure:minicpm-v46-remedy-table-v1

Router readiness artifact:

    python tools/minicpm_edge/eval_router_readiness.py \
      --metrics-dir tools/minicpm_edge/eval_runs/router_metrics_v1 \
      --include-multitask \
      --out tools/minicpm_edge/eval_runs/router_readiness_v1_policy.json

V2 diagnostics:

    python tools/minicpm_edge/analyze_v2_misses.py \
      --out-dir tools/minicpm_edge/eval_runs/v2_miss_analysis_h100

V2 status from the H100 sprint:

- Heading v2 remains experimental: valid JSON 1.0, status 0.9952, exact
  correction 0.8476 against a 0.90 v2 gate.
- Reading-order v2 remains experimental: valid JSON 1.0 but status 0.50 because
  every fail validation example was predicted as pass.
- Do not build or promote multitask v2 until new per-task v2 candidates beat
  their gates.

Promotion Gates
---------------

- valid JSON rate >= 0.90 for every task
- pass false-positive rate <= 0.10
- table status accuracy >= 1.00
- contrast status accuracy >= 0.90 and near-threshold status accuracy >= 0.85
- reading order status accuracy >= 0.80
- heading status accuracy >= 0.95 and exact correction accuracy >= 0.85
