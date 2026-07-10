# MiniCPM High-Resolution Training Note

## Finding

`4x/36` high-resolution generation works for MiniCPM-V-4.6 on the Remedy page
renders, but the original training path failed because `downsample_mode` was
only passed to `processor.apply_chat_template`. It was not passed into the model
forward call made by `Trainer`.

MiniCPM-V-4.6 uses `downsample_mode` twice:

- processor side: controls image placeholder count in the tokenized prompt;
- model side: controls whether the vision tower emits 4x or 16x visual tokens.

If the processor uses `4x` and forward silently defaults to `16x`, the image
placeholder count and visual feature count diverge. That matches the observed
H100 failures:

- `RuntimeError: shape '[21, 1023, 1152]' is invalid for input of size 24754176`
- `ValueError: Multimodal features and tokens do not match, tokens: 252, features: 63`

## Fix

`train_lora_minicpm.py` now carries `downsample_mode` in the collator output so
`Trainer` passes it into `model.forward(...)`:

```python
out = {
    "input_ids": torch.stack(input_ids),
    "attention_mask": torch.stack(attention_mask),
    "labels": torch.stack(labels),
    "downsample_mode": self.downsample_mode,
}
```

`max_slice_nums` remains processor-only.

## Verification

Local non-GPU verification confirms the collator sends:

- processor kwargs: `{"downsample_mode": "4x", "max_slice_nums": 36}`
- forward batch: `downsample_mode="4x"`

GPU verification on the Heidi L4 also passed:

- artifact: `tools/minicpm_edge/eval_runs/highres_training_probe_l4/summary.json`
- fixed path: `4x/36` forward passed with loss `0.641650915145874`
- omitted-forward path: reproduced the original shape error

Run this on the next GPU workbench before a high-res training job:

```bash
PYTHONPATH=tools/minicpm_edge python tools/minicpm_edge/probe_training_forward.py \
  --train tools/minicpm_edge/data/tasks/heading_hierarchy/train.jsonl \
  --downsample-mode 4x \
  --max-slice-nums 36
```

To prove the old failure mode is gone for the right reason, this command should
fail or reproduce the mismatch:

```bash
PYTHONPATH=tools/minicpm_edge python tools/minicpm_edge/probe_training_forward.py \
  --train tools/minicpm_edge/data/tasks/heading_hierarchy/train.jsonl \
  --downsample-mode 4x \
  --max-slice-nums 36 \
  --omit-forward-downsample
```

Do not spend on a full H100 high-res v3 training run until the first command
passes.
