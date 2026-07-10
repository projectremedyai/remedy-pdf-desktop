# Desktop packaging spike — MiniCPM-V-4.6 LoRA adapters (macOS / Apple Silicon)

Plan step 7: "Package the winning route for desktop: PEFT sidecar first, GGUF/MLX second."
Date: 2026-07-09. Host: macOS, Apple Silicon. All artifacts under `eval_runs/` (gitignored).

## TL;DR

**The plan's open risk is resolved: it does not materialize.** llama.cpp converts our
language-tower PEFT LoRAs to GGUF losslessly, and **llama-server hot-swaps multiple adapters
per request** in a single process. Ship **GGUF adapters**, not a PEFT sidecar.

| Route | Verdict |
|---|---|
| 1. GGUF conversion (`convert_lora_to_gguf.py`) | **WORKS** — all 8 adapters, 0 tensors dropped |
| 2a. Ollama `ADAPTER <file>.gguf` | **WORKS** — vision inference confirmed |
| 2b. Ollama `ADAPTER <peft-dir>` (raw safetensors) | **BLOCKED** — Ollama can't read PEFT dirs |
| 2c. Ollama per-request adapter hot-swap | **BLOCKED** — needs one model entry per task |
| 2d. llama-server per-request adapter hot-swap | **WORKS** — N adapters, 1 process, per-request scale |
| 3a. MLX inference (`mlx-vlm`) | **WORKS** — `minicpmv4_6` supported, 4-bit runs |
| 3b. MLX LoRA loading of our PEFT adapters | **BLOCKED** — incompatible adapter format |
| 4. Merged model → GGUF (fallback) | **SUPPORTED** (not needed; assessed only) |

Recommendation: **llama.cpp / llama-server with multi-adapter GGUF hot-swap.**

---

## Route 1 — GGUF conversion: WORKS

`MiniCPMV4_6ForConditionalGeneration` **is** registered in the converter
(`conversion/__init__.py`, `conversion/minicpm.py`). The text tower subclasses
`Qwen3_5TextModel` and emits `general.architecture = qwen35`.

Setup:
```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp     # build 9910 (f5525f7e7)
python3 -m venv venv && ./venv/bin/pip install torch transformers safetensors numpy huggingface_hub peft gguf
```

Base model config only (~20 MB, no weights, public — no HF auth needed):
```python
snapshot_download("openbmb/MiniCPM-V-4.6",
    allow_patterns=["*.json","*.py","*.jinja","*.txt","*.model"],
    local_dir="base_minicpm_v46")
```

Command that worked (per adapter):
```bash
./venv/bin/python llama.cpp/convert_lora_to_gguf.py \
  --base base_minicpm_v46 --outtype f16 \
  --outfile gguf/alt-v1.gguf \
  tools/minicpm_edge/outputs/remedy-minicpm-v46-alt-v1-lora
```

All 8 adapters converted. **Lossless: 192/192 tensors, 96/96 modules preserved.**

| Adapter | rank | GGUF size |
|---|---|---|
| alt-v1, contrast-v1, heading-v1, multitask-v1, reading-order-v1, table-v1 | r=8 | 6.40 MB each |
| heading-v2, reading-order-v2 | r=16 | 12.79 MB each |

### Why it lines up (the thing that could have broken)

MiniCPM-V-4.6's text core is a Qwen3.5 **hybrid** stack: linear-attention layers fuse Q/K/V into
a single `attn_qkv.weight` (18 of 24 layers), while **full-attention layers keep separate
`attn_q/attn_k/attn_v`**. Our LoRAs target q/k/v on exactly layers **3, 7, 11, 15, 19, 23** —
precisely the full-attention layers — plus `gate/up/down_proj` on all 24.

Verified against Ollama's base GGUF: **0 of 96 LoRA target tensors missing.** Had the LoRAs
touched linear-attention layers, GGUF would have been a dead end.

### Proof the adapter is actually applied (not silently ignored)

Differential test at fixed `--temp 0` on the same prompt:

| Config | Output |
|---|---|
| no `--lora` | `...describe what the chart shows` |
| `--lora-scaled X:0.0` | `...describe what the chart shows` (identical → correct null behavior) |
| `--lora-scaled X:1.0` | `...describe what the chart` (diverges → applied) |
| `--lora-scaled X:6.0` | `Losalt remedios remedios remedios...` (degenerate → *our* trained weights dominate) |

The `remedios` collapse at high scale is the fine-tuning signal from our own `remedy-*` training data.

Verbose load confirms:
```
llama_adapter_lora_init_impl: loading lora adapter from 'gguf/alt-v1.gguf'
  general.architecture str = qwen35
  adapter.type         str = lora
  general.name         str = Remedy Minicpm v46 Alt v1 Lora
```

### Vision + LoRA end-to-end

```bash
llama-mtmd-cli -m <ollama-text-blob> --mmproj <ollama-projector-blob> \
  --lora gguf/alt-v1.gguf --image test_chart.png \
  -p "Write concise alt text for this chart." -n 80 --temp 0 -ngl 99
```
→ `A bar graph of quarterly revenue for the year 2025, with values for Q1 to Q4.`

The Ollama-pulled blobs are plain GGUF and reusable directly by llama.cpp:
- text: `~/.ollama/models/blobs/sha256-6b0c74962c44...` (529 MB, arch `qwen35`)
- mmproj: `~/.ollama/models/blobs/sha256-ca931d861d08...` (1.11 GB, arch `clip`)

---

## Route 2 — Ollama

### 2a. GGUF adapter: WORKS
```
# Modelfile
FROM minicpm-v4.6:latest
ADAPTER /abs/path/gguf/alt-v1.gguf
```
```bash
ollama create remedy-alt-v1 -f Modelfile   # → success
```
Vision prompt via `/api/generate` with base64 image returned a correct chart description, and
output **differs from the base model** on the same prompt at `temperature=0` → adapter is live.

**Storage is cheap:** Ollama reuses the base blobs. `remedy-alt-v1`'s manifest =
529 MB model + 1.11 GB projector (both shared) + **6.4 MB adapter**. `ollama list` shows
"1.6 GB" per entry but that is shared. 8 task models ≈ 1.65 GB total, not 13 GB.

### 2b. Raw PEFT safetensors dir: BLOCKED
| Attempt | Result |
|---|---|
| `ADAPTER /path/to/peft-dir/` | `Error: no Modelfile or safetensors files found` |
| `ADAPTER /path/adapter_model.safetensors` | `converting adapter` → `Error: open adapter_config.json: no such file or directory` |
| same, with cwd = adapter dir (config present) | same error |

Ollama globs for model-weight-style safetensors and resolves `adapter_config.json` relative to
the **server's** working directory, not the Modelfile. Unusable. Pre-convert to GGUF instead.
(Ollama v0.31.2)

### 2c. Per-request hot-swap: BLOCKED
Passing `"adapters": [...]` to `/api/generate` is **silently ignored** — the request succeeds and
returns base-model output. Ollama binds an adapter at `ollama create` time.

→ **Ollama needs one model entry per task** (`remedy-alt-v1`, `remedy-table-v1`, …).
Swapping tasks means swapping models, which triggers a model reload/unload cycle.

### 2d. llama-server per-request hot-swap: WORKS ✅
```bash
llama-server -m <text-blob> --mmproj <mmproj-blob> \
  --lora gguf/alt-v1.gguf --lora gguf/table-v1.gguf --lora gguf/heading-v2.gguf \
  --port 18231 -ngl 99
```
`GET /lora-adapters` → `[{id:0,...alt-v1}, {id:1,...table-v1}, {id:2,...heading-v2}]`

Per-request selection via `"lora":[{"id":N,"scale":S}]` on `/completion`, same prompt, `temp=0`:

| Active adapter | Output |
|---|---|
| alt-v1 (id 0) | ` A bar chart comparing the average number of hours spent on social media by` |
| table-v1 (id 1) | ` A bar chart showing the number of people who voted in the 2` |
| heading-v2 (id 2) | ` a bar chart of the number of people who voted in the 2` |
| all scale 0 (base) | ` A bar chart showing the distribution of different types of food consumed by the` |

Four distinct outputs, one loaded process, no reload. `--lora` also accepts comma-separated
lists and `--lora-scaled FNAME:SCALE`. **This is the multi-adapter hot-swap the plan feared was
unavailable.**

---

## Route 3 — MLX

**Inference: WORKS.** `mlx-vlm` 0.6.4 ships a `minicpmv4_6` model module
(`LanguageModel`, `VisionModel`, `MiniCPMVProcessor`, …). Downloaded
`mlx-community/MiniCPM-V-4.6-4bit` (2.18 GB) and ran the same test image:

> `A bar chart displaying quarterly revenue for the year 2025, with values for Q1, Q2, Q3, and Q4.`

Official MLX variants exist: `mlx-community/MiniCPM-V-4.6-{4bit,5bit,8bit,bf16,nvfp4,mxfp4,mxfp8}`.

**LoRA loading: BLOCKED (format incompatibility).**
`mlx_vlm.trainer.adapter_utils.load_adapters` expects MLX-native format:
`adapter_config.json` with `num_layers` + `lora_parameters`, and weights named
`adapters.safetensors`. Loading our PEFT dir fails:
```
AttributeError: 'types.SimpleNamespace' object has no attribute 'num_layers'
```
Worse than a rename: `linear_to_lora_layers(model, num_layers, config)` applies LoRA
**uniformly to the last N layers** with a single spec. It **cannot express our selective
targeting** (q/k/v on only layers 3/7/11/15/19/23, MLP on all 24). Using MLX would require
either merge-then-convert, or accepting a different adapter topology than what we trained.

---

## Route 4 — Merged model → GGUF (fallback, assessed not executed)

Supported; not needed. `convert_hf_to_gguf.py` resolves the architecture cleanly:

| mode | class | `general.architecture` |
|---|---|---|
| text (`--mmproj` off) | `MiniCPMV4_6TextModel` | `qwen35` |
| `--mmproj` | `MiniCPMV4_6VisionModel` | `clip` |

So `PEFT merge_and_unload()` → `convert_hf_to_gguf.py` → quantize would work. Costs: requires the
full ~9 GB base weights, and produces **one ~530 MB text GGUF per task** (vs a 6.4 MB adapter).
Only worth it if a single merged multitask model beats the adapter family on the eval gate.

---

## Recommendation

**Ship llama.cpp (`llama-server`) with multi-adapter GGUF hot-swap.** Do not ship a PEFT sidecar.

Rationale:
1. **No PyTorch/transformers in the desktop bundle.** llama.cpp is a ~21 MB brew formula
   (`brew install llama.cpp`); the PEFT sidecar drags in torch (GB-scale) and a Python runtime.
2. **One process, all tasks.** Load all task adapters at startup; select per request with
   `"lora":[{"id":N,"scale":1.0}]`. No model reload between routing decisions — the exact
   pattern the plan listed as an open risk.
3. **Tiny task artifacts.** 6.4 MB (r=8) / 12.8 MB (r=16) per task on a shared 1.64 GB base.
   Adding or updating a task ships megabytes, not gigabytes.
4. **Lossless.** 192/192 tensors, and target tensors align 1:1 with the base GGUF.
5. Verified on the vision path, not just text (`llama-mtmd-cli` with `--mmproj`).

**Ollama fallback (if the app must reuse the existing Ollama runtime):** works, but requires
**one `ollama create` model entry per task** and gives up per-request hot-swap. Given
`backend/app/ollama_runtime.py` already exists, this is the lower-friction path — at the cost of a
model swap per task. Adapter blobs are deduped, so the disk cost is ~6.4 MB per task.
Note Ollama would also need `ADAPTER` to point at a **pre-converted .gguf** — it cannot ingest
the PEFT dirs.

**MLX:** not recommended. Inference works, but adapter loading needs a bespoke PEFT→MLX
converter *and* our per-layer targeting doesn't fit its uniform last-N-layers model. No upside
over llama.cpp on this hardware.

---

## Reproduction artifacts

- `gguf/*.gguf` — 8 converted adapters
- `Modelfile.alt-v1` — working Ollama Modelfile (GGUF adapter)
- `Modelfile.alt-v1-safetensors` — the variant that fails (kept as evidence)
- `test_chart.png` — 560×420 synthetic bar chart used for all vision prompts
- Ollama model `remedy-alt-v1:latest` created during the spike (`ollama rm remedy-alt-v1` to clean up)

### Gotchas for whoever repeats this
- macOS has no `timeout`; use `gtimeout` or background+kill.
- `llama-cli` needs `-st` (single-turn). `-no-cnv` with `</dev/null` spins on EOF and floods the log.
- `llama-server` binding a port requires the sandbox to be disabled in this harness.
