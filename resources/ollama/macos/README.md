# Bundled macOS Ollama runtime

This directory holds the macOS Ollama standalone CLI distribution that the
Tauri shell bundles into the packaged `.app` so the local vision/text
runtime works without requiring users to install Ollama separately.

The binaries are **not** stored in git (they total ~430 MB and rotate
with each Ollama release). `scripts/release_macos.sh` and any future
release pipeline should populate this directory before `tauri build`.

**Currently bundled:** Ollama **0.31.2** (universal x86_64+arm64), sourced
from the local `/Applications/Ollama.app/Contents/Resources` install (the
GUI-only `*.png`/`*.icns` assets are intentionally excluded). Re-populate from
the official `ollama-darwin.tgz` for a clean release; keep this version note in
sync with what is actually shipped.

## What goes here

The full extracted contents of `ollama-darwin.tgz` from the
[ollama/ollama GitHub releases](https://github.com/ollama/ollama/releases),
specifically:

```
ollama                              # main CLI binary (~80 MB)
libggml-base.{0.0.0,0,}.dylib       # GGML base runtime
libggml-cpu*.so                     # CPU variant shared libraries
mlx_metal_v3/                       # MLX Metal compute (older GPUs)
mlx_metal_v4/                       # MLX Metal compute (newer GPUs)
```

## How to populate

```bash
# From repo root, replace VERSION with the desired tag
VERSION=v0.23.3
curl -L -o /tmp/ollama-darwin.tgz \
  "https://github.com/ollama/ollama/releases/download/${VERSION}/ollama-darwin.tgz"
tar -xzf /tmp/ollama-darwin.tgz -C resources/ollama/macos/
chmod +x resources/ollama/macos/ollama
resources/ollama/macos/ollama --version  # sanity check
```

## Fallback

If this directory is empty when the app runs, the Tauri shell falls back
to a system-installed Ollama at:

- `/usr/local/bin/ollama` (Intel Homebrew)
- `/opt/homebrew/bin/ollama` (Apple Silicon Homebrew)
- `/Applications/Ollama.app/Contents/Resources/ollama` (official installer)

See `src-tauri/src/lib.rs` for the resolution order.
