import { useState, useEffect, useRef } from "react";
import {
  getModelStatus,
  streamModelDownload,
  listLocalModels,
  getVisionSettings,
  putVisionSettings,
  type ModelStatus,
  type DownloadProgress,
  type LocalModelEntry,
  type VisionProvider,
  type VisionSettings,
} from "../api";

interface Props {
  onClose: () => void;
}

const PROVIDERS: Array<{ id: VisionProvider; label: string }> = [
  { id: "local", label: "Local Ollama" },
  { id: "ollama_cloud", label: "Ollama Cloud" },
  { id: "openrouter", label: "OpenRouter" },
];

export function ModelSettings({ onClose }: Props) {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [vision, setVision] = useState<VisionSettings | null>(null);
  const [localModels, setLocalModels] = useState<LocalModelEntry[]>([]);
  const [localModelsError, setLocalModelsError] = useState<string | null>(null);
  const [modelToPull, setModelToPull] = useState("");
  const [openrouterKeyDraft, setOpenrouterKeyDraft] = useState("");
  const [ollamaCloudKeyDraft, setOllamaCloudKeyDraft] = useState("");
  const [savingVision, setSavingVision] = useState(false);
  const [savedToast, setSavedToast] = useState<string | null>(null);

  async function refreshOllama() {
    try {
      const s = await getModelStatus();
      setStatus(s);
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }

  async function refreshLocalModels() {
    try {
      const list = await listLocalModels();
      if (list.reachable) {
        setLocalModels(list.models.filter((m) => m.name));
        setLocalModelsError(null);
      } else {
        setLocalModels([]);
        setLocalModelsError(list.error ?? "Local Ollama is not reachable");
      }
    } catch (err) {
      setLocalModels([]);
      setLocalModelsError(String(err));
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [s, v, list] = await Promise.all([
          getModelStatus(),
          getVisionSettings(),
          listLocalModels(),
        ]);
        if (cancelled) return;
        setStatus(s);
        setVision(v);
        setError(null);
        if (list.reachable) {
          setLocalModels(list.models.filter((m) => m.name));
          setLocalModelsError(null);
        } else {
          setLocalModels([]);
          setLocalModelsError(list.error ?? "Local Ollama is not reachable");
        }
      } catch (err) {
        if (!cancelled) setError(String(err));
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  function startDownload(modelName?: string) {
    const target = modelName?.trim() || undefined;
    setDownloading(true);
    setDownloadingModel(target ?? null);
    setError(null);
    setProgress({ downloaded_mb: 0, total_mb: 0 });

    abortRef.current = streamModelDownload((p) => {
      if (p.error) {
        setError(p.error);
        setDownloading(false);
        setDownloadingModel(null);
        return;
      }
      if (p.done) {
        setDownloading(false);
        setDownloadingModel(null);
        refreshOllama();
        refreshLocalModels();
        return;
      }
      setProgress(p);
    }, target);
  }

  function cancelDownload() {
    abortRef.current?.abort();
    setDownloading(false);
    setDownloadingModel(null);
    setProgress(null);
  }

  async function saveVision(patch: Parameters<typeof putVisionSettings>[0]) {
    if (!vision) return;
    setSavingVision(true);
    setError(null);
    try {
      const next = await putVisionSettings(patch);
      setVision(next);
      setOpenrouterKeyDraft("");
      setOllamaCloudKeyDraft("");
      setSavedToast("Saved");
      window.setTimeout(() => setSavedToast(null), 1800);
    } catch (err) {
      setError(String(err));
    } finally {
      setSavingVision(false);
    }
  }

  const model = status?.default_model;
  const pct =
    progress?.total_mb && progress.total_mb > 0
      ? Math.min(100, Math.round(((progress.downloaded_mb ?? 0) / progress.total_mb) * 100))
      : 0;

  const selectedLocalModelInstalled = Boolean(
    vision?.local_model && localModels.some((m) => m.name === vision.local_model),
  );

  return (
    <div className="mx-auto w-full max-w-3xl space-y-6 animate-fade-up">
      <div className="flex items-center justify-between gap-4">
        <h2 className="text-2xl font-bold text-text" style={{ fontFamily: "var(--font-heading)" }}>
          Model Settings
        </h2>
        <div className="flex items-center gap-3">
          {savedToast && <span className="text-xs text-conformant">{savedToast}</span>}
          <button
            onClick={onClose}
            className="rounded-lg border border-elevated bg-raised px-3 py-1.5 text-sm text-text-muted hover:text-text hover:bg-elevated transition-colors"
          >
            &larr; Back
          </button>
        </div>
      </div>

      {vision && (
        <div className="rounded-xl border border-elevated bg-raised p-6 space-y-5">
          <div>
            <h3 className="text-sm font-semibold text-text uppercase tracking-wider">
              Vision Provider
            </h3>
            <p className="mt-1 text-xs text-text-muted">
              Figure alt text, reading order, contrast checks, math tagging, and verification use this provider.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {PROVIDERS.map((provider) => (
              <button
                key={provider.id}
                onClick={() => saveVision({ provider: provider.id })}
                disabled={savingVision}
                className={
                  "rounded-lg border px-3 py-2 text-sm font-semibold transition-colors " +
                  (vision.provider === provider.id
                    ? "border-primary bg-primary/15 text-primary"
                    : "border-elevated bg-canvas text-text-muted hover:text-text hover:border-text-muted")
                }
              >
                {provider.label}
              </button>
            ))}
          </div>

          {vision.provider === "local" && (
            <div className="space-y-3">
              <div className="space-y-2">
                <label className="text-xs font-semibold text-text uppercase tracking-wider">
                  Local model
                </label>
                {localModelsError && <div className="text-xs text-failing">{localModelsError}</div>}
                <select
                  value={vision.local_model}
                  onChange={(event) => saveVision({ local_model: event.target.value })}
                  disabled={savingVision || localModels.length === 0}
                  className="w-full rounded-lg border border-elevated bg-canvas px-3 py-2 text-sm text-text focus:border-primary focus:outline-none"
                >
                  {!localModels.some((m) => m.name === vision.local_model) && (
                    <option value={vision.local_model}>{vision.local_model} (not installed)</option>
                  )}
                  {localModels.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                      {m.parameter_size ? ` - ${m.parameter_size}` : ""}
                      {m.size_mb ? ` (${m.size_mb} MB)` : ""}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
                <input
                  type="text"
                  value={modelToPull}
                  onChange={(event) => setModelToPull(event.target.value)}
                  placeholder="ollama model name"
                  className="rounded-lg border border-elevated bg-canvas px-3 py-2 text-sm text-text font-mono focus:border-primary focus:outline-none"
                />
                <button
                  onClick={() => {
                    const target = modelToPull.trim();
                    if (target) {
                      saveVision({ local_model: target });
                      startDownload(target);
                    }
                  }}
                  disabled={downloading || !modelToPull.trim()}
                  className="rounded-lg border border-primary bg-primary/10 px-4 py-2 text-sm font-semibold text-primary transition-colors hover:bg-primary/20 disabled:opacity-50"
                >
                  Pull & Use
                </button>
              </div>
            </div>
          )}

          {vision.provider === "openrouter" && (
            <div className="space-y-3">
              <label className="text-xs font-semibold text-text uppercase tracking-wider">
                OpenRouter model
              </label>
              <input
                type="text"
                value={vision.openrouter_model}
                onChange={(event) => setVision({ ...vision, openrouter_model: event.target.value })}
                onBlur={() => saveVision({ openrouter_model: vision.openrouter_model })}
                placeholder="openai/gpt-4o-mini"
                className="w-full rounded-lg border border-elevated bg-canvas px-3 py-2 text-sm text-text font-mono focus:border-primary focus:outline-none"
              />
              <p className="text-xs text-text-muted">
                Use the OpenRouter slug format (e.g. <span className="font-mono">openai/gpt-4o-mini</span>,
                {" "}<span className="font-mono">anthropic/claude-3-5-sonnet</span>,
                {" "}<span className="font-mono">google/gemini-2.0-flash</span>). Browse the catalog at
                {" "}<a className="underline" href="https://openrouter.ai/models" target="_blank" rel="noreferrer">openrouter.ai/models</a>.
              </p>
              <label className="text-xs font-semibold text-text uppercase tracking-wider">
                API key
                {vision.openrouter_api_key_set && (
                  <span className="ml-2 text-conformant">configured ({vision.openrouter_api_key})</span>
                )}
              </label>
              <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
                <input
                  type="password"
                  value={openrouterKeyDraft}
                  onChange={(event) => setOpenrouterKeyDraft(event.target.value)}
                  placeholder={vision.openrouter_api_key_set ? "Paste new key to replace" : "sk-or-v1-..."}
                  className="rounded-lg border border-elevated bg-canvas px-3 py-2 text-sm text-text font-mono focus:border-primary focus:outline-none"
                />
                <button
                  onClick={() => saveVision({ openrouter_api_key: openrouterKeyDraft })}
                  disabled={savingVision || !openrouterKeyDraft.trim()}
                  className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-primary/90 disabled:opacity-50"
                >
                  Save Key
                </button>
              </div>
            </div>
          )}

          {vision.provider === "ollama_cloud" && (
            <div className="space-y-3">
              <label className="text-xs font-semibold text-text uppercase tracking-wider">
                Ollama Cloud model
              </label>
              <input
                type="text"
                value={vision.ollama_cloud_model}
                onChange={(event) => setVision({ ...vision, ollama_cloud_model: event.target.value })}
                onBlur={() => saveVision({ ollama_cloud_model: vision.ollama_cloud_model })}
                className="w-full rounded-lg border border-elevated bg-canvas px-3 py-2 text-sm text-text font-mono focus:border-primary focus:outline-none"
              />
              <label className="text-xs font-semibold text-text uppercase tracking-wider">
                API key
                {vision.ollama_cloud_api_key_set && (
                  <span className="ml-2 text-conformant">configured ({vision.ollama_cloud_api_key})</span>
                )}
              </label>
              <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
                <input
                  type="password"
                  value={ollamaCloudKeyDraft}
                  onChange={(event) => setOllamaCloudKeyDraft(event.target.value)}
                  placeholder={vision.ollama_cloud_api_key_set ? "Paste new key to replace" : "ollama key"}
                  className="rounded-lg border border-elevated bg-canvas px-3 py-2 text-sm text-text font-mono focus:border-primary focus:outline-none"
                />
                <button
                  onClick={() => saveVision({ ollama_cloud_api_key: ollamaCloudKeyDraft })}
                  disabled={savingVision || !ollamaCloudKeyDraft.trim()}
                  className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-primary/90 disabled:opacity-50"
                >
                  Save Key
                </button>
              </div>
            </div>
          )}

          <div className="space-y-2 border-t border-elevated pt-4">
            <label className="text-xs font-semibold text-text uppercase tracking-wider">
              Per-page vision timeout
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={30}
                max={300}
                step={15}
                value={vision.page_timeout_seconds}
                onChange={(event) =>
                  setVision({ ...vision, page_timeout_seconds: Number(event.target.value) })
                }
                onMouseUp={() => saveVision({ page_timeout_seconds: vision.page_timeout_seconds })}
                onTouchEnd={() => saveVision({ page_timeout_seconds: vision.page_timeout_seconds })}
                className="flex-1"
              />
              <span className="w-16 text-right text-sm font-mono text-text">
                {vision.page_timeout_seconds}s
              </span>
            </div>
          </div>
        </div>
      )}

      {status?.installed && (
        <div className="rounded-xl border border-conformant/30 bg-conformant/10 px-5 py-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-conformant">
            Default Local Model Installed
          </div>
          <div className="mt-1 flex items-baseline gap-3">
            <span className="text-lg font-semibold text-text">{status.model_tag}</span>
            <span className="text-sm text-text-muted">{status.size_mb.toFixed(0)} MB on disk</span>
          </div>
          {vision?.provider === "local" && !selectedLocalModelInstalled && (
            <div className="mt-2 text-xs text-failing">
              Selected local model is not installed.
            </div>
          )}
          <div className="mt-1 text-xs text-text-muted font-mono">{status.models_dir}</div>
        </div>
      )}

      {!downloading && (
        <div className="rounded-xl border border-elevated bg-raised p-6 space-y-4">
          <h3 className="text-sm font-semibold text-text uppercase tracking-wider">
            Local Runtime
          </h3>
          {status && (
            <div className="rounded-lg border border-elevated bg-canvas px-4 py-4">
              <div className="text-sm font-semibold text-text">{status.model_tag}</div>
              <div className="mt-1 text-xs text-text-muted">
                {model?.params ?? "4.66B"} &middot; {model?.size_gb ?? "3.4"} GB download
              </div>
              <div className="mt-1 text-xs text-text-muted">
                Endpoint: <span className="font-mono">{status.endpoint}</span>
              </div>
              <div className="mt-1 text-xs text-text-muted">
                Model storage: <span className="font-mono">{status.models_dir}</span>
              </div>
              {status.error && (
                <div className="mt-3 rounded-lg bg-fail/10 px-3 py-2 text-xs text-fail">
                  {status.error}
                </div>
              )}
            </div>
          )}

          <button
            onClick={() => startDownload()}
            disabled={status?.installed}
            className="w-full rounded-lg bg-primary px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status?.installed
              ? `${status.model_tag} is already installed`
              : `Download ${status?.model_tag ?? "gemma4:e4b"} (${model?.size_gb ?? "9.6"} GB)`}
          </button>
        </div>
      )}

      {downloading && (
        <div className="rounded-xl border border-elevated bg-raised p-6 space-y-3">
          <h3 className="text-sm font-semibold text-text">
            Downloading {downloadingModel ?? status?.model_tag ?? "gemma4:e4b"}...
          </h3>
          <div className="h-3 overflow-hidden rounded-full bg-elevated">
            <div
              className="h-full rounded-full bg-primary transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-sm text-text-muted">
            <span>{progress?.downloaded_mb ?? 0} MB / {progress?.total_mb ?? "?"} MB ({pct}%)</span>
            <button
              onClick={cancelDownload}
              className="text-xs text-failing underline hover:text-failing/80"
            >
              Cancel
            </button>
          </div>
          {progress?.status && (
            <div className="text-xs text-text-muted">{progress.status}</div>
          )}
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-fail/10 px-4 py-3 text-sm text-fail">
          {error}
        </div>
      )}
    </div>
  );
}
