import { useState, useEffect, useRef, useMemo } from "react";
import {
  getModelStatus,
  streamModelDownload,
  type ModelStatus,
  type DownloadProgress,
  type AvailableLocalModel,
} from "../api";

interface Props {
  onReady: () => void;
  onConfigureProvider: () => void;
}

export function ModelSetup({ onReady, onConfigureProvider }: Props) {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadingTag, setDownloadingTag] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getModelStatus().then(setStatus).catch(() => {});
  }, []);

  const offerings: AvailableLocalModel[] = useMemo(() => {
    if (status?.available_models && status.available_models.length > 0) {
      return status.available_models;
    }
    // Fallback so the screen still renders if a stale backend is missing
    // the available_models field. Mirrors the recommended default.
    return [
      {
        tag: status?.model_tag ?? "gemma4:e4b",
        label: "Gemma 4 E4B",
        params: status?.default_model?.params ?? "4B effective",
        size_gb: status?.default_model?.size_gb ?? 9.6,
        ram_gb: status?.default_model?.ram_gb ?? 16,
        recommended: true,
        description: "Recommended local vision model.",
      },
    ];
  }, [status]);

  const activeTag =
    selectedTag ??
    offerings.find((m) => m.recommended)?.tag ??
    offerings[0]?.tag ??
    "gemma4:e4b";

  function startDownload(tag: string) {
    setDownloading(true);
    setDownloadingTag(tag);
    setError(null);
    setProgress({ downloaded_mb: 0, total_mb: 0 });

    abortRef.current = streamModelDownload((p) => {
      if (p.error) {
        setError(p.error);
        setDownloading(false);
        setDownloadingTag(null);
        return;
      }
      if (p.done) {
        setDownloading(false);
        setDownloadingTag(null);
        onReady();
        return;
      }
      setProgress(p);
    }, tag);
  }

  function cancelDownload() {
    abortRef.current?.abort();
    setDownloading(false);
    setDownloadingTag(null);
    setProgress(null);
  }

  const pct =
    progress?.total_mb && progress.total_mb > 0
      ? Math.min(100, Math.round((progress.downloaded_mb ?? 0) / progress.total_mb * 100))
      : 0;

  return (
    <div className="flex min-h-screen items-center justify-center bg-canvas">
      <div className="w-full max-w-lg rounded-2xl border border-elevated bg-raised p-8">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
            <svg className="h-7 w-7 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-text" style={{ fontFamily: "var(--font-heading)" }}>
            Set Up Vision Provider
          </h2>
          <p className="mt-2 text-sm text-text-muted">
            Remedy PDF Desktop ships a bundled local Ollama runtime. Pick a
            local vision model below or configure a cloud provider in Model
            Settings.
          </p>
        </div>

        {!downloading && (
          <>
            <div className="mb-4 space-y-3">
              {offerings.map((m) => {
                const checked = m.tag === activeTag;
                return (
                  <label
                    key={m.tag}
                    className={`block cursor-pointer rounded-xl border px-4 py-4 text-left transition-colors ${
                      checked
                        ? "border-primary bg-primary/5"
                        : "border-elevated bg-canvas hover:border-text-muted"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <input
                        type="radio"
                        name="local-model"
                        checked={checked}
                        onChange={() => setSelectedTag(m.tag)}
                        className="mt-1 h-4 w-4 accent-primary"
                      />
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-base font-semibold text-text">
                            {m.label}
                          </span>
                          {m.recommended && (
                            <span className="rounded-full bg-primary/15 px-2 py-0.5 text-xs font-medium text-primary">
                              Recommended
                            </span>
                          )}
                        </div>
                        <div className="mt-1 text-sm text-text-muted">
                          {m.params} &middot; {m.size_gb} GB download &middot; ~{m.ram_gb} GB RAM
                        </div>
                        <div className="mt-2 text-xs text-text-muted">
                          {m.description}
                        </div>
                        <div className="mt-1 text-xs font-mono text-text-muted">
                          {m.tag}
                        </div>
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>

            {status && (
              <div className="mb-4 rounded-lg border border-elevated bg-canvas px-3 py-2 text-xs text-text-muted">
                Runtime endpoint: <span className="font-mono">{status.endpoint}</span>
                {status.error && (
                  <div className="mt-1 text-fail">{status.error}</div>
                )}
              </div>
            )}

            <button
              onClick={() => startDownload(activeTag)}
              className="w-full rounded-lg bg-primary px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-primary/90"
            >
              Download {activeTag} (
              {offerings.find((m) => m.tag === activeTag)?.size_gb ?? "?"} GB)
            </button>

            <button
              onClick={onConfigureProvider}
              className="mt-3 w-full rounded-lg border border-elevated bg-canvas px-4 py-3 text-sm font-semibold text-text-muted transition-colors hover:border-text-muted hover:text-text"
            >
              Configure Cloud Provider
            </button>

            <p className="mt-3 text-center text-xs text-text-muted">
              Local setup needs a one-time model download. You can swap models
              later from Model Settings.
            </p>
          </>
        )}

        {downloading && (
          <div className="space-y-3">
            <div className="h-3 overflow-hidden rounded-full bg-elevated">
              <div
                className="h-full rounded-full bg-primary transition-all duration-300"
                style={{ width: `${pct}%` }}
              />
            </div>
            <p className="text-center text-sm text-text-muted">
              {progress?.downloaded_mb ?? 0} MB / {progress?.total_mb ?? "?"} MB ({pct}%)
            </p>
            <p className="text-center text-xs text-text-muted">
              Pulling {downloadingTag ?? activeTag} through the local Ollama runtime.
            </p>
            {progress?.status && (
              <p className="text-center text-xs text-text-muted">{progress.status}</p>
            )}
            <div className="text-center">
              <button
                onClick={cancelDownload}
                className="text-xs text-fail underline hover:text-fail/80"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-lg bg-fail/10 px-4 py-3">
            <p className="text-sm text-fail">{error}</p>
            <button
              onClick={() => { setError(null); setDownloading(false); }}
              className="mt-2 text-xs text-fail underline hover:text-fail/80"
            >
              Try again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
