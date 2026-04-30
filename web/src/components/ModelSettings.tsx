import { useState, useEffect, useRef } from "react";
import {
  getModelStatus,
  streamModelDownload,
  type ModelStatus,
  type DownloadProgress,
} from "../api";

interface Props {
  onClose: () => void;
}

export function ModelSettings({ onClose }: Props) {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function refresh() {
    try {
      const s = await getModelStatus();
      setStatus(s);
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }

  useEffect(() => {
    let cancelled = false;
    getModelStatus()
      .then((s) => {
        if (!cancelled) {
          setStatus(s);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function startDownload() {
    setDownloading(true);
    setError(null);
    setProgress({ downloaded_mb: 0, total_mb: 0 });

    abortRef.current = streamModelDownload((p) => {
      if (p.error) {
        setError(p.error);
        setDownloading(false);
        return;
      }
      if (p.done) {
        setDownloading(false);
        refresh();
        return;
      }
      setProgress(p);
    });
  }

  function cancelDownload() {
    abortRef.current?.abort();
    setDownloading(false);
    setProgress(null);
  }

  const model = status?.default_model;
  const pct =
    progress?.total_mb && progress.total_mb > 0
      ? Math.min(100, Math.round((progress.downloaded_mb ?? 0) / progress.total_mb * 100))
      : 0;

  return (
    <div className="mx-auto w-full max-w-2xl space-y-6 animate-fade-up">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-text" style={{ fontFamily: "var(--font-heading)" }}>
          Model Settings
        </h2>
        <button
          onClick={onClose}
          className="rounded-lg border border-elevated bg-raised px-3 py-1.5 text-sm text-text-muted hover:text-text hover:bg-elevated transition-colors"
        >
          &larr; Back
        </button>
      </div>

      {status?.installed && (
        <div className="rounded-xl border border-conformant/30 bg-conformant/10 px-5 py-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-conformant">
            Currently Installed
          </div>
          <div className="mt-1 flex items-baseline gap-3">
            <span className="text-lg font-semibold text-text">{status.model_tag}</span>
            <span className="text-sm text-text-muted">{status.size_mb.toFixed(0)} MB on disk</span>
          </div>
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
            onClick={startDownload}
            disabled={status?.installed}
            className="w-full rounded-lg bg-primary px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status?.installed
              ? `${status.model_tag} is already installed`
              : `Download ${status?.model_tag ?? "qwen3.5:4b"} (${model?.size_gb ?? "3.4"} GB)`}
          </button>
        </div>
      )}

      {downloading && (
        <div className="rounded-xl border border-elevated bg-raised p-6 space-y-3">
          <h3 className="text-sm font-semibold text-text">Downloading {status?.model_tag ?? "qwen3.5:4b"}...</h3>
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
