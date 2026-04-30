import { useState, useEffect, useRef } from "react";
import {
  getModelStatus,
  streamModelDownload,
  type ModelStatus,
  type DownloadProgress,
} from "../api";

interface Props {
  onReady: () => void;
}

export function ModelSetup({ onReady }: Props) {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getModelStatus().then(setStatus).catch(() => {});
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
        onReady();
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
    <div className="flex min-h-screen items-center justify-center bg-canvas">
      <div className="w-full max-w-lg rounded-2xl border border-elevated bg-raised p-8">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
            <svg className="h-7 w-7 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-text" style={{ fontFamily: "var(--font-heading)" }}>
            Set Up Local AI Runtime
          </h2>
          <p className="mt-2 text-sm text-text-muted">
            Remedy PDF Desktop uses a bundled local Ollama runtime with the
            <span className="font-medium text-text"> qwen3.5:4b </span>
            model for vision-assisted checks. Download once, then everything runs locally.
          </p>
        </div>

        {!downloading && (
          <>
            {status && (
              <div className="mb-4 rounded-xl border border-elevated bg-canvas px-4 py-4 text-left">
                <div className="text-xs font-semibold uppercase tracking-wider text-text-muted">
                  Default Model
                </div>
                <div className="mt-2 text-base font-semibold text-text">{status.model_tag}</div>
                <div className="mt-1 text-sm text-text-muted">
                  {model?.params ?? "4.66B"} &middot; {model?.size_gb ?? "3.4"} GB download
                </div>
                <div className="mt-1 text-xs text-text-muted">
                  Recommended for the local cross-platform desktop runtime.
                </div>
                <div className="mt-3 text-xs text-text-muted">
                  Runtime endpoint: <span className="font-mono">{status.endpoint}</span>
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
              className="w-full rounded-lg bg-primary px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-primary/90"
            >
              Download {status?.model_tag ?? "qwen3.5:4b"} ({model?.size_gb ?? "3.4"} GB)
            </button>

            <p className="mt-3 text-center text-xs text-text-muted">
              First-run internet is required once so the app can pull the local model.
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
              Pulling {status?.model_tag ?? "qwen3.5:4b"} through the local Ollama runtime.
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
