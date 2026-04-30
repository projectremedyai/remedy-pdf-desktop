import { useEffect, useMemo, useState } from "react";

import {
  getJob,
  getRepairStrategies,
  repairOutputDownloadUrl,
  runRepairStrategies,
  type JobStatus,
  type RepairStrategyState,
} from "../api";

interface Props {
  jobId: string;
  onJobUpdated?: (job: JobStatus) => void;
}

function statusTone(status: string) {
  switch (status) {
    case "success":
      return "border-conformant/20 bg-conformant/10 text-conformant";
    case "failed":
      return "border-failing/20 bg-failing/10 text-failing";
    case "flagged":
      return "border-partial/20 bg-partial/10 text-partial";
    default:
      return "border-elevated bg-canvas text-text-muted";
  }
}

export function RepairStrategyPicker({ jobId, onJobUpdated }: Props) {
  const [data, setData] = useState<RepairStrategyState | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      try {
        const next = await getRepairStrategies(jobId);
        if (cancelled) return;
        setData(next);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load repair strategies");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  useEffect(() => {
    if (!data) return;
    setSelectedIds(
      new Set(
        data.applicable_strategies
          .filter((strategy) => strategy.selected_by_default)
          .map((strategy) => strategy.id),
      ),
    );
    setRunning(data.running);
  }, [data]);

  const selectedInOrder = useMemo(
    () =>
      data?.applicable_strategies
        .filter((strategy) => selectedIds.has(strategy.id))
        .map((strategy) => strategy.id) ?? [],
    [data, selectedIds],
  );

  const isBusy = running || data?.running === true;
  const downloadUrl = repairOutputDownloadUrl(jobId);

  function toggleStrategy(strategyId: string) {
    setSelectedIds((previous) => {
      const next = new Set(previous);
      if (next.has(strategyId)) {
        next.delete(strategyId);
      } else {
        next.add(strategyId);
      }
      return next;
    });
  }

  async function handleRun() {
    if (selectedInOrder.length === 0) return;

    setRunning(true);
    setError(null);

    try {
      const next = await runRepairStrategies(jobId, selectedInOrder);
      setData(next);
      if (onJobUpdated) {
        const refreshedJob = await getJob(jobId);
        onJobUpdated(refreshedJob);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Strategy run failed");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="rounded-xl border border-elevated bg-raised p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-text-muted">
            Post-Remediation Strategies
          </div>
          <p className="mt-2 text-sm text-text-muted">
            Recommendations are based on{" "}
            {data?.analysis_basis === "strategy_output" ? "the latest strategy output" : "the current remediated PDF"}.
            Selected strategies run in the order shown.
          </p>
          <p className="mt-1 text-xs text-text-muted">
            The summary above refreshes after successful strategy runs.
          </p>
        </div>
        {data?.latest_output_available && (
          <a
            href={downloadUrl}
            download
            className="flex items-center gap-2 rounded-lg border border-conformant/30 bg-conformant/10 px-3 py-2 text-sm font-semibold text-conformant transition-colors hover:bg-conformant/20 no-underline"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
            </svg>
            Download Latest Repaired PDF
          </a>
        )}
      </div>

      {data?.latest_output_available && data.latest_output_label && (
        <div className="mt-3 text-xs text-text-muted">
          Latest output came from: <span className="text-text">{data.latest_output_label}</span>
        </div>
      )}

      {error && (
        <div className="mt-4 rounded-lg border border-failing/30 bg-failing/10 px-4 py-3 text-sm text-failing" role="alert">
          {error}
        </div>
      )}

      <div className="mt-4 space-y-3">
        {loading && !data ? (
          <div className="rounded-lg border border-elevated bg-canvas px-4 py-3 text-sm text-text-muted">
            Checking residual strategies…
          </div>
        ) : data?.applicable_strategies.length ? (
          <>
            {data.applicable_strategies.map((strategy) => (
              <label
                key={strategy.id}
                className={`flex cursor-pointer items-start gap-3 rounded-lg border px-4 py-3 transition-colors ${
                  selectedIds.has(strategy.id)
                    ? "border-primary/40 bg-primary/5"
                    : "border-elevated bg-canvas hover:bg-elevated/60"
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedIds.has(strategy.id)}
                  onChange={() => toggleStrategy(strategy.id)}
                  disabled={isBusy}
                  className="mt-0.5 h-4 w-4 rounded border-elevated bg-raised text-primary accent-primary disabled:opacity-50"
                />
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-text">{strategy.label}</div>
                  <div className="mt-1 text-sm text-text-muted">{strategy.reason}</div>
                </div>
              </label>
            ))}

            <button
              onClick={() => {
                void handleRun();
              }}
              disabled={isBusy || selectedInOrder.length === 0}
              aria-busy={isBusy}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-primary/20 transition-all hover:bg-primary-light disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isBusy ? (
                <>
                  <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24" aria-hidden="true">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Running selected strategies…
                </>
              ) : (
                "Run Selected Strategies"
              )}
            </button>
          </>
        ) : (
          <div className="rounded-lg border border-elevated bg-canvas px-4 py-3 text-sm text-text-muted">
            No applicable post-remediation strategies were detected for the current output.
          </div>
        )}
      </div>

      {data?.runs.length ? (
        <div className="mt-5">
          <div className="text-xs font-semibold uppercase tracking-wider text-text-muted">
            Run History
          </div>
          <div className="mt-3 space-y-2">
            {data.runs
              .slice()
              .reverse()
              .map((run, index) => (
                <div key={`${run.strategy_id}-${run.started_at}-${index}`} className="rounded-lg border border-elevated bg-canvas px-4 py-3">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-text">{run.label}</div>
                      <div className="mt-1 text-sm text-text-muted">{run.summary}</div>
                    </div>
                    <span className={`rounded-full border px-2.5 py-1 text-xs font-semibold uppercase tracking-wide ${statusTone(run.status)}`}>
                      {run.status}
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      ) : null}

      {data?.unavailable_strategies.length ? (
        <details className="mt-5 rounded-lg border border-elevated bg-canvas px-4 py-3">
          <summary className="cursor-pointer text-sm font-semibold text-text">
            Unavailable strategies ({data.unavailable_strategies.length})
          </summary>
          <div className="mt-3 space-y-3">
            {data.unavailable_strategies.map((strategy) => (
              <div key={strategy.id} className="rounded-lg border border-elevated bg-raised px-4 py-3">
                <div className="text-sm font-semibold text-text">{strategy.label}</div>
                <div className="mt-1 text-sm text-text-muted">{strategy.reason}</div>
              </div>
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}
