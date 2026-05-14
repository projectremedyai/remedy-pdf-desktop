import { useEffect, useRef } from "react";

interface ProgressEvent {
  type: string;
  step?: string;
  status?: string;
  message?: string;
  fixes_applied?: number;
  pages_reordered?: number;
  pages_analyzed?: number;
  tier_used?: string;
}

const STEPS = [
  { key: "xy_cut", label: "Reading Order" },
  { key: "remediation", label: "Fixing Issues" },
  { key: "report", label: "Generating Report" },
];

interface Props {
  filename: string;
  events: ProgressEvent[];
  pages?: number;
  verificationMode?: "sampled" | "full";
  fileSizeBytes?: number;
}

// Rough per-page budget, in seconds, for the on-device 4.7B vision model.
// Full = every page runs vision checks (alt text, reading order, tables,
// headings). Sampled = vision checks only on flagged pages.
const PER_PAGE_SECS = { full: 75, sampled: 20 };
const BASELINE_SECS = 45;  // XY-cut, non-vision fixes, report generation.

function estimateSeconds(pages: number, mode: "sampled" | "full", sizeMB: number): [number, number] {
  const base = PER_PAGE_SECS[mode] * Math.max(pages, 1) + BASELINE_SECS;
  // Size modifier: image-heavy PDFs send larger rendered pages to the model.
  const sizeFactor = sizeMB > 20 ? 1.4 : sizeMB > 5 ? 1.15 : 1;
  const mid = base * sizeFactor;
  return [Math.round(mid * 0.7), Math.round(mid * 1.4)];
}

function fmtRange(low: number, high: number): string {
  const toUnit = (s: number) =>
    s < 90 ? `${Math.round(s)}s` : `${Math.round(s / 60)} min`;
  return toUnit(low) === toUnit(high) ? toUnit(low) : `${toUnit(low)}–${toUnit(high)}`;
}

export function RemediationProgress({ filename, events, pages, verificationMode, fileSizeBytes }: Props) {
  const logRef = useRef<HTMLDivElement>(null);

  const completedSteps = new Set(
    events.filter((e) => e.status === "completed").map((e) => e.step)
  );
  const currentStep = events.findLast((e) => e.status === "running")?.step;

  const logMessages = events
    .filter((e) => e.message)
    .map((e, i) => ({
      id: i,
      message: e.message!,
      step: e.step,
      done: e.status === "completed",
    }));

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logMessages.length]);

  return (
    <div className="flex w-full max-w-2xl flex-col items-center gap-6 animate-fade-up" role="status" aria-live="polite" aria-label="Remediation progress">
      {/* File being processed */}
      <div className="flex items-center gap-3 rounded-xl border border-elevated bg-raised px-5 py-3">
        <svg className="h-5 w-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
        </svg>
        <span className="text-sm font-medium text-text">{filename}</span>
      </div>

      {/* Steps */}
      <div className="flex items-center gap-3">
        {STEPS.map((step, i) => {
          const done = completedSteps.has(step.key);
          const active = currentStep === step.key || (step.key === "remediation" && currentStep === "escalation");
          return (
            <div key={step.key} className="flex items-center gap-3">
              {i > 0 && (
                <div className={`h-px w-10 ${done ? "bg-conformant" : "bg-elevated"}`} />
              )}
              <div className="flex items-center gap-2">
                <div
                  className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-all ${
                    done
                      ? "bg-conformant text-canvas"
                      : active
                        ? "bg-primary text-white shadow-lg shadow-primary/30"
                        : "bg-elevated text-text-muted"
                  } ${active ? "animate-pulse" : ""}`}
                >
                  {done ? "\u2713" : i + 1}
                </div>
                <span
                  className={`text-xs font-medium ${
                    done ? "text-conformant" : active ? "text-primary" : "text-text-muted"
                  }`}
                >
                  {step.label}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Animated progress bar */}
      <div className="h-1 w-full max-w-sm overflow-hidden rounded-full bg-elevated" role="progressbar" aria-label="Processing document">
        <div className="h-full w-1/3 rounded-full bg-primary" style={{
          animation: "shimmer 1.5s ease-in-out infinite",
        }} />
      </div>

      {/* Activity log */}
      <div className="w-full rounded-xl border border-elevated bg-canvas">
        <div className="flex items-center gap-2 border-b border-elevated px-4 py-2">
          <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
          <span className="text-xs font-medium text-text-muted">Activity</span>
        </div>
        <div
          ref={logRef}
          className="max-h-48 overflow-y-auto px-4 py-2 font-mono text-xs leading-relaxed"
        >
          {logMessages.length === 0 && (
            <p className="text-text-muted">Starting remediation...</p>
          )}
          {logMessages.map((entry) => (
            <div key={entry.id} className="flex gap-2 py-0.5">
              <span className={entry.done ? "text-conformant" : "text-primary"}>
                {entry.done ? "\u2713" : "\u25b8"}
              </span>
              <span className={entry.done ? "text-text-muted" : "text-text"}>
                {entry.message}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Why this takes time */}
      <ProcessingEstimate
        pages={pages}
        verificationMode={verificationMode}
        fileSizeBytes={fileSizeBytes}
      />

      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); width: 33%; }
          50% { width: 66%; }
          100% { transform: translateX(300%); width: 33%; }
        }
      `}</style>
    </div>
  );
}

function ProcessingEstimate({
  pages,
  verificationMode,
  fileSizeBytes,
}: {
  pages?: number;
  verificationMode?: "sampled" | "full";
  fileSizeBytes?: number;
}) {
  const mode = verificationMode ?? "full";
  const sizeMB = (fileSizeBytes ?? 0) / (1024 * 1024);
  const estimate =
    pages && pages > 0 ? estimateSeconds(pages, mode, sizeMB) : null;

  return (
    <details className="w-full rounded-xl border border-elevated bg-raised text-sm">
      <summary className="cursor-pointer list-none px-5 py-3 text-xs font-medium text-text-muted hover:text-text">
        <span className="inline-flex items-center gap-2">
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
          </svg>
          Why this takes time
          {estimate && (
            <span className="ml-1 rounded-md bg-elevated px-2 py-0.5 text-[11px] text-text">
              ~{fmtRange(estimate[0], estimate[1])} for this file
            </span>
          )}
        </span>
      </summary>

      <div className="space-y-3 border-t border-elevated px-5 py-4 leading-relaxed text-text-muted">
        <p>
          Every PDF runs through 49 accessibility checks and fixes. Several of
          them can call the selected vision provider (reading order, figure alt
          text, heading detection, table structure, color contrast). Each call
          takes a few seconds, and some steps run per page.
        </p>

        <div className="rounded-lg border border-elevated bg-canvas p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-text">
            Rough time by page count ({mode} verification)
          </p>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-text-muted">
                <th className="pb-1 font-medium">Pages</th>
                <th className="pb-1 font-medium">Expected time</th>
              </tr>
            </thead>
            <tbody className="text-text">
              {[1, 5, 10, 25, 50].map((n) => {
                const [lo, hi] = estimateSeconds(n, mode, sizeMB);
                return (
                  <tr key={n} className="border-t border-elevated/60">
                    <td className="py-1">{n}</td>
                    <td className="py-1">{fmtRange(lo, hi)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <ul className="list-disc space-y-1 pl-5 text-xs">
          <li>
            <span className="text-text">Reading order pass</span> — geometry-only,
            ~5 s regardless of page count.
          </li>
          <li>
            <span className="text-text">Non-vision fixes</span> (MarkInfo, Lang,
            structure tree, bookmarks, form tagging, etc.) — ~15–30 s per document.
          </li>
          <li>
            <span className="text-text">Vision-backed fixes</span> (alt text,
            reading-order verification, tables, headings, contrast) — ~3–8 s per
            call, typically 1–4 calls per page in full mode.
          </li>
          <li>
            <span className="text-text">Accessibility report</span> — ~5–10 s at
            the end.
          </li>
          <li>
            Image-heavy or scanned PDFs take longer because rendered pages send
            more tokens to the model.
          </li>
          <li>
            First run of a session adds ~30–60 s for the model cold-load.
          </li>
        </ul>
      </div>
    </details>
  );
}
