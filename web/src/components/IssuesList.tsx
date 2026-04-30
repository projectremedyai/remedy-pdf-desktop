import { useId, useState } from "react";

interface Props {
  failedChecks: Array<{ rule_id: string; description: string; fixable: boolean }>;
  manualReviewChecks?: Array<{ rule_id: string; description: string; details: string[]; decision?: string; recommendation: string }>;
  sourceLimitedIssues?: Array<{ rule_id: string; description: string; details: string[]; recommendation: string }>;
  srIssues: Array<{ rule_id: string; severity: string; count: number }>;
  wcagFailures: Array<{ criterion_id: string; criterion_name: string; remarks: string }>;
  wcagManualReviews?: Array<{ criterion_id: string; criterion_name: string; remarks: string }>;
}

export function IssuesList({
  failedChecks,
  manualReviewChecks = [],
  sourceLimitedIssues = [],
  srIssues,
  wcagFailures,
  wcagManualReviews = [],
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const headingId = useId();
  const descriptionId = useId();
  const panelId = useId();

  const totalIssues =
    failedChecks.length
    + manualReviewChecks.length
    + sourceLimitedIssues.length
    + srIssues.length
    + wcagFailures.length
    + wcagManualReviews.length;
  if (totalIssues === 0) return null;

  return (
    <section className="overflow-hidden rounded-xl border border-elevated bg-raised" aria-labelledby={headingId}>
      <div className="px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 id={headingId} className="text-sm font-semibold text-text">
              Items to review
            </h3>
            <p id={descriptionId} className="mt-1 max-w-2xl text-sm leading-relaxed text-text-muted">
              Automated checks and vision-assisted review still flagged {totalIssues} item{totalIssues === 1 ? "" : "s"}.
              Use this list as a review queue, not as a final compliance ruling.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            aria-expanded={expanded}
            aria-controls={panelId}
            aria-describedby={descriptionId}
            className="rounded-lg border border-elevated bg-canvas/40 px-3 py-2 text-sm font-medium text-text transition-colors hover:bg-elevated/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-canvas"
          >
            {expanded ? "Hide details" : "Show details"}
          </button>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {failedChecks.length > 0 && (
            <SummaryPill label="Failed checks" count={failedChecks.length} tone="partial" />
          )}
          {manualReviewChecks.length > 0 && (
            <SummaryPill label="Manual review" count={manualReviewChecks.length} tone="partial" />
          )}
          {sourceLimitedIssues.length > 0 && (
            <SummaryPill label="Source limited" count={sourceLimitedIssues.length} tone="source" />
          )}
          {srIssues.length > 0 && (
            <SummaryPill label="Screen reader findings" count={srIssues.length} tone="partial" />
          )}
          {wcagFailures.length > 0 && (
            <SummaryPill label="WCAG criteria" count={wcagFailures.length} tone="failing" />
          )}
          {wcagManualReviews.length > 0 && (
            <SummaryPill label="WCAG review" count={wcagManualReviews.length} tone="partial" />
          )}
        </div>
      </div>

      {expanded && (
        <div id={panelId} role="region" aria-labelledby={headingId} className="space-y-5 border-t border-elevated px-5 py-4">
          {/* Failed checks */}
          {failedChecks.length > 0 && (
            <section>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-muted">
                Failed Checks
              </h4>
              <ul className="space-y-2">
                {failedChecks.map((c) => (
                  <li key={c.rule_id} className="flex items-start gap-3 rounded-lg border border-elevated/70 bg-canvas/40 px-3 py-2 text-sm">
                    <span className={`mt-0.5 inline-block h-2 w-2 rounded-full flex-shrink-0 ${c.fixable ? "bg-partial" : "bg-failing"}`} />
                    <div>
                      <p>
                        <span className="font-mono text-xs text-primary">{c.rule_id}</span>
                        <span className="ml-2 text-text-muted">{c.description}</span>
                      </p>
                      {!c.fixable && (
                        <p className="mt-1 text-xs text-failing">
                          May require source-document changes or manual tagging.
                        </p>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Manual review checks */}
          {manualReviewChecks.length > 0 && (
            <section>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-muted">
                Needs Manual Review
              </h4>
              <ul className="space-y-2">
                {manualReviewChecks.map((c) => (
                  <li key={c.rule_id} className="flex items-start gap-3 rounded-lg border border-elevated/70 bg-canvas/40 px-3 py-2 text-sm">
                    <span className="mt-0.5 inline-block h-2 w-2 rounded-full bg-partial flex-shrink-0" />
                    <div>
                      <p>
                        <span className="font-mono text-xs text-primary">{c.rule_id}</span>
                        <span className="ml-2 text-text-muted">{c.description}</span>
                      </p>
                      {c.decision && (
                        <span className="mt-1 inline-flex rounded-full border border-partial/30 bg-partial/10 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-partial">
                          {formatDecision(c.decision)}
                        </span>
                      )}
                      {(c.details ?? []).length > 0 && (
                        <p className="mt-1 text-xs leading-relaxed text-text-muted">
                          {(c.details ?? []).slice(0, 2).join("; ")}
                        </p>
                      )}
                      {c.recommendation && (
                        <p className="mt-1 text-xs leading-relaxed text-partial">
                          {c.recommendation}
                        </p>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Source-limited issues */}
          {sourceLimitedIssues.length > 0 && (
            <section>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-muted">
                Source Limited
              </h4>
              <ul className="space-y-2">
                {sourceLimitedIssues.map((issue) => (
                  <li key={issue.rule_id} className="flex items-start gap-3 rounded-lg border border-elevated/70 bg-canvas/40 px-3 py-2 text-sm">
                    <span className="mt-0.5 inline-block h-2 w-2 rounded-full bg-primary flex-shrink-0" />
                    <div>
                      <p>
                        <span className="font-mono text-xs text-primary">{issue.rule_id}</span>
                        <span className="ml-2 text-text-muted">{issue.description}</span>
                      </p>
                      {(issue.details ?? []).length > 0 && (
                        <p className="mt-1 text-xs leading-relaxed text-text-muted">
                          {(issue.details ?? []).slice(0, 2).join("; ")}
                        </p>
                      )}
                      {issue.recommendation && (
                        <p className="mt-1 text-xs leading-relaxed text-primary">
                          {issue.recommendation}
                        </p>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Screen reader issues */}
          {srIssues.length > 0 && (
            <section>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-muted">
                Screen Reader Issues
              </h4>
              <ul className="space-y-2">
                {srIssues.map((sr) => (
                  <li key={sr.rule_id} className="flex items-center gap-3 rounded-lg border border-elevated/70 bg-canvas/40 px-3 py-2 text-sm">
                    <span className={`inline-block h-2 w-2 rounded-full flex-shrink-0 ${sr.severity === "error" ? "bg-failing" : "bg-partial"}`} />
                    <span className="font-mono text-xs text-primary">{sr.rule_id}</span>
                    <span className="text-text-muted">{formatSeverity(sr.severity)}</span>
                    <span className="ml-auto font-mono text-xs text-text-muted">
                      {sr.count} finding{sr.count === 1 ? "" : "s"}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* WCAG manual review */}
          {wcagManualReviews.length > 0 && (
            <section>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-muted">
                WCAG Criteria Needing Manual Review
              </h4>
              <ul className="space-y-2">
                {wcagManualReviews.map((w) => (
                  <li key={w.criterion_id} className="flex items-start gap-3 rounded-lg border border-elevated/70 bg-canvas/40 px-3 py-2 text-sm">
                    <span className="mt-0.5 inline-block h-2 w-2 rounded-full bg-partial flex-shrink-0" />
                    <div>
                      <p>
                        <span className="font-mono text-xs text-primary">{w.criterion_id}</span>
                        <span className="ml-2 text-text">{w.criterion_name}</span>
                      </p>
                      {w.remarks && (
                        <p className="mt-1 text-xs leading-relaxed text-text-muted">
                          {w.remarks}
                        </p>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* WCAG failures */}
          {wcagFailures.length > 0 && (
            <section>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-muted">
                WCAG Criteria Not Met
              </h4>
              <ul className="space-y-2">
                {wcagFailures.map((w) => (
                  <li key={w.criterion_id} className="flex items-start gap-3 rounded-lg border border-elevated/70 bg-canvas/40 px-3 py-2 text-sm">
                    <span className="mt-0.5 inline-block h-2 w-2 rounded-full bg-failing flex-shrink-0" />
                    <div>
                      <p>
                        <span className="font-mono text-xs text-primary">{w.criterion_id}</span>
                        <span className="ml-2 text-text">{w.criterion_name}</span>
                      </p>
                      {w.remarks && (
                        <p className="mt-1 text-xs leading-relaxed text-text-muted">
                          {w.remarks}
                        </p>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}
    </section>
  );
}

function SummaryPill({
  label,
  count,
  tone,
}: {
  label: string;
  count: number;
  tone: "partial" | "failing" | "source";
}) {
  const toneClasses =
    tone === "source"
      ? "border-primary/30 bg-primary/10 text-primary"
      : tone === "failing"
      ? "border-failing/30 bg-failing/10 text-failing"
      : "border-partial/30 bg-partial/10 text-partial";

  return (
    <span className={`rounded-full border px-2.5 py-1 text-xs font-medium ${toneClasses}`}>
      {label}: {count}
    </span>
  );
}

function formatSeverity(severity: string) {
  return severity.charAt(0).toUpperCase() + severity.slice(1);
}

function formatDecision(decision: string) {
  switch (decision) {
    case "no_pass":
      return "No pass";
    case "rerun_full_verification":
      return "Rerun full verification";
    case "pass":
      return "Pass";
    default:
      return "Manual review";
  }
}
