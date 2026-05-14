import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { save } from "@tauri-apps/plugin-dialog";
import { writeFile } from "@tauri-apps/plugin-fs";

import type { FixSummary, JobStatus } from "../api";
import { apiFetch, pdfDownloadUrl, reportDownloadUrl } from "../api";
import { IssuesList } from "./IssuesList";
import { RepairStrategyPicker } from "./RepairStrategyPicker";

type DownloadKind = "pdf" | "report";
type DownloadResult =
  | { kind: "saved"; path: string }
  | { kind: "cancelled" }
  | { kind: "error"; message: string };

async function downloadFile(
  url: string,
  defaultName: string,
  ext: string,
): Promise<DownloadResult> {
  try {
    const filePath = await save({
      defaultPath: defaultName,
      filters: [{ name: ext.toUpperCase(), extensions: [ext] }],
    });
    if (!filePath) return { kind: "cancelled" };

    const res = await apiFetch(url);
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const buffer = await res.arrayBuffer();
    await writeFile(filePath, new Uint8Array(buffer));
    return { kind: "saved", path: filePath };
  } catch (err) {
    return {
      kind: "error",
      message: err instanceof Error ? err.message : String(err),
    };
  }
}

function conformanceBadge(status: string) {
  switch (status) {
    case "Conformant":
      return { bg: "bg-conformant/10 border-conformant/30", text: "text-conformant", glow: "shadow-conformant/10" };
    case "Partially Conformant":
      return { bg: "bg-partial/10 border-partial/30", text: "text-partial", glow: "shadow-partial/10" };
    default:
      return { bg: "bg-failing/10 border-failing/30", text: "text-failing", glow: "shadow-failing/10" };
  }
}

// The backend Conformance enum stays "Conformant / Partially Conformant /
// Not Conformant" for API stability, but the UI reframes those as triage
// verdicts so users don't confuse automated checks with legal certification.
// See compliance_report.py::_TRIAGE_LABEL for the matching HTML mapping.
function triageLabel(conformance: string): string {
  switch (conformance) {
    case "Conformant":
      return "Triage: Clear";
    case "Partially Conformant":
      return "Triage: Partial";
    case "Not Conformant":
      return "Triage: Failing";
    default:
      return conformance;
  }
}

interface Props {
  job: JobStatus;
  onReset: () => void;
}

export function ResultsSummary({ job, onReset }: Props) {
  const [currentJob, setCurrentJob] = useState(job);
  const [toast, setToast] = useState<{ kind: "ok" | "err"; message: string } | null>(null);
  const [saving, setSaving] = useState<DownloadKind | null>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const resultsHeadingId = useId();
  const assessmentNoteId = useId();
  const downloadsHeadingId = useId();

  useEffect(() => {
    setCurrentJob(job);
  }, [job]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      headingRef.current?.focus();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [currentJob.id]);

  useEffect(() => {
    if (!toast) return undefined;
    const timeout = window.setTimeout(
      () => setToast(null),
      toast.kind === "ok" ? 4000 : 6000,
    );
    return () => window.clearTimeout(timeout);
  }, [toast]);

  const report = currentJob.report;
  const fixes = currentJob.fix_summary;

  if (!report) return null;

  const badge = conformanceBadge(report.conformance);
  const manualReviewChecks = report.manual_review_checks ?? [];
  const sourceLimitedIssues = report.source_limited_issues ?? [];
  const sourceLimitedCount = report.source_limited_count ?? sourceLimitedIssues.length;
  const wcagManualReviews = report.wcag_manual_reviews ?? [];
  const wcagReviewCount = report.wcag_review_count ?? wcagManualReviews.length;
  const reviewCount =
    report.failed_checks.length
    + manualReviewChecks.length
    + sourceLimitedIssues.length
    + report.sr_issues.length
    + report.wcag_failures.length
    + wcagManualReviews.length;
  const wcagTotal = report.wcag_pass_count + report.wcag_fail_count;
  const reviewSummary =
    reviewCount === 0
      ? "No review items were flagged in this automated run."
      : `${reviewCount} item${reviewCount === 1 ? "" : "s"} still need review.`;

  async function handleDownload(kind: DownloadKind) {
    setSaving(kind);

    const base = currentJob.filename.replace(/\.(pdf|docx|pptx|xlsx)$/i, "");
    const outExt = (currentJob.filename.split(".").pop() || "pdf").toLowerCase();
    const result =
      kind === "pdf"
        ? await downloadFile(
            pdfDownloadUrl(currentJob.id),
            `${base}_remediated.${outExt}`,
            outExt,
          )
        : await downloadFile(
            reportDownloadUrl(currentJob.id),
            `${base}_acr.html`,
            "html",
          );

    setSaving(null);

    if (result.kind === "saved") {
      setToast({
        kind: "ok",
        message:
          kind === "pdf"
            ? `Saved remediated file to ${result.path}`
            : `Saved accessibility report to ${result.path}`,
      });
      return;
    }

    if (result.kind === "error") {
      setToast({
        kind: "err",
        message:
          kind === "pdf"
            ? `Could not save the remediated file: ${result.message}`
            : `Could not save the accessibility report: ${result.message}`,
      });
    }
  }

  return (
    <section className="relative w-full space-y-6" aria-labelledby={resultsHeadingId}>
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        Results ready for {currentJob.filename}. Automated triage verdict: {triageLabel(report.conformance)}.{" "}
        {reviewSummary} This is an automated triage, not a compliance certification.
      </div>

      {/* Persistent disclaimer — stays visible on every results view so the
          honest framing is unmissable. The pre-upload checkbox isn't enough;
          users return to this screen after the checkbox is out of sight. */}
      <div
        role="note"
        className="animate-fade-up rounded-xl border border-partial/40 bg-partial/10 px-5 py-4"
      >
        <p className="text-sm font-semibold text-text">
          This report is an automated triage, not a compliance certification.
        </p>
        <p className="mt-1 text-sm leading-relaxed text-text-muted">
          Roughly two-thirds of PDF/UA-1 accessibility requirements are
          machine-testable. The remaining third (semantic correctness,
          keyboard-trap behaviour, consistent navigation, language-of-parts)
          requires human judgment. Before publishing, review the remediated
          document in{" "}
          <a
            href="https://pac.pdf-accessibility.org/en"
            target="_blank"
            rel="noopener noreferrer"
            className="font-semibold text-primary underline"
          >
            PAC 2024
          </a>{" "}
          or Adobe Acrobat Pro, and listen to it with a screen reader.
        </p>
      </div>

      {toast && (
        <div
          role={toast.kind === "ok" ? "status" : "alert"}
          aria-live={toast.kind === "ok" ? "polite" : "assertive"}
          aria-atomic="true"
          className={`fixed right-6 top-6 z-50 max-w-md rounded-xl border px-4 py-3 text-sm shadow-lg animate-fade-up ${
            toast.kind === "ok"
              ? "border-conformant/30 bg-conformant/10 text-conformant"
              : "border-failing/30 bg-failing/10 text-failing"
          }`}
        >
          {toast.message}
        </div>
      )}

      <div className={`animate-fade-up flex flex-col items-center gap-3 rounded-2xl border ${badge.bg} px-8 py-6 text-center shadow-lg ${badge.glow}`}>
        <span className="text-xs font-semibold uppercase tracking-[0.2em] text-text-muted">
          Automated Triage
        </span>
        <h2
          ref={headingRef}
          id={resultsHeadingId}
          tabIndex={-1}
          className={`text-3xl font-bold ${badge.text} focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-4 focus-visible:ring-offset-canvas`}
          style={{ fontFamily: "var(--font-heading)" }}
        >
          {triageLabel(report.conformance)}
        </h2>
        <span className="text-sm text-text-muted">{currentJob.filename}</span>
        <span className="rounded-full border border-elevated bg-canvas px-3 py-1 text-xs font-semibold uppercase tracking-wide text-text-muted">
          {currentJob.verification_mode === "full" ? "Full verification" : "Sampled verification"}
        </span>
        <p id={assessmentNoteId} className="max-w-2xl text-sm leading-relaxed text-text-muted">
          Automated-check verdict only — this reflects the machine-testable
          subset of WCAG 2.1 AA and PDF/UA-1. Human review (see the banner
          above) is still required before publishing or making an accessibility
          claim.
        </p>
      </div>

      <div className="animate-fade-up stagger-1 rounded-xl border border-primary/20 bg-primary/5 px-5 py-4" role="note" aria-describedby={assessmentNoteId}>
        <p className="text-sm font-semibold text-text">Review before sharing</p>
        <p className="mt-1 text-sm leading-relaxed text-text-muted">
          {reviewSummary} Use the saved document and report to confirm reading
          order, headings, tables, alt text, form labels, and any source-content
          limitations the automation could not resolve.
        </p>
      </div>

      {report.verification_coverage && (
        <div className="animate-fade-up stagger-1 rounded-xl border border-elevated bg-raised px-5 py-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-text-muted">
                Verification Coverage
              </p>
              <p className="mt-1 text-sm font-semibold text-text">
                {report.verification_coverage.mode === "full"
                  ? "Full-document verification"
                  : "Sampled verification"}
              </p>
              <p className="mt-1 text-sm leading-relaxed text-text-muted">
                {report.verification_coverage.vision_checked
                  ? report.verification_coverage.covers_all_pages
                    ? `Vision verification analyzed all ${report.verification_coverage.total_pages} page(s).`
                    : `Vision verification analyzed ${report.verification_coverage.analyzed_page_count} of ${report.verification_coverage.total_pages} page(s).`
                  : "No vision verification data was available for this run."}
              </p>
            </div>
            {report.verification_coverage.vision_checked && !report.verification_coverage.covers_all_pages && (
              <span className="rounded-full border border-partial/30 bg-partial/10 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-partial">
                Partial page coverage
              </span>
            )}
          </div>
          {report.verification_coverage.unresolved_sampled_checks.length > 0 && (
            <div className="mt-3 rounded-lg border border-partial/20 bg-partial/5 px-4 py-3">
              <p className="text-xs font-semibold uppercase tracking-wider text-partial">
                Checks still unresolved because only part of the document was vision-verified
              </p>
              <p className="mt-1 text-sm text-text-muted">
                {report.verification_coverage.unresolved_sampled_checks.join(", ")}
              </p>
            </div>
          )}
        </div>
      )}

      <div className="animate-fade-up stagger-1 grid grid-cols-2 gap-3 sm:grid-cols-6">
        <StatCard label="Auto-fixed" value={fixes?.fixed_count ?? 0} color="text-conformant" />
        <StatCard
          label="Review Queue"
          value={reviewCount}
          color={reviewCount > 0 ? "text-partial" : "text-conformant"}
        />
        <StatCard
          label="WCAG Auto Pass"
          value={wcagTotal > 0 ? `${report.wcag_pass_count}/${wcagTotal}` : "N/A"}
          color="text-primary"
        />
        <StatCard
          label="WCAG Manual Review"
          value={wcagReviewCount}
          color={wcagReviewCount > 0 ? "text-partial" : "text-conformant"}
        />
        <StatCard
          label="Source Limited"
          value={sourceLimitedCount}
          color={sourceLimitedCount > 0 ? "text-primary" : "text-conformant"}
        />
        <StatCard
          label="SR Readability"
          value={report.screen_reader_readability != null ? `${Math.round(report.screen_reader_readability)}%` : "N/A"}
          color="text-primary"
        />
      </div>

      <section className="animate-fade-up stagger-2 space-y-3" aria-labelledby={downloadsHeadingId}>
        <div>
          <h3 id={downloadsHeadingId} className="text-sm font-semibold text-text">
            Downloads
          </h3>
          <p className="mt-1 text-sm text-text-muted">
            Save the updated file and the report that explains what this run
            changed, checked, and still could not verify automatically.
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <DownloadCard
            title="Remediated document"
            description="The updated output from this run. Review the file before publishing, distributing, or making accessibility claims about it."
            buttonLabel="Save remediated file"
            busyLabel="Saving remediated file..."
            disabled={saving !== null}
            busy={saving === "pdf"}
            tone="primary"
            onClick={() => handleDownload("pdf")}
            icon={
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
              </svg>
            }
          />
          <DownloadCard
            title="Accessibility review report"
            description="An HTML summary of the automated checks, screen-reader findings, and review items that may still need human inspection."
            buttonLabel="Save accessibility report"
            busyLabel="Saving accessibility report..."
            disabled={saving !== null}
            busy={saving === "report"}
            tone="secondary"
            onClick={() => handleDownload("report")}
            icon={
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25zM6.75 12h.008v.008H6.75V12zm0 3h.008v.008H6.75V15zm0 3h.008v.008H6.75V18z" />
              </svg>
            }
          />
        </div>
      </section>

      {/* Visual fidelity check (diff % + vision artifact scan) */}
      {fixes && (fixes.visual_diff_pct != null || fixes.visual_artifact_check) && (
        <div className="animate-fade-up stagger-2">
          <VisualFidelityCard fixes={fixes} />
        </div>
      )}

      {/* Issues detail */}
      {reviewCount > 0 && (
        <div className="animate-fade-up stagger-3">
          <IssuesList
            failedChecks={report.failed_checks}
            manualReviewChecks={manualReviewChecks}
            sourceLimitedIssues={sourceLimitedIssues}
            srIssues={report.sr_issues}
            wcagFailures={report.wcag_failures}
            wcagManualReviews={wcagManualReviews}
          />
        </div>
      )}

      {/* Residual-driven post-remediation strategy picker */}
      <div className="animate-fade-up stagger-3">
        <RepairStrategyPicker jobId={currentJob.id} onJobUpdated={setCurrentJob} />
      </div>

      {/* Remediate another */}
      <div className="animate-fade-up stagger-4 flex justify-center pt-6 pb-8">
        <button
          type="button"
          onClick={onReset}
          className="text-sm text-text-muted hover:text-primary transition-colors"
        >
          &larr; Start another remediation
        </button>
      </div>
    </section>
  );
}

function StatCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="rounded-xl border border-elevated bg-raised p-4" aria-label={`${label}: ${value}`}>
      <div className="text-xs font-medium uppercase tracking-wider text-text-muted">{label}</div>
      <div className={`mt-1 text-xl font-bold ${color}`} style={{ fontFamily: "var(--font-heading)" }}>
        {value}
      </div>
    </div>
  );
}

function DownloadCard({
  title,
  description,
  buttonLabel,
  busyLabel,
  disabled,
  busy,
  tone,
  onClick,
  icon,
}: {
  title: string;
  description: string;
  buttonLabel: string;
  busyLabel: string;
  disabled: boolean;
  busy: boolean;
  tone: "primary" | "secondary";
  onClick: () => void;
  icon: ReactNode;
}) {
  const descriptionId = useId();
  const buttonClasses =
    tone === "primary"
      ? "bg-primary text-white shadow-lg shadow-primary/20 hover:bg-primary-light"
      : "border border-elevated bg-raised text-text hover:bg-elevated";

  return (
    <div className="rounded-xl border border-elevated bg-raised p-4">
      <h4 className="text-sm font-semibold text-text">{title}</h4>
      <p id={descriptionId} className="mt-1 text-sm leading-relaxed text-text-muted">
        {description}
      </p>
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        aria-describedby={descriptionId}
        aria-busy={busy}
        className={`mt-4 flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-canvas disabled:cursor-not-allowed disabled:opacity-60 ${buttonClasses}`}
      >
        {busy ? (
          <>
            <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24" aria-hidden="true">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            {busyLabel}
          </>
        ) : (
          <>
            {icon}
            {buttonLabel}
          </>
        )}
      </button>
    </div>
  );
}

/**
 * Visual fidelity signals that the pixel-level stats cards don't surface:
 *  - visual_diff_pct from fix_and_verify's diff gate (how much the remediated
 *    render differs from the source)
 *  - manual-review flag when diff exceeds the configured threshold
 *  - results of the post-remediation vision artifact check (the selected vision provider inspects
 *    rendered pages for gray tints, stray boxes, missing text)
 *
 * Purpose: give the user a "should I eyeball this output?" signal before
 * they trust the conformance badge. The pixel diff catches aggregate
 * regressions; the vision check catches subtle localized issues the pixel
 * diff misses.
 */
function VisualFidelityCard({ fixes }: { fixes: FixSummary }) {
  const diffPct = fixes.visual_diff_pct ?? 0;
  const diffPercent = diffPct * 100;
  const check = fixes.visual_artifact_check;
  const needsReview = fixes.needs_manual_review;

  // Color the diff value: green <5%, yellow 5-10%, red >10% (matches the
  // 10% manual-review threshold we configured on the backend).
  const diffColor =
    diffPercent < 5 ? "text-conformant" : diffPercent < 10 ? "text-partial" : "text-failing";

  const hasArtifacts = !!check?.has_artifacts;
  const checkSummary = !check
    ? null
    : !check.checked
    ? "Vision check couldn't run"
    : hasArtifacts
    ? `Vision flagged ${check.flagged_pages.length} page${check.flagged_pages.length === 1 ? "" : "s"}`
    : `Vision inspected ${check.pages_checked.length} page${check.pages_checked.length === 1 ? "" : "s"}, no artifacts detected`;

  return (
    <div className="rounded-xl border border-elevated bg-raised overflow-hidden">
      <div className="px-5 py-4">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-text-muted">
              Visual Fidelity
            </div>
            <div className="mt-1 flex items-baseline gap-2">
              <span
                className={`text-xl font-bold ${diffColor}`}
                style={{ fontFamily: "var(--font-heading)" }}
              >
                {diffPercent.toFixed(2)}%
              </span>
              <span className="text-xs text-text-muted">pixel diff vs. source</span>
            </div>
          </div>
          {checkSummary && (
            <div
              className={`flex items-center gap-2 text-xs ${
                hasArtifacts ? "text-partial" : "text-text-muted"
              }`}
            >
              <svg
                className="h-4 w-4 flex-shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
                aria-hidden="true"
              >
                {hasArtifacts ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
                  />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
                )}
              </svg>
              <span>{checkSummary}</span>
            </div>
          )}
        </div>

        {/* Manual-review banner */}
        {needsReview && fixes.manual_review_reason && (
          <div
            className="mt-3 rounded-lg border border-partial/30 bg-partial/10 px-3 py-2 text-xs text-partial"
            role="alert"
          >
            <span className="font-semibold">Needs manual review:</span>{" "}
            {fixes.manual_review_reason}
          </div>
        )}

        {/* Per-page artifact findings */}
        {hasArtifacts && check && check.flagged_pages.length > 0 && (
          <div className="mt-3 space-y-1.5">
            {check.flagged_pages.map((p) => (
              <div
                key={p.page}
                className="rounded-lg border border-partial/20 bg-partial/5 px-3 py-2 text-xs"
              >
                <div className="font-semibold text-partial">Page {p.page}</div>
                {p.description && (
                  <div className="mt-0.5 text-text-muted">{p.description}</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
