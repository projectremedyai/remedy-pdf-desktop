import { useState, useEffect } from "react";
import { AppShell } from "./components/layout/AppShell";
import { ModelSetup } from "./components/ModelSetup";
import { ModelSettings } from "./components/ModelSettings";
import { UploadZone } from "./components/UploadZone";
import { RemediationProgress } from "./components/RemediationProgress";
import { ResultsSummary } from "./components/ResultsSummary";
import { HowItWorksPage } from "./components/HowItWorksPage";
import { useRemediation } from "./hooks/useRemediation";
import { getModelStatus } from "./api";

export type AppPage = "home" | "how-it-works" | "model-settings";

export default function App() {
  const { phase, job, progress, error, uploadInfo, upload, reset } =
    useRemediation();
  const [accepted, setAccepted] = useState(false);
  const [page, setPage] = useState<AppPage>("home");
  const [modelReady, setModelReady] = useState<boolean | null>(null);
  const [verificationMode, setVerificationMode] = useState<"sampled" | "full">(() => {
    if (typeof window === "undefined") return "sampled";
    const saved = window.localStorage.getItem("verification-mode");
    return saved === "full" ? "full" : "sampled";
  });

  useEffect(() => {
    let cancelled = false;

    async function loadModelStatus() {
      try {
        const s = await getModelStatus();
        if (!cancelled) setModelReady(s.installed);
      } catch {
        if (!cancelled) window.setTimeout(loadModelStatus, 1000);
      }
    }

    loadModelStatus();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem("verification-mode", verificationMode);
  }, [verificationMode]);

  if (modelReady === null) return null;
  if (!modelReady) return <ModelSetup onReady={() => setModelReady(true)} />;

  const centerContent = page === "home" && (phase === "upload" || phase === "progress" || phase === "error");

  return (
    <AppShell centerContent={centerContent} page={page} onNavigate={setPage}>
      {page === "how-it-works" && <HowItWorksPage />}
      {page === "model-settings" && <ModelSettings onClose={() => setPage("home")} />}

      {page === "home" && phase === "upload" && (
        <div className="flex w-full flex-col items-center gap-8 animate-fade-up">
          <div className="text-center">
            <h2
              className="text-3xl font-bold text-text"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              Remediate a document
            </h2>
            <p className="mt-2 text-sm text-text-muted">
              Upload a document to apply accessibility fixes and generate an
              accessibility review report
            </p>
          </div>

          {/* Disclaimer */}
          <div
            className="w-full max-w-3xl rounded-xl border border-partial/30 bg-partial/5 px-6 py-5"
            role="note"
            aria-labelledby="important-notice-title"
          >
            <div className="flex items-start gap-3">
              <svg className="mt-0.5 h-5 w-5 flex-shrink-0 text-partial" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
              <div className="flex-1 text-sm leading-relaxed text-text-muted">
                <p id="important-notice-title" className="font-semibold text-partial">Important Notice</p>
                <p className="mt-2">
                  Documents are remediated <strong className="text-text">on this device</strong>.
                  Core fixes use deterministic parsing and layout analysis, and
                  vision-assisted checks use the local Ollama runtime and downloaded model.
                </p>
                <p className="mt-2">
                  Installing or updating that local model can require a one-time
                  network download, but the document itself is not uploaded to an
                  external accessibility service during local remediation.
                </p>
                <p className="mt-2">
                  <strong className="text-text">This tool is experimental and under active development.</strong>{" "}
                  Automated fixes and reports can reduce manual work, but they do
                  not guarantee compliance with WCAG 2.1 AA, ADA Title II, PDF/UA,
                  or any other standard. Always review the remediated file before
                  publishing, distributing, or making accessibility claims about it.
                </p>
                <label className="mt-4 flex cursor-pointer items-start gap-3">
                  <input
                    type="checkbox"
                    checked={accepted}
                    onChange={(e) => setAccepted(e.target.checked)}
                    className="mt-0.5 h-4 w-4 rounded border-elevated bg-raised text-primary accent-primary focus:ring-primary focus:ring-offset-0"
                    aria-describedby="disclaimer-label"
                  />
                  <span id="disclaimer-label" className="text-text">
                    I understand that this run produces an automated assessment
                    only, and that the remediated document still needs manual
                    review before it is shared or relied on.
                  </span>
                </label>
              </div>
            </div>
          </div>

          {/* Upload zone */}
          <div className={`w-full max-w-3xl transition-all duration-300 ${!accepted ? "opacity-40 pointer-events-none select-none" : ""}`} aria-disabled={!accepted}>
            <div className="mb-5 rounded-xl border border-elevated bg-raised px-5 py-4">
              <div className="text-xs font-semibold uppercase tracking-wider text-text-muted">
                Verification Mode
              </div>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <label className={`cursor-pointer rounded-lg border px-4 py-3 transition-colors ${
                  verificationMode === "sampled"
                    ? "border-primary bg-primary/10"
                    : "border-elevated bg-canvas hover:border-primary/40"
                }`}>
                  <input
                    type="radio"
                    name="verification-mode"
                    value="sampled"
                    checked={verificationMode === "sampled"}
                    onChange={() => setVerificationMode("sampled")}
                    className="sr-only"
                  />
                  <div className="text-sm font-semibold text-text">Sampled verification</div>
                  <div className="mt-1 text-sm text-text-muted">
                    Faster run. Vision verification checks a representative subset of pages on large PDFs.
                  </div>
                </label>
                <label className={`cursor-pointer rounded-lg border px-4 py-3 transition-colors ${
                  verificationMode === "full"
                    ? "border-primary bg-primary/10"
                    : "border-elevated bg-canvas hover:border-primary/40"
                }`}>
                  <input
                    type="radio"
                    name="verification-mode"
                    value="full"
                    checked={verificationMode === "full"}
                    onChange={() => setVerificationMode("full")}
                    className="sr-only"
                  />
                  <div className="text-sm font-semibold text-text">Full verification</div>
                  <div className="mt-1 text-sm text-text-muted">
                    Slower run. Vision verification analyzes every page before the final report is generated.
                  </div>
                </label>
              </div>
            </div>
            <UploadZone
              onUpload={(file) => upload(file, verificationMode)}
              disabled={!accepted}
            />
          </div>
        </div>
      )}

      {page === "home" && phase === "progress" && (
        <RemediationProgress
          filename={uploadInfo?.filename ?? "document.pdf"}
          events={progress}
          pages={uploadInfo?.pages}
          verificationMode={uploadInfo?.verification_mode}
        />
      )}

      {page === "home" && phase === "results" && job && (
        <ResultsSummary job={job} onReset={reset} />
      )}

      {page === "home" && phase === "error" && (
        <div className="flex flex-col items-center gap-4 animate-fade-up">
          <div className="rounded-2xl border border-failing/30 bg-failing/10 px-8 py-6 text-center" role="alert">
            <p className="text-lg font-semibold text-failing">
              Remediation Failed
            </p>
            <p className="mt-2 text-sm text-text-muted">{error}</p>
          </div>
          <button
            onClick={reset}
            className="text-sm text-text-muted hover:text-primary transition-colors"
          >
            &larr; Try again
          </button>
        </div>
      )}
    </AppShell>
  );
}
