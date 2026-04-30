export function HowItWorksPage() {
  return (
    <div className="w-full max-w-4xl mx-auto space-y-16 py-4">

      {/* ── Hero ── */}
      <section className="text-center animate-fade-up">
        <h2 className="text-3xl font-bold text-text" style={{ fontFamily: "var(--font-heading)" }}>
          How Remedy PDF Desktop Works
        </h2>
        <p className="mt-3 text-base text-text-muted max-w-2xl mx-auto leading-relaxed">
          A multi-tier automated PDF accessibility remediation system that combines
          deterministic fixes, AI-assisted escalation, and research-backed reading order
          analysis. No black box — every fix is traceable and verifiable.
        </p>
        <p className="mt-2 text-xs text-text-muted">
          Based on peer-reviewed research — paper forthcoming on arXiv
        </p>
      </section>

      {/* ── The Problem ── */}
      <section className="animate-fade-up stagger-1">
        <SectionHeading>The Problem</SectionHeading>
        <div className="rounded-xl border border-elevated bg-raised p-6 space-y-3 text-sm text-text-muted leading-relaxed">
          <p>
            The <strong className="text-text">ADA Title II Final Rule</strong> requires state and local
            government web content to be accessible by <strong className="text-text">April 24, 2026</strong>.
            For higher education institutions, this means remediating thousands of existing PDFs —
            course catalogs, class schedules, forms, maps, calendars — to comply with WCAG 2.1 Level AA
            and PDF/UA-1.
          </p>
          <p>
            Manual remediation at scale costs hundreds of thousands of dollars. Existing tools like
            CommonLook, Equidox, and Adobe Acrobat Pro are semi-automated — they still require a human
            operator making judgment calls on every file. <strong className="text-text">Remedy PDF Desktop is fully automated.</strong>
          </p>
        </div>
      </section>

      {/* ── The Pipeline ── */}
      <section className="animate-fade-up stagger-2">
        <SectionHeading>The Pipeline</SectionHeading>
        <p className="text-sm text-text-muted mb-6">
          Every document goes through three stages. The key insight: most documents can be fixed with
          cheap, deterministic operations — AI only handles the residual hard cases.
        </p>
        <div className="space-y-4">
          <PipelineStep
            number={1}
            title="XY-Cut++ Reading Order"
            cost="$0 — No AI"
            color="text-conformant"
            description="Deterministic geometric analysis fixes multi-column reading order before anything else runs. Handles academic papers, newsletters, brochures, and forms."
            details={[
              "4 phases: pre-mask cross-layout elements, compute density ratio, recursive gap segmentation, merge",
              "Based on arXiv:2504.10258, ported from OpenDataLoader PDF (Apache 2.0)",
              "Handles multi-column, full-width headers/footers, sidebars, label-field forms",
              "Pure math — no API calls, no external dependencies",
            ]}
          />
          <PipelineStep
            number={2}
            title="240+ Accessibility Fixes"
            cost="Mostly $0 — AI optional"
            color="text-primary"
            description="Multi-tier escalation: deterministic fixes first (fast, free), then local model-assisted checks when the on-device runtime is available, then closed-loop repair strategies for the hardest cases."
            details={[
              "Tier 1: 240+ deterministic structural repairs — tagging, headings, tables, lists, forms, language, bookmarks, font encoding, BDC/EMC balance",
              "Tier 2: Model-assisted escalation with vision (alt text, contrast, reading order verification)",
              "Tier 3: Agentic loop — 17 tools, automatic veraPDF verification after each mutation, Pareto-optimal config evolution",
              "Font fix suite: GS encoding cleanup, precise CIDSet, 4-layer ToUnicode recovery",
            ]}
          />
          <PipelineStep
            number={3}
            title="Conformance Report (ACR)"
            cost="$0 — Deterministic"
            color="text-partial"
            description="Triple-layer validation generates a detailed Accessibility Conformance Report mapped to 23 WCAG 2.1 AA criteria."
            details={[
              "32 Adobe-equivalent accessibility checks with pass/fail status",
              "13 screen reader simulation checks (NVDA/VoiceOver behavior)",
              "veraPDF PDF/UA-1 (ISO 14289-1) structure validation",
              "Continuous readability score (0–100) replaces binary pass/fail",
              "Clear breakdown: what's fixable vs source-document limitations",
            ]}
          />
        </div>
      </section>

      {/* ── What Makes It Different ── */}
      <section className="animate-fade-up stagger-3">
        <SectionHeading>What Makes It Different</SectionHeading>
        <div className="grid gap-4 sm:grid-cols-2">
          <DiffCard
            title="Deterministic First, AI Second"
            description="240+ fixes run without any AI or API calls. Most tools require AI for everything — slow, expensive, unpredictable. Our core engine is instant and free."
          />
          <DiffCard
            title="Research-Backed Reading Order"
            description="XY-Cut++ from arXiv:2504.10258. Other tools guess column order or skip it entirely. We use peer-reviewed geometric analysis."
          />
          <DiffCard
            title="Font Fix Suite"
            description="GS encoding cleanup, precise CIDSet via fontTools, 4-layer ToUnicode recovery. Fixes font violations that other tools can't touch. 488+ PDFs fixed, 617+ violations eliminated."
          />
          <DiffCard
            title="Real Conformance Testing"
            description="Not just a checkbox. Triple-layer validation with screen reader simulation and a continuous 0–100 readability score — not binary pass/fail."
          />
          <DiffCard
            title="Built-In AI Assistant"
            description="Ask questions about your results in plain English. The agent queries actual report data with tool calling — no hallucination. Free models, no subscription."
          />
          <DiffCard
            title="Open Source"
            description="MIT license. 50,000+ lines across 82 Python modules. No black box. Built with AI-assisted engineering under human direction."
          />
        </div>
      </section>

      {/* ── Fix Categories ── */}
      <section className="animate-fade-up stagger-4">
        <SectionHeading>What Gets Fixed</SectionHeading>
        <p className="text-sm text-text-muted mb-4">
          Nine categories of accessibility repairs, applied automatically:
        </p>
        <div className="grid gap-3 sm:grid-cols-3">
          <CategoryCard title="Document Metadata" examples="Language tag, display title, bookmarks, viewer preferences" />
          <CategoryCard title="Structure Tagging" examples="Untagged content, orphan MCIDs, artifact marking" />
          <CategoryCard title="Headings" examples="Nesting repair (H1→H3 becomes H1→H2→H3)" />
          <CategoryCard title="Tables" examples="Header cells, scope attributes, TR/TH/TD structure, regularity" />
          <CategoryCard title="Lists" examples="Container nesting (L/LI/Lbl/LBody)" />
          <CategoryCard title="Forms" examples="Field tagging, descriptions, widget annotations" />
          <CategoryCard title="Links & Annotations" examples="Link tagging, annotation descriptions, tab order" />
          <CategoryCard title="Content Streams" examples="BDC/EMC marker balance, pagination artifacts" />
          <CategoryCard title="Font Encoding" examples="GS cleanup, CIDSet precision, ToUnicode recovery" />
        </div>
      </section>

      {/* ── AI Layer ── */}
      <section className="animate-fade-up">
        <SectionHeading>The AI Layer (Optional)</SectionHeading>
        <div className="rounded-xl border border-primary/20 bg-primary/5 p-6 space-y-3 text-sm text-text-muted leading-relaxed">
          <p>
            The core engine works without any AI or cloud provider key. When the bundled
            <strong className="text-text"> local Ollama runtime</strong> and downloaded model are available,
            additional vision-powered capabilities activate:
          </p>
          <ul className="space-y-2 ml-4">
            <li className="flex items-start gap-2">
              <span className="text-primary mt-1">+</span>
              <span><strong className="text-text">Alt text generation</strong> for images and figures</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary mt-1">+</span>
              <span><strong className="text-text">Color contrast detection and repair</strong> (WCAG AA 4.5:1 ratio)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary mt-1">+</span>
              <span><strong className="text-text">Visual reading order verification</strong> — confirms or overrides XY-Cut++</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary mt-1">+</span>
              <span><strong className="text-text">Closed-loop agentic system</strong> — 17 tool primitives with automatic verification after each mutation</span>
            </li>
          </ul>
        </div>
      </section>

      {/* ── Standards ── */}
      <section className="animate-fade-up">
        <SectionHeading>Standards & Compliance</SectionHeading>
        <div className="grid gap-3 sm:grid-cols-2">
          <StandardCard title="WCAG 2.1 Level AA" detail="23 success criteria mapped and evaluated" />
          <StandardCard title="PDF/UA-1 (ISO 14289-1)" detail="Structure validation via veraPDF" />
          <StandardCard title="Section 508" detail="Federal accessibility standard (covered by WCAG mapping)" />
          <StandardCard title="ADA Title II" detail="State/local government web accessibility requirement" />
        </div>
        <p className="mt-3 text-xs text-text-muted text-center">
          This application itself is built to WCAG 2.2 AA standards — skip links, ARIA landmarks,
          keyboard navigation, focus management, and reduced cognitive load.
        </p>
      </section>

      {/* ── Research ── */}
      <section className="animate-fade-up text-center pb-8">
        <SectionHeading>The Research</SectionHeading>
        <div className="rounded-xl border border-elevated bg-raised p-6 text-sm text-text-muted leading-relaxed max-w-2xl mx-auto">
          <p>
            Remedy PDF Desktop is part of a remediation research effort currently in preparation for arXiv submission.
            The paper details the multi-tier architecture, benchmark results (100% compliance on Adobe-equivalent
            checks, 37% reduction in PDF/UA-1 violations), and provides a case study in multi-model AI-assisted
            software development.
          </p>
          <p className="mt-3 text-xs text-text-muted">
            Phung, J. (2026). <em>Remedy PDF Desktop: An AI-Built System for Multi-Tier PDF Accessibility
            Remediation at Scale.</em> In preparation.
          </p>
        </div>
      </section>
    </div>
  );
}


/* ── Subcomponents ── */

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-lg font-semibold text-text mb-4" style={{ fontFamily: "var(--font-heading)" }}>
      {children}
    </h3>
  );
}

function PipelineStep({
  number,
  title,
  cost,
  color,
  description,
  details,
}: {
  number: number;
  title: string;
  cost: string;
  color: string;
  description: string;
  details: string[];
}) {
  return (
    <div className="rounded-xl border border-elevated bg-raised p-5">
      <div className="flex items-start gap-4">
        <div className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-primary/10 text-lg font-bold ${color}`} style={{ fontFamily: "var(--font-heading)" }}>
          {number}
        </div>
        <div className="flex-1">
          <div className="flex items-baseline gap-3">
            <h4 className="text-sm font-semibold text-text">{title}</h4>
            <span className={`text-xs font-medium ${color}`}>{cost}</span>
          </div>
          <p className="mt-1 text-sm text-text-muted leading-relaxed">{description}</p>
          <ul className="mt-3 space-y-1.5">
            {details.map((d, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-text-muted">
                <span className="mt-1 h-1 w-1 rounded-full bg-elevated flex-shrink-0" />
                {d}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function DiffCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="rounded-xl border border-elevated bg-raised p-5">
      <h4 className="text-sm font-semibold text-text">{title}</h4>
      <p className="mt-2 text-xs text-text-muted leading-relaxed">{description}</p>
    </div>
  );
}

function CategoryCard({ title, examples }: { title: string; examples: string }) {
  return (
    <div className="rounded-lg border border-elevated bg-raised px-4 py-3">
      <div className="text-xs font-semibold text-text">{title}</div>
      <div className="mt-1 text-xs text-text-muted">{examples}</div>
    </div>
  );
}

function StandardCard({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="rounded-lg border border-elevated bg-raised px-4 py-3">
      <div className="text-sm font-semibold text-primary">{title}</div>
      <div className="mt-1 text-xs text-text-muted">{detail}</div>
    </div>
  );
}
