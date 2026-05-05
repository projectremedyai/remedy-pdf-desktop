# Remedy PDF Desktop

Remedy PDF Desktop is a local-first document accessibility remediation application. It accepts PDF, Word, PowerPoint, and Excel files, produces a remediated output plus an HTML accessibility report, and surfaces signals for manual review before publication.

## What this tool is / is not

**It is** an automated triage for the machine-testable subset of PDF/UA-1 and WCAG 2.1 AA:

- Structure-tree repair (tags, headings, lists, tables, reading order)
- Alt-text synthesis (OCR fallback + optional on-device vision)
- Metadata + language + title normalization
- Screen-reader simulation against NVDA/VoiceOver patterns
- PDF/UA-1 validation via veraPDF
- Honest reporting of what it changed and what it couldn't

**It is not** a compliance certification. The Matterhorn Protocol 1.1 — the PDF Association's reference failure catalog for PDF/UA-1 — classifies roughly **two thirds** of accessibility requirements as machine-testable. The remaining third (whether headings convey the *correct* structure, whether alt text is *semantically* correct, keyboard-trap behaviour, language-of-parts, consistent navigation and identification) requires **human judgment**. No automated tool — including this one, PAC 2024, Adobe Acrobat Pro, or veraPDF — can close that third on its own.

**Before publishing a remediated PDF:**

1. Run it through [PAC 2024](https://pac.pdf-accessibility.org/en) and/or Adobe Acrobat Pro's built-in Accessibility Checker.
2. Listen to it with a screen reader (NVDA on Windows, VoiceOver on macOS).
3. Walk the five WCAG criteria this tool cannot test — see [`docs/HUMAN_REVIEW_GUIDE.md`](docs/HUMAN_REVIEW_GUIDE.md).

Do not claim WCAG, PDF/UA, Section 508, ADA Title II, or EAA compliance on the strength of this app's "Triage: Clear" verdict alone. That verdict means "the automated checks we *can* run passed" — nothing more.

## Current Behavior

- Upload formats: `.pdf`, `.docx`, `.pptx`, `.xlsx`
- Outputs: a remediated document, an HTML report (`*_acr.html`), and a results summary in the UI
- Review signals surfaced in the UI: fixes applied, remaining issues, WCAG mapping summary, screen-reader readability score, visual-diff/manual-review flags, and an experimental faithful rebuild action for completed PDF jobs
- Processing model: remediation runs on the local machine. If the on-device vision model is not installed yet, the frontend offers a one-time download from Hugging Face and then uses that model locally.

## Pipelines

### PDF documents

1. `XY-Cut++` pre-pass for reading order
2. `fix_and_verify()` remediation, with optional on-device vision assistance and escalation
3. `generate_document_report()` HTML report generation
4. Best-effort post-remediation visual artifact check when the local vision model is available

### Office documents

Word, PowerPoint, and Excel uploads use a separate Office remediator and still flow through the same upload, progress, and results screens. The PDF-specific report detail and experimental rebuild path do not apply in the same way to those formats.

## Running Locally

### Prerequisites

- Python 3.11+
- Node.js 22
- Rust 1.77.2+ for the native Tauri shell
- Optional: Java 17+ plus veraPDF for PDF/UA validation data in PDF reports
- Optional: Ghostscript for additional PDF preprocessing and repair paths

### Install

```bash
git clone https://github.com/projectremedyai/remedy-pdf-desktop.git
cd remedy-pdf-desktop

pip install -e .[dev]
cd web && npm install && cd ..

cp .env.example .env
```

`.env.example` contains optional local overrides. The app ships with no API keys required. The desktop workflow does not use Ollama Cloud. All LLM activity runs against a local Ollama runtime (default base URL `http://127.0.0.1:11500/v1`) with downloaded models.

### Run The Web UI During Development

```bash
# Backend
uvicorn backend.app.main:app --reload

# Frontend
cd web && npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

### Run The Native App During Development

```bash
npm install
npm run tauri:dev
```

The Tauri shell starts the local backend and local Ollama runtime for the desktop app. On first launch, the frontend may ask you to download a local vision/text model (default `qwen3.5:4b`, ~3.4 GB). That download needs network access once; document processing after installation stays local to the machine running the app.

### Verify

```bash
pytest tests/ -v
cd web && npm run lint && npm run build
```

## Interpreting Results

- The HTML report summarizes automated findings from the current remediation run.
- For PDF jobs, the report may include grouped failed checks, grouped screen-reader issues, WCAG criteria mapped from those findings, and veraPDF results when veraPDF is installed.
- Conformance labels shown by the app are outputs of this toolchain's current checks and heuristics. Treat them as triage signals, not as publication approval.
- If the UI flags a document for manual review, inspect the remediated output directly before publishing or distributing it.
- The faithful rebuild action is experimental and user-invoked. It is not part of the default automatic remediation path.

## Repository Layout

- `backend/app/` — FastAPI backend and job orchestration
- `lib/project_remedy/` — bundled remediation engine (import path retained; not renamed)
- `web/` — React/Vite/Tailwind frontend (npm package `remedy-pdf-desktop-web`)
- `src-tauri/` — Tauri desktop shell
- `backend/project_remedy_backend.spec` — PyInstaller spec for the bundled backend sidecar

## Packaging Identity

These values are sourced from `src-tauri/tauri.conf.json` and the workspace manifests; keep them in sync if anything changes:

- Root npm package: `remedy-pdf-desktop`
- Web npm package: `remedy-pdf-desktop-web`
- Python project (`pyproject.toml`): `remedy-pdf-desktop`
- Tauri `productName`: `Remedy PDF Desktop`
- Tauri `identifier`: `com.projectremedy.app` (bundle/updater identity intentionally retained from the prior `Project Remedy` naming to preserve install/update continuity)

## License

MIT. See `LICENSE`.
