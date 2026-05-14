function isBrowserHttpOrigin(): boolean {
  if (typeof window === "undefined") return true;
  return window.location.protocol === "http:" || window.location.protocol === "https:";
}

function backendOrigin(): string {
  if (isBrowserHttpOrigin()) return "";
  return "http://127.0.0.1:8000";
}

const APP_API_KEY = (import.meta.env.VITE_APP_API_KEY ?? "").trim();

function withAuthHeaders(headers?: HeadersInit): Headers {
  const next = new Headers(headers);
  if (APP_API_KEY) next.set("X-API-Key", APP_API_KEY);
  return next;
}

export function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  return fetch(input, {
    ...init,
    headers: withAuthHeaders(init.headers),
  });
}

export function withApiKeyQuery(url: string): string {
  if (!APP_API_KEY) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}api_key=${encodeURIComponent(APP_API_KEY)}`;
}

export function backendBaseUrl(): string {
  return `${backendOrigin()}/api`;
}

export function backendWebSocketOrigin(): string {
  if (typeof window === "undefined") return "ws://127.0.0.1:8000";
  if (isBrowserHttpOrigin()) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}`;
  }
  return "ws://127.0.0.1:8000";
}

const BASE = backendBaseUrl();

export interface UploadResult {
  job_id: string;
  filename: string;
  pages: number;
  verification_mode: "sampled" | "full";
}

export interface FlaggedPage {
  page: number;
  has_artifacts?: boolean;
  description?: string;
  error?: string;
}

export interface VisualArtifactCheck {
  checked: boolean;
  pages_checked: number[];
  has_artifacts: boolean;
  flagged_pages: FlaggedPage[];
  error_count: number;
}

export interface FixSummary {
  fixed_count: number;
  skipped_count: number;
  needs_manual_review: boolean;
  manual_review_reason: string;
  visual_diff_pct?: number;
  tier_used?: string;
  escalation_attempted?: boolean;
  gs_was_used?: boolean;
  visual_artifact_check?: VisualArtifactCheck;
}

export interface ReportData {
  document_name: string;
  conformance: string;
  tag_count: number;
  pages: number;
  failed_checks: Array<{ rule_id: string; description: string; fixable: boolean }>;
  reviewable_checks?: Array<{ rule_id: string; description: string; details: string[]; decision?: "pass" | "no_pass" | "rerun_full_verification" | "manual_review"; recommendation: string; status?: string }>;
  manual_review_checks?: Array<{ rule_id: string; description: string; details: string[]; decision?: string; recommendation: string }>;
  source_limited_issues?: Array<{ rule_id: string; description: string; details: string[]; recommendation: string }>;
  source_limited_count?: number;
  sr_issues: Array<{ rule_id: string; severity: string; count: number }>;
  wcag_results: Array<{ criterion_id: string; criterion_name: string; level: string; status: string; remarks: string }>;
  wcag_failures: Array<{ criterion_id: string; criterion_name: string; remarks: string }>;
  wcag_manual_reviews?: Array<{ criterion_id: string; criterion_name: string; remarks: string }>;
  wcag_pass_count: number;
  wcag_fail_count: number;
  wcag_review_count?: number;
  total_issues: number;
  screen_reader_readability: number;
  verapdf_checked: boolean;
  verapdf_passed: boolean | null;
  verapdf_violation_count: number;
  verification_coverage?: {
    mode: "sampled" | "full";
    vision_checked: boolean;
    total_pages: number;
    analyzed_page_count: number;
    analyzed_pages: number[];
    covers_all_pages: boolean;
    unresolved_sampled_checks: string[];
  };
}

export interface JobStatus {
  id: string;
  filename: string;
  status: "queued" | "running" | "completed" | "failed";
  current_step: string;
  error: string | null;
  verification_mode: "sampled" | "full";
  fix_summary?: FixSummary;
  report?: ReportData;
}

export async function uploadPdf(
  file: File,
  verificationMode: "sampled" | "full",
): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("verification_mode", verificationMode);
  const res = await apiFetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const res = await apiFetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Job fetch failed: ${res.statusText}`);
  return res.json();
}

export function pdfDownloadUrl(jobId: string): string {
  return `${BASE}/jobs/${jobId}/pdf`;
}

export function reportDownloadUrl(jobId: string): string {
  return `${BASE}/jobs/${jobId}/report`;
}

export interface RepairStrategyOption {
  id: string;
  label: string;
  reason: string;
  selected_by_default: boolean;
}

export interface RepairStrategyRun {
  strategy_id: string;
  label: string;
  status: "success" | "skipped" | "failed" | "flagged";
  summary: string;
  error: string | null;
  started_at: number;
  completed_at: number | null;
}

export interface RepairStrategyState {
  running: boolean;
  analysis_basis: "remediated_pdf" | "strategy_output";
  latest_output_available: boolean;
  latest_output_label: string | null;
  download_url?: string;
  applicable_strategies: RepairStrategyOption[];
  unavailable_strategies: RepairStrategyOption[];
  runs: RepairStrategyRun[];
}

export async function getRepairStrategies(jobId: string): Promise<RepairStrategyState> {
  const res = await apiFetch(`${BASE}/jobs/${jobId}/repair-strategies`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function runRepairStrategies(
  jobId: string,
  strategies: string[],
): Promise<RepairStrategyState> {
  const res = await apiFetch(`${BASE}/jobs/${jobId}/repair-strategies/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ strategies }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export function repairOutputDownloadUrl(jobId: string): string {
  return withApiKeyQuery(`${BASE}/jobs/${jobId}/repair-output`);
}

// --- Model management ---

export interface LocalModelInfo {
  tag: string;
  size_gb: number;
  params: string;
  ram_gb: number;
}

export interface ModelStatus {
  reachable: boolean;
  installed: boolean;
  endpoint: string;
  models_dir: string;
  model_tag: string;
  size_mb: number;
  default_model: LocalModelInfo;
  error?: string | null;
}

export async function getModelStatus(): Promise<ModelStatus> {
  const res = await apiFetch(`${BASE}/model/status`);
  if (!res.ok) throw new Error("Failed to fetch model status");
  return res.json();
}

export interface DownloadProgress {
  downloaded_mb?: number;
  total_mb?: number;
  done?: boolean;
  error?: string;
  status?: string;
  digest?: string;
}

export interface LocalModelEntry {
  name: string;
  size_mb: number;
  parameter_size: string;
  family: string;
}

export interface LocalModelList {
  reachable: boolean;
  models: LocalModelEntry[];
  error?: string;
}

export async function listLocalModels(): Promise<LocalModelList> {
  const res = await apiFetch(`${BASE}/model/list`);
  if (!res.ok) throw new Error("Failed to list local models");
  return res.json();
}

export type VisionProvider = "local" | "ollama_cloud" | "openrouter";

export interface VisionSettings {
  provider: VisionProvider;
  local_model: string;
  openrouter_model: string;
  ollama_cloud_model: string;
  openrouter_api_key: string;
  openrouter_api_key_set: boolean;
  ollama_cloud_api_key: string;
  ollama_cloud_api_key_set: boolean;
  page_timeout_seconds: number;
}

export async function getVisionSettings(): Promise<VisionSettings> {
  const res = await apiFetch(`${BASE}/settings/vision`);
  if (!res.ok) throw new Error("Failed to load vision settings");
  return res.json();
}

export async function putVisionSettings(
  patch: Partial<{
    provider: VisionProvider;
    local_model: string;
    openrouter_model: string;
    ollama_cloud_model: string;
    openrouter_api_key: string;
    ollama_cloud_api_key: string;
    page_timeout_seconds: number;
  }>,
): Promise<VisionSettings> {
  const res = await apiFetch(`${BASE}/settings/vision`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export function streamModelDownload(
  onProgress: (p: DownloadProgress) => void,
  modelName?: string,
): AbortController {
  const controller = new AbortController();
  const url = modelName
    ? `${BASE}/model/download?model=${encodeURIComponent(modelName)}`
    : `${BASE}/model/download`;
  apiFetch(url, {
    method: "POST",
    signal: controller.signal,
  }).then(async (res) => {
    if (!res.ok || !res.body) {
      onProgress({ error: `Download failed: ${res.statusText}` });
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            onProgress(JSON.parse(line.slice(6)));
          } catch { /* skip malformed */ }
        }
      }
    }
  }).catch((err) => {
    if (err.name !== "AbortError") onProgress({ error: String(err) });
  });
  return controller;
}
