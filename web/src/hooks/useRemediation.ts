import { useCallback, useRef, useState } from "react";
import {
  backendWebSocketOrigin,
  uploadPdf,
  getJob,
  withApiKeyQuery,
  type JobStatus,
  type UploadResult,
} from "../api";

interface ProgressEvent {
  type: string;
  step?: string;
  status?: string;
  message?: string;
  fixes_applied?: number;
  fixes_skipped?: number;
  conformance?: string;
  issues_remaining?: number;
}

export type AppPhase = "upload" | "progress" | "results" | "error";

export function useRemediation() {
  const [phase, setPhase] = useState<AppPhase>("upload");
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [progress, setProgress] = useState<ProgressEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploadInfo, setUploadInfo] = useState<UploadResult | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const upload = useCallback(async (file: File, verificationMode: "sampled" | "full") => {
    setError(null);
    setProgress([]);
    setJob(null);

    try {
      const result = await uploadPdf(file, verificationMode);
      setUploadInfo(result);
      setJobId(result.job_id);
      setPhase("progress");

      // Connect to progress WebSocket
      const ws = new WebSocket(
        withApiKeyQuery(`${backendWebSocketOrigin()}/api/ws/progress/${result.job_id}`),
      );
      wsRef.current = ws;

      const startPolling = () => {
        const poll = setInterval(async () => {
          try {
            const j = await getJob(result.job_id);
            if (j.status === "completed") {
              setJob(j);
              setPhase("results");
              clearInterval(poll);
            } else if (j.status === "failed") {
              setError(j.error || "Remediation failed");
              setPhase("error");
              clearInterval(poll);
            }
          } catch { /* retry */ }
        }, 3000);
      };

      ws.onmessage = (evt) => {
        const event: ProgressEvent = JSON.parse(evt.data);
        setProgress((prev) => [...prev, event]);

        if (event.type === "complete") {
          getJob(result.job_id).then((j) => {
            setJob(j);
            setPhase("results");
          });
          ws.close();
        } else if (event.type === "error") {
          // WS-side error may just be an idle timeout while the backend job
          // keeps running. Verify against the actual job status before giving up.
          ws.close();
          getJob(result.job_id).then((j) => {
            if (j.status === "failed") {
              setError(j.error || event.message || "Remediation failed");
              setPhase("error");
            } else if (j.status === "completed") {
              setJob(j);
              setPhase("results");
            } else {
              startPolling();
            }
          }).catch(() => startPolling());
        }
      };

      ws.onerror = () => {
        startPolling();
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setPhase("error");
    }
  }, []);

  const reset = useCallback(() => {
    wsRef.current?.close();
    setPhase("upload");
    setJobId(null);
    setJob(null);
    setProgress([]);
    setError(null);
    setUploadInfo(null);
  }, []);

  return { phase, jobId, job, progress, error, uploadInfo, upload, reset };
}
