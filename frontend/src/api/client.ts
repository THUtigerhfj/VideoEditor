export type Mode = "prompt" | "image" | "sketch";
export type PointLabel = "Positive" | "Negative";
export type JobState = "queued" | "running" | "succeeded" | "failed";

export interface SessionPayload {
  session_id: string;
  video_info: Record<string, unknown>;
  points: Array<{ x: number; y: number; label: PointLabel }>;
  has_tracked_mask: boolean;
  has_reference: boolean;
  latest_output_url: string | null;
}

export interface JobPayload {
  job_id: string;
  session_id: string;
  type: string;
  state: JobState;
  stage: string;
  progress: number;
  message: string;
  result: Record<string, unknown>;
  error: string;
}

export interface UploadVideoPayload {
  session_id: string;
  video_info: Record<string, unknown>;
  frame_url: string;
}

export interface PointPayload {
  overlay_url?: string;
  frame_url?: string;
  preview_url?: string;
  points?: Array<{ x: number; y: number; label: PointLabel }>;
  has_reference_mask?: boolean;
}

function defaultApiBase() {
  if (typeof window === "undefined") return "http://127.0.0.1:7862";
  if (window.location.port && window.location.port !== "5173") return window.location.origin;
  return `${window.location.protocol}//${window.location.hostname}:7862`;
}

export const apiBase = (import.meta.env.VITE_API_BASE || defaultApiBase()).replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(String(body.detail || response.statusText));
  }
  return response.json() as Promise<T>;
}

export function assetUrl(path?: unknown): string {
  if (!path || typeof path !== "string") return "";
  if (path.startsWith("http")) return path;
  return `${apiBase}${path}`;
}

export const api = {
  createSession: () => request<SessionPayload>("/api/sessions", { method: "POST" }),
  getSession: (sessionId: string) => request<SessionPayload>(`/api/sessions/${sessionId}`),
  uploadVideo: (sessionId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadVideoPayload>(`/api/sessions/${sessionId}/video`, {
      method: "POST",
      body: form,
    });
  },
  addTargetPoint: (sessionId: string, x: number, y: number, label: PointLabel) =>
    request<PointPayload>(`/api/sessions/${sessionId}/target-points`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, label }),
    }),
  clearTargetPoints: (sessionId: string) =>
    request<PointPayload>(`/api/sessions/${sessionId}/target-points`, { method: "DELETE" }),
  startTrack: (sessionId: string) => request<JobPayload>(`/api/sessions/${sessionId}/track`, { method: "POST" }),
  uploadReference: (sessionId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ reference_url: string }>(`/api/sessions/${sessionId}/reference-image`, {
      method: "POST",
      body: form,
    });
  },
  addReferencePoint: (sessionId: string, x: number, y: number, label: PointLabel) =>
    request<PointPayload>(`/api/sessions/${sessionId}/reference-points`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, label }),
    }),
  generatePrompt: (sessionId: string, body: Record<string, unknown>) =>
    request<JobPayload>(`/api/sessions/${sessionId}/generate/prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  generateImage: (sessionId: string, body: Record<string, unknown>) =>
    request<JobPayload>(`/api/sessions/${sessionId}/generate/image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  generateSketch: (sessionId: string, body: Record<string, unknown>) =>
    request<JobPayload>(`/api/sessions/${sessionId}/generate/sketch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  getJob: (jobId: string) => request<JobPayload>(`/api/jobs/${jobId}`),
};
