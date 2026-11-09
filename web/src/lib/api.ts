// Typed client for the FastAPI backend in ../api (see api/main.py).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export type VehicleClass = "car" | "motorcycle" | "bus" | "truck";

export interface AppConfig {
  models: string[];
  default_model: string;
  devices: string[];
  cuda_available: boolean;
  auto_device: string;
  image_sizes: number[];
  defaults: { confidence: number; iou: number; imgsz: number };
  vehicle_classes: { id: number; name: VehicleClass }[];
  tracking: { available: boolean; message: string; disclaimer: string };
  supported_extensions: string[];
}

export interface DetectionSettings {
  model_name: string;
  device: "auto" | "cpu" | "cuda";
  imgsz: number;
  confidence: number;
  iou: number;
  all_classes: boolean;
  classes: VehicleClass[];
  enable_tracking: boolean;
  save_records: boolean;
  max_frames: number;
}

export interface VideoInfo {
  path: string;
  name: string;
  width: number;
  height: number;
  resolution: string;
  fps: number;
  frame_count: number;
  duration_s: number;
}

export interface VideoRef {
  video_id: string;
  original_name: string;
  info: VideoInfo;
}

export type Row = Record<string, string | number | null>;

export interface JobResult {
  input_name: string;
  output_name: string | null;
  output_available: boolean;
  browser_compatible: boolean;
  stopped_early: boolean;
  warnings: string[];
  video_info: VideoInfo;
  model_name: string;
  device: string;
  frames_processed: number;
  processing_time_s: number;
  average_fps: number;
  frames_with_detections: number;
  all_classes_mode: boolean;
  tracking: boolean;
  detection_counts: Record<string, number>;
  total_detections: number;
  total_vehicle_detections: number;
  unique_counts: Record<string, number> | null;
  unique_vehicle_total: number | null;
  summary_table: Row[];
  run_metadata: Row[];
  track_table: Row[] | null;
  has_frame_records: boolean;
  has_track_table: boolean;
  summary_report_path: string | null;
}

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface Job {
  id: string;
  kind: "video" | "live";
  source: string;
  status: JobStatus;
  stage: string;
  error: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  progress: {
    frames_processed: number;
    total_frames: number;
    fraction: number | null;
    elapsed_s: number;
    fps: number;
    eta_s: number | null;
    vehicle_detections: number;
  };
  preview_seq: number;
  max_seconds: number | null;
  result: JobResult | null;
}

export const isFinished = (job: Job) => ["completed", "failed", "cancelled"].includes(job.status);

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      // FastAPI validation errors
      return body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
    }
  } catch {
    /* not JSON */
  }
  return `${res.status} ${res.statusText}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(`Cannot reach the API at ${API_URL}. Is the backend running?`, 0);
  }
  if (!res.ok) throw new ApiError(await errorMessage(res), res.status);
  return res.json() as Promise<T>;
}

const postJson = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const api = {
  config: () => request<AppConfig>("/api/config"),
  registerLocalVideo: (path: string) => postJson<VideoRef>("/api/videos/local", { path }),
  getVideo: (videoId: string) => request<VideoRef>(`/api/videos/${encodeURIComponent(videoId)}`),
  startVideoJob: (videoId: string, settings: DetectionSettings) =>
    postJson<Job>("/api/jobs/video", { video_id: videoId, settings }),
  startLiveJob: (body: { source: number | string; max_seconds: number; save_output: boolean; settings: DetectionSettings }) =>
    postJson<Job>("/api/jobs/live", body),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  cancelJob: (id: string) => postJson<Job>(`/api/jobs/${id}/cancel`),
  records: (id: string, limit = 500) => request<Row[]>(`/api/jobs/${id}/records?limit=${limit}`),
  saveReport: (id: string) => postJson<{ path: string }>(`/api/jobs/${id}/report`),

  /** Upload with progress reporting (fetch has no upload progress events). */
  uploadVideo(file: File, onProgress: (fraction: number) => void): Promise<VideoRef> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const form = new FormData();
      form.append("file", file);
      xhr.open("POST", `${API_URL}/api/videos`);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onerror = () => reject(new ApiError(`Cannot reach the API at ${API_URL}. Is the backend running?`, 0));
      xhr.onload = () => {
        let body: { detail?: unknown } & Partial<VideoRef> = {};
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* ignore */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body as VideoRef);
        else reject(new ApiError(typeof body.detail === "string" ? body.detail : `Upload failed (${xhr.status})`, xhr.status));
      };
      xhr.send(form);
    });
  },
};

export const urls = {
  stream: (id: string) => `${API_URL}/api/jobs/${id}/stream`,
  video: (id: string, download = false) => `${API_URL}/api/jobs/${id}/video${download ? "?download=true" : ""}`,
  summaryCsv: (id: string) => `${API_URL}/api/jobs/${id}/summary.csv`,
  recordsCsv: (id: string) => `${API_URL}/api/jobs/${id}/records.csv`,
  tracksCsv: (id: string) => `${API_URL}/api/jobs/${id}/tracks.csv`,
};
