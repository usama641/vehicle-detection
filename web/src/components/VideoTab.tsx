"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useStoredState } from "@/hooks/useStoredState";
import { api, type AppConfig, type DetectionSettings, type VideoRef } from "@/lib/api";
import { formatInt, formatSeconds } from "@/lib/format";
import { JobSection } from "./JobSection";
import { Alert, Button, Metric, ProgressBar, Segmented, inputClass } from "./ui";

type Source = "upload" | "local";

export function VideoTab({
  config,
  settings,
  onRunningChange,
}: {
  config: AppConfig;
  settings: DetectionSettings;
  onRunningChange: (running: boolean) => void;
}) {
  const [source, setSource] = useState<Source>("upload");
  const [video, setVideo] = useStoredState<VideoRef | null>("vd.video", null);
  const [jobId, setJobId] = useStoredState<string | null>("vd.videoJob", null);
  const [localPath, setLocalPath] = useState("");
  const [uploadFraction, setUploadFraction] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const accept = config.supported_extensions.join(",");

  const handleRunning = useCallback(
    (r: boolean) => {
      setRunning(r);
      onRunningChange(r);
    },
    [onRunningChange],
  );

  // Re-validate a remembered video (it may have been deleted since).
  useEffect(() => {
    if (!video) return;
    api.getVideo(video.video_id).catch(() => setVideo(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only on first load
  }, []);

  const upload = async (file: File) => {
    setError(null);
    setVideo(null);
    setUploadFraction(0);
    try {
      setVideo(await api.uploadVideo(file, setUploadFraction));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploadFraction(null);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const loadLocal = async () => {
    setError(null);
    setBusy(true);
    try {
      setVideo(await api.registerLocalVideo(localPath));
    } catch (e) {
      setVideo(null);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    if (!video) return;
    setError(null);
    setBusy(true);
    try {
      const job = await api.startVideoJob(video.video_id, settings);
      setJobId(job.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const info = video?.info;

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <Segmented
          value={source}
          onChange={setSource}
          disabled={running}
          options={[
            { value: "upload", label: "Upload a video" },
            { value: "local", label: "Video file on the API machine" },
          ]}
        />

        {source === "upload" ? (
          <label
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 glass rounded-2xl border-2 border-dashed border-accent/30 px-4 py-10 text-center transition hover:border-accent hover:glow ${
              running || uploadFraction !== null ? "pointer-events-none opacity-60" : ""
            }`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const file = e.dataTransfer.files?.[0];
              if (file) upload(file);
            }}
          >
            <span className="grid h-14 w-14 place-items-center rounded-2xl bg-gradient-accent text-2xl text-white shadow-[0_8px_24px_-8px_rgba(124,58,237,0.6)]" aria-hidden>
              ⬆
            </span>
            <span className="text-base font-semibold">Drop a video here or <span className="text-gradient">browse files</span></span>
            <span className="label-tech">
              {config.supported_extensions.map((e) => e.slice(1).toUpperCase()).join(", ")}
            </span>
            <input
              ref={fileInput}
              type="file"
              accept={accept}
              className="sr-only"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) upload(file);
              }}
            />
          </label>
        ) : (
          <div className="flex flex-col gap-2 sm:flex-row">
            <input
              className={inputClass}
              placeholder="C:\Videos\traffic.mp4  or  /home/me/traffic.mp4"
              value={localPath}
              disabled={running}
              onChange={(e) => setLocalPath(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && localPath.trim() && loadLocal()}
            />
            <Button onClick={loadLocal} disabled={!localPath.trim() || busy || running}>
              Load
            </Button>
          </div>
        )}

        {uploadFraction !== null && (
          <div className="space-y-1">
            <ProgressBar fraction={uploadFraction} />
            <p className="text-sm text-muted">
              {uploadFraction < 1 ? `Uploading… ${Math.round(uploadFraction * 100)}%` : "Validating video…"}
            </p>
          </div>
        )}
      </div>

      {error && <Alert tone="error">{error}</Alert>}

      {video && info && (
        <div className="space-y-3">
          <p className="text-sm">
            <span className="text-muted">Selected:</span> <strong>{video.original_name}</strong>
          </p>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Metric label="Resolution" value={info.resolution} />
            <Metric label="Frame rate" value={`${info.fps.toFixed(2)} fps`} />
            <Metric label="Frames" value={info.frame_count ? formatInt(info.frame_count) : "unknown"} />
            <Metric label="Duration" value={info.duration_s ? formatSeconds(info.duration_s) : "unknown"} />
          </div>
        </div>
      )}

      <Button variant="primary" onClick={start} disabled={!video || busy || running || uploadFraction !== null}>
        ▶️ Start Detection
      </Button>

      {jobId && (
        <>
          <hr className="border-border" />
          <JobSection
            key={jobId}
            jobId={jobId}
            trackingDisclaimer={config.tracking.disclaimer}
            onClear={() => setJobId(null)}
            onRunningChange={handleRunning}
          />
        </>
      )}
    </div>
  );
}
