"use client";

import { urls, type Job } from "@/lib/api";
import { formatInt, formatSeconds } from "@/lib/format";
import { Button, ProgressBar } from "./ui";

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-white px-3 py-2">
      <div className="label-tech">{label}</div>
      <div className="font-mono text-lg font-semibold tabular-nums text-foreground">{value}</div>
    </div>
  );
}

/** Progress bar, live MJPEG preview and cancel/stop button for a queued or running job. */
export function JobMonitor({ job, onCancel, cancelling }: { job: Job; onCancel: () => void; cancelling: boolean }) {
  const p = job.progress;
  const live = job.kind === "live";
  const queued = job.status === "queued";
  const fraction = live ? (job.max_seconds ? Math.min(p.elapsed_s / job.max_seconds, 1) : null) : p.fraction;
  const percent = fraction !== null && !queued ? `${Math.round(fraction * 100)}%` : "--";

  const readouts = [
    { label: "Frames", value: p.total_frames > 0 && !live ? `${formatInt(p.frames_processed)}/${formatInt(p.total_frames)}` : formatInt(p.frames_processed) },
    { label: "Speed", value: `${p.fps.toFixed(1)} FPS` },
    live
      ? { label: "Elapsed", value: `${Math.floor(p.elapsed_s)}s / ${job.max_seconds}s` }
      : { label: "Remaining", value: p.eta_s !== null ? `~${formatSeconds(p.eta_s)}` : "--" },
    { label: "Vehicle detections", value: formatInt(p.vehicle_detections) },
  ];

  return (
    <div className="glass hud glow space-y-4 rounded-2xl p-4 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="label-tech flex items-center gap-2">
            <span className="h-2 w-2 animate-pulse rounded-full bg-accent shadow-[0_0_8px_var(--accent)]" aria-hidden />
            {queued ? "Queued" : live ? "Live session" : "Processing"}
          </div>
          <div className="mt-0.5 truncate font-semibold">{queued ? "Waiting for the previous job to finish…" : job.stage}</div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-gradient font-display text-2xl font-bold tabular-nums">{percent}</span>
          <Button variant="danger" onClick={onCancel} disabled={cancelling}>
            {live ? "⏹ Stop" : "✖ Cancel"}
          </Button>
        </div>
      </div>

      <ProgressBar fraction={job.status === "running" && p.frames_processed > 0 ? fraction : null} />

      <div className="grid grid-cols-2 gap-2.5 lg:grid-cols-4">
        {readouts.map((r) => (
          <Readout key={r.label} {...r} />
        ))}
      </div>

      {job.status === "running" && (
        <div className="relative overflow-hidden rounded-xl border border-border bg-[#05080f]">
          {/* eslint-disable-next-line @next/next/no-img-element -- MJPEG stream, not a static image */}
          <img
            src={urls.stream(job.id)}
            alt="Live annotated preview"
            className="mx-auto max-h-[60vh] w-full object-contain"
          />
          <div className="scanline pointer-events-none absolute inset-x-0 top-0 h-[10%] bg-linear-to-b from-transparent via-cyan-300/15 to-transparent" aria-hidden />
          <div className="pointer-events-none absolute inset-3 rounded-lg border border-cyan-300/25" aria-hidden />
          <span className="absolute left-4 top-4 inline-flex items-center gap-1.5 rounded-md bg-black/60 px-2 py-1 font-mono text-[0.68rem] tracking-widest text-white backdrop-blur">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-500" aria-hidden />
            {live ? "LIVE" : "REC"}
          </span>
        </div>
      )}
    </div>
  );
}
