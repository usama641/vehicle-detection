"use client";

import { useEffect, useState } from "react";
import { useJob } from "@/hooks/useJob";
import { api, isFinished, type Job } from "@/lib/api";
import { formatInt } from "@/lib/format";
import { JobMonitor } from "./JobMonitor";
import { Results } from "./Results";
import { Alert, Button } from "./ui";

/** Follows one job from queued to finished and renders progress, errors and results. */
export function JobSection({
  jobId,
  trackingDisclaimer,
  onClear,
  onRunningChange,
}: {
  jobId: string;
  trackingDisclaimer: string;
  onClear: () => void;
  onRunningChange: (running: boolean) => void;
}) {
  const { job, setJob, error } = useJob(jobId);
  const [cancelling, setCancelling] = useState(false);
  const running = job !== null && !isFinished(job);

  useEffect(() => onRunningChange(running), [running, onRunningChange]);

  const cancel = async () => {
    setCancelling(true);
    try {
      setJob((await api.cancelJob(jobId)) as Job);
    } catch {
      /* the poller will surface connection problems */
    } finally {
      setCancelling(false);
    }
  };

  if (!job) {
    return error ? (
      <div className="space-y-3">
        <Alert tone="error">{error}</Alert>
        <Button onClick={onClear}>🗑️ Clear</Button>
      </div>
    ) : (
      <p className="text-sm text-muted">Loading job…</p>
    );
  }

  return (
    <div className="space-y-5">
      {error && <Alert tone="warning">Lost contact with the API: {error}. Retrying…</Alert>}
      {running && <JobMonitor job={job} onCancel={cancel} cancelling={cancelling} />}
      {job.status === "failed" && (
        <Alert tone="error">
          {job.error}
          {"\n"}
          <span className="text-muted">If you are using a GPU and ran out of memory, choose a smaller model/image size or the CPU.</span>
        </Alert>
      )}
      {job.status === "cancelled" && (
        <Alert tone="warning">
          Processing was cancelled after {formatInt(job.progress.frames_processed)} frames. No output was kept.
        </Alert>
      )}
      {job.status === "completed" && <Alert tone="success">{job.stage}</Alert>}
      {job.status === "completed" && job.result && <Results job={job} trackingDisclaimer={trackingDisclaimer} />}
      {!running && (
        <Button onClick={onClear} variant="secondary">
          🗑️ Clear results
        </Button>
      )}
    </div>
  );
}
