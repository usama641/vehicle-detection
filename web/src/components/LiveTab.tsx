"use client";

import { useCallback, useState } from "react";
import { useStoredState } from "@/hooks/useStoredState";
import { api, type AppConfig, type DetectionSettings } from "@/lib/api";
import { JobSection } from "./JobSection";
import { Alert, Button, Field, Segmented, Toggle, inputClass } from "./ui";

export function LiveTab({
  config,
  settings,
  onRunningChange,
}: {
  config: AppConfig;
  settings: DetectionSettings;
  onRunningChange: (running: boolean) => void;
}) {
  const [kind, setKind] = useState<"webcam" | "url">("webcam");
  const [cameraIndex, setCameraIndex] = useState(0);
  const [url, setUrl] = useState("");
  const [maxSeconds, setMaxSeconds] = useState(60);
  const [saveOutput, setSaveOutput] = useState(true);
  const [jobId, setJobId] = useStoredState<string | null>("vd.liveJob", null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const handleRunning = useCallback(
    (r: boolean) => {
      setRunning(r);
      onRunningChange(r);
    },
    [onRunningChange],
  );

  const start = async () => {
    setError(null);
    setBusy(true);
    try {
      const job = await api.startLiveJob({
        source: kind === "webcam" ? cameraIndex : url.trim(),
        max_seconds: maxSeconds,
        save_output: saveOutput,
        settings,
      });
      setJobId(job.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <p className="text-sm text-muted">
        Run detection on a webcam or network stream (RTSP/HTTP). The camera must be attached to the machine running the
        API, not the browser&apos;s machine. Click <strong>Stop</strong> (or wait for the time limit) to finish; the
        recording is finalised either way.
      </p>

      <fieldset disabled={running} className="grid gap-5 sm:grid-cols-2">
        <Field label="Source">
          <Segmented
            value={kind}
            onChange={setKind}
            disabled={running}
            options={[
              { value: "webcam", label: "Webcam" },
              { value: "url", label: "Stream URL" },
            ]}
          />
        </Field>
        {kind === "webcam" ? (
          <Field label="Camera index" hint="Usually 0; try 1 or 2 if you have several cameras.">
            <input
              type="number"
              min={0}
              max={10}
              className={inputClass}
              value={cameraIndex}
              onChange={(e) => setCameraIndex(Math.min(10, Math.max(0, Math.floor(Number(e.target.value) || 0))))}
            />
          </Field>
        ) : (
          <Field label="Stream URL">
            <input
              className={inputClass}
              placeholder="rtsp://user:pass@192.168.1.10:554/stream"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
          </Field>
        )}
        <Field label={`Time limit: ${maxSeconds} seconds`}>
          <input
            type="range"
            className="w-full"
            min={10}
            max={600}
            step={10}
            value={maxSeconds}
            onChange={(e) => setMaxSeconds(Number(e.target.value))}
          />
        </Field>
        <div className="flex items-end">
          <Toggle label="Save annotated recording" checked={saveOutput} onChange={setSaveOutput} />
        </div>
      </fieldset>

      {error && <Alert tone="error">{error}</Alert>}

      <Button variant="primary" onClick={start} disabled={busy || running || (kind === "url" && !url.trim())}>
        ▶️ Start camera
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
