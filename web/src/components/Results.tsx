"use client";

import { useState } from "react";
import { api, urls, type Job, type JobResult, type Row, type VehicleClass } from "@/lib/api";
import { CLASS_COLOR, CLASS_EMOJI, VEHICLE_CLASSES, capitalize, formatInt, formatSeconds } from "@/lib/format";
import { Alert, Button, Card, DataTable, Expander, LinkButton, Metric } from "./ui";

export function Results({ job, trackingDisclaimer }: { job: Job; trackingDisclaimer: string }) {
  const result = job.result as JobResult;
  const [records, setRecords] = useState<Row[] | null>(null);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const [reportPath, setReportPath] = useState<string | null>(result.summary_report_path);
  const [savingReport, setSavingReport] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);

  const counts = result.detection_counts;
  const unique = result.unique_counts;

  const loadRecords = async () => {
    if (records || !result.has_frame_records) return;
    try {
      setRecords(await api.records(job.id, 500));
    } catch (e) {
      setRecordsError((e as Error).message);
    }
  };

  const saveReport = async () => {
    setSavingReport(true);
    setReportError(null);
    try {
      setReportPath((await api.saveReport(job.id)).path);
    } catch (e) {
      setReportError((e as Error).message);
    } finally {
      setSavingReport(false);
    }
  };

  return (
    <div className="space-y-5">
      <h2 className="font-display text-lg font-bold uppercase tracking-[0.14em]">
        <span className="text-gradient">Detection</span> results
      </h2>

      {result.warnings.map((w) => (
        <Alert key={w} tone="warning">
          {w}
        </Alert>
      ))}

      {result.output_available &&
        (result.browser_compatible ? (
          <video
            key={job.id}
            src={urls.video(job.id)}
            controls
            className="glow mx-auto max-h-[70vh] w-full rounded-2xl border border-border bg-black"
          />
        ) : (
          <Alert>The browser preview may not play this mp4v file; download it and open it in VLC or another player.</Alert>
        ))}

      <Card title="Detections across frames">
        <p className="mb-3 text-sm text-muted">
          These are <strong>frame-level detections</strong>: a vehicle visible for 100 frames counts 100 times. They are
          not counts of unique vehicles.
        </p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          <Metric label="Total vehicle detections" value={formatInt(result.total_vehicle_detections)} />
          {VEHICLE_CLASSES.map((name) => (
            <Metric
              key={name}
              label={`${CLASS_EMOJI[name]} ${capitalize(name)} detections`}
              value={formatInt(counts[name] ?? 0)}
              accent={CLASS_COLOR[name]}
            />
          ))}
        </div>
        {result.all_classes_mode && result.total_detections !== result.total_vehicle_detections && (
          <p className="mt-3 text-sm text-muted">
            All-classes mode: {formatInt(result.total_detections)} detections in total including non-vehicle objects.
          </p>
        )}
      </Card>

      {unique && (
        <Card title="Unique tracked vehicles (estimate)">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            <Metric label="Unique vehicle track IDs" value={formatInt(result.unique_vehicle_total ?? 0)} />
            {VEHICLE_CLASSES.map((name: VehicleClass) => (
              <Metric
                key={name}
                label={`${CLASS_EMOJI[name]} Unique ${name}s`}
                value={formatInt(unique[name] ?? 0)}
                accent={CLASS_COLOR[name]}
              />
            ))}
          </div>
          <div className="mt-3">
            <Alert>{trackingDisclaimer}</Alert>
          </div>
        </Card>
      )}

      <Card title="Performance">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Metric label="Frames processed" value={formatInt(result.frames_processed)} />
          <Metric label="Average processing FPS" value={result.average_fps.toFixed(1)} />
          <Metric label="Processing time" value={formatSeconds(result.processing_time_s)} />
          <Metric label="Device" value={<span className="text-base">{result.device}</span>} />
        </div>
      </Card>

      <Card title="Detection summary">
        <DataTable rows={result.summary_table} />
      </Card>

      <Card title="Downloads">
        <div className="flex flex-wrap gap-2">
          {result.output_available && <LinkButton href={urls.video(job.id, true)}>⬇️ Annotated video</LinkButton>}
          <LinkButton href={urls.summaryCsv(job.id)}>⬇️ Summary CSV</LinkButton>
          {result.has_frame_records && <LinkButton href={urls.recordsCsv(job.id)}>⬇️ Frame-level records CSV</LinkButton>}
          {result.has_track_table && <LinkButton href={urls.tracksCsv(job.id)}>⬇️ Track table CSV</LinkButton>}
        </div>
        <div className="mt-4 space-y-2">
          <Button onClick={saveReport} disabled={savingReport}>
            💾 {savingReport ? "Saving…" : "Save summary report to reports/ folder"}
          </Button>
          {reportPath && <Alert tone="success">Saved {reportPath}</Alert>}
          {reportError && <Alert tone="error">{reportError}</Alert>}
        </div>
      </Card>

      <div className="space-y-2">
        <Expander title="Run details">
          <DataTable rows={result.run_metadata} />
          {result.output_name && <p className="mt-2 text-xs text-muted">Output video: {result.output_name}</p>}
        </Expander>
        {result.has_frame_records && (
          <Expander title="Frame-level detection records (first 500 rows)" onOpen={loadRecords}>
            {recordsError ? (
              <Alert tone="error">{recordsError}</Alert>
            ) : records ? (
              <DataTable rows={records} />
            ) : (
              <p className="text-sm text-muted">Loading…</p>
            )}
          </Expander>
        )}
        {result.has_track_table && result.track_table && (
          <Expander title="Tracked objects">
            <DataTable rows={result.track_table} />
          </Expander>
        )}
      </div>
    </div>
  );
}
