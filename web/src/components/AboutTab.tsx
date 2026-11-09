import { API_URL } from "@/lib/api";

export function AboutTab({ trackingDisclaimer }: { trackingDisclaimer: string }) {
  return (
    <div className="max-w-3xl space-y-4 text-sm leading-relaxed">
      <p>
        <strong>Pipeline:</strong> video → OpenCV reads one frame at a time → YOLO detects objects (restricted to the
        selected classes) → optional ByteTrack assigns IDs → boxes, labels and confidences are drawn → frames are written
        to an MP4 → the file is re-encoded to H.264 so browsers can play it → statistics and CSV reports are produced.
      </p>
      <p>
        <strong>Classes (COCO IDs):</strong> car (2), motorcycle (3), bus (5), truck (7).
      </p>
      <p>
        <strong>Detections vs. unique vehicles:</strong> detection counts are summed over every frame. Unique counts are
        only available with tracking enabled and are estimates. {trackingDisclaimer}
      </p>
      <p>
        <strong>Architecture:</strong> this Next.js frontend talks to the FastAPI backend at{" "}
        <code className="rounded bg-panel-muted px-1.5 py-0.5">{API_URL}</code> (API docs at{" "}
        <a className="text-accent underline" href={`${API_URL}/docs`} target="_blank" rel="noreferrer">
          /docs
        </a>
        ). Jobs run one at a time on the API; you can leave the page and come back while a video is processing.
      </p>
      <p>
        <strong>Files:</strong> uploads are stored in <code>api/input_videos/</code>, annotated videos in{" "}
        <code>api/output_videos/</code>, CSV reports in <code>api/reports/</code>, and model weights in{" "}
        <code>api/models/</code>.
      </p>
    </div>
  );
}
