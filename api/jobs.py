"""Background processing jobs for the REST API.

Each request to process a video or a live stream becomes a ``Job``. Jobs run
one at a time on a single worker thread: the YOLO model (and ByteTrack state
stored on it) is cached and shared, so running jobs concurrently would mix
tracker state between videos. Progress, the latest annotated preview frame and
the final ``ProcessingResult`` are kept in memory for the frontend to poll.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from detector import ModelLoadError, VehicleDetector
from utils import InsufficientDiskSpaceError, VideoValidationError, format_seconds
from video_processor import ProcessingConfig, ProcessingProgress, ProcessingResult, VideoProcessor

logger = logging.getLogger("vehicle_detection")

JobKind = Literal["video", "live"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]

PREVIEW_MAX_WIDTH = 960
PREVIEW_JPEG_QUALITY = 80
LIVE_PREVIEW_INTERVAL_S = 0.08  # ~12 preview frames per second for live streams
MAX_CACHED_DETECTORS = 3


class JobCancelled(Exception):
    """Raised from the progress callback to abort a video run."""


@dataclass
class DetectorSettings:
    model_name: str
    device: str
    imgsz: int


@dataclass
class Job:
    id: str
    kind: JobKind
    source_label: str
    detector_settings: DetectorSettings
    class_names: list[str] | None  # None = all COCO classes
    config: ProcessingConfig
    # video jobs
    video_path: Path | None = None
    # live jobs
    stream_source: int | str | None = None
    max_seconds: float = 60.0
    save_output: bool = True

    status: JobStatus = "queued"
    stage: str = "Queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    frames_processed: int = 0
    total_frames: int = 0
    elapsed_s: float = 0.0
    vehicle_detections: int = 0
    error: str | None = None
    result: ProcessingResult | None = None
    summary_report_path: str | None = None
    preview_jpeg: bytes | None = None
    preview_seq: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _last_preview_at: float = 0.0

    @property
    def finished(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    @property
    def fps(self) -> float:
        return self.frames_processed / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def progress_dict(self) -> dict:
        fraction = min(self.frames_processed / self.total_frames, 1.0) if self.total_frames > 0 else None
        eta = (self.total_frames - self.frames_processed) / self.fps if self.total_frames > 0 and self.fps > 0 else None
        return {
            "frames_processed": self.frames_processed,
            "total_frames": self.total_frames,
            "fraction": fraction,
            "elapsed_s": round(self.elapsed_s, 2),
            "fps": round(self.fps, 2),
            "eta_s": round(eta, 1) if eta is not None else None,
            "vehicle_detections": self.vehicle_detections,
        }

    def set_preview(self, frame_bgr: np.ndarray) -> None:
        height, width = frame_bgr.shape[:2]
        if width > PREVIEW_MAX_WIDTH:
            frame_bgr = cv2.resize(frame_bgr, (PREVIEW_MAX_WIDTH, int(height * PREVIEW_MAX_WIDTH / width)))
        ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
        if ok:
            self.preview_jpeg = buffer.tobytes()
            self.preview_seq += 1


class JobManager:
    """Owns the job registry, the single worker thread and the detector cache."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        # Only touched from the worker thread, so it needs no lock.
        self._detectors: OrderedDict[tuple[str, str, int], VehicleDetector] = OrderedDict()

    # ------------------------------------------------------------------ #
    # Registry
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Create the worker thread pool (called on app startup; safe to call again after shutdown)."""
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detection-worker")

    def submit(self, job: Job) -> Job:
        self.start()
        with self._lock:
            self._jobs[job.id] = job
            executor = self._executor
        assert executor is not None
        executor.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job: Job) -> None:
        job.cancel_event.set()
        if job.status == "queued":
            job.status, job.stage = "cancelled", "Cancelled before start"
            job.finished_at = time.time()
        elif job.status == "running":
            job.stage = "Finalising recording…" if job.kind == "live" else "Cancelling…"

    def shutdown(self) -> None:
        for job in self.list():
            job.cancel_event.set()
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #
    def _detector(self, settings: DetectorSettings) -> VehicleDetector:
        """Load the YOLO model once per (model, device, image size) and reuse it across jobs."""
        key = (settings.model_name, settings.device, settings.imgsz)
        if key in self._detectors:
            self._detectors.move_to_end(key)
            return self._detectors[key]
        detector = VehicleDetector(model_name=settings.model_name, device=settings.device, imgsz=settings.imgsz)
        self._detectors[key] = detector
        while len(self._detectors) > MAX_CACHED_DETECTORS:
            self._detectors.popitem(last=False)
        return detector

    def _run(self, job: Job) -> None:
        if job.cancel_event.is_set():
            return
        job.status, job.started_at = "running", time.time()
        try:
            job.stage = f"Loading {Path(job.detector_settings.model_name).name} (downloads on first use)…"
            detector = self._detector(job.detector_settings)
            if job.class_names is not None:
                job.config.class_ids = detector.class_ids_for(job.class_names)
                if not job.config.class_ids:
                    raise ModelLoadError("This model contains none of the selected vehicle classes.")
            if job.cancel_event.is_set():
                raise JobCancelled
            processor = VideoProcessor(detector)
            job.result = self._run_video(job, processor) if job.kind == "video" else self._run_live(job, processor)
            job.frames_processed = job.result.frames_processed
            job.elapsed_s = job.result.processing_time_s
            job.status = "completed"
            job.stage = (
                f"Done - {job.result.frames_processed:,} frames in {format_seconds(job.result.processing_time_s)} "
                f"({job.result.average_fps:.1f} FPS)"
            )
        except JobCancelled:
            job.status, job.stage = "cancelled", "Cancelled"
        except (VideoValidationError, InsufficientDiskSpaceError, ModelLoadError, ValueError) as exc:
            job.status, job.stage, job.error = "failed", "Processing failed", str(exc)
        except Exception as exc:  # unexpected (e.g. broken torch install, CUDA out of memory)
            logger.exception("Job %s failed", job.id)
            job.status, job.stage = "failed", "Processing failed"
            job.error = (
                f"Unexpected error during processing: {exc}. If you are using a GPU and ran out of memory, "
                "choose a smaller model/image size or the CPU."
            )
        finally:
            job.finished_at = time.time()

    def _run_video(self, job: Job, processor: VideoProcessor) -> ProcessingResult:
        assert job.video_path is not None
        job.stage = "Processing video…"

        def on_progress(p: ProcessingProgress) -> None:
            if job.cancel_event.is_set():
                raise JobCancelled  # process_video's finally-block releases handles and removes the partial file
            job.frames_processed, job.total_frames = p.frames_processed, p.total_frames
            job.elapsed_s, job.vehicle_detections = p.elapsed_s, p.vehicle_detections
            if p.preview_bgr is not None:
                job.set_preview(p.preview_bgr)
            if p.total_frames and p.frames_processed >= p.total_frames:
                job.stage = "Encoding H.264 video…"

        return processor.process_video(job.video_path, job.config, progress_callback=on_progress)

    def _run_live(self, job: Job, processor: VideoProcessor) -> ProcessingResult:
        assert job.stream_source is not None
        job.stage = f"Opening camera/stream {job.stream_source}…"

        def on_frame(frame: np.ndarray, p: ProcessingProgress) -> None:
            if job.stage.startswith("Opening"):
                job.stage = "Live detection running…"
            job.frames_processed, job.elapsed_s, job.vehicle_detections = p.frames_processed, p.elapsed_s, p.vehicle_detections
            now = time.perf_counter()
            if now - job._last_preview_at >= LIVE_PREVIEW_INTERVAL_S:
                job._last_preview_at = now
                job.set_preview(frame)

        # Stopping is graceful: process_stream checks should_stop between frames and still finalises the recording.
        result = processor.process_stream(
            job.stream_source,
            job.config,
            on_frame,
            should_stop=job.cancel_event.is_set,
            max_seconds=job.max_seconds,
            save_output=job.save_output,
        )
        if job.cancel_event.is_set():
            result.warnings.insert(0, "Live session stopped by the user.")
        return result
