"""Frame-by-frame video processing: read -> detect -> draw -> write -> statistics.

Frames are streamed one at a time from ``cv2.VideoCapture`` to
``cv2.VideoWriter``, so memory use is independent of video length.
"""

from __future__ import annotations

import csv
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from detector import Detection, VehicleDetector, draw_detections, draw_frame_overlay
from reports import FRAME_RECORD_FIELDS
from tracker import TrackStatistics
from utils import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    OUTPUT_DIR,
    REPORTS_DIR,
    VEHICLE_CLASS_NAMES,
    VideoInfo,
    VideoValidationError,
    check_disk_space,
    describe_device,
    ensure_directories,
    estimate_output_bytes,
    probe_video,
    sanitize_fps,
    transcode_to_h264,
    unique_filename,
)

logger = logging.getLogger("vehicle_detection")

# Stop after this many consecutive unreadable frames (corrupted tail / broken stream).
MAX_CONSECUTIVE_READ_FAILURES = 10


@dataclass
class ProcessingConfig:
    """User-selected options for one processing run."""

    confidence: float = DEFAULT_CONFIDENCE
    iou: float = DEFAULT_IOU
    class_ids: list[int] | None = None  # None = all COCO classes
    enable_tracking: bool = False
    save_frame_records: bool = True
    max_frames: int = 0  # 0 = process the whole video
    draw_overlay: bool = True


@dataclass
class ProcessingProgress:
    """Snapshot passed to the progress callback."""

    frames_processed: int
    total_frames: int  # 0 when unknown
    elapsed_s: float
    vehicle_detections: int
    preview_bgr: np.ndarray | None = None

    @property
    def fraction(self) -> float:
        return min(self.frames_processed / self.total_frames, 1.0) if self.total_frames > 0 else 0.0


ProgressCallback = Callable[[ProcessingProgress], None]


@dataclass
class ProcessingResult:
    """Everything the UI and reports need after a run."""

    input_path: str
    output_path: str
    video_info: VideoInfo
    config: ProcessingConfig
    model_name: str
    device: str
    frames_processed: int
    processing_time_s: float
    detection_counts: dict[str, int]  # frame-level detections per class (NOT unique vehicles)
    frames_with_detections: int
    unique_counts: dict[str, int] | None = None  # unique track IDs per class (tracking only)
    track_table: pd.DataFrame | None = None
    frame_records_path: str | None = None
    browser_compatible: bool = True
    stopped_early: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def average_fps(self) -> float:
        return self.frames_processed / self.processing_time_s if self.processing_time_s > 0 else 0.0

    @property
    def total_detections(self) -> int:
        return sum(self.detection_counts.values())

    @property
    def total_vehicle_detections(self) -> int:
        return sum(self.detection_counts.get(name, 0) for name in VEHICLE_CLASS_NAMES)

    @property
    def unique_total(self) -> int | None:
        return None if self.unique_counts is None else sum(self.unique_counts.values())


class _RunState:
    """Mutable counters shared by file and stream processing."""

    def __init__(self, tracking: bool) -> None:
        self.counts: Counter[str] = Counter()
        self.frames = 0
        self.frames_with_detections = 0
        self.tracks = TrackStatistics() if tracking else None

    @property
    def vehicle_detections(self) -> int:
        return sum(self.counts.get(name, 0) for name in VEHICLE_CLASS_NAMES)

    def update(self, detections: list[Detection]) -> None:
        self.counts.update(det.class_name for det in detections)
        if detections:
            self.frames_with_detections += 1
        if self.tracks is not None:
            self.tracks.update(detections, self.frames)
        self.frames += 1


def stream_base_name(source: int | str) -> str:
    """Unique output base name for a live session, e.g. ``live_0_20261019_101500_ab12cd``."""
    return unique_filename(f"live_{source}", "")


class VideoProcessor:
    """Runs a ``VehicleDetector`` over video files or live streams."""

    def __init__(self, detector: VehicleDetector) -> None:
        self.detector = detector

    # ------------------------------------------------------------------ #
    # Shared per-frame step
    # ------------------------------------------------------------------ #
    def _process_frame(self, frame: np.ndarray, config: ProcessingConfig, state: _RunState) -> list[Detection]:
        """Detect, update statistics and annotate ``frame`` in place."""
        detections = self.detector.detect(
            frame, config.confidence, config.class_ids, iou=config.iou, track=config.enable_tracking
        )
        state.update(detections)
        draw_detections(frame, detections)
        if config.draw_overlay:
            vehicles_now = sum(1 for d in detections if d.class_name in VEHICLE_CLASS_NAMES)
            lines = [f"Frame {state.frames}", f"Vehicles in frame: {vehicles_now}"]
            if state.tracks is not None:
                lines.append(f"Unique track IDs so far: {state.tracks.unique_count}")
            draw_frame_overlay(frame, lines)
        return detections

    @staticmethod
    def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"OpenCV could not create the output video '{path.name}'. Check that the folder is writable."
            )
        return writer

    @staticmethod
    def _open_records(path: Path | None):
        if path is None:
            return None, None
        fh = open(path, "w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(FRAME_RECORD_FIELDS)
        return fh, writer

    @staticmethod
    def _write_records(writer, frame_index: int, fps: float, detections: list[Detection]) -> None:
        if writer is None:
            return
        timestamp = round(frame_index / fps, 3) if fps > 0 else ""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            writer.writerow(
                [frame_index, timestamp, det.class_id, det.class_name, round(det.confidence, 4),
                 x1, y1, x2, y2, "" if det.track_id is None else det.track_id]
            )

    def _finalize_output(self, raw_path: Path, final_path: Path, warnings: list[str]) -> bool:
        """Convert the mp4v intermediate to H.264; keep the mp4v file if that is impossible."""
        if transcode_to_h264(raw_path, final_path):
            raw_path.unlink(missing_ok=True)
            return True
        raw_path.replace(final_path)
        warnings.append(
            "Could not convert the output to H.264 (ffmpeg unavailable or failed). The video was saved in "
            "MPEG-4 (mp4v) format: it plays in VLC and most desktop players but may not play in the browser."
        )
        return False

    # ------------------------------------------------------------------ #
    # Video files
    # ------------------------------------------------------------------ #
    def process_video(
        self,
        input_path: str | Path,
        config: ProcessingConfig,
        output_dir: Path = OUTPUT_DIR,
        progress_callback: ProgressCallback | None = None,
        preview_every: int = 10,
    ) -> ProcessingResult:
        """Process a whole video file and return statistics.

        Raises:
            VideoValidationError: unreadable / unsupported input.
            InsufficientDiskSpaceError: not enough room for the output.
            RuntimeError: output video could not be created.
        """
        ensure_directories()
        input_path = Path(input_path)
        info = probe_video(input_path)
        check_disk_space(output_dir, estimate_output_bytes(input_path, info))

        base = unique_filename(input_path.stem + "_annotated", "")
        raw_path = output_dir / f"{base}_mp4v.mp4"
        final_path = output_dir / f"{base}.mp4"
        records_path = REPORTS_DIR / f"{base}_frames.csv" if config.save_frame_records else None

        total = info.frame_count
        if config.max_frames > 0:
            total = min(total, config.max_frames) if total > 0 else config.max_frames

        self.detector.prepare_run()

        state = _RunState(config.enable_tracking)
        warnings: list[str] = []
        cap = cv2.VideoCapture(str(input_path))
        writer = self._open_writer(raw_path, info.fps, (info.width, info.height))
        records_fh, records_writer = self._open_records(records_path)
        start = time.perf_counter()
        failures = 0
        completed = False
        try:
            while True:
                if config.max_frames > 0 and state.frames >= config.max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    # Skip past a few corrupt frames in the middle; otherwise this is the end of the file.
                    if info.frame_count and state.frames + failures < info.frame_count and failures < MAX_CONSECUTIVE_READ_FAILURES:
                        continue
                    break
                failures = 0
                if frame.shape[1] != info.width or frame.shape[0] != info.height:
                    frame = cv2.resize(frame, (info.width, info.height))

                frame_index = state.frames
                detections = self._process_frame(frame, config, state)
                self._write_records(records_writer, frame_index, info.fps, detections)
                writer.write(frame)

                if progress_callback is not None:
                    preview = frame if (state.frames == 1 or state.frames % preview_every == 0) else None
                    progress_callback(
                        ProcessingProgress(state.frames, total, time.perf_counter() - start, state.vehicle_detections, preview)
                    )
            completed = True
        finally:
            # Always release handles, even on errors or when Streamlit interrupts the run.
            cap.release()
            writer.release()
            if records_fh:
                records_fh.close()
            if not completed:
                raw_path.unlink(missing_ok=True)

        elapsed = time.perf_counter() - start
        if state.frames == 0:
            raw_path.unlink(missing_ok=True)
            if records_path:
                records_path.unlink(missing_ok=True)
            raise VideoValidationError("No frames could be decoded from the video.")

        stopped_early = bool(info.frame_count) and config.max_frames == 0 and state.frames < info.frame_count * 0.98
        if stopped_early:
            warnings.append(
                f"Only {state.frames} of the {info.frame_count} frames reported by the file could be read; "
                "the video may be truncated or partially corrupted."
            )

        browser_ok = self._finalize_output(raw_path, final_path, warnings)
        return self._build_result(
            input_path, final_path, info, config, state, elapsed, records_path, browser_ok, stopped_early, warnings
        )

    def _build_result(
        self,
        input_path: Path | str,
        output_path: Path,
        info: VideoInfo,
        config: ProcessingConfig,
        state: _RunState,
        elapsed: float,
        records_path: Path | None,
        browser_ok: bool,
        stopped_early: bool,
        warnings: list[str],
    ) -> ProcessingResult:
        return ProcessingResult(
            input_path=str(input_path),
            output_path=str(output_path),
            video_info=info,
            config=config,
            model_name=Path(self.detector.weights_path).name,
            device=describe_device(self.detector.device),
            frames_processed=state.frames,
            processing_time_s=elapsed,
            detection_counts=dict(state.counts),
            frames_with_detections=state.frames_with_detections,
            unique_counts=state.tracks.unique_counts_by_class() if state.tracks is not None else None,
            track_table=state.tracks.to_dataframe(info.fps) if state.tracks is not None else None,
            frame_records_path=str(records_path) if records_path else None,
            browser_compatible=browser_ok,
            stopped_early=stopped_early,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # Live camera / stream
    # ------------------------------------------------------------------ #
    def process_stream(
        self,
        source: int | str,
        config: ProcessingConfig,
        frame_callback: Callable[[np.ndarray, ProcessingProgress], None],
        should_stop: Callable[[], bool] = lambda: False,
        max_seconds: float = 60.0,
        save_output: bool = True,
        output_dir: Path = OUTPUT_DIR,
        base_name: str | None = None,
    ) -> ProcessingResult:
        """Process a webcam index (e.g. ``0``) or stream URL (RTSP/HTTP) until stopped or ``max_seconds`` elapse.

        ``frame_callback`` receives each annotated BGR frame for display.
        ``base_name`` fixes the output file names (``<base>_mp4v.mp4`` while
        recording, ``<base>.mp4`` when finished) so a caller can still find the
        recording if the run is interrupted before finalisation.
        """
        ensure_directories()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            raise VideoValidationError(
                f"Could not open camera/stream '{source}'. Check that the device is connected and not used by "
                "another application, or that the URL is correct."
            )
        ok, first = cap.read()
        if not ok or first is None:
            cap.release()
            raise VideoValidationError(f"Camera/stream '{source}' opened but returned no frames.")

        height, width = first.shape[:2]
        fps = sanitize_fps(cap.get(cv2.CAP_PROP_FPS), default=20.0)
        info = VideoInfo(str(source), width, height, fps, 0)

        base = base_name or stream_base_name(source)
        raw_path = output_dir / f"{base}_mp4v.mp4"
        final_path = output_dir / f"{base}.mp4"
        records_path = REPORTS_DIR / f"{base}_frames.csv" if config.save_frame_records else None
        if save_output:
            check_disk_space(output_dir, 200 * 1024 * 1024)

        self.detector.prepare_run()

        state = _RunState(config.enable_tracking)
        warnings: list[str] = []
        writer = self._open_writer(raw_path, fps, (width, height)) if save_output else None
        records_fh, records_writer = self._open_records(records_path)
        start = time.perf_counter()
        frame: np.ndarray | None = first
        failures = 0
        try:
            while time.perf_counter() - start < max_seconds and not should_stop():
                if frame is None:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        failures += 1
                        frame = None
                        if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                            warnings.append("The camera/stream stopped delivering frames.")
                            break
                        continue
                failures = 0
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                frame_index = state.frames
                detections = self._process_frame(frame, config, state)
                self._write_records(records_writer, frame_index, fps, detections)
                if writer is not None:
                    writer.write(frame)
                frame_callback(
                    frame,
                    ProcessingProgress(state.frames, 0, time.perf_counter() - start, state.vehicle_detections),
                )
                frame = None
        finally:
            # Runs even when Streamlit interrupts the script (Stop button / rerun).
            cap.release()
            if writer is not None:
                writer.release()
            if records_fh:
                records_fh.close()

        elapsed = time.perf_counter() - start
        browser_ok = True
        if save_output and state.frames > 0:
            browser_ok = self._finalize_output(raw_path, final_path, warnings)
        else:
            raw_path.unlink(missing_ok=True)
        return self._build_result(
            str(source), final_path if save_output else Path(""), info, config, state, elapsed,
            records_path, browser_ok, False, warnings,
        )
