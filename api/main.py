"""AI Vehicle Detection System - REST API (FastAPI).

Exposes the detection pipeline (detector.py, video_processor.py, tracker.py,
reports.py) to the Next.js frontend in ../web.

Run from this folder:
    uvicorn main:app --host 127.0.0.1 --port 8000
Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jobs import DetectorSettings, Job, JobManager
from reports import build_run_metadata, build_summary_table, load_frame_records, save_summary_csv, summary_csv_bytes
from tracker import TRACKING_DISCLAIMER, tracking_available
from utils import (
    AVAILABLE_MODELS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    INPUT_DIR,
    MODELS_DIR,
    SUPPORTED_VIDEO_EXTENSIONS,
    VEHICLE_CLASS_NAMES,
    VEHICLE_CLASSES,
    InsufficientDiskSpaceError,
    VideoInfo,
    VideoValidationError,
    check_disk_space,
    cuda_available,
    describe_device,
    ensure_directories,
    probe_video,
    select_device,
    unique_filename,
    validate_video_extension,
)
from video_processor import ProcessingConfig, ProcessingResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

IMAGE_SIZES = (320, 416, 480, 640, 800, 960, 1280)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("VD_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if origin.strip()
]

manager = JobManager()
# Videos known to the API: id -> path. Uploads are also found on disk by id after a restart.
videos: dict[str, Path] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_directories()
    manager.start()
    yield
    manager.shutdown()


app = FastAPI(title="AI Vehicle Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class DetectionSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow the "model_name" field

    model_name: str = DEFAULT_MODEL
    device: Literal["auto", "cpu", "cuda"] = "auto"
    imgsz: int = DEFAULT_IMAGE_SIZE
    confidence: float = Field(DEFAULT_CONFIDENCE, ge=0.05, le=0.95)
    iou: float = Field(DEFAULT_IOU, ge=0.10, le=0.90)
    all_classes: bool = False
    classes: list[str] = Field(default_factory=lambda: list(VEHICLE_CLASS_NAMES))
    enable_tracking: bool = False
    save_records: bool = True
    max_frames: int = Field(0, ge=0)

    @field_validator("imgsz")
    @classmethod
    def _valid_imgsz(cls, value: int) -> int:
        if value not in IMAGE_SIZES:
            raise ValueError(f"Image size must be one of {IMAGE_SIZES}")
        return value

    @field_validator("classes")
    @classmethod
    def _valid_classes(cls, value: list[str]) -> list[str]:
        unknown = [name for name in value if name not in VEHICLE_CLASS_NAMES]
        if unknown:
            raise ValueError(f"Unknown vehicle classes: {unknown}")
        return value

    @field_validator("model_name")
    @classmethod
    def _valid_model(cls, value: str) -> str:
        value = value.strip().strip('"').strip("'")
        if not value:
            raise ValueError("Enter a model name or the path to custom .pt weights.")
        return value


class VideoJobRequest(BaseModel):
    video_id: str
    settings: DetectionSettings = Field(default_factory=DetectionSettings)


class LiveJobRequest(BaseModel):
    source: int | str = 0  # webcam index or RTSP/HTTP URL
    max_seconds: int = Field(60, ge=5, le=3600)
    save_output: bool = True
    settings: DetectionSettings = Field(default_factory=DetectionSettings)


class LocalPathRequest(BaseModel):
    path: str


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def records(df: pd.DataFrame | None) -> list[dict] | None:
    """DataFrame -> JSON-safe list of rows (numpy scalars and NaN handled by pandas)."""
    if df is None:
        return None
    return json.loads(df.to_json(orient="records"))


def video_info_dict(info: VideoInfo) -> dict:
    return {
        "path": info.path,
        "name": Path(info.path).name,
        "width": info.width,
        "height": info.height,
        "resolution": info.resolution,
        "fps": round(info.fps, 3),
        "frame_count": info.frame_count,
        "duration_s": round(info.duration_s, 3),
    }


def result_dict(job: Job, result: ProcessingResult) -> dict:
    output = Path(result.output_path) if result.output_path else None
    records_path = Path(result.frame_records_path) if result.frame_records_path else None
    unique_vehicles = (
        sum(result.unique_counts.get(n, 0) for n in VEHICLE_CLASS_NAMES) if result.unique_counts is not None else None
    )
    return {
        "input_name": Path(result.input_path).name,
        "output_name": output.name if output else None,
        "output_available": bool(output and output.is_file()),
        "browser_compatible": result.browser_compatible,
        "stopped_early": result.stopped_early,
        "warnings": result.warnings,
        "video_info": video_info_dict(result.video_info),
        "model_name": result.model_name,
        "device": result.device,
        "frames_processed": result.frames_processed,
        "processing_time_s": round(result.processing_time_s, 3),
        "average_fps": round(result.average_fps, 2),
        "frames_with_detections": result.frames_with_detections,
        "all_classes_mode": result.config.class_ids is None,
        "tracking": result.config.enable_tracking,
        "detection_counts": result.detection_counts,
        "total_detections": result.total_detections,
        "total_vehicle_detections": result.total_vehicle_detections,
        "unique_counts": result.unique_counts,
        "unique_vehicle_total": unique_vehicles,
        "summary_table": records(build_summary_table(result)),
        "run_metadata": records(build_run_metadata(result)),
        "track_table": records(result.track_table) if result.track_table is not None else None,
        "has_frame_records": bool(records_path and records_path.is_file()),
        "has_track_table": result.track_table is not None and not result.track_table.empty,
        "summary_report_path": job.summary_report_path,
    }


def job_dict(job: Job, include_result: bool = True) -> dict:
    data = {
        "id": job.id,
        "kind": job.kind,
        "source": job.source_label,
        "status": job.status,
        "stage": job.stage,
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "progress": job.progress_dict(),
        "preview_seq": job.preview_seq,
        "max_seconds": job.max_seconds if job.kind == "live" else None,
        "settings": {
            "model_name": job.detector_settings.model_name,
            "device": job.detector_settings.device,
            "imgsz": job.detector_settings.imgsz,
            "confidence": job.config.confidence,
            "iou": job.config.iou,
            "classes": job.class_names,
            "enable_tracking": job.config.enable_tracking,
            "max_frames": job.config.max_frames,
        },
        "result": None,
    }
    if include_result and job.result is not None:
        data["result"] = result_dict(job, job.result)
    return data


# --------------------------------------------------------------------------- #
# Lookups
# --------------------------------------------------------------------------- #
def get_job(job_id: str) -> Job:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return job


def get_result(job_id: str) -> tuple[Job, ProcessingResult]:
    job = get_job(job_id)
    if job.result is None:
        raise HTTPException(409, "This job has no result yet.")
    return job, job.result


def resolve_video(video_id: str) -> Path:
    path = videos.get(video_id)
    if path is None:
        candidate = INPUT_DIR / Path(video_id).name  # strips any directory components
        if candidate.is_file():
            path = videos[video_id] = candidate
    if path is None:
        raise HTTPException(404, "Video not found. Upload it again.")
    return path


def build_job_parts(settings: DetectionSettings) -> tuple[DetectorSettings, list[str] | None, ProcessingConfig]:
    """Validate settings that would otherwise only fail inside the worker."""
    if not settings.all_classes and not settings.classes:
        raise HTTPException(422, "Select at least one vehicle class, or switch to 'All COCO classes'.")
    if settings.enable_tracking:
        ok, message = tracking_available()
        if not ok:
            raise HTTPException(422, message)
    try:
        select_device(settings.device)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    detector_settings = DetectorSettings(settings.model_name, settings.device, settings.imgsz)
    class_names = None if settings.all_classes else list(settings.classes)
    config = ProcessingConfig(
        confidence=settings.confidence,
        iou=settings.iou,
        class_ids=None,  # resolved from class_names by the worker once the model is loaded
        enable_tracking=settings.enable_tracking,
        save_frame_records=settings.save_records,
        max_frames=settings.max_frames,
    )
    return detector_settings, class_names, config


def file_response(path: Path, media_type: str, download: bool) -> FileResponse:
    if not path.is_file():
        raise HTTPException(404, "File not found.")
    disposition = "attachment" if download else "inline"
    return FileResponse(path, media_type=media_type, filename=path.name, content_disposition_type=disposition)


def csv_response(content: bytes, filename: str) -> Response:
    return Response(
        content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict:
    """Everything the frontend needs to build the settings panel."""
    local_weights = sorted(p.name for p in MODELS_DIR.glob("*.pt") if p.name not in AVAILABLE_MODELS)
    has_cuda = cuda_available()
    track_ok, track_msg = tracking_available()
    return {
        "models": list(AVAILABLE_MODELS) + local_weights,
        "default_model": DEFAULT_MODEL,
        "devices": ["auto", "cpu"] + (["cuda"] if has_cuda else []),
        "cuda_available": has_cuda,
        "auto_device": describe_device(select_device("auto")),
        "image_sizes": list(IMAGE_SIZES),
        "defaults": {
            "confidence": DEFAULT_CONFIDENCE,
            "iou": DEFAULT_IOU,
            "imgsz": DEFAULT_IMAGE_SIZE,
        },
        "vehicle_classes": [{"id": cid, "name": name} for cid, name in VEHICLE_CLASSES.items()],
        "tracking": {"available": track_ok, "message": track_msg, "disclaimer": TRACKING_DISCLAIMER},
        "supported_extensions": list(SUPPORTED_VIDEO_EXTENSIONS),
    }


# --------------------------------------------------------------------------- #
# Videos
# --------------------------------------------------------------------------- #
@app.post("/api/videos", status_code=201)
def upload_video(file: UploadFile = File(...)) -> dict:
    """Save an uploaded video to input_videos/, validate it and return its metadata."""
    name = file.filename or "video"
    try:
        validate_video_extension(name)
        ensure_directories()
        check_disk_space(INPUT_DIR, (file.size or 0) + 50 * 1024 * 1024)
    except (VideoValidationError, InsufficientDiskSpaceError) as exc:
        raise HTTPException(422, str(exc)) from exc

    original = Path(name)
    destination = INPUT_DIR / unique_filename(original.stem, original.suffix.lower())
    with open(destination, "wb") as fh:
        shutil.copyfileobj(file.file, fh, length=4 * 1024 * 1024)
    try:
        info = probe_video(destination)
    except VideoValidationError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    videos[destination.name] = destination
    return {"video_id": destination.name, "original_name": original.name, "info": video_info_dict(info)}


@app.post("/api/videos/local")
def register_local_video(body: LocalPathRequest) -> dict:
    """Use a video that already exists on the API machine (no upload, any size)."""
    raw = body.path.strip().strip('"').strip("'")
    if not raw:
        raise HTTPException(422, "Enter the full path to a video file.")
    path = Path(raw).expanduser().resolve()
    try:
        info = probe_video(path)
    except VideoValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    video_id = f"local-{abs(hash(str(path))):x}"
    videos[video_id] = path
    return {"video_id": video_id, "original_name": path.name, "info": video_info_dict(info)}


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict:
    path = resolve_video(video_id)
    try:
        info = probe_video(path)
    except VideoValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"video_id": video_id, "original_name": path.name, "info": video_info_dict(info)}


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@app.post("/api/jobs/video", status_code=202)
def start_video_job(body: VideoJobRequest) -> dict:
    path = resolve_video(body.video_id)
    detector_settings, class_names, config = build_job_parts(body.settings)
    job = Job(
        id=manager.new_id(),
        kind="video",
        source_label=path.name,
        detector_settings=detector_settings,
        class_names=class_names,
        config=config,
        video_path=path,
    )
    return job_dict(manager.submit(job))


@app.post("/api/jobs/live", status_code=202)
def start_live_job(body: LiveJobRequest) -> dict:
    source: int | str = body.source
    if isinstance(source, str):
        source = source.strip()
        if source.isdigit():
            source = int(source)
        elif not source:
            raise HTTPException(422, "Enter a camera index or stream URL.")
    detector_settings, class_names, config = build_job_parts(body.settings)
    job = Job(
        id=manager.new_id(),
        kind="live",
        source_label=f"Camera {source}" if isinstance(source, int) else str(source),
        detector_settings=detector_settings,
        class_names=class_names,
        config=config,
        stream_source=source,
        max_seconds=float(body.max_seconds),
        save_output=body.save_output,
    )
    return job_dict(manager.submit(job))


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [job_dict(job, include_result=False) for job in manager.list()]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return job_dict(get_job(job_id))


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Cancel a video job, or stop a live session (its recording is still finalised)."""
    job = get_job(job_id)
    if not job.finished:
        manager.cancel(job)
    return job_dict(job)


@app.get("/api/jobs/{job_id}/preview.jpg")
def job_preview(job_id: str) -> Response:
    job = get_job(job_id)
    if job.preview_jpeg is None:
        raise HTTPException(404, "No preview frame yet.")
    return Response(job.preview_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/jobs/{job_id}/stream")
def job_stream(job_id: str) -> StreamingResponse:
    """MJPEG stream of annotated preview frames (use as <img src>), ends when the job finishes."""
    job = get_job(job_id)

    def frames() -> Iterator[bytes]:
        last_seq = -1
        while True:
            if job.preview_jpeg is not None and job.preview_seq != last_seq:
                last_seq = job.preview_seq
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + job.preview_jpeg + b"\r\n"
            elif job.finished:
                return
            time.sleep(0.04)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


# --------------------------------------------------------------------------- #
# Results and downloads
# --------------------------------------------------------------------------- #
@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str, download: bool = False) -> FileResponse:
    _, result = get_result(job_id)
    if not result.output_path:
        raise HTTPException(404, "This run did not save a video.")
    return file_response(Path(result.output_path), "video/mp4", download)


@app.get("/api/jobs/{job_id}/summary.csv")
def job_summary_csv(job_id: str) -> Response:
    _, result = get_result(job_id)
    return csv_response(summary_csv_bytes(result), f"{Path(result.input_path).stem}_summary.csv")


@app.get("/api/jobs/{job_id}/records.csv")
def job_records_csv(job_id: str) -> FileResponse:
    _, result = get_result(job_id)
    if not result.frame_records_path:
        raise HTTPException(404, "Frame-level records were not saved for this run.")
    return file_response(Path(result.frame_records_path), "text/csv", download=True)


@app.get("/api/jobs/{job_id}/tracks.csv")
def job_tracks_csv(job_id: str) -> Response:
    _, result = get_result(job_id)
    if result.track_table is None or result.track_table.empty:
        raise HTTPException(404, "No tracked objects for this run.")
    return csv_response(result.track_table.to_csv(index=False).encode("utf-8"), f"{Path(result.input_path).stem}_tracks.csv")


@app.get("/api/jobs/{job_id}/records")
def job_records(job_id: str, limit: int = Query(500, ge=1, le=10_000)) -> list[dict]:
    """First ``limit`` frame-level detection records as JSON."""
    _, result = get_result(job_id)
    path = Path(result.frame_records_path) if result.frame_records_path else None
    return records(load_frame_records(path, max_rows=limit)) or []


@app.post("/api/jobs/{job_id}/report")
def save_report(job_id: str) -> dict:
    """Write the summary + run metadata CSV to the reports/ folder."""
    job, result = get_result(job_id)
    path = save_summary_csv(result)
    job.summary_report_path = str(path)
    return {"path": str(path)}
