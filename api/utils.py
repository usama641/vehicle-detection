"""Shared constants, paths, device selection, validation and file helpers.

Nothing in this module depends on Streamlit, so it can be reused by the
command-line validation script and the test-suite.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

logger = logging.getLogger("vehicle_detection")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
INPUT_DIR = PROJECT_ROOT / "input_videos"
OUTPUT_DIR = PROJECT_ROOT / "output_videos"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --------------------------------------------------------------------------- #
# Detection configuration
# --------------------------------------------------------------------------- #
SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov")
DEFAULT_MODEL = "yolov8n.pt"
# Official Ultralytics COCO-pretrained weights. All are downloaded automatically
# on first use and share the same 80-class COCO label map.
AVAILABLE_MODELS: tuple[str, ...] = (
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
)
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU = 0.45
DEFAULT_IMAGE_SIZE = 640

# COCO class IDs used by every standard COCO-trained YOLO model.
VEHICLE_CLASSES: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASS_NAMES: tuple[str, ...] = tuple(VEHICLE_CLASSES.values())

# OpenCV draws in BGR order.
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "car": (75, 180, 60),  # green
    "motorcycle": (0, 165, 255),  # orange
    "bus": (255, 140, 0),  # blue
    "truck": (60, 20, 220),  # red
}
_FALLBACK_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (180, 119, 31),
    (14, 127, 255),
    (44, 160, 44),
    (40, 39, 214),
    (189, 103, 148),
    (75, 86, 140),
    (194, 119, 227),
    (127, 127, 127),
    (34, 189, 188),
    (207, 190, 23),
)


class VideoValidationError(ValueError):
    """Raised when a video file is missing, unsupported or unreadable."""


class InsufficientDiskSpaceError(OSError):
    """Raised when there is not enough free disk space to write the output."""


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata about a video file or stream."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int  # 0 means "unknown" (some containers do not report it)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 and self.frame_count > 0 else 0.0

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


# --------------------------------------------------------------------------- #
# General helpers
# --------------------------------------------------------------------------- #
def ensure_directories() -> None:
    """Create the managed project directories if they do not exist yet."""
    for directory in (MODELS_DIR, INPUT_DIR, OUTPUT_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def color_for_class(class_name: str, class_id: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for a class (fixed colours for vehicles)."""
    return CLASS_COLORS_BGR.get(class_name, _FALLBACK_PALETTE_BGR[class_id % len(_FALLBACK_PALETTE_BGR)])


def unique_filename(stem: str, suffix: str) -> str:
    """Build a filesystem-safe, collision-free filename such as ``clip_20261019_142233_ab12cd.mp4``."""
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:60] or "video"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_stem}_{timestamp}_{uuid.uuid4().hex[:6]}{suffix}"


def format_seconds(seconds: float) -> str:
    """Format a duration as ``1h 02m 03s`` / ``2m 05s`` / ``4.2s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


# --------------------------------------------------------------------------- #
# Video validation
# --------------------------------------------------------------------------- #
def validate_video_extension(path: str | Path) -> None:
    """Reject files whose extension is not MP4, AVI or MOV."""
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        allowed = ", ".join(ext.upper().lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS)
        raise VideoValidationError(f"Unsupported file type '{suffix or '(none)'}'. Please use one of: {allowed}.")


def probe_video(path: str | Path) -> VideoInfo:
    """Open a video file, read its first frame and return its metadata.

    Raises:
        VideoValidationError: if the file is missing, has an unsupported
            extension, or cannot be decoded by OpenCV.
    """
    path = Path(path)
    if not path.is_file():
        raise VideoValidationError(f"Video file not found: {path}")
    validate_video_extension(path)
    if path.stat().st_size == 0:
        raise VideoValidationError(f"Video file is empty: {path.name}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoValidationError(
                f"OpenCV could not open '{path.name}'. The file may be corrupted or use an unsupported codec."
            )
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoValidationError(
                f"'{path.name}' was opened but no frames could be decoded. The file may be corrupted or truncated."
            )
        height, width = frame.shape[:2]
        fps = sanitize_fps(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return VideoInfo(str(path), width, height, fps, max(frame_count, 0))
    finally:
        cap.release()


def sanitize_fps(fps: float, default: float = 30.0) -> float:
    """Some files/cameras report 0, NaN or absurd frame rates; fall back to a sane default."""
    if fps is None or math.isnan(fps) or fps <= 0 or fps > 240:
        return default
    return float(fps)


# --------------------------------------------------------------------------- #
# Disk space and uploads
# --------------------------------------------------------------------------- #
def estimate_output_bytes(input_path: str | Path, info: VideoInfo) -> int:
    """Rough upper bound for the space needed by the intermediate and final videos."""
    input_size = Path(input_path).stat().st_size
    # mp4v intermediate is usually ~1-3x the source; the H.264 copy is similar to the source.
    pixel_estimate = int(info.width * info.height * max(info.frame_count, 1) * 0.15)
    return max(input_size * 4, pixel_estimate, 100 * 1024 * 1024)


def check_disk_space(directory: str | Path, required_bytes: int) -> None:
    """Raise InsufficientDiskSpaceError if ``directory`` lacks ``required_bytes`` of free space."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < required_bytes:
        raise InsufficientDiskSpaceError(
            f"Not enough free disk space in '{directory}': {free / 1e9:.2f} GB free, "
            f"about {required_bytes / 1e9:.2f} GB needed. Free up space or process a shorter clip."
        )


def save_uploaded_bytes(data: bytes | memoryview, original_name: str) -> Path:
    """Persist an uploaded video to ``input_videos/`` under a unique name and return its path."""
    validate_video_extension(original_name)
    ensure_directories()
    check_disk_space(INPUT_DIR, len(data) + 50 * 1024 * 1024)
    original = Path(original_name)
    destination = INPUT_DIR / unique_filename(original.stem, original.suffix.lower())
    with open(destination, "wb") as fh:
        fh.write(data)
    return destination


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #
def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # torch missing or broken CUDA install
        return False


def select_device(preference: str = "auto") -> str:
    """Resolve a device preference ("auto", "cpu", "cuda") to an actual torch device string.

    "auto" picks the first CUDA GPU when one is usable, otherwise the CPU.
    Requesting "cuda" without a usable GPU raises ValueError instead of
    silently falling back, so the UI never claims GPU use it doesn't have.
    """
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return "cpu"
    if preference.startswith("cuda"):
        if not cuda_available():
            raise ValueError(
                "CUDA was requested but no usable NVIDIA GPU was found. Install a CUDA-enabled PyTorch build "
                "(see README) or choose CPU."
            )
        return "cuda:0" if preference == "cuda" else preference
    return "cuda:0" if cuda_available() else "cpu"


def describe_device(device: str) -> str:
    """Human-readable device description, e.g. 'GPU: NVIDIA GeForce RTX 3060 (cuda:0)'."""
    if device.startswith("cuda"):
        try:
            import torch

            index = int(device.split(":")[1]) if ":" in device else 0
            return f"GPU: {torch.cuda.get_device_name(index)} ({device})"
        except Exception:
            return f"GPU ({device})"
    return "CPU"


# --------------------------------------------------------------------------- #
# Browser-compatible output (H.264)
# --------------------------------------------------------------------------- #
def get_ffmpeg_executable() -> str | None:
    """Return a path to an ffmpeg binary (bundled imageio-ffmpeg first, then the system PATH)."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def transcode_to_h264(source: Path, destination: Path) -> bool:
    """Re-encode ``source`` to H.264/yuv420p MP4 so browsers and common players can play it.

    OpenCV's pip wheels can reliably write only MPEG-4 Part 2 ("mp4v"), which
    plays in VLC but not in Chrome/Firefox/Edge. Returns True on success.
    """
    ffmpeg = get_ffmpeg_executable()
    if ffmpeg is None:
        logger.warning("ffmpeg not found; output will stay in mp4v format.")
        return False
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # yuv420p requires even dimensions
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(destination),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        logger.warning("Could not run ffmpeg: %s", exc)
        return False
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        logger.warning("ffmpeg transcoding failed: %s", completed.stderr.strip()[:500])
        destination.unlink(missing_ok=True)
        return False
    return True
