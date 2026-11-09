"""Command-line environment check and end-to-end smoke test.

Usage:
    python validate_setup.py                 # checks + run on a generated sample video
    python validate_setup.py path/to/video.mp4 [--track] [--max-frames 300]

It verifies the installation, loads the model, processes a video and prints
the statistics, without starting the Streamlit UI.
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def make_sample_video(path: Path, frames: int = 45) -> Path:
    """Create a short MP4 that slides Ultralytics' bundled bus.jpg sample across the frame."""
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    if image is None:
        raise RuntimeError("Could not read the ultralytics sample image bus.jpg")
    height, width = 480, 854
    image = cv2.resize(image, (360, 480))  # keep the whole scene (bus + people) visible
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))
    for i in range(frames):
        canvas = np.full((height, width, 3), 40, dtype=np.uint8)
        x = int(i * (width - image.shape[1]) / max(frames - 1, 1))
        canvas[:, x:x + image.shape[1]] = image
        writer.write(canvas)
    writer.release()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", help="video to process (default: generated sample)")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--track", action="store_true", help="enable ByteTrack tracking")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    print("1. Environment")
    print(f"  Python {platform.python_version()} on {platform.system()} {platform.release()}")
    if sys.version_info < (3, 10):
        fail("Python 3.10 or newer is required")
        return 1
    try:
        import pandas
        import streamlit
        import torch
        import ultralytics

        ok(f"ultralytics {ultralytics.__version__}, torch {torch.__version__}, opencv {cv2.__version__}, "
           f"streamlit {streamlit.__version__}, pandas {pandas.__version__}")
    except ImportError as exc:
        fail(f"Missing package: {exc}. Run: pip install -r requirements.txt")
        return 1

    from detector import ModelLoadError, VehicleDetector
    from reports import build_summary_table
    from tracker import tracking_available
    from utils import describe_device, get_ffmpeg_executable, probe_video
    from video_processor import ProcessingConfig, VideoProcessor

    print(f"  CUDA available: {torch.cuda.is_available()}")
    ffmpeg = get_ffmpeg_executable()
    (ok if ffmpeg else fail)(f"ffmpeg for H.264 output: {ffmpeg or 'not found (output stays mp4v)'}")
    track_ok, track_msg = tracking_available()
    (ok if track_ok else fail)(f"Tracking: {track_msg}")

    print("2. Model")
    try:
        detector = VehicleDetector(args.model, device=args.device)
        ok(f"Loaded {detector.weights_path} on {describe_device(detector.device)}")
        ok(f"Vehicle class IDs: {detector.vehicle_class_ids()}")
    except (ModelLoadError, ValueError) as exc:
        fail(str(exc))
        return 1

    print("3. Video")
    tmp_dir = tempfile.TemporaryDirectory()
    video = Path(args.video) if args.video else make_sample_video(Path(tmp_dir.name) / "sample.mp4")
    try:
        info = probe_video(video)
        ok(f"Opened {video.name}: {info.resolution} @ {info.fps:.1f} fps, {info.frame_count} frames")
    except Exception as exc:
        fail(str(exc))
        return 1

    print("4. Processing")
    config = ProcessingConfig(
        confidence=args.conf,
        class_ids=detector.class_ids_for(["car", "motorcycle", "bus", "truck"]),
        enable_tracking=args.track and track_ok,
        max_frames=args.max_frames,
    )

    def progress(p) -> None:
        if p.total_frames:
            print(f"\r  {p.frames_processed}/{p.total_frames} frames", end="", flush=True)

    result = VideoProcessor(detector).process_video(video, config, progress_callback=progress)
    print()
    out = probe_video(result.output_path)
    ok(f"Output: {result.output_path} ({out.frame_count} frames, H.264={result.browser_compatible})")
    ok(f"{result.frames_processed} frames in {result.processing_time_s:.1f}s ({result.average_fps:.1f} FPS)")
    for warning in result.warnings:
        print(f"  [WARN] {warning}")
    print(build_summary_table(result).to_string(index=False))
    tmp_dir.cleanup()

    if result.total_vehicle_detections == 0:
        print("  [WARN] No vehicles were detected. Try a lower confidence or a different video.")
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
