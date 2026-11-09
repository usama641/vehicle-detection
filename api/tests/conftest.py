"""Shared pytest fixtures. Model-dependent tests are skipped if the model cannot be loaded (e.g. offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 15.0) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "OpenCV cannot write mp4v videos on this system"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def bus_image() -> np.ndarray:
    """Ultralytics' bundled sample image containing a bus and pedestrians."""
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    assert image is not None
    return image


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory, bus_image) -> Path:
    """30-frame video with the bus scene sliding horizontally across a grey canvas."""
    height, width = 480, 854
    scene = cv2.resize(bus_image, (360, 480))
    frames = []
    n = 30
    for i in range(n):
        canvas = np.full((height, width, 3), 40, dtype=np.uint8)
        x = int(i * (width - scene.shape[1]) / (n - 1))
        canvas[:, x:x + scene.shape[1]] = scene
        frames.append(canvas)
    return _write_video(tmp_path_factory.mktemp("videos") / "bus_scene.mp4", frames)


@pytest.fixture(scope="session")
def blank_video(tmp_path_factory) -> Path:
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(10)]
    return _write_video(tmp_path_factory.mktemp("videos") / "blank.mp4", frames)


@pytest.fixture(scope="session")
def detector():
    from detector import ModelLoadError, VehicleDetector

    try:
        return VehicleDetector("yolov8n.pt", device="cpu")
    except ModelLoadError as exc:
        pytest.skip(f"YOLO model unavailable: {exc}")


@pytest.fixture()
def isolated_dirs(tmp_path, monkeypatch):
    """Redirect output and report folders to a temp dir so tests don't litter the project."""
    import video_processor

    out_dir, rep_dir = tmp_path / "out", tmp_path / "reports"
    out_dir.mkdir()
    rep_dir.mkdir()
    monkeypatch.setattr(video_processor, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(video_processor, "REPORTS_DIR", rep_dir)
    return out_dir, rep_dir
