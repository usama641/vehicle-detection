"""Tests for validation, detection, drawing, video processing and statistics.

Run from the project folder:  python -m pytest -v
The first run downloads yolov8n.pt (~6 MB) into models/.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from detector import Detection, draw_detections
from reports import FRAME_RECORD_FIELDS, build_summary_table
from tracker import TrackStatistics
from utils import (
    VEHICLE_CLASSES,
    VideoValidationError,
    color_for_class,
    probe_video,
    select_device,
    unique_filename,
    validate_video_extension,
)
from video_processor import ProcessingConfig, VideoProcessor

TEST_CONF = 0.25


# --------------------------------------------------------------------------- #
# Input validation (no model needed)
# --------------------------------------------------------------------------- #
def test_probe_valid_video(sample_video: Path) -> None:
    info = probe_video(sample_video)
    assert (info.width, info.height) == (854, 480)
    assert info.frame_count == 30
    assert info.fps == pytest.approx(15, abs=0.5)


@pytest.mark.parametrize("name", ["clip.mkv", "image.jpg", "noext"])
def test_unsupported_extension_rejected(name: str) -> None:
    with pytest.raises(VideoValidationError, match="Unsupported file type"):
        validate_video_extension(name)


def test_missing_file_gives_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(VideoValidationError, match="not found"):
        probe_video(tmp_path / "does_not_exist.mp4")


def test_empty_file_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp4"
    empty.touch()
    with pytest.raises(VideoValidationError, match="empty"):
        probe_video(empty)


def test_corrupted_file_rejected(tmp_path: Path) -> None:
    corrupted = tmp_path / "corrupted.mp4"
    corrupted.write_bytes(np.random.default_rng(0).bytes(50_000))
    with pytest.raises(VideoValidationError):
        probe_video(corrupted)


def test_cpu_device_selection() -> None:
    assert select_device("cpu") == "cpu"
    assert select_device("auto") in {"cpu", "cuda:0"}


def test_unique_filename_is_safe() -> None:
    name = unique_filename("my video (1)/../x", ".mp4")
    assert name.endswith(".mp4") and "/" not in name and " " not in name


def test_vehicle_class_ids_match_coco() -> None:
    assert VEHICLE_CLASSES == {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert len({color_for_class(n, i) for i, n in VEHICLE_CLASSES.items()}) == 4  # distinct colours


# --------------------------------------------------------------------------- #
# Drawing and statistics (no model needed)
# --------------------------------------------------------------------------- #
def test_draw_detections_draws_box_in_class_colour() -> None:
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    det = Detection(2, "car", 0.9, (100, 120, 300, 260))
    draw_detections(frame, [det])
    expected = np.array(color_for_class("car", 2), dtype=int)
    # Bottom edge of the box (well away from the label) has the class colour (allowing for anti-aliasing)...
    edge = frame[258:263, 200].astype(int)
    assert any(np.abs(px - expected).max() < 60 for px in edge)
    # ...the box interior is untouched, and the label background is drawn above the box.
    assert not frame[200, 200].any()
    assert frame[110, 105].any()


def test_draw_detections_label_inside_box_at_top_edge() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_detections(frame, [Detection(7, "truck", 0.5, (10, 0, 150, 150), track_id=3)])
    assert frame[5, 15].any()  # label drawn inside the box instead of off-screen


def test_track_statistics_majority_vote() -> None:
    stats = TrackStatistics()
    frames = [
        [Detection(2, "car", 0.8, (0, 0, 10, 10), 1), Detection(5, "bus", 0.9, (20, 20, 40, 40), 2)],
        [Detection(7, "truck", 0.6, (0, 0, 10, 10), 1), Detection(5, "bus", 0.9, (20, 20, 40, 40), 2)],
        [Detection(2, "car", 0.7, (0, 0, 10, 10), 1), Detection(2, "car", 0.7, (50, 50, 60, 60), None)],
    ]
    for i, dets in enumerate(frames):
        stats.update(dets, i)
    assert stats.unique_count == 2  # untracked box ignored
    assert stats.unique_counts_by_class() == {"car": 1, "bus": 1}  # track 1 is mostly "car"
    table = stats.to_dataframe(fps=10)
    assert list(table["frames_seen"]) == [3, 2]
    assert table.loc[0, "last_seen_s"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Model and end-to-end pipeline
# --------------------------------------------------------------------------- #
def test_model_loads_with_coco_vehicle_classes(detector) -> None:
    assert detector.vehicle_class_ids() == {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
    assert detector.device == "cpu"


def test_detects_bus_in_sample_image(detector, bus_image) -> None:
    detections = detector.detect(bus_image, TEST_CONF, class_ids=[2, 3, 5, 7])
    names = {d.class_name for d in detections}
    assert "bus" in names
    assert names <= {"car", "motorcycle", "bus", "truck"}  # vehicle-only mode excludes people
    h, w = bus_image.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 < w and 0 <= y1 < y2 < h
        assert d.confidence >= TEST_CONF


def test_all_classes_mode_includes_people(detector, bus_image) -> None:
    names = {d.class_name for d in detector.detect(bus_image, TEST_CONF, class_ids=None)}
    assert "person" in names


def test_process_video_end_to_end(detector, sample_video, isolated_dirs) -> None:
    out_dir, _ = isolated_dirs
    config = ProcessingConfig(confidence=TEST_CONF, class_ids=[2, 3, 5, 7], save_frame_records=True)
    progress_calls = []
    result = VideoProcessor(detector).process_video(
        sample_video, config, output_dir=out_dir, progress_callback=progress_calls.append
    )

    # Output exists, can be reopened, and keeps resolution and frame count.
    output = Path(result.output_path)
    assert output.is_file() and output.parent == out_dir
    info = probe_video(output)
    assert (info.width, info.height) == (854, 480)
    assert info.frame_count == 30
    assert not list(out_dir.glob("*_mp4v.mp4"))  # intermediate file cleaned up

    # Statistics are consistent.
    assert result.frames_processed == 30
    assert result.detection_counts.get("bus", 0) > 0
    assert result.total_vehicle_detections == sum(result.detection_counts.values())
    assert 0 < result.frames_with_detections <= 30
    assert result.average_fps > 0
    assert result.unique_counts is None  # no tracking => no "unique" numbers
    assert progress_calls[-1].frames_processed == 30

    # Frame records add up to the reported totals.
    with open(result.frame_records_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0].keys()) == FRAME_RECORD_FIELDS
    assert len(rows) == result.total_detections
    assert sum(r["class_name"] == "bus" for r in rows) == result.detection_counts["bus"]

    table = build_summary_table(result)
    total_row = table[table["Class"] == "TOTAL (vehicles)"].iloc[0]
    assert total_row["Detections across frames"] == result.total_vehicle_detections
    assert "Unique tracked IDs (estimate)" not in table.columns


def test_max_frames_limit(detector, sample_video, isolated_dirs) -> None:
    out_dir, _ = isolated_dirs
    config = ProcessingConfig(confidence=TEST_CONF, class_ids=[5], save_frame_records=False, max_frames=5)
    result = VideoProcessor(detector).process_video(sample_video, config, output_dir=out_dir)
    assert result.frames_processed == 5
    assert set(result.detection_counts) <= {"bus"}
    assert result.frame_records_path is None


def test_blank_video_yields_zero_detections(detector, blank_video, isolated_dirs) -> None:
    out_dir, _ = isolated_dirs
    result = VideoProcessor(detector).process_video(blank_video, ProcessingConfig(class_ids=[2, 3, 5, 7]), output_dir=out_dir)
    assert result.frames_processed == 10
    assert result.total_detections == 0


def test_tracking_assigns_ids(detector, sample_video, isolated_dirs) -> None:
    from tracker import tracking_available

    available, message = tracking_available()
    if not available:
        pytest.skip(message)
    out_dir, _ = isolated_dirs
    config = ProcessingConfig(confidence=TEST_CONF, class_ids=[2, 3, 5, 7], enable_tracking=True)
    result = VideoProcessor(detector).process_video(sample_video, config, output_dir=out_dir)
    assert result.unique_counts is not None
    assert result.unique_total >= 1
    # A single moving bus produces many frame-level detections but few track IDs.
    assert result.unique_counts.get("bus", 0) < result.detection_counts["bus"]
    assert result.track_table is not None and not result.track_table.empty
    assert "Unique tracked IDs (estimate)" in build_summary_table(result).columns


def test_invalid_input_to_processor(detector, tmp_path, isolated_dirs) -> None:
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not a video")
    with pytest.raises(VideoValidationError):
        VideoProcessor(detector).process_video(bogus, ProcessingConfig(), output_dir=isolated_dirs[0])
