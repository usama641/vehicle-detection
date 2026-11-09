"""Detection statistics tables and CSV export (Pandas)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from utils import REPORTS_DIR, VEHICLE_CLASS_NAMES, ensure_directories, unique_filename

if TYPE_CHECKING:
    from video_processor import ProcessingResult

# Column order of the optional frame-level CSV written during processing.
FRAME_RECORD_FIELDS: tuple[str, ...] = (
    "frame_index",
    "timestamp_s",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "track_id",
)


def build_summary_table(result: "ProcessingResult") -> pd.DataFrame:
    """One row per class with frame-level detections and (if tracking) unique track IDs.

    Vehicle classes are always listed (with 0 when absent) so the table has a
    stable shape; other COCO classes appear only when detected in all-classes mode.
    """
    class_names = list(VEHICLE_CLASS_NAMES) + sorted(
        name for name in result.detection_counts if name not in VEHICLE_CLASS_NAMES
    )
    rows = []
    for name in class_names:
        row = {
            "Class": name,
            "Vehicle": "yes" if name in VEHICLE_CLASS_NAMES else "no",
            "Detections across frames": result.detection_counts.get(name, 0),
        }
        if result.unique_counts is not None:
            row["Unique tracked IDs (estimate)"] = result.unique_counts.get(name, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    total = {"Class": "TOTAL (vehicles)", "Vehicle": "", "Detections across frames": result.total_vehicle_detections}
    if result.unique_counts is not None:
        total["Unique tracked IDs (estimate)"] = sum(result.unique_counts.get(n, 0) for n in VEHICLE_CLASS_NAMES)
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def build_run_metadata(result: "ProcessingResult") -> pd.DataFrame:
    """Key/value table describing the run (input, settings, performance)."""
    info = result.video_info
    items = {
        "Input video": Path(result.input_path).name,
        "Output video": Path(result.output_path).name,
        "Resolution": info.resolution,
        "Source FPS": round(info.fps, 2),
        "Frames processed": result.frames_processed,
        "Processing time (s)": round(result.processing_time_s, 2),
        "Average processing FPS": round(result.average_fps, 2),
        "Device": result.device,
        "Model": result.model_name,
        "Confidence threshold": result.config.confidence,
        "IoU threshold": result.config.iou,
        "Mode": "All COCO classes" if result.config.class_ids is None else "Selected vehicle classes",
        "Tracking": "ByteTrack" if result.config.enable_tracking else "off",
        "Total detections (all classes, across frames)": result.total_detections,
        "Vehicle detections (across frames)": result.total_vehicle_detections,
        "Frames containing a detection": result.frames_with_detections,
    }
    if result.unique_total is not None:
        items["Unique tracked IDs (estimate)"] = result.unique_total
    return pd.DataFrame({"Metric": list(items.keys()), "Value": [str(v) for v in items.values()]})


def save_summary_csv(result: "ProcessingResult", directory: Path = REPORTS_DIR) -> Path:
    """Write the class summary followed by run metadata to ``reports/`` and return the path."""
    ensure_directories()
    path = directory / unique_filename(Path(result.input_path).stem + "_summary", ".csv")
    summary = build_summary_table(result)
    metadata = build_run_metadata(result)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        summary.to_csv(fh, index=False)
        fh.write("\n")
        metadata.to_csv(fh, index=False)
    return path


def summary_csv_bytes(result: "ProcessingResult") -> bytes:
    """CSV content of the class summary table for a download button."""
    return build_summary_table(result).to_csv(index=False).encode("utf-8")


def load_frame_records(path: Path | None, max_rows: int | None = None) -> pd.DataFrame:
    """Read the frame-level CSV (optionally only the first ``max_rows`` rows for preview)."""
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=list(FRAME_RECORD_FIELDS))
    return pd.read_csv(path, nrows=max_rows)
