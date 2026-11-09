"""YOLO model loading, inference and bounding-box drawing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tracker import TRACKER_CONFIG
from utils import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    MODELS_DIR,
    VEHICLE_CLASSES,
    color_for_class,
    ensure_directories,
    select_device,
)

logger = logging.getLogger("vehicle_detection")


class ModelLoadError(RuntimeError):
    """Raised when the YOLO weights cannot be found, downloaded or loaded."""


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coordinates
    track_id: int | None = None

    @property
    def label(self) -> str:
        prefix = f"#{self.track_id} " if self.track_id is not None else ""
        return f"{prefix}{self.class_name} {self.confidence:.2f}"


class VehicleDetector:
    """Wraps an Ultralytics YOLO model.

    The model is loaded once in ``__init__`` and reused for every frame. In the
    Streamlit app the whole detector is cached with ``st.cache_resource``.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto", imgsz: int = DEFAULT_IMAGE_SIZE) -> None:
        self.device = select_device(device)
        self.imgsz = imgsz
        self.half = self.device.startswith("cuda")  # FP16 is only beneficial (and supported) on GPU
        self.weights_path = self._resolve_weights(model_name)
        self.model = self._load_model(self.weights_path)
        self.names: dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}
        self._warmup()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_weights(model_name: str) -> str:
        """Find the weights file, preferring ``models/``.

        * An existing path is used as-is.
        * Official names (``yolov8n.pt``, ``yolo11n.pt``, ...) that are not
          present yet resolve to ``models/<name>``; Ultralytics downloads the
          file to that location on first load.
        """
        ensure_directories()
        candidate = Path(model_name).expanduser()
        if candidate.is_file():
            return str(candidate)
        local = MODELS_DIR / candidate.name
        if local.is_file():
            return str(local)
        if candidate.suffix != ".pt":
            raise ModelLoadError(f"Model weights must be a .pt file, got '{model_name}'.")
        if not candidate.name.lower().startswith("yolo") or candidate.parent != Path("."):
            raise ModelLoadError(
                f"Model file '{model_name}' was not found. Check the path, or use an official model name "
                f"such as '{DEFAULT_MODEL}' which is downloaded automatically."
            )
        return str(local)

    @staticmethod
    def _load_model(weights: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError("The 'ultralytics' package is not installed. Run: pip install -r requirements.txt") from exc
        try:
            return YOLO(weights)
        except Exception as exc:
            name = Path(weights).name
            raise ModelLoadError(
                f"Could not load YOLO weights '{name}': {exc}\n"
                f"If you are offline, download {name} from "
                f"https://github.com/ultralytics/assets/releases and place it in '{MODELS_DIR}'."
            ) from exc

    def _warmup(self) -> None:
        """Run one dummy inference so device/weights errors surface at load time, not mid-video."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(dummy, device=self.device, imgsz=self.imgsz, half=self.half, verbose=False)
        except Exception as exc:
            raise ModelLoadError(f"Model loaded but inference failed on device '{self.device}': {exc}") from exc

    # ------------------------------------------------------------------ #
    # Class helpers
    # ------------------------------------------------------------------ #
    def vehicle_class_ids(self) -> dict[str, int]:
        """Map vehicle names to this model's class IDs (looked up by name, so non-standard label maps work too)."""
        by_name = {name.lower(): cid for cid, name in self.names.items()}
        found = {name: by_name[name] for name in VEHICLE_CLASSES.values() if name in by_name}
        if not found:
            raise ModelLoadError("This model does not contain any of the classes car, motorcycle, bus or truck.")
        return found

    def class_ids_for(self, class_names: list[str]) -> list[int]:
        mapping = self.vehicle_class_ids()
        return [mapping[name] for name in class_names if name in mapping]

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def detect(
        self,
        frame: np.ndarray,
        confidence: float,
        class_ids: list[int] | None = None,
        iou: float = DEFAULT_IOU,
        track: bool = False,
    ) -> list[Detection]:
        """Detect (and optionally track) objects in a BGR frame.

        ``class_ids`` restricts inference to those classes (vehicle-only mode);
        ``None`` keeps all 80 COCO classes. Filtering happens inside the model's
        NMS step, so unrelated classes are never post-processed or tracked.
        """
        kwargs = dict(
            conf=confidence,
            iou=iou,
            classes=class_ids,
            device=self.device,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
        )
        if track:
            results = self.model.track(frame, persist=True, tracker=TRACKER_CONFIG, **kwargs)
        else:
            results = self.model.predict(frame, **kwargs)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().round().astype(int)
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.int().cpu().tolist() if (track and boxes.id is not None) else [None] * len(classes)

        height, width = frame.shape[:2]
        detections: list[Detection] = []
        for (x1, y1, x2, y2), conf, cid, tid in zip(xyxy, confs, classes, ids):
            bbox = (
                int(np.clip(x1, 0, width - 1)),
                int(np.clip(y1, 0, height - 1)),
                int(np.clip(x2, 0, width - 1)),
                int(np.clip(y2, 0, height - 1)),
            )
            detections.append(Detection(int(cid), self.names.get(int(cid), str(cid)), float(conf), bbox, tid))
        return detections

    def prepare_run(self) -> None:
        """Reset per-video inference state before processing a new video or stream.

        ``model.track(persist=True)`` stores ByteTrack state on the predictor and
        registers tracker callbacks on the model. Because the model is cached and
        reused, we drop the predictor (rebuilt cheaply from the already-loaded
        weights), restore the default callbacks and restart the ID counter so
        each video starts with fresh track IDs and plain detection runs are not
        affected by an earlier tracking run.
        """
        self.model.predictor = None
        reset_callbacks = getattr(self.model, "reset_callbacks", None)
        if callable(reset_callbacks):
            reset_callbacks()
        try:
            from ultralytics.trackers.basetrack import BaseTrack

            BaseTrack.reset_id()
        except Exception:  # pragma: no cover - older/newer ultralytics layouts
            pass


# ---------------------------------------------------------------------- #
# Drawing
# ---------------------------------------------------------------------- #
def draw_detections(frame: np.ndarray, detections: list[Detection], show_confidence: bool = True) -> np.ndarray:
    """Draw boxes and labels onto ``frame`` in place and return it.

    Line width and font size scale with the frame resolution so labels stay
    legible on both 480p and 4K footage.
    """
    height, width = frame.shape[:2]
    scale = max(min(width, height) / 720.0, 0.5)
    thickness = max(int(round(2 * scale)), 1)
    font_scale = 0.55 * scale
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = color_for_class(det.class_name, det.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        label = det.label if show_confidence else (f"#{det.track_id} " if det.track_id is not None else "") + det.class_name
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, max(thickness - 1, 1))
        pad = max(int(3 * scale), 2)
        # Put the label above the box, or inside it when the box touches the top edge.
        top = y1 - th - baseline - 2 * pad
        if top < 0:
            top = y1
        cv2.rectangle(frame, (x1, top), (x1 + tw + 2 * pad, top + th + baseline + 2 * pad), color, cv2.FILLED)
        cv2.putText(
            frame,
            label,
            (x1 + pad, top + th + pad),
            font,
            font_scale,
            _text_color_for(color),
            max(thickness - 1, 1),
            lineType=cv2.LINE_AA,
        )
    return frame


def draw_frame_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a small semi-transparent info panel in the top-left corner."""
    if not lines:
        return frame
    height, width = frame.shape[:2]
    scale = max(min(width, height) / 720.0, 0.5)
    font_scale = 0.55 * scale
    thickness = max(int(round(scale)), 1)
    line_height = int(26 * scale)
    text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] for line in lines)
    panel_w, panel_h = text_width + int(20 * scale), line_height * len(lines) + int(10 * scale)
    panel_w, panel_h = min(panel_w, width), min(panel_h, height)

    roi = frame[0:panel_h, 0:panel_w]
    frame[0:panel_h, 0:panel_w] = cv2.addWeighted(roi, 0.45, np.zeros_like(roi), 0.55, 0)
    for i, line in enumerate(lines):
        y = int(10 * scale) + line_height * (i + 1) - int(8 * scale)
        cv2.putText(frame, line, (int(10 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def _text_color_for(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black text on light label backgrounds, white on dark ones."""
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)
