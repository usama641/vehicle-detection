"""Optional multi-object tracking support (Ultralytics + ByteTrack).

The actual association of boxes across frames is done by Ultralytics'
built-in ByteTrack implementation (``model.track(..., tracker="bytetrack.yaml")``).
This module provides:

* the tracker configuration and an availability check, and
* ``TrackStatistics``, which turns per-frame track IDs into an *estimate*
  of the number of unique vehicles seen in a video.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import pandas as pd

if TYPE_CHECKING:  # avoid a circular import at runtime
    from detector import Detection

# Bundled with the ultralytics package; no extra file is needed.
TRACKER_CONFIG = "bytetrack.yaml"

TRACKING_DISCLAIMER = (
    "Unique counts are estimated from ByteTrack IDs. An ID can change when a vehicle is occluded, "
    "leaves and re-enters the frame, or is missed for several frames, so one real vehicle may receive "
    "more than one ID (and IDs do not identify real-world vehicles)."
)


def tracking_available() -> tuple[bool, str]:
    """Check that ByteTrack's dependencies are importable.

    Ultralytics tries to pip-install ``lap`` on the fly when it is missing,
    which fails offline, so we check up front and give a clear message.
    """
    try:
        import lap  # noqa: F401
    except ImportError:
        try:
            import scipy  # noqa: F401  (newer ultralytics versions can fall back to scipy)
        except ImportError:
            return False, "Tracking needs the 'lap' package. Install it with: pip install lap"
    try:
        from ultralytics.trackers import byte_tracker  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the installed ultralytics
        return False, f"Ultralytics tracking module is unavailable: {exc}"
    return True, "ByteTrack is available."


@dataclass
class TrackSummary:
    track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    frames_seen: int
    max_confidence: float


class TrackStatistics:
    """Accumulates track IDs over a video to estimate unique vehicle counts."""

    def __init__(self) -> None:
        self._class_votes: dict[int, Counter[str]] = defaultdict(Counter)
        self._first_seen: dict[int, int] = {}
        self._last_seen: dict[int, int] = {}
        self._max_conf: dict[int, float] = defaultdict(float)

    def update(self, detections: Iterable["Detection"], frame_index: int) -> None:
        """Register the tracked detections of one frame (untracked boxes are ignored)."""
        for det in detections:
            if det.track_id is None:
                continue
            tid = det.track_id
            self._class_votes[tid][det.class_name] += 1
            self._first_seen.setdefault(tid, frame_index)
            self._last_seen[tid] = frame_index
            self._max_conf[tid] = max(self._max_conf[tid], det.confidence)

    @property
    def unique_count(self) -> int:
        """Number of distinct track IDs observed."""
        return len(self._class_votes)

    def class_of(self, track_id: int) -> str:
        """Majority-vote class of a track (the detector may flip e.g. car/truck between frames)."""
        return self._class_votes[track_id].most_common(1)[0][0]

    def unique_counts_by_class(self) -> dict[str, int]:
        counts: Counter[str] = Counter(self.class_of(tid) for tid in self._class_votes)
        return dict(counts)

    def summaries(self) -> list[TrackSummary]:
        return [
            TrackSummary(
                track_id=tid,
                class_name=self.class_of(tid),
                first_frame=self._first_seen[tid],
                last_frame=self._last_seen[tid],
                frames_seen=sum(votes.values()),
                max_confidence=round(self._max_conf[tid], 4),
            )
            for tid, votes in sorted(self._class_votes.items())
        ]

    def to_dataframe(self, fps: float | None = None) -> pd.DataFrame:
        """Per-track table; adds first/last-seen timestamps when ``fps`` is known."""
        columns = ["track_id", "class_name", "first_frame", "last_frame", "frames_seen", "max_confidence"]
        df = pd.DataFrame([s.__dict__ for s in self.summaries()], columns=columns)
        if fps and fps > 0 and not df.empty:
            df["first_seen_s"] = (df["first_frame"] / fps).round(2)
            df["last_seen_s"] = (df["last_frame"] / fps).round(2)
        return df
