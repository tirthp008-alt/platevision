"""Per-camera multi-object vehicle tracking (ByteTrack-style).

Implements the two-stage association that gives ByteTrack most of its value:

1. associate high-confidence detections with active tracks;
2. associate the remaining low-confidence detections with still-unmatched
   tracks (recovering occluded vehicles before spawning new IDs);
3. start tracks from leftover high-confidence detections.

Track motion is predicted with a constant-velocity model on the bounding-box
centre, which is enough for short camera views and keeps the module dependency
free. The local track id is per-camera only; a global identity is produced by
the fusion engine.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.grid.vehicle_detector import VehicleDetection, bbox_iou
from app.schemas.detection import BoundingBox


@dataclass
class VehicleTrack:
    track_id: int
    bbox: BoundingBox
    confidence: float
    vehicle_type: str = "unknown"
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    state: str = "tracked"  # tracked | lost

    # Emitted sighting bookkeeping (set by the pipeline)
    sighting_id: Optional[str] = None
    sighting_plate_conf: float = 0.0
    plate_readings: List = field(default_factory=list)

    _cx: float = 0.0
    _cy: float = 0.0
    _vx: float = 0.0
    _vy: float = 0.0
    _initialised: bool = False

    # Associated evidence
    plate_hypothesis = None
    embedding: List[float] = field(default_factory=list)
    vehicle_color: str = "unknown"
    crop_ref: Optional[str] = None
    best_crop = None
    sightings_emitted: int = 0

    def _update_velocity(self) -> None:
        cx = self.bbox.x + self.bbox.width / 2.0
        cy = self.bbox.y + self.bbox.height / 2.0
        if not self._initialised:
            self._cx, self._cy = cx, cy
            self._vx = self._vy = 0.0
            self._initialised = True
        else:
            self._vx = 0.6 * self._vx + 0.4 * (cx - self._cx)
            self._vy = 0.6 * self._vy + 0.4 * (cy - self._cy)
            self._cx, self._cy = cx, cy

    def predict(self) -> BoundingBox:
        return BoundingBox(
            x=int(self._cx + self._vx - self.bbox.width / 2.0),
            y=int(self._cy + self._vy - self.bbox.height / 2.0),
            width=self.bbox.width,
            height=self.bbox.height,
        )

    def apply(self, det: VehicleDetection, now: float) -> None:
        self.bbox = det.bbox
        self.confidence = det.confidence
        if det.vehicle_type != "unknown":
            self.vehicle_type = det.vehicle_type
        self.hits += 1
        self.time_since_update = 0
        self.last_seen = now
        self.state = "tracked"
        self._update_velocity()

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.state = "lost"

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


class VehicleTracker:
    """ByteTrack-style tracker for a single camera."""

    def __init__(
        self,
        track_thresh: float = 0.45,
        match_thresh: float = 0.30,
        low_thresh: float = 0.10,
        max_time_lost: int = 45,
        min_hits_to_confirm: int = 2,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.max_time_lost = max_time_lost
        self.min_hits_to_confirm = min_hits_to_confirm
        self.tracks: Dict[int, VehicleTrack] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    @staticmethod
    def _associate(
        tracks: List[VehicleTrack], dets: List[VehicleDetection], iou_thr: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU association between predicted track boxes and detections."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        iou_matrix = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for i, t in enumerate(tracks):
            pred = t.predict()
            for j, d in enumerate(dets):
                iou_matrix[i, j] = bbox_iou(pred, d.bbox)

        matches: List[Tuple[int, int]] = []
        used_t, used_d = set(), set()
        while True:
            if iou_matrix.size == 0:
                break
            idx = np.argmax(iou_matrix)
            i, j = np.unravel_index(idx, iou_matrix.shape)
            if iou_matrix[i, j] < iou_thr:
                break
            matches.append((int(i), int(j)))
            used_t.add(int(i))
            used_d.add(int(j))
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
        unmatched_d = [j for j in range(len(dets)) if j not in used_d]
        return matches, unmatched_t, unmatched_d

    def update(self, detections: List[VehicleDetection], now: Optional[float] = None) -> List[VehicleTrack]:
        now = now or time.time()
        for t in self.tracks.values():
            t.age += 1

        high = [d for d in detections if d.confidence >= self.track_thresh]
        low = [d for d in detections if self.low_thresh <= d.confidence < self.track_thresh]

        active = [t for t in self.tracks.values() if t.time_since_update == 0]
        lost = [t for t in self.tracks.values() if t.time_since_update > 0]

        # Stage 1: high-confidence detections vs all tracks
        pool = active + lost
        matches, unmatched_tracks, unmatched_high = self._associate(pool, high, self.match_thresh)
        for ti, dj in matches:
            pool[ti].apply(high[dj], now)

        # Stage 2: low-confidence detections vs remaining (mostly active) tracks
        remaining = [pool[i] for i in unmatched_tracks]
        matches2, unmatched_remaining, _ = self._associate(remaining, low, 0.15)
        for ti, dj in matches2:
            remaining[ti].apply(low[dj], now)

        tracked_ids = {id(t) for t in pool if t.time_since_update == 0}

        # Stage 3: new tracks from unmatched high-confidence detections
        for dj in unmatched_high:
            det = high[dj]
            track = VehicleTrack(
                track_id=self.next_id,
                bbox=det.bbox,
                confidence=det.confidence,
                vehicle_type=det.vehicle_type,
                first_seen=now,
                last_seen=now,
            )
            track._update_velocity()
            self.tracks[self.next_id] = track
            self.next_id += 1

        # Mark missed and prune
        for t in self.tracks.values():
            if t.time_since_update > 0:
                t.mark_missed()
        dead = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_time_lost]
        for tid in dead:
            del self.tracks[tid]

        # Return currently tracked (updated this frame) tracks
        return [t for t in self.tracks.values() if t.time_since_update == 0]
