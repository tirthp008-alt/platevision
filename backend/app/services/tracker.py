"""High-performance ByteTrack / IoU Multi-Object Tracker for vehicle license plates.
Tracks plate bounding boxes across stream frames and selectively triggers OCR on high-confidence frames.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.schemas.detection import BoundingBox


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculates IoU between two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / float(union_area) if union_area > 0 else 0.0


class PlateTrack:
    def __init__(self, track_id: int, bbox: BoundingBox, confidence: float):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        
        # OCR caching
        self.ocr_count = 0  # Number of times OCR was executed
        self.best_ocr_text: str = ""
        self.best_formatted_text: str = ""
        self.best_format_status: str = "uncertain"
        self.best_ocr_confidence: float = 0.0
        self.best_crop_b64: str = ""
        self.best_box_area: int = bbox.width * bbox.height
        self.last_seen: float = time.time()

    def update(self, bbox: BoundingBox, confidence: float):
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.last_seen = time.time()

    def should_trigger_ocr(self, max_ocr_runs: int = 3) -> bool:
        """Determines if OCR should be run for this track on the current frame.
        Triggers OCR on the first detection, and subsequent frames only if the plate is larger/clearer.
        """
        if self.ocr_count == 0:
            return True
        if self.ocr_count >= max_ocr_runs:
            return False
        # Trigger again if plate area grew by > 20% (vehicle approached closer)
        current_area = self.bbox.width * self.bbox.height
        if current_area > self.best_box_area * 1.20 and self.confidence >= 0.40:
            return True
        return False

    def update_ocr_result(
        self,
        raw_text: str,
        formatted_text: str,
        format_status: str,
        confidence: float,
        crop_b64: str,
    ):
        self.ocr_count += 1
        current_area = self.bbox.width * self.bbox.height
        
        # Update if higher confidence or higher quality format
        is_better = (
            (format_status == "valid" and self.best_format_status != "valid")
            or (confidence > self.best_ocr_confidence)
            or (not self.best_ocr_text and raw_text)
        )
        
        if is_better or self.ocr_count == 1:
            self.best_ocr_text = raw_text
            self.best_formatted_text = formatted_text
            self.best_format_status = format_status
            self.best_ocr_confidence = confidence
            self.best_crop_b64 = crop_b64
            self.best_box_area = current_area


class PlateTracker:
    """ByteTrack / SORT-inspired lightweight spatial tracker for video and CCTV stream feeds."""

    def __init__(self, max_age: int = 15, iou_threshold: float = 0.30):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.tracks: Dict[int, PlateTrack] = {}

    def update(self, detections: List[Tuple[BoundingBox, float]]) -> List[PlateTrack]:
        """Matches incoming detections to existing tracks using spatial IoU association."""
        # Increment age for all existing tracks
        for track in self.tracks.values():
            track.age += 1
            track.time_since_update += 1

        active_tracks: List[PlateTrack] = []
        unmatched_detections = list(range(len(detections)))

        if self.tracks and detections:
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

            for i, tid in enumerate(track_ids):
                t_box = (
                    self.tracks[tid].bbox.x,
                    self.tracks[tid].bbox.y,
                    self.tracks[tid].bbox.width,
                    self.tracks[tid].bbox.height,
                )
                for j, (det_box, _) in enumerate(detections):
                    d_box = (det_box.x, det_box.y, det_box.width, det_box.height)
                    iou_matrix[i, j] = calculate_iou(t_box, d_box)

            # Greedy Hungarian / IoU matching
            matched_tracks = set()
            matched_dets = set()

            while True:
                max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0
                if max_val < self.iou_threshold:
                    break

                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                det_box, det_conf = detections[j]

                self.tracks[tid].update(det_box, det_conf)
                matched_tracks.add(tid)
                matched_dets.add(j)

                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0

            unmatched_detections = [j for j in range(len(detections)) if j not in matched_dets]

        # Create new tracks for unmatched detections
        for j in unmatched_detections:
            det_box, det_conf = detections[j]
            new_track = PlateTrack(self.next_track_id, det_box, det_conf)
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # Prune dead tracks
        dead_ids = [
            tid
            for tid, track in self.tracks.items()
            if track.time_since_update > self.max_age
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        # Return currently visible tracks (updated on this frame)
        return [track for track in self.tracks.values() if track.time_since_update == 0]
