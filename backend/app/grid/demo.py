"""Synthetic demo scene generator and shared demo trajectory schedule.

This is the *demo/simulation* path, kept clearly separate from the production
pipeline. It renders a road scene with animated vehicles carrying real Indian
registration plates so the full pipeline can be exercised end-to-end when no
physical camera or trained detector weights are available.

The generator only draws pixels. Everything downstream (vehicle detection,
tracking, plate localization, PlateVision OCR, embeddings, fusion, identity
resolution) is the same real code used for live cameras — nothing is faked.

The schedule makes vehicles follow a shared route through the demo cameras with
timing consistent with the road-network travel-time model, so the cross-camera
fusion engine can genuinely accept or abstain on the resulting sightings.
"""

import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from app.core.config import settings

DEMO_CAMERA_NAMES = [
    "Camera 01 - Market Road",
    "Camera 02 - University Junction",
    "Camera 03 - Flyover",
    "Camera 04 - Stadium Cross",
    "Camera 05 - Highway Toll",
]

# Demo fleet. Plates follow real Indian formats. `route` vehicles traverse
# every demo camera; non-route vehicles appear at a single camera only (these
# never produce cross-camera links, which is useful for demonstrating abstention).
DEMO_FLEET: List[Dict] = [
    {"plate": "GJ01AB1234", "color": (29, 78, 216), "type": "car", "route": True, "offset": 0.0},
    {"plate": "GJ05CD5678", "color": (216, 78, 29), "type": "car", "route": True, "offset": 70.0},
    {"plate": "GJ18EF9012", "color": (40, 140, 60), "type": "truck", "route": True, "offset": 150.0},
    {"plate": "GJ27DS4837", "color": (180, 50, 50), "type": "car", "route": True, "offset": 240.0},
    {"plate": "GJ01XY9999", "color": (30, 41, 59), "type": "car", "route": False, "offset": 30.0},
    {"plate": "GJ06MN3456", "color": (25, 90, 180), "type": "car", "route": False, "offset": 120.0},
]

DEMO_SEGMENT_SECONDS = 30.0
DEMO_DWELL_SECONDS = 14.0


def _segment_seconds() -> float:
    return float(getattr(settings, "GRID_DEMO_SEGMENT_SECONDS", DEMO_SEGMENT_SECONDS))


def _dwell_seconds() -> float:
    return float(getattr(settings, "GRID_DEMO_DWELL_SECONDS", DEMO_DWELL_SECONDS))


def demo_camera_count() -> int:
    return len(DEMO_CAMERA_NAMES)


def _vehicle_progress(veh: Dict, camera_index: int, now: float) -> Optional[float]:
    """Return progress (0..1) for a vehicle at a camera, or None if not visible."""
    n = demo_camera_count()
    seg = _segment_seconds()
    dwell = _dwell_seconds()
    if not veh["route"]:
        target = int(hash(veh["plate"]) % n)
        if camera_index != target:
            return None
        period = n * seg
        phase = (now + veh["offset"]) % period
    else:
        period = n * seg
        phase = (now - (camera_index * seg + veh["offset"])) % period
    if phase <= dwell:
        return phase / dwell
    return None


class DemoSceneGenerator:
    """Renders a synthetic multi-vehicle scene for one demo camera."""

    WIDTH = 960
    HEIGHT = 540

    def __init__(self, camera_name: str, camera_index: int = 0):
        self.camera_name = camera_name
        self.camera_index = camera_index
        self._t0 = time.time()

    def next_frame(self) -> np.ndarray:
        frame = np.full((self.HEIGHT, self.WIDTH, 3), 28, dtype=np.uint8)
        cv2.rectangle(frame, (0, 130), (self.WIDTH, 470), (58, 62, 70), -1)
        cv2.line(frame, (0, 130), (self.WIDTH, 130), (200, 200, 200), 2)
        cv2.line(frame, (0, 470), (self.WIDTH, 470), (200, 200, 200), 2)
        for lane_y in (290, 380):
            for mx in range(0, self.WIDTH, 90):
                cv2.rectangle(frame, (mx, lane_y), (mx + 45, lane_y + 4), (235, 235, 235), -1)

        now = time.time() - self._t0
        for idx, veh in enumerate(DEMO_FLEET):
            progress = _vehicle_progress(veh, self.camera_index, now)
            if progress is None:
                continue
            body_w = 120 if veh["type"] == "motorcycle" else 300
            body_h = 150 if veh["type"] == "truck" else 110
            # Keep the whole vehicle (and therefore its plate) inside the frame
            # for the duration of the pass, so the demo yields clean plate reads.
            max_x = max(1, self.WIDTH - body_w)
            cx = int(progress * max_x)
            cy = 210 + (idx % 3) * 80
            x1, y1 = cx, cy
            x2, y2 = cx + body_w, cy + body_h
            cv2.rectangle(frame, (x1, y1), (x2, y2), veh["color"], -1)
            cv2.rectangle(frame, (x1 + 40, y1 - 32), (x2 - 40, y1), (18, 24, 38), -1)

            pw, ph = 176, 40
            px = x1 + (body_w - pw) // 2
            py = y2 - ph - 6
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (250, 250, 250), -1)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (20, 20, 20), 2)
            cv2.rectangle(frame, (px, py), (px + 16, py + ph), (150, 60, 30), -1)
            cv2.putText(
                frame,
                veh["plate"],
                (px + 20, py + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )
        return frame
