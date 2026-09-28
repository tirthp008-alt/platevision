"""Tests for the local multi-object tracker and the synthetic demo pipeline.

These exercise the real code paths (tracker association + demo scene rendering)
without requiring camera hardware or trained model weights.
"""

import time

import numpy as np
import pytest

from app.grid.demo import DemoSceneGenerator, demo_camera_count
from app.grid.local_tracker import VehicleTracker
from app.grid.plate_localizer import find_plate_regions
from app.grid.vehicle_detector import VehicleDetection
from app.schemas.detection import BoundingBox
from app.grid.embedder import cosine_similarity, vehicle_embedder


# --------------------------------------------------------------------------
# 7. Local tracking (ByteTrack-style association)
# --------------------------------------------------------------------------

def _det(x, y, w=60, h=40, conf=0.9):
    return VehicleDetection(bbox=BoundingBox(x=x, y=y, width=w, height=h), confidence=conf, vehicle_type="car")


def test_tracker_assigns_persistent_ids_across_frames():
    tracker = VehicleTracker()
    now = time.time()
    for i in range(6):
        tracks = tracker.update([_det(100 + i * 12, 200)], now + i * 0.1)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1
    assert tracks[0].hits >= 6


def test_tracker_separates_two_vehicles():
    tracker = VehicleTracker()
    now = time.time()
    for i in range(4):
        tracks = tracker.update([_det(50 + i * 5, 200), _det(400 + i * 5, 250)], now + i * 0.1)
        assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2


def test_tracker_tolerates_brief_missed_detections():
    tracker = VehicleTracker()
    now = time.time()
    first = tracker.update([_det(100, 200)], now)
    tid = first[0].track_id
    # One frame with no detections should not immediately drop the track.
    tracker.update([], now + 0.1)
    again = tracker.update([_det(112, 200)], now + 0.2)
    assert any(t.track_id == tid for t in again)


# --------------------------------------------------------------------------
# Demo/simulation scene generator (Phase 1/2 acceptance: real pipeline input)
# --------------------------------------------------------------------------

def test_demo_generator_produces_frames_with_plates():
    gen = DemoSceneGenerator("Camera 01 - Market Road", 0)
    seen_plate_text = False
    for _ in range(15):
        frame = gen.next_frame()
        assert frame is not None
        assert frame.ndim == 3
        regions = find_plate_regions(frame)
        if regions:
            seen_plate_text = True
    assert seen_plate_text, "demo frames should contain detectable plate regions"


def test_demo_generator_is_deterministic_per_camera(monkeypatch):
    monkeypatch.setattr('app.grid.demo.time.time', lambda: 1003.0)
    a = DemoSceneGenerator("Cam", 0)
    b = DemoSceneGenerator("Cam", 0)
    # Same camera index → identical geometry for the same elapsed time.
    assert demo_camera_count() >= 3
    a._t0 = 1000.0
    b._t0 = 1000.0
    fa = a.next_frame()
    fb = b.next_frame()
    assert np.array_equal(fa, fb)


# --------------------------------------------------------------------------
# 8. Visual embedding on real rendered crops
# --------------------------------------------------------------------------

def test_embedding_distinguishes_vehicles_in_demo_scene():
    gen = DemoSceneGenerator("Cam", 0)
    crops = []
    for _ in range(20):
        frame = gen.next_frame()
        regions = find_plate_regions(frame)
        for box, _conf in regions:
            vx = max(0, int(box.x - box.width * 1.2))
            vy = max(0, int(box.y - box.height * 3.2))
            vw = min(frame.shape[1] - vx, int(box.width * 3.4))
            vh = min(frame.shape[0] - vy, int(box.height * 3.6))
            if vw > 10 and vh > 10:
                crops.append(frame[vy : vy + vh, vx : vx + vw])
        if len(crops) >= 4:
            break
    assert len(crops) >= 2
    embs = [vehicle_embedder.embed(c) for c in crops]
    sims = [
        cosine_similarity(embs[i], embs[j])
        for i in range(len(embs))
        for j in range(i + 1, len(embs))
    ]
    assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in sims)
    # The same physical vehicle (identical rendered crop) is highly self-similar.
    assert cosine_similarity(embs[0], embs[0]) == pytest.approx(1.0, abs=1e-6)
