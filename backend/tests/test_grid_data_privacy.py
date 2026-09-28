"""Tests for camera registration, sighting persistence, privacy and audit."""

import time

import pytest

from app.grid import repository as repo
from app.grid.privacy import (
    ROLE_PERMISSIONS,
    has_permission,
    mask_plate,
    resolve_role,
)
from app.grid.trajectory import build_trajectory, search_plate


# --------------------------------------------------------------------------
# 9. Camera registration
# --------------------------------------------------------------------------

def test_camera_registration_persists_metadata(camera_factory):
    cam = camera_factory(0, "Camera 01 - Market Road")
    assert cam.id.startswith("cam-")
    assert cam.name == "Camera 01 - Market Road"
    assert cam.status in ("ONLINE", "OFFLINE", "CONNECTING", "PROCESSING", "ERROR")
    assert cam.enabled in (True, False)


def test_camera_crud_update_and_delete(fresh_db, camera_factory):
    cam = camera_factory(0)
    updated = repo.update_camera(fresh_db, cam.id, name="Renamed", enabled=False)
    assert updated.name == "Renamed"
    assert updated.enabled is False
    assert repo.delete_camera(fresh_db, cam.id) is True
    assert repo.get_camera(fresh_db, cam.id) is None


def test_camera_status_transitions(fresh_db, camera_factory):
    cam = camera_factory(0)
    repo.set_camera_status(fresh_db, cam.id, "PROCESSING")
    reloaded = repo.get_camera(fresh_db, cam.id)
    assert reloaded.status == "PROCESSING"
    assert reloaded.last_seen is not None


# --------------------------------------------------------------------------
# 10. Sighting creation
# --------------------------------------------------------------------------

def test_sighting_creation_roundtrip(fresh_db, camera_factory):
    cam = camera_factory(0)
    s = repo.create_sighting(
        fresh_db,
        camera_id=cam.id,
        vehicle_local_track_id=17,
        timestamp=time.time(),
        plate_raw="GJ01AB1234",
        plate_normalized="GJ01AB1234",
        plate_confidence=0.91,
        visual_embedding=[0.1, 0.2, 0.3],
        vehicle_type="car",
        vehicle_color="blue",
        vehicle_bbox=[1, 2, 3, 4],
        plate_bbox=[10, 11, 12, 13],
        detection_confidence=0.9,
        frame_reference="cam-frames/abc.jpg",
    )
    assert s.id.startswith("sgt-")
    assert repo.get_sighting(fresh_db, s.id).plate_normalized == "GJ01AB1234"


def test_sightings_filter_by_camera_and_plate(fresh_db, camera_factory):
    c1 = camera_factory(0, "C1")
    c2 = camera_factory(1, "C2")
    t = time.time()
    for cam, plate in ((c1, "GJ01AB1234"), (c2, "GJ06MN3456")):
        repo.create_sighting(
            fresh_db,
            camera_id=cam.id,
            vehicle_local_track_id=1,
            timestamp=t,
            plate_normalized=plate,
            plate_confidence=0.9,
            visual_embedding=[0.1, 0.2],
        )
    assert len(repo.list_sightings(fresh_db, camera_id=c1.id)) == 1
    assert len(repo.list_sightings(fresh_db, plate="GJ06MN3456")) == 1
    assert repo.count_sightings_for_camera(fresh_db, c1.id) == 1


# --------------------------------------------------------------------------
# 21. Privacy / RBAC
# --------------------------------------------------------------------------

def test_mask_plate_hides_middle():
    masked = mask_plate("GJ01AB1234")
    assert masked.startswith("GJ01")
    assert masked.endswith("34")
    assert "*" in masked
    assert "AB12" not in masked


def test_mask_plate_short_plate():
    masked = mask_plate("GJ01")
    assert masked.endswith("**")
    assert mask_plate("") == ""


def test_role_permissions_matrix():
    assert has_permission("admin", "view_full_plate")
    assert not has_permission("operator", "view_full_plate")
    assert not has_permission("viewer", "write_camera")
    assert has_permission("operator", "control_stream")
    assert "read" in ROLE_PERMISSIONS["viewer"]


def test_resolve_role_unknown_key_is_viewer(monkeypatch):
    monkeypatch.setenv("DRISHTI_ADMIN_KEY", "sekret")
    assert resolve_role("sekret") == "admin"
    assert resolve_role("wrong") == "viewer"
    assert resolve_role(None) == "viewer"


def test_audit_log_records_entries(fresh_db, camera_factory):
    cam = camera_factory(0)
    repo.add_audit_log(
        fresh_db,
        actor="operator",
        role="operator",
        action="vehicle_search",
        target=cam.id,
        detail='{"plate": "GJ01****34"}',
    )
    logs = repo.list_audit_logs(fresh_db)
    assert logs
    assert logs[0].action == "vehicle_search"
    assert logs[0].actor == "operator"
    assert "GJ01****34" in (logs[0].detail or "")


# --------------------------------------------------------------------------
# 12. Trajectory construction + plate search
# --------------------------------------------------------------------------

def _seed_chain(session, cameras, plate="GJ01AB1234", base_t=None, step=16.0, emb=None):
    t = base_t if base_t is not None else time.time()
    emb = emb or [0.2, 0.3, 0.4, 0.5]
    ids = []
    for i, cam in enumerate(cameras):
        s = repo.create_sighting(
            session,
            camera_id=cam.id,
            vehicle_local_track_id=10 + i,
            timestamp=t + i * step,
            plate_raw=plate,
            plate_normalized=plate,
            plate_confidence=0.95,
            visual_embedding=emb,
            vehicle_type="car",
            vehicle_color="blue",
        )
        ids.append(s.id)
    return ids


def test_trajectory_construction_end_to_end(fresh_db, camera_factory):
    from app.grid.fusion import FusionConfig, fusion_engine

    cams = [camera_factory(i, f"C{i}") for i in range(3)]
    _seed_chain(fresh_db, cams)
    fusion_engine.recompute(fresh_db, FusionConfig.from_settings("full"))

    identity = repo.list_identities(fresh_db)[0]
    traj = build_trajectory(fresh_db, identity.id)
    assert traj is not None
    assert [s["camera_id"] for s in traj["route"]] == [c.id for c in cams]
    assert len(traj["transitions"]) >= 2
    assert all(e["decision"] == "ACCEPTED" for e in traj["transitions"])
    # Route geometry must come from the road network, not a bare straight line.
    assert any(e.get("path_geometry") for e in traj["transitions"])


def test_trajectory_never_fabricated_for_single_sighting(fresh_db, camera_factory):
    cam = camera_factory(0)
    repo.create_sighting(
        fresh_db,
        camera_id=cam.id,
        vehicle_local_track_id=1,
        timestamp=time.time(),
        plate_normalized="GJ01AB1234",
        plate_confidence=0.9,
        visual_embedding=[0.1],
    )
    # A single sighting alone must not create a global identity.
    assert repo.list_identities(fresh_db) == []

    # Even when an identity is created explicitly, no transitions may be inferred.
    identity = repo.create_identity(
        fresh_db,
        display_plate="GJ01AB1234",
        status="PROVISIONAL",
        first_seen=time.time(),
        last_seen=time.time(),
        sighting_count=1,
    )
    traj = build_trajectory(fresh_db, identity.id)
    assert traj is not None
    assert traj["transitions"] == []


def test_plate_search_finds_identity(fresh_db, camera_factory):
    from app.grid.fusion import FusionConfig, fusion_engine

    cams = [camera_factory(i, f"C{i}") for i in range(2)]
    _seed_chain(fresh_db, cams)
    fusion_engine.recompute(fresh_db, FusionConfig.from_settings("full"))

    result = search_plate(fresh_db, "GJ01AB1234")
    assert result["matches"]
    top = result["matches"][0]
    assert top["identity"]["display_plate"] == "GJ01AB1234"


def test_plate_search_no_false_match(fresh_db, camera_factory):
    cam = camera_factory(0)
    repo.create_sighting(
        fresh_db,
        camera_id=cam.id,
        vehicle_local_track_id=1,
        timestamp=time.time(),
        plate_normalized="GJ01AB1234",
        plate_confidence=0.9,
        visual_embedding=[0.1],
    )
    result = search_plate(fresh_db, "MH12DE1433")
    assert result["matches"] == []
