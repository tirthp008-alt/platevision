"""API tests for the Drishti Grid endpoints (routing, RBAC, masking, search)."""

import time

import pytest
from fastapi.testclient import TestClient

ADMIN = {"X-API-Key": "test-admin"}
OPERATOR = {"X-API-Key": "test-operator"}
VIEWER = {"X-API-Key": "test-viewer"}


@pytest.fixture()
def client():
    from app.grid.db import reset_for_tests
    from app.main import app

    reset_for_tests()
    with TestClient(app) as c:
        yield c


def _register_camera(client, index=0, name=None, headers=ADMIN):
    return client.post(
        "/api/grid/cameras",
        headers=headers,
        json={
            "name": name or f"Camera {index:02d} - Test",
            "source_type": "demo",
            "source_uri": "demo",
            "latitude": 23.0225 + index * 0.001019,
            "longitude": 72.5714,
            "is_demo": True,
        },
    )


def _seed_sightings(client, cameras, plate="GJ01AB1234"):
    """Insert sightings + run fusion directly against the test DB."""
    from app.grid import repository as repo
    from app.grid.db import session_scope
    from app.grid.fusion import FusionConfig, fusion_engine

    with session_scope() as s:
        t = time.time()
        for i, cam_id in enumerate(cameras):
            repo.create_sighting(
                s,
                camera_id=cam_id,
                vehicle_local_track_id=10 + i,
                timestamp=t + i * 16,
                plate_raw=plate,
                plate_normalized=plate,
                plate_confidence=0.95,
                visual_embedding=[0.2, 0.3, 0.4, 0.5],
                vehicle_type="car",
                vehicle_color="blue",
            )
        fusion_engine.recompute(s, FusionConfig.from_settings("full"))


def test_grid_routes_are_registered(client):
    spec = client.get("/api/openapi.json").json()
    grid_paths = [p for p in spec["paths"] if p.startswith("/api/grid")]
    for required in (
        "/api/grid/cameras",
        "/api/grid/cameras/{camera_id}",
        "/api/grid/cameras/{camera_id}/start",
        "/api/grid/cameras/{camera_id}/stop",
        "/api/grid/cameras/{camera_id}/status",
        "/api/grid/cameras/{camera_id}/events",
        "/api/grid/vehicles/search",
        "/api/grid/vehicles/{vehicle_id}",
        "/api/grid/vehicles/{vehicle_id}/trajectory",
        "/api/grid/analytics/traffic",
        "/api/grid/analytics/heatmap",
    ):
        assert required in grid_paths, f"missing route {required}"


def test_camera_crud_endpoints(client):
    r = _register_camera(client, 0, "Camera 01 - Market Road")
    assert r.status_code == 200, r.text
    cam = r.json()
    assert cam["name"] == "Camera 01 - Market Road"
    assert cam["latitude"] == pytest.approx(23.0225, abs=1e-6)

    listing = client.get("/api/grid/cameras").json()
    assert listing["count"] == 1
    assert listing["cameras"][0]["id"] == cam["id"]

    upd = client.put(f"/api/grid/cameras/{cam['id']}", headers=ADMIN, json={"name": "Renamed"})
    assert upd.status_code == 200
    assert upd.json()["name"] == "Renamed"

    dele = client.delete(f"/api/grid/cameras/{cam['id']}", headers=ADMIN)
    assert dele.status_code == 200
    assert client.get("/api/grid/cameras").json()["count"] == 0


def test_camera_write_requires_permission(client):
    # Viewer lacks write_camera.
    r = _register_camera(client, 0, headers=VIEWER)
    assert r.status_code == 403


def test_camera_lifecycle_status(client):
    cam_id = _register_camera(client, 0, "Camera 01 - Demo").json()["id"]
    start = client.post(f"/api/grid/cameras/{cam_id}/start", headers=OPERATOR)
    assert start.status_code == 200

    status = client.get(f"/api/grid/cameras/{cam_id}/status").json()
    assert status["status"] in ("PROCESSING", "ONLINE", "CONNECTING")

    stop = client.post(f"/api/grid/cameras/{cam_id}/stop", headers=OPERATOR)
    assert stop.status_code == 200


def test_camera_devices_endpoint(client):
    r = client.get("/api/grid/cameras/devices")
    assert r.status_code == 200
    body = r.json()
    assert "devices" in body
    assert "demo" in body["supported_source_types"]


def test_plate_search_masks_for_viewer(client):
    cam_ids = [_register_camera(client, i).json()["id"] for i in range(2)]
    _seed_sightings(client, cam_ids)

    admin = client.get("/api/grid/vehicles/search", params={"plate": "GJ01AB1234"}, headers=ADMIN)
    assert admin.status_code == 200
    assert admin.json()["matches"]
    assert admin.json()["matches"][0]["identity"]["display_plate"] == "GJ01AB1234"

    viewer = client.get("/api/grid/vehicles/search", params={"plate": "GJ01AB1234"}, headers=VIEWER)
    assert viewer.status_code == 200
    assert viewer.json()["matches"]
    assert "*" in viewer.json()["matches"][0]["identity"]["display_plate"]


def test_plate_trajectory_endpoint(client):
    cam_ids = [_register_camera(client, i).json()["id"] for i in range(3)]
    _seed_sightings(client, cam_ids)

    r = client.get("/api/grid/trajectory/plate", params={"plate": "GJ01AB1234"}, headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert len(body["route"]) == 3
    assert len(body["transitions"]) >= 2


def test_vehicle_detail_and_trajectory_endpoints(client):
    cam_ids = [_register_camera(client, i).json()["id"] for i in range(2)]
    _seed_sightings(client, cam_ids)
    vehicles = client.get("/api/grid/vehicles", headers=ADMIN).json()["vehicles"]
    assert vehicles
    vid = vehicles[0]["id"]

    detail = client.get(f"/api/grid/vehicles/{vid}", headers=ADMIN)
    assert detail.status_code == 200
    traj = client.get(f"/api/grid/vehicles/{vid}/trajectory", headers=ADMIN)
    assert traj.status_code == 200
    assert traj.json()["identity"]["id"] == vid


def test_search_for_unknown_plate_returns_no_matches(client):
    _register_camera(client, 0)
    r = client.get("/api/grid/vehicles/search", params={"plate": "MH12DE1433"}, headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["matches"] == []


def test_search_prefers_identity_with_accepted_links(client):
    """A plate seen at several cameras must rank above a lone sighting.

    Both identities carry the same exact plate, so the tie-break is the number
    of accepted cross-camera links in the reconstructed trajectory.
    """
    from app.grid import repository as repo
    from app.grid.db import session_scope
    from app.grid.fusion import FusionConfig, fusion_engine

    cam_ids = [_register_camera(client, i).json()["id"] for i in range(3)]
    with session_scope() as s:
        t = time.time()
        # Multi-camera identity: three sightings chained across the corridor.
        for i, cam_id in enumerate(cam_ids):
            repo.create_sighting(
                s,
                camera_id=cam_id,
                vehicle_local_track_id=100 + i,
                timestamp=t + i * 16,
                plate_raw="GJ01AB1234",
                plate_normalized="GJ01AB1234",
                plate_confidence=0.95,
                visual_embedding=[0.2, 0.3, 0.4, 0.5],
            )
        # Lone single-camera sighting of the same plate, earlier in time.
        repo.create_sighting(
            s,
            camera_id=cam_ids[0],
            vehicle_local_track_id=200,
            timestamp=t - 300,
            plate_raw="GJ01AB1234",
            plate_normalized="GJ01AB1234",
            plate_confidence=0.95,
            visual_embedding=[0.2, 0.3, 0.4, 0.5],
        )
        fusion_engine.recompute(s, FusionConfig.from_settings("full"))

    matches = client.get(
        "/api/grid/vehicles/search", params={"plate": "GJ01AB1234"}, headers=ADMIN
    ).json()["matches"]
    assert matches
    assert matches[0]["accepted_links"] >= 2
    assert matches[0]["sighting_count"] >= 3


def test_unresolved_endpoint(client):
    r = client.get("/api/grid/unresolved")
    assert r.status_code == 200
    assert "unresolved" in r.json()


def test_analytics_endpoints(client):
    cam_ids = [_register_camera(client, i).json()["id"] for i in range(3)]
    _seed_sightings(client, cam_ids)

    traffic = client.get("/api/grid/analytics/traffic").json()
    assert "totals" in traffic
    assert "segments" in traffic

    heat = client.get("/api/grid/analytics/heatmap").json()
    assert "points" in heat or "cells" in heat


def test_fusion_config_endpoint_roundtrip(client):
    cfg = client.get("/api/grid/fusion/config").json()
    assert "tau" in cfg and "delta" in cfg
    assert "lambda_plate" in cfg

    updated = client.put(
        "/api/grid/fusion/config",
        headers=ADMIN,
        json={"tau": 0.75, "delta": 0.2, "lambda_plate": 0.6},
    )
    assert updated.status_code == 200
    cfg2 = client.get("/api/grid/fusion/config").json()
    assert cfg2["tau"] == pytest.approx(0.75)
    assert cfg2["delta"] == pytest.approx(0.2)


def test_system_status_endpoint(client):
    r = client.get("/api/grid/system/status")
    assert r.status_code == 200
    body = r.json()
    assert "aggregate" in body
    assert "vehicle_detector" in body


def test_audit_log_requires_permission(client):
    assert client.get("/api/grid/audit", headers=VIEWER).status_code == 403
    assert client.get("/api/grid/audit", headers=ADMIN).status_code == 200


def test_vehicle_search_is_audited(client):
    cam_ids = [_register_camera(client, i).json()["id"] for i in range(2)]
    _seed_sightings(client, cam_ids)
    client.get("/api/grid/vehicles/search", params={"plate": "GJ01AB1234"}, headers=OPERATOR)
    logs = client.get("/api/grid/audit", headers=ADMIN).json()
    assert any(entry["action"] == "vehicle.search" for entry in logs["audit_logs"])
