"""Drishti Grid API endpoints: cameras, vehicles, trajectories, analytics.

These endpoints implement the city-wide multi-camera ANPR trajectory tracking
contract. All mutating/privacy-sensitive operations are permission checked and
audited. Plate text is masked for callers without the ``view_full_plate``
permission.
"""

import time
from typing import List, Optional

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.grid import repository as repo
from app.grid.analytics import heatmap as analytics_heatmap
from app.grid.analytics import traffic_analytics
from app.grid.camera_discovery import discover_cameras
from app.grid.camera_manager import camera_manager
from app.grid.db import get_db, session_scope
from app.grid.embedder import vehicle_embedder
from app.grid.fusion import FusionConfig, fusion_engine
from app.grid.frame_store import frame_store
from app.grid.metrics import LatencyTimer, metrics
from app.grid.privacy import enforce, get_access_context
from app.grid.road_network import road_network
from app.grid.trajectory import build_trajectory, search_plate
from app.schemas.grid import (
    CameraCreate,
    CameraUpdate,
    DemoSetupRequest,
    ErrorResponse,
    FusionConfigRequest,
    FusionRecomputeRequest,
    StreamProbeRequest,
)

router = APIRouter(prefix="/grid", tags=["Drishti Grid"])


# --------------------------------------------------------------------------
# Camera discovery & probing
# --------------------------------------------------------------------------

@router.get("/cameras/devices")
async def list_camera_devices():
    """Enumerate local USB/built-in camera devices available to this machine."""
    devices = discover_cameras()
    return {
        "devices": devices,
        "supported_source_types": ["device", "rtsp", "file", "demo"],
        "note": "On headless servers no local devices may appear; RTSP/file/demo sources still work.",
    }


@router.post("/cameras/probe")
async def probe_camera_source(
    payload: StreamProbeRequest,
    ctx: dict = Depends(get_access_context),
):
    """Test whether a camera source is reachable before registering it."""
    enforce(ctx, "write_camera")
    if payload.source_type == "demo":
        return {"status": "ok", "message": "Demo source is always available.", "source_type": "demo"}

    start = time.time()
    try:
        if payload.source_type == "device":
            index = int((payload.device_id or "device:0").replace("device:", ""))
            cap = cv2.VideoCapture(index)
        else:
            import os

            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            if not payload.source_uri:
                raise ValueError("source_uri is required for rtsp/http/file sources")
            cap = cv2.VideoCapture(payload.source_uri)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        opened = cap.isOpened()
        frame = None
        if opened:
            ok, frame = cap.read()
            opened = bool(ok and frame is not None)
        cap.release()
        latency_ms = int((time.time() - start) * 1000)
        if not opened:
            return {
                "status": "error",
                "message": "Source could not be opened or produced no frames.",
                "latency_ms": latency_ms,
            }
        h, w = frame.shape[:2]
        return {
            "status": "ok",
            "message": f"Source reachable ({w}x{h}).",
            "resolution": f"{w}x{h}",
            "latency_ms": latency_ms,
        }
    except Exception as e:
        return {"status": "error", "message": f"Probe failed: {e}", "latency_ms": int((time.time() - start) * 1000)}


# --------------------------------------------------------------------------
# Camera CRUD + lifecycle
# --------------------------------------------------------------------------

@router.get("/cameras")
async def list_cameras(db: Session = Depends(get_db)):
    cameras = repo.list_cameras(db)
    statuses = camera_manager.all_status()
    out = []
    for cam in cameras:
        d = cam.to_dict()
        d["runtime"] = statuses.get(cam.id)
        d["sighting_count"] = repo.count_sightings_for_camera(db, cam.id)
        out.append(d)
    return {"cameras": out, "count": len(out)}


@router.post("/cameras", responses={403: {"model": ErrorResponse}})
async def create_camera(
    payload: CameraCreate,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    enforce(ctx, "write_camera")
    with LatencyTimer("api.create_camera"):
        cam = repo.create_camera(
            db,
            name=payload.name,
            source_type=payload.source_type,
            source_uri=payload.source_uri,
            device_id=payload.device_id,
            latitude=payload.latitude,
            longitude=payload.longitude,
            road_segment_id=payload.road_segment_id,
            enabled=payload.enabled,
            status="OFFLINE",
            is_demo=payload.is_demo,
        )
        repo.add_audit_log(
            db,
            action="camera.create",
            actor=ctx["actor"],
            role=ctx["role"],
            target=cam.id,
            detail=f"name={payload.name} source={payload.source_type}",
            client_ip=ctx["client_ip"],
        )
        db.commit()
        road_network.register_cameras(
            [{"id": c.id, "latitude": c.latitude, "longitude": c.longitude} for c in repo.list_cameras(db)]
        )
        return cam.to_dict()


@router.put("/cameras/{camera_id}")
async def update_camera(
    camera_id: str,
    payload: CameraUpdate,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    enforce(ctx, "write_camera")
    fields = payload.model_dump(exclude_unset=True)
    cam = repo.update_camera(db, camera_id, **fields)
    if cam is None:
        raise HTTPException(status_code=404, detail={"code": "CAMERA_NOT_FOUND", "message": "No such camera"})
    repo.add_audit_log(
        db, action="camera.update", actor=ctx["actor"], role=ctx["role"],
        target=camera_id, detail=str(fields), client_ip=ctx["client_ip"],
    )
    db.commit()
    road_network.register_cameras(
        [{"id": c.id, "latitude": c.latitude, "longitude": c.longitude} for c in repo.list_cameras(db)]
    )
    return cam.to_dict()


@router.delete("/cameras/{camera_id}")
async def delete_camera(
    camera_id: str,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    enforce(ctx, "write_camera")
    camera_manager.remove(camera_id)
    ok = repo.delete_camera(db, camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail={"code": "CAMERA_NOT_FOUND", "message": "No such camera"})
    repo.add_audit_log(
        db, action="camera.delete", actor=ctx["actor"], role=ctx["role"], target=camera_id,
        client_ip=ctx["client_ip"],
    )
    db.commit()
    road_network.register_cameras(
        [{"id": c.id, "latitude": c.latitude, "longitude": c.longitude} for c in repo.list_cameras(db)]
    )
    return {"deleted": camera_id}


@router.post("/cameras/{camera_id}/start")
async def start_camera(
    camera_id: str,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    enforce(ctx, "control_stream")
    with LatencyTimer("api.camera_start"):
        cam = repo.get_camera(db, camera_id)
        if cam is None:
            raise HTTPException(status_code=404, detail={"code": "CAMERA_NOT_FOUND", "message": "No such camera"})
        if not cam.enabled:
            raise HTTPException(
                status_code=400,
                detail={"code": "CAMERA_DISABLED", "message": "Camera is disabled; enable it before starting."},
            )
        # Provide the demo index so demo cameras show a consistent shared route.
        camera_row = cam.to_dict()
        if cam.is_demo:
            demo_ids = [c.id for c in repo.list_cameras(db) if c.is_demo]
            camera_row["demo_index"] = demo_ids.index(camera_id) if camera_id in demo_ids else 0
        repo.set_camera_status(db, camera_id, "CONNECTING")
        repo.add_audit_log(
            db, action="camera.start", actor=ctx["actor"], role=ctx["role"], target=camera_id,
            client_ip=ctx["client_ip"],
        )
        db.commit()
        runtime = camera_manager.start(camera_row)
        fusion_engine.mark_dirty()
        return {"camera_id": camera_id, "runtime": runtime}


@router.post("/cameras/{camera_id}/stop")
async def stop_camera(
    camera_id: str,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    enforce(ctx, "control_stream")
    cam = repo.get_camera(db, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail={"code": "CAMERA_NOT_FOUND", "message": "No such camera"})
    runtime = camera_manager.stop(camera_id)
    repo.set_camera_status(db, camera_id, "OFFLINE")
    repo.add_audit_log(
        db, action="camera.stop", actor=ctx["actor"], role=ctx["role"], target=camera_id,
        client_ip=ctx["client_ip"],
    )
    db.commit()
    return {"camera_id": camera_id, "runtime": runtime}


@router.get("/cameras/{camera_id}/status")
async def camera_status(
    camera_id: str,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    cam = repo.get_camera(db, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail={"code": "CAMERA_NOT_FOUND", "message": "No such camera"})
    runtime = camera_manager.status(camera_id) or {
        "camera_id": camera_id, "name": cam.name, "status": cam.status, "is_running": False,
    }
    recent = repo.list_sightings(db, camera_id=camera_id, limit=10)
    runtime["recent_sightings"] = [
        {
            "sighting_id": s.id,
            "plate": s.plate_normalized if ctx["can_view_full_plate"] else _mask(s.plate_normalized),
            "plate_confidence": round(s.plate_confidence, 3),
            "timestamp": s.timestamp,
            "track_id": s.vehicle_local_track_id,
        }
        for s in recent
    ]
    runtime["sighting_count"] = repo.count_sightings_for_camera(db, camera_id)
    return runtime


@router.get("/cameras/{camera_id}/events")
async def camera_events(
    camera_id: str,
    limit: int = Query(50, ge=1, le=200),
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    sightings = repo.list_sightings(db, camera_id=camera_id, limit=limit)
    events = [s.to_dict(mask_plate=not ctx["can_view_full_plate"]) for s in sightings]
    runtime = camera_manager.status(camera_id)
    return {"camera_id": camera_id, "events": events, "runtime": runtime, "count": len(events)}


@router.get("/cameras/{camera_id}/sightings")
async def camera_sightings(
    camera_id: str,
    limit: int = Query(100, ge=1, le=500),
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    sightings = repo.list_sightings(db, camera_id=camera_id, limit=limit)
    return {
        "camera_id": camera_id,
        "sightings": [s.to_dict(mask_plate=not ctx["can_view_full_plate"]) for s in sightings],
    }


@router.get("/cameras/{camera_id}/frame")
async def camera_frame(camera_id: str):
    """Return the latest annotated JPEG snapshot for a camera (never raw video)."""
    data = frame_store.get_frame(camera_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "NO_FRAME", "message": "No frame available yet for this camera."},
        )
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------------
# Vehicle search & trajectories
# --------------------------------------------------------------------------

@router.get("/vehicles/search")
async def vehicles_search(
    plate: str = Query(..., min_length=2, description="Plate to search for, e.g. GJ01AB1234"),
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    with LatencyTimer("api.vehicle_search"):
        result = search_plate(db, plate, limit=20)
    # Audit every plate search (privacy requirement).
    repo.add_audit_log(
        db,
        action="vehicle.search",
        actor=ctx["actor"],
        role=ctx["role"],
        target=result["query"]["normalized"],
        detail=f"query={plate} matches={len(result['matches'])}",
        client_ip=ctx["client_ip"],
    )
    db.commit()
    if not ctx["can_view_full_plate"]:
        result = _mask_search(result)
    return result


@router.get("/vehicles/{vehicle_id}")
async def vehicle_detail(
    vehicle_id: str,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    identity = repo.get_identity(db, vehicle_id)
    if identity is None:
        raise HTTPException(status_code=404, detail={"code": "VEHICLE_NOT_FOUND", "message": "No such vehicle identity"})
    sightings = repo.sightings_for_identity(db, vehicle_id)
    return {
        "identity": identity.to_dict(mask_plate=not ctx["can_view_full_plate"]),
        "sightings": [s.to_dict(mask_plate=not ctx["can_view_full_plate"]) for s in sightings],
    }


@router.get("/vehicles/{vehicle_id}/trajectory")
async def vehicle_trajectory(
    vehicle_id: str,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    traj = build_trajectory(db, vehicle_id)
    if traj is None:
        raise HTTPException(status_code=404, detail={"code": "VEHICLE_NOT_FOUND", "message": "No such vehicle identity"})
    if not ctx["can_view_full_plate"]:
        traj = _mask_trajectory(traj)
    return traj


@router.get("/trajectory/plate")
async def trajectory_by_plate(
    plate: str = Query(..., min_length=2),
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    with LatencyTimer("api.trajectory_plate"):
        result = search_plate(db, plate, limit=5)
        if not result["matches"]:
            repo.add_audit_log(
                db, action="vehicle.search", actor=ctx["actor"], role=ctx["role"],
                target=result["query"]["normalized"], detail="no match", client_ip=ctx["client_ip"],
            )
            db.commit()
            return {
                "query": result["query"],
                "found": False,
                "matches": [],
                "message": "No vehicle with sufficient evidence matches this plate.",
                "unresolved_links": result["unresolved_links"],
            }
        best_id = result["matches"][0]["identity"]["id"]
        traj = build_trajectory(db, best_id)
    repo.add_audit_log(
        db, action="vehicle.search", actor=ctx["actor"], role=ctx["role"],
        target=result["query"]["normalized"], detail=f"best={best_id}", client_ip=ctx["client_ip"],
    )
    db.commit()
    if traj is None:
        return {"query": result["query"], "found": False, "matches": [], "message": "No trajectory found."}
    traj["found"] = True
    traj["matches"] = result["matches"]
    if not ctx["can_view_full_plate"]:
        traj = _mask_trajectory(traj)
    return traj


@router.get("/vehicles")
async def list_vehicles(
    limit: int = Query(100, ge=1, le=500),
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    identities = repo.list_identities(db, limit=limit)
    return {
        "vehicles": [i.to_dict(mask_plate=not ctx["can_view_full_plate"]) for i in identities],
        "count": len(identities),
    }


@router.get("/unresolved")
async def unresolved_links(
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Candidate links the confidence gate refused to commit to (human review)."""
    edges = repo.list_unresolved_edges(db, limit=limit)
    return {
        "unresolved": [e.to_dict() for e in edges],
        "count": len(edges),
        "note": "These links were abstained/rejected; they are deliberately not part of any trajectory.",
    }


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

@router.get("/evidence/{frame_reference}")
async def get_evidence(frame_reference: str):
    data = frame_store.get_crop(frame_reference)
    if data is None:
        raise HTTPException(status_code=404, detail={"code": "EVIDENCE_NOT_FOUND", "message": "Evidence crop expired or not found."})
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------

@router.get("/analytics/traffic")
async def analytics_traffic(
    window_seconds: float = Query(3600.0, gt=0, le=86400),
    db: Session = Depends(get_db),
):
    return traffic_analytics(db, window_seconds=window_seconds)


@router.get("/analytics/heatmap")
async def analytics_heatmap_endpoint(
    window_seconds: float = Query(3600.0, gt=0, le=86400),
    db: Session = Depends(get_db),
):
    return analytics_heatmap(db, window_seconds=window_seconds)


@router.get("/system/status")
async def system_status(db: Session = Depends(get_db)):
    cameras = repo.list_cameras(db)
    aggregate = camera_manager.aggregate()
    return {
        "cameras": len(cameras),
        "cameras_enabled": sum(1 for c in cameras if c.enabled),
        "cameras_processing": aggregate["cameras_processing"],
        "aggregate": aggregate,
        "fusion": fusion_engine.metrics,
        "database": settings.GRID_DATABASE_URL.split("://")[0],
        "detector": type(__import__("app.services.detector.factory", fromlist=["get_detector"]).get_detector()).__name__,
        "vehicle_detector": settings.VEHICLE_DETECTOR_BACKEND,
        "reid_backend": vehicle_embedder.active,
        "demo_mode": settings.GRID_DEMO_MODE,
        "event_backend": settings.EVENT_BACKEND,
        "uptime_note": "Timestamps are server epoch seconds.",
    }


@router.get("/metrics")
async def get_metrics():
    return metrics.snapshot()


# --------------------------------------------------------------------------
# Fusion configuration
# --------------------------------------------------------------------------

@router.get("/fusion/config")
async def get_fusion_config():
    cfg = FusionConfig.from_settings()
    return {
        "lambda_plate": cfg.lambda_plate,
        "lambda_visual": cfg.lambda_visual,
        "lambda_time": cfg.lambda_time,
        "sigma_t": cfg.sigma_t,
        "tau": cfg.tau,
        "delta": cfg.delta,
        "mode": cfg.mode,
        "modes_available": ["plate_only", "plate_visual", "full"],
        "equation": "S_ij = λp·s_plate + λv·cos(vi,vj) − λt·|Δt − t_G|/σt",
        "acceptance": "S(1) ≥ τ AND S(1) − S(2) ≥ δ",
    }


@router.put("/fusion/config")
async def update_fusion_config(
    payload: FusionConfigRequest,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    enforce(ctx, "manage_fusion")
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    for key, value in updates.items():
        setattr(settings, f"FUSION_{key.upper()}", value)
    repo.add_audit_log(
        db, action="fusion.config", actor=ctx["actor"], role=ctx["role"], detail=str(updates),
        client_ip=ctx["client_ip"],
    )
    db.commit()
    fusion_engine.mark_dirty()
    return await get_fusion_config()


@router.post("/fusion/recompute")
async def recompute_fusion(
    payload: FusionRecomputeRequest,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    enforce(ctx, "manage_fusion")
    config = FusionConfig.from_settings(mode=payload.mode)
    with LatencyTimer("api.fusion_recompute"):
        result = fusion_engine.recompute(db, config=config)
    repo.add_audit_log(
        db, action="fusion.recompute", actor=ctx["actor"], role=ctx["role"],
        detail=f"mode={config.mode}", client_ip=ctx["client_ip"],
    )
    db.commit()
    return {"mode": config.mode, "metrics": result}


# --------------------------------------------------------------------------
# Demo / simulation setup
# --------------------------------------------------------------------------

@router.post("/demo/setup")
async def setup_demo(
    payload: DemoSetupRequest,
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    """Register a set of demo cameras on a synthetic road network and optionally start them."""
    enforce(ctx, "write_camera")
    from app.grid.demo import DEMO_CAMERA_NAMES

    # Only create cameras that do not already exist (by name).
    existing = {c.name: c for c in repo.list_cameras(db)}
    # Demo cameras sit on a short corridor so the road model's expected travel
    # time matches the demo segment time. This keeps accepted cross-camera links
    # physically plausible (the demo is clearly labelled as synthetic).
    lat0 = payload.center_latitude
    lon0 = payload.center_longitude
    step = settings.GRID_DEMO_CORRIDOR_STEP_DEG
    created = []
    demo_cameras = []
    for i, name in enumerate(DEMO_CAMERA_NAMES):
        if name in existing:
            cam = existing[name]
        else:
            cam = repo.create_camera(
                db,
                name=name,
                source_type="demo",
                source_uri="demo",
                latitude=lat0 + i * step,
                longitude=lon0,
                is_demo=True,
                status="OFFLINE",
            )
            created.append(cam.id)
        demo_cameras.append(cam)
    db.commit()

    road_network.register_cameras(
        [{"id": c.id, "latitude": c.latitude, "longitude": c.longitude} for c in repo.list_cameras(db)]
    )

    runtime = []
    if payload.start_processing:
        for idx, cam in enumerate(demo_cameras):
            camera_row = cam.to_dict()
            camera_row["demo_index"] = idx
            repo.set_camera_status(db, cam.id, "CONNECTING")
            db.commit()
            runtime.append(camera_manager.start(camera_row))

    repo.add_audit_log(
        db, action="demo.setup", actor=ctx["actor"], role=ctx["role"],
        detail=f"created={len(created)} started={len(runtime)}", client_ip=ctx["client_ip"],
    )
    db.commit()
    fusion_engine.mark_dirty()
    return {
        "created_camera_ids": created,
        "demo_cameras": [
            {"id": c.id, "name": c.name, "latitude": c.latitude, "longitude": c.longitude}
            for c in demo_cameras
        ],
        "runtime": runtime,
        "note": "Demo cameras render synthetic traffic and run the real detection/OCR/fusion pipeline.",
    }


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------

@router.get("/audit")
async def get_audit_log(
    limit: int = Query(200, ge=1, le=1000),
    ctx: dict = Depends(get_access_context),
    db: Session = Depends(get_db),
):
    enforce(ctx, "view_audit")
    logs = repo.list_audit_logs(db, limit=limit)
    return {"audit_logs": [l.to_dict() for l in logs], "count": len(logs)}


# --------------------------------------------------------------------------
# Masking helpers
# --------------------------------------------------------------------------

def _mask(plate: str) -> str:
    from app.grid.privacy import mask_plate

    return mask_plate(plate) if plate else ""


def _mask_search(result: dict) -> dict:
    for m in result.get("matches", []):
        ident = m.get("identity", {})
        if ident.get("display_plate"):
            ident["display_plate"] = _mask(ident["display_plate"])
    for s in result.get("raw_sightings", []):
        if s.get("plate"):
            s["plate"] = _mask(s["plate"])
        if s.get("plate_raw"):
            s["plate_raw"] = _mask(s["plate_raw"])
    result["plate_masked"] = True
    return result


def _mask_trajectory(traj: dict) -> dict:
    ident = traj.get("identity", {})
    if ident.get("display_plate"):
        ident["display_plate"] = _mask(ident["display_plate"])
    for s in traj.get("route", []):
        if s.get("plate"):
            s["plate"] = _mask(s["plate"])
        if s.get("plate_raw"):
            s["plate_raw"] = _mask(s["plate_raw"])
    traj["plate_masked"] = True
    return traj
