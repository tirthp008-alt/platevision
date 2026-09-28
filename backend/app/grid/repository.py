"""Repository layer: all persistence access for Drishti Grid.

Keeping every query here means the fusion/ingestion code has no SQLAlchemy
dependency and can be unit-tested against an in-memory database.
"""

import time
import uuid
from typing import Iterable, List, Optional, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.grid.models import (
    AuditLog,
    Camera,
    PlateObservation,
    ProcessingSession,
    TrajectoryEdge,
    VehicleIdentity,
    VehicleSighting,
)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------

def create_camera(session: Session, **fields) -> Camera:
    cam = Camera(id=fields.pop("id", None) or new_id("cam"), **fields)
    session.add(cam)
    session.flush()
    return cam


def get_camera(session: Session, camera_id: str) -> Optional[Camera]:
    return session.get(Camera, camera_id)


def list_cameras(session: Session) -> List[Camera]:
    return list(session.scalars(select(Camera).order_by(Camera.created_at)))


def update_camera(session: Session, camera_id: str, **fields) -> Optional[Camera]:
    cam = session.get(Camera, camera_id)
    if cam is None:
        return None
    for k, v in fields.items():
        if v is not None and hasattr(cam, k):
            setattr(cam, k, v)
    cam.updated_at = time.time()
    session.flush()
    return cam


def delete_camera(session: Session, camera_id: str) -> bool:
    cam = session.get(Camera, camera_id)
    if cam is None:
        return False
    session.delete(cam)
    session.flush()
    return True


def set_camera_status(
    session: Session, camera_id: str, status: str, error: Optional[str] = None
) -> None:
    cam = session.get(Camera, camera_id)
    if cam is None:
        return
    cam.status = status
    cam.last_error = error
    if status in ("ONLINE", "PROCESSING"):
        cam.last_seen = time.time()
    session.flush()


# --------------------------------------------------------------------------
# Sightings
# --------------------------------------------------------------------------

def create_sighting(session: Session, **fields) -> VehicleSighting:
    fields.setdefault("id", new_id("sgt"))
    sighting = VehicleSighting(**fields)
    session.add(sighting)
    session.flush()
    return sighting


def get_sighting(session: Session, sighting_id: str) -> Optional[VehicleSighting]:
    return session.get(VehicleSighting, sighting_id)


def list_sightings(
    session: Session,
    camera_id: Optional[str] = None,
    plate: Optional[str] = None,
    since: Optional[float] = None,
    limit: int = 200,
) -> List[VehicleSighting]:
    stmt = select(VehicleSighting)
    if camera_id:
        stmt = stmt.where(VehicleSighting.camera_id == camera_id)
    if plate:
        stmt = stmt.where(VehicleSighting.plate_normalized == plate)
    if since is not None:
        stmt = stmt.where(VehicleSighting.timestamp >= since)
    stmt = stmt.order_by(VehicleSighting.timestamp.desc()).limit(limit)
    return list(session.scalars(stmt))


def recent_unassigned_sightings(session: Session, limit: int = 500) -> List[VehicleSighting]:
    stmt = (
        select(VehicleSighting)
        .order_by(VehicleSighting.timestamp.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def sightings_for_identity(session: Session, identity_id: str) -> List[VehicleSighting]:
    stmt = (
        select(VehicleSighting)
        .where(VehicleSighting.vehicle_identity_id == identity_id)
        .order_by(VehicleSighting.timestamp)
    )
    return list(session.scalars(stmt))


def count_sightings_for_camera(session: Session, camera_id: str) -> int:
    return int(
        session.scalar(
            select(func.count()).select_from(VehicleSighting).where(
                VehicleSighting.camera_id == camera_id
            )
        )
        or 0
    )


# --------------------------------------------------------------------------
# Plate observations
# --------------------------------------------------------------------------

def create_plate_observation(session: Session, **fields) -> PlateObservation:
    fields.setdefault("id", new_id("pob"))
    obs = PlateObservation(**fields)
    session.add(obs)
    session.flush()
    return obs


def observations_for_track(
    session: Session, camera_id: str, track_id: int, limit: int = 50
) -> List[PlateObservation]:
    stmt = (
        select(PlateObservation)
        .where(
            PlateObservation.camera_id == camera_id,
            PlateObservation.vehicle_local_track_id == track_id,
        )
        .order_by(PlateObservation.timestamp.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


# --------------------------------------------------------------------------
# Vehicle identities
# --------------------------------------------------------------------------

def create_identity(session: Session, **fields) -> VehicleIdentity:
    fields.setdefault("id", new_id("veh"))
    identity = VehicleIdentity(**fields)
    session.add(identity)
    session.flush()
    return identity


def get_identity(session: Session, identity_id: str) -> Optional[VehicleIdentity]:
    return session.get(VehicleIdentity, identity_id)


def find_identities_by_plate(session: Session, plate: str) -> List[VehicleIdentity]:
    if not plate:
        return []
    stmt = select(VehicleIdentity).where(VehicleIdentity.display_plate == plate)
    return list(session.scalars(stmt))


def list_identities(session: Session, limit: int = 200) -> List[VehicleIdentity]:
    stmt = select(VehicleIdentity).order_by(VehicleIdentity.last_seen.desc().nullslast()).limit(limit)
    return list(session.scalars(stmt))


# --------------------------------------------------------------------------
# Trajectory edges
# --------------------------------------------------------------------------

def create_edge(session: Session, **fields) -> TrajectoryEdge:
    fields.setdefault("id", new_id("edge"))
    edge = TrajectoryEdge(**fields)
    session.add(edge)
    session.flush()
    return edge


def delete_edges_for_identity(session: Session, identity_id: str) -> None:
    session.execute(delete(TrajectoryEdge).where(TrajectoryEdge.vehicle_identity_id == identity_id))
    session.flush()


def list_edges(
    session: Session,
    identity_id: Optional[str] = None,
    decision: Optional[str] = None,
    limit: int = 500,
) -> List[TrajectoryEdge]:
    stmt = select(TrajectoryEdge)
    if identity_id:
        stmt = stmt.where(TrajectoryEdge.vehicle_identity_id == identity_id)
    if decision:
        stmt = stmt.where(TrajectoryEdge.decision == decision)
    stmt = stmt.order_by(TrajectoryEdge.created_at.desc()).limit(limit)
    return list(session.scalars(stmt))


def list_unresolved_edges(session: Session, limit: int = 200) -> List[TrajectoryEdge]:
    stmt = (
        select(TrajectoryEdge)
        .where(TrajectoryEdge.decision != "ACCEPTED")
        .order_by(TrajectoryEdge.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


# --------------------------------------------------------------------------
# Processing sessions
# --------------------------------------------------------------------------

def create_session_record(session: Session, camera_id: str, is_demo: bool = False) -> ProcessingSession:
    rec = ProcessingSession(id=new_id("sess"), camera_id=camera_id, is_demo=is_demo)
    session.add(rec)
    session.flush()
    return rec


def update_session_record(session: Session, session_id: str, **fields) -> None:
    rec = session.get(ProcessingSession, session_id)
    if rec is None:
        return
    for k, v in fields.items():
        if hasattr(rec, k):
            setattr(rec, k, v)
    session.flush()


def list_sessions(session: Session, camera_id: Optional[str] = None, limit: int = 100) -> List[ProcessingSession]:
    stmt = select(ProcessingSession)
    if camera_id:
        stmt = stmt.where(ProcessingSession.camera_id == camera_id)
    stmt = stmt.order_by(ProcessingSession.started_at.desc()).limit(limit)
    return list(session.scalars(stmt))


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------

def add_audit_log(
    session: Session,
    action: str,
    actor: str = "anonymous",
    role: str = "viewer",
    target: str = "",
    detail: Optional[str] = None,
    client_ip: Optional[str] = None,
    result: str = "ok",
) -> AuditLog:
    entry = AuditLog(
        action=action,
        actor=actor,
        role=role,
        target=target,
        detail=detail,
        client_ip=client_ip,
        result=result,
    )
    session.add(entry)
    session.flush()
    return entry


def list_audit_logs(session: Session, limit: int = 200) -> List[AuditLog]:
    stmt = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)
    return list(session.scalars(stmt))


# --------------------------------------------------------------------------
# Analytics helpers
# --------------------------------------------------------------------------

def sightings_between(session: Session, start: float, end: float, limit: int = 5000) -> List[VehicleSighting]:
    stmt = (
        select(VehicleSighting)
        .where(VehicleSighting.timestamp >= start, VehicleSighting.timestamp <= end)
        .order_by(VehicleSighting.timestamp)
        .limit(limit)
    )
    return list(session.scalars(stmt))


def accepted_edges(session: Session, limit: int = 5000) -> List[TrajectoryEdge]:
    stmt = (
        select(TrajectoryEdge)
        .where(TrajectoryEdge.decision == "ACCEPTED")
        .order_by(TrajectoryEdge.created_at)
        .limit(limit)
    )
    return list(session.scalars(stmt))
