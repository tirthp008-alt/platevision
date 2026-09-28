"""SQLAlchemy ORM models for the Drishti Grid trajectory-tracking system.

The schema is portable across SQLite (default local development) and
PostgreSQL/PostGIS (production). Geographic coordinates are stored as plain
``latitude``/``longitude`` floats so SQLite works out of the box; when a
PostgreSQL URL is configured the repository creates a PostGIS ``geometry``
column and spatial index as well (see :mod:`app.grid.db`).
"""

import time
from typing import Optional

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _now() -> float:
    return time.time()


class Camera(Base):
    """A registered camera location bound to a local/RTSP video source."""

    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), default="device")  # device|rtsp|file|demo
    source_uri: Mapped[str] = mapped_column(Text, default="")
    device_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    road_segment_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(24), default="OFFLINE")  # ONLINE/OFFLINE/CONNECTING/PROCESSING/ERROR
    created_at: Mapped[float] = mapped_column(Float, default=_now)
    updated_at: Mapped[float] = mapped_column(Float, default=_now, onupdate=_now)
    last_seen: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    sightings: Mapped[list["VehicleSighting"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "device_id": self.device_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "road_segment_id": self.road_segment_id,
            "enabled": self.enabled,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_seen": self.last_seen,
            "last_error": self.last_error,
            "is_demo": self.is_demo,
        }


class VehicleSighting(Base):
    """One observation of one vehicle at one camera (the tuple o_j = (p_j, v_j, t_j, c_j))."""

    __tablename__ = "vehicle_sightings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id"), index=True)
    vehicle_local_track_id: Mapped[int] = mapped_column(Integer, index=True)
    vehicle_identity_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("vehicle_identities.id"), index=True, nullable=True
    )

    timestamp: Mapped[float] = mapped_column(Float, index=True)
    plate_raw: Mapped[str] = mapped_column(Text, default="")
    plate_normalized: Mapped[str] = mapped_column(String(24), default="", index=True)
    plate_candidates: Mapped[list] = mapped_column(JSON, default=list)
    plate_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    plate_format_status: Mapped[str] = mapped_column(String(16), default="uncertain")

    visual_embedding: Mapped[list] = mapped_column(JSON, default=list)
    vehicle_type: Mapped[str] = mapped_column(String(32), default="unknown")
    vehicle_color: Mapped[str] = mapped_column(String(32), default="unknown")
    vehicle_bbox: Mapped[list] = mapped_column(JSON, default=list)
    plate_bbox: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    frame_reference: Mapped[Optional[str]] = mapped_column(String(96), nullable=True)
    detection_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    camera: Mapped["Camera"] = relationship(back_populates="sightings")
    identity: Mapped[Optional["VehicleIdentity"]] = relationship(back_populates="sightings")

    __table_args__ = (
        Index("ix_sighting_cam_time", "camera_id", "timestamp"),
        Index("ix_sighting_plate_time", "plate_normalized", "timestamp"),
    )

    def to_dict(self, mask_plate: bool = False) -> dict:
        from app.grid.privacy import mask_plate as _mask

        plate = self.plate_normalized or self.plate_raw
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "vehicle_local_track_id": self.vehicle_local_track_id,
            "vehicle_identity_id": self.vehicle_identity_id,
            "timestamp": self.timestamp,
            "plate_raw": self.plate_raw if not mask_plate else _mask(self.plate_raw),
            "plate_normalized": plate if not mask_plate else _mask(plate),
            "plate_confidence": round(self.plate_confidence, 3),
            "plate_format_status": self.plate_format_status,
            "plate_candidates": self.plate_candidates or [],
            "vehicle_type": self.vehicle_type,
            "vehicle_color": self.vehicle_color,
            "vehicle_bbox": self.vehicle_bbox,
            "plate_bbox": self.plate_bbox,
            "frame_reference": self.frame_reference,
            "detection_confidence": round(self.detection_confidence, 3),
            "embedding_dim": len(self.visual_embedding or []),
        }


class VehicleIdentity(Base):
    """A global vehicle identity assembled from cross-camera associations."""

    __tablename__ = "vehicle_identities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_plate: Mapped[str] = mapped_column(String(24), default="", index=True)
    status: Mapped[str] = mapped_column(String(24), default="PROVISIONAL")  # PROVISIONAL|CONFIRMED|AMBIGUOUS
    first_seen: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_seen: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sighting_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    plate_hypotheses: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[float] = mapped_column(Float, default=_now)

    sightings: Mapped[list["VehicleSighting"]] = relationship(back_populates="identity")

    def to_dict(self, mask_plate: bool = False) -> dict:
        from app.grid.privacy import mask_plate as _mask

        plate = self.display_plate
        return {
            "id": self.id,
            "display_plate": plate if not mask_plate else _mask(plate),
            "status": self.status,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "sighting_count": self.sighting_count,
            "confidence": round(self.confidence, 3),
            "plate_hypotheses": self.plate_hypotheses or [],
        }


class TrajectoryEdge(Base):
    """A candidate transition between two sightings (trajectory graph edge)."""

    __tablename__ = "trajectory_edges"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_sighting_id: Mapped[str] = mapped_column(String(64), index=True)
    destination_sighting_id: Mapped[str] = mapped_column(String(64), index=True)
    source_camera_id: Mapped[str] = mapped_column(String(64), index=True)
    destination_camera_id: Mapped[str] = mapped_column(String(64), index=True)
    vehicle_identity_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)

    time_delta: Mapped[float] = mapped_column(Float, default=0.0)
    expected_travel_time: Mapped[float] = mapped_column(Float, default=0.0)
    road_distance_m: Mapped[float] = mapped_column(Float, default=0.0)
    plate_score: Mapped[float] = mapped_column(Float, default=0.0)
    visual_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    temporal_score: Mapped[float] = mapped_column(Float, default=0.0)
    combined_score: Mapped[float] = mapped_column(Float, default=0.0)
    runner_up_score: Mapped[float] = mapped_column(Float, default=0.0)
    decision: Mapped[str] = mapped_column(String(16), default="ABSTAINED")  # ACCEPTED|ABSTAINED|REJECTED
    decision_reason: Mapped[str] = mapped_column(Text, default="")
    path_geometry: Mapped[list] = mapped_column(JSON, default=list)
    fusion_mode: Mapped[str] = mapped_column(String(24), default="full")
    created_at: Mapped[float] = mapped_column(Float, default=_now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_sighting_id": self.source_sighting_id,
            "destination_sighting_id": self.destination_sighting_id,
            "source_camera_id": self.source_camera_id,
            "destination_camera_id": self.destination_camera_id,
            "vehicle_identity_id": self.vehicle_identity_id,
            "time_delta": round(self.time_delta, 2),
            "expected_travel_time": round(self.expected_travel_time, 2),
            "road_distance_m": round(self.road_distance_m, 1),
            "plate_score": round(self.plate_score, 4),
            "visual_similarity": round(self.visual_similarity, 4),
            "temporal_score": round(self.temporal_score, 4),
            "combined_score": round(self.combined_score, 4),
            "runner_up_score": round(self.runner_up_score, 4),
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "path_geometry": self.path_geometry or [],
            "fusion_mode": self.fusion_mode,
            "created_at": self.created_at,
        }


class PlateObservation(Base):
    """A single-frame OCR observation for a tracked vehicle (multi-frame fusion input)."""

    __tablename__ = "plate_observations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    vehicle_local_track_id: Mapped[int] = mapped_column(Integer, index=True)
    timestamp: Mapped[float] = mapped_column(Float, index=True)
    plate_raw: Mapped[str] = mapped_column(Text, default="")
    plate_normalized: Mapped[str] = mapped_column(String(24), default="")
    ocr_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    plate_bbox: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    vehicle_bbox: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    def to_dict(self) -> dict:
        from app.grid.privacy import mask_plate as _mask

        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "vehicle_local_track_id": self.vehicle_local_track_id,
            "timestamp": self.timestamp,
            "plate_raw": self.plate_raw,
            "plate_normalized": self.plate_normalized,
            "ocr_confidence": round(self.ocr_confidence, 3),
            "plate_bbox": self.plate_bbox,
            "vehicle_bbox": self.vehicle_bbox,
        }


class ProcessingSession(Base):
    """A start/stop lifecycle record for a camera processing run."""

    __tablename__ = "processing_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    started_at: Mapped[float] = mapped_column(Float, default=_now)
    stopped_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    frames_dropped: Mapped[int] = mapped_column(Integer, default=0)
    sightings_created: Mapped[int] = mapped_column(Integer, default=0)
    plates_read: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "sightings_created": self.sightings_created,
            "plates_read": self.plates_read,
            "error": self.error,
            "is_demo": self.is_demo,
        }


class AuditLog(Base):
    """Tamper-evident access/audit log for privacy-sensitive operations."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[float] = mapped_column(Float, default=_now, index=True)
    actor: Mapped[str] = mapped_column(String(80), default="anonymous")
    role: Mapped[str] = mapped_column(String(32), default="viewer")
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str] = mapped_column(String(160), default="")
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    client_ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    result: Mapped[str] = mapped_column(String(24), default="ok")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "role": self.role,
            "action": self.action,
            "target": self.target,
            "detail": self.detail,
            "client_ip": self.client_ip,
            "result": self.result,
        }
