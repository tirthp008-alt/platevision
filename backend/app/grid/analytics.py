"""Traffic analytics derived from the same sighting and trajectory-edge data.

Implements the metrics from the specification:

  average segment speed   v̄_e = d_e / Δt_e
  flow rate               q_e = N_e / ΔT
  congestion ratio        r_e = q_e / q_cap_e

Segment capacity ``q_cap_e`` is a configurable baseline; without real detector
data it is an approximation and is labelled as such in the response.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.grid import repository as repo
from app.grid.models import VehicleSighting

# Approximate per-lane capacity in vehicles/hour used for the congestion ratio.
DEFAULT_SEGMENT_CAPACITY_VPH = 1800.0


def traffic_analytics(session: Session, window_seconds: float = 3600.0) -> dict:
    end = time.time()
    start = end - window_seconds
    sightings = repo.sightings_between(session, start, end)
    edges = [e for e in repo.accepted_edges(session) if e.created_at >= start or True]

    cameras = {c.id: c for c in repo.list_cameras(session)}

    # Sightings per camera
    per_camera: Dict[str, int] = defaultdict(int)
    for s in sightings:
        per_camera[s.camera_id] += 1

    # Segment metrics from accepted transitions
    segments: Dict[str, dict] = {}
    for e in edges:
        key = f"{e.source_camera_id}->{e.destination_camera_id}"
        seg = segments.setdefault(
            key,
            {
                "segment": key,
                "source_camera": e.source_camera_id,
                "destination_camera": e.destination_camera_id,
                "distance_m": e.road_distance_m,
                "expected_travel_time": e.expected_travel_time,
                "vehicles": 0,
                "sum_travel_time": 0.0,
                "sum_speed_ms": 0.0,
                "samples": 0,
            },
        )
        seg["vehicles"] += 1
        if e.time_delta > 0 and e.road_distance_m > 0:
            speed = e.road_distance_m / e.time_delta
            # Ignore physically impossible speeds (> 200 km/h) as noise.
            if 0 < speed <= 55.0:
                seg["sum_speed_ms"] += speed
                seg["sum_travel_time"] += e.time_delta
                seg["samples"] += 1

    segment_list = []
    for key, seg in segments.items():
        n = seg["vehicles"]
        samples = max(1, seg["samples"])
        avg_speed_ms = seg["sum_speed_ms"] / samples if seg["samples"] else 0.0
        avg_travel_time = seg["sum_travel_time"] / samples if seg["samples"] else 0.0
        flow_vph = (n / window_seconds) * 3600.0
        congestion = flow_vph / DEFAULT_SEGMENT_CAPACITY_VPH
        src = cameras.get(seg["source_camera"])
        dst = cameras.get(seg["destination_camera"])
        segment_list.append(
            {
                "segment": key,
                "source_camera": seg["source_camera"],
                "source_camera_name": src.name if src else seg["source_camera"],
                "destination_camera": seg["destination_camera"],
                "destination_camera_name": dst.name if dst else seg["destination_camera"],
                "distance_m": round(seg["distance_m"], 1),
                "expected_travel_time": round(seg["expected_travel_time"], 1),
                "vehicle_count": n,
                "average_speed_kmh": round(avg_speed_ms * 3.6, 1),
                "average_speed_ms": round(avg_speed_ms, 2),
                "average_travel_time": round(avg_travel_time, 1),
                "flow_vph": round(flow_vph, 1),
                "capacity_vph": DEFAULT_SEGMENT_CAPACITY_VPH,
                "congestion_ratio": round(congestion, 3),
                "units": "si",
                "approximate": True,
            }
        )

    segment_list.sort(key=lambda s: s["vehicle_count"], reverse=True)

    camera_activity = []
    for cam_id, cam in cameras.items():
        activity = {
            "camera_id": cam_id,
            "name": cam.name,
            "status": cam.status,
            "sightings": per_camera.get(cam_id, 0),
            "latitude": cam.latitude,
            "longitude": cam.longitude,
        }
        camera_activity.append(activity)
    camera_activity.sort(key=lambda c: c["sightings"], reverse=True)

    total_vehicles = len({s.vehicle_identity_id for s in sightings if s.vehicle_identity_id})
    return {
        "window_seconds": window_seconds,
        "generated_at": end,
        "totals": {
            "sightings": len(sightings),
            "accepted_transitions": len(edges),
            "unique_vehicles": total_vehicles,
            "active_cameras": sum(1 for c in camera_activity if c["sightings"] > 0),
        },
        "segments": segment_list,
        "camera_activity": camera_activity,
        "notes": "Segment capacity is a baseline approximation; congestion ratio is indicative.",
    }


def heatmap(session: Session, window_seconds: float = 3600.0) -> dict:
    """Vehicle density heatmap points weighted by sightings per camera."""
    end = time.time()
    start = end - window_seconds
    sightings = repo.sightings_between(session, start, end)
    cameras = {c.id: c for c in repo.list_cameras(session)}

    counts: Dict[str, int] = defaultdict(int)
    for s in sightings:
        counts[s.camera_id] += 1

    points = []
    max_count = max(counts.values()) if counts else 1
    for cam_id, cam in cameras.items():
        count = counts.get(cam_id, 0)
        points.append(
            {
                "latitude": cam.latitude,
                "longitude": cam.longitude,
                "camera_id": cam_id,
                "name": cam.name,
                "count": count,
                "intensity": round(count / float(max_count), 3) if max_count else 0.0,
                "status": cam.status,
            }
        )
    return {"window_seconds": window_seconds, "max_count": max_count, "points": points}
