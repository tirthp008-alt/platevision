"""Trajectory reconstruction and plate search.

Given a plate query, this module finds matching global vehicle identities and
builds the complete observed/inferred route: the ordered sightings, the
accepted trajectory edges between them (with plate/visual/temporal evidence),
and any unresolved (ABSTAINED) candidate links that the system deliberately did
not commit to. It never fabricates a route when evidence is insufficient — an
identity with a single sighting yields a route with no transitions.
"""

from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.grid import repository as repo
from app.grid.models import VehicleSighting
from app.grid.road_network import road_network
from app.services.normalizer import clean_raw_ocr, normalize_indian_plate
from app.services.validator import validate_indian_plate_format


def normalize_query_plate(query: str) -> Dict:
    """Normalize a user-entered plate, returning the normalized form plus variants."""
    cleaned = clean_raw_ocr(query)
    normalized, formatted, status = normalize_indian_plate(query)
    cand_status, _ = validate_indian_plate_format(normalized)
    variants = {cleaned, normalized, query.upper().replace(" ", "")}
    variants.discard("")
    return {
        "raw": query,
        "cleaned": cleaned,
        "normalized": normalized or cleaned,
        "formatted": formatted,
        "format_status": status,
        "variants": sorted(variants),
    }


def _sighting_dict(s: VehicleSighting, camera_lookup: Dict[str, dict]) -> dict:
    cam = camera_lookup.get(s.camera_id, {})
    return {
        "sighting_id": s.id,
        "camera_id": s.camera_id,
        "camera_name": cam.get("name", s.camera_id),
        "latitude": cam.get("latitude"),
        "longitude": cam.get("longitude"),
        "timestamp": s.timestamp,
        "vehicle_local_track_id": s.vehicle_local_track_id,
        "plate": s.plate_normalized,
        "plate_raw": s.plate_raw,
        "plate_confidence": round(s.plate_confidence, 3),
        "plate_format_status": s.plate_format_status,
        "plate_candidates": s.plate_candidates or [],
        "visual_embedding_dim": len(s.visual_embedding or []),
        "vehicle_type": s.vehicle_type,
        "vehicle_color": s.vehicle_color,
        "vehicle_bbox": s.vehicle_bbox,
        "frame_reference": s.frame_reference,
        "detection_confidence": s.detection_confidence,
        "observation_type": "OBSERVED",
    }


def search_plate(session: Session, query: str, limit: int = 20) -> dict:
    """Search historical sightings/identities for a plate and rank matches."""
    norm = normalize_query_plate(query)
    cameras = {c.id: c.to_dict() for c in repo.list_cameras(session)}

    # Exact identity match on the dominant plate.
    identities = repo.find_identities_by_plate(session, norm["normalized"])

    # Fall back to sighting-level exact/variant search.
    matched_sighting_ids: set = set()
    plate_hit_counts: Dict[str, int] = {}
    for variant in norm["variants"]:
        for s in repo.list_sightings(session, plate=variant, limit=500):
            matched_sighting_ids.add(s.id)
            plate_hit_counts[s.plate_normalized] = plate_hit_counts.get(s.plate_normalized, 0) + 1

    # Collect identities reachable from the matched sightings.
    identity_ids = {i.id for i in identities}
    for sid in matched_sighting_ids:
        s = repo.get_sighting(session, sid)
        if s and s.vehicle_identity_id:
            identity_ids.add(s.vehicle_identity_id)

    results: List[dict] = []
    matched_identity_objs = []
    for iid in identity_ids:
        identity = repo.get_identity(session, iid)
        if identity is None:
            continue
        sightings = repo.sightings_for_identity(session, iid)
        matched_sighting_objs = [s for s in sightings if s.id in matched_sighting_ids]
        exact = identity.display_plate == norm["normalized"]
        best_conf = max([s.plate_confidence for s in matched_sighting_objs] or [identity.confidence])
        score = (1.0 if exact else 0.6) * (0.5 + 0.5 * best_conf)
        # Count accepted transitions that actually connect sightings of this
        # identity, so a fully reconstructed multi-camera route is surfaced
        # ahead of a lone single-camera sighting of the same plate.
        identity_sighting_ids = {s.id for s in sightings}
        accepted_links = sum(
            1
            for e in repo.list_edges(session, identity_id=identity.id, limit=1000)
            if e.decision == "ACCEPTED"
            and e.source_sighting_id in identity_sighting_ids
            and e.destination_sighting_id in identity_sighting_ids
        )
        results.append(
            {
                "identity": identity.to_dict(),
                "matched": exact,
                "score": round(score, 3),
                "sighting_count": len(sightings),
                "accepted_links": accepted_links,
                "cameras": sorted({s.camera_id for s in sightings}),
                "first_seen": identity.first_seen,
                "last_seen": identity.last_seen,
            }
        )
        matched_identity_objs.append(identity)

    # Prefer exact plate matches, then identities with the most confirmed
    # cross-camera links, then the richest (most sightings) trajectory.
    results.sort(
        key=lambda r: (r["matched"], r["accepted_links"], r["sighting_count"], r["score"]),
        reverse=True,
    )

    # Also expose raw sighting matches (may be unassigned single sightings).
    raw_matches = []
    for sid in list(matched_sighting_ids)[:limit]:
        s = repo.get_sighting(session, sid)
        if s is not None:
            raw_matches.append(_sighting_dict(s, cameras))
    raw_matches.sort(key=lambda m: m["timestamp"], reverse=True)

    return {
        "query": norm,
        "matches": results[:limit],
        "raw_sightings": raw_matches[:limit],
        "unresolved_links": [
            e.to_dict() for e in repo.list_unresolved_edges(session, limit=limit)
        ],
    }


def build_trajectory(session: Session, identity_id: str) -> Optional[dict]:
    """Assemble the full observed/inferred trajectory for one identity."""
    identity = repo.get_identity(session, identity_id)
    if identity is None:
        return None

    cameras = {c.id: c.to_dict() for c in repo.list_cameras(session)}
    sightings = repo.sightings_for_identity(session, identity_id)
    sighting_dicts = [_sighting_dict(s, cameras) for s in sightings]

    sighting_ids = {s.id for s in sightings}
    edges = [
        e
        for e in repo.list_edges(session, identity_id=identity_id, limit=1000)
        if e.source_sighting_id in sighting_ids and e.destination_sighting_id in sighting_ids
    ]
    accepted = [e for e in edges if e.decision == "ACCEPTED"]

    # The identity is a simple path over distinct cameras, but the edge set also
    # contains "skip" links between non-adjacent sightings (e.g. C1->C3 while
    # C1->C2->C3 is the real chain). Walk the accepted edges as a directed path
    # starting from the chain head (a source that is never a destination) and
    # take the earliest still-forward edge at each hop, so the displayed route is
    # a single monotonic chain rather than a tangle of overlapping segments.
    # Skipped edges remain in the graph but are not part of the route.
    sighting_order = sorted(sightings, key=lambda s: (s.timestamp, s.id))
    time_of = {s.id: s.timestamp for s in sighting_order}
    edges_by_source: Dict[str, list] = {}
    for e in accepted:
        edges_by_source.setdefault(e.source_sighting_id, []).append(e)
    for lst in edges_by_source.values():
        lst.sort(key=lambda e: time_of.get(e.destination_sighting_id, float("inf")))

    destination_ids = {e.destination_sighting_id for e in accepted}
    heads = [sid for sid in edges_by_source if sid not in destination_ids]
    if heads:
        start = min(heads, key=lambda sid: time_of.get(sid, float("inf")))
    else:
        start = sighting_order[0].id if sighting_order else None

    chain = []
    current, visited = start, set()
    while current is not None and current not in visited:
        visited.add(current)
        nxt = edges_by_source.get(current)
        if not nxt:
            break
        edge = nxt[0]
        chain.append(edge)
        current = edge.destination_sighting_id

    transition_dicts = [
        {
            **e.to_dict(),
            "source_camera_name": cameras.get(e.source_camera_id, {}).get("name", e.source_camera_id),
            "destination_camera_name": cameras.get(e.destination_camera_id, {}).get(
                "name", e.destination_camera_id
            ),
            "inference": "INFERRED",
        }
        for e in chain
    ]

    # Unresolved candidate links involving these sightings (for human review).
    # Only ABSTAINED edges belong here: a REJECTED edge was confidently ruled out
    # and must not be presented as an open question.
    unresolved = []
    for e in repo.list_unresolved_edges(session, limit=200):
        if e.decision != "ABSTAINED":
            continue
        if e.source_sighting_id in sighting_ids or e.destination_sighting_id in sighting_ids:
            unresolved.append(
                {
                    **e.to_dict(),
                    "source_camera_name": cameras.get(e.source_camera_id, {}).get(
                        "name", e.source_camera_id
                    ),
                    "destination_camera_name": cameras.get(e.destination_camera_id, {}).get(
                        "name", e.destination_camera_id
                    ),
                }
            )

    # Geometry for accepted transitions (road-network path where available).
    for t in transition_dicts:
        if not t.get("path_geometry"):
            t["path_geometry"] = road_network.path_geometry(
                t["source_camera_id"], t["destination_camera_id"]
            )

    route_camera_ids = [s["camera_id"] for s in sighting_dicts]
    travel_time = None
    if sightings:
        travel_time = sightings[-1].timestamp - sightings[0].timestamp

    return {
        "identity": identity.to_dict(),
        "route": sighting_dicts,
        "transitions": transition_dicts,
        "unresolved_links": unresolved,
        "route_camera_ids": route_camera_ids,
        "first_seen": identity.first_seen,
        "last_seen": identity.last_seen,
        "inferred_total_travel_time": round(travel_time, 1) if travel_time is not None else None,
        "has_inferred_links": bool(transition_dicts),
        "notes": (
            "OBSERVED entries are actual camera detections. INFERRED transitions are "
            "reconstructed from plate + visual + temporal + road-network evidence and "
            "may include road-network paths. UNRESOLVED links abstained by the confidence gate."
        ),
    }


def trajectory_for_plate(session: Session, query: str) -> Optional[dict]:
    """Convenience: search a plate and return the trajectory of the best match."""
    search = search_plate(session, query, limit=5)
    if not search["matches"]:
        return None
    best = search["matches"][0]
    traj = build_trajectory(session, best["identity"]["id"])
    if traj is not None:
        traj["search"] = search["query"]
    return traj
