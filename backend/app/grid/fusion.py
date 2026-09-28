"""Cross-camera fusion engine.

This is the core of Drishti Grid. For two sightings i and j from different
cameras it computes the candidate association score

    S_ij = λ_p · s_plate_ij
         + λ_v · cos(v_i, v_j)
         − λ_t · |Δt_ij − t_G(c_i, c_j)| / σ_t

and applies the confidence gate

    accept  ⟺  S(1) ≥ τ   AND   S(1) − S(2) ≥ δ

where S(1)/S(2) are the best/second-best candidate scores for a source
sighting. Otherwise the link is ABSTAINED (unresolved) or REJECTED. Accepted
edges form the trajectory graph; their connected components become global
vehicle identities.

The three research modes required for comparison are selectable via ``mode``:
``plate_only`` (λ_v = λ_t = 0), ``plate_visual`` (λ_t = 0) and ``full``.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.grid import repository as repo
from app.grid.embedder import cosine_similarity
from app.grid.events import event_bus
from app.grid.models import TrajectoryEdge, VehicleIdentity, VehicleSighting
from app.grid.plate_ocr import PlateHypothesis, plate_agreement_score
from app.grid.road_network import road_network

MODE_WEIGHTS = {
    # mode -> (use_visual, use_temporal)
    "plate_only": (False, False),
    "plate_visual": (True, False),
    "full": (True, True),
}


@dataclass
class FusionConfig:
    lambda_plate: float = 0.50
    lambda_visual: float = 0.35
    lambda_time: float = 0.15
    sigma_t: float = 180.0
    tau: float = 0.70
    delta: float = 0.15
    mode: str = "full"

    @classmethod
    def from_settings(cls, mode: Optional[str] = None) -> "FusionConfig":
        return cls(
            lambda_plate=settings.FUSION_LAMBDA_PLATE,
            lambda_visual=settings.FUSION_LAMBDA_VISUAL,
            lambda_time=settings.FUSION_LAMBDA_TIME,
            sigma_t=settings.FUSION_SIGMA_T_SECONDS,
            tau=settings.FUSION_TAU,
            delta=settings.FUSION_DELTA,
            mode=mode or settings.FUSION_MODE,
        )


def _hypothesis_from_sighting(s: VehicleSighting) -> PlateHypothesis:
    candidates = s.plate_candidates or []
    if s.plate_normalized:
        status = "confident"
    elif candidates:
        status = "ambiguous"
    else:
        status = "unknown"
    return PlateHypothesis(
        plate_normalized=s.plate_normalized or "",
        plate_raw=s.plate_raw or "",
        confidence=float(s.plate_confidence or 0.0),
        observation_count=1,
        status=status,
        candidates=candidates,
    )


def candidate_score(
    si: VehicleSighting,
    sj: VehicleSighting,
    cam_i: str,
    cam_j: str,
    config: FusionConfig,
    plate_hypotheses: Optional[Dict[str, PlateHypothesis]] = None,
    route=None,
) -> Dict:
    """Compute S_ij and its components for one candidate pair."""
    use_visual, use_temporal = MODE_WEIGHTS.get(config.mode, (True, True))

    hyp = plate_hypotheses or {}
    hi = hyp.get(si.id) or _hypothesis_from_sighting(si)
    hj = hyp.get(sj.id) or _hypothesis_from_sighting(sj)

    s_plate = plate_agreement_score(hi, hj)
    plate_known = s_plate is not None
    if s_plate is None:
        s_plate = 0.0

    # Visual appearance similarity (cosine), only if both embeddings exist.
    visual = 0.0
    visual_known = bool(si.visual_embedding) and bool(sj.visual_embedding)
    if visual_known:
        cos = cosine_similarity(si.visual_embedding, sj.visual_embedding)
        visual = max(0.0, cos)  # negative similarity contributes nothing, never a bonus

    # Temporal / road-network feasibility.
    if route is None:
        route = road_network.route(cam_i, cam_j)
    dt = abs(float(sj.timestamp) - float(si.timestamp))
    t_expected = route.expected_travel_time
    sigma = max(1.0, config.sigma_t)
    temporal_term = abs(dt - t_expected) / sigma
    temporal_score = math.exp(-0.5 * temporal_term * temporal_term)  # 1.0 when dt == t_expected

    lam_t = config.lambda_time if use_temporal else 0.0

    # Distribute weight only across evidence channels that are actually known,
    # so the score keeps a comparable scale across fusion modes.
    plate_weight = config.lambda_plate if plate_known else 0.0
    visual_weight = config.lambda_visual if (use_visual and visual_known) else 0.0
    total_weight = plate_weight + visual_weight
    if total_weight > 0:
        lam_p_n = plate_weight / total_weight
        lam_v_n = visual_weight / total_weight
    else:
        lam_p_n = lam_v_n = 0.0

    combined = lam_p_n * s_plate + lam_v_n * visual
    if use_temporal:
        combined -= lam_t * temporal_term

    combined = float(max(0.0, min(1.0, combined)))

    return {
        "plate_score": float(s_plate),
        "plate_known": plate_known,
        "visual_similarity": float(visual),
        "visual_known": visual_known,
        "temporal_score": float(temporal_score),
        "time_delta": dt,
        "expected_travel_time": float(t_expected),
        "road_distance_m": float(route.distance_m),
        "combined_score": combined,
        "route_source": route.source,
        "geometry": route.geometry,
        "hops": max(0, len(getattr(route, "path_node_ids", []) or []) - 2),
        "time_residual": float(abs(dt - t_expected)),
    }


def _plate_of(hyp) -> str:
    """Best plate string for a hypothesis (normalized, falling back to top candidate)."""
    if hyp is None:
        return ""
    if getattr(hyp, "plate_normalized", ""):
        return hyp.plate_normalized
    candidates = getattr(hyp, "candidates", None) or []
    return candidates[0] if candidates else ""


def _same_identity_plate(hyp_a, hyp_b) -> bool:
    """True when two hypotheses are consistent with the same vehicle identity."""
    return _plates_match(_plate_of(hyp_a), _plate_of(hyp_b))


def _plates_match(a: str, b: str) -> bool:
    if not a or not b:
        # An unknown plate is not treated as a mismatch: if the other side is
        # also unknown we cannot distinguish them, so they are not competitors.
        return not a and not b
    return a == b


def decision_for(
    best: float, second: float, config: FusionConfig
) -> Tuple[str, str]:
    """Apply the confidence gate. Returns (decision, reason)."""
    if best >= config.tau and (best - second) >= config.delta:
        return (
            "ACCEPTED",
            f"S(1)={best:.3f} ≥ τ={config.tau:.2f} and margin {best - second:.3f} ≥ δ={config.delta:.2f}",
        )
    if best >= config.tau:
        return (
            "ABSTAINED",
            f"top score {best:.3f} ≥ τ but margin {best - second:.3f} < δ={config.delta:.2f} (ambiguous)",
        )
    return (
        "ABSTAINED",
        f"top score {best:.3f} < τ={config.tau:.2f} (insufficient evidence)",
    )


class FusionEngine:
    """Recomputes cross-camera associations, trajectory edges and identities."""

    WINDOW_SECONDS = 6 * 3600
    MAX_SIGHTINGS = 3000

    def __init__(self):
        self._lock = threading.Lock()
        self._dirty = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_run = 0.0
        self.metrics = {
            "runs": 0,
            "edges_accepted": 0,
            "edges_abstained": 0,
            "edges_rejected": 0,
            "identities": 0,
        }

    # ------------------------------------------------------------------
    def mark_dirty(self) -> None:
        self._dirty.set()

    def start_background(self, interval: float = 2.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(interval,), name="drishti-fusion", daemon=True
        )
        self._thread.start()

    def stop_background(self) -> None:
        self._stop.set()

    def _loop(self, interval: float) -> None:
        while not self._stop.is_set():
            if self._dirty.wait(timeout=interval):
                self._dirty.clear()
                try:
                    from app.grid.db import session_scope

                    with session_scope() as session:
                        self.recompute(session)
                except Exception as e:
                    logger.error(f"Fusion recompute failed: {e}")

    # ------------------------------------------------------------------
    def recompute(self, session: Session, config: Optional[FusionConfig] = None) -> Dict:
        """Recompute trajectory edges and global identities from stored sightings."""
        config = config or FusionConfig.from_settings()
        with self._lock:
            sightings = repo.recent_unassigned_sightings(session, limit=self.MAX_SIGHTINGS)
            if not sightings:
                self.metrics["runs"] += 1
                return dict(self.metrics)

            now = max(s.timestamp for s in sightings)
            window_start = now - self.WINDOW_SECONDS
            sighting_list = [s for s in sightings if s.timestamp >= window_start]
            sighting_list.sort(key=lambda s: s.timestamp)

            cameras = repo.list_cameras(session)
            cam_lookup = {c.id: c for c in cameras}
            road_network.register_cameras(
                [
                    {"id": c.id, "latitude": c.latitude, "longitude": c.longitude}
                    for c in cameras
                ]
            )

            hyp = {s.id: _hypothesis_from_sighting(s) for s in sighting_list}

            # Cache routes between camera pairs.
            route_cache: Dict[Tuple[str, str], object] = {}

            def get_route(a: str, b: str):
                key = (a, b)
                if key not in route_cache:
                    route_cache[key] = road_network.route(a, b)
                return route_cache[key]

            # ----------------------------------------------------------
            # Candidate generation + gating, per source sighting.
            # ----------------------------------------------------------
            edges: List[TrajectoryEdge] = []
            accepted_pairs: List[Tuple[str, str, dict]] = []

            # Clear previous derived state for a consistent snapshot. Edges and
            # identities are fully derived from sightings, so they are rebuilt
            # on every recompute (sightings themselves are never removed).
            session.query(TrajectoryEdge).delete()
            session.query(VehicleIdentity).delete()
            for s in sighting_list:
                s.vehicle_identity_id = None
            session.flush()

            for i, si in enumerate(sighting_list):
                candidates = []
                for j, sj in enumerate(sighting_list):
                    if i == j:
                        continue
                    if sj.camera_id == si.camera_id:
                        continue
                    if sj.timestamp <= si.timestamp:
                        continue
                    route = get_route(si.camera_id, sj.camera_id)
                    max_dt = 4.0 * route.expected_travel_time + 600.0
                    if max_dt > self.WINDOW_SECONDS:
                        max_dt = self.WINDOW_SECONDS
                    if (sj.timestamp - si.timestamp) > max_dt:
                        continue
                    score = candidate_score(
                        si, sj, si.camera_id, sj.camera_id, config, hyp, route
                    )
                    candidates.append((sj.id, score))

                if not candidates:
                    continue

                # Rank by combined score, then by temporal adjacency and fewer
                # road-graph hops as documented tie-breaks when scores are
                # effectively equal.
                candidates.sort(
                    key=lambda c: (
                        round(c[1]["combined_score"], 2),
                        -c[1]["time_residual"],
                        -c[1]["hops"],
                    ),
                    reverse=True,
                )
                best_id = candidates[0][0]
                best_score = candidates[0][1]["combined_score"]
                top_plate = _plate_of(hyp.get(best_id))

                # The margin gate guards against *identity* ambiguity: only
                # candidates that do not share the top candidate's plate are
                # genuine competitors. Candidates with the same plate are the
                # same vehicle seen elsewhere (a corroborating link) and must not
                # force an abstention, or multi-hop chains could never be built.
                competitor_scores = [
                    sc["combined_score"]
                    for _cid, sc in candidates[1:]
                    if not _plates_match(_plate_of(hyp.get(_cid)), top_plate)
                ]
                second = max(competitor_scores) if competitor_scores else 0.0

                # Same-plate candidates are only corroborating if they sit at
                # distinct cameras. Two same-plate sightings at one camera are
                # mutually exclusive choices (which physical vehicle followed?),
                # so the group is ambiguous and must be abstained.
                top_group = [
                    cid
                    for cid, _sc in candidates
                    if _plates_match(_plate_of(hyp.get(cid)), top_plate)
                ]
                group_cameras = {
                    s.camera_id for s in sighting_list if s.id in top_group
                }
                group_ambiguous = len(group_cameras) < len(top_group)

                for rank, (dest_id, score) in enumerate(candidates):
                    same_plate = _plates_match(_plate_of(hyp.get(dest_id)), top_plate)
                    if same_plate and group_ambiguous:
                        decision = "ABSTAINED"
                        reason = (
                            f"ambiguous: {len(top_group)} same-plate candidates across "
                            f"{len(group_cameras)} camera(s); cannot resolve identity"
                        )
                    elif rank == 0:
                        decision, reason = decision_for(best_score, second, config)
                    elif same_plate and score["combined_score"] >= config.tau:
                        decision = "ACCEPTED"
                        reason = (
                            f"corroborating link: same plate identity as the top candidate "
                            f"(S={score['combined_score']:.3f})"
                        )
                    elif score["combined_score"] >= config.tau:
                        decision = "ABSTAINED"
                        reason = (
                            f"alternative candidate S={score['combined_score']:.3f} ≥ τ but conflicts "
                            f"with the accepted identity"
                        )
                    else:
                        decision = "REJECTED"
                        reason = f"S={score['combined_score']:.3f} below τ={config.tau:.2f}"

                    sj = next(s for s in sighting_list if s.id == dest_id)
                    edge = TrajectoryEdge(
                        id=repo.new_id("edge"),
                        source_sighting_id=si.id,
                        destination_sighting_id=sj.id,
                        source_camera_id=si.camera_id,
                        destination_camera_id=sj.camera_id,
                        time_delta=score["time_delta"],
                        expected_travel_time=score["expected_travel_time"],
                        road_distance_m=score["road_distance_m"],
                        plate_score=score["plate_score"],
                        visual_similarity=score["visual_similarity"],
                        temporal_score=score["temporal_score"],
                        combined_score=score["combined_score"],
                        runner_up_score=second,
                        decision=decision,
                        decision_reason=reason,
                        path_geometry=score["geometry"],
                        fusion_mode=config.mode,
                    )
                    session.add(edge)
                    edges.append(edge)
                    if decision == "ACCEPTED":
                        accepted_pairs.append((si.id, sj.id, edge_debug(score, route_source=score["route_source"])))

            session.flush()

            # ----------------------------------------------------------
            # Global identities = connected components of ACCEPTED edges.
            # ----------------------------------------------------------
            parent: Dict[str, str] = {s.id: s.id for s in sighting_list}

            def find(x: str) -> str:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: str, b: str) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            # A vehicle traverses each camera at most once in a trip, so a
            # physically meaningful identity is a simple path: no camera is
            # revisited. Accepted edges are merged in descending score order and
            # any union that would introduce a duplicate camera is dropped, which
            # keeps the reconstructed route a clean single pass through the city
            # instead of chaining several same-camera sightings together.
            def _camera_after_merge(ra: str, rb: str) -> bool:
                cams_a = {s.camera_id for s in sighting_list if find(s.id) == ra}
                cams_b = {s.camera_id for s in sighting_list if find(s.id) == rb}
                return bool(cams_a & cams_b)

            for a, b, _ in sorted(accepted_pairs, key=lambda t: t[2]["combined_score"], reverse=True):
                ra, rb = find(a), find(b)
                if ra == rb:
                    continue
                if _camera_after_merge(ra, rb):
                    continue
                union(ra, rb)

            components: Dict[str, List[VehicleSighting]] = {}
            for s in sighting_list:
                components.setdefault(find(s.id), []).append(s)

            sighting_by_id = {s.id: s for s in sighting_list}
            identities: List[VehicleIdentity] = []
            for root, members in components.items():
                if len(members) < 2 and not any(m.plate_normalized for m in members):
                    continue  # nothing meaningful to persist
                members.sort(key=lambda m: m.timestamp)

                plate_counts: Dict[str, float] = {}
                for m in members:
                    if m.plate_normalized:
                        plate_counts[m.plate_normalized] = plate_counts.get(
                            m.plate_normalized, 0.0
                        ) + max(0.1, m.plate_confidence)
                display_plate = ""
                if plate_counts:
                    display_plate = max(plate_counts.items(), key=lambda kv: kv[1])[0]

                cam_chain = [m.camera_id for m in members]
                chain_ok = len(set(cam_chain)) == len(cam_chain)  # no camera revisited
                has_accepted = any(
                    find(a) == root and find(b) == root for a, b, _ in accepted_pairs
                )
                if display_plate and chain_ok and len(members) >= 2:
                    status = "CONFIRMED"
                elif len(members) >= 2:
                    status = "PROVISIONAL"
                else:
                    status = "PROVISIONAL"

                conf = sum(m.plate_confidence for m in members) / max(1, len(members))
                # Derive a stable identity id from the initial sighting so that
                # re-running fusion does not invalidate identities a client just
                # fetched (sightings are never removed, only derived edges and
                # identities are rebuilt).
                anchor = min(members, key=lambda m: (m.timestamp, m.id))
                identity = VehicleIdentity(
                    id=f"veh-{anchor.id}",
                    display_plate=display_plate,
                    status=status,
                    first_seen=members[0].timestamp,
                    last_seen=members[-1].timestamp,
                    sighting_count=len(members),
                    confidence=float(conf),
                    plate_hypotheses=[
                        hyp[m.id].to_dict() for m in members if hyp.get(m.id)
                    ],
                )
                session.add(identity)
                identities.append(identity)
                for m in members:
                    m.vehicle_identity_id = identity.id

            # Attach identity ids to accepted edges where possible.
            session.flush()
            sighting_identity = {s.id: s.vehicle_identity_id for s in sighting_list}
            for edge in edges:
                sid = sighting_identity.get(edge.source_sighting_id)
                did = sighting_identity.get(edge.destination_sighting_id)
                if sid and sid == did:
                    edge.vehicle_identity_id = sid

            accepted = sum(1 for e in edges if e.decision == "ACCEPTED")
            abstained = sum(1 for e in edges if e.decision == "ABSTAINED")
            rejected = sum(1 for e in edges if e.decision == "REJECTED")
            self.metrics.update(
                {
                    "runs": self.metrics["runs"] + 1,
                    "edges_accepted": accepted,
                    "edges_abstained": abstained,
                    "edges_rejected": rejected,
                    "identities": len(identities),
                }
            )
            self._last_run = time.time()
            logger.info(
                f"Fusion: {len(sighting_list)} sightings → {accepted} accepted, "
                f"{abstained} abstained, {rejected} rejected, {len(identities)} identities"
            )
            return dict(self.metrics)


def edge_debug(score: dict, route_source: str = "") -> dict:
    return {**score, "route_source": route_source}


fusion_engine = FusionEngine()
