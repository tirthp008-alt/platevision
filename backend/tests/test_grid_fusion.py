"""Tests for the fusion engine: candidate score, confidence gate, abstention."""

import time

import pytest

from app.grid import repository as repo
from app.grid.fusion import FusionConfig, candidate_score, decision_for, fusion_engine
from app.grid.road_network import RoadRoute


def _route(distance_m=150.0, travel_time=16.0, source="graph_fallback"):
    return RoadRoute(distance_m=distance_m, expected_travel_time=travel_time, source=source, approximate=True)


def _make_sighting(session, camera, track_id, timestamp, plate, conf, embedding):
    return repo.create_sighting(
        session,
        camera_id=camera.id,
        vehicle_local_track_id=track_id,
        timestamp=timestamp,
        plate_raw=plate,
        plate_normalized=plate,
        plate_confidence=conf,
        visual_embedding=embedding,
        vehicle_type="car",
        vehicle_color="blue",
        vehicle_bbox=[10, 10, 50, 50],
        detection_confidence=0.9,
    )


# --------------------------------------------------------------------------
# 6. Candidate score calculation
# --------------------------------------------------------------------------

def test_candidate_score_high_for_matching_pair(fresh_db, camera_factory):
    c1 = camera_factory(0, "C1")
    c2 = camera_factory(1, "C2")
    t = time.time()
    emb = [0.2, 0.3, 0.4, 0.5]
    s1 = _make_sighting(fresh_db, c1, 17, t, "GJ01AB1234", 0.95, emb)
    s2 = _make_sighting(fresh_db, c2, 42, t + 16, "GJ01AB1234", 0.93, emb)

    cfg = FusionConfig.from_settings("full")
    score = candidate_score(s1, s2, c1.id, c2.id, cfg, route=_route())
    assert score["plate_score"] > 0.8
    assert score["visual_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert score["temporal_score"] > 0.9  # observed delta matches expected
    assert score["combined_score"] > 0.8


def test_candidate_score_penalised_for_impossible_timing(fresh_db, camera_factory):
    c1 = camera_factory(0, "C1")
    c2 = camera_factory(4, "C2")
    t = time.time()
    emb = [0.2, 0.3, 0.4, 0.5]
    # Cameras ~20 km apart, observed delta only 30 seconds.
    s1 = _make_sighting(fresh_db, c1, 1, t, "GJ01AB1234", 0.95, emb)
    s2 = _make_sighting(fresh_db, c2, 2, t + 30, "GJ01AB1234", 0.95, emb)

    cfg = FusionConfig.from_settings("full")
    far_route = _route(distance_m=20_000.0, travel_time=2100.0)
    score = candidate_score(s1, s2, c1.id, c2.id, cfg, route=far_route)
    # Temporal penalty must dominate and pull the score below the accept gate.
    assert score["temporal_score"] < 0.5
    assert score["combined_score"] < cfg.tau


def test_candidate_score_low_for_different_plate(fresh_db, camera_factory):
    c1 = camera_factory(0, "C1")
    c2 = camera_factory(1, "C2")
    t = time.time()
    s1 = _make_sighting(fresh_db, c1, 1, t, "GJ01AB1234", 0.95, [1.0, 0.0, 0.0, 0.0])
    s2 = _make_sighting(fresh_db, c2, 2, t + 16, "GJ01AB9999", 0.95, [0.0, 1.0, 0.0, 0.0])

    cfg = FusionConfig.from_settings("full")
    score = candidate_score(s1, s2, c1.id, c2.id, cfg, route=_route())
    # Matching state/district but a different series → partial plate evidence only.
    assert score["plate_score"] <= 0.45
    assert score["combined_score"] < cfg.tau


def test_candidate_score_near_zero_for_unrelated_plate(fresh_db, camera_factory):
    c1 = camera_factory(0, "C1")
    c2 = camera_factory(1, "C2")
    t = time.time()
    s1 = _make_sighting(fresh_db, c1, 1, t, "GJ01AB1234", 0.95, [1.0, 0.0, 0.0, 0.0])
    s2 = _make_sighting(fresh_db, c2, 2, t + 16, "MH12DE1433", 0.95, [0.0, 1.0, 0.0, 0.0])

    cfg = FusionConfig.from_settings("full")
    score = candidate_score(s1, s2, c1.id, c2.id, cfg, route=_route())
    assert score["plate_score"] <= 0.15


# --------------------------------------------------------------------------
# 7/8. Confidence gate and abstention
# --------------------------------------------------------------------------

def test_gate_accepts_clear_winner():
    cfg = FusionConfig(tau=0.70, delta=0.15)
    decision, reason = decision_for(0.82, 0.54, cfg)
    assert decision == "ACCEPTED"


def test_gate_abstains_without_margin():
    cfg = FusionConfig(tau=0.70, delta=0.15)
    decision, reason = decision_for(0.82, 0.79, cfg)
    assert decision == "ABSTAINED"
    assert "margin" in reason


def test_gate_abstains_below_threshold():
    cfg = FusionConfig(tau=0.70, delta=0.15)
    decision, reason = decision_for(0.60, 0.20, cfg)
    assert decision == "ABSTAINED"


# --------------------------------------------------------------------------
# 11. Trajectory construction through the engine
# --------------------------------------------------------------------------

def test_fusion_accepts_valid_chain(fresh_db, camera_factory):
    c1 = camera_factory(0, "C1")
    c2 = camera_factory(1, "C2")
    c3 = camera_factory(2, "C3")
    t = time.time()
    emb = [0.2, 0.3, 0.4, 0.5]
    _make_sighting(fresh_db, c1, 17, t, "GJ01AB1234", 0.95, emb)
    _make_sighting(fresh_db, c2, 42, t + 16, "GJ01AB1234", 0.93, [0.2, 0.31, 0.4, 0.49])
    _make_sighting(fresh_db, c3, 8, t + 32, "GJ01AB1234", 0.94, [0.21, 0.3, 0.41, 0.5])

    result = fusion_engine.recompute(fresh_db, FusionConfig.from_settings("full"))
    # C1->C2 and C2->C3 accepted (the direct C1->C3 link is abstained as redundant).
    assert result["edges_accepted"] >= 2
    identities = repo.list_identities(fresh_db)
    assert len(identities) == 1
    identity = identities[0]
    assert identity.display_plate == "GJ01AB1234"
    assert identity.sighting_count == 3
    assert identity.status == "CONFIRMED"

    sightings = repo.sightings_for_identity(fresh_db, identity.id)
    cameras = [s.camera_id for s in sightings]
    assert cameras == [c1.id, c2.id, c3.id]


def test_fusion_abstains_when_two_candidates_close(fresh_db, camera_factory):
    """A sighting with two plausible follow-ups whose scores are near-equal must not produce a link."""
    c1 = camera_factory(0, "C1")
    c2 = camera_factory(1, "C2")
    t = time.time()
    emb = [0.2, 0.3, 0.4, 0.5]
    _make_sighting(fresh_db, c1, 1, t, "GJ01AB1234", 0.95, emb)
    # Two vehicles at C2 with near-identical timing and appearance → near-equal scores.
    _make_sighting(fresh_db, c2, 2, t + 16, "GJ01AB1234", 0.90, [0.2, 0.3, 0.4, 0.5])
    _make_sighting(fresh_db, c2, 3, t + 16.001, "GJ01AB1234", 0.90, [0.2, 0.3, 0.4, 0.5])

    result = fusion_engine.recompute(fresh_db, FusionConfig.from_settings("full"))
    assert result["edges_accepted"] == 0
    assert result["edges_abstained"] >= 1


def test_fusion_rejected_transition_persists_reason(fresh_db, camera_factory):
    c1 = camera_factory(0, "C1")
    c2 = camera_factory(1, "C2")
    c3 = camera_factory(2, "C3")
    t = time.time()
    # A strong chain C1->C2->C3 for one plate, plus a different-plate sighting at
    # C2 that should be rejected/abstained but still recorded with a reason.
    emb = [0.2, 0.3, 0.4, 0.5]
    _make_sighting(fresh_db, c1, 1, t, "GJ01AB1234", 0.95, emb)
    _make_sighting(fresh_db, c2, 2, t + 16, "GJ01AB1234", 0.95, emb)
    _make_sighting(fresh_db, c3, 3, t + 32, "GJ01AB1234", 0.95, emb)
    _make_sighting(fresh_db, c2, 4, t + 15, "GJ01XY9999", 0.95, [0.9, 0.1, 0.1, 0.1])

    fusion_engine.recompute(fresh_db, FusionConfig.from_settings("full"))
    edges = repo.list_edges(fresh_db)
    assert edges
    assert any(e.decision in ("REJECTED", "ABSTAINED") and e.decision_reason for e in edges)


def test_fusion_modes_change_weighting(fresh_db, camera_factory):
    """plate_only mode must ignore visual similarity."""
    c1 = camera_factory(0, "C1")
    c2 = camera_factory(1, "C2")
    t = time.time()
    s1 = _make_sighting(fresh_db, c1, 1, t, "GJ01AB1234", 0.95, [1.0, 0.0, 0.0, 0.0])
    s2 = _make_sighting(fresh_db, c2, 2, t + 16, "GJ01AB1234", 0.95, [0.0, 1.0, 0.0, 0.0])
    score_plate = candidate_score(s1, s2, c1.id, c2.id, FusionConfig.from_settings("plate_only"), route=_route())
    score_full = candidate_score(s1, s2, c1.id, c2.id, FusionConfig.from_settings("full"), route=_route())
    assert score_plate["combined_score"] == pytest.approx(score_plate["plate_score"], abs=0.05)
    assert score_full["combined_score"] < score_plate["combined_score"]


def test_identity_route_does_not_revisit_a_camera(fresh_db, camera_factory):
    """A reconstructed route must be a simple path through distinct cameras.

    Two sightings of the same plate at one camera plus one at a second camera
    must not chain into a route that visits the first camera twice.
    """
    c1 = camera_factory(0, "C1")
    c2 = camera_factory(1, "C2")
    t = time.time()
    emb = [0.2, 0.3, 0.4, 0.5]
    # Two distinct tracks at C1 (same plate + appearance), then one at C2.
    _make_sighting(fresh_db, c1, 1, t, "GJ01AB1234", 0.95, emb)
    _make_sighting(fresh_db, c1, 2, t + 0.5, "GJ01AB1234", 0.95, emb)
    _make_sighting(fresh_db, c2, 3, t + 16, "GJ01AB1234", 0.95, emb)

    fusion_engine.recompute(fresh_db, FusionConfig.from_settings("full"))
    for identity in repo.list_identities(fresh_db):
        cams = [s.camera_id for s in repo.sightings_for_identity(fresh_db, identity.id)]
        assert len(cams) == len(set(cams)), f"route revisits a camera: {cams}"
