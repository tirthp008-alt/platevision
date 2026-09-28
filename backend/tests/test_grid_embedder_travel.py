"""Unit tests for visual embeddings, cosine similarity and road travel time."""

import numpy as np
import pytest

from app.grid.embedder import cosine_similarity, dominant_color_name
from app.grid.road_network import haversine_m, road_network


# --------------------------------------------------------------------------
# 4. Cosine visual similarity
# --------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors():
    v = [0.1, 0.2, 0.3, 0.4]
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_similarity_handles_empty():
    assert cosine_similarity([], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_embedding_is_normalised_and_discriminative():
    from app.grid.embedder import _classical_embedding

    blue = np.zeros((80, 60, 3), np.uint8)
    blue[:, :] = (200, 60, 30)  # BGR blue-ish
    red = np.zeros((80, 60, 3), np.uint8)
    red[:, :] = (30, 40, 210)  # BGR red-ish

    eb = _classical_embedding(blue)
    er = _classical_embedding(red)
    norm = float(np.linalg.norm(np.asarray(eb)))
    assert norm == pytest.approx(1.0, abs=1e-4)
    assert cosine_similarity(eb, _classical_embedding(blue)) > cosine_similarity(eb, er)


def test_dominant_color_name():
    blue = np.zeros((40, 40, 3), np.uint8)
    blue[:, :] = (200, 60, 30)
    assert dominant_color_name(blue) == "blue"
    grey = np.full((40, 40, 3), 128, np.uint8)
    assert dominant_color_name(grey) in ("grey", "black", "white")


# --------------------------------------------------------------------------
# 5. Travel-time / road network
# --------------------------------------------------------------------------

def test_haversine_known_distance():
    # ~111 km per degree of latitude.
    d = haversine_m(23.0, 72.0, 24.0, 72.0)
    assert 110_000 < d < 112_500


def test_expected_travel_time_positive_and_ordered():
    road_network.register_cameras(
        [
            {"id": "A", "latitude": 23.0225, "longitude": 72.5714},
            {"id": "B", "latitude": 23.023519, "longitude": 72.5714},
            {"id": "C", "latitude": 23.024538, "longitude": 72.5714},
        ]
    )
    t1 = road_network.expected_travel_time("A", "B")
    t2 = road_network.expected_travel_time("A", "C")
    assert t1 > 0
    assert t2 > t1  # farther camera takes longer


def test_travel_time_direct_fallback_matches_speed_model():
    road_network.register_cameras(
        [
            {"id": "D", "latitude": 0.0, "longitude": 0.0},
            {"id": "E", "latitude": 0.0, "longitude": 0.01},
        ]
    )
    route = road_network.route("D", "E")
    assert route.distance_m > 0
    # 34 km/h -> 9.44 m/s
    assert route.expected_travel_time == pytest.approx(route.distance_m / (34.0 / 3.6), rel=1e-3)


def test_road_network_geometry_is_densified():
    road_network.register_cameras(
        [
            {"id": "F", "latitude": 23.0, "longitude": 72.0},
            {"id": "G", "latitude": 23.01, "longitude": 72.0},
        ]
    )
    geom = road_network.path_geometry("F", "G")
    assert len(geom) >= 2
    assert all(len(pt) == 2 for pt in geom)
