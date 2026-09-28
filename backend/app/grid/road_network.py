"""Road-network abstraction and travel-time estimation.

Provides :func:`expected_travel_time` and :func:`path_geometry` behind a single
replaceable interface. Resolution order:

1. **OSM / routing engine** — if a real routing engine is configured it is used
   first. Two integrations are supported without changing call sites:
   * ``ORS_API_KEY`` (openrouteservice) queried over HTTPS, and
   * an ``osmnx``-derived drivable graph cached locally (used when the package
     and graph are available).
2. **Graph fallback approximation** — a Dijkstra search over a camera-node graph
   whose edge cost is ``haversine_distance × detour_factor / assumed_speed``.
   The assumed speed model is clearly an approximation, not real traffic data.
3. **Direct fallback** — straight-line distance with the detour factor.

Every result carries a ``source`` field so the dashboard can label inferred
segments as approximate rather than observed.
"""

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

from app.core.config import settings
from app.core.logging import logger

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class RoadRoute:
    distance_m: float
    expected_travel_time: float  # seconds
    source: str  # 'openrouteservice' | 'osmnx' | 'graph_fallback' | 'direct_fallback'
    geometry: List[List[float]] = field(default_factory=list)  # [[lat, lon], ...]
    approximate: bool = True
    path_node_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "distance_m": round(self.distance_m, 1),
            "expected_travel_time": round(self.expected_travel_time, 1),
            "source": self.source,
            "approximate": self.approximate,
            "geometry": self.geometry,
            "path_node_ids": self.path_node_ids,
        }


def _densify(a: Tuple[float, float], b: Tuple[float, float], segments: int = 8) -> List[List[float]]:
    lat1, lon1 = a
    lat2, lon2 = b
    pts = []
    for i in range(segments + 1):
        t = i / float(segments)
        pts.append([lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t])
    return pts


class RoadNetwork:
    """Replaceable road-network / travel-time provider."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cameras: Dict[str, Tuple[float, float]] = {}
        self._route_cache: Dict[Tuple[str, str], RoadRoute] = {}
        self._ors_key = os.getenv("ORS_API_KEY", "").strip()
        self._osmnx_graph = None
        self._osmnx_nodes = None
        self._load_osmnx_graph()
        if self._ors_key:
            logger.info("Road network: openrouteservice routing enabled.")
        elif self._osmnx_graph is not None:
            logger.info("Road network: local osmnx drivable graph loaded.")
        else:
            logger.info(
                "Road network: using graph/direct fallback approximation "
                f"(speed={settings.ROAD_FALLBACK_SPEED_KMH} km/h, detour={settings.ROAD_FALLBACK_DETOUR_FACTOR})."
            )

    # ------------------------------------------------------------------
    def _load_osmnx_graph(self) -> None:
        if not settings.PYTHON_ROUTING_ENABLED:
            return
        graph_path = os.getenv("OSMNX_GRAPH_PATH", "").strip()
        if not graph_path or not os.path.exists(graph_path):
            return
        try:
            import osmnx as ox  # type: ignore
            import networkx as nx  # type: ignore

            self._osmnx_graph = ox.load_graphml(graph_path)
            self._osmnx_nodes = nx
            logger.info(f"Loaded osmnx graph from {graph_path}")
        except Exception as e:
            logger.warning(f"Could not load osmnx graph ({e}); using fallback.")

    def register_cameras(self, cameras: List[dict]) -> None:
        """Provide camera coordinates so the graph fallback can route and cache."""
        with self._lock:
            self._cameras = {
                c["id"]: (float(c["latitude"]), float(c["longitude"])) for c in cameras
            }
            self._route_cache.clear()

    # ------------------------------------------------------------------
    def _ors_route(self, a: Tuple[float, float], b: Tuple[float, float]) -> Optional[RoadRoute]:
        if not self._ors_key:
            return None
        try:
            url = "https://api.openrouteservice.org/v2/directions/driving-car"
            headers = {"Authorization": self._ors_key, "Content-Type": "application/json"}
            body = {"coordinates": [[a[1], a[0]], [b[1], b[0]]]}
            resp = requests.post(url, headers=headers, json=body, timeout=8)
            if resp.status_code != 200:
                return None
            data = resp.json()
            feat = data["features"][0]
            summary = feat["properties"]["summary"]
            coords = feat["geometry"]["coordinates"]  # [lon, lat]
            geometry = [[c[1], c[0]] for c in coords]
            return RoadRoute(
                distance_m=float(summary["distance"]),
                expected_travel_time=float(summary["duration"]),
                source="openrouteservice",
                geometry=geometry,
                approximate=False,
            )
        except Exception as e:
            logger.debug(f"openrouteservice route failed: {e}")
            return None

    def _osmnx_route(self, a: Tuple[float, float], b: Tuple[float, float]) -> Optional[RoadRoute]:
        if self._osmnx_graph is None:
            return None
        try:
            import osmnx as ox  # type: ignore

            orig = ox.distance.nearest_nodes(self._osmnx_graph, a[1], a[0])
            dest = ox.distance.nearest_nodes(self._osmnx_graph, b[1], b[0])
            route = ox.shortest_path(self._osmnx_graph, orig, dest, weight="length")
            if not route:
                return None
            length = sum(
                self._osmnx_graph[u][v][0].get("length", 0.0)
                for u, v in zip(route[:-1], route[1:])
            )
            speed_ms = settings.ROAD_FALLBACK_SPEED_KMH / 3.6
            geometry = [
                [self._osmnx_graph.nodes[n]["y"], self._osmnx_graph.nodes[n]["x"]] for n in route
            ]
            return RoadRoute(
                distance_m=float(length),
                expected_travel_time=float(length / speed_ms),
                source="osmnx",
                geometry=geometry,
                approximate=True,
            )
        except Exception as e:
            logger.debug(f"osmnx route failed: {e}")
            return None

    def _fallback(self, a: Tuple[float, float], b: Tuple[float, float]) -> RoadRoute:
        speed_ms = max(1.0, settings.ROAD_FALLBACK_SPEED_KMH / 3.6)
        detour = settings.ROAD_FALLBACK_DETOUR_FACTOR
        straight = haversine_m(a[0], a[1], b[0], b[1])
        distance = straight * detour
        return RoadRoute(
            distance_m=distance,
            expected_travel_time=distance / speed_ms,
            source="direct_fallback",
            geometry=_densify(a, b, segments=10),
            approximate=True,
        )

    def _graph_route(self, cam_a: str, cam_b: str) -> Optional[RoadRoute]:
        """Dijkstra over camera nodes using the fallback speed model."""
        if cam_a not in self._cameras or cam_b not in self._cameras:
            return None
        if cam_a == cam_b:
            return None
        nodes = list(self._cameras.keys())
        speed_ms = max(1.0, settings.ROAD_FALLBACK_SPEED_KMH / 3.6)
        detour = settings.ROAD_FALLBACK_DETOUR_FACTOR

        dist: Dict[str, float] = {n: float("inf") for n in nodes}
        prev: Dict[str, Optional[str]] = {n: None for n in nodes}
        dist[cam_a] = 0.0
        visited = set()
        while True:
            current = None
            best = float("inf")
            for n in nodes:
                if n not in visited and dist[n] < best:
                    best, current = dist[n], n
            if current is None or current == cam_b:
                break
            visited.add(current)
            clat, clon = self._cameras[current]
            for n in nodes:
                if n in visited or n == current:
                    continue
                nlat, nlon = self._cameras[n]
                w = haversine_m(clat, clon, nlat, nlon) * detour
                if dist[current] + w < dist[n]:
                    dist[n] = dist[current] + w
                    prev[n] = current

        if math.isinf(dist[cam_b]):
            return None
        path: List[str] = []
        node: Optional[str] = cam_b
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()

        geometry: List[List[float]] = []
        for u, v in zip(path[:-1], path[1:]):
            seg = _densify(self._cameras[u], self._cameras[v], segments=8)
            geometry.extend(seg[:-1])
        geometry.append(list(self._cameras[path[-1]]))
        return RoadRoute(
            distance_m=dist[cam_b],
            expected_travel_time=dist[cam_b] / speed_ms,
            source="graph_fallback",
            geometry=geometry,
            approximate=True,
            path_node_ids=path,
        )

    # ------------------------------------------------------------------
    def route(self, cam_a: str, cam_b: str) -> RoadRoute:
        """Return the road route between two camera ids (cached)."""
        cache_key = (cam_a, cam_b)
        with self._lock:
            cached = self._route_cache.get(cache_key)
        if cached is not None:
            return cached

        loc_a = self._cameras.get(cam_a)
        loc_b = self._cameras.get(cam_b)
        if loc_a is None or loc_b is None:
            result = RoadRoute(0.0, 0.0, "unknown", [], True)
            return result

        result = self._ors_route(loc_a, loc_b) or self._osmnx_route(loc_a, loc_b)
        if result is None:
            result = self._graph_route(cam_a, cam_b) or self._fallback(loc_a, loc_b)

        with self._lock:
            self._route_cache[cache_key] = result
        return result

    def expected_travel_time(self, cam_a: str, cam_b: str) -> float:
        """Expected travel time in seconds between two cameras (t_G(c_i, c_j))."""
        return self.route(cam_a, cam_b).expected_travel_time

    def path_geometry(self, cam_a: str, cam_b: str) -> List[List[float]]:
        return self.route(cam_a, cam_b).geometry

    def road_distance(self, cam_a: str, cam_b: str) -> float:
        return self.route(cam_a, cam_b).distance_m


road_network = RoadNetwork()
