import { describe, expect, it } from "vitest";
import {
  acceptedEdges,
  buildCameraLookup,
  classifyLink,
  edgePositions,
  isContiguousRoute,
  observedCameraIds,
  observedRoutePositions,
  unresolvedEdges,
} from "../lib/grid-trajectory";
import { GridCamera, Trajectory, TrajectoryEdge } from "../lib/grid-types";

function cam(id: string, lat: number, lon: number): GridCamera {
  return {
    id,
    name: id,
    source_type: "demo",
    source_uri: "demo",
    latitude: lat,
    longitude: lon,
    enabled: true,
    status: "PROCESSING",
    is_demo: true,
    created_at: 0,
  };
}

function edge(partial: Partial<TrajectoryEdge>): TrajectoryEdge {
  return {
    id: partial.id || "e1",
    source_camera_id: partial.source_camera_id || "C1",
    destination_camera_id: partial.destination_camera_id || "C2",
    source_sighting_id: partial.source_sighting_id || "s1",
    destination_sighting_id: partial.destination_sighting_id || "s2",
    time_delta: partial.time_delta ?? 16,
    expected_travel_time: partial.expected_travel_time ?? 16,
    plate_score: partial.plate_score ?? 0.9,
    visual_similarity: partial.visual_similarity ?? 0.8,
    temporal_score: partial.temporal_score ?? 0.9,
    combined_score: partial.combined_score ?? 0.85,
    decision: partial.decision || "ACCEPTED",
    decision_reason: partial.decision_reason || "",
    path_geometry: partial.path_geometry ?? null,
  };
}

describe("edge geometry", () => {
  const lookup = buildCameraLookup([cam("C1", 23.0, 72.0), cam("C2", 23.01, 72.0)]);

  it("uses road-network geometry when available", () => {
    const e = edge({
      path_geometry: [
        [23.0, 72.0],
        [23.005, 72.001],
        [23.01, 72.0],
      ],
    });
    const pos = edgePositions(e, lookup);
    expect(pos).toHaveLength(3);
    expect(pos[1]).toEqual([23.005, 72.001]);
  });

  it("falls back to a straight line between known cameras", () => {
    const pos = edgePositions(edge({}), lookup);
    expect(pos).toEqual([
      [23.0, 72.0],
      [23.01, 72.0],
    ]);
  });

  it("returns no path for unknown cameras instead of fabricating points", () => {
    const pos = edgePositions(edge({ source_camera_id: "X", destination_camera_id: "Y" }), lookup);
    expect(pos).toEqual([]);
  });
});

describe("link classification", () => {
  it("maps accepted -> INFERRED, abstained -> UNRESOLVED, else REJECTED", () => {
    expect(classifyLink("ACCEPTED")).toBe("INFERRED");
    expect(classifyLink("ABSTAINED")).toBe("UNRESOLVED");
    expect(classifyLink("REJECTED")).toBe("REJECTED");
    expect(classifyLink("anything")).toBe("REJECTED");
  });

  it("never labels an inferred segment as OBSERVED", () => {
    expect(classifyLink("ACCEPTED")).not.toBe("OBSERVED");
  });
});

describe("trajectory route helpers", () => {
  const traj: Trajectory = {
    identity: {
      id: "veh1",
      display_plate: "GJ01AB1234",
      status: "CONFIRMED",
      sighting_count: 3,
      confidence: 0.9,
    },
    route: [
      {
        sighting_id: "s1",
        camera_id: "C1",
        camera_name: "C1",
        latitude: 23.0,
        longitude: 72.0,
        timestamp: 100,
        vehicle_local_track_id: 1,
        plate: "GJ01AB1234",
        plate_confidence: 0.9,
        observation_type: "OBSERVED",
      },
      {
        sighting_id: "s2",
        camera_id: "C2",
        camera_name: "C2",
        latitude: 23.01,
        longitude: 72.0,
        timestamp: 116,
        vehicle_local_track_id: 2,
        plate: "GJ01AB1234",
        plate_confidence: 0.9,
        observation_type: "OBSERVED",
      },
      {
        sighting_id: "s3",
        camera_id: "C3",
        camera_name: "C3",
        latitude: 23.02,
        longitude: 72.0,
        timestamp: 132,
        vehicle_local_track_id: 3,
        plate: "GJ01AB1234",
        plate_confidence: 0.9,
        observation_type: "OBSERVED",
      },
    ],
    transitions: [
      edge({ id: "a", source_camera_id: "C1", destination_camera_id: "C2", source_sighting_id: "s1", destination_sighting_id: "s2" }),
      edge({ id: "b", source_camera_id: "C2", destination_camera_id: "C3", source_sighting_id: "s2", destination_sighting_id: "s3" }),
      edge({ id: "u", source_camera_id: "C1", destination_camera_id: "C3", decision: "ABSTAINED" }),
    ],
    unresolved_links: [edge({ id: "u2", source_camera_id: "C1", destination_camera_id: "C3", decision: "ABSTAINED" })],
    route_camera_ids: ["C1", "C2", "C3"],
    has_inferred_links: true,
  };

  it("extracts observed camera order", () => {
    expect(observedCameraIds(traj)).toEqual(["C1", "C2", "C3"]);
  });

  it("returns observed positions with coordinates", () => {
    expect(observedRoutePositions(traj)).toEqual([
      [23.0, 72.0],
      [23.01, 72.0],
      [23.02, 72.0],
    ]);
  });

  it("splits accepted and unresolved edges", () => {
    expect(acceptedEdges(traj).map((e) => e.id)).toEqual(["a", "b"]);
    expect(unresolvedEdges(traj)).toHaveLength(1);
  });

  it("recognises a contiguous forward chain", () => {
    expect(isContiguousRoute(traj)).toBe(true);
  });

  it("flags a non-contiguous chain", () => {
    const broken: Trajectory = {
      ...traj,
      transitions: [
        edge({ id: "x", source_camera_id: "C2", destination_camera_id: "C1" }),
        edge({ id: "y", source_camera_id: "C1", destination_camera_id: "C3" }),
      ],
    };
    expect(isContiguousRoute(broken)).toBe(false);
  });

  it("handles null trajectory without throwing", () => {
    expect(observedRoutePositions(null)).toEqual([]);
    expect(acceptedEdges(null)).toEqual([]);
    expect(isContiguousRoute(null)).toBe(true);
  });
});
