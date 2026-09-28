import { GridCamera, Sighting, Trajectory, TrajectoryEdge } from "./grid-types";

/**
 * Pure helpers for trajectory rendering. Kept separate from the React/Leaflet
 * components so the route/edge logic is unit-testable.
 */

export type CameraLookup = Map<string, GridCamera>;

export function buildCameraLookup(cameras: GridCamera[]): CameraLookup {
  const m: CameraLookup = new Map();
  cameras.forEach((c) => m.set(c.id, c));
  return m;
}

/**
 * Resolve the geographic path for a trajectory edge. Uses the road-network
 * geometry when present, otherwise falls back to a straight line between the
 * two camera coordinates. Never invents points for unknown cameras.
 */
export function edgePositions(
  edge: TrajectoryEdge,
  lookup: CameraLookup
): [number, number][] {
  if (edge.path_geometry && edge.path_geometry.length >= 2) {
    return edge.path_geometry.map((p) => [p[0], p[1]] as [number, number]);
  }
  const a = lookup.get(edge.source_camera_id);
  const b = lookup.get(edge.destination_camera_id);
  if (!a || !b) return [];
  return [
    [a.latitude, a.longitude],
    [b.latitude, b.longitude],
  ];
}

/** Sightings that have usable coordinates, in the order the backend returned them. */
export function observedRoutePositions(trajectory: Trajectory | null): [number, number][] {
  if (!trajectory) return [];
  return trajectory.route
    .filter((s) => s.latitude != null && s.longitude != null)
    .map((s) => [s.latitude as number, s.longitude as number]);
}

export type LinkState = "OBSERVED" | "INFERRED" | "UNRESOLVED" | "REJECTED";

/**
 * Classify how a link should be presented. Accepted edges are INFERRED (the
 * movement between two observations was reconstructed), abstained edges are
 * UNRESOLVED, and rejected edges are REJECTED. Observed camera sightings are
 * always OBSERVED — an inferred segment is never shown as directly observed.
 */
export function classifyLink(decision: string): LinkState {
  switch (decision) {
    case "ACCEPTED":
      return "INFERRED";
    case "ABSTAINED":
      return "UNRESOLVED";
    default:
      return "REJECTED";
  }
}

export function acceptedEdges(trajectory: Trajectory | null): TrajectoryEdge[] {
  return (trajectory?.transitions || []).filter((e) => e.decision === "ACCEPTED");
}

export function unresolvedEdges(trajectory: Trajectory | null): TrajectoryEdge[] {
  return (trajectory?.unresolved_links || []).filter(
    (e) => e.decision === "ABSTAINED" || e.decision === "REJECTED"
  );
}

/** Ordered list of cameras actually observed, derived from sightings. */
export function observedCameraIds(trajectory: Trajectory | null): string[] {
  return (trajectory?.route || []).map((s: Sighting) => s.camera_id);
}

/**
 * True only when every accepted edge forms a contiguous chain over the observed
 * route — i.e. the rendered path cannot skip an observation. Used to avoid
 * drawing a route that the evidence does not support.
 */
export function isContiguousRoute(trajectory: Trajectory | null): boolean {
  if (!trajectory || trajectory.transitions.length < 2) return true;
  const order = observedCameraIds(trajectory);
  return trajectory.transitions.every((e, i) => {
    const from = order.indexOf(e.source_camera_id);
    const to = order.indexOf(e.destination_camera_id);
    return from !== -1 && to !== -1 && to > from;
  });
}
