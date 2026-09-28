import {
  AuditLogResponse,
  CameraListResponse,
  DeviceListResponse,
  FusionConfigShape,
  GridCamera,
  HeatmapResponse,
  PlateSearchResponse,
  ProbeResponse,
  SystemStatus,
  TrafficAnalytics,
  Trajectory,
  VehicleIdentity,
} from "./grid-types";

const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "";

/**
 * API key storage for role-based access control. The key is kept in
 * sessionStorage (never localStorage) and only sent to the backend it was
 * entered for, so it is not persisted beyond the browser session.
 */
const KEY_STORAGE = "drishti.apiKey";

export function getApiKey(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(KEY_STORAGE);
}

export function setApiKey(key: string | null): void {
  if (typeof window === "undefined") return;
  if (key && key.trim()) {
    window.sessionStorage.setItem(KEY_STORAGE, key.trim());
  } else {
    window.sessionStorage.removeItem(KEY_STORAGE);
  }
}

export interface ApiErrorBody {
  detail?: string | { code?: string; message?: string };
  error?: { code?: string; message?: string };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const key = getApiKey();
  const headers: Record<string, string> = {
    ...(init?.body ? { "Content-Type": "application/json" } : {}),
    ...((init?.headers as Record<string, string>) || {}),
  };
  if (key) headers["X-API-Key"] = key;

  const res = await fetch(`${BASE_URL}/api/grid${path}`, { ...init, headers });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const body = data as ApiErrorBody;
    const detail = body?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message || body?.error?.message || res.statusText;
    const err = new Error(message) as Error & { status?: number; code?: string };
    err.status = res.status;
    err.code = (typeof detail === "object" ? detail?.code : undefined) || body?.error?.code;
    throw err;
  }
  return data as T;
}

// ---------------------------------------------------------------------------
// Cameras
// ---------------------------------------------------------------------------

export async function listCameraDevices(): Promise<DeviceListResponse> {
  return request<DeviceListResponse>("/cameras/devices");
}

export async function probeCameraSource(payload: {
  source_type: string;
  source_uri?: string;
  device_id?: string;
}): Promise<ProbeResponse> {
  return request<ProbeResponse>("/cameras/probe", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function listCameras(): Promise<CameraListResponse> {
  return request<CameraListResponse>("/cameras");
}

export async function createCamera(payload: {
  name: string;
  source_type: string;
  source_uri?: string;
  device_id?: string;
  latitude: number;
  longitude: number;
  road_segment_id?: string;
  enabled?: boolean;
  is_demo?: boolean;
}): Promise<GridCamera> {
  return request<GridCamera>("/cameras", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function updateCamera(
  id: string,
  payload: Partial<{
    name: string;
    source_type: string;
    source_uri: string;
    device_id: string;
    latitude: number;
    longitude: number;
    enabled: boolean;
  }>
): Promise<GridCamera> {
  return request<GridCamera>(`/cameras/${id}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function deleteCamera(id: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(`/cameras/${id}`, { method: "DELETE" });
}

export async function startCamera(id: string) {
  return request<{ camera_id: string; runtime: unknown }>(`/cameras/${id}/start`, {
    method: "POST",
  });
}

export async function stopCamera(id: string) {
  return request<{ camera_id: string; runtime: unknown }>(`/cameras/${id}/stop`, {
    method: "POST",
  });
}

export async function cameraStatus(id: string): Promise<import("./grid-types").CameraRuntime> {
  return request(`/cameras/${id}/status`);
}

export function cameraFrameUrl(id: string, cacheBust?: number): string {
  const suffix = cacheBust ? `?t=${cacheBust}` : "";
  return `${BASE_URL}/api/grid/cameras/${id}/frame${suffix}`;
}

// ---------------------------------------------------------------------------
// Vehicles / trajectories
// ---------------------------------------------------------------------------

export async function searchPlate(plate: string): Promise<PlateSearchResponse> {
  return request<PlateSearchResponse>(
    `/vehicles/search?plate=${encodeURIComponent(plate)}`
  );
}

export async function listVehicles(): Promise<{ vehicles: VehicleIdentity[]; count: number }> {
  return request<{ vehicles: VehicleIdentity[]; count: number }>("/vehicles");
}

export async function vehicleTrajectory(id: string): Promise<Trajectory> {
  return request<Trajectory>(`/vehicles/${id}/trajectory`);
}

export async function trajectoryByPlate(plate: string): Promise<Trajectory & { found: boolean }> {
  return request<Trajectory & { found: boolean }>(
    `/trajectory/plate?plate=${encodeURIComponent(plate)}`
  );
}

export async function listUnresolved(): Promise<{
  unresolved: import("./grid-types").TrajectoryEdge[];
  count: number;
  note: string;
}> {
  return request("/unresolved");
}

// ---------------------------------------------------------------------------
// Analytics / system
// ---------------------------------------------------------------------------

export async function trafficAnalytics(windowSeconds = 3600): Promise<TrafficAnalytics> {
  return request<TrafficAnalytics>(`/analytics/traffic?window_seconds=${windowSeconds}`);
}

export async function heatmap(windowSeconds = 3600): Promise<HeatmapResponse> {
  return request<HeatmapResponse>(`/analytics/heatmap?window_seconds=${windowSeconds}`);
}

export async function systemStatus(): Promise<SystemStatus> {
  return request<SystemStatus>("/system/status");
}

export async function getFusionConfig(): Promise<FusionConfigShape> {
  return request<FusionConfigShape>("/fusion/config");
}

export async function updateFusionConfig(
  payload: Partial<FusionConfigShape>
): Promise<FusionConfigShape> {
  return request<FusionConfigShape>("/fusion/config", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function recomputeFusion(mode?: string) {
  return request<{ metrics: Record<string, unknown> }>("/fusion/recompute", {
    method: "POST",
    body: JSON.stringify({ mode }),
  });
}

export async function auditLog(limit = 200): Promise<AuditLogResponse> {
  return request<AuditLogResponse>(`/audit?limit=${limit}`);
}

export async function setupDemoCameras(): Promise<{
  created_camera_ids: string[];
  demo_cameras: { id: string; name: string; latitude: number; longitude: number }[];
  runtime: unknown[];
  note: string;
}> {
  return request("/demo/setup", {
    method: "POST",
    body: JSON.stringify({ start_processing: true }),
  });
}
