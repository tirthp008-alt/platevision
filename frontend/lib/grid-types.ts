/**
 * Types for the Drishti Grid city-wide multi-camera ANPR trajectory system.
 * Mirrors the FastAPI schemas in backend/app/schemas/grid.py.
 */

export type CameraStatus =
  | "ONLINE"
  | "OFFLINE"
  | "CONNECTING"
  | "PROCESSING"
  | "ERROR";

export type SourceType = "device" | "rtsp" | "file" | "demo";

export interface CameraDeviceInfo {
  id: string;
  index: number;
  name: string;
  available: boolean;
  backend?: string;
}

export interface CameraRuntime {
  camera_id: string;
  name?: string;
  status: CameraStatus;
  is_running: boolean;
  fps?: number;
  frames_processed?: number;
  tracked_vehicles?: number;
  sightings_created?: number;
  last_frame_at?: number;
  error?: string | null;
  recent_sightings?: CameraRecentSighting[];
  sighting_count?: number;
}

export interface CameraRecentSighting {
  sighting_id: string;
  plate: string;
  plate_confidence: number;
  timestamp: number;
  track_id: number;
}

export interface GridCamera {
  id: string;
  name: string;
  source_type: SourceType;
  source_uri: string;
  device_id?: string | null;
  latitude: number;
  longitude: number;
  road_segment_id?: string | null;
  enabled: boolean;
  status: CameraStatus;
  is_demo: boolean;
  created_at: number;
  last_seen?: number | null;
  last_error?: string | null;
  runtime?: CameraRuntime | null;
  sighting_count?: number;
}

export interface CameraListResponse {
  cameras: GridCamera[];
  count: number;
}

export interface DeviceListResponse {
  devices: CameraDeviceInfo[];
  supported_source_types: SourceType[];
  note: string;
}

export interface ProbeResponse {
  status: "ok" | "error";
  message: string;
  source_type?: string;
  resolution?: string;
  latency_ms?: number;
}

export interface Sighting {
  sighting_id: string;
  camera_id: string;
  camera_name: string;
  latitude?: number | null;
  longitude?: number | null;
  timestamp: number;
  vehicle_local_track_id: number;
  plate: string;
  plate_raw?: string;
  plate_confidence: number;
  plate_format_status?: string;
  plate_candidates?: string[];
  visual_embedding_dim?: number;
  vehicle_type?: string;
  vehicle_color?: string;
  vehicle_bbox?: number[] | null;
  frame_reference?: string | null;
  detection_confidence?: number;
  observation_type: "OBSERVED";
}

export type EdgeDecision = "ACCEPTED" | "ABSTAINED" | "REJECTED";

export interface TrajectoryEdge {
  id: string;
  source_camera_id: string;
  destination_camera_id: string;
  source_camera_name?: string;
  destination_camera_name?: string;
  source_sighting_id: string;
  destination_sighting_id: string;
  time_delta: number;
  expected_travel_time: number;
  plate_score: number;
  visual_similarity: number;
  temporal_score: number;
  combined_score: number;
  decision: EdgeDecision;
  decision_reason: string;
  path_geometry?: [number, number][] | null;
  inference?: "INFERRED";
}

export interface VehicleIdentity {
  id: string;
  display_plate: string;
  status: "PROVISIONAL" | "CONFIRMED" | "AMBIGUOUS";
  first_seen?: number | null;
  last_seen?: number | null;
  sighting_count: number;
  confidence: number;
  plate_hypotheses?: unknown[];
}

export interface Trajectory {
  identity: VehicleIdentity;
  route: Sighting[];
  transitions: TrajectoryEdge[];
  unresolved_links: TrajectoryEdge[];
  route_camera_ids: string[];
  first_seen?: number | null;
  last_seen?: number | null;
  inferred_total_travel_time?: number | null;
  has_inferred_links: boolean;
  notes?: string;
}

export interface PlateSearchMatch {
  identity: VehicleIdentity;
  score: number;
  matched_sightings: number;
  cameras: string[];
}

export interface PlateSearchResponse {
  query: {
    raw: string;
    cleaned: string;
    normalized: string;
    formatted?: string;
    format_status?: string;
    variants: string[];
  };
  matches: PlateSearchMatch[];
  count: number;
}

export interface FusionConfigShape {
  lambda_plate: number;
  lambda_visual: number;
  lambda_temporal: number;
  sigma_t: number;
  tau: number;
  delta: number;
  mode: string;
}

export interface TrafficSegment {
  segment: string;
  source_camera: string;
  source_camera_name?: string;
  destination_camera: string;
  destination_camera_name?: string;
  distance_m: number;
  expected_travel_time: number;
  vehicle_count: number;
  average_speed_kmh: number;
  average_speed_ms: number;
  average_travel_time: number;
  flow_vph: number;
  capacity_vph: number;
  congestion_ratio: number;
  approximate: boolean;
}

export interface CameraActivity {
  camera_id: string;
  name: string;
  status: CameraStatus;
  sightings: number;
  latitude: number;
  longitude: number;
}

export interface TrafficAnalytics {
  window_seconds: number;
  generated_at: number;
  totals: {
    sightings: number;
    accepted_transitions: number;
    unique_vehicles: number;
    active_cameras: number;
  };
  segments: TrafficSegment[];
  camera_activity: CameraActivity[];
  notes?: string;
}

export interface HeatmapPoint {
  latitude: number;
  longitude: number;
  camera_id: string;
  name: string;
  count: number;
  intensity: number;
  status: CameraStatus;
}

export interface HeatmapResponse {
  window_seconds: number;
  max_count: number;
  points: HeatmapPoint[];
}

export interface SystemStatus {
  cameras: number;
  cameras_enabled: number;
  cameras_processing: number;
  aggregate: {
    cameras_registered: number;
    cameras_processing: number;
    frames_processed: number;
    sightings_created: number;
    plates_read: number;
    tracked_vehicles: number;
    average_fps: number;
  };
  fusion: Record<string, unknown>;
  database: string;
  detector: string;
  vehicle_detector: string;
  reid_backend: string;
  demo_mode: boolean;
  event_backend: string;
  uptime_note?: string;
}

export interface AuditLogEntry {
  id: string;
  timestamp: number;
  actor: string;
  role: string;
  action: string;
  target: string;
  detail?: string | null;
  client_ip?: string | null;
  result: string;
}

export interface AuditLogResponse {
  audit_logs: AuditLogEntry[];
  count: number;
}
