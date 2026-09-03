export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface NormalizedBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ImageMeta {
  width: number;
  height: number;
}

export interface DetectionItem {
  id: string;
  bounding_box: BoundingBox;
  normalized_box: NormalizedBox;
  detection_confidence: number;
  raw_text: string;
  normalized_text: string;
  formatted_text: string;
  ocr_confidence: number;
  format_status: "valid" | "possible" | "uncertain";
  crop_url: string;
  crop_base64?: string;
  preprocessed_base64?: string;
}

export interface DetectionResponse {
  request_id: string;
  processing_time_ms: number;
  image: ImageMeta;
  detections: DetectionItem[];
  warnings: string[];
}

export interface ApiError {
  error: {
    code: string;
    message: string;
    details?: any;
  };
}

export interface ScanHistoryRecord {
  id: string;
  timestamp: number;
  source: "camera" | "upload";
  detection: DetectionItem;
  crop_thumbnail?: string;
  user_notes?: string;
}

export interface HealthStatus {
  status: string;
  detector_ready: boolean;
  ocr_ready: boolean;
  version: string;
  backend_type: string;
}
