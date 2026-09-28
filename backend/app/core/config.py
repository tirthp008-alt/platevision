"""Configuration settings for PlateVision API."""

import os
from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Accept a .env in the current working directory (backend/) or at the repository
# root, so both `cp .env.example backend/.env` and a root-level .env work when
# the server is started from backend/.
_ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    PROJECT_NAME: str = "PlateVision API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # CORS Configuration
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return [str(v)]

    # Detector settings
    DETECTOR_BACKEND: str = "auto"  # 'auto', 'onnx', or 'mock'
    ONNX_MODEL_PATH: str = os.getenv(
        "ONNX_MODEL_PATH",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "models",
            "plate_detector.onnx",
        ),
    )
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.25
    NMS_IOU_THRESHOLD: float = 0.45
    BBOX_PADDING_RATIO: float = 0.06

    # Security & Limits
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_IMAGE_MIME_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    ]
    TEMP_CROP_TTL_SECONDS: int = 300
    ENABLE_REDACTED_LOGS: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 120

    # ------------------------------------------------------------------
    # Drishti Grid — city-wide multi-camera ANPR trajectory tracking
    # ------------------------------------------------------------------

    # Persistence. SQLite by default so the prototype runs with zero setup.
    # A PostgreSQL/PostGIS URL (postgresql+psycopg://user:pw@host/db) enables
    # the PostGIS-backed geographic storage path.
    GRID_DATABASE_URL: str = "sqlite:///./drishti_grid.db"

    # Processing pipeline
    GRID_ENABLED: bool = True
    VEHICLE_DETECTOR_BACKEND: str = "hybrid"  # 'hybrid', 'motion', 'plate_anchored', 'yolo', 'synthetic', 'mock'
    VEHICLE_MODEL_PATH: str = os.getenv(
        "VEHICLE_MODEL_PATH",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "models",
            "vehicle_detector.onnx",
        ),
    )
    VEHICLE_DETECTION_CONFIDENCE: float = 0.30
    MOTION_MIN_AREA_RATIO: float = 0.0025
    MOTION_WARMUP_FRAMES: int = 25

    # Vehicle Re-ID / visual embedding
    REID_BACKEND: str = "auto"  # 'auto', 'onnx', 'classical'
    REID_MODEL_PATH: str = os.getenv(
        "REID_MODEL_PATH",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "models",
            "vehicle_reid.onnx",
        ),
    )
    REID_EMBEDDING_DIM: int = 512

    # Multi-frame OCR aggregation
    OCR_MIN_OBSERVATIONS: int = 2
    OCR_MIN_AGREEMENT: float = 0.55
    # Readings below this raw OCR confidence are stored for audit but never
    # accepted as plate evidence (degraded OCR must not create identity).
    OCR_MIN_READING_CONFIDENCE: float = 0.45

    # Cross-camera fusion weights — S_ij = λp·s_plate + λv·cos(vi,vj) − λt·|Δt − t_G| / σt
    FUSION_LAMBDA_PLATE: float = 0.50
    FUSION_LAMBDA_VISUAL: float = 0.35
    FUSION_LAMBDA_TIME: float = 0.15
    FUSION_SIGMA_T_SECONDS: float = 180.0
    # Confidence gate / abstention
    FUSION_TAU: float = 0.70
    FUSION_DELTA: float = 0.15
    # Default fusion mode can be overridden per-request for research comparison:
    # 'plate_only', 'plate_visual', 'full'
    FUSION_MODE: str = "full"

    # Road network / travel-time model
    ROAD_FALLBACK_SPEED_KMH: float = 34.0
    ROAD_FALLBACK_DETOUR_FACTOR: float = 1.35
    PYTHON_ROUTING_ENABLED: bool = True
    PYTHON_ROUTING_MAX_NODES: int = 25000
    # Extra checkpoints used to densify straight road edges when OSM graph is unavailable
    ROAD_NETWORK_EXTRA_CHARGERS: int = 0

    # Camera ingestion
    CAMERA_DISCOVERY_ENABLED: bool = True
    CAMERA_MAX_DEVICE_PROBE: int = 6
    PROCESSING_TARGET_FPS: float = 4.0
    PROCESSING_MAX_PROCESSING_MS: int = 250
    RECONNECT_BACKOFF_SECONDS: float = 2.0
    MAX_RECONNECT_ATTEMPTS: int = 10

    # Privacy / security
    ENABLE_AUDIT_LOG: bool = True
    PLATE_MASK_FOR_VIEWER: bool = True
    GRID_FRAME_JPEG_QUALITY: int = 80
    FRAME_STORE_TTL_SECONDS: int = 300
    FRAME_STORE_MAX_FRAMES: int = 400

    # Events (in-process default; 'kafka' when configured)
    EVENT_BACKEND: str = "memory"  # 'memory' | 'kafka'
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_SIGHTINGS: str = "drishti.sightings"

    # Demo / simulation mode (clearly separated from production pipeline).
    # Demo cameras sit ~63 m apart on a corridor (expected travel time ≈9 s at
    # the fallback model's 34 km/h), and the demo segment time matches it, so
    # accepted cross-camera links are physically plausible rather than tuned to
    # always pass the gate. Each route vehicle cycles the whole camera corridor.
    GRID_DEMO_MODE: bool = True
    GRID_DEMO_SPEED_FACTOR: float = 8.0
    GRID_DEMO_SEGMENT_SECONDS: float = 9.0
    GRID_DEMO_DWELL_SECONDS: float = 7.5
    GRID_DEMO_CORRIDOR_STEP_DEG: float = 0.000567  # ~63 m north per camera
    GRID_SEED_DEMO_ON_STARTUP: bool = False

    model_config = SettingsConfigDict(
        env_file=(str(_ROOT_ENV), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
