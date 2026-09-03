"""Configuration settings for PlateVision API."""

import os
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
