from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    baseline_model: bool = True
    app_env: str = 'development'
    detector_mode: str = 'onnx'
    model_path: str = '../models/plate-detector.onnx'
    detector_name: str = 'YOLOv8n baseline'
    detector_retrained: bool = False
    vehicle_model_path: str = ''
    vehicle_context_enabled: bool = True
    vehicle_taxonomy: str = 'coco'
    vehicle_reid_model_path: str = '../models/reid/veri-sbs-r50-ibn.onnx'
    vehicle_confidence_threshold: float = Field(default=.30, ge=0, le=1)
    vehicle_plate_search_enabled: bool = True
    vehicle_plate_max_crops: int = Field(default=8, ge=1, le=64)
    region_confidence_threshold: float = Field(default=.25, ge=0, le=1)
    region_supplement_path: str = ''
    plate_tile_size: int = Field(default=480, ge=320, le=1280)
    plate_tile_overlap: float = Field(default=.25, ge=.1, le=.5)
    plate_refinement_enabled: bool = True
    max_plate_refinements: int = Field(default=12, ge=0, le=50)
    onnx_provider: str = 'CPUExecutionProvider'
    inference_threads: int = Field(default=4, ge=1, le=32)
    acceleration: str = 'auto'
    openvino_device: str = 'GPU'
    fast_ocr_batch_size: int = Field(default=16, ge=1, le=32)
    resnet_ocr_model_path: str = '../models/resnet34/inference.pdmodel'
    video_max_upload_mb: int = 512
    video_max_seconds: int = 900
    video_output_dir: str = 'video-results'
    video_result_ttl_seconds: int = 3600
    input_size: int = Field(default=640, ge=320, le=1280)
    confidence_threshold: float = Field(default=.25, ge=0, le=1)
    unverified_plate_threshold: float = Field(default=.55, ge=0, le=1)
    plate_crop_padding: float = Field(default=.25, ge=0, le=.4)
    max_detections: int = Field(default=100, ge=1, le=500)
    max_upload_mb: int = 10
    cors_origins: str = 'http://localhost:3000,http://localhost:8080,http://127.0.0.1:8080'
    result_ttl_seconds: int = 60
    max_cached_crops: int = 200

settings = Settings()
