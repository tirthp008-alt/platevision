"""Pydantic schemas for detection and OCR requests & responses."""

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(..., description="Top-left X coordinate in pixels")
    y: int = Field(..., description="Top-left Y coordinate in pixels")
    width: int = Field(..., description="Width in pixels")
    height: int = Field(..., description="Height in pixels")


class NormalizedBox(BaseModel):
    x: float = Field(..., description="Normalized top-left X (0.0 - 1.0)")
    y: float = Field(..., description="Normalized top-left Y (0.0 - 1.0)")
    width: float = Field(..., description="Normalized width (0.0 - 1.0)")
    height: float = Field(..., description="Normalized height (0.0 - 1.0)")


class ImageMeta(BaseModel):
    width: int
    height: int


class DetectionItem(BaseModel):
    id: str = Field(..., description="Unique UUID for this detection")
    bounding_box: BoundingBox
    normalized_box: NormalizedBox
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    raw_text: str = Field(..., description="Unmodified OCR output text")
    normalized_text: str = Field(..., description="Alphanumeric uppercase normalized registration number")
    formatted_text: str = Field(..., description="Prettified display format e.g. 'GJ 01 AB 1234'")
    ocr_confidence: float = Field(..., ge=0.0, le=1.0)
    format_status: str = Field(..., description="'valid', 'possible', or 'uncertain'")
    crop_url: str = Field(..., description="Endpoint URL to fetch cropped plate")
    crop_base64: Optional[str] = Field(None, description="Base64 encoded cropped plate image")
    preprocessed_base64: Optional[str] = Field(None, description="Base64 encoded preprocessed OCR variant for comparison")


class DetectionResponse(BaseModel):
    request_id: str
    processing_time_ms: int
    image: ImageMeta
    detections: List[DetectionItem]
    warnings: List[str] = []


class HealthResponse(BaseModel):
    status: str
    detector_ready: bool
    ocr_ready: bool
    version: str
    backend_type: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
