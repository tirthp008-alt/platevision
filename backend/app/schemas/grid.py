"""Pydantic request/response schemas for the Drishti Grid APIs."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)
    source_type: str = Field("device", description="device | rtsp | file | demo")
    source_uri: str = Field("", description="RTSP/HTTP URL or file path (empty for local device)")
    device_id: Optional[str] = Field(None, description="e.g. 'device:0'")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    road_segment_id: Optional[str] = None
    enabled: bool = True
    is_demo: bool = False


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    source_type: Optional[str] = None
    source_uri: Optional[str] = None
    device_id: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    road_segment_id: Optional[str] = None
    enabled: Optional[bool] = None


class StreamProbeRequest(BaseModel):
    source_type: str = "rtsp"
    source_uri: str = ""
    device_id: Optional[str] = None


class FusionConfigRequest(BaseModel):
    lambda_plate: Optional[float] = Field(None, ge=0, le=1)
    lambda_visual: Optional[float] = Field(None, ge=0, le=1)
    lambda_time: Optional[float] = Field(None, ge=0, le=5)
    sigma_t: Optional[float] = Field(None, gt=0)
    tau: Optional[float] = Field(None, ge=0, le=1)
    delta: Optional[float] = Field(None, ge=0, le=1)
    mode: Optional[str] = Field(None, description="plate_only | plate_visual | full")


class FusionRecomputeRequest(BaseModel):
    mode: Optional[str] = None


class DemoSetupRequest(BaseModel):
    center_latitude: float = 23.0225
    center_longitude: float = 72.5714
    start_processing: bool = True
    auto_fuse: bool = True


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
