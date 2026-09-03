"""Live Traffic and CCTV Surveillance stream endpoints with connection probing and frame rate controls."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.cctv_stream import cctv_manager

router = APIRouter(prefix="/cctv", tags=["Traffic & CCTV Surveillance"])


class CCTVStartRequest(BaseModel):
    stream_url: str = "demo_traffic"  # Can be "demo_traffic", "rtsp://...", ".m3u8", "http://...", or YouTube link
    analysis_fps: int = 3  # Frame analysis rate (1 to 10 FPS)


class CCTVProbeRequest(BaseModel):
    stream_url: str


@router.post("/probe")
async def probe_traffic_stream(payload: CCTVProbeRequest):
    """Probes the target stream URL without starting ingestion, checking reachability, codec, and latency."""
    res = cctv_manager.probe_stream(url=payload.stream_url)
    return res


@router.post("/start")
async def start_cctv_stream(payload: CCTVStartRequest):
    """Starts live traffic stream ingestion from RTSP, HLS, MJPEG, YouTube, or built-in highway simulations."""
    res = cctv_manager.start_stream(stream_url=payload.stream_url, analysis_fps=payload.analysis_fps)
    return res


@router.post("/stop")
async def stop_cctv_stream():
    """Stops the active traffic stream ingestion."""
    return cctv_manager.stop_stream()


@router.get("/status")
async def get_cctv_status():
    """Returns the current connection state, FPS, analysis rate, protocol, and recent plate detection events."""
    return cctv_manager.get_status()


@router.get("/feed")
async def get_cctv_mjpeg_feed():
    """Streams live CCTV video frames with real-time HUD annotations via MJPEG."""
    if not cctv_manager.is_running:
        cctv_manager.start_stream("demo_traffic", analysis_fps=3)

    return StreamingResponse(
        cctv_manager.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/events")
async def get_cctv_events(
    limit: int = Query(50, ge=1, le=200),
    state_code: Optional[str] = Query(None, description="Filter by 2-letter state code e.g. GJ, MH, DL, BH"),
    status_filter: Optional[str] = Query(None, description="Filter by valid, possible, or uncertain"),
):
    """Returns the list of recently recognized vehicle plates from the traffic stream with filtering."""
    status_data = cctv_manager.get_status()
    all_events = [e.to_dict() for e in cctv_manager.detected_events]

    # Filter by state code if provided
    if state_code:
        code_upper = state_code.upper()
        all_events = [
            e for e in all_events
            if e["plate_number"].startswith(code_upper) or code_upper in e["formatted_number"]
        ]

    # Filter by validation status if provided
    if status_filter:
        all_events = [e for e in all_events if e["format_status"] == status_filter]

    return {
        "is_running": status_data["is_running"],
        "status_state": status_data["status_state"],
        "total_events": len(all_events),
        "events": all_events[:limit],
    }
