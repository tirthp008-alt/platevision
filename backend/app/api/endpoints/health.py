"""Health check endpoint."""

from fastapi import APIRouter
from app.core.config import settings
from app.schemas.detection import HealthResponse
from app.services.detector.factory import get_detector
from app.services.ocr.engine import ocr_engine

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    detector = get_detector()
    return HealthResponse(
        status="ok",
        detector_ready=detector.is_ready(),
        ocr_ready=True,
        version=settings.VERSION,
        backend_type=type(detector).__name__,
    )
