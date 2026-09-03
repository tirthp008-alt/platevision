"""Main API router combining all endpoint modules."""

from fastapi import APIRouter
from app.api.endpoints import cctv, detect, health, results

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(detect.router, tags=["Detection"])
api_router.include_router(results.router, tags=["Results"])
api_router.include_router(cctv.router, tags=["CCTV Surveillance"])
