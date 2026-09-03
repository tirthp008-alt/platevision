"""Main FastAPI application entrypoint for PlateVision."""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import logger
from app.core.security import SecurityHeadersMiddleware
from app.services.detector.factory import get_detector
from app.services.ocr.engine import ocr_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm detector and OCR models on application startup."""
    logger.info("Initializing PlateVision backend services...")
    try:
        det = get_detector()
        logger.info(f"Detector ready: {det.is_ready()} ({type(det).__name__})")
        logger.info(f"OCR Engine backend: {ocr_engine._engine_type}")
    except Exception as e:
        logger.error(f"Error during startup model loading: {e}")
    yield
    logger.info("Shutting down PlateVision backend...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# Security hardening headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


# Custom error formatting according to spec
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "details": None,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request parameters or payload.",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error: {exc}", exc_info=settings.DEBUG)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during image processing.",
                "details": str(exc) if settings.DEBUG else None,
            }
        },
    )


@app.get("/")
async def root():
    return {
        "app": "PlateVision API",
        "version": settings.VERSION,
        "docs": "/api/docs",
        "health": "/api/health",
    }
