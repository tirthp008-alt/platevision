"""Plate detection and recognition endpoints for uploaded photos and live camera frames."""

import base64
import time
import uuid
from typing import List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger
from app.core.security import rate_limiter
from app.schemas.detection import (
    BoundingBox,
    DetectionItem,
    DetectionResponse,
    ErrorResponse,
    ImageMeta,
)
from app.services.crop_store import crop_store
from app.services.cropper import extract_plate_crop, to_normalized_box
from app.services.detector.factory import get_detector
from app.services.ocr.engine import ocr_engine
from app.utils.image_ops import (
    decode_image_safely,
    encode_image_to_base64,
    validate_image_bytes,
)

router = APIRouter()


class FrameBase64Payload(BaseModel):
    frame_base64: str


def process_image_pipeline(
    image_bgr: np.ndarray, request_id: str
) -> DetectionResponse:
    """Executes the full pipeline: Detection -> Crop -> Preprocess -> OCR -> Normalization."""
    start_time = time.time()
    h, w = image_bgr.shape[:2]

    detector = get_detector()
    try:
        raw_detections = detector.detect(image_bgr)
    except Exception as e:
        logger.error(f"Detector error for request {request_id}: {e}")
        raw_detections = []

    detection_items: List[DetectionItem] = []
    warnings: List[str] = []

    if not raw_detections:
        warnings.append("No number plate was detected in this image.")

    for raw_det in raw_detections:
        det_id = str(uuid.uuid4())
        # Crop with padding and clamping
        crop_bgr, clamped_bbox = extract_plate_crop(
            image_bgr,
            raw_det.bbox,
            apply_padding=True,
            padding_ratio=settings.BBOX_PADDING_RATIO,
        )

        if crop_bgr.size == 0 or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 15:
            continue

        # Run OCR with multi-stage preprocessing variants
        ocr_res = ocr_engine.recognize(crop_bgr)

        # Verification: If no text was recognized and raw detection confidence is very low, skip false positive
        if not ocr_res.normalized_text and raw_det.confidence < 0.25:
            continue

        # Deduplication check against already confirmed detections
        is_duplicate = False
        for existing in detection_items:
            # Same normalized text or high bounding box overlap
            if (
                ocr_res.normalized_text
                and ocr_res.normalized_text == existing.normalized_text
            ):
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Store in transient cache
        crop_store.store_crop(request_id, det_id, crop_bgr, format_type="jpeg")

        # Encode crop and preprocessed variant as base64 for immediate UI rendering
        crop_b64 = encode_image_to_base64(crop_bgr, format_type="jpeg", quality=92)
        preprocessed_b64 = encode_image_to_base64(
            ocr_res.best_variant_image, format_type="jpeg", quality=88
        )

        # Normalized coordinates
        norm_box = to_normalized_box(clamped_bbox, w, h)

        crop_url = f"/api/results/{request_id}/{det_id}/crop"

        detection_items.append(
            DetectionItem(
                id=det_id,
                bounding_box=clamped_bbox,
                normalized_box=norm_box,
                detection_confidence=round(raw_det.confidence, 2),
                raw_text=ocr_res.raw_text,
                normalized_text=ocr_res.normalized_text,
                formatted_text=ocr_res.formatted_text,
                ocr_confidence=round(ocr_res.confidence, 2),
                format_status=ocr_res.format_status,
                crop_url=crop_url,
                crop_base64=crop_b64,
                preprocessed_base64=preprocessed_b64,
            )
        )

    duration_ms = int((time.time() - start_time) * 1000)

    # Privacy-conscious logging
    logger.info(
        f"Processed request {request_id}: {len(detection_items)} plates found in {duration_ms}ms"
    )

    return DetectionResponse(
        request_id=request_id,
        processing_time_ms=duration_ms,
        image=ImageMeta(width=w, height=h),
        detections=detection_items,
        warnings=warnings,
    )


@router.post(
    "/detect/image",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image format or size"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_image(request: Request, image: UploadFile = File(...)):
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests. Please slow down.",
                "details": None,
            },
        )

    content = await image.read()
    valid, err_msg = validate_image_bytes(content, max_size_mb=settings.MAX_IMAGE_SIZE_MB)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_IMAGE", "message": err_msg, "details": None},
        )

    img_bgr = decode_image_safely(content)
    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "DECODE_ERROR",
                "message": "Failed to decode image data.",
                "details": None,
            },
        )

    req_id = str(uuid.uuid4())
    return process_image_pipeline(img_bgr, req_id)


@router.post(
    "/detect/video",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid video format or size"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_video(request: Request, video: UploadFile = File(...)):
    """Extracts keyframes from an uploaded video file and detects all visible number plates."""
    import tempfile
    import os

    start_time = time.time()
    req_id = str(uuid.uuid4())
    content = await video.read()

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_VIDEO", "message": "Uploaded video file is empty.", "details": None},
        )

    suffix = os.path.splitext(video.filename or "temp.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    all_detections_map = {}
    last_frame_meta = ImageMeta(width=1280, height=720)

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "DECODE_ERROR", "message": "Could not open video file.", "details": None},
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(fps * 0.75))
        frame_idx = 0
        samples_processed = 0

        while cap.isOpened() and samples_processed < 30:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_interval == 0:
                samples_processed += 1
                fh, fw = frame.shape[:2]
                last_frame_meta = ImageMeta(width=fw, height=fh)
                try:
                    f_resp = process_image_pipeline(frame, f"{req_id}_{samples_processed}")
                    for det in f_resp.detections:
                        key = det.normalized_text or det.id
                        if key not in all_detections_map or det.ocr_confidence > all_detections_map[key].ocr_confidence:
                            all_detections_map[key] = det
                except Exception:
                    pass

            frame_idx += 1

        cap.release()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    total_time_ms = int((time.time() - start_time) * 1000)
    detections_list = list(all_detections_map.values())

    return DetectionResponse(
        request_id=req_id,
        processing_time_ms=total_time_ms,
        image=last_frame_meta,
        detections=detections_list,
        warnings=[],
    )


@router.post(
    "/detect/frame",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid frame data"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_frame(
    request: Request,
):
    """Low-latency detection endpoint for live phone camera frames."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests. Please throttle frame rate.",
                "details": None,
            },
        )

    req_id = str(uuid.uuid4())
    content_type = request.headers.get("content-type", "")

    img_bgr = None
    if "application/json" in content_type:
        try:
            body = await request.json()
            raw_b64 = body.get("frame_base64", "")
            if not raw_b64:
                raise ValueError("Empty frame_base64")
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            img_bgr = decode_image_safely(img_bytes)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_BASE64",
                    "message": "Malformed base64 frame data.",
                    "details": None,
                },
            )
    else:
        # Try parsing as form/multipart
        try:
            form = await request.form()
            image_file = form.get("image")
            if image_file and hasattr(image_file, "read"):
                content = await image_file.read()
                valid, err_msg = validate_image_bytes(
                    content, max_size_mb=settings.MAX_IMAGE_SIZE_MB
                )
                if not valid:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "INVALID_IMAGE",
                            "message": err_msg,
                            "details": None,
                        },
                    )
                img_bgr = decode_image_safely(content)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error parsing frame request: {e}")

    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MISSING_OR_CORRUPT_PAYLOAD",
                "message": "Must provide a valid JSON body with 'frame_base64' or a valid multipart image file.",
                "details": None,
            },
        )

    return process_image_pipeline(img_bgr, req_id)
