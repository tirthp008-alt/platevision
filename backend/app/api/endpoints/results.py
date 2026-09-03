"""Crop image retrieval endpoint."""

from fastapi import APIRouter, HTTPException, Response, status
from app.schemas.detection import ErrorResponse
from app.services.crop_store import crop_store

router = APIRouter()


@router.get(
    "/results/{request_id}/{detection_id}/crop",
    responses={
        200: {
            "content": {"image/jpeg": {}, "image/png": {}},
            "description": "Cropped plate image binary.",
        },
        404: {"model": ErrorResponse, "description": "Crop not found or expired."},
    },
)
async def get_detection_crop(request_id: str, detection_id: str):
    crop_data = crop_store.get_crop(request_id, detection_id)
    if not crop_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "CROP_NOT_FOUND",
                "message": "The requested plate crop was not found or has expired from temporary memory.",
                "details": None,
            },
        )

    image_bytes, mime_type = crop_data
    return Response(
        content=image_bytes,
        media_type=mime_type,
        headers={
            "Cache-Control": "private, max-age=300",
            "Content-Disposition": f'inline; filename="plate_{detection_id[:8]}.jpg"',
        },
    )
