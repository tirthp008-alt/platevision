"""Bounding box adjustment, coordinate clamping, and high-resolution plate cropping service."""

from typing import Dict, Tuple
import numpy as np
from app.core.config import settings
from app.schemas.detection import BoundingBox, NormalizedBox


def clamp_coordinate(val: int, min_val: int, max_val: int) -> int:
    """Clamps an integer coordinate within [min_val, max_val]."""
    return max(min_val, min(val, max_val))


def expand_and_clamp_bbox(
    bbox: BoundingBox,
    img_width: int,
    img_height: int,
    padding_ratio: float = None,
) -> BoundingBox:
    """Expands bounding box by padding_ratio and clamps to image dimensions."""
    if padding_ratio is None:
        padding_ratio = settings.BBOX_PADDING_RATIO

    pad_x = int(bbox.width * padding_ratio)
    pad_y = int(bbox.height * padding_ratio)

    x1 = clamp_coordinate(bbox.x - pad_x, 0, img_width - 1)
    y1 = clamp_coordinate(bbox.y - pad_y, 0, img_height - 1)
    x2 = clamp_coordinate(bbox.x + bbox.width + pad_x, 1, img_width)
    y2 = clamp_coordinate(bbox.y + bbox.height + pad_y, 1, img_height)

    # Ensure width and height are strictly positive
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    return BoundingBox(x=x1, y=y1, width=width, height=height)


def to_normalized_box(bbox: BoundingBox, img_width: int, img_height: int) -> NormalizedBox:
    """Converts pixel bounding box to normalized coordinates (0.0 to 1.0)."""
    if img_width <= 0 or img_height <= 0:
        return NormalizedBox(x=0.0, y=0.0, width=0.0, height=0.0)

    nx = max(0.0, min(1.0, float(bbox.x) / img_width))
    ny = max(0.0, min(1.0, float(bbox.y) / img_height))
    nw = max(0.0, min(1.0, float(bbox.width) / img_width))
    nh = max(0.0, min(1.0, float(bbox.height) / img_height))

    return NormalizedBox(
        x=round(nx, 4),
        y=round(ny, 4),
        width=round(nw, 4),
        height=round(nh, 4),
    )


def extract_plate_crop(
    image_np: np.ndarray,
    bbox: BoundingBox,
    apply_padding: bool = True,
    padding_ratio: float = None,
) -> Tuple[np.ndarray, BoundingBox]:
    """Extracts a cropped plate sub-image from the full-resolution image array."""
    h, w = image_np.shape[:2]

    if apply_padding:
        adjusted_box = expand_and_clamp_bbox(bbox, w, h, padding_ratio)
    else:
        x1 = clamp_coordinate(bbox.x, 0, w - 1)
        y1 = clamp_coordinate(bbox.y, 0, h - 1)
        x2 = clamp_coordinate(bbox.x + bbox.width, 1, w)
        y2 = clamp_coordinate(bbox.y + bbox.height, 1, h)
        adjusted_box = BoundingBox(x=x1, y=y1, width=max(1, x2 - x1), height=max(1, y2 - y1))

    crop = image_np[
        adjusted_box.y : adjusted_box.y + adjusted_box.height,
        adjusted_box.x : adjusted_box.x + adjusted_box.width,
    ]

    # Return safe copy
    return crop.copy(), adjusted_box
