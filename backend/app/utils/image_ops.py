"""Image operations, decoding, orientation fix, and validation utilities."""

import base64
import io
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

from app.core.config import settings
from app.core.logging import logger


def validate_image_bytes(
    data: bytes, max_size_mb: int = 10
) -> Tuple[bool, Optional[str]]:
    """Validate image size and basic magic bytes signature."""
    if not data:
        return False, "Image payload is empty."

    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"Image size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_size_mb} MB."

    # Validate magic bytes for JPEG, PNG, WEBP, HEIC
    if len(data) < 8:
        return False, "Corrupted image data header."

    is_jpeg = data[:2] == b"\xff\xd8"
    is_png = data[:8] == b"\x89PNG\r\n\x1a\n"
    is_webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    is_heic = len(data) >= 12 and (data[4:12] == b"ftypheic" or data[4:12] == b"ftypmif1" or data[4:12] == b"ftypheix")

    if not (is_jpeg or is_png or is_webp or is_heic):
        # We can still try PIL verification as fallback
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except Exception:
            return False, "Unsupported or corrupted image format. Please upload JPEG, PNG, WEBP, or HEIC."

    return True, None


def decode_image_safely(data: bytes) -> Optional[np.ndarray]:
    """Safely decode image bytes into a BGR OpenCV numpy array with EXIF orientation correction."""
    try:
        pil_image = Image.open(io.BytesIO(data))
        # Correct orientation based on EXIF tag
        pil_image = ImageOps.exif_transpose(pil_image)

        # Convert to RGB (in case of RGBA, Palette, or Grayscale)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert RGB PIL to BGR OpenCV ndarray
        rgb_arr = np.array(pil_image)
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return bgr_arr
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        # Fallback to OpenCV direct decode
        try:
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e2:
            logger.error(f"Fallback cv2 imdecode failed: {e2}")
            return None


def encode_image_to_base64(
    image_np: np.ndarray, format_type: str = "jpeg", quality: int = 90
) -> str:
    """Encode OpenCV BGR or Grayscale image into a data URI base64 string."""
    try:
        ext = f".{format_type}"
        params = []
        if format_type.lower() in ["jpg", "jpeg"]:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif format_type.lower() == "png":
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
        elif format_type.lower() == "webp":
            params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]

        success, buffer = cv2.imencode(ext, image_np, params)
        if not success:
            return ""
        b64_str = base64.b64encode(buffer).decode("utf-8")
        mime = f"image/{'jpeg' if format_type.lower() in ['jpg', 'jpeg'] else format_type.lower()}"
        return f"data:{mime};base64,{b64_str}"
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return ""


def resize_maintaining_aspect_ratio(
    image_np: np.ndarray, max_dim: int = 1280
) -> Tuple[np.ndarray, float]:
    """Resize image so its longest side is at most max_dim, returning (resized_image, scale_factor)."""
    h, w = image_np.shape[:2]
    if max(h, w) <= max_dim:
        return image_np, 1.0

    if w > h:
        scale = max_dim / float(w)
        new_w = max_dim
        new_h = int(h * scale)
    else:
        scale = max_dim / float(h)
        new_h = max_dim
        new_w = int(w * scale)

    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale
