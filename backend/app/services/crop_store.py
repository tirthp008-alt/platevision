"""Transient in-memory crop cache with automatic TTL cleanup."""

import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from app.core.config import settings
from app.core.logging import logger


class TransientCropStore:
    """Thread-safe transient in-memory cache for cropped plate images."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        # Key: (request_id, detection_id) -> (image_bytes, timestamp, mime_type)
        self._cache: Dict[Tuple[str, str], Tuple[bytes, float, str]] = {}

    def _cleanup_expired(self):
        now = time.time()
        expired_keys = [
            k for k, (_, ts, _) in self._cache.items() if now - ts > self.ttl_seconds
        ]
        for k in expired_keys:
            del self._cache[k]

    def store_crop(
        self, request_id: str, detection_id: str, crop_bgr: np.ndarray, format_type: str = "jpeg"
    ) -> None:
        self._cleanup_expired()
        ext = f".{format_type}"
        success, buffer = cv2.imencode(ext, crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success:
            mime = "image/jpeg" if format_type in ["jpg", "jpeg"] else "image/png"
            self._cache[(request_id, detection_id)] = (buffer.tobytes(), time.time(), mime)

    def get_crop(self, request_id: str, detection_id: str) -> Optional[Tuple[bytes, str]]:
        self._cleanup_expired()
        item = self._cache.get((request_id, detection_id))
        if item:
            return item[0], item[2]
        return None

    def clear(self):
        self._cache.clear()


crop_store = TransientCropStore(ttl_seconds=settings.TEMP_CROP_TTL_SECONDS)
