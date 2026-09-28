"""Transient store for the most recent annotated frames and evidence crops.

Only in-memory, TTL-bound frames are kept so raw video is never persisted to
disk and never transmitted as a continuous stream. The dashboard polls a
single JPEG snapshot per camera for the live view.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings


class FrameStore:
    def __init__(self, ttl: int = None, max_frames: int = None):
        self.ttl = ttl or settings.FRAME_STORE_TTL_SECONDS
        self.max_frames = max_frames or settings.FRAME_STORE_MAX_FRAMES
        self._lock = threading.Lock()
        self._frames: Dict[str, Tuple[bytes, float]] = {}
        self._crops: Dict[str, Tuple[bytes, float]] = {}

    def _prune(self) -> None:
        now = time.time()
        for store in (self._frames, self._crops):
            expired = [k for k, (_, ts) in store.items() if now - ts > self.ttl]
            for k in expired:
                del store[k]
        if len(self._crops) > self.max_frames:
            ordered = sorted(self._crops.items(), key=lambda kv: kv[1][1])
            for k, _ in ordered[: len(self._crops) - self.max_frames]:
                del self._crops[k]

    def put_frame(self, camera_id: str, frame_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), settings.GRID_FRAME_JPEG_QUALITY]
        )
        if not ok:
            return
        with self._lock:
            self._frames[camera_id] = (buf.tobytes(), time.time())
            self._prune()

    def get_frame(self, camera_id: str) -> Optional[bytes]:
        with self._lock:
            self._prune()
            item = self._frames.get(camera_id)
            return item[0] if item else None

    def put_crop(self, key: str, crop_bgr: np.ndarray) -> None:
        if crop_bgr is None or crop_bgr.size == 0:
            return
        ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        with self._lock:
            self._crops[key] = (buf.tobytes(), time.time())
            self._prune()

    def get_crop(self, key: str) -> Optional[bytes]:
        with self._lock:
            self._prune()
            item = self._crops.get(key)
            return item[0] if item else None

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._crops.clear()


frame_store = FrameStore()
