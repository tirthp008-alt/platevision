"""Visual vehicle embedding / Re-ID module.

Plate matching alone is insufficient, so each tracked vehicle yields a visual
appearance embedding. The module has two interchangeable backends:

  * ``classical``  a deterministic HOG-shape + HSV-colour descriptor. Not a
                   learned Re-ID net, but it captures body shape and colour in
                   a way that is far more robust than raw RGB averaging.
  * ``onnx``       a proper vehicle Re-ID / image-embedding ONNX model when one
                   is supplied at ``REID_MODEL_PATH`` (input NCHW RGB 0-1).

Both return L2-normalised float vectors so the fusion engine can use cosine
similarity ``cos(v_i, v_j) = (v_i·v_j)/(||v_i|| ||v_j||)`` directly.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embedding vectors. Returns 0.0 for empty input."""
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.size != vb.size:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))


def dominant_color_name(bgr_crop: np.ndarray) -> str:
    """Return a coarse colour label for display, ignoring road/background pixels."""
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown"
    small = cv2.resize(bgr_crop, (32, 32))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (s > 40) & (v > 40)
    if mask.sum() < 20:
        # Low-saturation: greyscale vehicle
        mean_v = float(v.mean())
        if mean_v < 55:
            return "black"
        if mean_v > 190:
            return "white"
        return "grey"
    mean_h = float(h[mask].mean())
    if mean_h < 10 or mean_h >= 170:
        return "red"
    if mean_h < 22:
        return "orange"
    if mean_h < 33:
        return "yellow"
    if mean_h < 78:
        return "green"
    if mean_h < 130:
        return "blue"
    return "purple"


def _classical_embedding(bgr_crop: np.ndarray) -> List[float]:
    """Shape + colour descriptor, L2-normalised.

    A central crop is used to reduce background influence, and the colour
    histogram is computed only over saturated pixels (the vehicle body) so that
    grey road/background pixels do not dominate the descriptor.
    """
    h, w = bgr_crop.shape[:2]
    # Central 70% crop keeps the vehicle body and discards the surrounding scene.
    cy0, cy1 = int(h * 0.15), int(h * 0.85)
    cx0, cx1 = int(w * 0.15), int(w * 0.85)
    core = bgr_crop[cy0:cy1, cx0:cx1]
    if core.size == 0:
        core = bgr_crop

    img = cv2.resize(core, (48, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # HOG-like: per 8x8 cell gradient orientation histogram (9 bins)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    cell, bins = 8, 9
    hog: List[float] = []
    for cy in range(0, 64, cell):
        for cx in range(0, 48, cell):
            m = mag[cy : cy + cell, cx : cx + cell].ravel()
            a = (ang[cy : cy + cell, cx : cx + cell] % 180.0).ravel()
            hist = np.zeros(bins, dtype=np.float32)
            idx = np.clip((a / 20.0).astype(np.int32), 0, bins - 1)
            np.add.at(hist, idx, m)
            norm = float(np.linalg.norm(hist)) + 1e-6
            hog.extend((hist / norm).tolist())

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mask = (hsv[..., 1] > 45) & (hsv[..., 2] > 40)
    if int(sat_mask.sum()) < 30:
        sat_mask = None
    hue = cv2.calcHist([hsv], [0], sat_mask, [30], [0, 180]).flatten()
    sat = cv2.calcHist([hsv], [1], sat_mask, [8], [0, 256]).flatten()
    val = cv2.calcHist([hsv], [2], sat_mask, [8], [0, 256]).flatten()
    hue = hue / (float(hue.sum()) + 1e-6)
    sat = sat / (float(sat.sum()) + 1e-6)
    val = val / (float(val.sum()) + 1e-6)

    # Colour carries more identity than coarse shape here, so it is up-weighted.
    shape = np.asarray(hog, dtype=np.float32)
    shape = shape / (float(np.linalg.norm(shape)) + 1e-6)
    colour = np.concatenate([hue * 2.0, sat, val])
    colour = colour / (float(np.linalg.norm(colour)) + 1e-6)

    vec = np.concatenate([shape, colour]).astype(np.float32)
    norm = float(np.linalg.norm(vec)) + 1e-6
    return (vec / norm).tolist()


class _OnnxReID:
    def __init__(self, path: str):
        import onnxruntime as ort  # type: ignore

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.size = (256, 128)
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.size = (shape[2], shape[3])

    def embed(self, bgr_crop: np.ndarray) -> List[float]:
        h, w = self.size
        img = cv2.resize(bgr_crop, (w, h))
        blob = img[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        out = self.session.run(None, {self.input_name: blob})[0].ravel()
        norm = float(np.linalg.norm(out)) + 1e-6
        return (out / norm).tolist()


class VehicleEmbedder:
    """Facade selecting an embedding backend, with graceful fallback."""

    def __init__(self, backend: Optional[str] = None, model_path: Optional[str] = None):
        self.backend = (backend or settings.REID_BACKEND).lower()
        self.model_path = model_path or settings.REID_MODEL_PATH
        self._onnx: Optional[_OnnxReID] = None
        self.active = "classical"
        if self.backend in ("auto", "onnx") and os.path.exists(self.model_path):
            try:
                self._onnx = _OnnxReID(self.model_path)
                self.active = "onnx"
                logger.info(f"Vehicle Re-ID using ONNX model {self.model_path}")
            except Exception as e:
                logger.warning(f"Re-ID ONNX model failed to load ({e}); using classical descriptor.")
        elif self.backend == "onnx":
            logger.warning("REID_BACKEND=onnx but no model path found; using classical descriptor.")

    @property
    def dim(self) -> int:
        return settings.REID_EMBEDDING_DIM

    def embed(self, bgr_crop: np.ndarray) -> List[float]:
        if bgr_crop is None or bgr_crop.size == 0 or bgr_crop.shape[0] < 8 or bgr_crop.shape[1] < 8:
            return []
        try:
            if self._onnx is not None:
                return self._onnx.embed(bgr_crop)
            return _classical_embedding(bgr_crop)
        except Exception as e:
            logger.debug(f"Embedding failed: {e}")
            return []

    def embed_with_color(self, bgr_crop: np.ndarray) -> Tuple[List[float], str]:
        return self.embed(bgr_crop), dominant_color_name(bgr_crop)


vehicle_embedder = VehicleEmbedder()
