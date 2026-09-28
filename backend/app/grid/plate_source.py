"""Composite plate source for the grid pipeline.

The stock PlateVision detector factory may return a ``MockPlateDetector`` when
no trained ONNX weights are present. The mock is a heuristic that only draws a
fixed fallback box; using it as a *plate* source would inject junk OCR into
sightings. For the city-wide pipeline we therefore combine:

  * a trained ONNX plate detector when one is actually available, and
  * the contrast-based :func:`find_plate_regions` localizer, which works on any
    real frame without trained weights.

This keeps the plate evidence genuinely data-driven and honest about the absence
of trained weights, and it is the single plate source used for both plate OCR
and vehicle anchoring.
"""

from typing import List, Tuple

import numpy as np

from app.core.config import settings
from app.schemas.detection import BoundingBox
from app.services.detector.base import BasePlateDetector, RawDetection
from app.services.detector.factory import get_detector
from app.grid.plate_localizer import find_plate_regions


def _is_trained_detector(detector: BasePlateDetector) -> bool:
    name = type(detector).__name__.lower()
    if "mock" in name:
        return False
    try:
        return bool(detector.is_ready())
    except Exception:
        return False


class CompositePlateDetector(BasePlateDetector):
    """Merges a trained plate detector (if present) with the contrast localizer."""

    def __init__(self, detector: BasePlateDetector = None):
        raw = detector if detector is not None else get_detector()
        self._trained = raw if _is_trained_detector(raw) else None
        self.backend_name = type(self._trained).__name__ if self._trained else "contrast_localizer"

    def is_ready(self) -> bool:
        return True

    def detect(self, image_bgr: np.ndarray) -> List[RawDetection]:
        dets: List[RawDetection] = []
        if self._trained is not None:
            try:
                dets.extend(self._trained.detect(image_bgr))
            except Exception:
                pass
        for box, conf in find_plate_regions(image_bgr):
            dets.append(RawDetection(bbox=box, confidence=conf))
        return _merge(dets)


def _merge(dets: List[RawDetection], iou_thr: float = 0.4) -> List[RawDetection]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: List[RawDetection] = []
    for d in dets:
        dup = False
        for k in kept:
            x1 = max(d.bbox.x, k.bbox.x)
            y1 = max(d.bbox.y, k.bbox.y)
            x2 = min(d.bbox.x + d.bbox.width, k.bbox.x + k.bbox.width)
            y2 = min(d.bbox.y + d.bbox.height, k.bbox.y + k.bbox.height)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = d.bbox.width * d.bbox.height + k.bbox.width * k.bbox.height - inter
            if union > 0 and inter / union > iou_thr:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def build_composite_plate_detector() -> CompositePlateDetector:
    return CompositePlateDetector()
