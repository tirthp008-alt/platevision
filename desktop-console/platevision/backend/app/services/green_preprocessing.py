"""Optional OCR alternative for green plates with faint, unpainted characters.

This does not detect plates or reconstruct missing characters. Keep the original
crop and its reading: edge enhancement can also amplify screws and background.
The colour gate is a heuristic, not a calibrated plate classification score.
"""
from __future__ import annotations

import cv2
import numpy as np
import time

from app.services.normalization import normalize_plate


_EMBOSS = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)
_GRADIENT_ELEMENT = np.ones((3, 3), np.uint8)


def _valid_crop(crop: np.ndarray) -> bool:
    return (isinstance(crop, np.ndarray) and crop.dtype == np.uint8
            and crop.ndim == 3 and crop.shape[2] == 3
            and min(crop.shape[:2]) >= 8)


def green_plate_features(crop: np.ndarray) -> dict[str, float]:
    """Measure green coverage and visible white paint on a small BGR probe.

    Border pixels are excluded to reduce the influence of white plate frames.
    A dark green plate can qualify even when white paint is dim under shadow.
    """
    if not _valid_crop(crop):
        return {"green_fraction": 0.0, "white_fraction": 0.0}
    h, w = crop.shape[:2]
    margin_x, margin_y = max(1, round(w * .05)), max(1, round(h * .08))
    inner = crop[margin_y:h-margin_y, margin_x:w-margin_x]
    scale = min(1., 160 / inner.shape[1], 64 / inner.shape[0])
    if scale < 1:
        size = (max(1, round(inner.shape[1]*scale)), max(1, round(inner.shape[0]*scale)))
        inner = cv2.resize(inner, size,
                           interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 50, 25), (100, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 160), (179, 65, 255))
    count = hsv.shape[0] * hsv.shape[1]
    return {"green_fraction": cv2.countNonZero(green) / count,
            "white_fraction": cv2.countNonZero(white) / count}


def needs_green_emboss(crop: np.ndarray) -> bool:
    """Gate an uncertain OCR retry; call only after checking original OCR."""
    features = green_plate_features(crop)
    return features["green_fraction"] >= .25 and features["white_fraction"] < .04


def emboss_morphological_gradient(crop: np.ndarray) -> np.ndarray:
    """Apply an emboss kernel, then dilation-minus-erosion, to a plate crop.

    Float filtering preserves signed responses until the gradient is computed.
    A percentile stretch removes lighting offsets without clipping the emboss
    response first. Invert to dark strokes on a light field for an OCR retry.
    Processing is bounded at 640x160 pixels and never changes the input.
    """
    if not _valid_crop(crop):
        raise ValueError("Expected a nonempty uint8 BGR plate crop of at least 8x8 pixels")
    h, w = crop.shape[:2]
    scale = min(3., 640 / w, 160 / h, max(1., 80 / h))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if scale != 1:
        size = (max(1, round(w*scale)), max(1, round(h*scale)))
        gray = cv2.resize(gray, size,
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    embossed = cv2.filter2D(gray, cv2.CV_32F, _EMBOSS,
                           borderType=cv2.BORDER_REPLICATE)
    gradient = cv2.morphologyEx(embossed, cv2.MORPH_GRADIENT, _GRADIENT_ELEMENT,
                               borderType=cv2.BORDER_REPLICATE)
    low, high = np.percentile(gradient, (2, 98))
    if high - low < 1:
        # A blank plate contains no recoverable lettering. Return a blank image.
        result = np.full(gray.shape, 255, np.uint8)
    else:
        result = 255 - np.clip((gradient - low) * (255 / (high - low)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def green_emboss_candidate(crop: np.ndarray) -> np.ndarray | None:
    """Return one bounded alternative only when colour/paint heuristics match."""
    return emboss_morphological_gradient(crop) if needs_green_emboss(crop) else None


def prepare_green_emboss(crop: np.ndarray, original_reading: tuple[str, float]):
    """Prepare an explicitly requested experimental alternative and metadata.

    Caller keeps the original text/confidence and measures candidate OCR time
    separately. A confident original does not block a user-requested comparison:
    this helper never chooses, merges, or changes readings. Do not call it on the
    default path; the optional request should control all extra work.
    """
    started = time.perf_counter()
    features = green_plate_features(crop)
    if not _valid_crop(crop):
        reason = 'invalid_crop'
    elif features['green_fraction'] < .25:
        reason = 'insufficient_green_background'
    elif features['white_fraction'] >= .04:
        reason = 'white_lettering_visible'
    else:
        reason = 'faint_or_unpainted_green_candidate'
    applied = reason == 'faint_or_unpainted_green_candidate'
    enhanced = emboss_morphological_gradient(crop) if applied else None
    text, confidence = original_reading
    metadata = dict(method='emboss_morphological_gradient', experimental=True,
                    applied=applied, reason=reason, requires_review=applied,
                    original_preserved=True, original_text=str(text),
                    original_ocr_confidence=float(confidence),
                    original_uncertain=(confidence < .88 or normalize_plate(text)[2] != 'valid'),
                    features=features,
                    preprocessing_ms=round((time.perf_counter()-started)*1000, 4),
                    confidence_note='OCR scores are uncalibrated; enhanced text requires manual review.')
    return enhanced, metadata
