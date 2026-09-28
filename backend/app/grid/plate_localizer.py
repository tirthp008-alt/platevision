"""Heuristic number-plate localization.

When a trained plate detector is unavailable (no ONNX weights present) this
module still localises plates in a genuinely data-driven way: it segments
bright, low-saturation, high-contrast rectangular regions (white HSRP and
black-on-yellow commercial plates) and returns their bounding boxes. It is a
real computer-vision step, not a hard-coded answer, and its output is fed
through the same PlateVision OCR path as the ONNX detector.

It is used as a *supplementary* plate source alongside the configured plate
detector, which improves recall on high-contrast plates without weakening the
pipeline when a proper detector is present.
"""

from typing import List, Tuple

import cv2
import numpy as np

from app.schemas.detection import BoundingBox


def _refine_character_contrast(gray: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    """Return the dark-character-on-light-plate contrast in [0,1]."""
    x, y, w, h = box
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    # A plate is mostly light with dark glyphs; measure the spread.
    thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    dark_ratio = float(np.count_nonzero(thr == 0)) / float(roi.size)
    std = float(roi.std())
    if 0.05 <= dark_ratio <= 0.6:
        return float(min(1.0, std / 90.0))
    return 0.0


def find_plate_regions(
    image_bgr: np.ndarray,
    expect_vehicle_bbox: BoundingBox = None,
    max_results: int = 3,
) -> List[Tuple[BoundingBox, float]]:
    """Locate candidate plate regions, optionally restricted to a vehicle box."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    h, w = image_bgr.shape[:2]
    search = image_bgr
    ox, oy = 0, 0
    if expect_vehicle_bbox is not None:
        x = max(0, expect_vehicle_bbox.x)
        y = max(0, expect_vehicle_bbox.y)
        x2 = min(w, expect_vehicle_bbox.x + expect_vehicle_bbox.width)
        y2 = min(h, expect_vehicle_bbox.y + expect_vehicle_bbox.height)
        if x2 - x < 20 or y2 - y < 20:
            return []
        search = image_bgr[y:y2, x:x2]
        ox, oy = x, y

    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)

    results: List[Tuple[BoundingBox, float]] = []
    frame_area = float(search.shape[0] * search.shape[1])

    # White plates: bright, low saturation.
    white = cv2.inRange(hsv, (0, 0, 150), (180, 70, 255))
    # Yellow plates: hue ~20-35, high saturation/value.
    yellow = cv2.inRange(hsv, (15, 90, 120), (38, 255, 255))

    for mask in (white, yellow):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bh < 10 or bw < 28:
                continue
            aspect = bw / float(bh)
            rel = (bw * bh) / frame_area
            if not (1.8 <= aspect <= 6.5 and 0.001 <= rel <= 0.5):
                continue
            # Require a real glyph pattern inside (dark characters).
            if _refine_character_contrast(gray, (x, y, bw, bh)) < 0.15:
                continue
            conf = float(min(0.9, 0.5 + 0.4 * min(1.0, aspect / 4.0)))
            results.append(
                (
                    BoundingBox(x=x + ox, y=y + oy, width=bw, height=bh),
                    conf,
                )
            )

    if not results:
        return []

    # Deduplicate overlapping boxes, keep highest confidence / larger area.
    results.sort(key=lambda r: (r[1], r[0].width * r[0].height), reverse=True)
    kept: List[Tuple[BoundingBox, float]] = []
    for box, conf in results:
        overlap = False
        for kb, _ in kept:
            x1 = max(box.x, kb.x)
            y1 = max(box.y, kb.y)
            x2 = min(box.x + box.width, kb.x + kb.width)
            y2 = min(box.y + box.height, kb.y + kb.height)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = box.width * box.height + kb.width * kb.height - inter
            if union > 0 and inter / union > 0.4:
                overlap = True
                break
        if not overlap:
            kept.append((box, conf))
    return kept[:max_results]
