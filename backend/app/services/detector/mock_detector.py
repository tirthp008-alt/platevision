"""Mock detector implementation for development, testing, and offline fallback."""

from typing import List
import cv2
import numpy as np

from app.core.logging import logger
from app.schemas.detection import BoundingBox
from app.services.detector.base import BasePlateDetector, RawDetection


class MockPlateDetector(BasePlateDetector):
    """Mock detector using morphological edge analysis and heuristic vehicle localization."""

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        logger.info("Initialized MockPlateDetector (Development / Offline mode).")

    def is_ready(self) -> bool:
        return True

    def detect(self, image_bgr: np.ndarray) -> List[RawDetection]:
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return []

        # Try to locate plate regions using edge gradients
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Morphological blackhat
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        # Sobel X gradient
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        if max_val > min_val:
            grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")
        else:
            grad_x = np.zeros_like(grad_x, dtype="uint8")

        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Close gaps
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        )
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[RawDetection] = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh) if bh > 0 else 0
            rel_area = (bw * bh) / float(w * h)

            # Indian standard plates aspect ratio is ~ 2.2 to 5.5, area between 0.5% and 35% of frame
            if 2.0 <= aspect_ratio <= 6.0 and 0.005 <= rel_area <= 0.35 and bw > 40 and bh > 15:
                # Plausible plate contour
                conf = round(min(0.96, 0.82 + (aspect_ratio / 20.0)), 2)
                detections.append(
                    RawDetection(
                        bbox=BoundingBox(x=bx, y=by, width=bw, height=bh),
                        confidence=conf,
                    )
                )

        if detections:
            # Sort by area/confidence descending, limit to top 3
            detections.sort(key=lambda d: d.bbox.width * d.bbox.height, reverse=True)
            return detections[:3]

        # Default realistic fallback for test images where edges are smooth or mock is forced
        # Place standard plate box in lower-center third (typical vehicle bumper position)
        box_w = int(w * 0.32)
        box_h = int(box_w * 0.28)
        box_x = int((w - box_w) / 2)
        box_y = int(h * 0.62)

        fallback_bbox = BoundingBox(x=box_x, y=box_y, width=box_w, height=box_h)
        return [RawDetection(bbox=fallback_bbox, confidence=0.92)]
