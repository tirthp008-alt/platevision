"""Conservative, bounded plate-crop rectification; never changes scene boxes.

This is an image transform, not a detector or character reconstruction step.
Without a credible plate outline the original crop is returned unchanged.
"""

import math
import time

import cv2
import numpy as np


def _ordered_quad(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
    points = points[order]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    return points


def rectify_plate(crop):
    """Return ``(crop, metadata)`` with any quadrilateral in input-crop pixels.

    Analysis and output are bounded to 640 pixels on their longest edge. No
    mutation is made to the supplied image, including on an unsuccessful warp.
    """
    started = time.perf_counter()
    valid = isinstance(crop, np.ndarray) and crop.ndim == 3 and crop.shape[2] == 3
    h, w = crop.shape[:2] if valid else (0, 0)
    metadata = dict(applied=False, angle_degrees=0.0, reason="no_credible_outline",
                    quad=None, method="none", source_size=[w, h], output_size=[w, h])

    def finish(result, reason=None):
        if reason:
            metadata["reason"] = reason
        metadata["time_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return result, metadata

    if not valid or crop.dtype != np.uint8 or min(h, w) < 12:
        return finish(crop, "crop_too_small_or_invalid")
    scale = min(1.0, 640 / max(h, w))
    small = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else crop
    sh, sw = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, (0, 0, 115), (179, 110, 255)),
        cv2.inRange(hsv, (35, 45, 30), (95, 255, 255)),
        cv2.inRange(hsv, (15, 70, 45), (40, 255, 255)),
    ]
    candidates = []
    kernel = np.ones((3, 3), np.uint8)
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            area = cv2.contourArea(contour)
            coverage = area / (sh * sw)
            if not .20 <= coverage <= .94:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, .025 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = _ordered_quad(approx)
            # A clipped outline cannot establish a trustworthy perspective.
            if (quad[:, 0].min() < 1 or quad[:, 1].min() < 1
                    or quad[:, 0].max() > sw - 2 or quad[:, 1].max() > sh - 2):
                continue
            edge = np.roll(quad, -1, axis=0) - quad
            lengths = np.linalg.norm(edge, axis=1)
            if lengths.min() < 8 or lengths.max() / lengths.min() > 12:
                continue
            if not (.35 < lengths[0] / lengths[2] < 2.85
                    and .35 < lengths[1] / lengths[3] < 2.85):
                continue
            cosines = np.sum((-np.roll(edge, 1, axis=0)) * edge, axis=1) / (np.roll(lengths, 1) * lengths)
            if np.max(np.abs(cosines)) > .82:
                continue
            fill = area / max(1, cv2.contourArea(quad))
            if fill < .88:
                continue
            candidates.append((coverage * min(fill, 1), quad, lengths))
    if not candidates:
        return finish(crop)
    _, quad, lengths = max(candidates, key=lambda item: item[0])
    angle = math.degrees(math.atan2(quad[1, 1] - quad[0, 1], quad[1, 0] - quad[0, 0]))
    perspective = max(abs(lengths[0] - lengths[2]) / max(lengths[0], lengths[2]),
                      abs(lengths[1] - lengths[3]) / max(lengths[1], lengths[3]))
    # Preserve padding and avoid resampling when an outline is already level.
    if abs(angle) < 1.5 and perspective < .04:
        return finish(crop, "already_level")
    source_quad = quad * np.array([w / sw, h / sh], np.float32)
    source_lengths = np.linalg.norm(np.roll(source_quad, -1, axis=0) - source_quad, axis=1)
    out_w = max(source_lengths[0], source_lengths[2])
    out_h = max(source_lengths[1], source_lengths[3])
    out_scale = min(1, 640 / max(out_w, out_h))
    out_w, out_h = max(2, round(out_w * out_scale)), max(2, round(out_h * out_scale))
    destination = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    transform = cv2.getPerspectiveTransform(source_quad, destination)
    if not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-8:
        return finish(crop, "unstable_transform")
    result = cv2.warpPerspective(crop, transform, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    metadata.update(applied=True, angle_degrees=round(angle, 2), reason="plate_outline",
                    quad=source_quad.round(2).tolist(), method="quadrilateral_perspective",
                    output_size=[out_w, out_h])
    return finish(result)
