"""Bounded secondary plate-model searches inside vehicles lacking plate evidence.

Whole-scene plate results are always preserved. A vehicle rectangle is only a
search region: the plate-trained model must return an actual qualifying box.
OCR, vehicle category and generic rectangle geometry never manufacture plates.
"""
from itertools import islice
import math
import time

import numpy as np

from app.core.config import settings


def _box(item, fields=5):
    try:
        if len(item) < fields or any(isinstance(value, (bool, str, bytes)) for value in item[:5]):
            return None
        x, y, width, height, score = map(float, item[:5])
    except (TypeError, ValueError, OverflowError):
        return None
    if (not all(math.isfinite(value) for value in (x, y, width, height, score))
            or width <= 0 or height <= 0 or not 0 <= score <= 1):
        return None
    return x, y, width, height, score


def _contains_centre(vehicle, plate):
    x, y, width, height = vehicle[:4]
    cx, cy = plate[0] + plate[2] / 2, plate[1] + plate[3] / 2
    return x <= cx <= x + width and y <= cy <= y + height


def _duplicate(first, second):
    x, y, width, height = first[:4]
    bx, by, bw, bh = second[:4]
    intersection = max(0., min(x + width, bx + bw) - max(x, bx)) * max(
        0., min(y + height, by + bh) - max(y, by))
    small, large = sorted((width * height, bw * bh))
    return (intersection / max(1e-9, small + large - intersection) > .4
            or (intersection / max(1e-9, small) > .8 and small / max(1e-9, large) >= .4))


def recover_vehicle_plates(image, detector, plate_boxes, vehicle_objects, *,
                           start_index=0, max_regions=None):
    """Return existing plus model-detected plates, with bounded-search evidence.

    Each eligible vehicle is attempted once, in rotating input order. An
    optional max_regions can reduce the configured crop budget. Individual
    inference failures are contained. New box centres and at least 75% of
    their area must lie in the parent vehicle; modest bumper clipping is
    allowed, while padding cannot supply an unrelated neighbouring plate.
    """
    started = time.perf_counter()
    merged = list(plate_boxes)
    capability = getattr(detector, 'detect_vehicle_regions', None)
    enabled = bool(settings.vehicle_plate_search_enabled and callable(capability))
    report = dict(enabled=enabled, eligible_vehicles=0, searched_vehicles=0,
                  added_regions=0, time_ms=0., failed_crops=0, remaining_vehicles=0,
                  next_start_index=0)

    def finish(reason=None):
        if reason:
            report['reason'] = reason
        report['time_ms'] = round((time.perf_counter() - started) * 1000, 3)
        return merged, report

    if not enabled:
        return finish('disabled' if not settings.vehicle_plate_search_enabled else 'unsupported_detector')
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        return finish('invalid_image')
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError('start_index must be a nonnegative integer.')
    if max_regions is not None and (isinstance(max_regions, bool) or not isinstance(max_regions, int)
                                     or max_regions < 0):
        raise ValueError('max_regions must be a nonnegative integer or None.')
    budget = settings.vehicle_plate_max_crops
    if max_regions is not None:
        budget = min(budget, max_regions)
    height, width = image.shape[:2]
    vehicles, seen = [], set()
    for item in vehicle_objects:
        box = _box(item, fields=6)
        if box is None:
            continue
        x, y, vw, vh, score = box
        left, top = max(0., x), max(0., y)
        right, bottom = min(float(width), x + vw), min(float(height), y + vh)
        if right <= left or bottom <= top:
            continue
        clipped = (left, top, right - left, bottom - top, score)
        if clipped[:4] not in seen:
            seen.add(clipped[:4])
            vehicles.append(clipped)
    existing = [box for item in merged if (box := _box(item)) is not None]
    owned = set()
    for plate in existing:
        enclosing = [index for index, vehicle in enumerate(vehicles) if _contains_centre(vehicle, plate)]
        if enclosing:
            owned.add(min(enclosing, key=lambda index: vehicles[index][2] * vehicles[index][3]))
    eligible = [vehicle for index, vehicle in enumerate(vehicles) if index not in owned]
    report['eligible_vehicles'] = len(eligible)
    if not eligible:
        return finish('no_eligible_vehicles')
    start = start_index % len(eligible)
    selected = (eligible[start:] + eligible[:start])[:budget]
    report['remaining_vehicles'] = len(eligible) - len(selected)
    report['next_start_index'] = (start + len(selected)) % len(eligible)
    proposals = []
    for x, y, vw, vh, _ in selected:
        left, top = max(0, math.floor(x - .08 * vw)), max(0, math.floor(y - .08 * vh))
        right = min(width, math.ceil(x + vw + .08 * vw))
        bottom = min(height, math.ceil(y + vh + .08 * vh))
        crop = image[top:bottom, left:right]
        report['searched_vehicles'] += 1
        try:
            candidates = []
            for item in islice(capability(crop), settings.max_detections):
                candidate = _box(item)
                if candidate is None:
                    continue
                px, py, pw, ph, score = candidate
                if (score < settings.region_confidence_threshold or not .1 <= pw / ph <= 10
                        or px < 0 or py < 0 or px + pw > crop.shape[1] or py + ph > crop.shape[0]):
                    continue
                absolute = (px + left, py + top, pw, ph, score)
                intersection = max(0., min(absolute[0] + pw, x + vw) - max(absolute[0], x)) * max(
                    0., min(absolute[1] + ph, y + vh) - max(absolute[1], y))
                if (not _contains_centre((x, y, vw, vh), absolute)
                        or intersection / (pw * ph) < .75):
                    continue
                candidates.append(absolute)
            proposals.extend(candidates)
        except Exception:
            report['failed_crops'] += 1
    for candidate in sorted(proposals, key=lambda box: -box[4]):
        if any(_duplicate(candidate, previous) for previous in existing):
            continue
        merged.append(candidate)
        existing.append(candidate)
        report['added_regions'] += 1
    return finish('crop_budget_zero' if not selected else None)
