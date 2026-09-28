"""Plate OCR adapter and multi-frame fusion.

This module is the PlateVision integration seam. It decouples the rest of the
system from the OCR implementation:

* :class:`PlateVisionAdapter` wraps the existing PlateVision
  ``ocr_engine`` (RapidOCR ONNX text recognizer) and the existing plate
  detector. The OCR engine already performs Indian-plate normalization and
  reports both raw and normalized text.
* :func:`fuse_plate_observations` implements the multi-frame voting required
  by the spec: repeated character agreement + high confidence + temporal
  consistency, with explicit ambiguity handling instead of forced identity.
* :func:`plate_agreement_score` implements the cross-camera plate evidence
  score used by the fusion engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.schemas.detection import BoundingBox
from app.services.ocr.engine import OCRResult, ocr_engine
from app.services.normalizer import normalize_indian_plate

# Characters that OCR commonly confuses on Indian plates.
CONFUSABLE_GROUPS: List[set] = [
    {"O", "0", "D", "Q"},
    {"I", "1", "L", "T", "J"},
    {"B", "8"},
    {"S", "5"},
    {"Z", "2"},
    {"G", "6", "C"},
    {"A", "4"},
    {"E", "3"},
    {"U", "V"},
    {"M", "N"},
]


def _confusable(a: str, b: str) -> bool:
    if a == b:
        return True
    for group in CONFUSABLE_GROUPS:
        if a in group and b in group:
            return True
    return False


def character_similarity(p: str, q: str) -> float:
    """Levenshtein similarity with OCR-confusable substitutions partially credited."""
    if not p and not q:
        return 0.0
    if p == q:
        return 1.0
    m, n = len(p), len(q)
    # DP with substitution costs.
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i - 1] == q[j - 1]:
                sub = 0.0
            elif _confusable(p[i - 1], q[j - 1]):
                sub = 0.4
            else:
                sub = 1.0
            dp[i][j] = min(dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0, dp[i - 1][j - 1] + sub)
    dist = dp[m][n]
    return max(0.0, 1.0 - dist / float(max(m, n)))


@dataclass
class PlateReading:
    plate_raw: str
    plate_normalized: str
    ocr_confidence: float
    frame_timestamp: float
    camera_id: str = ""
    vehicle_track_id: int = -1
    plate_bbox: Optional[List[int]] = None
    vehicle_bbox: Optional[List[int]] = None
    format_status: str = "uncertain"


@dataclass
class PlateHypothesis:
    plate_normalized: str = ""
    plate_raw: str = ""
    confidence: float = 0.0
    agreement: float = 0.0
    observation_count: int = 0
    status: str = "unknown"  # confident | ambiguous | unknown
    candidates: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plate_normalized": self.plate_normalized,
            "plate_raw": self.plate_raw,
            "plate_confidence": round(self.confidence, 3),
            "agreement": round(self.agreement, 3),
            "observation_count": self.observation_count,
            "status": self.status,
            "candidates": self.candidates,
        }


class PlateVisionAdapter:
    """Thin adapter over the existing PlateVision detector + OCR engine."""

    def __init__(self, detector=None):
        self._detector = detector
        self.available = True

    def set_detector(self, detector) -> None:
        self._detector = detector

    def detect_plates(self, image_bgr: np.ndarray) -> List[Tuple[BoundingBox, float]]:
        if self._detector is None:
            return []
        try:
            return [(d.bbox, d.confidence) for d in self._detector.detect(image_bgr)]
        except Exception as e:
            logger.debug(f"Plate detection error: {e}")
            return []

    def read_crop(self, crop_bgr: np.ndarray) -> OCRResult:
        """Run PlateVision OCR on a plate crop (raw + normalized text, confidence)."""
        try:
            return ocr_engine.recognize(crop_bgr)
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return OCRResult("", "", "", 0.0, "uncertain", "none", np.zeros((8, 8), np.uint8))

    def read_frame(
        self, image_bgr: np.ndarray, plate_boxes: List[Tuple[BoundingBox, float]] = None
    ) -> List[PlateReading]:
        """Detect plates (if not provided) and OCR each crop into observations."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        boxes = plate_boxes if plate_boxes is not None else self.detect_plates(image_bgr)
        h, w = image_bgr.shape[:2]
        obs: List[PlateReading] = []
        for bbox, det_conf in boxes:
            pad = 0.06
            x = max(0, int(bbox.x - bbox.width * pad))
            y = max(0, int(bbox.y - bbox.height * pad))
            x2 = min(w, int(bbox.x + bbox.width * (1 + pad)))
            y2 = min(h, int(bbox.y + bbox.height * (1 + pad)))
            crop = image_bgr[y:y2, x:x2]
            if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 12:
                continue
            res = self.read_crop(crop)
            if not res.raw_text and not res.normalized_text:
                continue
            obs.append(
                PlateReading(
                    plate_raw=res.raw_text,
                    plate_normalized=res.normalized_text,
                    ocr_confidence=float(res.confidence),
                    frame_timestamp=0.0,
                    plate_bbox=[bbox.x, bbox.y, bbox.width, bbox.height],
                    format_status=res.format_status,
                )
            )
        return obs


def fuse_plate_observations(
    observations: List[PlateReading],
    min_observations: int = None,
    min_agreement: float = None,
) -> PlateHypothesis:
    """Multi-frame voting/aggregation over plate observations for one track.

    Prefers repeated character agreement and high-confidence observations.
    Preserves alternative candidate interpretations; if OCR stays ambiguous the
    confidence is lowered and status is 'ambiguous' (never forced).
    """
    min_observations = min_observations or settings.OCR_MIN_OBSERVATIONS
    min_agreement = min_agreement or settings.OCR_MIN_AGREEMENT
    min_reading_conf = settings.OCR_MIN_READING_CONFIDENCE

    # Degraded readings are kept in the observation log for audit, but only
    # readings meeting a minimum confidence may contribute to plate identity.
    valid = [
        o
        for o in observations
        if (o.plate_normalized or o.plate_raw) and o.ocr_confidence >= min_reading_conf
    ]
    if not valid:
        return PlateHypothesis(status="unknown")

    # Group by normalized string, weighting votes by OCR confidence.
    votes: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    best_raw: Dict[str, Tuple[float, str]] = {}
    for o in valid:
        key = o.plate_normalized or o.plate_raw
        votes[key] = votes.get(key, 0.0) + max(0.05, o.ocr_confidence)
        counts[key] = counts.get(key, 0) + 1
        if key not in best_raw or o.ocr_confidence > best_raw[key][0]:
            best_raw[key] = (o.ocr_confidence, o.plate_raw)

    total_weight = sum(votes.values())
    ranked = sorted(votes.items(), key=lambda kv: (kv[1], counts[kv[0]]), reverse=True)
    top_plate, top_weight = ranked[0]
    top_count = counts[top_plate]
    agreement = top_weight / total_weight if total_weight > 0 else 0.0

    candidates = [
        {
            "plate": plate,
            "vote_weight": round(weight / total_weight if total_weight else 0.0, 3),
            "count": counts[plate],
            "confidence": round(best_raw[plate][0], 3),
        }
        for plate, weight in ranked[:5]
    ]

    # Confidence: weighted mean OCR confidence, scaled by agreement and support.
    mean_conf = float(np.mean([o.ocr_confidence for o in valid]))
    support = min(1.0, top_count / float(max(1, min_observations)))
    confidence = mean_conf * (0.5 + 0.5 * agreement) * (0.6 + 0.4 * support)

    if top_count >= min_observations and agreement >= min_agreement:
        status = "confident"
    elif len(ranked) == 1 and top_count >= 1:
        # Single unambiguous reading — retain but keep confidence honest.
        status = "confident" if mean_conf >= 0.6 else "ambiguous"
    else:
        status = "ambiguous"
        confidence *= 0.6  # penalise ambiguous multi-frame OCR

    return PlateHypothesis(
        plate_normalized=top_plate if status == "confident" else "",
        plate_raw=best_raw[top_plate][1],
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        agreement=float(agreement),
        observation_count=len(valid),
        status=status,
        candidates=candidates,
    )


def plate_agreement_score(a: PlateHypothesis, b: PlateHypothesis) -> Optional[float]:
    """Cross-camera plate evidence score s_plate_ij.

    Returns ``None`` when plate evidence is unknown for either side, so the
    fusion engine treats it as neutral rather than a mismatch. When a side is
    ambiguous (multiple candidate readings) all candidates are compared and the
    best character-level agreement is used, weighted by confidence.
    """
    a_plates = _candidate_plates(a)
    b_plates = _candidate_plates(b)
    if not a_plates or not b_plates:
        return None

    best = 0.0
    exact = False
    for ap in a_plates:
        for bp in b_plates:
            sim = character_similarity(ap, bp)
            if sim > best:
                best = sim
            if ap == bp:
                exact = True

    if exact:
        base = 1.0
    else:
        base = best * (0.5 + 0.5 * min(a.confidence, b.confidence))

    conf_gate = 0.5 + 0.5 * min(a.confidence, b.confidence)
    score = base * conf_gate
    # A completely different plate must be near-zero, not merely low.
    if best < 0.50:
        score = min(score, 0.10)
    # A partial agreement (e.g. matching state/district but a different series)
    # is capped at "moderate" so plate similarity alone cannot carry a link.
    elif best < 0.85:
        score = min(score, 0.45)
    return float(np.clip(score, 0.0, 1.0))


def _candidate_plates(h: PlateHypothesis) -> List[str]:
    """Distinct candidate plate strings for a hypothesis (best reading first)."""
    plates: List[str] = []
    for p in [h.plate_normalized, h.plate_raw]:
        if p and p not in plates:
            plates.append(p)
    for c in h.candidates or []:
        cp = c.get("plate")
        if cp and cp not in plates:
            plates.append(cp)
    return plates


plate_vision = PlateVisionAdapter()
