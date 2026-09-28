"""Vehicle detection providers for the Drishti Grid ingestion pipeline.

These detectors localise *vehicles* (not plates). The plate detector from the
existing PlateVision package is anchored separately for OCR, so vehicle
detection and plate recognition stay decoupled and individually replaceable.

Provider ``VEHICLE_DETECTOR_BACKEND``:
  * ``motion``          background-subtraction of moving objects (works on any
                        live feed without trained weights)
  * ``plate_anchored``  projects plate boxes up to an estimated vehicle region
  * ``yolo``            generic YOLO vehicle ONNX model (COCO classes 2,3,5,7)
  * ``hybrid``          default: motion ∪ plate-anchored, merged with NMS
  * ``synthetic``       colour/rectangle based detector for the demo generator
  * ``mock``            deterministic single box (tests only)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.schemas.detection import BoundingBox

# COCO class ids that correspond to vehicles
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class VehicleDetection:
    bbox: BoundingBox
    confidence: float
    vehicle_type: str = "unknown"
    source: str = "hybrid"


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / float(union) if union > 0 else 0.0


def nms_detections(dets: List[VehicleDetection], iou_thr: float = 0.45) -> List[VehicleDetection]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: List[VehicleDetection] = []
    for d in dets:
        if all(bbox_iou(d.bbox, k.bbox) <= iou_thr for k in kept):
            kept.append(d)
    return kept


def classify_vehicle_type(bbox: BoundingBox, frame_area: float) -> str:
    """Coarse size/aspect heuristic. Not a trained classifier; used for display only."""
    if bbox.height <= 0:
        return "unknown"
    aspect = bbox.width / float(bbox.height)
    rel = (bbox.width * bbox.height) / max(1.0, frame_area)
    if aspect > 2.6 and rel > 0.08:
        return "bus"
    if aspect > 2.0 and rel > 0.04:
        return "truck"
    if aspect < 1.3 and rel < 0.02:
        return "motorcycle"
    if 1.2 <= aspect <= 2.6:
        return "car"
    return "vehicle"


class BaseVehicleDetector:
    name = "base"

    def detect(self, image_bgr: np.ndarray) -> List[VehicleDetection]:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset temporal state (called when a camera restarts)."""


class MotionVehicleDetector(BaseVehicleDetector):
    """Background-subtraction detector. Requires warmup frames to learn background."""

    name = "motion"

    def __init__(self, min_area_ratio: float = None, warmup: int = None):
        self.min_area_ratio = min_area_ratio or settings.MOTION_MIN_AREA_RATIO
        self.warmup = warmup or settings.MOTION_WARMUP_FRAMES
        self._bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)
        self._frames = 0

    def reset(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)
        self._frames = 0

    def detect(self, image_bgr: np.ndarray) -> List[VehicleDetection]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        self._frames += 1
        small = cv2.resize(image_bgr, (320, int(320 * h / max(1, w))))
        fg = self._bg.apply(small)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((7, 7), np.uint8), iterations=2)

        if self._frames < self.warmup:
            return []

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sx = w / small.shape[1]
        sy = h / small.shape[0]
        frame_area = float(h * w)
        dets: List[VehicleDetection] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < self.min_area_ratio * small.shape[0] * small.shape[1]:
                continue
            aspect = bw / float(bh) if bh else 0
            if not (0.4 <= aspect <= 5.0):
                continue
            solidity = cv2.contourArea(c) / float(area) if area else 0
            conf = float(min(0.95, 0.45 + 0.5 * solidity))
            box = BoundingBox(
                x=int(x * sx), y=int(y * sy), width=int(bw * sx), height=int(bh * sy)
            )
            dets.append(
                VehicleDetection(
                    bbox=box,
                    confidence=conf,
                    vehicle_type=classify_vehicle_type(box, frame_area),
                    source="motion",
                )
            )
        return nms_detections(dets)


class PlateAnchoredVehicleDetector(BaseVehicleDetector):
    """Derives vehicle regions from plate detections (plate sits near the lower part of a vehicle)."""

    name = "plate_anchored"

    def __init__(self, plate_detector):
        self.plate_detector = plate_detector

    def detect(self, image_bgr: np.ndarray) -> List[VehicleDetection]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        try:
            plates = self.plate_detector.detect(image_bgr)
        except Exception as e:
            logger.debug(f"Plate-anchored detector: plate detector error {e}")
            return []

        frame_area = float(h * w)
        dets: List[VehicleDetection] = []
        for p in plates:
            pb = p.bbox
            # Extend the plate box into the full vehicle body.
            vw = int(pb.width * 3.4)
            vh = int(pb.height * 3.4)
            vx = int(pb.x - (vw - pb.width) / 2)
            vy = int(pb.y - vh + pb.height)  # plate is near the bottom of the vehicle
            vx = max(0, vx)
            vy = max(0, vy)
            vw = min(w - vx, vw)
            vh = min(h - vy, vh)
            if vw < 20 or vh < 20:
                continue
            box = BoundingBox(x=vx, y=vy, width=vw, height=vh)
            dets.append(
                VehicleDetection(
                    bbox=box,
                    confidence=float(min(0.9, p.confidence)),
                    vehicle_type=classify_vehicle_type(box, frame_area),
                    source="plate_anchored",
                )
            )
        return nms_detections(dets)


class OnnxVehicleDetector(BaseVehicleDetector):
    """Generic YOLO vehicle detector exported to ONNX (classes 2,3,5,7)."""

    name = "yolo"

    def __init__(self, model_path: str = None, conf: float = None):
        self.model_path = model_path or settings.VEHICLE_MODEL_PATH
        self.conf = conf or settings.VEHICLE_DETECTION_CONFIDENCE
        self.session = None
        self.input_name = None
        self.input_shape = (640, 640)
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning(f"Vehicle ONNX model not found at '{self.model_path}'.")
            return
        try:
            import onnxruntime as ort  # type: ignore

            self.session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self.input_shape = (shape[2], shape[3])
            logger.info(f"Loaded vehicle ONNX detector from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load vehicle ONNX model: {e}")
            self.session = None

    def is_ready(self) -> bool:
        return self.session is not None

    def detect(self, image_bgr: np.ndarray) -> List[VehicleDetection]:
        if not self.is_ready() or image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        th, tw = self.input_shape
        r = min(tw / w, th / h)
        rw, rh = int(round(w * r)), int(round(h * r))
        pad_w, pad_h = (tw - rw) // 2, (th - rh) // 2
        canvas = np.full((th, tw, 3), 114, np.uint8)
        canvas[pad_h : pad_h + rh, pad_w : pad_w + rw] = cv2.resize(image_bgr, (rw, rh))

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        out = self.session.run(None, {self.input_name: blob})[0]
        if out.ndim == 3:
            out = out[0]
        if out.shape[0] < out.shape[1]:
            out = out.T

        frame_area = float(h * w)
        dets: List[VehicleDetection] = []
        for row in out:
            if len(row) < 5:
                continue
            cx, cy, bw, bh = row[:4]
            scores = row[4:]
            cls = int(np.argmax(scores))
            score = float(scores[cls])
            if score < self.conf or (len(scores) > 1 and cls not in VEHICLE_CLASS_IDS and len(scores) > 5):
                continue
            vtype = VEHICLE_CLASS_IDS.get(cls, "vehicle")
            x1 = int((cx - bw / 2 - pad_w) / r)
            y1 = int((cy - bh / 2 - pad_h) / r)
            x2 = int((cx + bw / 2 - pad_w) / r)
            y2 = int((cy + bh / 2 - pad_h) / r)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(
                VehicleDetection(
                    bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                    confidence=score,
                    vehicle_type=vtype,
                    source="yolo",
                )
            )
        return nms_detections(dets)


class HybridVehicleDetector(BaseVehicleDetector):
    """Default detector: union of motion and plate-anchored regions, merged via NMS."""

    name = "hybrid"

    def __init__(self, motion: MotionVehicleDetector, anchored: PlateAnchoredVehicleDetector):
        self.motion = motion
        self.anchored = anchored

    def reset(self) -> None:
        self.motion.reset()

    def detect(self, image_bgr: np.ndarray) -> List[VehicleDetection]:
        dets = self.motion.detect(image_bgr)
        dets.extend(self.anchored.detect(image_bgr))
        # Favour plate-anchored boxes on overlapping regions.
        dets.sort(key=lambda d: (d.source == "plate_anchored", d.confidence), reverse=True)
        return nms_detections(dets, iou_thr=0.40)


class SyntheticVehicleDetector(BaseVehicleDetector):
    """Detects the coloured rectangles drawn by the synthetic demo generator.

    Used only in demo mode so the UI always has real detections to display
    when no physical camera is attached. It is intentionally separate from
    the production detector path.
    """

    name = "synthetic"

    def __init__(self, motion: MotionVehicleDetector = None):
        self.motion = motion or MotionVehicleDetector()

    def reset(self) -> None:
        self.motion.reset()

    def detect(self, image_bgr: np.ndarray) -> List[VehicleDetection]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Vehicle bodies use saturated colours on a muted road background.
        mask = cv2.inRange(hsv, (0, 90, 60), (180, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(h * image_bgr.shape[1])
        dets: List[VehicleDetection] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < 0.01 * frame_area or bw < 40 or bh < 30:
                continue
            box = BoundingBox(x=x, y=y, width=bw, height=bh)
            dets.append(
                VehicleDetection(
                    bbox=box,
                    confidence=0.9,
                    vehicle_type=classify_vehicle_type(box, frame_area),
                    source="synthetic",
                )
            )
        return nms_detections(dets)


def build_vehicle_detector(plate_detector, backend: Optional[str] = None) -> BaseVehicleDetector:
    backend = (backend or settings.VEHICLE_DETECTOR_BACKEND).lower()
    if backend == "mock":

        class _Mock(BaseVehicleDetector):
            name = "mock"

            def detect(self, image_bgr):
                h, w = image_bgr.shape[:2]
                return [
                    VehicleDetection(
                        bbox=BoundingBox(x=int(w * 0.3), y=int(h * 0.4), width=int(w * 0.4), height=int(h * 0.25)),
                        confidence=0.8,
                        vehicle_type="car",
                        source="mock",
                    )
                ]

        return _Mock()

    if backend == "motion":
        return MotionVehicleDetector()
    if backend == "plate_anchored":
        return PlateAnchoredVehicleDetector(plate_detector)
    if backend == "synthetic":
        return SyntheticVehicleDetector()
    if backend == "yolo":
        det = OnnxVehicleDetector()
        if det.is_ready():
            return det
        logger.warning("Vehicle YOLO model unavailable; falling back to hybrid detector.")
        return HybridVehicleDetector(MotionVehicleDetector(), PlateAnchoredVehicleDetector(plate_detector))

    # hybrid (default)
    return HybridVehicleDetector(MotionVehicleDetector(), PlateAnchoredVehicleDetector(plate_detector))
