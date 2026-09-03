"""Production-grade Multi-Scale Tiled (SAHI-style) YOLOv8 ONNX Plate Detector.
Captures all two-wheelers, motorcycles, distant vehicles, and multi-vehicle traffic scenes
by combining full-frame global inference with high-resolution overlapping tile slices.
"""

import os
from typing import List, Tuple
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.schemas.detection import BoundingBox
from app.services.detector.base import BasePlateDetector, RawDetection


def compute_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    x1 = max(b1.x, b2.x)
    y1 = max(b1.y, b2.y)
    x2 = min(b1.x + b1.width, b2.x + b2.width)
    y2 = min(b1.y + b1.height, b2.y + b2.height)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    b1_area = b1.width * b1.height
    b2_area = b2.width * b2.height
    union_area = b1_area + b2_area - inter_area

    return inter_area / float(union_area) if union_area > 0 else 0.0


class OnnxPlateDetector(BasePlateDetector):
    """High-accuracy multi-scale tiled ONNX YOLOv8 detector for cars, two-wheelers & distant plates."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.ONNX_MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = (640, 640)
        self._load_session()

    def _load_session(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"ONNX model file not found at '{self.model_path}'.")
            return

        try:
            import onnxruntime as ort  # type: ignore

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self.input_shape = (shape[2], shape[3])

            logger.info(f"Loaded high-performance ONNX YOLO detector from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.session = None

    def is_ready(self) -> bool:
        return self.session is not None

    def _letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color=(114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img, r, (int(round(dw)), int(round(dh)))

    def _infer_yolo(
        self, image_bgr: np.ndarray, conf_threshold: float = 0.10, iou_threshold: float = 0.40
    ) -> List[Tuple[BoundingBox, float]]:
        if not self.session or image_bgr is None or image_bgr.size == 0:
            return []

        orig_h, orig_w = image_bgr.shape[:2]
        img_letterboxed, ratio, (pad_w, pad_h) = self._letterbox(image_bgr, self.input_shape)

        img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
        input_tensor = img_rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        raw_output = outputs[0]

        if len(raw_output.shape) == 3:
            if raw_output.shape[1] < raw_output.shape[2]:
                raw_output = np.transpose(raw_output[0], (1, 0))
            else:
                raw_output = raw_output[0]

        boxes = []
        confidences = []

        for row in raw_output:
            cx, cy, w, h = row[0:4]
            scores = row[4:]
            max_score = float(np.max(scores)) if len(scores) > 0 else float(row[4])

            if max_score >= conf_threshold:
                x1 = cx - w / 2
                y1 = cy - h / 2

                x1_orig = (x1 - pad_w) / ratio
                y1_orig = (y1 - pad_h) / ratio
                w_orig = w / ratio
                h_orig = h / ratio

                x1_clamped = max(0, min(orig_w - 1, int(x1_orig)))
                y1_clamped = max(0, min(orig_h - 1, int(y1_orig)))
                w_clamped = max(1, min(orig_w - x1_clamped, int(w_orig)))
                h_clamped = max(1, min(orig_h - y1_clamped, int(h_orig)))

                # Geometric validation: supports two-wheelers (square/small) & cars (rectangular)
                aspect = w_clamped / float(h_clamped) if h_clamped > 0 else 0
                if 1.0 <= aspect <= 7.5 and w_clamped >= 12 and h_clamped >= 6:
                    boxes.append([x1_clamped, y1_clamped, w_clamped, h_clamped])
                    confidences.append(float(max_score))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                b = boxes[idx]
                results.append((BoundingBox(x=b[0], y=b[1], width=b[2], height=b[3]), confidences[idx]))

        return results

    def detect(self, image_bgr: np.ndarray) -> List[RawDetection]:
        if not self.is_ready():
            raise RuntimeError("ONNX Plate Detector is not initialized.")

        h, w = image_bgr.shape[:2]
        all_candidates: List[Tuple[BoundingBox, float]] = []

        # 1. Global Full-Image Pass (catches nearby/large vehicles)
        global_dets = self._infer_yolo(image_bgr, conf_threshold=0.10, iou_threshold=0.40)
        all_candidates.extend(global_dets)

        # 2. Multi-Scale Overlapping Tiles for High-Resolution Scenes:
        # Slices image into 4 overlapping quadrants (magnifying small two-wheeler & motorcycle plates by 2x-3x)
        if w >= 640 or h >= 480:
            tile_h = int(h * 0.60)
            tile_w = int(w * 0.60)

            # 4 Corner Tiles + 1 Center Tile
            tiles_coords = [
                (0, 0),  # Top-Left
                (0, w - tile_w),  # Top-Right
                (h - tile_h, 0),  # Bottom-Left
                (h - tile_h, w - tile_w),  # Bottom-Right
                (int((h - tile_h) / 2), int((w - tile_w) / 2)),  # Center Zoom
            ]

            for y0, x0 in tiles_coords:
                tile = image_bgr[y0 : y0 + tile_h, x0 : x0 + tile_w]
                tile_dets = self._infer_yolo(tile, conf_threshold=0.10, iou_threshold=0.35)

                for tb, tc in tile_dets:
                    mapped_box = BoundingBox(
                        x=tb.x + x0,
                        y=tb.y + y0,
                        width=tb.width,
                        height=tb.height,
                    )
                    # Check deduplication against existing candidates
                    if not any(compute_iou(mapped_box, ex_b) > 0.35 for ex_b, _ in all_candidates):
                        all_candidates.append((mapped_box, tc))

        if not all_candidates:
            return []

        # 3. Global NMS Merge & Spatial Ordering
        boxes_list = [[b.x, b.y, b.width, b.height] for b, _ in all_candidates]
        confs_list = [c for _, c in all_candidates]

        indices = cv2.dnn.NMSBoxes(boxes_list, confs_list, 0.10, 0.35)
        merged: List[Tuple[BoundingBox, float]] = []
        if len(indices) > 0:
            for idx in indices.flatten():
                b = boxes_list[idx]
                merged.append((BoundingBox(x=b[0], y=b[1], width=b[2], height=b[3]), confs_list[idx]))

        # Sort spatially: left-to-right, top-to-bottom
        merged.sort(key=lambda item: (item[0].y // 60, item[0].x))

        return [
            RawDetection(bbox=box, confidence=round(conf, 2))
            for box, conf in merged[:25]
        ]
