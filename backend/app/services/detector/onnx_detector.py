"""Hybrid Multi-Plate Detector combining YOLOv8 ONNX object detection, DBNet text proposals,
and morphological gradient analysis for maximum recall across all visible vehicle plates.
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
    """Computes Intersection over Union (IoU) between two bounding boxes."""
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
    """Production Multi-Plate Hybrid Detector (YOLOv8 + Scene Text DBNet + Edge Contours)."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.ONNX_MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = (640, 640)
        self._rapidocr = None
        self._load_session()
        self._load_rapidocr()

    def _load_session(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"ONNX model file not found at '{self.model_path}'.")
            return

        try:
            import onnxruntime as ort  # type: ignore

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
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

            logger.info(f"Loaded ONNX YOLO detector from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.session = None

    def _load_rapidocr(self):
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._rapidocr = RapidOCR()
        except Exception as e:
            self._rapidocr = None

    def is_ready(self) -> bool:
        return self.session is not None or self._rapidocr is not None

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

    def _detect_yolo(self, image_bgr: np.ndarray, conf_threshold: float = 0.08) -> List[Tuple[BoundingBox, float]]:
        """Pass 1: High-sensitivity YOLOv8 detection."""
        if not self.session:
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

                boxes.append([x1_clamped, y1_clamped, w_clamped, h_clamped])
                confidences.append(float(max_score))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.35)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                b = boxes[idx]
                results.append((BoundingBox(x=b[0], y=b[1], width=b[2], height=b[3]), confidences[idx]))

        return results

    def _detect_scene_text(self, image_bgr: np.ndarray) -> List[Tuple[BoundingBox, float]]:
        """Pass 2: DBNet deep text detector on the full frame to catch any plates missed by YOLO."""
        if not self._rapidocr:
            return []

        orig_h, orig_w = image_bgr.shape[:2]
        results = []

        try:
            ocr_res, _ = self._rapidocr(image_bgr)
            if ocr_res:
                for item in ocr_res:
                    pts = np.array(item[0], dtype=np.int32)
                    text = item[1].strip()
                    conf = float(item[2])

                    # Plate text generally has at least 4 alphanumeric characters
                    cleaned = "".join(filter(str.isalnum, text))
                    if len(cleaned) >= 3 and conf >= 0.30:
                        bx, by, bw, bh = cv2.boundingRect(pts)
                        # Expand slightly to cover full plate bezel
                        pad_x = int(bw * 0.12)
                        pad_y = int(bh * 0.18)
                        x1 = max(0, bx - pad_x)
                        y1 = max(0, by - pad_y)
                        w1 = min(orig_w - x1, bw + 2 * pad_x)
                        h1 = min(orig_h - y1, bh + 2 * pad_y)

                        aspect = w1 / float(h1) if h1 > 0 else 0
                        if 1.5 <= aspect <= 7.0 and w1 >= 30 and h1 >= 12:
                            results.append((BoundingBox(x=x1, y=y1, width=w1, height=h1), conf))
        except Exception as e:
            logger.debug(f"Scene text proposal error: {e}")

        return results

    def _detect_morphological(self, image_bgr: np.ndarray) -> List[Tuple[BoundingBox, float]]:
        """Pass 3: Morphological edge gradient plate finder for high-contrast plates."""
        orig_h, orig_w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_v, max_v = np.min(grad_x), np.max(grad_x)
        if max_v > min_v:
            grad_x = (255 * ((grad_x - min_v) / (max_v - min_v))).astype("uint8")
        else:
            return []

        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)))
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bw / float(bh) if bh > 0 else 0
            rel_area = (bw * bh) / float(orig_w * orig_h)

            if 2.2 <= aspect <= 6.5 and 0.003 <= rel_area <= 0.30 and bw >= 45 and bh >= 15:
                results.append((BoundingBox(x=bx, y=by, width=bw, height=bh), 0.70))

        return results

    def detect(self, image_bgr: np.ndarray) -> List[RawDetection]:
        if not self.is_ready():
            raise RuntimeError("Detector not ready.")

        # 1. Gather proposals from all 3 complementary detectors
        yolo_dets = self._detect_yolo(image_bgr, conf_threshold=0.08)
        text_dets = self._detect_scene_text(image_bgr)
        morph_dets = self._detect_morphological(image_bgr)

        # Merge candidate pools: YOLO first (highest priority), then Text, then Morph
        all_candidates: List[Tuple[BoundingBox, float]] = []
        all_candidates.extend(yolo_dets)

        for t_box, t_conf in text_dets:
            # Check overlap with existing YOLO detections
            if not any(compute_iou(t_box, ex_box) > 0.35 for ex_box, _ in all_candidates):
                all_candidates.append((t_box, t_conf))

        for m_box, m_conf in morph_dets:
            if not any(compute_iou(m_box, ex_box) > 0.35 for ex_box, _ in all_candidates):
                all_candidates.append((m_box, m_conf))

        if not all_candidates:
            return []

        # Sort spatially: left-to-right, top-to-bottom
        all_candidates.sort(key=lambda item: (item[0].y // 60, item[0].x))

        return [
            RawDetection(bbox=box, confidence=round(conf, 2))
            for box, conf in all_candidates[:15]
        ]
