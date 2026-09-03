"""High-performance, high-accuracy OCR recognition engine with RapidOCR (PaddleOCR ONNX),
tight crop normalization, multi-line Indian plate aggregation, and fast-exit caching.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.core.logging import logger
from app.services.normalizer import clean_raw_ocr, normalize_indian_plate
from app.services.ocr.preprocessor import PlateImagePreprocessor
from app.services.validator import validate_indian_plate_format


class OCRResult:
    def __init__(
        self,
        raw_text: str,
        normalized_text: str,
        formatted_text: str,
        confidence: float,
        format_status: str,
        best_variant_name: str,
        best_variant_image: np.ndarray,
    ):
        self.raw_text = raw_text
        self.normalized_text = normalized_text
        self.formatted_text = formatted_text
        self.confidence = confidence
        self.format_status = format_status
        self.best_variant_name = best_variant_name
        self.best_variant_image = best_variant_image


class OCREngine:
    """Ultra-fast, high-accuracy OCR Engine with RapidOCR (PaddleOCR ONNX), tight crop normalization, and fast-exit."""

    def __init__(self):
        self._rapidocr = None
        self._engine_type = "heuristic"
        self._init_engine()

    def _init_engine(self):
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._rapidocr = RapidOCR()
            self._engine_type = "rapidocr"
            logger.info("OCR Engine initialized with high-accuracy RapidOCR (PaddleOCR ONNX).")
            return
        except Exception as e:
            logger.warning(f"RapidOCR initialization failed ({e}), falling back to heuristic engine.")
            self._engine_type = "heuristic"

    def _preprocess_crop(self, bgr_crop: np.ndarray, target_height: int = 48) -> np.ndarray:
        """Tightly resizes and enhances the plate crop for optimal text recognition."""
        h, w = bgr_crop.shape[:2]
        if h == 0 or w == 0:
            return bgr_crop

        # Scale to standard height (48px) maintaining aspect ratio
        scale = target_height / float(h)
        target_width = max(32, int(w * scale))
        resized = cv2.resize(bgr_crop, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

        # Apply subtle CLAHE contrast enhancement
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        return enhanced_bgr

    def _run_rapidocr(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        """Runs RapidOCR with vertical line grouping and 2-line plate aggregation."""
        if not self._rapidocr:
            return "", 0.0

        try:
            results, _ = self._rapidocr(img_bgr)
            if not results or len(results) == 0:
                return "", 0.0

            # Sort detected text boxes top-to-bottom, then left-to-right (handles both 1-line and 2-line plates)
            sorted_items = sorted(
                results,
                key=lambda item: (
                    np.mean([p[1] for p in item[0]]),  # Y coordinate (line ordering)
                    np.mean([p[0] for p in item[0]]),  # X coordinate
                ),
            )

            text_chunks = []
            confs = []
            for item in sorted_items:
                text = item[1].strip()
                score = float(item[2])
                if text:
                    text_chunks.append(text)
                    confs.append(score)

            if text_chunks:
                combined_text = " ".join(text_chunks)
                avg_conf = float(np.mean(confs)) if confs else 0.85
                return combined_text.strip(), max(0.35, min(0.99, avg_conf))

        except Exception as e:
            logger.error(f"RapidOCR execution error: {e}")

        return "", 0.0

    def recognize(self, bgr_crop: np.ndarray) -> OCRResult:
        """Fast-path optimized OCR pipeline with early-exit on confident detections."""
        if bgr_crop is None or bgr_crop.size == 0:
            return OCRResult("", "", "", 0.0, "uncertain", "none", np.zeros((10, 10), dtype=np.uint8))

        # 1. Tight crop preprocessing
        enhanced_bgr = self._preprocess_crop(bgr_crop, target_height=48)

        # FAST PATH (Primary evaluation):
        raw_text, ocr_conf = self._run_rapidocr(enhanced_bgr)
        norm_text, fmt_text, status = normalize_indian_plate(raw_text)

        # Early exit if highly confident and valid format
        if status == "valid" and ocr_conf >= 0.70:
            return OCRResult(
                raw_text=raw_text,
                normalized_text=norm_text,
                formatted_text=fmt_text,
                confidence=round(ocr_conf, 2),
                format_status=status,
                best_variant_name="fast_clahe",
                best_variant_image=cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY),
            )

        # 2. Secondary Pass: Raw crop direct pass
        raw_text_2, ocr_conf_2 = self._run_rapidocr(bgr_crop)
        norm_text_2, fmt_text_2, status_2 = normalize_indian_plate(raw_text_2)

        if status_2 == "valid" and ocr_conf_2 >= 0.70:
            return OCRResult(
                raw_text=raw_text_2,
                normalized_text=norm_text_2,
                formatted_text=fmt_text_2,
                confidence=round(ocr_conf_2, 2),
                format_status=status_2,
                best_variant_name="raw_crop",
                best_variant_image=cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY),
            )

        # 3. Multi-Variant fallback for degraded/noisy plates
        variants = PlateImagePreprocessor.generate_all_variants(enhanced_bgr)
        candidates = [
            (raw_text, ocr_conf, norm_text, fmt_text, status, "fast_clahe", enhanced_bgr),
            (raw_text_2, ocr_conf_2, norm_text_2, fmt_text_2, status_2, "raw_crop", enhanced_bgr),
        ]

        for var_name, var_img in variants.items():
            v_bgr = cv2.cvtColor(var_img, cv2.COLOR_GRAY2BGR) if len(var_img.shape) == 2 else var_img
            t, c = self._run_rapidocr(v_bgr)
            if t:
                nt, ft, st = normalize_indian_plate(t)
                candidates.append((t, c, nt, ft, st, var_name, var_img))

        # Select best candidate with weighted scoring
        best_candidate = None
        best_score = -1.0

        for t, c, nt, ft, st, v_name, v_img in candidates:
            if not t and not nt:
                continue

            format_bonus = 1.0 if st == "valid" else (0.70 if st == "possible" else 0.3)
            length_bonus = 1.0 if (8 <= len(nt) <= 11) else 0.5
            score = (c * 0.5) + (format_bonus * 0.35) + (length_bonus * 0.15)

            if score > best_score:
                best_score = score
                best_candidate = OCRResult(
                    raw_text=t,
                    normalized_text=nt,
                    formatted_text=ft,
                    confidence=round(c, 2),
                    format_status=st,
                    best_variant_name=v_name,
                    best_variant_image=v_img,
                )

        if best_candidate is not None:
            return best_candidate

        # Fallback safe default
        return OCRResult(
            raw_text=raw_text or "",
            normalized_text=norm_text or "",
            formatted_text=fmt_text or "",
            confidence=round(ocr_conf, 2),
            format_status=status or "uncertain",
            best_variant_name="fast_clahe",
            best_variant_image=cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY),
        )


# Singleton instance
ocr_engine = OCREngine()
