"""OCR image preprocessing variants pipeline for vehicle registration plates."""

from typing import Dict, List, Tuple
import cv2
import numpy as np


class PlateImagePreprocessor:
    """Provides multiple specialized computer-vision preprocessing pipelines for license plate crops."""

    @staticmethod
    def upscale_if_needed(image: np.ndarray, min_height: int = 70) -> np.ndarray:
        """Upscales small license plate crops to ensure characters are distinct for OCR."""
        h, w = image.shape[:2]
        if h < min_height:
            scale = min_height / float(h)
            new_w = int(w * scale)
            new_h = min_height
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return image

    @staticmethod
    def correct_skew(gray: np.ndarray) -> np.ndarray:
        """Attempts skew angle correction using minAreaRect on high-gradient plate contours."""
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            coords = np.column_stack(np.where(edges > 0))
            if len(coords) < 10:
                return gray
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            if abs(angle) > 0.5 and abs(angle) < 25.0:
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
        except Exception:
            pass
        return gray

    @staticmethod
    def variant_clahe_otsu(bgr_crop: np.ndarray) -> np.ndarray:
        """Variant A: Upscale + CLAHE contrast enhancement + Otsu adaptive threshold."""
        upscaled = PlateImagePreprocessor.upscale_if_needed(bgr_crop)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        gray = PlateImagePreprocessor.correct_skew(gray)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def variant_bilateral_adaptive(bgr_crop: np.ndarray) -> np.ndarray:
        """Variant B: Upscale + Bilateral Filter (edge-preserving denoising) + Adaptive Gaussian Threshold."""
        upscaled = PlateImagePreprocessor.upscale_if_needed(bgr_crop)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )
        return thresh

    @staticmethod
    def variant_morph_sharpen(bgr_crop: np.ndarray) -> np.ndarray:
        """Variant C: Upscale + Sharpening kernel + TopHat / BlackHat morphological enhancement."""
        upscaled = PlateImagePreprocessor.upscale_if_needed(bgr_crop)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        # Sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(sharpened, cv2.MORPH_TOPHAT, rect_kernel)
        blackhat = cv2.morphologyEx(sharpened, cv2.MORPH_BLACKHAT, rect_kernel)
        enhanced = cv2.add(sharpened, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)
        return enhanced

    @staticmethod
    def variant_enhanced_grayscale(bgr_crop: np.ndarray) -> np.ndarray:
        """Variant D: High-fidelity normalized grayscale with contrast stretching."""
        upscaled = PlateImagePreprocessor.upscale_if_needed(bgr_crop)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return normalized

    @classmethod
    def generate_all_variants(cls, bgr_crop: np.ndarray) -> Dict[str, np.ndarray]:
        """Generates all 4 preprocessed variants for OCR evaluation."""
        return {
            "clahe_otsu": cls.variant_clahe_otsu(bgr_crop),
            "bilateral_adaptive": cls.variant_bilateral_adaptive(bgr_crop),
            "morph_sharpen": cls.variant_morph_sharpen(bgr_crop),
            "enhanced_grayscale": cls.variant_enhanced_grayscale(bgr_crop),
        }
