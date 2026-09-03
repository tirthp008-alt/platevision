"""Detector factory and singleton management."""

import os
from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.services.detector.base import BasePlateDetector
from app.services.detector.mock_detector import MockPlateDetector
from app.services.detector.onnx_detector import OnnxPlateDetector

_detector_instance: Optional[BasePlateDetector] = None


def get_detector() -> BasePlateDetector:
    """Returns the configured plate detector singleton instance."""
    global _detector_instance
    if _detector_instance is not None:
        return _detector_instance

    backend = settings.DETECTOR_BACKEND.lower()

    if backend == "mock":
        logger.info("Instantiating MockPlateDetector per configuration.")
        _detector_instance = MockPlateDetector()
    elif backend == "onnx":
        logger.info(f"Instantiating OnnxPlateDetector from {settings.ONNX_MODEL_PATH}")
        _detector_instance = OnnxPlateDetector()
        if not _detector_instance.is_ready():
            raise RuntimeError(
                f"Production ONNX model required but failed to load from '{settings.ONNX_MODEL_PATH}'. "
                "Please run scripts/setup_models.py or download weights."
            )
    else:  # 'auto'
        if os.path.exists(settings.ONNX_MODEL_PATH):
            logger.info(f"Found ONNX model at {settings.ONNX_MODEL_PATH}, initializing ONNX detector.")
            onnx_det = OnnxPlateDetector()
            if onnx_det.is_ready():
                _detector_instance = onnx_det
                return _detector_instance
        logger.info("ONNX model file not found in auto mode; falling back to MockPlateDetector for development/demo.")
        _detector_instance = MockPlateDetector()

    return _detector_instance
