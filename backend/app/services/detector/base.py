"""Base plate detector provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np

from app.schemas.detection import BoundingBox


@dataclass
class RawDetection:
    bbox: BoundingBox
    confidence: float
    class_id: int = 0


class BasePlateDetector(ABC):
    """Abstract Base Class for License Plate Detection Providers."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Returns True if the detector model is loaded and ready for inference."""
        pass

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> List[RawDetection]:
        """Runs plate detection on an input OpenCV BGR image and returns detected bounding boxes."""
        pass
