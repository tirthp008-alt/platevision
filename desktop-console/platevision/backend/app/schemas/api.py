from pydantic import BaseModel
from typing import Literal

class Box(BaseModel): x: int; y: int; width: int; height: int
class NormalizedBox(BaseModel): x: float; y: float; width: float; height: float
class Detection(BaseModel):
    id: str; bounding_box: Box; normalized_box: NormalizedBox; detection_confidence: float
    raw_text: str; normalized_text: str; formatted_text: str; ocr_confidence: float
    format_status: Literal["valid", "possible", "uncertain"]; crop_url: str
class DetectResponse(BaseModel):
    request_id: str; processing_time_ms: int; image: dict; detections: list[Detection]; warnings: list[str]
