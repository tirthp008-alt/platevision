"""Integration tests for PlateVision FastAPI endpoints."""

import io
import cv2
import numpy as np
from PIL import Image
import pytest
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def generate_test_vehicle_image() -> bytes:
    """Generates a test image with a simulated license plate region."""
    # 600x800 background
    img = np.full((600, 800, 3), 40, dtype=np.uint8)

    # Car body rectangle
    cv2.rectangle(img, (100, 150), (700, 500), (80, 80, 80), -1)

    # License plate rectangle (White background)
    cv2.rectangle(img, (260, 360), (540, 430), (255, 255, 255), -1)
    cv2.rectangle(img, (260, 360), (540, 430), (0, 0, 0), 2)

    # Plate text
    cv2.putText(
        img,
        "GJ 01 AB 1234",
        (280, 410),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    success, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["detector_ready"] is True
    assert data["ocr_ready"] is True


def test_detect_image_upload():
    img_bytes = generate_test_vehicle_image()
    files = {"image": ("test_car.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    response = client.post("/api/detect/image", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "detections" in data
    assert len(data["detections"]) > 0

    first_det = data["detections"][0]
    assert "id" in first_det
    assert "bounding_box" in first_det
    assert "normalized_box" in first_det
    assert "detection_confidence" in first_det
    assert "raw_text" in first_det
    assert "normalized_text" in first_det
    assert "format_status" in first_det
    assert "crop_url" in first_det
    assert first_det["crop_base64"] is not None

    # Test crop retrieval
    crop_url = first_det["crop_url"]
    crop_resp = client.get(crop_url)
    assert crop_resp.status_code == 200
    assert crop_resp.headers["content-type"].startswith("image/")


def test_detect_frame_endpoint():
    import base64

    img_bytes = generate_test_vehicle_image()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    payload = {"frame_base64": f"data:image/jpeg;base64,{b64_str}"}

    response = client.post("/api/detect/frame", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["detections"]) > 0


def test_invalid_image_upload():
    files = {"image": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/api/detect/image", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_IMAGE"


def test_nonexistent_crop_404():
    response = client.get("/api/results/fake-req/fake-det/crop")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "CROP_NOT_FOUND"
