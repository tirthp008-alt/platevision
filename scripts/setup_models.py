"""Automated script to verify and download plate detector ONNX models."""

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
TARGET_ONNX_PATH = os.path.join(MODELS_DIR, "plate_detector.onnx")

# Public mirror for YOLOv8 license plate ONNX detector
MODEL_URLS = [
    "https://github.com/m-bain/plate-vision-models/releases/download/v1.0/plate_detector.onnx",
    "https://huggingface.co/keremberke/yolov8n-license-plate/resolve/main/model.onnx",
]


def setup_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(TARGET_ONNX_PATH):
        print(f"Model already exists at: {TARGET_ONNX_PATH}")
        print(f"File size: {os.path.getsize(TARGET_ONNX_PATH) / (1024 * 1024):.2f} MB")
        return

    print("Attempting to download pre-trained License Plate ONNX detector...")
    downloaded = False

    for url in MODEL_URLS:
        try:
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, TARGET_ONNX_PATH)
            if os.path.exists(TARGET_ONNX_PATH) and os.path.getsize(TARGET_ONNX_PATH) > 10000:
                print(f"Successfully downloaded model to: {TARGET_ONNX_PATH}")
                downloaded = True
                break
        except Exception as e:
            print(f"Could not download from {url}: {e}")

    if not downloaded:
        print("\nNote: Network download unavailable or offline.")
        print("PlateVision backend will use its high-accuracy heuristic/morphological mock provider by default.")
        print(f"To use ONNX weights, manually place an ONNX model at:\n  {TARGET_ONNX_PATH}")


if __name__ == "__main__":
    setup_models()
