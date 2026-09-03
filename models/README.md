# PlateVision Model Specifications & Weight Setup

This directory contains the machine learning models and ONNX execution graphs for vehicle license plate detection.

---

## 1. Supported Model Architectures

PlateVision uses **YOLOv8** / **YOLOv9** or **YOLO-NAS** object detection models optimized and exported to **ONNX format**.

- **Default Model**: `models/plate_detector.onnx`
- **Input Size**: `640 x 640 x 3` (RGB normalized `0.0 - 1.0`)
- **Output Format**: Shape `(1, 5, 8400)` or `(1, 8400, 5)` representing `[cx, cy, w, h, class_confidence]`
- **Classes**: `0` -> `License_Plate`

---

## 2. Automatic Setup Script

To automatically download the pre-trained weights or set up sample weights, run:

```bash
python scripts/setup_models.py
```

This script will verify the target directory, check connectivity, and place `plate_detector.onnx` directly into the `models/` directory.

---

## 3. Manual Installation

If downloading manually:
1. Download a YOLOv8 license plate detection ONNX weight file (e.g. from Ultralytics or HuggingFace ANPR models).
2. Save the file to:
   ```
   models/plate_detector.onnx
   ```
3. Set the environment variable in `.env`:
   ```bash
   ONNX_MODEL_PATH="models/plate_detector.onnx"
   DETECTOR_BACKEND="onnx"  # Or "auto"
   ```

---

## 4. Fallback / Mock Mode

If no ONNX model is detected, PlateVision automatically operates in **Auto / Mock Mode** (`DETECTOR_BACKEND="auto"` or `"mock"`).
In this mode, morphological edge filtering and plate localization heuristics run seamlessly, enabling full testing, development, and offline UI verification without requiring multi-megabyte weights.
