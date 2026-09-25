# Model sources and configuration

Downloaded for local evaluation on 2026-09-15:

| Local file | Source | SHA256 |
|---|---|---|
| plate-detector.onnx | https://huggingface.co/ml-debi/yolov8-license-plate-detection/resolve/main/best.onnx | 85D236280A1301AD98907947D284951DD2B20C23A6786FF50F7E6A8EC515BD50 |
| vehicle-detector.onnx | https://huggingface.co/webnn/yolo11n/resolve/main/onnx/yolo11n.onnx | 7D8FD1717D9D5BBAB6986CD134AFB620649C7A394303D55B1E09FC00804CC5C1 |

The plate repository declares MIT but has an empty model description; the ONNX metadata itself declares AGPL-3.0, so do not assume the weight file is MIT-licensed. The vehicle repository declares AGPL-3.0. Preserve applicable source/model licence obligations when redistributing. These are evaluation weights, not a validated Indian CCTV model.

Set `MODEL_PATH` and `VEHICLE_MODEL_PATH` in `backend/.env`. Set `ONNX_PROVIDER` to an installed ONNX execution provider; unavailable providers fail visibly. The default is CPU with two inference threads per detector. Sessions warm up once at startup. The optional `BASELINE_MODEL=false` setting removes the baseline notice only after a replacement model has been independently validated.

The adapter in `backend/app/services/detector.py` supports single-class plate and COCO-80 vehicle YOLOv8/11 FP32 ONNX detection exports, `batch=1`, `nms=False`. It letterboxes the image, restores source coordinates, filters scores and suppresses duplicates. Vehicle suppression is class-agnostic across car/motorcycle/bus/truck to avoid counting one vehicle twice.

OCR uses rapidocr-onnxruntime 1.4.4 and its bundled models. Single-line crops bypass text detection; compact/two-line crops use bounded text detection. See https://rapidai.github.io/RapidOCRDocs/v1.4.4/install_usage/api/RapidOCR/ and https://docs.ultralytics.com/models/yolo11/ for the source APIs.

The selected detector is unchanged after evaluating YOLO11 and YOLOv9 alternatives. Vehicle crop refinement and preserving source image detail fix the reproduced miss; see the root README for measured recovery and false-positive counts.

## Active search strategy

The selected plate model now searches the full frame and overlapping tiles directly; no vehicle boxes are used to choose search areas. Vehicle context is disabled by default. The older vehicle-refinement behavior above describes the previous implementation and is superseded by this strategy.


## September 2026 accuracy and recording update

The original plate ONNX weights are unchanged. The active path uses 480 px overlapping tiles, a bounded close-up pass on weak plate candidates, confidence-aware merging, and contextual/tight OCR crops. Uploaded MP4/WebM recordings use the same model at selected timestamps with pause/resume, temporal OCR voting and a recording register. See the root README for configuration, usage, measured before/after results and limitations. Historical descriptions above are superseded by this update.
