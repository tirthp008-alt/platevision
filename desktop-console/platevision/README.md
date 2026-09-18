# PlateSight backend and web client

See the workspace root `README.md` for setup, measured latency, accuracy limitations and verification commands.

`docker compose up --build` starts the API and the Next.js wrapper at http://localhost:3000. The API mounts `./models` read-only. Put the plate model at `models/plate-detector.onnx` and vehicle model at `models/vehicle-detector.onnx` before starting. The browser's API address defaults to http://localhost:8000; set it to a reachable HTTPS API when deployed elsewhere.

The model adapter supports YOLOv8/11 raw detection exports with FP32 NCHW input and `[1, 4 + classes, anchors]` output. Plate models use one class; vehicle models use COCO's 80 classes. Use `nms=False`, `batch=1` when exporting. Other output formats fail explicitly.

Training scripts and the existing video-separated dataset split remain available. The downloaded baseline performs poorly on small plates and is not a substitute for camera-specific training and validation.

## Plate-first update (2026-09-16)

The runtime now calls `OnnxPlateDetector.detect_plates`: whole-frame inference plus overlapping image tiles. `refine_vehicles` has been removed. `VEHICLE_CONTEXT_ENABLED=false` is the default, so the vehicle model is not loaded or run. When explicitly enabled, vehicle context is supplementary and its inference errors do not remove plate results.

`PLATE_TILE_SIZE=640`, `PLATE_TILE_OVERLAP=0.25` control coverage. All image boundaries are covered; edge-cut fragments are recovered in neighbouring tiles. Default maximum merged regions: 100. More coverage costs additional inference time; there is no sub-100 ms claim for this mode.


## September 2026 accuracy and recording update

The original plate ONNX weights are unchanged. The active path uses 480 px overlapping tiles, a bounded close-up pass on weak plate candidates, confidence-aware merging, and contextual/tight OCR crops. Uploaded MP4/WebM recordings use the same model at selected timestamps with pause/resume, temporal OCR voting and a recording register. See the root README for configuration, usage, measured before/after results and limitations. Historical descriptions above are superseded by this update.
