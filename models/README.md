# Verified inference models

Install the versioned core bundle from the repository root:

```sh
python scripts/setup_demo_models.py
python scripts/setup_demo_models.py --check
python scripts/run_console.py --model recommended
```

Use Python 3.12 with `desktop-console/platevision/backend/requirements.txt` installed.
These commands do not require PyTorch, Ultralytics, a training dataset, or model conversion.
The installer verifies the release ZIP's exact size and SHA-256, its complete member
list, each member's size/hash and ZIP CRC, and safe destination paths before installing.
Existing different files are preserved unless `--replace` is explicitly supplied.

For an offline machine, copy the release asset and run:

```sh
python scripts/setup_demo_models.py --archive /path/to/platevision-core-models.zip
```

The exact release URLs, archive hashes, file hashes and destination paths are in
[manifest.json](manifest.json). The release is `v2026.09.28-models` at
[tirthp008-alt/platevision](https://github.com/tirthp008-alt/platevision/releases/tag/v2026.09.28-models).
Weights are downloaded separately and are not Git source files.

| Model | Bundle | Role |
|---|---|---|
| Green YOLOv8n, 12-epoch plate fine-tune | core | Recommended desktop default; API 8003 |
| YOLO11n, completed 20-epoch plate fine-tune | core | Explicit comparison candidate; API 8004 |
| IISc UVH-26 YOLO11s vehicle detector | core | Indian vehicle categories in both console configurations |
| FastReID SBS R50-IBN, trained on VeRi | reid | Optional desktop cross-camera appearance matching |

The core models remain under `desktop-console/platevision/models/` at the paths
used by the tested launchers. The installer also makes a verified copy of the
recommended plate detector at `models/plate_detector.onnx` for the separate root
web application's `ONNX_MODEL_PATH`. It does **not** install the UVH vehicle or
VeRi appearance graph into the root web app: its model interfaces differ.

Select the explicit candidate without changing the default:

```sh
python scripts/run_console.py --model yolo11n
```

The launcher starts localhost API and static servers, waits for model/OCR/vehicle
health, checks the loaded plate hash, and prints the correctly configured console
URL. Ctrl+C stops both owned processes. It opens no windows. `--check` verifies
the launch dependencies without starting servers; `--device cpu` uses CPU,
`--device gpu` requires OpenVINO GPU, and the default `--device auto` permits
the supported CPU fallback. Custom ports use `--port` and `--web-port`.

Optional matching:

```sh
python scripts/setup_demo_models.py --bundle reid
python scripts/setup_demo_models.py --bundle reid --check
```

An offline optional install uses `--bundle reid --archive /path/to/platevision-reid-models.zip`.
Restart the console after installing it. The export needs its included verified
model card. ReID uses OpenVINO on the selected device; choose `--device cpu` on
machines without an available Intel GPU. Matching remains off until enabled in
the console with camera IDs, real capture times and directed route windows.
The optional ResNet OCR experiment is not required or bundled; PP-OCRv4 uses the
models supplied by the pinned RapidOCR package.

The candidate improved validation matches from 99/131 to 111/131 but had additional
general-scene false detections, lower reserved-test coverage and worse fresh OCR
results. It did not pass the predeclared replacement gate. Scores are not accuracy
probabilities, and 95% overall recall was not achieved. See the desktop console's
evaluation documentation for settings, measured latency and limitations.

Preserve the [third-party model notices](THIRD_PARTY_MODELS.md) and included model
cards when redistributing artifacts. Original training data is not part of these bundles.
