# Third-party model provenance and notices

This file records source/model declarations, not a new license grant or an
independent assurance about upstream dataset rights. Fine-tuning does not change
the underlying model's license. Keep upstream notices and applicable source
obligations with redistributed artifacts; supplied private dataset images and
annotations are not included in this release.

Full upstream license texts are retained as
[Ultralytics AGPL-3.0](licenses/Ultralytics-AGPL-3.0.txt) and
[FastReID Apache-2.0](licenses/FastReID-Apache-2.0.txt). The latter is copied
from the exact pinned source revision named below.

## Plate detectors

- Recommended green YOLOv8n: derived from the plate export at
  [ml-debi/yolov8-license-plate-detection](https://huggingface.co/ml-debi/yolov8-license-plate-detection).
  Original SHA-256: `85d236280a1301ad98907947d284951dd2b20c23a6786ff50f7e6a8ec515bd50`.
  The repository card labels that source MIT, while its ONNX metadata declares
  **AGPL-3.0**. This release retains the AGPL-3.0 declaration; it does not relabel
  the weights MIT. The published green fine-tune is
  `658d763e2a603cde6269bb45272e87795d46f206aa18c06655de7f54d41437a5`.
- YOLO11n candidate: Ultralytics architecture and training framework, recorded
  **AGPL-3.0** in its training model card. Completed 20 epochs on the supplied
  local ZIP data, with validation-only checkpoint selection; OCR was not trained.
  Export SHA-256: `906fc0b38309e1d4c55459575c5eaa7eef84c74baf6e21d55f4f289cbff77e1e`.
- Framework/source: [Ultralytics](https://github.com/ultralytics/ultralytics).
  The repository retains training/evaluation scripts and a sanitized model card
  containing the parent checkpoint hash, dataset-manifest hash and run settings.

## Vehicle categories

[IISc UVH-26](https://huggingface.co/iisc-aim/UVH-26), pinned source revision
`4a22412775adb6f97f22735647afee976b4638a0`, YOLO11s architecture. The local export's
provenance records **IISc repository Apache-2.0; Ultralytics architecture AGPL-3.0**.
These are recorded upstream declarations, not a claim that architecture and
weight obligations are interchangeable. No vehicle-model training was performed
for this release. Source checkpoint SHA-256:
`12dbeb8f6fa207d19446fa652fc4313359fbec267027c8c01641004dcf1d0150`.
ONNX SHA-256: `91f8aab79100ca910c095a62e3c3a079d01d81c78f6f1b2b47c100f868c2883b`.

## Optional appearance matching

[JDAI-CV FastReID](https://github.com/JDAI-CV/fast-reid), source revision
`c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d`, source license **Apache-2.0**.
The supplied graph exports the official `veri_sbs_R50-ibn.pth` checkpoint trained
on VeRi; it was not retrained here. Source checkpoint SHA-256:
`57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd`.
ONNX SHA-256: `36155fa20d749f3ca95dfa7f99ac1286cdc98ba3874f577b2b9eeab146b3d1d5`.
Its sidecar records strict checkpoint loading and successful export parity.
Published upstream benchmark scores do not measure this application's CCTV
accuracy; appearance matches remain candidates for human review.

## OCR and other software

The console uses the pinned `rapidocr-onnxruntime` package and its bundled
PP-OCRv4 models. It does not redistribute a separately trained OCR checkpoint in
these ZIPs. Preserve the licenses/notices provided by RapidOCR, PaddleOCR,
OpenVINO, ONNX Runtime, OpenCV and the other installed dependencies.
