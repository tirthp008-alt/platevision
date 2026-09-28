# Evaluation and model choice

[Overview](../README.md) · [Architecture](ARCHITECTURE.md) · [Aggregate data](evaluation-2026-09-28.json)

**Scope: the independent desktop console.** These results do not measure the
root PlateVision web API, Drishti Grid's association accuracy, or city-scale
throughput. Grid's cross-camera end-to-end accuracy remains unmeasured.

## Training and release

The YOLO11n candidate completed 20 additional fine-tuning epochs from an earlier
plate-trained YOLO11n checkpoint. Training used only the supplied ZIP data,
took **24 minutes 20.98 seconds** on CPU, and selected epoch 14 by validation.
OCR and vehicle-category weights were not retrained.

| Split | Images/examples | Plate boxes |
|---|---:|---:|
| Training | 382 | 458 |
| Trainer validation, including context crops | 155 | 189 |
| Whole-image validation used below | 97 | 131 |
| Reserved test | 23 | 28 |

The 58 validation context crops are excluded from the whole-image comparison.
The best checkpoint's trainer mAP50 was **0.8514**; that is a trainer-validation
metric, not exact OCR accuracy or the application recall reported below.

| Role | Model | ONNX SHA-256 |
|---|---|---|
| Recommended | Green YOLOv8n | `658d763e2a603cde6269bb45272e87795d46f206aa18c06655de7f54d41437a5` |
| Candidate | YOLO11n, 20 additional epochs | `906fc0b38309e1d4c55459575c5eaa7eef84c74baf6e21d55f4f289cbff77e1e` |

Install the [versioned release](https://github.com/tirthp008-alt/platevision/releases/tag/v2026.09.28-models)
with the hash-checking [model installer](../scripts/setup_demo_models.py).
No dataset is included in the release.

## Localization

Confidence threshold **0.25**, one-to-one matching at **IoU ≥ 0.5**, identical
whole-image inputs for both models:

| Measure | Recommended green YOLOv8n | YOLO11n candidate |
|---|---:|---:|
| Validation plates matched | 99/131 (75.6%) | 111/131 (84.7%) |
| Green validation plates matched | 34/40 | 40/40 |
| General validation plates matched | 65/91 | 71/91 |
| Green validation unmatched detections | 29 | 15 |
| General validation unmatched detections | 63 | 65 |
| Reserved-test plates matched | 22/28 (78.6%) | 21/28 (75.0%) |
| Reserved-test unmatched detections on reviewed scenes | 10 | 8 |

General annotations are assumed complete. Precision scoring excludes two
incompletely labelled green test scenes. The reserved test has been evaluated
previously; it is not a fresh blind road test.

**Decision: retain green YOLOv8n as the recommended model.** The candidate
improves validation coverage, but adds general-scene unmatched predictions and
regresses on reserved-test coverage and the manual OCR diagnostic. Its additional
matches do not form a superset of the recommended model's matches. The
predeclared replacement rule disallows regression in either validation category.
Overall 95% plate recall has not been achieved.

## Full application OCR diagnostic

Both console APIs used PP-OCRv4, `profile=fast`, OCR and angle correction on,
and experimental emboss and cross-camera ReID off. Requests included the
vehicle detector and bounded vehicle-crop search.

Twenty whole validation images were selected for a manual diagnostic. Twelve
distinct clear registrations were transcribed before inference. Eight occluded,
ambiguous, unreadable, duplicate, or country-unverified targets were excluded
only from exact-text scoring. One target per image was annotated for this check;
other returned detections are **unscored**, not declared false positives.

| Measure | Recommended green YOLOv8n | YOLO11n candidate |
|---|---:|---:|
| Selected target boxes found | 16/20 | 14/20 |
| Raw OCR exact, including detection misses | 6/12 | 5/12 |
| Application-normalized exact, including detection misses | 9/12 | 7/12 |

**Raw exact** ignores case and whitespace only. **Application-normalized exact**
also reflects the backend's punctuation removal and positional O/Q/I-to-digit
repairs. Neither uses the OCR confidence score as measured accuracy. The small
purposive subset is an OCR diagnostic, not representative road accuracy.

## Synthetic load and timing

Hardware: **Intel Core Ultra 5 225H, Intel Arc 130T**, OpenVINO GPU. The 1080p
fixture repeats three source vehicles into ten labelled plate regions; those
regions are not ten independent road observations.

| Workload | Recommended median / p95 | Candidate median / p95 |
|---|---:|---:|
| Detector only, 30 warmed rotating trials | 19.48 / 25.42 ms | 21.44 / 26.95 ms |
| Detector + OCR pipeline, same trials | 150.58 / 165.47 ms | 146.23 / 167.92 ms |
| Full application HTTP, 1 warmup + 5 measured requests | 213.70 / 253.77 ms | 204.27 / 536.19 ms |

Detector + OCR timing excludes vehicle inference/search, angle correction,
HTTP, and browser rendering. Full HTTP timing includes the vehicle and OCR
pipeline plus request/response handling, but excludes browser drawing. Different
stage percentiles must not be added together. Five HTTP samples show timing
jitter, not a proven latency improvement.

Both models returned **10/10 target plate boxes and 8 vehicle detections** per
synthetic image. Raw OCR was exact on **5/10** regions for green YOLOv8n and
**4/10** for YOLO11n; normalized exact was **6/10** and **7/10**, respectively.
The vehicle detector missed two visible target buses. Plate and vehicle counts
have different denominators and do not form a combined accuracy percentage.

## Recording check

Each model processed a four-second, 120-frame synthetic recording with OCR and
angle correction on. Every frame was searched and the exported MP4 decoded all
120 frames at the original dimensions, with HTTP byte-range playback verified.
Both produced three plate tracks and three vehicle tracks.

Processing averaged **59.61 ms/frame** for green YOLOv8n and **59.46 ms/frame**
for YOLO11n. This is one short functional fixture, not sustained throughput or
time-to-first-playback. The bus was classified as a truck and flagged for review;
vehicle-category accuracy remains unresolved.

## Evidence and reproduction limits

The [public aggregate](evaluation-2026-09-28.json) preserves counts, timings,
model hashes, and evaluation scope without registrations, local user paths, or
raw traffic images. Its source report is fingerprinted for traceability.

The local evaluation recorded 356 core/training regression tests and 14
HTTP-evaluator tests passing before publication consolidation. These historical
test counts are not a claim about the current checkout's complete test count;
run the [current checks](SETUP.md#checks) to validate it.

The console's comparison and HTTP-evaluation scripts are retained under
`desktop-console/scripts/`. Full benchmark reproduction requires the original
authorized dataset ZIPs and private manual labels, which are not distributed.
The existing public synthetic fixtures remain under `desktop-console/tests/`
for functional checks. Use new output filenames, verify image and
model hashes, keep model/settings fixed, and run inference without competing
training or GPU workloads.

The next accuracy study should use previously unseen recordings split by
vehicle/capture session, complete plate boxes, verified readable transcripts,
vehicle categories, and cross-camera identity labels. No results in this document
establish Drishti Grid's accepted-link precision or production readiness.
