## Single-photo fast path — 1080p / ten plates (2026-09-19)

Implemented a GPU full-scene detector with a static 1536 x 864 input (portrait equivalent), unchanged learned weights, batch GPU OCR, and on-device CTC argmax/probability reduction. Original-resolution crops go to OCR. All returned candidates are processed; ten is a benchmark load, not a cutoff. There is no image-result cache or vehicle-first gate. CPU/tiled fallback remains available. Static model initialization is warmed at startup; it is not included in per-request times.

`POST /api/detect/image` and `/frame` accept `profile=fast` (default) or `profile=detailed`. Fast retries uncertain single-line crops once in a batch and exposes uncertain text; detailed retains full text-localization retries and overlapping tiled detection. Stacked plates still use the slower full OCR reader in fast mode. Unsupported model exports fall back to tiled search instead of guessing anchor layouts. OpenCV decoding preserves EXIF orientation.

The interface shows separate detection and OCR confidence, fast/detailed choices, actual client elapsed time and backend time. Scores are uncalibrated model outputs, not accuracy probabilities. Original image bytes avoid unnecessary browser PNG encoding; plate previews render directly to canvases. Client elapsed includes request preparation, network, OCR, and synchronous result rendering, not a guaranteed display-paint deadline.

### Measured limits

Intel Arc 130T / Core Ultra 5 225H, synthetic 1920 x 1080 collage with ten plates, 30 warmed HTTP requests:

| Stage | Median | p95 |
|---|---:|---:|
| Complete local HTTP result | 39.12 ms | 44.25 ms |
| Backend decode + detection + OCR + crops | 35.31 ms | 40.44 ms |
| Detection | 13.03 ms | 15.74 ms |
| OCR | 14.34 ms | 16.33 ms |

All ten target regions were returned; nine texts were exact and one O/0 ambiguity remained. This is a synthetic load test using repeated source vehicles, **not an independent road accuracy test**. Timing stages have separate percentiles and cannot be added as percentiles. First request is stored separately. The warmed browser check showed 46 ms (36 ms server); its first request took 357 ms during page initialization. Browser elapsed is higher than HTTP inference; a strict <40 ms complete visible result is **not achieved**.

On 50 existing development frames / 84 labelled plates, fast mode matched 45, missed 39, and returned four unmatched boxes: precision 91.8%, recall 53.6%, backend median 33.85 ms / p95 116.93 ms. The 1920-wide model had recall 54.8% and p95 138.28 ms; it did not solve small-plate recall. The older detailed pipeline matched 53/84 (historical comparison below). These frames have been used during development and are not an independent test set. The fast mode trades coverage for speed; it is not production-ready all-plate accuracy.

Evidence: `tests/ten-plate-http-benchmark.json`, `tests/fast-image-development.json`, and `tests/image-before-40ms.json`. Reproduce the direct model load test with `scripts/benchmark_ten_plates.py`, and development comparison with `scripts/evaluate_fast_images.py`. All 36 backend tests, six scheduler tests, and the frontend build passed. Annotated video export still decodes all output frames and supports byte-range playback.

### Missing inputs for the next improvement

1. Original 1080p clips from the fixed camera, with every visible plate boxed and exact text transcribed where legible, across day/night, near/far, and moving traffic. Split by recording/vehicle, not adjacent frames, to avoid evaluation leakage.
2. Complete annotations for white, yellow and green plates. The supplied green-plate dataset omits other visible plates; training on those incomplete labels would teach the model to ignore them. No retraining was claimed or performed here.
3. A smaller detector trained for this camera and an Indian-plate OCR recognizer, including stacked and angled plates. Validate on held-out recordings, with both recall and exact-text accuracy; calibrate confidence on those labels.
4. Camera placement/exposure that produces enough readable plate pixels and limits motion blur. Measure sustained p95/p99 latency under the actual ten-plate workload and concurrent video load. A user-defined road region could reduce work but must not silently exclude plates from a full-scene claim.

# PlateSight

A local plate-recognition console for images, uploaded recordings and browser-accessible cameras. The root client is shared by both web projects through `scripts/sync-client.ps1`.

## Detection and recognition

The existing `plate-detector.onnx` weights are unchanged (SHA256 `85d236280a1301ad98907947d284951dd2b20c23a6786ff50f7e6a8ec515bd50`). The improved pipeline:

1. Searches the complete frame and overlapping 480 px tiles for plates, independently of vehicles.
2. Rechecks up to 12 small, weak plate proposals in enlarged views using the same detector.
3. Merges duplicate boxes by confidence, preserving complete stacked plates when confidence is similar. Large weak boxes cannot swallow neighbouring strong plates.
4. Reads both contextual and tight plate crops when needed. OCR groups sloping text into lines, removes small country marks/manufacturer words, and retries uncertain crops with contrast enhancement.
5. Requires text evidence for weak detections. Strong detections remain visible even when unreadable. When OCR is unavailable, candidates remain visible with a warning.

Vehicle detection is disabled by default and is never required to find plates. The API is the default engine. Green boxes have a format-matching reading above the OCR confidence threshold; amber boxes require review. Format matching and OCR confidence do not prove that a registration is correct.

## Video: accelerated detection and annotated MP4

Upload a recording through **Choose your source**, with **Model API** selected. The server receives the file once, searches **every decoded frame** for multiple plates, reads the clearest crops per track, and exports a playable H.264 MP4 with boxes and readings burned into the pixels. Use **Download MP4** or **Detection report** after processing. Optional source audio is re-encoded to AAC. Pause/resume preserves processing; Reset cancels it.

- **Fast** (default): fit within 1280 px on the longest side, overlapping 480 px plate searches, no extra proposal-refinement passes. No frames are skipped.
- **Detailed**: fit within 1920 px, including additional weak-proposal refinement. Slower, and extra resolution does not guarantee better recognition.
- OpenVINO runs the unchanged ONNX weights on the Intel GPU. Colour conversion and normalization are fused into the device graph, avoiding repeated CPU tensor copies. `ACCELERATION=auto` falls back to ONNX CPU when unavailable; use `openvino` to require the accelerator, or `onnx` to force CPU. `/api/health` reports the actual runtime.
- OCR runs on at most two best observations per track. Exported text is retrospective: a later, clearer reading can label earlier frames of the same track. No registration is synthesized.
- The console reports **overall processing time / decoded frame**, including OCR and export, and **detector p95** separately. User pause time and file upload/download time are excluded from the processing average. The benchmark also records upload/poll/download wall time.
- Inputs are limited to 512 MB and 15 minutes, with bounded tracks and observations. Temporary uploaded files are deleted after processing. Results expire after one hour and require the same server session.
- Browser mode retains its sampled-frame workflow. Live-camera `/api/detect/frame` still includes synchronous OCR and has no sub-50 ms guarantee. Direct RTSP requires a gateway.

### Measured video performance on this machine

Intel Core Ultra 5 225H + Intel Arc 130T, OpenVINO GPU, same plate weights. A four-second, 30 fps **synthetic three-plate motion fixture** was processed at 1280 × 372:

| Measurement | Run 1 | Run 2 |
|---|---:|---:|
| Full-scene detector p95 | 27.67 ms | 32.69 ms |
| Decode + detect + track p95 | 32.82 ms | 38.36 ms |
| Overall processing average, including OCR + MP4 export | 36.39 ms/frame | 47.58 ms/frame |
| Plate regions found | 3 in all 120 frames | 3 in all 120 frames |

`tests/video-export-30fps-benchmark.json` contains the complete measurements. The browser run also measured 34.4 ms/frame. This meets the 50 ms **average processing** target for this fixture, not a guarantee for every frame or CCTV recording. One OCR reading remains uncertain. The shorter 20-frame fixture averaged 59–86 ms/frame because OCR and export setup account for more of its total time. Taller, crowded scenes require more tiles; the attached portrait photos took approximately 57–77 ms for fast detection alone.

The output is available only after the recording is processed. A four-second input took 4.37–5.71 seconds to finish in the two API runs; average time per frame is not time-to-first-playback.

### Attached green-plate photos

`tests/attached-green-plates.json` evaluates the 19 supplied photographs against their existing VOC annotations. Fast mode matched **12/19 annotated green plates** at IoU ≥ 0.5; detailed mode matched 11/19. Several angled, blurred or sideways plates remain missed. These are box-recall diagnostics, not exact-text accuracy or independent test results.

The archive contains 209 annotations, but they label green plates only. White/yellow plates in the same scenes are often unlabelled, so treating this as a complete all-plate training dataset would teach false negatives. The model was **not retrained**. Further training should complete all plate annotations, add exact OCR text, and split by vehicle/capture session so near-duplicate images cannot leak between training and validation.

## Run locally (Windows)

The project's `.venv` and model files are already present in this workspace.

Terminal 1, from the project root:

```powershell
Set-Location platevision/backend
../../.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2, from the project root:

```powershell
./.venv/Scripts/python.exe -m http.server 8080 --bind 127.0.0.1
```

Open http://127.0.0.1:8080/?engine=api&v=8. Browser heuristics are available explicitly with `?engine=browser`.

For a clean installation use Python 3.12, create a virtual environment, install `platevision/backend/requirements.txt`, copy `platevision/.env.example` to `platevision/backend/.env`, and obtain the model described in `platevision/models/README.md`. API uploads preserve up to 1920 px width; unchanged still images retain their original bytes and resized images/video frames use high-quality JPEG. OpenVINO GPU is selected automatically when available; CPU fallback uses four inference threads.

## Earlier image accuracy diagnostics

On the same 50 development images containing 84 annotated plates, with IoU >= 0.5 and the actual API output boxes:

| Measurement | Previous pipeline | Improved pipeline |
|---|---:|---:|
| Matched plates | 40 | 53 |
| Unmatched detections | 54 | 15 |
| Missed labelled plates | 44 | 31 |
| Precision | 42.6% | 77.9% |
| Recall | 47.6% | 63.1% |

These are development diagnostics, not independent final accuracy. Only 9 of 29 multi-plate scenes have every annotated plate matched. The comparison measures boxes, not exact text accuracy. Both sides include their actual API crop padding; older raw-box-only reports are not directly comparable.

- `tests/accuracy-comparison.json`: reproducible before/after comparison using cached model predictions.
- `tests/accuracy-current.json`: historical CPU image-pipeline run before the video acceleration work; confirms the comparison. Median processing 1,956 ms, p95 2,520 ms. These are not the new GPU video timings.
- `tests/plate-first-regression.json`: original scooter photo returns two regions/two readings without the earlier extra candidate; the stacked reading is `GJ27DS4837`. The other reading remains ambiguous. The private image is not copied into the repository.
- `tests/video-regression.json`: the three-plate composite recording returns three plate regions in all nine sampled frames. This is a functional video fixture, not a real CCTV accuracy benchmark.

Camera-specific training and an independent dataset with exact plate-text labels are still needed for production accuracy. The API is a localhost development service without authentication. No official government affiliation is claimed.

## Checks

```powershell
$env:PYTHONPATH = "$PWD/platevision/backend"
./.venv/Scripts/python.exe -m pytest platevision/backend/tests -q -p no:cacheprovider
node tests/video-analysis.test.cjs
node --check app.js
./.venv/Scripts/python.exe scripts/evaluate.py
./.venv/Scripts/python.exe scripts/verify_plate_first.py
./.venv/Scripts/python.exe scripts/verify_video.py
./.venv/Scripts/python.exe scripts/benchmark_video_export.py --video tests/multi-plate-30fps.mp4 --output tests/video-export-30fps-benchmark.json --runs 2
```

There are 36 backend tests and six browser-scheduler/temporal-reading tests. The image regression script accepts an optional path to the original scooter photo. Browser checks cover upload, multi-plate output, scheduled frames, pause/resume and timestamped results.

Run `./scripts/sync-client.ps1` after client edits. Run `npm run build` inside `plate-sight-live` for the production build. A static deployment needs a reachable configured API for real inference.

The video benchmark checks the real upload/job/download API, all output frames, three persistent plate tracks, and HTTP byte-range playback. Browser verification covers upload, annotated playback, download links and per-track crops. Backend tests cover tracking, pause/resume, cancellation, damaged media, concurrent uploads, unready downloads and byte-range responses.
