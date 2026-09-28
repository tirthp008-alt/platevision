# Judge walkthrough

[Setup](SETUP.md) · [Architecture](ARCHITECTURE.md) · [Evaluation](EVALUATION.md)

Use the Grid walkthrough to demonstrate camera relationships. Use the separate console to inspect the evaluated plate/OCR models. Neither demonstration is a claim of independent road accuracy.

## 1. Drishti Grid: from sightings to a candidate journey

Start the web application using [Setup](SETUP.md#drishti-grid-web-application). Open `http://localhost:3000/grid` and enter the local demo administrator key in **Set access key**.

Create the five-camera synthetic corridor. This registers demo cameras in the local database and starts their processing workers.

macOS/Linux:

```bash
curl -X POST http://127.0.0.1:8000/api/grid/demo/setup \
  -H 'X-API-Key: admin-demo' -H 'Content-Type: application/json' \
  -d '{"start_processing":true}'
```

Windows PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/grid/demo/setup -Headers @{'X-API-Key'='admin-demo'} -ContentType 'application/json' -Body '{"start_processing":true}'
```

1. **Show the camera map and feed status.** Demo cameras are labelled `SYNTHETIC DEMO`. They render traffic scenes; detection, OCR, tracking, and fusion process those pixels through the configured backends.
2. **Inspect recent sightings.** Point out the local track ID, capture time, available plate evidence, and vehicle appearance. Check system status so model-backed and fallback runs are identified accurately.
3. **Search an observed reading.** Use a plate actually returned by the run, rather than assuming every rendered registration was read correctly. The demo fleet includes `GJ01AB1234`; that is an input example, not a guaranteed OCR output.
4. **Review the map and timeline.** Explain which sightings are observed, which connecting movements are inferred, and which associations remain unresolved. Inspect evidence for a transition instead of presenting the displayed route as a verified identity.
5. **Show a role change.** Clear the access key to use the viewer role and demonstrate masked plate text. Audit access and fusion control are reserved for an administrator.

Fusion runs in the background. An administrator can explicitly recompute it after sightings arrive with `POST /api/grid/fusion/recompute` and an empty JSON body. Do not lower thresholds merely to force a route during a demonstration. If the run yields no accepted journey, use its evidence and unresolved links to explain the result.

### Explain one ambiguous match

Suppose Camera A sees a white hatchback with an unreadable plate, and Camera B sees a similar vehicle three minutes later. Grid compares appearance and any usable plate evidence, then checks time against the configured road model. The best candidate must pass both a score threshold and a margin over the next candidate. Appearance similarity alone is not proof that the vehicles match.

The current identity model builds a monotonic path across distinct cameras. It does not model a vehicle returning to the same camera during a single journey. When no router is configured, travel time uses a labelled geographic approximation.

### Traffic panel

Show segment flow, estimated speed, congestion indicators, and the heatmap. These derive from processed sightings and inferred links. Congestion uses a baseline capacity assumption; origin–destination matrices and watchlist alerts are not implemented.

## 2. Desktop console: inspect evidence and compare models

Install the core bundle and start the console as described in [Setup](SETUP.md#platevision-desktop-console). Open the launcher URL.

For a repeatable functional check, use the existing synthetic [ten-plate image](../desktop-console/tests/ten-plate-1080p.jpg) or [short recording](../desktop-console/tests/multi-plate-30fps.mp4). These repeat source vehicles and do not represent independent road samples.

1. Upload an image you are authorized to process. Keep **Read plate text** and **Plate angle correction** on; PP-OCRv4 is the default reader.
2. Inspect plate boxes independently of vehicle boxes. A vehicle with no associated plate may receive a bounded extra crop search. Those added candidates stay marked for localization review.
3. Compare the original crop, any corrected view, OCR attempts, and review status. An unreadable plate region remains visible. A valid format or high model score does not establish that its characters are correct.
4. Upload an authorized short recording. After processing, inspect plate and vehicle tracks, grouped readings, and unresolved entries. Play or download the annotated MP4 and detection report.
5. Stop the console and restart with `--model yolo11n` to try the candidate. Use the same inputs and settings. Explain why the recommended model remains green YOLOv8n using the [published comparison](EVALUATION.md).

Plate regions, vehicle tracks, and grouped registration strings count different things. Simultaneous tracks sharing a reading remain ambiguous; grouping does not establish a real-world vehicle identity.

## Suggested presentation order

| Segment | Show | Explain |
|---|---|---|
| Problem | Two sightings at different cameras | A reading or appearance match alone can be misleading |
| Grid workflow | Map, camera states, candidate journey | Plate, appearance, time, and travel evidence are combined |
| Review | Unresolved transition and viewer masking | Uncertainty and access controls are visible |
| Console | Crop evidence and annotated recording | Detection, OCR, and vehicle categories are separate outputs |
| Evaluation | Validation/test and raw/normalized OCR table | Improvements and regressions both influence model choice |

This walkthrough does not require private road imagery. Use the Grid's labelled synthetic scenes, architectural diagrams, and aggregate evaluation tables for public presentations. Review screenshots for registrations, source URLs, credentials, and local file paths before sharing them.
