# PlateVision desktop console

The independent plate/OCR console complements [Drishti Grid](../README.md).
It analyses photos and recordings, preserves plate crops and uncertain readings,
shows vehicle-category evidence, and exports annotated MP4. Its API and client
are separate from the repository's root web application.

[Install and run](../docs/SETUP.md#platevision-desktop-console) ·
[Demo walkthrough](../docs/DEMO.md#2-desktop-console-inspect-evidence-and-compare-models) ·
[Measured results](../docs/EVALUATION.md)

## Start

Create and activate a separate Python 3.12 environment as described in the
setup guide. Run these commands from the **repository root**:

```bash
python -m pip install -r desktop-console/platevision/backend/requirements.txt
python scripts/setup_demo_models.py --bundle core
python scripts/run_console.py --model recommended
```

The launcher prints the console URL and starts the recommended API on port 8003
with the static client on port 8080. Ctrl+C stops its services. To try the trained
candidate, stop it and run `python scripts/run_console.py --model yolo11n`;
the candidate API defaults to port 8004.

If changing the client address manually, use **Model API address**. Keep OCR and
angle correction on for the current workflow. PP-OCRv4 is the default reader;
experimental emboss and appearance matching are optional.

## What to inspect

- **Plate geometry:** every returned region stays visible even when text is
  unreadable. Additional vehicle-crop proposals are marked for review.
- **OCR evidence:** original and eligible corrected crops, bounded retry
  attempts, selected reading, and conflict flags. Raw text and normalized text
  are different evidence; normalization can repair characters.
- **Counts:** plate regions/tracks, grouped registration readings, unresolved
  entries, and vehicles are separate quantities. They are not accuracy scores.
- **Recordings:** every decoded frame is searched; inspect local tracks and
  download the annotated MP4 or JSON report after processing.

The YOLO11n candidate improves whole-image validation matches from 99/131 to
111/131, but lowers reserved-test matches from 22/28 to 21/28 and manual raw
OCR exact from 6/12 to 5/12. Green YOLOv8n therefore remains recommended.
The [evaluation report](../docs/EVALUATION.md) gives workload-specific timings,
model hashes, denominators, and known limitations.

## Source layout

| Path | Purpose |
|---|---|
| `index.html`, `app.js`, `styles.css` | Canonical console interface |
| `video-analysis.js`, `video-export.js`, `reid-ui.js` | Recording and optional cross-camera client modules |
| `platevision/backend/` | Independent FastAPI inference and recording API |
| `scripts/`, `tests/` | Evaluation tools and regression checks |

Use the root installer and launcher for normal operation. Historical setup and
evaluation tools may require private datasets or old fixtures; those inputs are
not distributed. See the [current checks](../docs/SETUP.md#checks).

The console is a local prototype, not Grid's authenticated service. Its measured
results do not validate Grid's camera association, and its track IDs are not
verified real-world identities.
