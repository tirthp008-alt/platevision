# Drishti Grid — City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking and Urban Traffic Analytics

**Problem Statement 26SIH127 (SIH26127) · Smart India Hackathon 2026**
Organization: Bharat Electronics Limited (BEL) · Theme: Transportation & Logistics · Category: Software

Drishti Grid is a working end-to-end prototype of a centralized, city-wide ANPR
platform. It ingests multi-camera feeds, recognizes Indian number plates, tracks
vehicles locally, associates the same vehicle across geographically distributed
cameras using **plate + visual Re-ID + temporal + road-network** evidence, and
reconstructs the vehicle's route on an interactive GIS map — with explicit
abstention when the evidence is insufficient.

It is built on **PlateVision**, the repository's existing production ANPR
detector/OCR/cropper stack, which remains fully intact and independently usable.

> **Intended use.** Authorised, synthetic, or demo traffic footage only. The
> system enforces role-based access control, masks plates for low-privilege
> roles, audit-logs every search, transmits no raw video to the browser, and
> reports "unresolved" rather than guessing.

---

## Alignment with the official problem statement

The statement defines three core deliverables plus an expected alert system.
This table records exactly what is implemented in this repository.

| # | Statement requirement | Status in this repository |
|---|---|---|
| 1 | **High-accuracy ANPR & OCR engine** (>90% in adverse conditions) | Implemented end-to-end: plate localization → PlateVision OCR → multi-frame voting → position-aware Indian-plate normalization (standard + Bharat Series). Runnable with the repo's ONNX detector or documented mock/classical backends. **Accuracy is not yet benchmarked** — the 90% figure is a target, not a measured result (see *Honest status*). |
| 2 | **Single-plate trajectory tracking** on a GIS map with timestamps, direction, route | Implemented: `/grid` dashboard, plate search, GIS route with OBSERVED / INFERRED / UNRESOLVED states, per-transition evidence, timestamps, directional arrows, visual timeline. No fabricated routes. |
| 3 | **Macro traffic flow & movement analytics** — density, congestion, heatmaps | Implemented: segment speed, flow rate, congestion ratio, camera-node heatmap on the map. |
| 3b | Origin–destination patterns | Not implemented — see *Roadmap*. |
| 4 | **Alert system** for blacklisted vehicles & route anomalies | Not implemented — see *Roadmap*. |

### Honest status

| Claim | Reality |
|---|---|
| Plate accuracy > 90% | **Not measured.** No benchmark has been run against a labelled dataset in this repo. The pipeline is real and runnable; the accuracy target requires evaluation on Indian ANPR datasets. |
| Cross-camera tracking | Fully implemented and exercised on a **synthetic 5-camera demo corridor** driving the real detection/OCR/tracking/fusion code. A production city deployment requires real camera infrastructure. |
| Road-network travel time | Uses OpenRouteService or an `osmnx` graph when configured; otherwise a **clearly labelled geographic fallback** (34 km/h, 1.35 detour factor). |
| Detector / OCR / Re-ID weights | `models/*.onnx` are **not committed**. Without them the system stays runnable via mock/classical backends and says so in `GET /api/grid/system/status`. Demo camera scenes are synthetic; their outputs are genuine outputs of the real pipeline, never fabricated. |

---

## Architecture

```
CAMERA SOURCES (USB webcam / local device / RTSP / file / synthetic demo)
        |
        v
VIDEO INGESTION .......... per-camera worker threads, isolated failures
        |
        v
VEHICLE DETECTION ........ motion | plate-anchored | ONNX | hybrid
        |
        +-----------------------------+
        |                             |
        v                             v
LOCAL TRACKING                 PLATE LOCALIZATION
(ByteTrack-style IoU tracker,         |
 persistent per-camera track IDs)     v
        |                      PLATEVISION OCR
        |                             |
        |                      MULTI-FRAME OCR VOTING
        |                             |
        |                      PLATE CONFIDENCE
        |                             |
        v                             v
VEHICLE CROP  ->  VISUAL EMBEDDING (Re-ID)
        |
        v
SIGHTING EVENT
        |
        v
EVENT / DATABASE LAYER ... in-process bus or Kafka; SQLAlchemy + SQLite/PostGIS
        |
        v
CROSS-CAMERA FUSION ENGINE
        |
        +--------------+---------------+
        |              |               |
        v              v               v
   Plate evidence  Visual Re-ID   Temporal + road graph
        |              |               |
        +--------------+---------------+
                       |
                       v
              Candidate score S_ij
                       |
                       v
              Confidence gate (tau, delta)
                       |
              +--------+--------+
              |                 |
           ACCEPT            ABSTAIN
              |                 |
              v                 v
       Trajectory edge    Unresolved link
              |
              v
       TRAJECTORY GRAPH  ->  GIS DASHBOARD
```

Every stage is a separate module, so a detector, OCR engine, Re-ID model, road
provider, tracker, or event bus can be replaced without redesigning the system.

---

## The primary demonstrated capability

```
REGISTER CAMERAS ON MAP
        ↓
PROCESS CAMERA FEEDS
        ↓
RECOGNIZE PLATES WITH PLATEVISION
        ↓
TRACK VEHICLES (local, per camera)
        ↓
EXTRACT VEHICLE VISUAL FEATURES
        ↓
FUSE PLATE + VISUAL + TIME + ROAD NETWORK
        ↓
RECONSTRUCT VEHICLE TRAJECTORIES
        ↓
SEARCH A NUMBER PLATE
        ↓
DISPLAY THE VEHICLE'S COMPLETE OBSERVED/INFERRED ROUTE ON THE CITY MAP
```

---

## Key capabilities

**ANPR core (PlateVision)**
- Live phone-camera scanning, drag-and-drop photo/video upload, CCTV/RTSP stream page.
- High-resolution plate cropping with configurable padding and boundary clamping.
- Multi-variant OCR preprocessing (CLAHE + Otsu, bilateral denoising, morphological sharpening, super-resolution).
- Position-aware Indian plate normalization (`O/0`, `I/1`, `B/8`, `S/5`, `Z/2`, `G/6`, `A/4`) for standard state and Bharat Series formats. Raw OCR is always preserved alongside the normalized value and the transformations applied.

**Drishti Grid (multi-camera layer)**
- Camera discovery (USB devices via OpenCV/OS), RTSP/HTTP/file sources, and feed probing before registration.
- Map-based camera registration with persisted coordinates and full lifecycle status (`OFFLINE`/`CONNECTING`/`PROCESSING`/`ONLINE`/`ERROR`).
- ByteTrack-style local multi-object tracking with persistent per-camera track IDs. A local track ID is **never** treated as a global vehicle identity.
- Vehicle Re-ID embedding (ONNX adapter or classical backend) compared by cosine similarity.
- Cross-camera fusion with configurable weights and an explicit confidence gate + abstention.
- Global vehicle identities assembled from plate hypotheses, visual embeddings, temporal consistency, and road feasibility.
- Plate search returning the full reconstructed route with per-transition evidence.
- Traffic analytics: segment speed, flow rate, congestion ratio, and a city heatmap.

---

## Mathematical model

### Cross-camera association score

```
S_ij = λ_p · s_plate_ij  +  λ_v · cos(v_i, v_j)  −  λ_t · |Δt_ij − t_G(c_i, c_j)| / σ_t
```

| Symbol | Meaning |
|---|---|
| `s_plate_ij` | OCR plate agreement score in `[0, 1]` |
| `cos(v_i, v_j)` | Cosine similarity of vehicle Re-ID embeddings: `(v_i · v_j) / (‖v_i‖·‖v_j‖)` |
| `Δt_ij` | Observed timestamp difference between sightings |
| `t_G(c_i, c_j)` | Expected travel time between the two cameras on the road graph |
| `σ_t` | Allowed temporal variance |

Weights and thresholds are configuration-driven (never hard-coded), and can be
changed at runtime through `PUT /api/grid/fusion/config`:

`λ_p = 0.50` · `λ_v = 0.35` · `λ_t = 0.15` · `σ_t = 180 s`

### Confidence gate & abstention

A link is accepted only if **both** hold:

```
S(1) ≥ τ          AND          S(1) − S(2) ≥ δ
```

with `τ = 0.70` (minimum confidence) and `δ = 0.15` (required margin over the
second-best candidate). Otherwise the link is **ABSTAINED** and surfaced for
human review.

```
Candidates 0.82 and 0.54  →  margin 0.28  →  ACCEPT
Candidates 0.82 and 0.79  →  margin 0.03  →  ABSTAIN
```

This is deliberate: the system prefers *"unknown / unresolved"* over an
incorrectly identified vehicle.

### Evaluation modes

`FUSION_MODE` enables the research comparison:
`plate_only` → `plate_visual` → `full` (plate + visual + spatio-temporal graph).

### Traffic analytics

```
Average speed:     v̄_e = d_e / Δt_e
Flow rate:         q_e = N_e / ΔT
Congestion ratio:  r_e = q_e / q_cap_e
```

Segment capacity `q_cap_e` is a baseline approximation and is labelled as such in the UI.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy |
| Vision | OpenCV (headless), NumPy, Pillow, ONNX Runtime |
| Database | SQLite (default, zero setup) · PostgreSQL + PostGIS (optional) |
| Eventing | In-process bus (default) · Kafka (optional) |
| Routing | OpenRouteService API · `osmnx` graph · geographic fallback |
| Frontend | Next.js 14.2.15, React 18.3.1, TypeScript 5.5.4, Tailwind CSS |
| GIS | Leaflet 1.9.4, React-Leaflet 4.2.1, OpenStreetMap tiles |
| Testing | pytest (backend) · Vitest + jsdom (frontend) |

---

## Repository structure

```
platevision/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── endpoints/
│   │   │   │   ├── detect.py       # /api/detect/image, /api/detect/frame
│   │   │   │   ├── cctv.py         # /api/cctv/* live stream control
│   │   │   │   ├── results.py      # /api/results/{id}/crop
│   │   │   │   ├── health.py       # /api/health
│   │   │   │   └── grid.py         # /api/grid/*  (Drishti Grid, 25 routes)
│   │   │   └── router.py
│   │   ├── core/                   # config, redacted logging, security headers
│   │   ├── schemas/                # detection.py, grid.py
│   │   ├── services/               # cropper, normalizer, validator, detector/, ocr/
│   │   ├── grid/                   # Drishti Grid subsystem
│   │   │   ├── camera_discovery.py camera_manager.py pipeline.py
│   │   │   ├── vehicle_detector.py local_tracker.py plate_localizer.py
│   │   │   ├── plate_ocr.py        plate_source.py  embedder.py
│   │   │   ├── fusion.py           trajectory.py    road_network.py
│   │   │   ├── analytics.py        privacy.py       events.py
│   │   │   ├── metrics.py          demo.py          repository.py
│   │   │   ├── models.py           db.py            frame_store.py
│   │   └── main.py
│   ├── tests/                      # 103 pytest tests
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # ANPR dashboard
│   │   ├── grid/page.tsx           # Drishti Grid command dashboard
│   │   ├── cctv/page.tsx           # Live traffic/CCTV stream
│   │   ├── camera/page.tsx         # Phone-camera scanner
│   │   ├── upload/page.tsx         # Photo/video upload
│   │   ├── history/page.tsx        # Local audit ledger
│   │   └── about/page.tsx          # Format standards
│   ├── components/
│   │   ├── grid/                   # CityMap, CameraManagerPanel, VehicleSearchPanel,
│   │   │                           # LiveCameraGrid, TrajectoryTimeline, TrafficAnalyticsPanel
│   │   ├── camera/ upload/ results/ history/ common/ ui/
│   ├── hooks/                      # useCamera, useLiveDetection, useLocalHistory
│   ├── lib/                        # grid-api.ts, grid-types.ts, grid-trajectory.ts
│   ├── tests/                      # 16 vitest tests
│   ├── Dockerfile
│   └── package.json
├── models/                         # ONNX weights (not committed) + README
├── scripts/                        # setup_models.py, run_dev.sh, run_dev.bat
├── desktop-console/                # standalone GPU multi-plate console (separate README)
├── docker-compose.yml
├── .env.example
├── DRISHTI_GRID.md                 # full subsystem reference
└── README.md
```

---

## Installation & running

This section covers every supported way to install and run the project. If you
just want the fastest path, follow **steps 1–7** below — no database, model
weights, or GPU are required to get a working demo.

### 0. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | **3.10+** (3.11–3.13 tested) | Backend runtime |
| Node.js | **18+** (20+ recommended) | Frontend build/run |
| npm | bundled with Node | Frontend deps |
| Git | any recent | Clone |
| Docker + Compose | optional | Containerised run |
| ONNX model weights | optional | See [step 4](#4-optional-model-weights) |
| GPU | optional | CPU-only works; the pipeline falls back automatically |

Check your toolchain:

```bash
python --version        # 3.10 or newer
node --version          # v18 or newer
npm --version
```

### 1. Clone the repository

```bash
git clone <repository-url> platevision
cd platevision
```

### 2. Configure environment variables

```bash
cp .env.example .env            # repo root — loaded automatically
# or: cp .env.example backend/.env
```

Every variable in `.env.example` has a working default, so **this step is
optional** — the app runs without a `.env`. The backend loads `.env` from the
repository root *or* from `backend/` (a `backend/.env` takes precedence when both
exist). To enable authenticated features, set at least:

```ini
DRISHTI_ADMIN_KEY=admin-demo        # role: admin  — full plates, audit, fusion control
DRISHTI_OPERATOR_KEY=operator-demo  # role: operator — camera + stream control
DRISHTI_VIEWER_KEY=viewer-demo      # role: viewer  — read-only, plates masked
```

If no key is set, the API falls back to a masked, read-only **viewer** role.
Environment variables exported in the shell override the `.env` file. See
[Configuration](#configuration) for the full list of variables.

### 3. Install backend dependencies

```bash
cd backend
python -m venv .venv

# Activate the virtual environment:
source .venv/bin/activate           # macOS / Linux
# .venv\Scripts\activate            # Windows (cmd)
# .venv\Scripts\Activate.ps1        # Windows (PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs FastAPI, Uvicorn, Pydantic v2, SQLAlchemy, OpenCV
(headless), NumPy, Pillow, ONNX Runtime, and the pytest toolchain.

> **Debian/Ubuntu note:** the `opencv-python-headless` wheel needs `libglib2.0-0`.
> If OpenCV fails to import, run `sudo apt-get install -y libglib2.0-0`.

### 4. (Optional) Model weights

The system runs without model weights using documented mock/classical backends,
and reports which backend is active at `GET /api/grid/system/status`. For
production-grade accuracy, fetch the ONNX weights:

```bash
python scripts/setup_models.py       # from the repo root
```

Or place them manually (see [`models/README.md`](models/README.md) for sources):

```
models/plate_detector.onnx     # plate detection + OCR
models/vehicle_detector.onnx   # vehicle detection (optional)
models/vehicle_reid.onnx       # vehicle Re-ID embeddings (optional)
```

### 5. Start the backend

```bash
# Still inside backend/ with the venv active
export DRISHTI_ADMIN_KEY="admin-demo"     # Windows: set DRISHTI_ADMIN_KEY=admin-demo
export DRISHTI_VIEWER_KEY="viewer-demo"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Purpose |
|---|---|
| `http://localhost:8000` | API root |
| `http://localhost:8000/api/health` | Health check |
| `http://localhost:8000/api/docs` | Swagger UI |
| `http://localhost:8000/api/redoc` | ReDoc |

Verify it is up:

```bash
curl http://localhost:8000/api/health
```

### 6. Install and start the frontend

In a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

| URL | Purpose |
|---|---|
| `http://localhost:3000` | ANPR dashboard |
| `http://localhost:3000/grid` | **Drishti Grid command dashboard** |
| `http://localhost:3000/cctv` | Live traffic / CCTV stream |
| `http://localhost:3000/camera` | Phone-camera scanner |
| `http://localhost:3000/upload` | Photo / video upload |

The frontend reads the backend location from `NEXT_PUBLIC_BACKEND_URL`
(default `http://localhost:8000`, set in `frontend/.env.local` or the shell):

```bash
NEXT_PUBLIC_BACKEND_URL="http://localhost:8000" npm run dev
```

### 7. Verify the full stack

```bash
# 1. Backend health
curl http://localhost:8000/api/health

# 2. Start the 5-camera synthetic demo corridor (real pipeline, synthetic scenes)
curl -X POST http://localhost:8000/api/grid/demo/setup \
  -H "X-API-Key: admin-demo" -H "Content-Type: application/json" \
  -d '{"start_processing": true}'

# 3. Confirm cameras are processing
curl http://localhost:8000/api/grid/system/status -H "X-API-Key: admin-demo"
```

Then open `http://localhost:3000/grid`, set the access key to `admin-demo`, and
search a plate (e.g. `GJ01AB1234`). The full walkthrough is in
[Running the Drishti Grid demo](#running-the-drishti-grid-demo).

### Option A — one-command local start

After completing steps 1–4 once, you can start both services together:

```bash
./scripts/run_dev.sh          # macOS / Linux
scripts\run_dev.bat           # Windows
```

`run_dev.sh` activates `backend/.venv`, starts Uvicorn on port 8000 and
`next dev` on port 3000, and terminates both on `Ctrl+C`. It assumes the venv
created in step 3 already exists.

### Option B — Docker Compose

Runs the backend and frontend in containers with health checks and no local
Python/Node setup. Docker and Compose are required.

```bash
cp .env.example .env          # optional: add DRISHTI_* keys
docker compose up --build -d
docker compose ps             # check container health
docker compose logs -f backend
```

| Service | URL |
|---|---|
| Frontend | `http://localhost:3000` |
| Backend health | `http://localhost:8000/api/health` |

The Compose file mounts `./models` read-only into the backend, so place ONNX
weights there before starting if you want real model inference. Stop and remove
the containers with:

```bash
docker compose down           # add -v to also drop volumes
```

> The Compose backend does not pass `DRISHTI_*_KEY` by default. Add them under the
> backend service's `environment:` block in `docker-compose.yml`, or export them
> and use `docker compose run -e DRISHTI_ADMIN_KEY=admin-demo ...`, to enable
> authenticated (non-viewer) access.

### Option C — production build (no Docker)

```bash
# Backend — multiple workers, no auto-reload
cd backend && source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

# Frontend — build once, then serve the optimised bundle
cd frontend
npm run build
NEXT_PUBLIC_BACKEND_URL="http://localhost:8000" npm run start
```

### Ports and firewall

| Port | Service | Override |
|---|---|---|
| 8000 | Backend API | `--port` on `uvicorn` |
| 3000 | Frontend | `-p 3000` in the npm scripts, or `next dev -p <port>` |

To reach the dashboard from another device on your LAN, bind both services to
`0.0.0.0` (as shown above) and allow those ports through your firewall, then
browse to `http://<your-ip>:3000/grid`. Camera access over plain HTTP requires
the browser secure-context workaround described in
[Mobile camera over HTTPS](#mobile-camera-over-https).

### Running the test suites

```bash
cd backend  && python -m pytest -q      # 103 tests
cd frontend && npm test                 # 16 tests
cd frontend && npm run build            # production build check
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: cv2` / missing `libGL` | `sudo apt-get install -y libgl1 libglib2.0-0` (Debian/Ubuntu) |
| `Address already in use` on 8000/3000 | Another instance is running; stop it or pass a different `--port` / `-p` |
| Frontend shows no cameras or data | Confirm the backend is reachable and `NEXT_PUBLIC_BACKEND_URL` is correct, then reload |
| `403 Forbidden` on camera/search actions | Set the `X-API-Key` in the dashboard's **Set access key** field; defaults to a masked viewer otherwise |
| Dashboard shows `MockPlateDetector` | No ONNX weights present — expected; run `scripts/setup_models.py` or accept demo backends |
| Plate search returns no matches | Ensure cameras are `PROCESSING` and `POST /api/grid/fusion/recompute` has run since sightings were created |
| `docker compose` fails a health check | Give the backend ~30 s to start, then `docker compose logs backend` |

---

## Running the Drishti Grid demo

The fastest way to see the full pipeline is the synthetic demo corridor, which
runs the **real** detection, OCR, tracking, embedding, and fusion code over
rendered scenes.

```bash
# 5 demo cameras start processing a short corridor
curl -X POST http://localhost:8000/api/grid/demo/setup \
  -H "X-API-Key: admin-demo" -H "Content-Type: application/json" \
  -d '{"start_processing": true}'
```

Then open `http://localhost:3000/grid`:

1. Set the access key to `admin-demo` in the **Set access key** field.
2. Watch the 5 cameras transition to `PROCESSING` with live annotated snapshots
   (vehicle boxes, track IDs, plate boxes, recognized text).
3. Enter a plate from the recent reads (e.g. `GJ01AB1234`) in
   **"Enter vehicle number plate"** and press **Search**.
4. Inspect the reconstructed route, the per-transition evidence table, the
   visual timeline, and the unresolved candidates on the map.

Camera scenes labelled `SYNTHETIC DEMO` are demo data, and the dashboard says so.

---

## Camera setup

**Discover local cameras**

```bash
curl http://localhost:8000/api/grid/cameras/devices -H "X-API-Key: admin-demo"
```

**Probe a feed before registering**

```bash
curl -X POST http://localhost:8000/api/grid/cameras/probe \
  -H "X-API-Key: admin-demo" -H "Content-Type: application/json" \
  -d '{"source_type":"rtsp","source_uri":"rtsp://user:pass@10.0.0.5:554/stream1"}'
```

**Register a camera on the map**

1. Open `http://localhost:3000/grid` and click **Add Camera**.
2. Choose the source type: local webcam, RTSP/HTTP stream, video file, or demo.
3. Click **Pick location on map**, then click the map to set latitude/longitude.
4. Enter a name (e.g. `Camera 06 - Ring Road`) and click **Save Camera**.
5. Start/stop processing with the per-camera controls.

Camera metadata persists across restarts. Status transitions through
`OFFLINE → CONNECTING → PROCESSING`, with `ERROR` on failure; a single camera
failing never crashes the rest of the system.

### Mobile camera over HTTPS

Mobile browsers require a secure context for `getUserMedia`. For local Wi-Fi
testing, either use Chrome's
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` flag, or deploy behind
an SSL reverse proxy (Caddy, Nginx + Let's Encrypt, Cloudflare) in production.

---

## API reference

**PlateVision core**

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Service health |
| POST | `/api/detect/image` | Detect + OCR from an uploaded image |
| POST | `/api/detect/frame` | Detect + OCR from a live frame |
| POST | `/api/detect/video` | Detect + OCR across a video file |
| GET | `/api/results/{request_id}/{detection_id}/crop` | Retrieve a plate crop |
| POST | `/api/cctv/probe` \| `/start` \| `/stop` | Probe / control a stream |
| GET | `/api/cctv/status` \| `/feed` \| `/events` | Stream status, frames, events |

**Drishti Grid** (`/api/grid`, 25 routes)

| Method | Path | Purpose |
|---|---|---|
| GET/POST | `/cameras` | List / register cameras |
| PUT/DELETE | `/cameras/{id}` | Update / delete a camera |
| GET | `/cameras/devices` | Enumerate local camera devices |
| POST | `/cameras/probe` | Test feed accessibility |
| POST | `/cameras/{id}/start` \| `/stop` | Start / stop processing |
| GET | `/cameras/{id}/status` \| `/events` \| `/sightings` | Runtime status and events |
| GET | `/cameras/{id}/frame` | Latest annotated JPEG snapshot |
| GET | `/vehicles` | List vehicle identities |
| GET | `/vehicles/search?plate=...` | Search by plate (audited) |
| GET | `/vehicles/{id}` \| `/vehicles/{id}/trajectory` | Identity detail / trajectory |
| GET | `/trajectory/plate?plate=...` | Trajectory of the best plate match |
| GET | `/unresolved` | Abstained candidate links |
| GET/PUT | `/fusion/config` | Read / update fusion weights |
| POST | `/fusion/recompute` | Re-run fusion (optional `mode`) |
| GET | `/analytics/traffic` \| `/analytics/heatmap` | Traffic metrics / heatmap |
| GET | `/system/status` \| `/metrics` | System status / instrumentation |
| GET | `/audit` | Audit log (operator/admin) |
| POST | `/demo/setup` | Register + start the demo corridor |

Full endpoint and schema details: [`DRISHTI_GRID.md`](DRISHTI_GRID.md).

---

## Configuration

All settings have working defaults; copy `.env.example` to `.env` to override.

| Variable | Default | Purpose |
|---|---|---|
| `GRID_DATABASE_URL` | `sqlite:///./drishti_grid.db` | Persistence (SQLite or PostgreSQL/PostGIS) |
| `DRISHTI_ADMIN_KEY` / `DRISHTI_OPERATOR_KEY` / `DRISHTI_VIEWER_KEY` | — | Role API keys (`X-API-Key`) |
| `VEHICLE_DETECTOR_BACKEND` | `hybrid` | Vehicle detector backend |
| `REID_BACKEND` | `auto` | Visual embedding backend |
| `FUSION_LAMBDA_PLATE` / `_VISUAL` / `_TIME` | `0.50` / `0.35` / `0.15` | `λ_p`, `λ_v`, `λ_t` |
| `FUSION_SIGMA_T_SECONDS` | `180.0` | `σ_t` |
| `FUSION_TAU` / `FUSION_DELTA` | `0.70` / `0.15` | Accept threshold `τ`, margin `δ` |
| `FUSION_MODE` | `full` | `plate_only` \| `plate_visual` \| `full` |
| `ORS_API_KEY` | — | OpenRouteService routing |
| `OSMNX_GRAPH_PATH` | — | Local `osmnx` road graph |
| `ROAD_FALLBACK_SPEED_KMH` / `_DETOUR_FACTOR` | `34.0` / `1.35` | Fallback travel-time model |
| `EVENT_BACKEND` | `memory` | `memory` or `kafka` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers |
| `GRID_DEMO_MODE` | `true` | Enable labelled synthetic demo cameras |
| `PLATE_MASK_FOR_VIEWER` | `true` | Mask plates for the viewer role |
| `ENABLE_AUDIT_LOG` | `true` | Record searches and camera actions |

**Roles and permissions** (resolved from `X-API-Key`; unset keys default to a masked viewer):

| Role | Permissions |
|---|---|
| `viewer` | `read` |
| `operator` | `read`, `write_camera`, `control_stream` |
| `admin` | `read`, `write_camera`, `control_stream`, `view_full_plate`, `view_audit`, `manage_fusion` |

---

## Testing

```bash
# Backend — 103 tests
cd backend && python -m pytest -v

# Frontend — 16 tests across 3 files
cd frontend && npm test

# Production build check
cd frontend && npm run build
```

**Backend coverage:** plate normalization and validation · multi-frame OCR
aggregation · character-level plate similarity · cosine visual similarity ·
travel-time calculation · candidate scoring · the confidence gate and abstention ·
camera registration · sighting creation · trajectory construction (including the
no-camera-revisit invariant and route ranking) · plate search · database
persistence · API endpoints, RBAC, and plate masking.

**Frontend coverage:** trajectory rendering logic — edge geometry resolution,
OBSERVED/INFERRED/UNRESOLVED classification, route ordering, and the
contiguous-chain check, asserting an inferred segment is never labelled
`OBSERVED` and no path is fabricated for unknown cameras.

### Instrumentation

`GET /api/grid/metrics` exposes processing FPS, camera latency, API latency
percentiles, OCR throughput, candidate rejection rate, and abstention rate so the
performance targets below can eventually be measured.

**Prototype acceptance targets:** exact plate accuracy > 90% · accepted-link
precision ≥ 95% · dashboard latency ≤ 5 s p95. These are targets; the current
implementation instruments them rather than asserting them.

---

## Privacy, security & responsible use

Vehicle registration numbers may be personal or sensitive information. Scan only
vehicles you are authorised to process.

- **Role-based access control** — `admin` / `operator` / `viewer` via `X-API-Key`;
  write, stream-control, audit, and full-plate access are permission-gated.
- **Plate masking** — viewers receive hash-masked plates (e.g. `GJ01****34`).
- **Audit logging** — every plate search, camera action, and fusion run is
  recorded with actor, role, target, and client IP.
- **No raw video exposure** — only lightweight annotated JPEG snapshots reach the
  browser; events carry lightweight descriptors, never raw video.
- **Edge-first processing** — cameras are processed locally; only descriptors and
  embeddings are stored centrally.
- **Explicit abstention** — the system reports "unresolved" rather than guessing.
- **Transient buffers** — PlateVision core crop buffers use a short-lived
  5-minute TTL cache; plate characters are redacted in system logs.
- **No owner lookup** — the system contains no vehicle-owner registry lookup.
- **Demo separation** — synthetic cameras are explicitly labelled and never
  mixed with production sources.

---

## Deployment notes

- **Database** — SQLite works with zero setup for a single-node prototype. For a
  city deployment use PostgreSQL + PostGIS (`GRID_DATABASE_URL=postgresql+psycopg://...`)
  for geographic operations. TimescaleDB is optional and never required.
- **Event bus** — the in-process bus is the default. Set `EVENT_BACKEND=kafka`
  for the distributed Camera → Edge AI → Event Stream → Fusion → Store → GIS topology.
- **Routing** — configure `ORS_API_KEY` or `OSMNX_GRAPH_PATH` for real
  road-network travel times; otherwise the labelled geographic fallback applies.
- **CORS** — set `CORS_ORIGINS` to your deployed frontend origin.

See [`DRISHTI_GRID.md`](DRISHTI_GRID.md) for the full database schema, event
contract, and deployment guidance.

---

## Known limitations

- Plate-recognition accuracy is **not benchmarked** against a labelled dataset;
  the >90% requirement is a target pending evaluation on Indian ANPR data.
- The cross-camera half is demonstrated on a synthetic corridor, not a real city
  camera network.
- No alert/watchlist system for blacklisted vehicles or route anomalies yet.
- Origin–destination matrices are not yet computed.
- Without ONNX weights, the detector/OCR/Re-ID use documented mock/classical
  backends — real outputs of those backends, but not production-grade accuracy.
- The road-network fallback is an approximation, clearly marked in the UI.
- Congestion ratio uses a fixed baseline capacity, not measured capacity.
- Local tracker IDs are per-camera by design and are only linked into global
  identities by the fusion engine.

## Roadmap

- Benchmark and tune plate recognition on Indian ANPR datasets toward the 90% target.
- Alert system: blacklist/watchlist matching with real-time notifications.
- Origin–destination pattern extraction and bottleneck detection.
- Trained vehicle Re-ID ONNX model and tracklet-level association.
- TimescaleDB for high-volume time-series sightings.
- Real road-graph routing as the default provider.

---

## Additional documentation

- [`DRISHTI_GRID.md`](DRISHTI_GRID.md) — full subsystem reference: architecture,
  fusion mathematics, API, database schema, event stream, privacy model, testing.
- [`models/README.md`](models/README.md) — model weights and download sources.
- [`desktop-console/README.md`](desktop-console/README.md) — standalone GPU
  multi-plate detection console with batched OCR and annotated video export.
  Its separate API and console are independent of the web application above; the
  ten-plate 1080p synthetic benchmark measured 39.1 ms median / 44.2 ms p95 for
  the complete local HTTP response (a strict sub-40 ms accurate all-plate result
  is not achieved). Model weights, private uploads, datasets, and credentials are
  excluded from version control.

---

*Built on the PlateVision ANPR stack. Drishti Grid is a technology prototype
submitted to Smart India Hackathon 2026 (Problem Statement 26SIH127) and does not
represent or claim official government affiliation.*
