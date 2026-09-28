# Drishti Grid — City-Wide Multi-Camera ANPR Trajectory Tracking

Drishti Grid is the city-scale extension of PlateVision. It registers cameras on a
GIS map, processes their feeds through the existing PlateVision detection/OCR
stack, tracks vehicles locally, and reconstructs a vehicle's route across cameras
by fusing **plate evidence + visual Re-ID + temporal + road-network** constraints.

It is intended for **authorised, synthetic, or demo traffic footage**. The
dashboard never presents an inferred movement as a direct observation and never
fabricates a route when the evidence is insufficient.

---

## 1. Architecture

```
CAMERA SOURCES (USB / RTSP / file / synthetic demo)
        |
        v
VIDEO INGESTION (per-camera worker thread, isolated failures)
        |
        v
VEHICLE DETECTION  (app/grid/vehicle_detector.py)
        |
        +-----------------------+
        |                       |
        v                       v
LOCAL TRACKING (ByteTrack-    PLATE LOCALIZATION
style IoU tracker,            (app/grid/plate_localizer.py)
app/grid/local_tracker.py)          |
        |                           v
        |                    PLATE OCR (app/grid/plate_ocr.py)
        |                           |
        |                    MULTI-FRAME VOTING
        |                    (plate_ocr.py aggregation)
        |                           |
        |                    PLATE CONFIDENCE
        |                           |
        v                           v
VEHICLE CROP  ->  VISUAL EMBEDDING (app/grid/embedder.py)
        |
        v
SIGHTING EVENT  (app/grid/models.py: VehicleSighting)
        |
        v
EVENT / DB LAYER  (app/grid/events.py, SQLAlchemy + optional PostGIS)
        |
        v
CROSS-CAMERA FUSION ENGINE  (app/grid/fusion.py)
        |
        +--------------+--------------+
        |              |              |
        v              v              v
   Plate evidence  Visual        Temporal /
   s_plate         similarity    road graph
        |              |              |
        +--------------+--------------+
                       |
                       v
              Candidate score S_ij
                       |
                       v
              Confidence gate (tau, delta)
                       |
              +--------+--------+
              |                 |
           Accept            Abstain
              |                 |
              v                 v
       Trajectory edge     Unresolved link
              |
              v
       TRAJECTORY GRAPH -> GIS DASHBOARD (frontend/app/grid/page.tsx)
```

Every stage is a separate module so a detector, OCR engine, embedder, road
provider, or tracker can be replaced without redesigning the application.

### Backend modules (`backend/app/grid/`)

| Module | Responsibility |
| --- | --- |
| `camera_discovery.py` | Enumerate local webcams via OpenCV/OS backends |
| `camera_manager.py` | Per-camera worker threads, status, health, frame store |
| `vehicle_detector.py` | Vehicle detection (hybrid motion / plate-anchored / model) |
| `local_tracker.py` | ByteTrack-style multi-object tracker (persistent local IDs) |
| `plate_localizer.py` | Plate region localization inside a vehicle crop |
| `plate_ocr.py` | OCR adapter + multi-frame voting + confidence estimation |
| `embedder.py` | Vehicle Re-ID embedding (classical or ONNX adapter) |
| `pipeline.py` | Frame -> sighting orchestration |
| `plate_source.py` | Indian plate normalization + validation |
| `road_network.py` | Camera graph, routing, `expected_travel_time` |
| `fusion.py` | Candidate scoring, confidence gate, identity assembly |
| `trajectory.py` | Plate search + trajectory assembly |
| `analytics.py` | Traffic metrics + heatmap |
| `privacy.py` | RBAC, plate masking, audit logging |
| `events.py` | `EventPublisher`/`EventConsumer` (in-process or Kafka) |
| `demo.py` | Synthetic demo corridor (clearly separated from production) |
| `repository.py` | Database access layer |
| `models.py` | SQLAlchemy tables |
| `metrics.py` | Performance instrumentation |

---

## 2. Mathematical model

### Cross-camera association score

```
S_ij = λ_p · s_plate_ij  +  λ_v · cos(v_i, v_j)  −  λ_t · |Δt_ij − t_G(c_i,c_j)| / σ_t
```

- `s_plate_ij` — OCR plate agreement score in `[0, 1]`
- `cos(v_i, v_j)` — cosine similarity of visual embeddings:
  `(v_i · v_j) / (||v_i|| · ||v_j||)`
- `Δt_ij` — observed timestamp difference
- `t_G(c_i, c_j)` — expected travel time between cameras on the road graph
- `σ_t` — allowed temporal variance

Weights `λ_p`, `λ_v`, `λ_t` and `σ_t` are configuration values
(`FUSION_LAMBDA_PLATE`, `FUSION_LAMBDA_VISUAL`, `FUSION_LAMBDA_TIME`,
`FUSION_SIGMA_T_SECONDS`) and can also be set at runtime through
`PUT /api/grid/fusion/config`. They are never hard-coded in the pipeline.

### Confidence gate / abstention

Accept a link only if:

```
S(1) >= τ          AND          S(1) − S(2) >= δ
```

where `S(1)` is the best candidate score and `S(2)` the second-best. Defaults:
`τ = 0.70`, `δ = 0.15` (`FUSION_TAU`, `FUSION_DELTA`). These are starting values,
not universal truths.

Example:

```
Candidate A = 0.82,  Candidate B = 0.54  -> margin 0.28 -> ACCEPT
Candidate A = 0.82,  Candidate B = 0.79  -> margin 0.03 -> ABSTAIN
```

An abstained link is surfaced in the dashboard as **UNRESOLVED** for human
review rather than being forced into a route.

### Research comparison modes

`FUSION_MODE` (or `POST /api/grid/fusion/recompute {"mode": ...}`) selects:

1. `plate_only` — plate evidence only
2. `plate_visual` — plate + visual Re-ID
3. `full` — plate + visual Re-ID + spatio-temporal road-graph fusion

### Traffic analytics

```
Average speed:     v̄_e = d_e / Δt_e
Flow rate:         q_e = N_e / ΔT
Congestion ratio:  r_e = q_e / q_cap_e
```

Segment capacity is a baseline approximation and is labelled as such in the UI.

---

## 3. Installation

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Python 3.10+ is required. By default the system uses:
- SQLite persistence (zero setup)
- the repository's existing detector/OCR stack
- the in-process event bus

### Frontend

```bash
cd frontend
npm install
```

Required frontend packages for the map: `leaflet`, `react-leaflet` (already added
to `package.json`).

### Database (optional PostgreSQL/PostGIS)

```bash
export GRID_DATABASE_URL="postgresql+psycopg://user:pw@host:5432/drishti"
```

SQLite is used when `GRID_DATABASE_URL` is unset or points at a `sqlite://` URL.
TimescaleDB is **not** required.

---

## 4. Environment variables

Add these to `.env` (see `.env.example`) — all have working defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GRID_ENABLED` | `true` | Enable the Drishti Grid subsystem |
| `GRID_DATABASE_URL` | `sqlite:///./drishti_grid.db` | Persistence (SQLite or PostgreSQL/PostGIS) |
| `DRISHTI_ADMIN_KEY` | — | API key for the `admin` role |
| `DRISHTI_OPERATOR_KEY` | — | API key for the `operator` role |
| `DRISHTI_VIEWER_KEY` | — | API key for the `viewer` role |
| `VEHICLE_DETECTOR_BACKEND` | `hybrid` | Vehicle detector backend |
| `REID_BACKEND` | `auto` | Visual embedding backend (`auto`/`onnx`/`classical`) |
| `FUSION_LAMBDA_PLATE` | `0.50` | `λ_p` |
| `FUSION_LAMBDA_VISUAL` | `0.35` | `λ_v` |
| `FUSION_LAMBDA_TIME` | `0.15` | `λ_t` |
| `FUSION_SIGMA_T_SECONDS` | `180.0` | `σ_t` |
| `FUSION_TAU` | `0.70` | Accept threshold `τ` |
| `FUSION_DELTA` | `0.15` | Required margin `δ` |
| `FUSION_MODE` | `full` | `plate_only` / `plate_visual` / `full` |
| `ROAD_FALLBACK_SPEED_KMH` | `34.0` | Fallback speed model (approximation) |
| `ROAD_FALLBACK_DETOUR_FACTOR` | `1.35` | Fallback detour factor |
| `EVENT_BACKEND` | `memory` | `memory` (in-process) or `kafka` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers when `EVENT_BACKEND=kafka` |
| `KAFKA_TOPIC_SIGHTINGS` | `drishti.sightings` | Kafka sightings topic |
| `GRID_DEMO_MODE` | `true` | Enable clearly-labelled synthetic demo cameras |
| `PLATE_MASK_FOR_VIEWER` | `true` | Mask plates for the viewer role |
| `ENABLE_AUDIT_LOG` | `true` | Record searches and camera actions |

Without any `DRISHTI_*_KEY` set, the default role is a masked **viewer**. Set at
least the admin key to register cameras, start streams, run fusion, and read the
audit log.

---

## 5. Running the application

### Backend

```bash
cd backend
export DRISHTI_ADMIN_KEY="admin-demo" DRISHTI_VIEWER_KEY="viewer-demo"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health: `GET /api/health`  •  Grid OpenAPI: `GET /api/openapi.json`

### Frontend

```bash
cd frontend
NEXT_PUBLIC_BACKEND_URL="http://localhost:8000" npm run dev
```

Open the dashboard at **http://localhost:3000/grid**.

### Docker

```bash
docker compose up --build
```

---

## 6. Camera setup

### Discover local cameras

`GET /api/grid/cameras/devices` lists USB webcams detected through OpenCV/OS
backends. If none are found locally the response explains how to attach one.

### Test a feed before registering

`POST /api/grid/cameras/probe`

```json
{ "source_type": "rtsp", "source_uri": "rtsp://user:pass@10.0.0.5:554/stream1" }
```

### Register a camera on the map

1. Open **/grid**.
2. Click **Add Camera**.
3. Pick a source: *Local webcam*, *RTSP / HTTP stream*, *Video file*, or *Demo*.
4. Click **Pick location on map**, then click the map to set lat/lon.
5. Enter a name (e.g. `Camera 06 - Ring Road`) and **Save Camera**.

The marker appears immediately; camera metadata persists in the database.

### Start / stop processing

Use the ▶ / ■ buttons on each camera card (or
`POST /api/grid/cameras/{id}/start` / `/stop`). Status transitions through
`OFFLINE → CONNECTING → PROCESSING`, with `ERROR` on failure. One camera failing
never crashes the others.

---

## 7. PlateVision setup

Drishti Grid reuses the repository's existing detector + OCR stack rather than
introducing a parallel model path:

- **Plate detector**: the PlateVision ONNX detector (`models/plate_detector.onnx`)
  when available, otherwise `MockPlateDetector` for development.
- **OCR**: `app/grid/plate_ocr.py` wraps the same normalizer/validator used by the
  core PlateVision endpoints (`app/services`), and applies multi-frame voting.

To supply real weights, place them at `models/plate_detector.onnx`
(add `models/vehicle_detector.onnx` and `models/vehicle_reid.onnx` for vehicle
detection and Re-ID). `scripts/setup_models.py` documents download sources. With
no weights present the system stays runnable using mock/classical backends —
this is clearly reported in `GET /api/grid/system/status`, never hidden.

**Plate normalization** preserves the raw OCR string, the normalized string, and
the recorded transformations; characters are never silently altered.

---

## 8. The complete workflow

```
MAP  ->  select camera location  ->  attach camera feed  ->  start processing
     ->  vehicle detection  ->  plate detection + OCR  ->  local tracking
     ->  visual embedding  ->  sighting event  ->  cross-camera candidates
     ->  plate + visual + temporal + road fusion  ->  trajectory graph
     ->  search plate  ->  display route on map
```

Search a plate in the dashboard's **"Enter vehicle number plate"** field. The
result panel shows the identity, first/last seen, reconstructed route, each
transition's plate evidence, visual similarity, observed Δt, expected travel
time, combined score, and decision — plus the unresolved candidates.

---

## 9. API reference

Base path: `/api/grid`

| Method | Path | Description |
| --- | --- | --- |
| GET | `/cameras` | List registered cameras |
| POST | `/cameras` | Register a camera |
| PUT | `/cameras/{id}` | Update a camera |
| DELETE | `/cameras/{id}` | Delete a camera |
| GET | `/cameras/devices` | Enumerate local camera devices |
| POST | `/cameras/probe` | Test a feed's accessibility |
| POST | `/cameras/{id}/start` | Start processing |
| POST | `/cameras/{id}/stop` | Stop processing |
| GET | `/cameras/{id}/status` | Runtime status + recent detections |
| GET | `/cameras/{id}/events` | Recent sighting events |
| GET | `/cameras/{id}/sightings` | Recent sightings |
| GET | `/cameras/{id}/frame` | Latest annotated JPEG snapshot |
| GET | `/vehicles` | List vehicle identities |
| GET | `/vehicles/search?plate=...` | Search by plate (audited) |
| GET | `/vehicles/{id}` | Vehicle identity detail |
| GET | `/vehicles/{id}/trajectory` | Reconstructed trajectory |
| GET | `/trajectory/plate?plate=...` | Trajectory of the best plate match |
| GET | `/unresolved` | Abstained candidate links |
| GET | `/fusion/config` | Read fusion weights |
| PUT | `/fusion/config` | Update fusion weights (admin) |
| POST | `/fusion/recompute` | Re-run fusion (optional `mode`) |
| GET | `/analytics/traffic` | Traffic segment metrics |
| GET | `/analytics/heatmap` | Vehicle density heatmap |
| GET | `/system/status` | Aggregate system status |
| GET | `/metrics` | Performance instrumentation |
| GET | `/audit` | Audit log (admin/operator) |
| POST | `/demo/setup` | Register + start the synthetic demo corridor |

Swagger UI is available at `/docs`.

---

## 10. Database schema

Tables (SQLAlchemy; add PostGIS geometry for camera coordinates in a
PostgreSQL deployment):

- **cameras** — id, name, source_type, source_uri, device_id, latitude, longitude,
  road_segment_id, enabled, status, created_at, updated_at, last_seen, last_error,
  is_demo
- **vehicle_sightings** — id, camera_id, vehicle_local_track_id, timestamp,
  plate_raw, plate_normalized, plate_confidence, visual_embedding, vehicle_type,
  vehicle_color, vehicle_bbox, plate_bbox, frame_reference, detection_confidence,
  vehicle_identity_id
- **vehicle_tracks** — per-camera local track metadata
- **vehicle_identities** — id, display_plate, status, first_seen, last_seen,
  sighting_count, confidence, plate_hypotheses
- **trajectory_edges** — id, source/destination camera + sighting, time_delta,
  expected_travel_time, road_distance_m, plate_score, visual_similarity,
  temporal_score, combined_score, runner_up_score, decision, decision_reason,
  path_geometry, fusion_mode
- **plate_observations** — per-frame OCR readings behind a sighting
- **processing_sessions** — camera processing runs
- **audit_logs** — actor, role, action, target, detail, client_ip, result

---

## 11. Event stream

`app/grid/events.py` defines `EventPublisher`/`EventConsumer`:

- **`InProcessEventBus`** (default, `EVENT_BACKEND=memory`) for local development.
- **`KafkaEventPublisher`** (`EVENT_BACKEND=kafka`) for the full deployment.

Events carry lightweight descriptors (camera_id, timestamp, track_id, plate,
plate_confidence, visual_embedding, vehicle_type, bbox) — never raw video.

---

## 12. Privacy & security

- **RBAC**: `admin` / `operator` / `viewer` roles selected by `X-API-Key`
  (`DRISHTI_ADMIN_KEY`, `DRISHTI_OPERATOR_KEY`, `DRISHTI_VIEWER_KEY`). Write and
  stream-control endpoints require operator/admin; audit log requires
  operator/admin.
- **Plate masking**: viewers receive hash-masked plates (`GJ01****34`) and cannot
  read the audit log.
- **Audit logging**: every plate search, camera registration, stream control, and
  fusion recompute is recorded with actor, role, target, and client IP.
- **No raw video exposure**: only lightweight annotated JPEG snapshots are served;
  the browser never receives raw video streams.
- **Edge-first**: cameras are processed locally; only descriptors/embeddings are
  stored centrally.

---

## 13. Testing

```bash
# Backend
cd backend && python -m pytest -q

# Frontend
cd frontend && npm test
```

Backend suites cover plate normalization, multi-frame OCR aggregation,
character-level plate similarity, cosine visual similarity, travel-time
calculation, candidate scoring, the confidence gate, abstention, camera
registration, sighting creation, trajectory construction (including the
no-camera-revisit invariant and route ranking), plate search, database
persistence, and API endpoints/RBAC.

Frontend suites (`tests/grid_trajectory.test.ts`) cover trajectory rendering
logic: edge geometry resolution, observed/inferred/unresolved classification,
route ordering, and the contiguous-chain check — asserting an inferred segment
is never labelled `OBSERVED` and no path is fabricated for unknown cameras.

### End-to-end demo

```bash
export DRISHTI_ADMIN_KEY="admin-demo"
python -m uvicorn app.main:app --port 8000 &

curl -X POST http://localhost:8000/api/grid/demo/setup \
  -H "X-API-Key: admin-demo" -H "Content-Type: application/json" \
  -d '{"start_processing": true}'
```

Five synthetic cameras start on a short corridor; the real detection/OCR/
tracking/fusion pipeline runs over their rendered frames. Then search a plate in
the dashboard (or `GET /api/grid/vehicles/search?plate=GJ01AB1234`) and inspect
the reconstructed route on the map.

---

## 14. Known limitations

- The road network falls back to a geodesic distance + assumed-speed model when
  no OSM router is available; the fallback is explicitly marked as approximate.
- Segment capacity in the congestion ratio is a fixed baseline, not measured.
- With no ONNX weights present, the detector/OCR/embedder use documented
  mock/classical backends; results are real outputs of those backends, not
  fabricated values.
- Demo camera scenes are synthetic; they exercise the genuine pipeline but are
  not real traffic.
- PostGIS geometry columns are not created automatically for SQLite; geographic
  operations use the Python road-network module.

## 15. Future improvements

- Real road-network routing (OSRM/Valhalla/OSM graph) as the default provider.
- Trained Indian-plate and vehicle Re-ID ONNX models.
- TimescaleDB for high-volume time-series sightings.
- Cross-camera tracklet-level (not only sighting-level) association.
- Active-learning loop over abstained candidates.
- Model-drift and OCR-accuracy monitoring wired to `/api/grid/metrics`.
