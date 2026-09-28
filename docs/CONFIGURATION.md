# Configuration and operational reference

[Setup](SETUP.md) · [Architecture](ARCHITECTURE.md) · [Full Grid reference](../DRISHTI_GRID.md)

## Web application

Application settings load from root `.env` and then `backend/.env`; shell
environment values override them. Role keys and routing-provider credentials
are read directly from the process environment. Keep secrets out of committed
files and shell examples used in public presentations.

| Setting | Default / purpose |
|---|---|
| `DETECTOR_BACKEND`, `ONNX_MODEL_PATH` | `auto`; compatible root plate detector or reported fallback |
| `GRID_DATABASE_URL` | `sqlite:///./drishti_grid.db`; local persistence |
| `VEHICLE_DETECTOR_BACKEND` | `hybrid`; Grid vehicle proposal backend |
| `VEHICLE_MODEL_PATH` | Configured Grid vehicle model path; separate from the console's UVH adapter |
| `REID_BACKEND`, `REID_MODEL_PATH` | `auto`; Grid appearance adapter and model |
| `FUSION_LAMBDA_PLATE`, `FUSION_LAMBDA_VISUAL`, `FUSION_LAMBDA_TIME` | `0.50`, `0.35`, `0.15` |
| `FUSION_SIGMA_T_SECONDS` | `180`; travel-time penalty scale |
| `FUSION_TAU`, `FUSION_DELTA` | `0.70`, `0.15`; score threshold and candidate margin |
| `FUSION_MODE` | `full`; also `plate_only` or `plate_visual` |
| `ORS_API_KEY` | Optional routing-provider key, exported in the process environment |
| `OSMNX_GRAPH_PATH` | Optional local road graph; requires its routing dependencies |
| `ROAD_FALLBACK_SPEED_KMH`, `ROAD_FALLBACK_DETOUR_FACTOR` | `34.0`, `1.35`; labelled approximation when routing is unavailable |
| `EVENT_BACKEND` | `memory`; optional Kafka configuration is separate |
| `GRID_DEMO_MODE` | `true`; labelled synthetic camera source |
| `PLATE_MASK_FOR_VIEWER`, `ENABLE_AUDIT_LOG` | `true`, `true` |
| `CORS_ORIGINS` | Set only origins appropriate to the deployment |
| `NEXT_PUBLIC_BACKEND_URL` | Browser-reachable API origin; empty/unset uses the frontend proxy |

Fusion weights can be inspected or changed through `/api/grid/fusion/config`
with the required role. A configuration change is not an accuracy improvement;
record it when evaluating a run.

### Roles

| Role | Permissions |
|---|---|
| Viewer / unknown key | Read, with masked plate text |
| Operator | Read, camera changes, and stream control |
| Administrator | Operator actions, full plate text, audit access, and fusion management |

Export `DRISHTI_ADMIN_KEY`, `DRISHTI_OPERATOR_KEY`, and/or `DRISHTI_VIEWER_KEY`
in the backend shell and send the corresponding value in `X-API-Key`. An
operator is not automatically entitled to full plates or the audit log.
The dashboard stores the entered key for its browser session.

### Cameras and API discovery

The API base is `/api`, with Grid mounted at `/api/grid`. Browse
`/api/docs` or `/api/openapi.json` for the current schema.

| Action | Grid endpoint |
|---|---|
| Discover local devices | `GET /api/grid/cameras/devices` |
| Probe a source | `POST /api/grid/cameras/probe` |
| Register / list cameras | `POST /api/grid/cameras`, `GET /api/grid/cameras` |
| Start / stop a camera | `POST /api/grid/cameras/{id}/start`, `/stop` |
| Inspect snapshots / sightings | `GET /api/grid/cameras/{id}/frame`, `/sightings` |
| Search observed plate evidence | `GET /api/grid/vehicles/search?plate=...` |
| Inspect a journey | `GET /api/grid/vehicles/{id}/trajectory` |
| Inspect abstained links | `GET /api/grid/unresolved` |
| Recompute fusion | `POST /api/grid/fusion/recompute` |
| Inspect traffic / heatmap | `GET /api/grid/analytics/traffic`, `/heatmap` |
| Check runtime / metrics | `GET /api/grid/system/status`, `/metrics` |

Use the map's **Add Camera** flow to choose a device, RTSP/file source, or
labelled demo source, probe it, set its coordinates, and start processing.
Camera metadata persists in the database; camera runtime is process-local.
Do not expose a credential-bearing camera URL in screenshots or logs.

### Optional infrastructure

SQLite and the in-process bus are the documented local path. PostgreSQL/PostGIS,
Kafka, and external/local road routers require their own drivers, services,
permissions, and configuration. Geographic fallbacks remain explicitly marked.
Metrics such as processing FPS, candidate rejection, and abstention are
instrumentation, not validated service-level guarantees.

## Desktop console

Use `scripts/run_console.py` from the repository root with the console Python
environment active. It configures its independent backend without changing
Drishti Grid's environment or running API.

| Option | Meaning |
|---|---|
| `--model recommended` | Green YOLOv8n; API port 8003 by default |
| `--model yolo11n` | Evaluated candidate; API port 8004 by default |
| `--port` | Override the selected API port |
| `--web-port` | Static console port; default 8080 |
| `--device auto` / `cpu` / `gpu` | Select runtime policy; inspect `/api/health` for the actual engine |
| `--check` | Verify the launch configuration without starting services |

The console defaults to PP-OCRv4 with OCR and plate-angle correction on. It
enables UVH vehicle categories and bounds extra vehicle-crop searches to eight
per frame. Experimental emboss and cross-camera matching remain off unless
explicitly selected. The optional ReID bundle is for this console's appearance
adapter; it does not automatically configure Grid's distinct embedding adapter.

Console image endpoints are `/api/detect/image` and `/api/detect/frame`; recording
jobs use `/api/videos`. The root web app has its own API contracts. Avoid pointing
one application's client at the other's API solely because both use `/api`.

The console API is a localhost development service without Grid's role system.
Keep uploads, result directories, camera metadata, and registration strings
private. Expiring local results and review flags are workflow features, not a
substitute for deployment access control or a data-retention policy.
