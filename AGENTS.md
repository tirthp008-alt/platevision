# AGENTS.md

Repository-specific knowledge for agents working on PlateVision / Drishti Grid.

## Layout

- `backend/` — FastAPI app. Core PlateVision endpoints under `app/api/endpoints/`;
  the Drishti Grid subsystem lives entirely in `app/grid/` and is mounted at
  `/api/grid` (`app/api/endpoints/grid.py`, registered in `app/api/router.py`).
- `frontend/` — Next.js 14 App Router. The Drishti Grid dashboard is
  `app/grid/page.tsx` with components in `components/grid/` and API/type clients
  in `lib/grid-*.ts`.
- `desktop-console/` — separate GPU console; do not couple it to the web app.
- `models/` — ONNX weights. Not committed; see `models/README.md`.

## Build / test / run

```bash
# Backend tests (isolated SQLite configured in tests/conftest.py)
cd backend && python -m pytest -q

# Backend server
cd backend && DRISHTI_ADMIN_KEY=admin-demo python -m uvicorn app.main:app --port 8000

# Frontend
cd frontend && npm install && npm test && npm run build
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev   # dashboard at /grid
```

Conventions: the API base is `/api` and OpenAPI is served at `/api/openapi.json`
(not `/api/openapi.json` under a version prefix). Frontend calls the backend
through `NEXT_PUBLIC_BACKEND_URL` (empty string = same origin, proxied).

## Drishti Grid specifics

- Fusion weights and the confidence gate are settings-driven
  (`FUSION_LAMBDA_*`, `FUSION_TAU`, `FUSION_DELTA`, `FUSION_MODE`); never hard-code
  them. Update via `PUT /api/grid/fusion/config`.
- Fusion is a full rebuild over the recent-sighting window: edges and identities
  are deleted and recomputed each run. Identity IDs are derived from the earliest
  sighting (`veh-<sighting_id>`) so re-running fusion doesn't invalidate IDs a
  client just fetched. Sightings themselves are never deleted.
- An identity is a **simple path** over distinct cameras (a vehicle doesn't
  revisit a camera in one trip). The identity union enforces this; the displayed
  route is a single monotonic chain built in `trajectory.build_trajectory`.
- Only `ABSTAINED` edges belong in `unresolved_links`; `REJECTED` edges were
  confidently ruled out and are not open questions.
- Plate search ranking: exact match, then most accepted links, then most
  sightings. Real multi-hop routes rank above lone single-camera sightings.
- Demo cameras are clearly labelled (`is_demo`, `SYNTHETIC DEMO`) and run the real
  detection/OCR/fusion pipeline over rendered frames — outputs are genuine, not
  fabricated. Keep this separation if adding new demo behaviour.
- RBAC: roles come from `X-API-Key` (`DRISHTI_ADMIN_KEY`/`OPERATOR`/`VIEWER`).
  Unset keys ⇒ masked viewer. Viewers get masked plates and no audit access.
- Camera frames are served as annotated JPEG snapshots only; never stream raw
  video to the browser.
