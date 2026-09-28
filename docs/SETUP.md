# Local setup

[Overview](../README.md) · [Judge walkthrough](DEMO.md) · [Configuration](CONFIGURATION.md)

Drishti Grid and the desktop console are separate applications. Start with the web app for camera journeys or the console for the measured model comparison. Keep their Python environments separate because their vision dependencies differ.

## Requirements

- Python **3.12** and a virtual environment.
- Node.js **20 or newer** with npm for `frontend/`; the console does not need Node.
- A modern browser. Localhost supports browser camera access; remote camera access requires an appropriate secure deployment.
- A GPU is optional. Published console timings used Intel Arc 130T; CPU timings will differ. No training dependencies are required to use released ONNX models.

```bash
git clone https://github.com/tirthp008-alt/platevision.git
cd platevision
```

## Drishti Grid web application

### Install the backend

From the repository root:

```bash
python -m venv backend/.venv
```

Activate it on macOS/Linux with `source backend/.venv/bin/activate`, or on Windows PowerShell with `./backend/.venv/Scripts/Activate.ps1`. Then:

```bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

SQLite is the default database. No database server or camera is needed for the synthetic demo. Copy `.env.example` to `.env` only when overriding settings; `backend/.env` takes precedence over the root file for application settings.

Set role keys in the **backend process environment**. They are resolved from environment variables, so copying a role key into `.env` alone is insufficient. These example values are for a localhost demo only.

macOS/Linux:

```bash
export DRISHTI_ADMIN_KEY=admin-demo
export DRISHTI_VIEWER_KEY=viewer-demo
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows PowerShell, starting at the repository root with the environment active:

```powershell
$env:DRISHTI_ADMIN_KEY = 'admin-demo'
$env:DRISHTI_VIEWER_KEY = 'viewer-demo'
Set-Location backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Use one backend process for the local camera demo: camera workers, frame buffers, and the default event bus live in that process. Multiple workers are not a documented deployment shortcut for this stateful pipeline.

### Start the frontend

For a judge demonstration, use the built frontend. In a second terminal, from the repository root:

```bash
cd frontend
npm ci
npm run build
npm start
```

Open **http://localhost:3000/grid**. The default Next.js rewrite proxies `/api` to `http://127.0.0.1:8000`. For a different backend, set `NEXT_PUBLIC_BACKEND_URL` to a URL reachable from the browser **before** starting or building the frontend. Empty/unset keeps the same-origin proxy path.

For development with live code updates, use `npm run dev` instead of the build/start steps. Stop the running frontend before switching modes on the same port.

Other pages remain available: `/` for PlateVision, `/camera` for browser-camera input, `/upload` for files, and `/cctv` for stream controls.

### Verify the active engines

- `http://127.0.0.1:8000/api/health` reports the core detector.
- `http://127.0.0.1:8000/api/grid/system/status` reports camera processing and the selected detector/appearance backends.
- `http://127.0.0.1:8000/api/docs` provides interactive API documentation.
- `http://127.0.0.1:8000/api/openapi.json` provides the schema.

Without weights the web application can use mock/classical fallbacks. Check the reported backends and OCR startup messages; an HTTP 200 health response alone does not demonstrate trained OCR or accurate recognition.

The verified core model bundle described below also installs the recommended plate detector at `models/plate_detector.onnx` for the web adapter. This does not make the web pipeline equivalent to the separately measured console, and does not replace Grid's vehicle or appearance adapter configuration.

If overriding `ONNX_MODEL_PATH` in an environment file, resolve it from the backend's working directory: use `../models/plate_detector.onnx` when starting from `backend/`, or an absolute path on your own machine.

Follow [the demo walkthrough](DEMO.md) once the services are running.

## PlateVision desktop console

Stop/deactivate the web Python environment before creating a separate console environment. From the repository root:

```bash
python -m venv .venv-console
```

Activate with `source .venv-console/bin/activate` on macOS/Linux, or `./.venv-console/Scripts/Activate.ps1` on Windows PowerShell. Then:

```bash
python -m pip install --upgrade pip
python -m pip install -r desktop-console/platevision/backend/requirements.txt
python scripts/setup_demo_models.py --bundle core
python scripts/setup_demo_models.py --bundle core --check
python scripts/run_console.py --model recommended --check
python scripts/run_console.py --model recommended
```

The launcher's `--check` verifies dependencies/models and prints settings without starting services. The normal launcher starts the API on **8003** and the static console on **8080**, prints the correct browser URL, and stops its child services on Ctrl+C. It does not start or reconfigure Drishti Grid's API on port 8000.

To compare the trained YOLO11n candidate, stop the console and run:

```bash
python scripts/run_console.py --model yolo11n
```

The candidate API defaults to **8004**; use the URL printed by the launcher. Ports can be changed with `--port` and `--web-port`. If selecting an API by hand, set the console's **Model API address** to `http://127.0.0.1:8003` for the recommended model or `http://127.0.0.1:8004` for the candidate.

`--device auto` is the default. Use `--device cpu` for a CPU run or `--device gpu` to request the accelerator. Confirm the actual runtime at `/api/health`; the published GPU latency must not be applied to a CPU run.

### Models and offline installation

Release: [v2026.09.28-models](https://github.com/tirthp008-alt/platevision/releases/tag/v2026.09.28-models).

| Bundle | Purpose |
|---|---|
| `platevision-core-models.zip` | Recommended green plate detector, YOLO11n candidate, and UVH vehicle detector |
| `platevision-reid-models.zip` | Optional console vehicle-appearance model |

To install a previously downloaded core archive:

```bash
python scripts/setup_demo_models.py --bundle core --archive /path/to/platevision-core-models.zip
```

Install optional appearance matching with `python scripts/setup_demo_models.py --bundle reid`. It is not needed for photo OCR or video export. Existing files are checked; `--replace` explicitly permits replacing an installed bundle. See [model documentation](../models/README.md) for the manifest, hashes, sources, and licensing information. Datasets and private audit images are not distributed with the models.

## Checks

From the repository root, installer and launcher unit checks require only pytest and do not download models, run inference, or start live services:

```bash
python -m pytest tests/test_demo_models.py tests/test_run_console.py -q -p no:cacheprovider
```

For the web app, use its Python environment and run from `backend/`:

```bash
python -m pytest -q
```

From `frontend/`:

```bash
npm test
npm run build
```

For console backend and training/evaluation-tool checks, use the console Python environment and run from `desktop-console/`.

macOS/Linux:

```bash
PYTHONPATH=platevision/backend python -m pytest platevision/backend/tests tests -q -p no:cacheprovider
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = 'platevision/backend'
python -m pytest platevision/backend/tests tests -q -p no:cacheprovider
```

Pure client checks run from `desktop-console/` with Node:

```bash
node --test tests/video-analysis.test.cjs tests/region-mode.test.cjs tests/reid-ui.test.cjs tests/green-enhancement.test.cjs
```

Model benchmarks and HTTP evaluations perform real inference. Run them separately from training and other GPU workloads; they are not part of these unit checks. See [evaluation](EVALUATION.md) for data requirements and measurement scope.

## Other deployment paths

The existing `scripts/run_dev.sh` / `run_dev.bat` are conveniences for the web app after its environment and npm packages have been installed. Export role keys before using them. The repository's Docker files are a separate deployment path; they do not launch the desktop console or reproduce its GPU benchmarks.

Before using Compose, configure role-key environment variables, writable database storage, model mounts, and a browser-reachable backend URL. A Docker service name such as `backend` is not a public browser hostname. For a built Next.js frontend, configure public environment variables during the build. Review container health and actual model readiness after startup.

Use a secure reverse proxy, scoped CORS origins, and a suitable authentication and retention policy before connecting remote or real camera sources. This repository is a prototype, not a hardened deployment package.

## Troubleshooting

| Symptom | Check |
|---|---|
| Grid operations return 403 | Export `DRISHTI_ADMIN_KEY` in the backend shell and enter the same key in the dashboard; unknown keys receive viewer permissions |
| Grid has cameras but no journey | Check processing status, recent sightings, active OCR/appearance backends, and fusion evidence; the system may correctly abstain |
| API reports a mock or classical backend | Install/configure compatible weights, restart, and inspect startup logs |
| Console model missing or hash mismatch | Run the bundle installer with `--check`; do not rename an arbitrary detector into the expected model path |
| Port already in use | Stop the old process or choose unused API/web ports; the web and console APIs are separate |
| OpenCV cannot import on Linux | Install the system libraries required by its wheel, commonly `libglib2.0-0` and `libgl1` |
| Camera access unavailable | Use localhost or a secure browser origin and grant camera permission |
| Empty map tiles | Check browser network access; map tiles are separate from local detection and fusion |
