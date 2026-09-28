"""Run the separate desktop console API and website; Ctrl+C stops both.

Install models with scripts/setup_demo_models.py first. Use the same Python
interpreter that has desktop-console/platevision/backend/requirements.txt.
No browser opens, and the root PlateVision/Drishti web app is unchanged.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from setup_demo_models import checked_file, safe_target

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['recommended', 'yolo11n'], default='recommended')
    parser.add_argument('--port', type=int, help='API port; defaults to 8003 (recommended) or 8004 (candidate).')
    parser.add_argument('--web-port', type=int, default=8080)
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--startup-timeout', type=float, default=180.)
    parser.add_argument('--check', action='store_true', help='Verify models/dependencies and show settings without starting servers.')
    args = parser.parse_args(argv)
    if args.port is None:
        args.port = 8003 if args.model == 'recommended' else 8004
    if not 1 <= args.port <= 65535 or not 1 <= args.web_port <= 65535 or args.port == args.web_port:
        parser.error('API and website ports must be distinct numbers from 1 to 65535.')
    if not 1 <= args.startup_timeout <= 600:
        parser.error('--startup-timeout must be between 1 and 600 seconds.')
    return args


def launch_plan(args, root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'models/manifest.json').read_text(encoding='utf-8'))
    model = manifest['models'][args.model]
    files = {item['path']: item for item in manifest['bundles']['core']['files']}
    # Changing the root web app's compatibility copy must not affect this app.
    for name in model['required_files']:
        if not checked_file(safe_target(root, name), files[name]):
            raise ValueError(f'Missing or changed model artifact: {name}. Run scripts/setup_demo_models.py.')
    console = root / 'desktop-console'
    if args.model == 'recommended':
        command = [sys.executable, '-u', str(console / 'scripts/run_regions_model.py'), '--port', str(args.port)]
    else:
        command = [sys.executable, '-u', str(console / 'scripts/run_plate_checkpoint.py'),
                   '--weights', str(safe_target(root, model['path'])), '--port', str(args.port)]
    website = [sys.executable, '-u', '-m', 'http.server', str(args.web_port),
               '--bind', '127.0.0.1', '--directory', str(console)]
    api_url = f'http://127.0.0.1:{args.port}'
    url = f'http://127.0.0.1:{args.web_port}/?' + urllib.parse.urlencode(
        {'engine': 'api', 'detector': 'regions', 'api': api_url})
    environment = os.environ.copy()
    environment.update(ACCELERATION='openvino' if args.device == 'gpu' else 'auto',
                       OPENVINO_DEVICE='CPU' if args.device == 'cpu' else 'GPU',
                       ONNX_PROVIDER='CPUExecutionProvider',
                       CORS_ORIGINS=f'http://127.0.0.1:{args.web_port},http://localhost:{args.web_port}')
    return dict(api=command, website=website, cwd=console, environment=environment,
                api_url=api_url, url=url, expected_sha256=model['sha256'], label=model['label'])


def check_dependencies():
    missing = [name for name in ['uvicorn', 'fastapi', 'cv2', 'numpy', 'onnx',
                                'rapidocr_onnxruntime', 'imageio_ffmpeg', 'onnxruntime']
               if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError('Missing inference dependencies: ' + ', '.join(missing)
                         + '. Install desktop-console/platevision/backend/requirements.txt with this Python.')


def check_ports(*ports):
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(('127.0.0.1', port))


def wait_ready(processes, plan, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError('A console server exited during startup; inspect its error above.')
        try:
            with urllib.request.urlopen(plan['api_url'] + '/api/health', timeout=2) as response:
                health = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(.25)
            continue
        if health.get('model_sha256') != plan['expected_sha256']:
            raise RuntimeError('API did not load the selected verified model: ' + json.dumps(health.get('errors', [])))
        if not all(health.get(key) for key in ('detector_ready', 'ocr_ready', 'vehicle_detector_ready')):
            raise RuntimeError('API dependencies are not ready: ' + json.dumps(health.get('errors', [])))
        try:
            with urllib.request.urlopen(plan['url'], timeout=2) as response:
                if response.status == 200:
                    return health
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.25)
    raise RuntimeError(f'API startup exceeded {timeout:g} seconds. No server was left running.')


def stop_processes(processes):
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run(plan, timeout):
    processes = []
    try:
        for command in (plan['api'], plan['website']):
            processes.append(subprocess.Popen(command, cwd=plan['cwd'], env=plan['environment'],
                             stdin=subprocess.DEVNULL,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
        health = wait_ready(processes, plan, timeout)
        print(f'\nReady: {plan["label"]} ({health.get("engine", "runtime")})\n{plan["url"]}\n'
              'OCR and angle correction are on. Ctrl+C stops both servers.', flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(.5)
        raise RuntimeError('A console server stopped; shutting down its companion.')
    except KeyboardInterrupt:
        print('\nStopping the desktop console.', flush=True)
    finally:
        stop_processes(processes)


def main(argv=None):
    args = parse_args(argv)
    try:
        plan = launch_plan(args)
        check_dependencies()
        print(f'Selected {plan["label"]}; SHA-256 {plan["expected_sha256"]}', flush=True)
        if args.check:
            print(plan['url'])
            return 0
        check_ports(args.port, args.web_port)
        run(plan, args.startup_timeout)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f'Cannot run desktop console: {error}') from error


if __name__ == '__main__':
    raise SystemExit(main())
