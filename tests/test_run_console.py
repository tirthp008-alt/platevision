"""Launcher behavior without loading a detector or opening a GUI."""
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_console as console


def checkout(tmp_path):
    files = []
    paths = ['desktop-console/platevision/models/green.onnx',
             'desktop-console/platevision/models/new.onnx',
             'desktop-console/platevision/models/new-card.json',
             'desktop-console/platevision/models/vehicle.onnx']
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = name.encode()
        path.write_bytes(payload)
        files.append(dict(path=name, size_bytes=len(payload), sha256=sha256(payload).hexdigest()))
    models = {name: dict(path=paths[index], sha256=files[index]['sha256'], label=name,
                        required_files=[paths[index], paths[3]] + ([paths[2]] if index else []))
              for name,index in [('recommended',0), ('yolo11n',1)]}
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/manifest.json').write_text(json.dumps(dict(models=models,bundles={'core':{'files':files}})))
    return tmp_path


@pytest.mark.parametrize('name,port,script', [('recommended',8003,'run_regions_model.py'),
                                           ('yolo11n',8004,'run_plate_checkpoint.py')])
def test_model_choice_uses_absolute_paths_same_interpreter_and_correct_api(tmp_path, name, port, script):
    root = checkout(tmp_path)
    plan = console.launch_plan(console.parse_args(['--model', name]), root)
    assert plan['api'][0] == plan['website'][0] == sys.executable
    assert Path(plan['api'][2]).is_absolute() and Path(plan['api'][2]).name == script
    assert plan['cwd'] == root / 'desktop-console'
    assert parse_qs(urlparse(plan['url']).query)['api'] == [f'http://127.0.0.1:{port}']
    assert '--weights' in plan['api'] if name == 'yolo11n' else '--weights' not in plan['api']


def test_custom_ports_cpu_and_web_compatibility_copy_are_independent(tmp_path):
    root = checkout(tmp_path)
    (root / 'models/plate_detector.onnx').write_bytes(b'different-root-web-model')
    plan = console.launch_plan(console.parse_args(['--port','8013','--web-port','8090','--device','cpu']),root)
    assert plan['environment']['OPENVINO_DEVICE'] == 'CPU'
    assert plan['environment']['ACCELERATION'] == 'auto'
    assert '8090' in plan['environment']['CORS_ORIGINS']
    assert plan['api_url'].endswith(':8013')


@pytest.mark.parametrize('args', [['--port','0'], ['--port','8080'], ['--web-port','65536'],
                                ['--startup-timeout','nan'], ['--startup-timeout','0']])
def test_invalid_ports_and_wait_limits_are_rejected(args):
    with pytest.raises(SystemExit): console.parse_args(args)


def test_changed_model_fails_before_process_start(tmp_path):
    root = checkout(tmp_path)
    (root / 'desktop-console/platevision/models/green.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Missing or changed'):
        console.launch_plan(console.parse_args([]),root)


class Process:
    def __init__(self): self.running=True; self.terminated=False
    def poll(self): return None if self.running else 0
    def terminate(self): self.running=False; self.terminated=True
    def wait(self, timeout): return 0


def test_health_rejects_another_model_even_when_api_is_ready(monkeypatch):
    health=dict(model_sha256='wrong',detector_ready=True,ocr_ready=True,vehicle_detector_ready=True)
    monkeypatch.setattr(console.urllib.request,'urlopen',lambda *args,**kwargs:io.BytesIO(json.dumps(health).encode()))
    with pytest.raises(RuntimeError,match='selected verified model'):
        console.wait_ready([Process()],dict(api_url='http://localhost:8013',expected_sha256='expected'),1)


@pytest.mark.parametrize('failure', [KeyboardInterrupt, RuntimeError])
def test_shutdown_cleans_only_owned_children_on_interrupt_or_startup_failure(monkeypatch, failure):
    children=[]
    def spawn(command, **kwargs):
        assert kwargs['stdin'] == subprocess.DEVNULL
        child=Process();children.append(child);return child
    def fail(*args): raise failure('fixture')
    monkeypatch.setattr(console.subprocess,'Popen',spawn)
    monkeypatch.setattr(console,'wait_ready',fail)
    plan=dict(api=['api'],website=['web'],cwd=Path('.'),environment={})
    if failure is KeyboardInterrupt:
        console.run(plan,1)
    else:
        with pytest.raises(RuntimeError): console.run(plan,1)
    assert len(children)==2 and all(child.terminated for child in children)
