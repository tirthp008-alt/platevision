"""Local tiny ONNX metadata checks; no inference, GPU use, downloads or services."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import onnx
from onnx import TensorProto, helper
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_plate_checkpoint as launcher
import run_regions_model as regions


def checkpoint(tmp_path, *, names="{0: 'number_plate'}", task='detect', card=True, external=False):
    path = tmp_path / 'run/weights/best.onnx'
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = helper.make_tensor('boxes', TensorProto.FLOAT, [1, 5, 1], [0.] * 5)
    if external:
        tensor.data_location = TensorProto.EXTERNAL
        item = tensor.external_data.add()
        item.key, item.value = 'location', 'unverified-tensor.bin'
    graph = helper.make_graph([helper.make_node('Constant', [], ['output'], value=tensor)], 'fixture',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 5, 1])])
    model = helper.make_model(graph)
    metadata = {}
    if names is not None:
        metadata['names'] = names
    if task is not None:
        metadata['task'] = task
    helper.set_model_props(model, metadata)
    path.write_bytes(model.SerializeToString())
    digest = sha256(path.read_bytes()).hexdigest()
    record = dict(architecture='yolo11n', weights_retrained=True, onnx_sha256=digest)
    if card:
        (path.parent.parent / 'model-card.json').write_text(json.dumps(record), encoding='utf-8')
    return path, digest, record


def test_default_port_and_invalid_options():
    args = launcher.parse_args(['--weights', 'candidate.onnx'])
    assert args.port == 8004 and args.sha256 is None
    for option in (['--port', '0'], ['--port', '65536'], ['--sha256', 'bad']):
        with pytest.raises(SystemExit):
            launcher.parse_args(['--weights', 'candidate.onnx', *option])


def test_model_card_and_actual_bytes_are_verified_without_inference(tmp_path):
    path, digest, _ = checkpoint(tmp_path)
    record = launcher.verify_checkpoint(path)
    assert record['sha256'] == digest
    assert record['path'] == path.resolve()
    assert record['architecture'] == 'yolo11n'
    assert record['class_name'] == 'number_plate'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='differs from its model card'):
        launcher.verify_checkpoint(path)


def test_explicit_hash_can_replace_missing_card_but_not_override_conflicting_card(tmp_path):
    path, digest, _ = checkpoint(tmp_path, card=False)
    with pytest.raises(ValueError, match='Missing model card'):
        launcher.verify_checkpoint(path)
    assert launcher.verify_checkpoint(path, digest.upper())['model_card'] is None
    with pytest.raises(ValueError, match='differs from --sha256'):
        launcher.verify_checkpoint(path, '0' * 64)
    card = dict(architecture='yolo11n', weights_retrained=True, onnx_sha256='0' * 64)
    (path.parent.parent / 'model-card.json').write_text(json.dumps(card), encoding='utf-8')
    with pytest.raises(ValueError, match='differs from its model card'):
        launcher.verify_checkpoint(path, digest)


@pytest.mark.parametrize('names,task', [
    (None, 'detect'), ("{0: 'car'}", 'detect'), ("{0: 'dinner_plate'}", 'detect'),
    ("{0: 'plate', 1: 'car'}", 'detect'), ("{1: 'plate'}", 'detect'),
    ("{0: 'plate'}", None), ("{0: 'plate'}", 'classify'),
])
def test_missing_or_nonplate_metadata_is_rejected_even_with_explicit_hash(tmp_path, names, task):
    path, digest, _ = checkpoint(tmp_path, names=names, task=task, card=False)
    with pytest.raises(ValueError):
        launcher.verify_checkpoint(path, digest)


def test_external_constant_tensor_is_rejected_before_reading_sidecar(tmp_path):
    path, digest, _ = checkpoint(tmp_path, external=True)
    with pytest.raises(ValueError, match='External ONNX tensor data'):
        launcher.verify_checkpoint(path, digest)


@pytest.mark.parametrize('key,value', [('weights_retrained', False), ('architecture', ''), ('onnx_sha256', '')])
def test_incomplete_training_card_cannot_be_bypassed(tmp_path, key, value):
    path, digest, card = checkpoint(tmp_path)
    card[key] = value
    (path.parent.parent / 'model-card.json').write_text(json.dumps(card), encoding='utf-8')
    with pytest.raises(ValueError):
        launcher.verify_checkpoint(path, digest)


def test_shared_regions_configuration_and_candidate_output_are_isolated(tmp_path, monkeypatch):
    path, digest, _ = checkpoint(tmp_path)
    vehicle = tmp_path / 'uvh.onnx'
    vehicle.write_bytes(b'local vehicle fixture')
    monkeypatch.setattr(regions, 'VEHICLE', vehicle)
    monkeypatch.setattr(regions, 'PLATE', path)
    monkeypatch.setattr(regions, 'PLATE_SHA', digest)
    environment = {'MODEL_PATH': 'existing/default.onnx', 'VEHICLE_PLATE_MAX_CROPS': '64'}
    monkeypatch.setattr(os, 'environ', environment)
    regions.configure()
    defaults = environment.copy()
    record = launcher.verify_checkpoint(path)
    output = launcher.configure_checkpoint(record, 8004)
    assert output == f'video-results-checkpoint-{digest[:12]}-8004'
    for key, value in defaults.items():
        if key not in {'DETECTOR_NAME', 'VIDEO_OUTPUT_DIR'}:
            assert environment[key] == value
    assert environment['VEHICLE_TAXONOMY'] == 'uvh'
    assert environment['VEHICLE_CONTEXT_ENABLED'] == 'true'
    assert environment['VEHICLE_PLATE_SEARCH_ENABLED'] == 'true'
    assert environment['VEHICLE_PLATE_MAX_CROPS'] == '8'
    assert environment['REGION_SUPPLEMENT_PATH'] == ''
    assert environment['REGION_CONFIDENCE_THRESHOLD'] == '.25'
    assert 'candidate' in environment['DETECTOR_NAME']
    assert sha256(path.read_bytes()).hexdigest() == digest
    assert list(path.parent.iterdir()) == [path]


def test_launcher_starts_only_local_candidate_after_checks(tmp_path, monkeypatch):
    path, digest, _ = checkpoint(tmp_path)
    events = []
    backend = tmp_path / 'platevision/backend'
    backend.mkdir(parents=True)
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'check_port', lambda port: events.append(('port', port)))
    monkeypatch.setattr(launcher, 'configure_checkpoint', lambda record, port: events.append(('configure', record['sha256'])) or 'isolated')
    monkeypatch.setattr(os, 'chdir', lambda directory: events.append(('cwd', directory)))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    fake = ModuleType('uvicorn')
    fake.run = lambda app, **kwargs: events.append(('serve', app, kwargs))
    monkeypatch.setitem(sys.modules, 'uvicorn', fake)
    launcher.main(['--weights', str(path)])
    assert events == [('port', 8004), ('configure', digest), ('cwd', backend),
                      ('serve', 'app.main:app', dict(host='127.0.0.1', port=8004, access_log=False))]


def test_busy_port_fails_before_configuration_or_api_import(tmp_path, monkeypatch):
    path, _, _ = checkpoint(tmp_path)
    def busy(port):
        raise OSError('Address already in use')
    monkeypatch.setattr(launcher, 'check_port', busy)
    monkeypatch.setattr(launcher, 'configure_checkpoint', lambda *_: pytest.fail('Must not configure an occupied port'))
    with pytest.raises(SystemExit, match='Address already in use'):
        launcher.main(['--weights', str(path)])


@pytest.mark.parametrize('path', ['missing.onnx', 'https://example.com/model.onnx', r'\\server\share\model.onnx'])
def test_only_existing_local_onnx_files_are_accepted(path):
    with pytest.raises(ValueError):
        launcher.verify_checkpoint(path, '0' * 64)
