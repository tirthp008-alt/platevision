"""Serve a verified local plate ONNX with the regions API on localhost:8004.

Use the app's ?detector=regions view, then set Model API address to
http://127.0.0.1:8004. There is currently no frontend query for port 8004.
OCR (PP-OCRv4), angle correction and bounded vehicle-crop search retain the
regions API defaults. This script neither downloads nor activates weights in
another API process. It uses a checkpoint-specific video output directory.

By default, weights/../model-card.json supplies the expected ONNX SHA-256.
An explicit --sha256 permits a local export without a model card, but never
bypasses the required single-plate-class ONNX metadata. An existing model card
must agree with the actual file even when --sha256 is supplied.
"""
import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import sys

from run_regions_model import ROOT, configure


def checksum(value):
    if not re.fullmatch(r'[0-9a-fA-F]{64}', value):
        raise argparse.ArgumentTypeError('SHA-256 must contain exactly 64 hexadecimal characters.')
    return value.lower()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True, help='Existing local plate ONNX export.')
    parser.add_argument('--port', type=int, default=8004)
    parser.add_argument('--sha256', type=checksum, help='Expected hash; otherwise require an adjacent model card.')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535.')
    return args


def plate_metadata(model):
    """Inspect export metadata without constructing an inference session."""
    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata.get('task') != 'detect':
        raise ValueError('ONNX metadata must identify task=detect.')
    serialized = metadata.get('names', '')
    if not serialized or len(serialized) > 20000:
        raise ValueError('ONNX export is missing valid plate class metadata.')
    try:
        names = ast.literal_eval(serialized)
    except (SyntaxError, ValueError) as error:
        raise ValueError('ONNX class metadata is invalid.') from error
    if isinstance(names, dict):
        if len(names) != 1 or set(names) not in ({0}, {'0'}):
            raise ValueError('ONNX export must contain exactly one class with ID 0.')
        names = list(names.values())
    if not isinstance(names, (list, tuple)) or len(names) != 1 or not isinstance(names[0], str):
        raise ValueError('ONNX export must contain exactly one plate class.')
    normalized = re.sub(r'[^a-z]', '', names[0].lower())
    if normalized not in {'plate', 'numberplate', 'licenseplate', 'licenceplate', 'registrationplate', 'vehicleplate'}:
        raise ValueError('ONNX class metadata does not identify a number-plate detector.')
    return names[0]


def has_external_tensors(message):
    """Include constants and nested subgraphs, not only top-level initializers."""
    if message.DESCRIPTOR.full_name == 'onnx.TensorProto':
        return bool(message.data_location == 1 or message.external_data)
    for field, value in message.ListFields():
        if field.message_type is not None:
            values = value if field.is_repeated else (value,)
            if any(has_external_tensors(item) for item in values):
                return True
    return False


def verify_checkpoint(weights, expected_sha=None):
    raw = str(weights)
    if '://' in raw or raw.startswith(('\\\\', '//')):
        raise ValueError('Only local ONNX files are accepted; no remote weights or downloads.')
    path = Path(weights).expanduser().resolve()
    if path.suffix.lower() != '.onnx' or not path.is_file():
        raise ValueError('An existing local .onnx file is required.')
    # The training script keeps the card beside weights/, while standalone
    # exports may keep it beside the ONNX. Never search unrelated directories.
    card_path = (path.parent.parent if path.parent.name == 'weights' else path.parent) / 'model-card.json'
    card = None
    if card_path.is_file():
        card = json.loads(card_path.read_text(encoding='utf-8'))
        if not isinstance(card, dict) or card.get('weights_retrained') is not True:
            raise ValueError('Model card must identify plate-trained weights (weights_retrained=true).')
        if not isinstance(card.get('architecture'), str) or not card['architecture'].strip():
            raise ValueError('Model card is missing its detector architecture.')
        try:
            card_sha = checksum(card.get('onnx_sha256', ''))
        except (argparse.ArgumentTypeError, TypeError) as error:
            raise ValueError('Model card is missing a valid onnx_sha256.') from error
    elif expected_sha is None:
        raise ValueError(f'Missing model card: {card_path}. Supply --sha256 only for an independently verified local export.')
    if expected_sha is not None:
        expected_sha = checksum(expected_sha)
    data = path.read_bytes()
    actual_sha = sha256(data).hexdigest()
    if card is not None and card_sha != actual_sha:
        raise ValueError('ONNX hash differs from its model card; refusing the changed checkpoint.')
    if expected_sha is not None and expected_sha != actual_sha:
        raise ValueError('ONNX hash differs from --sha256; refusing the changed checkpoint.')
    try:
        import onnx
    except ImportError as error:
        raise ValueError('The API interpreter needs the installed ONNX package for checkpoint verification.') from error
    # Parse the same bytes that were hashed, without loading external tensors.
    try:
        model = onnx.load_model_from_string(data)
    except Exception as error:
        raise ValueError('The local checkpoint is not a readable ONNX model.') from error
    if has_external_tensors(model):
        raise ValueError('External ONNX tensor data is not supported; export a self-contained model.')
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as error:
        raise ValueError(f'Invalid ONNX checkpoint: {error}') from error
    class_name = plate_metadata(model)
    architecture = card['architecture'].strip() if card is not None else 'Local ONNX'
    return dict(path=path, sha256=actual_sha, architecture=architecture, class_name=class_name,
                model_card=str(card_path) if card is not None else None)


def configure_checkpoint(record, port):
    output = f"video-results-checkpoint-{record['sha256'][:12]}-{port}"
    configure(plate=record['path'], plate_sha=record['sha256'],
              detector_name=f"{record['architecture']} local plate candidate ({record['sha256'][:12]})",
              video_output_dir=output)
    return output


def check_port(port):
    # Fail before model warm-up if another API already owns this port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', port))


def main(argv=None):
    args = parse_args(argv)
    try:
        record = verify_checkpoint(args.weights, args.sha256)
        check_port(args.port)
        output = configure_checkpoint(record, args.port)
    except (OSError, ValueError, argparse.ArgumentTypeError) as error:
        raise SystemExit(f'Cannot start candidate API: {error}') from error
    print(json.dumps(dict(weights=str(record['path']), sha256=record['sha256'],
                          model_card=record['model_card'], video_output_dir=output,
                          api=f'http://127.0.0.1:{args.port}', read_text=True,
                          angle_correction=True, vehicle_taxonomy='uvh',
                          vehicle_plate_max_crops=8), indent=2), flush=True)
    print(f'Open the frontend with ?detector=regions, then set Model API address to http://127.0.0.1:{args.port}.', flush=True)
    backend = ROOT / 'platevision/backend'
    os.chdir(backend)
    sys.path.insert(0, str(backend))
    import uvicorn
    uvicorn.run('app.main:app', host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
