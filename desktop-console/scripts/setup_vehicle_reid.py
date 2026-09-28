"""Download and strictly export the official VeRi FastReID checkpoint; no training.

Run with .venv-train/Scripts/python.exe. Then use --verify-runtime with .venv
Python to compare OpenVINO/ONNX against the exported PyTorch references.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'platevision/models/reid'
CACHE = ROOT / '.cache/fastreid-source'
DEPS = ROOT / '.cache/fastreid-deps'
REVISION = 'c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d'
SOURCE_SHA = 'ead96dfb58cbe906a9277e4d00a6fdf1a51ee0ba6933bd62041d8b7da1837dad'
CHECKPOINT_SHA = '57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd'
SOURCE_URL = f'https://github.com/JDAI-CV/fast-reid/archive/{REVISION}.zip'
CHECKPOINT_URL = 'https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth'
ONNX_PATH = TARGET / 'veri-sbs-r50-ibn.onnx'


def checked_download(url, path, digest):
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.download')
        print(f'Downloading {url}', flush=True)
        urllib.request.urlretrieve(url, temporary)
        if sha256(temporary.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Hash mismatch for {url}; original path was not replaced.')
        temporary.replace(path)
    if sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'Hash mismatch: {path}')


def setup_source():
    TARGET.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / f'{REVISION}.zip'
    checked_download(SOURCE_URL, archive, SOURCE_SHA)
    source = CACHE / f'fast-reid-{REVISION}'
    # Re-extract the pinned files so export cannot silently use modified code.
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            if not (CACHE / item.filename).resolve().is_relative_to(CACHE.resolve()):
                raise RuntimeError('Unsafe source archive path.')
        zipped.extractall(CACHE)
    checkpoint = TARGET / 'veri_sbs_R50-ibn.pth'
    checked_download(CHECKPOINT_URL, checkpoint, CHECKPOINT_SHA)
    if not (DEPS / 'yacs').is_dir() or not (DEPS / 'termcolor').is_dir() or not (DEPS / 'tabulate').is_dir():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(DEPS), '--no-deps',
                        'yacs==0.1.8', 'termcolor==2.5.0', 'tabulate==0.9.0'], check=True)
    sys.path[:0] = [str(DEPS), str(source), str(ROOT / 'platevision/backend')]
    return source, checkpoint


def export():
    source, checkpoint = setup_source()
    import cv2
    import numpy as np
    import onnx
    import torch
    from fastreid.config import get_cfg
    from fastreid.modeling import build_model
    from app.services.vehicle_reid import prepare_crop
    torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    cfg = get_cfg()
    cfg.merge_from_file(str(source / 'configs/VeRi/sbs_R50-ibn.yml'))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = state['heads.classifier.weight'].shape[0]
    model = build_model(cfg).eval()
    # Exact legacy release -> current official source key migrations.
    migrated = dict(state)
    migrated['heads.weight'] = migrated.pop('heads.classifier.weight')
    migrated['heads.bottleneck.0.num_batches_tracked'] = migrated.pop('heads.bnneck.num_batches_tracked')
    for name in ('pixel_mean', 'pixel_std'):
        value = migrated.pop(name)
        if not torch.allclose(value, getattr(model, name), atol=1e-6, rtol=0):
            raise RuntimeError(f'Official config normalization differs from checkpoint: {name}')
        getattr(model, name).copy_(value)
    # All learned tensors and all persistent buffers must match exactly.
    model.load_state_dict(migrated, strict=True)

    class Embeddings(torch.nn.Module):
        def __init__(self, baseline):
            super().__init__()
            self.baseline = baseline

        def forward(self, images):
            normalized = (images - self.baseline.pixel_mean) / self.baseline.pixel_std
            embedding = self.baseline.heads(self.baseline.backbone(normalized))
            return torch.nn.functional.normalize(embedding, p=2, dim=1)

    wrapped = Embeddings(model).eval()
    dummy = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    print('Strict checkpoint load succeeded. Exporting official architecture.', flush=True)
    with torch.inference_mode():
        torch.onnx.export(wrapped, dummy, str(ONNX_PATH), input_names=['images_rgb_255'],
                          output_names=['embedding'], opset_version=17, do_constant_folding=True,
                          dynamo=False)
    graph = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(graph)
    onnx.helper.set_model_props(graph, dict(model='FastReID VeRi SBS R50-IBN',
        source_revision=REVISION, checkpoint_sha256=CHECKPOINT_SHA,
        input='RGB FP32 NCHW 0..255; PIL bicubic resize256x256', output='L2-normalized2048D'))
    onnx.save(graph, str(ONNX_PATH))
    fixture = cv2.imread(str(ROOT / 'tests/three-vehicle-composite.jpg'))
    if fixture is None:
        raise RuntimeError('Missing vehicle fixture for mandatory export parity check.')
    cell = fixture.shape[1] // 3
    crops = [fixture[:, :cell], fixture[5:-5, 8:cell-8], fixture[:, cell:2*cell], fixture[:, 2*cell:]]
    inputs = np.concatenate([prepare_crop(crop) for crop in crops])
    with torch.inference_mode():
        outputs = np.concatenate([wrapped(torch.from_numpy(tensor[None])).numpy() for tensor in inputs])
    np.savez_compressed(TARGET / 'parity-fixture.npz', inputs=inputs, expected=outputs)
    card = dict(model='FastReID SBS R50-IBN', task='vehicle appearance re-identification',
        training_dataset='VeRi', source_revision=REVISION, source_url=SOURCE_URL,
        source_archive_sha256=SOURCE_SHA, checkpoint_url=CHECKPOINT_URL,
        checkpoint_sha256=CHECKPOINT_SHA, onnx_sha256=sha256(ONNX_PATH.read_bytes()).hexdigest(),
        config='configs/VeRi/sbs_R50-ibn.yml', export_opset=17, torch_version=torch.__version__,
        strict_load=True, learned_tensors_skipped=0, state_key_migrations={
            'heads.classifier.weight':'heads.weight',
            'heads.bnneck.num_batches_tracked':'heads.bottleneck.0.num_batches_tracked'},
        normalization_buffers='Checkpoint pixel_mean/std checked and copied; nonpersistent in current official source.',
        input_shape=[1,3,256,256], input_color='RGB', resize='PIL bicubic',
        input_range=[0,255], mean=model.pixel_mean.flatten().tolist(), std=model.pixel_std.flatten().tolist(),
        embedding_dimension=2048, l2_normalized=True, training_performed=False,
        published_benchmark=dict(dataset='VeRi',rank1_percent=97.0,mAP_percent=81.9,
            source='https://github.com/JDAI-CV/fast-reid/blob/'+REVISION+'/MODEL_ZOO.md',
            note='Published official benchmark; not measured accuracy on this CCTV system.'),
        limitations=['Uncalibrated cosine similarity is not a verified identity.',
            'No multi-camera identity-labelled CCTV validation has been supplied.',
            'Published VeRi vehicle performance does not establish two/three-wheeler performance.',
            'Use embeddings on bounded keyframes; this model adds inference cost.'],
        source_license='Apache-2.0', parity_status='awaiting_runtime_verification')
    (TARGET / 'model-card.json').write_text(json.dumps(card, indent=2), encoding='utf-8')
    print(json.dumps({'export':str(ONNX_PATH),'onnx_sha256':card['onnx_sha256'],
                      'parameters':sum(p.numel() for p in model.parameters())}, indent=2), flush=True)


def verify(device):
    import time
    import numpy as np
    import onnxruntime as ort
    sys.path.insert(0, str(ROOT / 'platevision/backend'))
    from app.services.vehicle_reid import VehicleReID
    fixture = np.load(TARGET / 'parity-fixture.npz')
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(ONNX_PATH), sess_options=options, providers=['CPUExecutionProvider'])
    actual = np.concatenate([session.run(None, {'images_rgb_255':tensor[None]})[0] for tensor in fixture['inputs']])
    expected = fixture['expected']
    difference = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=2e-4, rtol=2e-3):
        raise RuntimeError(f'PyTorch/ONNX export parity failed: max_abs={difference}')
    runtime = VehicleReID(ONNX_PATH, device=device, _allow_unverified=True)
    crops = [np.ascontiguousarray(t.transpose(1,2,0)[:,:,::-1], dtype=np.uint8) for t in fixture['inputs']]
    embeddings = runtime.embed(crops)
    cosine_parity = np.sum(embeddings * expected, axis=1)
    if float(cosine_parity.min()) < .999:
        raise RuntimeError(f'OpenVINO/PyTorch cosine parity failed: {cosine_parity.tolist()}')
    times = []
    for _ in range(10):
        start=time.perf_counter();runtime.embed(crops[:1]);times.append((time.perf_counter()-start)*1000)
    metrics = dict(onnx_max_absolute_error=difference, openvino_min_cosine_to_torch=float(cosine_parity.min()),
        device=device, runtime=runtime.runtime, same_vehicle_perturbed_crop_cosine=float(embeddings[0]@embeddings[1]),
        different_vehicle_cosines=[float(embeddings[0]@embeddings[2]),float(embeddings[0]@embeddings[3])],
        timing_runs=10, crop_embedding_median_ms=float(np.median(times)),crop_embedding_p95_ms=float(np.percentile(times,95)),
        note='Sanity check on three source vehicles and a perturbed crop, not cross-camera matching accuracy or calibrated threshold.')
    card_path=TARGET/'model-card.json';card=json.loads(card_path.read_text(encoding='utf-8'))
    card.update(parity_status='passed',verification=metrics)
    card_path.write_text(json.dumps(card,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-runtime', action='store_true')
    parser.add_argument('--device',default='GPU')
    args=parser.parse_args()
    verify(args.device) if args.verify_runtime else export()
