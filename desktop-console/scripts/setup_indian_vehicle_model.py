"""Fetch a pinned IISc Indian-traffic checkpoint and export locally, without training."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = '4a22412775adb6f97f22735647afee976b4638a0'
SOURCE_SHA = '12dbeb8f6fa207d19446fa652fc4313359fbec267027c8c01641004dcf1d0150'
SOURCE = f'https://huggingface.co/iisc-aim/UVH-26/resolve/{REVISION}/weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt'
FOLDER = ROOT / 'platevision/models/indian-vehicles'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    checkpoint = FOLDER / 'UVH-26-MV-YOLOv11-S.pt'
    if not checkpoint.exists():
        if args.export_only:
            parser.error('Download the pinned checkpoint first.')
        partial = checkpoint.with_suffix('.download')
        with urllib.request.urlopen(SOURCE, timeout=90) as response, partial.open('wb') as target:
            while chunk := response.read(1024*1024):
                target.write(chunk)
        if sha256(partial.read_bytes()).hexdigest() != SOURCE_SHA:
            raise RuntimeError('Downloaded weight checksum does not match the pinned IISc artifact.')
        partial.replace(checkpoint)
    if sha256(checkpoint.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError('Checkpoint differs from the pinned source.')
    print(f'Pinned source verified: {checkpoint}', flush=True)
    if args.download_only:
        return
    overlay = ROOT / '.cache/nano-training-deps'
    if overlay.is_dir():
        sys.path.insert(0, str(overlay))
    import torch
    from ultralytics import YOLO
    import onnx
    torch.set_num_threads(4)
    model = YOLO(str(checkpoint))
    print(f'Class vocabulary: {model.names}', flush=True)
    output = Path(model.export(format='onnx', imgsz=640, batch=1, dynamic=True,
                              simplify=False, opset=17, nms=False, device='cpu'))
    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)
    record = dict(source=SOURCE, repository='https://huggingface.co/iisc-aim/UVH-26',
                  revision=REVISION, source_sha256=SOURCE_SHA,
                  onnx_sha256=sha256(output.read_bytes()).hexdigest(),
                  names=model.names, trained_in_this_task=False, task='vehicle_detection',
                  architecture='YOLO11s', taxonomy='uvh', input_size=640,
                  license='IISc repository Apache-2.0; Ultralytics architecture AGPL-3.0',
                  source_weights_are_vehicle_not_plate=True)
    (FOLDER / 'model-card.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
