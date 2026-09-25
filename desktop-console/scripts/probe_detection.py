"""Cache low-threshold model predictions for reproducible scale diagnostics."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platevision/backend'))
from app.core.config import settings
from app.services.detector import OnnxPlateDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--sizes', type=int, nargs='+', default=[480, 640])
    args = parser.parse_args()
    settings.confidence_threshold = .08
    model = OnnxPlateDetector(str(ROOT / 'platevision/models/plate-detector.onnx'))
    output = ROOT / 'tests/scale-diagnostic-cache.json'
    cached = json.loads(output.read_text()) if output.exists() else {}
    paths = sorted((ROOT / 'platevision/data/yolo-indian/val/images').glob('*.jpg'))
    for index, path in enumerate(paths[:args.limit]):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        scene = cached.setdefault(path.name, {})
        if 'truth' not in scene:
            scene['truth'] = []
            for line in (path.parents[1] / 'labels' / f'{path.stem}.txt').read_text().splitlines():
                _, x, y, w, h = map(float, line.split())
                scene['truth'].append([(x-w/2)*width, (y-h/2)*height, w*width, h*height])
        if 'full' not in scene:
            scene['full'] = model.detect(image)
        for size in args.sizes:
            if str(size) in scene:
                continue
            start = time.perf_counter()
            boxes = []
            for top in model.tile_starts(height, size, .25):
                for left in model.tile_starts(width, size, .25):
                    tile = image[top:top+size, left:left+size]
                    th, tw = tile.shape[:2]
                    for x,y,w,h,score in model.detect(tile):
                        if ((left>0 and x<=2) or (top>0 and y<=2)
                            or (left+tw<width and x+w>=tw-2)
                            or (top+th<height and y+h>=th-2)):
                            continue
                        boxes.append([x+left,y+top,w,h,score])
            scene[str(size)] = boxes
            scene[f'{size}_ms'] = round((time.perf_counter()-start)*1000, 1)
        output.write_text(json.dumps(cached))
        print(f'{index+1}/{min(args.limit,len(paths))}: {path.name}', flush=True)


if __name__ == '__main__':
    main()
