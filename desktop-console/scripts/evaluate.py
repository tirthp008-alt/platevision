"""Evaluate the actual plate/OCR pipeline on labelled development images."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'platevision/backend'))
from app.core.config import settings
from app.services.detector import OnnxPlateDetector
from app.services.ocr import PlateOCR
from app.services.pipeline import process,CropStore
from analyze_scales import iou


def main():
    model_path=ROOT/'platevision/models/plate-detector.onnx'
    model=OnnxPlateDetector(str(model_path));ocr=PlateOCR();store=CropStore()
    tp=fp=fn=0;scenes=[];timings=[]
    for path in sorted((ROOT/'platevision/data/yolo-indian/val/images').glob('*.jpg')):
        result=process(path.read_bytes(),model,store,ocr)
        width,height=result['image']['width'],result['image']['height']
        truth=[]
        for line in (path.parents[1]/'labels'/f'{path.stem}.txt').read_text().splitlines():
            _,cx,cy,w,h=map(float,line.split())
            truth.append(((cx-w/2)*width,(cy-h/2)*height,w*width,h*height))
        labelled=len(truth);matched=0
        for detection in result['detections']:
            box=detection['bounding_box']
            scores=[iou([box['x'],box['y'],box['width'],box['height']],b) for b in truth]
            if scores and max(scores)>=.5:
                truth.pop(scores.index(max(scores)));tp+=1;matched+=1
            else:fp+=1
        fn+=len(truth);timings.append(result['processing_time_ms'])
        scenes.append(dict(file=path.name,labelled=labelled,matched=matched,predicted=len(result['detections']),processing_time_ms=result['processing_time_ms']))
        print(f'{len(scenes)}/50 {path.name}: {matched}/{labelled}',flush=True)
    report=dict(images=len(scenes),model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),iou_threshold=.5,
        configuration=dict(tile_size=settings.plate_tile_size,confidence_threshold=settings.confidence_threshold,
            refinement=settings.plate_refinement_enabled,crop_padding=settings.plate_crop_padding,unverified_threshold=settings.unverified_plate_threshold),
        true_positives=tp,false_positives=fp,false_negatives=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
        multi_plate_scenes=sum(s['labelled']>=2 for s in scenes),complete_multi_plate_scenes=sum(s['labelled']>=2 and s['matched']==s['labelled'] for s in scenes),
        latency_ms=dict(p50=float(np.percentile(timings,50)),p95=float(np.percentile(timings,95))),
        note='Development diagnostic, not independent final accuracy. Measures returned API boxes including crop padding and OCR-supported filtering; no exact-text labels available.',scenes=scenes)
    (ROOT/'tests/accuracy-current.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))


if __name__=='__main__':main()
