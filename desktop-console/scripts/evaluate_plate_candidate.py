"""Compare baseline and candidate plate boxes on held-out whole images.

Uses the same runtime, confidence, NMS, resolution and IoU rule for both models.
No OCR transcripts exist for the green archive: this is localization accuracy.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'platevision/backend'))
from app.services.detector import OpenVINOPlateDetector


def overlap(a,b):
    ax,ay,aw,ah=a[:4];bx,by,bw,bh=b[:4]
    intersection=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return intersection/max(1e-9,aw*ah+bw*bh-intersection)


def match(predictions,truth):
    remaining=list(truth);matched=[];unmatched=[]
    for box in sorted(predictions,key=lambda b:b[4],reverse=True):
        scores=[overlap(box,t) for t in remaining]
        if scores and max(scores)>=.5:
            matched.append(box);remaining.pop(scores.index(max(scores)))
        else:unmatched.append(box)
    return len(matched),len(unmatched),len(remaining)


def load_truth(label,width,height):
    truth=[]
    for line in label.read_text().splitlines():
        cls,cx,cy,w,h=map(float,line.split())
        if cls!=0:raise ValueError('Only plate class 0 is supported')
        truth.append([(cx-w/2)*width,(cy-h/2)*height,w*width,h*height])
    return truth


def load_detector(path,high_resolution=True):
    detector=OpenVINOPlateDetector(str(path))
    if high_resolution:
        detector.fast_models={
            'landscape':OpenVINOPlateDetector(str(path),input_shape=(864,1536)),
            'portrait':OpenVINOPlateDetector(str(path),input_shape=(1536,864))}
    return detector


def evaluate(model_path,dataset,split,high_resolution):
    detector=load_detector(model_path,high_resolution)
    metadata=json.loads((dataset/'manifest.json').read_text())
    green={r['id']:r for r in metadata['green_records']}
    scenes=[]
    for path in sorted((dataset/split/'images').glob('*.jpg')):
        # Context crops are for training/selection, never full-scene reporting.
        if '_context_' in path.stem:continue
        image=cv2.imread(str(path));h,w=image.shape[:2]
        truth=load_truth(dataset/split/'labels'/f'{path.stem}.txt',w,h)
        # Include letterbox, inference, NMS and the app's plate merge/filter.
        tick=time.perf_counter()
        boxes=detector.detect_fast(image) if high_resolution else detector.merge_plates(detector.detect(image))
        elapsed=(time.perf_counter()-tick)*1000
        tp,unmatched,fn=match(boxes,truth)
        category='green' if path.stem in green else 'general'
        complete=category!='green' or green[path.stem]['full_scene_reviewed']
        scenes.append(dict(image=path.name,category=category,complete_labels=complete,
            labelled=len(truth),matched=tp,missed=fn,unmatched=unmatched,predictions=[list(b) for b in boxes],
            all_labelled_found=fn==0,detection_ms=elapsed))
    categories={}
    for category in ('green','general'):
        rows=[r for r in scenes if r['category']==category]
        tp=sum(r['matched'] for r in rows);fn=sum(r['missed'] for r in rows)
        complete=[r for r in rows if r['complete_labels']]
        ctp=sum(r['matched'] for r in complete);fp=sum(r['unmatched'] for r in complete)
        categories[category]=dict(images=len(rows),labelled=tp+fn,matched=tp,missed=fn,
            recall=tp/max(1,tp+fn),all_labelled_found=sum(r['all_labelled_found'] for r in rows),
            precision_on_reviewed_scenes=ctp/max(1,ctp+fp),precision_scene_count=len(complete),
            false_positives_on_reviewed_scenes=fp)
    return dict(model=str(model_path),sha256=sha256(model_path.read_bytes()).hexdigest(),
        runtime='OpenVINO GPU',input_resolution='1536x864 / 864x1536' if high_resolution else '640x640',
        confidence=.25,iou_threshold=.5,split=split,categories=categories,scenes=scenes,
        notes=['Whole images only; ground-truth boxes are not supplied to the detector.',
               'Green recall uses provided target annotations; scene-wide precision is omitted for incompletely labelled scenes.',
               'Confidence scores are uncalibrated. No exact-text OCR accuracy is claimed.',
               'Per-scene timings are diagnostic; run the isolated benchmark after training for performance claims.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--resolution',choices=['640','1536'],default='1536')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=evaluate(args.model,args.dataset,args.split,args.resolution=='1536')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))
