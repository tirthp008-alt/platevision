"""Compare old/new pipelines on the same labelled development set and weights.

The legacy path reproduces the previous threshold, merge and crop padding.
No text accuracy is inferred from these bounding-box annotations.
"""
import json
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'platevision/backend'))
from app.services.pipeline import process, CropStore
from app.services.ocr import PlateOCR
from app.services.detector import OnnxPlateDetector
from analyze_scales import iou


def legacy_boxes(scene):
    boxes=[b for key in ['full','640'] for b in scene[key] if b[4]>=.35]
    kept=[]
    for box in sorted(boxes,key=lambda b:b[2]*b[3],reverse=True):
        x,y,w,h,score=box
        duplicate=False
        for bx,by,bw,bh,_ in kept:
            area=max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by))
            if area/max(1,w*h+bw*bh-area)>.4 or (area/max(1,min(w*h,bw*bh))>.8 and min(w*h,bw*bh)/max(1,max(w*h,bw*bh))>=.2):
                duplicate=True
                break
        if not duplicate:kept.append(box)
    return sorted(kept,key=lambda b:b[4],reverse=True)[:100]


def score(truth, boxes):
    remaining=list(truth);tp=0
    for box in boxes:
        scores=[iou(box,b) for b in remaining]
        if scores and max(scores)>=.5:
            remaining.pop(scores.index(max(scores)));tp+=1
    return tp,len(boxes)-tp,len(remaining)


def main():
    import app.services.pipeline as pipeline
    from app.core.config import settings
    parser=argparse.ArgumentParser()
    parser.add_argument('--refine',action='store_true')
    parser.add_argument('--extra-scale',type=int)
    parser.add_argument('--padding',type=float,default=settings.plate_crop_padding)
    args=parser.parse_args()
    settings.plate_crop_padding=args.padding
    cache=json.loads((ROOT/'tests/scale-diagnostic-cache.json').read_text())
    reader=PlateOCR();store=CropStore()
    output=ROOT/'tests/accuracy-comparison.json'
    scenes=[];totals={name:[0,0,0] for name in ['before','after']}
    for name,scene in cache.items():
        # Baseline API returned boxes expanded by 8%, independent of OCR output.
        old=[pipeline.clamp(b[:4],1920,1080,padding=.08) for b in legacy_boxes(scene)]
        class CachedDetector:
            def detect_plates(self,image):
                scales=['full',str(settings.plate_tile_size)]+(['refined'] if args.refine else [])
                if args.extra_scale:scales.append(str(args.extra_scale))
                return OnnxPlateDetector.merge_plates([b for scale in scales
                    for b in scene[scale] if b[4]>=settings.confidence_threshold])
        result=process((ROOT/'platevision/data/yolo-indian/val/images'/name).read_bytes(),CachedDetector(),store,reader)
        new=[list(d['bounding_box'].values()) for d in result['detections']]
        scores={key:score(scene['truth'],boxes) for key,boxes in [('before',old),('after',new)]}
        for key,values in scores.items():totals[key]=[a+b for a,b in zip(totals[key],values)]
        scenes.append(dict(file=name,ground_truth=len(scene['truth']),scores=scores,readings=[dict(box=d['bounding_box'],text=d['normalized_text'],confidence=d['ocr_confidence'],status=d['format_status'],detection=d['detection_confidence']) for d in result['detections']]))
        print(f'{len(scenes)}/{len(cache)} {name}: {scores}',flush=True)
    report=dict(images=len(scenes),iou_threshold=.5,model='plate-detector.onnx (unchanged)',configuration=dict(tile_size=settings.plate_tile_size,extra_scale=args.extra_scale,threshold=settings.confidence_threshold,padding=settings.plate_crop_padding,refinement=args.refine),note='Development diagnostic, not independent test accuracy. Measures returned plate boxes, not exact text recognition. Before/after both include actual API crop padding.',scenes=scenes)
    for key,(tp,fp,fn) in totals.items():
        report[key]=dict(true_positives=tp,false_positives=fp,false_negatives=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))


if __name__=='__main__':main()
