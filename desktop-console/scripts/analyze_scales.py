"""Compare candidate selection on cached development images (not a test score)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def iou(a, b):
    area = max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])) * max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return area/max(1,a[2]*a[3]+b[2]*b[3]-area)


def merge(boxes, ordering):
    result = []
    for box in sorted(boxes,key=lambda b:b[4] if ordering=='score' else b[2]*b[3],reverse=True):
        if any(iou(box, b)>.4 for b in result): continue
        result.append(box)
    return result


def evaluate(scenes, scales, threshold, ordering, pad):
    tp = fp = fn = 0
    for scene in scenes.values():
        truth = list(scene['truth'])
        boxes = [b for scale in scales for b in scene[scale] if b[4]>=threshold and .65<=b[2]/b[3]<=7]
        for box in merge(boxes,ordering):
            x,y,w,h,_ = box
            candidate = [x-w*pad,y-h*pad,w*(1+2*pad),h*(1+2*pad)]
            scores = [iou(candidate,b) for b in truth]
            if scores and max(scores)>=.5:
                truth.pop(scores.index(max(scores)));tp+=1
            else: fp+=1
        fn+=len(truth)
    return dict(scales=scales,threshold=threshold,ordering=ordering,pad=pad,tp=tp,fp=fp,fn=fn,precision=round(tp/max(1,tp+fp),3),recall=round(tp/max(1,tp+fn),3),f1=round(2*tp/max(1,2*tp+fp+fn),3))


if __name__ == '__main__':
    scenes = json.loads((ROOT/'tests/scale-diagnostic-cache.json').read_text())
    rows = [evaluate(scenes,scales,threshold,ordering,pad)
        for scales in [['full','640'],['full','480'],['full','480','640']]
        for threshold in [.15,.25,.35,.45,.55]
        for ordering in ['score','area']
        for pad in [0,.08,.15,.25]]
    print('Images:',len(scenes))
    print(json.dumps(sorted(rows,key=lambda r:r['f1'],reverse=True)[:12],indent=2))
