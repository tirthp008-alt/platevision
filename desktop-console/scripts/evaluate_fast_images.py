import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'platevision/backend'))
from app.services.detector import OpenVINOPlateDetector
from app.services.ocr import PlateOCR
from app.services.pipeline import process,CropStore
from analyze_scales import iou
o=PlateOCR();reports=[]
for shape in [(864,1536),(1088,1920)]:
 d=OpenVINOPlateDetector(str(ROOT/'platevision/models/plate-detector.onnx'),input_shape=shape);d.fast_models={'landscape':d,'portrait':d};tp=fp=fn=0;scenes=[]
 for path in sorted((ROOT/'platevision/data/yolo-indian/val/images').glob('*.jpg')):
  r=process(path.read_bytes(),d,CropStore(),o,profile='fast');w,h=r['image']['width'],r['image']['height'];truth=[]
  for l in (path.parents[1]/'labels'/f'{path.stem}.txt').read_text().splitlines():
   _,cx,cy,bw,bh=map(float,l.split());truth.append(((cx-bw/2)*w,(cy-bh/2)*h,bw*w,bh*h))
  n=len(truth);matched=0
  for p in r['detections']:
   b=p['bounding_box'];scores=[iou([b['x'],b['y'],b['width'],b['height']],t) for t in truth]
   if scores and max(scores)>=.5:truth.pop(scores.index(max(scores)));tp+=1;matched+=1
   else:fp+=1
  fn+=len(truth);scenes.append(dict(file=path.name,labelled=n,matched=matched,ms=r['processing_time_ms'],predicted=len(r['detections'])))
 report=dict(shape=shape,tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),p50=float(np.median([s['ms'] for s in scenes])),p95=float(np.percentile([s['ms'] for s in scenes],95)),scenes=scenes)
 reports.append(report);print({k:v for k,v in report.items() if k!='scenes'},flush=True)
(ROOT/'tests/fast-image-development.json').write_text(json.dumps(reports,indent=2))
