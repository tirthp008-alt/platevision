"""Reproducible synthetic 1080p/10-plate load test; not a road accuracy test."""
import sys,json,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'platevision/backend'))
from app.services.detector import OpenVINOPlateDetector
from app.services.pipeline import process,CropStore
from app.services.ocr import PlateOCR
src=cv2.imread(str(ROOT/'tests/three-vehicle-composite.jpg'))
scene=np.full((1080,1920,3),60,np.uint8)
originals=[(101,162,75,21,'OD02BJ0758'),(593,184,104,27,'OD02BG1396'),(1240,322,93,19,'OD11H7299')]
truth=[]
for i in range(10):
    x,y,w,h,text=originals[i%3]
    left=max(0,min(src.shape[1]-480,x+w//2-240));top=max(0,min(src.shape[0]-360,y+h//2-180))
    dx=(i%4)*480;dy=(i//4)*360
    scene[dy:dy+360,dx:dx+480]=src[top:top+360,left:left+480]
    truth.append(dict(box=[dx+x-left,dy+y-top,w,h],text=text))
cv2.imwrite(str(ROOT/'tests/ten-plate-1080p.jpg'),scene)
# Crops include neighbouring cars in two panels; evaluate all actual detections and the ten labelled target plates.
d=OpenVINOPlateDetector(str(ROOT/'platevision/models/plate-detector.onnx'),input_shape=(864,1536));d.fast_models={'landscape':d,'portrait':d}
o=PlateOCR();data=(ROOT/'tests/ten-plate-1080p.jpg').read_bytes();runs=[]
for i in range(16):
    r=process(data,d,CropStore(),o,profile='fast');runs.append(r)
    print(i,r['processing_time_ms'],r['detection_time_ms'],r['ocr_time_ms'],len(r['detections']),flush=True)
report=dict(note=__doc__,truth=truth,cold=runs[0],runs=runs[1:],p50=float(np.median([r['processing_time_ms'] for r in runs[1:]])),p95=float(np.percentile([r['processing_time_ms'] for r in runs[1:]],95)))
(ROOT/'tests/ten-plate-1536-benchmark.json').write_text(json.dumps(report,indent=2))
print(report['p50'],report['p95'])
