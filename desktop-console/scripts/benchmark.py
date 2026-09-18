import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'platevision/backend'))
from app.services.detector import OnnxPlateDetector
from app.services.ocr import PlateOCR
from app.services.pipeline import process,CropStore
root=Path(__file__).resolve().parents[1]
p=OnnxPlateDetector(str(root/'platevision/models/plate-detector.onnx'))
v=OnnxPlateDetector(str(root/'platevision/models/vehicle-detector.onnx'),80,{2,3,5,7})
o=PlateOCR();store=CropStore();results=[]
files=sorted((root/'platevision/data/indian-plates').rglob('*.jpg'))[::8][:20]
for path in files:
    result=process(path.read_bytes(),p,store,o,v)
    results.append(dict(file=str(path.relative_to(root)),total=result['processing_time_ms'],detection=result['detection_time_ms'],ocr=result['ocr_time_ms'],plates=len(result['detections']),vehicles=len(result['vehicles']),text=[d['normalized_text'] for d in result['detections']]))
summary={key:{'p50':float(np.percentile([r[key] for r in results],50)),'p95':float(np.percentile([r[key] for r in results],95))} for key in ['total','detection','ocr']}
report=dict(sample_count=len(results),timings_ms=summary,results=results,accuracy='Not established; plate text ground truth is unavailable.')
(root/'tests/benchmark.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
