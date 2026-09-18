"""Measure annotated green-plate recall, not precision on partially labelled images."""
import argparse
import json
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'platevision/backend'))
from app.services.detector import OpenVINOPlateDetector
from app.services.pipeline import decode
from app.services.video import iou

prefixes='''20220725_17_54_49_922 20220725_17_54_56_374 20220723_16_54_39_492
20220724_11_20_11_768 20220724_13_51_07_465 20220724_19_33_51_663
20220725_09_40_20_665 20220725_11_15_15_253 20220725_11_15_29_428
20220725_11_15_49_466 20220725_11_38_19_981 20220725_11_38_31_614
20220725_11_43_56_412 20220725_11_44_57_903 20220725_13_52_02_558
20220725_13_57_27_861 20220725_15_39_26_969 20220725_15_39_35_347
20220725_15_39_53_908'''.split()
parser=argparse.ArgumentParser();parser.add_argument('--images',required=True);parser.add_argument('--labels',required=True)
args=parser.parse_args();rows=[]
detector=OpenVINOPlateDetector('platevision/models/plate-detector.onnx');cv2.setNumThreads(2)
xmls={p.stem:p for p in Path(args.labels).rglob('*.xml')}
for prefix in prefixes:
    path=next(Path(args.images).glob(prefix+'*.jpg'))
    tree=ET.parse(xmls[path.stem]);image=decode(path.read_bytes())
    aw=int(tree.findtext('size/width'));ah=int(tree.findtext('size/height'))
    if (image.shape[1],image.shape[0])!=(aw,ah):
        rows.append(dict(file=path.name,error='EXIF/annotation dimensions differ; needs label review'));continue
    labels=[]
    for obj in tree.findall('object'):
        b=obj.find('bndbox');x,y,x2,y2=[int(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
        labels.append((x,y,x2-x,y2-y))
    row=dict(file=path.name,labelled_green_plates=len(labels))
    for profile,limit,refine in [('fast',1280,False),('detailed',1920,True)]:
        ratio=min(1,limit/max(aw,ah));scaled=cv2.resize(image,(round(aw*ratio),round(ah*ratio)))
        started=time.perf_counter();boxes=detector.detect_plates(scaled,refine=refine);ms=(time.perf_counter()-started)*1000
        boxes=[tuple(v/ratio for v in box[:4])+(box[4],) for box in boxes]
        matched=sum(any(iou(label,box)>=.5 for box in boxes) for label in labels)
        row[profile]=dict(matched_green_plates=matched,total_predicted_regions=len(boxes),detection_ms=round(ms,2))
    rows.append(row);print(path.name[:23],row.get('fast'),row.get('detailed'),flush=True)
report=dict(note='Only green plates are annotated. Unlabelled white/yellow plates prevent a valid precision estimate. These images were used for development checks, not independent accuracy claims.',images=rows)
for profile in ('fast','detailed'):
    valid=[r for r in rows if profile in r];total=sum(r['labelled_green_plates'] for r in valid);matched=sum(r[profile]['matched_green_plates'] for r in valid)
    report[profile]=dict(matched=matched,labelled=total,recall=matched/max(1,total),detection_p50_p95_ms=np.percentile([r[profile]['detection_ms'] for r in valid],[50,95]).tolist())
Path('tests/attached-green-plates.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({p:report[p] for p in ('fast','detailed')},indent=2))
