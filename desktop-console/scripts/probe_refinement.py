"""Measure plate-candidate zoom refinement with the existing ONNX weights."""
import json
import sys
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'platevision/backend'))
from app.services.detector import OnnxPlateDetector
from app.core.config import settings


def main():
    settings.confidence_threshold=.15
    model=OnnxPlateDetector(str(ROOT/'platevision/models/plate-detector.onnx'))
    path=ROOT/'tests/scale-diagnostic-cache.json'
    scenes=json.loads(path.read_text())
    for index,(name,scene) in enumerate(scenes.items()):
        if 'refined' in scene: continue
        image=cv2.imread(str(ROOT/'platevision/data/yolo-indian/val/images'/name))
        height,width=image.shape[:2]
        proposals=model.merge_plates([b for scale in ['full','480'] for b in scene[scale] if b[4]>=.15])
        refined=[]
        for x,y,w,h,score in proposals:
            if score>=.65 or max(w,h)>180: continue
            size=max(192,round(max(w,h)*4))
            left=max(0,min(width-size,round(x+w/2-size/2)))
            top=max(0,min(height-size,round(y+h/2-size/2)))
            for bx,by,bw,bh,bs in model.detect(image[top:top+size,left:left+size]):
                # Keep only a result over the proposed plate, not unrelated
                # objects found anywhere in the expanded crop.
                ix=max(0,min(bx+bw+left,x+w)-max(bx+left,x))
                iy=max(0,min(by+bh+top,y+h)-max(by+top,y))
                if ix*iy/max(1,min(w*h,bw*bh))>=.5:
                    refined.append([bx+left,by+top,bw,bh,bs])
        scene['refined']=refined
        path.write_text(json.dumps(scenes))
        print(f'{index+1}/{len(scenes)} {len(refined)} refined',flush=True)


if __name__=='__main__':main()
