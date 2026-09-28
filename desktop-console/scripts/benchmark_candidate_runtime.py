"""Measure warmed whole-photo latency and a complete annotated video export.

Run after training has stopped. This synthetic ten-plate workload measures load,
not road accuracy. Photo totals include decode, detection, OCR and result crops;
video averages also include both decoding passes and MP4 encoding.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'platevision/backend'))
from app.services.ocr import PlateOCR
from app.services.pipeline import process,CropStore
from app.services.video import VideoJob,process_video
from evaluate_plate_candidate import load_detector,match


def make_video(photo,destination,frames=90):
    image=cv2.imread(str(photo));h,w=image.shape[:2]
    writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
    if not writer.isOpened():raise RuntimeError('Cannot create benchmark clip')
    for i in range(frames):
        transform=np.float32([[1,0,3*np.sin(i/10)],[0,1,2*np.cos(i/12)]])
        writer.write(cv2.warpAffine(image,transform,(w,h),borderMode=cv2.BORDER_REPLICATE))
    writer.release()


def benchmark(model_path,output,runs=30,include_video=True,legacy_video=False):
    detector=load_detector(model_path)
    detector.weights_retrained='runs' in model_path.parts
    detector.model_id=model_path.parent.parent.name if detector.weights_retrained else 'baseline'
    reader=PlateOCR();photo=ROOT/'tests/ten-plate-1080p.jpg';data=photo.read_bytes()
    measured=[];store=CropStore()
    for i in range(runs+3):
        wall=time.perf_counter();result=process(data,detector,store,reader,profile='fast')
        result['wall_ms']=(time.perf_counter()-wall)*1000
        if i>=3:measured.append(result)
    summary={key:dict(p50=float(np.median([r[key] for r in measured])),
                      p95=float(np.percentile([r[key] for r in measured],95)))
             for key in ('wall_ms','processing_time_ms','detection_time_ms','ocr_time_ms')}
    report=dict(model=str(model_path),sha256=sha256(model_path.read_bytes()).hexdigest(),
        photo_runs=runs,timings_ms=summary,plate_counts=[len(r['detections']) for r in measured],
        photo_result=measured[-1],notes=[__doc__,
            'No HTTP upload/browser rendering is included in the offline photo timing.',
            'Confidence is an uncalibrated model score; detection count alone is not recall.'])
    truth_report=json.loads((ROOT/'tests/ten-plate-fast-benchmark.json').read_text())
    image=cv2.imread(str(photo));predictions=detector.detect_fast(image)
    tp,fp,fn=match(predictions,[r['box'] for r in truth_report['truth']])
    report['ten_target_plate_localization']=dict(matched=tp,unmatched=fp,missed=fn,
                                               note='Fixture target boxes; neighbouring unlabelled plates may be visible.')
    if include_video:
        folder=ROOT/'.cache'/('candidate-video-'+uuid.uuid4().hex)
        folder.mkdir(parents=True);source=folder/'input.mp4';make_video(photo,source)
        if legacy_video:detector.fast_models=None
        job=VideoJob(folder.name,folder,source,'fast')
        process_video(job,detector,reader)
        status=job.snapshot()
        if status['status']!='complete':raise RuntimeError(status)
        report['video']=json.loads((folder/'results.json').read_text())
        report['video_file']=str(folder/'annotated.mp4')
        capture=cv2.VideoCapture(str(folder/'annotated.mp4'));frames=0
        while True:
            ok,frame=capture.read()
            if not ok:break
            if frames==15:cv2.imwrite(str(folder/'preview.jpg'),frame)
            frames+=1
        capture.release()
        if frames!=report['video']['frames']:raise AssertionError('Export lost frames')
        report['decoded_export_frames']=frames
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runs',type=int,default=30);parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--legacy-video',action='store_true')
    args=parser.parse_args();report=benchmark(args.model,args.output,args.runs,not args.no_video,args.legacy_video)
    print(json.dumps({k:v for k,v in report.items() if k not in ('photo_result','video')},indent=2))
    if 'video' in report:print(json.dumps(report['video'],indent=2))
