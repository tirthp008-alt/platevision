"""Exercise the real upload/export API and verify every output frame."""
import argparse
import json
import time
from pathlib import Path
import cv2
import httpx

parser=argparse.ArgumentParser()
parser.add_argument('--video',default='tests/multi-plate-video.webm')
parser.add_argument('--output',default='tests/video-export-benchmark.json')
parser.add_argument('--runs',type=int,default=3)
args=parser.parse_args()
reports=[]
with httpx.Client(base_url='http://127.0.0.1:8000',timeout=60) as client:
    for run in range(args.runs):
        start=time.perf_counter()
        with open(args.video,'rb') as source:
            response=client.post('/api/videos',files={'video':(Path(args.video).name,source,'video/webm')},data={'profile':'fast'})
        response.raise_for_status();job=response.json()
        while job['status'] not in {'complete','failed','cancelled'}:
            time.sleep(.15);job=client.get('/api/videos/'+job['id']).json()
        assert job['status']=='complete',job
        download=client.get(job['video_url']);download.raise_for_status()
        destination=Path('tests/video-results/api-annotated.mp4');destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(download.content)
        capture=cv2.VideoCapture(str(destination));frames=0
        while True:
            ok,image=capture.read()
            if not ok:break
            if frames==0:cv2.imwrite('tests/video-export-preview.jpg',image)
            frames+=1
        capture.release()
        assert frames==job['frames'],(frames,job['frames'])
        ranged=client.get(job['video_url'],headers={'Range':'bytes=0-255'})
        assert ranged.status_code==206 and len(ranged.content)==256
        assert len(job['tracks'])>=3,job['tracks']
        assert all(t['observations']==frames for t in job['tracks'][:3])
        job['upload_poll_download_wall_ms']=round((time.perf_counter()-start)*1000,2)
        reports.append(job)
        print(json.dumps({k:job[k] for k in ('detection_ms','decode_detect_track_ms','processing_ms_per_frame','total_ms','tracks')}),flush=True)
Path(args.output).write_text(json.dumps(reports,indent=2),encoding='utf-8')
