"""Run a multi-plate video fixture through the same local model frame endpoint."""
import json
from pathlib import Path
import cv2
import httpx

ROOT=Path(__file__).resolve().parents[1]
video=cv2.VideoCapture(str(ROOT/'tests/multi-plate-video.webm'))
duration=video.get(cv2.CAP_PROP_FRAME_COUNT)/video.get(cv2.CAP_PROP_FPS)
times=[i*.5 for i in range(int(duration/.5))]+[duration-.001]
results=[]
try:
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=60) as client:
        for time in times:
            video.set(cv2.CAP_PROP_POS_MSEC,time*1000)
            ok,frame=video.read()
            if not ok:
                video.set(cv2.CAP_PROP_POS_FRAMES,video.get(cv2.CAP_PROP_FRAME_COUNT)-1)
                ok,frame=video.read()
            assert ok,'Could not decode fixture'
            ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,95])
            response=client.post('/api/detect/frame',files={'image':('frame.jpg',encoded.tobytes(),'image/jpeg')})
            response.raise_for_status();data=response.json()
            readings=[d['normalized_text'] for d in data['detections'] if d['normalized_text']]
            assert data['vehicles']==[] and len(data['detections'])>=3,(time,data)
            assert 'OD11H7299' in readings,(time,readings)
            results.append(dict(timestamp=time,plates=len(data['detections']),text_readings=len(readings),processing_time_ms=data['processing_time_ms']))
finally:
    video.release()
report=dict(fixture='multi-plate-video.webm',note='Functional composite-video regression, not an independent accuracy benchmark.',frames=len(results),results=results)
(ROOT/'tests/video-regression.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
