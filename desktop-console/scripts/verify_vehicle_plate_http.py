"""Verify live supplemental plate rectangles without invoking OCR."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid

import cv2

from benchmark_reid_http import photo_request
from evaluate_plate_ocr_http import endpoint

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8003'


def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        return json.load(response)


def main():
    global BASE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',type=lambda value:endpoint('api='+value)[1],default=BASE)
    parser.add_argument('--image',type=Path,required=True,help='User-owned known missed-plate scene; this functional test expects secondary recovery.')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    BASE=args.base_url
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for attempt in range(30):
        try:
            health = get_json('/api/health')
            break
        except OSError:
            if attempt == 29:
                raise
            time.sleep(1)
    assert health['detector_ready'] and health['vehicle_detector_ready']
    source = args.image
    fields = dict(profile='fast',read_text='false',angle_correction='true',reidentify='false')
    photo, elapsed = photo_request(BASE, source.read_bytes(), fields)
    extra = [plate for plate in photo['detections'] if plate['localization_source'] == 'vehicle_crop']
    assert photo['read_text'] is False and photo['ocr_time_ms'] == 0
    assert photo['vehicle_plate_search']['added_regions'] == len(extra) > 0
    assert all(plate['localization_requires_review'] and not plate['raw_text'] for plate in extra)
    assert all(plate['tight_plate_box']['width'] > 0 and plate['tight_plate_box']['height'] > 0 for plate in extra)
    data = cv2.imread(str(source))
    height, width = data.shape[:2]
    scale = min(1, 1920 / max(height, width))
    frame = cv2.resize(data,(int(width*scale)//2*2,int(height*scale)//2*2))
    clip = args.output_dir / 'vehicle-plate-recovery-smoke.mp4'
    writer = cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),6.,(frame.shape[1],frame.shape[0]))
    assert writer.isOpened()
    for _ in range(6):
        writer.write(frame)
    writer.release()
    boundary='recovery-'+uuid.uuid4().hex
    parts=[f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode() for key,value in fields.items()]
    parts.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="recovery.mp4"\r\nContent-Type: video/mp4\r\n\r\n'.encode(),clip.read_bytes(),f'\r\n--{boundary}--\r\n'.encode()])
    request=urllib.request.Request(BASE+'/api/videos',data=b''.join(parts),headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(request,timeout=30) as response:
        state=json.load(response)
    deadline=time.monotonic()+90
    while state['status'] not in {'complete','failed','cancelled'}:
        if time.monotonic()>deadline:
            raise TimeoutError('Supplemental video check did not complete')
        time.sleep(.5)
        state=get_json('/api/videos/'+state['id'])
    assert state['status']=='complete',state.get('error')
    assert state['frames']==6 and state['all_frames_searched']
    assert state['read_text'] is False and state['ocr_total_ms']==0
    # Encoding/resizing can make the whole-frame detector find the formerly
    # missed plate already. Extra boxes are not mandatory when none are needed.
    assert state['vehicle_plate_search']['enabled']
    assert state['vehicle_plate_search']['searched_vehicles']>0
    assert state['vehicle_plate_search']['failed_crops']==0
    assert state['plate_summary']['max_visible']>0
    assert all(not track['text'] for track in state['tracks'])
    report=dict(note='Known missed-plate scene reused from development data; functional check only.',
                photo_http_ms=round(elapsed,2),photo=photo,
                video={key:state[key] for key in ('id','frames','read_text','all_frames_searched','vehicle_plate_search','vehicle_plate_search_ms','plate_summary','tracks','processing_ms_per_frame','video_url')})
    (args.output_dir/'vehicle-plate-recovery-http.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(photo_plates=photo['plate_count'],photo_vehicles=photo['vehicle_summary']['total_vehicles'],
                         photo_extra=photo['vehicle_plate_search'],video_frames=state['frames'],
                         video_extra=state['vehicle_plate_search']),indent=2))


if __name__=='__main__':
    main()
