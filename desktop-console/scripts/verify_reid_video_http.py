"""Exercise two synthetic recording uploads with real local detection/ReID.

Adds explicitly named demo camera routes/observations to the regions API.
Does not clear existing observations or replace existing routes.
"""
import argparse
import json
from pathlib import Path
import time
import uuid
import urllib.request

import cv2
from evaluate_plate_ocr_http import endpoint

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8003'


def get_json(path, data=None, method=None):
    payload = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(BASE + path, data=payload, method=method,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def upload(camera, captured):
    boundary = 'video-' + uuid.uuid4().hex
    fields = dict(profile='fast', read_text='true', angle_correction='true',
                  reidentify='true', camera_id=camera, observed_at=captured)
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
             for name, value in fields.items()]
    parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="fixture.mp4"\r\nContent-Type: video/mp4\r\n\r\n'.encode(),
              (ROOT / 'tests/multi-plate-30fps.mp4').read_bytes(), f'\r\n--{boundary}--\r\n'.encode()]
    request = urllib.request.Request(BASE + '/api/videos', data=b''.join(parts),
                                     headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    global BASE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',type=lambda value:endpoint('api='+value)[1],default=BASE)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    BASE=args.base_url
    args.output_dir.mkdir(parents=True,exist_ok=False)
    token = uuid.uuid4().hex[:8]
    camera_a, camera_b = 'Demo-video-A-' + token, 'Demo-video-B-' + token
    routes = get_json('/api/reid/routes')['routes']
    routes.append(dict(source_camera=camera_a, target_camera=camera_b, min_seconds=60, max_seconds=300))
    get_json('/api/reid/routes', {'routes': routes}, 'PUT')
    results = []
    for camera, captured in [(camera_a, '2026-09-27T10:00:00Z'), (camera_b, '2026-09-27T10:03:00Z')]:
        state = upload(camera, captured)
        job_id = state['id']
        deadline = time.monotonic() + 180
        while state['status'] not in {'complete', 'failed', 'cancelled'}:
            if time.monotonic() > deadline:
                raise TimeoutError(f'Video {job_id} did not finish in three minutes')
            time.sleep(.5)
            state = get_json('/api/videos/' + job_id)
        assert state['status'] == 'complete', state.get('error')
        assert state['reidentification']['status'] == 'complete', state['reidentification']
        assert state['reidentification']['observation_count'] > 0
        assert state['reidentification']['observation_count'] == state['vehicle_summary']['confirmed_tracks']
        assert state['frames'] == 120 and state['all_frames_searched'] is True
        observations = state['reidentification']['observations']
        if camera == camera_b:
            assert any(item['status'] == 'candidate' for item in observations), observations
            for item in observations:
                candidate = item.get('selected_candidate')
                if candidate:
                    assert 60 <= candidate['travel_seconds'] <= 300
                    assert candidate['plate_evidence']['kind'] in {'none','agreement','low_confidence','insufficient_known_characters','partial_conflict'}
        result = {key: state[key] for key in ('frames', 'all_frames_searched', 'processing_ms_per_frame',
                  'plate_detection_ms', 'decode_detect_track_ms', 'vehicle_summary', 'plate_summary', 'recognized_plate_summary', 'read_text', 'ocr_total_ms', 'reidentification')}
        results.append(result)
        print(camera, state['status'], state['reidentification']['observation_count'], 'observations', flush=True)
    video_url = BASE + state['video_url']
    with urllib.request.urlopen(urllib.request.Request(video_url, headers={'Range': 'bytes=0-63'}), timeout=10) as ranged:
        range_status = ranged.status
        assert range_status == 206 and len(ranged.read()) == 64
    export = args.output_dir / ('reid-export-' + token + '.mp4')
    with urllib.request.urlopen(video_url, timeout=30) as response:
        export.write_bytes(response.read())
    capture = cv2.VideoCapture(str(export))
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    assert decoded == 120
    report = dict(fixture='Same synthetic four-second three-vehicle recording submitted to two demo cameras 180s apart; functional test, not cross-camera accuracy',
                  decoded_export_frames=decoded, range_request_status=range_status, recordings=results)
    (args.output_dir / 'reid-video-http-smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(decoded_export_frames=decoded, export=str(export),
                         second_camera_statuses=[item['status'] for item in observations]), indent=2))


if __name__ == '__main__':
    main()
