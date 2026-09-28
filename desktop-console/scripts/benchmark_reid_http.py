"""Measure the local 1080p fixture with/without optional vehicle descriptors.

Requires the regions API on localhost. Uses synthetic benchmark camera IDs,
adds observations to that server's in-memory gallery, and changes no routes.
This measures latency and output integrity, not cross-camera accuracy.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid
from evaluate_plate_ocr_http import endpoint

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def photo_request(base, data, fields):
    boundary = 'platesight-' + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="fixture.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode(),
              data, f'\r\n--{boundary}--\r\n'.encode()]
    request = urllib.request.Request(base + '/api/detect/image', data=b''.join(parts),
              headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    return result, (time.perf_counter()-started)*1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--base-url',default='http://127.0.0.1:8003',type=lambda value:endpoint('api='+value)[1])
    parser.add_argument('--image',type=Path,default=ROOT/'tests/ten-plate-1080p.jpg')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-plates',type=int,help='Optional explicit fixture regression expectation; not an accuracy measurement.')
    args = parser.parse_args()
    if args.runs < 5:
        parser.error('Use at least five runs.')
    base = args.base_url
    with urllib.request.urlopen(base + '/api/reid/health') as response:
        health = json.load(response)
    if not health['ready']:
        raise RuntimeError('The verified appearance model must be ready.')
    data = args.image.read_bytes()
    camera = 'Benchmark-' + uuid.uuid4().hex[:8]
    samples = {False: [], True: []}
    for index in range(args.runs+3):
        for enabled in ([False, True] if index % 2 == 0 else [True, False]):
            fields = dict(profile='fast', read_text='false', angle_correction='true',
                          reidentify=str(enabled).lower())
            if enabled:
                fields.update(camera_id=camera, observed_at='2026-09-27T10:00:00Z')
            result, wall = photo_request(base, data, fields)
            if args.expected_plates is not None:
                assert result['plate_count'] == args.expected_plates, result['plate_count']
            assert result['ocr_time_ms'] == 0
            if enabled:
                assert result['reidentification']['status'] == 'complete'
                assert result['reidentification']['observation_count'] == result['vehicle_summary']['total_vehicles']
            if index >= 3:
                samples[enabled].append(dict(http_ms=wall, backend_ms=result['processing_time_ms'],
                    detection_ms=result['detection_time_ms'], reid_ms=(result.get('reidentification') or {}).get('processing_time_ms', 0),
                    vehicles=result['vehicle_summary']['total_vehicles'], plates=result['plate_count']))
    report = dict(fixture=str(args.image),
                  model=health['model'], runtime=health['engine'], measured_runs_per_mode=args.runs,
                  warmup_runs_per_mode=3, read_text=False, angle_correction=True,
                  note='Local HTTP includes request/response serialization; excludes browser rendering. Not real CCTV accuracy. No routes changed.', modes={})
    for enabled, rows in samples.items():
        report['modes']['reid_on' if enabled else 'reid_off'] = {
            name: dict(median=round(float(np.median([row[name] for row in rows])), 2),
                       p95=round(float(np.percentile([row[name] for row in rows], 95)), 2))
            for name in ('http_ms', 'backend_ms', 'detection_ms', 'reid_ms')}
        report['modes']['reid_on' if enabled else 'reid_off'].update(
            observed_plate_counts=sorted({row['plates'] for row in rows}),
            observed_vehicle_counts=sorted({row['vehicles'] for row in rows}))
    output = args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
