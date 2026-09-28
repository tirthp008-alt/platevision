"""One local video upload with OCR and angle correction, followed by export checks.

Uses the existing 120-frame synthetic fixture. Counts are model outputs, not
ground-truth accuracy. Run only after training and model comparison finish.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import cv2
import httpx

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FRAMES = 120
REQUEST_FIELDS = dict(profile='fast', read_text='true', angle_correction='true',
                      reidentify='false', ocr_model='ppocr', green_filter='false')


def validate_preflight(health):
    missing = [key for key in ('detector_ready', 'ocr_ready', 'video_export_ready', 'vehicle_detector_ready')
               if health.get(key) is not True]
    if missing:
        raise AssertionError('Candidate API is not ready for the plate/vehicle smoke: ' + ', '.join(missing))
    digest = health.get('model_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in '0123456789abcdefABCDEF' for char in digest):
        raise AssertionError('Candidate API did not report a valid model SHA256.')


def inspect_video(path, *, decode=False, preview=None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if decode:
            frames = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape[:2] != (height, width):
                    raise AssertionError('Decoded export dimensions changed within the recording.')
                if frames == 0 and preview is not None and not cv2.imwrite(str(preview), image):
                    raise OSError(f'Cannot save preview: {preview}')
                frames += 1
        return dict(frames=frames, width=width, height=height, fps=fps)
    finally:
        capture.release()


def validate_completed_job(job, source):
    if job.get('status') != 'complete':
        raise AssertionError(f'Video job failed: {job.get("status")}: {job.get("error")}')
    if job.get('read_text') is not True or job.get('angle_correction') is not True:
        raise AssertionError('OCR and angle correction must both be enabled.')
    if job.get('frames') != EXPECTED_FRAMES or job.get('all_frames_searched') is not True:
        raise AssertionError('The job did not search and retain all 120 fixture frames.')
    if (job.get('width'), job.get('height')) != (source['width'], source['height']):
        raise AssertionError('Unexpected fixture output dimensions.')
    recognized = job.get('recognized_plate_summary')
    if not isinstance(recognized, dict) or recognized.get('enabled') is not True:
        raise AssertionError('OCR-enabled recording is missing its recognized-plate summary.')
    if job.get('recognized_plates') != recognized.get('groups'):
        raise AssertionError('Recognized-plate export differs from its grouped summary.')
    for track in job.get('tracks', []):
        review = track.get('ocr_review') or {}
        if 'rectified_crop' in review or any(key.startswith('_') for key in review):
            raise AssertionError('Private OCR processing metadata leaked into the response.')


def summarize_job(job):
    tracks = job.get('tracks', [])
    groups = job['recognized_plate_summary'].get('groups', [])
    return dict(
        plate_summary=job.get('plate_summary'),
        vehicle_summary=job.get('vehicle_summary'),
        recognized_plate_summary=job['recognized_plate_summary'],
        raw_plate_tracks=len(tracks), grouped_registration_rows=len(groups),
        review_counts=dict(
            localization=sum(bool(track.get('localization_requires_review')) for track in tracks),
            ocr=sum(bool((track.get('ocr_review') or {}).get('requires_review')) for track in tracks),
            experimental=sum(bool(track.get('experimental_review_only')) for track in tracks)),
        localization_sources=dict(Counter(track.get('localization_source', 'unspecified') for track in tracks)),
        format_statuses=dict(Counter(track.get('format_status', 'unspecified') for track in tracks)),
        vehicle_plate_search=job.get('vehicle_plate_search'),
        warnings=job.get('warnings', []),
        note='Plate tracks, vehicle tracks, per-frame region observations and grouped registrations have different meanings. All counts are model-derived; this fixture has no asserted count or category accuracy in this smoke.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='Warmed local candidate API, e.g. http://127.0.0.1:8003')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report path within workspace .cache')
    parser.add_argument('--video', type=Path, default=ROOT / 'tests/multi-plate-30fps.mp4')
    parser.add_argument('--timeout', type=float, default=180., help='Maximum job polling seconds')
    args = parser.parse_args()
    endpoint = urlparse(args.base_url)
    if endpoint.scheme not in ('http', 'https') or endpoint.hostname not in ('localhost', '127.0.0.1', '::1'):
        parser.error('Use the local candidate API; external upload destinations are not supported.')
    if not 0 < args.timeout <= 600:
        parser.error('Timeout must be positive and at most 600 seconds.')
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / '.cache').resolve()):
        parser.error('Store smoke artifacts under workspace .cache.')
    paths = {name: output.with_name(output.stem + suffix) for name, suffix in
             [('job', '-job.json'), ('video', '-annotated.mp4'), ('preview', '-preview.jpg')]}
    if any(path.exists() for path in [output, *paths.values()]):
        raise FileExistsError('Refusing to overwrite prior smoke artifacts; choose a new output path.')
    source = inspect_video(args.video)
    if source['frames'] != EXPECTED_FRAMES or source['width'] > 1920 or source['height'] > 1920:
        raise ValueError('Use the 120-frame fixture with dimensions at most 1920 pixels.')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fields = dict(REQUEST_FIELDS)
    with httpx.Client(base_url=args.base_url.rstrip('/'), timeout=60.) as client:
        response = client.get('/api/health')
        response.raise_for_status()
        health = response.json()
        validate_preflight(health)
        with args.video.open('rb') as source_file:
            response = client.post('/api/videos', files={'video': (args.video.name, source_file, 'video/mp4')}, data=fields)
        response.raise_for_status()
        job = response.json()
        deadline = time.monotonic() + args.timeout
        while job['status'] not in ('complete', 'failed', 'cancelled'):
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Video job {job["id"]} exceeded {args.timeout:g} seconds.')
            time.sleep(.5)
            response = client.get('/api/videos/' + job['id'])
            response.raise_for_status()
            job = response.json()
        paths['job'].write_text(json.dumps(job, indent=2), encoding='utf-8')
        validate_completed_job(job, source)
        video_url = job['video_url']
        if not video_url.startswith('/api/videos/') or urlparse(video_url).netloc:
            raise AssertionError('Unexpected export URL outside the local video API.')
        ranged = client.get(video_url, headers={'Range': 'bytes=0-255'})
        ranged.raise_for_status()
        if ranged.status_code != 206 or len(ranged.content) != 256 or not ranged.headers.get('Content-Range', '').startswith('bytes 0-255/'):
            raise AssertionError('Export does not support the expected byte-range response.')
        with client.stream('GET', video_url) as download:
            download.raise_for_status()
            with paths['video'].open('xb') as target:
                for chunk in download.iter_bytes():
                    target.write(chunk)
    decoded = inspect_video(paths['video'], decode=True, preview=paths['preview'])
    if decoded['frames'] != EXPECTED_FRAMES or (decoded['width'], decoded['height']) != (source['width'], source['height']):
        raise AssertionError('Decoded annotated export lost frames or changed fixture dimensions.')
    report = dict(
        status='passed', base_url=args.base_url, fixture=str(args.video.resolve()),
        source=source, decoded_export=decoded, request_fields=fields, job_id=job['id'],
        preflight_health=health, model_sha256=health['model_sha256'],
        preflight_engine=health.get('engine'), vehicle_detector_ready=health['vehicle_detector_ready'],
        range_status=ranged.status_code, upload_poll_download_decode_wall_ms=round((time.perf_counter() - started) * 1000, 2),
        paths={name: str(path) for name, path in paths.items()},
        model=job.get('detector_model'), engine=job.get('engine'),
        processing_ms_per_frame=job.get('processing_ms_per_frame'),
        ocr_total_ms=job.get('ocr_total_ms'), rectification_total_ms=job.get('rectification_total_ms'),
        **summarize_job(job),
    )
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='passed', report=str(output), frames=decoded['frames'],
                         plate_summary=report['plate_summary'], vehicle_summary=report['vehicle_summary'],
                         recognized_plate_summary=report['recognized_plate_summary'],
                         review_counts=report['review_counts']), indent=2))


if __name__ == '__main__':
    main()
