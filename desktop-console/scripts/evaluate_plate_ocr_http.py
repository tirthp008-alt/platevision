"""Evaluate existing local APIs against frozen manual plate truth; never tune models.

The manual audit contains 20 whole validation images, one labelled target each,
and 12 explicitly eligible exact OCR strings. Raw OCR exact is the primary
text metric; application-normalized exact is reported separately because the
backend can repair positional O/Q/I characters to digits. It is a diagnostic on
previously used validation images, not independent road accuracy. Other returned
boxes are unscored, never called false positives. Optional synthetic timing uses
the existing ten-plate fixture (three repeated source vehicles), 1 warmup and 5
measured runs per endpoint. Neither workload measures vehicle accuracy.

Run only after training/comparison finishes and the GPU is otherwise idle.
This script contacts already-running localhost APIs; it starts no service and
changes no model, threshold, camera route, or registration gallery.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
IOU_THRESHOLD = .5
FIELDS = dict(profile='fast', ocr_model='ppocr', read_text='true',
              angle_correction='true', green_filter='false', reidentify='false')
TIMING_FIELDS = ('processing_time_ms', 'decode_time_ms', 'detection_time_ms',
                 'vehicle_detection_time_ms', 'rectification_time_ms', 'crop_time_ms', 'ocr_time_ms')


def endpoint(value):
    alias, separator, address = value.partition('=')
    parsed = urlsplit(address)
    if (not separator or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', alias)
            or parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'}
            or parsed.username or parsed.password or parsed.path not in {'', '/'}
            or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError('Use ALIAS=http://127.0.0.1:PORT (local API origin only).')
    try:
        parsed.port
    except ValueError as error:
        raise argparse.ArgumentTypeError('Invalid endpoint port.') from error
    return alias, address.rstrip('/')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', type=endpoint, action='append', required=True,
                        help='Repeat for each ALIAS=http://127.0.0.1:PORT to evaluate.')
    parser.add_argument('--truth', type=Path, required=True, help='User-owned manual truth JSON; private validation images and labels are not bundled')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report; an existing file is never overwritten.')
    parser.add_argument('--synthetic', action='store_true', help='Also run the existing ten-target synthetic fixture.')
    parser.add_argument('--timeout', type=float, default=120., help='Per-request timeout in seconds; no automatic retries.')
    args = parser.parse_args(argv)
    if len({alias for alias, _ in args.endpoint}) != len(args.endpoint):
        parser.error('Endpoint aliases must be unique.')
    if len({address for _, address in args.endpoint}) != len(args.endpoint):
        parser.error('Endpoint addresses must be distinct.')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive.')
    return args


def normalized_text(text):
    """Ignore case/separators only; never repair or infer registration characters."""
    return re.sub(r'[\s.\-]', '', text.upper()) if isinstance(text, str) else ''


def raw_text_for_exact(text):
    """Raw OCR comparison ignores case and whitespace only, preserving all characters."""
    return re.sub(r'\s', '', text.upper()) if isinstance(text, str) else ''


def xywh(value):
    if isinstance(value, dict):
        value = [value[key] for key in ('x', 'y', 'width', 'height')]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('Expected a four-value pixel xywh box.')
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise ValueError('Box coordinates must be finite numbers.')
    if min(value[2:]) <= 0:
        raise ValueError('Box width and height must be positive.')
    return list(value)


def overlap(a, b):
    intersection = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return intersection / (a[2]*a[3]+b[2]*b[3]-intersection)


def evaluate_targets(response, targets):
    """Match by descending IoU, one to one, without seeing predicted text."""
    detections = response['detections']
    # Padded evidence crops are not detector geometry. Fail on older APIs that
    # do not expose tight boxes instead of silently changing the scoring rule.
    boxes = [xywh(plate['tight_plate_box']) for plate in detections]
    pairs = [(overlap(xywh(target['box']), box), i, j)
             for i, target in enumerate(targets) for j, box in enumerate(boxes)]
    matched, used = {}, set()
    for iou, i, j in sorted(pairs, key=lambda pair: (-pair[0], pair[1], pair[2])):
        if iou >= IOU_THRESHOLD and i not in matched and j not in used:
            matched[i] = (j, iou)
            used.add(j)
    rows = []
    for index, target in enumerate(targets):
        assignment = matched.get(index)
        plate = detections[assignment[0]] if assignment else {}
        review = plate.get('ocr_review') or {}
        eligible = target['score_exact'] is True
        expected = normalized_text(target.get('text'))
        observed = normalized_text(plate.get('normalized_text'))
        raw_observed = raw_text_for_exact(plate.get('raw_text'))
        application_exact = bool(assignment and observed == expected) if eligible else None
        rows.append(dict(target_id=target['id'], target_box=xywh(target['box']),
                         detection_index=assignment[0] if assignment else None,
                         detected=assignment is not None, match_iou=assignment[1] if assignment else None,
                         best_available_iou=max((overlap(xywh(target['box']), box) for box in boxes), default=0.),
                         score_exact=eligible, expected_text=expected if eligible else None,
                         observed_text=observed, raw_observed_text=raw_observed,
                         raw_exact=bool(assignment and raw_observed == raw_text_for_exact(target.get('text'))) if eligible else None,
                         application_normalized_exact=application_exact,
                         exact=application_exact,  # Legacy alias, explicitly defined in report metadata.
                         detection_confidence=plate.get('detection_confidence'), ocr_confidence=plate.get('ocr_confidence'),
                         format_status=plate.get('format_status'), localization_source=plate.get('localization_source'),
                         localization_requires_review=plate.get('localization_requires_review'),
                         ocr_requires_review=review.get('requires_review'), ocr_review=review,
                         rectification=plate.get('rectification'), raw_text=plate.get('raw_text')))
    return dict(targets=rows, detections_outside_scored_targets=len(detections)-len(used))


def load_manual(path):
    data = path.read_bytes()
    document = json.loads(data)
    entries = document.get('entries', [])
    if len(entries) != 20 or sum(entry.get('use_for_exact_ocr_score') is True for entry in entries) != 12:
        raise ValueError('Expected the frozen audit with 20 whole images and 12 exact-OCR targets.')
    ids, image_hashes, exact_strings, prepared = set(), set(), set(), []
    for entry in entries:
        identifier = entry['audit_id']
        if identifier in ids or type(entry.get('use_for_exact_ocr_score')) is not bool:
            raise ValueError('Audit IDs must be unique and exact-OCR flags must be booleans.')
        ids.add(identifier)
        image = Path(entry['image'])
        if not image.is_absolute():
            image = path.parent / image
        content = image.read_bytes()
        digest = sha256(content).hexdigest()
        if digest != entry['image_sha256']:
            raise ValueError(f'Whole image changed after manual review: {image}')
        if digest in image_hashes:
            raise ValueError('The manual audit contains duplicate whole-image bytes.')
        image_hashes.add(digest)
        eligible = entry['use_for_exact_ocr_score']
        text = normalized_text(entry.get('manual_text'))
        if eligible and (not re.fullmatch(r'[A-Z0-9]+', text) or text in exact_strings):
            raise ValueError('Scored manual strings must be nonempty and distinct; keep duplicate registrations excluded.')
        if eligible:
            exact_strings.add(text)
        prepared.append(dict(id=identifier, path=image, data=content, sha256=digest, metadata=entry,
                             targets=[dict(id=identifier, box=xywh(entry['box']), text=text, score_exact=eligible)]))
    provenance = {key: value for key, value in document.items() if key != 'entries'}
    provenance.update(path=str(path.resolve()), sha256=sha256(data).hexdigest(), whole_images=20, exact_targets=12)
    return prepared, provenance


def load_synthetic():
    path = ROOT / 'tests/ten-plate-1080p.jpg'
    truth_path = ROOT / 'tests/ten-plate-fast-benchmark.json'
    data, truth_data = path.read_bytes(), truth_path.read_bytes()
    truth = json.loads(truth_data)['truth']
    if len(truth) != 10:
        raise ValueError('Expected the existing synthetic fixture with ten labelled targets.')
    targets = [dict(id=str(i+1), box=xywh(item['box']), text=normalized_text(item['text']), score_exact=True)
               for i, item in enumerate(truth)]
    if any(not target['text'] for target in targets):
        raise ValueError('Synthetic target strings must be nonempty.')
    return dict(id='synthetic-ten', path=path, data=data, sha256=sha256(data).hexdigest(), targets=targets,
                truth_path=str(truth_path), truth_sha256=sha256(truth_data).hexdigest())


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


def request_json(url, timeout, data=None, headers=None):
    # Explicit local endpoints should not use ambient proxy or redirect uploads.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(url, data=data, headers=headers or {})
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def photo_request(address, sample, timeout):
    boundary = 'plate-audit-' + uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
             for key, value in FIELDS.items()]
    mime = mimetypes.guess_type(sample['path'].name)[0]
    if mime not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise ValueError('The audit image must be JPEG, PNG or WebP.')
    parts.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="whole-image{sample["path"].suffix}"\r\nContent-Type: {mime}\r\n\r\n'.encode(),
                  sample['data'], f'\r\n--{boundary}--\r\n'.encode()])
    body = b''.join(parts)
    started = time.perf_counter()
    result = request_json(address+'/api/detect/image', timeout, body,
                          {'Content-Type': f'multipart/form-data; boundary={boundary}'})
    return result, (time.perf_counter()-started)*1000


def validate_health(health):
    if not all(health.get(key) is True for key in ('detector_ready', 'ocr_ready', 'vehicle_detector_ready')):
        raise ValueError('Detector, OCR and vehicle engines must all be ready for this end-to-end audit.')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', health.get('model_sha256', '')):
        raise ValueError('API health must report the actual detector model SHA-256.')


def capture(sample, response, elapsed, health):
    if response.get('read_text') is not True or response.get('angle_correction') is not True:
        raise ValueError('API did not preserve requested OCR and angle correction.')
    if response.get('detector_model') != health.get('detector_model'):
        raise ValueError('Response detector identity differs from preflight health.')
    if response.get('plate_count') != len(response['detections']):
        raise ValueError('Returned plate count disagrees with the preserved detection boxes.')
    if any(not isinstance(plate.get('raw_text'), str) for plate in response['detections']):
        raise ValueError('API must return raw_text for every detected region; normalized text cannot substitute for raw OCR.')
    if response.get('vehicle_detector_ready') is not True:
        raise ValueError('Vehicle detection became unavailable during the audit.')
    analysis = evaluate_targets(response, sample['targets'])
    detections = response['detections']
    review = dict(ocr_requires_review=sum((plate.get('ocr_review') or {}).get('requires_review') is True for plate in detections),
                  ocr_review_missing=sum(not isinstance(plate.get('ocr_review'), dict) for plate in detections),
                  localization_requires_review=sum(plate.get('localization_requires_review') is True for plate in detections),
                  format_status_counts=dict(Counter(plate.get('format_status', 'missing') for plate in detections)))
    return dict(image=str(sample['path']), image_sha256=sample['sha256'], id=sample['id'],
                audit_metadata=sample.get('metadata'), **analysis,
                counters=dict(returned_plate_regions=len(detections), returned_vehicles=len(response.get('vehicles', [])),
                              api_plate_count=response['plate_count'], vehicle_summary=response.get('vehicle_summary'),
                              recognized_plate_summary=response.get('recognized_plate_summary'), review=review),
                latencies_ms=dict(http_ms=elapsed, **{key: response.get(key) for key in TIMING_FIELDS}),
                response=response)


def distribution(values):
    values = sorted(value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))
    def percentile(fraction):
        position = (len(values)-1)*fraction
        lower = int(position)
        upper = min(lower+1, len(values)-1)
        return values[lower] + (values[upper]-values[lower])*(position-lower)
    return dict(samples=len(values), p50=percentile(.5) if values else None,
                p95=percentile(.95) if values else None, min=values[0] if values else None, max=values[-1] if values else None)


def summarize(rows):
    targets = [target for row in rows for target in row['targets']]
    eligible = [target for target in targets if target['score_exact']]
    detected = sum(target['detected'] for target in targets)
    detected_eligible = sum(target['detected'] for target in eligible)
    raw_exact = sum(target['raw_exact'] is True for target in eligible)
    application_exact = sum(target['application_normalized_exact'] is True for target in eligible)
    def metric(numerator, denominator):
        return dict(numerator=numerator, denominator=denominator, rate=numerator/denominator if denominator else None)
    return dict(requests=len(rows), target_detection=metric(detected, len(targets)),
                primary_ocr_metric='raw_ocr_exact_end_to_end',
                raw_ocr_exact_end_to_end=metric(raw_exact, len(eligible)),
                raw_ocr_exact_given_target_detected=metric(raw_exact, detected_eligible),
                application_normalized_exact_end_to_end=metric(application_exact, len(eligible)),
                application_normalized_exact_given_target_detected=metric(application_exact, detected_eligible),
                exact_targets_missed_by_detector=len(eligible)-detected_eligible,
                targets_excluded_from_exact_text=len(targets)-len(eligible),
                returned_plate_region_observations=sum(row['counters']['returned_plate_regions'] for row in rows),
                returned_vehicle_observations=sum(row['counters']['returned_vehicles'] for row in rows),
                review_observations={key: sum(row['counters']['review'][key] for row in rows)
                                     for key in ('ocr_requires_review', 'ocr_review_missing', 'localization_requires_review')},
                detections_outside_scored_targets=sum(row['detections_outside_scored_targets'] for row in rows),
                latencies_ms={key: distribution([row['latencies_ms'][key] for row in rows]) for key in ('http_ms', *TIMING_FIELDS)},
                counting_note='Counters sum observations across requests; they are not unique vehicles or registrations. Unmatched boxes are unscored, not false positives.')


def save_report(path, report):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05*(attempt+1))


def run(args):
    manual, provenance = load_manual(args.truth)
    synthetic = load_synthetic() if args.synthetic else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve an unused output before issuing requests; later writes are atomic.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(dict(status='preparing'), handle)
    aliases = [alias for alias, _ in args.endpoint]
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), command=sys.argv,
                  manual_truth=provenance, request_fields=FIELDS, match_iou=IOU_THRESHOLD,
                  matching='Descending geometric IoU, one to one; tight_plate_box in original-image pixel xywh; text never influences assignment.',
                  primary_ocr_metric='raw_ocr_exact_end_to_end',
                  exact_text_definitions={
                      'raw_ocr_exact': 'Compare raw_text with manual truth after uppercasing and removing whitespace only. No punctuation stripping or character repair; unknown markers remain mismatches.',
                      'application_normalized_exact': 'Compare normalized_text with manual truth. The backend removes non-alphanumeric characters and can substitute positional O/Q/I with 0/1 before this field is returned. The evaluator only uppercases and removes whitespace, periods and hyphens.',
                      'legacy_target_exact_field': 'Each target exact field is an alias for application_normalized_exact, never raw_exact.'},
                  metric_notes=[__doc__, 'Manual latencies are one request per different whole image, including the first/cold request; not a repeated-load benchmark.',
                                'Both raw and application-normalized end-to-end exact metrics include detection misses in their denominator. Their conditional metrics exclude them. Unflagged targets never contribute to either exact metric.',
                                'Full API responses, model hashes, timings and review evidence are preserved; model scores are not accuracy.',
                                'No precision, false-positive rate, vehicle accuracy or automatic model selection is computed.'],
                  models={alias: dict(endpoint=address, manual=[], synthetic_warmup=[], synthetic=[]) for alias, address in args.endpoint})
    if synthetic:
        report['synthetic_fixture'] = dict(image=str(synthetic['path']), image_sha256=synthetic['sha256'],
                                           truth_path=synthetic['truth_path'], truth_sha256=synthetic['truth_sha256'],
                                           distinct_targets=10, warmup_runs=1, measured_runs=5,
                                           note='Fifty target observations per model repeat ten labelled regions from three source vehicles; neighbouring plates remain unscored.')
    current = dict(stage='health')
    def persist():
        for model in report['models'].values():
            model['manual_summary'] = summarize(model['manual'])
            if synthetic:
                model['synthetic_summary'] = summarize(model['synthetic'])
        save_report(args.output, report)
    try:
        for alias in aliases:
            current = dict(stage='health', model=alias)
            model = report['models'][alias]
            health = request_json(model['endpoint']+'/api/health', args.timeout)
            model['health_before'] = health
            validate_health(health)
        persist()
        workloads = [('manual', index, sample, False) for index, sample in enumerate(manual)]
        if synthetic:
            workloads += [('synthetic', index, synthetic, index == 0) for index in range(6)]
        for stage, index, sample, warmup in workloads:
            order = aliases[index % len(aliases):] + aliases[:index % len(aliases)]
            for alias in order:
                current = dict(stage=stage, model=alias, sample=sample['id'], run=index, warmup=warmup)
                model = report['models'][alias]
                response, elapsed = photo_request(model['endpoint'], sample, args.timeout)
                row = capture(sample, response, elapsed, model['health_before'])
                row.update(run=index, warmup=warmup)
                model['synthetic_warmup' if warmup else stage].append(row)
                persist()
                print(json.dumps(dict(**current, targets_detected=sum(item['detected'] for item in row['targets']),
                                      raw_ocr_exact=sum(item['raw_exact'] is True for item in row['targets']),
                                      application_normalized_exact=sum(item['application_normalized_exact'] is True for item in row['targets']),
                                      http_ms=round(elapsed, 2))), flush=True)
        for alias in aliases:
            current = dict(stage='health_after', model=alias)
            model = report['models'][alias]
            after = request_json(model['endpoint']+'/api/health', args.timeout)
            model['health_after'] = after
            validate_health(after)
            for key in ('model_sha256', 'detector_model', 'engine', 'ocr_engine', 'vehicle_engine', 'vehicle_taxonomy', 'region_confidence_threshold'):
                if after.get(key) != model['health_before'].get(key):
                    raise ValueError(f'API {key} changed during evaluation; results cannot be treated as a fixed configuration.')
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=dict(type=type(error).__name__, message=str(error), context=current))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        persist()
    print(json.dumps({alias: {key: model[key] for key in ('manual_summary', 'synthetic_summary') if key in model}
                      for alias, model in report['models'].items()}, indent=2), flush=True)
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f'HTTP audit failed: {error}. Completed observations, if any, are retained in {args.output}.') from error


if __name__ == '__main__':
    main()
