"""Compare plate-trained nano exports using warmed production detector and OCR code.

Selection uses validation only; optional held-out reporting happens afterwards.
It does not activate a model. Detector timing includes resize, inference and postprocessing;
Photo timing adds decoding, OCR and evidence crops. It excludes vehicle detection,
vehicle-crop search, angle correction, cross-camera matching and HTTP/UI.
Run alone after training stops so competing GPU/CPU jobs do not distort results.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platevision/backend'))
from evaluate_plate_candidate import load_truth, match, overlap


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def alternating_order(names, round_index):
    """Rotate first position so one model is not always measured first."""
    offset = round_index % len(names)
    return list(names[offset:]) + list(names[:offset])


def percentiles(values):
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Latency samples must be finite, nonnegative and nonempty.')
    return {'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95))}


def confidence_summary(values):
    return {'count': len(values), 'mean': float(np.mean(values)) if values else None,
            'median': float(np.median(values)) if values else None}


def match_assignments(predictions, truth):
    """Same confidence-first IoU>=.5 matching as the existing count evaluator."""
    remaining = list(range(len(truth)))
    matched, unmatched = [], []
    for index in sorted(range(len(predictions)), key=lambda i: predictions[i][4], reverse=True):
        scores = [overlap(predictions[index], truth[target]) for target in remaining]
        if scores and max(scores) >= .5:
            matched.append((index, remaining.pop(scores.index(max(scores)))))
        else:
            unmatched.append(index)
    return matched, unmatched, remaining


def scene_confidence_reports(scenes_by_model):
    """Compare scores on shared truth targets, separately from measured accuracy."""
    by_image = {name: {row['image']: row for row in rows if row['complete_labels']}
                for name, rows in scenes_by_model.items()}
    common_images = set.intersection(*(set(rows) for rows in by_image.values()))
    common_targets = {image: set.intersection(*(set(item['truth_index'] for item in rows[image]['truth_matches'])
                      for rows in by_image.values())) for image in common_images}
    reports = {}
    for name, images in by_image.items():
        categories = {}
        for category in ('all', 'green', 'general'):
            rows = [row for row in images.values() if category == 'all' or row['category'] == category]
            tp = [item['confidence'] for row in rows for item in row['truth_matches']]
            fp = [score for row in rows for score in row['unmatched_confidences']]
            common = [item['confidence'] for row in rows for item in row['truth_matches']
                      if item['truth_index'] in common_targets.get(row['image'], set())]
            categories[category] = {'true_positives': confidence_summary(tp),
                                    'false_positives': confidence_summary(fp),
                                    'common_true_positives': confidence_summary(common)}
        reports[name] = {'reviewed_scenes_only': True, 'compared_models': list(scenes_by_model),
                         'categories': categories,
                         'note': 'Scores are uncalibrated and not accuracy. Common true-positive scores use exactly the same matched truth targets across every compared model.'}
    return reports


def summarize_scenes(scenes):
    categories = {}
    for category in ('green', 'general'):
        rows = [row for row in scenes if row['category'] == category]
        complete = [row for row in rows if row['complete_labels']]
        tp = sum(row['matched'] for row in rows)
        fn = sum(row['missed'] for row in rows)
        ctp = sum(row['matched'] for row in complete)
        fp = sum(row['unmatched'] for row in complete)
        all_found = sum(row['all_labelled_found'] for row in rows)
        categories[category] = {
            'images': len(rows), 'labelled': tp + fn, 'matched': tp, 'missed': fn,
            'recall': tp / (tp + fn) if tp + fn else None,
            'all_labelled_found': all_found,
            'all_labelled_found_rate': all_found / len(rows) if rows else None,
            'precision_on_reviewed_scenes': ctp / (ctp + fp) if ctp + fp else None,
            'precision_scene_count': len(complete), 'false_positives_on_reviewed_scenes': fp,
            'matched_on_reviewed_scenes': ctp,
            'labelled_on_reviewed_scenes': sum(row['labelled'] for row in complete),
        }
    return categories


def accuracy_eligibility(entry, reference, target_count):
    """Conservative, zero-tolerance evidence checks, not statistical equivalence."""
    issues = []
    coverage = {}
    for stage in ('detector', 'photo'):
        counts = entry[stage].get('target_matches_by_run', [])
        minimum = min(counts) if counts else None
        coverage[stage] = {'minimum_matched': minimum, 'required_targets': target_count,
                           'maximum_missed': target_count - minimum if minimum is not None else None}
        if not counts or any(count != target_count for count in counts):
            issues.append(f'{stage}: complete target coverage was not demonstrated on every measured run.')
    def signature(report):
        return sorted((row['image'], row['category'], row['complete_labels'], row['labelled'])
                      for row in report['validation']['scenes'])
    comparable = signature(entry) == signature(reference) and bool(signature(entry))
    if not comparable:
        issues.append('Validation images/label counts/reviewed-scene membership differ from the reference.')
    deltas = {}
    for category in ('green', 'general'):
        actual = entry['validation']['categories'][category]
        expected = reference['validation']['categories'][category]
        metrics = ('matched', 'missed', 'matched_on_reviewed_scenes',
                   'false_positives_on_reviewed_scenes', 'all_labelled_found')
        deltas[category] = {key: actual[key] - expected[key] for key in metrics}
        deltas[category]['recall_delta'] = (actual['recall'] - expected['recall']
            if actual['recall'] is not None and expected['recall'] is not None else None)
        ap, ep = actual['precision_on_reviewed_scenes'], expected['precision_on_reviewed_scenes']
        deltas[category]['reviewed_precision_delta'] = ap - ep if ap is not None and ep is not None else None
        if not actual['labelled'] or not actual['precision_scene_count']:
            issues.append(f'{category}: insufficient labelled and reviewed-scene evidence.')
        if (actual['matched'] < expected['matched'] or
                actual['matched_on_reviewed_scenes'] < expected['matched_on_reviewed_scenes'] or
                actual['false_positives_on_reviewed_scenes'] > expected['false_positives_on_reviewed_scenes'] or
                actual['all_labelled_found'] < expected['all_labelled_found']):
            issues.append(f'{category}: at least one localization count regressed against the reference; inspect deltas.')
    return {'eligible_for_recommendation': not issues, 'issues': issues,
            'synthetic_coverage': coverage, 'validation_comparable': comparable,
            'validation_count_deltas_vs_reference': deltas}


def rank_candidates(reports, candidates, reference=None, target_count=10):
    """Retain raw speed rankings, but never recommend a win caused by misses."""
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('Selection candidates must be nonempty and unique.')
    reference = reference or next(iter(reports))
    if reference not in reports:
        raise ValueError(f'Unknown accuracy reference: {reference}')
    ranking = []
    for name in candidates:
        if name not in reports:
            raise ValueError(f'Unknown candidate: {name}')
        entry = reports[name]
        p95 = entry['photo']['timings_ms']['wall_ms']['p95']
        if not math.isfinite(p95) or p95 <= 0:
            raise ValueError(f'Invalid photo p95 for {name}')
        ranking.append({
            'name': name, 'full_photo_p95_ms': p95,
            'detector_p95_ms': entry['detector']['timings_ms']['p95'],
            'synthetic_target_localization': entry['detector']['target_localization'],
            'returned_plate_counts': entry['photo']['plate_counts'],
            'validation': entry['validation']['categories'],
            'accuracy_eligibility': accuracy_eligibility(entry, reports[reference], target_count),
        })
    ranking.sort(key=lambda row: (row['full_photo_p95_ms'], row['name']))
    eligible = [row for row in ranking if row['accuracy_eligibility']['eligible_for_recommendation']]
    recommendation = min(eligible, key=lambda row: (row['detector_p95_ms'], row['name']))['name'] if eligible else None
    return {
        'fastest_by_full_photo_p95': ranking[0]['name'],
        'fastest_by_detector_p95': min(ranking, key=lambda row: row['detector_p95_ms'])['name'],
        'raw_ranking_criterion': 'Lowest warmed full-photo p95 on the same synthetic 1080p/10-target fixture, without accuracy gating.',
        'recommendation_criterion': 'Lowest warmed detector p95 among candidates passing every accuracy eligibility check against the named reference.',
        'ranking': ranking, 'accuracy_reference': reference, 'default_model_changed': False,
        'accuracy_rule': 'Every synthetic target retained on every detector and photo run; identical whole-image validation/reviewed labels; no lower TP/all-plates-found counts or higher reviewed-scene FP in either category. No tolerance margin or statistical equivalence claim.',
        'recommended_for_review': recommendation,
        'fastest_eligible_by_photo_p95': eligible[0]['name'] if eligible else None,
        'recommendation': (f'{recommendation} has the lowest detector p95 among models clearing the reference accuracy checks. Review full-photo/OCR costs before activation.'
                           if eligible else 'No candidate clears the accuracy checks. Keep the reference active and review coverage/false-positive regressions before switching.'),
        'note': 'Raw speed rankings are not accuracy approvals. Missing plates can reduce OCR work. This report does not activate a model.',
    }


def hardware_info(device):
    import openvino as ov
    core = ov.Core()
    packages = {}
    for package in ('openvino', 'onnxruntime', 'numpy', 'opencv-python', 'rapidocr-onnxruntime'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        'platform': platform.platform(), 'processor': platform.processor(),
        'cpu_identifier': os.environ.get('PROCESSOR_IDENTIFIER'),
        'logical_cpu_count': os.cpu_count(), 'python': sys.version,
        'openvino_device': device, 'device_name': str(core.get_property(device, 'FULL_DEVICE_NAME')),
        'available_devices': core.available_devices, 'packages': packages,
    }


def load_model(path, name, device, height, width):
    from app.services.detector import OpenVINOPlateDetector
    detector = OpenVINOPlateDetector(str(path), device=device)
    detector.fast_models = {
        'landscape': OpenVINOPlateDetector(str(path), device=device, input_shape=(height, width)),
        'portrait': OpenVINOPlateDetector(str(path), device=device, input_shape=(width, height)),
    }
    detector.model_id = name
    # Model provenance is reported separately; do not infer training from a filename.
    return detector


def runtime_description(detector):
    return {name: {'runtime': graph.runtime, 'output_format': graph.output_format,
                   'input_hw': [graph.height, graph.width],
                   'compiled_input_shape': list(map(int, graph.compiled.input(0).shape)),
                   'compiled_output_shape': list(map(int, graph.compiled.output(0).shape)),
                   'application_overlap_suppression': graph.output_format != 'end_to_end'}
            for name, graph in {'base': detector, **detector.fast_models}.items()}


class ObservedDetector:
    """Observe fresh model boxes used by the pipeline, without another inference."""
    def __init__(self, detector):
        self.detector, self.boxes = detector, []

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def detect_fast(self, image):
        self.boxes = self.detector.detect_fast(image)
        return self.boxes

    def detect_regions(self, image, profile='fast'):
        self.boxes = self.detector.detect_regions(image, profile=profile)
        return self.boxes


def retained_observations(raw_boxes, detections, width, height, padding):
    """Map final readings one-to-one to their original, unpadded model boxes."""
    from app.services.pipeline import clamp
    buckets = {}
    for box in raw_boxes:
        buckets.setdefault((clamp(box[:4], width, height, padding), box[4]), []).append(box)
    observations = []
    for row in detections:
        key = (tuple(row['bounding_box'][key] for key in ('x', 'y', 'width', 'height')),
               row['detection_confidence'])
        if buckets.get(key):
            observations.append((buckets[key].pop(0), row))
    return observations


def retained_raw_boxes(raw_boxes, detections, width, height, padding):
    return [box for box, _ in retained_observations(raw_boxes, detections, width, height, padding)]


def fixture_ocr_report(observations, truth, expected_texts):
    """Exact normalized-string score only for the explicitly transcribed fixture."""
    if len(expected_texts) != len(truth) or not all(expected_texts):
        raise ValueError('Every fixture target needs its explicit expected registration.')
    matched, unmatched, missed = match_assignments([box for box, _ in observations], truth)
    readings = {target: observations[index][1] for index, target in matched}
    targets = []
    for index, expected in enumerate(expected_texts):
        reading = readings.get(index)
        actual = reading['normalized_text'] if reading else None
        targets.append({'target_index': index, 'expected_text': expected, 'detected': reading is not None,
                        'normalized_text': actual, 'raw_text': reading['raw_text'] if reading else None,
                        'exact_text': actual == expected,
                        'detection_confidence': reading['detection_confidence'] if reading else None,
                        'ocr_confidence': reading['ocr_confidence'] if reading else None})
    return {'labelled_targets': len(truth), 'detected_targets': len(matched),
            'exact_text_targets': sum(row['exact_text'] for row in targets),
            'missed_targets': len(missed), 'unmatched_predictions': len(unmatched),
            'distinct_expected_registrations': len(set(expected_texts)), 'targets': targets,
            'note': 'One-to-one IoU>=.5 matching uses unpadded boxes retained by the real photo pipeline. Exact text is evaluated only on this synthetic, explicitly transcribed fixture; repeated source plates are not independent road examples.'}


def measure_fixed_ocr(reader, fixture, truth, runs, warmup):
    """Shared OCR load from every known fixture target, independent of model misses."""
    from app.services.pipeline import clamp
    image = cv2.imread(str(fixture))
    if image is None:
        raise ValueError(f'Unreadable OCR fixture: {fixture}')
    height, width = image.shape[:2]
    pairs = []
    for box in truth:
        x, y, w, h = clamp(box, width, height, .05)
        tx, ty, tw, th = clamp(box, width, height, 0)
        if not tw or not th:
            raise ValueError('A fixed OCR target is outside the fixture.')
        pairs.append((image[y:y+h, x:x+w].copy(), image[ty:ty+th, tx:tx+tw].copy()))
    samples = []
    for index in range(warmup + runs):
        tick = time.perf_counter()
        readings = reader.read_many(pairs)
        elapsed = (time.perf_counter() - tick) * 1000
        if len(readings) != len(truth):
            raise RuntimeError('Fixed OCR benchmark did not return every target.')
        if index >= warmup:
            samples.append(elapsed)
    return {'timings_ms': percentiles(samples), 'samples_ms': samples, 'plate_count': len(pairs),
            'last_readings': [{'raw_text': text, 'ocr_confidence': confidence} for text, confidence in readings],
            'note': 'Shared identical ground-truth crops are used only for this isolated OCR diagnostic. They are never supplied to detection/full-photo/validation. Includes all OCR retries, excludes fixture decode/crop preparation; no OCR accuracy is claimed.'}


def measure_photos(detectors, reader, fixture, truth, runs, warmup, expected_texts=None):
    from app.core.config import settings
    from app.services.pipeline import CropStore, process
    data = fixture.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unreadable fixture: {fixture}')
    names = list(detectors)
    records = {name: {'detector_samples': [], 'photo_samples': []} for name in names}
    stores = {name: CropStore() for name in names}
    observed = {name: ObservedDetector(detector) for name, detector in detectors.items()}
    # Separate stages avoid mixing raw detection timing with preceding OCR costs.
    for stage in ('detector', 'photo'):
        for index in range(warmup + runs):
            for name in alternating_order(names, index):
                detector = detectors[name]
                tick = time.perf_counter()
                if stage == 'detector':
                    result = detector.detect_regions(image, profile='fast')
                else:
                    result = process(data, observed[name], stores[name], reader,
                                     vehicle_detector=None, profile='fast', read_text=True,
                                     angle_correction=False, reid_context=None)
                elapsed = (time.perf_counter() - tick) * 1000
                if index < warmup:
                    continue
                if stage == 'detector':
                    tp, fp, fn = match(result, truth)
                    records[name]['detector_samples'].append(elapsed)
                    records[name].setdefault('target_matches_by_run', []).append(tp)
                    records[name]['detector_boxes'] = [list(box) for box in result]
                    records[name]['target_localization'] = {'matched': tp, 'unmatched': fp, 'missed': fn}
                else:
                    observations = retained_observations(observed[name].boxes, result['detections'],
                                                        image.shape[1], image.shape[0], settings.plate_crop_padding)
                    retained = [box for box, _ in observations]
                    matched, _, _ = match(retained, truth)
                    records[name].setdefault('photo_target_matches_by_run', []).append(matched)
                    if expected_texts is not None:
                        score = fixture_ocr_report(observations, truth, expected_texts)
                        records[name]['fixture_ocr'] = score
                        records[name].setdefault('exact_text_counts_by_run', []).append(score['exact_text_targets'])
                    sample = {key: result[key] for key in (
                        'decode_time_ms', 'detection_time_ms', 'ocr_time_ms', 'processing_time_ms')}
                    sample.update(wall_ms=elapsed, plate_count=len(result['detections']))
                    records[name]['photo_samples'].append(sample)
                    records[name]['last_readings'] = [{key: detection[key] for key in (
                        'bounding_box', 'detection_confidence', 'raw_text', 'normalized_text',
                        'ocr_confidence', 'format_status')} for detection in result['detections']]
    reports = {}
    for name, record in records.items():
        reports[name] = {
            'detector': {
                'timings_ms': percentiles(record['detector_samples']),
                'samples_ms': record['detector_samples'],
                'target_matches_by_run': record['target_matches_by_run'],
                'target_localization': record['target_localization'],
                'last_boxes': record['detector_boxes'],
            },
            'photo': {
                'timings_ms': {key: percentiles([sample[key] for sample in record['photo_samples']])
                               for key in ('wall_ms', 'processing_time_ms', 'decode_time_ms',
                                           'detection_time_ms', 'ocr_time_ms')},
                'plate_counts': [sample['plate_count'] for sample in record['photo_samples']],
                'target_matches_by_run': record['photo_target_matches_by_run'],
                'samples': record['photo_samples'], 'last_readings': record['last_readings'],
                'last_reading_confidence': {key: confidence_summary([row[key] for row in record['last_readings']])
                                            for key in ('detection_confidence', 'ocr_confidence')},
            },
        }
        if 'fixture_ocr' in record:
            reports[name]['photo']['fixture_ocr'] = {**record['fixture_ocr'],
                'exact_text_counts_by_run': record['exact_text_counts_by_run']}
    return reports


def evaluate_validation(detectors, dataset, split='val'):
    if split not in ('val', 'test'):
        raise ValueError('Only val or test evaluation is supported.')
    metadata = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    green = {row['id']: row for row in metadata['green_records']}
    paths = sorted(path for path in (dataset / split / 'images').glob('*.jpg')
                   if '_context_' not in path.stem)
    if not paths:
        raise ValueError('No whole-image validation fixtures found.')
    names = list(detectors)
    scenes = {name: [] for name in names}
    fixtures = []
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable validation image: {path}')
        height, width = image.shape[:2]
        label = dataset / split / 'labels' / f'{path.stem}.txt'
        truth = load_truth(label, width, height)
        category = 'green' if path.stem in green else 'general'
        complete = category != 'green' or green[path.stem]['full_scene_reviewed']
        fixtures.append({'image': path.name, 'image_sha256': digest(path), 'label_sha256': digest(label)})
        for name in alternating_order(names, index):
            boxes = detectors[name].detect_regions(image, profile='fast')
            assignments, unmatched, missed = match_assignments(boxes, truth)
            tp, fp, fn = len(assignments), len(unmatched), len(missed)
            scenes[name].append({
                'image': path.name, 'category': category, 'complete_labels': complete,
                'labelled': len(truth), 'matched': tp, 'unmatched': fp, 'missed': fn,
                'all_labelled_found': fn == 0, 'predictions': [list(box) for box in boxes],
                'truth_matches': [{'truth_index': target, 'confidence': float(boxes[prediction][4])}
                                  for prediction, target in assignments],
                'unmatched_confidences': [float(boxes[index][4]) for index in unmatched],
            })
    confidences = scene_confidence_reports(scenes)
    return ({name: {'split': split, 'categories': summarize_scenes(rows), 'scenes': rows,
                    'detector_confidence': confidences[name]}
             for name, rows in scenes.items()}, fixtures)


def benchmark_video(detector, reader, source, folder):
    from app.services.video import VideoJob, process_video
    folder.mkdir(parents=True, exist_ok=False)
    # Production jobs own and delete their upload after processing. Each model
    # must own a separate copy; never give a job the user's/shared source path.
    local_source = folder / ('input' + source.suffix)
    shutil.copyfile(source, local_source)
    job = VideoJob(folder.name, folder, local_source, 'fast')
    tick = time.perf_counter()
    process_video(job, detector, reader)
    elapsed = (time.perf_counter() - tick) * 1000
    if job.snapshot()['status'] != 'complete':
        raise RuntimeError(job.snapshot())
    report = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
    capture = cv2.VideoCapture(str(folder / 'annotated.mp4'))
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    capture.release()
    if frames != report['frames'] or not report['all_frames_searched']:
        raise RuntimeError('Video benchmark did not retain/search every frame.')
    report.update(call_wall_ms=elapsed, decoded_export_frames=frames,
                  export_file=str(folder / 'annotated.mp4'))
    return report


def parse_models(values):
    models = {}
    for value in values:
        name, separator, filename = value.partition('=')
        if not separator or not name.strip() or not filename.strip() or name in models:
            raise ValueError('Each --model must have a unique NAME=PATH value.')
        path = Path(filename).resolve()
        if not path.is_file():
            raise ValueError(f'Model does not exist: {path}')
        models[name] = path
    if len(models) < 2:
        raise ValueError('Provide at least two models for comparison.')
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--candidate', action='append', help='Eligible speed-selection name; repeat. Defaults to all models.')
    parser.add_argument('--reference', help='Accuracy reference name; defaults to the first --model.')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'platevision/data/combined-green-20260919')
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/ten-plate-1080p.jpg')
    parser.add_argument('--truth', type=Path, default=ROOT / 'tests/ten-plate-fast-benchmark.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--device', default='GPU')
    parser.add_argument('--input-height', type=int, default=864)
    parser.add_argument('--input-width', type=int, default=1536)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--report-test', action='store_true', help='Report held-out test AFTER validation-only selection; never change selection using test.')
    videos = parser.add_mutually_exclusive_group()
    videos.add_argument('--video', type=Path, help='Optional identical video source for all models.')
    videos.add_argument('--synthetic-video', action='store_true', help='Also export a 90-frame ten-plate fixture video.')
    args = parser.parse_args()
    if args.runs < 2 or args.warmup < 1 or not 0 < args.confidence < 1 or not 1 <= args.threads <= 32:
        parser.error('Use runs >= 2, warmup >= 1, confidence in (0,1), and threads in [1,32].')
    if any(size < 320 or size % 32 for size in (args.input_height, args.input_width)):
        parser.error('Input sides must be multiples of 32, at least 320.')
    try:
        models = parse_models(args.model)
    except ValueError as error:
        parser.error(str(error))
    candidates = args.candidate or list(models)
    if len(set(candidates)) != len(candidates) or any(name not in models for name in candidates):
        parser.error('Candidates must be unique names provided with --model.')
    if args.reference is not None and args.reference not in models:
        parser.error('--reference must be a name provided with --model.')
    from app.core.config import settings
    from app.services.ocr import PlateOCR
    settings.confidence_threshold = args.confidence
    settings.region_confidence_threshold = args.confidence
    settings.openvino_device = args.device
    settings.inference_threads = args.threads
    settings.acceleration = 'openvino'
    cv2.setNumThreads(args.threads)
    detectors = {}
    for name, path in models.items():
        print(f'Loading {name}: {path}', flush=True)
        detectors[name] = load_model(path, name, args.device, args.input_height, args.input_width)
    reader = PlateOCR(backend='ppocr')
    truth_rows = json.loads(args.truth.read_text(encoding='utf-8'))['truth']
    truth = [row['box'] for row in truth_rows]
    expected_texts = [row.get('text', '') for row in truth_rows]
    expected_texts = expected_texts if all(expected_texts) else None
    reports = measure_photos(detectors, reader, args.fixture, truth, args.runs, args.warmup, expected_texts)
    fixed_ocr = measure_fixed_ocr(reader, args.fixture, truth, args.runs, args.warmup)
    print('Photo timing complete; evaluating whole-image validation split.', flush=True)
    validation, fixtures = evaluate_validation(detectors, args.dataset)
    for name, path in models.items():
        reports[name].update(model=str(path), sha256=digest(path), validation=validation[name],
                             runtime_graphs=runtime_description(detectors[name]))
    selection = rank_candidates(reports, candidates, args.reference, len(truth))
    test_fixtures = None
    if args.report_test:
        print('Validation selection frozen; reporting held-out test without changing the selection.', flush=True)
        test_reports, test_fixtures = evaluate_validation(detectors, args.dataset, split='test')
        for name in reports:
            reports[name]['held_out_test'] = test_reports[name]
    source = args.video
    if args.synthetic_video or source:
        folder = args.output.parent / f'nano-videos-{uuid.uuid4().hex}'
        folder.mkdir(parents=True)
        if args.synthetic_video:
            from benchmark_candidate_runtime import make_video
            source = folder / 'input.mp4'
            make_video(args.fixture, source)
        source = source.resolve()
        for index, (name, detector) in enumerate(detectors.items()):
            print(f'Exporting identical video with {name}.', flush=True)
            reports[name]['video'] = benchmark_video(detector, reader, source, folder / str(index))
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'hardware': hardware_info(args.device),
        'implementation_sha256': {name: digest(ROOT / path) for name, path in {
            'comparison': 'scripts/compare_nano_models.py',
            'detector': 'platevision/backend/app/services/detector.py',
            'pipeline': 'platevision/backend/app/services/pipeline.py',
            'ocr': 'platevision/backend/app/services/ocr.py',
            'video': 'platevision/backend/app/services/video.py',
        }.items()},
        'settings': {
            'input_landscape_hw': [args.input_height, args.input_width],
            'input_portrait_hw': [args.input_width, args.input_height], 'profile': 'fast',
            'confidence_threshold': settings.confidence_threshold, 'raw_nms_iou': .45,
            'region_confidence_threshold': settings.region_confidence_threshold,
            'match_iou': .5, 'max_detections': settings.max_detections,
            'unverified_plate_threshold': settings.unverified_plate_threshold,
            'plate_crop_padding': settings.plate_crop_padding, 'opencv_threads': args.threads,
            'inference_threads': args.threads, 'ocr_runtime': reader.runtime,
            'ocr_batch_size': settings.fast_ocr_batch_size, 'runs_per_model': args.runs,
            'warmup_per_stage_per_model': args.warmup,
            'constructor_warmup': 'Additional runtime graph warmup excluded from samples.',
            'measurement_order': 'Stages: detector then full photo; model first position rotates every round.',
        },
        'photo_measurement_scope': {
            'pipeline': 'detector_and_ocr', 'detector_method': 'detect_regions',
            'profile': 'fast', 'read_text': True, 'angle_correction': False,
            'vehicle_detector': False, 'vehicle_crop_search': False,
            'vehicle_roi_status': 'unavailable_without_vehicle_detector',
            'cross_camera_matching': False,
            'note': 'Production detector and OCR code with these explicit options; not the complete application configuration. Raw detector timing and validation use the same detect_regions policy.',
        },
        'fixture': {'image': str(args.fixture.resolve()), 'sha256': digest(args.fixture),
                    'truth_sha256': digest(args.truth), 'labelled_targets': len(truth)},
        'dataset': {'path': str(args.dataset.resolve()), 'manifest_sha256': digest(args.dataset / 'manifest.json'),
                    'split': 'val', 'whole_image_fixtures': fixtures},
        'models': reports, 'fixed_target_ocr': fixed_ocr,
        'selection': selection,
        'notes': [
            'All photo requests decode and infer afresh. CropStore retains evidence only; no recognition-result cache or benchmark plate-count cutoff is used.',
            'Synthetic ten-plate timings measure load, not real-road accuracy. HTTP/browser and cold model loading are excluded.',
            'Photo measurements run OCR on, angle correction off, without a vehicle detector, vehicle-crop search or cross-camera matching. They do not measure the complete app configuration.',
            'Fixed-target OCR is an isolated diagnostic with 5% crop padding; photo OCR uses the configured plate_crop_padding and detected regions.',
            'Validation uses whole images only, without ground-truth crop hints. Optional test reporting runs AFTER validation-only selection is frozen; test scores never feed ranking.',
            'Green recall covers supplied labels; precision is computed only on reviewed complete scenes. General scenes use existing annotations as complete.',
            'Detection and OCR confidences are separate, uncalibrated model scores. Real-image OCR exact-string accuracy is not claimed without verified transcripts; synthetic fixture exact text is explicitly reported separately.',
            'Optional video timing includes CPU decoding, every-frame detection, track OCR, overlay rendering and H.264 encoding. Its amortized ms/frame is not single-frame end-to-end latency.',
            'Video export is measured once per model in listed order; photo/detector p95 use alternating repeated measurements.',
        ],
    }
    if test_fixtures is not None:
        report['held_out_test'] = {'whole_image_fixtures': test_fixtures, 'used_for_selection': False,
            'note': 'Reserved split previously evaluated for YOLOv8; never used for this training. Reported after selection; no test-based threshold or model tuning performed by this harness.'}
    if source:
        report['video_source'] = {'path': str(source), 'sha256': digest(source)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['selection'], indent=2))
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
