"""CPU-only checks of fair comparison and honest metric reporting."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_nano_models import (alternating_order, benchmark_video, parse_models, percentiles,
                                 rank_candidates, retained_raw_boxes, summarize_scenes,
                                 fixture_ocr_report, scene_confidence_reports, match_assignments,
                                 measure_photos, evaluate_validation)
from evaluate_plate_candidate import match


def test_rotation_balances_first_position_and_keeps_every_model():
    names = ['yolo11n', 'yolo26n', 'baseline']
    orders = [alternating_order(names, index) for index in range(6)]
    assert [order[0] for order in orders] == names * 2
    assert all(sorted(order) == sorted(names) for order in orders)
    assert names == ['yolo11n', 'yolo26n', 'baseline']


def test_partial_annotations_affect_recall_but_not_precision():
    scenes = [
        dict(category='green', complete_labels=True, labelled=1, matched=1, missed=0, unmatched=1, all_labelled_found=True),
        dict(category='green', complete_labels=False, labelled=2, matched=1, missed=1, unmatched=8, all_labelled_found=False),
    ]
    result = summarize_scenes(scenes)
    assert result['green']['recall'] == pytest.approx(2 / 3)
    assert result['green']['precision_on_reviewed_scenes'] == .5
    assert result['green']['false_positives_on_reviewed_scenes'] == 1
    assert result['green']['all_labelled_found_rate'] == .5
    assert result['general']['recall'] is None
    assert result['general']['precision_on_reviewed_scenes'] is None


def model_report(photo_p95, detector_p95, matches):
    scenes = [dict(image=f'{category}.jpg', category=category, complete_labels=True,
                   labelled=10, matched=matches, missed=10 - matches, unmatched=0,
                   all_labelled_found=matches == 10) for category in ('green', 'general')]
    return {
        'photo': {'timings_ms': {'wall_ms': {'p95': photo_p95}}, 'plate_counts': [matches] * 3,
                  'target_matches_by_run': [matches] * 3},
        'detector': {'timings_ms': {'p95': detector_p95}, 'target_localization': {
            'matched': matches, 'missed': 10 - matches, 'unmatched': 0},
                     'target_matches_by_run': [matches] * 3},
        'validation': {'categories': summarize_scenes(scenes), 'scenes': scenes},
    }


def test_selection_uses_full_pipeline_p95_and_retains_accuracy_regression():
    reports = {
        'baseline': model_report(10, 5, 10),
        'yolo11n': model_report(80, 10, 10),
        'yolo26n': model_report(50, 15, 3),
    }
    result = rank_candidates(reports, ['yolo11n', 'yolo26n'])
    assert result['fastest_by_full_photo_p95'] == 'yolo26n'
    assert result['fastest_by_detector_p95'] == 'yolo11n'
    assert result['ranking'][0]['synthetic_target_localization']['missed'] == 7
    assert result['ranking'][0]['validation']['green']['recall'] == .3
    assert result['ranking'][0]['accuracy_eligibility']['eligible_for_recommendation'] is False
    assert result['recommended_for_review'] == 'yolo11n'
    assert result['fastest_eligible_by_photo_p95'] == 'yolo11n'
    assert result['default_model_changed'] is False
    with pytest.raises(ValueError):
        rank_candidates(reports, ['missing'])


def test_photo_filter_can_fail_coverage_even_when_detector_finds_every_target():
    reports = {'baseline': model_report(80, 20, 10), 'candidate': model_report(40, 10, 10)}
    reports['candidate']['photo']['target_matches_by_run'] = [10, 9, 10]
    result = rank_candidates(reports, ['candidate'])
    eligibility = result['ranking'][0]['accuracy_eligibility']
    assert eligibility['synthetic_coverage']['photo']['maximum_missed'] == 1
    assert eligibility['eligible_for_recommendation'] is False
    assert result['recommended_for_review'] is None


def test_reviewed_false_positive_regression_is_not_hidden_by_full_recall():
    reports = {'baseline': model_report(80, 20, 10), 'candidate': model_report(40, 10, 10)}
    scenes = reports['candidate']['validation']['scenes']
    scenes[0]['unmatched'] = 4
    reports['candidate']['validation']['categories'] = summarize_scenes(scenes)
    eligibility = rank_candidates(reports, ['candidate'])['ranking'][0]['accuracy_eligibility']
    assert eligibility['validation_comparable'] is True
    assert eligibility['validation_count_deltas_vs_reference']['green']['false_positives_on_reviewed_scenes'] == 4
    assert eligibility['eligible_for_recommendation'] is False


def test_different_validation_scenes_cannot_receive_accuracy_approval():
    reports = {'baseline': model_report(80, 20, 10), 'candidate': model_report(40, 10, 10)}
    reports['candidate']['validation']['scenes'][0]['image'] = 'different.jpg'
    result = rank_candidates(reports, ['candidate'])
    assert result['recommended_for_review'] is None
    assert result['ranking'][0]['accuracy_eligibility']['validation_comparable'] is False


def test_final_photo_matching_uses_original_boxes_not_padded_evidence_crops():
    boxes = [(0, 10, 20, 10, .9), (50, 10, 20, 10, .8)]
    detections = [{'bounding_box': dict(x=0, y=8, width=25, height=14), 'detection_confidence': .9}]
    assert retained_raw_boxes(boxes, detections, 100, 100, .25) == [boxes[0]]
    assert retained_raw_boxes([boxes[0], boxes[0]], detections, 100, 100, .25) == [boxes[0]]


def test_photo_benchmark_observes_regions_called_by_the_current_pipeline(tmp_path):
    import cv2
    import numpy as np

    class Detector:
        def __init__(self):
            self.calls = []

        def detect_fast(self, image):
            raise AssertionError('Raw timing must use the same region policy as photo processing.')

        def detect_regions(self, image, profile='fast'):
            self.calls.append(('regions', profile))
            return [(12, 10, 40, 20, .86)]

        def detect_vehicle_regions(self, image):
            raise AssertionError('No vehicle detector was supplied to this benchmark.')

    class Reader:
        def read(self, crop):
            return 'GJ01AB1234', .96

    fixture = tmp_path / 'fixture.jpg'
    encoded, image = cv2.imencode('.jpg', np.zeros((60, 100, 3), np.uint8))
    assert encoded
    fixture.write_bytes(image.tobytes())
    detector = Detector()
    report = measure_photos({'candidate': detector}, Reader(), fixture,
                            [[12, 10, 40, 20]], runs=2, warmup=1,
                            expected_texts=['GJ01AB1234'])['candidate']
    assert report['photo']['target_matches_by_run'] == [1, 1]
    assert report['photo']['plate_counts'] == [1, 1]
    assert report['photo']['fixture_ocr']['exact_text_counts_by_run'] == [1, 1]
    assert report['detector']['target_matches_by_run'] == [1, 1]
    assert detector.calls == [('regions', 'fast')] * 6


def test_validation_uses_whole_images_with_the_current_region_policy(tmp_path):
    import cv2
    import numpy as np

    images, labels = tmp_path / 'val/images', tmp_path / 'val/labels'
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    assert cv2.imwrite(str(images / 'scene.jpg'), np.zeros((100, 100, 3), np.uint8))
    (labels / 'scene.txt').write_text('0 .3 .2 .4 .2\n', encoding='utf-8')
    (tmp_path / 'manifest.json').write_text(json.dumps({'green_records': []}), encoding='utf-8')

    class Detector:
        def detect_fast(self, image):
            raise AssertionError('Validation must use the measured photo region policy.')

        def detect_regions(self, image, profile='fast'):
            assert image.shape == (100, 100, 3)
            assert profile == 'fast'
            return [(10, 10, 40, 20, .86)]

    reports, fixtures = evaluate_validation({'candidate': Detector()}, tmp_path)
    scene = reports['candidate']['scenes'][0]
    assert (scene['matched'], scene['unmatched'], scene['missed']) == (1, 0, 0)
    assert len(fixtures) == 1


def test_cli_confidence_applies_equally_to_raw_and_region_pipeline_before_model_loading(tmp_path, monkeypatch):
    import compare_nano_models
    from app.core.config import settings

    first, second = tmp_path / 'first.onnx', tmp_path / 'second.onnx'
    first.write_bytes(b'fixture'); second.write_bytes(b'fixture')
    monkeypatch.setattr(settings, 'confidence_threshold', .11)
    monkeypatch.setattr(settings, 'region_confidence_threshold', .22)
    monkeypatch.setattr(settings, 'openvino_device', settings.openvino_device)
    monkeypatch.setattr(settings, 'inference_threads', settings.inference_threads)
    monkeypatch.setattr(settings, 'acceleration', settings.acceleration)
    monkeypatch.setattr(compare_nano_models.cv2, 'setNumThreads', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['compare_nano_models.py', '--model', f'old={first}',
        '--model', f'new={second}', '--confidence', '.37', '--output', str(tmp_path / 'result.json')])

    class SettingsObserved(Exception):
        pass

    def observe_settings(*args):
        assert settings.confidence_threshold == .37
        assert settings.region_confidence_threshold == .37
        raise SettingsObserved  # Stop before any model or GPU is initialized.

    monkeypatch.setattr(compare_nano_models, 'load_model', observe_settings)
    with pytest.raises(SettingsObserved):
        compare_nano_models.main()


def test_fixture_exact_text_requires_correct_one_to_one_localization():
    truth = [[0, 0, 20, 10], [40, 0, 20, 10], [80, 0, 20, 10]]
    def reading(text, score):
        return dict(normalized_text=text, raw_text=text, detection_confidence=score, ocr_confidence=.85)
    observations = [
        ((0, 0, 20, 10, .9), reading('WRONG', .9)),
        ((0, 0, 20, 10, .8), reading('AB12CD3456', .8)),
        ((80, 0, 20, 10, .7), reading('AB12CD3456', .7)),
    ]
    result = fixture_ocr_report(observations, truth, ['AB12CD3456'] * 3)
    assert result['detected_targets'] == 2
    assert result['exact_text_targets'] == 1  # Duplicate correct text cannot rescue higher-score wrong text.
    assert result['unmatched_predictions'] == 1
    assert result['missed_targets'] == 1
    assert result['targets'][1]['ocr_confidence'] is None
    assert result['targets'][2]['detection_confidence'] == .7
    assert result['distinct_expected_registrations'] == 1


def test_confidence_comparison_separates_false_positives_and_uses_common_targets():
    def scene(image, matches, fps, complete=True):
        return dict(image=image, category='green', complete_labels=complete,
                    truth_matches=[dict(truth_index=index, confidence=score) for index, score in matches],
                    unmatched_confidences=fps)
    rows = {
        'v8': [scene('same.jpg', [(0, .9), (1, .4)], []),
               scene('partial.jpg', [(0, .99)], [.99], complete=False)],
        'v26': [scene('same.jpg', [(0, .6)], [.95]),
                scene('partial.jpg', [(0, .1)], [], complete=False)],
    }
    result = scene_confidence_reports(rows)
    original = result['v8']['categories']['green']
    candidate = result['v26']['categories']['green']
    assert original['true_positives']['count'] == 2
    assert original['true_positives']['mean'] == pytest.approx(.65)
    assert original['common_true_positives'] == dict(count=1, mean=.9, median=.9)
    assert candidate['common_true_positives'] == dict(count=1, mean=.6, median=.6)
    assert candidate['false_positives'] == dict(count=1, mean=.95, median=.95)
    assert result['v8']['categories']['general']['true_positives']['mean'] is None


def test_assignment_counts_match_existing_iou_evaluator():
    truth = [[5, 5, 20, 10], [45, 5, 20, 10]]
    predictions = [(5, 5, 20, 10, .9), (5, 5, 20, 10, .8)]
    assigned, unmatched, missed = match_assignments(predictions, truth)
    assert (len(assigned), len(unmatched), len(missed)) == match(predictions, truth)
    assert assigned == [(0, 0)]


def test_optional_test_report_cannot_change_validation_selection():
    reports = {'baseline': model_report(80, 20, 10), 'candidate': model_report(40, 10, 10)}
    before = rank_candidates(reports, ['candidate'])
    reports['candidate']['held_out_test'] = {'matched': 0, 'missed': 1000, 'false_positives': 1000}
    reports['baseline']['held_out_test'] = {'matched': 1000, 'missed': 0, 'false_positives': 0}
    assert rank_candidates(reports, ['candidate']) == before


def test_overlapping_duplicates_never_count_as_additional_true_positives():
    assert match([(5, 5, 20, 10, .9), (5, 5, 20, 10, .8)], [[5, 5, 20, 10], [45, 5, 20, 10]]) == (1, 1, 1)


def test_model_names_are_unique_and_windows_drive_colons_are_preserved(tmp_path):
    first = tmp_path / 'yolo11.onnx'; first.write_bytes(b'one')
    second = tmp_path / 'yolo26.onnx'; second.write_bytes(b'two')
    assert parse_models([f'11={first}', f'26={second}']) == {'11': first, '26': second}
    with pytest.raises(ValueError):
        parse_models([f'11={first}', f'11={second}'])
    with pytest.raises(ValueError):
        parse_models([f'11={first}'])


def test_latency_summary_does_not_hide_invalid_samples():
    assert percentiles([10, 20, 30])['p50'] == 20
    for values in ([], [float('nan')], [-1]):
        with pytest.raises(ValueError):
            percentiles(values)


def test_video_jobs_delete_only_their_own_copy_and_each_model_receives_the_source(tmp_path, monkeypatch):
    from app.services import video
    import compare_nano_models
    source = tmp_path / 'user-recording.mov'
    payload = b'original recording bytes'
    source.write_bytes(payload)
    received = []

    def fake_process(job, detector, reader):
        assert job.source != source
        assert job.source.parent == job.folder
        received.append(job.source.read_bytes())
        job.source.unlink()  # Mirror the real production job's upload cleanup.
        (job.folder / 'results.json').write_text(json.dumps({'frames': 2, 'all_frames_searched': True}))
        job.update(status='complete')

    class FakeCapture:
        def __init__(self, path):
            self.remaining = 2

        def read(self):
            self.remaining -= 1
            return self.remaining >= 0, None

        def release(self):
            pass

    monkeypatch.setattr(video, 'process_video', fake_process)
    monkeypatch.setattr(compare_nano_models.cv2, 'VideoCapture', FakeCapture)
    for name in ('yolo11n', 'yolo26n'):
        report = benchmark_video(object(), object(), source, tmp_path / name)
        assert report['decoded_export_frames'] == 2
        assert source.read_bytes() == payload
    assert received == [payload, payload]
