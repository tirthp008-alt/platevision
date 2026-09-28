"""Pure scoring and fake-HTTP orchestration; never contact an API or run inference."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_plate_ocr_http as audit


def detection(box, text, **extra):
    return dict(tight_plate_box=dict(zip(('x', 'y', 'width', 'height'), box)),
                bounding_box=dict(x=0, y=0, width=1000, height=1000), normalized_text=text,
                raw_text=extra.pop('raw_text', text),
                detection_confidence=.8, ocr_confidence=.9, format_status='valid',
                ocr_review=dict(requires_review=False, attempts=[dict(method='original')]),
                localization_requires_review=False, **extra)


def test_geometric_match_does_not_choose_the_reading_that_happens_to_match_truth():
    response = dict(detections=[detection([1, 0, 10, 10], 'KA01AB1234'),
                                detection([0, 0, 10, 10], 'WRONG'),
                                detection([100, 100, 10, 10], 'OTHER')])
    target = dict(id='one', box=[0, 0, 10, 10], text='KA01AB1234', score_exact=True)
    result = audit.evaluate_targets(response, [target])
    assert result['targets'][0]['detection_index'] == 1
    assert result['targets'][0]['detected'] is True
    assert result['targets'][0]['exact'] is False
    assert result['detections_outside_scored_targets'] == 2
    assert 'false_positives' not in result


def test_matching_uses_tight_xywh_named_keys_and_one_to_one_half_iou_boundary():
    plate = detection([0, 0, 10, 10], 'AB123')
    plate['tight_plate_box'] = dict(height=10, width=10, y=0, x=0)
    targets = [dict(id='a', box=[0, 0, 20, 10], text='AB123', score_exact=True),
               dict(id='b', box=[0, 0, 20, 10], text='AB123', score_exact=True)]
    result = audit.evaluate_targets(dict(detections=[plate]), targets)
    assert [row['detected'] for row in result['targets']] == [True, False]
    assert result['targets'][0]['match_iou'] == .5
    targets[0]['box'] = [0, 0, 20.01, 10]
    assert audit.evaluate_targets(dict(detections=[plate]), targets[:1])['targets'][0]['detected'] is False
    del plate['tight_plate_box']
    with pytest.raises(KeyError):
        audit.evaluate_targets(dict(detections=[plate]), targets)


def health():
    # A baseline may report weights_retrained=False; that is not a scoring gate.
    return dict(detector_ready=True, ocr_ready=True, vehicle_detector_ready=True,
                model_sha256='a'*64, detector_model='fixture', weights_retrained=False,
                engine='fake', ocr_engine='fake', vehicle_engine='fake', vehicle_taxonomy='uvh')


def response(plates):
    return dict(detections=plates, plate_count=len(plates), read_text=True, angle_correction=True,
                detector_model='fixture', vehicle_detector_ready=True, vehicles=[dict(id='vehicle')],
                processing_time_ms=10., ocr_time_ms=3.)


def sample(targets):
    return dict(id='fixture', path=Path('whole.jpg'), sha256='0'*64, data=b'whole-image', targets=targets)


def test_exact_denominators_include_detection_misses_and_exclude_unflagged_targets():
    targets = [dict(id='scored_hit', box=[0, 0, 10, 10], text='KA01AB1234', score_exact=True),
               dict(id='scored_miss', box=[20, 0, 10, 10], text='KA01AB2345', score_exact=True),
               dict(id='excluded', box=[40, 0, 10, 10], text='KA01AB3456', score_exact=False)]
    result = response([detection([0, 0, 10, 10], 'KA01AB1234'), detection([40, 0, 10, 10], 'KA01AB3456')])
    row = audit.capture(sample(targets), result, 15., health())
    summary = audit.summarize([row])
    assert summary['target_detection'] == dict(numerator=2, denominator=3, rate=2/3)
    assert summary['raw_ocr_exact_end_to_end'] == dict(numerator=1, denominator=2, rate=.5)
    assert summary['raw_ocr_exact_given_target_detected'] == dict(numerator=1, denominator=1, rate=1.)
    assert summary['application_normalized_exact_end_to_end'] == dict(numerator=1, denominator=2, rate=.5)
    assert summary['application_normalized_exact_given_target_detected'] == dict(numerator=1, denominator=1, rate=1.)
    assert summary['targets_excluded_from_exact_text'] == 1
    assert summary['exact_targets_missed_by_detector'] == 1
    assert row['targets'][2]['exact'] is None and row['targets'][2]['expected_text'] is None
    assert row['targets'][2]['raw_exact'] is None and row['targets'][2]['application_normalized_exact'] is None
    assert summary['returned_vehicle_observations'] == 1


def test_unknown_character_markers_and_o_zero_are_never_repaired():
    assert audit.normalized_text('ka 01-ab.1234') == 'KA01AB1234'
    assert audit.normalized_text('KAO1AB1234') != 'KA01AB1234'
    assert audit.normalized_text('?KA01AB1234') != 'KA01AB1234'
    assert audit.raw_text_for_exact('ka 01 ab1234') == 'KA01AB1234'
    assert audit.raw_text_for_exact('KA01-AB1234') != 'KA01AB1234'


def test_review_state_does_not_hide_geometric_or_exact_matches():
    plate = detection([0, 0, 10, 10], 'KA01AB1234')
    plate.update(localization_requires_review=True,
                 ocr_review=dict(requires_review=True, selected_method='grayscale_clahe',
                                 conflicting_readings=True, attempts=[{}, {}]))
    target = dict(id='target', box=[0, 0, 10, 10], text='KA01AB1234', score_exact=True)
    row = audit.capture(sample([target]), response([plate]), 15., health())
    assert row['targets'][0]['exact'] is True
    assert row['targets'][0]['ocr_review']['selected_method'] == 'grayscale_clahe'
    assert row['counters']['review']['localization_requires_review'] == 1
    assert row['counters']['review']['ocr_requires_review'] == 1


def manual_fixture(tmp_path):
    entries = []
    for i in range(20):
        image = tmp_path / f'whole{i}.jpg'
        image.write_bytes(f'whole original bytes {i}'.encode())
        entries.append(dict(audit_id=str(i), image=str(image), image_sha256=sha256(image.read_bytes()).hexdigest(),
                            box=[0, 0, 10, 10], manual_text=f'KA01AB{i:04d}' if i < 12 else None,
                            use_for_exact_ocr_score=i < 12))
    path = tmp_path / 'truth.json'
    path.write_text(json.dumps(dict(entries=entries)), encoding='utf-8')
    return path


def test_frozen_manifest_count_flags_and_image_hashes_are_enforced(tmp_path):
    path = manual_fixture(tmp_path)
    samples, provenance = audit.load_manual(path)
    assert len(samples) == 20 and provenance['exact_targets'] == 12
    assert samples[0]['data'] == b'whole original bytes 0'
    samples[0]['path'].write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='changed after manual review'):
        audit.load_manual(path)
    path = manual_fixture(tmp_path)
    data = json.loads(path.read_text())
    data['entries'][12]['use_for_exact_ocr_score'] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='20 whole images and 12'):
        audit.load_manual(path)


def test_endpoint_validation_and_latency_distribution():
    assert audit.endpoint('candidate=http://127.0.0.1:8004/') == ('candidate', 'http://127.0.0.1:8004')
    for endpoint in ('http://127.0.0.1', 'x=http://example.com:8004', 'x=http://127.0.0.1:8004/api', 'x=http://user@localhost:8004'):
        with pytest.raises(audit.argparse.ArgumentTypeError):
            audit.endpoint(endpoint)
    assert audit.distribution([1., 2., 3., 4., 5.]) == dict(samples=5, p50=3., p95=4.8, min=1., max=5.)
    assert audit.distribution([None])['p95'] is None


def test_fake_http_run_scores_all_images_and_excludes_synthetic_warmup(tmp_path, monkeypatch):
    path = manual_fixture(tmp_path)
    args = audit.parse_args(['--endpoint', 'green=http://127.0.0.1:8003', '--endpoint', 'new=http://127.0.0.1:8004',
                             '--truth', str(path), '--output', str(tmp_path/'report.json'), '--synthetic'])
    synthetic = sample([dict(id=str(i), box=[i*20, 0, 10, 10], text=f'KA01AB{i:04d}', score_exact=True) for i in range(10)])
    synthetic.update(id='synthetic-ten', truth_path='fixture.json', truth_sha256='b'*64)
    monkeypatch.setattr(audit, 'load_synthetic', lambda: synthetic)
    monkeypatch.setattr(audit, 'request_json', lambda *_, **__: health())
    calls = []
    def fake_photo(address, sample, timeout):
        calls.append((address, sample['id']))
        return response([detection(target['box'], target['text']) for target in sample['targets']]), 15.
    monkeypatch.setattr(audit, 'photo_request', fake_photo)
    report = audit.run(args)
    assert report['status'] == 'complete' and len(calls) == 52
    for model in report['models'].values():
        assert model['manual_summary']['target_detection']['denominator'] == 20
        assert model['manual_summary']['raw_ocr_exact_end_to_end'] == dict(numerator=12, denominator=12, rate=1.)
        assert model['manual_summary']['application_normalized_exact_end_to_end'] == dict(numerator=12, denominator=12, rate=1.)
        assert model['synthetic_summary']['target_detection']['denominator'] == 50
        assert model['synthetic_summary']['raw_ocr_exact_end_to_end']['denominator'] == 50
        assert model['synthetic_summary']['application_normalized_exact_end_to_end']['denominator'] == 50
        assert len(model['synthetic_warmup']) == 1 and len(model['synthetic']) == 5
        assert model['synthetic_summary']['latencies_ms']['http_ms']['samples'] == 5
    assert calls[:4] == [('http://127.0.0.1:8003', '0'), ('http://127.0.0.1:8004', '0'),
                         ('http://127.0.0.1:8004', '1'), ('http://127.0.0.1:8003', '1')]
    with pytest.raises(FileExistsError):
        audit.run(args)
    assert len(calls) == 52


def test_failed_request_preserves_partial_report_without_retry(tmp_path, monkeypatch):
    path = manual_fixture(tmp_path)
    args = audit.parse_args(['--endpoint', 'new=http://127.0.0.1:8004', '--truth', str(path), '--output', str(tmp_path/'partial.json')])
    monkeypatch.setattr(audit, 'request_json', lambda *_, **__: health())
    calls = []
    def fail_second(address, sample, timeout):
        calls.append(sample['id'])
        if len(calls) == 2:
            raise TimeoutError('fixture timed out')
        return response([detection(target['box'], target['text']) for target in sample['targets']]), 15.
    monkeypatch.setattr(audit, 'photo_request', fail_second)
    with pytest.raises(TimeoutError):
        audit.run(args)
    saved = json.loads(args.output.read_text())
    assert saved['status'] == 'failed'
    assert saved['models']['new']['manual_summary']['requests'] == 1
    assert saved['error']['context']['sample'] == '1'
    assert calls == ['0', '1']


def test_unready_engine_or_changed_response_identity_is_rejected():
    bad = health()
    bad['ocr_ready'] = False
    with pytest.raises(ValueError, match='all be ready'):
        audit.validate_health(bad)
    audit.validate_health(health())
    result = response([])
    result['detector_model'] = 'changed'
    with pytest.raises(ValueError, match='identity differs'):
        audit.capture(sample([]), result, 1., health())


def test_http_upload_contains_original_whole_image_and_flags_but_no_truth(monkeypatch):
    captured = []
    def fake_request(url, timeout, data, headers):
        captured.append((url, timeout, data, headers))
        return response([])
    monkeypatch.setattr(audit, 'request_json', fake_request)
    source = sample([dict(id='target', box=[1, 2, 3, 4], text='SECRETTRUTH123', score_exact=True)])
    audit.photo_request('http://127.0.0.1:8004', source, 120.)
    url, timeout, body, headers = captured[0]
    assert url == 'http://127.0.0.1:8004/api/detect/image'
    assert b'whole-image' in body and b'SECRETTRUTH123' not in body
    assert b'name="read_text"\r\n\r\ntrue' in body
    assert b'name="angle_correction"\r\n\r\ntrue' in body
    assert b'name="reidentify"\r\n\r\nfalse' in body
    assert b'name="ocr_model"\r\n\r\nppocr' in body
    assert 'multipart/form-data' in headers['Content-Type']


@pytest.mark.parametrize('raw,corrected', [('MH12AB1O34', 'MH12AB1034'), ('MHO1AB1234', 'MH01AB1234')])
def test_application_character_repair_does_not_count_as_raw_ocr_exact(raw, corrected):
    target = dict(id='target', box=[0, 0, 10, 10], text=corrected, score_exact=True)
    row = audit.capture(sample([target]), response([detection(target['box'], corrected, raw_text=raw)]), 10., health())
    scored = row['targets'][0]
    assert scored['raw_text'] == raw and scored['raw_observed_text'] == raw
    assert scored['raw_exact'] is False
    assert scored['application_normalized_exact'] is True
    assert scored['exact'] is True  # Legacy alias is explicitly application-normalized.
    summary = audit.summarize([row])
    assert summary['primary_ocr_metric'] == 'raw_ocr_exact_end_to_end'
    assert summary['raw_ocr_exact_end_to_end'] == dict(numerator=0, denominator=1, rate=0.)
    assert summary['raw_ocr_exact_given_target_detected']['numerator'] == 0
    assert summary['application_normalized_exact_end_to_end'] == dict(numerator=1, denominator=1, rate=1.)
    assert summary['application_normalized_exact_given_target_detected']['numerator'] == 1


def test_raw_ocr_must_be_present_and_never_falls_back_to_normalized_text():
    target = dict(id='target', box=[0, 0, 10, 10], text='MH12AB1034', score_exact=True)
    plate = detection(target['box'], target['text'])
    del plate['raw_text']
    with pytest.raises(ValueError, match='normalized text cannot substitute for raw OCR'):
        audit.capture(sample([target]), response([plate]), 10., health())
