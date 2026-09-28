"""Bounded crop retries must retain actual OCR evidence and ambiguous text."""
import json

import cv2
import numpy as np
import pytest

import app.services.ocr as module
from app.services.ocr import PlateOCR, preprocessing_candidates


def crop():
    image = np.full((40, 180, 3), 190, np.uint8)
    cv2.putText(image, 'MH12AB1234', (3, 28), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (90, 90, 90), 2, cv2.LINE_AA)
    return image


def reader(readings):
    value = object.__new__(PlateOCR)
    calls = []
    sequence = iter(readings)
    def recognize(image, **kwargs):
        calls.append((image.shape, kwargs))
        result = next(sequence)
        if isinstance(result, Exception):
            raise result
        return result
    value._recognize_crop = recognize
    return value, calls


def variants(monkeypatch):
    image = crop()
    monkeypatch.setattr(module, 'preprocessing_candidates', lambda original: [
        ('grayscale_clahe', image), ('adaptive_threshold', image), ('morphology_open', image)])


def test_confident_original_skips_filters_rectification_and_context(monkeypatch):
    def forbidden(*args):
        raise AssertionError('A confident original must not trigger transforms')
    monkeypatch.setattr(module, 'preprocessing_candidates', forbidden)
    import app.services.rectification as correction
    monkeypatch.setattr(correction, 'rectify_plate', forbidden)
    value, calls = reader([('MH12AB1234', .96)])
    detail = value.read_plate_details(crop(), crop()[2:-2, 2:-2], angle_correction=True)
    assert len(calls) == 1
    assert detail['retry_count'] == 0 and not detail['requires_review']
    assert detail['text'] == detail['original_text'] == 'MH12AB1234'
    assert detail['confidence'] == detail['original_confidence'] == .96
    assert 'rectified_crop' not in detail
    assert '_rectification_attempt' not in detail


def test_filtered_agreement_selects_an_actual_score_without_boost(monkeypatch):
    variants(monkeypatch)
    value, calls = reader([('MH12AB1234', .95), ('MH12AB1234', .93)])
    image = crop()
    detail = value.read_plate_details(image, image, initial_reading=('BLUR', .2))
    assert len(calls) == 2  # The seeded original is not recognized again.
    assert detail['text'] == 'MH12AB1234' and detail['confidence'] == .95
    assert detail['original_text'] == 'BLUR' and detail['original_confidence'] == .2
    assert detail['agreeing_attempts'] == 2 and not detail['requires_review']
    assert detail['selected_method'] == 'grayscale_clahe'
    assert [attempt['method'] for attempt in detail['attempts']] == [
        'tight_original', 'grayscale_clahe', 'adaptive_threshold']
    assert detail['preprocessing_time_ms'] >= 0
    json.dumps(detail)


def test_one_filtered_guess_remains_uncertain_even_at_high_model_confidence(monkeypatch):
    variants(monkeypatch)
    value, calls = reader([('MH12AB1234', .99), ('', 0.), ('', 0.)])
    image = crop()
    detail = value.read_plate_details(image, image, initial_reading=('BLUR', .2))
    assert detail['text'] == 'MH12AB1234' and detail['confidence'] == .99
    assert detail['requires_review'] and detail['agreeing_attempts'] == 1
    assert len(calls) == 3 and detail['retry_count'] == 3


def test_conflicting_actual_registrations_remain_uncertain_despite_filter_votes(monkeypatch):
    variants(monkeypatch)
    value, calls = reader([('MH12AB1235', .98), ('MH12AB1235', .96), ('MH12AB1235', .95)])
    image = crop()
    detail = value.read_plate_details(image, image, initial_reading=('MH12AB1234', .81))
    assert detail['text'] == 'MH12AB1235' and detail['confidence'] == .98
    assert detail['requires_review']
    assert detail['conflicting_readings'] == ['MH12AB1234', 'MH12AB1235']
    assert detail['original_text'] == 'MH12AB1234'
    assert len(calls) == 3


def test_normalization_does_not_hide_o_zero_disagreement(monkeypatch):
    variants(monkeypatch)
    value, _ = reader([('MH12AB1204', .98)] * 3)
    image = crop()
    detail = value.read_plate_details(image, image, initial_reading=('MH12AB12O4', .81))
    assert detail['requires_review']
    assert detail['conflicting_readings'] == ['MH12AB12O4', 'MH12AB1204']


def test_failed_optional_retries_preserve_original_score_and_text(monkeypatch):
    variants(monkeypatch)
    value, calls = reader([RuntimeError('fixture failure')] * 3)
    image = crop()
    detail = value.read_plate_details(image, image, initial_reading=('MH12AB1234', .81))
    assert detail['text'] == 'MH12AB1234' and detail['confidence'] == .81
    assert detail['requires_review'] and len(calls) == 3
    assert all(attempt.get('error') == 'unavailable' for attempt in detail['attempts'][1:])


def test_rectification_requires_opt_in_and_filters_use_credible_corrected_crop(monkeypatch):
    import app.services.rectification as correction
    original = crop()
    corrected = np.full((32, 140, 3), 123, np.uint8)
    rectified_calls = []
    def rectify(image):
        rectified_calls.append(image)
        return corrected, dict(applied=True, method='quadrilateral_perspective', angle_degrees=7.)
    monkeypatch.setattr(correction, 'rectify_plate', rectify)
    seen = []
    def filters(image):
        seen.append(image)
        return [('grayscale_clahe', image)]
    monkeypatch.setattr(module, 'preprocessing_candidates', filters)
    off, _ = reader([('BLUR', .2)])
    off_detail = off.read_plate_details(original, original, initial_reading=('BLUR', .2))
    assert rectified_calls == [] and 'rectified_crop' not in off_detail
    value, _ = reader([('MH12AB1234', .96), ('MH12AB1234', .95)])
    detail = value.read_plate_details(original, original, initial_reading=('BLUR', .2), angle_correction=True)
    assert detail['rectified_crop'] is corrected and seen[-1] is corrected
    assert detail['rectification']['applied'] and not detail['requires_review']
    assert detail['_rectification_attempt'] == dict(source='context', transform='rectify_plate')
    assert detail['attempts'][-1]['source'] == 'rectified'
    json.dumps({key: val for key, val in detail.items() if key != 'rectified_crop'})


def test_batch_confident_crops_stay_batched_and_return_matching_tuple_api():
    class Batch:
        def __init__(self): self.sizes = []
        def recognize(self, crops):
            self.sizes.append(len(crops))
            return [('MH12AB1234', .96)] * len(crops)
    value = object.__new__(PlateOCR)
    value.batch = Batch()
    image = crop()
    detail = value.read_many_details([(image, image)] * 12)
    assert value.batch.sizes == [12]
    assert len(detail) == 12 and all(not item['requires_review'] for item in detail)
    assert value.read_many([(image, image)]) == [('MH12AB1234', .96)]


@pytest.mark.parametrize('seeded', [False, True])
def test_retry_timing_includes_context_read_and_excludes_original(monkeypatch, seeded):
    now = [0.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    context = crop()
    tight = context[2:-2, 2:-2]
    value = object.__new__(PlateOCR)
    value.engine = object()  # Enable the localized retry without a real session.

    def recognize(image, *, localized=False):
        if localized:
            now[0] += .007
            return 'MH12AB1234', .95
        now[0] += .070 if image is tight else .011
        return 'BLUR', .2

    value._recognize_crop = recognize
    detail = value.read_plate_details(context, tight,
                                     initial_reading=('BLUR', .2) if seeded else None)
    assert detail['retry_time_ms'] == 18.
    assert detail['retry_count'] == 2
    assert detail['selected_method'] == 'context_localized'


@pytest.mark.parametrize('batch_fails', [False, True])
def test_retry_timing_shares_context_batch_cost_without_counting_other_plates(monkeypatch, batch_fails):
    now = [0.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])

    class Batch:
        def __init__(self): self.calls = 0
        def recognize(self, images):
            self.calls += 1
            if self.calls == 1:
                now[0] += .100
                return [('BLUR', .2), ('BLUR', .2), ('MH12AB1234', .96)]
            assert len(images) == 2
            now[0] += .020
            if batch_fails:
                raise RuntimeError('optional context batch unavailable')
            return [('BLUR', .2)] * 2

    value = object.__new__(PlateOCR)
    value.batch = Batch()
    value.engine = object()
    durations = iter([.004, .006])
    def recognize(image, *, localized=False):
        assert localized
        now[0] += next(durations)
        return 'MH12AB1234', .95
    value._recognize_crop = recognize
    image = crop()
    details = value.read_many_details([(image, image[2:-2, 2:-2])] * 3)
    assert [detail['retry_time_ms'] for detail in details] == [14., 16., 0.]
    assert [detail['retry_count'] for detail in details] == [2, 2, 0]
    assert sum(detail['retry_time_ms'] for detail in details) == 30.
    assert all(not detail['requires_review'] for detail in details)


def test_preprocessing_is_bounded_nonmutating_and_only_returns_image_transforms():
    image = cv2.resize(crop(), (2400, 540))
    snapshot = image.copy()
    outputs = preprocessing_candidates(image)
    assert [name for name, _ in outputs][:2] == ['grayscale_clahe', 'adaptive_threshold']
    assert 2 <= len(outputs) <= 3
    assert np.array_equal(image, snapshot)
    assert all(max(result.shape[:2]) <= 640 and result.dtype == np.uint8
               and result.shape[2] == 3 for _, result in outputs)
    assert set(np.unique(outputs[1][1])).issubset({0, 255})


def test_morphology_is_reserved_for_speckled_crops_and_preserves_character_ink():
    image = np.full((96, 400, 3), 230, np.uint8)
    cv2.putText(image, 'MH12AB1234', (8, 68), cv2.FONT_HERSHEY_SIMPLEX,
                1.65, (20, 20, 20), 4, cv2.LINE_AA)
    assert 'morphology_open' not in dict(preprocessing_candidates(image))
    rng = np.random.default_rng(42)
    points = rng.integers([0, 0], [96, 400], size=(400, 2))
    image[points[:, 0], points[:, 1]] = 0
    filtered = dict(preprocessing_candidates(image))
    assert 'morphology_open' in filtered
    before = np.count_nonzero(filtered['adaptive_threshold'][:, :, 0] == 0)
    after = np.count_nonzero(filtered['morphology_open'][:, :, 0] == 0)
    assert .8 * before <= after < before


@pytest.mark.parametrize('image', [None, np.zeros((0, 4, 3), np.uint8),
                                  np.zeros((10, 30, 3), np.float32), np.full((20, 80, 3), 120, np.uint8)])
def test_blank_or_invalid_crops_do_not_generate_filter_evidence(image):
    assert preprocessing_candidates(image) == []
