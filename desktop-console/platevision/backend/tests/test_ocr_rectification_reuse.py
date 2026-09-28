"""Photo preview may reuse correction only when OCR tried the same input."""
import json

import cv2
import numpy as np
import pytest

from app.core.config import settings
from app.services.ocr import PlateOCR
from app.services.pipeline import CropStore, process


class Detector:
    def detect(self, image):
        return [(60, 30, 80, 20, .9)]


def photo():
    image = np.arange(120 * 240 * 3, dtype=np.uint8).reshape(120, 240, 3)
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    return encoded.tobytes()


def stub_correction(monkeypatch):
    import app.services.ocr as ocr
    import app.services.pipeline as pipeline
    import app.services.rectification as rectification
    calls = []

    def rectify(image):
        calls.append(image.copy())
        return image, dict(applied=False, reason='fixture_no_quad', attempt=len(calls))

    monkeypatch.setattr(rectification, 'rectify_plate', rectify)
    monkeypatch.setattr(pipeline, 'rectify_plate', rectify)
    monkeypatch.setattr(ocr, 'preprocessing_candidates', lambda image: [])
    monkeypatch.setattr(settings, 'plate_crop_padding', .25)
    return calls


def reader(reading=('BLUR', .2)):
    value = object.__new__(PlateOCR)
    value._recognize_crop = lambda image, **kwargs: reading
    return value


def test_failed_same_crop_correction_runs_once_and_private_metadata_stays_internal(monkeypatch):
    calls = stub_correction(monkeypatch)
    result = process(photo(), Detector(), CropStore(), reader(), angle_correction=True)
    plate = result['detections'][0]
    assert len(calls) == 1
    assert plate['rectification'] == plate['ocr_review']['rectification']
    assert plate['rectification']['applied'] is False
    assert plate['crop_url'] and plate['rectified_crop_url'] is None
    assert plate['raw_text'] == 'BLUR' and plate['format_status'] == 'uncertain'
    assert not any(key.startswith('_') for key in plate['ocr_review'])
    json.dumps(result)


def test_confident_ocr_skips_correction_but_angle_preview_still_runs(monkeypatch):
    calls = stub_correction(monkeypatch)
    result = process(photo(), Detector(), CropStore(), reader(('MH12AB1234', .96)),
                     angle_correction=True)
    plate = result['detections'][0]
    assert len(calls) == 1
    assert 'rectification' not in plate['ocr_review']
    assert plate['ocr_review']['retry_count'] == 0
    assert plate['rectification']['attempt'] == 1


@pytest.mark.parametrize('same_size', [False, True])
def test_different_preview_crop_gets_its_own_correction(monkeypatch, same_size):
    calls = stub_correction(monkeypatch)
    value = reader()
    original = value.read_many_details

    def details(pairs, **kwargs):
        result = original(pairs, **kwargs)
        # Simulate a preview crop configuration differing from the OCR input.
        if same_size:
            import app.services.pipeline as pipeline
            original_clamp = pipeline.clamp
            def shifted_clamp(box, width, height, padding=.08):
                x, y, w, h = original_clamp(box, width, height, padding)
                return (x + 1, y, w, h) if padding == .25 else (x, y, w, h)
            monkeypatch.setattr(pipeline, 'clamp', shifted_clamp)
        else:
            monkeypatch.setattr(settings, 'plate_crop_padding', .1)
        return result

    value.read_many_details = details
    result = process(photo(), Detector(), CropStore(), value, angle_correction=True)
    plate = result['detections'][0]
    assert len(calls) == 2 and not np.array_equal(calls[0], calls[1])
    assert (calls[0].shape == calls[1].shape) is same_size
    assert plate['ocr_review']['rectification']['attempt'] == 1
    assert plate['rectification']['attempt'] == 2


@pytest.mark.parametrize('marker', [None, dict(source='tight', transform='rectify_plate'),
                                    dict(source='context', transform='other_transform')])
def test_unknown_input_or_transform_does_not_suppress_angle_preview(monkeypatch, marker):
    calls = stub_correction(monkeypatch)

    class DetailsReader:
        def read_many_details(self, pairs, **kwargs):
            detail = dict(text='BLUR', confidence=.2, requires_review=True,
                          rectification=dict(applied=False, reason='earlier_attempt'))
            if marker is not None:
                detail['_rectification_attempt'] = marker
            return [detail]

    result = process(photo(), Detector(), CropStore(), DetailsReader(), angle_correction=True)
    assert len(calls) == 1
    assert result['detections'][0]['rectification']['reason'] == 'fixture_no_quad'
