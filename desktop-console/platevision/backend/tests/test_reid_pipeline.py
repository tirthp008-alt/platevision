"""Photo matching must leave independent plate capture and counts intact."""
import json

import cv2
import numpy as np

from app.services.camera_matches import CameraMatchGallery
from app.services.pipeline import CropStore, process


class Plates:
    def detect_regions(self, image, profile):
        # One unreadable plate belongs to a car; the other has no vehicle box.
        return [(30, 70, 50, 18, .3), (210, 75, 50, 18, .4)]

    def detect_fast(self, image):
        return [(30, 70, 50, 18, .9), (210, 75, 50, 18, .9)]


class Vehicles:
    taxonomy = 'uvh'

    def objects(self, image):
        return [(10, 10, 110, 90, .9, 'car'), (130, 10, 50, 90, .8, 'two_wheeler')]


class Embeddings:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def embed(self, crops):
        vectors = []
        for crop in crops:
            self.calls.append(crop.shape)
            if len(self.calls) == self.fail_at:
                raise RuntimeError('optional runtime failure')
            vector = np.zeros(2048, np.float32)
            vector[0 if crop.shape[1] > 80 else 1] = 1
            vectors.append(vector)
        return np.stack(vectors)


def photo():
    ok, encoded = cv2.imencode('.png', np.zeros((120, 300, 3), np.uint8))
    assert ok
    return encoded.tobytes()


def capture(camera, timestamp, engine, gallery, **kwargs):
    return process(photo(), Plates(), CropStore(), vehicle_detector=Vehicles(),
                   profile='fast', read_text=False, angle_correction=True,
                   reid_context=dict(engine=engine, gallery=gallery, camera_id=camera,
                                     observed_at=timestamp), **kwargs)


def test_unreadable_plates_match_three_minutes_later_without_losing_unowned_plate():
    gallery = CameraMatchGallery()
    gallery.set_routes([dict(source_camera='A', target_camera='B', min_seconds=60, max_seconds=300)])
    engine = Embeddings()
    first = capture('A', 1000, engine, gallery)
    second = capture('B', 1180, engine, gallery)
    for result in (first, second):
        assert result['plate_count'] == 2
        assert result['vehicle_summary']['total_vehicles'] == 2
        assert result['vehicle_summary']['counts_by_category']['two_wheeler'] == 1
        assert result['detections'][1]['vehicle_id'] is None
        assert all(item['crop_url'] and item['raw_text'] == '' for item in result['detections'])
        assert result['ocr_time_ms'] == 0
        assert result['reidentification']['status'] == 'complete'
        assert 'descriptor' not in json.dumps(result)
    matches = second['reidentification']['observations']
    assert len(matches) == 2 and len(engine.calls) == 4
    assert all(item['status'] == 'candidate' for item in matches)
    assert all(item['selected_candidate']['travel_seconds'] == 180 for item in matches)
    assert all(item['selected_candidate']['plate_evidence']['kind'] == 'none' for item in matches)


def test_photo_reid_failure_preserves_regions_and_never_partially_registers_batch():
    gallery = CameraMatchGallery()
    result = capture('A', 1000, Embeddings(fail_at=2), gallery)
    assert result['plate_count'] == 2
    assert result['vehicle_summary']['total_vehicles'] == 2
    assert result['reidentification']['status'] == 'failed'
    assert gallery.stats()['observations'] == 0
    assert any('matching failed' in warning for warning in result['warnings'])


def test_missing_reid_runtime_is_optional_and_does_not_invent_observations():
    gallery = CameraMatchGallery()
    result = capture('A', 1000, None, gallery)
    assert result['plate_count'] == 2
    assert result['reidentification']['status'] == 'unavailable'
    assert gallery.stats()['observations'] == 0


def test_photo_with_no_vehicle_boxes_retains_every_plate_without_embedding():
    gallery, engine = CameraMatchGallery(), Embeddings()
    result = process(photo(), Plates(), CropStore(), profile='fast', read_text=False,
                     reid_context=dict(engine=engine, gallery=gallery, camera_id='A', observed_at=1000))
    assert result['plate_count'] == 2
    assert engine.calls == [] and gallery.stats()['observations'] == 0
    assert result['reidentification']['observation_count'] == 0


def test_photo_ocr_evidence_can_strengthen_appearance_match():
    class OCR:
        def read_many(self, crops):
            return [('MH12AB1234', .95) for _ in crops]

    gallery, engine = CameraMatchGallery(), Embeddings()
    gallery.set_routes([dict(source_camera='A', target_camera='B', min_seconds=60, max_seconds=300)])
    for camera, timestamp in [('A', 1000), ('B', 1180)]:
        result = process(photo(), Plates(), CropStore(), OCR(), Vehicles(), profile='fast', read_text=True,
                         reid_context=dict(engine=engine, gallery=gallery, camera_id=camera, observed_at=timestamp))
    observations = result['reidentification']['observations']
    assert observations[0]['selected_candidate']['plate_evidence']['kind'] == 'agreement'
    assert observations[0]['selected_candidate']['plate_evidence']['boost'] > 0
    assert observations[1]['selected_candidate']['plate_evidence']['kind'] == 'none'


def test_photo_reports_filtered_conflicting_ocr_as_uncertain_and_preserves_plate_crops():
    class DetailsOCR:
        def read_many_details(self, pairs, *, angle_correction=False):
            assert angle_correction is True
            return [dict(text='MH12AB1234',confidence=.98,requires_review=True,
                         conflicting_readings=['MH12AB1234','MH12AB1284'],selected_method='morphology',
                         attempts=[dict(method='original',text='MH12AB1284',confidence=.92),
                                   dict(method='morphology',text='MH12AB1234',confidence=.98)],
                         preprocessing_time_ms=1.2,rectified_crop=np.zeros((20,50,3),np.uint8),
                         rectification=dict(applied=True,method='quadrilateral_perspective')) for _ in pairs]

    result = process(photo(), Plates(), CropStore(), DetailsOCR(), Vehicles(), profile='fast',
                     read_text=True, angle_correction=True)
    assert result['plate_count']==2
    for detection in result['detections']:
        assert detection['format_status']=='uncertain'
        assert detection['ocr_confidence']==.98
        assert detection['ocr_review']['selected_method']=='morphology'
        assert detection['ocr_review']['requires_review'] is True
        assert detection['crop_url'] and detection['rectified_crop_url']
        assert 'rectified_crop' not in detection['ocr_review']
    json.dumps(result)  # No image arrays leak into the API response.
