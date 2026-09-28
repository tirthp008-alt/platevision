"""Recovery must affect real photo results, without depending on text."""
import cv2
import numpy as np
import pytest

from app.services.pipeline import CropStore, process


class PlateDetector:
    def detect_regions(self, image, profile):
        return [(20, 20, 40, 12, .8)]

    def detect_vehicle_regions(self, crop):
        # Second vehicle's padded crop starts at x=148; this is learned
        # localization inside that crop, not an assumed bumper position.
        return [(32, 60, 60, 18, .35)]


class Vehicles:
    def objects(self, image):
        return [(0, 0, 100, 120, .9, 2), (160, 0, 140, 120, .9, 2)]


class UnreadableOCR:
    def read_plate(self, context, tight):
        return '', 0.


@pytest.mark.parametrize('read_text', [False, True])
def test_recovered_rectangle_is_saved_and_associated_even_without_readable_text(read_text):
    _, encoded = cv2.imencode('.png', np.zeros((120, 300, 3), np.uint8))
    store = CropStore()
    result = process(encoded.tobytes(), PlateDetector(), store, UnreadableOCR(), Vehicles(),
                     profile='fast', read_text=read_text, angle_correction=True)
    assert result['plate_count'] == 2
    assert result['vehicle_summary']['total_vehicles'] == 2
    assert result['vehicle_plate_search']['searched_vehicles'] == 1
    assert result['vehicle_plate_search']['added_regions'] == 1
    recovered = result['detections'][1]
    assert recovered['tight_plate_box'] == dict(x=180, y=60, width=60, height=18)
    assert recovered['vehicle_id'] == 'vehicle-2'
    assert recovered['localization_source'] == 'vehicle_crop'
    assert recovered['localization_requires_review'] is True
    assert result['detections'][0]['localization_requires_review'] is False
    assert result['vehicles'][1]['plate_ids'] == [recovered['id']]
    assert recovered['raw_text'] == recovered['normalized_text'] == ''
    assert store.get((result['request_id'], recovered['id']))
    assert result['ocr_time_ms'] == 0 if not read_text else recovered['format_status'] == 'uncertain'


def test_failed_secondary_pass_preserves_whole_scene_plate_and_vehicle_count():
    class FailedRecovery(PlateDetector):
        def detect_vehicle_regions(self, crop):
            raise RuntimeError('secondary model request failed')

    _, encoded = cv2.imencode('.png', np.zeros((120, 300, 3), np.uint8))
    result = process(encoded.tobytes(), FailedRecovery(), CropStore(), None, Vehicles(), read_text=False)
    assert result['plate_count'] == 1
    assert result['vehicle_summary']['total_vehicles'] == 2
    assert result['vehicle_plate_search']['failed_crops'] == 1
    assert result['vehicles'][1]['plate_ids'] == []


def test_crop_only_candidate_is_not_promoted_to_trusted_identity_by_plausible_ocr():
    from app.services.plate_summary import summarize_recognized_plates
    from app.services.reidentification import strongest_plate

    plate = dict(id='crop-only', raw_text='GJ01AB1234', normalized_text='GJ01AB1234',
                 ocr_confidence=.99, format_status='valid', localization_requires_review=True)
    summary = summarize_recognized_plates([plate], scope='frame')
    assert summary['unique_registrations'] == 0
    assert summary['unresolved_track_ids'] == ['crop-only']
    assert strongest_plate([plate]) == (None, 0.)
