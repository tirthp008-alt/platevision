"""Secondary boxes require plate-model evidence, bounded geometry and budget."""
import numpy as np
import pytest

from app.core.config import settings
from app.services.detector import OnnxPlateDetector
from app.services.vehicle_plate_search import recover_vehicle_plates


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(settings, 'vehicle_plate_search_enabled', True)
    monkeypatch.setattr(settings, 'vehicle_plate_max_crops', 8)
    monkeypatch.setattr(settings, 'region_confidence_threshold', .25)
    monkeypatch.setattr(settings, 'max_detections', 100)


class Detector:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.crops = []

    def detect_vehicle_regions(self, crop):
        self.crops.append(crop)
        result = next(self.responses, [])
        if isinstance(result, Exception):
            raise result
        return result


def test_single_pass_capability_uses_base_plate_model_and_existing_threshold(monkeypatch):
    value = object.__new__(OnnxPlateDetector)
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError('A vehicle crop must not invoke tiling, OCR or high-resolution models')
    def detect(image, threshold=None):
        calls.append((image.shape, threshold))
        return [(10, 20, 50, 15, .8)]
    value.detect = detect
    value.detect_plates = value.detect_regions = value.refine_plates = forbidden
    value.fast_models = {'landscape': object(), 'portrait': object()}
    monkeypatch.setattr(settings, 'region_confidence_threshold', .6)
    result = value.detect_vehicle_regions(np.zeros((180, 250, 3), np.uint8))
    assert result == [(10, 20, 50, 15, .8)]
    assert calls == [((180, 250, 3), .6)]


def test_existing_vehicle_is_skipped_and_secondary_box_maps_from_padded_crop():
    original = (20, 40, 40, 10, .7)
    image = np.zeros((200, 400, 3), np.uint8)
    snapshot = image.copy()
    detector = Detector([[(32, 48, 50, 20, .8)]])
    vehicles = [(10, 10, 100, 80, .9, 'car'), (160, 10, 150, 100, .9, 'bus')]
    boxes, report = recover_vehicle_plates(image, detector, [original], vehicles)
    assert boxes == [original, (180., 50., 50., 20., .8)]
    assert boxes[0] is original and np.array_equal(image, snapshot)
    assert detector.crops[0].shape == (116, 174, 3)
    assert report['eligible_vehicles'] == report['searched_vehicles'] == report['added_regions'] == 1
    assert report['failed_crops'] == report['remaining_vehicles'] == 0
    assert report['enabled'] and report['time_ms'] >= 0


def test_smallest_enclosing_vehicle_owns_an_existing_plate():
    original = (45, 55, 30, 10, .7)
    detector = Detector([[(210, 100, 50, 15, .8)]])
    boxes, report = recover_vehicle_plates(np.zeros((200, 400, 3), np.uint8), detector,
        [original], [(0, 0, 300, 150, .9, 'bus'), (30, 30, 80, 60, .9, 'car')])
    assert report['eligible_vehicles'] == 1
    assert detector.crops[0].shape == (162, 324, 3)
    assert boxes == [original, (210., 100., 50., 15., .8)]


def test_no_plate_model_evidence_never_creates_a_vehicle_rectangle_as_a_plate():
    detector = Detector()
    boxes, report = recover_vehicle_plates(np.zeros((100, 300, 3), np.uint8), detector, [],
        [(0, 0, 80, 90, .99, 'car'), (100, 0, 80, 90, .99, 'bus'), (200, 0, 80, 90, .99, 'truck')])
    assert boxes == [] and report['added_regions'] == 0
    assert report['eligible_vehicles'] == report['searched_vehicles'] == len(detector.crops) == 3


def test_overlapping_vehicle_crops_merge_duplicate_secondary_plates():
    detector = Detector([[(130, 80, 40, 10, .7)],
                         [(45, 80, 40, 10, .9), (125, 80, 40, 10, .6)]])
    boxes, report = recover_vehicle_plates(np.zeros((200, 400, 3), np.uint8), detector, [],
        [(0, 0, 180, 150, .9, 'bus'), (100, 0, 180, 150, .9, 'truck')])
    assert boxes == [(130., 80., 40., 10., .9), (210., 80., 40., 10., .6)]
    assert report['searched_vehicles'] == report['added_regions'] == 2


def test_secondary_duplicate_never_replaces_original_even_at_higher_score():
    original = (130, 80, 40, 10, .3)
    detector = Detector([[(45, 80, 40, 10, .99)]])
    boxes, report = recover_vehicle_plates(np.zeros((200, 400, 3), np.uint8), detector, [original],
        [(0, 0, 180, 150, .9, 'bus'), (100, 0, 180, 150, .9, 'truck')])
    assert boxes == [original] and boxes[0] is original
    assert report['searched_vehicles'] == 1 and report['added_regions'] == 0


@pytest.mark.parametrize('global_x,width,accepted', [(142, 10, True), (45, 20, True),
                                                   (44, 20, False), (44, 6, False)])
def test_bumper_clipping_requires_centre_inside_and_three_quarters_plate_area(global_x, width, accepted):
    # Vehicle x=50..150, padded crop x=42..158. Modest bumper-edge clipping
    # is accepted; a centre in padding or <75% vehicle overlap is rejected.
    detector = Detector([[(global_x - 42, 24, width, 10, .8)]])
    boxes, report = recover_vehicle_plates(np.zeros((140, 220, 3), np.uint8), detector, [],
                                            [(50, 50, 100, 50, .9, 'car')])
    assert len(boxes) == int(accepted) and report['added_regions'] == int(accepted)
    if accepted:
        assert boxes[0] == (float(global_x), 70., float(width), 10., .8)


def test_clipped_vehicle_crop_stays_inside_real_image():
    image = np.zeros((100, 150, 3), np.uint8)
    detector = Detector([[(5, 70, 30, 10, .8)]])
    boxes, report = recover_vehicle_plates(image, detector, [], [(-20, -10, 100, 100, .9, 'car')])
    assert detector.crops[0].shape == (98, 87, 3)
    assert boxes == [(5., 70., 30., 10., .8)]
    assert report['failed_crops'] == 0


def test_rotation_and_per_call_budget_visit_different_eligible_vehicles():
    image = np.zeros((100, 500, 3), np.uint8)
    image[:, :, 0] = np.arange(500, dtype=np.uint16) // 4
    vehicles = [(10 + 90 * index, 10, 60, 70, .9, 'car') for index in range(5)]
    first = Detector()
    _, report = recover_vehicle_plates(image, first, [], vehicles, start_index=7, max_regions=2)
    assert [int(crop[0, 0, 0]) for crop in first.crops] == [46, 68]
    assert report['next_start_index'] == 4 and report['remaining_vehicles'] == 3
    second = Detector()
    recover_vehicle_plates(image, second, [], vehicles, start_index=report['next_start_index'], max_regions=2)
    assert [int(crop[0, 0, 0]) for crop in second.crops] == [91, 1]


def test_default_crop_budget_is_not_expanded_by_override(monkeypatch):
    monkeypatch.setattr(settings, 'vehicle_plate_max_crops', 2)
    detector = Detector()
    _, report = recover_vehicle_plates(np.zeros((100, 400, 3), np.uint8), detector, [],
        [(index * 90, 0, 80, 90, .9, 'car') for index in range(4)], max_regions=1000)
    assert len(detector.crops) == 2 and report['remaining_vehicles'] == 2


def test_duplicate_vehicle_geometry_is_attempted_once():
    detector = Detector()
    _, report = recover_vehicle_plates(np.zeros((100, 150, 3), np.uint8), detector, [],
        [(10, 10, 80, 70, .9, 'car'), (10, 10, 80, 70, .8, 'truck')])
    assert report['eligible_vehicles'] == report['searched_vehicles'] == 1


def test_crop_failure_preserves_all_original_regions_and_continues_other_vehicles():
    original = (350, 160, 20, 10, .7)
    detector = Detector([RuntimeError('fixture inference failure'), [(18, 18, 30, 10, .8)]])
    boxes, report = recover_vehicle_plates(np.zeros((200, 400, 3), np.uint8), detector, [original],
        [(0, 0, 100, 100, .9, 'car'), (150, 0, 100, 100, .9, 'bus')])
    assert boxes == [original, (160., 18., 30., 10., .8)]
    assert report['searched_vehicles'] == 2 and report['failed_crops'] == 1


def test_invalid_geometry_and_scores_do_not_become_plate_evidence():
    bad = [(0, 0, 20, 10, .24), (0, 0, 20, 10, 1.1), (0, 0, 20, 10, float('nan')),
           (float('inf'), 0, 20, 10, .8), (-1, 0, 20, 10, .8), (0, 0, 0, 10, .8),
           (0, 0, 20, 10, '0.8'), (80, 80, 200, 10, .8), (0, 0, 100, 1, .8)]
    detector = Detector([bad + [(20, 20, 30, 10, .8)]])
    boxes, report = recover_vehicle_plates(np.zeros((100, 150, 3), np.uint8), detector, [],
        [(0, 0, 100, 90, .9, 'car'), (0, 0, -10, 80, .9, 'car'),
         (float('nan'), 0, 20, 20, .9, 'car'), (0, 0, 20, 20, 2., 'car')])
    assert boxes == [(20., 20., 30., 10., .8)]
    assert report['eligible_vehicles'] == 1 and report['failed_crops'] == 0


@pytest.mark.parametrize('disabled,detector', [(True, Detector()), (False, object())])
def test_opt_out_or_unsupported_detector_preserves_base(monkeypatch, disabled, detector):
    monkeypatch.setattr(settings, 'vehicle_plate_search_enabled', not disabled)
    original = [(10, 10, 20, 10, .5)]
    boxes, report = recover_vehicle_plates(np.zeros((100, 150, 3), np.uint8), detector, original,
                                            [(50, 10, 80, 70, .9, 'car')])
    assert boxes == original and not report['enabled']
    assert report['searched_vehicles'] == report['added_regions'] == 0


def test_zero_budget_retains_pending_count_without_inference():
    detector = Detector()
    boxes, report = recover_vehicle_plates(np.zeros((100, 150, 3), np.uint8), detector, [],
                                            [(10, 10, 80, 70, .9, 'car')], max_regions=0)
    assert boxes == [] and detector.crops == []
    assert report['eligible_vehicles'] == report['remaining_vehicles'] == 1
    assert report['reason'] == 'crop_budget_zero'
