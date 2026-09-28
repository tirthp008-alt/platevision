"""Keep overlapping categories, reconciling only near-identical vehicle boxes."""
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.detector import OnnxPlateDetector, VehicleDetector
from app.services.vehicles import attach_vehicles


@pytest.mark.parametrize('classes,allowed', [(80, {2, 5}), (12, set(range(12)))])
def test_filtered_vehicle_vocabulary_keeps_overlapping_categories_and_removes_duplicates(classes, allowed):
    detector = object.__new__(OnnxPlateDetector)
    detector.classes = classes
    detector.allowed = allowed
    detector.output_format = 'raw'
    output = np.zeros((1, 4+classes, 3), np.float32)
    # Two categories share a large overlap; a duplicate of category 2 is
    # removed, while category 5 remains visible for downstream tracking.
    output[0, :4, :] = np.array([[100, 102, 104], [100, 101, 102],
                                [120, 120, 120], [100, 100, 100]], np.float32)
    output[0, 4+2, 0] = .95
    output[0, 4+2, 1] = .70
    output[0, 4+5, 2] = .85
    result = detector.postprocess(output, 1, 0, 0, 300, 200, threshold=.25)
    assert [box[5] for box in result] == [2, 5]
    assert [box[4] for box in result] == pytest.approx([.95, .85])
    detector.runtime = 'test'
    detector.objects = lambda image, threshold: result
    # The separate vehicle stage must also preserve this dense-scene overlap;
    # merely switching back to class-agnostic .45 NMS would lose a vehicle.
    vehicles = VehicleDetector(detector, 'coco').objects(np.zeros((200, 300, 3), np.uint8))
    assert [box[5] for box in vehicles] == [2, 5]
    _, summary = attach_vehicles([], vehicles, 300, 200)
    assert summary['total_vehicles'] == 2


@pytest.mark.parametrize('taxonomy,names,expected', [('coco', None, 5),
                                                    ('uvh', {5: 'bus', 7: 'truck'}, 'bus')])
def test_vehicle_duplicates_keep_strongest_category_before_name_mapping(taxonomy, names, expected):
    # An engine may return unsorted observations; only the strongest category
    # for identical geometry should reach either the frame counter or tracker.
    engine = SimpleNamespace(runtime='test', objects=lambda image, threshold: [
        (20, 30, 60, 40, .8, 7), (20, 30, 60, 40, .9, 5)])
    result = VehicleDetector(engine, taxonomy, names).objects(np.zeros((100, 200, 3), np.uint8))
    assert len(result) == 1
    assert result[0][4] == pytest.approx(.9)
    assert result[0][5] == expected


def test_vehicle_duplicates_do_not_remove_a_smaller_occluding_vehicle():
    engine = SimpleNamespace(runtime='test', objects=lambda image, threshold: [
        (0, 0, 180, 100, .9, 5), (50, 20, 90, 70, .8, 2)])
    result = VehicleDetector(engine, 'coco').objects(np.zeros((100, 200, 3), np.uint8))
    assert [box[5] for box in result] == [5, 2]
