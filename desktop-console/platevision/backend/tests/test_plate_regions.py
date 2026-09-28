import cv2
import numpy as np

from app.services.detector import MockPlateDetector
from app.services.pipeline import CropStore, process
from app.services.rectification import rectify_plate


def image_bytes(image=None):
    if image is None:
        image = np.zeros((120, 300, 3), np.uint8)
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    return encoded.tobytes()


class NeverOCR:
    def read_many(self, *args):
        raise AssertionError('Region capture must not invoke batch OCR')

    def read_plate(self, *args):
        raise AssertionError('Region capture must not invoke plate OCR')

    def read(self, *args):
        raise AssertionError('Region capture must not invoke OCR')


def test_regions_keep_weak_and_unreadable_plates_without_ocr_or_normalization(monkeypatch):
    import app.services.pipeline as pipeline

    class Detector:
        def detect_regions(self, image, profile):
            assert profile == 'fast'
            return [(10, 10, 50, 20, .16), (100, 10, 50, 20, .3), (200, 10, 50, 20, .9)]

        def detect_fast(self, image):
            raise AssertionError('Use independent region search')

    def never_normalize(text):
        raise AssertionError('No text exists to normalize')

    monkeypatch.setattr(pipeline, 'normalize_plate', never_normalize)
    result = process(image_bytes(), Detector(), CropStore(), NeverOCR(), profile='fast',
                     green_filter=True, read_text=False)
    assert result['plate_count'] == 3
    assert result['ocr_time_ms'] == 0
    assert result['read_text'] is False
    assert [d['detection_confidence'] for d in result['detections']] == [.16, .3, .9]
    for detection in result['detections']:
        assert detection['raw_text'] == detection['normalized_text'] == detection['formatted_text'] == ''
        assert detection['ocr_confidence'] is None
        assert detection['format_status'] == 'not_requested'
        assert 'green_enhancement' not in detection


def test_mock_regions_cannot_invent_a_registration():
    detector = MockPlateDetector()
    result = process(image_bytes(), detector, CropStore(), NeverOCR(), read_text=False)
    assert result['detections']
    assert all(item['raw_text'] == '' and item['ocr_confidence'] is None for item in result['detections'])
    assert 'DEMO MODE' in ' '.join(result['warnings'])


def test_crop_encoding_failure_does_not_discard_plate(monkeypatch):
    import app.services.pipeline as pipeline

    class Detector:
        def detect(self, image):
            return [(10, 10, 50, 20, .2), (100, 10, 50, 20, .9)]

    data = image_bytes()
    monkeypatch.setattr(pipeline.cv2, 'imencode', lambda *args, **kwargs: (False, None))
    result = process(data, Detector(), CropStore(), NeverOCR(), read_text=False)
    assert result['plate_count'] == 2
    assert all(item['crop_url'] is None for item in result['detections'])


def tilted_plate(width=240, height=120):
    image = np.zeros((height, width, 3), np.uint8)
    quad = cv2.boxPoints(((width / 2, height / 2), (width * .72, height * .40), -12)).astype(np.int32)
    cv2.fillConvexPoly(image, quad, (40, 170, 50))
    cv2.line(image, tuple(quad[0]), tuple(quad[1]), (255, 255, 255), 2)
    return image


def test_rectification_preserves_source_and_reports_bounded_local_quad():
    crop = tilted_plate()
    original = crop.copy()
    corrected, metadata = rectify_plate(crop)
    assert metadata['applied'] is True
    assert 5 < abs(metadata['angle_degrees']) < 20
    assert metadata['method'] == 'quadrilateral_perspective'
    assert np.array_equal(crop, original)
    assert not np.shares_memory(crop, corrected)
    assert metadata['output_size'] == [corrected.shape[1], corrected.shape[0]]
    quad = np.array(metadata['quad'])
    assert (quad[:, 0] >= 0).all() and (quad[:, 0] < crop.shape[1]).all()
    assert (quad[:, 1] >= 0).all() and (quad[:, 1] < crop.shape[0]).all()
    assert metadata['time_ms'] >= 0


def test_rectification_leaves_uniform_or_clipped_crop_unchanged():
    for crop in [np.zeros((100, 200, 3), np.uint8), np.full((80, 240, 3), 255, np.uint8)]:
        corrected, metadata = rectify_plate(crop)
        assert metadata['applied'] is False
        assert corrected is crop
        assert metadata['quad'] is None


def test_rectification_bounds_analysis_and_output_for_large_crop():
    crop = tilted_plate(2400, 1200)
    corrected, metadata = rectify_plate(crop)
    assert metadata['applied'] is True
    assert max(corrected.shape[:2]) <= 640
    assert metadata['source_size'] == [2400, 1200]
    assert max(point[0] for point in metadata['quad']) > 640


def test_angle_correction_keeps_original_scene_box_and_caches_both_views(monkeypatch):
    import app.services.pipeline as pipeline

    class Detector:
        def detect(self, image):
            return [(40, 30, 70, 25, .4)]

    original_crops = []

    def fake_rectify(crop):
        original_crops.append(crop.copy())
        return np.full((20, 60, 3), 220, np.uint8), dict(applied=True, angle_degrees=-12,
            reason='plate_outline', quad=[[0, 0], [60, 0], [60, 20], [0, 20]],
            method='quadrilateral_perspective', source_size=[crop.shape[1], crop.shape[0]],
            output_size=[60, 20], time_ms=.5)

    monkeypatch.setattr(pipeline, 'rectify_plate', fake_rectify)
    store = CropStore()
    uncorrected = process(image_bytes(), Detector(), store, NeverOCR(), read_text=False)
    corrected = process(image_bytes(), Detector(), store, NeverOCR(), read_text=False, angle_correction=True)
    detection = corrected['detections'][0]
    assert detection['bounding_box'] == uncorrected['detections'][0]['bounding_box']
    assert detection['tight_plate_box'] == dict(x=40, y=30, width=70, height=25)
    assert corrected['angle_correction'] is True and corrected['plate_count'] == 1
    assert detection['crop_url'] != detection['rectified_crop_url']
    original_jpeg = store.get((corrected['request_id'], detection['id']))
    corrected_jpeg = store.get((corrected['request_id'], detection['id'] + '-rectified'))
    assert original_jpeg and corrected_jpeg and original_jpeg != corrected_jpeg
    assert corrected['ocr_time_ms'] == 0


def test_region_boxes_remain_available_when_vehicle_model_fails():
    class Detector:
        def detect(self, image):
            return [(10, 10, 50, 20, .2)]

    class Vehicles:
        def objects(self, image):
            raise RuntimeError('GPU context unavailable')

    result = process(image_bytes(), Detector(), CropStore(), NeverOCR(), Vehicles(), read_text=False)
    assert result['plate_count'] == 1
    assert result['vehicle_summary']['available'] is False
    assert result['vehicle_summary']['total_vehicles'] is None


def test_regions_filter_only_invalid_geometry_not_confidence():
    class Detector:
        def detect(self, image):
            return [(10, 10, 50, 20, .1), (500, 10, 50, 20, .9), (10, 10, -1, 20, .9),
                    (float('nan'), 10, 50, 20, .9)]

    result = process(image_bytes(), Detector(), CropStore(), NeverOCR(), read_text=False)
    assert result['plate_count'] == 1
    assert result['detections'][0]['detection_confidence'] == .1
