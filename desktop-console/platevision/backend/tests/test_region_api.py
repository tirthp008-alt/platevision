"""HTTP region-capture contracts, using CPU fixtures and no model weights."""
import asyncio
import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.services.pipeline import CropStore, InvalidImage, decode
from app.services.video import VideoJob


def png(image=None):
    image = np.zeros((120, 240, 3), np.uint8) if image is None else image
    success, encoded = cv2.imencode('.png', image)
    assert success
    return encoded.tobytes()


@pytest.fixture
def client(monkeypatch):
    # Do not enter the lifespan context: model startup belongs to integration
    # tests, and these assertions must not initialize any inference engines.
    monkeypatch.setattr(main, 'crops', CropStore())
    monkeypatch.setattr(main, 'inference_lock', asyncio.Lock())
    monkeypatch.setattr(main, 'video_upload_lock', asyncio.Lock())
    monkeypatch.setattr(main, 'video_jobs', {})
    monkeypatch.setattr(main, 'video_tasks', set())
    monkeypatch.setattr(main, 'vehicle_detector', None)
    return TestClient(main.app)


@pytest.mark.parametrize('route', ['/api/detect/image', '/api/detect/frame'])
def test_explicit_region_requests_ignore_ocr_selection_and_keep_all_plate_boxes(client, monkeypatch, route):
    class Plates:
        def detect_regions(self, image, profile):
            assert profile == 'fast'
            return [(10, 10, 45, 20, .19), (175, 70, 45, 20, .22)]

    class Vehicles:
        taxonomy = 'uvh'

        def objects(self, image):
            return [(0, 0, 90, 100, .9, 'Tempo-traveller')]

    def forbidden_ocr(*args, **kwargs):
        raise AssertionError('Region requests must not select, create or invoke OCR')

    monkeypatch.setattr(main, 'detector', Plates())
    monkeypatch.setattr(main, 'vehicle_detector', Vehicles())
    monkeypatch.setattr(main, 'selected_ocr', forbidden_ocr)
    monkeypatch.setattr(main, 'PlateOCR', forbidden_ocr)
    response = client.post(route, data={'read_text':'false','ocr_model': 'does-not-exist', 'green_filter': 'true'},
                           files={'image': ('frame.png', png(), 'image/png')})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['read_text'] is False and result['angle_correction'] is True
    assert result['ocr_model'] == 'not_requested' and result['ocr_time_ms'] == 0
    assert result['plate_count'] == 2
    assert result['vehicle_summary']['total_vehicles'] == 1
    assert result['vehicle_summary']['counts_by_category']['tempo'] == 1
    assert result['vehicle_summary']['unsupported_categories'] == []
    assert result['vehicle_detection_time_ms'] is not None
    assert result['detections'][0]['vehicle_id'] == 'vehicle-1'
    assert result['detections'][1]['vehicle_id'] is None
    for plate in result['detections']:
        assert plate['ocr_confidence'] is None and plate['raw_text'] == ''
        assert plate['format_status'] == 'not_requested'
        assert 'green_enhancement' not in plate


def test_ocr_is_enabled_by_default_and_preserves_reading(client, monkeypatch):
    class Plates:
        def detect_fast(self, image):
            return [(10, 10, 45, 20, .19)]

    class Reader:
        runtime = 'unit-test'

        def read(self, crop):
            return 'MH12AB1234', .96

    monkeypatch.setattr(main, 'detector', Plates())
    monkeypatch.setattr(main, 'ocr', Reader())
    response = client.post('/api/detect/image', data={'angle_correction': 'false'},
                           files={'image': ('frame.png', png(), 'image/png')})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['read_text'] is True and result['angle_correction'] is False
    assert result['ocr_model'] == 'ppocr' and result['ocr_engine'] == 'unit-test'
    assert result['plate_count'] == 1
    assert result['detections'][0]['normalized_text'] == 'MH12AB1234'
    assert result['detections'][0]['ocr_confidence'] == .96
    assert 'rectification' not in result['detections'][0]
    invalid = client.post('/api/detect/image', data={'read_text': 'true', 'ocr_model': 'invalid'},
                          files={'image': ('frame.png', png(), 'image/png')})
    assert invalid.status_code == 422


def test_original_and_rectified_photo_crops_are_separately_retrievable(client, monkeypatch):
    image = np.zeros((120, 240, 3), np.uint8)
    quad = cv2.boxPoints(((120, 60), (170, 48), -12)).astype(np.int32)
    cv2.fillConvexPoly(image, quad, (40, 170, 50))

    class Plates:
        def detect_regions(self, image, profile):
            return [(0, 0, 240, 120, .9)]

    monkeypatch.setattr(main, 'detector', Plates())
    response = client.post('/api/detect/image', data={'read_text':'false'}, files={'image': ('slanted.png', png(image), 'image/png')})
    assert response.status_code == 200, response.text
    plate = response.json()['detections'][0]
    assert plate['rectification']['applied'] is True
    assert plate['bounding_box'] == dict(x=0, y=0, width=240, height=120)
    original = client.get(plate['crop_url'])
    corrected = client.get(plate['rectified_crop_url'])
    assert original.status_code == corrected.status_code == 200
    assert original.headers['content-type'] == corrected.headers['content-type'] == 'image/jpeg'
    assert original.headers['cache-control'] == 'no-store'
    original_image = cv2.imdecode(np.frombuffer(original.content, np.uint8), cv2.IMREAD_COLOR)
    corrected_image = cv2.imdecode(np.frombuffer(corrected.content, np.uint8), cv2.IMREAD_COLOR)
    assert original_image.shape == image.shape
    assert corrected_image.shape[1] < original_image.shape[1]
    assert corrected_image.shape[0] < original_image.shape[0]
    assert client.get('/api/results/missing/missing/crop').status_code == 404


def test_decode_rotation_is_applied_before_plate_coordinates_are_generated(client, monkeypatch):
    seen = []

    class Plates:
        def detect_regions(self, image, profile):
            seen.append(image.shape[:2])
            return [(2, 5, 10, 20, .8)]

    source = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new('RGB', (40, 20)).save(source, format='JPEG', exif=exif)
    monkeypatch.setattr(main, 'detector', Plates())
    response = client.post('/api/detect/image', data={'read_text':'false'}, files={'image': ('rotated.jpg', source.getvalue(), 'image/jpeg')})
    assert response.status_code == 200, response.text
    result = response.json()
    assert seen == [(40, 20)]
    assert result['image'] == dict(width=20, height=40)
    assert result['detections'][0]['tight_plate_box'] == dict(x=2, y=5, width=10, height=20)


def test_decoder_rejects_corrupt_and_oversized_images_before_inference(client, monkeypatch):
    import app.services.pipeline as pipeline

    class Plates:
        def detect_regions(self, image, profile):
            raise AssertionError('Invalid images must not reach inference')

    monkeypatch.setattr(main, 'detector', Plates())
    response = client.post('/api/detect/frame', files={'image': ('broken.jpg', b'not a JPEG', 'image/jpeg')})
    assert response.status_code == 422
    monkeypatch.setattr(pipeline.Image, 'open', lambda data: SimpleNamespace(width=5001, height=4000))

    def forbidden_decode(*args):
        raise AssertionError('Oversized header must be rejected before full decoding')

    monkeypatch.setattr(pipeline.cv2, 'imdecode', forbidden_decode)
    with pytest.raises(InvalidImage, match='20 megapixels'):
        decode(b'header checked before payload decode')


def test_region_video_upload_preserves_flags_without_selecting_ocr(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, 'video_output_dir', str(tmp_path))
    monkeypatch.setattr(main, 'detector', object())
    calls = []

    def forbidden_ocr(*args):
        raise AssertionError('Video region requests must not select an OCR engine')

    async def capture_job(job, reader):
        calls.append((job.snapshot(), reader))

    monkeypatch.setattr(main, 'selected_ocr', forbidden_ocr)
    monkeypatch.setattr(main, 'run_video', capture_job)
    response = client.post('/api/videos', data={'read_text':'false','ocr_model': 'invalid', 'green_filter': 'true'},
                           files={'video': ('clip.mp4', b'fixture bytes; decoder is tested separately', 'video/mp4')})
    assert response.status_code == 202, response.text
    state = response.json()
    assert state['read_text'] is False and state['angle_correction'] is True
    assert state['green_filter'] is False and state['ocr_model'] == 'not_requested'
    assert len(calls) == 1 and calls[0][1] is None
    assert calls[0][0]['read_text'] is False


def test_video_crop_variant_is_validated_and_original_is_preserved(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, 'video_output_dir', str(tmp_path))
    job = VideoJob('crop-test', tmp_path, tmp_path / 'input.mp4')
    job.update(status='complete')
    monkeypatch.setattr(main, 'video_jobs', {job.id: job})
    original = np.full((40, 100, 3), 60, np.uint8)
    corrected = np.full((25, 80, 3), 180, np.uint8)
    assert cv2.imwrite(str(tmp_path / 'plate-1.jpg'), original)
    assert cv2.imwrite(str(tmp_path / 'plate-1-rectified.jpg'), corrected)
    raw = client.get(f'/api/videos/{job.id}/crops/1')
    fixed = client.get(f'/api/videos/{job.id}/crops/1?variant=rectified')
    assert raw.status_code == fixed.status_code == 200
    assert raw.content != fixed.content
    assert client.get(f'/api/videos/{job.id}/crops/1?variant=../../anything').status_code == 422
    assert client.get(f'/api/videos/{job.id}/crops/2?variant=rectified').status_code == 404
    job.update(status='processing')
    assert client.get(f'/api/videos/{job.id}/crops/1?variant=rectified').status_code == 404


def test_uvh_metadata_maps_class_indices_to_real_vehicle_names_without_loading_models(monkeypatch):
    import onnx
    import app.services.detector as detectors

    names = dict(enumerate(['Hatchback', 'Sedan', 'SUV', 'MUV', 'Bus', 'Truck',
                            'Three-wheeler', 'Two-wheeler', 'LCV', 'Mini-bus',
                            'tempo-traveller', 'bicycle', 'Van', 'Others']))
    monkeypatch.setattr(detectors.settings, 'vehicle_context_enabled', True)
    monkeypatch.setattr(detectors.settings, 'vehicle_model_path', 'unit-test-model.onnx')
    monkeypatch.setattr(detectors.settings, 'vehicle_taxonomy', 'uvh')
    monkeypatch.setattr(detectors.settings, 'acceleration', 'onnx')
    monkeypatch.setattr(onnx, 'load', lambda *args, **kwargs: SimpleNamespace(
        metadata_props=[SimpleNamespace(key='names', value=repr(names))]))

    class Engine:
        def __init__(self, path, classes, allowed):
            assert classes == 14 and allowed == set(range(14))

        def objects(self, image, threshold):
            assert threshold == detectors.settings.vehicle_confidence_threshold
            return [(5, 10, 40, 60, .9, 10), (65, 10, 40, 60, .8, 8)]

    monkeypatch.setattr(detectors, 'OnnxPlateDetector', Engine)
    vehicle_detector = detectors.get_vehicle_detector()
    assert vehicle_detector.taxonomy == 'uvh'
    assert [item[-1] for item in vehicle_detector.objects(np.zeros((100, 120, 3), np.uint8))] == ['tempo-traveller', 'LCV']


def test_uvh_configuration_rejects_generic_vehicle_vocabulary_before_loading_engine(monkeypatch):
    import onnx
    import app.services.detector as detectors

    monkeypatch.setattr(detectors.settings, 'vehicle_context_enabled', True)
    monkeypatch.setattr(detectors.settings, 'vehicle_model_path', 'wrong-model.onnx')
    monkeypatch.setattr(detectors.settings, 'vehicle_taxonomy', 'uvh')
    monkeypatch.setattr(onnx, 'load', lambda *args, **kwargs: SimpleNamespace(
        metadata_props=[SimpleNamespace(key='names', value="{0: 'car', 1: 'truck'}")]))
    with pytest.raises(RuntimeError, match='vocabulary'):
        detectors.get_vehicle_detector()
