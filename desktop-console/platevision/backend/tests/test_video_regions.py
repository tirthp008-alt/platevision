"""Region-only recording contracts; no model weights or GPU needed."""
import json

import cv2
import numpy as np
import pytest

from app.services.video import (
    VideoJob, PlateTracks, annotate, detect_video_frame, process_video, read_tracks, visible,
)


class ForbiddenOCR:
    def __getattr__(self, name):
        raise AssertionError(f'Region mode must not access OCR: {name}')


def test_region_reader_never_uses_ocr_even_with_green_filter_enabled(tmp_path):
    job = VideoJob('regions', tmp_path, tmp_path/'input.mp4')
    job.update(read_text=False, green_filter=True)
    tracker = PlateTracks(12)
    observations = tracker.update([(15, 40, 40, 15, .26)], 0, np.zeros((100, 100, 3), np.uint8))
    assert read_tracks(tracker.tracks, ForbiddenOCR(), job) == 0.
    track = tracker.tracks[0]
    assert track['confidence'] is None
    assert track['text'] == ''
    assert track['format_status'] == 'not_requested'
    assert visible(observations[0], track)
    assert 'green_enhancement' not in track


def test_region_video_uses_plate_regions_directly():
    class Detector:
        def detect_regions(self, image, profile):
            return [(10, 10, 30, 10, .26)]

        def detect_fast(self, image):
            raise AssertionError('Do not use ANPR proposal filtering')

    image = np.zeros((100, 100, 3), np.uint8)
    assert len(detect_video_frame(Detector(), image, 'fast', read_text=False)) == 1
    assert len(detect_video_frame(Detector(), image, 'fast', read_text=True)) == 1


def test_region_annotations_include_vehicle_category_plate_score_and_visible_count(monkeypatch):
    labels = []
    monkeypatch.setattr(cv2, 'putText', lambda image, text, *args, **kwargs: labels.append(text))
    tracks = [dict(id=1, text='', confidence=None, format_status='not_requested')]
    observations = [dict(track_id=1, box=[30, 60, 40, 10], score=.27)]
    vehicles = [dict(id='vehicle-1', track_id=1, box=[10, 35, 90, 60], category='two_wheeler', confidence=.85)]
    annotate(np.zeros((120, 240, 3), np.uint8), observations, tracks, vehicles)
    assert labels == ['V1 two wheeler 85%', 'Plate #1 27%', 'Visible: 1 vehicles | 1 plates']


def test_region_video_preserves_all_frames_tracks_vehicles_and_corrects_once_per_plate(tmp_path, monkeypatch):
    import app.services.rectification as rectification

    source = tmp_path/'input.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 12., (320, 180))
    assert writer.isOpened()
    for frame in range(4):
        image = np.full((180, 320, 3), 30+frame, np.uint8)
        cv2.rectangle(image, (25+frame, 90), (70+frame, 105), (245, 245, 245), -1)
        cv2.rectangle(image, (200+frame, 95), (245+frame, 110), (65, 180, 65), -1)
        writer.write(image)
    writer.release()

    class Plates:
        calls = 0
        runtime = 'test'
        fast_models = {'landscape': object()}

        def detect_fast(self, image):
            raise AssertionError('Region detector is required')

        def detect_regions(self, image, profile):
            shift = self.calls
            self.calls += 1
            return [(25+shift, 90, 45, 15, .27), (200+shift, 95, 45, 15, .28)]

    class Vehicles:
        calls = 0

        def objects(self, image):
            shift = self.calls
            self.calls += 1
            objects = [(10+shift, 40, 100, 110, .85, 2), (170+shift, 45, 100, 110, .80, 3)]
            return objects if shift % 2 else objects[::-1]

    corrected = []

    def rectify(crop):
        corrected.append(crop.copy())
        return crop.copy(), dict(applied=True, angle_degrees=5., reason='test contour',
                                method='perspective', time_ms=.5)

    monkeypatch.setattr(rectification, 'rectify_plate', rectify)
    plates = Plates(); vehicles = Vehicles()
    job = VideoJob('regions', tmp_path, source)
    job.update(read_text=False, angle_correction=True, green_filter=True)
    process_video(job, plates, ForbiddenOCR(), vehicles)
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert plates.calls == vehicles.calls == 4
    assert state['frames'] == 4 and state['fps'] == 12.
    assert state['source_width'] == state['width'] == 320
    assert state['source_height'] == state['height'] == 180
    assert state['all_frames_searched'] is True
    assert state['ocr_total_ms'] == 0.
    assert state['ocr_engine'] == 'not_requested'
    assert state['mode'] == 'regions'
    assert state['rectification_total_ms'] == 1.
    assert len(corrected) == len(state['tracks']) == 2
    assert state['plate_summary']['max_visible'] == 2
    assert state['vehicle_summary']['total_tracks'] == 2
    assert state['vehicle_summary']['max_visible'] == 2
    assert state['vehicle_summary']['counts_by_category']['car'] == 1
    assert state['vehicle_summary']['counts_by_category']['two_wheeler'] == 1
    assert state['analysis_sample_interval_frames'] == 12
    assert [frame['frame'] for frame in state['analysis_timeline']] == [0, 3]
    assert all(frame['visible_plates'] == frame['visible_vehicles'] == 2 for frame in state['analysis_timeline'])
    assert all(t['observations'] == 4 for t in state['vehicle_tracks'])
    for track in state['tracks']:
        assert track['text'] == '' and track['ocr_confidence'] is None
        assert track['format_status'] == 'not_requested'
        assert track['detection_confidence'] >= .27
        assert track['rectified_crop_url'].endswith('?variant=rectified')
        assert (tmp_path/f"plate-{track['id']}.jpg").is_file()
        assert (tmp_path/f"plate-{track['id']}-rectified.jpg").is_file()
    assert not source.exists()
    saved = json.loads((tmp_path/'results.json').read_text())
    assert saved['vehicle_summary'] == state['vehicle_summary']
    capture = cv2.VideoCapture(str(tmp_path/'annotated.mp4'))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert frames == 4


def test_optional_vehicle_failure_and_crop_write_failure_do_not_erase_plate_video(tmp_path, monkeypatch):
    import app.services.video as video

    source = tmp_path/'input.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 12., (160, 100))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((100, 160, 3), 150, np.uint8))
    writer.release()

    class Plates:
        def detect_regions(self, image, profile):
            return [(45, 55, 45, 15, .30)]

    class Vehicles:
        taxonomy = 'uvh'
        calls = 0

        def objects(self, image):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError('Optional vehicle device unavailable')
            return [(20, 30, 100, 60, .9, 'Tempo/Traveller')]

    monkeypatch.setattr(video, '_save_crop', lambda path, crop: False)
    vehicles = Vehicles()
    job = VideoJob('regions', tmp_path, source)
    job.update(read_text=False)
    process_video(job, Plates(), ForbiddenOCR(), vehicles)
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert state['frames'] == 3
    assert state['tracks'][0]['observations'] == 3
    assert state['tracks'][0]['crop_url'] is None
    assert state['vehicle_summary']['incomplete'] is True
    assert state['vehicle_summary']['complete'] is False
    assert state['vehicle_summary']['total_vehicles'] is None
    assert state['vehicle_summary']['counts_by_category']['car'] is None
    assert 'tempo' in state['vehicle_summary']['supported_categories']
    assert state['warnings'] and 'Plate capture continued' in state['warnings'][0]
    assert state['analysis_timeline'][0]['visible_vehicles'] == 1
    assert state['analysis_timeline'][1]['visible_vehicles'] is None
    assert state['analysis_timeline'][1]['counts_by_category'] is None
    assert vehicles.calls == 2
    assert (tmp_path/'annotated.mp4').is_file()


def _reid_recording(tmp_path, *, read_text=False):
    source = tmp_path/'input.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10., (160, 100))
    assert writer.isOpened()
    for index in range(3):
        image = np.full((100, 160, 3), 40, np.uint8)
        if index == 1:
            for x in range(10, 150, 6):
                cv2.line(image, (x, 20), (x, 79), (220, 220, 220), 2)
        writer.write(image)
    writer.release()

    class Plates:
        def detect_regions(self, image, profile):
            return [(25, 65, 30, 10, .91), (105, 65, 30, 10, .92)]

        def detect_plates(self, image, refine=False):
            return self.detect_regions(image, 'detailed')

    class Vehicles:
        def objects(self, image):
            return [(10, 20, 60, 60, .95, 2), (90, 20, 60, 60, .95, 2)]

    job = VideoJob('recording', tmp_path, source)
    job.update(read_text=read_text)
    return job, Plates(), Vehicles()


class RecordingReID:
    def __init__(self, callback=None):
        self.crops = []
        self.callback = callback

    def embed(self, crops):
        self.crops.extend(crop.copy() for crop in crops)
        if self.callback:
            self.callback()
        descriptor = np.zeros(2048, np.float32)
        descriptor[0] = 1.
        return np.array([descriptor])


def test_recording_reid_embeds_best_unannotated_crop_once_per_vehicle_track(tmp_path):
    from app.services.camera_matches import CameraMatchGallery

    job, plates, vehicles = _reid_recording(tmp_path)
    engine = RecordingReID()
    gallery = CameraMatchGallery()
    context = dict(engine=engine, gallery=gallery, camera_id='Camera A', observed_at=1000.)
    process_video(job, plates, ForbiddenOCR(), vehicles, context)
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert state['reidentification']['status'] == 'complete'
    assert state['reidentification']['observation_count'] == len(engine.crops) == 2
    assert state['vehicle_summary']['total_tracks'] == state['plate_summary']['total_tracks'] == 2
    assert [track['plate_ids'] for track in state['vehicle_tracks']] == [[1], [2]]
    assert gallery.stats()['observations'] == 2
    for observation in state['reidentification']['observations']:
        assert observation['track_id'].startswith('recording:vehicle-')
        assert observation['observed_at'] == pytest.approx(1000.1)
        assert observation['first_seen_at'] == 1000.
        assert observation['last_seen_at'] == pytest.approx(1000.2)
    for crop in engine.crops:
        assert crop.shape == (60, 60, 3)
        assert crop.max() <= 235  # No colored annotation boxes or labels.
        assert np.max(np.ptp(crop.astype(int), axis=2)) < 10
    saved = (tmp_path/'results.json').read_text()
    assert 'descriptor' not in saved and 'crop":' not in saved
    assert json.loads(saved)['reidentification'] == state['reidentification']


def test_recording_reid_preserves_actual_partial_ocr_evidence(tmp_path):
    from app.services.camera_matches import CameraMatchGallery

    class Gallery(CameraMatchGallery):
        staged = None

        def register_batch(self, observations):
            self.staged = observations
            return super().register_batch(observations)

    class OCR:
        def read_plate(self, context, tight):
            return 'GJ01??1234', .91

    job, plates, vehicles = _reid_recording(tmp_path, read_text=True)
    gallery = Gallery()
    context = dict(engine=RecordingReID(), gallery=gallery, camera_id='A', observed_at=1000.)
    process_video(job, plates, OCR(), vehicles, context)
    assert job.snapshot()['status'] == 'complete', job.snapshot().get('error')
    assert [observation['plate_text'] for observation in gallery.staged] == ['GJ01??1234'] * 2
    assert [observation['plate_confidence'] for observation in gallery.staged] == [.91] * 2


def test_tentative_vehicle_is_counted_and_boxed_without_embedding_or_registration(tmp_path, monkeypatch):
    from app.services.camera_matches import CameraMatchGallery
    import app.services.video as video

    job, plates, vehicles = _reid_recording(tmp_path)
    original_objects = vehicles.objects
    frames = []

    def one_tentative_vehicle(image):
        objects = original_objects(image)
        frames.append(None)
        return objects if len(frames) == 1 else objects[:1]

    vehicles.objects = one_tentative_vehicle
    drawn_vehicle_ids = []
    original_annotate = video.annotate

    def capture_annotations(image, observations, tracks, frame_vehicles=None):
        drawn_vehicle_ids.append([vehicle['track_id'] for vehicle in frame_vehicles or []])
        return original_annotate(image, observations, tracks, frame_vehicles)

    monkeypatch.setattr(video, 'annotate', capture_annotations)
    engine = RecordingReID()
    gallery = CameraMatchGallery()
    process_video(job, plates, ForbiddenOCR(), vehicles,
                  dict(engine=engine, gallery=gallery, camera_id='A', observed_at=1000.))
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert state['vehicle_summary']['total_tracks'] == 2
    assert state['vehicle_summary']['confirmed_tracks'] == state['vehicle_summary']['tentative_tracks'] == 1
    assert state['vehicle_summary']['max_visible'] == 2
    assert state['vehicle_tracks'][1]['observations'] == 1
    assert state['vehicle_tracks'][1]['confirmed'] is False
    assert drawn_vehicle_ids == [[1, 2], [1], [1]]
    assert len(engine.crops) == gallery.stats()['observations'] == 1
    assert state['reidentification']['observation_count'] == 1
    assert state['reidentification']['observations'][0]['track_id'] == 'recording:vehicle-1'
    assert state['plate_summary']['total_tracks'] == 2


@pytest.mark.parametrize('failure', ['missing', 'inference', 'invalid_descriptor'])
def test_optional_recording_reid_failure_keeps_regions_counts_and_export(tmp_path, failure):
    from app.services.camera_matches import CameraMatchGallery

    job, plates, vehicles = _reid_recording(tmp_path)
    gallery = CameraMatchGallery()

    class BrokenReID(RecordingReID):
        def embed(self, crops):
            self.crops.extend(crop.copy() for crop in crops)
            if failure == 'inference' and len(self.crops) == 2:
                raise RuntimeError('Optional appearance device unavailable')
            return np.array([[float('nan') if failure == 'invalid_descriptor' else 1.] * 2048])

    engine = None if failure == 'missing' else BrokenReID()
    process_video(job, plates, ForbiddenOCR(), vehicles,
                  dict(engine=engine, gallery=gallery, camera_id='A', observed_at=1000.))
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert state['reidentification']['status'] == ('unavailable' if failure == 'missing' else 'failed')
    assert state['vehicle_summary']['total_tracks'] == state['plate_summary']['total_tracks'] == 2
    assert state['frames'] == 3 and (tmp_path/'annotated.mp4').is_file()
    assert gallery.stats()['observations'] == 0
    assert state['reidentification']['warning'] in state['warnings']


def test_incomplete_vehicle_analysis_skips_reid_without_affecting_plate_capture(tmp_path):
    from app.services.camera_matches import CameraMatchGallery

    job, plates, vehicles = _reid_recording(tmp_path)
    valid_objects = vehicles.objects
    calls = []

    def fail_second_frame(image):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError('Vehicle analysis failed mid-recording')
        return valid_objects(image)

    vehicles.objects = fail_second_frame
    gallery = CameraMatchGallery()
    engine = RecordingReID()
    process_video(job, plates, ForbiddenOCR(), vehicles,
                  dict(engine=engine, gallery=gallery, camera_id='A', observed_at=1000.))
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert state['vehicle_summary']['total_tracks'] is None
    assert state['vehicle_summary']['incomplete'] is True
    assert state['plate_summary']['total_tracks'] == 2
    assert state['reidentification']['status'] == 'failed'
    assert engine.crops == [] and gallery.stats()['observations'] == 0


def test_cancel_during_recording_reid_does_not_register_or_publish(tmp_path):
    from app.services.camera_matches import CameraMatchGallery

    job, plates, vehicles = _reid_recording(tmp_path)
    gallery = CameraMatchGallery()
    engine = RecordingReID(callback=job.cancel.set)
    process_video(job, plates, ForbiddenOCR(), vehicles,
                  dict(engine=engine, gallery=gallery, camera_id='A', observed_at=1000.))
    assert job.snapshot()['status'] == 'cancelled'
    assert len(engine.crops) == 1
    assert gallery.stats()['observations'] == 0
    assert not (tmp_path/'annotated.mp4').exists()
    assert not (tmp_path/'results.json').exists()


def test_failed_export_never_registers_staged_vehicle_descriptors(tmp_path, monkeypatch):
    from io import BytesIO
    from app.services.camera_matches import CameraMatchGallery
    import app.services.video as video

    class FailedEncoder:
        returncode = 1

        def __init__(self, *args, **kwargs):
            self.stdin = BytesIO()

        def wait(self, timeout):
            return self.returncode

        def poll(self):
            return self.returncode

    job, plates, vehicles = _reid_recording(tmp_path)
    gallery = CameraMatchGallery()
    engine = RecordingReID()
    monkeypatch.setattr(video.subprocess, 'Popen', FailedEncoder)
    process_video(job, plates, ForbiddenOCR(), vehicles,
                  dict(engine=engine, gallery=gallery, camera_id='A', observed_at=1000.))
    assert job.snapshot()['status'] == 'failed'
    assert len(engine.crops) == 2
    assert gallery.stats()['observations'] == 0


def test_vehicle_plate_evidence_uses_temporal_ownership_and_omits_ties():
    from app.services.video import _vehicle_plate_ids

    frames = [
        [dict(track_id=1, plate_ids=[1, 2, 3])],
        [dict(track_id=2, plate_ids=[1, 2]), dict(track_id=1, plate_ids=[3])],
        [dict(track_id=1, plate_ids=[1])],
    ]
    assert _vehicle_plate_ids(frames, [dict(id=1), dict(id=2)]) == {1: [1]}


@pytest.mark.parametrize('simultaneous', [False, True])
def test_recording_groups_repeated_readings_but_retains_geometric_tracks_and_collisions(tmp_path, simultaneous):
    job, plates, vehicles = _reid_recording(tmp_path, read_text=True)
    calls = []

    def regions(image, profile):
        index = len(calls); calls.append(None)
        if simultaneous:
            return [(25, 65, 30, 10, .91), (105, 65, 30, 10, .92)]
        return [(25 if index == 0 else 105, 65, 30, 10, .91)] if index != 1 else []

    class Reader:
        def read_plate(self, context, tight):
            return 'GJ01AB1234', .95

    plates.detect_regions = regions
    process_video(job, plates, Reader(), vehicles)
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert len(state['tracks']) == state['plate_summary']['total_tracks'] == 2
    summary = state['recognized_plate_summary']
    assert summary['recognized_track_count'] == 2
    assert state['recognized_plates'] == summary['groups']
    if simultaneous:
        assert summary['unique_registrations'] == 0 and summary['ambiguous_track_count'] == 2
        assert len(summary['groups']) == 2 and summary['suppressed_repeat_count'] == 0
    else:
        assert summary['unique_registrations'] == 1 and summary['suppressed_repeat_count'] == 1
        assert summary['groups'][0]['track_ids'] == [1, 2]
        assert summary['groups'][0]['observations'] == 2
    assert all(track['crop_url'] for track in state['tracks'])


def test_ocr_enabled_recording_keeps_weak_unreadable_regions(tmp_path):
    job, plates, vehicles = _reid_recording(tmp_path, read_text=True)
    original_regions = plates.detect_regions
    plates.detect_regions = lambda image, profile: [(*box[:4], .26) for box in original_regions(image, profile)]

    class Reader:
        def read_plate(self, context, tight):
            return 'BLUR', .1

    process_video(job, plates, Reader(), vehicles)
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert len(state['tracks']) == state['plate_summary']['max_visible'] == 2
    assert all(track['observations'] == 3 and track['format_status'] == 'uncertain' for track in state['tracks'])
    assert state['recognized_plate_summary']['unresolved_track_ids'] == [1, 2]
    assert state['recognized_plates'] == []


def test_recording_uses_details_once_and_reuses_correction_for_export(tmp_path, monkeypatch):
    import app.services.rectification as rectification

    job, plates, vehicles = _reid_recording(tmp_path, read_text=True)
    job.update(angle_correction=True)
    calls = []

    class Reader:
        def read_many_details(self, pairs, *, angle_correction):
            calls.append(len(pairs))
            assert angle_correction is True
            return [dict(text='GJ01AB1234', confidence=.91, original_text='BLUR', original_confidence=.2,
                         requires_review=False, conflicting_readings=False, selected_method='rectified',
                         attempts=[dict(method='original', text='BLUR', confidence=.2),
                                   dict(method='rectified', text='GJ01AB1234', confidence=.91)],
                          preprocessing_time_ms=1., rectification=dict(applied=True, time_ms=.5, method='perspective'),
                          _rectification_attempt=dict(source='context', transform='rectify_plate'),
                          rectified_crop=np.full((20, 60, 3), 120, np.uint8)) for _ in pairs]

        def read_plate_details(self, *args, **kwargs):
            raise AssertionError('Confident detailed batch results should not repeat OCR')

    def repeated_rectification(crop):
        raise AssertionError('Export must reuse the already corrected OCR crop')

    monkeypatch.setattr(rectification, 'rectify_plate', repeated_rectification)
    process_video(job, plates, Reader(), vehicles)
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert calls == [2]
    assert state['rectification_total_ms'] == 1.
    for track in state['tracks']:
        assert track['ocr_review']['original_text'] == 'BLUR'
        assert track['ocr_review']['selected_method'] == 'rectified'
        assert track['rectified_crop_url'].endswith('?variant=rectified')
        assert 'rectified_crop' not in track['ocr_review']
        assert '_rectification_attempt' not in track['ocr_review']
    json.dumps(state)


def test_contradictory_ocr_details_stay_uncertain_and_cannot_strengthen_reid(tmp_path):
    from app.services.reidentification import strongest_plate

    job = VideoJob('details', tmp_path, tmp_path/'unused')
    job.update(angle_correction=False)
    crop = np.zeros((20, 80, 3), np.uint8)
    tracks = [dict(crops=[(1, crop, crop)])]

    class Reader:
        def read_plate_details(self, context, tight, *, angle_correction):
            assert angle_correction is False
            return dict(text='GJ01AB1234', confidence=.98, original_text='GJ01AB1235', original_confidence=.91,
                        requires_review=True, conflicting_readings=True, selected_method='contrast', attempts=[])

    read_tracks(tracks, Reader(), job)
    assert tracks[0]['format_status'] == 'uncertain'
    assert tracks[0]['ocr_review']['requires_review'] is True
    assert strongest_plate(tracks) == (None, 0.)


def test_real_ocr_detail_strategy_is_used_by_recording_batches(tmp_path, monkeypatch):
    from app.services.ocr import PlateOCR
    import app.services.rectification as rectification

    responses = iter([('BLUR', .2), ('GJ01AB1234', .86), ('GJ01AB1234', .95)])
    calls = []

    class Recognizer:
        def recognize(self, crops):
            calls.append(len(crops))
            return [next(responses) for _ in crops]

    ocr = PlateOCR.__new__(PlateOCR)
    ocr.batch = Recognizer()
    context = np.full((24, 90, 3), 30, np.uint8)
    tight = np.full((20, 80, 3), 30, np.uint8)
    corrected = np.full((20, 80, 3), 120, np.uint8)
    rectifications = []

    def rectify(crop):
        rectifications.append(crop)
        return corrected, dict(applied=True, time_ms=.5, method='perspective')

    monkeypatch.setattr(rectification, 'rectify_plate', rectify)
    tracks = [dict(crops=[(1, context, tight)] * 2)]
    job = VideoJob('details', tmp_path, tmp_path/'unused')
    job.update(angle_correction=True)
    read_tracks(tracks, ocr, job)
    assert calls == [1, 1, 1] and len(rectifications) == 1
    assert tracks[0]['text'] == 'GJ01AB1234' and tracks[0]['confidence'] == .95
    assert tracks[0]['format_status'] == 'valid'
    assert tracks[0]['ocr_review']['selected_method'] == 'rectified_original'
    assert tracks[0]['ocr_review']['original_confidence'] == .2
    assert tracks[0]['ocr_review']['requires_review'] is False
    assert tracks[0]['ocr_rectified_crop'] is corrected


def test_video_recovers_unreadable_vehicle_plates_before_tracking_and_reports_cost(tmp_path, monkeypatch):
    import app.services.video as video
    import app.services.vehicle_plate_search as search

    job, plates, vehicles = _reid_recording(tmp_path, read_text=True)
    plates.detect_regions = lambda image, profile: [(25, 65, 30, 10, .91)]
    calls = []
    drawn = []
    original_annotate = video.annotate
    monkeypatch.setattr(video.settings, 'vehicle_plate_max_crops', 2)

    def recover(image, detector, plate_boxes, vehicle_objects, *, start_index=0, max_regions=None):
        calls.append(start_index)
        assert detector is plates and image.shape == (100, 160, 3)
        assert plate_boxes == [(25, 65, 30, 10, .91)]
        assert len(vehicle_objects) == 2
        return plate_boxes + [(105, 65, 30, 10, .26)], dict(
            enabled=True, eligible_vehicles=1, searched_vehicles=1, added_regions=1, time_ms=999999.,
            failed_crops=0, remaining_vehicles=0, next_start_index=0)

    def annotated(image, observations, tracks, frame_vehicles=None):
        drawn.append([item['box'] for item in observations])
        return original_annotate(image, observations, tracks, frame_vehicles)

    class Reader:
        def read_plate(self, context, tight):
            return '', .05

    monkeypatch.setattr(search, 'recover_vehicle_plates', recover)
    monkeypatch.setattr(video, 'annotate', annotated)
    process_video(job, plates, Reader(), vehicles)
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert calls == [0, 2, 4]
    assert all(boxes == [[25, 65, 30, 10], [105, 65, 30, 10]] for boxes in drawn)
    assert len(state['tracks']) == state['plate_summary']['total_tracks'] == 2
    assert all(track['text'] == '' and track['observations'] == 3 for track in state['tracks'])
    assert state['tracks'][0]['localization_source'] == 'whole_frame'
    assert state['tracks'][0]['localization_requires_review'] is False
    assert state['tracks'][1]['localization_source'] == 'vehicle_crop'
    assert state['tracks'][1]['localization_requires_review'] is True
    assert state['tracks'][1]['has_vehicle_crop_evidence'] is True
    assert state['tracks'][1]['has_whole_frame_evidence'] is False
    assert state['recognized_plate_summary']['unresolved_track_count'] == 2
    report = state['vehicle_plate_search']
    assert report['scope'] == 'recording' and report['frames_searched'] == 3
    assert report['searched_vehicles'] == report['eligible_vehicles'] == report['added_regions'] == 3
    assert report['failed_crops'] == report['remaining_vehicles'] == report['failed_frames'] == 0
    assert 0 <= report['time_ms'] < 999999.
    assert state['plate_detection_ms']['p50'] >= state['whole_frame_plate_detection_ms']['p50']
    assert [item['vehicle_plate_search']['added_regions'] for item in state['analysis_timeline']] == [1, 1]
    assert all(item['vehicle_plate_search']['scope'] == 'frame' for item in state['analysis_timeline'])


def test_vehicle_inference_failure_never_reuses_stale_crop_search_regions(tmp_path, monkeypatch):
    import app.services.vehicle_plate_search as search

    job, plates, vehicles = _reid_recording(tmp_path)
    original_objects = vehicles.objects
    detected = []
    searched = []

    def fail_after_first(image):
        detected.append(None)
        if len(detected) > 1:
            raise RuntimeError('Optional vehicle inference failed')
        return original_objects(image)

    def recover(image, detector, plate_boxes, vehicle_objects, *, start_index=0, max_regions=None):
        searched.append(list(vehicle_objects))
        return plate_boxes, dict(enabled=True, eligible_vehicles=0, searched_vehicles=0, added_regions=0,
                                 time_ms=0., failed_crops=0, remaining_vehicles=0, next_start_index=0)

    vehicles.objects = fail_after_first
    monkeypatch.setattr(search, 'recover_vehicle_plates', recover)
    process_video(job, plates, ForbiddenOCR(), vehicles)
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert [len(objects) for objects in searched] == [2, 0, 0]
    assert len(detected) == 2
    assert state['plate_summary']['total_tracks'] == 2
    assert state['vehicle_summary']['total_vehicles'] is None


def test_optional_vehicle_plate_search_error_preserves_original_detections(tmp_path, monkeypatch):
    import app.services.vehicle_plate_search as search

    job, plates, vehicles = _reid_recording(tmp_path)

    def broken_search(image, detector, plate_boxes, vehicle_objects, **kwargs):
        plate_boxes.clear()
        raise RuntimeError('Optional crop detector failed')

    monkeypatch.setattr(search, 'recover_vehicle_plates', broken_search)
    process_video(job, plates, ForbiddenOCR(), vehicles)
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert state['plate_summary']['total_tracks'] == 2
    assert all(track['observations'] == 3 for track in state['tracks'])
    assert state['vehicle_plate_search']['failed_frames'] == 3
    assert state['vehicle_plate_search']['added_regions'] == 0
    assert any('whole-frame plate regions were retained' in warning for warning in state['warnings'])
    assert (tmp_path/'annotated.mp4').is_file()


@pytest.mark.parametrize('crop_fails', [False, True])
def test_real_supplemental_search_only_visits_vehicle_without_plate_and_keeps_empty_ocr(tmp_path, crop_fails):
    job, plates, vehicles = _reid_recording(tmp_path, read_text=True)
    plates.detect_regions = lambda image, profile: [(25, 65, 30, 10, .91)]
    searched_crops = []

    def vehicle_plate_regions(crop):
        searched_crops.append(crop.shape)
        if crop_fails:
            raise RuntimeError('Optional cropped plate inference unavailable')
        # The unmatched vehicle starts at (90,20); the helper's .08 padding
        # gives a crop origin of (85,15), placing this actual plate at (105,65).
        return [(20, 50, 30, 10, .26)]

    class Reader:
        def read_plate(self, context, tight):
            return '', 0.

    plates.detect_vehicle_regions = vehicle_plate_regions
    process_video(job, plates, Reader(), vehicles)
    state = job.snapshot()
    assert state['status'] == 'complete', state.get('error')
    assert searched_crops == [(70, 70, 3)] * 3
    assert state['vehicle_plate_search']['searched_vehicles'] == 3
    assert state['vehicle_plate_search']['failed_crops'] == (3 if crop_fails else 0)
    assert state['vehicle_plate_search']['added_regions'] == (0 if crop_fails else 3)
    assert state['plate_summary']['total_tracks'] == (1 if crop_fails else 2)
    assert all(track['text'] == '' and track['observations'] == 3 for track in state['tracks'])
    assert state['vehicle_summary']['total_tracks'] == 2
    if not crop_fails:
        assert state['tracks'][1]['localization_requires_review'] is True
        saved = json.loads((tmp_path/'results.json').read_text())
        assert saved['tracks'][1]['localization_source'] == 'vehicle_crop'


def test_supplemental_plate_annotations_remain_amber_even_with_confident_ocr(monkeypatch):
    import app.services.video as video

    track = dict(id=1, text='GJ01AB1234', confidence=.99, format_status='valid')
    observations = [dict(track_id=1, box=[20, 30, 60, 20], score=.95)]
    video._mark_plate_sources(observations, [track], 0)
    drawn = []
    monkeypatch.setattr(video, '_draw_box', lambda frame, box, text, color: drawn.append((box, text, color)))
    video.annotate(np.zeros((100, 120, 3), np.uint8), observations, [track])
    assert drawn[0][0] == (20, 30, 60, 20)
    assert 'GJ01AB1234' in drawn[0][1] and 'vehicle crop review' in drawn[0][1]
    assert drawn[0][2] == (80, 185, 245)
    assert track['format_status'] == 'valid' and track['confidence'] == .99


def test_later_whole_frame_evidence_clears_track_review_without_rewriting_frame_provenance():
    import app.services.video as video

    track = dict(id=1)
    first = [dict(track_id=1)]
    video._mark_plate_sources(first, [track], 0)
    assert track['localization_requires_review'] is True
    later = [dict(track_id=1)]
    video._mark_plate_sources(later, [track], 1)
    assert track['localization_source'] == 'mixed'
    assert track['has_vehicle_crop_evidence'] is True and track['has_whole_frame_evidence'] is True
    assert track['localization_requires_review'] is False
    assert first[0]['localization_source'] == 'vehicle_crop' and first[0]['localization_requires_review'] is True
    assert later[0]['localization_source'] == 'whole_frame' and later[0]['localization_requires_review'] is False
