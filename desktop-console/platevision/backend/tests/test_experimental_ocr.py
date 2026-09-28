import io
import numpy as np
from PIL import Image
from app.services.pipeline import process, CropStore
from app.services.video import PlateTracks, VideoJob, read_tracks


class Detector:
    def detect_fast(self, image):
        return [(10, 10, 80, 24, .91), (110, 10, 80, 24, .88)]


class Reader:
    def read_many(self, pairs):
        return [('MH12AB1234', .82)] * len(pairs)

    def read(self, crop):
        return 'WRONG123', .99


def photo():
    data = io.BytesIO()
    Image.new('RGB', (220, 60), (0, 110, 30)).save(data, format='PNG')
    return data.getvalue()


def test_filter_default_adds_no_extra_ocr_or_changes(monkeypatch):
    import app.services.experimental_ocr as experiment
    def unexpected(*args):
        raise AssertionError('Default requests must not run experimental filtering')
    monkeypatch.setattr(experiment, 'green_alternative', unexpected)
    result = process(photo(), Detector(), CropStore(), Reader(), profile='fast')
    assert len(result['detections']) == 2
    assert all('green_enhancement' not in item for item in result['detections'])


def test_opt_in_retains_all_original_text_scores_and_separate_crop():
    store = CropStore()
    result = process(photo(), Detector(), store, Reader(), profile='fast', green_filter=True)
    assert len(result['detections']) == 2
    for item in result['detections']:
        assert item['normalized_text'] == 'MH12AB1234'
        assert item['ocr_confidence'] == .82
        extra = item['green_enhancement']
        assert extra['candidate_text'] == 'WR0NG123'
        assert extra['candidate_ocr_confidence'] == .99
        assert extra['requires_review'] and extra['original_preserved']
        assert store.get((result['request_id'], item['id'] + '-emboss'))


def test_comparison_ocr_failure_preserves_original():
    from app.services.experimental_ocr import green_alternative
    class FailingReader:
        def read(self, crop):
            raise RuntimeError('unavailable')
    _, metadata = green_alternative(FailingReader(), np.full((24, 80, 3), (30, 110, 0), np.uint8), ('MH12AB1234', .82))
    assert metadata['candidate_status'] == 'unavailable'
    assert metadata['original_text'] == 'MH12AB1234'
    assert metadata['original_ocr_confidence'] == .82


def test_weak_unreadable_green_photo_is_retained_with_optional_review_evidence():
    class WeakDetector:
        def detect_fast(self,image):
            return [(10,10,80,24,.4)]
    class WeakReader:
        def __init__(self):self.alternative_calls=0
        def read_many(self,pairs):return [('BLUR',.2)]*len(pairs)
        def read(self,crop):
            self.alternative_calls+=1
            return 'KA25AB1234',.99
    reader=WeakReader();store=CropStore()
    original=process(photo(),WeakDetector(),store,reader,profile='fast')
    assert len(original['detections'])==1 and reader.alternative_calls==0
    assert original['detections'][0]['format_status']=='uncertain'
    result=process(photo(),WeakDetector(),store,reader,profile='fast',green_filter=True)
    assert len(result['detections'])==1 and reader.alternative_calls==1
    item=result['detections'][0]
    assert item['experimental_review_only'] and item['format_status']=='uncertain'
    assert item['raw_text']=='BLUR' and item['normalized_text']=='BLUR' and item['ocr_confidence']==.2
    assert item['green_enhancement']['candidate_text']=='KA25AB1234'
    assert store.get((result['request_id'],item['id']))
    assert store.get((result['request_id'],item['id']+'-emboss'))


def test_requested_review_keeps_non_green_regions_without_inventing_alternatives():
    class WeakDetector:
        def detect_fast(self,image):return [(10,10,80,24,.4)]
    class WeakReader:
        def read_many(self,pairs):return [('BLUR',.2)]*len(pairs)
        def read(self,crop):raise AssertionError('Non-green crops must not get an alternative')
    data=io.BytesIO();Image.new('RGB',(120,50),(230,230,230)).save(data,format='PNG')
    result=process(data.getvalue(),WeakDetector(),CropStore(),WeakReader(),profile='fast',green_filter=True)
    assert len(result['detections'])==1
    assert result['detections'][0]['format_status']=='uncertain'


def test_weak_video_alternative_is_visible_as_candidate_and_preserves_original(tmp_path,monkeypatch):
    import app.services.video as video
    frame=np.full((60,120,3),(30,110,0),np.uint8)
    class WeakReader:
        def read_plate(self,*args):return 'BLUR',.2
        def read(self,crop):return 'KA25AB1234',.99
    def read(opt_in):
        tracker=PlateTracks(30)
        observations=tracker.update([(10,10,80,24,.4)],0,frame)
        job=VideoJob('review',tmp_path,tmp_path/'input.mp4',profile='detailed')
        job.update(green_filter=opt_in)
        read_tracks(tracker.tracks,WeakReader(),job)
        return tracker.tracks,observations
    tracks,observations=read(False)
    assert video.visible(observations[0],tracks[0])
    tracks,observations=read(True)
    track=tracks[0]
    assert video.visible(observations[0],track)
    assert track['experimental_review_only'] and track['format_status']=='uncertain'
    assert track['text']=='BLUR' and track['confidence']==.2
    assert track['green_enhancement']['candidate_text']=='KA25AB1234'
    drawn=[]
    monkeypatch.setattr(video.cv2,'putText',lambda image,text,*args,**kwargs:drawn.append(text))
    video.annotate(frame.copy(),observations,tracks)
    assert len(drawn)==1 and 'Candidate' in drawn[0] and 'BLUR' in drawn[0]
    assert 'KA25AB1234' not in drawn[0]


def test_confident_video_observation_does_not_become_review_only(tmp_path):
    frame=np.full((60,120,3),(30,110,0),np.uint8)
    tracker=PlateTracks(30)
    tracker.update([(10,10,80,24,.9)],0,frame)
    tracker.update([(10,10,80,24,.4)],1,frame)
    class WeakReader:
        def read_plate(self,*args):return 'BLUR',.2
        def read(self,crop):return 'KA25AB1234',.99
    job=VideoJob('review',tmp_path,tmp_path/'input.mp4',profile='detailed')
    job.update(green_filter=True)
    read_tracks(tracker.tracks,WeakReader(),job)
    assert not tracker.tracks[0].get('experimental_review_only')
    assert tracker.tracks[0]['max_detection_confidence']==.9


def test_video_comparison_uses_chosen_observation_and_preserves_vote(tmp_path, monkeypatch):
    import app.services.experimental_ocr as experiment
    first = np.full((24, 80, 3), 10, np.uint8)
    second = np.full((24, 80, 3), 20, np.uint8)
    tracks = [dict(crops=[(100, first, first), (90, second, second)])]
    class VideoReader:
        def __init__(self):
            self.readings = iter([('BLUR', .2), ('MH12AB1234', .95)])
        def read_plate(self, *args):
            return next(self.readings)
    def compare(ocr, crop, original):
        assert np.array_equal(crop, second)
        assert original == ('MH12AB1234', .95)
        return None, {'candidate_text': 'WRONG123', 'candidate_ocr_confidence': .99}
    monkeypatch.setattr(experiment, 'green_alternative', compare)
    job = VideoJob('test', tmp_path, tmp_path/'input.mp4', profile='detailed')
    job.update(green_filter=True)
    read_tracks(tracks, VideoReader(), job)
    assert tracks[0]['text'] == 'MH12AB1234'
    assert tracks[0]['confidence'] == .95
    assert tracks[0]['green_enhancement']['candidate_text'] == 'WRONG123'


def test_api_passes_green_filter_explicitly(monkeypatch):
    import app.main as main
    from fastapi.testclient import TestClient
    calls = []
    def fake_process(*args):
        calls.append(args[6])
        return {'detections': [], 'processing_time_ms': 1}
    monkeypatch.setattr(main, 'detector', object())
    monkeypatch.setattr(main, 'ocr', object())
    monkeypatch.setattr(main, 'process', fake_process)
    client = TestClient(main.app)
    for data in ({}, {'green_filter': 'true'}):
        response = client.post('/api/detect/image', data=data,
                               files={'image': ('frame.png', photo(), 'image/png')})
        assert response.status_code == 200
    assert calls == [False, True]
