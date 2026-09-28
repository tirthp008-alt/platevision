import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.services.ocr import PlateOCR
from app.services.resnet_ocr import ResNetPlateRecognizer
from app.services.video import VideoJob, read_tracks


def track(number, observations=2):
    crop = np.full((16,80,3), number, np.uint8)
    return dict(crops=[(1,crop,crop)] * observations, text='', confidence=0, format_status='uncertain')


def test_video_batches_every_track_preserving_order_and_skips_unneeded_second_crops(tmp_path):
    class Reader:
        sizes = []
        def read_many(self,pairs):
            self.sizes.append(len(pairs))
            return [(f'MH12AB{int(tight[0,0,0]):04}',.97) for _,tight in pairs]
        def read_plate(self,*args):
            raise AssertionError('Confident batch results should not repeat CPU OCR')
    tracks = [track(i) for i in range(21)]
    reader = Reader()
    read_tracks(tracks,reader,VideoJob('test',tmp_path,tmp_path/'unused'))
    assert reader.sizes == [16,5]
    assert [t['text'] for t in tracks] == [f'MH12AB{i:04}' for i in range(21)]
    assert all(t['confidence']==.97 for t in tracks)


def test_video_keeps_localized_retries_for_uncertain_batch_text(tmp_path):
    class Reader:
        rank = staticmethod(PlateOCR.rank)
        def read_many(self,pairs):return [('BLUR',.2)]*len(pairs)
        def read_plate(self,context,tight):return ('MH12AB1234',.96)
    tracks = [track(1)]
    read_tracks(tracks,Reader(),VideoJob('test',tmp_path,tmp_path/'unused'))
    assert tracks[0]['text']=='MH12AB1234' and tracks[0]['confidence']==.96


def test_video_cancel_is_checked_between_batches(tmp_path):
    job = VideoJob('test',tmp_path,tmp_path/'unused')
    class Reader:
        def read_many(self,pairs):
            job.cancel.set()
            return [('MH12AB1234',.97)]*len(pairs)
    with pytest.raises(InterruptedError):
        read_tracks([track(i) for i in range(20)],Reader(),job)


def test_detailed_video_retains_full_reader(tmp_path):
    class Reader:
        def read_many(self,pairs):raise AssertionError('Detailed mode must use full OCR')
        def read_plate(self,*args):return ('MH12AB1234',.97)
    tracks = [track(1)]
    read_tracks(tracks,Reader(),VideoJob('test',tmp_path,tmp_path/'unused',profile='detailed'))
    assert tracks[0]['text']=='MH12AB1234'


def test_cpu_fallback_does_not_run_the_full_reader_twice(tmp_path):
    class Reader:
        batch = None
        calls = 0
        def read_many(self,pairs):raise AssertionError('No accelerated batch available')
        def read_plate(self,*args):
            self.calls += 1
            return ('MH12AB1234',.97)
    reader = Reader()
    read_tracks([track(1)],reader,VideoJob('test',tmp_path,tmp_path/'unused'))
    assert reader.calls==1


def test_stacked_crop_does_not_repeat_localization_after_read_many(tmp_path):
    class Reader:
        def read_many(self,pairs):return [('BLUR',.4)]*len(pairs)
        def read_plate(self,*args):raise AssertionError('Stacked crop already used the full reader')
    crop = np.zeros((40,60,3),np.uint8)
    tracks = [dict(crops=[(1,crop,crop)])]
    read_tracks(tracks,Reader(),VideoJob('test',tmp_path,tmp_path/'unused'))
    assert tracks[0]['format_status']=='uncertain'


def test_resnet_preprocessing_preserves_bgr_and_does_not_stretch_short_crops():
    crop = np.zeros((16,40,3),np.uint8);crop[:,:,2]=255
    tensor = ResNetPlateRecognizer.preprocess(crop)
    assert tensor.shape==(3,32,320) and tensor.dtype==np.float32
    assert np.all(tensor[2,:,:80]==1) and np.all(tensor[0,:,:80]==-1)
    assert np.all(tensor[:,:,80:]==0)
    with pytest.raises(ValueError):ResNetPlateRecognizer.preprocess(crop[:0])


def test_requesting_unavailable_resnet_never_silently_uses_default_reader(monkeypatch):
    import app.main as main
    monkeypatch.setattr(main,'ocr',object())
    monkeypatch.setattr(main,'resnet_ocr',None)
    client = TestClient(main.app)
    for model,code in [('resnet34',503),('unknown',422)]:
        assert client.post('/api/detect/image',data={'ocr_model':model,'read_text':'true'},
                           files={'image':('a.jpg',b'jpeg','image/jpeg')}).status_code==code
        assert client.post('/api/videos',data={'ocr_model':model,'read_text':'true'},
                           files={'video':('a.mp4',b'video','video/mp4')}).status_code==code


def test_image_request_uses_selected_resnet_and_reports_its_confidence(monkeypatch):
    import app.main as main
    class Detector:
        def detect_fast(self,image):return [(2,2,70,15,.91)]
    class Reader:
        runtime='ResNet34 test'
        def read_many(self,pairs):return [('MH12AB1234',.83)]*len(pairs)
    monkeypatch.setattr(main,'detector',Detector())
    monkeypatch.setattr(main,'resnet_ocr',Reader())
    monkeypatch.setattr(main,'vehicle_detector',None)
    data=io.BytesIO();Image.new('RGB',(100,50)).save(data,format='JPEG')
    response=TestClient(main.app).post('/api/detect/image',data={'ocr_model':'resnet34','read_text':'true'},
                                     files={'image':('a.jpg',data.getvalue(),'image/jpeg')})
    assert response.status_code==200,response.text
    result=response.json()
    assert result['ocr_model']=='resnet34' and result['ocr_engine']=='ResNet34 test'
    assert result['detections'][0]['ocr_confidence']==.83
