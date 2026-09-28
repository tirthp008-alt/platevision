from threading import Thread
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.services.video import PlateTracks, VideoJob, visible, process_video, annotate
from app.services.video import detect_video_frame,video_search_strategy


def test_fast_video_searches_all_frames_with_whole_scene_gpu_model():
    class Detector:
        fast_models={'landscape':object(),'portrait':object()}
        calls=0
        def detect_fast(self,image):
            self.calls+=1
            return [(10,20,70,20,.9),(150,20,70,20,.8)]
        def detect_plates(self,*args,**kwargs):
            raise AssertionError('Fast GPU frames must not repeat tiled inference')
    detector=Detector()
    for _ in range(10):
        assert len(detect_video_frame(detector,np.zeros((108,192,3),np.uint8),'fast'))==2
    assert detector.calls==10
    assert video_search_strategy(detector,'fast')=='high_resolution_whole_frame'


def test_detailed_video_and_unsupported_models_retain_tiled_search():
    class Detector:
        fast_models=None
        def detect_fast(self,image):raise AssertionError('Unavailable GPU graph')
        def detect_plates(self,image,refine):return [('tiled',refine)]
    detector=Detector();image=np.zeros((10,10,3),np.uint8)
    assert detect_video_frame(detector,image,'fast')==[('tiled',False)]
    detector.fast_models={'landscape':object()}
    assert detect_video_frame(detector,image,'detailed')==[('tiled',True)]


def test_model_restart_retains_only_unexpired_completed_downloads(tmp_path,monkeypatch):
    import json
    import os
    from app.core.config import settings
    from app.services.video import restore_completed_jobs
    monkeypatch.setattr(settings,'video_output_dir',str(tmp_path))
    for ident in ('a'*32,'b'*32,'c'*32,'not-a-job'):
        folder=tmp_path/ident;folder.mkdir()
        (folder/'results.json').write_text(json.dumps({'frames':90,'profile':'fast','tracks':[]}))
        if ident!='c'*32:(folder/'annotated.mp4').write_bytes(b'completed video')
    expired=tmp_path/('b'*32)/'results.json'
    old=time.time()-settings.video_result_ttl_seconds-1;os.utime(expired,(old,old))
    restored=restore_completed_jobs()
    assert list(restored)==['a'*32]
    assert restored['a'*32].snapshot()['status']=='complete'
    assert restored['a'*32].snapshot()['video_url']==f"/api/videos/{'a'*32}/video"


def make_job(tmp_path):
    source=tmp_path/'input.mp4';source.write_bytes(b'invalid')
    return VideoJob('test',tmp_path,source)


def test_multiple_tracks_keep_identity_when_confidence_order_changes():
    tracker=PlateTracks(30);image=np.zeros((100,300,3),np.uint8)
    a=(10,20,70,20,.9);b=(150,20,70,20,.8)
    first=tracker.update([a,b],0,image)
    second=tracker.update([(152,20,70,20,.95),(12,20,70,20,.8)],1,image)
    assert [x['track_id'] for x in first]==[1,2]
    assert [x['track_id'] for x in second]==[2,1]
    assert len(tracker.tracks)==2


def test_departed_plate_does_not_reuse_old_track():
    tracker=PlateTracks(30);image=np.zeros((100,300,3),np.uint8)
    tracker.update([(10,20,70,20,.9)],0,image)
    assert tracker.update([(10,20,70,20,.9)],30,image)[0]['track_id']==2


def test_plate_regions_remain_visible_even_when_ocr_has_no_plate_text_evidence():
    observation={'score':.3,'track_id':1,'box':[10,20,70,20]}
    track={'text':'MOTOR','confidence':.99,'format_status':'uncertain','id':1}
    assert visible(observation,track)
    image=np.zeros((100,100,3),np.uint8)
    assert annotate(image.copy(),[observation],[track]).any()
    track['text']='MH12AB1234'
    assert visible(observation,track)
    assert annotate(image.copy(),[observation],[track]).any()


def test_pause_resume_and_cancel_are_cooperative(tmp_path):
    job=make_job(tmp_path);job.resume.clear();finished=[]
    thread=Thread(target=lambda:(job.checkpoint(),finished.append(True)))
    thread.start();time.sleep(.03)
    assert not finished
    job.resume.set();thread.join(1)
    assert finished and job.paused_seconds>=.02
    job.cancel.set()
    with pytest.raises(InterruptedError):job.checkpoint()


def test_bad_recording_fails_visibly_and_removes_uploaded_source(tmp_path):
    job=make_job(tmp_path)
    process_video(job,object(),None)
    assert job.snapshot()['status']=='failed'
    assert 'decode' in job.snapshot()['error'].lower()
    assert not job.source.exists()


def test_video_api_blocks_conflicting_uploads_and_unready_downloads(tmp_path,monkeypatch):
    import app.main as main
    job=make_job(tmp_path);job.update(status='processing')
    monkeypatch.setattr(main,'video_jobs',{'test':job})
    monkeypatch.setattr(main,'detector',object())
    client=TestClient(main.app)
    assert client.get('/api/videos/test/video').status_code==409
    assert client.get('/api/videos/missing').status_code==404
    assert client.post('/api/videos',files={'video':('x.mp4',b'video','video/mp4')}).status_code==429
    assert client.post('/api/videos',files={'video':('x.txt',b'text','text/plain')}).status_code==415
    assert client.post('/api/videos/test/pause').json()['status']=='paused'
    assert not job.resume.is_set()
    assert client.post('/api/videos/test/resume').json()['status']=='processing'
    assert job.resume.is_set()
    client.post('/api/videos/test/cancel')
    assert job.cancel.is_set()


def test_complete_video_supports_browser_seeking_and_attachment(tmp_path,monkeypatch):
    import app.main as main
    job=make_job(tmp_path);job.update(status='complete')
    content=bytes(range(256))*4;(tmp_path/'annotated.mp4').write_bytes(content)
    monkeypatch.setattr(main,'video_jobs',{'test':job})
    client=TestClient(main.app)
    response=client.get('/api/videos/test/video',headers={'Range':'bytes=10-29'})
    assert response.status_code==206 and response.content==content[10:30]
    response=client.get('/api/videos/test/video?download=true')
    assert 'attachment' in response.headers['content-disposition']
    assert response.content==content
