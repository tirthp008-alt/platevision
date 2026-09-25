from threading import Thread
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.services.video import PlateTracks, VideoJob, visible, process_video, annotate


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


def test_weak_candidate_needs_plate_text_evidence():
    observation={'score':.3,'track_id':1,'box':[10,20,70,20]}
    track={'text':'MOTOR','confidence':.99,'format_status':'uncertain','id':1}
    assert not visible(observation,track)
    image=np.zeros((100,100,3),np.uint8)
    assert not annotate(image.copy(),[observation],[track]).any()
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
