"""Cross-camera prototype API contracts without model startup or inference."""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.services.camera_matches import CameraMatchGallery
from app.services.video import VideoJob


def photo():
    ok,data=cv2.imencode('.png',np.zeros((80,120,3),np.uint8))
    assert ok
    return {'image':('frame.png',data.tobytes(),'image/png')}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main,'camera_gallery',CameraMatchGallery())
    monkeypatch.setattr(main,'reid_engine',SimpleNamespace(model_id='Fake verified Re-ID',runtime='CPU test'))
    monkeypatch.setattr(main,'reid_startup_error',None)
    monkeypatch.setattr(main,'detector',object())
    monkeypatch.setattr(main,'vehicle_detector',object())
    monkeypatch.setattr(main,'video_jobs',{})
    monkeypatch.setattr(main,'video_tasks',set())
    monkeypatch.setattr(main,'inference_lock',asyncio.Lock())
    monkeypatch.setattr(main,'video_upload_lock',asyncio.Lock())
    return TestClient(main.app)


def test_photo_opt_in_passes_utc_context_and_default_does_no_reid(client,monkeypatch):
    calls=[]
    def process(*args):
        calls.append(args)
        return dict(detections=[{'id':'plate-1'}],plate_count=1,processing_time_ms=1)
    monkeypatch.setattr(main,'process',process)
    off=client.post('/api/detect/image',data={'camera_id':'ignored','observed_at':'invalid'},files=photo())
    assert off.status_code==200 and len(calls[-1])==9
    enabled=client.post('/api/detect/image',data=dict(reidentify='true',camera_id=' Camera A ',
                         observed_at='2026-09-27T10:30:00+05:30'),files=photo())
    assert enabled.status_code==200,enabled.text
    assert len(calls[-1])==10
    context=calls[-1][-1]
    assert context['camera_id']=='Camera A'
    assert context['observed_at']==datetime(2026,9,27,5,tzinfo=timezone.utc).timestamp()
    assert context['engine'] is main.reid_engine and context['gallery'] is main.camera_gallery
    assert enabled.json()['plate_count']==1


@pytest.mark.parametrize('camera,observed',[('', '2026-09-27T10:00:00Z'),('A'*101,'2026-09-27T10:00:00Z'),
    ('camera\nA','2026-09-27T10:00:00Z'),('A',''),('A','2026-09-27T10:00:00'),
    ('A','not-a-time'),('A','1960-01-01T00:00:00Z')])
def test_enabled_reid_rejects_invalid_camera_or_unzoned_timestamp(client,monkeypatch,camera,observed):
    def forbidden(*args):
        raise AssertionError('Invalid match metadata must be rejected before inference')
    monkeypatch.setattr(main,'process',forbidden)
    response=client.post('/api/detect/image',data=dict(reidentify='true',camera_id=camera,
                        observed_at=observed),files=photo())
    assert response.status_code==422


def test_missing_optional_reid_model_does_not_reject_plate_request(client,monkeypatch):
    monkeypatch.setattr(main,'reid_engine',None)
    monkeypatch.setattr(main,'reid_startup_error','Model not installed')
    def process(*args):
        assert args[-1]['engine'] is None
        return dict(detections=[{'id':'plate-1'}],plate_count=1,processing_time_ms=1,
                    reidentification={'enabled':True,'status':'unavailable'})
    monkeypatch.setattr(main,'process',process)
    response=client.post('/api/detect/image',data=dict(reidentify='true',camera_id='A',
                        observed_at='2026-09-27T10:00:00Z'),files=photo())
    assert response.status_code==200 and response.json()['plate_count']==1
    health=client.get('/api/reid/health').json()
    assert health['ready'] is False and health['error']=='Model not installed'
    assert health['gallery_size']==0


def test_live_frame_skips_reid_without_metadata_or_extra_work(client,monkeypatch):
    def process(*args):
        assert len(args)==9
        return dict(detections=[{'id':'plate-1'}],plate_count=1,processing_time_ms=1)
    monkeypatch.setattr(main,'process',process)
    response=client.post('/api/detect/frame',data={'reidentify':'true'},files=photo())
    assert response.status_code==200
    assert response.json()['reidentification']['status']=='unsupported_for_live_frames'
    assert response.json()['plate_count']==1 and response.json()['warnings']
    assert main.camera_gallery.stats()['observations']==0


def test_routes_are_explicit_directed_and_invalid_updates_are_atomic(client):
    assert client.get('/api/reid/routes').json()=={'routes':[]}
    route=dict(source_camera='A',target_camera='B',min_seconds=30,max_seconds=180)
    assert client.put('/api/reid/routes',json={'routes':[route]}).status_code==200
    assert client.get('/api/reid/routes').json()['routes']==[route]
    invalid=[dict(route,target_camera='A'),dict(route,min_seconds=-1),dict(route,min_seconds=190),dict(route,min_seconds=True),
             dict(route,target_camera='\nB')]
    for item in invalid:
        assert client.put('/api/reid/routes',json={'routes':[item]}).status_code==422
        assert client.get('/api/reid/routes').json()['routes']==[route]
    assert client.put('/api/reid/routes',json={'routes':[route,route]}).status_code==422
    assert client.get('/api/reid/routes').json()['routes']==[route]


def test_match_history_has_no_descriptors_and_clear_erases_observations_not_routes(client):
    route=dict(source_camera='A',target_camera='B',min_seconds=30,max_seconds=180)
    client.put('/api/reid/routes',json={'routes':[route]})
    descriptor=np.zeros(2048,np.float32);descriptor[0]=1
    for camera,timestamp in [('A',1000),('B',1100)]:
        main.camera_gallery.register_batch([dict(camera_id=camera,track_id='local-track-1',
            observed_at=timestamp,descriptor=descriptor,category='car')])
    health=client.get('/api/reid/health').json()
    assert health['ready'] is True and health['gallery_size']==2
    response=client.get('/api/reid/matches')
    assert response.status_code==200
    assert len(response.json()['observations'])==2
    assert response.json()['observations'][0]['status']=='candidate'
    assert 'descriptor' not in response.text
    cleared=client.delete('/api/reid/gallery').json()
    assert cleared==dict(cleared=True,removed_observations=2,gallery_size=0)
    assert client.get('/api/reid/matches').json()=={'observations':[]}
    assert client.get('/api/reid/routes').json()['routes']==[route]


def test_routes_and_gallery_mutations_allow_frontend_cors(client):
    for method in ('PUT','DELETE'):
        response=client.options('/api/reid/routes',headers={'Origin':'http://127.0.0.1:8080',
                                'Access-Control-Request-Method':method})
        assert response.status_code==200
        assert method in response.headers['access-control-allow-methods']


def test_recording_context_is_not_serialized_and_worker_receives_it_only_when_enabled(client,monkeypatch,tmp_path):
    monkeypatch.setattr(main.settings,'video_output_dir',str(tmp_path))
    captured=[]
    async def capture_job(job,reader):
        captured.append(job)
    monkeypatch.setattr(main,'run_video',capture_job)
    response=client.post('/api/videos',data=dict(reidentify='true',camera_id='A',
                         observed_at='2026-09-27T10:30:00+05:30'),
                         files={'video':('clip.mp4',b'fixture bytes','video/mp4')})
    assert response.status_code==202,response.text
    state=response.json()
    assert state['camera_id']=='A' and isinstance(state['observed_at'],float)
    assert 'gallery' not in state and 'engine' not in state
    assert captured and captured[0].snapshot()['reidentify'] is True


def test_video_worker_appends_fifth_context_only_when_requested(client,monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(main,'process_video',lambda *args:calls.append(args))
    plain=VideoJob('plain',tmp_path,tmp_path/'plain.mp4')
    asyncio.run(main.run_video(plain,None))
    assert len(calls[-1])==4
    linked=VideoJob('linked',tmp_path,tmp_path/'linked.mp4')
    linked.update(reidentify=True,camera_id='B',observed_at=1234.)
    asyncio.run(main.run_video(linked,None))
    assert len(calls[-1])==5
    assert calls[-1][-1]==dict(engine=main.reid_engine,gallery=main.camera_gallery,
                              camera_id='B',observed_at=1234.)


@pytest.mark.parametrize('present',[False,True])
def test_optional_reid_startup_failure_keeps_plate_service_available(client,monkeypatch,tmp_path,present):
    plate=object();vehicle=object()
    monkeypatch.setattr(main,'get_detector',lambda:plate)
    monkeypatch.setattr(main,'get_vehicle_detector',lambda:vehicle)
    monkeypatch.setattr(main,'PlateOCR',lambda:object())
    monkeypatch.setattr(main,'restore_completed_jobs',lambda:{})
    monkeypatch.setattr(main.settings,'resnet_ocr_model_path',str(tmp_path/'no-resnet'))
    path=tmp_path/'optional-reid.onnx'
    if present:path.write_bytes(b'fixture model')
    monkeypatch.setattr(main.settings,'vehicle_reid_model_path',str(path))
    calls=[]
    def broken(*args):
        calls.append(args)
        raise RuntimeError('Optional model unavailable')
    monkeypatch.setattr(main,'VehicleReID',broken)
    async def start():
        async with main.lifespan(main.app):
            assert main.detector is plate and main.vehicle_detector is vehicle
            assert main.reid_engine is None and main.reid_startup_error
    asyncio.run(start())
    assert bool(calls)==present
