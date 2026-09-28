import asyncio
import logging
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from app.core.config import settings
from app.services.detector import get_detector, get_vehicle_detector
from app.services.ocr import PlateOCR
from app.services.pipeline import process, CropStore, InvalidImage
from app.services.video import new_job, process_video, restore_completed_jobs
from app.services.vehicle_reid import VehicleReID
from app.services.camera_matches import CameraMatchGallery

logger=logging.getLogger(__name__)
detector=None
ocr=None
resnet_ocr=None
resnet_ocr_error=None
vehicle_detector=None
reid_engine=None
reid_startup_error=None
camera_gallery=CameraMatchGallery()
startup_errors=[]
crops=CropStore(settings.result_ttl_seconds,settings.max_cached_crops)
inference_lock=asyncio.Lock()
video_jobs={}
video_tasks=set()
video_upload_lock=asyncio.Lock()

@asynccontextmanager
async def lifespan(app):
    global detector,ocr,vehicle_detector,resnet_ocr,resnet_ocr_error,reid_engine,reid_startup_error
    startup_errors.clear()
    video_jobs.update(restore_completed_jobs())
    for name, factory in [('plate',get_detector),('ocr',PlateOCR),('vehicle',get_vehicle_detector)]:
        try:
            value=await run_in_threadpool(factory)
            if name=='plate':detector=value
            elif name=='ocr':ocr=value
            else:vehicle_detector=value
        except Exception as error:
            startup_errors.append(f'{name}: {error}')
            logger.warning('Engine startup: %s',error)
    resnet_ocr=None
    resnet_ocr_error=None
    if Path(settings.resnet_ocr_model_path).is_file():
        try:
            resnet_ocr=await run_in_threadpool(PlateOCR,'resnet34')
        except Exception as error:
            resnet_ocr_error=str(error)
            logger.warning('Optional ResNet OCR unavailable: %s',error)
    reid_engine=None
    reid_startup_error=None
    if Path(settings.vehicle_reid_model_path).is_file():
        try:
            reid_engine=await run_in_threadpool(VehicleReID,settings.vehicle_reid_model_path,settings.openvino_device)
        except Exception as error:
            reid_startup_error=str(error)
            logger.warning('Optional vehicle Re-ID unavailable; plate capture remains available: %s',error)
    else:
        reid_startup_error='Vehicle Re-ID model is not installed.'
    yield
    for job in video_jobs.values():
        job.cancel.set(); job.resume.set()
    if video_tasks:
        await asyncio.gather(*video_tasks,return_exceptions=True)

app=FastAPI(title='PlateSight API',version='0.4.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=settings.cors_origins.split(','),allow_methods=['GET','POST','PUT','DELETE'],allow_headers=['Content-Type'])

REID_NOTE='Prototype appearance matching with camera-route and time constraints. Similarity is not a verified identity or calibrated probability.'


def camera_identifier(value):
    if not isinstance(value,str) or not value.strip() or len(value)>100 or any(ord(character)<32 for character in value):
        raise ValueError('Camera ID must be a nonempty string of at most 100 characters without control characters.')
    return value.strip()


class CameraRoute(BaseModel):
    model_config=ConfigDict(extra='forbid')
    source_camera:str=Field(max_length=100)
    target_camera:str=Field(max_length=100)
    min_seconds:float=Field(ge=0,allow_inf_nan=False,strict=True)
    max_seconds:float=Field(ge=0,allow_inf_nan=False,strict=True)

    @field_validator('source_camera','target_camera',mode='before')
    @classmethod
    def check_camera(cls,value):
        return camera_identifier(value)

    @model_validator(mode='after')
    def check_route(self):
        if self.source_camera==self.target_camera:
            raise ValueError('A cross-camera route must connect different cameras.')
        if self.max_seconds<self.min_seconds:
            raise ValueError('max_seconds must be greater than or equal to min_seconds.')
        return self


class CameraRoutes(BaseModel):
    model_config=ConfigDict(extra='forbid')
    routes:list[CameraRoute]=Field(default_factory=list,max_length=1000)


@app.get('/api/reid/health')
def reid_health():
    return dict(ready=reid_engine is not None,model=getattr(reid_engine,'model_id',None),
                engine=getattr(reid_engine,'runtime','unavailable'),note=REID_NOTE,
                gallery_size=camera_gallery.stats()['observations'],error=reid_startup_error)


@app.get('/api/reid/routes')
def get_reid_routes():
    return dict(routes=camera_gallery.get_routes())


@app.put('/api/reid/routes')
def update_reid_routes(payload:CameraRoutes):
    routes=[route.model_dump() for route in payload.routes]
    try:
        camera_gallery.set_routes(routes)
    except ValueError as error:
        raise HTTPException(422,str(error)) from error
    return dict(routes=camera_gallery.get_routes())


@app.get('/api/reid/matches')
def reid_matches():
    return dict(observations=camera_gallery.recent_matches())


@app.delete('/api/reid/gallery')
def clear_reid_gallery():
    cleared=camera_gallery.clear()
    return dict(cleared=True,removed_observations=cleared['cleared'],gallery_size=camera_gallery.stats()['observations'])


def reid_context(camera_id,observed_at):
    try:
        camera_id=camera_identifier(camera_id)
    except ValueError as error:
        raise HTTPException(422,str(error)) from error
    try:
        observed=datetime.fromisoformat(observed_at.strip().replace('Z','+00:00'))
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise ValueError('Timezone is required')
        timestamp=observed.astimezone(timezone.utc).timestamp()
        if not math.isfinite(timestamp) or timestamp<0:
            raise ValueError('Invalid timestamp')
    except (ValueError,TypeError,AttributeError,OverflowError) as error:
        raise HTTPException(422,'Re-ID requires observed_at as a timezone-aware ISO timestamp, for example 2026-09-27T10:30:00+05:30.') from error
    return dict(engine=reid_engine,gallery=camera_gallery,camera_id=camera_id,observed_at=timestamp)

@app.get('/api/health')
def health():
    return dict(status='ok' if detector and (vehicle_detector or not settings.vehicle_context_enabled) else 'degraded',detector_ready=detector is not None,ocr_ready=ocr is not None,vehicle_detector_ready=vehicle_detector is not None,vehicle_engine=getattr(vehicle_detector,'runtime','unavailable'),vehicle_taxonomy=getattr(vehicle_detector,'taxonomy',None),default_read_text=True,region_confidence_threshold=settings.region_confidence_threshold,region_supplement_ready=bool(getattr(detector,'region_supplement',None)),mode=settings.detector_mode,engine=getattr(detector,'runtime','unavailable'),ocr_engine=getattr(ocr,'runtime','unavailable'),ocr_models={'ppocr':getattr(ocr,'runtime',None),'resnet34':getattr(resnet_ocr,'runtime',None)},resnet_ocr_error=resnet_ocr_error,detector_model=getattr(detector,'model_id','unavailable'),weights_retrained=bool(getattr(detector,'weights_retrained',False)),model_sha256=getattr(detector,'model_sha256',None),fast_video_ready=bool(getattr(detector,'fast_models',None)),fast_image_ready=bool(getattr(detector,'fast_models',None)),video_export_ready=detector is not None,errors=startup_errors,version='0.4.0')

def selected_ocr(model):
    if model == 'ppocr': return ocr
    if model != 'resnet34': raise HTTPException(422,'Unknown OCR model.')
    if resnet_ocr is None:
        raise HTTPException(503,'ResNet OCR is unavailable. Run scripts/setup_resnet_ocr.py and restart the API; see /api/health.')
    return resnet_ocr


async def detect(image,profile="fast",ocr_model='ppocr',green_filter=False,read_text=True,angle_correction=True,reidentify=False,camera_id='',observed_at=''):
    matching=reid_context(camera_id,observed_at) if reidentify else None
    reader=selected_ocr(ocr_model) if read_text else None
    if profile not in {"fast","detailed"}:raise HTTPException(422,"Unknown image profile.")
    if image.content_type not in {'image/jpeg','image/png','image/webp'}:raise HTTPException(415,'Upload JPG, PNG or WEBP.')
    data=await image.read(settings.max_upload_mb*1024*1024+1)
    if len(data)>settings.max_upload_mb*1024*1024:raise HTTPException(413,'Image exceeds size limit.')
    if detector is None:raise HTTPException(503,'Plate model unavailable. Configure MODEL_PATH; see /api/health.')
    if inference_lock.locked():raise HTTPException(429,'Inference is busy. Send the next available frame; do not queue frames.')
    async with inference_lock:
        try:
            arguments=(data,detector,crops,reader,vehicle_detector,profile,green_filter,read_text,angle_correction)
            result=await run_in_threadpool(process,*arguments,matching) if reidentify else await run_in_threadpool(process,*arguments)
            result['ocr_engine']=getattr(reader,'runtime','unavailable') if read_text else 'not_requested'
            result['ocr_model']=ocr_model if read_text else 'not_requested'
            return result
        except InvalidImage as error:raise HTTPException(422,str(error)) from error
        except Exception as error:
            logger.exception('Inference failed')
            raise HTTPException(500,'Inference failed. Check server logs and model compatibility.') from error

@app.post('/api/detect/image')
async def detect_image(image:UploadFile=File(...),profile:str=Form("fast"),ocr_model:str=Form('ppocr'),green_filter:bool=Form(False),read_text:bool=Form(True),angle_correction:bool=Form(True),reidentify:bool=Form(False),camera_id:str=Form(''),observed_at:str=Form('')):
    return await detect(image,profile,ocr_model,green_filter,read_text,angle_correction,reidentify,camera_id,observed_at)
@app.post('/api/detect/frame')
async def detect_frame(image:UploadFile=File(...),profile:str=Form("fast"),ocr_model:str=Form('ppocr'),green_filter:bool=Form(False),read_text:bool=Form(True),angle_correction:bool=Form(True),reidentify:bool=Form(False),camera_id:str=Form(''),observed_at:str=Form('')):
    result=await detect(image,profile,ocr_model,green_filter,read_text,angle_correction)
    if reidentify:
        result.setdefault('warnings',[]).append('Cross-camera Re-ID is available for photos and uploaded recordings; live frame capture skips it.')
        result['reidentification']=dict(enabled=False,status='unsupported_for_live_frames',note=REID_NOTE)
    return result
@app.get('/api/results/{request_id}/{detection_id}/crop')
def crop(request_id:str,detection_id:str):
    item=crops.get((request_id,detection_id))
    if item is None:raise HTTPException(404,'Crop expired or unavailable.')
    return Response(item,media_type='image/jpeg',headers={'Cache-Control':'no-store'})


def clean_video_jobs():
    root=Path(settings.video_output_dir).resolve()
    for key,job in list(video_jobs.items()):
        if job.snapshot()['status'] in {'complete','failed','cancelled'} and time.monotonic()-job.created>settings.video_result_ttl_seconds:
            folder=job.folder.resolve()
            if folder.parent==root and folder.name==job.id:
                shutil.rmtree(folder,ignore_errors=True)
            del video_jobs[key]


def video_job(job_id):
    clean_video_jobs()
    job=video_jobs.get(job_id)
    if job is None:
        raise HTTPException(404,'Recording expired or unavailable. Upload it again.')
    return job


async def run_video(job,reader):
    async with inference_lock:
        state=job.snapshot()
        if state.get('reidentify'):
            matching=dict(engine=reid_engine,gallery=camera_gallery,camera_id=state['camera_id'],observed_at=state['observed_at'])
            await run_in_threadpool(process_video,job,detector,reader,vehicle_detector,matching)
        else:
            await run_in_threadpool(process_video,job,detector,reader,vehicle_detector)


@app.post('/api/videos',status_code=202)
async def upload_video(video:UploadFile=File(...),profile:str=Form('fast'),ocr_model:str=Form('ppocr'),green_filter:bool=Form(False),read_text:bool=Form(True),angle_correction:bool=Form(True),reidentify:bool=Form(False),camera_id:str=Form(''),observed_at:str=Form('')):
    matching=reid_context(camera_id,observed_at) if reidentify else None
    reader=selected_ocr(ocr_model) if read_text else None
    if detector is None:
        raise HTTPException(503,'Plate detector is unavailable.')
    suffix=Path(video.filename or '').suffix.lower()
    if suffix not in {'.mp4','.webm','.mov','.m4v','.avi','.mkv'}:
        raise HTTPException(415,'Upload MP4, WebM, MOV, M4V, AVI or MKV.')
    if profile not in {'fast','detailed'}:
        raise HTTPException(422,'Unknown video profile.')
    if video_upload_lock.locked():
        raise HTTPException(429,'Another recording is uploading.')
    async with video_upload_lock:
        clean_video_jobs()
        if any(j.snapshot()['status'] in {'queued','processing','paused'} for j in video_jobs.values()):
            raise HTTPException(429,'Finish or cancel the current recording first.')
        job=new_job(suffix,profile)
        try:
            size=0
            with job.source.open('wb') as target:
                while chunk:=await video.read(1024*1024):
                    size+=len(chunk)
                    if size>settings.video_max_upload_mb*1024*1024:
                        raise HTTPException(413,f'Video exceeds {settings.video_max_upload_mb} MB.')
                    target.write(chunk)
            if not size:
                raise HTTPException(422,'Video is empty.')
        except BaseException:
            job.source.unlink(missing_ok=True)
            job.folder.rmdir()
            raise
        finally:
            await video.close()
        video_jobs[job.id]=job
        job.update(ocr_model=ocr_model if read_text else 'not_requested',green_filter=green_filter and read_text,read_text=read_text,angle_correction=angle_correction)
        if matching:
            job.update(reidentify=True,camera_id=matching['camera_id'],observed_at=matching['observed_at'])
        task=asyncio.create_task(run_video(job,reader));video_tasks.add(task)
        task.add_done_callback(video_tasks.discard)
        return job.snapshot()


@app.get('/api/videos/{job_id}')
def video_status(job_id:str):
    return video_job(job_id).snapshot()


@app.post('/api/videos/{job_id}/{action}')
def control_video(job_id:str,action:str):
    job=video_job(job_id)
    state=job.snapshot()['status']
    if action not in {'pause','resume','cancel'}:
        raise HTTPException(404,'Unknown recording action.')
    if state in {'complete','failed','cancelled'}:
        return job.snapshot()
    if action=='cancel':
        job.cancel.set();job.resume.set()
    elif action=='pause':
        job.resume.clear();job.update(status='paused')
    else:
        job.resume.set();job.update(status='processing')
    return job.snapshot()


@app.get('/api/videos/{job_id}/video')
def annotated_video(job_id:str,download:bool=False):
    job=video_job(job_id)
    if job.snapshot()['status']!='complete':
        raise HTTPException(409,'Video export is not ready.')
    return FileResponse(job.folder/'annotated.mp4',media_type='video/mp4',
                        filename='platesight-annotated.mp4' if download else None,
                        headers={'Cache-Control':'private, max-age=3600'})


@app.get('/api/videos/{job_id}/report')
def video_report(job_id:str):
    job=video_job(job_id)
    if job.snapshot()['status']!='complete':
        raise HTTPException(409,'Report is not ready.')
    return FileResponse(job.folder/'results.json',media_type='application/json',filename='platesight-results.json')


@app.get('/api/videos/{job_id}/crops/{track_id}')
def video_crop(job_id:str,track_id:int,variant:str='original'):
    if variant not in {'original','rectified'}:
        raise HTTPException(422,'Unknown crop variant.')
    job=video_job(job_id)
    path=job.folder/f'plate-{track_id}{"-rectified" if variant=="rectified" else ""}.jpg'
    if job.snapshot()['status']!='complete' or not path.is_file():
        raise HTTPException(404,'Plate crop unavailable.')
    return FileResponse(path,media_type='image/jpeg')
