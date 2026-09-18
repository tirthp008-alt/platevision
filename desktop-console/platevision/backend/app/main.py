import asyncio
import logging
import shutil
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from app.core.config import settings
from app.services.detector import get_detector, OnnxPlateDetector
from app.services.ocr import PlateOCR
from app.services.pipeline import process, CropStore, InvalidImage
from app.services.video import new_job, process_video

logger=logging.getLogger(__name__)
detector=None
ocr=None
vehicle_detector=None
startup_errors=[]
crops=CropStore(settings.result_ttl_seconds,settings.max_cached_crops)
inference_lock=asyncio.Lock()
video_jobs={}
video_tasks=set()
video_upload_lock=asyncio.Lock()

@asynccontextmanager
async def lifespan(app):
    global detector,ocr,vehicle_detector
    startup_errors.clear()
    for name, factory in [('plate',get_detector),('ocr',PlateOCR),('vehicle',lambda:OnnxPlateDetector(settings.vehicle_model_path,classes=80,allowed={2,3,5,7}) if settings.vehicle_context_enabled and settings.vehicle_model_path else None)]:
        try:
            value=await run_in_threadpool(factory)
            if name=='plate':detector=value
            elif name=='ocr':ocr=value
            else:vehicle_detector=value
        except Exception as error:
            startup_errors.append(f'{name}: {error}')
            logger.warning('Engine startup: %s',error)
    yield
    for job in video_jobs.values():
        job.cancel.set(); job.resume.set()
    if video_tasks:
        await asyncio.gather(*video_tasks,return_exceptions=True)

app=FastAPI(title='PlateSight API',version='0.3.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=settings.cors_origins.split(','),allow_methods=['GET','POST'],allow_headers=['Content-Type'])

@app.get('/api/health')
def health():
    return dict(status='ok' if detector and ocr else 'degraded',detector_ready=detector is not None,ocr_ready=ocr is not None,vehicle_detector_ready=vehicle_detector is not None,mode=settings.detector_mode,engine=getattr(detector,'runtime','unavailable'),ocr_engine=getattr(ocr,'runtime','unavailable'),fast_image_ready=bool(getattr(detector,'fast_models',None)),video_export_ready=detector is not None,errors=startup_errors,version='0.3.0')

async def detect(image,profile="fast"):
    if profile not in {"fast","detailed"}:raise HTTPException(422,"Unknown image profile.")
    if image.content_type not in {'image/jpeg','image/png','image/webp'}:raise HTTPException(415,'Upload JPG, PNG or WEBP.')
    data=await image.read(settings.max_upload_mb*1024*1024+1)
    if len(data)>settings.max_upload_mb*1024*1024:raise HTTPException(413,'Image exceeds size limit.')
    if detector is None:raise HTTPException(503,'Plate model unavailable. Configure MODEL_PATH; see /api/health.')
    if inference_lock.locked():raise HTTPException(429,'Inference is busy. Send the next available frame; do not queue frames.')
    async with inference_lock:
        try:return await run_in_threadpool(process,data,detector,crops,ocr,vehicle_detector,profile)
        except InvalidImage as error:raise HTTPException(422,str(error)) from error
        except Exception as error:
            logger.exception('Inference failed')
            raise HTTPException(500,'Inference failed. Check server logs and model compatibility.') from error

@app.post('/api/detect/image')
async def detect_image(image:UploadFile=File(...),profile:str=Form("fast")):return await detect(image,profile)
@app.post('/api/detect/frame')
async def detect_frame(image:UploadFile=File(...),profile:str=Form("fast")):return await detect(image,profile)
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


async def run_video(job):
    async with inference_lock:
        await run_in_threadpool(process_video,job,detector,ocr)


@app.post('/api/videos',status_code=202)
async def upload_video(video:UploadFile=File(...),profile:str=Form('fast')):
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
        task=asyncio.create_task(run_video(job));video_tasks.add(task)
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
def video_crop(job_id:str,track_id:int):
    job=video_job(job_id)
    path=job.folder/f'plate-{track_id}.jpg'
    if job.snapshot()['status']!='complete' or not path.is_file():
        raise HTTPException(404,'Plate crop unavailable.')
    return FileResponse(path,media_type='image/jpeg')
