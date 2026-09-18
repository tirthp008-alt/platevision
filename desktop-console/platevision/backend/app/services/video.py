"""Bounded, local recording jobs with per-frame plate search and H.264 output.

OCR runs on each track's best crops instead of on every video frame.
The second pass burns the final observed reading into every matching frame.
"""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
import json
import logging
import subprocess
import time
import uuid

import cv2
import numpy as np

from app.core.config import settings
from app.services.normalization import normalize_plate
from app.services.pipeline import clamp

logger = logging.getLogger(__name__)


def iou(a, b):
    x, y, w, h = a[:4]; bx, by, bw, bh = b[:4]
    area = max(0, min(x+w, bx+bw)-max(x, bx))*max(0, min(y+h, by+bh)-max(y, by))
    return area/max(1, w*h+bw*bh-area)


def percentiles(values):
    return dict(p50=round(float(np.percentile(values, 50)), 2),
                p95=round(float(np.percentile(values, 95)), 2)) if values else dict(p50=0, p95=0)


@dataclass
class VideoJob:
    id: str
    folder: Path
    source: Path
    profile: str = 'fast'
    created: float = field(default_factory=time.monotonic)
    paused_seconds: float = 0.
    cancel: Event = field(default_factory=Event)
    resume: Event = field(default_factory=Event)
    lock: Lock = field(default_factory=Lock)
    state: dict = field(default_factory=lambda: dict(status='queued', progress=0, phase='Queued'))

    def __post_init__(self):
        self.resume.set()

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            return dict(id=self.id, **self.state)

    def checkpoint(self):
        if not self.resume.is_set():
            started=time.perf_counter()
            while not self.resume.wait(.1):
                if self.cancel.is_set():
                    raise InterruptedError('Recording cancelled.')
            self.paused_seconds += time.perf_counter()-started
        if self.cancel.is_set():
            raise InterruptedError('Recording cancelled.')


class PlateTracks:
    """One-to-one spatial matching; nearby plates cannot claim the same track."""
    def __init__(self, fps):
        self.fps = fps
        self.tracks = []

    def update(self, boxes, frame_index, image):
        available = {t['id']: t for t in self.tracks if frame_index-t['last_frame'] <= max(1, self.fps*.25)}
        pairs = sorted(((iou(box, t['box']), index, t['id']) for index, box in enumerate(boxes)
                        for t in available.values()), reverse=True)
        matches, used = {}, set()
        for score, index, track_id in pairs:
            if score < .25 or index in matches or track_id in used:
                continue
            matches[index] = available[track_id]; used.add(track_id)
        observations = []
        for index, box in enumerate(boxes):
            track = matches.get(index)
            if track is None:
                if len(self.tracks)>=5000:
                    raise ValueError('Recording exceeds 5,000 plate tracks. Split it into shorter clips.')
                track = dict(id=len(self.tracks)+1, first_frame=frame_index, observations=0,
                             best_quality=-1, crops=[], text='', confidence=0., format_status='uncertain')
                self.tracks.append(track)
            track.update(box=box, last_frame=frame_index)
            track['observations'] += 1
            x, y, w, h = clamp(box[:4], image.shape[1], image.shape[0], .15)
            tight_x, tight_y, tight_w, tight_h = map(int, box[:4])
            crop = image[y:y+h, x:x+w]
            if crop.size:
                quality = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) * np.sqrt(w*h)
                # At most two diverse high-quality observations per track.
                if quality > track['best_quality']*1.12 or not track['crops']:
                    track['best_quality'] = max(quality, track['best_quality'])
                    track['crops'].append((quality, crop.copy(), image[tight_y:tight_y+tight_h,tight_x:tight_x+tight_w].copy()))
                    track['crops'] = sorted(track['crops'], key=lambda c:c[0], reverse=True)[:2]
            observations.append(dict(track_id=track['id'], box=[int(v) for v in box[:4]], score=float(box[4])))
        return observations


def read_tracks(tracks, ocr, job):
    started = time.perf_counter()
    for index, track in enumerate(tracks):
        job.checkpoint()
        readings = []
        for _, context, tight in track['crops']:
            raw, confidence = ocr.read_plate(context, tight) if ocr else ('', 0.)
            text, _, status = normalize_plate(raw)
            readings.append((text, float(confidence), status, context))
            if status == 'valid' and confidence >= .94:
                break
        if readings:
            votes = Counter(text for text, confidence, _, _ in readings if confidence >= .7)
            text, confidence, status, chosen_crop = max(readings, key=lambda r:(r[2]=='valid' and r[1]>=.7, votes[r[0]], r[1]))
            track.update(text=text, confidence=confidence, format_status=status if confidence>=.7 else 'uncertain',chosen_crop=chosen_crop)
        job.update(progress=65+10*(index+1)/max(1,len(tracks)), phase='Reading best plate crops')
    return (time.perf_counter()-started)*1000


def visible(observation, track):
    text = track['text']
    supported = track['confidence']>=.7 and 6<=len(text)<=12 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text)
    return observation['score']>=settings.unverified_plate_threshold or supported


def annotate(frame, observations, tracks):
    for observation in observations:
        track = tracks[observation['track_id']-1]
        if not visible(observation, track):
            continue
        x, y, w, h = observation['box']
        good = track['format_status']=='valid' and track['confidence']>=.7
        color = (130, 220, 70) if good else (80, 185, 245)
        text = f"#{track['id']} {track['text'] or 'Plate / unreadable'}"
        scale = max(.45, min(.8, frame.shape[1]/1800))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        lx = max(0, min(x, frame.shape[1]-tw-10)); ly = max(th+10, y)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.rectangle(frame, (lx,ly-th-10), (lx+tw+10,ly), color, -1)
        cv2.putText(frame,text,(lx+5,ly-5),cv2.FONT_HERSHEY_SIMPLEX,scale,(20,35,30),1,cv2.LINE_AA)
    return frame


def process_video(job, detector, ocr):
    started = time.perf_counter()
    cap = None; encoder = None; error_log = None
    try:
        import imageio_ffmpeg
        cv2.setNumThreads(2)
        cap = cv2.VideoCapture(str(job.source))
        if not cap.isOpened():
            raise ValueError('Cannot decode this video. Use a valid MP4, MOV or WebM recording.')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or not 1<=fps<=120 or total<=0:
            raise ValueError('Video must have a valid frame rate (1–120 fps) and duration.')
        if total/fps>settings.video_max_seconds:
            raise ValueError(f'Video exceeds {settings.video_max_seconds//60} minutes.')
        original_w, original_h = int(cap.get(3)), int(cap.get(4))
        if not 0<original_w*original_h<=16_000_000:
            raise ValueError('Video frame exceeds the supported size.')
        limit = 1280 if job.profile=='fast' else 1920
        ratio = min(1, limit/max(original_w,original_h))
        width, height = max(2,int(original_w*ratio)//2*2), max(2,int(original_h*ratio)//2*2)
        tracker = PlateTracks(fps); timeline = []; frame_times=[]; detector_times=[]
        job.update(status='processing', phase='Detecting plates in every frame', fps=fps,
                   width=width,height=height,total_frames=total,engine=getattr(detector,'runtime','ONNX'),
                   profile=job.profile, source_duration_seconds=round(total/fps,3))
        index=0; observation_count=0
        while True:
            job.checkpoint(); tick=time.perf_counter()
            ok, image = cap.read()
            if not ok: break
            if index>=fps*settings.video_max_seconds:
                raise ValueError('Video exceeds the duration limit.')
            if image.shape[1]!=width or image.shape[0]!=height:
                image=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
            detection_start=time.perf_counter()
            boxes=detector.detect_plates(image,refine=job.profile=='detailed')
            detector_times.append((time.perf_counter()-detection_start)*1000)
            timeline.append(tracker.update(boxes,index,image))
            observation_count+=len(timeline[-1])
            if observation_count>1_000_000:
                raise ValueError('Recording contains too many plate observations. Split it into shorter clips.')
            frame_times.append((time.perf_counter()-tick)*1000)
            index+=1
            if index%5==0 or index==total:
                job.update(frames_processed=index,progress=min(65,65*index/total),
                           detection_ms=round(detector_times[-1],2),frame_ms=round(frame_times[-1],2))
        cap.release(); cap=None
        if not timeline:
            raise ValueError('Video has no decodable frames.')
        if index<total-1:
            raise ValueError('Video ended before its declared frame count; it may be damaged.')
        detection_wall=(time.perf_counter()-started-job.paused_seconds)*1000
        paused_before_ocr=job.paused_seconds
        ocr_ms=read_tracks(tracker.tracks,ocr,job)-(job.paused_seconds-paused_before_ocr)*1000
        # No frames are skipped. A second sequential pass attaches the best
        # observed reading to the corresponding track throughout the video.
        cap=cv2.VideoCapture(str(job.source))
        output=job.folder/'annotated.mp4'
        error_log=open(job.folder/'encoder.log','wb')
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
                 '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0',
                 '-i',str(job.source),'-map','0:v:0','-map','1:a?','-c:v','libx264',
                 '-preset','ultrafast','-crf','20','-pix_fmt','yuv420p','-threads','2',
                 '-c:a','aac','-movflags','+faststart','-t',str(len(timeline)/fps),str(output)]
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=error_log,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        render_started=time.perf_counter(); paused_before_render=job.paused_seconds
        for index, observations in enumerate(timeline):
            job.checkpoint()
            ok, image=cap.read()
            if not ok:
                raise ValueError('Video decode failed during export.')
            if image.shape[1]!=width or image.shape[0]!=height:
                image=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
            encoder.stdin.write(annotate(image,observations,tracker.tracks).tobytes())
            if index%10==0:
                job.update(progress=75+24*(index+1)/len(timeline),phase='Rendering annotated MP4')
        encoder.stdin.close(); encoder.wait(timeout=60)
        if encoder.returncode:
            raise RuntimeError('Video encoder failed. See the server log.')
        render_ms=(time.perf_counter()-render_started-job.paused_seconds+paused_before_render)*1000
        track_results=[]
        seen_by_track={t['id']:[] for t in tracker.tracks}
        for index, observations in enumerate(timeline):
            for observation in observations:
                if visible(observation,tracker.tracks[observation['track_id']-1]):
                    seen_by_track[observation['track_id']].append(index)
        for track in tracker.tracks:
            seen=seen_by_track[track['id']]
            if not seen: continue
            crop_name=f"plate-{track['id']}.jpg"
            if track['crops']:
                cv2.imwrite(str(job.folder/crop_name),track.get('chosen_crop',track['crops'][0][1]))
            track_results.append(dict(id=track['id'],text=track['text'],ocr_confidence=track['confidence'],
                                      format_status=track['format_status'],first_seen=round(seen[0]/fps,3),
                                      last_seen=round(seen[-1]/fps,3),observations=len(seen),
                                      crop_url=f'/api/videos/{job.id}/crops/{track["id"]}'))
        wall_elapsed=(time.perf_counter()-started)*1000
        elapsed=wall_elapsed-job.paused_seconds*1000
        report=dict(frames=len(timeline),fps=fps,width=width,height=height,profile=job.profile,
                    engine=getattr(detector,'runtime','ONNX'),detection_ms=percentiles(detector_times),
                    decode_detect_track_ms=percentiles(frame_times),ocr_total_ms=round(ocr_ms,2),
                    render_total_ms=round(render_ms,2),total_ms=round(elapsed,2),
                    paused_ms=round(job.paused_seconds*1000,2),wall_ms=round(wall_elapsed,2),
                    processing_ms_per_frame=round(elapsed/len(timeline),2),
                    detection_phase_ms=round(detection_wall,2),tracks=track_results,
                    all_frames_searched=True,weights_retrained=False,
                    note='Final track readings are applied retrospectively. OCR and export are included in the overall processing average.')
        (job.folder/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        job.update(status='complete',phase='Ready to play',progress=100,frames_processed=len(timeline),
                   video_url=f'/api/videos/{job.id}/video',report_url=f'/api/videos/{job.id}/report',**report)
    except InterruptedError:
        job.update(status='cancelled',phase='Cancelled')
    except Exception as error:
        logger.exception('Video job failed')
        job.update(status='failed',phase='Failed',error=str(error))
    finally:
        if cap: cap.release()
        if encoder and encoder.poll() is None:
            encoder.kill(); encoder.wait(timeout=10)
        if error_log: error_log.close()
        if job.snapshot()['status']!='complete':
            (job.folder/'annotated.mp4').unlink(missing_ok=True)
        job.source.unlink(missing_ok=True)


def new_job(suffix, profile):
    job_id=uuid.uuid4().hex
    folder=Path(settings.video_output_dir).resolve()/job_id
    folder.mkdir(parents=True)
    return VideoJob(job_id,folder,folder/f'input{suffix}',profile)
