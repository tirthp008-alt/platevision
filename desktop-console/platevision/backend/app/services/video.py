"""Local recording jobs with plate/vehicle regions and annotated H.264 output.

Region capture does not run OCR. Optional ANPR reads each track's best crops.
Every decoded frame is searched; the export preserves the recording frame rate.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
import json
import logging
import re
import subprocess
import time
import uuid

import cv2
import numpy as np

from app.core.config import settings
from app.services.normalization import normalize_plate
from app.services.pipeline import clamp

logger = logging.getLogger(__name__)


def restore_completed_jobs():
    """Keep unexpired annotated downloads available across a model restart."""
    root=Path(settings.video_output_dir).resolve()
    jobs={}
    if not root.is_dir():return jobs
    for folder in root.iterdir():
        if not re.fullmatch(r'[0-9a-f]{32}',folder.name) or folder.is_symlink() or not folder.is_dir():continue
        if folder.resolve().parent!=root:continue
        report=folder/'results.json';video=folder/'annotated.mp4'
        try:
            if not report.is_file() or report.stat().st_size>16_000_000 or not video.is_file():continue
            age=max(0,time.time()-report.stat().st_mtime)
            if age>=settings.video_result_ttl_seconds:continue
            result=json.loads(report.read_text(encoding='utf-8'))
            if not isinstance(result,dict) or not isinstance(result.get('frames'),int) or result['frames']<=0:continue
            job=VideoJob(folder.name,folder,folder/'input.mp4',result.get('profile','fast'))
            job.created=time.monotonic()-age
            job.update(**result)
            job.update(status='complete',phase='Ready to play',progress=100,frames_processed=result['frames'],
                       video_url=f'/api/videos/{job.id}/video',report_url=f'/api/videos/{job.id}/report')
            jobs[job.id]=job
        except (OSError,ValueError,TypeError):
            logger.warning('Could not restore completed video %s',folder.name)
    return jobs


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
            track['max_detection_confidence']=max(track.get('max_detection_confidence',0.),float(box[4]))
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
    if not job.snapshot().get('read_text', True):
        # Region retention must never depend on text confidence or validity.
        for track in tracks:
            job.checkpoint()
            track.update(text='', confidence=None, format_status='not_requested')
            if track['crops']:
                track['chosen_crop'] = track['crops'][0][1]
        job.update(progress=75, phase='Plate regions captured')
        return 0.
    started = time.perf_counter()
    angle_correction = bool(job.snapshot().get('angle_correction', False))
    batch_details = callable(getattr(ocr, 'read_many_details', None))
    plate_details = callable(getattr(ocr, 'read_plate_details', None))
    batched = (job.profile == 'fast' and (batch_details or callable(getattr(ocr, 'read_many', None)))
               and getattr(ocr, 'batch', True) is not None)
    cached = {}
    if batched:
        # Each round visits one crop per track, so every plate gets a turn.
        # The second observation is only needed if the first is uncertain.
        for observation in range(2):
            candidates = [i for i,t in enumerate(tracks) if len(t['crops']) > observation
                          and not (observation and _confident_reading(cached[(i,0)]))]
            for start in range(0, len(candidates), settings.fast_ocr_batch_size):
                job.checkpoint()
                indices = candidates[start:start + settings.fast_ocr_batch_size]
                pairs = [tracks[i]['crops'][observation][1:] for i in indices]
                readings = (ocr.read_many_details(pairs, angle_correction=angle_correction)
                            if batch_details else ocr.read_many(pairs))
                if len(readings) != len(indices):
                    raise ValueError('OCR returned an incomplete plate batch.')
                for i, pair, reading in zip(indices, pairs, readings):
                    job.checkpoint()
                    # Keep the older localization/contrast retries for hard crops.
                    # A faster batch must not turn an uncertain guess into a fact.
                    single_line = pair[1].shape[1] / max(1,pair[1].shape[0]) >= 2.5
                    if not batch_details and single_line and not _confident_reading(reading):
                        reading = max([reading, ocr.read_plate(*pair)], key=ocr.rank)
                    cached[(i, observation)] = reading
                job.update(progress=65+5*(observation+(start+len(indices))/max(1,len(candidates))),
                           phase='Reading plate crops in batches')
    for index, track in enumerate(tracks):
        job.checkpoint()
        readings = []
        for observation, (_, context, tight) in enumerate(track['crops']):
            reading = (cached[(index,observation)] if batched else
                       ocr.read_plate_details(context, tight, angle_correction=angle_correction) if plate_details else
                       ocr.read_plate(context, tight) if ocr else ('', 0.))
            raw, confidence = _reading_pair(reading)
            text, _, status = normalize_plate(raw)
            if isinstance(reading, dict) and (reading.get('requires_review') or reading.get('conflicting_readings')):
                status = 'uncertain'
            readings.append((text, float(confidence), status, context, tight, raw, reading))
            if _confident_reading(reading):
                break
        if readings:
            votes = Counter(text for text, confidence, _, _, _, _, _ in readings if confidence >= .7)
            text, confidence, status, chosen_crop, chosen_tight, raw, detail = max(readings, key=lambda r:(r[2]=='valid' and r[1]>=.7, votes[r[0]], r[1]))
            track.update(text=text, raw_text=raw, confidence=confidence, format_status=status if confidence>=.7 else 'uncertain',chosen_crop=chosen_crop)
            if isinstance(detail, dict):
                track['ocr_review'] = {key: value for key, value in detail.items()
                                       if key != 'rectified_crop' and not key.startswith('_')}
                if detail.get('rectification') is not None:
                    track['ocr_rectification'] = detail['rectification']
                    track['ocr_rectified_crop'] = detail.get('rectified_crop')
            if job.snapshot().get('green_filter',False):
                from app.services.experimental_ocr import green_alternative
                job.checkpoint()
                _, comparison=green_alternative(ocr,chosen_tight,(text,confidence))
                track['green_enhancement']=comparison
                if (comparison.get('applied') and not supported_reading(track)
                        and track.get('max_detection_confidence',0.)<settings.unverified_plate_threshold):
                    # Keep a requested alternative discoverable, without
                    # treating its OCR as evidence for the original reading.
                    track.update(experimental_review_only=True,format_status='uncertain')
        if not batched:
            job.update(progress=65+10*(index+1)/max(1,len(tracks)), phase='Reading best plate crops')
    job.update(progress=75, phase='Plate readings ready')
    return (time.perf_counter()-started)*1000


def _reading_pair(reading):
    return (reading['text'], reading['confidence']) if isinstance(reading, dict) else reading


def _confident_reading(reading):
    text, confidence = _reading_pair(reading)
    if isinstance(reading, dict) and (reading.get('requires_review') or reading.get('conflicting_readings')):
        return False
    return confidence >= (.88 if isinstance(reading, dict) else .94) and normalize_plate(text)[2] == 'valid'


def supported_reading(track):
    text = track['text']
    return (track.get('confidence') or 0.)>=.7 and 6<=len(text)<=12 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text)


def visible(observation, track):
    # Plate geometry is retained independently of OCR readability or score.
    return True


def _mark_plate_sources(observations, tracks, original_count):
    """The supplemental helper preserves the original prefix and appends boxes."""
    for index, observation in enumerate(observations):
        recovered = index >= original_count
        observation.update(localization_source='vehicle_crop' if recovered else 'whole_frame',
                           localization_requires_review=recovered)
        track = tracks[observation['track_id']-1]
        track['has_vehicle_crop_evidence'] = track.get('has_vehicle_crop_evidence', False) or recovered
        track['has_whole_frame_evidence'] = track.get('has_whole_frame_evidence', False) or not recovered
        track['localization_requires_review'] = not track['has_whole_frame_evidence']
        track['localization_source'] = ('mixed' if track['has_vehicle_crop_evidence'] and track['has_whole_frame_evidence']
                                        else 'vehicle_crop' if track['has_vehicle_crop_evidence'] else 'whole_frame')


def annotate(frame, observations, tracks, vehicles=None):
    # Draw the larger vehicle boxes first so the plate region remains visible.
    for vehicle in vehicles or []:
        x, y, w, h = vehicle['box']
        category = vehicle['category'].replace('_', ' ')
        score = vehicle.get('classification_confidence', vehicle['confidence'])
        review = ' | category review' if vehicle.get('classification_review', {}).get('requires_review') else ''
        label = f"V{vehicle['track_id']} {category} {score:.0%}{review}"
        _draw_box(frame, (x, y, w, h), label, (235, 185, 85))
    for observation in observations:
        track = tracks[observation['track_id']-1]
        if not visible(observation, track):
            continue
        x, y, w, h = observation['box']
        review_only=bool(track.get('experimental_review_only'))
        localization_review = bool(observation.get('localization_requires_review'))
        region_only = track.get('format_status')=='not_requested'
        good = not localization_review and (region_only or (not review_only and track['format_status']=='valid' and track['confidence']>=.7))
        color = (130, 220, 70) if good else (80, 185, 245)
        label='Candidate | ' if review_only else ''
        text = (f"Plate #{track['id']} {observation['score']:.0%}" if region_only
                else f"#{track['id']} {label}{track['text'] or 'Plate / unreadable'}")
        if localization_review:
            text += ' | vehicle crop review'
        _draw_box(frame, (x, y, w, h), text, color)
    if vehicles is not None:
        count = sum(visible(o, tracks[o['track_id']-1]) for o in observations)
        text = f"Visible: {len(vehicles)} vehicles | {count} plates"
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 460), 30), (22, 34, 47), -1)
        cv2.putText(frame, text, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 245, 250), 1, cv2.LINE_AA)
    return frame


def _draw_box(frame, box, text, color):
    x, y, w, h = map(int, box)
    scale = max(.45, min(.8, frame.shape[1]/1800))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    lx = max(0, min(x, frame.shape[1]-tw-10)); ly = max(th+10, y)
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
    cv2.rectangle(frame, (lx,ly-th-10), (lx+tw+10,ly), color, -1)
    cv2.putText(frame,text,(lx+5,ly-5),cv2.FONT_HERSHEY_SIMPLEX,scale,(20,35,30),1,cv2.LINE_AA)


def video_search_strategy(detector,profile):
    return ('high_resolution_whole_frame' if profile=='fast'
            and getattr(detector,'fast_models',None)
            and callable(getattr(detector,'detect_fast',None))
            else 'whole_frame_and_overlapping_tiles')


def detect_video_frame(detector,image,profile,read_text=True):
    if callable(getattr(detector, 'detect_regions', None)):
        return detector.detect_regions(image, profile)
    if video_search_strategy(detector,profile)=='high_resolution_whole_frame':
        return detector.detect_fast(image)
    return detector.detect_plates(image,refine=profile=='detailed')


def _timed_call(function, *args):
    started = time.perf_counter()
    result = function(*args)
    return result, (time.perf_counter()-started)*1000


def _attach_plates(vehicles, observations):
    """Containment is an association, never a gate on finding plate regions."""
    for observation in observations:
        x, y, w, h = observation['box']
        cx, cy = x+w/2, y+h/2
        enclosing = [v for v in vehicles if v['box'][0] <= cx <= v['box'][0]+v['box'][2]
                     and v['box'][1] <= cy <= v['box'][1]+v['box'][3]]
        if enclosing:
            vehicle = min(enclosing, key=lambda v:v['box'][2]*v['box'][3])
            vehicle.setdefault('plate_ids', []).append(observation['track_id'])
            observation['vehicle_id'] = vehicle['id']


def _vehicle_summary(tracker, available, error=None):
    result = tracker.summary(available=available and error is None)
    result['complete'] = bool(available and error is None)
    result['incomplete'] = error is not None
    if error:
        result['warning'] = 'Vehicle analysis stopped after an inference error. Plate capture continued; vehicle totals are unavailable.'
    return result


def _select_vehicle_frames(selections, vehicles, image, frame_index):
    """Keep frame references only; embed the best crop during the export pass."""
    for vehicle in vehicles or []:
        x, y, w, h = clamp(vehicle['box'], image.shape[1], image.shape[0], padding=0.)
        if min(w, h) < 16:
            continue
        crop = image[y:y+h, x:x+w]
        # Score at bounded resolution so large frames do not add another
        # expensive full-resolution image operation for each vehicle.
        ratio = min(1., 128 / max(w, h))
        thumbnail = cv2.resize(crop, (max(2, round(w*ratio)), max(2, round(h*ratio))))
        sharpness = cv2.Laplacian(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        quality = float((1 + sharpness) * np.sqrt(w*h) * vehicle['confidence'])
        previous = selections.get(vehicle['track_id'])
        if previous is None or quality > previous['quality']:
            selections[vehicle['track_id']] = dict(frame=frame_index, box=(x, y, w, h), quality=quality)


def _vehicle_plate_ids(vehicle_timeline, plate_results):
    """Use temporal ownership, avoiding duplicate evidence across local tracks."""
    retained = {plate['id'] for plate in plate_results}
    owners = {}
    for vehicles in vehicle_timeline:
        for vehicle in vehicles or []:
            for plate_id in vehicle.get('plate_ids', []):
                if plate_id in retained:
                    owners.setdefault(plate_id, Counter())[vehicle['track_id']] += 1
    linked = {}
    for plate_id, votes in owners.items():
        ranked = votes.most_common()
        # Equal ownership is ambiguous, so it cannot strengthen a match.
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            linked.setdefault(ranked[0][0], []).append(plate_id)
    return linked


def _save_crop(path, crop):
    try:
        return bool(cv2.imwrite(str(path), crop))
    except cv2.error:
        logger.warning('Could not save video evidence crop %s', path.name)
        return False


def process_video(job, detector, ocr, vehicle_detector=None, reid_context=None):
    started = time.perf_counter()
    cap = None; encoder = None; error_log = None; inference_pool = None
    try:
        import imageio_ffmpeg
        from app.services.vehicles import VehicleTracker
        from app.services.vehicle_plate_search import recover_vehicle_plates
        options = job.snapshot()
        read_text = bool(options.get('read_text', True))
        angle_correction = bool(options.get('angle_correction', False))
        vehicle_available = vehicle_detector is not None
        vehicle_error = None
        reid_selections = {}; reid_descriptors = {}; reid_error = None; reid_ms = 0.
        reid_enabled = bool(reid_context is not None and reid_context.get('engine') is not None
                            and reid_context.get('gallery') is not None)
        if vehicle_detector is not None:
            inference_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='video-regions')
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
        search_strategy=video_search_strategy(detector,job.profile)
        limit = 1280 if job.profile=='fast' and search_strategy=='whole_frame_and_overlapping_tiles' else 1920
        ratio = min(1, limit/max(original_w,original_h))
        width, height = max(2,int(original_w*ratio)//2*2), max(2,int(original_h*ratio)//2*2)
        tracker = PlateTracks(fps)
        vehicle_tracker = VehicleTracker(fps, taxonomy=getattr(vehicle_detector, 'taxonomy', 'coco'))
        timeline = []; vehicle_timeline = []; frame_times=[]; detector_times=[]; vehicle_times=[]
        whole_frame_detector_times = []; plate_search_times = []; plate_search_reports = {}
        analysis_interval = max(1, round(fps))
        plate_search_summary = dict(enabled=False, scope='recording', eligible_vehicles=0, searched_vehicles=0,
                                    added_regions=0, failed_crops=0, remaining_vehicles=0, frames_searched=0,
                                    failed_frames=0, time_ms=0.,
                                    counting_note='Supplemental crop searches and added regions count per-frame observations, '
                                                  'not unique vehicles or plates. Whole-frame plate search is always retained.')
        job.update(status='processing', phase='Detecting plates and vehicles in every frame' if vehicle_detector else 'Detecting plates in every frame', fps=fps,
                   width=width,height=height,source_width=original_w,source_height=original_h,
                   total_frames=total,engine=getattr(detector,'runtime','ONNX'),
                   profile=job.profile,search_strategy=search_strategy, source_duration_seconds=round(total/fps,3),
                   read_text=read_text,angle_correction=angle_correction,
                   mode='anpr' if read_text else 'regions')
        index=0; observation_count=0
        while True:
            job.checkpoint(); tick=time.perf_counter()
            paused_at_frame_start = job.paused_seconds
            ok, image = cap.read()
            if not ok: break
            if index>=fps*settings.video_max_seconds:
                raise ValueError('Video exceeds the duration limit.')
            if image.shape[1]!=width or image.shape[0]!=height:
                image=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
            vehicles = None; vehicle_objects = []
            if inference_pool and vehicle_error is None:
                plate_future = inference_pool.submit(_timed_call, detect_video_frame, detector, image, job.profile, read_text)
                vehicle_future = inference_pool.submit(_timed_call, vehicle_detector.objects, image)
                boxes, plate_ms = plate_future.result()
                try:
                    vehicle_objects, vehicle_ms = vehicle_future.result()
                    vehicles = vehicle_tracker.update(vehicle_objects,index,width,height)
                    vehicle_times.append(vehicle_ms)
                except Exception:
                    logger.exception('Optional vehicle analysis failed; continuing plate capture')
                    vehicle_error = 'inference_failed'
                    vehicle_ms = None
                    vehicle_objects = []
            else:
                boxes, plate_ms = _timed_call(detect_video_frame, detector, image, job.profile, read_text)
                vehicle_ms = None
            whole_frame_detector_times.append(plate_ms)
            original_plate_count = len(boxes)
            job.checkpoint()
            search_started = time.perf_counter()
            try:
                boxes, plate_search_report = recover_vehicle_plates(
                    image, detector, list(boxes), vehicle_objects, start_index=index*settings.vehicle_plate_max_crops)
            except InterruptedError:
                raise
            except Exception:
                logger.exception('Optional vehicle-crop plate search failed; retaining whole-frame detections')
                plate_search_report = dict(enabled=settings.vehicle_plate_search_enabled, eligible_vehicles=0,
                                           searched_vehicles=0, added_regions=0, failed_crops=0,
                                           remaining_vehicles=0, error='search_failed')
            search_ms = (time.perf_counter()-search_started)*1000
            job.checkpoint()
            plate_search_report = dict(plate_search_report, scope='frame', time_ms=round(search_ms, 3))
            plate_ms += search_ms
            plate_search_times.append(search_ms)
            plate_search_summary['enabled'] |= bool(plate_search_report.get('enabled'))
            for key in ('eligible_vehicles', 'searched_vehicles', 'added_regions', 'failed_crops', 'remaining_vehicles'):
                plate_search_summary[key] += plate_search_report.get(key, 0)
            plate_search_summary['frames_searched'] += plate_search_report.get('searched_vehicles', 0) > 0
            plate_search_summary['failed_frames'] += bool(plate_search_report.get('error') or plate_search_report.get('failed_crops'))
            plate_search_summary['time_ms'] += search_ms
            if index % analysis_interval == 0:
                plate_search_reports[index] = plate_search_report
            detector_times.append(plate_ms)
            timeline.append(tracker.update(boxes,index,image))
            _mark_plate_sources(timeline[-1], tracker.tracks, original_plate_count)
            _attach_plates(vehicles or [], timeline[-1])
            vehicle_timeline.append(vehicles)
            if reid_enabled and reid_error is None and vehicle_error is None:
                reid_started = time.perf_counter()
                try:
                    _select_vehicle_frames(reid_selections, vehicles, image, index)
                except Exception:
                    logger.exception('Optional vehicle crop selection failed; continuing plate capture')
                    reid_error = 'crop_selection_failed'
                reid_ms += (time.perf_counter()-reid_started)*1000
            observation_count+=len(timeline[-1])+len(vehicles or [])
            if observation_count>1_000_000:
                raise ValueError('Recording contains too many region observations. Split it into shorter clips.')
            frame_times.append((time.perf_counter()-tick-job.paused_seconds+paused_at_frame_start)*1000)
            index+=1
            if index%5==0 or index==total:
                job.update(frames_processed=index,progress=min(65,65*index/total),
                           detection_ms=round(detector_times[-1],2),plate_detection_ms=round(plate_ms,2),
                           whole_frame_plate_detection_ms=round(whole_frame_detector_times[-1],2),
                           vehicle_plate_search=plate_search_report,
                           vehicle_detection_ms=round(vehicle_ms,2) if vehicle_ms is not None else None,frame_ms=round(frame_times[-1],2),
                           plates_visible=len(timeline[-1]),vehicle_summary=_vehicle_summary(vehicle_tracker,vehicle_available,vehicle_error))
        cap.release(); cap=None
        if not timeline:
            raise ValueError('Video has no decodable frames.')
        plate_search_reports[len(timeline)-1] = plate_search_report
        plate_search_summary['time_ms'] = round(plate_search_summary['time_ms'], 3)
        if index<total-1:
            raise ValueError('Video ended before its declared frame count; it may be damaged.')
        detection_wall=(time.perf_counter()-started-job.paused_seconds)*1000
        paused_before_ocr=job.paused_seconds
        ocr_ms=max(0.,read_tracks(tracker.tracks,ocr,job)-(job.paused_seconds-paused_before_ocr)*1000)
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
        vehicle_results = vehicle_tracker.results()
        selected_by_frame = {}
        if reid_enabled and reid_error is None and vehicle_error is None:
            confirmed_track_ids = {vehicle['track_id'] for vehicle in vehicle_results if vehicle['confirmed']}
            for track_id, selection in reid_selections.items():
                if track_id in confirmed_track_ids:
                    selected_by_frame.setdefault(selection['frame'], []).append((track_id, selection['box']))
        render_started=time.perf_counter(); paused_before_render=job.paused_seconds
        for index, observations in enumerate(timeline):
            job.checkpoint()
            ok, image=cap.read()
            if not ok:
                raise ValueError('Video decode failed during export.')
            if image.shape[1]!=width or image.shape[0]!=height:
                image=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
            # The source crop is read before annotation. Stage descriptors only;
            # no gallery writes occur until the MP4 export succeeds.
            if reid_error is None:
                for track_id, (x, y, w, h) in selected_by_frame.get(index, []):
                    job.checkpoint()
                    reid_started = time.perf_counter()
                    try:
                        reid_descriptors[track_id] = reid_context['engine'].embed([image[y:y+h,x:x+w]])[0]
                    except InterruptedError:
                        raise
                    except Exception:
                        logger.exception('Optional video appearance matching failed; continuing export')
                        reid_error = 'inference_failed'
                        reid_descriptors.clear()
                    reid_ms += (time.perf_counter()-reid_started)*1000
                    job.checkpoint()
                    if reid_error:
                        break
            encoder.stdin.write(annotate(image,observations,tracker.tracks,
                                        vehicle_timeline[index]).tobytes())
            if index%10==0:
                job.update(progress=75+24*(index+1)/len(timeline),phase='Rendering annotated MP4')
        encoder.stdin.close(); encoder.wait(timeout=60)
        if encoder.returncode:
            raise RuntimeError('Video encoder failed. See the server log.')
        job.checkpoint()
        render_ms=(time.perf_counter()-render_started-job.paused_seconds+paused_before_render)*1000
        track_results=[]; rectification_ms=0.
        seen_by_track={t['id']:[] for t in tracker.tracks}
        for index, observations in enumerate(timeline):
            for observation in observations:
                if visible(observation,tracker.tracks[observation['track_id']-1]):
                    seen_by_track[observation['track_id']].append(index)
        for track in tracker.tracks:
            job.checkpoint()
            seen=seen_by_track[track['id']]
            if not seen: continue
            crop_name=f"plate-{track['id']}.jpg"
            crop_saved = False
            if track['crops']:
                original_crop = track.get('chosen_crop',track['crops'][0][1])
                crop_saved = _save_crop(job.folder/crop_name, original_crop)
            track_results.append(dict(id=track['id'],text=track['text'],raw_text=track.get('raw_text',''),ocr_confidence=track['confidence'],
                                      format_status=track['format_status'],first_seen=round(seen[0]/fps,3),
                                      last_seen=round(seen[-1]/fps,3),observations=len(seen),
                                      detection_confidence=track['max_detection_confidence'],
                                      localization_source=track.get('localization_source', 'whole_frame'),
                                      localization_requires_review=track.get('localization_requires_review', False),
                                      has_vehicle_crop_evidence=track.get('has_vehicle_crop_evidence', False),
                                      has_whole_frame_evidence=track.get('has_whole_frame_evidence', True),
                                      crop_url=f'/api/videos/{job.id}/crops/{track["id"]}' if crop_saved else None))
            if 'ocr_review' in track:
                track_results[-1]['ocr_review'] = track['ocr_review']
            if angle_correction and track['crops']:
                from app.services.rectification import rectify_plate
                job.checkpoint()
                if 'ocr_rectification' in track:
                    metadata = track['ocr_rectification']
                    corrected = track.get('ocr_rectified_crop')
                else:
                    corrected, metadata = rectify_plate(original_crop)
                rectification_ms += float(metadata.get('time_ms', 0.))
                track_results[-1]['rectification'] = metadata
                if metadata.get('applied') and corrected is not None:
                    path = job.folder/f"plate-{track['id']}-rectified.jpg"
                    track_results[-1]['rectified_crop_url'] = (f'/api/videos/{job.id}/crops/{track["id"]}?variant=rectified'
                                                               if _save_crop(path, corrected) else None)
            if 'green_enhancement' in track:
                track_results[-1]['green_enhancement']=track['green_enhancement']
            if track.get('experimental_review_only'):
                track_results[-1]['experimental_review_only']=True
        plate_ids = _vehicle_plate_ids(vehicle_timeline, track_results)
        for vehicle in vehicle_results:
            vehicle['plate_ids'] = plate_ids.get(vehicle['track_id'], [])
        from app.services.plate_summary import summarize_recognized_plates
        recognized_plate_summary = summarize_recognized_plates(track_results, read_text=read_text,
                                                               frames_by_track=seen_by_track)
        reidentification = None
        if reid_context is not None:
            from app.services.reidentification import MODEL_NAME, REVIEW_NOTE, match_observations, strongest_plate
            if reid_error or vehicle_error:
                reidentification = dict(status='failed', model=MODEL_NAME, observations=[], observation_count=0,
                                        processing_time_ms=0., note=REVIEW_NOTE,
                                        warning='Cross-camera appearance matching failed; plate and vehicle results remain available.'
                                        if reid_error else 'Cross-camera matching was skipped because vehicle analysis is incomplete; plate results remain available.')
            else:
                by_id = {plate['id']: plate for plate in track_results}
                observations = []
                for vehicle in vehicle_results:
                    track_id = vehicle['track_id']
                    if track_id not in reid_descriptors:
                        continue
                    text, confidence = strongest_plate([by_id[plate_id] for plate_id in vehicle['plate_ids']]) if read_text else (None, 0.)
                    start = reid_context['observed_at']
                    observations.append(dict(descriptor=reid_descriptors[track_id],
                                             track_id=f"{job.id}:{vehicle['id']}", category=vehicle['category'],
                                             observed_at=start+reid_selections[track_id]['frame']/fps,
                                             first_seen_at=start+vehicle['first_frame']/fps,
                                             last_seen_at=start+vehicle['last_frame']/fps,
                                             plate_text=text, plate_confidence=confidence))
                reidentification = match_observations(observations, reid_context, checkpoint=job.checkpoint)
            reidentification['processing_time_ms'] = round(reidentification['processing_time_ms']+reid_ms, 2)
        wall_elapsed=(time.perf_counter()-started)*1000
        elapsed=wall_elapsed-job.paused_seconds*1000
        # The exported video retains every observed box. The JSON analytics
        # samples at roughly 1 Hz (plus the last frame), keeping long recordings
        # small enough for polling, downloads and restart restoration.
        report=dict(frames=len(timeline),fps=fps,width=width,height=height,
                    source_width=original_w,source_height=original_h,profile=job.profile,
                    engine=getattr(detector,'runtime','ONNX'),detection_ms=percentiles(detector_times),
                    plate_detection_ms=percentiles(detector_times),vehicle_detection_ms=percentiles(vehicle_times),
                    whole_frame_plate_detection_ms=percentiles(whole_frame_detector_times),
                    vehicle_plate_search=plate_search_summary,vehicle_plate_search_ms=percentiles(plate_search_times),
                    ocr_engine=getattr(ocr,'runtime','unavailable') if read_text else 'not_requested',
                    read_text=read_text,angle_correction=angle_correction,mode='anpr' if read_text else 'regions',
                    decode_detect_track_ms=percentiles(frame_times),ocr_total_ms=round(ocr_ms,2),
                    rectification_total_ms=round(rectification_ms,2),
                    render_total_ms=round(render_ms,2),total_ms=round(elapsed,2),
                    paused_ms=round(job.paused_seconds*1000,2),wall_ms=round(wall_elapsed,2),
                    processing_ms_per_frame=round(elapsed/len(timeline),2),
                    detection_phase_ms=round(detection_wall,2),tracks=track_results,
                    recognized_plate_summary=recognized_plate_summary,
                    recognized_plates=recognized_plate_summary['groups'],
                    plate_summary=dict(total_tracks=len(track_results),
                                       max_visible=max(sum(visible(o,tracker.tracks[o['track_id']-1]) for o in frame) for frame in timeline),
                                       counting_note='Geometry tracks can fragment after occlusion; track counts are not guaranteed unique plates.'),
                    vehicle_summary=_vehicle_summary(vehicle_tracker,vehicle_available,vehicle_error),
                    vehicle_tracks=vehicle_results,reidentification=reidentification,
                    analysis_sample_interval_frames=analysis_interval,
                    analysis_timeline_note='Counts sampled approximately once per second, including first and last frames. Detection and annotated video retain every frame.',
                    analysis_timeline=[dict(frame=i,time_seconds=round(i/fps,3),
                                            vehicle_plate_search=plate_search_reports[i],
                                            visible_vehicles=len(vehicles) if vehicles is not None else None,
                                            counts_by_category=dict(Counter(v['category'] for v in vehicles)) if vehicles is not None else None,
                                            visible_plates=sum(visible(o,tracker.tracks[o['track_id']-1]) for o in timeline[i]))
                                       for i,vehicles in enumerate(vehicle_timeline)
                                       if i % analysis_interval == 0 or i == len(timeline)-1],
                    all_frames_searched=True,search_strategy=search_strategy,
                    weights_retrained=bool(getattr(detector,'weights_retrained',False)),
                    detector_model=getattr(detector,'model_id','baseline'),
                    warnings=([_vehicle_summary(vehicle_tracker,vehicle_available,vehicle_error)['warning']] if vehicle_error else []),
                    note=('Final track readings are applied retrospectively. OCR and export are included in the overall processing average.'
                          if read_text else 'Plate regions are retained without OCR. Every frame is searched; overall processing includes vehicle detection, tracking, optional crop correction and export.'))
        if reidentification and reidentification.get('warning'):
            report['warnings'].append(reidentification['warning'])
        if plate_search_summary['failed_frames']:
            report['warnings'].append('Some supplemental vehicle-crop plate searches failed; whole-frame plate regions were retained.')
        (job.folder/'results.json').write_text(json.dumps(report,separators=(',',':')),encoding='utf-8')
        job.update(status='complete',phase='Ready to play',progress=100,frames_processed=len(timeline),
                   video_url=f'/api/videos/{job.id}/video',report_url=f'/api/videos/{job.id}/report',**report)
    except InterruptedError:
        job.update(status='cancelled',phase='Cancelled')
    except Exception as error:
        logger.exception('Video job failed')
        job.update(status='failed',phase='Failed',error=str(error))
    finally:
        if inference_pool:
            inference_pool.shutdown(wait=True, cancel_futures=True)
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
