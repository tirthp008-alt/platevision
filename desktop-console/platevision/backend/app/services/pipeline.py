from concurrent.futures import ThreadPoolExecutor
import io
import time
import uuid
from collections import OrderedDict
from threading import Lock
from PIL import Image, ImageOps, UnidentifiedImageError
import cv2
import numpy as np
from app.services.detector import MockPlateDetector
from app.services.normalization import normalize_plate
from app.core.config import settings

class CropStore:
    def __init__(self, ttl=60, limit=200):
        self.ttl, self.limit, self.items, self.lock = ttl, limit, OrderedDict(), Lock()
    def _expire(self):
        now=time.monotonic()
        while self.items and next(iter(self.items.values()))[0] <= now:
            self.items.popitem(last=False)
    def put(self,key,data):
        with self.lock:
            self._expire();self.items[key]=(time.monotonic()+self.ttl,data)
            while len(self.items)>self.limit:self.items.popitem(last=False)
    def get(self,key):
        with self.lock:
            self._expire();item=self.items.get(key)
            return item[1] if item else None

INFERENCE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="detector")
class InvalidImage(ValueError): pass

def decode(data):
    try:
        image=Image.open(io.BytesIO(data))
        if image.width*image.height>20_000_000: raise InvalidImage('Image exceeds 20 megapixels.')
        image.close()
        decoded=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if decoded is None: raise InvalidImage('Image could not be decoded.')
        return decoded
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
        raise InvalidImage('Image could not be decoded.') from error

def clamp(box,width,height,padding=.08):
    x,y,w,h=box;px,py=int(w*padding),int(h*padding)
    x1,y1=max(0,int(x-px)),max(0,int(y-py))
    x2,y2=min(width,int(x+w+px)),min(height,int(y+h+py))
    return x1,y1,max(0,x2-x1),max(0,y2-y1)

def process(data,detector,store,ocr=None,vehicle_detector=None,profile="detailed"):
    started=time.perf_counter();image=decode(data);h,w=image.shape[:2]
    decode_ms=(time.perf_counter()-started)*1000
    detection_started=time.perf_counter()
    request_id=str(uuid.uuid4());detections=[];ocr_ms=0
    vehicle_future=INFERENCE_POOL.submit(vehicle_detector.objects,image) if vehicle_detector else None
    context_warning=None
    try:
        boxes=detector.detect_fast(image) if profile=='fast' and hasattr(detector,'detect_fast') else (detector.detect_plates(image) if hasattr(detector,'detect_plates') else detector.detect(image))
    finally:
        try:
            objects=vehicle_future.result() if vehicle_future else []
        except Exception:
            objects=[]
            context_warning='Optional vehicle context failed; plate results remain available.'
    detection_ms=(time.perf_counter()-detection_started)*1000
    prepared=[]
    for x,y,bw,bh,score in boxes:
        tx,ty,tw,th=clamp((x,y,bw,bh),w,h,padding=0)
        cx,cy,cw,ch=clamp((x,y,bw,bh),w,h,padding=.05)
        if tw and th:
            prepared.append(((x,y,bw,bh,score),image[cy:cy+ch,cx:cx+cw],image[ty:ty+th,tx:tx+tw]))
    batch_readings=None
    if profile=='fast' and ocr and hasattr(ocr,'read_many') and not isinstance(detector,MockPlateDetector):
        ocr_started=time.perf_counter()
        batch_readings=ocr.read_many([(context,tight) for _,context,tight in prepared])
        ocr_ms=(time.perf_counter()-ocr_started)*1000
    for index,((x,y,bw,bh,score),_,_) in enumerate(prepared):
        tx,ty,tw,th=clamp((x,y,bw,bh),w,h,padding=0)
        x,y,bw,bh=clamp((x,y,bw,bh),w,h,padding=settings.plate_crop_padding)
        if not bw or not bh:continue
        crop=image[y:y+bh,x:x+bw];detection_id=str(uuid.uuid4())
        ok,encoded=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:continue
        ocr_started=time.perf_counter()
        if batch_readings is not None:raw,confidence=batch_readings[index]
        elif isinstance(detector,MockPlateDetector):raw,confidence='GJ01AB1234',.65
        elif ocr and hasattr(ocr,'read_plate'):raw,confidence=ocr.read_plate(crop,image[ty:ty+th,tx:tx+tw])
        else:raw,confidence=ocr.read(crop) if ocr else ('',0)
        if batch_readings is None:ocr_ms+=(time.perf_counter()-ocr_started)*1000
        normalized,formatted,status=normalize_plate(raw)
        if confidence<.7:status='uncertain'
        # Retain high-confidence regions even when text is unreadable. Weak
        # proposals need OCR support; an arbitrary word is not a registration.
        text_supported=(confidence>=.7 and 6<=len(normalized)<=12
                        and any(c.isalpha() for c in normalized)
                        and any(c.isdigit() for c in normalized))
        if ocr and score<settings.unverified_plate_threshold and not text_supported:
            continue
        store.put((request_id,detection_id),encoded.tobytes())
        detections.append(dict(id=detection_id,bounding_box=dict(x=x,y=y,width=bw,height=bh),normalized_box=dict(x=x/w,y=y/h,width=bw/w,height=bh/h),detection_confidence=score,raw_text=raw,normalized_text=normalized,formatted_text=formatted,ocr_confidence=confidence,format_status=status,crop_url=f'/api/results/{request_id}/{detection_id}/crop'))
    labels={2:'car',3:'motorcycle',5:'bus',7:'truck'}
    vehicles=[dict(bounding_box=dict(x=x,y=y,width=bw,height=bh),confidence=score,label=labels.get(cls,'vehicle')) for x,y,bw,bh,score,cls in objects]
    # Each plate belongs to the smallest containing vehicle, so overlapping
    # vehicle boxes cannot claim the same registration.
    for index, vehicle in enumerate(vehicles):
        vehicle['id']=f'vehicle-{index+1}'
        vehicle['plate_ids']=[]
    for detection in detections:
        b=detection['bounding_box'];cx,cy=b['x']+b['width']/2,b['y']+b['height']/2
        owners=[v for v in vehicles if v['bounding_box']['x'] <= cx <= v['bounding_box']['x']+v['bounding_box']['width']
                and v['bounding_box']['y'] <= cy <= v['bounding_box']['y']+v['bounding_box']['height']]
        owner=min(owners,key=lambda v:v['bounding_box']['width']*v['bounding_box']['height']) if owners else None
        detection['vehicle_id']=owner['id'] if owner else None
        if owner:owner['plate_ids'].append(detection['id'])
    warnings=[context_warning] if context_warning else []
    if settings.baseline_model and not isinstance(detector,MockPlateDetector):warnings.append('Baseline model: CCTV accuracy is not sufficient for unattended operation. Verify all readings.')
    if isinstance(detector,MockPlateDetector):warnings.append('DEMO MODE: synthetic results, not real detections.')
    elif not ocr:warnings.append('OCR unavailable: plate regions only.')
    return dict(profile=profile,decode_time_ms=round(decode_ms,2),latency_target_ms=40,confidence_note='Detector and OCR scores are uncalibrated model scores, not measured accuracy.',request_id=request_id,processing_time_ms=round((time.perf_counter()-started)*1000,2),detection_time_ms=round(detection_ms,2),ocr_time_ms=round(ocr_ms,2),image=dict(width=w,height=h),search_strategy='high_resolution_whole_frame' if profile=='fast' and getattr(detector,'fast_models',None) else 'whole_frame_and_overlapping_tiles',detections=detections,vehicles=vehicles,vehicle_detector_ready=vehicle_detector is not None,warnings=warnings)
