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
from app.services.rectification import rectify_plate
from app.services.vehicles import attach_vehicles
from app.services.vehicle_plate_search import recover_vehicle_plates
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

def process(data,detector,store,ocr=None,vehicle_detector=None,profile="detailed",green_filter=False,read_text=True,angle_correction=False,reid_context=None):
    started=time.perf_counter();image=decode(data);h,w=image.shape[:2]
    decode_ms=(time.perf_counter()-started)*1000
    detection_started=time.perf_counter()
    request_id=str(uuid.uuid4());detections=[];ocr_ms=0;crop_ms=0;rectification_ms=0
    vehicle_timing={}
    def detect_vehicles():
        vehicle_started=time.perf_counter()
        try:
            return vehicle_detector.objects(image)
        finally:
            vehicle_timing['ms']=(time.perf_counter()-vehicle_started)*1000
    vehicle_future=INFERENCE_POOL.submit(detect_vehicles) if vehicle_detector else None
    context_warning=None
    try:
        if hasattr(detector,'detect_regions'):
            boxes=detector.detect_regions(image,profile=profile)
        else:
            boxes=detector.detect_fast(image) if profile=='fast' and hasattr(detector,'detect_fast') else (detector.detect_plates(image) if hasattr(detector,'detect_plates') else detector.detect(image))
    finally:
        try:
            objects=vehicle_future.result() if vehicle_future else []
        except Exception:
            objects=[]
            context_warning='Optional vehicle context failed; plate results remain available.'
    # Search unmatched vehicle crops at the plate model's native resolution.
    # OCR is downstream and cannot decide whether a recovered box is retained.
    original_box_count=len(boxes)
    boxes,vehicle_plate_search=recover_vehicle_plates(image,detector,boxes,objects)
    recovered_boxes={tuple(box) for box in boxes[original_box_count:]}
    detection_ms=(time.perf_counter()-detection_started)*1000
    prepared=[]
    # The detector may clip an edge character. Give the detail reader the same
    # contextual margin as the saved crop, while keeping the original tight box
    # as its first OCR input. No scene box is expanded or synthesized for recall.
    context_padding=settings.plate_crop_padding if read_text and hasattr(ocr,'read_many_details') else .05
    for x,y,bw,bh,score in boxes:
        if not np.isfinite([x,y,bw,bh,score]).all() or bw<=0 or bh<=0:
            continue
        tx,ty,tw,th=clamp((x,y,bw,bh),w,h,padding=0)
        cx,cy,cw,ch=clamp((x,y,bw,bh),w,h,padding=context_padding)
        if tw and th:
            prepared.append(((x,y,bw,bh,score),image[cy:cy+ch,cx:cx+cw],image[ty:ty+th,tx:tx+tw]))
    batch_readings=None
    batch_details=None
    if read_text and ocr and hasattr(ocr,'read_many_details') and not isinstance(detector,MockPlateDetector):
        ocr_started=time.perf_counter()
        batch_details=ocr.read_many_details([(context,tight) for _,context,tight in prepared],angle_correction=angle_correction)
        if len(batch_details)!=len(prepared):
            raise ValueError('OCR returned an incomplete plate batch.')
        batch_readings=[(item['text'],item['confidence']) for item in batch_details]
        ocr_ms=(time.perf_counter()-ocr_started)*1000
    elif read_text and profile=='fast' and ocr and hasattr(ocr,'read_many') and not isinstance(detector,MockPlateDetector):
        ocr_started=time.perf_counter()
        batch_readings=ocr.read_many([(context,tight) for _,context,tight in prepared])
        ocr_ms=(time.perf_counter()-ocr_started)*1000
    for index,((x,y,bw,bh,score),context,_) in enumerate(prepared):
        tx,ty,tw,th=clamp((x,y,bw,bh),w,h,padding=0)
        x,y,bw,bh=clamp((x,y,bw,bh),w,h,padding=settings.plate_crop_padding)
        if not bw or not bh:continue
        crop=image[y:y+bh,x:x+bw];detection_id=str(uuid.uuid4())
        crop_started=time.perf_counter()
        try:
            ok,encoded=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,85])
        except cv2.error:
            ok,encoded=False,None
        crop_ms+=(time.perf_counter()-crop_started)*1000
        raw=normalized=formatted='';confidence=None;status='not_requested';ocr_review=None
        if read_text:
            ocr_started=time.perf_counter()
            if batch_readings is not None:
                raw,confidence=batch_readings[index]
                if batch_details is not None:ocr_review=batch_details[index]
            elif isinstance(detector,MockPlateDetector):raw,confidence='GJ01AB1234',.65
            elif ocr and hasattr(ocr,'read_plate'):raw,confidence=ocr.read_plate(crop,image[ty:ty+th,tx:tx+tw])
            else:raw,confidence=ocr.read(crop) if ocr else ('',0)
            if batch_readings is None:ocr_ms+=(time.perf_counter()-ocr_started)*1000
            normalized,formatted,status=normalize_plate(raw)
            if confidence<.8 or (ocr_review and ocr_review.get('requires_review')):status='uncertain'
        # OCR annotates every independently detected plate region. An unreadable
        # or conflicting reading must never remove its original box or crop.
        text_supported=(read_text and confidence>=.7 and 6<=len(normalized)<=12
                        and any(c.isalpha() for c in normalized)
                        and any(c.isdigit() for c in normalized))
        enhanced,comparison=None,None
        if green_filter and read_text:
            from app.services.experimental_ocr import green_alternative
            enhanced,comparison=green_alternative(ocr,image[ty:ty+th,tx:tx+tw],(raw,confidence))
            ocr_ms+=comparison['ocr_ms']
        otherwise_filtered=bool(read_text and ocr and score<settings.unverified_plate_threshold and not text_supported)
        review_only=bool(otherwise_filtered and comparison and comparison.get('applied'))
        if otherwise_filtered or review_only:status='uncertain'
        crop_url=None
        if ok:
            store.put((request_id,detection_id),encoded.tobytes())
            crop_url=f'/api/results/{request_id}/{detection_id}/crop'
        detections.append(dict(id=detection_id,bounding_box=dict(x=x,y=y,width=bw,height=bh),tight_plate_box=dict(x=tx,y=ty,width=tw,height=th),normalized_box=dict(x=x/w,y=y/h,width=bw/w,height=bh/h),detection_confidence=score,raw_text=raw,normalized_text=normalized,formatted_text=formatted,ocr_confidence=confidence,format_status=status,crop_url=crop_url))
        recovered=tuple(prepared[index][0]) in recovered_boxes
        detections[-1]['localization_source']='vehicle_crop' if recovered else 'whole_frame'
        detections[-1]['localization_requires_review']=recovered
        if ocr_review is not None:
            detections[-1]['ocr_review']={key:value for key,value in ocr_review.items()
                                         if key!='rectified_crop' and not key.startswith('_')}
        if angle_correction:
            correction_started=time.perf_counter()
            if ocr_review is not None and ocr_review.get('rectified_crop') is not None:
                corrected,rectification=ocr_review['rectified_crop'],ocr_review['rectification']
            elif (ocr_review is not None
                  and ocr_review.get('_rectification_attempt')==dict(source='context',transform='rectify_plate')
                  and isinstance(ocr_review.get('rectification'),dict)
                  and ocr_review['rectification'].get('applied') is False
                  and context.shape==crop.shape and np.array_equal(context,crop)):
                # Reuse a failed attempt only for this exact preview input.
                # Missing metadata means OCR skipped it, so the preview still runs.
                corrected,rectification=None,ocr_review['rectification']
            else:
                corrected,rectification=rectify_plate(crop)
            rectification_ms+=(time.perf_counter()-correction_started)*1000
            detections[-1]['rectification']=rectification
            detections[-1]['rectified_crop_url']=None
            if rectification['applied']:
                crop_started=time.perf_counter()
                try:
                    corrected_ok,corrected_encoded=cv2.imencode('.jpg',corrected,[cv2.IMWRITE_JPEG_QUALITY,90])
                except cv2.error:
                    corrected_ok,corrected_encoded=False,None
                if corrected_ok:
                    corrected_id=detection_id+'-rectified'
                    store.put((request_id,corrected_id),corrected_encoded.tobytes())
                    detections[-1]['rectified_crop_url']=f'/api/results/{request_id}/{corrected_id}/crop'
                crop_ms+=(time.perf_counter()-crop_started)*1000
        if review_only:
            detections[-1]['experimental_review_only']=True
        if green_filter and read_text:
            if enhanced is not None:
                encoded_ok, encoded_enhanced=cv2.imencode('.jpg',enhanced,[cv2.IMWRITE_JPEG_QUALITY,90])
                if encoded_ok:
                    enhanced_id=detection_id+'-emboss'
                    store.put((request_id,enhanced_id),encoded_enhanced.tobytes())
                    comparison['crop_url']=f'/api/results/{request_id}/{enhanced_id}/crop'
            detections[-1]['green_enhancement']=comparison
    vehicles,vehicle_summary=attach_vehicles(detections,objects,w,h,available=vehicle_detector is not None and context_warning is None,taxonomy=getattr(vehicle_detector,'taxonomy','coco'))
    warnings=[context_warning] if context_warning else []
    reidentification = None
    if reid_context is not None:
        from app.services.reidentification import match_observations, strongest_plate
        observations = []
        if reid_context.get('engine') is not None:
            by_id = {plate['id']:plate for plate in detections}
            for vehicle in vehicles:
                vx,vy,vw,vh = clamp(vehicle['box'],w,h,padding=.02)
                if min(vw,vh) < 16:
                    continue
                text,confidence = strongest_plate([by_id[plate_id] for plate_id in vehicle['plate_ids'] if plate_id in by_id]) if read_text else (None,0.)
                observations.append(dict(crop=image[vy:vy+vh,vx:vx+vw],
                                         track_id=f"{request_id}:{vehicle['id']}",
                                         observed_at=reid_context['observed_at'],
                                         category=vehicle['category'],plate_text=text,plate_confidence=confidence))
        reidentification = match_observations(observations,reid_context)
        if reidentification.get('warning'):
            warnings.append(reidentification['warning'])
    if green_filter and read_text:
        warnings.append('Experimental emboss + gradient comparison is separate from the original reading. It adds OCR time and can reduce accuracy.')
    elif green_filter:
        warnings.append('Experimental OCR comparison is disabled while text recognition is off.')
    if angle_correction:
        warnings.append('Angle correction applies only to captured plate crops with a credible outline; it cannot recover hidden plates or invisible characters.')
    if settings.baseline_model and not isinstance(detector,MockPlateDetector):warnings.append('Camera-specific CCTV accuracy has not been established. Verify detected regions.' if not read_text else 'Camera-specific CCTV accuracy has not been established. Verify all readings.')
    if isinstance(detector,MockPlateDetector):warnings.append('DEMO MODE: synthetic results, not real detections.')
    elif read_text and not ocr:warnings.append('OCR unavailable: plate regions only.')
    from app.services.plate_summary import summarize_recognized_plates
    recognized_plate_summary=summarize_recognized_plates(detections,read_text=read_text,scope='frame')
    return dict(detector_model=getattr(detector,'model_id','Plate detector'),weights_retrained=bool(getattr(detector,'weights_retrained',False)),profile=profile,read_text=bool(read_text),angle_correction=bool(angle_correction),plate_count=len(detections),recognized_plate_summary=recognized_plate_summary,recognized_plates=recognized_plate_summary['groups'],rectification_time_ms=round(rectification_ms,2),crop_time_ms=round(crop_ms,2),decode_time_ms=round(decode_ms,2),latency_target_ms=40,confidence_note='Detector and OCR scores are uncalibrated model scores, not measured accuracy.',request_id=request_id,processing_time_ms=round((time.perf_counter()-started)*1000,2),detection_time_ms=round(detection_ms,2),vehicle_detection_time_ms=round(vehicle_timing['ms'],2) if 'ms' in vehicle_timing else None,vehicle_plate_search=vehicle_plate_search,ocr_time_ms=round(ocr_ms,2),image=dict(width=w,height=h),search_strategy='high_resolution_whole_frame' if profile=='fast' and getattr(detector,'fast_models',None) else 'whole_frame_and_overlapping_tiles',detections=detections,vehicles=vehicles,vehicle_summary=vehicle_summary,vehicle_detector_ready=vehicle_summary['available'],warnings=warnings,reidentification=reidentification)
