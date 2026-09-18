import io
import numpy as np
import pytest
from PIL import Image
from app.services.detector import OnnxPlateDetector
from app.services.pipeline import clamp, CropStore, process, InvalidImage


def test_padding_preserves_original_right_edge():
    assert clamp((-10,-5,100,40),200,100)==(0,0,98,38)

def test_nms_keeps_two_plates_and_clips_bounds():
    detector=object.__new__(OnnxPlateDetector);detector.classes=1;detector.allowed=None
    output=np.array([[[50,51,180],[30,31,60],[80,80,80],[20,20,20],[.9,.8,.95]]],dtype=np.float32)
    result=detector.postprocess(output,1,0,0,200,100)
    assert len(result)==2
    assert result[0][:4]==(140,50,60,20)

def test_output_adapter_rejects_unknown_layout():
    detector=object.__new__(OnnxPlateDetector);detector.classes=1;detector.allowed=None
    with pytest.raises(RuntimeError):detector.postprocess(np.zeros((1,20,6)),1,0,0,100,100)

def test_bounded_expiring_crops(monkeypatch):
    import app.services.pipeline as pipeline
    now=[100];monkeypatch.setattr(pipeline.time,'monotonic',lambda:now[0])
    store=CropStore(ttl=5,limit=2)
    for key in 'abc':store.put(key,b'data')
    assert store.get('a') is None and store.get('c')==b'data'
    now[0]=106
    assert store.get('c') is None

def test_multiple_plate_pipeline_uses_ocr_and_never_invents_text():
    class Detector:
        def detect(self,image):return [(10,10,60,20,.9),(100,60,60,20,.8)]
    class OCR:
        def read(self,crop):return 'MH12AB1234',.95
    data=io.BytesIO();Image.new('RGB',(200,100)).save(data,format='PNG')
    store=CropStore()
    result=process(data.getvalue(),Detector(),store,OCR())
    assert len(result['detections'])==2
    assert all(item['normalized_text']=='MH12AB1234' for item in result['detections'])
    result=process(data.getvalue(),Detector(),store)
    assert all(item['raw_text']=='' for item in result['detections'])
    with pytest.raises(InvalidImage):process(b'bad image',Detector(),store)

def test_api_errors_are_distinct(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    client=TestClient(main.app)
    monkeypatch.setattr(main,'detector',None)
    assert client.post('/api/detect/frame',files={'image':('x.png',b'bad','image/png')}).status_code==503
    assert client.post('/api/detect/image',files={'image':('x.txt',b'bad','text/plain')}).status_code==415
    class Detector:
        def detect(self,image):raise RuntimeError('bad model output')
    monkeypatch.setattr(main,'detector',Detector())
    assert client.post('/api/detect/frame',files={'image':('x.png',b'bad','image/png')}).status_code==422
    data=io.BytesIO();Image.new('RGB',(100,100)).save(data,format='PNG')
    assert client.post('/api/detect/image',files={'image':('x.png',data.getvalue(),'image/png')}).status_code==500

def test_vehicle_classes_do_not_double_count_same_vehicle():
    detector=object.__new__(OnnxPlateDetector);detector.classes=80;detector.allowed={2,3,5,7}
    output=np.zeros((1,84,2),np.float32)
    output[0,:4,:]=np.array([[50,51],[50,51],[60,60],[40,40]])
    output[0,4+5,0]=.9;output[0,4+7,1]=.8
    result=detector.postprocess(output,1,0,0,200,100)
    assert len(result)==1 and result[0][-1]==5

def test_tile_starts_cover_edges_with_overlap():
    starts=OnnxPlateDetector.tile_starts(1600,640,.25)
    assert starts==[0,480,960]
    assert OnnxPlateDetector.tile_starts(1200,640,.25)==[0,480,560]
    assert OnnxPlateDetector.tile_starts(320,640,.25)==[0]


def test_tiling_detects_plates_without_any_vehicle_input(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings,'plate_tile_size',640)
    detector=object.__new__(OnnxPlateDetector)
    calls=[]
    def detect(crop,threshold=None):
        calls.append(crop.shape)
        if crop.shape[1]>640:return []
        return [(30,100,100,40,.8)]
    detector.detect=detect
    result=detector.detect_plates(np.zeros((640,1600,3),np.uint8))
    assert len(result)==3
    assert [b[0] for b in result]==[30,510,990]
    assert len(calls)==4


def test_tile_merge_keeps_separate_plates_and_drops_stacked_fragment():
    boxes=[(10,10,100,100,.8),(10,10,100,48,.9),(130,10,100,100,.8)]
    result=OnnxPlateDetector.merge_plates(boxes)
    assert len(result)==2
    assert result[0][:4]==(10,10,100,100)


def test_internal_tile_edge_fragment_is_discarded(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings,'plate_tile_size',640)
    detector=object.__new__(OnnxPlateDetector)
    def detect(crop,threshold=None):
        return [] if crop.shape[1]>640 else [(0,50,60,20,.8)]
    detector.detect=detect
    result=detector.detect_plates(np.zeros((640,1600,3),np.uint8))
    assert len(result)==1 and result[0][0]==0


def test_plate_pipeline_ignores_vehicle_detector_failure():
    class Detector:
        def detect_plates(self,image):return [(20,30,60,20,.8)]
        def detect(self,image):raise AssertionError('Must use full-image plate search')
    class Vehicles:
        def objects(self,image):raise RuntimeError('vehicle model failed')
    data=io.BytesIO();Image.new('RGB',(200,100)).save(data,format='PNG')
    result=process(data.getvalue(),Detector(),CropStore(),vehicle_detector=Vehicles())
    assert len(result['detections'])==1 and result['vehicles']==[]
    assert 'vehicle context failed' in result['warnings'][0]


def test_plate_association_uses_single_smallest_vehicle():
    class Detector:
        def detect(self,image):return [(80,50,30,15,.9),(220,60,30,15,.9)]
    class Vehicles:
        def objects(self,image):return [(0,0,180,100,.9,2),(50,20,90,70,.9,2),(200,10,90,80,.8,2)]
    data=io.BytesIO();Image.new('RGB',(300,120)).save(data,format='PNG')
    result=process(data.getvalue(),Detector(),CropStore(),vehicle_detector=Vehicles())
    assert [d['vehicle_id'] for d in result['detections']]==['vehicle-2','vehicle-3']
    assert [len(v['plate_ids']) for v in result['vehicles']]==[0,1,1]

def test_uncertain_ocr_retries_localization():
    from app.services.ocr import PlateOCR
    reader=object.__new__(PlateOCR)
    calls=[]
    def engine(crop,**kwargs):
        calls.append(kwargs)
        if kwargs.get('use_det') is False:return [['MH12AB134',.6]],None
        return [[[[0,0],[60,0],[60,20],[0,20]],'MH12AB1234',.96]],None
    reader.engine=engine
    assert reader.read(np.zeros((20,100,3),np.uint8))==('MH12AB1234',.96)
    assert len(calls)==2


def test_stacked_plate_ocr_ignores_brand_and_small_country_mark():
    from app.services.ocr import PlateOCR
    reader=object.__new__(PlateOCR)
    reader.engine=lambda *args,**kwargs:([
        [[[0,0],[65,0],[65,20],[0,20]],'HONDA',.99],
        [[[10,30],[100,30],[100,70],[10,70]],'GJ27D',.98],
        [[[0,65],[10,65],[10,75],[0,75]],'IND',.99],
        [[[10,75],[100,75],[100,115],[10,115]],'S4837',.97]],None)
    text,score=reader.read(np.zeros((120,130,3),np.uint8))
    assert text=='GJ27DS4837' and score>.97


def test_weak_large_box_cannot_replace_two_strong_neighbouring_plates():
    boxes=[(0,0,250,80,.26),(10,10,80,30,.9),(145,10,80,30,.91)]
    result=OnnxPlateDetector.merge_plates(boxes)
    assert (10,10,80,30,.9) in result and (145,10,80,30,.91) in result


def test_refinement_maps_to_original_plate_and_excludes_unrelated_objects():
    detector=object.__new__(OnnxPlateDetector)
    # 192px crop centred on (120,110) has origin (24,14).
    detector.detect=lambda crop:[(76,86,40,20,.9),(0,0,30,15,.9)]
    result=detector.refine_plates(np.zeros((400,600,3),np.uint8),[(100,100,40,20,.3)])
    assert result==[(100,100,40,20,.9)]


def test_weak_proposal_is_promoted_only_after_local_model_confirmation(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings,'plate_refinement_enabled',True)
    detector=object.__new__(OnnxPlateDetector)
    def detect(image,threshold=None):
        if image.shape[0]==300:return [(100,100,40,20,.18)]
        return [(76,86,40,20,.8)]
    detector.detect=detect
    assert detector.detect_plates(np.zeros((300,300,3),np.uint8))==[(100,100,40,20,.8)]


def test_weak_false_candidate_filtered_but_readable_and_strong_regions_retained():
    class Detector:
        def detect(self,image):return [(10,10,50,20,.3),(100,10,50,20,.3),(200,10,50,20,.8)]
    class OCR:
        def __init__(self):self.rows=iter([('HONDA',.99),('MH12AB1234',.92),('',0)])
        def read(self,crop):return next(self.rows)
    data=io.BytesIO();Image.new('RGB',(300,100)).save(data,format='PNG')
    result=process(data.getvalue(),Detector(),CropStore(),OCR())
    assert [d['normalized_text'] for d in result['detections']]==['MH12AB1234','']
    assert len(process(data.getvalue(),Detector(),CropStore())['detections'])==3


def test_slanted_same_line_text_is_read_left_to_right():
    from app.services.ocr import PlateOCR
    rows=[[[[0,5],[35,5],[35,25],[0,25]],'MH12',.95],
          [[[40,0],[100,0],[100,20],[40,20]],'AB1234',.95]]
    assert PlateOCR.join_rows(rows)[0]=='MH12AB1234'


def test_tight_crop_can_rescue_context_ocr_without_inventing_characters():
    from app.services.ocr import PlateOCR
    reader=object.__new__(PlateOCR)
    readings=iter([('127EB2006',.64),('GJ27EBZ005',.84)])
    reader.read=lambda crop:next(readings)
    assert reader.read_plate(np.zeros((20,80,3),np.uint8),np.zeros((15,60,3),np.uint8))==('GJ27EBZ005',.84)


def test_valid_localized_text_wins_over_high_confidence_incomplete_fast_text():
    from app.services.ocr import PlateOCR
    assert PlateOCR.rank(('MH12AB1234',.92))>PlateOCR.rank(('MH12AB123',.98))
    assert PlateOCR.rank(('MH12AB1234',.4))<PlateOCR.rank(('MH12AB123',.8))
