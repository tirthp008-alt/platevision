import io
import numpy as np
from PIL import Image
from app.services.pipeline import process, CropStore, decode
from app.services.ocr import PlateOCR
from app.services.model_shape import anchor_grid


def test_fast_pipeline_reads_every_plate_and_keeps_separate_scores():
    class Detector:
        fast_models=True
        def detect_fast(self,image):return [(i*30,10,25,8,.91) for i in range(12)]
    class OCR:
        def read_many(self,pairs):
            assert len(pairs)==12
            return [('MH12AB1234',.82)]*12
    data=io.BytesIO();Image.new('RGB',(400,60)).save(data,format='PNG')
    result=process(data.getvalue(),Detector(),CropStore(),OCR(),profile='fast')
    assert len(result['detections'])==12
    assert all(d['detection_confidence']==.91 and d['ocr_confidence']==.82 for d in result['detections'])
    assert result['search_strategy']=='high_resolution_whole_frame'
    assert result['processing_time_ms']>=result['ocr_time_ms']


def test_fast_ocr_does_not_invent_confident_text_on_failed_retry():
    class Batch:
        def recognize(self,crops):return [('BLUR',.2)]*len(crops)
    o=object.__new__(PlateOCR);o.batch=Batch()
    crop=np.zeros((20,100,3),np.uint8)
    result=o.read_many([(crop,crop)]*12)
    assert result==[('BLUR',.2)]*12


def test_decode_honours_exif_rotation():
    data=io.BytesIO();image=Image.new('RGB',(40,20));exif=Image.Exif();exif[274]=6
    image.save(data,format='JPEG',exif=exif)
    assert decode(data.getvalue()).shape[:2]==(40,20)


def test_rectangular_anchor_grid_has_expected_strides_and_extent():
    points,strides=anchor_grid(864,1536)
    assert points.shape==(1,2,27216)
    assert strides.shape==(1,27216)
    np.testing.assert_array_equal(points[0,:,0],[.5,.5])
    np.testing.assert_array_equal(points[0,:,108*192-1],[191.5,107.5])
    assert strides[0,0]==8 and strides[0,-1]==32
