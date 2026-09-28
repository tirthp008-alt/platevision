"""Regression checks for annotation correctness and evaluation isolation."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from import_plate_archive import assign_splits,parse_annotation,prepare
from build_combined_plate_dataset import context_region
from evaluate_plate_candidate import match


def record(number,group=None,text='',review=''):
    return dict(id=str(number),image_sha256=f'hash{number}',vehicle_group=group,
                plates=[dict(text=text)],review_required=review)


def test_related_photos_and_crops_cannot_cross_splits():
    rows=[record(i) for i in range(24)]
    rows[0]['vehicle_group']=rows[1]['vehicle_group']='same motorcycle'
    rows[2]['plates']=rows[3]['plates']=[dict(text='AB12CD3456')]
    rows[4]['image_sha256']=rows[5]['image_sha256']='duplicate photo'
    result=assign_splits(deepcopy(rows))
    assert all(result[a]['split']==result[b]['split'] for a,b in [(0,1),(2,3),(4,5)])
    assert result==assign_splits(deepcopy(rows))
    assert {r['split'] for r in result}=={'train','val','test'}


def test_incomplete_annotations_quarantine_related_images():
    rows=[record(i) for i in range(24)]
    rows[0]['vehicle_group']=rows[1]['vehicle_group']='same scene'
    rows[1]['review_required']='missing plate labels'
    result=assign_splits(rows)
    assert result[0]['split']==result[1]['split']=='review'


def annotation(width=80,height=40,xmax=60):
    return f'''<annotation><size><width>{width}</width><height>{height}</height></size>
      <object><name>number_plate</name><bndbox><xmin>20</xmin><ymin>10</ymin>
      <xmax>{xmax}</xmax><ymax>30</ymax></bndbox></object></annotation>'''.encode()


def test_bad_boxes_and_xml_entities_are_rejected():
    with pytest.raises(ValueError):parse_annotation(annotation(xmax=100))
    with pytest.raises(ValueError):parse_annotation(b'<!DOCTYPE x>'+annotation())
    size,plates=parse_annotation(annotation())
    assert size==(80,40) and plates[0]['text_status']=='missing'
    assert not plates[0]['transcription_verified']


def test_orientation_is_applied_before_annotation_coordinates(tmp_path):
    archive=tmp_path/'photos.zip'
    with ZipFile(archive,'w') as z:
        for i in range(8):
            # Stored portrait; EXIF rotation makes the labelled image landscape.
            image=Image.new('RGB',(40,80),(i*25,0,0));exif=Image.Exif();exif[274]=6
            buffer=BytesIO();image.save(buffer,format='JPEG',exif=exif)
            z.writestr(f'images/{i}.jpg',buffer.getvalue());z.writestr(f'labels/{i}.xml',annotation())
    output=tmp_path/'dataset';report=prepare(archive,output,include_existing=False)
    assert report['orientation_corrections']==8
    for row in report['records']:
        assert Image.open(output/'oriented'/f"{row['id']}.jpg").size==(80,40)
        values=(output/row['split']/'labels'/f"{row['id']}.txt").read_text().split()
        assert [float(v) for v in values]==[0,.5,.5,.5,.5]


def test_context_region_never_clips_its_target_plate():
    for box in ([0,0,50,15],[80,70,100,100],[10,40,90,60]):
        l,t,r,b=context_region(box,100,100)
        assert 0<=l<=box[0]<box[2]<=r<=100
        assert 0<=t<=box[1]<box[3]<=b<=100


def test_duplicate_predictions_cannot_inflate_recall():
    truth=[[10,10,30,10],[70,10,30,10]]
    assert match([(10,10,30,10,.9),(10,10,30,10,.8)],truth)==(1,1,1)
