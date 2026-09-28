"""Validate a VOC ZIP and prepare grouped train/val/test splits without executing it."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def parse_annotation(data):
    if len(data)>1_000_000:
        raise ValueError('Annotation exceeds import limit.')
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('XML document types and entities are not accepted.')
    root = ET.fromstring(data)
    width, height = (int(root.findtext('size/'+name)) for name in ('width','height'))
    if not 0 < width * height <= 20_000_000:
        raise ValueError('Unsupported image dimensions.')
    boxes = []
    for obj in root.findall('object'):
        if obj.findtext('name') != 'number_plate':
            raise ValueError('Unrecognized class in annotation.')
        box = [float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
        x1,y1,x2,y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'Invalid plate box: {box}')
        text = obj.findtext("attributes/attribute[name='number_plate_text']/value", default='')
        clean = re.sub('[^A-Z0-9]','',text.upper())
        boxes.append(dict(box=box,text=clean,source_text=text,
                          text_status='provided' if len(clean)>=8 else 'partial' if clean else 'missing',
                          transcription_verified=False))
    if not boxes:
        raise ValueError('Annotation contains no plates; review negative images explicitly.')
    return (width,height),boxes


def assign_splits(records, seed=42):
    """Keep exact duplicates, shared text and manually reviewed vehicles together."""
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    owners = {}
    for i,record in enumerate(records):
        keys = ['sha:'+record['image_sha256']]
        if record.get('vehicle_group'):keys.append('vehicle:'+record['vehicle_group'])
        keys += ['text:'+b['text'] for b in record['plates'] if len(b['text'])>=8]
        for key in keys:
            if key in owners:parents[find(i)] = find(owners[key])
            else:owners[key] = i
    groups = {}
    for i,record in enumerate(records):groups.setdefault(find(i),[]).append(record)
    usable = [g for g in groups.values() if not any(r.get('review_required') for r in g)]
    random.Random(seed).shuffle(usable)
    reserved = max(1,round(len(usable)*.16))
    if len(usable)<6:raise ValueError('Need at least six independent groups for train/val/test.')
    for index, group in enumerate(usable):
        split = 'test' if index<reserved else 'val' if index<2*reserved else 'train'
        for record in group:record['split']=split;record['group']=min(r['id'] for r in group)
    for group in groups.values():
        if any(r.get('review_required') for r in group):
            for record in group:record['split']='review';record['group']=min(r['id'] for r in group)
    return records


def prepare(archive, destination, review=None, include_existing=True):
    archive = Path(archive);destination = Path(destination)
    if destination.exists():raise FileExistsError(f'Refusing to overwrite an existing dataset: {destination}')
    review = review or {}
    records = []
    with ZipFile(archive) as source:
        infos = source.infolist()
        if len(infos)>50_000 or sum(i.file_size for i in infos)>5_000_000_000:
            raise ValueError('Archive exceeds dataset import limits.')
        images = {}
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
                raise ValueError('Unsafe archive member path.')
            if path.suffix.lower() in {'.jpg','.jpeg','.png'}:
                if path.stem in images:raise ValueError('Ambiguous duplicate image name.')
                images[path.stem]=info
        annotations = sorted(i.filename for i in infos if i.filename.lower().endswith('.xml'))
        if len(annotations)!=len(images):raise ValueError('Every image needs a matching annotation.')
        for index,name in enumerate(annotations):
            stem=PurePosixPath(name).stem
            if source.getinfo(name).file_size>1_000_000:raise ValueError('Annotation exceeds import limit.')
            size,plates=parse_annotation(source.read(name))
            info=images.pop(stem)
            if info.file_size>40_000_000:raise ValueError('Individual image is too large.')
            data=source.read(info)
            with Image.open(BytesIO(data)) as raw:
                orientation=raw.getexif().get(274,1)
                image=ImageOps.exif_transpose(raw).convert('RGB')
            if image.size!=size:raise ValueError(f'Annotation dimensions do not match oriented photo: {name}')
            ident=f'archive_{index+1:03}'
            record=dict(id=ident,source_image=info.filename,source_annotation=name,image_sha256=sha256(data).hexdigest(),
                        width=size[0],height=size[1],orientation_corrected=orientation in {2,3,4,5,6,7,8},plates=plates,
                        **review.get(stem,{}))
            # Never extract arbitrary archive paths. Only generate known image/crop files.
            folder=destination/'oriented';folder.mkdir(parents=True,exist_ok=True)
            image.save(folder/f'{ident}.jpg',quality=95)
            crops=destination/'ocr-crops';crops.mkdir(exist_ok=True)
            for number,plate in enumerate(plates):
                plate['crop']=f'ocr-crops/{ident}_{number}.jpg'
                image.crop(tuple(plate['box'])).save(destination/plate['crop'],quality=98)
            records.append(record)
    assign_splits(records)
    for split in ('train','val','test'):
        for kind in ('images','labels'):(destination/split/kind).mkdir(parents=True,exist_ok=True)
    for record in records:
        split=record['split']
        if split=='review':continue
        shutil.copyfile(destination/'oriented'/f"{record['id']}.jpg",destination/split/'images'/f"{record['id']}.jpg")
        labels=[];w=record['width'];h=record['height']
        for plate in record['plates']:
            x1,y1,x2,y2=plate['box']
            labels.append(f'0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
        (destination/split/'labels'/f"{record['id']}.txt").write_text('\n'.join(labels)+'\n')
    existing_counts={}
    if include_existing:
        existing=ROOT/'platevision/data/yolo-indian'
        seen={r['image_sha256'] for r in records}
        for split in ('train','val'):
            count=0
            for image in sorted((existing/split/'images').glob('*.jpg')):
                label=existing/split/'labels'/f'{image.stem}.txt'
                if not label.is_file():raise ValueError(f'Missing existing label: {label}')
                digest=sha256(image.read_bytes()).hexdigest()
                if digest in seen:raise ValueError('Archive overlaps an existing split; regroup before training.')
                seen.add(digest)
                shutil.copyfile(image,destination/split/'images'/('base_'+image.name))
                shutil.copyfile(label,destination/split/'labels'/('base_'+label.name));count+=1
            existing_counts[split]=count
    report=dict(archive_sha256=sha256(archive.read_bytes()).hexdigest(),seed=42,
                image_count=len(records),plate_count=sum(len(r['plates']) for r in records),
                orientation_corrections=sum(r['orientation_corrected'] for r in records),
                splits=dict(Counter(r['split'] for r in records)),existing_images_added=existing_counts,
                text_labels=dict(Counter(b['text_status'] for r in records for b in r['plates'])),
                note='Provided OCR labels are not verified transcripts. Review scenes are excluded from full-frame training and precision evaluation. Test images are never used for checkpoint selection.',records=records)
    (destination/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (destination/'data.yaml').write_text('path: '+json.dumps(destination.resolve().as_posix())+'\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: number_plate\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review',type=Path)
    parser.add_argument('--no-existing',action='store_true')
    args=parser.parse_args()
    review=json.loads(args.review.read_text()) if args.review else {}
    report=prepare(args.archive,args.output,review,not args.no_existing)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
