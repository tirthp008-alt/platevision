"""Import green-plate images with their external VOC labels for visual review.

Archive paths and XML path/filename fields are data, never extraction destinations.
Output is local training data; no images are sent to another service.
"""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps


def import_green(archive, annotations, output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    (output/'images').mkdir(parents=True)
    records = []; rejected = []
    with ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries)>5000 or sum(i.file_size for i in entries)>5_000_000_000:
            raise ValueError('Archive exceeds import limits')
        for source_index,name in enumerate(sorted(i.filename for i in entries if i.filename.lower().endswith('.jpg'))):
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('Unsafe archive member')
            if source.getinfo(name).file_size>50_000_000:
                raise ValueError('Image exceeds import limit')
            data=source.read(name)
            try:
                with Image.open(BytesIO(data)) as raw:
                    if raw.width*raw.height>40_000_000:
                        raise ValueError('Image exceeds pixel limit')
                    orientation=raw.getexif().get(274,1)
                    image=ImageOps.exif_transpose(raw).convert('RGB')
            except OSError as error:
                rejected.append(dict(source_image=name,reason=str(error)))
                continue
            width,height=image.size
            xml=annotations/(path.stem+'.xml')
            plates=[]; review=''
            if xml.exists():
                xml_data=xml.read_bytes()
                if b'<!DOCTYPE' in xml_data.upper() or b'<!ENTITY' in xml_data.upper():
                    raise ValueError('XML entities/document types are not accepted')
                tree=ET.fromstring(xml_data)
                dims=tuple(int(tree.findtext('size/'+k)) for k in ('width','height'))
                if dims!=image.size:
                    raise ValueError(f'Orientation/annotation mismatch: {path.name}')
                for obj in tree.findall('object'):
                    if obj.findtext('name')!='green license plate':
                        raise ValueError('Unexpected label class')
                    box=[float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
                    x1,y1,x2,y2=box
                    if not (0<=x1<x2<=width and 0<=y1<y2<=height):
                        raise ValueError('Out of bounds annotation')
                    plates.append(dict(box=box,text='',text_status='missing',transcription_verified=False))
            else:
                review='Missing annotation; never assume an unlabelled photo is a negative.'
            ident=f'green_{source_index+1:03}'
            image.thumbnail((1600,1600),Image.Resampling.LANCZOS)
            sx,sy=image.width/width,image.height/height
            for plate in plates:plate['box']=[v*(sx if i%2==0 else sy) for i,v in enumerate(plate['box'])]
            image.save(output/'images'/f'{ident}.jpg',quality=95)
            # The capture identity is deliberately conservative: all photos from
            # the same contributor stay together, including repeated vehicles.
            tokens=path.stem.split('_')
            record=dict(id=ident,source_image=name,image_sha256=sha256(data).hexdigest(),
                        width=image.width,height=image.height,original_size=[width,height],
                        orientation_corrected=orientation in {2,3,4,5,6,7,8},
                        vehicle_group='green-contributor:'+tokens[6],plates=plates,
                        labels_scope='green plates only; other colours require review',
                        review_required=review)
            records.append(record)
            if len(records)%30==0:print(f'Imported {len(records)} green photos',flush=True)
    report=dict(archive_sha256=sha256(archive.read_bytes()).hexdigest(),
                images=len(records),plates=sum(len(r['plates']) for r in records),
                groups=len({r['vehicle_group'] for r in records}),rejected=rejected,records=records)
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    render_sheets(output,records)
    return report


def render_sheets(output,records):
    sheets=output/'review';sheets.mkdir(exist_ok=True)
    for start in range(0,len(records),30):
        sheet=Image.new('RGB',(1500,1800),'#e8edf2');draw=ImageDraw.Draw(sheet)
        for n,record in enumerate(records[start:start+30]):
            image=Image.open(output/'images'/f"{record['id']}.jpg")
            image.thumbnail((294,268))
            x=(n%5)*300+(300-image.width)//2;y=(n//5)*300+25
            sheet.paste(image,(x,y))
            for plate in record['plates']:
                box=plate['box'];sx=image.width/record['width'];sy=image.height/record['height']
                draw.rectangle((x+box[0]*sx,y+box[1]*sy,x+box[2]*sx,y+box[3]*sy),outline='#ff2040',width=3)
            draw.text(((n%5)*300+8,(n//5)*300+6),record['id']+f" | {len(record['plates'])} boxes",fill='black')
        sheet.save(sheets/f'sheet-{start//30+1}.jpg',quality=92)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=import_green(args.archive,args.annotations,args.output)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
