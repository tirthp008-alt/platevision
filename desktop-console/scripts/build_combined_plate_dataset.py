"""Combine reviewed Indian plate data; keep capture groups and test images apart."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

from PIL import Image
from import_plate_archive import assign_splits

# Visual review of the green archive contact sheets. Other scenes contribute
# local plate context only, because their other plate colours are unlabelled.
FULL_SCENES={1,2,3,4,5,7,8,14,15,17,20,23,24,25,26,27,29,30,31,32,
    37,38,40,43,44,45,46,48,51,52,53,54,55,58,60,64,66,67,68,69,70,
    71,72,73,74,77,78,79,88,91,94,95,97,98,99,100,102,103,104,105,
    106,107,108,109,111,112,113,115,116,117,118,120,122,124,125,126,
    127,128,129,131,132,134,135,136,137,140,141,144,145,146,148,150,
    151,152,153,154,155,156,157,158,159,160,162,163,164,165,166,172,
    174,176,177,178,179,180,181,182,184,185,186,187,191,192,193,194,
    195,196,202,203,204,205,206,207,208,209,210}
REVIEWED_GREEN_ARCHIVE='465aaf98848c8d24f4fec7cc030fa5323c70fa11989b321938a487aea1058d8f'


def save_example(image,plates,output,split,name):
    image.save(output/split/'images'/f'{name}.jpg',quality=95)
    w,h=image.size;lines=[]
    for p in plates:
        x1,y1,x2,y2=p['box']
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):
            raise ValueError(f'Invalid transformed box in {name}')
        lines.append(f'0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
    (output/split/'labels'/f'{name}.txt').write_text('\n'.join(lines)+'\n')


def context_region(box,width,height,factor=2.2):
    x1,y1,x2,y2=box;cx=(x1+x2)/2;cy=(y1+y2)/2
    w=max(96,(x2-x1)*factor);h=max(96,(y2-y1)*factor)
    return (max(0,int(cx-w/2)),max(0,int(cy-h/2)),
            min(width,int(cx+w/2+.999)),min(height,int(cy+h/2+.999)))


def build(general,green,output):
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    imported=json.loads((green/'manifest.json').read_text())
    if imported['archive_sha256']!=REVIEWED_GREEN_ARCHIVE:
        raise ValueError('This visual-review list applies only to the supplied green ZIP. Review a different archive before building its training set.')
    for split in ('train','val','test'):
        for kind in ('images','labels'):(output/split/kind).mkdir(parents=True)
    original=json.loads((general/'manifest.json').read_text())
    records=deepcopy(imported['records'])
    # The two source boxes in this image describe the same physical plate.
    for r in records:
        if r['id']=='green_015':r['plates']=r['plates'][:1]
        r['full_scene_reviewed']=int(r['id'].split('_')[1]) in FULL_SCENES
    assign_splits(records,seed=42)
    # Preserve the first archive's existing split and its separate video-source
    # train/validation sets. The untouched test set is never a training source.
    for split in ('train','val','test'):
        for kind in ('images','labels'):
            for file in (general/split/kind).iterdir():
                shutil.copyfile(file,output/split/kind/file.name)
    for r in records:
        split=r['split']
        if split=='review':continue
        image=Image.open(green/'images'/f"{r['id']}.jpg").convert('RGB')
        r['evaluation_image']=str((green/'images'/f"{r['id']}.jpg").resolve())
        if split=='test':
            # Whole images, not enlarged crops, for the final green recall test.
            save_example(image,r['plates'],output,split,r['id'])
            continue
        if r['full_scene_reviewed']:
            save_example(image,r['plates'],output,split,r['id'])
        # Green context examples increase green-plate exposure without deleting
        # white/yellow training images. Crops inherit their parent split.
        for i,plate in enumerate(r['plates']):
            left,top,right,bottom=context_region(plate['box'],*image.size)
            included=[]
            for p in r['plates']:
                x1,y1,x2,y2=p['box']
                if left<=x1<x2<=right and top<=y1<y2<=bottom:
                    included.append(dict(box=[x1-left,y1-top,x2-left,y2-top]))
                elif max(left,x1)<min(right,x2) and max(top,y1)<min(bottom,y2):
                    raise ValueError('Crop would leave another plate partially unlabelled')
            save_example(image.crop((left,top,right,bottom)),included,output,split,f"{r['id']}_context_{i}")
    summary=dict(source_archives=[original['archive_sha256'],imported['archive_sha256']],
        seed=42,green_image_splits=dict(Counter(r['split'] for r in records)),
        examples={s:len(list((output/s/'images').glob('*.jpg'))) for s in ('train','val','test')},
        duplicate_box_removed='green_015',rejected_images=imported['rejected'],
        green_records=records,general_manifest=str((general/'manifest.json').resolve()),
        notes=['Capture contributor groups stay in one split; crops follow their parents.',
               'Green holdout is evaluated on whole images, including background plates.',
               'Unreviewed green scenes have incomplete labels for other colours: report green recall, not scene-wide precision.',
               'Labels localize plates; they do not provide verified OCR transcripts.'])
    (output/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'data.yaml').write_text('path: '+json.dumps(output.resolve().as_posix())+
        '\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: number_plate\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('general','green','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();report=build(args.general,args.green,args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in {'green_records','rejected_images'}},indent=2))
