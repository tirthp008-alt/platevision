"""Add audited new plate images to training while freezing all existing splits.

Read ZIP members as data, never extract their paths. The source `test` folder is
kept together as one training provenance group under the user's training request;
it is not used as a new validation/test split. No active dataset is modified.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'val', 'test')


def digest(data):
    return sha256(data).hexdigest()


def fingerprint(data):
    with Image.open(BytesIO(data)) as raw:
        if raw.width * raw.height > 40_000_000:
            raise ValueError('Image exceeds 40 million pixels.')
        raw.load()
        orientation = int(raw.getexif().get(274, 1))
        rgb = np.asarray(ImageOps.exif_transpose(raw).convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(thumb)[:8, :8]
    bits = dct > np.median(dct.ravel()[1:])
    bits[0, 0] = False
    phash = sum(int(bit) << i for i, bit in enumerate(bits.ravel()))
    return dict(encoded_sha256=digest(data), decoded_sha256=digest(str(rgb.shape).encode() + rgb.tobytes()),
                perceptual_hash=phash, thumbnail=thumb, width=rgb.shape[1], height=rgb.shape[0],
                orientation=orientation)


def read_labels(xml, txt, width, height):
    """Validate both formats using the ZIP's consistent one-pixel VOC offset."""
    if len(xml) > 1_000_000 or b'<!DOCTYPE' in xml.upper() or b'<!ENTITY' in xml.upper():
        raise ValueError('Unsafe or oversized annotation XML.')
    tree = ET.fromstring(xml)
    dimensions = tuple(int(tree.findtext('size/' + key)) for key in ('width', 'height'))
    if dimensions != (width, height):
        raise ValueError('Image and XML dimensions disagree.')
    expected = []
    for obj in tree.findall('object'):
        if obj.findtext('name') not in {'LP', 'number_plate'}:
            raise ValueError('Unsupported annotation class.')
        x1, y1, x2, y2 = [float(obj.findtext('bndbox/' + key)) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        if (not all(math.isfinite(value) for value in (x1, y1, x2, y2))
                or not (1 <= x1 < x2 <= width and 1 <= y1 < y2 <= height)):
            raise ValueError('Invalid, degenerate or out-of-bounds VOC box.')
        # Every supplied usable TXT box in this ZIP follows this converter.
        expected.append((0., ((x1 + x2) / 2 - 1) / width, ((y1 + y2) / 2 - 1) / height,
                         (x2 - x1) / width, (y2 - y1) / height))
    if not expected:
        raise ValueError('No labelled plate; do not silently create a negative image.')
    if txt is not None:
        rows = [tuple(map(float, line.split())) for line in txt.decode('utf-8-sig').splitlines() if line.strip()]
        if len(rows) != len(expected):
            raise ValueError('VOC and YOLO label counts disagree.')
        remaining = list(expected)
        for row in rows:
            if len(row) != 5 or row[0] != 0 or not all(math.isfinite(v) for v in row):
                raise ValueError('Invalid YOLO label.')
            match = next((i for i, target in enumerate(remaining)
                          if np.allclose(row, target, atol=1e-7, rtol=0)), None)
            if match is None:
                raise ValueError('VOC and YOLO coordinates disagree with the audited one-pixel convention.')
            remaining.pop(match)
        return txt, len(expected), 'provided_yolo_validated_against_voc'
    generated = ''.join('0 ' + ' '.join(f'{value:.10f}' for value in row[1:]) + '\n' for row in expected)
    return generated.encode('utf-8'), len(expected), 'generated_from_voc_matching_supplied_yolo_offset_minus_one_pixel'


def overlap_reason(candidate, references):
    for previous in references:
        exact = (candidate['encoded_sha256'] == previous['encoded_sha256']
                 or candidate['decoded_sha256'] == previous['decoded_sha256'])
        distance = (candidate['perceptual_hash'] ^ previous['perceptual_hash']).bit_count()
        near = distance <= 8
        reencoded = distance <= 4 and float(np.mean(np.abs(candidate['thumbnail'] - previous['thumbnail']))) <= 8
        if exact or reencoded or (previous['split'] in {'val', 'test', 'review'} and near):
            kind = 'exact_duplicate' if exact else 'probable_reencoded_duplicate' if reencoded else 'possible_related_holdout'
            return dict(reason=kind, reference=previous['name'], split=previous['split'], phash_distance=distance)
    return None


def prepare(base, archive, original_archive, output):
    base, archive, original_archive, output = map(Path, (base, archive, original_archive, output))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    base_manifest_data = (base / 'manifest.json').read_bytes()
    manifest = json.loads(base_manifest_data)
    source_hash = digest(archive.read_bytes())
    original_hash = digest(original_archive.read_bytes())
    if original_hash not in manifest['source_archives']:
        raise ValueError('archive.zip does not match an archive already represented in the frozen dataset.')
    cv2.setNumThreads(1)
    references, files = [], []
    for split in SPLITS:
        images = sorted(path for path in (base / split / 'images').iterdir() if path.is_file())
        if len(images) != manifest['examples'][split]:
            raise ValueError(f'Frozen {split} image count differs from its manifest.')
        for image in images:
            label = base / split / 'labels' / (image.stem + '.txt')
            image_data, label_data = image.read_bytes(), label.read_bytes()
            references.append(dict(fingerprint(image_data), name=f'{split}/{image.name}', split=split))
            files.append(dict(split=split, image=image.name, image_sha256=digest(image_data),
                              label=label.name, label_sha256=digest(label_data)))
    # Include original green whole scenes that only contributed context crops.
    # Their full images must not leak back into training through a new archive.
    source_hashes = {}
    for record in manifest.get('green_records', []):
        source_hashes[record['image_sha256']] = dict(name=record['source_image'], split=record['split'])
        path = record.get('evaluation_image')
        if path and record['split'] in {'val', 'test', 'review'}:
            path = Path(path)
            if not path.resolve().is_relative_to((ROOT / 'platevision/data').resolve()):
                raise ValueError('Historical evaluation image is outside the dataset workspace.')
            references.append(dict(fingerprint(path.read_bytes()), name=record['source_image'], split=record['split']))
    group = 'source-archive:' + source_hash
    records, accepted, quarantined = [], [], []
    with ZipFile(archive) as zipped:
        infos = zipped.infolist()
        if len(infos) > 5000 or sum(item.file_size for item in infos) > 1_000_000_000:
            raise ValueError('Archive exceeds bounded import size.')
        names = set()
        for item in infos:
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in item.filename
                    or ':' in item.filename or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Unsafe archive member path or symbolic link.')
            if item.filename in names:
                raise ValueError('Duplicate archive member name.')
            names.add(item.filename)
        images = sorted(item.filename for item in infos
                        if not item.is_dir() and PurePosixPath(item.filename).suffix.lower() in {'.jpg', '.jpeg', '.png'})
        for index, name in enumerate(images):
            member = PurePosixPath(name)
            if zipped.getinfo(name).file_size > 40_000_000:
                raise ValueError('Individual image exceeds bounded import size.')
            image_data = zipped.read(name)
            xml_name, txt_name = str(member.with_suffix('.xml')), str(member.with_suffix('.txt'))
            xml = zipped.read(xml_name) if xml_name in names else None
            txt = zipped.read(txt_name) if txt_name in names else None
            identifier = f'additional_{source_hash[:8]}_{index + 1:03}'
            record = dict(id=identifier, source_image=name, image_sha256=digest(image_data),
                          source_annotation=xml_name, annotation_sha256=digest(xml) if xml else None,
                          supplied_yolo_sha256=digest(txt) if txt is not None else None,
                          source_archive_sha256=source_hash, provenance_group=group, split='train')
            try:
                fp = fingerprint(image_data)
                record.update(width=fp['width'], height=fp['height'], decoded_sha256=fp['decoded_sha256'],
                              perceptual_hash=f'{fp["perceptual_hash"]:016x}')
                if fp['orientation'] != 1:
                    raise ValueError('Nontrivial EXIF orientation requires annotation review.')
                if xml is None:
                    raise ValueError('Missing VOC annotation.')
                labels, count, label_source = read_labels(xml, txt, fp['width'], fp['height'])
                overlap = source_hashes.get(fp['encoded_sha256'])
                if overlap:
                    overlap = dict(reason='exact_source_archive_duplicate', reference=overlap['name'], split=overlap['split'])
                else:
                    overlap = overlap_reason(fp, references)
                if overlap:
                    record['overlap'] = overlap
                    raise ValueError(overlap['reason'])
                record.update(plate_count=count, label_source=label_source,
                              output_image=f'train/images/{identifier}{member.suffix.lower()}',
                              output_label=f'train/labels/{identifier}.txt')
                accepted.append((record, image_data, labels))
                references.append(dict(fp, name=name, split='train'))
            except (ValueError, OSError, ET.ParseError, TypeError) as error:
                record.update(split='quarantine', reason=str(error))
                quarantined.append((record, image_data, xml, txt, member.suffix.lower()))
            records.append(record)
    # All decisions precede mutation. Existing dataset files are only copied.
    for split in SPLITS:
        for kind in ('images', 'labels'):
            (output / split / kind).mkdir(parents=True, exist_ok=False)
    for item in files:
        for kind, name_key, hash_key in (('images', 'image', 'image_sha256'), ('labels', 'label', 'label_sha256')):
            source = base / item['split'] / kind / item[name_key]
            target = output / item['split'] / kind / item[name_key]
            shutil.copyfile(source, target)
            if digest(target.read_bytes()) != item[hash_key] or digest(source.read_bytes()) != item[hash_key]:
                raise ValueError('A frozen source changed during copying.')
    for record, data, labels in accepted:
        (output / record['output_image']).write_bytes(data)
        (output / record['output_label']).write_bytes(labels)
    if quarantined:
        (output / 'quarantine').mkdir()
    for record, data, xml, txt, extension in quarantined:
        stem = output / 'quarantine' / record['id']
        stem.with_suffix(extension).write_bytes(data)
        if xml is not None:
            stem.with_suffix('.xml').write_bytes(xml)
        if txt is not None:
            stem.with_suffix('.txt').write_bytes(txt)
    updated = deepcopy(manifest)
    updated.update(dataset_version=output.name, base_dataset=str(base.resolve()),
                   base_manifest_sha256=digest(base_manifest_data), frozen_split_files=files,
                   frozen_splits_preserved_byte_for_byte=True,
                   source_archives=list(dict.fromkeys([*manifest['source_archives'], source_hash])),
                   additional_records=records,
                   additional_data=dict(archive=str(archive.resolve()), archive_sha256=source_hash,
                                        already_included_archive_sha256=original_hash,
                                        source_images=len(images), retained_images=len(accepted),
                                        retained_plate_boxes=sum(row[0]['plate_count'] for row in accepted),
                                        quarantined_images=len(quarantined),
                                        quarantine_reasons=dict(Counter(row[0]['reason'] for row in quarantined)),
                                        provenance_group=group, split_policy='All usable new source images stay in train; prior splits never move.',
                                        label_policy='Validate supplied YOLO against VOC with -1 pixel centre convention; generate missing TXT identically.',
                                        class_mapping={'LP': 'number_plate', 'number_plate': 'number_plate'}),
                   examples={split: manifest['examples'][split] + (len(accepted) if split == 'train' else 0) for split in SPLITS})
    updated['notes'] = [*manifest.get('notes', []),
                        'New archive is training-only as one source group; old validation/test images and labels are byte-for-byte frozen.',
                        'Exact encoded/decoded and conservative perceptual duplicate checks cannot rule out all same-vehicle images from unrelated angles.',
                        'VOC/TXT labels localize plates; this new archive contains no verified OCR transcripts.']
    (output / 'base-manifest.json').write_bytes(base_manifest_data)
    (output / 'manifest.json').write_text(json.dumps(updated, indent=2), encoding='utf-8')
    (output / 'data.yaml').write_text('path: ' + json.dumps(output.resolve().as_posix()) +
                                     '\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: number_plate\n', encoding='utf-8')
    return updated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'platevision/data/combined-green-20260919')
    parser.add_argument('--archive', type=Path, required=True, help='User-supplied additional labelled ZIP archive')
    parser.add_argument('--original-archive', type=Path, required=True, help='Original archive already included in the frozen base dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'platevision/data/combined-yolo11-20260927')
    args = parser.parse_args()
    report = prepare(args.base, args.archive, args.original_archive, args.output)
    print(json.dumps(dict(examples=report['examples'], additional_data=report['additional_data'],
                          yaml=str((args.output / 'data.yaml').resolve())), indent=2))
