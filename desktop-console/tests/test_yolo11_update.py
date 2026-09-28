"""Frozen holdouts, annotation consistency and safe archive handling."""
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_yolo11_update import digest, fingerprint, overlap_reason, prepare, read_labels


def photo(seed, kind='PNG'):
    pixels = np.random.default_rng(seed).integers(0, 256, (5, 10, 3), dtype=np.uint8)
    image = Image.fromarray(pixels).resize((80, 40), Image.Resampling.BILINEAR)
    stream = BytesIO()
    image.save(stream, format=kind, quality=92)
    return stream.getvalue()


def annotation(xmax=60):
    return f'<annotation><size><width>80</width><height>40</height></size><object><name>LP</name><bndbox><xmin>20</xmin><ymin>10</ymin><xmax>{xmax}</xmax><ymax>30</ymax></bndbox></object></annotation>'.encode()


def base_dataset(tmp_path):
    base = tmp_path / 'base'
    original = tmp_path / 'original.zip'
    original.write_bytes(b'existing archive hash fixture')
    for index, split in enumerate(('train', 'val', 'test')):
        (base / split / 'images').mkdir(parents=True)
        (base / split / 'labels').mkdir()
        (base / split / 'images' / 'old.png').write_bytes(photo(index))
        (base / split / 'labels' / 'old.txt').write_bytes(b'0 .5 .5 .5 .5\n')
    (base / 'manifest.json').write_text(json.dumps(dict(source_archives=[digest(original.read_bytes())],
        examples=dict(train=1, val=1, test=1), green_records=[])))
    return base, original


def test_missing_txt_uses_proven_converter_and_conflicting_or_degenerate_labels_are_rejected():
    generated, count, source = read_labels(annotation(), None, 80, 40)
    assert list(map(float, generated.split())) == [0., .4875, .475, .5, .5]
    assert count == 1 and source.startswith('generated_from_voc')
    assert read_labels(annotation(), generated, 80, 40)[0] == generated
    with pytest.raises(ValueError, match='coordinates disagree'):
        read_labels(annotation(), b'0 .5 .5 .5 .5', 80, 40)
    with pytest.raises(ValueError, match='degenerate'):
        read_labels(annotation(20), None, 80, 40)


def test_new_source_stays_train_and_frozen_files_survive_byte_for_byte(tmp_path):
    base, original = base_dataset(tmp_path)
    archive = tmp_path / 'new.zip'
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr('test/181.png', photo(10))
        zipped.writestr('test/181.xml', annotation())
        zipped.writestr('test/183.png', photo(11))
        zipped.writestr('test/183.xml', annotation(20))
    output = tmp_path / 'updated'
    report = prepare(base, archive, original, output)
    assert report['examples'] == dict(train=2, val=1, test=1)
    assert report['additional_data']['retained_images'] == report['additional_data']['quarantined_images'] == 1
    assert {row['split'] for row in report['additional_records']} == {'train', 'quarantine'}
    for path in base.rglob('*'):
        if path.is_file() and path.name != 'manifest.json':
            assert (output / path.relative_to(base)).read_bytes() == path.read_bytes()
    assert (output / 'base-manifest.json').read_bytes() == (base / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        prepare(base, archive, original, output)


def test_lossy_reencoding_and_holdout_duplicates_are_screened():
    png, jpeg = fingerprint(photo(33)), fingerprint(photo(33, 'JPEG'))
    assert png['encoded_sha256'] != jpeg['encoded_sha256']
    result = overlap_reason(jpeg, [dict(png, split='test', name='frozen-test.png')])
    assert result['reason'] in {'probable_reencoded_duplicate', 'possible_related_holdout'}
    assert result['split'] == 'test'
    exact = overlap_reason(png, [dict(png, split='val', name='frozen-val.png')])
    assert exact['reason'] == 'exact_duplicate'


def test_archive_path_escape_is_rejected_without_creating_destination(tmp_path):
    base, original = base_dataset(tmp_path)
    archive = tmp_path / 'bad.zip'
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr('../outside.png', photo(9))
    output = tmp_path / 'updated'
    with pytest.raises(ValueError, match='Unsafe archive'):
        prepare(base, archive, original, output)
    assert not output.exists() and not (tmp_path / 'outside.png').exists()
