"""Install the official trained ResNet34 OCR model, with pinned checksums.

Run from the repository root: .venv/Scripts/python scripts/setup_resnet_ocr.py
Optionally reuse a previously downloaded --archive and --dictionary.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar'
DICT_URL = 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt'
MODEL_SHA = 'd7e7d43d4fd635644fba06c5b43239ecf6d2160aec156df1ab3bd86935acd7c5'
DICT_SHA = '28b2362ad4ab2dc38769aa72feb535e3a9ddb3fd2a7585a05920e6393b1dc7f7'


def checked_file(path, url, expected):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(url, timeout=180) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Download checksum mismatch: {url}')
        partial.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Checksum mismatch: {path}. Remove the incomplete download and retry.')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'.cache/resnet-ocr/model.tar')
    parser.add_argument('--dictionary', type=Path, default=ROOT/'.cache/resnet-ocr/ppocr_keys_v1.txt')
    args = parser.parse_args()
    archive = checked_file(args.archive, MODEL_URL, MODEL_SHA)
    dictionary = checked_file(args.dictionary, DICT_URL, DICT_SHA)
    destination = ROOT/'platevision/models/resnet34'
    destination.mkdir(parents=True, exist_ok=True)
    # Extract only the two named regular files; never honor archive paths/links.
    with tarfile.open(archive) as source:
        for name in ('inference.pdmodel', 'inference.pdiparams'):
            member = source.getmember('ch_ppocr_server_v2.0_rec_infer/' + name)
            if not member.isfile():
                raise ValueError(f'Expected a regular model file: {name}')
            with source.extractfile(member) as model, (destination/name).open('wb') as target:
                shutil.copyfileobj(model, target)
    shutil.copyfile(dictionary, destination/'ppocr_keys_v1.txt')
    manifest = dict(architecture='ResNet34 + BiLSTM + CTC', source=MODEL_URL,
                    archive_sha256=MODEL_SHA, dictionary_source=DICT_URL, dictionary_sha256=DICT_SHA,
                    plate_finetuned=False, license='Apache-2.0 (PaddleOCR)',
                    config='https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml')
    (destination/'source.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Installed trained ResNet OCR at {destination}. Restart the API to enable it.')


if __name__ == '__main__':
    main()
