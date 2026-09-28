"""Install pinned inference models without training libraries.

Default: python scripts/setup_demo_models.py
Offline: python scripts/setup_demo_models.py --archive platevision-core-models.zip
Optional matching: python scripts/setup_demo_models.py --bundle reid
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'models/manifest.json'
CHUNK = 1024 * 1024


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_target(root, relative):
    """Reject traversal, Windows drive/ADS names and symlink escapes."""
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or any(ord(char) < 32 for char in relative)):
        raise ValueError(f'Unsafe model path: {relative!r}')
    path = PurePosixPath(relative)
    if (path.is_absolute() or any(part in ('', '.', '..') for part in relative.split('/'))
            or any(part.endswith((' ', '.')) for part in path.parts)
            or any(re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', part)
                   for part in path.parts)):
        raise ValueError(f'Unsafe model path: {relative!r}')
    root = Path(root).resolve()
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Model path escapes installation directory: {relative}')
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError(f'Symlink model destination is not supported: {relative}')
        if current != target and current.exists() and not current.is_dir():
            raise ValueError(f'Model destination parent is not a directory: {relative}')
        current = current.parent
    return target


def validate_bundle(bundle, root):
    if not re.fullmatch('[0-9a-f]{64}', bundle.get('sha256', '')):
        raise ValueError('Bundle requires a pinned SHA-256.')
    if not isinstance(bundle.get('size_bytes'), int) or bundle['size_bytes'] <= 0:
        raise ValueError('Bundle requires a positive exact byte size.')
    if not bundle.get('files'):
        raise ValueError('Bundle has no model files.')
    seen = set()
    for item in bundle['files']:
        if (not re.fullmatch('[0-9a-f]{64}', item.get('sha256', ''))
                or not isinstance(item.get('size_bytes'), int) or item['size_bytes'] <= 0):
            raise ValueError('Every model member requires a hash and exact positive size.')
        for name in [item['path'], *item.get('copies', [])]:
            safe_target(root, name)
            if name.casefold() in seen:
                raise ValueError(f'Duplicate model destination: {name}')
            seen.add(name.casefold())


def checked_file(path, record):
    path = Path(path)
    return (path.is_file() and path.stat().st_size == record['size_bytes']
            and file_hash(path) == record['sha256'])


def verify_installed(root, bundle):
    validate_bundle(bundle, root)
    failures = []
    for item in bundle['files']:
        for name in [item['path'], *item.get('copies', [])]:
            if not checked_file(safe_target(root, name), item):
                failures.append(name)
    return failures


def download_bundle(bundle, destination):
    url = bundle['url']
    if not url.startswith('https://github.com/tirthp008-alt/platevision/releases/download/'):
        raise ValueError('Only the pinned project release URL is supported.')
    request = urllib.request.Request(url, headers={'User-Agent': 'PlateVision-model-setup/1'})
    with urllib.request.urlopen(request, timeout=90) as response, Path(destination).open('wb') as output:
        if not response.geturl().startswith('https://'):
            raise ValueError('Model download redirected to an insecure URL.')
        count = 0
        while block := response.read(CHUNK):
            count += len(block)
            if count > bundle['size_bytes']:
                raise ValueError('Downloaded bundle exceeds its pinned byte size.')
            output.write(block)
    if not checked_file(destination, bundle):
        raise ValueError('Downloaded model bundle failed its size/SHA-256 check.')


def install_archive(archive, bundle, root, *, replace=False):
    """Validate every member before replacing any installed file."""
    root = Path(root).resolve()
    validate_bundle(bundle, root)
    if not checked_file(archive, bundle):
        raise ValueError('Model archive failed its pinned size/SHA-256 check; nothing was installed.')
    expected = {item['path']: item for item in bundle['files']}
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        names = [item.filename for item in members]
        if len(set(name.casefold() for name in names)) != len(names) or set(names) != set(expected):
            raise ValueError('Archive members differ from the pinned manifest.')
        for info in members:
            safe_target(root, info.filename)
            kind = stat.S_IFMT(info.external_attr >> 16)
            if info.is_dir() or kind not in (0, stat.S_IFREG) or info.flag_bits & 1:
                raise ValueError('Only unencrypted regular model files are supported.')
            if info.file_size != expected[info.filename]['size_bytes']:
                raise ValueError(f'Wrong uncompressed member size: {info.filename}')
        # Cleanup is restricted to this newly-created, owned staging tree.
        with tempfile.TemporaryDirectory(prefix='.model-stage-', dir=root) as temporary:
            stage = Path(temporary).resolve()
            if not stage.is_relative_to(root):
                raise ValueError('Invalid model staging directory.')
            pending = []
            for index, info in enumerate(members):
                item = expected[info.filename]
                staged = stage / str(index)
                with zipped.open(info) as source, staged.open('wb') as target:
                    shutil.copyfileobj(source, target, CHUNK)
                # ZipFile also verifies member CRC while reading to EOF.
                if not checked_file(staged, item):
                    raise ValueError(f'Member SHA-256 mismatch: {info.filename}')
                for name in [item['path'], *item.get('copies', [])]:
                    target = safe_target(root, name)
                    if target.exists() and not target.is_file():
                        raise ValueError(f'Model destination is not a regular file: {name}')
                    if target.exists() and not checked_file(target, item) and not replace:
                        raise ValueError(f'Existing file differs: {name}. Preserve it or explicitly use --replace.')
                    if not checked_file(target, item):
                        pending.append((staged, name))
            for staged, name in pending:
                target = safe_target(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Create beside the destination so Windows inherits its normal
                # ACL, rather than moving a private TemporaryDirectory ACL out.
                transfer = target.parent / f'.{target.name}.{uuid.uuid4().hex}.install'
                try:
                    with staged.open('rb') as source, transfer.open('xb') as output:
                        shutil.copyfileobj(source, output, CHUNK)
                    os.replace(transfer, target)
                finally:
                    if transfer.exists():
                        transfer.unlink()
    return len(pending)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', choices=['core', 'reid'], default='core')
    parser.add_argument('--archive', type=Path, help='Use a local release ZIP without downloading.')
    parser.add_argument('--check', action='store_true', help='Verify installed files without downloading or changing them.')
    parser.add_argument('--replace', action='store_true', help='Explicitly replace differing installed files after all checks pass.')
    args = parser.parse_args(argv)
    if args.check and (args.archive or args.replace):
        parser.error('--check cannot be combined with --archive or --replace.')
    try:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        bundle = manifest['bundles'][args.bundle]
        failures = verify_installed(ROOT, bundle)
        if args.check:
            if failures:
                raise ValueError('Missing or changed model files:\n' + '\n'.join(failures))
            print(f'{args.bundle}: every installed file matches {manifest["release_tag"]}.')
            return 0
        if not failures:
            print(f'{args.bundle}: verified models are already installed.')
            return 0
        if args.archive:
            count = install_archive(args.archive.resolve(), bundle, ROOT, replace=args.replace)
        else:
            with tempfile.TemporaryDirectory(prefix='.model-download-', dir=ROOT) as temporary:
                temporary = Path(temporary).resolve()
                if not temporary.is_relative_to(ROOT.resolve()):
                    raise ValueError('Invalid download staging directory.')
                archive = temporary / bundle['asset']
                print(f'Downloading pinned {args.bundle} model bundle ({bundle["size_bytes"] / 1e6:.1f} MB).', flush=True)
                download_bundle(bundle, archive)
                count = install_archive(archive, bundle, ROOT, replace=args.replace)
        print(f'{args.bundle}: installed {count} verified files. No training or model conversion ran.')
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Model setup failed: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
