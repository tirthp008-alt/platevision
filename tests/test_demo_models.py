"""Installer tests use tiny fixtures: no model downloads or inference."""
from hashlib import sha256
import io
import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import setup_demo_models as setup


def record(path, content, **extra):
    return dict(path=path, size_bytes=len(content), sha256=sha256(content).hexdigest(), **extra)


def bundle(tmp_path, members=None):
    members = members or {'desktop-console/platevision/models/default.onnx': b'fixture-model',
                          'desktop-console/platevision/models/model-card.json': b'{"fixture":true}'}
    archive = tmp_path / 'bundle.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        for path, content in members.items():
            zipped.writestr(path, content)
    records = [record(path, content) for path, content in members.items()]
    records[0]['copies'] = ['models/plate_detector.onnx']
    return archive, dict(size_bytes=archive.stat().st_size, sha256=setup.file_hash(archive),
                         files=records, url='https://github.com/tirthp008-alt/platevision/releases/download/v-test/core.zip')


def test_offline_install_checks_members_and_compatibility_copy_and_is_repeatable(tmp_path):
    archive, expected = bundle(tmp_path)
    target = tmp_path / 'clean-checkout'
    assert setup.install_archive(archive, expected, target) == 3
    assert setup.verify_installed(target, expected) == []
    assert (target / 'models/plate_detector.onnx').read_bytes() == b'fixture-model'
    assert setup.install_archive(archive, expected, target) == 0


def test_bad_archive_hash_never_creates_model_destinations(tmp_path):
    archive, expected = bundle(tmp_path)
    expected['sha256'] = '0' * 64
    target = tmp_path / 'checkout'
    with pytest.raises(ValueError, match='SHA-256'):
        setup.install_archive(archive, expected, target)
    assert not target.exists()


def test_atomic_transfer_inherits_destination_permissions_not_private_staging_acl(tmp_path, monkeypatch):
    archive, expected = bundle(tmp_path)
    moves = []
    original = setup.os.replace
    def replace(source, target):
        moves.append((source, target))
        assert source.parent == target.parent
        return original(source, target)
    monkeypatch.setattr(setup.os, 'replace', replace)
    checkout = tmp_path / 'checkout'
    assert setup.install_archive(archive, expected, checkout) == 3
    assert len(moves) == 3
    assert not list(checkout.rglob('*.install'))


def test_bad_member_hash_does_not_install_earlier_valid_member(tmp_path):
    archive, expected = bundle(tmp_path)
    expected['files'][1]['sha256'] = '0' * 64
    target = tmp_path / 'checkout'
    with pytest.raises(ValueError, match='Member SHA-256'):
        setup.install_archive(archive, expected, target)
    assert not (target / expected['files'][0]['path']).exists()
    assert not (target / 'models/plate_detector.onnx').exists()


def test_changed_existing_files_are_preserved_until_explicit_replace(tmp_path):
    archive, expected = bundle(tmp_path)
    target = tmp_path / 'checkout'
    existing = target / expected['files'][1]['path']
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'user-custom-card')
    with pytest.raises(ValueError, match='Existing file differs'):
        setup.install_archive(archive, expected, target)
    assert existing.read_bytes() == b'user-custom-card'
    assert not (target / expected['files'][0]['path']).exists()
    assert setup.install_archive(archive, expected, target, replace=True) == 3
    assert setup.verify_installed(target, expected) == []


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../escape', 'a//escape',
                                 'C:/escape', 'a\\escape', 'models/a:stream', 'models/CON.onnx',
                                 'models/name.', 'models/name '])
def test_unsafe_manifest_paths_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        setup.safe_target(tmp_path, name)


def test_extra_traversal_member_is_rejected_even_if_archive_hash_matches(tmp_path):
    archive, expected = bundle(tmp_path)
    with zipfile.ZipFile(archive, 'a') as zipped:
        zipped.writestr('../escape', b'bad')
    expected.update(size_bytes=archive.stat().st_size, sha256=setup.file_hash(archive))
    with pytest.raises(ValueError, match='members differ'):
        setup.install_archive(archive, expected, tmp_path / 'checkout')
    assert not (tmp_path / 'escape').exists()


def test_symbolic_link_zip_entry_is_rejected(tmp_path):
    archive = tmp_path / 'link.zip'
    content = b'../../outside'
    info = zipfile.ZipInfo('models/fake.onnx')
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr(info, content)
    expected = dict(size_bytes=archive.stat().st_size, sha256=setup.file_hash(archive),
                    files=[record(info.filename, content)])
    with pytest.raises(ValueError, match='regular model files'):
        setup.install_archive(archive, expected, tmp_path / 'checkout')


def test_directory_collision_does_not_modify_other_members_even_with_replace(tmp_path):
    archive, expected = bundle(tmp_path)
    target = tmp_path / 'checkout'
    (target / expected['files'][1]['path']).mkdir(parents=True)
    with pytest.raises(ValueError, match='not a regular file'):
        setup.install_archive(archive, expected, target, replace=True)
    assert not (target / expected['files'][0]['path']).exists()


@pytest.mark.parametrize('payload', [b'bad-data', b'very-long-bad-data'])
def test_download_rejects_tampered_and_oversized_response(tmp_path, monkeypatch, payload):
    class Response(io.BytesIO):
        def geturl(self): return 'https://release-assets.githubusercontent.com/fixture'
    expected = dict(url='https://github.com/tirthp008-alt/platevision/releases/download/v-test/core.zip',
                    size_bytes=8, sha256=sha256(b'gooddata').hexdigest())
    monkeypatch.setattr(setup.urllib.request, 'urlopen', lambda *args, **kwargs: Response(payload))
    with pytest.raises(ValueError, match='pinned|SHA-256'):
        setup.download_bundle(expected, tmp_path / 'download.zip')


def test_installed_check_and_offline_cli_never_access_network(tmp_path, monkeypatch):
    archive, expected = bundle(tmp_path)
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(release_tag='fixture', bundles={'core': expected})))
    monkeypatch.setattr(setup, 'ROOT', checkout)
    monkeypatch.setattr(setup, 'MANIFEST', manifest)
    def forbidden(*args, **kwargs): raise AssertionError('Offline installer attempted network')
    monkeypatch.setattr(setup.urllib.request, 'urlopen', forbidden)
    assert setup.main(['--archive', str(archive)]) == 0
    assert setup.main(['--check']) == 0
