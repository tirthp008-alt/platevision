"""Tests for environment-file resolution in app.core.config.

The backend must load a ``.env`` placed either at the repository root or inside
``backend/``, because the README documents both. ``backend/.env`` wins when both
exist. These tests build a throwaway Settings subclass rather than mutating the
process-wide singleton.
"""

import os
from pathlib import Path

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import _ROOT_ENV


def _settings_from(env_files):
    class _Probe(BaseSettings):
        PROBE_ENV_VALUE: str = "default"
        model_config = SettingsConfigDict(
            env_file=env_files, env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
        )

    return _Probe(_env_file=env_files)


def test_root_env_path_points_at_repository_root():
    assert _ROOT_ENV.name == ".env"
    assert _ROOT_ENV.parent.name == "platevision"


def test_root_env_is_loaded(tmp_path):
    root = tmp_path / ".env"
    root.write_text("PROBE_ENV_VALUE=from-root\n")
    assert _settings_from((str(root),)).PROBE_ENV_VALUE == "from-root"


def test_local_env_takes_precedence_over_root(tmp_path):
    root = tmp_path / ".env"
    local = tmp_path / "backend.env"
    root.write_text("PROBE_ENV_VALUE=from-root\n")
    local.write_text("PROBE_ENV_VALUE=from-local\n")
    # Order matters: later entries override earlier ones.
    assert _settings_from((str(root), str(local))).PROBE_ENV_VALUE == "from-local"


def test_missing_env_file_falls_back_to_default(tmp_path):
    missing = tmp_path / "does-not-exist.env"
    assert _settings_from((str(missing),)).PROBE_ENV_VALUE == "default"


def test_shell_environment_overrides_env_file(tmp_path, monkeypatch):
    root = tmp_path / ".env"
    root.write_text("PROBE_ENV_VALUE=from-root\n")
    monkeypatch.setenv("PROBE_ENV_VALUE", "from-shell")
    assert _settings_from((str(root),)).PROBE_ENV_VALUE == "from-shell"


def test_real_config_wires_both_env_locations():
    """The application Settings must declare the root env plus the local one."""
    from app.core.config import settings

    env_file = type(settings).model_config.get("env_file")
    assert env_file is not None
    resolved = [os.fspath(p) for p in env_file]
    assert os.fspath(_ROOT_ENV) in resolved
    assert ".env" in resolved
    # Root is listed first so a local backend/.env overrides it.
    assert resolved[0] == os.fspath(_ROOT_ENV)
