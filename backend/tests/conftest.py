"""Test configuration for the Drishti Grid test suite.

Environment is configured before the application package is imported so the
settings singleton picks up an isolated SQLite database and test API keys.
"""

import os
import tempfile

# Isolated database per test session.
_TMP_DB = os.path.join(tempfile.gettempdir(), "drishti_grid_test.db")
for _p in (_TMP_DB, _TMP_DB + "-wal", _TMP_DB + "-shm"):
    if os.path.exists(_p):
        os.remove(_p)

os.environ["GRID_DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["DRISHTI_ADMIN_KEY"] = "test-admin"
os.environ["DRISHTI_OPERATOR_KEY"] = "test-operator"
os.environ["DRISHTI_VIEWER_KEY"] = "test-viewer"
os.environ["GRID_DEMO_SEGMENT_SECONDS"] = "10"
os.environ["GRID_DEMO_DWELL_SECONDS"] = "8"
os.environ["EVENT_BACKEND"] = "memory"

import pytest  # noqa: E402


@pytest.fixture()
def fresh_db():
    """Recreate all tables and yield a clean database session."""
    from app.grid.db import reset_for_tests, session_scope

    reset_for_tests()
    with session_scope() as session:
        yield session


@pytest.fixture()
def camera_factory(fresh_db):
    """Create a camera on the standard test corridor (~113 m spacing)."""
    from app.grid import repository as repo

    base = (23.0225, 72.5714)
    step = 0.001019

    def _make(index: int = 0, name: str = None, is_demo: bool = False):
        return repo.create_camera(
            fresh_db,
            name=name or f"Test Camera {index}",
            source_type="demo" if is_demo else "device",
            source_uri="demo" if is_demo else "0",
            latitude=base[0] + index * step,
            longitude=base[1],
            is_demo=is_demo,
        )

    return _make
