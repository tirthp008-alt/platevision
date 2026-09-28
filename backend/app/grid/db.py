"""Database engine/session management for Drishti Grid.

Defaults to SQLite for zero-setup local development. When ``GRID_DATABASE_URL``
points at PostgreSQL, an optional PostGIS ``geography(Point,4326)`` column and
GiST index are added to the ``cameras`` table so geographic operations can be
pushed down to the database. The application logic remains fully functional
without PostGIS.
"""

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.logging import logger
from app.grid.models import Base

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None
_init_lock = threading.Lock()
_initialized = False


def _make_engine() -> Engine:
    url = settings.GRID_DATABASE_URL
    connect_args = {}
    if url.startswith("sqlite"):
        # Allow the connection to be shared across the ingestion threads.
        connect_args = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True, future=True)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):  # pragma: no cover - driver hook
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = _make_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionLocal


def init_db() -> Engine:
    """Create all tables (idempotent) and enable PostGIS when available."""
    global _initialized
    with _init_lock:
        engine = get_engine()
        if _initialized:
            return engine
        Base.metadata.create_all(engine)
        _try_enable_postgis(engine)
        _initialized = True
        logger.info(f"Drishti Grid database ready at {settings.GRID_DATABASE_URL}")
        return engine


def _try_enable_postgis(engine: Engine) -> None:
    """Best-effort PostGIS enablement; silently skipped for SQLite."""
    if engine.dialect.name != "postgresql":
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            conn.execute(
                text(
                    "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS geom geography(Point,4326)"
                )
            )
            conn.execute(
                text(
                    "UPDATE cameras SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography "
                    "WHERE geom IS NULL AND latitude IS NOT NULL"
                )
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_cameras_geom ON cameras USING GIST (geom)")
            )
        logger.info("PostGIS geographic column and GiST index enabled on cameras table.")
    except Exception as e:  # pragma: no cover - depends on DB privileges
        logger.warning(f"PostGIS enablement skipped: {e}")


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context manager."""
    init_db()
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def reset_for_tests() -> None:
    """Drop and recreate all tables (test helper)."""
    global _engine, _SessionLocal, _initialized
    with _init_lock:
        if _engine is not None:
            _engine.dispose()
        _engine = _make_engine()
        _SessionLocal = None
        _initialized = False
        Base.metadata.drop_all(_engine)
        Base.metadata.create_all(_engine)
        _initialized = True
