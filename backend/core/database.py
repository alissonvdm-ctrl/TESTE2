"""
Async SQLAlchemy database setup.

Provides:
- Async engine and session factory backed by asyncpg.
- Declarative ``Base`` class shared by all ORM models.
- ``get_async_session`` dependency for FastAPI route injection.
- ``get_sync_session`` context-manager for synchronous code paths
  (e.g. Alembic migrations, Celery tasks).
- Utility helpers: ``init_db``, ``drop_db``, and ``check_db_connection``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy import create_engine

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Async engine
# ---------------------------------------------------------------------------

async_engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,          # Log SQL statements in debug mode
    pool_pre_ping=True,           # Verify connections before checkout
    pool_size=10,                 # Baseline connection pool size
    max_overflow=20,              # Extra connections allowed under load
    pool_recycle=1800,            # Recycle connections every 30 minutes
    pool_timeout=30,              # Seconds to wait for a pool connection
    future=True,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,       # Keep attribute access after commit
    autocommit=False,
    autoflush=False,
)

# ---------------------------------------------------------------------------
# Sync engine (Alembic / Celery tasks)
# ---------------------------------------------------------------------------

sync_engine = create_engine(
    settings.sync_database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,
    future=True,
)

SyncSessionLocal: sessionmaker[Session] = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """
    Shared declarative base for all SQLAlchemy ORM models.

    All models that inherit from this class are automatically registered with
    both the async and sync engines' metadata objects.
    """

    pass


# ---------------------------------------------------------------------------
# FastAPI dependency: async session
# ---------------------------------------------------------------------------


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a managed ``AsyncSession``.

    Usage::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_async_session)):
            ...

    The session is committed on success and rolled back on any exception.
    It is always closed when the request completes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Context-manager helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for obtaining a session outside of FastAPI.

    Usage::

        async with get_async_db() as db:
            result = await db.execute(select(Experiment))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    """
    Synchronous context manager for obtaining a session.

    Intended for Celery tasks and Alembic migration scripts that cannot use
    the async engine.

    Usage::

        with get_sync_db() as db:
            db.add(some_object)
    """
    session: Session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Database lifecycle helpers
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """
    Create all tables defined in ``Base.metadata``.

    This is a lightweight alternative to running Alembic migrations and is
    useful for integration tests or fresh development environments.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created (init_db).")


async def drop_db() -> None:
    """
    Drop all tables defined in ``Base.metadata``.

    **Warning**: This is destructive and intended for test teardown only.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables dropped (drop_db).")


async def check_db_connection() -> bool:
    """
    Verify that the async database engine can reach the server.

    Returns ``True`` if a round-trip query succeeds, ``False`` otherwise.
    Useful for liveness / readiness probe endpoints.
    """
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database connectivity check failed: %s", exc)
        return False


async def close_db() -> None:
    """
    Dispose the async engine's connection pool.

    Should be called during application shutdown to release all held
    connections gracefully.
    """
    await async_engine.dispose()
    logger.info("Async database engine disposed.")
