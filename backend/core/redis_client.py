"""
Redis client setup with connection pool, JSON serialization helpers,
pub/sub support, and task-queue utilities.

Public surface
--------------
- ``redis_client``   – shared async Redis instance (aioredis / redis-py ≥ 4)
- ``get_redis``      – FastAPI dependency that yields the shared client
- ``RedisCache``     – high-level async helper (get/set/delete/pub/sub)
- ``TaskQueue``      – lightweight wrapper for simple task-queue patterns
- ``check_redis_connection`` – liveness probe
- ``close_redis``    – graceful shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncGenerator, AsyncIterator, Optional, Union

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool, Redis
from redis.asyncio.client import PubSub

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL type alias
# ---------------------------------------------------------------------------
TTL = Union[int, timedelta, None]

# ---------------------------------------------------------------------------
# Connection pool & shared client
# ---------------------------------------------------------------------------

_pool: ConnectionPool = aioredis.ConnectionPool.from_url(
    settings.redis_url,
    password=settings.redis_password or None,
    decode_responses=True,          # Always work with str, not bytes
    max_connections=50,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True,
)

redis_client: Redis = aioredis.Redis(connection_pool=_pool)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_redis() -> AsyncGenerator[Redis, None]:
    """
    FastAPI dependency that yields the shared Redis client.

    Usage::

        @router.get("/ping")
        async def ping(r: Redis = Depends(get_redis)):
            return await r.ping()
    """
    yield redis_client


# ---------------------------------------------------------------------------
# JSON serialisation helpers (module-level convenience functions)
# ---------------------------------------------------------------------------


def _dumps(value: Any) -> str:
    """Serialise *value* to a JSON string."""
    return json.dumps(value, default=str)


def _loads(raw: Optional[str]) -> Any:
    """Deserialise *raw* from JSON; return ``None`` for missing keys."""
    if raw is None:
        return None
    return json.loads(raw)


def _ttl_seconds(ttl: TTL) -> Optional[int]:
    """Normalise *ttl* to an integer number of seconds (or ``None``)."""
    if ttl is None:
        return None
    if isinstance(ttl, timedelta):
        return int(ttl.total_seconds())
    return int(ttl)


# ---------------------------------------------------------------------------
# RedisCache – high-level async helper
# ---------------------------------------------------------------------------


class RedisCache:
    """
    High-level, JSON-aware async Redis helper.

    All keys accept an optional namespace prefix to avoid collisions between
    application modules::

        cache = RedisCache(namespace="experiments")
        await cache.set("abc123", {"status": "running"}, ttl=300)
        data = await cache.get("abc123")
    """

    def __init__(
        self,
        client: Redis = redis_client,
        namespace: str = "",
        default_ttl: TTL = None,
    ) -> None:
        self._client = client
        self._ns = f"{namespace}:" if namespace else ""
        self._default_ttl = _ttl_seconds(default_ttl)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _key(self, key: str) -> str:
        return f"{self._ns}{key}"

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any:
        """
        Retrieve and deserialise the value stored at *key*.

        Returns ``None`` if the key does not exist or has expired.
        """
        raw = await self._client.get(self._key(key))
        return _loads(raw)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: TTL = None,
    ) -> bool:
        """
        Serialise and store *value* at *key*.

        Parameters
        ----------
        key:
            Cache key (namespace prefix is prepended automatically).
        value:
            Any JSON-serialisable Python object.
        ttl:
            Time-to-live in seconds, as a ``timedelta``, or ``None`` for no
            expiry.  Falls back to ``default_ttl`` when not supplied.

        Returns ``True`` on success.
        """
        seconds = _ttl_seconds(ttl) if ttl is not None else self._default_ttl
        serialised = _dumps(value)
        if seconds:
            result = await self._client.setex(self._key(key), seconds, serialised)
        else:
            result = await self._client.set(self._key(key), serialised)
        return bool(result)

    async def delete(self, *keys: str) -> int:
        """
        Remove one or more keys.  Returns the number of keys deleted.
        """
        prefixed = [self._key(k) for k in keys]
        return await self._client.delete(*prefixed)

    async def exists(self, key: str) -> bool:
        """Return ``True`` if *key* exists in Redis."""
        return bool(await self._client.exists(self._key(key)))

    async def expire(self, key: str, ttl: TTL) -> bool:
        """Update the TTL of an existing key. Returns ``True`` on success."""
        seconds = _ttl_seconds(ttl)
        if seconds is None:
            return False
        return bool(await self._client.expire(self._key(key), seconds))

    async def ttl(self, key: str) -> int:
        """
        Return the remaining TTL of *key* in seconds.

        Returns:
        - Positive integer: remaining seconds.
        - ``-1``: key exists but has no expiry.
        - ``-2``: key does not exist.
        """
        return await self._client.ttl(self._key(key))

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    async def mget(self, *keys: str) -> dict[str, Any]:
        """
        Retrieve multiple keys in a single round-trip.

        Returns a ``{key: value}`` mapping; missing keys map to ``None``.
        """
        prefixed = [self._key(k) for k in keys]
        raws = await self._client.mget(*prefixed)
        return {k: _loads(r) for k, r in zip(keys, raws)}

    async def mset(self, mapping: dict[str, Any], ttl: TTL = None) -> None:
        """
        Store multiple key-value pairs in a single pipeline.

        If *ttl* is provided every key receives the same expiry.
        """
        seconds = _ttl_seconds(ttl) if ttl is not None else self._default_ttl
        async with self._client.pipeline(transaction=False) as pipe:
            for k, v in mapping.items():
                serialised = _dumps(v)
                if seconds:
                    pipe.setex(self._key(k), seconds, serialised)
                else:
                    pipe.set(self._key(k), serialised)
            await pipe.execute()

    # ------------------------------------------------------------------
    # Hash helpers (for structured objects)
    # ------------------------------------------------------------------

    async def hset(self, hash_key: str, field: str, value: Any) -> int:
        """Store a JSON-serialised field inside a Redis hash."""
        return await self._client.hset(self._key(hash_key), field, _dumps(value))

    async def hget(self, hash_key: str, field: str) -> Any:
        """Retrieve and deserialise a single hash field."""
        raw = await self._client.hget(self._key(hash_key), field)
        return _loads(raw)

    async def hgetall(self, hash_key: str) -> dict[str, Any]:
        """Retrieve and deserialise all fields of a Redis hash."""
        raw_map: dict[str, str] = await self._client.hgetall(self._key(hash_key))
        return {field: _loads(val) for field, val in raw_map.items()}

    # ------------------------------------------------------------------
    # Pub / Sub
    # ------------------------------------------------------------------

    async def publish(self, channel: str, message: Any) -> int:
        """
        Publish a JSON-serialised *message* to *channel*.

        Returns the number of subscribers that received the message.
        """
        return await self._client.publish(self._key(channel), _dumps(message))

    @asynccontextmanager
    async def subscribe(self, *channels: str) -> AsyncIterator[PubSub]:
        """
        Async context manager that returns a ``PubSub`` object subscribed to
        the given channels.

        Usage::

            async with cache.subscribe("updates") as pubsub:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        ...
        """
        pubsub: PubSub = self._client.pubsub()
        prefixed = [self._key(c) for c in channels]
        await pubsub.subscribe(*prefixed)
        try:
            yield pubsub
        finally:
            await pubsub.unsubscribe(*prefixed)
            await pubsub.close()


# ---------------------------------------------------------------------------
# TaskQueue – lightweight Redis-list based queue
# ---------------------------------------------------------------------------


class TaskQueue:
    """
    Simple FIFO task queue backed by a Redis list.

    This is intentionally minimal; for production workloads use Celery (which
    is already configured).  ``TaskQueue`` is suitable for lightweight,
    in-process coordination patterns.

    Usage::

        queue = TaskQueue("preprocessing")
        await queue.push({"experiment_id": "abc", "step": "validate"})
        task = await queue.pop()
    """

    def __init__(
        self,
        name: str,
        client: Redis = redis_client,
        namespace: str = "taskqueue",
    ) -> None:
        self._client = client
        self._key = f"{namespace}:{name}"
        self._processing_key = f"{namespace}:{name}:processing"

    async def push(self, task: Any) -> int:
        """
        Enqueue *task* (JSON-serialised) at the tail of the list.

        Returns the current queue length.
        """
        return await self._client.rpush(self._key, _dumps(task))

    async def pop(self, timeout: int = 0) -> Optional[Any]:
        """
        Dequeue a task from the head of the list.

        If *timeout* is 0 the call returns immediately (``None`` if empty).
        If *timeout* > 0 the call blocks up to *timeout* seconds (BLPOP).
        """
        if timeout > 0:
            result = await self._client.blpop(self._key, timeout=timeout)
            if result is None:
                return None
            _, raw = result
            return _loads(raw)
        else:
            raw = await self._client.lpop(self._key)
            return _loads(raw)

    async def length(self) -> int:
        """Return the number of tasks currently in the queue."""
        return await self._client.llen(self._key)

    async def peek(self, count: int = 10) -> list[Any]:
        """
        Return the first *count* tasks without removing them.
        """
        raws = await self._client.lrange(self._key, 0, count - 1)
        return [_loads(r) for r in raws]

    async def clear(self) -> None:
        """Remove all tasks from the queue."""
        await self._client.delete(self._key)


# ---------------------------------------------------------------------------
# Distributed lock helper
# ---------------------------------------------------------------------------


class DistributedLock:
    """
    Simple Redis-based distributed lock using SET NX EX.

    Usage::

        async with DistributedLock("train:experiment-123", ttl=120):
            await run_training(...)
    """

    def __init__(
        self,
        name: str,
        ttl: int = 60,
        client: Redis = redis_client,
        namespace: str = "lock",
    ) -> None:
        self._client = client
        self._key = f"{namespace}:{name}"
        self._ttl = ttl
        self._acquired = False

    async def acquire(self, retry: int = 3, retry_delay: float = 0.5) -> bool:
        """
        Attempt to acquire the lock.

        Retries up to *retry* times with *retry_delay* seconds between attempts.
        Returns ``True`` if the lock was acquired, ``False`` otherwise.
        """
        for attempt in range(retry):
            result = await self._client.set(
                self._key, "1", nx=True, ex=self._ttl
            )
            if result:
                self._acquired = True
                return True
            if attempt < retry - 1:
                await asyncio.sleep(retry_delay)
        return False

    async def release(self) -> None:
        """Release the lock if it was acquired by this instance."""
        if self._acquired:
            await self._client.delete(self._key)
            self._acquired = False

    async def __aenter__(self) -> "DistributedLock":
        acquired = await self.acquire()
        if not acquired:
            raise RuntimeError(f"Could not acquire distributed lock '{self._key}'")
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.release()


# ---------------------------------------------------------------------------
# Liveness probe
# ---------------------------------------------------------------------------


async def check_redis_connection() -> bool:
    """
    Verify connectivity to the Redis server.

    Returns ``True`` if the server responds to PING, ``False`` otherwise.
    Suitable for use in health-check / readiness-probe endpoints.
    """
    try:
        return await redis_client.ping()
    except Exception as exc:
        logger.error("Redis connectivity check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


async def close_redis() -> None:
    """
    Close the shared Redis client and dispose the underlying connection pool.

    Should be called during application shutdown (e.g. FastAPI lifespan).
    """
    await redis_client.aclose()
    await _pool.aclose()
    logger.info("Redis connection pool closed.")


# ---------------------------------------------------------------------------
# Pre-configured module-level instances
# ---------------------------------------------------------------------------

#: Default cache instance (no namespace).
cache: RedisCache = RedisCache(client=redis_client, default_ttl=timedelta(hours=1))

#: Namespace-isolated cache for experiment-level results.
experiment_cache: RedisCache = RedisCache(
    client=redis_client,
    namespace="experiments",
    default_ttl=timedelta(hours=6),
)

#: General-purpose task queue.
default_queue: TaskQueue = TaskQueue(name="default", client=redis_client)
