"""
FastAPI application entry point for the Numeric Sequence Analysis backend.

Initialises all middleware, routers, event handlers, WebSocket support,
static file serving, and global exception handlers.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import analysis, experiments, reports, uploads
from backend.core.config import settings
from backend.db.session import close_db, init_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upload / static directories
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Redis connection holder (module-level so routes can import it)
# ---------------------------------------------------------------------------

redis_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return the active Redis client, raising if not initialised."""
    if redis_client is None:
        raise RuntimeError("Redis is not connected. Application may not have started yet.")
    return redis_client


# ---------------------------------------------------------------------------
# Lifespan context manager (replaces deprecated on_event handlers)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Handle startup and shutdown side-effects."""
    # ---- Startup ----
    logger.info("Starting up: initialising database …")
    await init_db()
    logger.info("Database initialised.")

    logger.info("Starting up: connecting to Redis at %s …", settings.REDIS_URL)
    global redis_client  # noqa: PLW0603
    try:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
        )
        await redis_client.ping()
        logger.info("Redis connected.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not connect to Redis (%s). Real-time features may be unavailable.", exc)
        redis_client = None

    # Make redis_client available via app state for dependency injection
    app.state.redis = redis_client

    yield  # ---- Application running ----

    # ---- Shutdown ----
    logger.info("Shutting down: closing database connections …")
    await close_db()
    logger.info("Database closed.")

    if redis_client is not None:
        logger.info("Shutting down: closing Redis connection …")
        await redis_client.aclose()
        logger.info("Redis connection closed.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""

    app = FastAPI(
        title=settings.APP_TITLE,
        description=(
            "REST + WebSocket API for numeric sequence analysis: "
            "upload Excel data, run statistical diagnostics, "
            "change-point detection, machine-learning forecasting, "
            "ensemble modelling, and report generation."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # CORS
    # -----------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    # -----------------------------------------------------------------------
    # Routers
    # -----------------------------------------------------------------------
    app.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
    app.include_router(uploads.router, prefix="/uploads", tags=["Uploads"])
    app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
    app.include_router(reports.router, prefix="/reports", tags=["Reports"])

    # -----------------------------------------------------------------------
    # Static files (uploaded artefacts)
    # -----------------------------------------------------------------------
    app.mount(
        "/static/uploads",
        StaticFiles(directory=str(UPLOAD_DIR)),
        name="uploads",
    )

    # -----------------------------------------------------------------------
    # Global exception handlers
    # -----------------------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "body": exc.body,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        logger.warning(
            "HTTP %s on %s: %s", exc.status_code, request.url.path, exc.detail
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected internal error occurred."},
        )

    # -----------------------------------------------------------------------
    # Health-check endpoint
    # -----------------------------------------------------------------------

    @app.get(
        "/health",
        tags=["Health"],
        summary="Service liveness check",
        response_model=dict[str, Any],
    )
    async def health_check(request: Request) -> dict[str, Any]:
        """
        Returns the operational status of the service and its dependencies.
        Suitable for use by load-balancers and container orchestrators.
        """
        redis_ok = False
        if request.app.state.redis is not None:
            try:
                await request.app.state.redis.ping()
                redis_ok = True
            except Exception:  # noqa: BLE001
                pass

        return {
            "status": "ok",
            "version": settings.APP_VERSION,
            "redis": "connected" if redis_ok else "unavailable",
        }

    # -----------------------------------------------------------------------
    # Root redirect
    # -----------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            {"message": "Numeric Sequence Analysis API. See /docs for documentation."}
        )

    return app


# ---------------------------------------------------------------------------
# Application instance (used by Uvicorn / Gunicorn)
# ---------------------------------------------------------------------------

app = create_app()
