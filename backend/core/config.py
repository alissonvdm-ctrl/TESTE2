"""
Application configuration using Pydantic Settings.

Reads settings from environment variables and .env files, providing
typed, validated configuration for all application components.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration class for the NumericSequenceAnalyzer application.

    All settings can be overridden via environment variables or a .env file
    located at the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = Field(default="NumericSequenceAnalyzer", description="Human-readable application name")
    app_env: str = Field(default="development", description="Runtime environment (development | staging | production)")
    debug: bool = Field(default=True, description="Enable debug mode")
    secret_key: str = Field(
        default="change-me-in-production-use-strong-random-key",
        description="Secret key used for signing tokens and sessions",
    )

    # -------------------------------------------------------------------------
    # Backend server
    # -------------------------------------------------------------------------
    backend_host: str = Field(default="0.0.0.0", description="Host address the Uvicorn server binds to")
    backend_port: int = Field(default=8000, ge=1, le=65535, description="Port the Uvicorn server listens on")
    workers: int = Field(default=4, ge=1, description="Number of Uvicorn worker processes")

    # -------------------------------------------------------------------------
    # PostgreSQL / SQLAlchemy
    # -------------------------------------------------------------------------
    postgres_host: str = Field(default="postgres", description="PostgreSQL server hostname")
    postgres_port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL server port")
    postgres_db: str = Field(default="sequence_analyzer", description="PostgreSQL database name")
    postgres_user: str = Field(default="analyst", description="PostgreSQL username")
    postgres_password: str = Field(default="analyst_password", description="PostgreSQL password")

    database_url: str = Field(
        default="postgresql+asyncpg://analyst:analyst_password@postgres:5432/sequence_analyzer",
        description="Async SQLAlchemy database URL (uses asyncpg driver)",
    )
    sync_database_url: str = Field(
        default="postgresql://analyst:analyst_password@postgres:5432/sequence_analyzer",
        description="Sync SQLAlchemy database URL (used for Alembic migrations)",
    )

    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------
    redis_host: str = Field(default="redis", description="Redis server hostname")
    redis_port: int = Field(default=6379, ge=1, le=65535, description="Redis server port")
    redis_password: Optional[str] = Field(default=None, description="Redis AUTH password (empty means no auth)")
    redis_db: int = Field(default=0, ge=0, description="Redis logical database index for general use")
    redis_url: str = Field(default="redis://redis:6379/0", description="Full Redis connection URL")

    # -------------------------------------------------------------------------
    # Celery
    # -------------------------------------------------------------------------
    celery_broker_url: str = Field(
        default="redis://redis:6379/1",
        description="URL for the Celery message broker",
    )
    celery_result_backend: str = Field(
        default="redis://redis:6379/2",
        description="URL for the Celery result backend",
    )

    # -------------------------------------------------------------------------
    # File storage
    # -------------------------------------------------------------------------
    upload_dir: Path = Field(default=Path("/app/uploads"), description="Directory where uploaded files are stored")
    max_upload_size_mb: int = Field(default=100, ge=1, description="Maximum allowed upload size in megabytes")

    # -------------------------------------------------------------------------
    # Machine Learning
    # -------------------------------------------------------------------------
    max_training_time_seconds: int = Field(
        default=3600, ge=1, description="Hard wall-clock limit for a single model training run (seconds)"
    )
    default_cv_splits: int = Field(default=5, ge=2, description="Default number of cross-validation folds")
    random_seed: int = Field(default=42, description="Global random seed for reproducibility")

    # -------------------------------------------------------------------------
    # MLflow (optional)
    # -------------------------------------------------------------------------
    mlflow_tracking_uri: Optional[str] = Field(
        default="http://mlflow:5000", description="URI of the MLflow tracking server"
    )
    mlflow_experiment_name: str = Field(
        default="sequence_analyzer", description="MLflow experiment name to log runs under"
    )

    # -------------------------------------------------------------------------
    # API metadata
    # -------------------------------------------------------------------------
    app_title: str = Field(default="Numeric Sequence Analyzer API", description="FastAPI app title")
    app_version: str = Field(default="1.0.0", description="API version string")
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "http://localhost:80"],
        description="CORS allowed origins",
    )

    # -------------------------------------------------------------------------
    # Frontend (injected at build time via Vite)
    # -------------------------------------------------------------------------
    vite_api_base_url: str = Field(
        default="http://localhost:8000", description="Base URL the frontend uses to reach the API"
    )
    vite_ws_url: str = Field(
        default="ws://localhost:8000", description="WebSocket URL the frontend connects to"
    )

    # -------------------------------------------------------------------------
    # GPU / CUDA
    # -------------------------------------------------------------------------
    use_gpu: bool = Field(default=False, description="Whether to enable GPU acceleration for ML training")
    cuda_visible_devices: str = Field(default="0", description="Comma-separated list of CUDA device indices to expose")

    # -------------------------------------------------------------------------
    # Derived / computed helpers
    # -------------------------------------------------------------------------

    @field_validator("redis_password", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> Optional[str]:
        """Convert an empty string Redis password to None (no-auth mode)."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v  # type: ignore[return-value]

    @field_validator("upload_dir", mode="before")
    @classmethod
    def _coerce_upload_dir(cls, v: object) -> Path:
        """Accept both string and Path for upload_dir."""
        return Path(str(v))

    @property
    def max_upload_size_bytes(self) -> int:
        """Return the maximum upload size converted to bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Return True when running in production mode."""
        return self.app_env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Return True when running in development mode."""
        return self.app_env.lower() == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton of the application Settings.

    Using ``lru_cache`` ensures the .env file is parsed only once per process
    lifetime, which is important both for performance and for test isolation
    (tests can call ``get_settings.cache_clear()`` before patching env vars).
    """
    return Settings()


# Convenience alias used throughout the codebase
settings: Settings = get_settings()
