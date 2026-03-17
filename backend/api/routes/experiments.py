"""
Experiments router.

Provides CRUD operations and lifecycle management for analysis experiments,
including creating, updating, deleting experiments and triggering the
analysis pipeline.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_async_session as get_db

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ExperimentCreate(BaseModel):
    """Payload for creating a new experiment."""

    name: str = Field(..., min_length=1, max_length=255, description="Human-readable experiment name")
    description: Optional[str] = Field(default=None, max_length=2000, description="Optional description")
    upload_id: str = Field(..., description="ID of the previously uploaded dataset")
    sequence_column: str = Field(..., description="Column name containing the numeric sequence")
    n_value: Optional[int] = Field(default=None, ge=1, description="N parameter for analysis window")
    k_value: Optional[int] = Field(default=None, ge=1, description="k parameter for lag/step")
    config: Optional[dict[str, Any]] = Field(default=None, description="Additional algorithm configuration")


class ExperimentUpdate(BaseModel):
    """Payload for updating an existing experiment's configuration."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=2000)
    sequence_column: Optional[str] = None
    n_value: Optional[int] = Field(default=None, ge=1)
    k_value: Optional[int] = Field(default=None, ge=1)
    config: Optional[dict[str, Any]] = None


class ExperimentResponse(BaseModel):
    """Full experiment representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: Optional[str]
    upload_id: str
    sequence_column: str
    n_value: Optional[int]
    k_value: Optional[int]
    config: Optional[dict[str, Any]]
    status: str
    created_at: datetime
    updated_at: Optional[datetime]


class ExperimentListResponse(BaseModel):
    """Paginated list of experiments."""

    total: int
    page: int
    page_size: int
    items: list[ExperimentResponse]


class ExperimentStatusResponse(BaseModel):
    """Real-time status of an experiment's processing pipeline."""

    experiment_id: str
    status: str
    stage: Optional[str] = None
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class PredictionItem(BaseModel):
    """Single prediction data point."""

    index: int
    actual: Optional[float] = None
    predicted: float
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None
    model_name: Optional[str] = None


class PredictionsResponse(BaseModel):
    """Collection of predictions for an experiment."""

    experiment_id: str
    model_name: str
    horizon: int
    predictions: list[PredictionItem]
    metrics: Optional[dict[str, float]] = None


class ResultSummary(BaseModel):
    """High-level summary of analysis results."""

    experiment_id: str
    diagnostic: Optional[dict[str, Any]] = None
    best_model: Optional[str] = None
    ensemble_score: Optional[float] = None
    n_changepoints: Optional[int] = None
    features_count: Optional[int] = None
    completed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Dependency: resolve experiment or 404
# ---------------------------------------------------------------------------


async def get_experiment_or_404(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Fetch an experiment record by ID, raising HTTP 404 if not found.

    In a full implementation this would query the ORM model; here we return
    a stub dict so the route layer remains decoupled from ORM internals until
    the models module is populated.
    """
    # TODO: replace with actual ORM query once db/models.py is populated
    # e.g.:
    #   from backend.db.models import Experiment
    #   result = await db.get(Experiment, experiment_id)
    #   if result is None:
    #       raise HTTPException(status_code=404, detail="Experiment not found")
    #   return result

    # Stub: always raise 404 for unknown IDs during development
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Experiment '{experiment_id}' not found.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new experiment",
)
async def create_experiment(
    payload: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """
    Create a new analysis experiment linked to a validated upload.

    The experiment is created in *pending* state. Call ``POST
    /experiments/{id}/start`` to trigger the full analysis pipeline.
    """
    logger.info("Creating experiment '%s' for upload '%s'", payload.name, payload.upload_id)

    # TODO: persist to DB via ORM
    new_id = str(uuid.uuid4())
    now = datetime.utcnow()

    return ExperimentResponse(
        id=new_id,
        name=payload.name,
        description=payload.description,
        upload_id=payload.upload_id,
        sequence_column=payload.sequence_column,
        n_value=payload.n_value,
        k_value=payload.k_value,
        config=payload.config,
        status="pending",
        created_at=now,
        updated_at=None,
    )


@router.get(
    "",
    response_model=ExperimentListResponse,
    summary="List all experiments",
)
async def list_experiments(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(default=None, alias="status", description="Filter by status"),
    db: AsyncSession = Depends(get_db),
) -> ExperimentListResponse:
    """
    Return a paginated list of experiments, optionally filtered by status.

    Supported status values: ``pending``, ``running``, ``completed``, ``failed``.
    """
    logger.info("Listing experiments (page=%d, page_size=%d, status=%s)", page, page_size, status_filter)

    # TODO: query DB, apply filters and pagination
    return ExperimentListResponse(
        total=0,
        page=page,
        page_size=page_size,
        items=[],
    )


@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get experiment details",
)
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """Return the full details of a single experiment."""
    experiment = await get_experiment_or_404(experiment_id, db)
    return experiment  # type: ignore[return-value]


@router.put(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Update experiment configuration",
)
async def update_experiment(
    experiment_id: str,
    payload: ExperimentUpdate,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """
    Update mutable fields of an experiment.

    Only *pending* experiments may have their configuration changed; an attempt
    to modify a running or completed experiment will return HTTP 409.
    """
    experiment = await get_experiment_or_404(experiment_id, db)
    logger.info("Updating experiment '%s'", experiment_id)

    # TODO: apply updates and persist
    return experiment  # type: ignore[return-value]


@router.delete(
    "/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an experiment",
)
async def delete_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Permanently delete an experiment and all associated results.

    Running experiments are stopped before deletion.
    """
    await get_experiment_or_404(experiment_id, db)
    logger.info("Deleting experiment '%s'", experiment_id)
    # TODO: cancel running tasks, delete DB records, remove artefact files


@router.get(
    "/{experiment_id}/status",
    response_model=ExperimentStatusResponse,
    summary="Get processing status",
)
async def get_experiment_status(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ExperimentStatusResponse:
    """
    Return the current processing status and progress of the analysis pipeline
    for the given experiment.
    """
    await get_experiment_or_404(experiment_id, db)
    logger.info("Fetching status for experiment '%s'", experiment_id)

    # TODO: query Redis / Celery task state for live progress
    return ExperimentStatusResponse(
        experiment_id=experiment_id,
        status="pending",
        stage=None,
        progress_pct=0.0,
        message="Experiment has not been started yet.",
    )


@router.post(
    "/{experiment_id}/start",
    response_model=ExperimentStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start the analysis pipeline",
)
async def start_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ExperimentStatusResponse:
    """
    Enqueue the full analysis pipeline for the experiment.

    The pipeline runs asynchronously.  Poll ``GET /experiments/{id}/status``
    or subscribe to the ``/ws/{experiment_id}`` WebSocket for live updates.
    """
    await get_experiment_or_404(experiment_id, db)
    logger.info("Starting analysis pipeline for experiment '%s'", experiment_id)

    # TODO: enqueue Celery task and update experiment status to "running"
    # e.g.:
    #   from backend.tasks.pipeline import run_analysis_pipeline
    #   task = run_analysis_pipeline.delay(experiment_id)

    return ExperimentStatusResponse(
        experiment_id=experiment_id,
        status="queued",
        stage="initialising",
        progress_pct=0.0,
        message="Analysis pipeline has been queued.",
        started_at=datetime.utcnow(),
    )


@router.get(
    "/{experiment_id}/results",
    response_model=ResultSummary,
    summary="Get all analysis results",
)
async def get_experiment_results(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ResultSummary:
    """
    Return a consolidated summary of all analysis results for the experiment,
    including diagnostic statistics, best model, ensemble score, and change-
    point count.
    """
    await get_experiment_or_404(experiment_id, db)
    logger.info("Fetching results for experiment '%s'", experiment_id)

    # TODO: aggregate results from DB
    return ResultSummary(experiment_id=experiment_id)


@router.get(
    "/{experiment_id}/predictions",
    response_model=PredictionsResponse,
    summary="Get model predictions",
)
async def get_experiment_predictions(
    experiment_id: str,
    model_name: str = Query(default="ensemble", description="Model to fetch predictions for"),
    horizon: int = Query(default=10, ge=1, le=500, description="Forecast horizon (steps ahead)"),
    db: AsyncSession = Depends(get_db),
) -> PredictionsResponse:
    """
    Return the forecast predictions produced by a specific model (or the
    ensemble) for the given experiment.
    """
    await get_experiment_or_404(experiment_id, db)
    logger.info(
        "Fetching '%s' predictions (horizon=%d) for experiment '%s'",
        model_name,
        horizon,
        experiment_id,
    )

    # TODO: load predictions from DB / artefact store
    return PredictionsResponse(
        experiment_id=experiment_id,
        model_name=model_name,
        horizon=horizon,
        predictions=[],
        metrics=None,
    )
