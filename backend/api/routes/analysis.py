"""
Analysis router.

Exposes endpoints for triggering and retrieving the outputs of the full
numeric-sequence analysis pipeline: statistical diagnostics, change-point /
regime detection, individual model results, ensemble forecasts, feature
importance, and a WebSocket channel for real-time progress streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_async_session as get_db

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AnalysisRunRequest(BaseModel):
    """Optional overrides for triggering a full analysis run."""

    force_rerun: bool = Field(
        default=False,
        description="If True, discard cached results and rerun from scratch",
    )
    stages: Optional[list[str]] = Field(
        default=None,
        description=(
            "Subset of pipeline stages to execute. "
            "Omit to run all stages: diagnostic, changepoints, models, ensemble, features."
        ),
    )
    model_subset: Optional[list[str]] = Field(
        default=None,
        description="Restrict model training to this list of model names",
    )


class AnalysisRunResponse(BaseModel):
    """Immediate response after enqueueing an analysis run."""

    experiment_id: str
    task_id: Optional[str] = Field(default=None, description="Celery task ID for tracking")
    status: str
    message: str
    queued_at: datetime


# --- Diagnostic ---

class StationarityTest(BaseModel):
    """Result of a single stationarity test (e.g. ADF, KPSS)."""

    test_name: str
    statistic: float
    p_value: float
    critical_values: Optional[dict[str, float]] = None
    is_stationary: bool
    interpretation: str


class DiagnosticResponse(BaseModel):
    """Statistical diagnostic report for the sequence."""

    experiment_id: str
    series_length: int
    mean: float
    std: float
    skewness: float
    kurtosis: float
    autocorrelation_lag1: Optional[float] = None
    stationarity_tests: list[StationarityTest] = Field(default_factory=list)
    recommended_differencing: int = Field(
        default=0, description="Suggested number of differencing steps"
    )
    notes: list[str] = Field(default_factory=list)
    computed_at: Optional[datetime] = None


# --- Change-points / Regimes ---

class Changepoint(BaseModel):
    """A single detected change-point in the series."""

    index: int = Field(..., description="Row index in the original series")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp if a date column is available")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    regime_before: Optional[str] = None
    regime_after: Optional[str] = None
    description: Optional[str] = None


class RegimeStats(BaseModel):
    """Statistical summary for a single regime segment."""

    regime_id: int
    start_index: int
    end_index: int
    length: int
    mean: float
    std: float
    trend: str = Field(..., description="'increasing', 'decreasing', or 'stationary'")


class ChangepointsResponse(BaseModel):
    """Full change-point / regime analysis result."""

    experiment_id: str
    algorithm: str = Field(..., description="Algorithm used for detection (e.g. 'PELT', 'BinSeg', 'BOCPD')")
    n_changepoints: int
    changepoints: list[Changepoint]
    regimes: list[RegimeStats]
    computed_at: Optional[datetime] = None


# --- Model results ---

class ModelMetrics(BaseModel):
    """Evaluation metrics for a trained model."""

    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    smape: Optional[float] = None
    extra: Optional[dict[str, float]] = None


class ModelResult(BaseModel):
    """Result for a single trained model."""

    model_name: str
    model_type: str = Field(..., description="Family: e.g. 'linear', 'tree', 'neural', 'statistical'")
    status: str = Field(..., description="'success', 'failed', or 'skipped'")
    train_metrics: Optional[ModelMetrics] = None
    val_metrics: Optional[ModelMetrics] = None
    test_metrics: Optional[ModelMetrics] = None
    training_duration_seconds: Optional[float] = None
    hyperparameters: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None


class ModelsResponse(BaseModel):
    """All trained model results for an experiment."""

    experiment_id: str
    total_models: int
    successful_models: int
    best_model: Optional[str] = None
    results: list[ModelResult]
    computed_at: Optional[datetime] = None


# --- Ensemble ---

class EnsembleWeights(BaseModel):
    """Per-model weights used in the ensemble."""

    model_name: str
    weight: float = Field(..., ge=0.0, le=1.0)
    contribution_pct: float


class EnsembleResponse(BaseModel):
    """Ensemble model composition and overall performance."""

    experiment_id: str
    strategy: str = Field(..., description="Combination strategy: 'weighted_avg', 'stacking', 'voting'")
    metrics: Optional[ModelMetrics] = None
    weights: list[EnsembleWeights]
    improvement_over_best_single: Optional[float] = Field(
        default=None, description="RMSE improvement (%) vs best individual model"
    )
    computed_at: Optional[datetime] = None


# --- Feature importance ---

class FeatureImportanceItem(BaseModel):
    """Importance score for a single engineered feature."""

    feature_name: str
    importance_score: float
    rank: int
    category: Optional[str] = Field(
        default=None,
        description="Feature category: 'lag', 'rolling_stat', 'calendar', 'spectral', etc.",
    )
    description: Optional[str] = None


class FeaturesResponse(BaseModel):
    """Feature importance ranking for the experiment."""

    experiment_id: str
    model_name: str = Field(..., description="Model whose feature importances are reported")
    total_features: int
    features: list[FeatureImportanceItem]
    computed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Dependency: resolve experiment or 404
# ---------------------------------------------------------------------------


async def _require_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> str:
    """
    Verify the experiment exists and return its ID.

    Raises HTTP 404 if not found.
    """
    # TODO: replace with actual ORM lookup once models are defined
    # from backend.db.models import Experiment
    # result = await db.get(Experiment, experiment_id)
    # if result is None:
    #     raise HTTPException(status_code=404, detail="Experiment not found")
    # return experiment_id

    # Stub: always raise until ORM is wired up
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Experiment '{experiment_id}' not found.",
    )


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@router.post(
    "/{experiment_id}/run",
    response_model=AnalysisRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger the full analysis pipeline",
)
async def run_analysis(
    experiment_id: str,
    payload: AnalysisRunRequest,
    db: AsyncSession = Depends(get_db),
) -> AnalysisRunResponse:
    """
    Enqueue the complete analysis pipeline for the given experiment.

    The pipeline covers: statistical diagnostics → change-point detection →
    feature engineering → model training → ensemble → report artefacts.

    Subscribe to ``WS /ws/{experiment_id}`` for live progress, or poll
    ``GET /experiments/{experiment_id}/status``.
    """
    await _require_experiment(experiment_id, db)

    logger.info(
        "Enqueueing analysis run for experiment '%s' (force_rerun=%s, stages=%s)",
        experiment_id,
        payload.force_rerun,
        payload.stages,
    )

    # TODO: dispatch Celery task
    # from backend.tasks.pipeline import run_analysis_pipeline
    # task = run_analysis_pipeline.apply_async(
    #     args=[experiment_id],
    #     kwargs={"force_rerun": payload.force_rerun, "stages": payload.stages},
    # )
    # task_id = task.id

    return AnalysisRunResponse(
        experiment_id=experiment_id,
        task_id=None,  # replace with task.id once wired
        status="queued",
        message="Analysis pipeline has been queued successfully.",
        queued_at=datetime.utcnow(),
    )


@router.get(
    "/{experiment_id}/diagnostic",
    response_model=DiagnosticResponse,
    summary="Get statistical diagnostic report",
)
async def get_diagnostic(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> DiagnosticResponse:
    """
    Return the statistical diagnostic report for the experiment's sequence,
    including descriptive statistics, stationarity tests, and differencing
    recommendations.
    """
    await _require_experiment(experiment_id, db)
    logger.info("Fetching diagnostic for experiment '%s'", experiment_id)

    # TODO: load from DB / result store
    return DiagnosticResponse(
        experiment_id=experiment_id,
        series_length=0,
        mean=0.0,
        std=0.0,
        skewness=0.0,
        kurtosis=0.0,
    )


@router.get(
    "/{experiment_id}/changepoints",
    response_model=ChangepointsResponse,
    summary="Get regime / change-point analysis",
)
async def get_changepoints(
    experiment_id: str,
    algorithm: str = Query(default="PELT", description="Detection algorithm to retrieve results for"),
    db: AsyncSession = Depends(get_db),
) -> ChangepointsResponse:
    """
    Return the change-point detection results for the experiment, including
    the list of detected change-points and per-regime statistical summaries.
    """
    await _require_experiment(experiment_id, db)
    logger.info(
        "Fetching changepoints (%s) for experiment '%s'", algorithm, experiment_id
    )

    # TODO: load from DB / result store
    return ChangepointsResponse(
        experiment_id=experiment_id,
        algorithm=algorithm,
        n_changepoints=0,
        changepoints=[],
        regimes=[],
    )


@router.get(
    "/{experiment_id}/models",
    response_model=ModelsResponse,
    summary="Get all model training results",
)
async def get_models(
    experiment_id: str,
    model_type: Optional[str] = Query(default=None, description="Filter by model family"),
    db: AsyncSession = Depends(get_db),
) -> ModelsResponse:
    """
    Return results for all models trained during the experiment, optionally
    filtered by model family (e.g. ``linear``, ``tree``, ``neural``).
    """
    await _require_experiment(experiment_id, db)
    logger.info("Fetching model results for experiment '%s' (type=%s)", experiment_id, model_type)

    # TODO: load from DB / result store
    return ModelsResponse(
        experiment_id=experiment_id,
        total_models=0,
        successful_models=0,
        results=[],
    )


@router.get(
    "/{experiment_id}/ensemble",
    response_model=EnsembleResponse,
    summary="Get ensemble model results",
)
async def get_ensemble(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> EnsembleResponse:
    """
    Return the ensemble model composition, per-model weights, combined
    performance metrics, and improvement over the best individual model.
    """
    await _require_experiment(experiment_id, db)
    logger.info("Fetching ensemble results for experiment '%s'", experiment_id)

    # TODO: load from DB / result store
    return EnsembleResponse(
        experiment_id=experiment_id,
        strategy="weighted_avg",
        weights=[],
    )


@router.get(
    "/{experiment_id}/features",
    response_model=FeaturesResponse,
    summary="Get feature importance ranking",
)
async def get_features(
    experiment_id: str,
    model_name: str = Query(
        default="ensemble",
        description="Model whose feature importances to return",
    ),
    top_n: int = Query(default=50, ge=1, le=500, description="Return only the top N features"),
    db: AsyncSession = Depends(get_db),
) -> FeaturesResponse:
    """
    Return the ranked feature importance scores for the given model, limited
    to the top ``top_n`` features.

    Features cover lag embeddings, rolling statistics, calendar variables,
    and spectral decomposition components.
    """
    await _require_experiment(experiment_id, db)
    logger.info(
        "Fetching feature importance (model=%s, top_n=%d) for experiment '%s'",
        model_name,
        top_n,
        experiment_id,
    )

    # TODO: load from DB / result store
    return FeaturesResponse(
        experiment_id=experiment_id,
        model_name=model_name,
        total_features=0,
        features=[],
    )


# ---------------------------------------------------------------------------
# WebSocket — real-time progress
# ---------------------------------------------------------------------------


class _ProgressMessage(BaseModel):
    """Schema for a single progress event pushed over the WebSocket."""

    model_config = ConfigDict(from_attributes=True)

    experiment_id: str
    stage: str
    status: str
    progress_pct: float = Field(..., ge=0.0, le=100.0)
    message: str
    timestamp: str


@router.websocket("/ws/{experiment_id}")
async def websocket_progress(
    websocket: WebSocket,
    experiment_id: str,
) -> None:
    """
    WebSocket endpoint for real-time analysis progress updates.

    The server pushes JSON-serialised ``ProgressMessage`` objects as each
    pipeline stage starts, updates, and completes.  The connection is closed
    by the server once the pipeline finishes (status ``completed`` or
    ``failed``).

    **Client usage example (JavaScript):**

    ```js
    const ws = new WebSocket(`ws://localhost:8000/analysis/ws/${experimentId}`);
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        console.log(msg.stage, msg.progress_pct);
    };
    ```
    """
    await websocket.accept()
    logger.info("WebSocket client connected for experiment '%s'", experiment_id)

    try:
        # TODO: subscribe to Redis pub/sub channel for this experiment and
        #       forward messages to the WebSocket client.
        #
        # Pseudo-code:
        #   redis = websocket.app.state.redis
        #   pubsub = redis.pubsub()
        #   await pubsub.subscribe(f"experiment:{experiment_id}:progress")
        #   async for raw_msg in pubsub.listen():
        #       if raw_msg["type"] == "message":
        #           await websocket.send_text(raw_msg["data"])
        #           data = json.loads(raw_msg["data"])
        #           if data.get("status") in ("completed", "failed"):
        #               break

        # Stub: send an initial "connected" event, then keep alive until
        # the client disconnects.
        connected_msg = _ProgressMessage(
            experiment_id=experiment_id,
            stage="connected",
            status="waiting",
            progress_pct=0.0,
            message="Connected. Waiting for analysis to start.",
            timestamp=datetime.utcnow().isoformat(),
        )
        await websocket.send_text(connected_msg.model_dump_json())

        # Keep the connection alive until the client disconnects
        while True:
            try:
                # Receive and discard any client pings (heartbeat)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                logger.debug(
                    "WS heartbeat from experiment '%s': %s", experiment_id, data
                )
            except asyncio.TimeoutError:
                # Send a server-side keepalive ping
                ping_msg = _ProgressMessage(
                    experiment_id=experiment_id,
                    stage="heartbeat",
                    status="waiting",
                    progress_pct=0.0,
                    message="ping",
                    timestamp=datetime.utcnow().isoformat(),
                )
                await websocket.send_text(ping_msg.model_dump_json())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for experiment '%s'", experiment_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Unexpected error in WebSocket handler for experiment '%s': %s",
            experiment_id,
            exc,
        )
        try:
            await websocket.close(code=1011)  # Internal error
        except Exception:  # noqa: BLE001
            pass
