"""
Analysis Celery tasks – routed to the ``analysis`` queue.

Tasks
-----
- run_preprocessing          : Feature engineering and multi-hot matrix creation.
- run_statistical_diagnostic : Full statistical diagnostic (frequencies, entropy, etc.).
- run_changepoint_detection  : Regime / structural-break detection.
- run_ensemble               : Weighted ensemble combination of module outputs.
- run_full_pipeline          : Orchestrates the complete end-to-end pipeline,
                               chaining all analysis + training tasks and then
                               generating the default JSON report.

All tasks share a common pattern:
1. Load Experiment + Samples from the database (sync session).
2. Execute the relevant module.
3. Persist an AnalysisResult row with the JSON output and duration.
4. Return a lightweight summary dict so Celery can store it in Redis.

Time limits (enforced by celery_config):
  soft 15 min / hard 30 min per task
  ``run_full_pipeline`` uses the orchestrator which has its own sub-limits.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from celery import chain, chord, group, shared_task
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.database import SyncSessionLocal
from backend.core.orchestrator import AnalyticsOrchestrator, ExperimentConfig
from backend.db.models import (
    AnalysisResult,
    AnalysisStatus,
    Experiment,
    ExperimentStatus,
    Sample,
)
from backend.modules.changepoints.detector import RegimeDetector
from backend.modules.preprocessing.feature_engineer import FeatureEngineer
from backend.modules.statistics.diagnostic import StatisticalDiagnostic
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_experiment(session: Session, experiment_id: int) -> Experiment:
    """Load an Experiment row or raise ValueError if not found."""
    exp = session.get(Experiment, experiment_id)
    if exp is None:
        raise ValueError(f"Experiment {experiment_id} not found")
    return exp


def _load_samples(session: Session, experiment_id: int) -> List[List[int]]:
    """Return all valid Sample rows ordered by sequence_order."""
    stmt = (
        select(Sample)
        .join(Sample.dataset)
        .where(Sample.dataset.has(experiment_id=experiment_id))
        .where(Sample.is_valid.is_(True))
        .order_by(Sample.sequence_order)
    )
    rows = session.scalars(stmt).all()
    return [row.numbers_json for row in rows]


def _samples_to_array(samples: List[List[int]], k_count: int) -> np.ndarray:
    """Convert list-of-lists to a (N, k) numpy int array."""
    arr = np.zeros((len(samples), k_count), dtype=np.int32)
    for i, seq in enumerate(samples):
        arr[i, : len(seq)] = seq[:k_count]
    return arr


def _save_analysis_result(
    session: Session,
    experiment_id: int,
    module_name: str,
    result_json: Any,
    duration: float,
    status: AnalysisStatus = AnalysisStatus.SUCCESS,
) -> AnalysisResult:
    """Upsert an AnalysisResult for (experiment_id, module_name)."""
    existing = session.scalars(
        select(AnalysisResult).where(
            AnalysisResult.experiment_id == experiment_id,
            AnalysisResult.module_name == module_name,
        )
    ).first()

    if existing:
        existing.result_json = result_json
        existing.duration_seconds = duration
        existing.status = status
        record = existing
    else:
        record = AnalysisResult(
            experiment_id=experiment_id,
            module_name=module_name,
            result_json=result_json,
            duration_seconds=duration,
            status=status,
        )
        session.add(record)

    session.commit()
    session.refresh(record)
    return record


def _build_multihot(samples: np.ndarray, n_max: int) -> np.ndarray:
    """Build a (N, n_max) binary multi-hot matrix from (N, k) sequences."""
    n_samples = samples.shape[0]
    mh = np.zeros((n_samples, n_max), dtype=np.float32)
    for i, row in enumerate(samples):
        for num in row:
            idx = int(num) - 1
            if 0 <= idx < n_max:
                mh[i, idx] = 1.0
    return mh


# ---------------------------------------------------------------------------
# Task: preprocessing
# ---------------------------------------------------------------------------


@celery_app.task(
    name="tasks.run_preprocessing",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=900,   # 15 min
    time_limit=1800,       # 30 min
    acks_late=True,
)
def run_preprocessing(self, experiment_id: int) -> Dict[str, Any]:
    """
    Feature engineering task.

    Builds lag features, rolling statistics, multi-hot vectors, gap features,
    and positional features from the raw sample sequences.  The resulting
    feature summary is stored as an AnalysisResult.

    Parameters
    ----------
    experiment_id : int
        Primary key of the Experiment to process.

    Returns
    -------
    dict
        ``{experiment_id, module_name, n_samples, n_features, duration_seconds}``
    """
    t0 = time.perf_counter()
    module_name = "preprocessing"
    logger.info("run_preprocessing: experiment_id=%s", experiment_id)

    try:
        with SyncSessionLocal() as session:
            exp = _load_experiment(session, experiment_id)
            raw_samples = _load_samples(session, experiment_id)

            if not raw_samples:
                raise ValueError("No valid samples found for experiment")

            sequences = _samples_to_array(raw_samples, exp.k_count)

            engineer = FeatureEngineer(n_max=exp.n_max, k_count=exp.k_count)
            features_df = engineer.fit_transform(sequences)

            result = {
                "n_samples": len(raw_samples),
                "n_features": features_df.shape[1],
                "feature_names": list(features_df.columns),
                "sample_stats": {
                    "mean": float(features_df.mean().mean()),
                    "std": float(features_df.std().mean()),
                },
            }

            duration = time.perf_counter() - t0
            _save_analysis_result(session, experiment_id, module_name, result, duration)

        logger.info(
            "run_preprocessing done: experiment_id=%s n_features=%d in %.2fs",
            experiment_id, result["n_features"], duration,
        )
        return {
            "experiment_id": experiment_id,
            "module_name": module_name,
            "n_samples": result["n_samples"],
            "n_features": result["n_features"],
            "duration_seconds": duration,
        }

    except SoftTimeLimitExceeded:
        logger.warning("run_preprocessing soft time limit exceeded: experiment_id=%s", experiment_id)
        with SyncSessionLocal() as session:
            _save_analysis_result(session, experiment_id, module_name, {"error": "soft_time_limit_exceeded"}, time.perf_counter() - t0, AnalysisStatus.FAILED)
        raise

    except Exception as exc:
        logger.exception("run_preprocessing failed: experiment_id=%s", experiment_id)
        try:
            with SyncSessionLocal() as session:
                _save_analysis_result(session, experiment_id, module_name, {"error": str(exc)}, time.perf_counter() - t0, AnalysisStatus.FAILED)
        except Exception:
            pass
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: statistical diagnostic
# ---------------------------------------------------------------------------


@celery_app.task(
    name="tasks.run_statistical_diagnostic",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=900,
    time_limit=1800,
    acks_late=True,
)
def run_statistical_diagnostic(self, experiment_id: int) -> Dict[str, Any]:
    """
    Run the full statistical diagnostic suite on the experiment's sequences.

    Covers frequency analysis, chi-squared uniformity, autocorrelation,
    stationarity tests (ADF/KPSS), entropy measures, and more.

    Parameters
    ----------
    experiment_id : int
        Primary key of the Experiment to analyse.

    Returns
    -------
    dict
        Summary with ``{experiment_id, module_name, n_samples, duration_seconds}``.
    """
    t0 = time.perf_counter()
    module_name = "statistical_diagnostic"
    logger.info("run_statistical_diagnostic: experiment_id=%s", experiment_id)

    try:
        with SyncSessionLocal() as session:
            exp = _load_experiment(session, experiment_id)
            raw_samples = _load_samples(session, experiment_id)

            if not raw_samples:
                raise ValueError("No valid samples found for experiment")

            sequences = _samples_to_array(raw_samples, exp.k_count)
            multihot = _build_multihot(sequences, exp.n_max)

            diagnostic = StatisticalDiagnostic(n_max=exp.n_max, k_count=exp.k_count)
            result = diagnostic.run_full_diagnostic(sequences, multihot)

            # Convert numpy scalars for JSON serialisation
            result = _sanitise_result(result)

            duration = time.perf_counter() - t0
            _save_analysis_result(session, experiment_id, module_name, result, duration)

        logger.info(
            "run_statistical_diagnostic done: experiment_id=%s in %.2fs",
            experiment_id, duration,
        )
        return {
            "experiment_id": experiment_id,
            "module_name": module_name,
            "n_samples": len(raw_samples),
            "duration_seconds": duration,
        }

    except SoftTimeLimitExceeded:
        logger.warning("run_statistical_diagnostic soft time limit exceeded: experiment_id=%s", experiment_id)
        with SyncSessionLocal() as session:
            _save_analysis_result(session, experiment_id, module_name, {"error": "soft_time_limit_exceeded"}, time.perf_counter() - t0, AnalysisStatus.FAILED)
        raise

    except Exception as exc:
        logger.exception("run_statistical_diagnostic failed: experiment_id=%s", experiment_id)
        try:
            with SyncSessionLocal() as session:
                _save_analysis_result(session, experiment_id, module_name, {"error": str(exc)}, time.perf_counter() - t0, AnalysisStatus.FAILED)
        except Exception:
            pass
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: change-point / regime detection
# ---------------------------------------------------------------------------


@celery_app.task(
    name="tasks.run_changepoint_detection",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=900,
    time_limit=1800,
    acks_late=True,
)
def run_changepoint_detection(
    self,
    experiment_id: int,
    method: str = "auto",
    min_size: int = 10,
) -> Dict[str, Any]:
    """
    Detect structural breaks / regime changes in the draw sequence.

    Uses the ``RegimeDetector`` (ruptures library when available, with a
    variance-based fallback).

    Parameters
    ----------
    experiment_id : int
        Primary key of the Experiment.
    method : str
        Detection algorithm: ``'auto'``, ``'pelt'``, ``'binseg'``, ``'dynp'``.
    min_size : int
        Minimum segment size (samples) per detected regime.

    Returns
    -------
    dict
        ``{experiment_id, module_name, n_regimes, n_breakpoints, duration_seconds}``
    """
    t0 = time.perf_counter()
    module_name = "changepoint_detection"
    logger.info("run_changepoint_detection: experiment_id=%s method=%s", experiment_id, method)

    try:
        with SyncSessionLocal() as session:
            exp = _load_experiment(session, experiment_id)
            raw_samples = _load_samples(session, experiment_id)

            if not raw_samples:
                raise ValueError("No valid samples found for experiment")

            sequences = _samples_to_array(raw_samples, exp.k_count)
            multihot = _build_multihot(sequences, exp.n_max)

            detector = RegimeDetector(n_max=exp.n_max)
            result = detector.detect(multihot, method=method, min_size=min_size)
            result = _sanitise_result(result)

            duration = time.perf_counter() - t0
            _save_analysis_result(session, experiment_id, module_name, result, duration)

        n_regimes = result.get("n_regimes", 1)
        n_breakpoints = len(result.get("breakpoints", []))

        logger.info(
            "run_changepoint_detection done: experiment_id=%s n_regimes=%d in %.2fs",
            experiment_id, n_regimes, duration,
        )
        return {
            "experiment_id": experiment_id,
            "module_name": module_name,
            "n_regimes": n_regimes,
            "n_breakpoints": n_breakpoints,
            "duration_seconds": duration,
        }

    except SoftTimeLimitExceeded:
        logger.warning("run_changepoint_detection soft time limit exceeded: experiment_id=%s", experiment_id)
        with SyncSessionLocal() as session:
            _save_analysis_result(session, experiment_id, module_name, {"error": "soft_time_limit_exceeded"}, time.perf_counter() - t0, AnalysisStatus.FAILED)
        raise

    except Exception as exc:
        logger.exception("run_changepoint_detection failed: experiment_id=%s", experiment_id)
        try:
            with SyncSessionLocal() as session:
                _save_analysis_result(session, experiment_id, module_name, {"error": str(exc)}, time.perf_counter() - t0, AnalysisStatus.FAILED)
        except Exception:
            pass
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: ensemble combination
# ---------------------------------------------------------------------------


@celery_app.task(
    name="tasks.run_ensemble",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=900,
    time_limit=1800,
    acks_late=True,
)
def run_ensemble(
    self,
    experiment_id: int,
    module_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Combine upstream module outputs into a weighted ensemble prediction.

    Reads previously persisted AnalysisResult and ModelResult rows for the
    experiment and uses the AnalyticsOrchestrator's ensemble logic to blend
    them into a single ranked set of predicted numbers.

    Parameters
    ----------
    experiment_id : int
        Primary key of the Experiment.
    module_weights : dict, optional
        Override weights for specific modules.  Example:
        ``{"statistical": 1.5, "hmm": 0.8}``.

    Returns
    -------
    dict
        ``{experiment_id, module_name, predicted_numbers, confidence, duration_seconds}``
    """
    t0 = time.perf_counter()
    module_name = "ensemble"
    logger.info("run_ensemble: experiment_id=%s", experiment_id)

    try:
        with SyncSessionLocal() as session:
            exp = _load_experiment(session, experiment_id)
            raw_samples = _load_samples(session, experiment_id)

            if not raw_samples:
                raise ValueError("No valid samples found for experiment")

            cfg = ExperimentConfig(
                n_max=exp.n_max,
                k_count=exp.k_count,
                order_matters=exp.order_matters,
                sample_count=len(raw_samples),
                has_timestamps=True,
            )

            orchestrator = AnalyticsOrchestrator(config=cfg, raw_samples=raw_samples)
            pipeline_result = orchestrator.run()

            ensemble_result = pipeline_result.ensemble_result or {}
            if module_weights:
                # Re-run ensemble with custom weights from caller
                from backend.core.orchestrator import OrchestratorContext
                ctx = OrchestratorContext(
                    config=cfg,
                    raw_samples=raw_samples,
                    module_outputs=pipeline_result.module_results,
                    orchestrator_flags={
                        **pipeline_result.orchestrator_flags,
                        "module_weights": module_weights,
                    },
                )
                from backend.core.orchestrator import _run_ensemble as _orch_ensemble
                ensemble_result = _orch_ensemble(ctx)

            ensemble_result = _sanitise_result(ensemble_result)

            duration = time.perf_counter() - t0
            _save_analysis_result(session, experiment_id, module_name, ensemble_result, duration)

        predicted = ensemble_result.get("predicted_numbers", [])
        confidence = ensemble_result.get("confidence", 0.0)

        logger.info(
            "run_ensemble done: experiment_id=%s predicted=%s confidence=%.4f in %.2fs",
            experiment_id, predicted, confidence, duration,
        )
        return {
            "experiment_id": experiment_id,
            "module_name": module_name,
            "predicted_numbers": predicted,
            "confidence": confidence,
            "duration_seconds": duration,
        }

    except SoftTimeLimitExceeded:
        logger.warning("run_ensemble soft time limit exceeded: experiment_id=%s", experiment_id)
        with SyncSessionLocal() as session:
            _save_analysis_result(session, experiment_id, module_name, {"error": "soft_time_limit_exceeded"}, time.perf_counter() - t0, AnalysisStatus.FAILED)
        raise

    except Exception as exc:
        logger.exception("run_ensemble failed: experiment_id=%s", experiment_id)
        try:
            with SyncSessionLocal() as session:
                _save_analysis_result(session, experiment_id, module_name, {"error": str(exc)}, time.perf_counter() - t0, AnalysisStatus.FAILED)
        except Exception:
            pass
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: full pipeline
# ---------------------------------------------------------------------------


@celery_app.task(
    name="tasks.run_full_pipeline",
    bind=True,
    max_retries=1,
    default_retry_delay=60,
    soft_time_limit=1800,   # 30 min (orchestrator has its own sub-limits)
    time_limit=3600,        # 60 min
    acks_late=True,
)
def run_full_pipeline(
    self,
    experiment_id: int,
    include_report: bool = True,
    report_format: str = "json",
) -> Dict[str, Any]:
    """
    End-to-end orchestration task.

    Executes the full ``AnalyticsOrchestrator`` pipeline (all enabled modules
    in dependency order) and optionally generates a report.

    This task updates the Experiment ``status`` to RUNNING at the start and
    to COMPLETED (or FAILED) when it finishes.

    Parameters
    ----------
    experiment_id : int
        Primary key of the Experiment to run.
    include_report : bool
        If True, triggers ``generate_report`` after the pipeline completes.
    report_format : str
        Report output format (``'json'``, ``'html'``, ``'pdf'``, ``'excel'``).

    Returns
    -------
    dict
        Pipeline summary including all module results and timing.
    """
    t0 = time.perf_counter()
    logger.info(
        "run_full_pipeline START: experiment_id=%s include_report=%s",
        experiment_id, include_report,
    )

    try:
        # ---- 1. Mark experiment as RUNNING ---------------------------------
        with SyncSessionLocal() as session:
            exp = _load_experiment(session, experiment_id)
            exp.status = ExperimentStatus.RUNNING
            session.commit()

            raw_samples = _load_samples(session, experiment_id)

            if not raw_samples:
                raise ValueError("No valid samples found for experiment")

            cfg = ExperimentConfig(
                n_max=exp.n_max,
                k_count=exp.k_count,
                order_matters=exp.order_matters,
                sample_count=len(raw_samples),
                has_timestamps=True,
            )

        # ---- 2. Run orchestrator -------------------------------------------
        orchestrator = AnalyticsOrchestrator(config=cfg, raw_samples=raw_samples)
        pipeline_result = orchestrator.run()

        # ---- 3. Persist all module results ---------------------------------
        with SyncSessionLocal() as session:
            for mod_name, mod_output in pipeline_result.module_results.items():
                status = (
                    AnalysisStatus.FAILED
                    if isinstance(mod_output, dict) and "error" in mod_output
                    else AnalysisStatus.SUCCESS
                )
                _save_analysis_result(
                    session,
                    experiment_id,
                    mod_name,
                    _sanitise_result(mod_output),
                    pipeline_result.per_module_duration.get(mod_name, 0.0),
                    status,
                )

            # Mark experiment COMPLETED
            exp = _load_experiment(session, experiment_id)
            exp.status = ExperimentStatus.COMPLETED
            session.commit()

        duration = time.perf_counter() - t0

        summary: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": "completed",
            "n_samples": len(raw_samples),
            "modules_run": list(pipeline_result.module_results.keys()),
            "modules_skipped": pipeline_result.skipped_modules,
            "total_duration_seconds": duration,
            "per_module_duration": pipeline_result.per_module_duration,
            "ensemble": pipeline_result.ensemble_result,
            "orchestrator_flags": _sanitise_result(pipeline_result.orchestrator_flags),
        }

        logger.info(
            "run_full_pipeline COMPLETE: experiment_id=%s in %.2fs modules=%s",
            experiment_id,
            duration,
            summary["modules_run"],
        )

        # ---- 4. Optionally kick off report generation ----------------------
        if include_report:
            from backend.tasks.report_tasks import generate_report
            generate_report.apply_async(
                kwargs={"experiment_id": experiment_id, "report_format": report_format},
                countdown=2,  # slight delay to allow DB writes to propagate
            )

        return summary

    except SoftTimeLimitExceeded:
        logger.warning("run_full_pipeline soft time limit exceeded: experiment_id=%s", experiment_id)
        with SyncSessionLocal() as session:
            exp = _load_experiment(session, experiment_id)
            exp.status = ExperimentStatus.FAILED
            session.commit()
        raise

    except Exception as exc:
        logger.exception("run_full_pipeline FAILED: experiment_id=%s", experiment_id)
        try:
            with SyncSessionLocal() as session:
                exp = _load_experiment(session, experiment_id)
                exp.status = ExperimentStatus.FAILED
                session.commit()
        except Exception:
            pass
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _sanitise_result(obj: Any) -> Any:
    """
    Recursively convert numpy scalars and arrays to Python-native types so
    the result can be safely serialised to JSON by Celery / PostgreSQL JSONB.
    """
    if isinstance(obj, dict):
        return {k: _sanitise_result(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise_result(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
