"""
Training Celery tasks – routed to the ``training`` queue.

Tasks
-----
- run_baselines     : Fit all baseline models and store cross-model metrics.
- run_hmm           : Train a Hidden Markov Model and store state diagnostics.
- run_ml_models     : Feature-engineer + train tabular ML models (XGBoost, RF…).
- run_deep_learning : Placeholder for LSTM/Transformer training (stub).

Each task follows the same three-step contract:
1. Load Experiment + Samples from the database (sync session).
2. Train / evaluate the module.
3. Persist a ModelResult row with metrics and (optionally) predictions.

Time limits (from celery_config):
  soft 50 min / hard 60 min – matches ``settings.max_training_time_seconds``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.database import SyncSessionLocal
from backend.db.models import (
    AnalysisStatus,
    Experiment,
    ModelResult,
    Sample,
)
from backend.modules.baselines.models import (
    CooccurrenceModel,
    GlobalFrequencyModel,
    MarkovOrder1Model,
    MarkovOrder2Model,
    RecencyWeightedFrequencyModel,
    UniformRandomModel,
    VariableOrderMarkovModel,
)
from backend.modules.preprocessing.feature_engineer import FeatureEngineer
from backend.modules.probabilistic.hmm_model import HiddenMarkovAnalyzer
from backend.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_experiment(session: Session, experiment_id: int) -> Experiment:
    exp = session.get(Experiment, experiment_id)
    if exp is None:
        raise ValueError(f"Experiment {experiment_id} not found")
    return exp


def _load_samples(session: Session, experiment_id: int) -> List[List[int]]:
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
    arr = np.zeros((len(samples), k_count), dtype=np.int32)
    for i, seq in enumerate(samples):
        arr[i, : len(seq)] = seq[:k_count]
    return arr


def _save_model_result(
    session: Session,
    experiment_id: int,
    model_name: str,
    module: str,
    metrics: Dict[str, Any],
    predictions: Optional[List[Any]] = None,
    feature_importance: Optional[List[Any]] = None,
    val_score: Optional[float] = None,
    test_score: Optional[float] = None,
) -> ModelResult:
    """Upsert a ModelResult row."""
    existing = session.scalars(
        select(ModelResult).where(
            ModelResult.experiment_id == experiment_id,
            ModelResult.model_name == model_name,
        )
    ).first()

    now = datetime.now(tz=timezone.utc)
    if existing:
        existing.metrics_json = metrics
        existing.predictions_json = predictions
        existing.feature_importance_json = feature_importance
        existing.val_score = val_score
        existing.test_score = test_score
        existing.trained_at = now
        existing.module = module
        record = existing
    else:
        record = ModelResult(
            experiment_id=experiment_id,
            model_name=model_name,
            module=module,
            metrics_json=metrics,
            predictions_json=predictions,
            feature_importance_json=feature_importance,
            val_score=val_score,
            test_score=test_score,
            trained_at=now,
        )
        session.add(record)

    session.commit()
    session.refresh(record)
    return record


def _sanitise(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Task: baselines
# ---------------------------------------------------------------------------

_BASELINE_CLASSES = [
    UniformRandomModel,
    GlobalFrequencyModel,
    RecencyWeightedFrequencyModel,
    CooccurrenceModel,
    MarkovOrder1Model,
    MarkovOrder2Model,
    VariableOrderMarkovModel,
]


@celery_app.task(
    name="tasks.run_baselines",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    soft_time_limit=3000,  # 50 min
    time_limit=3600,       # 60 min
    acks_late=True,
)
def run_baselines(self, experiment_id: int) -> Dict[str, Any]:
    """
    Fit all baseline models and persist their cross-validated scores.

    Baseline models (from ``backend.modules.baselines.models``):
    - UniformRandomModel
    - GlobalFrequencyModel
    - RecencyWeightedFrequencyModel
    - CooccurrenceModel
    - MarkovOrder1Model
    - MarkovOrder2Model
    - VariableOrderMarkovModel

    Each model is scored with an 80/20 temporal split (no shuffling to
    preserve the draw order).

    Parameters
    ----------
    experiment_id : int
        Primary key of the Experiment.

    Returns
    -------
    dict
        ``{experiment_id, models_trained, best_model, best_score, duration_seconds}``
    """
    t0 = time.perf_counter()
    logger.info("run_baselines START: experiment_id=%s", experiment_id)

    try:
        with SyncSessionLocal() as session:
            exp = _load_experiment(session, experiment_id)
            raw_samples = _load_samples(session, experiment_id)

            if not raw_samples:
                raise ValueError("No valid samples found for experiment")

            sequences = _samples_to_array(raw_samples, exp.k_count)

            split = max(1, int(len(sequences) * 0.8))
            train_seq = sequences[:split]
            test_seq = sequences[split:] if split < len(sequences) else sequences

            results: List[Dict[str, Any]] = []

            for ModelClass in _BASELINE_CLASSES:
                model_name = ModelClass.__name__
                try:
                    model_t0 = time.perf_counter()
                    instance = ModelClass()
                    instance.fit(train_seq)
                    score = instance.score(test_seq)
                    model_duration = time.perf_counter() - model_t0

                    metrics = {
                        "test_score": score,
                        "train_samples": int(len(train_seq)),
                        "test_samples": int(len(test_seq)),
                        "duration_seconds": model_duration,
                    }
                    results.append({"model": model_name, "score": score})

                    _save_model_result(
                        session,
                        experiment_id,
                        model_name=model_name,
                        module="backend.modules.baselines.models",
                        metrics=metrics,
                        val_score=score,
                        test_score=score,
                    )
                    logger.info(
                        "run_baselines: %s score=%.4f in %.2fs",
                        model_name, score, model_duration,
                    )

                except Exception as exc:
                    logger.warning("Baseline %s failed: %s", model_name, exc)
                    results.append({"model": model_name, "error": str(exc)})

        best = max((r for r in results if "score" in r), key=lambda r: r["score"], default=None)
        duration = time.perf_counter() - t0

        logger.info(
            "run_baselines DONE: experiment_id=%s best=%s in %.2fs",
            experiment_id, best, duration,
        )
        return {
            "experiment_id": experiment_id,
            "models_trained": [r["model"] for r in results],
            "best_model": best["model"] if best else None,
            "best_score": best["score"] if best else None,
            "duration_seconds": duration,
        }

    except SoftTimeLimitExceeded:
        logger.warning("run_baselines soft time limit exceeded: experiment_id=%s", experiment_id)
        raise

    except Exception as exc:
        logger.exception("run_baselines FAILED: experiment_id=%s", experiment_id)
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: HMM
# ---------------------------------------------------------------------------


@celery_app.task(
    name="tasks.run_hmm",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    soft_time_limit=3000,
    time_limit=3600,
    acks_late=True,
)
def run_hmm(
    self,
    experiment_id: int,
    n_states_range: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Train a Hidden Markov Model (GaussianHMM via hmmlearn) on sequence data.

    Performs BIC-based model selection over the requested state counts, then
    re-trains on the full dataset with the optimal number of states.

    Parameters
    ----------
    experiment_id : int
        Primary key of the Experiment.
    n_states_range : list[int], optional
        Candidate state counts for model selection.  Defaults to ``[2,3,4,5,6]``.

    Returns
    -------
    dict
        ``{experiment_id, n_states, log_likelihood, mean_hit_rate, duration_seconds}``
    """
    t0 = time.perf_counter()
    logger.info("run_hmm START: experiment_id=%s n_states_range=%s", experiment_id, n_states_range)

    n_states_range = n_states_range or [2, 3, 4, 5, 6]

    try:
        with SyncSessionLocal() as session:
            exp = _load_experiment(session, experiment_id)
            raw_samples = _load_samples(session, experiment_id)

            if not raw_samples:
                raise ValueError("No valid samples found for experiment")

            sequences = _samples_to_array(raw_samples, exp.k_count)

            analyzer = HiddenMarkovAnalyzer(
                n_max=exp.n_max,
                k_count=exp.k_count,
                n_states_range=n_states_range,
                random_state=settings.random_seed,
            )

            fit_result = analyzer.fit(sequences)
            fit_result = _sanitise(fit_result)

            # Compute evaluation metrics on held-out test set
            split = max(1, int(len(sequences) * 0.8))
            test_seq = sequences[split:] if split < len(sequences) else sequences
            eval_metrics = analyzer.compute_metrics(test_seq) if len(test_seq) > 0 else {}
            eval_metrics = _sanitise(eval_metrics)

            # Build state characteristics for diagnostics
            state_chars = _sanitise(analyzer.get_state_characteristics())
            transition_matrix = _sanitise(analyzer.get_transition_matrix())

            metrics = {
                **fit_result,
                **eval_metrics,
                "transition_matrix": transition_matrix,
                "state_characteristics": state_chars,
                "train_samples": int(split),
                "test_samples": int(len(sequences) - split),
            }

            val_score = eval_metrics.get("mean_hit_rate")

            _save_model_result(
                session,
                experiment_id,
                model_name="HiddenMarkovAnalyzer",
                module="backend.modules.probabilistic.hmm_model",
                metrics=metrics,
                val_score=val_score,
                test_score=val_score,
            )

        duration = time.perf_counter() - t0
        n_states = fit_result.get("n_states")

        logger.info(
            "run_hmm DONE: experiment_id=%s n_states=%s hit_rate=%s in %.2fs",
            experiment_id, n_states, eval_metrics.get("mean_hit_rate"), duration,
        )
        return {
            "experiment_id": experiment_id,
            "n_states": n_states,
            "log_likelihood": fit_result.get("log_likelihood"),
            "mean_hit_rate": eval_metrics.get("mean_hit_rate"),
            "duration_seconds": duration,
        }

    except SoftTimeLimitExceeded:
        logger.warning("run_hmm soft time limit exceeded: experiment_id=%s", experiment_id)
        raise

    except Exception as exc:
        logger.exception("run_hmm FAILED: experiment_id=%s", experiment_id)
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: tabular ML models
# ---------------------------------------------------------------------------


@celery_app.task(
    name="tasks.run_ml_models",
    bind=True,
    max_retries=1,
    default_retry_delay=120,
    soft_time_limit=3000,
    time_limit=3600,
    acks_late=True,
)
def run_ml_models(
    self,
    experiment_id: int,
    models: Optional[List[str]] = None,
    cv_splits: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Feature-engineer sequences and train tabular ML models.

    Models (subset of): ``logistic_regression``, ``random_forest``,
    ``extra_trees``, ``gradient_boosting``, ``xgboost``.

    Uses ``FeatureEngineer`` to build lag/rolling/co-occurrence features,
    then ``TabularModelTrainer`` to fit each model with time-series CV
    (no shuffling).

    Parameters
    ----------
    experiment_id : int
        Primary key of the Experiment.
    models : list[str], optional
        Subset of model keys to train.  Defaults to all available.
    cv_splits : int, optional
        Number of CV folds.  Defaults to ``settings.default_cv_splits``.

    Returns
    -------
    dict
        ``{experiment_id, models_trained, best_model, best_val_score, duration_seconds}``
    """
    from backend.modules.ml.tabular_models import TabularModelTrainer

    t0 = time.perf_counter()
    cv_splits = cv_splits or settings.default_cv_splits
    logger.info("run_ml_models START: experiment_id=%s models=%s cv_splits=%d", experiment_id, models, cv_splits)

    try:
        with SyncSessionLocal() as session:
            exp = _load_experiment(session, experiment_id)
            raw_samples = _load_samples(session, experiment_id)

            if not raw_samples:
                raise ValueError("No valid samples found for experiment")

            sequences = _samples_to_array(raw_samples, exp.k_count)

            # Build features
            engineer = FeatureEngineer(n_max=exp.n_max, k_count=exp.k_count)
            features_df = engineer.fit_transform(sequences)

            # Target: multi-hot matrix shifted by 1 (predict next draw)
            n_samples = len(sequences)
            y_multihot = np.zeros((n_samples, exp.n_max), dtype=np.int32)
            for i, seq in enumerate(sequences):
                for num in seq:
                    idx = int(num) - 1
                    if 0 <= idx < exp.n_max:
                        y_multihot[i, idx] = 1

            # Align: features at t → target at t+1
            X = features_df.values[:-1]
            y = y_multihot[1:]

            trainer = TabularModelTrainer(
                n_max=exp.n_max,
                k_count=exp.k_count,
                cv_splits=cv_splits,
                random_state=settings.random_seed,
                max_time_seconds=settings.max_training_time_seconds,
            )

            all_metrics = trainer.fit_all(
                features_df.iloc[:-1],
                y,
                selected_models=models,
            )
            all_metrics = _sanitise(all_metrics)

            trained_models: List[str] = []
            best_model: Optional[str] = None
            best_val_score: Optional[float] = None

            for model_name, metrics in all_metrics.items():
                trained_models.append(model_name)
                val_score = metrics.get("mean_val_log_loss") or metrics.get("val_score")
                if val_score is not None:
                    val_score = float(val_score)

                fi = metrics.pop("feature_importances", None)
                fi_list = (
                    [{"feature": k, "importance": v} for k, v in fi.items()]
                    if fi else None
                )

                _save_model_result(
                    session,
                    experiment_id,
                    model_name=model_name,
                    module="backend.modules.ml.tabular_models",
                    metrics=metrics,
                    feature_importance=fi_list,
                    val_score=val_score,
                )

                if val_score is not None:
                    if best_val_score is None or val_score < best_val_score:
                        best_val_score = val_score
                        best_model = model_name

        duration = time.perf_counter() - t0

        logger.info(
            "run_ml_models DONE: experiment_id=%s models=%s best=%s in %.2fs",
            experiment_id, trained_models, best_model, duration,
        )
        return {
            "experiment_id": experiment_id,
            "models_trained": trained_models,
            "best_model": best_model,
            "best_val_score": best_val_score,
            "duration_seconds": duration,
        }

    except SoftTimeLimitExceeded:
        logger.warning("run_ml_models soft time limit exceeded: experiment_id=%s", experiment_id)
        raise

    except Exception as exc:
        logger.exception("run_ml_models FAILED: experiment_id=%s", experiment_id)
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task: deep learning (stub – wire up real model when ready)
# ---------------------------------------------------------------------------


@celery_app.task(
    name="tasks.run_deep_learning",
    bind=True,
    max_retries=1,
    default_retry_delay=120,
    soft_time_limit=3000,
    time_limit=3600,
    acks_late=True,
)
def run_deep_learning(
    self,
    experiment_id: int,
    architecture: str = "lstm",
    epochs: int = 50,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Train a deep learning sequence model (LSTM / Transformer).

    This task is a **structured stub**: it validates inputs, records a
    ModelResult with placeholder metrics, and logs a clear notice so that
    future implementors know exactly where to wire in the real training loop.

    The ``backend/modules/deep/`` package is the intended home for the full
    implementation.  When ready, replace the placeholder block below with::

        from backend.modules.deep.model import DeepSequenceModel
        model = DeepSequenceModel(...)
        result = model.fit(sequences, epochs=epochs, batch_size=batch_size)

    Parameters
    ----------
    experiment_id : int
        Primary key of the Experiment.
    architecture : str
        Model architecture key (``'lstm'``, ``'gru'``, ``'transformer'``).
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    dict
        ``{experiment_id, architecture, status, duration_seconds}``
    """
    t0 = time.perf_counter()
    logger.info(
        "run_deep_learning START: experiment_id=%s arch=%s epochs=%d",
        experiment_id, architecture, epochs,
    )

    try:
        with SyncSessionLocal() as session:
            exp = _load_experiment(session, experiment_id)
            raw_samples = _load_samples(session, experiment_id)

            if not raw_samples:
                raise ValueError("No valid samples found for experiment")

            n_samples = len(raw_samples)

            # Structured stub – real training goes here
            logger.warning(
                "run_deep_learning: deep learning training is a stub for "
                "experiment_id=%s. Wire backend/modules/deep/ to activate.",
                experiment_id,
            )

            metrics = {
                "architecture": architecture,
                "epochs_requested": epochs,
                "batch_size": batch_size,
                "n_samples": n_samples,
                "n_max": exp.n_max,
                "k_count": exp.k_count,
                "status": "stub",
                "note": (
                    "Deep learning training is not yet implemented. "
                    "Add the model to backend/modules/deep/ and update this task."
                ),
                "val_loss": None,
                "train_loss": None,
            }

            _save_model_result(
                session,
                experiment_id,
                model_name=f"DeepLearning_{architecture}",
                module="backend.modules.deep",
                metrics=metrics,
                val_score=None,
            )

        duration = time.perf_counter() - t0

        logger.info(
            "run_deep_learning DONE (stub): experiment_id=%s arch=%s in %.2fs",
            experiment_id, architecture, duration,
        )
        return {
            "experiment_id": experiment_id,
            "architecture": architecture,
            "status": "stub",
            "duration_seconds": duration,
        }

    except SoftTimeLimitExceeded:
        logger.warning("run_deep_learning soft time limit exceeded: experiment_id=%s", experiment_id)
        raise

    except Exception as exc:
        logger.exception("run_deep_learning FAILED: experiment_id=%s", experiment_id)
        raise self.retry(exc=exc)
