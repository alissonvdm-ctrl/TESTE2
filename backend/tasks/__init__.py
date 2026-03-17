"""
Celery tasks package for the NumericSequenceAnalyzer.

Importing this package causes all task modules to be registered with the
Celery application so that ``celery_app.autodiscover_tasks`` finds them.

Queue routing summary
---------------------
analysis queue : run_full_pipeline, run_preprocessing,
                 run_statistical_diagnostic, run_changepoint_detection,
                 run_ensemble
training queue : run_baselines, run_hmm, run_ml_models, run_deep_learning
reports  queue : generate_report
"""

from backend.tasks.analysis_tasks import (
    run_changepoint_detection,
    run_ensemble,
    run_full_pipeline,
    run_preprocessing,
    run_statistical_diagnostic,
)
from backend.tasks.report_tasks import generate_report
from backend.tasks.training_tasks import (
    run_baselines,
    run_deep_learning,
    run_hmm,
    run_ml_models,
)

__all__ = [
    # analysis queue
    "run_full_pipeline",
    "run_preprocessing",
    "run_statistical_diagnostic",
    "run_changepoint_detection",
    "run_ensemble",
    # training queue
    "run_baselines",
    "run_hmm",
    "run_ml_models",
    "run_deep_learning",
    # reports queue
    "generate_report",
]
