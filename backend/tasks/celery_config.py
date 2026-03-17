"""
Celery configuration for the NumericSequenceAnalyzer task queue.

All values are read from environment variables so that the same codebase
can be deployed in development, staging, and production without code
changes.  Default values are suitable for local Docker Compose development.

Queue layout
------------
- default   – general-purpose catch-all queue
- analysis  – orchestration, preprocessing, statistical, change-point tasks
- training  – long-running model training (baselines, HMM, ML, deep learning)
- reports   – report generation (PDF / HTML / JSON / Excel)

Time limits
-----------
- default queue        : 5 min soft / 10 min hard
- analysis queue       : 15 min soft / 30 min hard
- training queue       : 50 min soft / 60 min hard  (matches config.max_training_time_seconds)
- reports queue        : 10 min soft / 20 min hard
"""

from __future__ import annotations

import os

from kombu import Exchange, Queue

# ---------------------------------------------------------------------------
# Broker & result backend
# ---------------------------------------------------------------------------

#: Redis DB 1 is reserved for the Celery broker.
broker_url: str = os.environ.get(
    "CELERY_BROKER_URL",
    "redis://redis:6379/1",
)

#: Redis DB 2 is reserved for Celery task results.
result_backend: str = os.environ.get(
    "CELERY_RESULT_BACKEND",
    "redis://redis:6379/2",
)

# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

task_serializer: str = "json"
result_serializer: str = "json"
accept_content: list[str] = ["json"]
result_accept_content: list[str] = ["json"]

# ---------------------------------------------------------------------------
# Result settings
# ---------------------------------------------------------------------------

#: Keep results in Redis for 24 hours so the API can poll for task state.
result_expires: int = 86_400  # seconds

#: Store extended task metadata (args, kwargs, timestamps) in the result.
result_extended: bool = True

# ---------------------------------------------------------------------------
# Task behaviour
# ---------------------------------------------------------------------------

#: Acknowledge tasks only after they complete (avoids silent loss on crash).
task_acks_late: bool = True

#: Reject (and re-queue) tasks when the worker process is killed unexpectedly.
task_reject_on_worker_lost: bool = True

#: Maximum number of tasks a worker child process handles before recycling.
#: Prevents memory leaks in long-lived scientific computing workloads.
worker_max_tasks_per_child: int = 50

# ---------------------------------------------------------------------------
# Worker concurrency
# ---------------------------------------------------------------------------

#: Number of concurrent worker processes per Celery worker node.
#: Override via CELERY_WORKER_CONCURRENCY env var (e.g. match CPU count).
worker_concurrency: int = int(os.environ.get("CELERY_WORKER_CONCURRENCY", "4"))

#: Use ``prefork`` (multiprocessing) by default; switch to ``gevent`` or
#: ``eventlet`` for I/O-bound workloads.
worker_pool: str = os.environ.get("CELERY_WORKER_POOL", "prefork")

#: How many tasks each worker prefetches per process.  Keep low for long tasks
#: to allow fair distribution across workers.
worker_prefetch_multiplier: int = 1

# ---------------------------------------------------------------------------
# Exchanges & queues
# ---------------------------------------------------------------------------

_default_exchange = Exchange("default", type="direct")
_analysis_exchange = Exchange("analysis", type="direct")
_training_exchange = Exchange("training", type="direct")
_reports_exchange = Exchange("reports", type="direct")

task_queues: tuple[Queue, ...] = (
    Queue(
        "default",
        exchange=_default_exchange,
        routing_key="default",
    ),
    Queue(
        "analysis",
        exchange=_analysis_exchange,
        routing_key="analysis",
    ),
    Queue(
        "training",
        exchange=_training_exchange,
        routing_key="training",
    ),
    Queue(
        "reports",
        exchange=_reports_exchange,
        routing_key="reports",
    ),
)

task_default_queue: str = "default"
task_default_exchange: str = "default"
task_default_routing_key: str = "default"

# ---------------------------------------------------------------------------
# Task routing
# ---------------------------------------------------------------------------

task_routes: dict[str, dict[str, str]] = {
    # ---- Orchestration / analysis tasks ------------------------------------
    "tasks.run_full_pipeline": {"queue": "analysis"},
    "tasks.run_preprocessing": {"queue": "analysis"},
    "tasks.run_statistical_diagnostic": {"queue": "analysis"},
    "tasks.run_changepoint_detection": {"queue": "analysis"},
    "tasks.run_ensemble": {"queue": "analysis"},
    # ---- Training tasks ---------------------------------------------------
    "tasks.run_baselines": {"queue": "training"},
    "tasks.run_hmm": {"queue": "training"},
    "tasks.run_ml_models": {"queue": "training"},
    "tasks.run_deep_learning": {"queue": "training"},
    # ---- Report tasks ----------------------------------------------------
    "tasks.generate_report": {"queue": "reports"},
}

# ---------------------------------------------------------------------------
# Time limits  (soft → warning + graceful shutdown; hard → SIGKILL)
# ---------------------------------------------------------------------------

# Per-task overrides take precedence over these worker-wide defaults.
task_soft_time_limit: int = int(
    os.environ.get("CELERY_SOFT_TIME_LIMIT", str(15 * 60))  # 15 min
)
task_time_limit: int = int(
    os.environ.get("CELERY_HARD_TIME_LIMIT", str(30 * 60))  # 30 min
)

# Fine-grained per-queue time limits are enforced via task-level annotations
# (see analysis_tasks.py).  Reference values:
#
#   queue       soft_time_limit   time_limit
#   ---------   ---------------   ----------
#   default         5 min           10 min
#   analysis       15 min           30 min
#   training       50 min           60 min
#   reports        10 min           20 min

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------

timezone: str = "UTC"
enable_utc: bool = True

# ---------------------------------------------------------------------------
# Beat schedule (periodic tasks) – extend as needed
# ---------------------------------------------------------------------------

beat_schedule: dict = {}  # No periodic tasks at initial release

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

#: Worker log level; override with CELERY_LOG_LEVEL env var.
worker_log_format: str = (
    "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
)
worker_task_log_format: str = (
    "[%(asctime)s: %(levelname)s/%(processName)s]"
    "[%(task_name)s(%(task_id)s)] %(message)s"
)
