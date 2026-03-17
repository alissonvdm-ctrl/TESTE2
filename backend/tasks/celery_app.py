"""
Celery application factory for the NumericSequenceAnalyzer.

Creates and configures a Celery instance backed by Redis as both the
message broker and the result backend.  All configuration is loaded from
``backend.tasks.celery_config`` so that settings stay in a single place.

Usage (in other modules)::

    from backend.tasks.celery_app import celery_app

    @celery_app.task
    def my_task(): ...

Or via the shared-task decorator (preferred for reusable tasks)::

    from celery import shared_task

    @shared_task
    def my_task(): ...
"""

from __future__ import annotations

from celery import Celery
from kombu import Exchange, Queue


def create_celery_app() -> Celery:
    """
    Create and configure the Celery application with Redis broker.

    Configuration is read from ``backend.tasks.celery_config`` via
    ``app.config_from_object``.  This keeps the factory clean and makes
    the settings easy to override in tests (swap the config object).

    Returns
    -------
    Celery
        A fully-configured Celery application instance.
    """
    app = Celery("sequence_analyzer")

    # Load all settings from the config module
    app.config_from_object("backend.tasks.celery_config")

    # Autodiscover tasks in the tasks package so they are registered
    # automatically when the worker starts.
    app.autodiscover_tasks(["backend.tasks"])

    return app


# ---------------------------------------------------------------------------
# Module-level singleton – import this everywhere
# ---------------------------------------------------------------------------

celery_app: Celery = create_celery_app()
