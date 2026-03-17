"""
Route modules for the Numeric Sequence Analysis API.

Each sub-module defines a FastAPI ``APIRouter`` that is registered
in ``backend.api.main.create_app()``.
"""

from backend.api.routes import analysis, experiments, reports, uploads

__all__ = ["analysis", "experiments", "reports", "uploads"]
