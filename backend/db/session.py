"""
Re-export of database lifecycle helpers from backend.core.database.

api/main.py imports ``init_db`` and ``close_db`` from this module; they
are defined in backend.core.database and simply re-exported here for
backward-compatibility.
"""

from backend.core.database import close_db, init_db  # noqa: F401

__all__ = ["init_db", "close_db"]
