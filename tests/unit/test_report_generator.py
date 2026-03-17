"""Unit tests for backend.modules.reports.generator."""
import json
import tempfile
from pathlib import Path

import pytest
from backend.modules.reports.generator import (
    build_report_payload,
    render_json,
    render_html,
)


def _sample_payload():
    return build_report_payload(
        experiment_id=1,
        experiment_name="Test Exp",
        experiment_status="completed",
        n_max=60,
        k_count=6,
        order_matters=True,
        description="A test experiment",
        analysis_modules={
            "statistical": {
                "status": "success",
                "duration_seconds": 1.2,
                "created_at": "2024-01-01T00:00:00Z",
                "result": {"frequencies": {"1": 0.05}},
            }
        },
        models=[
            {
                "model_name": "RandomForest",
                "module": "ml",
                "val_score": 0.85,
                "test_score": 0.83,
                "trained_at": "2024-01-01T00:01:00Z",
                "metrics": {},
                "feature_importance": [],
            }
        ],
    )


class TestBuildReportPayload:
    def test_structure(self):
        p = _sample_payload()
        for key in ("report_metadata", "experiment_config", "summary", "analysis", "models"):
            assert key in p

    def test_best_model_selected(self):
        p = _sample_payload()
        assert p["summary"]["best_model"] == "RandomForest"

    def test_no_models(self):
        p = build_report_payload(
            experiment_id=2, experiment_name="Empty", experiment_status="completed",
            n_max=60, k_count=6, order_matters=True, description=None,
            analysis_modules={}, models=[],
        )
        assert p["summary"]["best_model"] is None


class TestRenderJson:
    def test_writes_valid_json(self):
        p = _sample_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            render_json(p, path)
            loaded = json.loads(path.read_text())
            assert loaded["report_metadata"]["experiment_id"] == 1

    def test_creates_parent_dirs(self):
        p = _sample_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "dir" / "report.json"
            render_json(p, path)
            assert path.exists()


class TestRenderHtml:
    def test_writes_html(self):
        p = _sample_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            render_html(p, path)
            content = path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "Test Exp" in content

    def test_contains_model_name(self):
        p = _sample_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            render_html(p, path)
            assert "RandomForest" in path.read_text()
