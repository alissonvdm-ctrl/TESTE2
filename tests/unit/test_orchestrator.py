"""Unit tests for backend.core.orchestrator."""
import pytest
from backend.core.orchestrator import (
    AnalyticsOrchestrator,
    ExperimentConfig,
    ModuleType,
    PipelineResult,
)


def _make_config(**kwargs) -> ExperimentConfig:
    defaults = dict(n_max=60, k_count=6, order_matters=True, sample_count=200, has_timestamps=True)
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)


def _make_samples(n: int = 100, k: int = 6, n_max: int = 60):
    import random
    rng = random.Random(42)
    return [sorted(rng.sample(range(1, n_max + 1), k)) for _ in range(n)]


class TestOrchestrator:
    def test_run_returns_pipeline_result(self):
        cfg = _make_config()
        samples = _make_samples()
        orch = AnalyticsOrchestrator(config=cfg, raw_samples=samples)
        result = orch.run()
        assert isinstance(result, PipelineResult)

    def test_statistical_module_always_runs(self):
        cfg = _make_config()
        samples = _make_samples()
        result = AnalyticsOrchestrator(config=cfg, raw_samples=samples).run()
        assert "statistical" in result.module_results

    def test_ensemble_module_always_runs(self):
        cfg = _make_config()
        samples = _make_samples()
        result = AnalyticsOrchestrator(config=cfg, raw_samples=samples).run()
        assert "ensemble" in result.module_results

    def test_deep_learning_disabled_on_low_samples(self):
        cfg = _make_config(sample_count=50)
        samples = _make_samples(n=50)
        result = AnalyticsOrchestrator(config=cfg, raw_samples=samples).run()
        assert "deep_learning" in result.skipped_modules

    def test_set_based_active_when_order_irrelevant(self):
        cfg = _make_config(order_matters=False)
        samples = _make_samples()
        result = AnalyticsOrchestrator(config=cfg, raw_samples=samples).run()
        assert "set_based" in result.module_results
        assert "sequential" not in result.module_results

    def test_timing_recorded(self):
        cfg = _make_config()
        samples = _make_samples()
        result = AnalyticsOrchestrator(config=cfg, raw_samples=samples).run()
        assert result.total_duration_seconds > 0
        assert len(result.per_module_duration) > 0

    def test_ensemble_result_has_predicted_numbers(self):
        cfg = _make_config()
        samples = _make_samples()
        result = AnalyticsOrchestrator(config=cfg, raw_samples=samples).run()
        ensemble = result.ensemble_result
        assert ensemble is not None
        assert "predicted_numbers" in ensemble
        assert len(ensemble["predicted_numbers"]) == cfg.k_count
