"""Unit tests for backend.modules.ensemble.combiner."""
import pytest
from backend.modules.ensemble.combiner import (
    EnsembleCombiner,
    ModuleOutput,
    combine_predictions,
)


def _make_output(name: str, scores: dict, weight: float = 1.0) -> ModuleOutput:
    return ModuleOutput(module_name=name, scores=scores, weight=weight)


class TestEnsembleCombiner:
    def test_weighted_avg_picks_top_k(self):
        scores = {i: float(i) for i in range(1, 11)}  # numbers 1-10
        out = _make_output("stat", scores)
        result = EnsembleCombiner(n_max=10, k_count=3).combine([out])
        assert len(result.predicted_numbers) == 3
        # Highest-scored numbers should be picked
        assert set(result.predicted_numbers) == {8, 9, 10}

    def test_confidence_in_range(self):
        scores = {i: 1.0 for i in range(1, 61)}
        out = _make_output("stat", scores)
        result = EnsembleCombiner(n_max=60, k_count=6).combine([out])
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_modules_returns_empty(self):
        result = EnsembleCombiner(n_max=60, k_count=6).combine([])
        assert result.predicted_numbers == []
        assert result.confidence == 0.0

    def test_voting_strategy(self):
        scores = {i: float(i) for i in range(1, 11)}
        out = _make_output("stat", scores)
        result = EnsembleCombiner(n_max=10, k_count=3, strategy="voting").combine([out])
        assert len(result.predicted_numbers) == 3

    def test_multiple_modules_weighted(self):
        a = _make_output("a", {1: 0.8, 2: 0.2}, weight=2.0)
        b = _make_output("b", {1: 0.1, 2: 0.9}, weight=1.0)
        result = EnsembleCombiner(n_max=5, k_count=1).combine([a, b])
        # Module a has 2x weight so number 1 should win
        assert result.predicted_numbers == [1]

    def test_invalid_k_count_raises(self):
        with pytest.raises(ValueError):
            EnsembleCombiner(n_max=10, k_count=0)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            EnsembleCombiner(n_max=10, k_count=3, strategy="magic")

    def test_stacking_falls_back_on_zero_scores(self):
        # Zero scores should trigger fallback to weighted_avg
        a = _make_output("a", {1: 0.0, 2: 1.0})
        result = EnsembleCombiner(n_max=5, k_count=1, strategy="stacking").combine([a])
        # Falls back – should still return a result
        assert len(result.predicted_numbers) == 1

    def test_convenience_function(self):
        scores = {i: float(i) for i in range(1, 7)}
        out = _make_output("stat", scores)
        result = combine_predictions([out], n_max=6, k_count=3)
        assert len(result.predicted_numbers) == 3
