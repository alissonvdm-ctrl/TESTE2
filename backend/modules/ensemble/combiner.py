"""
EnsembleCombiner – weighted combination of per-module predictions.

Supports three strategies:
* ``weighted_avg`` – L1-normalised weighted sum of per-module scores.
* ``voting``       – each module votes for its top-k; tallied by weight.
* ``stacking``     – log-odds meta-combination; falls back to weighted_avg.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModuleOutput:
    module_name: str
    scores: Dict[int, float]
    weight: float = 1.0
    top_numbers: Optional[List[int]] = None


@dataclass
class EnsembleResult:
    predicted_numbers: List[int]
    confidence: float
    strategy: str
    module_contributions: List[Dict[str, Any]] = field(default_factory=list)
    score_distribution: Dict[int, float] = field(default_factory=dict)


class EnsembleCombiner:
    def __init__(self, n_max: int, k_count: int, strategy: str = "weighted_avg") -> None:
        if n_max < 1:
            raise ValueError(f"n_max must be >= 1, got {n_max}")
        if k_count < 1 or k_count > n_max:
            raise ValueError(f"k_count must be in [1, n_max], got {k_count}")
        if strategy not in ("weighted_avg", "voting", "stacking"):
            raise ValueError(f"Unknown strategy {strategy!r}.")
        self.n_max = n_max
        self.k_count = k_count
        self.strategy = strategy

    def combine(self, module_outputs: List[ModuleOutput]) -> EnsembleResult:
        usable = [m for m in module_outputs if m.scores]
        if not usable:
            return EnsembleResult(predicted_numbers=[], confidence=0.0, strategy=self.strategy)
        total_w = sum(m.weight for m in usable)
        for m in usable:
            m.weight = m.weight / total_w if total_w > 0 else 1.0 / len(usable)
        if self.strategy == "weighted_avg":
            return self._weighted_avg(usable)
        if self.strategy == "voting":
            return self._voting(usable)
        try:
            return self._stacking(usable)
        except Exception as exc:
            logger.warning("Stacking failed (%s); falling back to weighted_avg.", exc)
            return self._weighted_avg(usable)

    def _weighted_avg(self, modules: List[ModuleOutput]) -> EnsembleResult:
        combined: Dict[int, float] = {}
        for m in modules:
            normed = self._l1_normalise(m.scores)
            for num, score in normed.items():
                combined[num] = combined.get(num, 0.0) + m.weight * score
        return self._finalise(combined, self.strategy, modules)

    def _voting(self, modules: List[ModuleOutput]) -> EnsembleResult:
        votes: Dict[int, float] = {}
        for m in modules:
            top = m.top_numbers[:self.k_count] if m.top_numbers else self._top_k(m.scores, self.k_count)
            vote_w = m.weight / max(len(top), 1)
            for num in top:
                votes[num] = votes.get(num, 0.0) + vote_w
        return self._finalise(votes, self.strategy, modules)

    def _stacking(self, modules: List[ModuleOutput]) -> EnsembleResult:
        log_prior = math.log(1.0 / self.n_max)
        all_numbers: set = set()
        for m in modules:
            all_numbers.update(m.scores.keys())
        log_odds: Dict[int, float] = {}
        for num in all_numbers:
            lo = log_prior
            for m in modules:
                score = m.scores.get(num, 0.0)
                if score <= 0:
                    raise ValueError(f"Non-positive score for {num} in '{m.module_name}'")
                lo += m.weight * math.log(score)
            log_odds[num] = lo
        max_lo = max(log_odds.values())
        linear = {n: math.exp(lo - max_lo) for n, lo in log_odds.items()}
        return self._finalise(linear, self.strategy, modules)

    @staticmethod
    def _l1_normalise(scores: Dict[int, float]) -> Dict[int, float]:
        total = sum(abs(v) for v in scores.values())
        if total == 0:
            n = len(scores)
            return {k: 1.0 / n for k in scores} if n > 0 else {}
        return {k: v / total for k, v in scores.items()}

    @staticmethod
    def _top_k(scores: Dict[int, float], k: int) -> List[int]:
        return sorted(scores, key=scores.__getitem__, reverse=True)[:k]

    def _confidence(self, combined: Dict[int, float], top_numbers: List[int]) -> float:
        total = sum(combined.values())
        if total <= 0:
            return 0.0
        top_mass = sum(combined.get(n, 0.0) for n in top_numbers)
        baseline = self.k_count / max(self.n_max, 1)
        scaled = min(1.0, (top_mass / total) / max(baseline, 1e-9) * baseline)
        return round(min(max(scaled, 0.0), 1.0), 4)

    def _finalise(self, combined: Dict[int, float], strategy: str, modules: List[ModuleOutput]) -> EnsembleResult:
        top_numbers = sorted(self._top_k(combined, self.k_count))
        contributions = [
            {
                "module_name": m.module_name,
                "weight": round(m.weight, 4),
                "top_number_overlap": len(set(top_numbers) & set(self._top_k(m.scores, self.k_count))),
                "score_sum": round(sum(m.scores.get(n, 0.0) for n in top_numbers), 4),
            }
            for m in modules
        ]
        return EnsembleResult(
            predicted_numbers=top_numbers,
            confidence=self._confidence(combined, top_numbers),
            strategy=strategy,
            module_contributions=contributions,
            score_distribution={n: round(combined.get(n, 0.0), 6) for n in sorted(combined)},
        )


def combine_predictions(
    module_outputs: List[ModuleOutput],
    n_max: int,
    k_count: int,
    strategy: str = "weighted_avg",
) -> EnsembleResult:
    """One-shot convenience wrapper around EnsembleCombiner."""
    return EnsembleCombiner(n_max=n_max, k_count=k_count, strategy=strategy).combine(module_outputs)
