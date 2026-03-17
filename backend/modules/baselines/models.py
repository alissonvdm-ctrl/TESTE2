import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from itertools import combinations
import random


class BaseModel(ABC):
    """Abstract base for all baseline models."""
    name: str
    description: str

    @abstractmethod
    def fit(self, sequences: np.ndarray) -> None:
        """Fit the model on historical sequences.

        Parameters
        ----------
        sequences : np.ndarray
            2-D array of shape (n_draws, k) containing integer numbers
            drawn in each round.  Numbers are in range [1, n_max].
        """
        ...

    @abstractmethod
    def predict_proba(self) -> np.ndarray:
        """Return probability vector of shape (N,) for numbers 1..N."""
        ...

    def predict_combination(self, k: int, n_max: int) -> List[int]:
        """Sample a combination of k numbers using predict_proba."""
        probs = self.predict_proba()
        numbers = np.arange(1, n_max + 1)
        # Sample without replacement weighted by probabilities
        probs = np.clip(probs, 1e-10, None)
        probs = probs / probs.sum()
        chosen = np.random.choice(numbers, size=k, replace=False, p=probs)
        return sorted(chosen.tolist())

    def score(self, sequences: np.ndarray) -> float:
        """Average overlap between predictions and actual sequences.

        For each row in *sequences* we sample a prediction of the same
        length and count how many numbers match.  Returns the mean
        match count across all rows (normalised to [0, 1]).
        """
        if sequences.ndim != 2:
            raise ValueError("sequences must be 2-D (n_draws, k)")
        n_draws, k = sequences.shape
        n_max = int(sequences.max())
        total_overlap = 0.0
        for row in sequences:
            pred_set = set(self.predict_combination(k, n_max))
            actual_set = set(row.tolist())
            total_overlap += len(pred_set & actual_set)
        return total_overlap / (n_draws * k)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_multihot(sequences: np.ndarray, n_max: int) -> np.ndarray:
    """Convert (n_draws, k) integer array to (n_draws, n_max) multi-hot."""
    n_draws = sequences.shape[0]
    mh = np.zeros((n_draws, n_max), dtype=np.float64)
    for i, row in enumerate(sequences):
        for num in row:
            idx = int(num) - 1
            if 0 <= idx < n_max:
                mh[i, idx] = 1.0
    return mh


# ---------------------------------------------------------------------------
# Concrete model implementations
# ---------------------------------------------------------------------------

class UniformRandomModel(BaseModel):
    """Baseline: completely uniform random - theoretical ceiling for noise."""

    name = "uniform_random"
    description = "Uniform random selection - benchmark for noise floor"

    def __init__(self) -> None:
        self._n_max: Optional[int] = None

    def fit(self, sequences: np.ndarray) -> None:
        """Infer n_max from data; no real fitting required."""
        self._n_max = int(sequences.max())

    def predict_proba(self) -> np.ndarray:
        """Uniform distribution over 1..n_max."""
        if self._n_max is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.ones(self._n_max, dtype=np.float64) / self._n_max


class GlobalFrequencyModel(BaseModel):
    """Baseline: historical frequency of each number."""

    name = "global_frequency"
    description = "Historical frequency weighting"

    def __init__(self) -> None:
        self._probs: Optional[np.ndarray] = None

    def fit(self, sequences: np.ndarray) -> None:
        n_max = int(sequences.max())
        counts = np.zeros(n_max, dtype=np.float64)
        for row in sequences:
            for num in row:
                idx = int(num) - 1
                if 0 <= idx < n_max:
                    counts[idx] += 1.0
        total = counts.sum()
        if total == 0:
            self._probs = np.ones(n_max) / n_max
        else:
            self._probs = counts / total

    def predict_proba(self) -> np.ndarray:
        if self._probs is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._probs.copy()


class RecencyWeightedFrequencyModel(BaseModel):
    """Baseline: frequency weighted by recency (exponential decay)."""

    name = "recency_weighted"
    description = "Recent draws weighted more heavily"

    def __init__(self, decay: float = 0.95) -> None:
        self.decay = decay
        self._probs: Optional[np.ndarray] = None

    def fit(self, sequences: np.ndarray) -> None:
        n_draws, _ = sequences.shape
        n_max = int(sequences.max())
        counts = np.zeros(n_max, dtype=np.float64)
        # Most recent draw has weight decay^0 = 1; oldest has weight decay^(n-1)
        for t, row in enumerate(sequences):
            weight = self.decay ** (n_draws - 1 - t)
            for num in row:
                idx = int(num) - 1
                if 0 <= idx < n_max:
                    counts[idx] += weight
        total = counts.sum()
        if total == 0:
            self._probs = np.ones(n_max) / n_max
        else:
            self._probs = counts / total

    def predict_proba(self) -> np.ndarray:
        if self._probs is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._probs.copy()


class CooccurrenceModel(BaseModel):
    """Baseline: selection based on co-occurrence matrix.

    For each number i, we count how often number j appeared together
    with i across all draws.  The score of number j is the sum over
    all recently drawn numbers i of cooccurrence[i, j].  We use the
    last observed draw as the 'context'.
    """

    name = "cooccurrence"
    description = "Co-occurrence based selection"

    def __init__(self) -> None:
        self._cooc: Optional[np.ndarray] = None
        self._last_draw: Optional[np.ndarray] = None
        self._n_max: Optional[int] = None

    def fit(self, sequences: np.ndarray) -> None:
        n_max = int(sequences.max())
        self._n_max = n_max
        cooc = np.zeros((n_max, n_max), dtype=np.float64)
        for row in sequences:
            nums = [int(x) - 1 for x in row if 1 <= int(x) <= n_max]
            for a, b in combinations(nums, 2):
                cooc[a, b] += 1.0
                cooc[b, a] += 1.0
        # Row-normalise; keep raw if row is all zeros
        row_sums = cooc.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self._cooc = cooc / row_sums
        # Store last draw for prediction context
        self._last_draw = sequences[-1]

    def predict_proba(self) -> np.ndarray:
        if self._cooc is None or self._last_draw is None:
            raise RuntimeError("Model has not been fitted yet.")
        n_max = self._n_max
        scores = np.zeros(n_max, dtype=np.float64)
        for num in self._last_draw:
            idx = int(num) - 1
            if 0 <= idx < n_max:
                scores += self._cooc[idx]
        # Zero out numbers that appeared in the last draw so we don't
        # recommend the same combination again (optional heuristic)
        total = scores.sum()
        if total == 0:
            return np.ones(n_max) / n_max
        return scores / total


class MarkovOrder1Model(BaseModel):
    """First-order Markov chain on individual numbers.

    State = set of numbers drawn in draw t (multi-hot representation).
    For each number j, its next-draw probability is estimated as the
    weighted average of transition rows for numbers that appeared in
    the last draw.
    """

    name = "markov_1"
    description = "1st order Markov chain"

    def __init__(self) -> None:
        self._transition: Optional[np.ndarray] = None
        self._last_draw: Optional[np.ndarray] = None
        self._n_max: Optional[int] = None

    def fit(self, sequences: np.ndarray) -> None:
        n_max = int(sequences.max())
        self._n_max = n_max
        # transition[i, j] = P(j appears in draw t+1 | i appeared in draw t)
        trans_count = np.zeros((n_max, n_max), dtype=np.float64)
        trans_total = np.zeros(n_max, dtype=np.float64)

        for t in range(len(sequences) - 1):
            curr = [int(x) - 1 for x in sequences[t] if 1 <= int(x) <= n_max]
            nxt = [int(x) - 1 for x in sequences[t + 1] if 1 <= int(x) <= n_max]
            for i in curr:
                trans_total[i] += 1.0
                for j in nxt:
                    trans_count[i, j] += 1.0

        # Normalise rows
        denom = np.where(trans_total == 0, 1.0, trans_total)
        self._transition = trans_count / denom[:, np.newaxis]
        self._last_draw = sequences[-1]

    def predict_proba(self) -> np.ndarray:
        if self._transition is None or self._last_draw is None:
            raise RuntimeError("Model has not been fitted yet.")
        n_max = self._n_max
        scores = np.zeros(n_max, dtype=np.float64)
        context = [int(x) - 1 for x in self._last_draw if 1 <= int(x) <= n_max]
        for i in context:
            scores += self._transition[i]
        if len(context) > 0:
            scores /= len(context)
        total = scores.sum()
        if total == 0:
            return np.ones(n_max) / n_max
        return scores / total


class MarkovOrder2Model(BaseModel):
    """Second-order Markov chain on individual numbers.

    Context = pair of consecutive draws (t-1, t).  We encode the
    context as a frozenset pair and store counts in a dictionary to
    avoid an (n_max^2 x n_max) dense array.
    """

    name = "markov_2"
    description = "2nd order Markov chain"

    def __init__(self) -> None:
        self._trans2: Optional[Dict] = None
        self._last_two: Optional[np.ndarray] = None
        self._n_max: Optional[int] = None
        self._order1_fallback: Optional[MarkovOrder1Model] = None

    def fit(self, sequences: np.ndarray) -> None:
        n_max = int(sequences.max())
        self._n_max = n_max
        # Sparse storage: key=(frozenset t-1, frozenset t), value=count array
        trans2: Dict[Tuple[frozenset, frozenset], np.ndarray] = {}

        for t in range(len(sequences) - 2):
            ctx_key = (
                frozenset(int(x) for x in sequences[t]),
                frozenset(int(x) for x in sequences[t + 1]),
            )
            nxt = [int(x) - 1 for x in sequences[t + 2] if 1 <= int(x) <= n_max]
            if ctx_key not in trans2:
                trans2[ctx_key] = np.zeros(n_max, dtype=np.float64)
            for j in nxt:
                trans2[ctx_key][j] += 1.0

        # Normalise
        for key in trans2:
            total = trans2[key].sum()
            if total > 0:
                trans2[key] /= total

        self._trans2 = trans2
        if len(sequences) >= 2:
            self._last_two = sequences[-2:]
        else:
            self._last_two = sequences[-1:]

        # Fit order-1 as fallback
        fallback = MarkovOrder1Model()
        fallback.fit(sequences)
        self._order1_fallback = fallback

    def predict_proba(self) -> np.ndarray:
        if self._trans2 is None or self._last_two is None:
            raise RuntimeError("Model has not been fitted yet.")
        if len(self._last_two) < 2:
            return self._order1_fallback.predict_proba()

        ctx_key = (
            frozenset(int(x) for x in self._last_two[0]),
            frozenset(int(x) for x in self._last_two[1]),
        )
        if ctx_key in self._trans2:
            probs = self._trans2[ctx_key].copy()
            total = probs.sum()
            if total > 0:
                return probs / total
        # Fall back to order-1 if context unseen
        return self._order1_fallback.predict_proba()


class VariableOrderMarkovModel(BaseModel):
    """Variable-order Markov model using context tree.

    Tries contexts of decreasing length (max_order down to 1) and uses
    the longest context that has been seen in training.  Unseen contexts
    fall back to global frequency.
    """

    name = "markov_variable"
    description = "Variable-order Markov (context tree)"

    def __init__(self, max_order: int = 5) -> None:
        self.max_order = max_order
        self._context_tables: Optional[List[Dict]] = None
        self._last_draws: Optional[np.ndarray] = None
        self._global_probs: Optional[np.ndarray] = None
        self._n_max: Optional[int] = None

    def fit(self, sequences: np.ndarray) -> None:
        n_max = int(sequences.max())
        self._n_max = n_max
        # context_tables[order] maps frozenset-tuple-of-length-order -> prob array
        context_tables: List[Dict] = [{} for _ in range(self.max_order + 1)]

        for order in range(1, self.max_order + 1):
            counts: Dict[Tuple, np.ndarray] = {}
            totals: Dict[Tuple, float] = {}
            for t in range(order, len(sequences)):
                # Build context tuple as tuple of frozensets
                ctx_key = tuple(
                    frozenset(int(x) for x in sequences[t - order + i])
                    for i in range(order)
                )
                nxt = [int(x) - 1 for x in sequences[t] if 1 <= int(x) <= n_max]
                if ctx_key not in counts:
                    counts[ctx_key] = np.zeros(n_max, dtype=np.float64)
                    totals[ctx_key] = 0.0
                for j in nxt:
                    counts[ctx_key][j] += 1.0
                totals[ctx_key] += 1.0
            # Normalise
            for key in counts:
                t_val = totals[key]
                if t_val > 0:
                    context_tables[order][key] = counts[key] / t_val

        self._context_tables = context_tables
        self._last_draws = sequences[-self.max_order:] if len(sequences) >= self.max_order else sequences

        # Global frequency as ultimate fallback
        freq_model = GlobalFrequencyModel()
        freq_model.fit(sequences)
        self._global_probs = freq_model.predict_proba()

    def predict_proba(self) -> np.ndarray:
        if self._context_tables is None or self._last_draws is None:
            raise RuntimeError("Model has not been fitted yet.")

        # Try longest context first
        for order in range(min(self.max_order, len(self._last_draws)), 0, -1):
            ctx_key = tuple(
                frozenset(int(x) for x in self._last_draws[-order + i])
                for i in range(order)
            )
            if ctx_key in self._context_tables[order]:
                probs = self._context_tables[order][ctx_key].copy()
                total = probs.sum()
                if total > 0:
                    return probs / total

        # Ultimate fallback: global frequency
        return self._global_probs.copy()
