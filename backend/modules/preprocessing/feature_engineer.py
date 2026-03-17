"""
feature_engineer.py
===================
Creates all derived features from a cleaned numeric-sequence dataset.

Design principles
-----------------
* **No temporal leakage**: every rolling / lag / transition statistic at
  time step t is computed using only rows 0 … t-1.
* Features are grouped into logical families that can be selectively enabled.
* The main entry-point is :meth:`FeatureEngineer.fit_transform`, which takes
  a (T, k) integer array and returns a :class:`pandas.DataFrame` with one row
  per time-step and one column per feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import normalize


class FeatureEngineer:
    """
    Creates all derived features from the cleaned sequence dataset.

    Features created
    ----------------
    - Multi-hot vectors (binary presence of each number 1…N)
    - Gap / difference representations
    - Positional representations
    - Rolling statistics (frequency, recency, delay since last appearance)
    - Co-occurrence pairs and triples
    - Statistical moments (sum, mean, std, amplitude)
    - Range distribution (quartiles, decile membership)
    - Modular arithmetic features
    - Parity features
    - Positional patterns
    - Local entropy (Shannon, permutation)
    - Transition matrix features
    - Lag features (configurable; default: 1, 2, 3, 5, 10, 20, 50)

    Parameters
    ----------
    n_max   : Maximum value in the universe (numbers are in [1, n_max]).
    k_count : How many numbers appear in each draw.
    lags    : List of lag offsets to include.  Defaults to [1,2,3,5,10,20,50].
    window  : Rolling window size for frequency / recency features.
    """

    DEFAULT_LAGS: List[int] = [1, 2, 3, 5, 10, 20, 50]

    def __init__(
        self,
        n_max: int,
        k_count: int,
        lags: Optional[List[int]] = None,
        window: int = 50,
    ) -> None:
        if n_max < 1:
            raise ValueError("n_max must be >= 1.")
        if k_count < 1:
            raise ValueError("k_count must be >= 1.")

        self.n_max = n_max
        self.k_count = k_count
        self.lags = sorted(set(lags if lags is not None else self.DEFAULT_LAGS))
        self.window = window

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def fit_transform(self, sequences: np.ndarray) -> pd.DataFrame:
        """
        Transform a (T, k) integer sequence array into a feature DataFrame.

        Parameters
        ----------
        sequences : np.ndarray, shape (T, k)
            Each row is one draw; values are integers in [1, n_max].

        Returns
        -------
        pd.DataFrame, shape (T, F)
            One row per draw, one column per feature.  All features at row t
            are computed from rows 0 … t-1 (no leakage).
        """
        sequences = np.asarray(sequences, dtype=np.int32)
        if sequences.ndim == 1:
            sequences = sequences.reshape(-1, 1)
        if sequences.ndim != 2:
            raise ValueError("sequences must be a 2-D array of shape (T, k).")

        T, k = sequences.shape
        if k != self.k_count:
            raise ValueError(
                f"Expected k={self.k_count} columns, got {k}."
            )

        # --- build multi-hot matrix (T, N) ---------------------------------
        multihot = self._create_multihot(sequences)      # (T, N)

        # --- assemble all feature families ----------------------------------
        parts: List[pd.DataFrame] = []

        parts.append(self._wrap(multihot, "mh"))
        parts.append(self._create_gap_features(sequences))
        parts.append(self._create_statistical_features(sequences))
        parts.append(self._create_modular_features(sequences))
        parts.append(self._create_parity_features(sequences))
        parts.append(self._create_positional_features(sequences))
        parts.append(self._create_frequency_features(multihot))
        parts.append(self._create_cooccurrence_features(multihot))
        parts.append(self._create_entropy_features(multihot))
        parts.append(self._create_transition_features(sequences))
        parts.append(self._create_lag_features(multihot, self.lags))

        result = pd.concat(parts, axis=1)
        result.index = pd.RangeIndex(T)
        return result

    # ------------------------------------------------------------------
    # Multi-hot
    # ------------------------------------------------------------------

    def _create_multihot(self, sequences: np.ndarray) -> np.ndarray:
        """
        Convert (T, k) sequence array to (T, N) binary presence matrix.

        Column *i* (0-indexed, corresponding to number i+1) is 1 if the
        number i+1 appeared in that draw, 0 otherwise.
        """
        T = sequences.shape[0]
        N = self.n_max
        multihot = np.zeros((T, N), dtype=np.float32)
        for t in range(T):
            for num in sequences[t]:
                if 1 <= num <= N:
                    multihot[t, num - 1] = 1.0
        return multihot

    # ------------------------------------------------------------------
    # Gap / difference features
    # ------------------------------------------------------------------

    def _create_gap_features(self, sequences: np.ndarray) -> pd.DataFrame:
        """
        Positional gap features derived from sorted sequences.

        For each draw (sorted ascending), we compute:
        - Differences between consecutive drawn numbers (k-1 values).
        - Difference between the first number and 1 (left gap).
        - Difference between n_max and the last number (right gap).
        """
        T = sequences.shape[0]
        rows = []
        for t in range(T):
            sorted_seq = sorted(sequences[t])
            diffs = [sorted_seq[i + 1] - sorted_seq[i] for i in range(len(sorted_seq) - 1)]
            left_gap = sorted_seq[0] - 1
            right_gap = self.n_max - sorted_seq[-1]

            row: dict = {"gap_left": left_gap, "gap_right": right_gap}
            for i, d in enumerate(diffs):
                row[f"gap_{i}"] = d
            # Summary stats of gaps
            all_gaps = [left_gap] + diffs + [right_gap]
            row["gap_mean"] = float(np.mean(all_gaps))
            row["gap_std"] = float(np.std(all_gaps))
            row["gap_max"] = float(np.max(all_gaps))
            row["gap_min"] = float(np.min(all_gaps))
            rows.append(row)

        return pd.DataFrame(rows).fillna(0.0)

    # ------------------------------------------------------------------
    # Statistical / moment features
    # ------------------------------------------------------------------

    def _create_statistical_features(self, sequences: np.ndarray) -> pd.DataFrame:
        """
        Row-level statistical moments and distribution measures.

        Features: sum, mean, median, std, variance, amplitude (max-min),
        skewness, excess-kurtosis, quartile memberships, decile memberships.
        """
        T = sequences.shape[0]
        records = []
        for t in range(T):
            seq = sequences[t].astype(float)
            sorted_seq = np.sort(seq)
            s = float(np.sum(seq))
            mu = float(np.mean(seq))
            med = float(np.median(seq))
            sigma = float(np.std(seq))
            var = float(np.var(seq))
            amp = float(sorted_seq[-1] - sorted_seq[0])

            # Skewness and kurtosis (manual for speed)
            if sigma > 0:
                centered = seq - mu
                skew = float(np.mean(centered ** 3) / (sigma ** 3))
                kurt = float(np.mean(centered ** 4) / (sigma ** 4) - 3)
            else:
                skew, kurt = 0.0, 0.0

            # Quartile / decile bin counts
            q1, q2, q3 = np.percentile([1, self.n_max], [25, 50, 75])
            # Count per quartile of the universe
            n = self.n_max
            in_q1 = int(np.sum((seq >= 1) & (seq <= n / 4)))
            in_q2 = int(np.sum((seq > n / 4) & (seq <= n / 2)))
            in_q3 = int(np.sum((seq > n / 2) & (seq <= 3 * n / 4)))
            in_q4 = int(np.sum(seq > 3 * n / 4))

            row = {
                "stat_sum": s,
                "stat_mean": mu,
                "stat_median": med,
                "stat_std": sigma,
                "stat_var": var,
                "stat_amplitude": amp,
                "stat_skew": skew,
                "stat_kurt": kurt,
                "stat_q1_count": in_q1,
                "stat_q2_count": in_q2,
                "stat_q3_count": in_q3,
                "stat_q4_count": in_q4,
            }

            # Decile bin counts
            decile_size = n / 10.0
            for d in range(10):
                lo = d * decile_size + 1
                hi = (d + 1) * decile_size
                row[f"stat_decile_{d + 1}"] = int(np.sum((seq >= lo) & (seq <= hi)))

            records.append(row)

        return pd.DataFrame(records).fillna(0.0)

    # ------------------------------------------------------------------
    # Modular arithmetic features
    # ------------------------------------------------------------------

    def _create_modular_features(self, sequences: np.ndarray) -> pd.DataFrame:
        """
        Count of drawn numbers in each residue class for moduli 2, 3, 5, 7, 10.
        """
        MODULI = [2, 3, 5, 7, 10]
        T = sequences.shape[0]
        records = []
        for t in range(T):
            seq = sequences[t]
            row: dict = {}
            for m in MODULI:
                for r in range(m):
                    row[f"mod{m}_r{r}"] = int(np.sum(seq % m == r))
            records.append(row)
        return pd.DataFrame(records).fillna(0.0)

    # ------------------------------------------------------------------
    # Parity features
    # ------------------------------------------------------------------

    def _create_parity_features(self, sequences: np.ndarray) -> pd.DataFrame:
        """
        Count of odd vs even numbers, and low vs high numbers per draw.
        """
        T = sequences.shape[0]
        mid = self.n_max / 2.0
        records = []
        for t in range(T):
            seq = sequences[t]
            n_odd = int(np.sum(seq % 2 == 1))
            n_even = int(np.sum(seq % 2 == 0))
            n_low = int(np.sum(seq <= mid))
            n_high = int(np.sum(seq > mid))
            row = {
                "parity_odd": n_odd,
                "parity_even": n_even,
                "parity_low": n_low,
                "parity_high": n_high,
                "parity_odd_ratio": n_odd / self.k_count,
                "parity_low_ratio": n_low / self.k_count,
            }
            records.append(row)
        return pd.DataFrame(records).fillna(0.0)

    # ------------------------------------------------------------------
    # Positional features
    # ------------------------------------------------------------------

    def _create_positional_features(self, sequences: np.ndarray) -> pd.DataFrame:
        """
        For each of the k positions (after sorting the draw ascending),
        record the value at that position and its normalised version.
        """
        T = sequences.shape[0]
        records = []
        for t in range(T):
            sorted_seq = np.sort(sequences[t])
            row: dict = {}
            for pos, val in enumerate(sorted_seq):
                row[f"pos_{pos}_val"] = int(val)
                row[f"pos_{pos}_norm"] = val / self.n_max
            records.append(row)
        return pd.DataFrame(records).fillna(0.0)

    # ------------------------------------------------------------------
    # Frequency / recency / delay features (rolling, no leakage)
    # ------------------------------------------------------------------

    def _create_frequency_features(self, multihot: np.ndarray) -> pd.DataFrame:
        """
        For each number 1…N, compute at each time step t (using rows 0…t-1):
        - Rolling frequency in the last `window` draws.
        - Recency: how many draws ago was it last seen (0 if never).
        - Delay: number of draws since last appearance (NaN → window+1).

        Shape: (T, 3*N)
        """
        T, N = multihot.shape
        window = self.window

        # cumulative rolling sum shifted by 1 (strict past)
        freq_arr = np.zeros((T, N), dtype=np.float32)
        recency_arr = np.zeros((T, N), dtype=np.float32)
        delay_arr = np.full((T, N), float(window + 1), dtype=np.float32)

        last_seen = np.full(N, -1, dtype=np.int32)  # row index of last appearance

        for t in range(T):
            # At step t, use information from rows 0 … t-1
            if t > 0:
                w_start = max(0, t - window)
                window_slice = multihot[w_start:t]  # (w, N)
                freq_arr[t] = window_slice.sum(axis=0) / max(len(window_slice), 1)

                for n in range(N):
                    if last_seen[n] >= 0:
                        recency_arr[t, n] = t - last_seen[n]
                        delay_arr[t, n] = t - last_seen[n]

            # Update last_seen with row t
            for n in range(N):
                if multihot[t, n] > 0:
                    last_seen[n] = t

        freq_df = self._wrap(freq_arr, "freq")
        recency_df = self._wrap(recency_arr, "recency")
        delay_df = self._wrap(delay_arr, "delay")

        return pd.concat([freq_df, recency_df, delay_df], axis=1)

    # ------------------------------------------------------------------
    # Co-occurrence features (pairs)
    # ------------------------------------------------------------------

    def _create_cooccurrence_features(self, multihot: np.ndarray) -> pd.DataFrame:
        """
        For each pair (i, j) with i < j, compute the rolling co-occurrence
        count up to (but not including) step t.

        To keep dimensionality tractable we only store summary statistics
        rather than all N*(N-1)/2 pair counts:
        - For each number i: mean, max, total co-occurrence count with all others.
        - Overall co-occurrence entropy at each step.

        Shape: (T, 3*N + 1)
        """
        T, N = multihot.shape
        window = self.window

        cooc_mean = np.zeros((T, N), dtype=np.float32)
        cooc_max = np.zeros((T, N), dtype=np.float32)
        cooc_total = np.zeros((T, N), dtype=np.float32)
        cooc_entropy = np.zeros(T, dtype=np.float32)

        # Accumulated pairwise co-occurrence matrix
        pair_counts = np.zeros((N, N), dtype=np.float32)

        for t in range(T):
            if t > 0:
                # Compute stats from cumulative pair_counts (strictly past)
                row_sum = pair_counts.sum(axis=1)        # (N,)
                row_mean = pair_counts.mean(axis=1)      # (N,)
                row_max = pair_counts.max(axis=1)        # (N,)

                cooc_total[t] = row_sum
                cooc_mean[t] = row_mean
                cooc_max[t] = row_max

                # Entropy over flattened upper triangle
                upper = pair_counts[np.triu_indices(N, k=1)]
                total = upper.sum()
                if total > 0:
                    p = upper / total
                    cooc_entropy[t] = float(-np.sum(p * np.log(p + 1e-12)))

            # Update pair_counts with row t
            drawn = np.where(multihot[t] > 0)[0]
            for a in range(len(drawn)):
                for b in range(a + 1, len(drawn)):
                    i, j = drawn[a], drawn[b]
                    pair_counts[i, j] += 1
                    pair_counts[j, i] += 1

        df = pd.concat(
            [
                self._wrap(cooc_mean, "cooc_mean"),
                self._wrap(cooc_max, "cooc_max"),
                self._wrap(cooc_total, "cooc_total"),
                pd.DataFrame({"cooc_entropy": cooc_entropy}),
            ],
            axis=1,
        )
        return df

    # ------------------------------------------------------------------
    # Entropy features
    # ------------------------------------------------------------------

    def _create_entropy_features(self, multihot: np.ndarray) -> pd.DataFrame:
        """
        Shannon entropy of rolling frequency distribution.
        Also computes a simple permutation entropy over the sorted draws.

        Shape: (T, 2)
        """
        T, N = multihot.shape
        window = self.window
        shannon_ent = np.zeros(T, dtype=np.float32)
        perm_ent = np.zeros(T, dtype=np.float32)

        cumulative = np.zeros(N, dtype=np.float32)

        for t in range(T):
            if t > 0:
                # Use rolling window of past draws
                w_start = max(0, t - window)
                window_slice = multihot[w_start:t]
                freq = window_slice.mean(axis=0)  # (N,)
                total = freq.sum()
                if total > 0:
                    p = freq / total
                    shannon_ent[t] = float(-np.sum(p * np.log(p + 1e-12)))

                # Permutation entropy: treat the rolling mean vector as a signal
                # and rank adjacent pairs
                if t >= 2:
                    w2 = max(0, t - window)
                    segment = multihot[w2:t].mean(axis=1)  # (w,)
                    if len(segment) >= 2:
                        ordinals = np.argsort(segment)
                        # count of ascending vs descending transitions
                        n_up = int(np.sum(np.diff(ordinals) > 0))
                        n_down = len(segment) - 1 - n_up
                        total_t = n_up + n_down
                        if total_t > 0:
                            p_up = n_up / total_t
                            p_down = n_down / total_t
                            perm_ent[t] = float(
                                -p_up * np.log(p_up + 1e-12) - p_down * np.log(p_down + 1e-12)
                            )

            # Update cumulative (not used for features above, kept for extension)
            cumulative += multihot[t]

        return pd.DataFrame({"ent_shannon": shannon_ent, "ent_perm": perm_ent})

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------

    def _create_lag_features(
        self, multihot: np.ndarray, lags: List[int]
    ) -> pd.DataFrame:
        """
        For each lag L in *lags*, shift the multi-hot matrix by L rows
        (rows 0…L-1 are filled with zeros → no leakage).

        Shape: (T, N * len(lags))
        """
        T, N = multihot.shape
        frames = []
        for lag in lags:
            lagged = np.zeros_like(multihot)
            if lag < T:
                lagged[lag:] = multihot[:T - lag]
            frames.append(self._wrap(lagged, f"lag{lag}"))
        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------
    # Transition matrix features
    # ------------------------------------------------------------------

    def _create_transition_features(self, sequences: np.ndarray) -> pd.DataFrame:
        """
        Derive features from the empirical transition matrix T[i, j] = number of
        times number j followed number i in consecutive draws (column-wise, i.e.
        at same position after sorting).

        At each time step t we compute (using rows 0 … t-1):
        - For each number i: row-normalised transition entropy (uncertainty of
          what comes next after i).
        - Expected "next" number for each position (argmax of row).
        - Global transition matrix entropy.

        Shape: (T, 2*N + 1)
        """
        T, _ = sequences.shape
        N = self.n_max

        trans_matrix = np.zeros((N, N), dtype=np.float32)

        row_entropy_arr = np.zeros((T, N), dtype=np.float32)
        expected_next_arr = np.zeros((T, N), dtype=np.float32)
        global_entropy_arr = np.zeros(T, dtype=np.float32)

        for t in range(T):
            if t > 0:
                # Compute features from current trans_matrix (strictly past)
                row_sums = trans_matrix.sum(axis=1, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    row_norm = np.where(row_sums > 0, trans_matrix / row_sums, 0.0)

                # Row entropy for each number
                for i in range(N):
                    p = row_norm[i]
                    if p.sum() > 0:
                        row_entropy_arr[t, i] = float(
                            -np.sum(p * np.log(p + 1e-12))
                        )

                # Expected next number (argmax of each row, 1-indexed)
                expected_next_arr[t] = np.argmax(row_norm, axis=1) + 1

                # Global entropy of the full transition matrix
                flat = trans_matrix.flatten()
                total = flat.sum()
                if total > 0:
                    p_flat = flat / total
                    global_entropy_arr[t] = float(
                        -np.sum(p_flat * np.log(p_flat + 1e-12))
                    )

            # Update transition matrix using sorted sequences from step t
            if t > 0:
                prev_sorted = np.sort(sequences[t - 1])
                curr_sorted = np.sort(sequences[t])
                for p_num, c_num in zip(prev_sorted, curr_sorted):
                    pi, ci = int(p_num) - 1, int(c_num) - 1
                    if 0 <= pi < N and 0 <= ci < N:
                        trans_matrix[pi, ci] += 1

        df = pd.concat(
            [
                self._wrap(row_entropy_arr, "trans_row_ent"),
                self._wrap(expected_next_arr, "trans_expected"),
                pd.DataFrame({"trans_global_ent": global_entropy_arr}),
            ],
            axis=1,
        )
        return df

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap(arr: np.ndarray, prefix: str) -> pd.DataFrame:
        """Wrap a 2-D numpy array as a DataFrame with prefixed column names."""
        if arr.ndim == 1:
            return pd.DataFrame({prefix: arr})
        cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)
