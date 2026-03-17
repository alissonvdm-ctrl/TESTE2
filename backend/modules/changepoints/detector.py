import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

try:
    from sklearn.decomposition import PCA
    HAS_PCA = True
except ImportError:
    HAS_PCA = False


class RegimeDetector:
    """
    Detects structural breaks and regime changes in sequence data.
    Uses ruptures library with multiple algorithms.

    When ruptures is unavailable, falls back to a simple variance-based
    sliding window detector so the class remains functional.
    """

    MIN_SAMPLES = 20  # minimum samples required for detection

    def __init__(self, n_max: int):
        self.n_max = n_max
        self.breakpoints: List[int] = []
        self.n_regimes: int = 1
        self.regime_labels: Optional[np.ndarray] = None
        self._signal: Optional[np.ndarray] = None  # cached 1-D signal

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        multihot: np.ndarray,
        method: str = "auto",
        min_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Detect regime changes in the multi-hot matrix.

        Parameters
        ----------
        multihot : np.ndarray, shape (n_samples, n_max)
            Binary indicator matrix.
        method : str
            One of 'pelt', 'binseg', 'dynp', 'window', 'auto'.
            'auto' tries PELT first and falls back to binseg.
        min_size : int
            Minimum segment size (number of samples) for any regime.

        Returns
        -------
        dict with keys:
            breakpoints, n_regimes, regime_labels, regime_stats,
            method_used, signal_used, elbow_scores
        """
        n_samples = multihot.shape[0]

        if n_samples < self.MIN_SAMPLES:
            # Not enough data: single regime
            self.breakpoints = []
            self.n_regimes = 1
            self.regime_labels = np.zeros(n_samples, dtype=int)
            return self._build_result(
                multihot, method_used="none", note="insufficient data"
            )

        # Build 1-D signal for change-point detection
        signal = self._build_signal(multihot)
        self._signal = signal

        # Choose and run detection algorithm
        bkps: List[int] = []
        method_used = method

        if not HAS_RUPTURES:
            bkps = self._fallback_window_detector(signal, min_size)
            method_used = "fallback_window"
        else:
            if method == "auto":
                try:
                    bkps = self._detect_pelt(signal, min_size)
                    method_used = "pelt"
                except Exception:
                    try:
                        n_bkps = self._select_n_breakpoints(signal, min_size)
                        bkps = self._detect_binseg(signal, n_bkps, min_size)
                        method_used = "binseg"
                    except Exception:
                        bkps = []
                        method_used = "none"
            elif method == "pelt":
                bkps = self._detect_pelt(signal, min_size)
            elif method == "binseg":
                n_bkps = self._select_n_breakpoints(signal, min_size)
                bkps = self._detect_binseg(signal, n_bkps, min_size)
            elif method == "dynp":
                n_bkps = self._select_n_breakpoints(signal, min_size)
                bkps = self._detect_dynp(signal, n_bkps, min_size)
            elif method == "window":
                bkps = self._detect_window(signal, min_size)
            else:
                raise ValueError(f"Unknown method: {method!r}")

        # Remove trailing breakpoint == n_samples (ruptures convention)
        bkps = [b for b in bkps if b < n_samples]
        bkps = sorted(set(bkps))

        self.breakpoints = bkps
        self.n_regimes = len(bkps) + 1
        self.regime_labels = self._assign_labels(n_samples, bkps)

        return self._build_result(multihot, method_used=method_used)

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signal(self, multihot: np.ndarray) -> np.ndarray:
        """
        Reduce multi-hot matrix to a 1-D signal suitable for
        change-point detection.

        Strategy:
        1. If sklearn PCA available: use first principal component.
        2. Otherwise: use mean frequency per draw (row mean).
        """
        n_samples, n_max = multihot.shape
        signal_2d = multihot.astype(float)

        if HAS_PCA and n_samples >= 2:
            try:
                pca = PCA(n_components=1, random_state=42)
                signal = pca.fit_transform(signal_2d).ravel()
                return signal
            except Exception:
                pass

        # Fallback: mean frequency
        return signal_2d.mean(axis=1)

    # ------------------------------------------------------------------
    # Detection algorithms
    # ------------------------------------------------------------------

    def _detect_pelt(self, signal: np.ndarray, min_size: int = 10) -> List[int]:
        """PELT algorithm — optimal for an unknown number of breakpoints."""
        model = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
        model.fit(signal.reshape(-1, 1))
        # pen controls number of breakpoints; try a range and pick by elbow
        pen = max(1.0, np.log(len(signal)) * signal.var())
        result = model.predict(pen=pen)
        return result[:-1]  # remove trailing n_samples

    def _detect_binseg(
        self, signal: np.ndarray, n_bkps: int, min_size: int = 10
    ) -> List[int]:
        """Binary segmentation."""
        if n_bkps == 0:
            return []
        model = rpt.Binseg(model="rbf", min_size=min_size, jump=1)
        model.fit(signal.reshape(-1, 1))
        result = model.predict(n_bkps=n_bkps)
        return result[:-1]

    def _detect_dynp(
        self, signal: np.ndarray, n_bkps: int, min_size: int = 10
    ) -> List[int]:
        """Dynamic programming (optimal for fixed number of breakpoints)."""
        if n_bkps == 0:
            return []
        model = rpt.Dynp(model="rbf", min_size=min_size, jump=1)
        model.fit(signal.reshape(-1, 1))
        result = model.predict(n_bkps=n_bkps)
        return result[:-1]

    def _detect_window(self, signal: np.ndarray, min_size: int = 10) -> List[int]:
        """Sliding window detection."""
        width = max(min_size, len(signal) // 10)
        model = rpt.Window(model="rbf", width=width)
        model.fit(signal.reshape(-1, 1))
        pen = max(1.0, np.log(len(signal)) * signal.var())
        result = model.predict(pen=pen)
        return result[:-1]

    def _fallback_window_detector(
        self, signal: np.ndarray, min_size: int = 10
    ) -> List[int]:
        """
        Simple variance-ratio change-point detector used when ruptures is
        not installed.  Compares rolling variance across a sliding window.
        """
        n = len(signal)
        half = max(min_size, n // 10)
        scores = np.zeros(n)

        for i in range(half, n - half):
            left = signal[i - half : i]
            right = signal[i : i + half]
            var_l = np.var(left) + 1e-10
            var_r = np.var(right) + 1e-10
            mean_diff = abs(left.mean() - right.mean())
            scores[i] = mean_diff / np.sqrt((var_l + var_r) / 2)

        # Find local maxima above threshold
        threshold = np.percentile(scores[half : n - half], 90)
        bkps: List[int] = []
        last = -min_size * 2
        for i in range(half, n - half):
            if scores[i] >= threshold and i - last >= min_size:
                bkps.append(i)
                last = i
        return bkps

    # ------------------------------------------------------------------
    # Breakpoint count selection (elbow method)
    # ------------------------------------------------------------------

    def _select_n_breakpoints(
        self, signal: np.ndarray, min_size: int = 10, max_bkps: int = 10
    ) -> int:
        """
        Use the elbow method on the residual sum of squares from binseg
        to select the optimal number of breakpoints.
        """
        if not HAS_RUPTURES:
            return 0

        n = len(signal)
        max_bkps = min(max_bkps, n // min_size - 1)
        if max_bkps <= 0:
            return 0

        model = rpt.Binseg(model="rbf", min_size=min_size, jump=1)
        model.fit(signal.reshape(-1, 1))

        costs: List[float] = []
        for k in range(0, max_bkps + 1):
            try:
                if k == 0:
                    # Baseline: single segment cost
                    cost = float(np.sum((signal - signal.mean()) ** 2))
                else:
                    bkps = model.predict(n_bkps=k)
                    segments = [0] + bkps
                    cost = 0.0
                    for start, end in zip(segments[:-1], segments[1:]):
                        seg = signal[start:end]
                        cost += float(np.sum((seg - seg.mean()) ** 2))
                costs.append(cost)
            except Exception:
                costs.append(costs[-1] if costs else 0.0)

        # Elbow: largest second derivative
        if len(costs) < 3:
            return 0

        costs_arr = np.array(costs)
        diffs = np.diff(costs_arr)
        second_diffs = np.diff(diffs)
        elbow = int(np.argmax(np.abs(second_diffs))) + 1  # +1 because of double diff
        return max(0, min(elbow, max_bkps))

    # ------------------------------------------------------------------
    # Label assignment and result construction
    # ------------------------------------------------------------------

    def _assign_labels(self, n_samples: int, bkps: List[int]) -> np.ndarray:
        """Assign integer regime label to each sample."""
        labels = np.zeros(n_samples, dtype=int)
        boundaries = [0] + sorted(bkps) + [n_samples]
        for regime_id, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            labels[start:end] = regime_id
        return labels

    def _build_result(
        self,
        multihot: np.ndarray,
        method_used: str = "none",
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble the full result dictionary."""
        n_samples = multihot.shape[0]
        bkps = self.breakpoints
        labels = self.regime_labels

        result: Dict[str, Any] = {
            "breakpoints": bkps,
            "n_regimes": self.n_regimes,
            "regime_labels": labels.tolist() if labels is not None else [],
            "method_used": method_used,
            "n_samples": n_samples,
        }
        if note:
            result["note"] = note

        result["regime_stats"] = self._compute_regime_stats(multihot, labels)
        result["visualization_data"] = self.visualize_data()

        return result

    def _compute_regime_stats(
        self,
        multihot: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Compute statistics for each regime segment."""
        if labels is None:
            return []

        n_samples, n_max = multihot.shape
        regime_stats: List[Dict[str, Any]] = []

        for regime_id in range(self.n_regimes):
            mask = labels == regime_id
            segment = multihot[mask]
            indices = np.where(mask)[0]

            if len(segment) == 0:
                continue

            freq = segment.mean(axis=0)  # shape (n_max,)
            top5_idx = np.argsort(freq)[-5:][::-1]
            bot5_idx = np.argsort(freq)[:5]

            regime_stats.append(
                {
                    "regime_id": regime_id,
                    "start_idx": int(indices[0]),
                    "end_idx": int(indices[-1]),
                    "n_samples": int(len(segment)),
                    "mean_frequency_per_number": freq.tolist(),
                    "overall_mean_frequency": float(freq.mean()),
                    "overall_std_frequency": float(freq.std(ddof=1)) if n_max > 1 else 0.0,
                    "top5_numbers": [int(i + 1) for i in top5_idx],
                    "bottom5_numbers": [int(i + 1) for i in bot5_idx],
                    "entropy": self._segment_entropy(freq),
                }
            )

        return regime_stats

    @staticmethod
    def _segment_entropy(freq: np.ndarray) -> float:
        """Shannon entropy of frequency distribution within a segment."""
        p = np.clip(freq, 1e-10, None)
        p = p / p.sum()
        return float(-np.sum(p * np.log2(p)))

    # ------------------------------------------------------------------
    # Public utility methods
    # ------------------------------------------------------------------

    def get_regime_segments(self, sequences: np.ndarray) -> List[Dict[str, Any]]:
        """
        Return a list of segment descriptors, one per regime.

        Parameters
        ----------
        sequences : np.ndarray, shape (n_samples, k_count)
            Raw drawn numbers per sample.

        Returns
        -------
        List of dicts with keys:
            start_idx, end_idx, regime_id, n_samples, stats
        """
        if self.regime_labels is None:
            raise RuntimeError("Call detect() before get_regime_segments().")

        n_samples = sequences.shape[0]
        segments: List[Dict[str, Any]] = []

        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            seg_sequences = sequences[mask]
            flat = seg_sequences.ravel()
            unique, counts = np.unique(flat, return_counts=True)
            freq_dict = {int(u): int(c) for u, c in zip(unique, counts)}

            segments.append(
                {
                    "start_idx": int(indices[0]),
                    "end_idx": int(indices[-1]),
                    "regime_id": regime_id,
                    "n_samples": int(len(indices)),
                    "stats": {
                        "number_counts": freq_dict,
                        "most_frequent": int(unique[np.argmax(counts)]) if len(unique) > 0 else None,
                    },
                }
            )

        return segments

    def get_regime_feature(self, n_samples: int) -> np.ndarray:
        """
        Return integer array of regime label per sample.

        Parameters
        ----------
        n_samples : int
            Expected number of samples (used for validation).

        Returns
        -------
        np.ndarray, shape (n_samples,), dtype int
        """
        if self.regime_labels is None:
            return np.zeros(n_samples, dtype=int)
        if len(self.regime_labels) != n_samples:
            # Realign if sizes differ (edge case)
            labels = np.zeros(n_samples, dtype=int)
            copy_len = min(len(self.regime_labels), n_samples)
            labels[:copy_len] = self.regime_labels[:copy_len]
            return labels
        return self.regime_labels.copy()

    def split_by_regime(
        self, sequences: np.ndarray, features: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Split sequences and features by regime for separate model training.

        Parameters
        ----------
        sequences : np.ndarray, shape (n_samples, k_count)
        features : pd.DataFrame, shape (n_samples, n_features)

        Returns
        -------
        List of dicts, one per regime, with keys:
            regime_id, sequences, features, start_idx, end_idx, n_samples
        """
        if self.regime_labels is None:
            raise RuntimeError("Call detect() before split_by_regime().")

        splits: List[Dict[str, Any]] = []

        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            splits.append(
                {
                    "regime_id": regime_id,
                    "sequences": sequences[mask],
                    "features": features.iloc[mask] if len(features) == len(mask) else features,
                    "start_idx": int(indices[0]),
                    "end_idx": int(indices[-1]),
                    "n_samples": int(mask.sum()),
                }
            )

        return splits

    def visualize_data(self) -> Dict[str, Any]:
        """
        Return data suitable for rendering a breakpoint chart.

        Returns
        -------
        dict with keys:
            signal (1-D list), breakpoints, regime_colors, regime_spans
        """
        signal = self._signal.tolist() if self._signal is not None else []
        n = len(signal)
        breakpoints = self.breakpoints

        # Generate a distinct color per regime (as hex strings)
        palette = [
            "#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
            "#CCB974", "#64B5CD",
        ]
        n_regimes = self.n_regimes

        # Build spans: (start, end, regime_id, color)
        boundaries = [0] + sorted(breakpoints) + [n]
        regime_spans = []
        for regime_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            regime_spans.append(
                {
                    "start": start,
                    "end": end - 1,
                    "regime_id": regime_id,
                    "color": palette[regime_id % len(palette)],
                }
            )

        return {
            "signal": signal,
            "breakpoints": breakpoints,
            "n_regimes": n_regimes,
            "regime_spans": regime_spans,
            "regime_colors": [palette[i % len(palette)] for i in range(n_regimes)],
        }
