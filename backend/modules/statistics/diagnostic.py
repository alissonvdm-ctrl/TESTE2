import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional

try:
    from statsmodels.stats.diagnostic import runs_test as sm_runs_test
    HAS_RUNS_TEST = True
except ImportError:
    HAS_RUNS_TEST = False

try:
    from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
    HAS_TSA = True
except ImportError:
    HAS_TSA = False

try:
    from statsmodels.stats.stattools import durbin_watson
    HAS_DW = True
except ImportError:
    HAS_DW = False

try:
    from sklearn.feature_selection import mutual_info_classif
    HAS_MI = True
except ImportError:
    HAS_MI = False

try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False


class StatisticalDiagnostic:
    """
    Comprehensive statistical analysis of numeric sequence data.

    Tests performed:
    1. Descriptive statistics per number
    2. Autocorrelation analysis (ACF/PACF)
    3. Mutual information between lags and positions
    4. Runs test (Wald-Wolfowitz)
    5. Serial dependency test
    6. Approximate entropy
    7. Permutation entropy
    8. Sample entropy
    9. Stationarity tests (ADF, KPSS)
    10. Distribution of gaps between appearances
    11. Overlap between consecutive samples
    12. Chi-squared test for uniformity
    13. Kolmogorov-Smirnov test
    """

    def __init__(self, n_max: int, k_count: int):
        self.n_max = n_max
        self.k_count = k_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_diagnostic(
        self, sequences: np.ndarray, multihot: np.ndarray
    ) -> Dict[str, Any]:
        """
        Full diagnostic pipeline.

        Parameters
        ----------
        sequences : np.ndarray, shape (n_samples, k_count)
            Raw drawn numbers per sample (1-indexed).
        multihot : np.ndarray, shape (n_samples, n_max)
            Binary indicator matrix: multihot[i, j-1] == 1 if j was drawn in sample i.

        Returns
        -------
        dict with keys:
            descriptive, temporal_memory, entropy, stationarity,
            uniformity, gap_analysis, mutual_information, regime,
            verdict, feature_importance_hints
        """
        results: Dict[str, Any] = {}

        results["descriptive"] = self._compute_descriptive(multihot)
        results["temporal_memory"] = self.test_temporal_memory(multihot)
        results["entropy"] = self.compute_entropy_measures(multihot)
        results["stationarity"] = self.test_stationarity(multihot)
        results["uniformity"] = self.test_uniformity(multihot)
        results["gap_analysis"] = self.analyze_gaps(sequences)
        results["mutual_information"] = self.compute_mutual_information(multihot)
        results["verdict"] = self.generate_verdict(results)
        results["feature_importance_hints"] = self._feature_importance_hints(results)

        return results

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    def _compute_descriptive(self, multihot: np.ndarray) -> Dict[str, Any]:
        """Basic stats per number and overall."""
        n_samples, n_max = multihot.shape
        freq = multihot.sum(axis=0)  # shape (n_max,)
        freq_rate = freq / n_samples

        per_number = {}
        for i in range(n_max):
            col = multihot[:, i].astype(float)
            per_number[i + 1] = {
                "count": int(freq[i]),
                "frequency_rate": float(freq_rate[i]),
                "mean": float(col.mean()),
                "std": float(col.std(ddof=1)) if n_samples > 1 else 0.0,
            }

        expected_rate = self.k_count / self.n_max
        return {
            "n_samples": n_samples,
            "n_max": n_max,
            "k_count": self.k_count,
            "expected_frequency_rate": expected_rate,
            "per_number": per_number,
            "overall_mean_freq_rate": float(freq_rate.mean()),
            "overall_std_freq_rate": float(freq_rate.std(ddof=1)) if n_max > 1 else 0.0,
        }

    # ------------------------------------------------------------------
    # Temporal memory / autocorrelation
    # ------------------------------------------------------------------

    def test_temporal_memory(self, multihot: np.ndarray) -> Dict[str, Any]:
        """Test if there is autocorrelation/memory in the sequences."""
        n_samples, n_max = multihot.shape
        result: Dict[str, Any] = {}

        # Aggregate signal: mean frequency per draw
        agg_signal = multihot.mean(axis=1)

        # ---------- ACF / PACF ----------
        acf_vals: Optional[np.ndarray] = None
        pacf_vals: Optional[np.ndarray] = None
        max_lag = min(40, n_samples // 4)

        if HAS_TSA and n_samples >= 20:
            try:
                acf_vals = acf(agg_signal, nlags=max_lag, fft=True)
                pacf_vals = pacf(agg_signal, nlags=max_lag)
                result["acf"] = acf_vals.tolist()
                result["pacf"] = pacf_vals.tolist()
                ci = 1.96 / np.sqrt(n_samples)
                result["acf_significant_lags"] = [
                    int(lag)
                    for lag in range(1, len(acf_vals))
                    if abs(acf_vals[lag]) > ci
                ]
            except Exception as exc:
                result["acf_error"] = str(exc)
        else:
            result["acf"] = None
            result["acf_significant_lags"] = []

        # ---------- Durbin-Watson ----------
        if HAS_DW and n_samples >= 4:
            try:
                dw_stat = durbin_watson(agg_signal)
                result["durbin_watson"] = float(dw_stat)
                # DW ~2 = no autocorrelation, <1.5 positive, >2.5 negative
                if dw_stat < 1.5:
                    result["durbin_watson_verdict"] = "positive_autocorrelation"
                elif dw_stat > 2.5:
                    result["durbin_watson_verdict"] = "negative_autocorrelation"
                else:
                    result["durbin_watson_verdict"] = "no_autocorrelation"
            except Exception as exc:
                result["durbin_watson_error"] = str(exc)

        # ---------- Runs test ----------
        result["runs_test"] = self._runs_test(agg_signal)

        # ---------- Ljung-Box (manual approximation) ----------
        if n_samples >= 20 and acf_vals is not None:
            result["ljung_box"] = self._ljung_box(agg_signal, acf_vals, max_lag=min(10, max_lag))

        # ---------- Per-number autocorrelation summary ----------
        sig_numbers: List[int] = []
        ci = 1.96 / np.sqrt(n_samples) if n_samples > 0 else 1.0
        if HAS_TSA and n_samples >= 20:
            for i in range(n_max):
                col = multihot[:, i].astype(float)
                if col.std() < 1e-10:
                    continue
                try:
                    col_acf = acf(col, nlags=5, fft=True)
                    if any(abs(col_acf[lag]) > ci for lag in range(1, len(col_acf))):
                        sig_numbers.append(i + 1)
                except Exception:
                    pass
        result["numbers_with_significant_acf"] = sig_numbers
        result["fraction_numbers_with_memory"] = (
            len(sig_numbers) / n_max if n_max > 0 else 0.0
        )

        return result

    def _runs_test(self, signal: np.ndarray) -> Dict[str, Any]:
        """Wald-Wolfowitz runs test (above/below median)."""
        if len(signal) < 4:
            return {"error": "insufficient data"}
        median = np.median(signal)
        binary = (signal > median).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            return {"error": "all values on one side of median"}

        if HAS_RUNS_TEST:
            try:
                stat, p_value = sm_runs_test(binary)
                return {"statistic": float(stat), "p_value": float(p_value),
                        "random": p_value > 0.05}
            except Exception:
                pass

        # Manual implementation
        n1 = int(binary.sum())
        n2 = int(len(binary) - n1)
        runs = 1 + np.sum(np.diff(binary) != 0)
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        var_runs = (
            (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2))
            / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        )
        if var_runs <= 0:
            return {"error": "zero variance in runs test"}
        z = (runs - expected_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return {
            "statistic": float(z),
            "p_value": float(p_value),
            "n_runs": int(runs),
            "expected_runs": float(expected_runs),
            "random": p_value > 0.05,
        }

    def _ljung_box(
        self, signal: np.ndarray, acf_vals: np.ndarray, max_lag: int = 10
    ) -> Dict[str, Any]:
        """Ljung-Box Q statistic."""
        n = len(signal)
        lags = range(1, min(max_lag + 1, len(acf_vals)))
        q = n * (n + 2) * sum(
            acf_vals[k] ** 2 / (n - k) for k in lags
        )
        df = len(list(lags))
        p_value = 1 - stats.chi2.cdf(q, df=df)
        return {
            "statistic": float(q),
            "p_value": float(p_value),
            "df": df,
            "significant": p_value < 0.05,
        }

    # ------------------------------------------------------------------
    # Uniformity tests
    # ------------------------------------------------------------------

    def test_uniformity(self, multihot: np.ndarray) -> Dict[str, Any]:
        """Chi-squared and KS tests to check if numbers appear uniformly."""
        n_samples, n_max = multihot.shape
        observed_counts = multihot.sum(axis=0)
        expected_count = n_samples * self.k_count / self.n_max
        expected_counts = np.full(n_max, expected_count)

        result: Dict[str, Any] = {}

        # ---------- Chi-squared ----------
        if expected_count >= 5:
            chi2_stat, chi2_p = stats.chisquare(observed_counts, f_exp=expected_counts)
            result["chi_squared"] = {
                "statistic": float(chi2_stat),
                "p_value": float(chi2_p),
                "df": n_max - 1,
                "uniform": chi2_p > 0.05,
            }
        else:
            result["chi_squared"] = {"error": "expected count < 5, test unreliable"}

        # ---------- Kolmogorov-Smirnov vs uniform ----------
        freq_rates = observed_counts / n_samples
        uniform_sample = np.linspace(0, 1, n_max)
        ks_stat, ks_p = stats.kstest(
            freq_rates / freq_rates.sum() if freq_rates.sum() > 0 else freq_rates,
            "uniform",
        )
        result["kolmogorov_smirnov"] = {
            "statistic": float(ks_stat),
            "p_value": float(ks_p),
            "uniform": ks_p > 0.05,
        }

        # ---------- Most and least frequent numbers ----------
        sorted_idx = np.argsort(observed_counts)
        result["most_frequent"] = [int(i + 1) for i in sorted_idx[-5:][::-1]]
        result["least_frequent"] = [int(i + 1) for i in sorted_idx[:5]]
        result["max_deviation_from_expected"] = float(
            np.max(np.abs(observed_counts - expected_count))
        )

        return result

    # ------------------------------------------------------------------
    # Entropy measures
    # ------------------------------------------------------------------

    def compute_entropy_measures(self, multihot: np.ndarray) -> Dict[str, float]:
        """Compute approximate entropy, permutation entropy, sample entropy."""
        n_samples, n_max = multihot.shape
        agg_signal = multihot.mean(axis=1)
        result: Dict[str, float] = {}

        if n_samples < 10:
            return {"error": "insufficient data for entropy measures"}

        # ---------- Shannon entropy of frequency distribution ----------
        freq = multihot.mean(axis=0)
        freq = np.clip(freq, 1e-10, None)
        freq /= freq.sum()
        result["shannon_entropy"] = float(-np.sum(freq * np.log2(freq)))
        result["max_shannon_entropy"] = float(np.log2(n_max))
        result["normalized_shannon"] = float(
            result["shannon_entropy"] / result["max_shannon_entropy"]
            if result["max_shannon_entropy"] > 0
            else 0.0
        )

        if HAS_ANTROPY:
            try:
                result["approximate_entropy"] = float(
                    ant.app_entropy(agg_signal)
                )
            except Exception:
                result["approximate_entropy"] = self._approx_entropy_manual(agg_signal)

            try:
                result["permutation_entropy"] = float(
                    ant.perm_entropy(agg_signal, normalize=True)
                )
            except Exception:
                result["permutation_entropy"] = self._perm_entropy_manual(agg_signal)

            try:
                result["sample_entropy"] = float(
                    ant.sample_entropy(agg_signal)
                )
            except Exception:
                result["sample_entropy"] = float("nan")

            try:
                result["spectral_entropy"] = float(
                    ant.spectral_entropy(agg_signal, sf=1.0, normalize=True)
                )
            except Exception:
                result["spectral_entropy"] = float("nan")
        else:
            result["approximate_entropy"] = self._approx_entropy_manual(agg_signal)
            result["permutation_entropy"] = self._perm_entropy_manual(agg_signal)
            result["sample_entropy"] = float("nan")

        return result

    def _approx_entropy_manual(
        self, signal: np.ndarray, m: int = 2, r_factor: float = 0.2
    ) -> float:
        """Manual approximate entropy implementation."""
        n = len(signal)
        if n < m + 2:
            return float("nan")
        r = r_factor * np.std(signal, ddof=1)
        if r == 0:
            return 0.0

        def phi(m_val: int) -> float:
            templates = np.array([signal[i : i + m_val] for i in range(n - m_val + 1)])
            count = np.sum(
                np.max(np.abs(templates[:, None] - templates[None, :]), axis=2) <= r,
                axis=1,
            )
            return float(np.mean(np.log(count / (n - m_val + 1))))

        try:
            return phi(m) - phi(m + 1)
        except Exception:
            return float("nan")

    def _perm_entropy_manual(self, signal: np.ndarray, order: int = 3) -> float:
        """Manual permutation entropy implementation."""
        n = len(signal)
        if n < order + 1:
            return float("nan")
        from math import factorial
        permutations: Dict[tuple, int] = {}
        for i in range(n - order + 1):
            pattern = tuple(np.argsort(signal[i : i + order]))
            permutations[pattern] = permutations.get(pattern, 0) + 1
        total = sum(permutations.values())
        probs = np.array([v / total for v in permutations.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(factorial(order))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # ------------------------------------------------------------------
    # Stationarity tests
    # ------------------------------------------------------------------

    def test_stationarity(self, multihot: np.ndarray) -> Dict[str, Any]:
        """ADF and KPSS tests on the aggregate frequency series."""
        n_samples = multihot.shape[0]
        agg_signal = multihot.mean(axis=1)
        result: Dict[str, Any] = {}

        if n_samples < 20:
            return {"error": "insufficient data for stationarity tests (need >= 20)"}

        if not HAS_TSA:
            return {"error": "statsmodels not available"}

        # ---------- ADF ----------
        try:
            adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(
                agg_signal, autolag="AIC"
            )
            result["adf"] = {
                "statistic": float(adf_stat),
                "p_value": float(adf_p),
                "n_lags": int(adf_lags),
                "critical_values": {k: float(v) for k, v in adf_crit.items()},
                "stationary": adf_p < 0.05,
            }
        except Exception as exc:
            result["adf"] = {"error": str(exc)}

        # ---------- KPSS ----------
        try:
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(agg_signal, regression="c")
            result["kpss"] = {
                "statistic": float(kpss_stat),
                "p_value": float(kpss_p),
                "n_lags": int(kpss_lags),
                "critical_values": {k: float(v) for k, v in kpss_crit.items()},
                # KPSS H0 is stationary; reject H0 (p<0.05) => non-stationary
                "stationary": kpss_p > 0.05,
            }
        except Exception as exc:
            result["kpss"] = {"error": str(exc)}

        # ---------- Combined verdict ----------
        adf_stat_flag = result.get("adf", {}).get("stationary", None)
        kpss_stat_flag = result.get("kpss", {}).get("stationary", None)

        if adf_stat_flag is True and kpss_stat_flag is True:
            result["combined_verdict"] = "stationary"
        elif adf_stat_flag is False and kpss_stat_flag is False:
            result["combined_verdict"] = "non_stationary"
        elif adf_stat_flag is True and kpss_stat_flag is False:
            result["combined_verdict"] = "trend_stationary"
        elif adf_stat_flag is False and kpss_stat_flag is True:
            result["combined_verdict"] = "difference_stationary"
        else:
            result["combined_verdict"] = "inconclusive"

        return result

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def analyze_gaps(self, sequences: np.ndarray) -> Dict[str, Any]:
        """Analyze gaps between consecutive appearances of each number."""
        n_samples, k_count = sequences.shape
        result: Dict[str, Any] = {"per_number": {}}

        all_gap_means: List[float] = []
        all_gap_stds: List[float] = []

        for num in range(1, self.n_max + 1):
            positions = np.where(
                np.any(sequences == num, axis=1)
            )[0]

            if len(positions) < 2:
                result["per_number"][num] = {
                    "appearances": int(len(positions)),
                    "gaps": [],
                    "mean_gap": float("nan"),
                    "std_gap": float("nan"),
                    "min_gap": float("nan"),
                    "max_gap": float("nan"),
                }
                continue

            gaps = np.diff(positions)
            mean_gap = float(gaps.mean())
            std_gap = float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0
            all_gap_means.append(mean_gap)
            all_gap_stds.append(std_gap)

            # Test if gaps follow geometric/exponential distribution
            geo_p = None
            ks_geo_p = None
            if len(gaps) >= 10:
                try:
                    geo_p = 1.0 / mean_gap if mean_gap > 0 else None
                    if geo_p is not None:
                        ks_stat, ks_geo_p = stats.kstest(
                            gaps, "geom", args=(geo_p,)
                        )
                except Exception:
                    pass

            result["per_number"][num] = {
                "appearances": int(len(positions)),
                "mean_gap": mean_gap,
                "std_gap": std_gap,
                "min_gap": int(gaps.min()),
                "max_gap": int(gaps.max()),
                "geometric_p_estimate": geo_p,
                "ks_vs_geometric_p": ks_geo_p,
            }

        # Overall summary
        expected_gap = self.n_max / self.k_count
        result["expected_mean_gap"] = expected_gap
        result["overall_mean_gap"] = float(np.mean(all_gap_means)) if all_gap_means else float("nan")
        result["overall_std_gap"] = float(np.mean(all_gap_stds)) if all_gap_stds else float("nan")

        # Consecutive overlap
        result["overlap_analysis"] = self._analyze_overlap(sequences)

        return result

    def _analyze_overlap(self, sequences: np.ndarray) -> Dict[str, Any]:
        """Analyze overlap between consecutive draws."""
        n_samples = sequences.shape[0]
        if n_samples < 2:
            return {"error": "insufficient samples"}

        overlaps = []
        for i in range(1, n_samples):
            prev_set = set(sequences[i - 1].tolist())
            curr_set = set(sequences[i].tolist())
            overlap = len(prev_set & curr_set)
            overlaps.append(overlap)

        overlaps_arr = np.array(overlaps, dtype=float)
        expected_overlap = self.k_count ** 2 / self.n_max

        return {
            "mean_overlap": float(overlaps_arr.mean()),
            "std_overlap": float(overlaps_arr.std(ddof=1)) if len(overlaps_arr) > 1 else 0.0,
            "expected_overlap": expected_overlap,
            "max_overlap": int(overlaps_arr.max()),
            "min_overlap": int(overlaps_arr.min()),
            "fraction_no_overlap": float((overlaps_arr == 0).mean()),
        }

    # ------------------------------------------------------------------
    # Mutual information
    # ------------------------------------------------------------------

    def compute_mutual_information(
        self, multihot: np.ndarray, max_lag: int = 10
    ) -> Dict[str, float]:
        """Mutual information between current and lagged multi-hot vectors."""
        n_samples, n_max = multihot.shape
        result: Dict[str, float] = {}

        if n_samples < max_lag + 10:
            return {"error": "insufficient data for mutual information"}

        if not HAS_MI:
            # Manual MI via histogram
            return self._manual_mutual_information(multihot, max_lag)

        for lag in range(1, max_lag + 1):
            X = multihot[: n_samples - lag]  # features: past
            y_matrix = multihot[lag:]  # targets: future
            mi_sum = 0.0
            count = 0
            for col_idx in range(n_max):
                y = y_matrix[:, col_idx]
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                try:
                    mi = mutual_info_classif(X, y, discrete_features=True, random_state=42)
                    mi_sum += float(mi.sum())
                    count += 1
                except Exception:
                    pass
            result[f"lag_{lag}"] = mi_sum / count if count > 0 else 0.0

        return result

    def _manual_mutual_information(
        self, multihot: np.ndarray, max_lag: int
    ) -> Dict[str, float]:
        """Approximate MI using joint entropy of aggregated signal."""
        n_samples = multihot.shape[0]
        agg = multihot.mean(axis=1)
        result: Dict[str, float] = {}

        for lag in range(1, max_lag + 1):
            x = agg[: n_samples - lag]
            y = agg[lag:]
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                result[f"lag_{lag}"] = 0.0
                continue
            # Use correlation as MI proxy
            corr = np.corrcoef(x, y)[0, 1]
            # MI lower bound for Gaussian: -0.5 * log(1 - corr^2)
            if abs(corr) < 1.0:
                mi = -0.5 * np.log(1 - corr ** 2 + 1e-10)
            else:
                mi = 10.0
            result[f"lag_{lag}"] = float(mi)

        return result

    # ------------------------------------------------------------------
    # Verdict generation
    # ------------------------------------------------------------------

    def generate_verdict(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate human-readable verdict with keys:
        - has_temporal_memory: yes/no/weak
        - dependency_type: linear/nonlinear/none
        - has_regime_change: yes/no/possible
        - most_informative_features: description
        - recommendation: model suggestions
        """
        verdict: Dict[str, str] = {}

        # ---- Temporal memory ----
        tm = results.get("temporal_memory", {})
        sig_lags = tm.get("acf_significant_lags", [])
        frac_mem = tm.get("fraction_numbers_with_memory", 0.0)
        dw_verdict = tm.get("durbin_watson_verdict", "")
        runs_random = tm.get("runs_test", {}).get("random", True)

        memory_score = 0
        if len(sig_lags) > 0:
            memory_score += 2
        if frac_mem > 0.3:
            memory_score += 1
        if dw_verdict in ("positive_autocorrelation", "negative_autocorrelation"):
            memory_score += 1
        if not runs_random:
            memory_score += 1

        if memory_score >= 4:
            verdict["has_temporal_memory"] = "yes"
        elif memory_score >= 2:
            verdict["has_temporal_memory"] = "weak"
        else:
            verdict["has_temporal_memory"] = "no"

        # ---- Dependency type ----
        mi = results.get("mutual_information", {})
        mi_vals = [v for k, v in mi.items() if k.startswith("lag_") and isinstance(v, float)]
        max_mi = max(mi_vals) if mi_vals else 0.0

        if verdict["has_temporal_memory"] == "no":
            verdict["dependency_type"] = "none"
        elif max_mi > 0.1 and len(sig_lags) > 0:
            verdict["dependency_type"] = "linear_and_nonlinear"
        elif len(sig_lags) > 0:
            verdict["dependency_type"] = "linear"
        elif max_mi > 0.05:
            verdict["dependency_type"] = "nonlinear"
        else:
            verdict["dependency_type"] = "none"

        # ---- Regime change ----
        stat = results.get("stationarity", {})
        combined = stat.get("combined_verdict", "inconclusive")
        if combined in ("non_stationary", "trend_stationary"):
            verdict["has_regime_change"] = "possible"
        elif combined == "difference_stationary":
            verdict["has_regime_change"] = "yes"
        else:
            verdict["has_regime_change"] = "no"

        # ---- Uniformity ----
        uni = results.get("uniformity", {})
        chi2_uniform = uni.get("chi_squared", {}).get("uniform", True)
        ks_uniform = uni.get("kolmogorov_smirnov", {}).get("uniform", True)
        if not chi2_uniform and not ks_uniform:
            verdict["distribution"] = "non_uniform"
        elif not chi2_uniform or not ks_uniform:
            verdict["distribution"] = "possibly_non_uniform"
        else:
            verdict["distribution"] = "uniform"

        # ---- Most informative features ----
        hints: List[str] = []
        if len(sig_lags) > 0:
            hints.append(f"lagged_draws (lags {sig_lags[:3]})")
        if max_mi > 0.05:
            hints.append("lagged_multihot_vectors")
        if verdict.get("has_regime_change") in ("yes", "possible"):
            hints.append("regime_label")
        gap_analysis = results.get("gap_analysis", {})
        if gap_analysis and not gap_analysis.get("error"):
            hints.append("gap_since_last_appearance")
        verdict["most_informative_features"] = ", ".join(hints) if hints else "none_identified"

        # ---- Recommendation ----
        mem = verdict["has_temporal_memory"]
        dep = verdict["dependency_type"]
        regime = verdict["has_regime_change"]

        if mem == "no" and dep == "none":
            verdict["recommendation"] = (
                "Data appears random. Baseline frequency models are appropriate. "
                "Complex models unlikely to outperform naive baselines."
            )
        elif dep in ("linear", "linear_and_nonlinear") and regime == "no":
            verdict["recommendation"] = (
                "Linear temporal dependency detected. ARIMA, logistic regression with "
                "lagged features, or GRU models recommended."
            )
        elif dep == "nonlinear" or regime in ("yes", "possible"):
            verdict["recommendation"] = (
                "Non-linear dependencies or regime changes detected. Gradient boosting "
                "(XGBoost/LightGBM) with lag features and regime labels, or LSTM models recommended. "
                "Consider separate models per regime."
            )
        else:
            verdict["recommendation"] = (
                "Weak temporal structure. Ensemble of frequency baseline and light ML "
                "(random forest with lag features) recommended."
            )

        return verdict

    # ------------------------------------------------------------------
    # Feature importance hints
    # ------------------------------------------------------------------

    def _feature_importance_hints(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize which feature groups appear most informative."""
        hints: Dict[str, Any] = {}

        tm = results.get("temporal_memory", {})
        sig_lags = tm.get("acf_significant_lags", [])
        hints["recommended_lag_depth"] = max(sig_lags) if sig_lags else 0

        mi = results.get("mutual_information", {})
        mi_vals = {k: v for k, v in mi.items() if k.startswith("lag_") and isinstance(v, float)}
        if mi_vals:
            best_lag = max(mi_vals, key=mi_vals.get)
            hints["best_mi_lag"] = best_lag
            hints["best_mi_value"] = mi_vals[best_lag]

        ent = results.get("entropy", {})
        perm_ent = ent.get("permutation_entropy", None)
        if perm_ent is not None and not (isinstance(perm_ent, float) and np.isnan(perm_ent)):
            hints["permutation_entropy"] = perm_ent
            hints["high_complexity"] = perm_ent > 0.85

        uni = results.get("uniformity", {})
        hints["non_uniform_distribution"] = not uni.get("chi_squared", {}).get("uniform", True)

        gap = results.get("gap_analysis", {})
        overlap = gap.get("overlap_analysis", {})
        if overlap and not overlap.get("error"):
            hints["mean_consecutive_overlap"] = overlap.get("mean_overlap")
            hints["expected_overlap"] = overlap.get("expected_overlap")

        return hints
