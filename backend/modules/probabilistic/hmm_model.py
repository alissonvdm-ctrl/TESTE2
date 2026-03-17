import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

try:
    from hmmlearn import hmm as hmmlearn_hmm
    _HMM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HMM_AVAILABLE = False

from sklearn.model_selection import TimeSeriesSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sequences_to_feature_matrix(
    sequences: np.ndarray,
    n_max: int,
) -> Tuple[np.ndarray, List[int]]:
    """Convert (n_draws, k) integer array to GaussianHMM-compatible format.

    Each draw is represented as a length-*n_max* multi-hot vector (0/1) cast
    to float64.  Returns:
      - X      : (n_draws, n_max) float64 feature matrix (one row per draw)
      - lengths: [n_draws]  (single contiguous sequence)
    """
    n_draws, _ = sequences.shape
    X = np.zeros((n_draws, n_max), dtype=np.float64)
    for i, row in enumerate(sequences):
        for num in row:
            idx = int(num) - 1
            if 0 <= idx < n_max:
                X[i, idx] = 1.0
    lengths = [n_draws]
    return X, lengths


def _split_temporal(
    X: np.ndarray,
    val_fraction: float = 0.15,
) -> Tuple[np.ndarray, List[int], np.ndarray, List[int]]:
    """Temporal train/val split (no shuffling)."""
    n = len(X)
    split = max(1, int(n * (1.0 - val_fraction)))
    X_train, X_val = X[:split], X[split:]
    return X_train, [len(X_train)], X_val, [len(X_val)]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HiddenMarkovAnalyzer:
    """
    Hidden Markov Model for numeric sequence analysis.

    Models each lottery/numeric draw as an emission from a hidden state.
    Uses GaussianHMM from hmmlearn with multi-hot feature vectors.
    Selects the optimal number of hidden states via BIC over a temporal
    train / validation split.

    Parameters
    ----------
    n_max : int
        Maximum possible number (e.g. 60 for a 1-60 lottery).
    k_count : int
        Numbers drawn per round.
    max_states : int
        Upper bound on the grid search for n_states (2 .. max_states).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_max: int,
        k_count: int,
        max_states: int = 10,
        random_state: int = 42,
    ) -> None:
        if not _HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required.  Install with: pip install hmmlearn"
            )
        self.n_max = n_max
        self.k_count = k_count
        self.max_states = max_states
        self.random_state = random_state

        self.best_model: Optional[hmmlearn_hmm.GaussianHMM] = None
        self.best_n_states: Optional[int] = None
        self.state_sequence: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._X_full: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, sequences: np.ndarray) -> Dict[str, Any]:
        """Fit HMM with automatic state selection.

        Parameters
        ----------
        sequences : np.ndarray
            2-D array of shape (n_draws, k_count) with integer numbers
            in [1, n_max].

        Returns
        -------
        dict
            ``n_states``        – chosen number of hidden states
            ``bic_curve``       – {n_states: bic_value} for all candidates
            ``aic_curve``       – {n_states: aic_value}
            ``log_likelihood``  – log-likelihood on the training set
            ``converged``       – whether the best model converged
        """
        X_full, _ = _sequences_to_feature_matrix(sequences, self.n_max)
        self._X_full = X_full

        X_train, lengths_train, _X_val, _lengths_val = _split_temporal(
            X_full, val_fraction=0.15
        )
        self._X_train = X_train

        best_n, curves = self.select_n_states(X_train, lengths_train)
        self.best_n_states = best_n

        # Refit on the full dataset using the best n_states
        best_model = self._fit_single(X_full, [len(X_full)], best_n)
        self.best_model = best_model

        # Decode full sequence
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, state_seq = best_model.decode(X_full, lengths=[len(X_full)])
                self.state_sequence = state_seq
            except Exception:
                self.state_sequence = np.zeros(len(X_full), dtype=int)

        self._fitted = True

        log_likelihood = float(
            best_model.score(X_full, lengths=[len(X_full)])
        )

        return {
            "n_states": best_n,
            "bic_curve": curves["bic"],
            "aic_curve": curves["aic"],
            "log_likelihood": log_likelihood,
            "converged": best_model.monitor_.converged,
        }

    def select_n_states(
        self, X: np.ndarray, lengths: List[int]
    ) -> Tuple[int, Dict]:
        """Grid-search 2..max_states and select by BIC (lower is better).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_draws, n_max).
        lengths : list[int]
            Sequence-length list passed to hmmlearn.

        Returns
        -------
        best_n : int
        curves : dict with keys 'bic' and 'aic', each mapping n -> value
        """
        bic_curve: Dict[int, float] = {}
        aic_curve: Dict[int, float] = {}
        best_bic = np.inf
        best_n = 2

        n_draws = X.shape[0]

        for n_states in range(2, self.max_states + 1):
            # Skip if not enough data
            if n_draws < n_states * 2:
                break
            try:
                model = self._fit_single(X, lengths, n_states)
                ll = float(model.score(X, lengths=lengths))
                # Number of free parameters for GaussianHMM (full covariance)
                n_features = X.shape[1]
                # Transition matrix: n_states*(n_states-1), start probs: n_states-1
                # means: n_states*n_features, covars: n_states*n_features (diagonal)
                n_params = (
                    n_states * (n_states - 1)           # transition
                    + (n_states - 1)                    # start probs
                    + n_states * n_features             # means
                    + n_states * n_features             # diagonal covariance
                )
                bic = -2.0 * ll + n_params * np.log(n_draws)
                aic = -2.0 * ll + 2.0 * n_params
                bic_curve[n_states] = bic
                aic_curve[n_states] = aic
                if bic < best_bic:
                    best_bic = bic
                    best_n = n_states
            except Exception:
                # Model failed to converge for this n_states; skip
                continue

        if not bic_curve:
            # Fallback: use 2 states
            best_n = 2

        return best_n, {"bic": bic_curve, "aic": aic_curve}

    def predict_next_state(self) -> Tuple[int, np.ndarray]:
        """Predict the most likely next hidden state.

        Uses the last decoded state and the transition matrix to compute
        the distribution over next states.

        Returns
        -------
        next_state : int
            Most probable next hidden state index.
        state_probs : np.ndarray
            Probability vector of shape (n_states,) for the next state.
        """
        self._check_fitted()
        transmat = self.best_model.transmat_
        last_state = int(self.state_sequence[-1])
        state_probs = transmat[last_state].copy()
        next_state = int(np.argmax(state_probs))
        return next_state, state_probs

    def predict_proba(self) -> np.ndarray:
        """Return probability of each number 1..N appearing in the next draw.

        Combines the predicted next-state distribution with the emission
        means (treated as unnormalised number probabilities).

        Returns
        -------
        probs : np.ndarray of shape (n_max,)
        """
        self._check_fitted()
        _, state_probs = self.predict_next_state()
        # means shape: (n_states, n_max)
        means = self.best_model.means_
        # Weighted sum of emission means
        combined = np.dot(state_probs, means)  # (n_max,)
        combined = np.clip(combined, 0.0, None)
        total = combined.sum()
        if total == 0.0:
            return np.ones(self.n_max) / self.n_max
        return combined / total

    def decode_states(self, sequences: np.ndarray) -> np.ndarray:
        """Return most likely hidden-state sequence via Viterbi decoding.

        Parameters
        ----------
        sequences : np.ndarray
            2-D array of shape (n_draws, k_count).

        Returns
        -------
        state_seq : np.ndarray of shape (n_draws,) with state indices.
        """
        self._check_fitted()
        X, lengths = _sequences_to_feature_matrix(sequences, self.n_max)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, state_seq = self.best_model.decode(X, lengths=lengths)
            except Exception:
                state_seq = np.zeros(len(X), dtype=int)
        return state_seq

    def get_emission_matrix(self) -> np.ndarray:
        """Return (n_states, n_max) emission probability matrix.

        Each row is the softmax of the Gaussian means for that state,
        which gives an approximate probability over numbers 1..N.
        """
        self._check_fitted()
        means = self.best_model.means_.copy()  # (n_states, n_max)
        # Softmax per state
        means = means - means.max(axis=1, keepdims=True)
        exp_means = np.exp(means)
        emission = exp_means / exp_means.sum(axis=1, keepdims=True)
        return emission

    def get_transition_matrix(self) -> np.ndarray:
        """Return (n_states, n_states) transition probability matrix."""
        self._check_fitted()
        return self.best_model.transmat_.copy()

    def get_state_characteristics(self) -> List[Dict[str, Any]]:
        """Describe each hidden state.

        Returns
        -------
        list of dicts, one per state, each containing:
            ``state_index``     – int
            ``stationary_prob`` – stationary probability under the chain
            ``hot_numbers``     – top-5 most likely numbers (1-indexed)
            ``cold_numbers``    – bottom-5 least likely numbers (1-indexed)
            ``mean_vector``     – raw emission mean vector (n_max,)
            ``emission_probs``  – normalised emission probabilities (n_max,)
        """
        self._check_fitted()
        emission = self.get_emission_matrix()   # (n_states, n_max)
        transmat = self.best_model.transmat_

        # Stationary distribution via left eigenvector
        try:
            eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stat = np.abs(eigenvectors[:, idx])
            stat = stat / stat.sum()
        except np.linalg.LinAlgError:
            stat = np.ones(self.best_n_states) / self.best_n_states

        characteristics = []
        for s in range(self.best_n_states):
            em_row = emission[s]
            # Numbers are 1-indexed
            sorted_desc = np.argsort(em_row)[::-1]
            sorted_asc = np.argsort(em_row)
            top5 = (sorted_desc[:5] + 1).tolist()
            bottom5 = (sorted_asc[:5] + 1).tolist()
            characteristics.append(
                {
                    "state_index": s,
                    "stationary_prob": float(stat[s]),
                    "hot_numbers": top5,
                    "cold_numbers": bottom5,
                    "mean_vector": self.best_model.means_[s].tolist(),
                    "emission_probs": em_row.tolist(),
                }
            )
        return characteristics

    def compute_metrics(self, sequences: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics on held-out (or full) sequences.

        Parameters
        ----------
        sequences : np.ndarray
            2-D integer array of shape (n_draws, k_count).

        Returns
        -------
        dict with keys:
            ``log_likelihood``      – total log-likelihood
            ``log_likelihood_per_draw`` – per-draw log-likelihood
            ``mean_hit_rate``       – fraction of predicted top-k numbers
                                      that appear in the actual draw
            ``n_draws``             – number of draws evaluated
        """
        self._check_fitted()
        X, lengths = _sequences_to_feature_matrix(sequences, self.n_max)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                total_ll = float(self.best_model.score(X, lengths=lengths))
            except Exception:
                total_ll = float("-inf")

        n_draws = len(sequences)
        ll_per_draw = total_ll / n_draws if n_draws > 0 else float("-inf")

        # Hit-rate: for each draw, predict top-k and check overlap
        hit_rates = []
        for row in sequences:
            probs = self.predict_proba()
            top_k_indices = np.argsort(probs)[::-1][: self.k_count]
            top_k_numbers = set(top_k_indices + 1)
            actual_set = set(int(x) for x in row)
            hit = len(top_k_numbers & actual_set) / self.k_count
            hit_rates.append(hit)

        return {
            "log_likelihood": total_ll,
            "log_likelihood_per_draw": ll_per_draw,
            "mean_hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.0,
            "n_draws": n_draws,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_single(
        self,
        X: np.ndarray,
        lengths: List[int],
        n_states: int,
        n_iter: int = 200,
    ) -> "hmmlearn_hmm.GaussianHMM":
        """Fit a single GaussianHMM and return it."""
        model = hmmlearn_hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, lengths=lengths)
        return model

    def _check_fitted(self) -> None:
        if not self._fitted or self.best_model is None:
            raise RuntimeError(
                "HiddenMarkovAnalyzer has not been fitted yet.  Call fit() first."
            )
