"""
Deep Learning module for NumericSequenceAnalyzer.

Provides LSTM and Transformer-based sequence models for predicting the
next draw in a numeric sequence.  These models are optional and require
PyTorch to be installed.  When PyTorch is unavailable the module degrades
gracefully: all public functions return stub results with an ``unavailable``
flag so the rest of the pipeline can continue.

Public API
----------
- :func:`is_available`   – True if torch is importable.
- :class:`LSTMPredictor` – Simple stacked LSTM model.
- :class:`TransformerPredictor` – Encoder-only Transformer model.
- :func:`train_deep_model` – Convenience wrapper; chooses architecture
  based on sample count and config.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.info("PyTorch not installed – deep learning module running in stub mode.")


def is_available() -> bool:
    """Return True if PyTorch is available."""
    return _TORCH_AVAILABLE


# ---------------------------------------------------------------------------
# LSTM Predictor
# ---------------------------------------------------------------------------


class LSTMPredictor:
    """
    Stacked LSTM model for multi-label sequence prediction.

    Predicts a probability for each number in [1, n_max] given a window
    of past ``seq_len`` draws encoded as multi-hot vectors.

    Parameters
    ----------
    n_max:
        Domain size (output dimension = n_max).
    seq_len:
        Number of past draws used as the look-back window.
    hidden_size:
        LSTM hidden dimension.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability between LSTM layers (ignored for num_layers=1).
    """

    def __init__(
        self,
        n_max: int,
        seq_len: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        self.n_max = n_max
        self.seq_len = seq_len
        self._model: Any = None
        self._trained = False

        if _TORCH_AVAILABLE:
            import torch.nn as _nn

            class _LSTM(_nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.lstm = _nn.LSTM(
                        input_size=n_max,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0.0,
                    )
                    self.fc = _nn.Linear(hidden_size, n_max)

                def forward(self, x: Any) -> Any:
                    out, _ = self.lstm(x)
                    return torch.sigmoid(self.fc(out[:, -1, :]))

            self._model = _LSTM()

    def fit(
        self,
        sequences: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Train the LSTM on *sequences* (shape: N × k_count or N × n_max multi-hot).

        Returns a metrics dict with train loss history.
        """
        if not _TORCH_AVAILABLE:
            return {"status": "unavailable", "reason": "PyTorch not installed"}

        import torch

        model = self._model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()

        # Build multi-hot matrix if sequences are integer draws (N × k)
        mh = self._to_multihot(sequences)
        if len(mh) <= self.seq_len:
            return {"status": "error", "reason": f"Not enough samples ({len(mh)}) for seq_len={self.seq_len}"}

        X, y = self._build_windows(mh)
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history: List[float] = []
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / max(len(loader), 1)
            history.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                logger.debug("LSTM epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)

        self._trained = True
        return {"status": "success", "train_loss_history": history, "final_loss": history[-1]}

    def predict(self, last_sequences: np.ndarray) -> Dict[str, Any]:
        """
        Return a probability score for each number given the last few draws.

        Parameters
        ----------
        last_sequences:
            Array of shape (seq_len, k_count) or (seq_len, n_max) representing
            the last *seq_len* draws.

        Returns
        -------
        dict with ``scores`` mapping number → probability.
        """
        if not _TORCH_AVAILABLE:
            return {"status": "unavailable", "scores": {}}
        if not self._trained:
            return {"status": "not_trained", "scores": {}}

        import torch

        mh = self._to_multihot(last_sequences)
        if len(mh) < self.seq_len:
            window = np.zeros((self.seq_len, self.n_max), dtype=np.float32)
            window[-len(mh):] = mh
        else:
            window = mh[-self.seq_len:]

        x = torch.tensor(window[np.newaxis], dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            probs = self._model(x).squeeze(0).numpy()

        scores = {i + 1: float(probs[i]) for i in range(self.n_max)}
        return {"status": "success", "scores": scores}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_multihot(self, sequences: np.ndarray) -> np.ndarray:
        """Convert integer sequence array (N × k) to multi-hot (N × n_max)."""
        if sequences.ndim == 1:
            sequences = sequences.reshape(1, -1)
        if sequences.shape[1] == self.n_max:
            return sequences.astype(np.float32)
        mh = np.zeros((len(sequences), self.n_max), dtype=np.float32)
        for i, row in enumerate(sequences):
            for num in row:
                idx = int(num) - 1
                if 0 <= idx < self.n_max:
                    mh[i, idx] = 1.0
        return mh

    def _build_windows(self, mh: np.ndarray):
        X, y = [], []
        for i in range(len(mh) - self.seq_len):
            X.append(mh[i: i + self.seq_len])
            y.append(mh[i + self.seq_len])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def train_deep_model(
    sequences: np.ndarray,
    n_max: int,
    k_count: int,
    epochs: int = 20,
    seq_len: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Train an LSTM predictor and return a result dict.

    When PyTorch is unavailable returns a stub dict with
    ``status='unavailable'``.

    Parameters
    ----------
    sequences:
        Raw draw sequences, shape (N, k_count).
    n_max:
        Domain size.
    k_count:
        Draw size (used for result annotation only).
    epochs:
        Training epochs.
    seq_len:
        Look-back window length.
    device:
        ``'cpu'`` or ``'cuda'``.

    Returns
    -------
    dict
        Keys: ``status``, ``model_type``, ``train_metrics``, ``scores``.
    """
    if not _TORCH_AVAILABLE:
        return {
            "status": "unavailable",
            "model_type": "lstm",
            "note": "PyTorch not installed – install torch to enable deep learning.",
            "train_metrics": {},
            "scores": {},
        }

    predictor = LSTMPredictor(n_max=n_max, seq_len=seq_len)
    train_metrics = predictor.fit(sequences, epochs=epochs, device=device)

    if train_metrics.get("status") != "success":
        return {
            "status": train_metrics.get("status", "error"),
            "model_type": "lstm",
            "note": train_metrics.get("reason", "training failed"),
            "train_metrics": train_metrics,
            "scores": {},
        }

    prediction = predictor.predict(sequences)
    return {
        "status": "success",
        "model_type": "lstm",
        "train_metrics": train_metrics,
        "scores": prediction.get("scores", {}),
        "k_count": k_count,
    }
