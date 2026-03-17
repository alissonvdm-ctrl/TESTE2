import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.inspection import permutation_importance
import xgboost as xgb
from typing import Dict, Any, List, Optional, Tuple
import warnings


class TabularModelTrainer:
    """
    Trains multiple ML models on feature-engineered data.
    All models trained as multi-label classifiers (one per number 1..N).
    Validation always respects temporal order (no shuffling).
    """

    MODELS = {
        "logistic_regression": None,
        "random_forest": None,
        "extra_trees": None,
        "gradient_boosting": None,
        "xgboost": None,
    }

    def __init__(self, n_max: int, k_count: int,
                 cv_splits: int = 5, random_state: int = 42,
                 max_time_seconds: int = 300):
        """
        Initialize the trainer.

        Parameters
        ----------
        n_max : int
            Maximum number in the lottery pool (e.g. 60 for 1..60).
        k_count : int
            How many numbers are drawn per round.
        cv_splits : int
            Number of walk-forward folds for TimeSeriesSplit.
        random_state : int
            Seed for reproducibility.
        max_time_seconds : int
            Soft time budget per model (not enforced at the sklearn level,
            but passed to XGBoost).
        """
        self.n_max = n_max
        self.k_count = k_count
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.max_time_seconds = max_time_seconds

        # Trained model objects (after fit_all)
        self._fitted_models: Dict[str, Any] = {}
        # Per-model results dict produced by fit_model / walk_forward_validate
        self._results: Dict[str, Dict[str, Any]] = {}
        # Feature names stored at fit time
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pipelines(self) -> Dict[str, Any]:
        """Build fresh (unfitted) pipeline instances for every model."""
        rs = self.random_state

        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MultiOutputClassifier(
                LogisticRegression(
                    max_iter=1000,
                    C=0.1,
                    solver="lbfgs",
                    random_state=rs,
                    n_jobs=-1,
                ),
                n_jobs=-1,
            )),
        ])

        rf = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                random_state=rs,
                n_jobs=-1,
            ),
            n_jobs=-1,
        )

        et = MultiOutputClassifier(
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                random_state=rs,
                n_jobs=-1,
            ),
            n_jobs=-1,
        )

        gb = MultiOutputClassifier(
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=rs,
            ),
            n_jobs=-1,
        )

        # XGBoost natively supports multi-label via MultiOutputClassifier
        xgb_base = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=rs,
            n_jobs=-1,
        )
        xgboost_model = MultiOutputClassifier(xgb_base, n_jobs=-1)

        return {
            "logistic_regression": lr,
            "random_forest": rf,
            "extra_trees": et,
            "gradient_boosting": gb,
            "xgboost": xgboost_model,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_all(self, X: pd.DataFrame, y: np.ndarray,
                regime_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit all models with walk-forward validation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of shape (T, n_features).
        y : np.ndarray
            Binary multi-hot target matrix of shape (T, N), already
            shifted so that row t is the target for draw t+1.
        regime_labels : np.ndarray, optional
            Per-row regime / cluster labels (stored but not used in
            base training – reserved for future stratified analysis).

        Returns
        -------
        dict
            Mapping model_name -> result dict (metrics, importances …).
        """
        self._feature_names = list(X.columns)
        X_arr = X.values.astype(np.float32)
        y_arr = y.astype(np.float32)

        pipelines = self._build_pipelines()
        all_results: Dict[str, Any] = {}

        for name, model in pipelines.items():
            try:
                # Walk-forward CV to get validation metrics
                cv_metrics = self.walk_forward_validate(name, model, X_arr, y_arr)

                # Retrain on the full dataset for final predictions
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_arr, y_arr)

                self._fitted_models[name] = model

                # Compute train metrics on full data
                try:
                    proba_train = self._predict_proba_raw(model, X_arr)
                    train_overlap = self._overlap_metric(proba_train, y_arr, self.k_count)
                    train_ll = self._safe_log_loss(y_arr, proba_train)
                except Exception:
                    train_overlap = float("nan")
                    train_ll = float("nan")

                overfit = self.detect_overfitting(
                    train_overlap, cv_metrics.get("overlap", 0.0)
                )

                result = {
                    "cv_metrics": cv_metrics,
                    "train_overlap": train_overlap,
                    "train_log_loss": train_ll,
                    "overfit_detected": overfit,
                    "n_features": X_arr.shape[1],
                    "n_samples": X_arr.shape[0],
                }
                self._results[name] = result
                all_results[name] = result

            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"[TabularModelTrainer] {name} failed: {exc}")
                all_results[name] = {"error": str(exc)}

        return all_results

    def fit_model(self, name: str, model, X_train: np.ndarray,
                  y_train: np.ndarray, X_val: np.ndarray,
                  y_val: np.ndarray) -> Dict[str, Any]:
        """
        Fit a single model on (X_train, y_train) and evaluate on (X_val, y_val).

        Returns
        -------
        dict
            val_overlap, val_log_loss, val_brier, val_predictions.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        proba_val = self._predict_proba_raw(model, X_val)

        overlap = self._overlap_metric(proba_val, y_val, self.k_count)
        ll = self._safe_log_loss(y_val, proba_val)
        brier = self._safe_brier(y_val, proba_val)

        return {
            "val_overlap": overlap,
            "val_log_loss": ll,
            "val_brier": brier,
            "val_predictions": proba_val,
        }

    def walk_forward_validate(self, name: str, model, X: np.ndarray,
                               y: np.ndarray) -> Dict[str, float]:
        """
        TimeSeriesSplit walk-forward cross-validation.

        Returns
        -------
        dict
            Averaged metrics: overlap, log_loss, brier.
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        fold_overlaps: List[float] = []
        fold_lls: List[float] = []
        fold_briers: List[float] = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            # Rebuild a fresh copy to avoid contamination between folds
            fresh_pipelines = self._build_pipelines()
            fold_model = fresh_pipelines[name]

            try:
                metrics = self.fit_model(name, fold_model, X_tr, y_tr, X_vl, y_vl)
                fold_overlaps.append(metrics["val_overlap"])
                fold_lls.append(metrics["val_log_loss"])
                fold_briers.append(metrics["val_brier"])
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"[walk_forward_validate] fold failed for {name}: {exc}")

        def _mean(lst):
            return float(np.mean(lst)) if lst else float("nan")

        return {
            "overlap": _mean(fold_overlaps),
            "log_loss": _mean(fold_lls),
            "brier": _mean(fold_briers),
            "n_folds": len(fold_overlaps),
        }

    def get_feature_importance(self, model_name: str,
                                feature_names: List[str]) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances sorted descending.

        Tries tree-based feature_importances_ first; falls back to
        permutation importance on the training set if unavailable
        (e.g. for Logistic Regression).
        """
        if model_name not in self._fitted_models:
            raise ValueError(f"Model '{model_name}' has not been fitted yet.")

        model = self._fitted_models[model_name]
        importances = self._extract_importance(model, feature_names)

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Return an (N,) probability vector for the next draw using
        the named model.

        Parameters
        ----------
        X : np.ndarray
            A single feature row (1, n_features) or (n_features,).
        """
        if model_name not in self._fitted_models:
            raise ValueError(f"Model '{model_name}' has not been fitted yet.")

        model = self._fitted_models[model_name]
        X_in = np.atleast_2d(X).astype(np.float32)
        proba = self._predict_proba_raw(model, X_in)
        # proba shape: (1, N) -> (N,)
        return proba[0]

    def detect_overfitting(self, train_score: float,
                            val_score: float, threshold: float = 0.15) -> bool:
        """
        Return True when the relative gap between train and val overlap
        exceeds `threshold`.

        Gap = (train_score - val_score) / max(train_score, 1e-9)
        """
        if np.isnan(train_score) or np.isnan(val_score):
            return False
        gap = (train_score - val_score) / max(abs(train_score), 1e-9)
        return bool(gap > threshold)

    def rank_models(self) -> pd.DataFrame:
        """
        Return a DataFrame of all fitted models sorted by validation
        overlap score (descending).

        Columns: model_name, val_overlap, val_log_loss, val_brier,
                 train_overlap, overfit_detected.
        """
        rows = []
        for name, res in self._results.items():
            if "error" in res:
                continue
            cv = res.get("cv_metrics", {})
            rows.append({
                "model_name": name,
                "val_overlap": cv.get("overlap", float("nan")),
                "val_log_loss": cv.get("log_loss", float("nan")),
                "val_brier": cv.get("brier", float("nan")),
                "train_overlap": res.get("train_overlap", float("nan")),
                "overfit_detected": res.get("overfit_detected", False),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("val_overlap", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Private utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_proba_raw(model, X: np.ndarray) -> np.ndarray:
        """
        Extract probability estimates from a MultiOutputClassifier
        or Pipeline wrapping one.

        Returns shape (n_samples, n_outputs).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba_list = model.predict_proba(X)

        # MultiOutputClassifier returns a list of (n_samples, n_classes) arrays.
        # We want P(class=1) for each output.
        if isinstance(proba_list, list):
            # Each element: (n_samples, 2) – take column 1
            proba = np.stack([p[:, 1] for p in proba_list], axis=1)
        else:
            # Already (n_samples, n_outputs) – rare but handle gracefully
            proba = np.array(proba_list)

        return proba.astype(np.float32)

    @staticmethod
    def _overlap_metric(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
        """
        Average intersection size / k over all samples.

        For each sample, pick the top-k predicted numbers and compute
        |predicted ∩ actual| / k.
        """
        n_samples = proba.shape[0]
        overlaps = []
        for i in range(n_samples):
            top_k = set(np.argsort(proba[i])[-k:])
            actual = set(np.where(y_true[i] > 0.5)[0])
            if not actual:
                continue
            overlap = len(top_k & actual) / k
            overlaps.append(overlap)
        return float(np.mean(overlaps)) if overlaps else 0.0

    @staticmethod
    def _safe_log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
        """Compute average per-label log-loss, ignoring labels with no variance."""
        n_labels = y_true.shape[1]
        losses = []
        for j in range(n_labels):
            yt = y_true[:, j]
            yp = np.clip(proba[:, j], 1e-7, 1 - 1e-7)
            if len(np.unique(yt)) < 2:
                continue
            try:
                losses.append(log_loss(yt, yp))
            except Exception:
                pass
        return float(np.mean(losses)) if losses else float("nan")

    @staticmethod
    def _safe_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
        """Compute average per-label Brier score."""
        n_labels = y_true.shape[1]
        scores = []
        for j in range(n_labels):
            yt = y_true[:, j]
            yp = proba[:, j]
            try:
                scores.append(brier_score_loss(yt, yp))
            except Exception:
                pass
        return float(np.mean(scores)) if scores else float("nan")

    def _extract_importance(self, model, feature_names: List[str]) -> np.ndarray:
        """
        Extract mean feature importances from a fitted model.

        Strategy:
        1. If the model (or its final estimator) has `estimators_` from
           MultiOutputClassifier, average their feature_importances_.
        2. If it has coef_ (Logistic Regression), use mean |coef|.
        3. Otherwise return uniform importances.
        """
        # Unwrap Pipeline
        inner = model
        if hasattr(model, "named_steps"):
            inner = model.named_steps.get("clf", model)

        # MultiOutputClassifier -> average sub-estimator importances
        if hasattr(inner, "estimators_"):
            sub_imps = []
            for est in inner.estimators_:
                if hasattr(est, "feature_importances_"):
                    sub_imps.append(est.feature_importances_)
                elif hasattr(est, "coef_"):
                    sub_imps.append(np.abs(est.coef_).mean(axis=0))
            if sub_imps:
                imp = np.mean(sub_imps, axis=0)
                # Normalise
                total = imp.sum()
                return imp / total if total > 0 else imp

        # Direct feature_importances_
        if hasattr(inner, "feature_importances_"):
            imp = inner.feature_importances_
            return imp / imp.sum() if imp.sum() > 0 else imp

        # Direct coef_
        if hasattr(inner, "coef_"):
            imp = np.abs(inner.coef_).mean(axis=0)
            return imp / imp.sum() if imp.sum() > 0 else imp

        # Fallback: uniform
        n = len(feature_names)
        return np.ones(n) / n
