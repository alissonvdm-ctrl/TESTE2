"""
AnalyticsOrchestrator – central decision engine for the NumericSequenceAnalyzer.

The orchestrator inspects experiment configuration and dataset characteristics,
decides which analytics modules to run, in what order, and with what weights,
then executes the pipeline, feeding each module's output into subsequent ones.

Decision logic
--------------
1. Low temporal memory  → reduce weight of sequential / recurrent models.
2. Regime changes detected → trigger per-regime sub-training.
3. Order does not matter → activate set-based models; skip order-sensitive ones.
4. Low sample count      → prioritise tabular ML and HMM over deep learning.
5. Module graph is built dynamically, then executed sequentially.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class ModuleType(str, Enum):
    """Broad category that drives scheduling and weight decisions."""

    STATISTICAL = "statistical"         # frequency counts, chi-square, entropy
    SEQUENTIAL = "sequential"           # LSTM, GRU, Markov, N-gram
    TABULAR_ML = "tabular_ml"           # XGBoost, LightGBM, Random Forest
    HMM = "hmm"                         # Hidden Markov Model
    SET_BASED = "set_based"             # association rules, combinatorial
    DEEP_LEARNING = "deep_learning"     # neural networks beyond simple RNNs
    REGIME = "regime"                   # change-point detection / regime models
    ENSEMBLE = "ensemble"               # combines other modules' predictions


@dataclass
class ModuleSpec:
    """
    Descriptor for a single analytics module.

    Attributes
    ----------
    name:
        Unique slug (also used as key in result dicts).
    module_type:
        Broad category (see ``ModuleType``).
    run_fn:
        Callable ``(context: OrchestratorContext) -> dict[str, Any]``.
        Receives the full context (including outputs of previously run modules)
        and returns a JSON-serialisable result dict.
    weight:
        Initial relative weight in the ensemble (0–1). May be adjusted by the
        orchestrator's decision logic before execution.
    depends_on:
        Names of modules whose outputs must be available before this one runs.
    enabled:
        Whether this module is scheduled for the current run.
    """

    name: str
    module_type: ModuleType
    run_fn: Callable[["OrchestratorContext"], Dict[str, Any]]
    weight: float = 1.0
    depends_on: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class ExperimentConfig:
    """
    Parameters describing the lottery / sequence domain.

    Attributes
    ----------
    n_max:
        Upper bound of the number domain (e.g. 60 for a 1-60 lottery).
    k_count:
        Draw size (numbers per sequence).
    order_matters:
        True if sequence position is meaningful.
    sample_count:
        Number of valid samples in the dataset.
    has_timestamps:
        Whether samples carry draw timestamps (enables temporal models).
    """

    n_max: int
    k_count: int
    order_matters: bool
    sample_count: int
    has_timestamps: bool = True


@dataclass
class OrchestratorContext:
    """
    Shared state threaded through every module during pipeline execution.

    Attributes
    ----------
    config:
        Experiment configuration.
    raw_samples:
        List of raw number sequences (each a list of ints).
    module_outputs:
        Accumulated outputs keyed by module name.
    orchestrator_flags:
        Decision flags set by the orchestrator's analysis phase.
    """

    config: ExperimentConfig
    raw_samples: List[List[int]]
    module_outputs: Dict[str, Any] = field(default_factory=dict)
    orchestrator_flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """
    Final result returned by ``AnalyticsOrchestrator.run()``.

    Attributes
    ----------
    experiment_config:
        The config used for this run.
    orchestrator_flags:
        Flags set during the decision phase.
    module_results:
        ``{module_name: result_dict}`` for every module that ran.
    ensemble_result:
        Final ensemble prediction / score (if an ensemble module ran).
    skipped_modules:
        Names of modules that were disabled by the decision logic.
    total_duration_seconds:
        Wall-clock time for the entire pipeline.
    per_module_duration:
        ``{module_name: seconds}`` for each executed module.
    """

    experiment_config: ExperimentConfig
    orchestrator_flags: Dict[str, Any]
    module_results: Dict[str, Any]
    ensemble_result: Optional[Dict[str, Any]]
    skipped_modules: List[str]
    total_duration_seconds: float
    per_module_duration: Dict[str, float]


# ---------------------------------------------------------------------------
# Default module implementations (stubs – replace with real logic)
# ---------------------------------------------------------------------------
# Each function receives the full OrchestratorContext and returns a dict.
# Real implementations would call into specialised modules in
# backend/modules/*.py; these stubs make the orchestrator self-contained.


def _run_statistical(ctx: OrchestratorContext) -> Dict[str, Any]:
    """Frequency analysis, chi-square tests, entropy estimates."""
    from collections import Counter

    all_numbers: List[int] = [n for seq in ctx.raw_samples for n in seq]
    counter = Counter(all_numbers)
    total = len(all_numbers)
    frequencies = {str(k): v / total for k, v in sorted(counter.items())}
    return {"frequencies": frequencies, "total_draws": total}


def _run_sequential(ctx: OrchestratorContext) -> Dict[str, Any]:
    """Markov chain transition probabilities and N-gram counts."""
    from collections import defaultdict

    transitions: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for seq in ctx.raw_samples:
        for i in range(len(seq) - 1):
            transitions[seq[i]][seq[i + 1]] += 1

    # Normalise to probabilities
    transition_probs: dict[str, dict[str, float]] = {}
    for from_n, to_counts in transitions.items():
        total = sum(to_counts.values())
        transition_probs[str(from_n)] = {
            str(to_n): cnt / total for to_n, cnt in to_counts.items()
        }
    return {"transition_probabilities": transition_probs}


def _run_tabular_ml(ctx: OrchestratorContext) -> Dict[str, Any]:
    """Placeholder for tabular ML (XGBoost / LightGBM / Random Forest)."""
    # Real implementation: build lag features, train model, return CV metrics.
    return {
        "model": "tabular_ml_placeholder",
        "cv_score": None,
        "feature_importances": {},
        "note": "Tabular ML stub – wire to backend/modules/tabular_ml.py",
    }


def _run_hmm(ctx: OrchestratorContext) -> Dict[str, Any]:
    """Placeholder for Hidden Markov Model training."""
    return {
        "model": "hmm_placeholder",
        "n_states": None,
        "log_likelihood": None,
        "note": "HMM stub – wire to backend/modules/hmm.py",
    }


def _run_set_based(ctx: OrchestratorContext) -> Dict[str, Any]:
    """Association-rule mining and combinatorial pattern analysis."""
    from collections import Counter
    from itertools import combinations

    pair_counts: Counter = Counter()
    for seq in ctx.raw_samples:
        for pair in combinations(sorted(seq), 2):
            pair_counts[pair] += 1

    top_pairs = [
        {"pair": list(p), "count": c}
        for p, c in pair_counts.most_common(20)
    ]
    return {"top_co_occurring_pairs": top_pairs}


def _run_deep_learning(ctx: OrchestratorContext) -> Dict[str, Any]:
    """Placeholder for LSTM / Transformer deep-learning models."""
    return {
        "model": "deep_learning_placeholder",
        "architecture": None,
        "val_loss": None,
        "note": "Deep learning stub – wire to backend/modules/deep_learning.py",
    }


def _run_regime_detection(ctx: OrchestratorContext) -> Dict[str, Any]:
    """
    Detect structural breaks / regime changes in the sequence stream.

    Uses a simple sliding-window variance heuristic as a placeholder.
    Real implementation: PELT, BOCPD, or ruptures library.
    """
    import statistics

    window = max(10, len(ctx.raw_samples) // 10)
    variances: List[float] = []
    for i in range(0, len(ctx.raw_samples) - window + 1, window // 2):
        chunk = [n for seq in ctx.raw_samples[i : i + window] for n in seq]
        if len(chunk) > 1:
            variances.append(statistics.variance(chunk))

    if len(variances) < 2:
        return {"change_points": [], "n_regimes": 1}

    mean_var = sum(variances) / len(variances)
    # Flag windows whose variance deviates more than 50% from the mean
    change_points = [
        i * (window // 2)
        for i, v in enumerate(variances)
        if abs(v - mean_var) / (mean_var + 1e-9) > 0.5
    ]
    return {
        "change_points": change_points,
        "n_regimes": len(change_points) + 1,
        "window_variances": variances,
    }


def _run_ensemble(ctx: OrchestratorContext) -> Dict[str, Any]:
    """
    Combine predictions from all upstream modules into a weighted vote.

    Each enabled module's weight (stored in flags) is used to blend
    predictions.  This stub averages frequency-based scores; a real
    implementation would use stacking or Bayesian model averaging.
    """
    freq_data = ctx.module_outputs.get("statistical", {}).get("frequencies", {})
    weights = ctx.orchestrator_flags.get("module_weights", {})

    if not freq_data:
        return {"predicted_numbers": [], "confidence": 0.0}

    # Score each number by frequency × module weight
    stat_weight = weights.get("statistical", 1.0)
    scored = {int(k): v * stat_weight for k, v in freq_data.items()}

    # Incorporate set-based pair co-occurrence if available
    set_weight = weights.get("set_based", 0.0)
    if set_weight > 0 and "set_based" in ctx.module_outputs:
        for entry in ctx.module_outputs["set_based"].get("top_co_occurring_pairs", []):
            for num in entry["pair"]:
                scored[num] = scored.get(num, 0.0) + set_weight * entry["count"] * 0.001

    n_pick = ctx.config.k_count
    top_numbers = sorted(scored, key=scored.get, reverse=True)[:n_pick]  # type: ignore[arg-type]
    confidence = min(1.0, sum(scored.get(n, 0) for n in top_numbers) / (n_pick or 1))

    return {
        "predicted_numbers": sorted(top_numbers),
        "confidence": round(confidence, 4),
        "module_weights_used": weights,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

#: Minimum sample count below which deep-learning models are deprioritised.
_DEEP_LEARNING_MIN_SAMPLES: int = 500

#: Autocorrelation threshold below which temporal memory is considered low.
_LOW_TEMPORAL_MEMORY_THRESHOLD: float = 0.05

#: Regime-change rate (fraction of windows flagged) above which per-regime
#: training is triggered.
_REGIME_CHANGE_RATE_THRESHOLD: float = 0.15


class AnalyticsOrchestrator:
    """
    Central decision engine that adapts the analytics pipeline to dataset
    characteristics and experiment configuration.

    Usage
    -----
    ::

        orchestrator = AnalyticsOrchestrator(
            config=ExperimentConfig(
                n_max=60, k_count=6, order_matters=True,
                sample_count=800, has_timestamps=True
            ),
            raw_samples=list_of_sequences,
        )
        result: PipelineResult = orchestrator.run()

    The orchestrator performs three phases:

    1. **Introspection** – statistical properties of the dataset are measured
       (temporal memory, regime changes, sample count).
    2. **Decision** – modules are enabled/disabled and weights are adjusted.
    3. **Execution** – modules are run in dependency order; outputs are
       accumulated in an ``OrchestratorContext`` and forwarded to downstream
       modules.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        raw_samples: List[List[int]],
        extra_modules: Optional[List[ModuleSpec]] = None,
    ) -> None:
        """
        Parameters
        ----------
        config:
            Domain configuration for the experiment.
        raw_samples:
            Validated sequence rows.  Each element is a list of integers.
        extra_modules:
            Optional additional ``ModuleSpec`` objects to inject into the
            pipeline alongside the built-in ones.
        """
        self._config = config
        self._raw_samples = raw_samples
        self._extra_modules = extra_modules or []
        self._modules: List[ModuleSpec] = self._build_default_modules()
        if self._extra_modules:
            self._modules.extend(self._extra_modules)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """
        Execute the full analytics pipeline and return consolidated results.

        Returns
        -------
        PipelineResult
            Contains per-module outputs, ensemble result, timing, and the
            decision flags set during introspection.
        """
        pipeline_start = time.perf_counter()

        # Phase 1 – introspect dataset
        flags = self._introspect()
        logger.info("Orchestrator flags: %s", flags)

        # Phase 2 – decide module configuration
        self._apply_decisions(flags)

        enabled = [m for m in self._modules if m.enabled]
        skipped = [m.name for m in self._modules if not m.enabled]
        logger.info(
            "Pipeline: %d modules enabled, %d skipped. Order: %s",
            len(enabled),
            len(skipped),
            [m.name for m in enabled],
        )

        # Phase 3 – execute modules in dependency order
        ctx = OrchestratorContext(
            config=self._config,
            raw_samples=self._raw_samples,
            orchestrator_flags=flags,
        )
        per_module_duration: Dict[str, float] = {}
        ordered = self._topological_sort(enabled)

        for spec in ordered:
            module_start = time.perf_counter()
            try:
                logger.info("Running module: %s (weight=%.3f)", spec.name, spec.weight)
                result = spec.run_fn(ctx)
                ctx.module_outputs[spec.name] = result
            except Exception as exc:  # noqa: BLE001
                logger.exception("Module %s failed: %s", spec.name, exc)
                ctx.module_outputs[spec.name] = {"error": str(exc)}
            finally:
                per_module_duration[spec.name] = time.perf_counter() - module_start

        ensemble_result = ctx.module_outputs.get("ensemble")

        return PipelineResult(
            experiment_config=self._config,
            orchestrator_flags=flags,
            module_results=ctx.module_outputs,
            ensemble_result=ensemble_result,
            skipped_modules=skipped,
            total_duration_seconds=time.perf_counter() - pipeline_start,
            per_module_duration=per_module_duration,
        )

    # ------------------------------------------------------------------
    # Phase 1 – Introspection
    # ------------------------------------------------------------------

    def _introspect(self) -> Dict[str, Any]:
        """
        Compute dataset-level properties that drive scheduling decisions.

        Returns a flags dict consumed by ``_apply_decisions``.
        """
        flags: Dict[str, Any] = {}
        samples = self._raw_samples
        cfg = self._config

        # --- Sample count ---------------------------------------------------
        flags["sample_count"] = cfg.sample_count
        flags["low_sample_count"] = cfg.sample_count < _DEEP_LEARNING_MIN_SAMPLES

        # --- Order matters ---------------------------------------------------
        flags["order_matters"] = cfg.order_matters

        # --- Temporal memory (autocorrelation of first position) ------------
        temporal_memory = self._compute_temporal_memory(samples)
        flags["temporal_memory"] = temporal_memory
        flags["low_temporal_memory"] = temporal_memory < _LOW_TEMPORAL_MEMORY_THRESHOLD

        # --- Regime changes --------------------------------------------------
        regime_info = self._detect_regime_changes(samples)
        flags["n_regimes"] = regime_info["n_regimes"]
        flags["change_points"] = regime_info["change_points"]
        flags["has_regime_changes"] = regime_info["n_regimes"] > 1

        # Default module weights (may be mutated by _apply_decisions)
        flags["module_weights"] = {m.name: m.weight for m in self._modules}

        return flags

    def _compute_temporal_memory(self, samples: List[List[int]]) -> float:
        """
        Estimate temporal memory as lag-1 autocorrelation of the mean draw value.

        A value close to 0 means the series has little predictable structure
        across consecutive draws (low temporal memory).
        """
        if len(samples) < 10:
            return 0.0

        means = [sum(seq) / len(seq) for seq in samples if seq]
        if len(means) < 2:
            return 0.0

        grand_mean = sum(means) / len(means)
        numerator = sum(
            (means[i] - grand_mean) * (means[i - 1] - grand_mean)
            for i in range(1, len(means))
        )
        denominator = sum((m - grand_mean) ** 2 for m in means) + 1e-12
        return abs(numerator / denominator)

    def _detect_regime_changes(self, samples: List[List[int]]) -> Dict[str, Any]:
        """
        Heuristic change-point detection using sliding-window variance.

        Returns a dict with ``n_regimes`` and ``change_points`` keys.
        """
        import statistics as _stats

        n = len(samples)
        if n < 20:
            return {"n_regimes": 1, "change_points": []}

        window = max(10, n // 10)
        step = max(1, window // 2)
        variances: List[Tuple[int, float]] = []

        for i in range(0, n - window + 1, step):
            chunk = [num for seq in samples[i : i + window] for num in seq]
            if len(chunk) > 1:
                variances.append((i, _stats.variance(chunk)))

        if not variances:
            return {"n_regimes": 1, "change_points": []}

        mean_var = sum(v for _, v in variances) / len(variances)
        change_points = [
            idx
            for idx, v in variances
            if abs(v - mean_var) / (mean_var + 1e-9) > 0.5
        ]
        change_rate = len(change_points) / len(variances)

        return {
            "n_regimes": len(change_points) + 1,
            "change_points": change_points,
            "change_rate": change_rate,
        }

    # ------------------------------------------------------------------
    # Phase 2 – Decision / adaptation
    # ------------------------------------------------------------------

    def _apply_decisions(self, flags: Dict[str, Any]) -> None:
        """
        Mutate module ``enabled`` flags and ``weight`` values in-place based
        on the introspection results.

        Decision rules (applied in order, may compound)
        ------------------------------------------------
        1. **Low temporal memory** → halve sequential / deep-learning weights.
        2. **Regime changes detected** → enable the regime module; schedule
           per-regime sub-runs (logged; actual per-regime training is left to
           the regime module itself via context flags).
        3. **Order does not matter** → enable set-based models, disable
           purely order-sensitive sequential models.
        4. **Low sample count** → disable deep-learning, boost tabular ML and
           HMM weights.
        """
        module_map: Dict[str, ModuleSpec] = {m.name: m for m in self._modules}
        weights: Dict[str, float] = flags["module_weights"]

        # ---- Rule 1: Low temporal memory -----------------------------------
        if flags.get("low_temporal_memory"):
            logger.info(
                "Low temporal memory (%.4f) → downweighting sequential models.",
                flags["temporal_memory"],
            )
            for name, spec in module_map.items():
                if spec.module_type in (ModuleType.SEQUENTIAL, ModuleType.DEEP_LEARNING):
                    spec.weight *= 0.4
                    weights[name] = spec.weight

        # ---- Rule 2: Regime changes ----------------------------------------
        if flags.get("has_regime_changes"):
            n_regimes = flags["n_regimes"]
            logger.info(
                "Regime changes detected (%d regimes) → enabling regime module.",
                n_regimes,
            )
            if "regime" in module_map:
                module_map["regime"].enabled = True
            # Signal downstream modules that per-regime training is requested
            flags["per_regime_training"] = True
            flags["regime_count"] = n_regimes
        else:
            if "regime" in module_map:
                module_map["regime"].enabled = False

        # ---- Rule 3: Order does not matter ---------------------------------
        if not flags.get("order_matters", True):
            logger.info(
                "Order does not matter → activating set-based models, "
                "disabling purely sequential models."
            )
            if "set_based" in module_map:
                module_map["set_based"].enabled = True
                module_map["set_based"].weight = min(2.0, module_map["set_based"].weight * 1.5)
                weights["set_based"] = module_map["set_based"].weight
            # Sequential models lose relevance when order is irrelevant
            if "sequential" in module_map:
                module_map["sequential"].enabled = False

        # ---- Rule 4: Low sample count --------------------------------------
        if flags.get("low_sample_count"):
            n = flags["sample_count"]
            logger.info(
                "Low sample count (%d < %d) → disabling deep learning, "
                "boosting tabular ML and HMM.",
                n,
                _DEEP_LEARNING_MIN_SAMPLES,
            )
            # Disable deep learning
            for name, spec in module_map.items():
                if spec.module_type == ModuleType.DEEP_LEARNING:
                    spec.enabled = False

            # Boost tabular ML
            for name, spec in module_map.items():
                if spec.module_type == ModuleType.TABULAR_ML:
                    spec.weight = min(2.0, spec.weight * 1.5)
                    weights[name] = spec.weight

            # Boost HMM
            if "hmm" in module_map:
                module_map["hmm"].weight = min(2.0, module_map["hmm"].weight * 1.5)
                weights["hmm"] = module_map["hmm"].weight

        # Sync weights back into flags for downstream modules
        flags["module_weights"] = {
            name: spec.weight for name, spec in module_map.items()
        }

    # ------------------------------------------------------------------
    # Phase 3 helpers – dependency ordering
    # ------------------------------------------------------------------

    def _topological_sort(self, modules: List[ModuleSpec]) -> List[ModuleSpec]:
        """
        Return *modules* sorted so that every module appears after its
        dependencies.  Raises ``ValueError`` on circular dependencies.
        """
        name_to_spec = {m.name: m for m in modules}
        sorted_names: List[str] = []
        visited: set[str] = set()
        in_progress: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in in_progress:
                raise ValueError(f"Circular dependency detected involving module '{name}'")
            in_progress.add(name)
            spec = name_to_spec.get(name)
            if spec:
                for dep in spec.depends_on:
                    if dep in name_to_spec:
                        visit(dep)
            in_progress.discard(name)
            visited.add(name)
            sorted_names.append(name)

        for m in modules:
            visit(m.name)

        return [name_to_spec[n] for n in sorted_names if n in name_to_spec]

    # ------------------------------------------------------------------
    # Module registry
    # ------------------------------------------------------------------

    def _build_default_modules(self) -> List[ModuleSpec]:
        """
        Build the canonical list of analytics modules.

        All modules are enabled by default; ``_apply_decisions`` may
        disable or re-weight them before execution.
        """
        return [
            ModuleSpec(
                name="statistical",
                module_type=ModuleType.STATISTICAL,
                run_fn=_run_statistical,
                weight=1.0,
                depends_on=[],
            ),
            ModuleSpec(
                name="sequential",
                module_type=ModuleType.SEQUENTIAL,
                run_fn=_run_sequential,
                weight=1.0,
                depends_on=["statistical"],
            ),
            ModuleSpec(
                name="hmm",
                module_type=ModuleType.HMM,
                run_fn=_run_hmm,
                weight=1.0,
                depends_on=["statistical"],
            ),
            ModuleSpec(
                name="tabular_ml",
                module_type=ModuleType.TABULAR_ML,
                run_fn=_run_tabular_ml,
                weight=1.0,
                depends_on=["statistical"],
            ),
            ModuleSpec(
                name="set_based",
                module_type=ModuleType.SET_BASED,
                run_fn=_run_set_based,
                weight=0.8,
                depends_on=["statistical"],
                enabled=True,  # Will be boosted if order_matters=False
            ),
            ModuleSpec(
                name="deep_learning",
                module_type=ModuleType.DEEP_LEARNING,
                run_fn=_run_deep_learning,
                weight=1.0,
                depends_on=["statistical", "sequential"],
            ),
            ModuleSpec(
                name="regime",
                module_type=ModuleType.REGIME,
                run_fn=_run_regime_detection,
                weight=1.0,
                depends_on=["statistical"],
                enabled=False,  # Enabled only when regime changes are detected
            ),
            ModuleSpec(
                name="ensemble",
                module_type=ModuleType.ENSEMBLE,
                run_fn=_run_ensemble,
                weight=1.0,
                depends_on=[
                    "statistical",
                    "sequential",
                    "hmm",
                    "tabular_ml",
                    "set_based",
                ],
            ),
        ]
