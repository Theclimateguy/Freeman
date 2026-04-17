"""Optional data-driven estimators for numeric causal edge weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from freeman_librarian.causal.result import EdgeKey, EstimationResult
from freeman_librarian.core.types import CausalEdge


def _optional_import(name: str) -> Any:
    """Import an optional dependency and return ``None`` when unavailable."""

    try:
        module = __import__(name, fromlist=["*"])
    except ImportError:
        return None
    return module


@dataclass
class _SingleEdgeEstimate:
    """Internal container for one estimated edge."""

    weight: float
    confidence_interval: tuple[float, float]
    backend: str
    metadata: Dict[str, Any]


class EdgeWeightEstimator:
    """Estimate numeric weights for an existing causal DAG from data.

    The estimator treats the DAG structure as fixed. For each edge, it estimates only
    the numeric magnitude of the treatment effect. Binary treatments use either a
    meta-learner-style T-learner or a forest-based backend. Continuous treatments fall
    back to a partial linear effect estimate.
    """

    SUPPORTED_MODELS = {"t_learner", "s_learner", "causal_forest"}

    def __init__(
        self,
        model: str = "t_learner",
        *,
        bootstrap_samples: int = 128,
        ci_level: float = 0.95,
        n_estimators: int = 128,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        normalized_model = str(model).strip().lower()
        if normalized_model not in self.SUPPORTED_MODELS:
            supported = ", ".join(sorted(self.SUPPORTED_MODELS))
            raise ValueError(f"Unsupported model '{model}'. Expected one of: {supported}")

        self.model = normalized_model
        self.bootstrap_samples = int(max(0, bootstrap_samples))
        self.ci_level = float(np.clip(ci_level, 0.5, 0.999))
        self.n_estimators = int(max(16, n_estimators))
        self.min_samples_leaf = int(max(1, min_samples_leaf))
        self.random_state = int(random_state)

    def fit(
        self,
        data: Any,
        edges: Iterable[EdgeKey | CausalEdge | Sequence[str]],
        *,
        treatment_col: str | None = None,
        outcome_col: str | None = None,
        covariate_cols: Sequence[str] | None = None,
    ) -> EstimationResult:
        """Estimate point weights and confidence intervals for ``edges``.

        When ``treatment_col`` and ``outcome_col`` are omitted, each edge is interpreted
        as ``(source_column, target_column)`` inside the provided tabular data.
        """

        pd = _optional_import("pandas")
        if pd is None:
            raise ImportError("EdgeWeightEstimator requires pandas. Install with `pip install \"freeman[causal]\"`.")

        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        edge_list = [self._normalize_edge(edge) for edge in edges]
        if not edge_list:
            return EstimationResult(weights={}, confidence_intervals={}, edge_metadata={}, backend=self.model)

        weights: Dict[EdgeKey, float] = {}
        intervals: Dict[EdgeKey, tuple[float, float]] = {}
        edge_metadata: Dict[EdgeKey, Dict[str, Any]] = {}
        resolved_backends: Dict[str, int] = {}

        for edge_key in edge_list:
            resolved_treatment = treatment_col or edge_key[0]
            resolved_outcome = outcome_col or edge_key[1]
            resolved_covariates = self._resolve_covariates(
                frame,
                treatment_col=resolved_treatment,
                outcome_col=resolved_outcome,
                covariate_cols=covariate_cols,
            )
            estimate = self._estimate_single_edge(
                frame,
                treatment_col=resolved_treatment,
                outcome_col=resolved_outcome,
                covariate_cols=resolved_covariates,
            )
            weights[edge_key] = estimate.weight
            intervals[edge_key] = estimate.confidence_interval
            edge_metadata[edge_key] = {
                **estimate.metadata,
                "treatment_col": resolved_treatment,
                "outcome_col": resolved_outcome,
                "covariate_cols": list(resolved_covariates),
                "requested_backend": self.model,
                "resolved_backend": estimate.backend,
            }
            resolved_backends[estimate.backend] = resolved_backends.get(estimate.backend, 0) + 1

        return EstimationResult(
            weights=weights,
            confidence_intervals=intervals,
            edge_metadata=edge_metadata,
            backend=self.model,
            metadata={
                "requested_backend": self.model,
                "resolved_backends": resolved_backends,
                "edge_count": len(edge_list),
                "bootstrap_samples": self.bootstrap_samples,
                "ci_level": self.ci_level,
            },
        )

    def _normalize_edge(self, edge: EdgeKey | CausalEdge | Sequence[str]) -> EdgeKey:
        """Coerce a supported edge representation into a ``(source, target)`` tuple."""

        if isinstance(edge, CausalEdge):
            return (edge.source, edge.target)
        if isinstance(edge, tuple) and len(edge) == 2:
            return (str(edge[0]), str(edge[1]))
        if isinstance(edge, list) and len(edge) == 2:
            return (str(edge[0]), str(edge[1]))
        raise TypeError(f"Unsupported edge specification: {edge!r}")

    def _resolve_covariates(
        self,
        frame: Any,
        *,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: Sequence[str] | None,
    ) -> List[str]:
        """Select numeric covariates, defaulting to all remaining numeric columns."""

        if covariate_cols is not None:
            return [str(column) for column in covariate_cols]

        numeric_columns: List[str] = []
        for column in frame.columns:
            if column in {treatment_col, outcome_col}:
                continue
            series = frame[column]
            if getattr(series, "dtype", None) is not None and np.issubdtype(series.dtype, np.number):
                numeric_columns.append(str(column))
        return numeric_columns

    def _estimate_single_edge(
        self,
        frame: Any,
        *,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: Sequence[str],
    ) -> _SingleEdgeEstimate:
        """Estimate one edge weight from a clean sub-frame."""

        required_columns = [treatment_col, outcome_col, *covariate_cols]
        clean = frame.loc[:, required_columns].dropna()
        if len(clean) < 12:
            raise ValueError(
                f"EdgeWeightEstimator requires at least 12 non-null rows for {treatment_col}->{outcome_col}; "
                f"received {len(clean)}."
            )

        treatment = clean[treatment_col].to_numpy(dtype=float)
        outcome = clean[outcome_col].to_numpy(dtype=float)
        covariates = (
            clean.loc[:, list(covariate_cols)].to_numpy(dtype=float)
            if covariate_cols
            else np.zeros((len(clean), 0), dtype=float)
        )

        if self._is_binary_treatment(treatment):
            if self.model == "causal_forest":
                point, backend = self._estimate_binary_causal_forest(covariates, treatment, outcome)
            elif self.model == "s_learner":
                point, backend = self._estimate_binary_s_learner(covariates, treatment, outcome)
            else:
                point, backend = self._estimate_binary_t_learner(covariates, treatment, outcome)
        else:
            point, backend = self._estimate_continuous_effect(covariates, treatment, outcome)

        interval = self._bootstrap_interval(
            covariates,
            treatment,
            outcome,
            is_binary=self._is_binary_treatment(treatment),
        )
        return _SingleEdgeEstimate(
            weight=point,
            confidence_interval=interval,
            backend=backend,
            metadata={
                "sample_size": int(len(clean)),
                "treatment_mean": float(np.mean(treatment)),
                "outcome_mean": float(np.mean(outcome)),
            },
        )

    def _bootstrap_interval(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        *,
        is_binary: bool,
    ) -> tuple[float, float]:
        """Estimate a percentile confidence interval by bootstrap resampling."""

        if self.bootstrap_samples <= 0:
            point = self._estimate_effect(covariates, treatment, outcome, is_binary=is_binary)[0]
            return (point, point)

        rng = np.random.default_rng(self.random_state)
        boot = []
        n_obs = len(treatment)
        for _ in range(self.bootstrap_samples):
            indices = rng.integers(0, n_obs, size=n_obs)
            sample_x = covariates[indices]
            sample_t = treatment[indices]
            sample_y = outcome[indices]
            if is_binary and len(np.unique(sample_t)) < 2:
                continue
            estimate, _ = self._estimate_effect(sample_x, sample_t, sample_y, is_binary=is_binary)
            boot.append(float(estimate))

        if not boot:
            point = self._estimate_effect(covariates, treatment, outcome, is_binary=is_binary)[0]
            return (point, point)

        alpha = (1.0 - self.ci_level) / 2.0
        return (
            float(np.quantile(boot, alpha)),
            float(np.quantile(boot, 1.0 - alpha)),
        )

    def _estimate_effect(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        *,
        is_binary: bool,
    ) -> tuple[float, str]:
        """Dispatch to the configured point-estimate backend."""

        if is_binary:
            if self.model == "causal_forest":
                return self._estimate_binary_causal_forest(covariates, treatment, outcome)
            if self.model == "s_learner":
                return self._estimate_binary_s_learner(covariates, treatment, outcome)
            return self._estimate_binary_t_learner(covariates, treatment, outcome)
        return self._estimate_continuous_effect(covariates, treatment, outcome)

    def _estimate_binary_t_learner(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> tuple[float, str]:
        """Estimate an average treatment effect with a T-learner."""

        sklearn_ensemble = _optional_import("sklearn.ensemble")
        sklearn_linear = _optional_import("sklearn.linear_model")
        if sklearn_ensemble is None or sklearn_linear is None:
            raise ImportError("t_learner backend requires scikit-learn. Install with `pip install \"freeman[causal]\"`.")

        treated_mask = treatment >= 0.5
        control_mask = ~treated_mask
        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            raise ValueError("Binary treatment estimation requires both treated and control observations.")

        if covariates.shape[1] == 0:
            effect = float(np.mean(outcome[treated_mask]) - np.mean(outcome[control_mask]))
            return effect, "mean_difference"

        treated_model = sklearn_ensemble.RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        control_model = sklearn_ensemble.RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state + 1,
        )
        treated_model.fit(covariates[treated_mask], outcome[treated_mask])
        control_model.fit(covariates[control_mask], outcome[control_mask])
        effect = float(np.mean(treated_model.predict(covariates) - control_model.predict(covariates)))
        return effect, "t_learner"

    def _estimate_binary_s_learner(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> tuple[float, str]:
        """Estimate an average treatment effect with a single learner."""

        sklearn_ensemble = _optional_import("sklearn.ensemble")
        if sklearn_ensemble is None:
            raise ImportError("s_learner backend requires scikit-learn. Install with `pip install \"freeman[causal]\"`.")

        features = np.column_stack([covariates, treatment.reshape(-1, 1)])
        model = sklearn_ensemble.RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        model.fit(features, outcome)

        treated_features = np.column_stack([covariates, np.ones((len(treatment), 1), dtype=float)])
        control_features = np.column_stack([covariates, np.zeros((len(treatment), 1), dtype=float)])
        effect = float(np.mean(model.predict(treated_features) - model.predict(control_features)))
        return effect, "s_learner"

    def _estimate_binary_causal_forest(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> tuple[float, str]:
        """Estimate an average treatment effect with CausalML or a forest fallback."""

        causalml_tree = _optional_import("causalml.inference.tree")
        if causalml_tree is not None:
            try:
                forest = causalml_tree.CausalRandomForestRegressor(random_state=self.random_state)
                forest.fit(X=covariates, treatment=treatment.astype(int), y=outcome)
                effect = np.asarray(forest.predict(covariates)).reshape(-1)
                return float(np.mean(effect)), "causalml_causal_forest"
            except Exception:
                pass

        return self._estimate_binary_s_learner(covariates, treatment, outcome)[0], "forest_fallback"

    def _estimate_continuous_effect(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
    ) -> tuple[float, str]:
        """Estimate a partial linear effect for continuous treatments."""

        sklearn_linear = _optional_import("sklearn.linear_model")
        if sklearn_linear is None:
            raise ImportError(
                "Continuous-treatment estimation requires scikit-learn. Install with `pip install \"freeman[causal]\"`."
            )

        features = np.column_stack([treatment.reshape(-1, 1), covariates])
        model = sklearn_linear.LinearRegression()
        model.fit(features, outcome)
        return float(model.coef_[0]), "partial_linear"

    def _is_binary_treatment(self, treatment: np.ndarray) -> bool:
        """Return whether a treatment vector is effectively binary."""

        unique = np.unique(np.asarray(treatment, dtype=float))
        return len(unique) <= 2 and set(np.round(unique, 8)).issubset({0.0, 1.0})

