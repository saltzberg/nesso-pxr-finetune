"""Reusable statistics for Nesso representation and transfer ablations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nesso_pxr.train_cached import regression_metrics


@dataclass(frozen=True)
class RidgeProbeResult:
    """Predictions and fold-local choices from a nested ridge probe."""

    predictions: pd.DataFrame
    selections: pd.DataFrame


def huber_values(residual: np.ndarray, delta: float) -> np.ndarray:
    """Return element-wise Huber losses for signed residuals."""

    if delta <= 0:
        raise ValueError("delta must be positive")
    absolute = np.abs(np.asarray(residual, dtype=np.float64))
    return np.where(
        absolute <= delta,
        0.5 * np.square(absolute),
        delta * (absolute - 0.5 * delta),
    )


def mean_huber(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    delta: float,
) -> float:
    """Return mean Huber loss in the observed target's coordinates."""

    residual = np.asarray(predicted) - np.asarray(observed)
    return float(np.mean(huber_values(residual, delta)))


def _fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if alpha < 0:
        raise ValueError("ridge alpha must be nonnegative")
    estimator: Any
    if alpha == 0:
        estimator = LinearRegression()
    else:
        estimator = Ridge(alpha=alpha)
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=np.float64)


def select_grouped_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
) -> tuple[float, pd.DataFrame]:
    """Select ridge strength by pooled group-held-out MAE."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two groups are required")
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        predicted = np.full(len(y), np.nan, dtype=np.float64)
        for group in unique_groups:
            validation = groups == group
            training = ~validation
            predicted[validation] = _fit_ridge(
                x[training],
                y[training],
                x[validation],
                alpha=float(alpha),
            )
        if not np.isfinite(predicted).all():
            raise AssertionError("ridge inner predictions are incomplete")
        metrics = regression_metrics(y, predicted)
        rows.append({"alpha": float(alpha), **metrics})
    screen = pd.DataFrame(rows)
    best = screen.sort_values(
        ["mae", "spearman", "alpha"],
        ascending=[True, False, True],
        kind="stable",
    ).iloc[0]
    return float(best["alpha"]), screen


def nested_ridge_probe(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
) -> RidgeProbeResult:
    """Evaluate a ridge probe with group-nested hyperparameter selection."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError("x, y, and groups must have equal row counts")
    predicted = np.full(len(y), np.nan, dtype=np.float64)
    selection_rows: list[dict[str, float]] = []
    for outer_group in np.unique(groups):
        outer_validation = groups == outer_group
        outer_training = ~outer_validation
        selected_alpha, screen = select_grouped_ridge_alpha(
            x[outer_training],
            y[outer_training],
            groups[outer_training],
            alphas,
        )
        predicted[outer_validation] = _fit_ridge(
            x[outer_training],
            y[outer_training],
            x[outer_validation],
            alpha=selected_alpha,
        )
        selected_metrics = screen.loc[screen["alpha"].eq(selected_alpha)].iloc[0]
        selection_rows.append(
            {
                "outer_fold": int(outer_group),
                "selected_alpha": selected_alpha,
                "inner_mae": float(selected_metrics["mae"]),
                "inner_spearman": float(selected_metrics["spearman"]),
            }
        )
    if not np.isfinite(predicted).all():
        raise AssertionError("ridge outer predictions are incomplete")
    prediction_frame = pd.DataFrame(
        {
            "row_index": np.arange(len(y), dtype=int),
            "fold": groups,
            "observed": y,
            "predicted": predicted,
        }
    )
    return RidgeProbeResult(prediction_frame, pd.DataFrame(selection_rows))


def fit_full_ridge_after_group_selection(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    x_test: np.ndarray,
    alphas: Sequence[float],
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Select alpha on grouped development folds, then fit all development rows."""

    selected, screen = select_grouped_ridge_alpha(
        x_train,
        y_train,
        groups,
        alphas,
    )
    prediction = _fit_ridge(
        np.asarray(x_train),
        np.asarray(y_train),
        np.asarray(x_test),
        alpha=selected,
    )
    return prediction, selected, screen


def fit_convex_mae_weight(
    observed: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Fit ``w * first + (1-w) * second`` by MAE with ``0 <= w <= 1``."""

    from scipy.optimize import minimize_scalar

    observed = np.asarray(observed, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    def objective(weight: float) -> float:
        prediction = weight * first + (1.0 - weight) * second
        return float(np.mean(np.abs(prediction - observed)))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"convex blend optimization failed: {result.message}")
    return float(result.x)


def fit_linear_stack(
    observed: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Fit an intercept and two unconstrained model coefficients by least squares."""

    design = np.column_stack(
        [
            np.ones(len(observed), dtype=np.float64),
            np.asarray(first, dtype=np.float64),
            np.asarray(second, dtype=np.float64),
        ]
    )
    return np.linalg.lstsq(design, np.asarray(observed, dtype=np.float64), rcond=None)[
        0
    ]


def apply_linear_stack(
    coefficients: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Apply coefficients returned by :func:`fit_linear_stack`."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (3,):
        raise ValueError("linear stack requires intercept and two coefficients")
    return (
        coefficients[0]
        + coefficients[1] * np.asarray(first, dtype=np.float64)
        + coefficients[2] * np.asarray(second, dtype=np.float64)
    )


def crossfit_scalar_stacks(
    frame: pd.DataFrame,
    *,
    observed_column: str,
    first_column: str,
    second_column: str,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fit convex and unconstrained scalar stacks across chemical folds."""

    required = {observed_column, first_column, second_column, group_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing stack columns: {sorted(missing)}")
    result = frame.copy()
    result["convex_stack"] = np.nan
    result["linear_stack"] = np.nan
    rows: list[dict[str, float]] = []
    for group in sorted(frame[group_column].unique()):
        validation = frame[group_column].eq(group).to_numpy()
        training = ~validation
        observed = frame[observed_column].to_numpy(dtype=float)
        first = frame[first_column].to_numpy(dtype=float)
        second = frame[second_column].to_numpy(dtype=float)
        weight = fit_convex_mae_weight(
            observed[training], first[training], second[training]
        )
        coefficients = fit_linear_stack(
            observed[training], first[training], second[training]
        )
        result.loc[validation, "convex_stack"] = (
            weight * first[validation] + (1.0 - weight) * second[validation]
        )
        result.loc[validation, "linear_stack"] = apply_linear_stack(
            coefficients,
            first[validation],
            second[validation],
        )
        rows.append(
            {
                "held_out_fold": int(group),
                "convex_first_weight": weight,
                "convex_second_weight": 1.0 - weight,
                "linear_intercept": float(coefficients[0]),
                "linear_first_coefficient": float(coefficients[1]),
                "linear_second_coefficient": float(coefficients[2]),
            }
        )
    if result[["convex_stack", "linear_stack"]].isna().any().any():
        raise AssertionError("cross-fitted stack predictions are incomplete")
    return result, pd.DataFrame(rows)


def similarity_quantiles(
    frame: pd.DataFrame,
    *,
    similarity_column: str,
    group_columns: Iterable[str],
) -> pd.DataFrame:
    """Summarize nearest-neighbor similarity by one or more grouping columns."""

    groups = list(group_columns)
    rows: list[dict[str, Any]] = []
    for keys, subset in frame.groupby(groups, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = subset[similarity_column].to_numpy(dtype=float)
        row = dict(zip(groups, keys, strict=True))
        row.update(
            {
                "n": len(values),
                "minimum": float(np.min(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "maximum": float(np.max(values)),
                "fraction_ge_0_5": float(np.mean(values >= 0.5)),
                "fraction_ge_0_6": float(np.mean(values >= 0.6)),
                "fraction_ge_0_7": float(np.mean(values >= 0.7)),
                "fraction_ge_0_8": float(np.mean(values >= 0.8)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
