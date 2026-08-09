"""Shared statistics for publication-oriented PXR model comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def regression_metrics(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """Return interpretable regression and calibration statistics.

    Calibration uses ``observed = intercept + slope * predicted``. A calibrated
    model therefore has intercept zero and slope one.
    """

    y = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 1:
        raise ValueError("observed and predicted must be aligned one-dimensional arrays")
    if not len(y):
        return {
            key: float("nan")
            for key in (
                "n",
                "mae",
                "rmse",
                "rae",
                "pearson",
                "spearman",
                "bias",
                "calibration_intercept",
                "calibration_slope",
            )
        }
    if not (np.isfinite(y).all() and np.isfinite(pred).all()):
        raise ValueError("metrics require finite observations and predictions")

    residual = pred - y
    denominator = np.sum(np.abs(y - np.mean(y)))
    if np.std(pred) == 0:
        calibration_intercept = float(np.mean(y))
        calibration_slope = float("nan")
    else:
        design = np.column_stack([np.ones(len(pred)), pred])
        calibration_intercept, calibration_slope = np.linalg.lstsq(
            design,
            y,
            rcond=None,
        )[0]
    return {
        "n": float(len(y)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "rae": (
            float(np.sum(np.abs(residual)) / denominator)
            if denominator > 0
            else float("nan")
        ),
        "pearson": _correlation(y, pred),
        "spearman": _correlation(_rank(y), _rank(pred)),
        "bias": float(np.mean(residual)),
        "calibration_intercept": float(calibration_intercept),
        "calibration_slope": float(calibration_slope),
    }


def inverse_se_weights(
    standard_error: Sequence[float],
    curation_weight: Sequence[float] | None = None,
) -> np.ndarray:
    """Reproduce the established 2D branch's inverse-assay-SE weights."""

    se = np.asarray(standard_error, dtype=np.float64)
    finite_positive = np.isfinite(se) & (se > 0)
    if not finite_positive.any():
        raise ValueError("no finite positive standard errors")
    weights = np.empty_like(se)
    finite_weights = 1.0 / se[finite_positive]
    weights[finite_positive] = finite_weights
    weights[~finite_positive] = np.percentile(finite_weights, 99)
    if curation_weight is not None:
        curation = np.asarray(curation_weight, dtype=np.float64)
        if curation.shape != weights.shape:
            raise ValueError("curation weights are not aligned")
        weights *= curation
    return weights


def metric_rows(
    frame: pd.DataFrame,
    models: Mapping[str, str],
    *,
    dataset: str,
    observed_column: str = "pEC50",
    subset: str = "all",
) -> list[dict[str, Any]]:
    """Expand model metrics into a tidy table."""

    rows: list[dict[str, Any]] = []
    observed = frame[observed_column].to_numpy(dtype=float)
    for model, column in models.items():
        metrics = regression_metrics(observed, frame[column].to_numpy(dtype=float))
        rows.extend(
            {
                "dataset": dataset,
                "subset": subset,
                "model": model,
                "metric": metric,
                "estimate": value,
            }
            for metric, value in metrics.items()
        )
    return rows


def clustered_bootstrap(
    frame: pd.DataFrame,
    models: Mapping[str, str],
    *,
    group_column: str,
    observed_column: str = "pEC50",
    replicates: int = 2000,
    seed: int = 20260809,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Cluster-bootstrap metrics by resampling chemical families.

    Returns a tidy confidence-interval table and raw replicate arrays. The raw
    arrays allow paired model-difference intervals without another resampling
    pass.
    """

    if replicates < 1:
        raise ValueError("replicates must be positive")
    groups = frame[group_column].astype(str).to_numpy()
    unique_groups = pd.unique(groups)
    if not len(unique_groups):
        raise ValueError("no bootstrap groups")
    positions = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    metric_names = ("mae", "rmse", "rae", "pearson", "spearman", "bias")
    draws = {
        f"{model}:{metric}": np.empty(replicates, dtype=float)
        for model in models
        for metric in metric_names
    }
    for replicate in range(replicates):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled = np.concatenate([positions[group] for group in sampled_groups])
        observed = frame.iloc[sampled][observed_column].to_numpy(dtype=float)
        for model, column in models.items():
            metrics = regression_metrics(
                observed,
                frame.iloc[sampled][column].to_numpy(dtype=float),
            )
            for metric in metric_names:
                draws[f"{model}:{metric}"][replicate] = metrics[metric]

    rows: list[dict[str, Any]] = []
    observed = frame[observed_column].to_numpy(dtype=float)
    for model, column in models.items():
        point = regression_metrics(observed, frame[column].to_numpy(dtype=float))
        for metric in metric_names:
            values = draws[f"{model}:{metric}"]
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_lower": (
                        float(np.quantile(finite, 0.025)) if len(finite) else np.nan
                    ),
                    "ci_upper": (
                        float(np.quantile(finite, 0.975)) if len(finite) else np.nan
                    ),
                    "bootstrap_replicates": replicates,
                    "bootstrap_clusters": len(unique_groups),
                }
            )
    return pd.DataFrame(rows), draws


def paired_difference_rows(
    draws: Mapping[str, np.ndarray],
    point_metrics: Mapping[str, Mapping[str, float]],
    comparisons: Sequence[tuple[str, str]],
    *,
    metrics: Sequence[str] = ("mae", "rae", "spearman"),
) -> pd.DataFrame:
    """Summarize paired differences as first model minus second model."""

    rows: list[dict[str, Any]] = []
    for first, second in comparisons:
        for metric in metrics:
            difference = draws[f"{first}:{metric}"] - draws[f"{second}:{metric}"]
            finite = difference[np.isfinite(difference)]
            rows.append(
                {
                    "comparison": f"{first}_minus_{second}",
                    "metric": metric,
                    "estimate": (
                        point_metrics[first][metric] - point_metrics[second][metric]
                    ),
                    "ci_lower": (
                        float(np.quantile(finite, 0.025)) if len(finite) else np.nan
                    ),
                    "ci_upper": (
                        float(np.quantile(finite, 0.975)) if len(finite) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def error_complementarity(
    frame: pd.DataFrame,
    first_column: str,
    second_column: str,
    *,
    observed_column: str = "pEC50",
) -> dict[str, float]:
    """Quantify whether two models make the same compound-level mistakes."""

    observed = frame[observed_column].to_numpy(dtype=float)
    first = frame[first_column].to_numpy(dtype=float)
    second = frame[second_column].to_numpy(dtype=float)
    first_residual = first - observed
    second_residual = second - observed
    first_absolute = np.abs(first_residual)
    second_absolute = np.abs(second_residual)
    threshold_first = np.quantile(first_absolute, 0.9)
    threshold_second = np.quantile(second_absolute, 0.9)
    worst_first = first_absolute >= threshold_first
    worst_second = second_absolute >= threshold_second
    intersection = np.sum(worst_first & worst_second)
    union = np.sum(worst_first | worst_second)
    blend = (first + second) / 2.0
    return {
        "n": float(len(frame)),
        "signed_residual_pearson": _correlation(first_residual, second_residual),
        "signed_residual_spearman": _correlation(
            _rank(first_residual),
            _rank(second_residual),
        ),
        "absolute_error_pearson": _correlation(first_absolute, second_absolute),
        "absolute_error_spearman": _correlation(
            _rank(first_absolute),
            _rank(second_absolute),
        ),
        "first_lower_absolute_error_fraction": float(
            np.mean(first_absolute < second_absolute)
        ),
        "second_lower_absolute_error_fraction": float(
            np.mean(second_absolute < first_absolute)
        ),
        "exact_tie_fraction": float(np.mean(first_absolute == second_absolute)),
        "worst_decile_overlap_count": float(intersection),
        "worst_decile_jaccard": float(intersection / union) if union else np.nan,
        "oracle_mae": float(np.mean(np.minimum(first_absolute, second_absolute))),
        "equal_blend_mae": regression_metrics(observed, blend)["mae"],
        "equal_blend_spearman": regression_metrics(observed, blend)["spearman"],
    }
