"""Leakage-safe metrics for the Nesso binary-head correction study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROXY_THRESHOLDS = (4.0, 5.0, 6.0)
RELEASED_BINDER_CUTOFF = 0.5


def safe_logit(probability: Sequence[float], epsilon: float = 1e-6) -> np.ndarray:
    """Return a finite logit after explicit probability clipping."""

    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and one half")
    probability_array = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must be finite")
    if np.any((probability_array < 0) | (probability_array > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    clipped = np.clip(probability_array, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def functional_potency_proxy(
    observed_pec50: Sequence[float], threshold: float
) -> np.ndarray:
    """Label functional-potency proxy positives at a prespecified pEC50 cutoff.

    These labels are not physical binding labels. In particular, a value below
    the threshold means only "not potent at this threshold."
    """

    if threshold not in PROXY_THRESHOLDS:
        raise ValueError(
            f"threshold {threshold} was not prespecified: {PROXY_THRESHOLDS}"
        )
    observed = np.asarray(observed_pec50, dtype=np.float64)
    if observed.ndim != 1 or not np.isfinite(observed).all():
        raise ValueError("pEC50 observations must be finite and one-dimensional")
    return observed >= threshold


def functional_proxy_metrics(
    observed_pec50: Sequence[float],
    binder_probability: Sequence[float],
    threshold: float,
    *,
    binder_cutoff: float = RELEASED_BINDER_CUTOFF,
) -> dict[str, float]:
    """Score binder probability against a functional-potency proxy label."""

    labels = functional_potency_proxy(observed_pec50, threshold)
    probability = np.asarray(binder_probability, dtype=np.float64)
    if labels.shape != probability.shape:
        raise ValueError("pEC50 and binder probability must be aligned")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("binder probabilities must be finite and in [0, 1]")
    predicted_binder = probability >= binder_cutoff
    proxy_positive = int(labels.sum())
    proxy_negative = int((~labels).sum())
    positive_binder = int(np.sum(labels & predicted_binder))
    positive_not_binder = int(np.sum(labels & ~predicted_binder))
    negative_binder = int(np.sum(~labels & predicted_binder))
    negative_not_binder = int(np.sum(~labels & ~predicted_binder))
    has_two_classes = proxy_positive > 0 and proxy_negative > 0
    return {
        "n": float(len(labels)),
        "threshold": float(threshold),
        "proxy_positive_count": float(proxy_positive),
        "proxy_positive_prevalence": float(np.mean(labels)),
        "roc_auc": (
            float(roc_auc_score(labels, probability))
            if has_two_classes
            else float("nan")
        ),
        "average_precision": (
            float(average_precision_score(labels, probability))
            if has_two_classes
            else float("nan")
        ),
        "brier_score": float(brier_score_loss(labels, probability)),
        "proxy_positive_binder_positive": float(positive_binder),
        "proxy_positive_binder_negative": float(positive_not_binder),
        "proxy_negative_binder_positive": float(negative_binder),
        "proxy_negative_binder_negative": float(negative_not_binder),
        "proxy_sensitivity_at_released_cutoff": (
            float(positive_binder / proxy_positive)
            if proxy_positive
            else float("nan")
        ),
        "proxy_specificity_at_released_cutoff": (
            float(negative_not_binder / proxy_negative)
            if proxy_negative
            else float("nan")
        ),
        "binder_positive_fraction_at_released_cutoff": float(
            np.mean(predicted_binder)
        ),
    }


def functional_proxy_curves(
    observed_pec50: Sequence[float],
    binder_probability: Sequence[float],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ROC and precision-recall curve points for a proxy threshold."""

    labels = functional_potency_proxy(observed_pec50, threshold)
    probability = np.asarray(binder_probability, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return pd.DataFrame(), pd.DataFrame()
    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
        labels, probability
    )
    precision, recall, pr_thresholds = precision_recall_curve(labels, probability)
    roc_frame = pd.DataFrame(
        {
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "score_threshold": roc_thresholds,
        }
    )
    pr_frame = pd.DataFrame(
        {
            "recall": recall,
            "precision": precision,
            "score_threshold": np.append(pr_thresholds, np.nan),
        }
    )
    return roc_frame, pr_frame


def family_bootstrap_proxy_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    observed_column: str = "pEC50",
    probability_column: str = "binder_probability",
    thresholds: Sequence[float] = PROXY_THRESHOLDS,
    replicates: int = 2000,
    seed: int = 20260813,
) -> pd.DataFrame:
    """Chemical-family bootstrap intervals for ROC-AUC and average precision."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if frame[group_column].isna().any():
        raise ValueError("bootstrap groups must not be missing")
    groups = frame[group_column].astype(str).to_numpy()
    unique_groups = pd.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique_groups}
    observed = frame[observed_column].to_numpy(dtype=float)
    probability = frame[probability_column].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        point = functional_proxy_metrics(observed, probability, float(threshold))
        draws = {
            "roc_auc": np.full(replicates, np.nan, dtype=float),
            "average_precision": np.full(replicates, np.nan, dtype=float),
        }
        for replicate in range(replicates):
            sampled_groups = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            sampled = np.concatenate([positions[group] for group in sampled_groups])
            metrics = functional_proxy_metrics(
                observed[sampled], probability[sampled], float(threshold)
            )
            for metric in draws:
                draws[metric][replicate] = metrics[metric]
        for metric, values in draws.items():
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "threshold": float(threshold),
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_lower": (
                        float(np.quantile(finite, 0.025)) if len(finite) else np.nan
                    ),
                    "ci_upper": (
                        float(np.quantile(finite, 0.975)) if len(finite) else np.nan
                    ),
                    "bootstrap_replicates": replicates,
                    "finite_replicates": len(finite),
                    "bootstrap_clusters": len(unique_groups),
                }
            )
    return pd.DataFrame(rows)


def assert_outer_predictions_isolated(
    prediction_rows: pd.DataFrame,
    *,
    row_column: str = "feature_row",
    fold_column: str = "fold",
    expected_rows: Sequence[int] | None = None,
) -> None:
    """Assert exactly one finite held-out prediction per development row."""

    if prediction_rows[row_column].duplicated().any():
        raise AssertionError("a development row has multiple outer predictions")
    if prediction_rows[fold_column].isna().any():
        raise AssertionError("outer predictions require an assigned fold")
    predicted_columns = [
        column for column in prediction_rows if column.endswith("_predicted_pEC50")
    ]
    if not predicted_columns:
        raise AssertionError("no prediction columns were found")
    if not np.isfinite(prediction_rows[predicted_columns].to_numpy()).all():
        raise AssertionError("outer predictions are incomplete")
    if expected_rows is not None and set(prediction_rows[row_column]) != set(
        expected_rows
    ):
        raise AssertionError("outer prediction coverage differs from expectation")


def paired_proxy_bootstrap_difference(
    frame: pd.DataFrame,
    models: Mapping[str, str],
    comparisons: Sequence[tuple[str, str]],
    *,
    threshold: float,
    group_column: str,
    observed_column: str = "pEC50",
    replicates: int = 2000,
    seed: int = 20260813,
) -> pd.DataFrame:
    """Paired family-bootstrap differences for proxy classification models."""

    groups = frame[group_column].astype(str).to_numpy()
    unique_groups = pd.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique_groups}
    observed = frame[observed_column].to_numpy(dtype=float)
    labels = functional_potency_proxy(observed, threshold)
    probabilities = {
        name: frame[column].to_numpy(dtype=float) for name, column in models.items()
    }
    point = {
        name: {
            "roc_auc": float(roc_auc_score(labels, values)),
            "average_precision": float(average_precision_score(labels, values)),
        }
        for name, values in probabilities.items()
    }
    rng = np.random.default_rng(seed)
    draws = {
        (name, metric): np.full(replicates, np.nan, dtype=float)
        for name in models
        for metric in ("roc_auc", "average_precision")
    }
    for replicate in range(replicates):
        sampled_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        sampled = np.concatenate([positions[group] for group in sampled_groups])
        sampled_labels = labels[sampled]
        if len(np.unique(sampled_labels)) < 2:
            continue
        for name, values in probabilities.items():
            draws[(name, "roc_auc")][replicate] = roc_auc_score(
                sampled_labels, values[sampled]
            )
            draws[(name, "average_precision")][replicate] = average_precision_score(
                sampled_labels, values[sampled]
            )
    rows: list[dict[str, Any]] = []
    for first, second in comparisons:
        for metric in ("roc_auc", "average_precision"):
            difference = draws[(first, metric)] - draws[(second, metric)]
            finite = difference[np.isfinite(difference)]
            rows.append(
                {
                    "threshold": threshold,
                    "comparison": f"{first}_minus_{second}",
                    "metric": metric,
                    "estimate": point[first][metric] - point[second][metric],
                    "ci_lower": float(np.quantile(finite, 0.025)),
                    "ci_upper": float(np.quantile(finite, 0.975)),
                    "bootstrap_replicates": replicates,
                    "bootstrap_clusters": len(unique_groups),
                }
            )
    return pd.DataFrame(rows)
