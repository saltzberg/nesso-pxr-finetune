#!/usr/bin/env python3
"""Run the CPU-only Nesso binary-readout correction analyses and figures."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from nesso_pxr.binary_head_reanalysis import (
    PROXY_THRESHOLDS,
    assert_outer_predictions_isolated,
    family_bootstrap_proxy_metrics,
    functional_proxy_curves,
    functional_proxy_metrics,
    paired_proxy_bootstrap_difference,
)
from nesso_pxr.comparison import (
    clustered_bootstrap,
    paired_difference_rows,
    regression_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "artifacts" / "experiments" / "binary_head_reanalysis"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "transfer_ablation" / "binary_head_reanalysis"
RIDGE_ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
BOOTSTRAP_REPLICATES = 2000
MLP_CONFIG = {
    "hidden_layer_sizes": (32,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-3,
    "batch_size": 64,
    "learning_rate_init": 1e-3,
    "max_iter": 100,
    "shuffle": True,
    "random_state": 42,
    "early_stopping": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metric_row(
    dataset: str,
    cohort: str,
    model: str,
    observed: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "cohort": cohort,
        "model": model,
        **regression_metrics(
            np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
        ),
    }


def _fit_ridge(
    x_training: np.ndarray,
    y_training: np.ndarray,
    x_evaluation: np.ndarray,
    alpha: float,
) -> np.ndarray:
    estimator = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(x_training, y_training)
    return np.asarray(model.predict(x_evaluation), dtype=float)


def _select_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for alpha in RIDGE_ALPHAS:
        prediction = np.full(len(y), np.nan, dtype=float)
        for fold in sorted(np.unique(folds)):
            held_out = folds == fold
            prediction[held_out] = _fit_ridge(
                x[~held_out], y[~held_out], x[held_out], alpha
            )
        rows.append({"alpha": alpha, **regression_metrics(y, prediction)})
    screen = pd.DataFrame(rows)
    selected = screen.sort_values(
        ["mae", "spearman", "alpha"],
        ascending=[True, False, True],
        kind="stable",
    ).iloc[0]
    return float(selected["alpha"]), screen


def _nested_ridge_probe(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(len(y), np.nan, dtype=float)
    selection_rows: list[dict[str, Any]] = []
    for outer_fold in sorted(np.unique(folds)):
        held_out = folds == outer_fold
        selected_alpha, screen = _select_ridge_alpha(
            x[~held_out], y[~held_out], folds[~held_out]
        )
        prediction[held_out] = _fit_ridge(
            x[~held_out], y[~held_out], x[held_out], selected_alpha
        )
        selected = screen.loc[screen["alpha"].eq(selected_alpha)].iloc[0]
        selection_rows.append(
            {
                "outer_fold": int(outer_fold),
                "selected_alpha": selected_alpha,
                "inner_mae": float(selected["mae"]),
                "inner_spearman": float(selected["spearman"]),
            }
        )
    if not np.isfinite(prediction).all():
        raise AssertionError("nested ridge predictions are incomplete")
    return prediction, pd.DataFrame(selection_rows)


def _full_ridge_prediction(
    x_development: np.ndarray,
    y_development: np.ndarray,
    folds: np.ndarray,
    x_evaluation: np.ndarray,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    selected_alpha, screen = _select_ridge_alpha(x_development, y_development, folds)
    return (
        _fit_ridge(x_development, y_development, x_evaluation, selected_alpha),
        selected_alpha,
        screen,
    )


def _nested_ridge_variant(
    name: str,
    x_development: np.ndarray,
    x_validation: np.ndarray,
    x_challenge: np.ndarray,
    development: pd.DataFrame,
    validation: pd.DataFrame,
    challenge: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    y_development = development["pEC50"].to_numpy(dtype=float)
    folds = development["fold"].to_numpy(dtype=int)
    oof, nested_selections = _nested_ridge_probe(x_development, y_development, folds)
    validation_prediction, selected_alpha, screen = _full_ridge_prediction(
        x_development,
        y_development,
        folds,
        x_validation,
    )
    challenge_prediction, challenge_alpha, _ = _full_ridge_prediction(
        x_development,
        y_development,
        folds,
        x_challenge,
    )
    if selected_alpha != challenge_alpha:
        raise AssertionError("full-development ridge selection changed by destination")
    selected_screen = screen.loc[screen["alpha"].eq(selected_alpha)].iloc[0]
    selection_rows = nested_selections.to_dict("records")
    for row in selection_rows:
        row["model"] = name
        row["selection_scope"] = "outer_training_only"
    selection_rows.append(
        {
            "model": name,
            "selection_scope": "all_development_for_external_evaluation",
            "outer_fold": np.nan,
            "selected_alpha": selected_alpha,
            "inner_mae": float(selected_screen["mae"]),
            "inner_spearman": float(selected_screen["spearman"]),
        }
    )
    return oof, validation_prediction, challenge_prediction, selection_rows


def _fixed_mlp_predictions(
    x_development: np.ndarray,
    x_validation: np.ndarray,
    x_challenge: np.ndarray,
    development: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = development["pEC50"].to_numpy(dtype=float)
    folds = development["fold"].to_numpy(dtype=int)
    oof = np.full(len(development), np.nan, dtype=float)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for fold in sorted(np.unique(folds)):
        held_out = folds == fold
        model = make_pipeline(StandardScaler(), MLPRegressor(**MLP_CONFIG))
        model.fit(x_development[~held_out], observed[~held_out])
        oof[held_out] = model.predict(x_development[held_out])
    full_model = make_pipeline(StandardScaler(), MLPRegressor(**MLP_CONFIG))
    full_model.fit(x_development, observed)
    return oof, full_model.predict(x_validation), full_model.predict(x_challenge)


def _activity_bin(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        values,
        bins=[-np.inf, 3.0, 4.0, 5.0, 6.0, np.inf],
        labels=["<3", "3–4", "4–5", "5–6", "≥6"],
        right=False,
    )


def _cohort_frames(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    challenge: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, str, str]]:
    return {
        "development_all": (development, "cluster_component", "family"),
        "development_emax_qualified": (
            development.loc[development["role"].eq("development_primary")].copy(),
            "cluster_component",
            "family",
        ),
        "development_lower_emax": (
            development.loc[development["role"].eq("excluded_initial_emax")].copy(),
            "cluster_component",
            "family",
        ),
        "validation_all": (validation, "audit_scaffold_group", "family"),
        "validation_emax_qualified": (
            validation.loc[validation["role"].eq("lockbox_primary")].copy(),
            "audit_scaffold_group",
            "family",
        ),
        "validation_lower_emax": (
            validation.loc[validation["role"].eq("excluded_initial_emax")].copy(),
            "audit_scaffold_group",
            "family",
        ),
        "challenge_all": (challenge, "record_id", "compound"),
    }


def _run_proxy_analysis(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    challenge: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_replicates: int,
) -> None:
    metric_rows: list[dict[str, Any]] = []
    interval_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    cohorts = _cohort_frames(development, validation, challenge)
    for cohort, (frame, group_column, bootstrap_unit) in cohorts.items():
        for threshold in PROXY_THRESHOLDS:
            row = functional_proxy_metrics(
                frame["pEC50"], frame["binder_probability"], threshold
            )
            row.update(
                {
                    "cohort": cohort,
                    "bootstrap_unit": bootstrap_unit,
                    "interpretation": "functional-potency proxy; not binding truth",
                }
            )
            metric_rows.append(row)
            roc_points, pr_points = functional_proxy_curves(
                frame["pEC50"], frame["binder_probability"], threshold
            )
            if not roc_points.empty:
                roc_points["cohort"] = cohort
                roc_points["threshold"] = threshold
                roc_points["curve"] = "roc"
                curve_frames.append(roc_points)
                pr_points["cohort"] = cohort
                pr_points["threshold"] = threshold
                pr_points["curve"] = "precision_recall"
                curve_frames.append(pr_points)
        intervals = family_bootstrap_proxy_metrics(
            frame,
            group_column=group_column,
            thresholds=PROXY_THRESHOLDS,
            replicates=bootstrap_replicates,
            seed=20260813,
        )
        intervals["cohort"] = cohort
        intervals["bootstrap_unit"] = bootstrap_unit
        interval_frames.append(intervals)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "proxy_classification_metrics.csv", index=False
    )
    pd.concat(interval_frames, ignore_index=True).to_csv(
        output_dir / "proxy_classification_bootstrap_intervals.csv", index=False
    )
    pd.concat(curve_frames, ignore_index=True).to_csv(
        output_dir / "proxy_classification_curve_points.csv", index=False
    )

    distribution_frames: list[pd.DataFrame] = []
    for cohort, frame in (
        ("development", development),
        ("validation", validation),
        ("challenge", challenge),
    ):
        binned = frame[["pEC50", "binder_probability"]].copy()
        binned["activity_bin"] = _activity_bin(binned["pEC50"])
        summary = (
            binned.groupby("activity_bin", observed=True)["binder_probability"]
            .agg(
                n="size",
                mean="mean",
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            )
            .reset_index()
        )
        summary["cohort"] = cohort
        distribution_frames.append(summary)
    pd.concat(distribution_frames, ignore_index=True).to_csv(
        output_dir / "binder_probability_by_activity_bin.csv", index=False
    )


def _plot_proxy_curves(output_dir: Path) -> None:
    curves = pd.read_csv(output_dir / "proxy_classification_curve_points.csv")
    metrics = pd.read_csv(output_dir / "proxy_classification_metrics.csv")
    colors = {"development_all": "#2f5d7c", "validation_all": "#b64c3b"}
    labels = {"development_all": "Development", "validation_all": "Validation"}
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.1), constrained_layout=True)
    for column, threshold in enumerate(PROXY_THRESHOLDS):
        roc_axis = axes[0, column]
        pr_axis = axes[1, column]
        for cohort in colors:
            subset = curves.loc[
                curves["cohort"].eq(cohort) & curves["threshold"].eq(threshold)
            ]
            roc = subset.loc[subset["curve"].eq("roc")]
            pr = subset.loc[subset["curve"].eq("precision_recall")]
            row = metrics.loc[
                metrics["cohort"].eq(cohort) & metrics["threshold"].eq(threshold)
            ].iloc[0]
            roc_axis.plot(
                roc["false_positive_rate"],
                roc["true_positive_rate"],
                color=colors[cohort],
                linewidth=2,
                label=f"{labels[cohort]} AUC {row['roc_auc']:.2f}",
            )
            pr_axis.plot(
                pr["recall"],
                pr["precision"],
                color=colors[cohort],
                linewidth=2,
                label=f"{labels[cohort]} AP {row['average_precision']:.2f}",
            )
            pr_axis.axhline(
                row["proxy_positive_prevalence"],
                color=colors[cohort],
                linestyle=":",
                alpha=0.6,
            )
        roc_axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
        roc_axis.set(
            title=f"pEC50 ≥ {threshold:g} proxy",
            xlabel="False-positive rate vs proxy",
            ylabel="True-positive rate vs proxy",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        pr_axis.set(
            xlabel="Recall vs proxy",
            ylabel="Precision vs proxy",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        roc_axis.legend(frameon=False, fontsize=8, loc="lower right")
        pr_axis.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Nesso binder probability against functional-potency proxies\n"
        "These are not physical binding labels.",
        fontsize=13,
    )
    fig.savefig(output_dir / "binder_proxy_roc_pr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_proxy_matrices(output_dir: Path) -> None:
    metrics = pd.read_csv(output_dir / "proxy_classification_metrics.csv")
    cohorts = ("development_all", "validation_all")
    titles = ("Development", "Validation")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    for row_index, (cohort, title) in enumerate(zip(cohorts, titles, strict=True)):
        for column, threshold in enumerate(PROXY_THRESHOLDS):
            values = metrics.loc[
                metrics["cohort"].eq(cohort) & metrics["threshold"].eq(threshold)
            ].iloc[0]
            matrix = np.asarray(
                [
                    [
                        values["proxy_positive_binder_positive"],
                        values["proxy_positive_binder_negative"],
                    ],
                    [
                        values["proxy_negative_binder_positive"],
                        values["proxy_negative_binder_negative"],
                    ],
                ]
            )
            axis = axes[row_index, column]
            axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
            for matrix_row in range(2):
                for matrix_column in range(2):
                    axis.text(
                        matrix_column,
                        matrix_row,
                        f"{int(matrix[matrix_row, matrix_column]):,}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="black",
                    )
            axis.set_xticks([0, 1], ["Nesso p ≥ 0.5", "Nesso p < 0.5"], rotation=20)
            axis.set_yticks(
                [0, 1],
                [f"pEC50 ≥ {threshold:g}", f"pEC50 < {threshold:g}"],
            )
            axis.set_title(f"{title}: threshold {threshold:g}")
    fig.suptitle(
        "Proxy contingency matrices at the released binder cutoff\n"
        "Below-threshold compounds are not labeled non-binders.",
        fontsize=13,
    )
    fig.savefig(
        output_dir / "binder_proxy_contingency_matrices.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_probability_distribution(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
) -> None:
    frames = []
    for cohort, source in (("Development", development), ("Validation", validation)):
        frame = source[["pEC50", "binder_probability"]].copy()
        frame["cohort"] = cohort
        frame["activity_bin"] = _activity_bin(frame["pEC50"])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    categories = ["<3", "3–4", "4–5", "5–6", "≥6"]
    fig, axes = plt.subplots(
        1, 2, figsize=(10.6, 4.2), constrained_layout=True, sharey=True
    )
    for axis, cohort in zip(axes, ("Development", "Validation"), strict=True):
        subset = combined.loc[combined["cohort"].eq(cohort)]
        values = [
            subset.loc[
                subset["activity_bin"].astype(str).eq(category), "binder_probability"
            ].to_numpy()
            for category in categories
        ]
        axis.boxplot(values, tick_labels=categories, showfliers=False, widths=0.65)
        axis.axhline(0.5, color="#b64c3b", linestyle="--", linewidth=1.2)
        axis.set(
            title=cohort,
            xlabel="Observed functional activity bin (pEC50)",
            ylabel="Nesso binder probability",
            ylim=(0, 1),
        )
    fig.suptitle("Binder probability rises only modestly with PXR functional potency")
    fig.savefig(
        output_dir / "binder_probability_by_activity.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def _add_existing_baselines(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    challenge: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nested = pd.read_csv(
        REPO_ROOT / "reports" / "model_comparison" / "nested_aligned_predictions.csv"
    )
    development = development.merge(
        nested[["feature_row", "nesso_pEC50", "two_d_pEC50"]].rename(
            columns={
                "nesso_pEC50": "fine_tuned_continuous_head_predicted_pEC50",
                "two_d_pEC50": "lightgbm_predicted_pEC50",
            }
        ),
        on="feature_row",
        how="left",
        validate="one_to_one",
    )
    nesso_validation = pd.read_csv(
        REPO_ROOT / "data" / "published" / "nesso_holdout_predictions.csv"
    )
    validation = validation.merge(
        nesso_validation[["feature_row", "predicted_pEC50_ensemble"]].rename(
            columns={
                "predicted_pEC50_ensemble": "fine_tuned_continuous_head_predicted_pEC50"
            }
        ),
        on="feature_row",
        how="left",
        validate="one_to_one",
    )
    # Reuse the canonical completed validation table rather than the superseded
    # site-dossier workspace. This keeps the historical rerun self-contained.
    full_validation = pd.read_csv(
        REPO_ROOT
        / "reports"
        / "transfer_ablation"
        / "binary_head_reanalysis"
        / "validation_predictions.csv"
    )
    validation = validation.merge(
        full_validation[["feature_row", "lightgbm_predicted_pEC50"]],
        on="feature_row",
        how="left",
        validate="one_to_one",
    )
    nesso_challenge = pd.read_csv(
        REPO_ROOT / "data" / "published" / "nesso_challenge_predictions.csv"
    )
    challenge = challenge.merge(
        nesso_challenge[["record_id", "trained_ensemble_pEC50"]].rename(
            columns={
                "trained_ensemble_pEC50": "fine_tuned_continuous_head_predicted_pEC50"
            }
        ),
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    lightgbm_challenge = pd.read_csv(
        REPO_ROOT / "data" / "published" / "established_2d_challenge_predictions.csv"
    )
    challenge = challenge.merge(
        lightgbm_challenge[["Molecule Name", "pEC50"]].rename(
            columns={
                "Molecule Name": "original_id",
                "pEC50": "lightgbm_predicted_pEC50",
            }
        ),
        on="original_id",
        how="left",
        validate="one_to_one",
    )
    return development, validation, challenge


def _run_representation_and_scalar_probes(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    challenge: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    *,
    skip_mlp: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    member1 = np.load(input_dir / "labeled_member1_repr.npy")
    member2 = np.load(input_dir / "labeled_member2_repr.npy")
    challenge_member1 = np.load(input_dir / "challenge_member1_repr.npy")
    challenge_member2 = np.load(input_dir / "challenge_member2_repr.npy")
    development_positions = development["feature_row"].to_numpy(dtype=int)
    validation_positions = validation["feature_row"].to_numpy(dtype=int)
    challenge_positions = challenge["feature_row"].to_numpy(dtype=int)
    member_mean = (member1 + member2) / 2.0
    challenge_mean = (challenge_member1 + challenge_member2) / 2.0

    scalar_dev = development[["pIC50_equivalent", "binary_logit"]].to_numpy()
    scalar_validation = validation[["pIC50_equivalent", "binary_logit"]].to_numpy()
    scalar_challenge = challenge[["pIC50_equivalent", "binary_logit"]].to_numpy()
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    polynomial.fit(scalar_dev)
    variants = {
        "affinity_only_linear": (
            scalar_dev[:, :1],
            scalar_validation[:, :1],
            scalar_challenge[:, :1],
        ),
        "binder_logit_only_linear": (
            scalar_dev[:, 1:],
            scalar_validation[:, 1:],
            scalar_challenge[:, 1:],
        ),
        "continuous_plus_binary_linear": (
            scalar_dev,
            scalar_validation,
            scalar_challenge,
        ),
        "continuous_plus_binary_polynomial2": (
            polynomial.transform(scalar_dev),
            polynomial.transform(scalar_validation),
            polynomial.transform(scalar_challenge),
        ),
        "mean_repr_384d_ridge": (
            member_mean[development_positions],
            member_mean[validation_positions],
            challenge_mean[challenge_positions],
        ),
        "mean_repr_plus_readouts_386d_ridge": (
            np.column_stack([member_mean[development_positions], scalar_dev]),
            np.column_stack([member_mean[validation_positions], scalar_validation]),
            np.column_stack([challenge_mean[challenge_positions], scalar_challenge]),
        ),
        "concat_repr_768d_ridge": (
            np.column_stack(
                [member1[development_positions], member2[development_positions]]
            ),
            np.column_stack(
                [member1[validation_positions], member2[validation_positions]]
            ),
            np.column_stack(
                [
                    challenge_member1[challenge_positions],
                    challenge_member2[challenge_positions],
                ]
            ),
        ),
        "concat_repr_plus_member_readouts_772d_ridge": (
            np.column_stack(
                [
                    member1[development_positions],
                    member2[development_positions],
                    development[
                        [
                            "member1_affinity_pred_value",
                            "member1_binary_logit",
                            "member2_affinity_pred_value",
                            "member2_binary_logit",
                        ]
                    ].to_numpy(),
                ]
            ),
            np.column_stack(
                [
                    member1[validation_positions],
                    member2[validation_positions],
                    validation[
                        [
                            "member1_affinity_pred_value",
                            "member1_binary_logit",
                            "member2_affinity_pred_value",
                            "member2_binary_logit",
                        ]
                    ].to_numpy(),
                ]
            ),
            np.column_stack(
                [
                    challenge_member1[challenge_positions],
                    challenge_member2[challenge_positions],
                    challenge[
                        [
                            "member1_affinity_pred_value",
                            "member1_binary_logit",
                            "member2_affinity_pred_value",
                            "member2_binary_logit",
                        ]
                    ].to_numpy(),
                ]
            ),
        ),
    }
    selection_rows: list[dict[str, Any]] = []
    for name, (x_dev, x_validation, x_challenge) in variants.items():
        oof, validation_prediction, challenge_prediction, selections = (
            _nested_ridge_variant(
                name,
                x_dev,
                x_validation,
                x_challenge,
                development,
                validation,
                challenge,
            )
        )
        development[f"{name}_predicted_pEC50"] = oof
        validation[f"{name}_predicted_pEC50"] = validation_prediction
        challenge[f"{name}_predicted_pEC50"] = challenge_prediction
        selection_rows.extend(selections)

    if not skip_mlp:
        mlp_variants = {
            "mean_repr_384d_mlp32_fixed": variants["mean_repr_384d_ridge"],
            "mean_repr_plus_readouts_386d_mlp32_fixed": variants[
                "mean_repr_plus_readouts_386d_ridge"
            ],
        }
        for name, (x_dev, x_validation, x_challenge) in mlp_variants.items():
            oof, validation_prediction, challenge_prediction = _fixed_mlp_predictions(
                x_dev, x_validation, x_challenge, development
            )
            development[f"{name}_predicted_pEC50"] = oof
            validation[f"{name}_predicted_pEC50"] = validation_prediction
            challenge[f"{name}_predicted_pEC50"] = challenge_prediction
            selection_rows.append(
                {
                    "model": name,
                    "selection_scope": "prespecified_fixed_configuration",
                    "outer_fold": np.nan,
                    "selected_alpha": MLP_CONFIG["alpha"],
                    "inner_mae": np.nan,
                    "inner_spearman": np.nan,
                }
            )
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "probe_selections.csv", index=False
    )
    return development, validation, challenge


def _merge_dual_branch_adapter(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    challenge: pd.DataFrame,
    input_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = input_dir / "dual_branch_activity_adapter_predictions.csv"
    if not path.is_file():
        return development, validation, challenge
    predictions = pd.read_csv(path)
    for dataset, frame in (
        ("development", development),
        ("validation", validation),
        ("challenge", challenge),
    ):
        subset = predictions.loc[predictions["dataset"].eq(dataset)][
            ["feature_row", "predicted_pEC50"]
        ].rename(
            columns={
                "predicted_pEC50": (
                    "fine_tuned_dual_branch_activity_adapter_predicted_pEC50"
                )
            }
        )
        frame = frame.merge(subset, on="feature_row", how="left", validate="one_to_one")
        if dataset == "development":
            development = frame
        elif dataset == "validation":
            validation = frame
        else:
            challenge = frame
    return development, validation, challenge


def _regression_outputs(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    challenge: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_replicates: int,
) -> None:
    development["direct_affinity_predicted_pEC50"] = development["pIC50_equivalent"]
    validation["direct_affinity_predicted_pEC50"] = validation["pIC50_equivalent"]
    challenge["direct_affinity_predicted_pEC50"] = challenge["pIC50_equivalent"]
    model_columns = {
        column.removesuffix("_predicted_pEC50"): column
        for column in development.columns
        if column.endswith("_predicted_pEC50")
    }
    ordered_models = [
        "direct_affinity",
        "affinity_only_linear",
        "binder_logit_only_linear",
        "continuous_plus_binary_linear",
        "continuous_plus_binary_polynomial2",
        "mean_repr_384d_ridge",
        "mean_repr_plus_readouts_386d_ridge",
        "concat_repr_768d_ridge",
        "concat_repr_plus_member_readouts_772d_ridge",
        "mean_repr_384d_mlp32_fixed",
        "mean_repr_plus_readouts_386d_mlp32_fixed",
        "fine_tuned_continuous_head",
        "fine_tuned_dual_branch_activity_adapter",
        "lightgbm",
    ]
    models = [name for name in ordered_models if name in model_columns]
    metric_rows: list[dict[str, Any]] = []
    for dataset, frame in (
        ("development", development),
        ("validation", validation),
        ("challenge", challenge),
    ):
        cohorts = {"all": frame}
        if dataset != "challenge":
            primary_role = (
                "development_primary" if dataset == "development" else "lockbox_primary"
            )
            cohorts["Emax_qualified"] = frame.loc[frame["role"].eq(primary_role)]
            cohorts["lower_Emax"] = frame.loc[frame["role"].eq("excluded_initial_emax")]
        for cohort, subset in cohorts.items():
            for model in models:
                metric_rows.append(
                    _metric_row(
                        dataset,
                        cohort,
                        model,
                        subset["pEC50"],
                        subset[model_columns[model]],
                    )
                )
    pd.DataFrame(metric_rows).to_csv(output_dir / "regression_metrics.csv", index=False)

    interval_frames: list[pd.DataFrame] = []
    paired_frames: list[pd.DataFrame] = []
    comparisons = [
        ("continuous_plus_binary_linear", "affinity_only_linear"),
        ("continuous_plus_binary_polynomial2", "affinity_only_linear"),
        ("mean_repr_plus_readouts_386d_ridge", "mean_repr_384d_ridge"),
        ("concat_repr_plus_member_readouts_772d_ridge", "concat_repr_768d_ridge"),
        ("fine_tuned_dual_branch_activity_adapter", "fine_tuned_continuous_head"),
        ("fine_tuned_dual_branch_activity_adapter", "lightgbm"),
    ]
    comparisons = [pair for pair in comparisons if all(name in models for name in pair)]
    for dataset, frame, group_column in (
        ("development", development, "cluster_component"),
        ("validation", validation, "audit_scaffold_group"),
    ):
        mapping = {name: model_columns[name] for name in models}
        intervals, draws = clustered_bootstrap(
            frame,
            mapping,
            group_column=group_column,
            observed_column="pEC50",
            replicates=bootstrap_replicates,
            seed=20260813,
        )
        intervals["dataset"] = dataset
        interval_frames.append(intervals)
        point_metrics = {
            name: regression_metrics(frame["pEC50"], frame[column])
            for name, column in mapping.items()
        }
        paired = paired_difference_rows(
            draws,
            point_metrics,
            comparisons,
            metrics=("mae", "pearson", "spearman", "bias"),
        )
        paired["dataset"] = dataset
        paired_frames.append(paired)
    pd.concat(interval_frames, ignore_index=True).to_csv(
        output_dir / "regression_family_bootstrap_intervals.csv", index=False
    )
    pd.concat(paired_frames, ignore_index=True).to_csv(
        output_dir / "regression_paired_family_bootstrap_differences.csv", index=False
    )

    development.to_csv(output_dir / "development_predictions.csv", index=False)
    validation.to_csv(output_dir / "validation_predictions.csv", index=False)
    challenge.to_csv(output_dir / "challenge_predictions.csv", index=False)
    assert_outer_predictions_isolated(
        development,
        expected_rows=development["feature_row"].tolist(),
    )


def _run_pair_pool_extension(
    manifest: pd.DataFrame,
    pair_readouts: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_replicates: int,
) -> None:
    pair = pair_readouts.merge(
        manifest[
            [
                "feature_row",
                "pEC50",
                "fold",
                "cluster_component",
                "modeling_role",
                "role",
                "audit_scaffold_group",
            ]
        ],
        on="feature_row",
        how="left",
        validate="many_to_one",
    )
    development = pair.loc[pair["modeling_role"].eq("development")].copy()
    validation = pair.loc[pair["modeling_role"].eq("lockbox")].copy()
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    proxy_frames: list[pd.DataFrame] = []
    for variant in ("all_pairs", "ligand_ligand_only", "receptor_ligand_only"):
        dev = development.loc[development["pool_variant"].eq(variant)].sort_values(
            "feature_row"
        )
        val = validation.loc[validation["pool_variant"].eq(variant)].sort_values(
            "feature_row"
        )
        x_dev = dev[["pIC50_equivalent", "binary_logit"]].to_numpy()
        x_val = val[["pIC50_equivalent", "binary_logit"]].to_numpy()
        folds = dev["fold"].to_numpy(dtype=int)
        dev_prediction, nested_selections = _nested_ridge_probe(
            x_dev, dev["pEC50"].to_numpy(), folds
        )
        val_prediction, selected_alpha, _ = _full_ridge_prediction(
            x_dev,
            dev["pEC50"].to_numpy(),
            folds,
            x_val,
        )
        metric_rows.append(
            _metric_row(
                "development",
                "all",
                f"{variant}_combined_readouts",
                dev["pEC50"],
                dev_prediction,
            )
        )
        metric_rows.append(
            _metric_row(
                "validation",
                "all",
                f"{variant}_combined_readouts",
                val["pEC50"],
                val_prediction,
            )
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "dataset": "development",
                    "feature_row": dev["feature_row"].to_numpy(),
                    "pool_variant": variant,
                    "pEC50": dev["pEC50"].to_numpy(),
                    "predicted_pEC50": dev_prediction,
                    "selected_alpha": [
                        ";".join(
                            str(value)
                            for value in nested_selections["selected_alpha"].tolist()
                        )
                    ]
                    * len(dev),
                }
            )
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "dataset": "validation",
                    "feature_row": val["feature_row"].to_numpy(),
                    "pool_variant": variant,
                    "pEC50": val["pEC50"].to_numpy(),
                    "predicted_pEC50": val_prediction,
                    "selected_alpha": selected_alpha,
                }
            )
        )
        for dataset, frame in (
            ("development", dev),
            ("validation", val),
        ):
            for threshold in PROXY_THRESHOLDS:
                row = functional_proxy_metrics(
                    frame["pEC50"], frame["binder_probability"], threshold
                )
                row.update({"dataset": dataset, "pool_variant": variant})
                proxy_frames.append(pd.DataFrame([row]))
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "pair_pool_combined_readout_metrics.csv", index=False
    )
    pair_predictions = pd.concat(prediction_frames, ignore_index=True)
    pair_predictions.to_csv(
        output_dir / "pair_pool_combined_readout_predictions.csv", index=False
    )
    pd.concat(proxy_frames, ignore_index=True).to_csv(
        output_dir / "pair_pool_proxy_metrics.csv", index=False
    )

    paired_frames: list[pd.DataFrame] = []
    for dataset, frame, group_column in (
        ("development", development, "cluster_component"),
        ("validation", validation, "audit_scaffold_group"),
    ):
        wide = frame.pivot(
            index=["feature_row", "pEC50", group_column],
            columns="pool_variant",
            values="binder_probability",
        ).reset_index()
        for threshold in PROXY_THRESHOLDS:
            paired = paired_proxy_bootstrap_difference(
                wide,
                {
                    "ligand_ligand_only": "ligand_ligand_only",
                    "all_pairs": "all_pairs",
                },
                [("ligand_ligand_only", "all_pairs")],
                threshold=threshold,
                group_column=group_column,
                replicates=bootstrap_replicates,
                seed=20260813,
            )
            paired["dataset"] = dataset
            paired_frames.append(paired)
    pd.concat(paired_frames, ignore_index=True).to_csv(
        output_dir / "pair_pool_proxy_paired_family_bootstrap_differences.csv",
        index=False,
    )

    regression_paired_frames: list[pd.DataFrame] = []
    for dataset, group_column in (
        ("development", "cluster_component"),
        ("validation", "audit_scaffold_group"),
    ):
        wide = (
            pair_predictions.loc[pair_predictions["dataset"].eq(dataset)]
            .pivot(
                index=["feature_row", "pEC50"],
                columns="pool_variant",
                values="predicted_pEC50",
            )
            .reset_index()
        )
        wide = wide.merge(
            manifest[["feature_row", group_column]],
            on="feature_row",
            how="left",
            validate="one_to_one",
        )
        model_columns = {
            "ligand_ligand_only": "ligand_ligand_only",
            "all_pairs": "all_pairs",
        }
        _, draws = clustered_bootstrap(
            wide,
            model_columns,
            group_column=group_column,
            observed_column="pEC50",
            replicates=bootstrap_replicates,
            seed=20260813,
        )
        point_metrics = {
            model: regression_metrics(wide["pEC50"], wide[column])
            for model, column in model_columns.items()
        }
        paired = paired_difference_rows(
            draws,
            point_metrics,
            [("ligand_ligand_only", "all_pairs")],
            metrics=("mae", "spearman"),
        )
        paired["dataset"] = dataset
        regression_paired_frames.append(paired)
    pd.concat(regression_paired_frames, ignore_index=True).to_csv(
        output_dir
        / "pair_pool_combined_readout_paired_family_bootstrap_differences.csv",
        index=False,
    )


def _plot_regression_comparison(output_dir: Path) -> None:
    metrics = pd.read_csv(output_dir / "regression_metrics.csv")
    models = [
        "affinity_only_linear",
        "continuous_plus_binary_linear",
        "continuous_plus_binary_polynomial2",
        "mean_repr_384d_ridge",
        "mean_repr_plus_readouts_386d_ridge",
        "fine_tuned_continuous_head",
        "fine_tuned_dual_branch_activity_adapter",
        "lightgbm",
    ]
    labels = {
        "affinity_only_linear": "Affinity scalar",
        "continuous_plus_binary_linear": "Both scalars, linear",
        "continuous_plus_binary_polynomial2": "Both scalars, polynomial",
        "mean_repr_384d_ridge": "384D ridge",
        "mean_repr_plus_readouts_386d_ridge": "384D + readouts ridge",
        "fine_tuned_continuous_head": "FT continuous head",
        "fine_tuned_dual_branch_activity_adapter": "FT dual-branch adapter",
        "lightgbm": "LightGBM",
    }
    models = [model for model in models if model in set(metrics["model"])]
    subset = metrics.loc[
        metrics["cohort"].eq("all")
        & metrics["dataset"].isin(["development", "validation"])
        & metrics["model"].isin(models)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    x = np.arange(len(models))
    width = 0.36
    colors = {"development": "#2f5d7c", "validation": "#b64c3b"}
    for axis, metric, ylabel in zip(
        axes,
        ("mae", "spearman"),
        ("MAE (lower is better)", "Spearman (higher is better)"),
        strict=True,
    ):
        for offset, dataset in zip(
            (-width / 2, width / 2), ("development", "validation"), strict=True
        ):
            values = []
            for model in models:
                row = subset.loc[
                    subset["dataset"].eq(dataset) & subset["model"].eq(model)
                ]
                values.append(float(row.iloc[0][metric]))
            axis.bar(
                x + offset,
                values,
                width=width,
                color=colors[dataset],
                label=dataset.title(),
            )
        axis.set_xticks(x, [labels[model] for model in models], rotation=38, ha="right")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False)
    fig.suptitle("Where the transferable PXR signal resides")
    fig.savefig(
        output_dir / "corrected_readout_comparison.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction = json.loads(
        (args.input_dir / "reconstruction_provenance.json").read_text()
    )
    if reconstruction["status"] != "pass":
        raise RuntimeError("binary reconstruction provenance did not pass")

    manifest = pd.read_csv(REPO_ROOT / "data" / "published" / "modeling_manifest.csv")
    readouts = pd.read_csv(args.input_dir / "labeled_binary_readouts.csv")
    labeled = manifest.merge(
        readouts, on=["feature_row", "record_id", "original_id"], validate="one_to_one"
    )
    development = (
        labeled.loc[labeled["modeling_role"].eq("development")]
        .sort_values("feature_row")
        .reset_index(drop=True)
    )
    validation = (
        labeled.loc[labeled["modeling_role"].eq("lockbox")]
        .sort_values("feature_row")
        .reset_index(drop=True)
    )
    development["fold"] = development["fold"].astype(int)

    challenge_readouts = pd.read_csv(args.input_dir / "challenge_binary_readouts.csv")
    challenge_truth = pd.read_csv(
        REPO_ROOT / "data" / "published" / "nesso_challenge_predictions.csv"
    )
    challenge = (
        challenge_readouts.merge(
            challenge_truth[["record_id", "pEC50"]],
            on="record_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("feature_row")
        .reset_index(drop=True)
    )
    if (len(development), len(validation), len(challenge)) != (3344, 790, 513):
        raise AssertionError("cohort coverage changed")

    development, validation, challenge = _add_existing_baselines(
        development, validation, challenge
    )
    development, validation, challenge = _run_representation_and_scalar_probes(
        development,
        validation,
        challenge,
        args.input_dir,
        args.output_dir,
        skip_mlp=args.skip_mlp,
    )
    development, validation, challenge = _merge_dual_branch_adapter(
        development, validation, challenge, args.input_dir
    )
    _run_proxy_analysis(
        development,
        validation,
        challenge,
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    _regression_outputs(
        development,
        validation,
        challenge,
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    pair_readouts = pd.read_csv(args.input_dir / "pair_pool_binary_readouts.csv")
    _run_pair_pool_extension(
        manifest,
        pair_readouts,
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    _plot_proxy_curves(args.output_dir)
    _plot_proxy_matrices(args.output_dir)
    _plot_probability_distribution(development, validation, args.output_dir)
    _plot_regression_comparison(args.output_dir)

    metadata = {
        "status": "pass",
        "scope": "CPU-only Nesso binary-head correction; no external publication",
        "proxy_thresholds": list(PROXY_THRESHOLDS),
        "proxy_semantics": "functional potency proxies, not binding ground truth",
        "released_binder_cutoff": 0.5,
        "ridge_alphas": list(RIDGE_ALPHAS),
        "mlp_config": None if args.skip_mlp else MLP_CONFIG,
        "bootstrap_replicates": args.bootstrap_replicates,
        "development_rows": len(development),
        "validation_rows": len(validation),
        "challenge_rows": len(challenge),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(args.output_dir / "analysis_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
