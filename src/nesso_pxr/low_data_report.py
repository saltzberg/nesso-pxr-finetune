"""Auditable, partial-matrix reports; no fitting or fabricated result fallbacks.

All scores are recomputed from completed task predictions and checked against
prepared identities, labels, held-out assignments, label budgets and saved scores.
Repeated held-out predictions are descriptive observations, not independent samples.
"""

from __future__ import annotations

import hashlib
import html
import itertools
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEY = ["split", "outer_fold", "draw", "n_train", "method"]
PRED_COLUMNS = [
    "record_id",
    "row_index",
    *KEY,
    "y_true",
    "y_pred",
    "assay_se",
    "weight",
    "ensemble_std",
    "nearest_similarity",
    "chemical_group",
    "lower_80",
    "upper_80",
    "lower_90",
    "upper_90",
]
METRIC_COLUMNS = [
    "task_id",
    "status",
    *KEY,
    "n_fit",
    "n_calibration",
    "n_test",
    "weighted_mae",
    "mae",
    "spearman",
    "fit_seconds",
    "n_parameters",
    "coverage_80",
    "width_80",
    "coverage_90",
    "width_90",
    "selection",
    "fit_audit",
]
FLOWS = {
    "weighted_mean": "Fit labels + inverse-SE weights → constant weighted mean",
    "native_continuous": (
        "Frozen Nesso twin 384D vectors → native continuous heads → member mean; "
        "pIC50-equivalent output, not assay-matched pEC50"
    ),
    "scalar_ridge": (
        "Frozen Nesso vectors → reconstructed continuous + binary scalars → "
        "fit-only scaling / weighted ridge → pEC50"
    ),
    "repr_ridge": (
        "Frozen paired Nesso 384D vectors → fit-only scaling / weighted ridge → pEC50"
    ),
    "repr_mlp": (
        "Frozen paired 384D vectors → concatenated 768D → "
        "weighted 768→128→1 ReLU MLP → pEC50"
    ),
    "head_random": (
        "Frozen paired Nesso vectors → random twin 384–384–384–1 heads → "
        "target-label head fitting → member / seed ensemble"
    ),
    "head_pretrained": (
        "Frozen paired Nesso vectors → pretrained twin 384–384–384–1 heads → "
        "target-label head fitting → member / seed ensemble"
    ),
    "dual_head": (
        "Frozen paired Nesso vectors → continuous + binary pretrained readouts → "
        "target-label readout adaptation / combination → pEC50"
    ),
    "head_lora": (
        "Frozen paired Nesso vectors → frozen pretrained final readout + rank-4 "
        "LoRA update → member / seed ensemble; no upstream representation update"
    ),
    "morgan_ridge": "Canonical SMILES → Morgan radius-2 / 2048 bits → weighted ridge",
    "morgan_lightgbm": (
        "Canonical SMILES → Morgan radius-2 / 2048 bits → weighted LightGBM; "
        "Morgan-only 2D baseline, not the historical descriptor-rich model"
    ),
    "morgan_knn": "Canonical SMILES → Morgan radius-2 / 2048 bits → weighted kNN",
}
EFFECTS = [
    ("head_pretrained", "head_random", "Pretrained vs random matched heads"),
    ("scalar_ridge", "native_continuous", "Scalar calibration vs native endpoint"),
    ("repr_ridge", "scalar_ridge", "Vector vs scalar ridge inputs"),
    ("repr_mlp", "repr_ridge", "MLP vs ridge readout"),
    ("head_lora", "head_pretrained", "Final-readout LoRA vs head fine-tuning"),
    ("dual_head", "head_pretrained", "Dual vs continuous readout"),
]
FAMILIES = {
    "learning_curves": "Weighted learning curves",
    "paired_effects": "Paired controlled effects",
    "novelty_error": "Chemical novelty and error",
    "calibration": "Predictions and interval calibration",
    "accuracy_cost": "Accuracy and measured fitting cost",
}


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected JSON object")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, name: str, *, minimum=None) -> float:
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"invalid {name}")
    return number


def _same_number(actual, saved, name: str) -> None:
    if actual is None:
        if saved is not None and not pd.isna(saved):
            raise ValueError(f"{name}: expected undefined")
    elif saved is None or not np.isclose(actual, float(saved), rtol=1e-6, atol=1e-8):
        raise ValueError(f"{name}: saved score does not match predictions")


def _score(frame: pd.DataFrame, floor: float) -> dict:
    y = frame.y_true.to_numpy(float)
    pred = frame.y_pred.to_numpy(float)
    se = frame.assay_se.to_numpy(float)
    w = 1 / np.maximum(se, floor)
    err = pred - y
    rho = None
    if len(frame) > 1 and np.ptp(y) > 0 and np.ptp(pred) > 0:
        rho = float(pd.Series(y).rank().corr(pd.Series(pred).rank()))
    return {
        "weighted_mae": float(np.average(abs(err), weights=w)),
        "mae": float(np.mean(abs(err))),
        "spearman": rho,
        "bias": float(np.mean(err)),
        "weight_sum": float(w.sum()),
        "weighted_error_sum": float(np.dot(abs(err), w)),
        "error_sum": float(abs(err).sum()),
        "bias_sum": float(err.sum()),
        "weight_squared_sum": float(np.dot(w, w)),
        "effective_n": float(w.sum() ** 2 / np.dot(w, w)),
        "weighted_mae_floor_005": float(
            np.average(abs(err), weights=1 / np.maximum(se, 0.05))
        ),
        "weighted_mae_floor_020": float(
            np.average(abs(err), weights=1 / np.maximum(se, 0.20))
        ),
    }


def _interval(frame: pd.DataFrame, level: int) -> dict:
    lo = frame[f"lower_{level}"].to_numpy(float)
    hi = frame[f"upper_{level}"].to_numpy(float)
    y = frame.y_true.to_numpy(float)
    missing = np.isnan(lo) | np.isnan(hi)
    valid = ~missing
    if np.any(lo[valid] > hi[valid]):
        raise ValueError(f"reversed {level}% interval")
    if np.any(np.isposinf(lo[valid])) or np.any(np.isneginf(hi[valid])):
        raise ValueError(f"invalid directed infinity in {level}% interval")
    unbounded = valid & (~np.isfinite(lo) | ~np.isfinite(hi))
    finite = valid & ~unbounded
    count = int(valid.sum())
    coverage = (
        float(np.mean((y[valid] >= lo[valid]) & (y[valid] <= hi[valid])))
        if count
        else None
    )
    width = float(np.mean(hi[valid] - lo[valid])) if count else None
    state = "unavailable" if not count else "finite"
    if unbounded.any():
        state = "unsupported_unbounded"
    elif missing.any():
        state = "partially_unavailable"
    return {
        f"coverage_{level}": coverage,
        f"width_{level}": width,
        f"interval_status_{level}": state,
        f"interval_n_{level}": count,
        f"finite_interval_n_{level}": int(finite.sum()),
        f"unbounded_interval_n_{level}": int(unbounded.sum()),
        f"missing_interval_n_{level}": int(missing.sum()),
    }


def _verify(
    task_dir: Path,
    meta: dict,
    protocol: dict,
    inputs: pd.DataFrame,
    assignments: pd.DataFrame,
    subsets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    missing = set(METRIC_COLUMNS) - meta.keys()
    if missing:
        raise ValueError(f"missing metrics fields: {', '.join(sorted(missing))}")
    if meta["task_id"] != task_dir.name:
        raise ValueError("task_id does not match directory")
    frame = pd.read_csv(
        task_dir / "predictions.csv",
        dtype={"record_id": str},
        float_precision="round_trip",
    )
    missing = set(PRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing prediction fields: {', '.join(sorted(missing))}")
    if (
        frame.empty
        or frame.row_index.duplicated().any()
        or frame.record_id.duplicated().any()
    ):
        raise ValueError("empty predictions or duplicate held-out identity")
    for key in KEY:
        if not frame[key].eq(meta[key]).all():
            raise ValueError(f"prediction / metrics mismatch: {key}")
    for name in [
        "y_true",
        "y_pred",
        "assay_se",
        "weight",
        "ensemble_std",
        "nearest_similarity",
    ]:
        if not np.isfinite(frame[name].to_numpy(float)).all():
            raise ValueError(f"nonfinite {name}")
    if (frame.assay_se <= 0).any() or (frame.ensemble_std < 0).any():
        raise ValueError("invalid assay SE or ensemble SD")
    if not frame.nearest_similarity.between(0, 1).all():
        raise ValueError("nearest_similarity must be Tanimoto similarity in [0, 1]")
    floor = _finite_number(protocol["se_floor"], "se_floor", minimum=1e-12)
    if not np.allclose(frame.weight, 1 / np.maximum(frame.assay_se, floor), rtol=1e-6):
        raise ValueError("weights are not frozen inverse-assay-SE weights")
    if not frame.row_index.isin(inputs.row_index).all():
        raise ValueError("prediction row absent from prepared inputs")
    truth = inputs.set_index("row_index").loc[frame.row_index]
    if list(truth.record_id.astype(str)) != list(frame.record_id):
        raise ValueError("record identity does not match prepared inputs")
    for saved, original in [("y_true", "pEC50"), ("assay_se", "pEC50_standard_error")]:
        if not np.allclose(frame[saved], truth[original], rtol=1e-7, atol=1e-9):
            raise ValueError(f"{saved} does not match prepared inputs")
    test = assignments.loc[
        assignments.split.eq(meta["split"])
        & assignments.outer_fold.eq(meta["outer_fold"])
    ]
    if set(test.row_index) != set(frame.row_index):
        raise ValueError("predictions do not cover the exact held-out fold")
    groups = test.set_index("row_index").loc[frame.row_index, "chemical_group"]
    if list(groups.astype(str)) != list(frame.chemical_group.astype(str)):
        raise ValueError("chemical groups do not match prepared assignments")
    chosen = subsets.loc[
        subsets.split.eq(meta["split"])
        & subsets.outer_fold.eq(meta["outer_fold"])
        & subsets.draw.eq(meta["draw"])
        & subsets.n_train.eq(meta["n_train"])
    ]
    if chosen.empty or chosen.row_index.duplicated().any():
        raise ValueError("missing or duplicate acquired label subset")
    if set(chosen.row_index) & set(frame.row_index):
        raise ValueError("acquired / test overlap")
    if not chosen.row_index.isin(inputs.row_index).all():
        raise ValueError("acquired row absent from prepared inputs")
    if not chosen.role.isin(["fit", "calibration"]).all():
        raise ValueError("unknown acquired label role")
    nf = int(chosen.role.eq("fit").sum())
    nc = int(chosen.role.eq("calibration").sum())
    if (
        nf != meta["n_fit"]
        or nc != meta["n_calibration"]
        or len(frame) != meta["n_test"]
    ):
        raise ValueError("saved fit / calibration / test counts disagree")
    acquired = nf + nc
    if meta["n_train"] != -1 and acquired != meta["n_train"]:
        raise ValueError("N is not the acquired label budget")
    if meta["n_train"] == -1:
        policy_rows = set(
            assignments.loc[assignments.split.eq(meta["split"]), "row_index"]
        )
        if set(chosen.row_index) != policy_rows - set(frame.row_index):
            raise ValueError("full reference does not use the full outer training pool")
    if nc != math.ceil(acquired * float(protocol.get("calibration_fraction", 0.2))):
        raise ValueError("calibration reservation differs from frozen budget")
    _finite_number(meta["fit_seconds"], "fit_seconds", minimum=0)
    _finite_number(meta["n_parameters"], "n_parameters", minimum=0)
    scores = _score(frame, floor)
    for name in ["weighted_mae", "mae", "spearman"]:
        _same_number(scores[name], meta[name], name)
    interval = {}
    for level in [80, 90]:
        values = _interval(frame, level)
        for metric in [f"coverage_{level}", f"width_{level}"]:
            # Runner serializes nonfinite JSON widths as null with an explicit
            # interval_<level>_finite flag; the CSV preserves signed infinities.
            null_infinity = (
                metric == f"width_{level}"
                and values[metric] == math.inf
                and meta[metric] is None
                and meta.get(f"interval_{level}_finite") is False
            )
            if not null_infinity:
                _same_number(values[metric], meta[metric], metric)
        # A finite interval cannot claim a conformal rank beyond calibration N.
        rank = math.ceil((nc + 1) * (level / 100))
        if rank > nc and values[f"finite_interval_n_{level}"]:
            raise ValueError(f"finite {level}% intervals unsupported by calibration N")
        interval.update(values)
    result = {
        name: meta[name]
        for name in METRIC_COLUMNS
        if name not in {"selection", "fit_audit"}
    }
    result.update(scores)
    result.update(interval)
    result.update(
        verification="verified",
        n_acquired=acquired,
        n_unique_test=len(frame),
        prediction_sha256=_hash(task_dir / "predictions.csv"),
        metrics_sha256=_hash(task_dir / "metrics.json"),
        row_signature=hashlib.sha256(
            frame.sort_values("row_index")[
                ["row_index", "y_true", "assay_se", "weight"]
            ]
            .to_csv(index=False)
            .encode()
        ).hexdigest(),
    )
    return frame, result


def _expected(protocol: dict) -> list[tuple]:
    budgets = list(protocol["budgets"])
    if protocol.get("include_full_reference", False):
        budgets.append(-1)
    return list(
        itertools.product(
            protocol["split_policies"],
            range(int(protocol["n_folds"])),
            range(int(protocol["draws"])),
            budgets,
            protocol["methods"],
        )
    )


def _tables(rows: list[dict], compound_sets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = pd.DataFrame(rows)
    keys = ["split", "n_train", "method"]
    aggregate = []
    if tasks.empty:
        return pd.DataFrame(columns=keys + ["weighted_mae"]), pd.DataFrame()
    for key, group in tasks.groupby(keys, sort=True):
        wsum = group.weight_sum.sum()
        total = int(group.n_test.sum())
        # Kish N on repeated predictions is not an independent-compound sample size.
        # Unique-compound weights are retained once in compound_sets instead.
        weights = np.array(list(compound_sets[key].values()), dtype=float)
        aggregate.append(
            dict(
                zip(keys, key, strict=False),
                weighted_mae=group.weighted_error_sum.sum() / wsum,
                mae=group.error_sum.sum() / total,
                bias=group.bias_sum.sum() / total,
                n_tasks=len(group),
                n_prediction_observations=total,
                n_unique_compounds=len(weights),
                effective_n_unique_compounds=weights.sum() ** 2
                / np.dot(weights, weights),
                task_weighted_mae_min=group.weighted_mae.min(),
                task_weighted_mae_max=group.weighted_mae.max(),
                n_fit_min=group.n_fit.min(),
                n_fit_max=group.n_fit.max(),
                n_calibration_min=group.n_calibration.min(),
                n_calibration_max=group.n_calibration.max(),
                fit_seconds_median=group.fit_seconds.median(),
                fit_seconds_sum=group.fit_seconds.sum(),
                n_parameters_min=group.n_parameters.min(),
                n_parameters_max=group.n_parameters.max(),
            )
        )
    paired = []
    pair_keys = ["split", "outer_fold", "draw", "n_train"]
    for treatment, control, label in EFFECTS:
        left = tasks.loc[tasks.method.eq(treatment)]
        right = tasks.loc[tasks.method.eq(control)]
        if left.empty or right.empty:
            continue
        pairs = left.merge(
            right,
            on=pair_keys,
            suffixes=("_treatment", "_control"),
            validate="one_to_one",
        )
        for _, row in pairs.iterrows():
            if row.row_signature_treatment != row.row_signature_control:
                continue
            if (
                row.n_fit_treatment != row.n_fit_control
                or row.n_calibration_treatment != row.n_calibration_control
            ):
                continue
            paired.append(
                {
                    **{key: row[key] for key in pair_keys},
                    "effect": label,
                    "treatment": treatment,
                    "control": control,
                    "treatment_task_id": row.task_id_treatment,
                    "control_task_id": row.task_id_control,
                    "delta_weighted_mae": row.weighted_mae_treatment
                    - row.weighted_mae_control,
                    "n_test": row.n_test_treatment,
                    "n_fit": row.n_fit_treatment,
                    "n_calibration": row.n_calibration_treatment,
                }
            )
    return pd.DataFrame(aggregate), pd.DataFrame(paired)


def _bins(frame: pd.DataFrame, task: dict) -> tuple[list[dict], list[dict]]:
    """Task-level source bins, retaining repeated-observation denominators."""
    base = {name: task[name] for name in ["task_id", *KEY]}
    work = frame[["y_true", "y_pred", "nearest_similarity", "weight"]].copy()
    work["weighted_error"] = abs(work.y_pred - work.y_true) * work.weight
    work["absolute_error"] = abs(work.y_pred - work.y_true)
    work["novelty_bin"] = np.minimum((work.nearest_similarity * 10).astype(int), 9)
    novelty = []
    for index, group in work.groupby("novelty_bin"):
        novelty.append(
            {
                **base,
                "similarity_low": index / 10,
                "similarity_high": (index + 1) / 10,
                "n_observations": len(group),
                "weight_sum": group.weight.sum(),
                "weighted_error_sum": group.weighted_error.sum(),
                "absolute_error_sum": group.absolute_error.sum(),
            }
        )
    # Fixed-width bins extend without clipping, even for extreme extrapolations.
    work["observed_bin"] = np.floor(work.y_true / 0.25) * 0.25
    work["predicted_bin"] = np.floor(work.y_pred / 0.25) * 0.25
    calibration = []
    for (observed, predicted), group in work.groupby(["observed_bin", "predicted_bin"]):
        calibration.append(
            {
                **base,
                "observed_low": observed,
                "predicted_low": predicted,
                "bin_width": 0.25,
                "n_observations": len(group),
            }
        )
    return novelty, calibration


def _csv(rows, path: Path, columns=None) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False)
    return frame


def _draw_figures(
    output: Path,
    protocol: dict,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    novelty: pd.DataFrame,
    prediction_bins: pd.DataFrame,
    tasks: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#777777",
            "axes.linewidth": 0.6,
            "figure.facecolor": "#fffffb",
            "axes.facecolor": "#fffffb",
            "svg.fonttype": "none",
            "savefig.dpi": 150,
        }
    )
    methods = list(protocol["methods"])
    colors = dict(zip(methods, plt.get_cmap("tab20").colors, strict=False))
    splits = list(protocol["split_policies"])

    def layout(rows=1):
        fig, axes = plt.subplots(
            rows,
            max(1, len(splits)),
            figsize=(6 * max(1, len(splits)), 3.8 * rows),
            squeeze=False,
        )
        for col, split in enumerate(splits):
            axes[0, col].set_title(
                "Identity-grouped random"
                if split == "random"
                else "Chemical-cluster holdout"
            )
        return fig, axes

    def finish(fig, axes, name, handles=None):
        for ax in axes.flat:
            if not ax.has_data():
                ax.text(
                    0.5,
                    0.5,
                    "No verified observations",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="#666666",
                )
        if handles:
            fig.legend(
                handles.values(),
                handles.keys(),
                loc="lower center",
                ncol=3,
                frameon=False,
                fontsize=8,
            )
        fig.tight_layout(rect=(0, 0.16 if handles else 0, 1, 1))
        for extension in ["png", "svg"]:
            fig.savefig(output / "figures" / f"{name}.{extension}")
        plt.close(fig)

    fig, axes = layout()
    handles = {}
    for col, split in enumerate(splits):
        ax = axes[0, col]
        if not metrics.empty:
            subset = metrics.loc[metrics.split.eq(split) & metrics.n_train.ne(-1)]
            for method, group in subset.groupby("method"):
                group = group.sort_values("n_train")
                (line,) = ax.plot(
                    group.n_train,
                    group.weighted_mae,
                    "o-",
                    lw=1,
                    ms=3,
                    color=colors[method],
                )
                handles[method] = line
        ax.set(
            xlabel="Acquired labeled compounds (fit + calibration)",
            ylabel="Inverse-assay-SE weighted MAE (pEC50)",
        )
    finish(fig, axes, "learning_curves", handles)

    fig, axes = layout()
    handles = {}
    for col, split in enumerate(splits):
        ax = axes[0, col]
        if not paired.empty:
            subset = paired.loc[paired.split.eq(split) & paired.n_train.ne(-1)]
            for effect, group in subset.groupby("effect"):
                summary = group.groupby("n_train").delta_weighted_mae.agg(
                    ["mean", "min", "max"]
                )
                h = ax.errorbar(
                    summary.index,
                    summary["mean"],
                    yerr=[
                        summary["mean"] - summary["min"],
                        summary["max"] - summary["mean"],
                    ],
                    fmt="o-",
                    lw=0.8,
                    ms=3,
                )
                handles[effect] = h
        if ax.has_data():
            ax.axhline(0, color="#777777", lw=0.6)
        ax.set(
            xlabel="Acquired labeled compounds",
            ylabel="Paired Δ weighted MAE (treatment − control)",
        )
    finish(fig, axes, "paired_effects", handles)

    # Novelty and calibration are faceted by method, not overlaid across methods.
    # All acquired budgets remain explicit in downloadable source tables; choose
    # one common, protocol-largest observed non-full budget for the diagnostic view.
    available = (
        [] if tasks.empty else tasks.loc[tasks.n_train.ne(-1), "n_train"].tolist()
    )
    diagnostic_budget = max(available) if available else None
    active = (
        []
        if tasks.empty
        else [
            m
            for m in methods
            if (tasks.method.eq(m) & tasks.n_train.eq(diagnostic_budget)).any()
        ]
    )
    nrows = max(1, len(active))
    fig, axes = layout(nrows)
    for row, method in enumerate(active):
        for col, split in enumerate(splits):
            ax = axes[row, col]
            group = novelty.loc[
                novelty.method.eq(method)
                & novelty.split.eq(split)
                & novelty.n_train.eq(diagnostic_budget)
            ]
            if not group.empty:
                bins = group.groupby("similarity_low")[
                    ["weight_sum", "weighted_error_sum"]
                ].sum()
                ax.plot(
                    bins.index + 0.05,
                    bins.weighted_error_sum / bins.weight_sum,
                    "o-",
                    lw=1,
                    ms=3,
                    color=colors[method],
                )
            ax.set(
                xlim=(0, 1),
                xlabel="Nearest fit-set Morgan Tanimoto similarity",
                ylabel=f"{method}\nWeighted MAE (pEC50)",
            )
    finish(fig, axes, "novelty_error")

    # Each method gets two diagnostic rows: prediction density and interval data.
    fig, axes = layout(max(1, 2 * len(active)))
    for row, method in enumerate(active):
        for col, split in enumerate(splits):
            ax = axes[2 * row, col]
            group = prediction_bins.loc[
                prediction_bins.method.eq(method)
                & prediction_bins.split.eq(split)
                & prediction_bins.n_train.eq(diagnostic_budget)
            ]
            if not group.empty:
                bins = (
                    group.groupby(["observed_low", "predicted_low"])
                    .n_observations.sum()
                    .reset_index()
                )
                # Marker area is proportional to log1p repeated observation count.
                ax.scatter(
                    bins.observed_low + 0.125,
                    bins.predicted_low + 0.125,
                    s=8 * np.log1p(bins.n_observations),
                    color=colors[method],
                    alpha=0.5,
                    linewidths=0,
                    rasterized=True,
                )
                lo = min(bins.observed_low.min(), bins.predicted_low.min())
                hi = max(bins.observed_low.max(), bins.predicted_low.max()) + 0.25
                ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.6)
                ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
            ax.set(
                xlabel="Observed pEC50 (0.25-wide bins)",
                ylabel=f"{method}\nPredicted value",
            )
            ax = axes[2 * row + 1, col]
            group = tasks.loc[
                tasks.method.eq(method)
                & tasks.split.eq(split)
                & tasks.n_train.eq(diagnostic_budget)
            ]
            for level in [80, 90]:
                finite = group.loc[group[f"interval_status_{level}"].eq("finite")]
                if not finite.empty:
                    ax.scatter(
                        np.full(len(finite), level / 100),
                        finite[f"coverage_{level}"],
                        s=14,
                        color=colors[method],
                        alpha=0.5,
                    )
                unsupported = len(group) - len(finite)
                if unsupported:
                    ax.text(
                        level / 100,
                        0.07,
                        f"{unsupported} unsupported /\nunavailable tasks",
                        ha="center",
                        fontsize=7,
                    )
            if not group.empty:
                ax.plot([0.75, 0.95], [0.75, 0.95], color="#888888", lw=0.6)
            ax.set(
                xlim=(0.74, 0.96),
                ylim=(0, 1.03),
                xticks=[0.8, 0.9],
                xlabel="Nominal interval coverage",
                ylabel="Empirical held-out coverage\n(one mark per fold/draw task)",
            )
    finish(fig, axes, "calibration")

    fig, axes = layout()
    handles = {}
    for col, split in enumerate(splits):
        ax = axes[0, col]
        if not tasks.empty:
            for method, group in tasks.loc[
                tasks.split.eq(split) & tasks.n_train.ne(-1)
            ].groupby("method"):
                handles[method] = ax.scatter(
                    group.fit_seconds,
                    group.weighted_mae,
                    s=12,
                    color=colors[method],
                    alpha=0.65,
                )
        ax.set(
            xlabel="Measured fit time per task (seconds; linear scale)",
            ylabel="Weighted MAE (pEC50)",
        )
        ax.set_xlim(left=0)
    finish(fig, axes, "accuracy_cost", handles)


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        root = str(Path(__file__).resolve().parents[2]) + "/"
        return value.replace(root, "")
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_safe(value), indent=2, allow_nan=False) + "\n")


def _task_id(key: tuple) -> str:
    split, fold, draw, budget, method = key
    label = "full" if budget == -1 else budget
    return f"{split}__fold{fold}__draw{draw:02d}__n{label}__{method}"


def build_report(run_dir: Path, report_dir: Path) -> dict:
    """Read-only audit of a running matrix; invalid evidence never enters scores.

    Output may be refreshed while the runner works. Only complete markers seen
    in this snapshot count; started/in-flight tasks remain pending. No training,
    checkpoint unpickling, network access or site-navigation edits occur here.
    """
    run_dir, output = Path(run_dir).resolve(), Path(report_dir).resolve()
    if output == run_dir or run_dir in output.parents:
        raise ValueError("report_dir must be outside the immutable run directory")
    root = Path(__file__).resolve().parents[2]
    prepared = run_dir / "prepared"
    candidates = [
        run_dir / "protocol_snapshot.json",
        run_dir / "protocol.json",
        prepared / "protocol.json",
    ]
    protocol_path = next((p for p in candidates if p.is_file()), None)
    if protocol_path is None:
        raise FileNotFoundError("no run or prepared protocol snapshot")
    protocol = _json(protocol_path)
    expected = _expected(protocol)
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate protocol task keys")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    provenance, errors = [], []
    hash_cache = {}

    def check(path, wanted=None, category="input"):
        path = Path(path)
        try:
            resolved = path.resolve()
            if resolved not in hash_cache:
                hash_cache[resolved] = _hash(resolved)
            actual = hash_cache[resolved]
            ok = wanted is None or actual == wanted
            provenance.append(
                dict(
                    path=str(path),
                    category=category,
                    sha256=actual,
                    expected_sha256=wanted,
                    verified=ok,
                )
            )
            if not ok:
                raise ValueError(f"hash mismatch: {path}")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    check(protocol_path, category="protocol")
    for alternate in candidates:
        if alternate.is_file() and _json(alternate) != protocol:
            errors.append(f"protocol snapshots disagree: {alternate}")
    prep = _json(prepared / "preparation_manifest.json")
    check(prepared / "preparation_manifest.json")
    if prep.get("config") != protocol:
        errors.append("prepared config differs from run protocol")
    for name, digest in prep.get("artifact_sha256", {}).items():
        check(prepared / name, digest, "prepared_artifact")
    for name, digest in prep.get("code_sha256", {}).items():
        check(root / name, digest, "preparation_source")
    for source in prep.get("sources", {}).values():
        check(root / source["path"], source["sha256"], "original_input")
    binding_path = run_dir / "run_bindings.json"
    bindings = _json(binding_path) if binding_path.is_file() else None
    if bindings is not None:
        check(binding_path, category="run_bindings")
        if bindings.get("config") != protocol:
            errors.append("run bindings config differs from protocol")
        for name, digest in bindings.get("sources", {}).items():
            check(root / name, digest, "run_source")
        for name, digest in bindings.get("inputs", {}).items():
            check(root / name, digest, "run_input")
        if not bindings.get("sources") or not bindings.get("inputs"):
            errors.append("empty source/input run bindings")
    manifest_path = run_dir / "task_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if isinstance(manifest, dict):
            manifest = manifest.get("tasks", [])
        keys = [tuple(t[k] for k in KEY) for t in manifest]
        if len(keys) != len(expected) or set(keys) != set(expected):
            errors.append("task manifest differs from full protocol matrix")
        if any(t["task_id"] != _task_id(tuple(t[k] for k in KEY)) for t in manifest):
            errors.append("task manifest contains inconsistent task IDs")
        check(manifest_path, category="task_manifest")
    inputs = pd.read_csv(prepared / "inputs.csv", dtype={"record_id": str})
    assignments = pd.read_csv(prepared / "assignments.csv")
    subsets = pd.read_csv(prepared / "subsets.csv")
    audit = _json(prepared / "split_audit.json")
    for name in ["inputs.csv", "assignments.csv", "subsets.csv", "split_audit.json"]:
        if name not in prep.get("artifact_sha256", {}):
            errors.append(f"prepared artifact lacks hash: {name}")
    if inputs.row_index.duplicated().any() or inputs.record_id.duplicated().any():
        errors.append("duplicate prepared input identity")
    if assignments.duplicated(["split", "row_index"]).any():
        errors.append("duplicate prepared split assignment")
    rows, states, novelty_rows, calibration_rows, model_audits = [], [], [], [], []
    compound_sets = {}
    expected_ids = {_task_id(key) for key in expected}
    unexpected = sorted(
        p.name
        for p in (run_dir / "tasks").glob("*")
        if p.is_dir() and p.name not in expected_ids
    )
    for key in expected:
        tid = _task_id(key)
        directory = run_dir / "tasks" / tid
        state = dict(
            zip(KEY, key, strict=False),
            task_id=tid,
            status="pending",
            reason="not verified complete",
        )
        if (directory / "complete.json").is_file():
            try:
                if errors:
                    raise ValueError(
                        "global provenance verification failed; see provenance.json"
                    )
                if bindings is None or not manifest_path.is_file():
                    raise ValueError(
                        "completed evidence requires run bindings and task manifest"
                    )
                marker = _json(directory / "complete.json")
                if marker.get("task_id") != tid:
                    raise ValueError("complete marker task identity mismatch")
                hashes = marker.get("output_sha256", {})
                required = {
                    "metrics.json",
                    "predictions.csv",
                    "label_access.json",
                    "seed_predictions.npz",
                }
                if not required.issubset(hashes):
                    raise ValueError("complete marker omits required task artifacts")
                for name, digest in hashes.items():
                    if (
                        Path(name).name != name
                        or not name
                        or not isinstance(digest, str)
                    ):
                        raise ValueError("invalid complete marker artifact name/hash")
                    if _hash(directory / name) != digest:
                        raise ValueError(f"complete marker hash mismatch: {name}")
                meta = _json(directory / "metrics.json")
                if (
                    meta.get("status") != "complete"
                    or tuple(meta.get(k) for k in KEY) != key
                ):
                    raise ValueError("metrics task key/status differs from protocol")
                frame, row = _verify(
                    directory, meta, protocol, inputs, assignments, subsets
                )
                chosen = subsets.loc[
                    (
                        subsets[KEY[:4]]
                        == pd.Series(dict(zip(KEY[:4], key[:4], strict=False)))
                    ).all(axis=1)
                ]
                access = _json(directory / "label_access.json")
                for role in ["fit", "calibration", "test"]:
                    indices = (
                        frame.row_index
                        if role == "test"
                        else chosen.loc[chosen.role.eq(role), "row_index"]
                    )
                    ids = (
                        inputs.set_index("row_index")
                        .loc[indices, "record_id"]
                        .astype(str)
                        .tolist()
                    )
                    saved = list(map(str, access[f"{role}_record_ids"]))
                    if len(saved) != len(ids) or set(saved) != set(ids):
                        raise ValueError(f"label-access {role} identities differ")
                if (
                    access.get("test_labels_accessible_to_fit") is not False
                    or access.get("calibration_labels_accessible_to_fit") is not False
                ):
                    raise ValueError("label access declaration violates fit isolation")
                if access.get("acquired_label_count") != row["n_acquired"]:
                    raise ValueError("label access acquired count differs")
                if key[0] == "chemical_cluster":
                    policy = assignments.loc[assignments.split.eq(key[0])].set_index(
                        "row_index"
                    )
                    if set(policy.loc[chosen.row_index, "chemical_group"]) & set(
                        frame.chemical_group
                    ):
                        raise ValueError("chemical groups overlap acquired and test")
                with np.load(
                    directory / "seed_predictions.npz", allow_pickle=False
                ) as seeds:
                    indices = seeds["row_index"]
                    predictions = seeds["predictions"]
                    if (
                        predictions.ndim != 2
                        or predictions.shape[1] != len(indices)
                        or not np.isfinite(predictions).all()
                    ):
                        raise ValueError("invalid seed prediction shape/values")
                    if len(set(indices)) != len(indices) or set(indices) != set(
                        chosen.loc[chosen.role.eq("calibration"), "row_index"]
                    ) | set(frame.row_index):
                        raise ValueError("seed prediction identities differ")
                    positions = pd.Index(indices).get_indexer(frame.row_index)
                    if not np.allclose(
                        predictions[:, positions].mean(axis=0), frame.y_pred, atol=1e-6
                    ):
                        raise ValueError("seed mean differs from saved prediction")
                    if not np.allclose(
                        predictions[:, positions].std(axis=0),
                        frame.ensemble_std,
                        atol=1e-6,
                    ):
                        raise ValueError("seed SD differs from saved ensemble spread")
                rows.append(row)
                model_audits.append(
                    {
                        "task_id": tid,
                        "method": key[-1],
                        "selection": meta["selection"],
                        "fit_audit": meta["fit_audit"],
                    }
                )
                group_key = (row["split"], row["n_train"], row["method"])
                compound_sets.setdefault(group_key, {}).update(
                    zip(frame.record_id, frame.weight, strict=False)
                )
                nov, cal = _bins(frame, row)
                novelty_rows.extend(nov)
                calibration_rows.extend(cal)
                state.update(
                    status="complete",
                    reason="all task artifacts hashed; scores and identities verified",
                )
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                state.update(status="failed", reason=f"verification: {exc}")
        elif (directory / "failure.json").is_file():
            try:
                failure = _json(directory / "failure.json")
                reason = failure.get("error", "runner failure")
            except (OSError, ValueError) as exc:
                reason = str(exc)
            state.update(status="failed", reason=reason)
        states.append(state)
    metrics, paired = _tables(rows, compound_sets)
    tasks = _csv(rows, output / "task_metrics.csv", METRIC_COLUMNS)
    _csv(metrics, output / "metrics.csv")
    paired = _csv(
        paired,
        output / "paired_effects.csv",
        [*KEY[:4], "effect", "delta_weighted_mae"],
    )
    novelty = _csv(
        novelty_rows,
        output / "novelty_error.csv",
        ["task_id", *KEY, "similarity_low", "weight_sum", "weighted_error_sum"],
    )
    calibration = _csv(
        calibration_rows,
        output / "prediction_bins.csv",
        ["task_id", *KEY, "observed_low", "predicted_low", "n_observations"],
    )
    state_frame = _csv(states, output / "task_status.csv")
    counts = {
        name: int(state_frame.status.eq(name).sum())
        for name in ["complete", "failed", "pending"]
    }
    adapter_root = (
        root / protocol["adapter_pilot_dir"]
        if protocol.get("adapter_pilot_dir")
        else run_dir / "adapter_pilot"
    )
    adapter_path = adapter_root / "parent_replay" / "pilot.json"
    adapter = _json(adapter_path) if adapter_path.is_file() else None
    if adapter is not None:
        check(adapter_path, category="engineering_pilot_metadata")
    adapter_text = "No parent replay metadata available in this run."
    if adapter is not None:
        adapter_text = (
            f"Parent replay metadata reports status={adapter.get('status')}, "
            f"classification={adapter.get('classification')}, "
            f"trainable parameters={adapter.get('trainable_parameters')}, "
            f"peak CUDA allocated bytes={adapter.get('peak_cuda_allocated_bytes')}, "
            f"zero-update delta={adapter.get('zero_update_max_abs')}, "
            f"reload delta={adapter.get('reload_max_abs')}, "
            f"frozen unchanged={adapter.get('frozen_unchanged')}, "
            f"scientific fit={adapter.get('scientific_fit')}. "
            "This report hashes and reads pilot metadata; "
            "it does not rerun the GPU experiment."
        )
        _write_json(output / "adapter_pilot.json", adapter)
    summary = dict(
        schema_version="1.0",
        generated_at=datetime.now(UTC).isoformat(),
        run_dir=str(run_dir),
        total_tasks=len(expected),
        completed=counts["complete"],
        **counts,
        status="complete" if counts["complete"] == len(expected) else "partial",
        winner=None,
        conclusion=(
            "No winner declared; practical margin and inferential uncertainty "
            "are not established."
        ),
        provenance_errors=errors,
        unexpected_task_directories=unexpected,
        paired_variability=(
            "mean and observed min–max of matched fold/draw deltas; "
            "NOT confidence intervals"
        ),
        n_prepared_compounds=len(inputs),
        n_prediction_observations=int(tasks.n_test.sum()) if len(tasks) else 0,
        split_audit=audit,
        protocol=protocol,
    )
    runner_status_path = run_dir / "status.json"
    runner_status = _json(runner_status_path) if runner_status_path.is_file() else {}
    # Runner state is advisory: derive all scientific counts from artifacts.
    summary["runner_status"] = runner_status.get("status", "not_initialized")
    summary["runner_declared_counts"] = {
        k: runner_status.get(k)
        for k in ["total_tasks", "complete", "failed", "pending", "running"]
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "provenance.json",
        {
            "files": provenance,
            "errors": errors,
            "report_source_sha256": _hash(Path(__file__)),
        },
    )
    _write_json(output / "model_audits.json", model_audits)
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "split_audit.json", audit)
    _draw_figures(output, protocol, metrics, paired, novelty, calibration, tasks)
    fraction = audit.get("singleton_compound_fraction")
    singleton = "unavailable" if fraction is None else f"{100 * float(fraction):.4f}%"
    historical = protocol.get("historical_singleton_compound_percent", "unavailable")
    flows = "".join(
        f"<tr><td>{html.escape(m)}</td><td>"
        f"{html.escape(FLOWS.get(m, 'See recorded model audit'))}</td></tr>"
        for m in protocol["methods"]
    )
    figures = "".join(
        f'<section><h2>{title}</h2><a href="figures/{name}.svg">'
        f'<img src="figures/{name}.png" alt="{title}"></a></section>'
        for name, title in FAMILIES.items()
    )
    downloads = [
        "metrics.csv",
        "task_metrics.csv",
        "task_status.csv",
        "paired_effects.csv",
        "novelty_error.csv",
        "prediction_bins.csv",
        "summary.json",
        "provenance.json",
        "model_audits.json",
        "protocol.json",
        "split_audit.json",
    ]
    links = " · ".join(f'<a href="{name}">{name}</a>' for name in downloads)
    table = (
        "<p>No verified observations. No performance estimates are available.</p>"
        if metrics.empty
        else metrics.to_html(
            index=False, escape=True, float_format=lambda x: f"{x:.5g}"
        )
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nesso low-data adaptation — auditable partial results</title>
<style>
body{{font:16px/1.5 system-ui;max-width:1180px;margin:2rem auto;
padding:0 1rem;color:#222;background:#fffffb}}
h1,h2{{font-weight:550}}img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;font-size:13px}}
td,th{{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}}
.scroll{{overflow:auto}}.notice{{border-left:2px solid #999;padding:0 1rem}}
a{{color:#175b86}}summary{{cursor:pointer}}details{{margin:1.5rem 0}}
</style>
<nav><a href="../">Project summary</a> ·
<a href="#data">Source tables</a> · <a href="#protocol">Protocol</a></nav>
<h1>Nesso-1 low-data adaptation</h1>
<p>How much can a pretrained model learn from 25–500 PXR measurements?</p>
<p>Retrospective development-only study ·
{html.escape(summary["generated_at"])}</p>
<div class="notice"><strong>{counts["complete"]} completed /
{counts["failed"]} failed / {counts["pending"]} pending —
{len(expected)} planned tasks.</strong>
<p>{html.escape(summary["conclusion"])}</p></div>
<p>Frozen Nesso vectors or Morgan bits → selected readout → seed mean →
intervals calibrated within the acquired label budget → outer-test prediction.
The native anchor is unchanged pIC50-equivalent affinity, not cellular pEC50.</p>
<p>Primary score: assay-uncertainty-weighted MAE, SE floor
{protocol["se_floor"]}. N counts both fitting and calibration labels.
Partial panels are descriptive, not equal-coverage rankings.</p>
<details id="protocol"><summary>Protocol, model flows and limitations</summary>
<p>Weights are 1 / max(assay SE, {protocol["se_floor"]}); no curation multiplier.
N comprises fit plus ceil({protocol.get("calibration_fraction", 0.2)} × N)
calibration labels. Full reference is n_train = −1, separate in tables.</p>
<p>Paired effects use the same split, fold, draw, budget and test identities.
Error bars are observed min–max variation, <strong>not confidence intervals</strong>.
Repeated predictions are not independent compounds. Unique-compound Kish
 effective N describes weighting, not inferential sample size.</p>
<p>N=25 with five calibration labels cannot support a finite 90%
split-conformal interval. CSV retains infinite widths; JSON uses null with
unsupported_unbounded status. No distribution-free guarantee under chemical
shift is asserted. Seed SD is not a predictive interval.</p>
<p>Current singleton compound fraction: {singleton}.
Historical grouping: {historical}% (a different audit).
{html.escape(str(audit.get("caveat", "Consult the chemical-overlap audit.")))}
Neither grouping establishes strong series extrapolation.</p>
<p>All matrix methods use frozen upstream features. Head LoRA is final-readout
adaptation, not representation learning. Morgan-only LightGBM is not the
historical descriptor-rich baseline.</p>
<p>Cost is observed fit_seconds (selection, final fitting and replay), not
extraction or GPU cost. Parameter counts are per-seed trainable parameters;
trees count leaves and kNN has zero optimized parameters but stores examples.
Exact selection and fitting scope are in model_audits.json.</p>
<p>Novelty/calibration figures use the largest observed non-full budget,
not necessarily a complete common panel. All budgets remain downloadable.</p>
<h2>Model flows</h2><table>{flows}</table>
<h2>Upstream adapter engineering test</h2>
<p>{html.escape(adapter_text)}</p>
<p>Assay-label representation-adaptation performance remains unestablished.</p>
</details>
{figures}
<details id="data"><summary>Coverage, exact metrics and source data</summary>
<p>{links}</p>
<p>Verification failures count as failed, never successful evidence.
Global provenance errors: {html.escape(json.dumps(errors))}.
Unexpected task directories (not counted):
{html.escape(json.dumps(unexpected))}.</p>
<div class="scroll">{table}</div></details></html>"""
    (output / "index.html").write_text(page)
    artifacts = {
        str(p.relative_to(output)): _hash(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "artifact_manifest.json"
    }
    _write_json(output / "artifact_manifest.json", {"output_sha256": artifacts})
    return summary
