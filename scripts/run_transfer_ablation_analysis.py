#!/usr/bin/env python3
"""Run post-hoc transfer, probe, split, loss, and stacking diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nesso_pxr.chemistry import morgan_fingerprints
from nesso_pxr.comparison import (
    clustered_bootstrap,
    paired_difference_rows,
    regression_metrics,
)
from nesso_pxr.train_cached import (
    CachedHeadConfig,
    fit_cached_fold,
)
from nesso_pxr.transfer_ablation import (
    apply_linear_stack,
    crossfit_scalar_stacks,
    fit_convex_mae_weight,
    fit_full_ridge_after_group_selection,
    fit_linear_stack,
    mean_huber,
    nested_ridge_probe,
    similarity_quantiles,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = REPO_ROOT / "data" / "published"
COMPARISON = REPO_ROOT / "reports" / "model_comparison"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "transfer_ablation"
CHECKPOINT = REPO_ROOT / "models" / "nesso-1" / "v1.0.0" / "model.safetensors"

RIDGE_ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
LEARNING_RATES = (
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
)
HUBER_DELTAS = (0.25, 0.5, 1.0, 2.0, 10.0)
NEURAL_PROBE_LRS = (1e-4, 3e-4, 1e-3, 3e-3)
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "diagnostics", "probes", "sensitivity"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--skip-neural-probes", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metric_record(
    dataset: str,
    model: str,
    observed: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "model": model,
        "n": len(observed),
        **regression_metrics(
            np.asarray(observed, dtype=float),
            np.asarray(predicted, dtype=float),
        ),
    }


def load_bundle() -> tuple[pd.DataFrame, dict[str, Any]]:
    from safetensors.torch import load_file

    manifest = pd.read_csv(PUBLISHED / "modeling_manifest.csv")
    manifest = manifest.sort_values("feature_row", kind="stable").reset_index(drop=True)
    tensors = load_file(str(PUBLISHED / "nesso_features.safetensors"), device="cpu")
    if len(manifest) != len(tensors["member1_affinity_repr"]):
        raise ValueError("manifest and Nesso tensors have different row counts")
    if manifest["feature_row"].tolist() != list(range(len(manifest))):
        raise ValueError("feature rows are not contiguous")
    return manifest, tensors


def _nearest_development_similarity(manifest: pd.DataFrame) -> pd.DataFrame:
    from rdkit import DataStructs

    development = manifest.loc[manifest["modeling_role"].eq("development")].copy()
    lockbox = manifest.loc[manifest["modeling_role"].eq("lockbox")].copy()
    development = development.reset_index(drop=True)
    lockbox = lockbox.reset_index(drop=True)
    development_fps = morgan_fingerprints(
        development["identity_canonical_smiles"].tolist(), radius=2, n_bits=2048
    )
    lockbox_fps = morgan_fingerprints(
        lockbox["identity_canonical_smiles"].tolist(), radius=2, n_bits=2048
    )
    similarities: list[float] = []
    neighbor_indices: list[int] = []
    for fingerprint in lockbox_fps:
        values = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, development_fps),
            dtype=float,
        )
        index = int(np.argmax(values))
        similarities.append(float(values[index]))
        neighbor_indices.append(index)
    neighbors = development.iloc[neighbor_indices].reset_index(drop=True)
    result = lockbox[
        [
            "feature_row",
            "original_id",
            "identity_canonical_smiles",
            "prepared_smiles",
            "pEC50",
            "role",
        ]
    ].copy()
    result["nearest_development_similarity"] = similarities
    result["nearest_development_id"] = neighbors["original_id"].to_numpy()
    result["nearest_development_smiles"] = neighbors[
        "identity_canonical_smiles"
    ].to_numpy()
    result["nearest_development_pEC50"] = neighbors["pEC50"].to_numpy()
    return result


def _plot_similarity_distributions(
    nested: pd.DataFrame,
    lockbox: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    fold_values = [
        nested.loc[nested["fold"].eq(fold), "nearest_train_similarity"].to_numpy()
        for fold in sorted(nested["fold"].unique())
    ]
    axes[0].boxplot(
        fold_values, tick_labels=[str(i) for i in range(5)], showfliers=True
    )
    axes[0].axhline(0.6, color="crimson", linestyle="--", linewidth=1.3)
    axes[0].set(
        xlabel="Held-out chemical-family fold",
        ylabel="Nearest-training Morgan Tanimoto",
        title="Nested development",
    )
    bins = np.linspace(0.1, 1.0, 37)
    axes[1].hist(
        lockbox["nearest_development_similarity"],
        bins=bins,
        color="#315b7d",
        alpha=0.85,
    )
    axes[1].axvline(0.6, color="crimson", linestyle="--", linewidth=1.3)
    axes[1].set(
        xlabel="Nearest-development Morgan Tanimoto",
        ylabel="Lockbox compounds",
        title="Historical 790-compound lockbox",
    )
    fig.savefig(output_dir / "chemical_similarity_distributions.png", dpi=180)
    plt.close(fig)


def _similarity_bin(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        values,
        bins=[-np.inf, 0.3, 0.4, 0.5, 0.6, np.inf],
        labels=["<=0.3", "(0.3,0.4]", "(0.4,0.5]", "(0.5,0.6]", ">0.6"],
    )


def _smiles_preprocessing_audit(manifest: pd.DataFrame) -> dict[str, Any]:
    development = manifest.loc[manifest["modeling_role"].eq("development")]
    lockbox = manifest.loc[manifest["modeling_role"].eq("lockbox")]
    columns: dict[str, Any] = {}
    for column in ("raw_smiles", "identity_canonical_smiles", "prepared_smiles"):
        values = manifest[column].fillna("").astype(str)
        columns[column] = {
            "unique": int(values.nunique()),
            "multi_fragment_dot_count": int(
                values.str.contains(".", regex=False).sum()
            ),
            "duplicate_row_count": int(values.duplicated().sum()),
        }
    prepared_cross_fold = development.groupby("prepared_smiles")["fold"].nunique()
    identity_cross_fold = development.groupby("identity_canonical_smiles")[
        "fold"
    ].nunique()
    return {
        "rows": len(manifest),
        "columns": columns,
        "selection_reason_counts": {
            str(key): int(value)
            for key, value in manifest["selection_reason"].value_counts().items()
        },
        "formal_charge_counts": {
            str(key): int(value)
            for key, value in manifest["formal_charge"]
            .value_counts()
            .sort_index()
            .items()
        },
        "identity_groups_crossing_development_folds": int(
            (identity_cross_fold > 1).sum()
        ),
        "prepared_groups_crossing_development_folds": int(
            (prepared_cross_fold > 1).sum()
        ),
        "prepared_smiles_development_lockbox_overlap": int(
            len(set(development["prepared_smiles"]) & set(lockbox["prepared_smiles"]))
        ),
        "identity_processing": (
            "RDKit parse plus canonical isomeric SMILES; no repository salt "
            "stripper, tautomer canonicalizer, or neutralizer"
        ),
        "nesso_input_state": (
            "one precomputed top_solution_state per compound; alternate states were "
            "not used"
        ),
    }


def _summarize_expanded_nested_lr(output_dir: Path) -> None:
    """Compare an optional expanded-grid nested rerun with the historical grid."""

    expanded_path = output_dir / "expanded_lr_nested" / "nested_nesso_predictions.csv"
    historical_path = COMPARISON / "nested_nesso_predictions.csv"
    if not expanded_path.is_file():
        return
    expanded = pd.read_csv(expanded_path)[
        ["feature_row", "pEC50", "cluster_component", "nesso_pEC50"]
    ].rename(columns={"nesso_pEC50": "expanded_grid_pEC50"})
    historical = pd.read_csv(historical_path)[["feature_row", "nesso_pEC50"]].rename(
        columns={"nesso_pEC50": "historical_grid_pEC50"}
    )
    comparison = expanded.merge(
        historical,
        on="feature_row",
        how="inner",
        validate="one_to_one",
    )
    models = {
        "expanded_grid": "expanded_grid_pEC50",
        "historical_grid": "historical_grid_pEC50",
    }
    intervals, draws = clustered_bootstrap(
        comparison,
        models,
        group_column="cluster_component",
        observed_column="pEC50",
        replicates=2000,
        seed=20260812,
    )
    intervals.to_csv(
        output_dir / "expanded_lr_nested_bootstrap_intervals.csv", index=False
    )
    point_metrics = {
        model: regression_metrics(comparison["pEC50"], comparison[column])
        for model, column in models.items()
    }
    paired_difference_rows(
        draws,
        point_metrics,
        [("expanded_grid", "historical_grid")],
        metrics=("mae", "spearman"),
    ).to_csv(output_dir / "expanded_lr_nested_paired_differences.csv", index=False)


def run_diagnostics(output_dir: Path) -> None:
    manifest, _ = load_bundle()
    nested = pd.read_csv(COMPARISON / "nested_aligned_predictions.csv")
    common_lockbox = pd.read_csv(COMPARISON / "lockbox_aligned_predictions.csv")
    nesso_lockbox = pd.read_csv(PUBLISHED / "nesso_holdout_predictions.csv")
    all_lockbox = manifest.loc[manifest["modeling_role"].eq("lockbox")].merge(
        nesso_lockbox,
        on=["feature_row", "record_id", "original_id", "pEC50"],
        how="inner",
        validate="one_to_one",
    )
    if len(all_lockbox) != 790:
        raise ValueError("expected all 790 Nesso lockbox predictions")

    summary_rows: list[dict[str, Any]] = []
    development_models = {
        "Frozen Nesso-1": "frozen_pEC50",
        "Affine calibration": "affine_pEC50",
        "Fine-tuned Nesso-1": "nesso_pEC50",
        "2D LightGBM": "two_d_pEC50",
    }
    lockbox_models = {
        "Frozen Nesso-1": "frozen_pEC50",
        "Affine calibration": "affine_pEC50",
        "Fine-tuned Nesso-1": "nesso_pEC50",
        "2D LightGBM": "two_d_pEC50",
    }
    for model, column in development_models.items():
        summary_rows.append(
            _metric_record("nested_development", model, nested["pEC50"], nested[column])
        )
    for model, column in lockbox_models.items():
        summary_rows.append(
            _metric_record(
                "lockbox_common_714",
                model,
                common_lockbox["pEC50"],
                common_lockbox[column],
            )
        )
    all_columns = {
        "Frozen Nesso-1": "frozen_pEC50",
        "Affine calibration": "affine_pEC50",
        "Fine-tuned Nesso-1": "predicted_pEC50_ensemble",
    }
    for model, column in all_columns.items():
        summary_rows.append(
            _metric_record(
                "lockbox_all_790", model, all_lockbox["pEC50"], all_lockbox[column]
            )
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "development_lockbox_metrics.csv", index=False
    )

    lockbox_similarity = _nearest_development_similarity(manifest)
    lockbox_similarity.to_csv(output_dir / "lockbox_nearest_neighbors.csv", index=False)
    nested_quantiles = similarity_quantiles(
        nested,
        similarity_column="nearest_train_similarity",
        group_columns=["fold"],
    )
    nested_quantiles.to_csv(
        output_dir / "nested_fold_similarity_quantiles.csv", index=False
    )
    similarity_quantiles(
        lockbox_similarity.assign(dataset="lockbox_all_790"),
        similarity_column="nearest_development_similarity",
        group_columns=["dataset"],
    ).to_csv(output_dir / "lockbox_similarity_quantiles.csv", index=False)

    all_lockbox = all_lockbox.merge(
        lockbox_similarity[["feature_row", "nearest_development_similarity"]],
        on="feature_row",
        how="left",
        validate="one_to_one",
    )
    all_lockbox["similarity_bin"] = _similarity_bin(
        all_lockbox["nearest_development_similarity"]
    )
    common_with_similarity = common_lockbox.copy()
    common_with_similarity["similarity_bin"] = _similarity_bin(
        common_with_similarity["nearest_train_similarity"]
    )
    stratified_rows: list[dict[str, Any]] = []
    for similarity_bin, subset in all_lockbox.groupby(
        "similarity_bin", observed=True, sort=True
    ):
        for model, column in all_columns.items():
            row = _metric_record(
                "lockbox_all_790", model, subset["pEC50"], subset[column]
            )
            row["similarity_bin"] = str(similarity_bin)
            stratified_rows.append(row)
    for similarity_bin, subset in common_with_similarity.groupby(
        "similarity_bin", observed=True, sort=True
    ):
        for model, column in lockbox_models.items():
            row = _metric_record(
                "lockbox_common_714", model, subset["pEC50"], subset[column]
            )
            row["similarity_bin"] = str(similarity_bin)
            stratified_rows.append(row)
    pd.DataFrame(stratified_rows).to_csv(
        output_dir / "lockbox_similarity_stratified_metrics.csv", index=False
    )
    _plot_similarity_distributions(nested, lockbox_similarity, output_dir)
    _write_json(
        output_dir / "smiles_preprocessing_audit.json",
        _smiles_preprocessing_audit(manifest),
    )
    _summarize_expanded_nested_lr(output_dir)


def _load_development_tensors() -> tuple[
    pd.DataFrame, pd.DataFrame, Any, Any, Any, Any
]:
    manifest, tensors = load_bundle()
    development = manifest.loc[manifest["modeling_role"].eq("development")].copy()
    development = development.sort_values("feature_row", kind="stable").reset_index(
        drop=True
    )
    lockbox = manifest.loc[manifest["modeling_role"].eq("lockbox")].copy()
    lockbox = lockbox.sort_values("feature_row", kind="stable").reset_index(drop=True)
    development["fold"] = development["fold"].astype(int)
    dev_positions = development["feature_row"].to_numpy(dtype=int)
    lock_positions = lockbox["feature_row"].to_numpy(dtype=int)
    member1_dev = tensors["member1_affinity_repr"][dev_positions].float()
    member2_dev = tensors["member2_affinity_repr"][dev_positions].float()
    member1_lock = tensors["member1_affinity_repr"][lock_positions].float()
    member2_lock = tensors["member2_affinity_repr"][lock_positions].float()
    return development, lockbox, member1_dev, member2_dev, member1_lock, member2_lock


def _representation_variants(
    member1: np.ndarray, member2: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "ridge_member1_384d": member1,
        "ridge_member2_384d": member2,
        "ridge_mean_384d": (member1 + member2) / 2.0,
        "ridge_concat_768d": np.concatenate([member1, member2], axis=1),
    }


def _run_ridge_probes(
    development: pd.DataFrame,
    lockbox: pd.DataFrame,
    member1_dev: np.ndarray,
    member2_dev: np.ndarray,
    member1_lock: np.ndarray,
    member2_lock: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    y_dev = development["pEC50"].to_numpy(dtype=float)
    y_lock = lockbox["pEC50"].to_numpy(dtype=float)
    folds = development["fold"].to_numpy(dtype=int)
    development_variants = _representation_variants(member1_dev, member2_dev)
    lockbox_variants = _representation_variants(member1_lock, member2_lock)
    metrics: list[dict[str, Any]] = []
    selections: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    full_selection_rows: list[dict[str, Any]] = []
    for name, x_dev in development_variants.items():
        result = nested_ridge_probe(x_dev, y_dev, folds, RIDGE_ALPHAS)
        predictions = result.predictions.rename(
            columns={"observed": "pEC50", "predicted": "predicted_pEC50"}
        )
        predictions["model"] = name
        predictions["feature_row"] = development["feature_row"].to_numpy()
        predictions["dataset"] = "nested_development"
        prediction_frames.append(predictions)
        selection = result.selections.copy()
        selection["model"] = name
        selections.append(selection)
        metrics.append(
            _metric_record(
                "nested_development",
                name,
                predictions["pEC50"],
                predictions["predicted_pEC50"],
            )
        )
        lock_prediction, selected_alpha, screen = fit_full_ridge_after_group_selection(
            x_dev,
            y_dev,
            folds,
            lockbox_variants[name],
            RIDGE_ALPHAS,
        )
        lock_predictions = pd.DataFrame(
            {
                "feature_row": lockbox["feature_row"].to_numpy(),
                "pEC50": y_lock,
                "predicted_pEC50": lock_prediction,
                "model": name,
                "dataset": "lockbox_all_790",
                "fold": np.nan,
            }
        )
        prediction_frames.append(lock_predictions)
        metrics.append(_metric_record("lockbox_all_790", name, y_lock, lock_prediction))
        selected_screen = screen.loc[screen["alpha"].eq(selected_alpha)].iloc[0]
        full_selection_rows.append(
            {
                "model": name,
                "selected_alpha": selected_alpha,
                "development_oof_mae": float(selected_screen["mae"]),
                "development_oof_spearman": float(selected_screen["spearman"]),
            }
        )
    pd.concat(selections, ignore_index=True).to_csv(
        output_dir / "ridge_probe_outer_selections.csv", index=False
    )
    pd.DataFrame(full_selection_rows).to_csv(
        output_dir / "ridge_probe_full_development_selections.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "ridge_probe_predictions.csv", index=False
    )
    return pd.DataFrame(metrics)


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_twin_probe(hidden_layers: tuple[int, ...]) -> Any:
    from torch import nn

    def make_member() -> nn.Sequential:
        dimensions = (384, *hidden_layers, 1)
        layers: list[nn.Module] = []
        for position, (input_dim, output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if position < len(dimensions) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    class TwinProbe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.member1 = make_member()
            self.member2 = make_member()

        def forward(self, x1: Any, x2: Any) -> tuple[Any, Any, Any]:
            prediction1 = self.member1(x1).squeeze(-1)
            prediction2 = self.member2(x2).squeeze(-1)
            return prediction1, prediction2, (prediction1 + prediction2) / 2.0

    return TwinProbe()


def _probe_loss(
    prediction1: Any,
    prediction2: Any,
    ensemble: Any,
    target: Any,
    *,
    delta: float = 0.5,
) -> Any:
    from torch.nn import functional

    return (
        0.5 * functional.huber_loss(ensemble, target, delta=delta)
        + 0.25 * functional.huber_loss(prediction1, target, delta=delta)
        + 0.25 * functional.huber_loss(prediction2, target, delta=delta)
    )


def _standardize_pair(
    train1: np.ndarray,
    train2: np.ndarray,
    evaluation1: np.ndarray,
    evaluation2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean1 = np.mean(train1, axis=0, keepdims=True)
    mean2 = np.mean(train2, axis=0, keepdims=True)
    scale1 = np.std(train1, axis=0, keepdims=True)
    scale2 = np.std(train2, axis=0, keepdims=True)
    scale1[scale1 < 1e-7] = 1.0
    scale2[scale2 < 1e-7] = 1.0
    return (
        ((train1 - mean1) / scale1).astype(np.float32),
        ((train2 - mean2) / scale2).astype(np.float32),
        ((evaluation1 - mean1) / scale1).astype(np.float32),
        ((evaluation2 - mean2) / scale2).astype(np.float32),
    )


def _fit_neural_probe(
    train1: np.ndarray,
    train2: np.ndarray,
    y_train: np.ndarray,
    evaluation1: np.ndarray,
    evaluation2: np.ndarray,
    *,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    seed: int,
    device: str,
    y_evaluation: np.ndarray | None = None,
    fixed_epochs: int | None = None,
    max_epochs: int = 40,
    patience: int = 6,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if fixed_epochs is None and y_evaluation is None:
        raise ValueError("early stopping requires evaluation labels")
    _set_seed(seed)
    train1, train2, evaluation1, evaluation2 = _standardize_pair(
        train1, train2, evaluation1, evaluation2
    )
    target = np.asarray(y_train, dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(train1),
        torch.from_numpy(train2),
        torch.from_numpy(target),
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    model = _make_twin_probe(hidden_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    evaluation_tensor1 = torch.from_numpy(evaluation1).to(device)
    evaluation_tensor2 = torch.from_numpy(evaluation2).to(device)
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_mae = math.inf
    waiting = 0
    history: list[dict[str, float]] = []
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for batch1, batch2, batch_target in loader:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction1, prediction2, ensemble = model(batch1, batch2)
            loss = _probe_loss(prediction1, prediction2, ensemble, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_target)
            examples += len(batch_target)
        model.eval()
        with torch.inference_mode():
            _, _, evaluation_prediction = model(evaluation_tensor1, evaluation_tensor2)
        prediction_array = evaluation_prediction.detach().cpu().numpy()
        record: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": total_loss / examples,
        }
        if y_evaluation is not None:
            metrics = regression_metrics(y_evaluation, prediction_array)
            record["validation_mae"] = metrics["mae"]
            record["validation_spearman"] = metrics["spearman"]
        history.append(record)
        if fixed_epochs is not None:
            continue
        if record["validation_mae"] < best_mae - 0.002:
            best_mae = record["validation_mae"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("neural probe did not produce a checkpoint")
        model.load_state_dict(best_state, strict=True)
        best_epoch_value = best_epoch
    else:
        best_epoch_value = fixed_epochs
    model.to(device).eval()
    with torch.inference_mode():
        _, _, final_prediction = model(evaluation_tensor1, evaluation_tensor2)
    return (
        final_prediction.detach().cpu().numpy(),
        int(best_epoch_value),
        pd.DataFrame(history),
    )


def _run_neural_probe_architecture(
    name: str,
    hidden_layers: tuple[int, ...],
    member1_dev: np.ndarray,
    member2_dev: np.ndarray,
    y_dev: np.ndarray,
    folds: np.ndarray,
    member1_lock: np.ndarray,
    member2_lock: np.ndarray,
    y_lock: np.ndarray,
    *,
    output_dir: Path,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    screen_rows: list[dict[str, Any]] = []
    seed42_predictions: dict[float, np.ndarray] = {}
    seed42_epochs: dict[float, list[int]] = {}
    for learning_rate in NEURAL_PROBE_LRS:
        oof = np.full(len(y_dev), np.nan, dtype=float)
        epochs: list[int] = []
        for fold in sorted(np.unique(folds)):
            validation = folds == fold
            training = ~validation
            prediction, best_epoch, _ = _fit_neural_probe(
                member1_dev[training],
                member2_dev[training],
                y_dev[training],
                member1_dev[validation],
                member2_dev[validation],
                hidden_layers=hidden_layers,
                learning_rate=learning_rate,
                seed=42,
                device=device,
                y_evaluation=y_dev[validation],
            )
            oof[validation] = prediction
            epochs.append(best_epoch)
        metrics = regression_metrics(y_dev, oof)
        screen_rows.append(
            {
                "probe": name,
                "learning_rate": learning_rate,
                "seed": 42,
                "best_epochs": ";".join(str(value) for value in epochs),
                **metrics,
            }
        )
        seed42_predictions[learning_rate] = oof
        seed42_epochs[learning_rate] = epochs
    screen = pd.DataFrame(screen_rows)
    selected_lr = float(
        screen.sort_values(
            ["mae", "spearman"], ascending=[True, False], kind="stable"
        ).iloc[0]["learning_rate"]
    )
    seed_predictions = {42: seed42_predictions[selected_lr]}
    all_epochs = list(seed42_epochs[selected_lr])
    for seed in (43, 44):
        oof = np.full(len(y_dev), np.nan, dtype=float)
        for fold in sorted(np.unique(folds)):
            validation = folds == fold
            training = ~validation
            prediction, best_epoch, _ = _fit_neural_probe(
                member1_dev[training],
                member2_dev[training],
                y_dev[training],
                member1_dev[validation],
                member2_dev[validation],
                hidden_layers=hidden_layers,
                learning_rate=selected_lr,
                seed=seed,
                device=device,
                y_evaluation=y_dev[validation],
            )
            oof[validation] = prediction
            all_epochs.append(best_epoch)
        seed_predictions[seed] = oof
    ensemble_oof = np.mean(np.stack(list(seed_predictions.values())), axis=0)
    fixed_epochs = max(1, int(np.rint(np.median(all_epochs))))
    lock_predictions: list[np.ndarray] = []
    for seed in SEEDS:
        prediction, _, _ = _fit_neural_probe(
            member1_dev,
            member2_dev,
            y_dev,
            member1_lock,
            member2_lock,
            hidden_layers=hidden_layers,
            learning_rate=selected_lr,
            seed=seed,
            device=device,
            fixed_epochs=fixed_epochs,
        )
        lock_predictions.append(prediction)
    ensemble_lock = np.mean(np.stack(lock_predictions), axis=0)
    predictions = pd.DataFrame(
        {"fold": folds, "pEC50": y_dev, "predicted_pEC50": ensemble_oof}
    )
    lockbox_predictions = pd.DataFrame(
        {"pEC50": y_lock, "predicted_pEC50": ensemble_lock}
    )
    selected = pd.DataFrame(
        [
            {
                "probe": name,
                "selected_learning_rate": selected_lr,
                "fixed_all_development_epochs": fixed_epochs,
                "selection_note": (
                    "post-hoc five-fold screen; seeds 43/44 repeat selected LR"
                ),
            }
        ]
    )
    predictions.to_csv(output_dir / f"{name}_oof_predictions.csv", index=False)
    lockbox_predictions.to_csv(
        output_dir / f"{name}_lockbox_predictions.csv", index=False
    )
    metrics = pd.DataFrame(
        [
            _metric_record("nested_development", name, y_dev, ensemble_oof),
            _metric_record("lockbox_all_790", name, y_lock, ensemble_lock),
        ]
    )
    return screen, selected, metrics


def _run_stacking_and_residual_probe(
    tensors: dict[str, Any],
    output_dir: Path,
) -> None:
    nested = pd.read_csv(COMPARISON / "nested_aligned_predictions.csv")
    common_lockbox = pd.read_csv(COMPARISON / "lockbox_aligned_predictions.csv")
    nested = nested.sort_values("feature_row", kind="stable").reset_index(drop=True)
    common_lockbox = common_lockbox.sort_values(
        "feature_row", kind="stable"
    ).reset_index(drop=True)
    stacked, fold_coefficients = crossfit_scalar_stacks(
        nested,
        observed_column="pEC50",
        first_column="nesso_pEC50",
        second_column="two_d_pEC50",
        group_column="fold",
    )
    stacked["equal_blend"] = (stacked["nesso_pEC50"] + stacked["two_d_pEC50"]) / 2.0

    y_dev = nested["pEC50"].to_numpy(dtype=float)
    nesso_dev = nested["nesso_pEC50"].to_numpy(dtype=float)
    two_d_dev = nested["two_d_pEC50"].to_numpy(dtype=float)
    full_weight = fit_convex_mae_weight(y_dev, nesso_dev, two_d_dev)
    full_coefficients = fit_linear_stack(y_dev, nesso_dev, two_d_dev)
    nesso_lock = common_lockbox["nesso_pEC50"].to_numpy(dtype=float)
    two_d_lock = common_lockbox["two_d_pEC50"].to_numpy(dtype=float)
    common_lockbox["equal_blend"] = (nesso_lock + two_d_lock) / 2.0
    common_lockbox["convex_stack"] = (
        full_weight * nesso_lock + (1.0 - full_weight) * two_d_lock
    )
    common_lockbox["linear_stack"] = apply_linear_stack(
        full_coefficients, nesso_lock, two_d_lock
    )

    dev_positions = nested["feature_row"].to_numpy(dtype=int)
    lock_positions = common_lockbox["feature_row"].to_numpy(dtype=int)
    x_dev = np.concatenate(
        [
            tensors["member1_affinity_repr"][dev_positions].numpy(),
            tensors["member2_affinity_repr"][dev_positions].numpy(),
        ],
        axis=1,
    )
    x_lock = np.concatenate(
        [
            tensors["member1_affinity_repr"][lock_positions].numpy(),
            tensors["member2_affinity_repr"][lock_positions].numpy(),
        ],
        axis=1,
    )
    residual = y_dev - two_d_dev
    residual_result = nested_ridge_probe(
        x_dev,
        residual,
        nested["fold"].to_numpy(dtype=int),
        RIDGE_ALPHAS,
    )
    stacked["representation_residual_ridge"] = (
        two_d_dev + residual_result.predictions["predicted"].to_numpy()
    )
    lock_residual, residual_alpha, residual_screen = (
        fit_full_ridge_after_group_selection(
            x_dev,
            residual,
            nested["fold"].to_numpy(dtype=int),
            x_lock,
            RIDGE_ALPHAS,
        )
    )
    common_lockbox["representation_residual_ridge"] = two_d_lock + lock_residual

    model_columns = {
        "fine_tuned_nesso": "nesso_pEC50",
        "two_d_lightgbm": "two_d_pEC50",
        "equal_blend": "equal_blend",
        "convex_mae_stack": "convex_stack",
        "unconstrained_linear_stack": "linear_stack",
        "nesso_repr_residual_ridge": "representation_residual_ridge",
    }
    metric_rows: list[dict[str, Any]] = []
    for model, column in model_columns.items():
        metric_rows.append(
            _metric_record(
                "crossfit_development",
                model,
                stacked["pEC50"],
                stacked[column],
            )
        )
        metric_rows.append(
            _metric_record(
                "lockbox_common_714",
                model,
                common_lockbox["pEC50"],
                common_lockbox[column],
            )
        )
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "stacking_and_residual_metrics.csv", index=False
    )
    stacked.to_csv(output_dir / "stacked_development_predictions.csv", index=False)
    common_lockbox.to_csv(output_dir / "stacked_lockbox_predictions.csv", index=False)

    bootstrap_models = {
        "two_d_lightgbm": "two_d_pEC50",
        "convex_mae_stack": "convex_stack",
        "unconstrained_linear_stack": "linear_stack",
        "nesso_repr_residual_ridge": "representation_residual_ridge",
    }
    bootstrap_intervals, bootstrap_draws = clustered_bootstrap(
        common_lockbox,
        bootstrap_models,
        group_column="bootstrap_group",
        observed_column="pEC50",
        replicates=2000,
        seed=20260812,
    )
    bootstrap_intervals.insert(0, "dataset", "lockbox_common_714")
    bootstrap_intervals.to_csv(
        output_dir / "stacking_lockbox_cluster_bootstrap_intervals.csv", index=False
    )
    point_metrics = {
        model: regression_metrics(common_lockbox["pEC50"], common_lockbox[column])
        for model, column in bootstrap_models.items()
    }
    comparisons = [
        (model, "two_d_lightgbm")
        for model in bootstrap_models
        if model != "two_d_lightgbm"
    ]
    paired_difference_rows(
        bootstrap_draws,
        point_metrics,
        comparisons,
        metrics=("mae", "spearman"),
    ).to_csv(output_dir / "stacking_lockbox_paired_differences.csv", index=False)
    residual_result.selections.to_csv(
        output_dir / "residual_ridge_outer_selections.csv", index=False
    )
    fold_coefficients.to_csv(
        output_dir / "stacking_crossfit_coefficients.csv", index=False
    )
    selected_residual = residual_screen.loc[
        residual_screen["alpha"].eq(residual_alpha)
    ].iloc[0]
    full_row = {
        "convex_nesso_weight": full_weight,
        "convex_2d_weight": 1.0 - full_weight,
        "linear_intercept": float(full_coefficients[0]),
        "linear_nesso_coefficient": float(full_coefficients[1]),
        "linear_2d_coefficient": float(full_coefficients[2]),
        "residual_ridge_alpha": residual_alpha,
        "residual_ridge_development_oof_mae": float(selected_residual["mae"]),
        "method_note": (
            "Meta-models fit development OOF predictions and are evaluated "
            "retrospectively on the previously opened common lockbox."
        ),
    }
    _write_json(output_dir / "stacking_full_development_fit.json", full_row)


def run_probes(output_dir: Path, *, device: str, skip_neural: bool) -> None:
    (
        development,
        lockbox,
        member1_dev_tensor,
        member2_dev_tensor,
        member1_lock_tensor,
        member2_lock_tensor,
    ) = _load_development_tensors()
    member1_dev = member1_dev_tensor.numpy()
    member2_dev = member2_dev_tensor.numpy()
    member1_lock = member1_lock_tensor.numpy()
    member2_lock = member2_lock_tensor.numpy()
    ridge_metrics = _run_ridge_probes(
        development,
        lockbox,
        member1_dev,
        member2_dev,
        member1_lock,
        member2_lock,
        output_dir,
    )
    metric_frames = [ridge_metrics]
    if not skip_neural:
        screens: list[pd.DataFrame] = []
        selections: list[pd.DataFrame] = []
        for name, hidden_layers in {
            "random_one_hidden_twin_mlp": (384,),
            "random_two_hidden_twin_mlp": (384, 384),
        }.items():
            screen, selection, metrics = _run_neural_probe_architecture(
                name,
                hidden_layers,
                member1_dev,
                member2_dev,
                development["pEC50"].to_numpy(dtype=float),
                development["fold"].to_numpy(dtype=int),
                member1_lock,
                member2_lock,
                lockbox["pEC50"].to_numpy(dtype=float),
                output_dir=output_dir,
                device=device,
            )
            screens.append(screen)
            selections.append(selection)
            metric_frames.append(metrics)
        pd.concat(screens, ignore_index=True).to_csv(
            output_dir / "neural_probe_lr_screen.csv", index=False
        )
        pd.concat(selections, ignore_index=True).to_csv(
            output_dir / "neural_probe_selections.csv", index=False
        )
    pd.concat(metric_frames, ignore_index=True).to_csv(
        output_dir / "representation_probe_metrics.csv", index=False
    )
    _, tensors = load_bundle()
    _run_stacking_and_residual_probe(tensors, output_dir)


def _run_pretrained_head_cv(
    development: pd.DataFrame,
    member1: Any,
    member2: Any,
    checkpoint: Path,
    config: CachedHeadConfig,
    *,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    best_epochs: list[int] = []
    for fold in sorted(development["fold"].unique()):
        predictions, history, checkpoint_info = fit_cached_fold(
            member1,
            member2,
            development,
            checkpoint,
            int(fold),
            config,
            device=device,
        )
        prediction_frames.append(predictions)
        history_frames.append(history)
        best_epochs.append(int(checkpoint_info["best_epoch"]))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    history = pd.concat(history_frames, ignore_index=True)
    observed = predictions["pEC50"].to_numpy(dtype=float)
    predicted = predictions["predicted_pEC50"].to_numpy(dtype=float)
    metrics = regression_metrics(observed, predicted)
    active = observed > 6.0
    summary = {
        **asdict(config),
        **metrics,
        "validation_huber": mean_huber(observed, predicted, delta=config.huber_delta),
        "best_epochs": ";".join(str(value) for value in best_epochs),
        "active_n": int(active.sum()),
        "active_mae": float(np.mean(np.abs(predicted[active] - observed[active]))),
        "active_predicted_above_6": int(np.sum(predicted[active] > 6.0)),
        "active_recall_at_6": float(np.mean(predicted[active] > 6.0)),
        "prediction_maximum": float(np.max(predicted)),
    }
    return predictions, history, summary


def _target_coordinate_equivalence() -> dict[str, float]:
    import torch
    from torch.nn import functional

    torch.manual_seed(17)
    p_ec50 = torch.tensor([2.1, 4.7, 5.3, 6.4], dtype=torch.float64)
    member1 = torch.randn(4, dtype=torch.float64, requires_grad=True)
    member2 = torch.randn(4, dtype=torch.float64, requires_grad=True)
    ensemble = (member1 + member2) / 2.0
    nesso_target = 6.0 - p_ec50
    nesso_loss = (
        0.5 * functional.huber_loss(ensemble, nesso_target, delta=0.5)
        + 0.25 * functional.huber_loss(member1, nesso_target, delta=0.5)
        + 0.25 * functional.huber_loss(member2, nesso_target, delta=0.5)
    )
    nesso_gradients = torch.autograd.grad(
        nesso_loss, (member1, member2), retain_graph=True
    )
    direct_loss = (
        0.5 * functional.huber_loss(6.0 - ensemble, p_ec50, delta=0.5)
        + 0.25 * functional.huber_loss(6.0 - member1, p_ec50, delta=0.5)
        + 0.25 * functional.huber_loss(6.0 - member2, p_ec50, delta=0.5)
    )
    direct_gradients = torch.autograd.grad(direct_loss, (member1, member2))
    return {
        "nesso_coordinate_loss": float(nesso_loss.detach()),
        "direct_pEC50_coordinate_loss": float(direct_loss.detach()),
        "loss_absolute_difference": float(torch.abs(nesso_loss - direct_loss).detach()),
        "member1_gradient_max_absolute_difference": float(
            torch.max(torch.abs(nesso_gradients[0] - direct_gradients[0]))
        ),
        "member2_gradient_max_absolute_difference": float(
            torch.max(torch.abs(nesso_gradients[1] - direct_gradients[1]))
        ),
    }


def _plot_sensitivity(
    learning_rate_screen: pd.DataFrame,
    huber_screen: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(
        learning_rate_screen["learning_rate"],
        learning_rate_screen["validation_huber"],
        marker="o",
        color="#24557a",
        label="Huber validation loss",
    )
    axes[0].set_xscale("log")
    axes[0].set(
        xlabel="AdamW learning rate",
        ylabel="OOF validation Huber loss",
        title="Pretrained-head learning-rate sensitivity",
    )
    best = learning_rate_screen.loc[learning_rate_screen["mae"].idxmin()]
    axes[0].axvline(
        best["learning_rate"], color="crimson", linestyle="--", linewidth=1.2
    )
    second_axis = axes[0].twinx()
    second_axis.plot(
        learning_rate_screen["learning_rate"],
        learning_rate_screen["mae"],
        marker="s",
        color="#bd6b2c",
        label="MAE",
    )
    second_axis.set_ylabel("OOF pEC50 MAE")

    axes[1].plot(
        huber_screen["huber_delta"],
        huber_screen["mae"],
        marker="o",
        label="All development",
    )
    axes[1].plot(
        huber_screen["huber_delta"],
        huber_screen["active_mae"],
        marker="s",
        label="Observed pEC50 > 6",
    )
    axes[1].set_xscale("log")
    axes[1].set(
        xlabel="Huber delta (pEC50 units)",
        ylabel="OOF MAE",
        title="Robust-loss sensitivity",
    )
    axes[1].legend()
    fig.savefig(output_dir / "learning_rate_and_huber_sensitivity.png", dpi=180)
    plt.close(fig)


def run_sensitivity(
    output_dir: Path,
    *,
    device: str,
    checkpoint: Path,
) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"missing Nesso checkpoint: run scripts/download_nesso_checkpoint.py: "
            f"{checkpoint}"
        )
    (
        development,
        _,
        member1_dev,
        member2_dev,
        _,
        _,
    ) = _load_development_tensors()
    learning_rate_rows: list[dict[str, Any]] = []
    learning_rate_predictions: list[pd.DataFrame] = []
    learning_rate_histories: list[pd.DataFrame] = []
    for learning_rate in LEARNING_RATES:
        config = CachedHeadConfig(
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_epochs=30,
            patience=5,
            huber_delta=0.5,
            seed=42,
        )
        predictions, history, summary = _run_pretrained_head_cv(
            development,
            member1_dev,
            member2_dev,
            checkpoint,
            config,
            device=device,
        )
        predictions["learning_rate"] = learning_rate
        history["screen_learning_rate"] = learning_rate
        learning_rate_predictions.append(predictions)
        learning_rate_histories.append(history)
        learning_rate_rows.append(summary)
        print(f"completed pretrained LR {learning_rate:g}", flush=True)
    learning_rate_screen = pd.DataFrame(learning_rate_rows).sort_values("learning_rate")
    selected_lr = float(
        learning_rate_screen.sort_values(
            ["mae", "spearman"], ascending=[True, False], kind="stable"
        ).iloc[0]["learning_rate"]
    )

    huber_rows: list[dict[str, Any]] = []
    huber_predictions: list[pd.DataFrame] = []
    huber_histories: list[pd.DataFrame] = []
    for delta in HUBER_DELTAS:
        config = CachedHeadConfig(
            learning_rate=selected_lr,
            weight_decay=0.0,
            max_epochs=30,
            patience=5,
            huber_delta=delta,
            seed=42,
        )
        predictions, history, summary = _run_pretrained_head_cv(
            development,
            member1_dev,
            member2_dev,
            checkpoint,
            config,
            device=device,
        )
        predictions["huber_delta"] = delta
        history["screen_huber_delta"] = delta
        huber_predictions.append(predictions)
        huber_histories.append(history)
        huber_rows.append(summary)
        print(f"completed Huber delta {delta:g}", flush=True)
    huber_screen = pd.DataFrame(huber_rows).sort_values("huber_delta")
    learning_rate_screen.to_csv(
        output_dir / "learning_rate_sensitivity.csv", index=False
    )
    huber_screen.to_csv(output_dir / "huber_delta_sensitivity.csv", index=False)
    pd.concat(learning_rate_predictions, ignore_index=True).to_csv(
        output_dir / "learning_rate_oof_predictions.csv", index=False
    )
    pd.concat(huber_predictions, ignore_index=True).to_csv(
        output_dir / "huber_delta_oof_predictions.csv", index=False
    )
    pd.concat(learning_rate_histories, ignore_index=True).to_csv(
        output_dir / "learning_rate_training_history.csv", index=False
    )
    pd.concat(huber_histories, ignore_index=True).to_csv(
        output_dir / "huber_delta_training_history.csv", index=False
    )
    _plot_sensitivity(learning_rate_screen, huber_screen, output_dir)

    labels = development["pEC50"].to_numpy(dtype=float)
    standard_error = development["pEC50_standard_error"].to_numpy(dtype=float)
    label_context = {
        "pEC50_minimum": float(np.min(labels)),
        "pEC50_q25": float(np.quantile(labels, 0.25)),
        "pEC50_median": float(np.median(labels)),
        "pEC50_q75": float(np.quantile(labels, 0.75)),
        "pEC50_maximum": float(np.max(labels)),
        "pEC50_iqr": float(np.quantile(labels, 0.75) - np.quantile(labels, 0.25)),
        "assay_standard_error_median": float(np.median(standard_error)),
        "assay_standard_error_q75": float(np.quantile(standard_error, 0.75)),
        "huber_delta_0_5_fold_error": float(10**0.5),
        "huber_delta_0_5_over_median_standard_error": float(
            0.5 / np.median(standard_error)
        ),
        "selected_posthoc_learning_rate": selected_lr,
        "interpretation": (
            "Delta 0.5 is a fixed heuristic, not an assay-derived estimate. "
            "The expanded screens are post-hoc sensitivity analyses."
        ),
    }
    _write_json(output_dir / "loss_scale_context.json", label_context)
    _write_json(
        output_dir / "target_coordinate_equivalence.json",
        _target_coordinate_equivalence(),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "analysis_scope.json",
        {
            "status": "post_hoc_sensitivity",
            "lockbox_status": "previously_opened_retrospective_evaluation",
            "stage": args.stage,
            "device": args.device,
        },
    )
    if args.stage in {"all", "diagnostics"}:
        run_diagnostics(args.output_dir)
    if args.stage in {"all", "sensitivity"}:
        run_sensitivity(
            args.output_dir,
            device=args.device,
            checkpoint=args.checkpoint,
        )
    if args.stage in {"all", "probes"}:
        run_probes(
            args.output_dir,
            device=args.device,
            skip_neural=args.skip_neural_probes,
        )


if __name__ == "__main__":
    main()
