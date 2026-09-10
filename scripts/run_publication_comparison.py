#!/usr/bin/env python3
"""Run leakage-safe nested Nesso/2D comparisons and publication diagnostics."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nesso_pxr.chemistry import bemis_murcko_scaffold, scaffold_group_key
from nesso_pxr.comparison import (
    clustered_bootstrap,
    error_complementarity,
    inverse_se_weights,
    metric_rows,
    paired_difference_rows,
    regression_metrics,
)
from nesso_pxr.train_cached import (
    CachedHeadConfig,
    _sample_weights,
    _set_seed,
    combined_huber_loss,
    fit_cached_fold,
    load_pretrained_heads,
    make_twin_heads,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_INPUTS = REPO_ROOT / "data" / "published"
PUBLISHED_REPORT = REPO_ROOT / "reports" / "model_comparison"
NESSO_CHECKPOINT = REPO_ROOT / "models" / "nesso-1" / "v1.0.0" / "model.safetensors"

E2_LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4)
WEIGHT_DECAYS = (0.0, 1e-4, 1e-3)
E3_UNCERTAINTY_FLOORS = (0.10, 0.15, 0.20)
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "nesso", "2d", "analyze"),
        default="all",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PUBLISHED_INPUTS / "nesso_features.safetensors",
    )
    parser.add_argument(
        "--feature-metadata",
        type=Path,
        default=PUBLISHED_INPUTS / "nesso_feature_metadata.csv",
    )
    parser.add_argument(
        "--modeling-manifest",
        type=Path,
        default=PUBLISHED_INPUTS / "modeling_manifest.csv",
    )
    parser.add_argument(
        "--checkpoint-weights",
        type=Path,
        default=NESSO_CHECKPOINT,
    )
    parser.add_argument(
        "--reduced-2d-features",
        type=Path,
        default=PUBLISHED_INPUTS / "reduced_2d_features.parquet",
    )
    parser.add_argument(
        "--historical-2d-holdout",
        type=Path,
        default=PUBLISHED_INPUTS / "established_2d_holdout_predictions.csv",
    )
    parser.add_argument(
        "--historical-2d-challenge",
        type=Path,
        default=PUBLISHED_INPUTS / "established_2d_challenge_predictions.csv",
    )
    parser.add_argument(
        "--challenge-nesso",
        type=Path,
        default=PUBLISHED_INPUTS / "nesso_challenge_predictions.csv",
    )
    parser.add_argument(
        "--nesso-holdout",
        type=Path,
        default=PUBLISHED_INPUTS / "nesso_holdout_predictions.csv",
    )
    parser.add_argument(
        "--cliff-pairs",
        type=Path,
        default=PUBLISHED_INPUTS / "activity_cliff_pairs.csv",
    )
    parser.add_argument(
        "--analysis-input-dir",
        type=Path,
        default=PUBLISHED_REPORT,
        help="directory containing the nested prediction tables for --stage analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PUBLISHED_REPORT,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--nesso-learning-rates",
        type=float,
        nargs="+",
        default=E2_LEARNING_RATES,
        help=("Nesso E2 learning-rate grid; defaults to the historical bounded grid"),
    )
    parser.add_argument("--lgbm-threads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def portable_path(path: Path) -> str:
    """Return repository-relative provenance without leaking host paths."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def load_development_manifest(args: argparse.Namespace) -> pd.DataFrame:
    metadata = pd.read_csv(args.feature_metadata).sort_values("feature_row")
    modeling = pd.read_csv(args.modeling_manifest).sort_values("feature_row")
    if metadata["record_id"].tolist() != modeling["record_id"].tolist():
        raise ValueError("feature metadata and modeling manifest are not aligned")
    frame = modeling.loc[modeling["modeling_role"].eq("development")].copy()
    frame = frame.sort_values("feature_row", kind="stable").reset_index(drop=True)
    if len(frame) != 3344 or frame["fold"].nunique() != 5:
        raise ValueError("unexpected all-curated development cohort")
    frame["fold"] = frame["fold"].astype(int)
    return frame


def fit_fixed_nesso(
    member1: Any,
    member2: Any,
    frame: pd.DataFrame,
    checkpoint_weights: Path,
    config: CachedHeadConfig,
    *,
    device: str,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _set_seed(config.seed)
    model = make_twin_heads()
    load_pretrained_heads(model, checkpoint_weights)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    target = torch.as_tensor(
        6.0 - frame["pEC50"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    weights = torch.as_tensor(_sample_weights(frame, config), dtype=torch.float32)
    dataset = TensorDataset(member1, member2, target, weights)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=0,
    )
    for _ in range(config.max_epochs):
        model.train()
        for repr1, repr2, batch_target, batch_weight in loader:
            repr1 = repr1.to(device)
            repr2 = repr2.to(device)
            batch_target = batch_target.to(device)
            batch_weight = batch_weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred1, pred2, ensemble = model(repr1, repr2)
            loss = combined_huber_loss(
                pred1,
                pred2,
                ensemble,
                batch_target,
                batch_weight,
                config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def predict_nesso(
    state: dict[str, Any],
    member1: Any,
    member2: Any,
    *,
    device: str,
) -> np.ndarray:
    import torch

    model = make_twin_heads()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        _, _, ensemble = model(member1.to(device), member2.to(device))
    return 6.0 - ensemble.detach().cpu().numpy()


def evaluate_inner_nesso_config(
    member1: Any,
    member2: Any,
    frame: pd.DataFrame,
    checkpoint_weights: Path,
    config: CachedHeadConfig,
    *,
    device: str,
) -> tuple[dict[str, float], list[int]]:
    prediction_frames: list[pd.DataFrame] = []
    best_epochs: list[int] = []
    for fold in sorted(frame["fold"].unique()):
        predictions, _, checkpoint = fit_cached_fold(
            member1,
            member2,
            frame,
            checkpoint_weights,
            int(fold),
            config,
            device=device,
        )
        prediction_frames.append(predictions)
        best_epochs.append(int(checkpoint["best_epoch"]))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = regression_metrics(
        predictions["pEC50"],
        predictions["predicted_pEC50"],
    )
    return metrics, best_epochs


def run_nested_nesso(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_development_manifest(args)
    tensors = load_file(str(args.features), device="cpu")
    positions = frame["feature_row"].to_numpy(dtype=int)
    member1 = tensors["member1_affinity_repr"][positions].float()
    member2 = tensors["member2_affinity_repr"][positions].float()
    frozen = 6.0 - tensors["affinity_pred_value"][positions].numpy()

    predictions = frame.copy()
    predictions["frozen_pEC50"] = frozen
    predictions["mean_pEC50"] = np.nan
    predictions["affine_pEC50"] = np.nan
    for seed in SEEDS:
        predictions[f"nesso_seed_{seed}_pEC50"] = np.nan

    inner_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for outer_fold in sorted(frame["fold"].unique()):
        outer_train_positions = np.flatnonzero(frame["fold"].ne(outer_fold))
        outer_val_positions = np.flatnonzero(frame["fold"].eq(outer_fold))
        inner_frame = frame.iloc[outer_train_positions].reset_index(drop=True)
        inner_member1 = member1[outer_train_positions]
        inner_member2 = member2[outer_train_positions]

        candidates: list[tuple[str, CachedHeadConfig, dict[str, float], list[int]]] = []
        e2_candidates: list[
            tuple[str, CachedHeadConfig, dict[str, float], list[int]]
        ] = []
        for learning_rate in args.nesso_learning_rates:
            for weight_decay in WEIGHT_DECAYS:
                config = CachedHeadConfig(
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    seed=42,
                )
                metrics, epochs = evaluate_inner_nesso_config(
                    inner_member1,
                    inner_member2,
                    inner_frame,
                    args.checkpoint_weights,
                    config,
                    device=args.device,
                )
                name = f"e2_lr{learning_rate:g}_wd{weight_decay:g}"
                candidate = (name, config, metrics, epochs)
                e2_candidates.append(candidate)
                candidates.append(candidate)
        best_e2 = min(
            e2_candidates,
            key=lambda item: (item[2]["mae"], -item[2]["spearman"]),
        )
        for floor in E3_UNCERTAINTY_FLOORS:
            for use_curation in (False, True):
                config = replace(
                    best_e2[1],
                    uncertainty_floor=floor,
                    use_curation_weight=use_curation,
                )
                metrics, epochs = evaluate_inner_nesso_config(
                    inner_member1,
                    inner_member2,
                    inner_frame,
                    args.checkpoint_weights,
                    config,
                    device=args.device,
                )
                name = f"e3_floor{floor:g}_curation{int(use_curation)}"
                candidates.append((name, config, metrics, epochs))

        for name, config, metrics, epochs in candidates:
            inner_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "candidate": name,
                    **asdict(config),
                    **{f"inner_{key}": value for key, value in metrics.items()},
                    "inner_best_epochs": ";".join(str(value) for value in epochs),
                }
            )
        selected = min(
            candidates,
            key=lambda item: (item[2]["mae"], -item[2]["spearman"]),
        )
        selected_name, selected_config, selected_metrics, best_epochs = selected
        fixed_epochs = max(1, int(np.rint(np.median(best_epochs))))
        selected_rows.append(
            {
                "outer_fold": int(outer_fold),
                "selected_candidate": selected_name,
                "fixed_refit_epochs": fixed_epochs,
                "inner_best_epochs": ";".join(str(value) for value in best_epochs),
                **asdict(selected_config),
                **{f"inner_{key}": value for key, value in selected_metrics.items()},
            }
        )

        y_train = frame.iloc[outer_train_positions]["pEC50"].to_numpy(dtype=float)
        predictions.loc[outer_val_positions, "mean_pEC50"] = np.mean(y_train)
        affine_design = np.column_stack(
            [np.ones(len(outer_train_positions)), frozen[outer_train_positions]]
        )
        affine_intercept, affine_slope = np.linalg.lstsq(
            affine_design,
            y_train,
            rcond=None,
        )[0]
        predictions.loc[outer_val_positions, "affine_pEC50"] = (
            affine_intercept + affine_slope * frozen[outer_val_positions]
        )

        outer_train_frame = frame.iloc[outer_train_positions].reset_index(drop=True)
        for seed in SEEDS:
            fixed_config = replace(
                selected_config,
                max_epochs=fixed_epochs,
                patience=fixed_epochs,
                seed=seed,
            )
            state = fit_fixed_nesso(
                member1[outer_train_positions],
                member2[outer_train_positions],
                outer_train_frame,
                args.checkpoint_weights,
                fixed_config,
                device=args.device,
            )
            predictions.loc[
                outer_val_positions,
                f"nesso_seed_{seed}_pEC50",
            ] = predict_nesso(
                state,
                member1[outer_val_positions],
                member2[outer_val_positions],
                device=args.device,
            )
        print(f"completed nested Nesso outer fold {outer_fold}", flush=True)

    seed_columns = [f"nesso_seed_{seed}_pEC50" for seed in SEEDS]
    predictions["nesso_pEC50"] = predictions[seed_columns].mean(axis=1)
    if predictions[["mean_pEC50", "affine_pEC50", "nesso_pEC50"]].isna().any().any():
        raise AssertionError("nested Nesso predictions are incomplete")
    predictions.to_csv(args.output_dir / "nested_nesso_predictions.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(
        args.output_dir / "nested_nesso_inner_screen.csv",
        index=False,
    )
    pd.DataFrame(selected_rows).to_csv(
        args.output_dir / "nested_nesso_outer_selections.csv",
        index=False,
    )


def lgbm_params(threads: int, *, seed: int, n_estimators: int = 5000) -> dict[str, Any]:
    return {
        "objective": "regression",
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 5.0,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "n_jobs": threads,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
    }


def two_d_weights(frame: pd.DataFrame) -> np.ndarray:
    return inverse_se_weights(
        frame["pEC50_standard_error"],
        frame["curation_sample_weight"],
    )


def run_nested_2d(args: argparse.Namespace) -> None:
    import lightgbm as lgb

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_development_manifest(args)
    feature_table = pd.read_parquet(args.reduced_2d_features)
    feature_names = [
        column
        for column in feature_table.columns
        if column not in {"Molecule Name", "SMILES"}
    ]
    if feature_table["Molecule Name"].duplicated().any():
        raise ValueError("2D feature identifiers must be unique")
    feature_table = feature_table.set_index("Molecule Name")
    aligned = feature_table.reindex(frame["original_id"])
    if aligned.index.isna().any() or aligned["SMILES"].isna().any():
        raise ValueError("2D feature table does not cover the development cohort")
    x = aligned[feature_names].to_numpy(dtype=np.float32)
    x[~np.isfinite(x)] = 0.0
    del aligned, feature_table
    gc.collect()
    y = frame["pEC50"].to_numpy(dtype=float)
    oof = np.empty(len(frame), dtype=float)
    outer_rows: list[dict[str, Any]] = []
    selected_rows: list[pd.DataFrame] = []

    for outer_fold in sorted(frame["fold"].unique()):
        outer_train = np.flatnonzero(frame["fold"].ne(outer_fold))
        outer_val = np.flatnonzero(frame["fold"].eq(outer_fold))
        inner_folds = sorted(frame.iloc[outer_train]["fold"].unique())
        gain = np.zeros((len(inner_folds), len(feature_names)), dtype=np.float64)
        first_pass_rounds: list[int] = []
        for position, inner_fold in enumerate(inner_folds):
            train_idx = np.flatnonzero(
                frame["fold"].ne(outer_fold) & frame["fold"].ne(inner_fold)
            )
            val_idx = np.flatnonzero(
                frame["fold"].ne(outer_fold) & frame["fold"].eq(inner_fold)
            )
            model = lgb.LGBMRegressor(
                **lgbm_params(
                    args.lgbm_threads,
                    seed=20260809 + int(outer_fold) * 100 + int(inner_fold),
                )
            )
            model.fit(
                x[train_idx],
                y[train_idx],
                sample_weight=two_d_weights(frame.iloc[train_idx]),
                eval_set=[(x[val_idx], y[val_idx])],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            gain[position] = model.booster_.feature_importance(importance_type="gain")
            first_pass_rounds.append(int(model.best_iteration_))
        mean_gain = gain.mean(axis=0)
        selected_idx = np.argsort(-mean_gain, kind="stable")[: args.top_k]
        selected_rows.append(
            pd.DataFrame(
                {
                    "outer_fold": int(outer_fold),
                    "rank": np.arange(1, len(selected_idx) + 1),
                    "feature_index": selected_idx,
                    "feature": [feature_names[index] for index in selected_idx],
                    "mean_inner_gain": mean_gain[selected_idx],
                }
            )
        )

        second_pass_rounds: list[int] = []
        for inner_fold in inner_folds:
            train_idx = np.flatnonzero(
                frame["fold"].ne(outer_fold) & frame["fold"].ne(inner_fold)
            )
            val_idx = np.flatnonzero(
                frame["fold"].ne(outer_fold) & frame["fold"].eq(inner_fold)
            )
            model = lgb.LGBMRegressor(
                **lgbm_params(
                    args.lgbm_threads,
                    seed=20261809 + int(outer_fold) * 100 + int(inner_fold),
                )
            )
            model.fit(
                x[train_idx][:, selected_idx],
                y[train_idx],
                sample_weight=two_d_weights(frame.iloc[train_idx]),
                eval_set=[(x[val_idx][:, selected_idx], y[val_idx])],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            second_pass_rounds.append(int(model.best_iteration_))
        fixed_rounds = max(1, int(np.rint(np.median(second_pass_rounds))))
        final_model = lgb.LGBMRegressor(
            **lgbm_params(
                args.lgbm_threads,
                seed=20262809 + int(outer_fold),
                n_estimators=fixed_rounds,
            )
        )
        final_model.fit(
            x[outer_train][:, selected_idx],
            y[outer_train],
            sample_weight=two_d_weights(frame.iloc[outer_train]),
        )
        oof[outer_val] = final_model.predict(x[outer_val][:, selected_idx])
        fold_metrics = regression_metrics(y[outer_val], oof[outer_val])
        outer_rows.append(
            {
                "outer_fold": int(outer_fold),
                "outer_train_rows": len(outer_train),
                "outer_validation_rows": len(outer_val),
                "input_feature_count": len(feature_names),
                "selected_feature_count": len(selected_idx),
                "first_pass_best_iterations": ";".join(
                    str(value) for value in first_pass_rounds
                ),
                "second_pass_best_iterations": ";".join(
                    str(value) for value in second_pass_rounds
                ),
                "fixed_refit_iterations": fixed_rounds,
                **fold_metrics,
            }
        )
        print(f"completed nested 2D outer fold {outer_fold}", flush=True)

    output = frame[["feature_row", "record_id", "original_id", "fold", "pEC50"]].copy()
    output["two_d_pEC50"] = oof
    output.to_csv(args.output_dir / "nested_2d_predictions.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(
        args.output_dir / "nested_2d_outer_summary.csv",
        index=False,
    )
    pd.concat(selected_rows, ignore_index=True).to_csv(
        args.output_dir / "nested_2d_selected_features.csv",
        index=False,
    )
    metadata = {
        "model": "LightGBM regression",
        "cohort": "all 3344 curated development dose-response compounds",
        "outer_split": "shared Nesso scaffold-aware Butina cluster folds",
        "inner_split": "the four remaining outer cluster folds",
        "feature_selection": (
            "mean LightGBM gain in inner CV; top 1000 separately per outer fold"
        ),
        "stopping": (
            "median inner best iteration after inner-only feature selection, then "
            "fixed-round refit on all outer-training rows"
        ),
        "invalid_feature_imputation": "zero",
        "sample_weight": "inverse pEC50 standard error times curation weight",
        "input_features": portable_path(args.reduced_2d_features),
        "input_feature_count": len(feature_names),
        "top_k": args.top_k,
        "parameters": lgbm_params(args.lgbm_threads, seed=20260809),
    }
    (args.output_dir / "nested_2d_method.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def nearest_training_neighbors(
    evaluation: pd.DataFrame,
    training: pd.DataFrame,
    *,
    evaluation_smiles: str,
    training_smiles: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprints(values: pd.Series) -> list[Any]:
        result = []
        for value in values:
            molecule = Chem.MolFromSmiles(str(value))
            if molecule is None:
                raise ValueError(f"invalid SMILES in similarity analysis: {value}")
            result.append(generator.GetFingerprint(molecule))
        return result

    eval_fingerprints = fingerprints(evaluation[evaluation_smiles])
    train_fingerprints = fingerprints(training[training_smiles])
    train_ids = training["original_id"].astype(str).to_numpy()
    train_activity = training["pEC50"].to_numpy(dtype=float)
    similarities = np.empty(len(evaluation), dtype=float)
    neighbor_ids = np.empty(len(evaluation), dtype=object)
    neighbor_activity = np.empty(len(evaluation), dtype=float)
    for index, fingerprint in enumerate(eval_fingerprints):
        values = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, train_fingerprints),
            dtype=float,
        )
        nearest = int(np.argmax(values))
        similarities[index] = values[nearest]
        neighbor_ids[index] = train_ids[nearest]
        neighbor_activity[index] = train_activity[nearest]
    return similarities, neighbor_ids, neighbor_activity


def add_nested_nearest_neighbors(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["nearest_train_similarity"] = np.nan
    result["nearest_train_id"] = ""
    result["nearest_train_pEC50"] = np.nan
    for fold in sorted(result["fold"].unique()):
        val_mask = result["fold"].eq(fold)
        train_mask = ~val_mask
        similarities, ids, activity = nearest_training_neighbors(
            result.loc[val_mask],
            result.loc[train_mask],
            evaluation_smiles="identity_canonical_smiles",
            training_smiles="identity_canonical_smiles",
        )
        result.loc[val_mask, "nearest_train_similarity"] = similarities
        result.loc[val_mask, "nearest_train_id"] = ids
        result.loc[val_mask, "nearest_train_pEC50"] = activity
    return result


def make_lockbox_frame(
    args: argparse.Namespace, manifest: pd.DataFrame
) -> pd.DataFrame:
    nesso = pd.read_csv(args.nesso_holdout)
    two_d = pd.read_csv(args.historical_2d_holdout).rename(
        columns={
            "Molecule Name": "original_id",
            "predicted_pEC50": "two_d_pEC50",
        }
    )
    lockbox = nesso.merge(
        two_d[["original_id", "two_d_pEC50"]],
        on="original_id",
        how="inner",
        validate="one_to_one",
    )
    annotations = manifest[
        [
            "original_id",
            "identity_canonical_smiles",
            "pEC50_standard_error",
            "formal_charge",
            "audit_scaffold_group",
            "role",
        ]
    ]
    lockbox = lockbox.merge(
        annotations,
        on="original_id",
        how="left",
        validate="one_to_one",
    )
    if len(lockbox) != 714 or not lockbox["role"].eq("lockbox_primary").all():
        raise ValueError("historical 2D and Nesso holdout predictions do not align")
    development_primary = manifest.loc[
        manifest["role"].eq("development_primary")
    ].copy()
    lockbox["mean_pEC50"] = development_primary["pEC50"].mean()
    lockbox = lockbox.rename(
        columns={
            "predicted_pEC50_ensemble": "nesso_pEC50",
            "frozen_pEC50": "frozen_pEC50",
        }
    )
    similarities, ids, activity = nearest_training_neighbors(
        lockbox,
        development_primary,
        evaluation_smiles="identity_canonical_smiles",
        training_smiles="identity_canonical_smiles",
    )
    lockbox["nearest_train_similarity"] = similarities
    lockbox["nearest_train_id"] = ids
    lockbox["nearest_train_pEC50"] = activity
    lockbox["bootstrap_group"] = lockbox["audit_scaffold_group"]
    return lockbox


def make_challenge_frame(
    args: argparse.Namespace, manifest: pd.DataFrame
) -> pd.DataFrame:
    nesso = pd.read_csv(args.challenge_nesso).rename(
        columns={"trained_ensemble_pEC50": "nesso_pEC50"}
    )
    two_d = pd.read_csv(args.historical_2d_challenge).rename(
        columns={"Molecule Name": "original_id", "pEC50": "two_d_pEC50"}
    )
    challenge = nesso.merge(
        two_d[["original_id", "two_d_pEC50"]],
        on="original_id",
        how="inner",
        validate="one_to_one",
    )
    if len(challenge) != 513:
        raise ValueError("challenge Nesso and 2D predictions do not align")
    challenge = challenge.rename(columns={"untrained_pEC50": "frozen_pEC50"})
    full_primary = manifest.loc[
        manifest["role"].isin(["development_primary", "lockbox_primary"])
    ].copy()
    challenge["mean_pEC50"] = full_primary["pEC50"].mean()
    challenge["identity_canonical_smiles"] = challenge["SMILES"]
    challenge["formal_charge"] = np.nan
    similarities, ids, activity = nearest_training_neighbors(
        challenge,
        full_primary,
        evaluation_smiles="identity_canonical_smiles",
        training_smiles="identity_canonical_smiles",
    )
    challenge["nearest_train_similarity"] = similarities
    challenge["nearest_train_id"] = ids
    challenge["nearest_train_pEC50"] = activity
    scaffolds = challenge["identity_canonical_smiles"].map(bemis_murcko_scaffold)
    challenge["bootstrap_group"] = [
        scaffold_group_key(smiles, scaffold)
        for smiles, scaffold in zip(
            challenge["identity_canonical_smiles"],
            scaffolds,
            strict=True,
        )
    ]
    return challenge


def add_strata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["potency_stratum"] = pd.cut(
        result["pEC50"],
        bins=(-np.inf, 4.0, 5.0, 6.0, np.inf),
        labels=("<=4", "(4,5]", "(5,6]", ">6"),
    ).astype(str)
    result["similarity_stratum"] = pd.cut(
        result["nearest_train_similarity"],
        bins=(-np.inf, 0.3, 0.4, 0.5, 0.6, np.inf),
        labels=("<=0.3", "(0.3,0.4]", "(0.4,0.5]", "(0.5,0.6]", ">0.6"),
    ).astype(str)
    if (
        "pEC50_standard_error" in result
        and result["pEC50_standard_error"].notna().sum() >= 4
    ):
        result["assay_se_stratum"] = pd.qcut(
            result["pEC50_standard_error"],
            q=4,
            labels=("Q1_low", "Q2", "Q3", "Q4_high"),
            duplicates="drop",
        ).astype(str)
    if "formal_charge" in result:
        charge = pd.to_numeric(result["formal_charge"], errors="coerce")
        result["charge_stratum"] = np.select(
            [charge < 0, charge == 0, charge > 0],
            ["negative", "neutral", "positive"],
            default="unknown",
        )
    result["is_train_neighbor_cliff"] = result["nearest_train_similarity"].ge(0.5) & (
        result["pEC50"] - result["nearest_train_pEC50"]
    ).abs().ge(1.5)
    return result


def cliff_pair_rows(
    predictions: pd.DataFrame,
    models: dict[str, str],
    pairs: pd.DataFrame,
    *,
    dataset: str,
    require_same_fold: bool,
) -> list[dict[str, Any]]:
    lookup = predictions.set_index("original_id")
    pairs = pairs.loc[pairs["is_default_cliff"].astype(bool)].copy()
    pairs = pairs.loc[
        pairs["low_ligand_id"].isin(lookup.index)
        & pairs["high_ligand_id"].isin(lookup.index)
    ]
    if require_same_fold and len(pairs):
        low_fold = pairs["low_ligand_id"].map(lookup["fold"])
        high_fold = pairs["high_ligand_id"].map(lookup["fold"])
        pairs = pairs.loc[low_fold.eq(high_fold)]
    rows: list[dict[str, Any]] = []
    true_delta = pairs["high_pEC50"].to_numpy() - pairs["low_pEC50"].to_numpy()
    for model, column in models.items():
        low_prediction = pairs["low_ligand_id"].map(lookup[column]).to_numpy(float)
        high_prediction = pairs["high_ligand_id"].map(lookup[column]).to_numpy(float)
        predicted_delta = high_prediction - low_prediction
        rows.append(
            {
                "dataset": dataset,
                "analysis": "held_out_pair_resolution",
                "model": model,
                "n": len(pairs),
                "delta_mae": (
                    float(np.mean(np.abs(predicted_delta - true_delta)))
                    if len(pairs)
                    else np.nan
                ),
                "direction_accuracy": (
                    float(np.mean(predicted_delta > 0)) if len(pairs) else np.nan
                ),
                "median_predicted_delta": (
                    float(np.median(predicted_delta)) if len(pairs) else np.nan
                ),
                "median_true_delta": (
                    float(np.median(true_delta)) if len(pairs) else np.nan
                ),
            }
        )
    return rows


def run_analysis(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.modeling_manifest)
    nested_nesso = pd.read_csv(args.analysis_input_dir / "nested_nesso_predictions.csv")
    nested_2d = pd.read_csv(args.analysis_input_dir / "nested_2d_predictions.csv")
    nested = nested_nesso.merge(
        nested_2d[["original_id", "two_d_pEC50"]],
        on="original_id",
        how="inner",
        validate="one_to_one",
    )
    if len(nested) != 3344:
        raise ValueError("nested model predictions do not cover the same compounds")
    nested = add_nested_nearest_neighbors(nested)
    nested["bootstrap_group"] = nested["cluster_component"].astype(int).astype(str)
    lockbox = make_lockbox_frame(args, manifest)
    challenge = make_challenge_frame(args, manifest)
    nested = add_strata(nested)
    lockbox = add_strata(lockbox)
    challenge = add_strata(challenge)
    nested.to_csv(args.output_dir / "nested_aligned_predictions.csv", index=False)
    lockbox.to_csv(args.output_dir / "lockbox_aligned_predictions.csv", index=False)
    challenge.to_csv(args.output_dir / "challenge_aligned_predictions.csv", index=False)

    datasets = {
        "nested_development": (
            nested,
            {
                "mean": "mean_pEC50",
                "frozen_nesso": "frozen_pEC50",
                "affine_frozen_nesso": "affine_pEC50",
                "fine_tuned_nesso": "nesso_pEC50",
                "nested_2d_lightgbm": "two_d_pEC50",
            },
        ),
        "historical_lockbox_primary": (
            lockbox,
            {
                "mean": "mean_pEC50",
                "frozen_nesso": "frozen_pEC50",
                "fine_tuned_nesso": "nesso_pEC50",
                "established_2d_lightgbm": "two_d_pEC50",
            },
        ),
        "public_challenge": (
            challenge,
            {
                "mean": "mean_pEC50",
                "frozen_nesso": "frozen_pEC50",
                "fine_tuned_nesso": "nesso_pEC50",
                "established_2d_challenge": "two_d_pEC50",
            },
        ),
    }

    metric_table_rows: list[dict[str, Any]] = []
    ci_rows: list[pd.DataFrame] = []
    difference_rows: list[pd.DataFrame] = []
    complementarity_rows: list[dict[str, Any]] = []
    for dataset, (frame, models) in datasets.items():
        metric_table_rows.extend(metric_rows(frame, models, dataset=dataset))
        intervals, draws = clustered_bootstrap(
            frame,
            models,
            group_column="bootstrap_group",
            replicates=args.bootstrap_replicates,
            seed=20260809,
        )
        intervals.insert(0, "dataset", dataset)
        ci_rows.append(intervals)
        point = {
            model: regression_metrics(frame["pEC50"], frame[column])
            for model, column in models.items()
        }
        two_d_name = next(model for model in models if "2d" in model)
        comparisons = [
            ("fine_tuned_nesso", "frozen_nesso"),
            ("fine_tuned_nesso", two_d_name),
            (two_d_name, "mean"),
        ]
        differences = paired_difference_rows(draws, point, comparisons)
        differences.insert(0, "dataset", dataset)
        difference_rows.append(differences)
        complementarity_rows.append(
            {
                "dataset": dataset,
                "first_model": "fine_tuned_nesso",
                "second_model": two_d_name,
                **error_complementarity(
                    frame,
                    "nesso_pEC50",
                    "two_d_pEC50",
                ),
            }
        )

    nested_primary = nested.loc[nested["role"].eq("development_primary")]
    nested_low_emax = nested.loc[nested["role"].eq("excluded_initial_emax")]
    nested_models = datasets["nested_development"][1]
    metric_table_rows.extend(
        metric_rows(
            nested_primary,
            nested_models,
            dataset="nested_development",
            subset="Emax_qualified",
        )
    )
    metric_table_rows.extend(
        metric_rows(
            nested_low_emax,
            nested_models,
            dataset="nested_development",
            subset="lower_Emax_curated",
        )
    )

    pd.DataFrame(metric_table_rows).to_csv(
        args.output_dir / "metrics.csv",
        index=False,
    )
    pd.concat(ci_rows, ignore_index=True).to_csv(
        args.output_dir / "cluster_bootstrap_intervals.csv",
        index=False,
    )
    pd.concat(difference_rows, ignore_index=True).to_csv(
        args.output_dir / "paired_cluster_bootstrap_differences.csv",
        index=False,
    )
    pd.DataFrame(complementarity_rows).to_csv(
        args.output_dir / "error_complementarity.csv",
        index=False,
    )

    strata_rows: list[dict[str, Any]] = []
    for dataset, (frame, models) in datasets.items():
        stratum_columns = ["potency_stratum", "similarity_stratum"]
        stratum_columns.extend(
            column
            for column in ("assay_se_stratum", "charge_stratum")
            if column in frame
        )
        for stratum_column in stratum_columns:
            for level, subset in frame.groupby(stratum_column, observed=True):
                if len(subset) < 2:
                    continue
                rows = metric_rows(
                    subset,
                    models,
                    dataset=dataset,
                    subset=f"{stratum_column}:{level}",
                )
                strata_rows.extend(rows)
    pd.DataFrame(strata_rows).to_csv(
        args.output_dir / "stratified_metrics.csv",
        index=False,
    )

    active_rows: list[dict[str, Any]] = []
    cliff_rows: list[dict[str, Any]] = []
    for dataset, (frame, models) in datasets.items():
        active = frame.loc[frame["pEC50"].gt(6.0)]
        cliff_subset = frame.loc[frame["is_train_neighbor_cliff"]]
        for model, column in models.items():
            metrics = regression_metrics(active["pEC50"], active[column])
            active_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    **metrics,
                    "predicted_active_count": int(active[column].gt(6.0).sum()),
                    "active_recall_at_6": (
                        float(active[column].gt(6.0).mean()) if len(active) else np.nan
                    ),
                }
            )
            cliff_metrics = regression_metrics(
                cliff_subset["pEC50"],
                cliff_subset[column],
            )
            cliff_rows.append(
                {
                    "dataset": dataset,
                    "analysis": "test_points_cliffing_from_nearest_training_neighbor",
                    "model": model,
                    **cliff_metrics,
                }
            )
    pd.DataFrame(active_rows).to_csv(
        args.output_dir / "active_tail_metrics.csv",
        index=False,
    )

    pairs = pd.read_csv(args.cliff_pairs)
    cliff_rows.extend(
        cliff_pair_rows(
            nested,
            nested_models,
            pairs,
            dataset="nested_development",
            require_same_fold=True,
        )
    )
    cliff_rows.extend(
        cliff_pair_rows(
            lockbox,
            datasets["historical_lockbox_primary"][1],
            pairs,
            dataset="historical_lockbox_primary",
            require_same_fold=False,
        )
    )
    pd.DataFrame(cliff_rows).to_csv(
        args.output_dir / "activity_cliff_metrics.csv",
        index=False,
    )

    summary = {
        "status": "pass",
        "nested_rows": len(nested),
        "nested_primary_rows": len(nested_primary),
        "nested_lower_emax_rows": len(nested_low_emax),
        "lockbox_primary_rows": len(lockbox),
        "challenge_rows": len(challenge),
        "bootstrap_replicates": args.bootstrap_replicates,
        "confidence_interval_unit": "scaffold/Butina chemical-family cluster",
        "challenge_analysis_status": (
            "retrospective public-truth analysis; not a temporal blind test"
        ),
        "historical_2d_holdout_model": (
            "PXR-only LightGBM, Emax-qualified training, five-fold ensemble"
        ),
        "historical_2d_challenge_model": (
            "pre-existing ChemBL-augmented, globally offset LightGBM submission"
        ),
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def validate_stage_inputs(args: argparse.Namespace) -> None:
    common = {
        "modeling manifest": args.modeling_manifest,
    }
    nesso = {
        **common,
        "cached Nesso features": args.features,
        "Nesso feature metadata": args.feature_metadata,
        "released Nesso checkpoint": args.checkpoint_weights,
    }
    two_d = {
        **common,
        "reduced 2D features": args.reduced_2d_features,
    }
    analysis = {
        **common,
        "nested Nesso predictions": (
            args.analysis_input_dir / "nested_nesso_predictions.csv"
        ),
        "nested 2D predictions": (
            args.analysis_input_dir / "nested_2d_predictions.csv"
        ),
        "historical Nesso holdout predictions": args.nesso_holdout,
        "historical 2D holdout predictions": args.historical_2d_holdout,
        "retrospective Nesso challenge predictions": args.challenge_nesso,
        "retrospective 2D challenge predictions": args.historical_2d_challenge,
        "activity-cliff pairs": args.cliff_pairs,
    }
    if args.stage == "nesso":
        required = nesso
    elif args.stage == "2d":
        required = two_d
    elif args.stage == "analyze":
        required = analysis
    else:
        required = {**nesso, **two_d}
        required.update(
            {
                key: value
                for key, value in analysis.items()
                if not key.startswith("nested ")
            }
        )
    missing = [
        f"{label}: {path}" for label, path in required.items() if not path.is_file()
    ]
    if missing:
        message = "missing required inputs:\n- " + "\n- ".join(missing)
        if args.checkpoint_weights in required.values():
            message += "\nRun: python scripts/download_nesso_checkpoint.py"
        raise FileNotFoundError(message)


def main() -> None:
    args = parse_args()
    validate_stage_inputs(args)
    if args.stage in {"all", "nesso"}:
        run_nested_nesso(args)
    if args.stage in {"all", "2d"}:
        run_nested_2d(args)
    if args.stage in {"all", "analyze"}:
        if args.stage == "all":
            args.analysis_input_dir = args.output_dir
        run_analysis(args)


if __name__ == "__main__":
    main()
