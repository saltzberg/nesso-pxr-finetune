"""Scaffold-aware cached Nesso-head training for the all-curated PXR cohort."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nesso_pxr.chemistry import bemis_murcko_scaffold, scaffold_group_key
from nesso_pxr.labels import effective_uncertainty_weight
from nesso_pxr.protocol import sha256_file
from nesso_pxr.splits import (
    SplitConfig,
    assert_split_integrity,
    assign_scaffold_aware_butina_folds,
)


@dataclass(frozen=True)
class CachedHeadConfig:
    """Bounded cached-head optimization choices frozen before lockbox use."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 0.005
    huber_delta: float = 0.5
    ensemble_loss_weight: float = 0.5
    member_loss_weight: float = 0.25
    gradient_clip_norm: float = 1.0
    uncertainty_floor: float | None = None
    uncertainty_cap_quantile: float = 0.95
    use_curation_weight: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("learning rate, batch size, and epochs must be positive")
        if self.patience < 1 or self.min_delta < 0 or self.huber_delta <= 0:
            raise ValueError("invalid early-stopping or Huber parameters")
        total = self.ensemble_loss_weight + 2 * self.member_loss_weight
        if not math.isclose(total, 1.0):
            raise ValueError("ensemble plus two member loss weights must sum to one")


def build_all_curated_modeling_manifest(
    selection_path: Path,
    holdout_path: Path,
    output_dir: Path,
    *,
    split_config: SplitConfig | None = None,
    expected_rows: int = 4134,
) -> dict[str, Any]:
    """Restore immutable holdout roles and create five folds for all development DRC."""

    config = split_config or SplitConfig()
    selection = pd.read_csv(selection_path)
    holdout = pd.read_csv(holdout_path)
    if len(selection) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} selected rows, found {len(selection)}"
        )
    if selection["original_id"].duplicated().any():
        raise ValueError("selection original IDs must be unique")

    role_map = holdout[["Molecule Name", "holdout_role"]].rename(
        columns={"Molecule Name": "original_id"}
    )
    modeling = selection.merge(
        role_map,
        on="original_id",
        how="left",
        validate="one_to_one",
    )
    if modeling["holdout_role"].isna().any():
        missing = modeling.loc[modeling["holdout_role"].isna(), "original_id"].tolist()
        raise ValueError(f"missing immutable holdout assignments: {missing[:10]}")

    if "fold" in modeling:
        modeling = modeling.rename(columns={"fold": "legacy_fold"})
    modeling["modeling_role"] = modeling["holdout_role"].map(
        {"development": "development", "holdout": "lockbox"}
    )
    if modeling["modeling_role"].isna().any():
        raise ValueError("unexpected immutable holdout role")

    development = modeling.loc[modeling["modeling_role"].eq("development")]
    assignments = assign_scaffold_aware_butina_folds(
        development,
        id_column="original_id",
        smiles_column="identity_canonical_smiles",
        config=config,
    )
    assert_split_integrity(assignments)
    assignment_columns = [
        "original_id",
        "canonical_smiles",
        "bemis_murcko_scaffold",
        "scaffold_group",
        "butina_cluster",
        "cluster_component",
        "fold",
    ]
    modeling = modeling.merge(
        assignments[assignment_columns],
        on="original_id",
        how="left",
        validate="one_to_one",
    )
    modeling["fold"] = modeling["fold"].astype("Int64")

    canonical = modeling["identity_canonical_smiles"]
    modeling["audit_scaffold"] = canonical.map(bemis_murcko_scaffold)
    modeling["audit_scaffold_group"] = [
        scaffold_group_key(smiles, scaffold)
        for smiles, scaffold in zip(
            canonical,
            modeling["audit_scaffold"],
            strict=True,
        )
    ]
    development_scaffolds = set(
        modeling.loc[
            modeling["modeling_role"].eq("development"),
            "audit_scaffold_group",
        ]
    )
    lockbox_scaffolds = set(
        modeling.loc[
            modeling["modeling_role"].eq("lockbox"),
            "audit_scaffold_group",
        ]
    )
    scaffold_overlap = sorted(development_scaffolds & lockbox_scaffolds)
    if scaffold_overlap:
        raise AssertionError(
            f"immutable development/lockbox scaffold overlap: {scaffold_overlap[:10]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "modeling_manifest.csv"
    modeling = modeling.sort_values("feature_row", kind="stable")
    modeling.to_csv(manifest_path, index=False)

    fold_counts = {
        str(int(key)): int(value)
        for key, value in modeling.loc[
            modeling["modeling_role"].eq("development"), "fold"
        ]
        .value_counts()
        .sort_index()
        .items()
    }
    role_counts = {
        str(key): int(value)
        for key, value in modeling["modeling_role"].value_counts().sort_index().items()
    }
    audit = {
        "status": "pass",
        "rows": int(len(modeling)),
        "role_counts": role_counts,
        "fold_counts": fold_counts,
        "development_lockbox_scaffold_overlap": 0,
        "selection_sha256": sha256_file(selection_path),
        "holdout_sha256": sha256_file(holdout_path),
        "modeling_manifest_sha256": sha256_file(manifest_path),
        "split": asdict(config),
    }
    (output_dir / "split_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def _torch_modules() -> tuple[Any, Any, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as functional

    return torch, nn, functional


def make_regression_mlp() -> Any:
    """Return Nesso's exact 384-wide pretrained regression MLP architecture."""

    _, nn, _ = _torch_modules()
    return nn.Sequential(
        nn.Linear(384, 384),
        nn.ReLU(),
        nn.Linear(384, 384),
        nn.ReLU(),
        nn.Linear(384, 1),
    )


def make_twin_heads() -> Any:
    """Construct the two trainable Nesso regression members."""

    _, nn, _ = _torch_modules()

    class TwinAffinityHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.member1 = make_regression_mlp()
            self.member2 = make_regression_mlp()

        def forward(self, member1_repr: Any, member2_repr: Any) -> tuple[Any, Any, Any]:
            pred1 = self.member1(member1_repr).squeeze(-1)
            pred2 = self.member2(member2_repr).squeeze(-1)
            return pred1, pred2, (pred1 + pred2) / 2.0

    return TwinAffinityHeads()


def load_pretrained_heads(model: Any, checkpoint_weights: Path) -> None:
    """Load only the two released Nesso regression MLPs into the cached model."""

    from safetensors.torch import load_file

    state = load_file(str(checkpoint_weights), device="cpu")
    prefixes = {
        "member1": "affinity_module.affinity_heads.to_affinity_pred_value.",
        "member2": "affinity_module2.affinity_heads.to_affinity_pred_value.",
    }
    for member_name, prefix in prefixes.items():
        member_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        getattr(model, member_name).load_state_dict(member_state, strict=True)


def combined_huber_loss(
    prediction1: Any,
    prediction2: Any,
    ensemble: Any,
    target: Any,
    sample_weight: Any,
    config: CachedHeadConfig,
) -> Any:
    """Apply the frozen 50/25/25 ensemble/member robust-loss contract."""

    _, _, functional = _torch_modules()
    losses = (
        config.ensemble_loss_weight
        * functional.huber_loss(
            ensemble,
            target,
            delta=config.huber_delta,
            reduction="none",
        )
        + config.member_loss_weight
        * functional.huber_loss(
            prediction1,
            target,
            delta=config.huber_delta,
            reduction="none",
        )
        + config.member_loss_weight
        * functional.huber_loss(
            prediction2,
            target,
            delta=config.huber_delta,
            reduction="none",
        )
    )
    return (losses * sample_weight).sum() / sample_weight.sum()


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return deterministic primary regression metrics in pEC50 units."""

    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - observed
    observed_rank = pd.Series(observed).rank(method="average").to_numpy()
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy()
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "spearman": float(np.corrcoef(observed_rank, predicted_rank)[0, 1]),
        "pearson": float(np.corrcoef(observed, predicted)[0, 1]),
        "bias": float(np.mean(residual)),
    }


def _sample_weights(frame: pd.DataFrame, config: CachedHeadConfig) -> np.ndarray:
    if config.uncertainty_floor is None:
        return np.ones(len(frame), dtype=np.float32)
    curation = (
        frame["curation_sample_weight"].to_numpy(dtype=np.float64)
        if config.use_curation_weight
        else None
    )
    return effective_uncertainty_weight(
        frame["pEC50_standard_error"].to_numpy(dtype=np.float64),
        floor=config.uncertainty_floor,
        cap_quantile=config.uncertainty_cap_quantile,
        curation_weight=curation,
    ).astype(np.float32)


def _set_seed(seed: int) -> None:
    torch, _, _ = _torch_modules()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_cached_fold(
    member1_repr: Any,
    member2_repr: Any,
    frame: pd.DataFrame,
    checkpoint_weights: Path,
    validation_fold: int,
    config: CachedHeadConfig,
    *,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit one development fold with early stopping and return held-out predictions."""

    torch, _, _ = _torch_modules()
    from torch.utils.data import DataLoader, TensorDataset

    _set_seed(config.seed)
    train_mask = frame["fold"].ne(validation_fold).to_numpy()
    validation_mask = frame["fold"].eq(validation_fold).to_numpy()
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError(
            f"empty train or validation partition for fold {validation_fold}"
        )

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
    training_weights = torch.as_tensor(
        _sample_weights(frame.iloc[train_indices], config),
        dtype=torch.float32,
    )
    dataset = TensorDataset(
        member1_repr[train_indices],
        member2_repr[train_indices],
        target[train_indices],
        training_weights,
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    best_mae = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    waiting = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for representation1, representation2, batch_target, batch_weight in loader:
            representation1 = representation1.to(device)
            representation2 = representation2.to(device)
            batch_target = batch_target.to(device)
            batch_weight = batch_weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred1, pred2, ensemble = model(representation1, representation2)
            loss = combined_huber_loss(
                pred1,
                pred2,
                ensemble,
                batch_target,
                batch_weight,
                config,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            count = len(batch_target)
            total_loss += float(loss.detach().cpu()) * count
            examples += count

        model.eval()
        with torch.inference_mode():
            pred1, pred2, ensemble = model(
                member1_repr[validation_indices].to(device),
                member2_repr[validation_indices].to(device),
            )
        validation_loss = combined_huber_loss(
            pred1,
            pred2,
            ensemble,
            target[validation_indices].to(device),
            torch.ones(len(validation_indices), device=device),
            config,
        )
        predicted_pec50 = 6.0 - ensemble.detach().cpu().numpy()
        observed_pec50 = frame.iloc[validation_indices]["pEC50"].to_numpy()
        metrics = regression_metrics(observed_pec50, predicted_pec50)
        history_rows.append(
            {
                "fold": validation_fold,
                "seed": config.seed,
                "epoch": epoch,
                "train_loss": total_loss / examples,
                "validation_loss": float(validation_loss.detach().cpu()),
                "validation_mae": metrics["mae"],
                "validation_spearman": metrics["spearman"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm_last_batch": float(gradient_norm),
            }
        )
        if metrics["mae"] < best_mae - config.min_delta:
            best_mae = metrics["mae"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += 1
            if waiting >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training never produced a best checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        pred1, pred2, ensemble = model(
            member1_repr[validation_indices].to(device),
            member2_repr[validation_indices].to(device),
        )
    predictions = frame.iloc[validation_indices][
        ["feature_row", "record_id", "original_id", "fold", "pEC50"]
    ].copy()
    predictions["predicted_nesso_member1"] = pred1.detach().cpu().numpy()
    predictions["predicted_nesso_member2"] = pred2.detach().cpu().numpy()
    predictions["predicted_nesso_ensemble"] = ensemble.detach().cpu().numpy()
    predictions["predicted_pEC50"] = 6.0 - predictions["predicted_nesso_ensemble"]
    predictions["seed"] = config.seed
    predictions["best_epoch"] = best_epoch
    checkpoint = {
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
    }
    return predictions, pd.DataFrame(history_rows), checkpoint


def _fit_affine(
    frozen_prediction: np.ndarray,
    observed: np.ndarray,
) -> tuple[float, float]:
    design = np.column_stack(
        [np.ones(len(frozen_prediction), dtype=np.float64), frozen_prediction]
    )
    intercept, slope = np.linalg.lstsq(design, observed, rcond=None)[0]
    return float(intercept), float(slope)


def run_cached_cv(
    features_path: Path,
    feature_metadata_path: Path,
    modeling_manifest_path: Path,
    checkpoint_weights: Path,
    output_dir: Path,
    config: CachedHeadConfig,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run five-fold E0/E1/head comparisons without touching the lockbox."""

    torch, _, _ = _torch_modules()
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(features_path), device="cpu")
    metadata = pd.read_csv(feature_metadata_path).sort_values("feature_row")
    modeling = pd.read_csv(modeling_manifest_path).sort_values("feature_row")
    if metadata["record_id"].tolist() != modeling["record_id"].tolist():
        raise ValueError("feature metadata and modeling manifest record order differ")
    if modeling["feature_row"].tolist() != list(range(len(modeling))):
        raise ValueError("modeling feature rows must be contiguous and zero-based")

    development_indices = np.flatnonzero(
        modeling["modeling_role"].eq("development").to_numpy()
    )
    frame = modeling.iloc[development_indices].reset_index(drop=True)
    member1_repr = tensors["member1_affinity_repr"][development_indices].float()
    member2_repr = tensors["member2_affinity_repr"][development_indices].float()
    frozen_nesso = tensors["affinity_pred_value"][development_indices].numpy()
    frozen_pec50 = 6.0 - frozen_nesso

    parity_model = make_twin_heads()
    load_pretrained_heads(parity_model, checkpoint_weights)
    parity_model.eval()
    with torch.inference_mode():
        parity1, parity2, parity_ensemble = parity_model(
            member1_repr,
            member2_repr,
        )
    parity_error = max(
        float(
            (
                parity1
                - tensors["reconstructed_affinity_pred_value1"][
                    development_indices
                ].float()
            )
            .abs()
            .max()
        ),
        float(
            (
                parity2
                - tensors["reconstructed_affinity_pred_value2"][
                    development_indices
                ].float()
            )
            .abs()
            .max()
        ),
        float(
            (
                parity_ensemble
                - tensors["affinity_pred_value"][development_indices].float()
            )
            .abs()
            .max()
        ),
    )
    if parity_error > 1e-6:
        raise AssertionError(f"pretrained cached-head parity failed: {parity_error}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"cached_lr{config.learning_rate:g}_wd{config.weight_decay:g}"
        f"_floor{config.uncertainty_floor}_curation{int(config.use_curation_weight)}"
        f"_seed{config.seed}"
    )

    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    fold_summaries: dict[str, Any] = {}
    for fold in sorted(frame["fold"].astype(int).unique()):
        train_mask = frame["fold"].astype(int).ne(fold).to_numpy()
        validation_mask = ~train_mask
        intercept, slope = _fit_affine(
            frozen_pec50[train_mask],
            frame.loc[train_mask, "pEC50"].to_numpy(),
        )
        fold_predictions, fold_history, checkpoint = fit_cached_fold(
            member1_repr,
            member2_repr,
            frame,
            checkpoint_weights,
            fold,
            config,
            device=device,
        )
        fold_predictions["frozen_pEC50"] = frozen_pec50[validation_mask]
        fold_predictions["affine_pEC50"] = (
            intercept + slope * frozen_pec50[validation_mask]
        )
        fold_predictions["run_id"] = run_id
        fold_history["run_id"] = run_id
        prediction_frames.append(fold_predictions)
        history_frames.append(fold_history)

        state = {
            key: value.contiguous() for key, value in checkpoint["state_dict"].items()
        }
        save_file(state, checkpoint_dir / f"fold_{fold}.safetensors")
        fold_metrics = {
            "frozen": regression_metrics(
                fold_predictions["pEC50"].to_numpy(),
                fold_predictions["frozen_pEC50"].to_numpy(),
            ),
            "affine": regression_metrics(
                fold_predictions["pEC50"].to_numpy(),
                fold_predictions["affine_pEC50"].to_numpy(),
            ),
            "cached_head": regression_metrics(
                fold_predictions["pEC50"].to_numpy(),
                fold_predictions["predicted_pEC50"].to_numpy(),
            ),
            "affine_intercept": intercept,
            "affine_slope": slope,
            "best_epoch": int(checkpoint["best_epoch"]),
        }
        fold_summaries[str(fold)] = fold_metrics

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        "feature_row",
        kind="stable",
    )
    history = pd.concat(history_frames, ignore_index=True)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    history.to_csv(output_dir / "history.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    )

    observed = predictions["pEC50"].to_numpy()
    aggregate = {
        "frozen": regression_metrics(
            observed,
            predictions["frozen_pEC50"].to_numpy(),
        ),
        "affine": regression_metrics(
            observed,
            predictions["affine_pEC50"].to_numpy(),
        ),
        "cached_head": regression_metrics(
            observed,
            predictions["predicted_pEC50"].to_numpy(),
        ),
    }
    summary = {
        "status": "pass",
        "run_id": run_id,
        "development_rows": int(len(frame)),
        "lockbox_rows_used": 0,
        "folds": fold_summaries,
        "aggregate": aggregate,
        "delta_mae_vs_frozen": (
            aggregate["frozen"]["mae"] - aggregate["cached_head"]["mae"]
        ),
        "delta_mae_vs_affine": (
            aggregate["affine"]["mae"] - aggregate["cached_head"]["mae"]
        ),
        "delta_spearman_vs_frozen": (
            aggregate["cached_head"]["spearman"] - aggregate["frozen"]["spearman"]
        ),
        "delta_spearman_vs_affine": (
            aggregate["cached_head"]["spearman"] - aggregate["affine"]["spearman"]
        ),
        "pretrained_parity_max_abs": parity_error,
        "features_sha256": sha256_file(features_path),
        "feature_metadata_sha256": sha256_file(feature_metadata_path),
        "modeling_manifest_sha256": sha256_file(modeling_manifest_path),
        "checkpoint_weights_sha256": sha256_file(checkpoint_weights),
        "config": asdict(config),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--selection", type=Path, required=True)
    split.add_argument("--holdout", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)

    cv = commands.add_parser("cv")
    cv.add_argument("--features", type=Path, required=True)
    cv.add_argument("--feature-metadata", type=Path, required=True)
    cv.add_argument("--modeling-manifest", type=Path, required=True)
    cv.add_argument("--checkpoint-weights", type=Path, required=True)
    cv.add_argument("--output-dir", type=Path, required=True)
    cv.add_argument("--learning-rate", type=float, default=1e-4)
    cv.add_argument("--weight-decay", type=float, default=1e-4)
    cv.add_argument("--batch-size", type=int, default=64)
    cv.add_argument("--max-epochs", type=int, default=30)
    cv.add_argument("--patience", type=int, default=5)
    cv.add_argument("--min-delta", type=float, default=0.005)
    cv.add_argument("--huber-delta", type=float, default=0.5)
    cv.add_argument("--uncertainty-floor", type=float)
    cv.add_argument("--use-curation-weight", action="store_true")
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "split":
        audit = build_all_curated_modeling_manifest(
            args.selection,
            args.holdout,
            args.output_dir,
        )
        print(json.dumps(audit, indent=2, sort_keys=True))
    elif args.command == "cv":
        config = CachedHeadConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            huber_delta=args.huber_delta,
            uncertainty_floor=args.uncertainty_floor,
            use_curation_weight=args.use_curation_weight,
            seed=args.seed,
        )
        summary = run_cached_cv(
            args.features,
            args.feature_metadata,
            args.modeling_manifest,
            args.checkpoint_weights,
            args.output_dir,
            config,
            device=args.device,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
