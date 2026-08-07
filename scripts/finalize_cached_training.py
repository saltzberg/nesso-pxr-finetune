#!/usr/bin/env python3
"""Refit selected cached heads, check the lockbox, and train final models."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, TensorDataset

from nesso_pxr.protocol import sha256_file
from nesso_pxr.train_cached import (
    CachedHeadConfig,
    combined_huber_loss,
    load_pretrained_heads,
    make_twin_heads,
    regression_metrics,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_fixed_epochs(
    member1_repr: torch.Tensor,
    member2_repr: torch.Tensor,
    target: torch.Tensor,
    checkpoint_weights: Path,
    config: CachedHeadConfig,
    *,
    device: str,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    set_seed(config.seed)
    model = make_twin_heads()
    load_pretrained_heads(model, checkpoint_weights)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    weights = torch.ones(len(target), dtype=torch.float32)
    dataset = TensorDataset(member1_repr, member2_repr, target, weights)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        gradient_norm = torch.tensor(0.0)
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
        history.append(
            {
                "run_id": "fixed_refit",
                "seed": config.seed,
                "epoch": epoch,
                "train_loss": total_loss / examples,
                "validation_loss": float("nan"),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm_last_batch": float(gradient_norm),
            }
        )
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    history_frame = pd.DataFrame(history)
    history_frame["wall_seconds_total"] = time.perf_counter() - started
    return state, history_frame


def predict(
    state: dict[str, torch.Tensor],
    member1_repr: torch.Tensor,
    member2_repr: torch.Tensor,
    *,
    device: str,
) -> np.ndarray:
    model = make_twin_heads()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        _, _, ensemble = model(member1_repr.to(device), member2_repr.to(device))
    return 6.0 - ensemble.detach().cpu().numpy()


def affine_fit(predicted: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(predicted)), predicted])
    intercept, slope = np.linalg.lstsq(design, observed, rcond=None)[0]
    return float(intercept), float(slope)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-metadata", type=Path, required=True)
    parser.add_argument("--modeling-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    tensors = load_file(str(args.features), device="cpu")
    metadata = pd.read_csv(args.feature_metadata).sort_values("feature_row")
    modeling = pd.read_csv(args.modeling_manifest).sort_values("feature_row")
    if metadata["record_id"].tolist() != modeling["record_id"].tolist():
        raise ValueError("feature metadata and modeling manifest order differ")
    if modeling["feature_row"].tolist() != list(range(len(modeling))):
        raise ValueError("feature rows must be contiguous and zero-based")

    member1 = tensors["member1_affinity_repr"].float()
    member2 = tensors["member2_affinity_repr"].float()
    target = torch.as_tensor(
        6.0 - modeling["pEC50"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    frozen_pec50 = 6.0 - tensors["affinity_pred_value"].numpy()
    development = np.flatnonzero(modeling["modeling_role"].eq("development"))
    lockbox = np.flatnonzero(modeling["modeling_role"].eq("lockbox"))
    if len(development) != 3344 or len(lockbox) != 790:
        raise ValueError(
            f"unexpected role counts: development={len(development)} "
            f"lockbox={len(lockbox)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    development_dir = args.output_dir / "development_refit"
    final_dir = args.output_dir / "final_all_labels"
    development_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    lockbox_predictions: list[np.ndarray] = []
    development_metrics: dict[str, Any] = {}
    reload_errors: dict[str, float] = {}
    for seed in args.seeds:
        config = CachedHeadConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.epochs,
            patience=args.epochs,
            seed=seed,
        )
        state, history = fit_fixed_epochs(
            member1[development],
            member2[development],
            target[development],
            args.checkpoint_weights,
            config,
            device=args.device,
        )
        checkpoint_path = development_dir / f"seed_{seed}.safetensors"
        save_file(state, checkpoint_path)
        history.to_csv(development_dir / f"history_seed_{seed}.csv", index=False)
        reloaded = load_file(str(checkpoint_path), device="cpu")
        original_prediction = predict(
            state,
            member1[lockbox[:32]],
            member2[lockbox[:32]],
            device=args.device,
        )
        reloaded_prediction = predict(
            reloaded,
            member1[lockbox[:32]],
            member2[lockbox[:32]],
            device=args.device,
        )
        reload_errors[str(seed)] = float(
            np.max(np.abs(original_prediction - reloaded_prediction))
        )
        predicted = predict(
            reloaded,
            member1[lockbox],
            member2[lockbox],
            device=args.device,
        )
        lockbox_predictions.append(predicted)
        development_metrics[str(seed)] = regression_metrics(
            modeling.iloc[lockbox]["pEC50"].to_numpy(),
            predicted,
        )

    observed_lockbox = modeling.iloc[lockbox]["pEC50"].to_numpy()
    ensemble_prediction = np.mean(np.stack(lockbox_predictions), axis=0)
    affine_intercept, affine_slope = affine_fit(
        frozen_pec50[development],
        modeling.iloc[development]["pEC50"].to_numpy(),
    )
    affine_prediction = affine_intercept + affine_slope * frozen_pec50[lockbox]
    lockbox_frame = modeling.iloc[lockbox][
        ["feature_row", "record_id", "original_id", "pEC50"]
    ].copy()
    lockbox_frame["frozen_pEC50"] = frozen_pec50[lockbox]
    lockbox_frame["affine_pEC50"] = affine_prediction
    for seed, predicted in zip(args.seeds, lockbox_predictions, strict=True):
        lockbox_frame[f"predicted_pEC50_seed_{seed}"] = predicted
    lockbox_frame["predicted_pEC50_ensemble"] = ensemble_prediction
    lockbox_frame.to_csv(args.output_dir / "holdout_predictions.csv", index=False)

    lockbox_summary = {
        "status": "pass",
        "rows": int(len(lockbox)),
        "used_once_after_configuration_freeze": True,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "per_seed": development_metrics,
        "ensemble": regression_metrics(observed_lockbox, ensemble_prediction),
        "frozen": regression_metrics(observed_lockbox, frozen_pec50[lockbox]),
        "affine": regression_metrics(observed_lockbox, affine_prediction),
        "affine_intercept": affine_intercept,
        "affine_slope": affine_slope,
        "reload_max_abs_by_seed": reload_errors,
    }
    (args.output_dir / "holdout_summary.json").write_text(
        json.dumps(lockbox_summary, indent=2, sort_keys=True) + "\n"
    )

    final_histories: list[pd.DataFrame] = []
    final_checkpoints: list[str] = []
    for seed in args.seeds:
        config = CachedHeadConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.epochs,
            patience=args.epochs,
            seed=seed,
        )
        state, history = fit_fixed_epochs(
            member1,
            member2,
            target,
            args.checkpoint_weights,
            config,
            device=args.device,
        )
        checkpoint_path = final_dir / f"seed_{seed}.safetensors"
        save_file(state, checkpoint_path)
        history["scope"] = "all_4134_labels"
        final_histories.append(history)
        final_checkpoints.append(str(checkpoint_path))
    pd.concat(final_histories, ignore_index=True).to_csv(
        final_dir / "history.csv",
        index=False,
    )

    summary = {
        "status": "pass",
        "training_complete": True,
        "selected_configuration": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "loss": "unweighted_huber_50_25_25",
        },
        "development_rows": int(len(development)),
        "lockbox_rows": int(len(lockbox)),
        "final_training_rows": int(len(modeling)),
        "lockbox": lockbox_summary,
        "final_checkpoints": final_checkpoints,
        "features_sha256": sha256_file(args.features),
        "feature_metadata_sha256": sha256_file(args.feature_metadata),
        "modeling_manifest_sha256": sha256_file(args.modeling_manifest),
        "checkpoint_weights_sha256": sha256_file(args.checkpoint_weights),
        "config_template": asdict(
            CachedHeadConfig(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
                patience=args.epochs,
                seed=args.seeds[0],
            )
        ),
    }
    (args.output_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
