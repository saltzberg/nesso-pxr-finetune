#!/usr/bin/env python3
"""Fine-tune a bounded PXR activity adapter initialized from both Nesso branches.

The cached 384D representations remain frozen. Both pretrained continuous and
binary pathways initialize the activity readout, but the training label is
functional pEC50; this script does not fine-tune or evaluate a binding classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "artifacts" / "experiments" / "binary_head_reanalysis"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "models" / "nesso-1" / "v1.0.0" / "model.safetensors"
)
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--anchor-lambda", type=float, default=1e-4)
    parser.add_argument("--max-binary-shift", type=float, default=1.5)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_branch() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(384, 384),
        nn.ReLU(),
        nn.Linear(384, 384),
        nn.ReLU(),
        nn.Linear(384, 1),
    )


def load_branch(
    state: dict[str, torch.Tensor], prefix: str
) -> nn.Sequential:
    branch = make_branch()
    branch.load_state_dict(
        {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        },
        strict=True,
    )
    return branch


class MemberActivityAdapter(nn.Module):
    def __init__(
        self,
        checkpoint: dict[str, torch.Tensor],
        module_prefix: str,
        max_binary_shift: float,
    ) -> None:
        super().__init__()
        prefix = f"{module_prefix}.affinity_heads"
        self.continuous = load_branch(
            checkpoint, f"{prefix}.to_affinity_pred_value."
        )
        self.binary_score = load_branch(
            checkpoint, f"{prefix}.to_affinity_pred_score."
        )
        self.binary_logit = nn.Linear(1, 1)
        self.binary_logit.load_state_dict(
            {
                key.removeprefix(f"{prefix}.to_affinity_logits_binary."): value
                for key, value in checkpoint.items()
                if key.startswith(f"{prefix}.to_affinity_logits_binary.")
            },
            strict=True,
        )
        self.binary_scale = nn.Parameter(torch.zeros(()))
        self.binary_offset = nn.Parameter(torch.zeros(()))
        self.max_binary_shift = float(max_binary_shift)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        continuous = self.continuous(representation).squeeze(-1)
        binary_logit = self.binary_logit(
            self.binary_score(representation)
        ).squeeze(-1)
        adjustment = self.max_binary_shift * torch.tanh(
            self.binary_scale * binary_logit + self.binary_offset
        )
        return continuous + adjustment


class TwinActivityAdapter(nn.Module):
    def __init__(
        self,
        checkpoint: dict[str, torch.Tensor],
        max_binary_shift: float,
    ) -> None:
        super().__init__()
        self.member1 = MemberActivityAdapter(
            checkpoint, "affinity_module", max_binary_shift
        )
        self.member2 = MemberActivityAdapter(
            checkpoint, "affinity_module2", max_binary_shift
        )

    def forward(
        self, member1: torch.Tensor, member2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction1 = self.member1(member1)
        prediction2 = self.member2(member2)
        return prediction1, prediction2, (prediction1 + prediction2) / 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def branch_parameter_anchors(model: TwinActivityAdapter) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if "binary_scale" not in name and "binary_offset" not in name
    }


def anchor_penalty(
    model: TwinActivityAdapter, anchors: dict[str, torch.Tensor]
) -> torch.Tensor:
    numerator = torch.zeros((), dtype=torch.float32)
    count = 0
    for name, value in model.named_parameters():
        if name not in anchors:
            continue
        numerator = numerator + torch.sum(torch.square(value - anchors[name]))
        count += value.numel()
    return numerator / max(count, 1)


def combined_huber(
    member1: torch.Tensor,
    member2: torch.Tensor,
    ensemble: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (
        0.5 * functional.huber_loss(ensemble, target, delta=0.5)
        + 0.25 * functional.huber_loss(member1, target, delta=0.5)
        + 0.25 * functional.huber_loss(member2, target, delta=0.5)
    )


def train_model(
    checkpoint: dict[str, torch.Tensor],
    member1: torch.Tensor,
    member2: torch.Tensor,
    target_raw: torch.Tensor,
    training_indices: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    anchor_lambda: float,
    max_binary_shift: float,
) -> tuple[TwinActivityAdapter, list[dict[str, Any]]]:
    set_seed(seed)
    model = TwinActivityAdapter(checkpoint, max_binary_shift)
    anchors = branch_parameter_anchors(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=0.0
    )
    dataset = TensorDataset(
        member1[training_indices],
        member2[training_indices],
        target_raw[training_indices],
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_data_loss = 0.0
        total_examples = 0
        last_gradient_norm = np.nan
        for batch1, batch2, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction1, prediction2, ensemble = model(batch1, batch2)
            data_loss = combined_huber(
                prediction1, prediction2, ensemble, batch_target
            )
            penalty = anchor_penalty(model, anchors)
            loss = data_loss + anchor_lambda * penalty
            loss.backward()
            last_gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            )
            optimizer.step()
            total_data_loss += float(data_loss.detach()) * len(batch_target)
            total_examples += len(batch_target)
        history.append(
            {
                "epoch": epoch,
                "data_loss": total_data_loss / total_examples,
                "anchor_penalty": float(anchor_penalty(model, anchors).detach()),
                "gradient_norm_last_batch": last_gradient_norm,
                "member1_binary_scale": float(model.member1.binary_scale.detach()),
                "member1_binary_offset": float(model.member1.binary_offset.detach()),
                "member2_binary_scale": float(model.member2.binary_scale.detach()),
                "member2_binary_offset": float(model.member2.binary_offset.detach()),
            }
        )
    model.eval()
    return model, history


def predict(
    model: TwinActivityAdapter,
    member1: torch.Tensor,
    member2: torch.Tensor,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.inference_mode():
        raw1, raw2, ensemble_raw = model(member1[indices], member2[indices])
    return (
        (6.0 - raw1).numpy(),
        (6.0 - raw2).numpy(),
        (6.0 - ensemble_raw).numpy(),
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_num_interop_threads(1)
    checkpoint = load_file(str(args.checkpoint), device="cpu")
    manifest = read_csv(REPO_ROOT / "data" / "published" / "modeling_manifest.csv")
    member1 = torch.from_numpy(
        np.load(args.input_dir / "labeled_member1_repr.npy")
    ).float()
    member2 = torch.from_numpy(
        np.load(args.input_dir / "labeled_member2_repr.npy")
    ).float()
    challenge_member1 = torch.from_numpy(
        np.load(args.input_dir / "challenge_member1_repr.npy")
    ).float()
    challenge_member2 = torch.from_numpy(
        np.load(args.input_dir / "challenge_member2_repr.npy")
    ).float()
    development_positions = np.asarray(
        [
            int(row["feature_row"])
            for row in manifest
            if row["modeling_role"] == "development"
        ],
        dtype=int,
    )
    validation_positions = np.asarray(
        [
            int(row["feature_row"])
            for row in manifest
            if row["modeling_role"] == "lockbox"
        ],
        dtype=int,
    )
    development_rows = [
        row for row in manifest if row["modeling_role"] == "development"
    ]
    validation_rows = [row for row in manifest if row["modeling_role"] == "lockbox"]
    development_folds = np.asarray(
        [int(float(row["fold"])) for row in development_rows], dtype=int
    )
    development_target = torch.tensor(
        [6.0 - float(row["pEC50"]) for row in development_rows],
        dtype=torch.float32,
    )
    development_member1 = member1[development_positions]
    development_member2 = member2[development_positions]
    validation_member1 = member1[validation_positions]
    validation_member2 = member2[validation_positions]

    initial_model = TwinActivityAdapter(checkpoint, args.max_binary_shift).eval()
    with torch.inference_mode():
        _, _, initial_raw = initial_model(member1, member2)
    reconstructed = read_csv(args.input_dir / "labeled_binary_readouts.csv")
    expected_initial = np.asarray(
        [float(row["pIC50_equivalent"]) for row in reconstructed]
    )
    initialization_parity = float(
        np.max(np.abs((6.0 - initial_raw.numpy()) - expected_initial))
    )
    if initialization_parity > 1e-6:
        raise AssertionError("dual-branch adapter does not preserve initial affinity output")

    seed_development_predictions: dict[int, np.ndarray] = {}
    seed_validation_predictions: dict[int, np.ndarray] = {}
    seed_challenge_predictions: dict[int, np.ndarray] = {}
    history_rows: list[dict[str, Any]] = []
    final_parameter_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        oof = np.full(len(development_rows), np.nan, dtype=float)
        for fold in sorted(np.unique(development_folds)):
            training = np.flatnonzero(development_folds != fold)
            held_out = np.flatnonzero(development_folds == fold)
            model, history = train_model(
                checkpoint,
                development_member1,
                development_member2,
                development_target,
                training,
                seed=seed,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                anchor_lambda=args.anchor_lambda,
                max_binary_shift=args.max_binary_shift,
            )
            _, _, fold_prediction = predict(
                model, development_member1, development_member2, held_out
            )
            oof[held_out] = fold_prediction
            for row in history:
                history_rows.append(
                    {"seed": seed, "fit_scope": f"outer_fold_{fold}", **row}
                )
        seed_development_predictions[seed] = oof

        all_development = np.arange(len(development_rows), dtype=int)
        full_model, history = train_model(
            checkpoint,
            development_member1,
            development_member2,
            development_target,
            all_development,
            seed=seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            anchor_lambda=args.anchor_lambda,
            max_binary_shift=args.max_binary_shift,
        )
        for row in history:
            history_rows.append(
                {"seed": seed, "fit_scope": "all_development", **row}
            )
        _, _, validation_prediction = predict(
            full_model,
            validation_member1,
            validation_member2,
            np.arange(len(validation_rows), dtype=int),
        )
        _, _, challenge_prediction = predict(
            full_model,
            challenge_member1,
            challenge_member2,
            np.arange(len(challenge_member1), dtype=int),
        )
        seed_validation_predictions[seed] = validation_prediction
        seed_challenge_predictions[seed] = challenge_prediction
        final_parameter_rows.append(
            {
                "seed": seed,
                "member1_binary_scale": float(full_model.member1.binary_scale.detach()),
                "member1_binary_offset": float(full_model.member1.binary_offset.detach()),
                "member2_binary_scale": float(full_model.member2.binary_scale.detach()),
                "member2_binary_offset": float(full_model.member2.binary_offset.detach()),
                "final_anchor_penalty": history[-1]["anchor_penalty"],
            }
        )

    prediction_rows: list[dict[str, Any]] = []
    for dataset, rows, predictions in (
        ("development", development_rows, seed_development_predictions),
        ("validation", validation_rows, seed_validation_predictions),
    ):
        ensemble = np.mean(np.stack(list(predictions.values())), axis=0)
        for position, row in enumerate(rows):
            prediction_rows.append(
                {
                    "dataset": dataset,
                    "feature_row": int(row["feature_row"]),
                    "original_id": row["original_id"],
                    "pEC50": float(row["pEC50"]),
                    "fold": row["fold"] if dataset == "development" else "",
                    "seed_42_predicted_pEC50": predictions[42][position],
                    "seed_43_predicted_pEC50": predictions[43][position],
                    "seed_44_predicted_pEC50": predictions[44][position],
                    "predicted_pEC50": ensemble[position],
                }
            )
    challenge_metadata = read_csv(
        REPO_ROOT
        / "artifacts"
        / "experiments"
        / "challenge"
        / "run2"
        / "metadata.csv"
    )
    challenge_truth = {
        row["record_id"]: row
        for row in read_csv(
            REPO_ROOT / "data" / "published" / "nesso_challenge_predictions.csv"
        )
    }
    challenge_ensemble = np.mean(
        np.stack(list(seed_challenge_predictions.values())), axis=0
    )
    for position, row in enumerate(challenge_metadata):
        truth = challenge_truth[row["record_id"]]
        prediction_rows.append(
            {
                "dataset": "challenge",
                "feature_row": int(row["feature_row"]),
                "original_id": row["original_id"],
                "pEC50": float(truth["pEC50"]),
                "fold": "",
                "seed_42_predicted_pEC50": seed_challenge_predictions[42][position],
                "seed_43_predicted_pEC50": seed_challenge_predictions[43][position],
                "seed_44_predicted_pEC50": seed_challenge_predictions[44][position],
                "predicted_pEC50": challenge_ensemble[position],
            }
        )
    write_csv(
        args.input_dir / "dual_branch_activity_adapter_predictions.csv",
        prediction_rows,
    )
    write_csv(args.input_dir / "dual_branch_activity_adapter_history.csv", history_rows)
    write_csv(
        args.input_dir / "dual_branch_activity_adapter_parameters.csv",
        final_parameter_rows,
    )
    metadata = {
        "status": "pass",
        "name": "fine_tuned_dual_branch_activity_adapter",
        "label_semantics": "functional pEC50 only; not a binding-classifier fine-tune",
        "frozen_components": "cached 384D representations",
        "trainable_components": (
            "pretrained continuous and binary readout branches plus bounded scalar fusion"
        ),
        "initialization": (
            "continuous affinity output exactly; binary contribution initialized to zero"
        ),
        "initialization_parity_max_abs": initialization_parity,
        "max_binary_shift": args.max_binary_shift,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "epoch_choice": (
            "prespecified from the historical continuous-head development refit; "
            "no outer-fold or validation labels selected training duration"
        ),
        "huber_delta": 0.5,
        "anchor_lambda": args.anchor_lambda,
        "seeds": list(SEEDS),
        "cpu_threads": torch.get_num_threads(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.input_dir / "dual_branch_activity_adapter_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
