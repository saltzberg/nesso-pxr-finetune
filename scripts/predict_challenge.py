#!/usr/bin/env python3
"""Prepare, predict, and evaluate the 513-compound PXR challenge set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nesso_pxr.protocol import sha256_file
from nesso_pxr.smoke import record_id_for_smiles
from nesso_pxr.train_cached import make_twin_heads, regression_metrics


def prepare_selection(
    blinded_path: Path,
    prepared_path: Path,
    processed_root: Path,
    reference_root: Path,
    esm_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    blinded = pd.read_csv(blinded_path)
    prepared = pd.read_csv(prepared_path)
    prepared_columns = [
        "original_ligand_id",
        "prepared_ligand_id",
        "prepared_smiles",
        "formal_charge",
        "selection_reason",
    ]
    selection = blinded.rename(
        columns={"Molecule Name": "original_id", "SMILES": "source_smiles"}
    ).merge(
        prepared[prepared_columns],
        left_on="original_id",
        right_on="original_ligand_id",
        how="left",
        validate="one_to_one",
    )
    if len(selection) != 513 or selection["prepared_smiles"].isna().any():
        raise ValueError("expected 513 challenge compounds with prepared states")
    selection["record_id"] = selection["prepared_smiles"].map(record_id_for_smiles)
    if selection["record_id"].duplicated().any():
        raise ValueError("challenge record IDs must be unique")

    references: list[dict[str, float]] = []
    missing: list[str] = []
    for record_id in selection["record_id"]:
        required = [
            processed_root / "records" / f"{record_id}.json",
            processed_root / "structures" / f"{record_id}.npz",
            processed_root / "rdkit_conformers" / f"{record_id}__B.pkl",
        ]
        reference_path = reference_root / record_id / "affinity.json"
        if not all(path.is_file() for path in required) or not reference_path.is_file():
            missing.append(record_id)
            continue
        reference = json.loads(reference_path.read_text())
        references.append(
            {
                "record_id": record_id,
                "reference_affinity_pred_value1": float(
                    reference["affinity_pred_value1"]
                ),
                "reference_affinity_pred_value2": float(
                    reference["affinity_pred_value2"]
                ),
                "reference_affinity_pred_value": float(
                    reference["affinity_pred_value"]
                ),
            }
        )
    if missing:
        raise ValueError(f"missing challenge Nesso artifacts: {missing[:10]}")
    selection = selection.merge(
        pd.DataFrame(references), on="record_id", validate="one_to_one"
    )
    selection = selection.sort_values("record_id", kind="stable").reset_index(drop=True)
    selection.insert(0, "feature_row", np.arange(len(selection), dtype=np.int64))
    forbidden = {"pEC50", "phase", "truth"}.intersection(selection.columns)
    if forbidden:
        raise ValueError(f"label columns leaked into challenge selection: {forbidden}")

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.csv"
    selection.to_csv(selection_path, index=False)
    overlay = output_dir / "processed"
    overlay.mkdir(parents=True, exist_ok=True)
    for directory in ("records", "structures", "rdkit_conformers"):
        destination = overlay / directory
        source = (processed_root / directory).resolve(strict=True)
        if destination.is_symlink():
            if destination.resolve(strict=True) != source:
                raise FileExistsError(f"overlay target mismatch: {destination}")
        elif destination.exists():
            raise FileExistsError(f"refusing to replace overlay path: {destination}")
        else:
            destination.symlink_to(source, target_is_directory=True)
    esm_dir = overlay / "esm_embeddings"
    esm_dir.mkdir(parents=True, exist_ok=True)
    esm_destination = esm_dir / "9ee8e7ba7e2644473a2ccd2a9ff22928.safetensors"
    esm_source = esm_path.resolve(strict=True)
    if esm_destination.is_symlink():
        if esm_destination.resolve(strict=True) != esm_source:
            raise FileExistsError(f"ESM target mismatch: {esm_destination}")
    elif esm_destination.exists():
        raise FileExistsError(f"refusing to replace ESM path: {esm_destination}")
    else:
        esm_destination.symlink_to(esm_source)
    audit = {
        "status": "pass",
        "rows": int(len(selection)),
        "unique_original_ids": int(selection["original_id"].nunique()),
        "unique_record_ids": int(selection["record_id"].nunique()),
        "label_columns_present": [],
        "blinded_sha256": sha256_file(blinded_path),
        "prepared_sha256": sha256_file(prepared_path),
        "esm_sha256": sha256_file(esm_path),
        "selection_sha256": sha256_file(selection_path),
        "processed_overlay": str(overlay),
    }
    (output_dir / "selection_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def _predict_checkpoint(
    state_path: Path,
    member1: Any,
    member2: Any,
    device: str,
) -> np.ndarray:
    from safetensors.torch import load_file

    model = make_twin_heads()
    model.load_state_dict(load_file(str(state_path), device="cpu"), strict=True)
    model.to(device).eval()
    import torch

    with torch.inference_mode():
        _, _, prediction = model(member1.to(device), member2.to(device))
    return 6.0 - prediction.detach().cpu().numpy()


def evaluate_predictions(
    features_path: Path,
    metadata_path: Path,
    blinded_path: Path,
    truth_paths: list[Path],
    checkpoints: list[Path],
    output_dir: Path,
    *,
    device: str,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    if len(checkpoints) != 3:
        raise ValueError("expected exactly three final checkpoints")
    tensors = load_file(str(features_path), device="cpu")
    metadata = pd.read_csv(metadata_path).sort_values("feature_row")
    if len(metadata) != 513 or metadata["feature_row"].tolist() != list(range(513)):
        raise ValueError("challenge feature metadata must have 513 ordered rows")
    member1 = tensors["member1_affinity_repr"].float()
    member2 = tensors["member2_affinity_repr"].float()
    untrained = 6.0 - tensors["affinity_pred_value"].numpy()
    trained_members = [
        _predict_checkpoint(path, member1, member2, device) for path in checkpoints
    ]
    trained = np.mean(np.stack(trained_members), axis=0)

    predictions = metadata[
        ["feature_row", "record_id", "original_id", "source_smiles"]
    ].copy()
    predictions["untrained_pEC50"] = untrained
    for index, values in enumerate(trained_members, start=1):
        predictions[f"trained_member_{index}_pEC50"] = values
    predictions["trained_ensemble_pEC50"] = trained

    blinded = pd.read_csv(blinded_path)
    truth = pd.concat([pd.read_csv(path) for path in truth_paths], ignore_index=True)
    if len(truth) != 513 or truth["Molecule Name"].nunique() != 513:
        raise ValueError("unblinded truth must cover 513 unique challenge compounds")
    if set(truth["Molecule Name"]) != set(blinded["Molecule Name"]):
        raise ValueError("unblinded IDs differ from the original challenge set")
    predictions = predictions.merge(
        truth[["Molecule Name", "SMILES", "pEC50", "phase"]],
        left_on="original_id",
        right_on="Molecule Name",
        how="left",
        validate="one_to_one",
    )
    if not predictions["source_smiles"].eq(predictions["SMILES"]).all():
        raise ValueError("challenge SMILES changed between blinded and truth files")
    challenge_order = {
        molecule_id: index for index, molecule_id in enumerate(blinded["Molecule Name"])
    }
    predictions["challenge_row"] = predictions["original_id"].map(challenge_order)
    predictions = predictions.sort_values("challenge_row", kind="stable")
    observed = predictions["pEC50"].to_numpy()
    untrained_evaluation = predictions["untrained_pEC50"].to_numpy()
    trained_evaluation = predictions["trained_ensemble_pEC50"].to_numpy()
    summary: dict[str, Any] = {
        "status": "pass",
        "rows": int(len(predictions)),
        "truth_joined_after_prediction": True,
        "untrained": regression_metrics(observed, untrained_evaluation),
        "trained_ensemble": regression_metrics(observed, trained_evaluation),
        "delta_mae": float(
            regression_metrics(observed, untrained_evaluation)["mae"]
            - regression_metrics(observed, trained_evaluation)["mae"]
        ),
        "delta_spearman": float(
            regression_metrics(observed, trained_evaluation)["spearman"]
            - regression_metrics(observed, untrained_evaluation)["spearman"]
        ),
        "per_phase": {},
        "features_sha256": sha256_file(features_path),
        "metadata_sha256": sha256_file(metadata_path),
        "checkpoint_sha256": [sha256_file(path) for path in checkpoints],
        "truth_sha256": [sha256_file(path) for path in truth_paths],
    }
    for phase, frame in predictions.groupby("phase"):
        phase_observed = frame["pEC50"].to_numpy()
        summary["per_phase"][str(int(phase))] = {
            "rows": int(len(frame)),
            "untrained": regression_metrics(
                phase_observed, frame["untrained_pEC50"].to_numpy()
            ),
            "trained_ensemble": regression_metrics(
                phase_observed, frame["trained_ensemble_pEC50"].to_numpy()
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "challenge_predictions_with_truth.csv", index=False)
    predictions[["source_smiles", "original_id", "trained_ensemble_pEC50"]].rename(
        columns={
            "source_smiles": "SMILES",
            "original_id": "Molecule Name",
            "trained_ensemble_pEC50": "pEC50",
        }
    ).to_csv(output_dir / "challenge_submission_trained.csv", index=False)
    predictions[["source_smiles", "original_id", "untrained_pEC50"]].rename(
        columns={
            "source_smiles": "SMILES",
            "original_id": "Molecule Name",
            "untrained_pEC50": "pEC50",
        }
    ).to_csv(output_dir / "challenge_submission_untrained.csv", index=False)
    (output_dir / "challenge_evaluation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--blinded", type=Path, required=True)
    prepare.add_argument("--prepared", type=Path, required=True)
    prepare.add_argument("--processed-root", type=Path, required=True)
    prepare.add_argument("--reference-root", type=Path, required=True)
    prepare.add_argument("--esm", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--metadata", type=Path, required=True)
    evaluate.add_argument("--blinded", type=Path, required=True)
    evaluate.add_argument("--truth", type=Path, action="append", required=True)
    evaluate.add_argument("--checkpoint", type=Path, action="append", required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.command == "prepare":
        summary = prepare_selection(
            args.blinded,
            args.prepared,
            args.processed_root,
            args.reference_root,
            args.esm,
            args.output_dir,
        )
    else:
        summary = evaluate_predictions(
            args.features,
            args.metadata,
            args.blinded,
            args.truth,
            args.checkpoint,
            args.output_dir,
            device=args.device,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
