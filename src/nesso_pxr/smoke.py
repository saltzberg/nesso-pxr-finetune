"""Deterministic 16-compound GPU feature-extraction smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nesso_pxr.protocol import sha256_file

FEATURE_TENSOR_NAMES = (
    "member1_affinity_repr",
    "member2_affinity_repr",
    "affinity_pred_value1",
    "affinity_pred_value2",
    "affinity_pred_value",
    "reconstructed_affinity_pred_value1",
    "reconstructed_affinity_pred_value2",
)


def record_id_for_smiles(smiles: str) -> str:
    """Return the stable record ID used by the existing Nesso campaign."""

    return "nesso_" + hashlib.sha256(smiles.encode()).hexdigest()[:20]


def allocate_fold_counts(total: int, folds: list[int]) -> dict[int, int]:
    """Allocate a requested sample count as evenly as possible across folds."""

    if total < len(folds):
        raise ValueError("total must be at least the number of folds")
    quotient, remainder = divmod(total, len(folds))
    return {
        fold: quotient + (position < remainder)
        for position, fold in enumerate(sorted(folds))
    }


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    """Choose deterministic range-covering indices without duplicates."""

    if count < 1 or size < count:
        raise ValueError("require 1 <= count <= size")
    if count == 1:
        return [size // 2]
    raw = np.linspace(0, size - 1, num=count)
    indices = [int(round(value)) for value in raw]
    if len(set(indices)) != count:
        raise RuntimeError("evenly spaced index construction produced duplicates")
    return indices


def _required_record_artifacts(processed_root: Path, record_id: str) -> list[Path]:
    return [
        processed_root / "records" / f"{record_id}.json",
        processed_root / "structures" / f"{record_id}.npz",
        processed_root / "rdkit_conformers" / f"{record_id}__B.pkl",
    ]


def select_smoke_compounds(
    training_manifest_path: Path,
    prepared_path: Path,
    processed_root: Path,
    reference_root: Path,
    *,
    total: int = 16,
) -> pd.DataFrame:
    """Select a potency-range and fold-stratified DRC smoke subset."""

    manifest = pd.read_csv(training_manifest_path)
    prepared = pd.read_csv(prepared_path)
    primary = manifest.loc[
        manifest["source"].eq("drc")
        & manifest["role"].eq("development_primary")
        & manifest["fold"].notna()
    ].copy()
    primary["fold"] = primary["fold"].astype(int)

    prepared_columns = [
        "original_ligand_id",
        "prepared_ligand_id",
        "prepared_smiles",
        "formal_charge",
        "selection_reason",
    ]
    primary = primary.merge(
        prepared[prepared_columns],
        left_on="original_id",
        right_on="original_ligand_id",
        how="left",
        validate="one_to_one",
    )
    if primary["prepared_smiles"].isna().any():
        missing = primary.loc[primary["prepared_smiles"].isna(), "original_id"].tolist()
        raise ValueError(f"missing prepared top states: {missing[:10]}")

    primary["record_id"] = primary["prepared_smiles"].map(record_id_for_smiles)
    eligible: list[dict[str, Any]] = []
    for row in primary.to_dict(orient="records"):
        record_id = row["record_id"]
        required = _required_record_artifacts(processed_root, record_id)
        reference_path = reference_root / record_id / "affinity.json"
        if not all(path.is_file() for path in required) or not reference_path.is_file():
            continue
        reference = json.loads(reference_path.read_text())
        for key in (
            "affinity_pred_value",
            "affinity_pred_value1",
            "affinity_pred_value2",
        ):
            if key not in reference:
                raise ValueError(f"reference {reference_path} lacks {key}")
            row[f"reference_{key}"] = float(reference[key])
        eligible.append(row)

    eligible_frame = pd.DataFrame.from_records(eligible)
    folds = sorted(primary["fold"].unique().tolist())
    counts = allocate_fold_counts(total, folds)
    selected: list[pd.DataFrame] = []
    for fold in folds:
        candidates = eligible_frame.loc[eligible_frame["fold"].eq(fold)].sort_values(
            ["pEC50", "original_id"], kind="stable"
        )
        count = counts[fold]
        if len(candidates) < count:
            raise ValueError(
                f"fold {fold} has only {len(candidates)} reference-complete candidates"
            )
        selected.append(candidates.iloc[evenly_spaced_indices(len(candidates), count)])

    result = pd.concat(selected, ignore_index=True)
    result = result.sort_values(["record_id"], kind="stable")
    result = result.reset_index(drop=True)
    result.insert(0, "feature_row", np.arange(len(result), dtype=int))
    if len(result) != total or result["record_id"].duplicated().any():
        raise AssertionError("invalid smoke selection")
    return result


def _register_representation_hooks(model: Any, captured: dict[str, Any]) -> list[Any]:
    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: dict[str, Any]) -> None:
            captured[name] = output["affinity_repr"].detach()

        return hook

    return [
        model.affinity_module.affinity_heads.register_forward_hook(capture("member1")),
        model.affinity_module2.affinity_heads.register_forward_hook(capture("member2")),
    ]


def _feature_schema(
    *,
    selection_path: Path,
    checkpoint: Path,
    recycling_steps: int,
    reference_tolerance: float,
) -> dict[str, Any]:
    return {
        "format": "safetensors",
        "row_order": "metadata.csv:feature_row",
        "representation_dimension": 384,
        "representation_dtype": "float32",
        "tensors": {
            "member1_affinity_repr": {
                "shape": ["N", 384],
                "meaning": "Nesso affinity member 1 pre-regression representation",
            },
            "member2_affinity_repr": {
                "shape": ["N", 384],
                "meaning": "Nesso affinity member 2 pre-regression representation",
            },
            "affinity_pred_value1": {
                "shape": ["N"],
                "meaning": "standard member 1 output",
            },
            "affinity_pred_value2": {
                "shape": ["N"],
                "meaning": "standard member 2 output",
            },
            "affinity_pred_value": {
                "shape": ["N"],
                "meaning": "standard ensemble mean",
            },
            "reconstructed_affinity_pred_value1": {
                "shape": ["N"],
                "meaning": "member 1 MLP reapplied to cached representation",
            },
            "reconstructed_affinity_pred_value2": {
                "shape": ["N"],
                "meaning": "member 2 MLP reapplied to cached representation",
            },
        },
        "inference": {
            "recycling_steps": recycling_steps,
            "refine_protein_inference": True,
            "refine_protein_cutoff": 22.0,
            "refine_protein_tokens_budget": 256,
            "affinity_protein_cutoff": 15.0,
            "precision": "bf16-mixed trunk; upstream affinity autocast disabled",
            "model_mode": "eval",
            "num_workers": 1,
        },
        "selection_sha256": sha256_file(selection_path),
        "checkpoint_hparams_sha256": sha256_file(checkpoint / "hparams.json"),
        "checkpoint_weights_sha256": sha256_file(checkpoint / "model.safetensors"),
        "reference_tolerance": reference_tolerance,
    }


def compare_smoke_runs(
    first_dir: Path,
    second_dir: Path,
    output_path: Path,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare two feature bundles for row and numerical determinism."""

    from safetensors.torch import load_file

    first_metadata = pd.read_csv(first_dir / "metadata.csv")
    second_metadata = pd.read_csv(second_dir / "metadata.csv")
    ids_equal = (
        first_metadata["record_id"].tolist() == second_metadata["record_id"].tolist()
    )
    first = load_file(first_dir / "features.safetensors")
    second = load_file(second_dir / "features.safetensors")
    tensor_names_equal = set(first) == set(second)
    differences: dict[str, float] = {}
    shapes_equal = True
    for name in sorted(set(first) & set(second)):
        if first[name].shape != second[name].shape:
            shapes_equal = False
            differences[name] = math.inf
        else:
            differences[name] = float((first[name] - second[name]).abs().max().item())
    maximum = max(differences.values(), default=math.inf)
    summary = {
        "status": (
            "pass"
            if ids_equal
            and tensor_names_equal
            and shapes_equal
            and maximum <= tolerance
            else "fail"
        ),
        "record_ids_equal": ids_equal,
        "tensor_names_equal": tensor_names_equal,
        "tensor_shapes_equal": shapes_equal,
        "max_abs_by_tensor": differences,
        "max_abs_overall": maximum,
        "tolerance": tolerance,
        "first_features_sha256": sha256_file(first_dir / "features.safetensors"),
        "second_features_sha256": sha256_file(second_dir / "features.safetensors"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select")
    select.add_argument("--training-manifest", type=Path, required=True)
    select.add_argument("--prepared", type=Path, required=True)
    select.add_argument("--processed-root", type=Path, required=True)
    select.add_argument("--reference-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--total", type=int, default=16)

    extract = commands.add_parser("extract")
    extract.add_argument("--selection", type=Path, required=True)
    extract.add_argument("--processed-root", type=Path, required=True)
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--ccd", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--seed", type=int, default=42)
    extract.add_argument("--recycling-steps", type=int, default=5)
    extract.add_argument("--reference-tolerance", type=float, default=1e-5)
    extract.add_argument(
        "--allow-missing-references",
        action="store_true",
    )
    extract.add_argument("--progress", action="store_true")

    compare = commands.add_parser("compare")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--tolerance", type=float, default=1e-6)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "select":
        selection = select_smoke_compounds(
            args.training_manifest,
            args.prepared,
            args.processed_root,
            args.reference_root,
            total=args.total,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        selection.to_csv(args.output, index=False)
        print(
            json.dumps(
                {
                    "selected": len(selection),
                    "fold_counts": {
                        str(key): int(value)
                        for key, value in selection["fold"]
                        .value_counts()
                        .sort_index()
                        .items()
                    },
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "extract":
        from nesso_pxr.smoke_lightning import (
            extract_smoke_features_lightning,
        )

        summary = extract_smoke_features_lightning(
            args.selection,
            args.processed_root,
            args.checkpoint,
            args.ccd,
            args.output_dir,
            seed=args.seed,
            recycling_steps=args.recycling_steps,
            reference_tolerance=args.reference_tolerance,
            require_reference=not args.allow_missing_references,
            progress=args.progress,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if summary["status"] != "pass":
            raise SystemExit(2)
    else:
        summary = compare_smoke_runs(
            args.first,
            args.second,
            args.output,
            tolerance=args.tolerance,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if summary["status"] != "pass":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
