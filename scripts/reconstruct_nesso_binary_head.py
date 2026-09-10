#!/usr/bin/env python3
"""Reconstruct Nesso-1 binary readouts from cached 384D affinity representations.

This script intentionally depends only on Python's standard library, NumPy,
PyTorch, and safetensors so it can run inside the pinned Nesso 1.0.0
image. It does not run the structure trunk and does not require a GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "nesso-1" / "v1.0.0" / "model.safetensors"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "experiments" / "binary_head_reanalysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--historical-json-dir",
        type=Path,
        default=None,
        help="Optional earlier Nesso predictions directory for a provenance audit",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_branch(checkpoint: dict[str, torch.Tensor], prefix: str) -> nn.Sequential:
    branch = make_branch()
    state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }
    branch.load_state_dict(state, strict=True)
    branch.eval()
    return branch


def load_binary_logit_layer(
    checkpoint: dict[str, torch.Tensor], prefix: str
) -> nn.Linear:
    layer = nn.Linear(1, 1)
    state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }
    layer.load_state_dict(state, strict=True)
    layer.eval()
    return layer


def reconstruct_variant(
    member1_repr: torch.Tensor,
    member2_repr: torch.Tensor,
    checkpoint: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    outputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        member_probabilities: list[torch.Tensor] = []
        for index, representation in enumerate((member1_repr, member2_repr), start=1):
            module_prefix = "affinity_module" if index == 1 else "affinity_module2"
            head_prefix = f"{module_prefix}.affinity_heads"
            continuous = load_branch(
                checkpoint, f"{head_prefix}.to_affinity_pred_value."
            )(representation).squeeze(-1)
            score = load_branch(checkpoint, f"{head_prefix}.to_affinity_pred_score.")(
                representation
            )
            binary_logit = load_binary_logit_layer(
                checkpoint, f"{head_prefix}.to_affinity_logits_binary."
            )(score).squeeze(-1)
            probability = torch.sigmoid(binary_logit)
            outputs[f"member{index}_affinity_pred_value"] = continuous
            outputs[f"member{index}_binary_logit"] = binary_logit
            outputs[f"member{index}_binder_probability"] = probability
            member_probabilities.append(probability)
        outputs["affinity_pred_value"] = (
            outputs["member1_affinity_pred_value"]
            + outputs["member2_affinity_pred_value"]
        ) / 2.0
        ensemble_probability = (member_probabilities[0] + member_probabilities[1]) / 2.0
        outputs["binder_probability"] = ensemble_probability
        outputs["binary_logit"] = torch.logit(
            ensemble_probability.clamp(1e-6, 1.0 - 1e-6)
        )
    return {key: value.detach().cpu().numpy() for key, value in outputs.items()}


def _maximum_absolute(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "fraction_ge_0_5": float(np.mean(values >= 0.5)),
    }


def _records_with_outputs(
    metadata: list[dict[str, str]], outputs: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    row_count = len(metadata)
    if any(len(value) != row_count for value in outputs.values()):
        raise ValueError("metadata and reconstructed readouts are not aligned")
    records: list[dict[str, Any]] = []
    for position, metadata_row in enumerate(metadata):
        record: dict[str, Any] = {
            "feature_row": int(metadata_row["feature_row"]),
            "record_id": metadata_row["record_id"],
            "original_id": metadata_row.get("original_id", ""),
        }
        for key, value in outputs.items():
            record[key] = float(value[position])
        record["pIC50_equivalent"] = 6.0 - record["affinity_pred_value"]
        records.append(record)
    return records


def _historical_json_audit(
    records: list[dict[str, Any]], historical_root: Path | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if historical_root is None:
        return (
            {
                "historical_json_files": 0,
                "current_records": len(records),
                "exact_record_id_intersection": 0,
                "scope_note": (
                    "Historical JSON comparison was not requested. Pass "
                    "--historical-json-dir to audit an earlier run."
                ),
            },
            [],
        )
    by_record = {row["record_id"]: row for row in records}
    comparisons: list[dict[str, Any]] = []
    json_files = sorted(historical_root.glob("*/affinity.json"))
    for path in json_files:
        current = by_record.get(path.parent.name)
        if current is None:
            continue
        historical = json.loads(path.read_text())
        if not {
            "affinity_pred_value",
            "affinity_probability_binary",
            "affinity_logits_binary",
        }.issubset(historical):
            continue
        comparisons.append(
            {
                "record_id": path.parent.name,
                "current_affinity_pred_value": current["affinity_pred_value"],
                "historical_affinity_pred_value": historical["affinity_pred_value"],
                "absolute_affinity_difference": abs(
                    current["affinity_pred_value"] - historical["affinity_pred_value"]
                ),
                "current_binder_probability": current["binder_probability"],
                "historical_binder_probability": historical[
                    "affinity_probability_binary"
                ],
                "absolute_probability_difference": abs(
                    current["binder_probability"]
                    - historical["affinity_probability_binary"]
                ),
                "current_binary_logit": current["binary_logit"],
                "historical_binary_logit": historical["affinity_logits_binary"],
                "absolute_logit_difference": abs(
                    current["binary_logit"] - historical["affinity_logits_binary"]
                ),
                "historical_json_path": str(path),
            }
        )
    summary: dict[str, Any] = {
        "historical_json_files": len(json_files),
        "current_records": len(records),
        "exact_record_id_intersection": len(comparisons),
        "scope_note": (
            "Retained JSON is a partial earlier run and is not the current cached "
            "inference reference. Current-checkpoint parity is established by the "
            "continuous branch replay and exact architecture/ensemble reconstruction."
        ),
    }
    if comparisons:
        for column, label in (
            ("absolute_affinity_difference", "affinity"),
            ("absolute_probability_difference", "probability"),
            ("absolute_logit_difference", "logit"),
        ):
            values = np.asarray([row[column] for row in comparisons])
            summary[f"mean_absolute_{label}_difference"] = float(np.mean(values))
            summary[f"maximum_absolute_{label}_difference"] = float(np.max(values))
        current_affinity = np.asarray(
            [row["current_affinity_pred_value"] for row in comparisons]
        )
        historical_affinity = np.asarray(
            [row["historical_affinity_pred_value"] for row in comparisons]
        )
        current_probability = np.asarray(
            [row["current_binder_probability"] for row in comparisons]
        )
        historical_probability = np.asarray(
            [row["historical_binder_probability"] for row in comparisons]
        )
        summary["affinity_pearson"] = float(
            np.corrcoef(current_affinity, historical_affinity)[0, 1]
        )
        summary["probability_pearson"] = float(
            np.corrcoef(current_probability, historical_probability)[0, 1]
        )
    return summary, comparisons


def _export_representations(
    output_dir: Path,
    prefix: str,
    member1: torch.Tensor,
    member2: torch.Tensor,
) -> None:
    np.save(output_dir / f"{prefix}_member1_repr.npy", member1.cpu().numpy())
    np.save(output_dir / f"{prefix}_member2_repr.npy", member2.cpu().numpy())


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_file(str(args.checkpoint), device="cpu")

    labeled_features_path = (
        REPO_ROOT / "data" / "published" / "nesso_features.safetensors"
    )
    labeled_metadata_path = REPO_ROOT / "data" / "published" / "modeling_manifest.csv"
    labeled_tensors = load_file(str(labeled_features_path), device="cpu")
    labeled_metadata = read_csv(labeled_metadata_path)
    labeled_outputs = reconstruct_variant(
        labeled_tensors["member1_affinity_repr"].float(),
        labeled_tensors["member2_affinity_repr"].float(),
        checkpoint,
    )
    labeled_records = _records_with_outputs(labeled_metadata, labeled_outputs)
    write_csv(args.output_dir / "labeled_binary_readouts.csv", labeled_records)
    _export_representations(
        args.output_dir,
        "labeled",
        labeled_tensors["member1_affinity_repr"].float(),
        labeled_tensors["member2_affinity_repr"].float(),
    )

    continuous_parity = {
        "member1_max_abs": _maximum_absolute(
            labeled_outputs["member1_affinity_pred_value"],
            labeled_tensors["affinity_pred_value1"].cpu().numpy(),
        ),
        "member2_max_abs": _maximum_absolute(
            labeled_outputs["member2_affinity_pred_value"],
            labeled_tensors["affinity_pred_value2"].cpu().numpy(),
        ),
        "ensemble_max_abs": _maximum_absolute(
            labeled_outputs["affinity_pred_value"],
            labeled_tensors["affinity_pred_value"].cpu().numpy(),
        ),
    }

    challenge_features_path = (
        REPO_ROOT
        / "artifacts"
        / "experiments"
        / "challenge"
        / "run2"
        / "features.safetensors"
    )
    challenge_metadata_path = (
        REPO_ROOT / "artifacts" / "experiments" / "challenge" / "run2" / "metadata.csv"
    )
    challenge_tensors = load_file(str(challenge_features_path), device="cpu")
    challenge_metadata = read_csv(challenge_metadata_path)
    challenge_outputs = reconstruct_variant(
        challenge_tensors["member1_affinity_repr"].float(),
        challenge_tensors["member2_affinity_repr"].float(),
        checkpoint,
    )
    challenge_records = _records_with_outputs(challenge_metadata, challenge_outputs)
    write_csv(args.output_dir / "challenge_binary_readouts.csv", challenge_records)
    _export_representations(
        args.output_dir,
        "challenge",
        challenge_tensors["member1_affinity_repr"].float(),
        challenge_tensors["member2_affinity_repr"].float(),
    )
    challenge_continuous_parity = {
        "member1_max_abs": _maximum_absolute(
            challenge_outputs["member1_affinity_pred_value"],
            challenge_tensors["affinity_pred_value1"].cpu().numpy(),
        ),
        "member2_max_abs": _maximum_absolute(
            challenge_outputs["member2_affinity_pred_value"],
            challenge_tensors["affinity_pred_value2"].cpu().numpy(),
        ),
        "ensemble_max_abs": _maximum_absolute(
            challenge_outputs["affinity_pred_value"],
            challenge_tensors["affinity_pred_value"].cpu().numpy(),
        ),
    }

    pair_path = (
        REPO_ROOT
        / "artifacts"
        / "experiments"
        / "pair_pool_full"
        / "features.safetensors"
    )
    pair_tensors = load_file(str(pair_path), device="cpu")
    pair_records: list[dict[str, Any]] = []
    pair_parity: dict[str, Any] = {}
    for variant in ("all_pairs", "ligand_ligand_only", "receptor_ligand_only"):
        variant_outputs = reconstruct_variant(
            pair_tensors[f"member1_{variant}_repr"].float(),
            pair_tensors[f"member2_{variant}_repr"].float(),
            checkpoint,
        )
        pair_parity[variant] = {
            "member1_continuous_max_abs": _maximum_absolute(
                variant_outputs["member1_affinity_pred_value"],
                pair_tensors[f"member1_{variant}_pred_value"].cpu().numpy(),
            ),
            "member2_continuous_max_abs": _maximum_absolute(
                variant_outputs["member2_affinity_pred_value"],
                pair_tensors[f"member2_{variant}_pred_value"].cpu().numpy(),
            ),
        }
        for position, metadata_row in enumerate(labeled_metadata):
            row: dict[str, Any] = {
                "feature_row": int(metadata_row["feature_row"]),
                "record_id": metadata_row["record_id"],
                "original_id": metadata_row["original_id"],
                "pool_variant": variant,
            }
            for key, value in variant_outputs.items():
                row[key] = float(value[position])
            row["pIC50_equivalent"] = 6.0 - row["affinity_pred_value"]
            pair_records.append(row)
    write_csv(args.output_dir / "pair_pool_binary_readouts.csv", pair_records)

    historical_summary, historical_rows = _historical_json_audit(
        labeled_records, args.historical_json_dir
    )
    if historical_rows:
        write_csv(args.output_dir / "historical_json_comparison.csv", historical_rows)

    role_counts: dict[str, int] = {}
    for row in labeled_metadata:
        key = f"{row['modeling_role']}::{row['role']}"
        role_counts[key] = role_counts.get(key, 0) + 1
    provenance = {
        "status": (
            "pass"
            if max(continuous_parity.values()) <= 1e-6
            and max(challenge_continuous_parity.values()) <= 1e-6
            and all(max(values.values()) <= 1e-6 for values in pair_parity.values())
            else "fail"
        ),
        "semantics": {
            "affinity_pred_value": "log10(IC50 / micromolar)",
            "pIC50_equivalent": "6 - affinity_pred_value; not cellular pEC50",
            "binder_probability": (
                "mean of the two member sigmoid probabilities; ensemble logit is "
                "logit of that mean"
            ),
        },
        "architecture": {
            "cached_representation": "per-member affinity_repr, 384D",
            "continuous_branch": "384-384-1 MLP with ReLU",
            "binary_branch": (
                "384-384-1 score MLP with ReLU, followed by a learned 1-1 logit layer"
            ),
            "ensemble_rule": "mean member probabilities, then logit",
        },
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "labeled_features_sha256": sha256_file(labeled_features_path),
        "labeled_manifest_sha256": sha256_file(labeled_metadata_path),
        "challenge_features_sha256": sha256_file(challenge_features_path),
        "challenge_metadata_sha256": sha256_file(challenge_metadata_path),
        "coverage": {
            "labeled_total": len(labeled_records),
            "role_counts": role_counts,
            "challenge_total": len(challenge_records),
        },
        "continuous_replay": {
            "labeled": continuous_parity,
            "challenge": challenge_continuous_parity,
            "pair_pool": pair_parity,
        },
        "binary_distribution": {
            "labeled": _summarize(labeled_outputs["binder_probability"]),
            "challenge": _summarize(challenge_outputs["binder_probability"]),
        },
        "historical_json_audit": historical_summary,
        "cpu_threads": torch.get_num_threads(),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output_dir / "reconstruction_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    if provenance["status"] != "pass":
        raise SystemExit("binary reconstruction parity audit failed")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
