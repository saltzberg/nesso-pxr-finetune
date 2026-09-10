#!/usr/bin/env python3
"""Analyze the full Nesso direct pair-pooling ablation extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nesso_pxr.comparison import (
    clustered_bootstrap,
    paired_difference_rows,
    regression_metrics,
)
from nesso_pxr.protocol import sha256_file
from nesso_pxr.transfer_ablation import (
    fit_full_ridge_after_group_selection,
    nested_ridge_probe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION = REPO_ROOT / "artifacts" / "experiments" / "pair_pool_full"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "published" / "modeling_manifest.csv"
DEFAULT_HISTORICAL = REPO_ROOT / "data" / "published" / "nesso_features.safetensors"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "transfer_ablation" / "pair_pool_full"
DEFAULT_SELECTION = (
    REPO_ROOT / "artifacts" / "experiments" / "e0" / "full" / "selection.csv"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "nesso-1" / "v1.0.0" / "model.safetensors"
DEFAULT_HPARAMS = REPO_ROOT / "models" / "nesso-1" / "v1.0.0" / "hparams.json"
DEFAULT_CCD = REPO_ROOT / "models" / "nesso-1" / "ccd.pkl"
NESSO_HUB_REVISION = "1896c84c7186c506c7efd79051480809d51098bf"
NESSO_IMAGE_DIGEST = (
    "sha256:e3519ab0faa11d098f14f57b628530c21df748ab7c632f5b071754548e058a24"
)

POOL_VARIANTS = (
    "all_pairs",
    "ligand_ligand_only",
    "receptor_ligand_only",
)
REPRESENTATION_VARIANTS = (
    "ridge_member1_384d",
    "ridge_member2_384d",
    "ridge_mean_384d",
    "ridge_concat_768d",
)
RIDGE_ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
PRIMARY_REPRESENTATION = "ridge_mean_384d"
REPLAY_ABSOLUTE_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--historical-features", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hparams", type=Path, default=DEFAULT_HPARAMS)
    parser.add_argument("--ccd", type=Path, default=DEFAULT_CCD)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _maximum_absolute_difference(left: Any, right: Any) -> float:
    import torch

    if left.shape != right.shape:
        return float("inf")
    return float(torch.max(torch.abs(left.float() - right.float())).item())


def _load_and_validate(
    extraction_dir: Path,
    manifest_path: Path,
    historical_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    feature_path = extraction_dir / "features.safetensors"
    metadata_path = extraction_dir / "metadata.csv"
    summary_path = extraction_dir / "run_summary.json"
    for path in (
        feature_path,
        metadata_path,
        summary_path,
        manifest_path,
        historical_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = pd.read_csv(metadata_path).sort_values("feature_row", kind="stable")
    manifest = pd.read_csv(manifest_path).sort_values("feature_row", kind="stable")
    metadata = metadata.reset_index(drop=True)
    manifest = manifest.reset_index(drop=True)
    if len(metadata) != 4134 or len(manifest) != 4134:
        raise ValueError("the full ablation requires exactly 4,134 aligned rows")
    expected_rows = list(range(4134))
    if metadata["feature_row"].tolist() != expected_rows:
        raise ValueError("extraction feature rows are not contiguous and ordered")
    if manifest["feature_row"].tolist() != expected_rows:
        raise ValueError("manifest feature rows are not contiguous and ordered")
    if not metadata["record_id"].equals(manifest["record_id"]):
        raise ValueError("extraction metadata and modeling manifest do not align")

    tensors = load_file(str(feature_path), device="cpu")
    historical = load_file(str(historical_path), device="cpu")
    required = {
        "affinity_pred_value",
        "affinity_pred_value1",
        "affinity_pred_value2",
    }
    for member in ("member1", "member2"):
        for pool in POOL_VARIANTS:
            required.add(f"{member}_{pool}_repr")
            required.add(f"{member}_{pool}_pred_value")
    missing = sorted(required.difference(tensors))
    if missing:
        raise KeyError(f"missing extracted tensors: {missing}")
    for key in required:
        if len(tensors[key]) != len(manifest):
            raise ValueError(f"tensor row count mismatch: {key}")

    with summary_path.open() as handle:
        extraction_summary = json.load(handle)
    if extraction_summary.get("status") != "pass":
        raise ValueError("extraction summary is not passing")
    if extraction_summary.get("selected") != 4134:
        raise ValueError("extraction summary does not cover all 4,134 rows")
    return manifest, tensors, historical, extraction_summary


def _parity_audit(
    tensors: dict[str, Any],
    historical: dict[str, Any],
    extraction_summary: dict[str, Any],
) -> dict[str, Any]:
    comparisons = {
        "member1_all_pairs_repr_vs_historical": (
            tensors["member1_all_pairs_repr"],
            historical["member1_affinity_repr"],
        ),
        "member2_all_pairs_repr_vs_historical": (
            tensors["member2_all_pairs_repr"],
            historical["member2_affinity_repr"],
        ),
        "member1_direct_score_vs_historical": (
            tensors["affinity_pred_value1"],
            historical["affinity_pred_value1"],
        ),
        "member2_direct_score_vs_historical": (
            tensors["affinity_pred_value2"],
            historical["affinity_pred_value2"],
        ),
        "ensemble_direct_score_vs_historical": (
            tensors["affinity_pred_value"],
            historical["affinity_pred_value"],
        ),
        "member1_replayed_score_vs_historical": (
            tensors["member1_all_pairs_pred_value"],
            historical["reconstructed_affinity_pred_value1"],
        ),
        "member2_replayed_score_vs_historical": (
            tensors["member2_all_pairs_pred_value"],
            historical["reconstructed_affinity_pred_value2"],
        ),
    }
    maximum_differences = {
        name: _maximum_absolute_difference(*values)
        for name, values in comparisons.items()
    }
    representation_exact = all(
        maximum_differences[name] == 0.0
        for name in (
            "member1_all_pairs_repr_vs_historical",
            "member2_all_pairs_repr_vs_historical",
        )
    )
    direct_scores_exact = all(
        maximum_differences[name] == 0.0
        for name in (
            "member1_direct_score_vs_historical",
            "member2_direct_score_vs_historical",
            "ensemble_direct_score_vs_historical",
        )
    )
    historical_replays_within_tolerance = all(
        maximum_differences[name] <= REPLAY_ABSOLUTE_TOLERANCE
        for name in (
            "member1_replayed_score_vs_historical",
            "member2_replayed_score_vs_historical",
        )
    )
    internal_replay_difference = float(
        extraction_summary["baseline_head_parity_max_abs"]
    )
    passed = (
        representation_exact
        and direct_scores_exact
        and historical_replays_within_tolerance
        and internal_replay_difference == 0.0
    )
    return {
        "status": "pass" if passed else "fail",
        "representation_exact": representation_exact,
        "direct_historical_scores_exact": direct_scores_exact,
        "historical_replays_within_tolerance": historical_replays_within_tolerance,
        "historical_replay_absolute_tolerance": REPLAY_ABSOLUTE_TOLERANCE,
        "internal_head_replay_exact": internal_replay_difference == 0.0,
        "internal_head_replay_max_abs": internal_replay_difference,
        "maximum_absolute_differences": maximum_differences,
        "interpretation_gate": (
            "Ablation metrics are interpretable only when all-pairs reproduces "
            "the historical extraction and released head exactly."
        ),
    }


def _representations(tensors: dict[str, Any], pool: str) -> dict[str, np.ndarray]:
    member1 = tensors[f"member1_{pool}_repr"].float().numpy()
    member2 = tensors[f"member2_{pool}_repr"].float().numpy()
    return {
        "ridge_member1_384d": member1,
        "ridge_member2_384d": member2,
        "ridge_mean_384d": (member1 + member2) / 2.0,
        "ridge_concat_768d": np.concatenate([member1, member2], axis=1),
    }


def _metric_row(
    dataset: str,
    pool: str,
    representation: str,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "pool_variant": pool,
        "representation": representation,
        **regression_metrics(observed, predicted),
    }


def _run_ridge_probes(
    manifest: pd.DataFrame,
    tensors: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    development = manifest.loc[manifest["modeling_role"].eq("development")].copy()
    lockbox = manifest.loc[manifest["modeling_role"].eq("lockbox")].copy()
    development = development.sort_values("feature_row", kind="stable").reset_index(
        drop=True
    )
    lockbox = lockbox.sort_values("feature_row", kind="stable").reset_index(drop=True)
    if len(development) != 3344 or len(lockbox) != 790:
        raise ValueError("unexpected development/lockbox row counts")

    development_positions = development["feature_row"].to_numpy(dtype=int)
    lockbox_positions = lockbox["feature_row"].to_numpy(dtype=int)
    y_development = development["pEC50"].to_numpy(dtype=float)
    y_lockbox = lockbox["pEC50"].to_numpy(dtype=float)
    folds = development["fold"].to_numpy(dtype=int)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    outer_selections: list[pd.DataFrame] = []
    full_screens: list[pd.DataFrame] = []

    for pool in POOL_VARIANTS:
        all_representations = _representations(tensors, pool)
        for representation in REPRESENTATION_VARIANTS:
            matrix = all_representations[representation]
            x_development = matrix[development_positions]
            x_lockbox = matrix[lockbox_positions]
            nested = nested_ridge_probe(
                x_development,
                y_development,
                folds,
                RIDGE_ALPHAS,
            )
            development_predictions = development[
                [
                    "feature_row",
                    "record_id",
                    "original_id",
                    "pEC50",
                    "fold",
                    "cluster_component",
                    "audit_scaffold_group",
                ]
            ].copy()
            development_predictions["predicted_pEC50"] = nested.predictions[
                "predicted"
            ].to_numpy()
            development_predictions["dataset"] = "nested_development"
            development_predictions["pool_variant"] = pool
            development_predictions["representation"] = representation
            prediction_frames.append(development_predictions)
            metric_rows.append(
                _metric_row(
                    "nested_development",
                    pool,
                    representation,
                    y_development,
                    development_predictions["predicted_pEC50"].to_numpy(),
                )
            )

            selections = nested.selections.copy()
            selections.insert(0, "representation", representation)
            selections.insert(0, "pool_variant", pool)
            outer_selections.append(selections)

            lock_prediction, selected_alpha, screen = (
                fit_full_ridge_after_group_selection(
                    x_development,
                    y_development,
                    folds,
                    x_lockbox,
                    RIDGE_ALPHAS,
                )
            )
            screen = screen.copy()
            screen.insert(0, "selected", screen["alpha"].eq(selected_alpha))
            screen.insert(0, "representation", representation)
            screen.insert(0, "pool_variant", pool)
            full_screens.append(screen)
            lockbox_predictions = lockbox[
                [
                    "feature_row",
                    "record_id",
                    "original_id",
                    "pEC50",
                    "fold",
                    "cluster_component",
                    "audit_scaffold_group",
                ]
            ].copy()
            lockbox_predictions["predicted_pEC50"] = lock_prediction
            lockbox_predictions["dataset"] = "lockbox_all_790"
            lockbox_predictions["pool_variant"] = pool
            lockbox_predictions["representation"] = representation
            prediction_frames.append(lockbox_predictions)
            metric_rows.append(
                _metric_row(
                    "lockbox_all_790",
                    pool,
                    representation,
                    y_lockbox,
                    lock_prediction,
                )
            )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(outer_selections, ignore_index=True),
        pd.concat(full_screens, ignore_index=True),
    )


def _bootstrap_differences(
    predictions: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_frames: list[pd.DataFrame] = []
    difference_frames: list[pd.DataFrame] = []
    for representation in REPRESENTATION_VARIANTS:
        for dataset in ("nested_development", "lockbox_all_790"):
            subset = predictions.loc[
                predictions["representation"].eq(representation)
                & predictions["dataset"].eq(dataset)
            ].copy()
            identity_columns = ["feature_row", "pEC50"]
            group_column = (
                "cluster_component"
                if dataset == "nested_development"
                else "audit_scaffold_group"
            )
            identity_columns.append(group_column)
            wide = subset.pivot(
                index=identity_columns,
                columns="pool_variant",
                values="predicted_pEC50",
            ).reset_index()
            if len(wide) * len(POOL_VARIANTS) != len(subset):
                raise ValueError("pool predictions are not one-to-one and complete")
            models = {pool: pool for pool in POOL_VARIANTS}
            intervals, draws = clustered_bootstrap(
                wide,
                models,
                group_column=group_column,
                observed_column="pEC50",
                replicates=replicates,
                seed=seed,
            )
            intervals.insert(0, "representation", representation)
            intervals.insert(0, "dataset", dataset)
            interval_frames.append(intervals)
            point_metrics = {
                pool: regression_metrics(wide["pEC50"], wide[pool])
                for pool in POOL_VARIANTS
            }
            differences = paired_difference_rows(
                draws,
                point_metrics,
                [
                    ("ligand_ligand_only", "all_pairs"),
                    ("receptor_ligand_only", "all_pairs"),
                ],
                metrics=("mae", "spearman"),
            )
            differences.insert(0, "representation", representation)
            differences.insert(0, "dataset", dataset)
            differences["bootstrap_replicates"] = replicates
            differences["bootstrap_clusters"] = wide[group_column].nunique()
            difference_frames.append(differences)
    return (
        pd.concat(interval_frames, ignore_index=True),
        pd.concat(difference_frames, ignore_index=True),
    )


def _effect_statement(row: pd.Series) -> str:
    if row["metric"] == "mae":
        if row["ci_lower"] > 0:
            return "worse without direct receptor-ligand final-pool cells"
        if row["ci_upper"] < 0:
            return "better without direct receptor-ligand final-pool cells"
    elif row["metric"] == "spearman":
        if row["ci_upper"] < 0:
            return "worse without direct receptor-ligand final-pool cells"
        if row["ci_lower"] > 0:
            return "better without direct receptor-ligand final-pool cells"
    return "confidence interval includes no difference"


def _write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    differences: pd.DataFrame,
    parity: dict[str, Any],
    extraction_summary: dict[str, Any],
) -> None:
    primary_metrics = metrics.loc[
        metrics["representation"].eq(PRIMARY_REPRESENTATION),
        ["dataset", "pool_variant", "n", "mae", "spearman"],
    ]
    primary_differences = differences.loc[
        differences["representation"].eq(PRIMARY_REPRESENTATION)
        & differences["comparison"].eq("ligand_ligand_only_minus_all_pairs")
    ].copy()
    lines = [
        "# Nesso-1 direct final pair-pooling ablation",
        "",
        "## Positive control",
        "",
        f"All-pairs historical parity: **{parity['status']}**. ",
        f"Extraction covered {extraction_summary['succeeded']:,} / "
        f"{extraction_summary['selected']:,} compounds with "
        f"{extraction_summary['failed']} failures.",
        "",
        "## Prespecified primary probe: ridge on mean 384D representation",
        "",
        "| Dataset | Final pool | N | MAE | Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary_metrics.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.pool_variant} | {int(row.n)} | "
            f"{row.mae:.5f} | {row.spearman:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Paired ligand-ligand-only minus all-pairs differences",
            "",
            "| Dataset | Metric | Difference | 95% family-bootstrap CI | Reading |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for _, row in primary_differences.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['estimate']:.5f} | "
            f"[{row['ci_lower']:.5f}, {row['ci_upper']:.5f}] | "
            f"{_effect_statement(row)} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "This intervention tests whether direct receptor-ligand cells in the "
            "final affinity pooling operation add predictive information. "
            "Ligand-ligand cells were already protein-conditioned upstream. A null "
            "result therefore does not establish that Nesso ignores protein; it only "
            "shows that this direct final-pool pathway is unnecessary for fixed-PXR "
            "prediction under this probe. The receptor-ligand-only result is "
            "diagnostic rather than the principal control.",
            "",
            "The 790-compound lockbox is a previously opened retrospective test set. "
            "No lockbox labels were used for ridge selection or fitting.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, tensors, historical, extraction_summary = _load_and_validate(
        args.extraction_dir,
        args.manifest,
        args.historical_features,
    )
    parity = _parity_audit(tensors, historical, extraction_summary)
    _write_json(args.output_dir / "parity_audit.json", parity)
    if parity["status"] != "pass":
        raise RuntimeError(
            "all-pairs positive-control parity failed; refusing to interpret ablation"
        )

    metrics, predictions, outer_selections, full_screens = _run_ridge_probes(
        manifest, tensors
    )
    bootstrap_intervals, paired_differences = _bootstrap_differences(
        predictions,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    metrics.to_csv(args.output_dir / "ridge_probe_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "ridge_probe_predictions.csv", index=False)
    outer_selections.to_csv(
        args.output_dir / "ridge_outer_fold_selections.csv", index=False
    )
    full_screens.to_csv(
        args.output_dir / "ridge_full_development_screens.csv", index=False
    )
    bootstrap_intervals.to_csv(
        args.output_dir / "family_bootstrap_intervals.csv", index=False
    )
    paired_differences.to_csv(
        args.output_dir / "paired_family_bootstrap_differences.csv", index=False
    )
    _write_json(
        args.output_dir / "analysis_scope.json",
        {
            "status": "complete",
            "scientific_question": (
                "Do direct receptor-ligand pair cells in Nesso-1's final affinity "
                "pool add predictive signal beyond ligand-ligand pair cells?"
            ),
            "primary_comparison": "ligand_ligand_only_minus_all_pairs",
            "primary_representation": PRIMARY_REPRESENTATION,
            "representation_sensitivity_analyses": list(REPRESENTATION_VARIANTS),
            "ridge_alphas": list(RIDGE_ALPHAS),
            "development_evaluation": "five-fold nested chemical-family CV",
            "development_bootstrap_group": "cluster_component",
            "lockbox_evaluation": (
                "fit on development only; all 790 labels evaluation only"
            ),
            "lockbox_bootstrap_group": "audit_scaffold_group",
            "lockbox_status": "previously_opened_retrospective_evaluation",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "feature_sha256": sha256_file(args.extraction_dir / "features.safetensors"),
            "metadata_sha256": sha256_file(args.extraction_dir / "metadata.csv"),
            "manifest_sha256": sha256_file(args.manifest),
            "historical_features_sha256": sha256_file(args.historical_features),
            "selection_sha256": sha256_file(args.selection),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "hparams_sha256": sha256_file(args.hparams),
            "ccd_sha256": sha256_file(args.ccd),
            "nesso_hub_revision": NESSO_HUB_REVISION,
            "nesso_image_digest": NESSO_IMAGE_DIGEST,
            "evidence_boundary": (
                "Ligand-ligand z cells remain protein-conditioned upstream; this "
                "surgical test isolates only direct final receptor-ligand pooling."
            ),
        },
    )
    _write_report(
        args.output_dir,
        metrics,
        paired_differences,
        parity,
        extraction_summary,
    )
    print(metrics.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
