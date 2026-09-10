#!/usr/bin/env python3
"""Run the predeclared CPU-only Nesso/PXR follow-up experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = REPO_ROOT / "data" / "published"
TRANSFER = REPO_ROOT / "reports" / "transfer_ablation"
OUTPUT_ROOT = TRANSFER / "cpu_followups"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("capacity", "standardization", "loss-lr"),
        required=True,
    )
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--analysis-spec",
        type=Path,
        default=OUTPUT_ROOT / "analysis_spec.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PUBLISHED / "modeling_manifest.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PUBLISHED / "nesso_features.safetensors",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "models" / "nesso-1" / "v1.0.0" / "model.safetensors",
    )
    parser.add_argument(
        "--historical-dev",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "model_comparison"
        / "nested_nesso_predictions.csv",
    )
    parser.add_argument(
        "--historical-lock",
        type=Path,
        default=PUBLISHED / "nesso_holdout_predictions.csv",
    )
    parser.add_argument(
        "--historical-ridge",
        type=Path,
        default=TRANSFER / "ridge_probe_predictions.csv",
    )
    parser.add_argument(
        "--historical-lgbm",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "model_comparison"
        / "nested_2d_predictions.csv",
    )
    parser.add_argument(
        "--reduced-2d-features",
        type=Path,
        default=PUBLISHED / "reduced_2d_features.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from nesso_pxr.cpu_followups import (
        run_capacity_experiment,
        run_loss_lr_experiment,
        run_standardization_experiment,
    )

    if args.stage == "capacity":
        run_capacity_experiment(
            manifest_path=args.manifest,
            features_path=args.features,
            historical_dev_path=args.historical_dev,
            historical_lock_path=args.historical_lock,
            analysis_spec=args.analysis_spec,
            output_dir=args.output_root / "experiment_a_capacity",
            max_workers=args.max_workers,
        )
    elif args.stage == "standardization":
        run_standardization_experiment(
            manifest_path=args.manifest,
            features_path=args.features,
            reduced_2d_features_path=args.reduced_2d_features,
            historical_ridge_path=args.historical_ridge,
            historical_lgbm_path=args.historical_lgbm,
            analysis_spec=args.analysis_spec,
            output_dir=args.output_root / "experiment_b_standardization",
            max_workers=args.max_workers,
        )
    else:
        run_loss_lr_experiment(
            manifest_path=args.manifest,
            features_path=args.features,
            checkpoint_path=args.checkpoint,
            historical_dev_path=args.historical_dev,
            historical_lock_path=args.historical_lock,
            analysis_spec=args.analysis_spec,
            output_dir=args.output_root / "experiment_c_loss_lr",
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
