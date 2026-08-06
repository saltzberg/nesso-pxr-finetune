#!/usr/bin/env python3
"""Run the prespecified bounded E2/E3 cached-head development screen."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from nesso_pxr.train_cached import CachedHeadConfig, run_cached_cv

E2_LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4)
WEIGHT_DECAYS = (0.0, 1e-4, 1e-3)
E3_UNCERTAINTY_FLOORS = (0.10, 0.15, 0.20)


def _config_directory(root: Path, label: str, config: CachedHeadConfig) -> Path:
    floor = (
        "none" if config.uncertainty_floor is None else f"{config.uncertainty_floor:g}"
    )
    return root / (
        f"{label}_lr{config.learning_rate:g}_wd{config.weight_decay:g}"
        f"_floor{floor}_curation{int(config.use_curation_weight)}"
        f"_seed{config.seed}"
    )


def _run_or_load(
    *,
    label: str,
    config: CachedHeadConfig,
    args: argparse.Namespace,
) -> dict:
    output_dir = _config_directory(args.output_root, label, config)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary.get("config") != asdict(config):
            raise ValueError(f"existing config mismatch: {summary_path}")
        return summary
    return run_cached_cv(
        args.features,
        args.feature_metadata,
        args.modeling_manifest,
        args.checkpoint_weights,
        output_dir,
        config,
        device=args.device,
    )


def _row(label: str, summary: dict) -> dict:
    metrics = summary["aggregate"]["cached_head"]
    return {
        "label": label,
        "run_id": summary["run_id"],
        "mae": metrics["mae"],
        "spearman": metrics["spearman"],
        "rmse": metrics["rmse"],
        "delta_mae_vs_frozen": summary["delta_mae_vs_frozen"],
        "delta_mae_vs_affine": summary["delta_mae_vs_affine"],
        "delta_spearman_vs_frozen": summary["delta_spearman_vs_frozen"],
        "delta_spearman_vs_affine": summary["delta_spearman_vs_affine"],
        **summary["config"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-metadata", type=Path, required=True)
    parser.add_argument("--modeling-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-weights", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    e2_results: list[tuple[CachedHeadConfig, dict]] = []
    for learning_rate in E2_LEARNING_RATES:
        for weight_decay in WEIGHT_DECAYS:
            config = CachedHeadConfig(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                uncertainty_floor=None,
                use_curation_weight=False,
                seed=42,
            )
            summary = _run_or_load(label="e2", config=config, args=args)
            e2_results.append((config, summary))
            rows.append(_row("e2", summary))

    best_e2_config, _ = min(
        e2_results,
        key=lambda item: (
            item[1]["aggregate"]["cached_head"]["mae"],
            -item[1]["aggregate"]["cached_head"]["spearman"],
        ),
    )
    for floor in E3_UNCERTAINTY_FLOORS:
        for use_curation_weight in (False, True):
            config = CachedHeadConfig(
                learning_rate=best_e2_config.learning_rate,
                weight_decay=best_e2_config.weight_decay,
                uncertainty_floor=floor,
                use_curation_weight=use_curation_weight,
                seed=42,
            )
            summary = _run_or_load(label="e3", config=config, args=args)
            rows.append(_row("e3", summary))

    results = pd.DataFrame(rows).sort_values(
        ["mae", "spearman"],
        ascending=[True, False],
        kind="stable",
    )
    results.to_csv(args.output_root / "screen_summary.csv", index=False)
    best = results.iloc[0].to_dict()
    payload = {
        "status": "pass",
        "selection_rule": "minimum pooled development OOF MAE; Spearman tie-break",
        "tested_configurations": int(len(results)),
        "best": best,
    }
    (args.output_root / "screen_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
