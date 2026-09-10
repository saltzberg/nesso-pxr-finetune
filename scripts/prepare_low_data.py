#!/usr/bin/env python3
"""Prepare immutable low-data artifacts without fitting any model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nesso_pxr.low_data_contract import load_prepared, prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/20260905_low_data_adaptation/protocol.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    protocol = (
        args.protocol if args.protocol.is_absolute() else args.repo_root / args.protocol
    )
    config = json.loads(protocol.read_text())
    output = args.output_dir or Path(config["run_dir"]) / "prepared"
    if not output.is_absolute():
        output = args.repo_root / output
    if args.verify_only:
        inputs, features, assignments, subsets = load_prepared(output)
        stored = json.loads((output / "preparation_manifest.json").read_text())
        if stored["config"] != config:
            raise ValueError("current protocol differs from prepared protocol")
        print(
            json.dumps(
                dict(
                    status="pass",
                    n_inputs=len(inputs),
                    feature_shapes={k: list(v.shape) for k, v in features.items()},
                    n_assignments=len(assignments),
                    n_subset_rows=len(subsets),
                )
            )
        )
    else:
        result = prepare_data(args.repo_root, output, config)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
