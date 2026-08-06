#!/usr/bin/env python3
"""Update the microsite's JSON-driven loss chart from a tidy CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

REQUIRED_COLUMNS = {"run_id", "epoch", "train_loss", "validation_loss"}


def build_payload(source: Path) -> dict[str, object]:
    grouped: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            epoch = int(row["epoch"])
            if epoch < 0:
                raise ValueError("epoch must be non-negative")
            grouped[row["run_id"]].append(
                {
                    "epoch": epoch,
                    "train_loss": float(row["train_loss"]),
                    "validation_loss": float(row["validation_loss"]),
                }
            )

    runs = [
        {"run_id": run_id, "epochs": sorted(epochs, key=lambda item: item["epoch"])}
        for run_id, epochs in sorted(grouped.items())
    ]
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "1.0.0",
        "updated_at": now,
        "status": "running" if runs else "pending",
        "message": "Training observations loaded."
        if runs
        else "No training observations are available yet.",
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/assets/training-history.json"),
    )
    args = parser.parse_args()
    payload = build_payload(args.history_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
