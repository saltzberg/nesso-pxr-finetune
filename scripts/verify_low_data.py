#!/usr/bin/env python3
"""Read-only reconciliation of a low-data run's declared and completed tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_low_data import complete_task, digest, planned_tasks


def verify_run(run_dir: Path, *, require_complete: bool = False) -> dict:
    config = json.loads((run_dir / "protocol.json").read_text())
    manifest = json.loads((run_dir / "task_manifest.json").read_text())
    assert manifest == planned_tasks(config), "task manifest differs from protocol"
    planned = {t["task_id"]: t for t in manifest}
    assert len(planned) == len(manifest), "duplicate task IDs"
    states = {"complete": [], "failed": [], "pending": [], "invalid": []}
    observed = {p.name for p in (run_dir / "tasks").glob("*") if p.is_dir()}
    assert observed <= planned.keys(), f"unexpected tasks: {observed - planned.keys()}"
    for tid, task in planned.items():
        directory = run_dir / "tasks" / tid
        if not (directory / "complete.json").exists():
            state = "failed" if (directory / "failure.json").exists() else "pending"
            states[state].append(tid)
            continue
        try:
            assert complete_task(directory), "output checksum mismatch"
            metrics = json.loads((directory / "metrics.json").read_text())
            predictions = pd.read_csv(directory / "predictions.csv")
            access = json.loads((directory / "label_access.json").read_text())
            assert metrics["status"] == "complete"
            for key in ("split", "outer_fold", "draw", "n_train", "method"):
                assert metrics[key] == task[key]
                assert predictions[key].eq(task[key]).all()
            assert predictions["record_id"].is_unique
            assert len(predictions) == metrics["n_test"]
            y = predictions["y_true"].to_numpy(dtype=float)
            p = predictions["y_pred"].to_numpy(dtype=float)
            se = predictions["assay_se"].to_numpy(dtype=float)
            assert np.isfinite(y).all() and np.isfinite(p).all()
            assert np.isfinite(se).all() and (se > 0).all()
            weights = 1.0 / np.maximum(se, config["se_floor"])
            expected = float(np.sum(weights * np.abs(y - p)) / np.sum(weights))
            assert math.isclose(expected, metrics["weighted_mae"], abs_tol=1e-10)
            np.testing.assert_allclose(weights, predictions["weight"], atol=1e-10)
            fit = set(access["fit_record_ids"])
            cal = set(access["calibration_record_ids"])
            test = set(access["test_record_ids"])
            assert not (fit & cal or fit & test or cal & test)
            assert test == set(predictions["record_id"])
            assert len(fit) == metrics["n_fit"] and len(cal) == metrics["n_calibration"]
            assert len(fit | cal) == access["acquired_label_count"]
            assert not access["test_labels_accessible_to_fit"]
            assert not access["calibration_labels_accessible_to_fit"]
            if task["n_train"] != -1:
                assert len(fit | cal) == task["n_train"]
            for cov in config["interval_coverages"]:
                key = str(round(cov * 100))
                low = predictions[f"lower_{key}"]
                high = predictions[f"upper_{key}"]
                assert (low <= high).all()
                measured = float(((y >= low) & (y <= high)).mean())
                assert math.isclose(measured, metrics[f"coverage_{key}"], abs_tol=1e-10)
            states["complete"].append(tid)
        except Exception as error:
            states["invalid"].append({"task_id": tid, "error": str(error)})
    summary = {
        "status": "pass" if not states["invalid"] else "fail",
        "total_tasks": len(manifest),
        **{f"{key}_count": len(value) for key, value in states.items()},
        "invalid": states["invalid"],
        "science_complete": len(states["complete"]) == len(manifest),
        "protocol_sha256": digest(run_dir / "protocol.json"),
    }
    assert sum(summary[f"{s}_count"] for s in states) == len(manifest)
    if require_complete and not summary["science_complete"]:
        summary["status"] = "fail"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = verify_run(args.run_dir, require_complete=args.require_complete)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
