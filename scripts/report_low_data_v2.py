#!/usr/bin/env python3
"""Reporting-only interval replay; leave source-bound training runs unchanged."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np

from nesso_pxr import low_data_report as original


def replay_interval(frame, level, meta, audit):
    """Replay the runner's residual comparison, with exact endpoint checks.

    Doubling/halving a finite binary float is exact in this assay's range.
    Using width/2 avoids cancellation when recovering radius from endpoints.
    No coverage tolerance or widening of intervals is introduced.
    """
    values = original_interval(frame, level)
    width = meta[f"width_{level}"]
    if width is None and meta.get(f"interval_{level}_finite") is False:
        radius = math.inf
    elif width is not None and math.isfinite(width) and width >= 0:
        radius = width / 2
    else:
        raise ValueError("invalid saved conformal width")
    p = frame.y_pred.to_numpy(float)
    y = frame.y_true.to_numpy(float)
    lo = frame[f"lower_{level}"].to_numpy(float)
    hi = frame[f"upper_{level}"].to_numpy(float)
    if not (np.array_equal(p - radius, lo) and np.array_equal(p + radius, hi)):
        raise ValueError("saved endpoints do not exactly replay prediction +/- radius")
    residual_covered = np.abs(y - p) <= radius
    bound_covered = (y >= lo) & (y <= hi)
    disagreement = residual_covered != bound_covered
    if disagreement.any():
        audit.append(
            {
                "task_id": meta["task_id"],
                "level": level,
                "record_ids": frame.loc[disagreement, "record_id"].tolist(),
                "residual_coverage": float(residual_covered.mean()),
                "serialized_bound_coverage": float(bound_covered.mean()),
            }
        )
    values[f"coverage_{level}"] = float(residual_covered.mean())
    values[f"width_{level}"] = 2 * radius
    return values


original_interval = original._interval


@contextlib.contextmanager
def replay_verifier(audit):
    """Install scoped report-only hooks in this separate single-threaded process."""
    verify = original._verify
    interval = original._interval
    active = {}

    def scoped_verify(task_dir, meta, *args):
        active["meta"] = meta
        try:
            return verify(task_dir, meta, *args)
        finally:
            active.clear()

    def scoped_interval(frame, level):
        return replay_interval(frame, level, active["meta"], audit)

    original._verify = scoped_verify
    original._interval = scoped_interval
    try:
        yield
    finally:
        original._verify = verify
        original._interval = interval


def build_report(run_dir, output):
    audit = []
    with replay_verifier(audit):
        summary = original.build_report(run_dir, output)
    evidence = {
        "policy": "Exact residual <= radius replay of the frozen runner; no tolerance",
        "original_report_source_sha256": original._hash(Path(original.__file__)),
        "replay_report_source_sha256": original._hash(Path(__file__)),
        "training_artifacts_modified": False,
        "boundary_disagreements": audit,
    }
    original._write_json(output / "interval_replay.json", evidence)
    page = output / "index.html"
    text = page.read_text()
    note = (
        "<p>Reporting revision: interval coverage exactly replays the frozen "
        "runner’s absolute-residual comparison. Rounded endpoint comparisons "
        "can differ at a floating-point boundary; no intervals were widened "
        "and no training artifacts were changed. "
        '<a href="interval_replay.json">Boundary replay evidence</a>. '
        '<a href="../low-data/">Original report</a>.</p>'
    )
    text = text.replace('<details id="data">', note + '<details id="data">')
    page.write_text(text)
    manifest_path = output / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ["index.html", "interval_replay.json"]:
        manifest["output_sha256"][name] = original._hash(output / name)
    original._write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "generated_at",
                    "completed",
                    "failed",
                    "pending",
                    "total_tasks",
                    "provenance_errors",
                ]
            }
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--watch-seconds", type=float, default=0)
    parser.add_argument("--maximum-seconds", type=float, default=21600)
    args = parser.parse_args()
    start = time.monotonic()
    while True:
        result = build_report(args.run_dir.resolve(), args.report_dir.resolve())
        if not args.watch_seconds or result["runner_status"] != "running":
            break
        if time.monotonic() - start >= args.maximum_seconds:
            break
        time.sleep(args.watch_seconds)
    if result["failed"] or result["provenance_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
