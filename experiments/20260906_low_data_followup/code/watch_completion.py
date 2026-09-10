"""Read-only queue supervision; separate final artifact reconciliation, no fits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/experiments/low_data_followup_20260906"
LANES = {
    "readouts": "readouts",
    "baselines": "baselines",
    "strict": "strict_split/panel",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def inspect_lane(run, *, verify_outputs=False):
    tasks = json.loads((run / "task_manifest.json").read_text())
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate expected task identity")
    binding = digest(run / "run_bindings.json")
    counts = dict(expected=len(tasks), complete=0, failed=0, pending=0, invalid=0)
    errors, rows = [], []
    required = {
        "metrics.json",
        "predictions.csv",
        "label_access.json",
        "seed_predictions.npz",
    }
    for task in tasks:
        directory = run / "tasks" / task["task_id"]
        marker = directory / "complete.json"
        if not marker.exists():
            counts[
                "failed" if (directory / "failure.json").exists() else "pending"
            ] += 1
            continue
        try:
            data = json.loads(marker.read_text())
            if data.get("task") != task:
                raise ValueError("foreign task identity")
            if data.get("binding_sha256", data.get("run_bindings_sha256")) != binding:
                raise ValueError("foreign run binding")
            outputs = data["output_sha256"]
            if not required.issubset(outputs) or not {
                "model_state.pkl",
                "model_state.pt",
            }.intersection(outputs):
                raise ValueError("incomplete output declaration")
            if verify_outputs:
                for name, expected in outputs.items():
                    path = (directory / name).resolve()
                    if (
                        not path.is_relative_to(directory.resolve())
                        or digest(path) != expected
                    ):
                        raise ValueError(f"output mismatch: {name}")
            metrics = json.loads((directory / "metrics.json").read_text())
            for key, value in task.items():
                if metrics.get(key) != value:
                    raise ValueError(f"metrics identity mismatch: {key}")
            counts["complete"] += 1
            rows.append(
                {
                    key: metrics.get(key)
                    for key in (
                        "task_id",
                        "split",
                        "outer_fold",
                        "draw",
                        "n_train",
                        "method",
                        "n_fit",
                        "n_calibration",
                        "n_test",
                        "weighted_mae",
                        "mae",
                        "spearman",
                        "fit_seconds",
                    )
                }
            )
        except (ValueError, KeyError, OSError) as error:
            counts["invalid"] += 1
            errors.append({"task_id": task["task_id"], "error": str(error)})
    foreign = sorted(
        p.parent.name
        for p in (run / "tasks").glob("*/complete.json")
        if p.parent.name not in set(ids)
    )
    complete = counts["complete"] == counts["expected"] and not foreign
    return {
        **counts,
        "foreign_task_ids": foreign,
        "errors": errors,
        "output_hashes_checked": verify_outputs,
        "complete": counts["complete"],
        "matrix_complete": complete,
        "fully_verified": complete and verify_outputs,
    }, rows


def service_state(name):
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            f"nesso-pxr-followup-{name}.service",
            "--property=ActiveState,SubState,MainPID,Result,ExecMainStatus,RuntimeMaxUSec",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def aggregates(rows):
    groups = {}
    for row in rows:
        key = (row["split"], row["n_train"], row["method"])
        groups.setdefault(key, []).append(row)
    result = []
    for (split, budget, method), group in sorted(groups.items()):
        record = {
            "split": split,
            "n_train": budget,
            "method": method,
            "completed_tasks": len(group),
        }
        for metric in ["weighted_mae", "mae", "spearman", "fit_seconds"]:
            values = [row[metric] for row in group if row[metric] is not None]
            record[f"mean_task_{metric}"] = statistics.fmean(values) if values else None
        result.append(record)
    return result


def write_csv(path, rows):
    if not rows:
        return
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, stream.getvalue())


def service_is_running(state):
    # RemainAfterExit units retain ActiveState=active after their worker exits.
    if state.get("SubState") == "exited" and state.get("MainPID") == "0":
        return False
    return state.get("ActiveState") in {
        "active",
        "activating",
        "reloading",
        "deactivating",
    }


def refresh():
    out = RUN / "completion_monitor"
    report = {"updated_at": datetime.now(UTC).isoformat(), "lanes": {}}
    terminal = True
    all_verified = True
    for name, relative in LANES.items():
        state = service_state(name)
        active = service_is_running(state)
        terminal = terminal and not active
        previous_path = out / f"{name}_final.json"
        lane, rows = inspect_lane(RUN / relative, verify_outputs=not active)
        write_csv(out / f"{name}_tasks.csv", rows)
        write_csv(out / f"{name}_descriptive_task_means.csv", aggregates(rows))
        if not active:
            atomic(previous_path, json.dumps(lane, indent=2) + "\n")
        lane = {**lane, "service": state}
        all_verified = all_verified and lane["fully_verified"]
        report["lanes"][name] = lane
    report["all_services_terminal"] = terminal
    report["all_output_sets_verified"] = all_verified
    report["scope"] = (
        "Task identity, binding and final output hashes; numerical replay remains "
        "in each lane's own verification. Partial tables are unpaired and not "
        "comparative evidence."
    )
    atomic(out / "status.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return terminal, all_verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        terminal, verified = refresh()
        if not args.watch or terminal:
            if terminal and not verified:
                raise SystemExit(2)
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
