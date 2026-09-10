#!/usr/bin/env python3
"""Resumable local low-data study; fitting sees only the declared fit labels."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STOP = False


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def conformal_radius(residuals: np.ndarray, coverage: float) -> float:
    """Finite sample split-conformal rank; infinity is scientifically meaningful."""
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("calibration residuals must be a nonempty finite vector")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    rank = math.ceil((len(values) + 1) * coverage)
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else math.inf


def nearest_similarity(test: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Maximum Morgan-bit Tanimoto to fit rows only, in bounded chunks."""
    if not len(fit):
        raise ValueError("empty fit fingerprint matrix")
    b = np.asarray(fit, dtype=np.float32)
    bsum = b.sum(axis=1)
    answer = []
    for start in range(0, len(test), 256):
        a = np.asarray(test[start : start + 256], dtype=np.float32)
        intersection = a @ b.T
        union = a.sum(axis=1)[:, None] + bsum[None, :] - intersection
        sims = np.divide(intersection, union, out=np.ones_like(union), where=union > 0)
        answer.append(sims.max(axis=1))
    return np.concatenate(answer).astype(float)


def task_id(split: str, fold: int, draw: int, budget: int, method: str) -> str:
    label = "full" if budget == -1 else str(budget)
    return f"{split}__fold{fold}__draw{draw:02d}__n{label}__{method}"


def planned_tasks(config: dict) -> list[dict]:
    budgets = list(config["budgets"])
    if config.get("include_full_reference", False):
        budgets.append(-1)
    tasks = []
    # Breadth first: a complete paired method panel before another draw or fold.
    for draw in range(config["draws"]):
        for fold in range(config["n_folds"]):
            for budget in budgets:
                for split in config["split_policies"]:
                    for method in config["methods"]:
                        tasks.append(
                            {
                                "task_id": task_id(split, fold, draw, budget, method),
                                "split": split,
                                "outer_fold": fold,
                                "draw": draw,
                                "n_train": budget,
                                "method": method,
                            }
                        )
    return tasks


def build_bindings(config: dict, prepared: Path) -> dict:
    sources = list((ROOT / "src/nesso_pxr").glob("*.py"))
    sources += [Path(__file__), ROOT / "scripts/run_dual_branch_activity_adapter.py"]
    inputs = [ROOT / config[k] for k in ("manifest", "features", "checkpoint")]
    inputs += [
        prepared / n
        for n in (
            "inputs.csv",
            "features.npz",
            "assignments.csv",
            "subsets.csv",
            "preparation_manifest.json",
            "split_audit.json",
        )
    ]
    for path in sources + inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    from importlib.metadata import version

    return {
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "inputs": {str(p.resolve()): digest(p) for p in inputs},
        "config": config,
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "torch",
                    "rdkit",
                    "lightgbm",
                )
            },
        },
    }


def complete_task(directory: Path) -> bool:
    marker = directory / "complete.json"
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text())
        return all(
            (directory / n).is_file() and digest(directory / n) == h
            for n, h in data["output_sha256"].items()
        )
    except (ValueError, OSError, KeyError):
        return False


def run_task(
    task: dict,
    config: dict,
    inputs: pd.DataFrame,
    features: dict,
    assignments: pd.DataFrame,
    subsets: pd.DataFrame,
    run_dir: Path,
) -> dict:
    from nesso_pxr.low_data_contract import assay_weights, score_predictions
    from nesso_pxr.low_data_models import fit_predict

    directory = run_dir / "tasks" / task["task_id"]
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "started.json", {**task, "started_at": now()})
    local = subsets.loc[
        subsets["split"].eq(task["split"])
        & subsets["outer_fold"].eq(task["outer_fold"])
        & subsets["draw"].eq(task["draw"])
        & subsets["n_train"].eq(task["n_train"])
    ]
    fit = local.loc[local["role"].eq("fit"), "row_index"].to_numpy(dtype=int)
    calibration = local.loc[local["role"].eq("calibration"), "row_index"].to_numpy(
        dtype=int
    )
    policy = assignments.loc[assignments["split"].eq(task["split"])]
    test = policy.loc[
        policy["outer_fold"].eq(task["outer_fold"]), "row_index"
    ].to_numpy(dtype=int)
    assert len(fit) and len(calibration) and len(test), "empty partition"
    assert not (
        set(fit) & set(calibration)
        or set(fit) & set(test)
        or set(calibration) & set(test)
    ), "overlapping partitions"
    assert len(set(fit) | set(calibration)) == len(local), "duplicate acquired row"
    if task["n_train"] != -1:
        assert len(local) == task["n_train"], "label budget violation"
    groups = policy.set_index("row_index")["chemical_group"].reindex(inputs.index)
    if task["split"] == "chemical_cluster":
        assert not set(groups.iloc[np.r_[fit, calibration]]) & set(groups.iloc[test])
    evaluate = np.r_[calibration, test]
    train_features = {k: v[fit] for k, v in features.items()}
    train_features["groups"] = (
        groups.iloc[fit].to_numpy()
        if task["split"] == "chemical_cluster"
        else inputs.iloc[fit]["record_id"].to_numpy()
    )
    test_features = {k: v[evaluate] for k, v in features.items()}
    result = fit_predict(
        task["method"],
        train_features,
        inputs.iloc[fit]["pEC50"].to_numpy(dtype=float),
        inputs.iloc[fit]["pEC50_standard_error"].to_numpy(dtype=float),
        test_features,
        config=config,
        checkpoint_path=ROOT / config["checkpoint"],
        seeds=tuple(config["seeds"]),
    )
    predictions = np.asarray(result["pred"], dtype=float)
    spread = np.asarray(result["ensemble_std"], dtype=float)
    assert predictions.shape == spread.shape == (len(evaluate),)
    assert np.isfinite(predictions).all() and np.isfinite(spread).all()
    ycal = inputs.iloc[calibration]["pEC50"].to_numpy(dtype=float)
    ytest = inputs.iloc[test]["pEC50"].to_numpy(dtype=float)
    ptest = predictions[len(calibration) :]
    se = inputs.iloc[test]["pEC50_standard_error"].to_numpy(dtype=float)
    pred = pd.DataFrame(
        {
            "record_id": inputs.iloc[test]["record_id"].to_numpy(),
            "row_index": test,
            **{
                k: task[k] for k in ("split", "outer_fold", "draw", "n_train", "method")
            },
            "y_true": ytest,
            "y_pred": ptest,
            "assay_se": se,
            "weight": assay_weights(se, config["se_floor"]),
            "ensemble_std": spread[len(calibration) :],
            "nearest_similarity": nearest_similarity(
                features["morgan"][test], features["morgan"][fit]
            ),
            "chemical_group": groups.iloc[test].to_numpy(),
        }
    )
    metrics = {
        **task,
        "status": "complete",
        "completed_at": now(),
        "n_fit": len(fit),
        "n_calibration": len(calibration),
        "n_test": len(test),
        **score_predictions(ytest, ptest, se, config["se_floor"]),
        "fit_seconds": result["fit_seconds"],
        "n_parameters": result["n_parameters"],
        "selection": result.get("selection", {}),
        "fit_audit": result.get("fit_audit", {}),
    }
    for coverage in config["interval_coverages"]:
        key = str(round(coverage * 100))
        radius = conformal_radius(
            np.abs(ycal - predictions[: len(calibration)]), coverage
        )
        pred[f"lower_{key}"] = ptest - radius
        pred[f"upper_{key}"] = ptest + radius
        metrics[f"coverage_{key}"] = float(np.mean(np.abs(ytest - ptest) <= radius))
        metrics[f"width_{key}"] = 2 * radius
        metrics[f"interval_{key}_finite"] = math.isfinite(radius)
    temporary = directory / f"predictions.csv.tmp.{os.getpid()}"
    pred.to_csv(temporary, index=False)
    os.replace(temporary, directory / "predictions.csv")
    atomic_json(directory / "metrics.json", metrics)
    atomic_json(
        directory / "label_access.json",
        {
            "fit_record_ids": inputs.iloc[fit]["record_id"].tolist(),
            "calibration_record_ids": inputs.iloc[calibration]["record_id"].tolist(),
            "test_record_ids": inputs.iloc[test]["record_id"].tolist(),
            "test_labels_accessible_to_fit": False,
            "calibration_labels_accessible_to_fit": False,
            "acquired_label_count": len(local),
        },
    )
    np.savez_compressed(
        directory / "seed_predictions.npz",
        row_index=evaluate,
        predictions=np.asarray(result["seed_predictions"]),
    )
    # Save factory-provided replay artifacts without assuming their internal schema.
    if "model_state" in result:
        import torch

        torch.save(result["model_state"], directory / "model_state.pt")
    files = [
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.name not in {"complete.json", "failure.json"}
        and ".tmp." not in p.name
    ]
    atomic_json(
        directory / "complete.json",
        {
            "task_id": task["task_id"],
            "completed_at": now(),
            "output_sha256": {p.name: digest(p) for p in files},
        },
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "experiments/20260905_low_data_adaptation/protocol.json",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--draws", type=int, nargs="+")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--report-every", type=int, default=24)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def stop_requested(_signum: int, _frame: Any) -> None:
    global STOP
    STOP = True


def main() -> None:
    from nesso_pxr.low_data_contract import load_prepared

    args = parse_args()
    config = json.loads(args.protocol.read_text())
    run_dir = (args.run_dir or ROOT / config["run_dir"]).resolve()
    prepared = (args.prepared_dir or run_dir / "prepared").resolve()
    preparation = json.loads((prepared / "preparation_manifest.json").read_text())
    if preparation["config"] != config:
        raise RuntimeError(
            "prepared protocol differs: prepare a new immutable input set"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    bindings = build_bindings(config, prepared)
    frozen = run_dir / "run_bindings.json"
    if frozen.exists() and json.loads(frozen.read_text()) != bindings:
        raise RuntimeError("source/input/protocol drift: use a distinct verified run")
    if not frozen.exists():
        atomic_json(frozen, bindings)
    for name, expected_hash in bindings["sources"].items():
        snapshot = run_dir / "source_snapshot" / name
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, snapshot)
        if digest(snapshot) != expected_hash:
            raise RuntimeError(f"source snapshot differs: {name}")
    atomic_json(run_dir / "protocol.json", config)
    inputs, features, assignments, subsets = load_prepared(prepared)
    assert np.array_equal(inputs["row_index"].to_numpy(), np.arange(len(inputs)))
    all_tasks = planned_tasks(config)
    atomic_json(run_dir / "task_manifest.json", all_tasks)
    selected = [
        t
        for t in all_tasks
        if all(
            allowed is None or t[field] in allowed
            for field, allowed in (
                ("method", args.methods),
                ("n_train", args.budgets),
                ("outer_fold", args.folds),
                ("draw", args.draws),
                ("split", args.splits),
            )
        )
    ]
    if not selected:
        raise ValueError("no tasks match selection")
    states = {}
    for task in all_tasks:
        directory = run_dir / "tasks" / task["task_id"]
        if complete_task(directory):
            states[task["task_id"]] = "complete"
        elif (directory / "failure.json").exists():
            states[task["task_id"]] = "failed"
        else:
            states[task["task_id"]] = "pending"
    started = time.monotonic()
    attempted = 0
    max_seconds = args.max_seconds or config["maximum_queue_hours"] * 3600

    def publish(state: str, current: str | None = None) -> None:
        counts = {
            key: sum(s == key for s in states.values())
            for key in ("complete", "failed", "pending", "running")
        }
        atomic_json(
            run_dir / "status.json",
            {
                "status": state,
                "updated_at": now(),
                "pid": os.getpid(),
                "hostname": platform.node(),
                "total_tasks": len(all_tasks),
                "selected_tasks": len(selected),
                "attempted_this_invocation": attempted,
                "elapsed_seconds": time.monotonic() - started,
                "current_task": current,
                **counts,
                "tasks": states,
            },
        )

    def report() -> None:
        if args.report_dir:
            try:
                from nesso_pxr.low_data_report import build_report

                build_report(run_dir, args.report_dir)
            except Exception:
                atomic_json(
                    run_dir / "report_failure.json",
                    {"at": now(), "traceback": traceback.format_exc()},
                )
                print("REPORT_FAILED: see report_failure.json", flush=True)

    publish("initialized")
    if args.initialize_only:
        report()
        return
    for task in selected:
        tid = task["task_id"]
        if states[tid] == "complete" or (
            states[tid] == "failed" and not args.retry_failed
        ):
            continue
        if STOP or time.monotonic() - started >= max_seconds:
            break
        if args.max_tasks is not None and attempted >= args.max_tasks:
            break
        states[tid] = "running"
        publish("running", tid)
        print(f"START {tid}", flush=True)
        try:
            result = run_task(
                task, config, inputs, features, assignments, subsets, run_dir
            )
            states[tid] = "complete"
            print(
                f"COMPLETE {tid} weighted_mae={result['weighted_mae']:.6f}", flush=True
            )
        except Exception as error:
            states[tid] = "failed"
            atomic_json(
                run_dir / "tasks" / tid / "failure.json",
                {
                    **task,
                    "failed_at": now(),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "scientific_evidence": False,
                },
            )
            print(f"FAILED {tid}: {error}", flush=True)
        attempted += 1
        publish("running")
        if args.report_every > 0 and attempted % args.report_every == 0:
            report()
    final = "complete" if all(v == "complete" for v in states.values()) else "paused"
    if (
        all(v in {"complete", "failed"} for v in states.values())
        and final != "complete"
    ):
        final = "finished_with_failures"
    publish(final)
    report()
    print(f"{final.upper()} tasks_attempted={attempted}", flush=True)
    with contextlib.suppress(OSError):
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
