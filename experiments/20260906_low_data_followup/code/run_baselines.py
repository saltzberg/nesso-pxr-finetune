#!/usr/bin/env python3
"""Uncapped, source-bound, resumable Lane 3 runner. CPU only, two threads.

Use ordinary imports rather than patching the historical factory. --max-tasks is
an explicit small-pilot limit, never a queue elapsed-time limit. Failed tasks are
retained and completion is reconciled against the entire frozen manifest.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import io
import json
import math
import os
import pickle
import platform
import shutil
import signal
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nesso_pxr.low_data_contract import assay_weights, load_prepared, score_predictions
from scripts.run_low_data import atomic_json, conformal_radius, digest, nearest_similarity, now, planned_tasks

B = importlib.import_module("experiments.20260906_low_data_followup.code.baselines")
ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/20260906_low_data_followup"
DEFAULT_RUN = ROOT / "artifacts/experiments/low_data_followup_20260906/baselines"
ORIGINAL = ROOT / "artifacts/experiments/low_data_20260905_cut035_run2"
PREPARED = ORIGINAL / "prepared"
PROTOCOL = EXP / "protocol_baselines.json"
OUTPUTS = {"started.json", "predictions.csv", "calibration_predictions.csv", "metrics.json",
           "label_access.json", "seed_predictions.npz", "model_state.pkl"}
STOP = False


def source_paths():
    return sorted([Path(__file__).resolve(), Path(B.__file__).resolve(), ROOT / "scripts/run_low_data.py"]
                  + list((ROOT / "src/nesso_pxr").glob("*.py")))


def input_paths():
    return sorted([p for p in PREPARED.iterdir() if p.is_file()]
                  + [DEFAULT_RUN / "features/descriptors.npz", DEFAULT_RUN / "features/descriptor_metadata.json"])


def check_original():
    old = json.loads((ORIGINAL / "run_bindings.json").read_text())
    for name, expected in old["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"original bound source drift: {name}")
    # Preparation hashes/identity/subset semantics are independently checked on load.
    return {"original_run_bindings_sha256": digest(ORIGINAL / "run_bindings.json"),
            "original_sources_verified": len(old["sources"])}


def freeze_protocol():
    if PROTOCOL.exists():
        raise ValueError("protocol already frozen; preserve it and version any amendment")
    check_original()
    # Full validation before freezing task selection; no model outcomes accessed.
    inputs, _, _, _ = load_prepared(PREPARED)
    _, metadata = B.load_descriptors(DEFAULT_RUN / "features", inputs[["record_id", "canonical_smiles", "row_index"]])
    config = json.loads((ORIGINAL / "protocol.json").read_text())
    for key in ("maximum_queue_hours", "maximum_adapter_pilot_minutes"):
        config.pop(key, None)
    config.update(experiment_id="low_data_followup_20260906_baselines", threads=2,
                  methods=list(B.METHODS), run_dir=str(DEFAULT_RUN.relative_to(ROOT)),
                  status="frozen_before_new_model_outcomes", frozen_at=now(),
                  data_scope="development_only_3344_retrospective_hypothesis_generating",
                  descriptor_feature_identity=metadata["feature_identity_sha256"],
                  baseline_recipe=B.RECIPE, runtime_cap_seconds=None,
                  original_prepared_dir=str(PREPARED.relative_to(ROOT)),
                  descriptor_comparator_name="canonical-SMILES RDKit2D + Mordred2D + Morgan, training-only reduced LightGBM",
                  historical_reconstruction=False,
                  historical_unavailable="Unreduced all_ligands feature_registry.csv and source tables; exact prepared-top lineage/order/settings. Published reduced matrix cannot invert fitted column removal.",
                  preprocessing="finite-value mean imputation; nanvar ddof0 >.01; abs(mean-imputed standardized cross-product/n)>.90 connected components; representative missingness asc, variance desc, raw-index asc; 150-round gain top1000; every operation refit within each inner-training and final-fit subset",
                  tuning="six fixed rounds/capacity pairs; pooled inner-OOF inverse-SE weighted MAE; first candidate breaks exact ties; no early stopping or outer-aware selection",
                  affine="fit-only inverse-SE weighted squared regression, slope [0,2], unconstrained intercept; zero input variance slope=0; no prediction clipping",
                  median="fit-only weighted lower median using inverse max(SE,.10)",
                  deterministic_seeds="one actual final fit; arrays repeated for 42/43/44, no stochastic ensemble interpretation",
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths()},
                  input_sha256={str(p.resolve()): digest(p) for p in input_paths()},
                  runtime={"python": platform.python_version(), "packages": {k: version(k) for k in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "rdkit", "mordredcommunity", "torch")}})
    atomic_json(PROTOCOL, config)
    print(f"FROZEN {PROTOCOL} sha256={digest(PROTOCOL)}", flush=True)


def assert_bindings(bindings, *, inputs=True):
    if digest(PROTOCOL) != bindings["protocol_sha256"]:
        raise ValueError("protocol drift")
    for rel, h in bindings["config"]["source_sha256"].items():
        if digest(ROOT / rel) != h:
            raise ValueError(f"source drift: {rel}")
    if inputs:
        for name, h in bindings["config"]["input_sha256"].items():
            if digest(Path(name)) != h:
                raise ValueError(f"input drift: {name}")


def initialize(run):
    config = json.loads(PROTOCOL.read_text())
    bindings = {"protocol_sha256": digest(PROTOCOL), "config": config}
    assert_bindings(bindings)
    run.mkdir(parents=True, exist_ok=True)
    path = run / "run_bindings.json"
    if path.exists() and json.loads(path.read_text()) != bindings:
        raise ValueError("run binding drift; use a distinct versioned attempt")
    if not path.exists():
        atomic_json(path, bindings)
    for rel, h in config["source_sha256"].items():
        target = run / "source_snapshot" / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
        if digest(target) != h:
            raise ValueError(f"source snapshot drift: {rel}")
    tasks = planned_tasks(config)
    for name, data in (("protocol.json", config), ("task_manifest.json", tasks)):
        target = run / name
        if target.exists() and json.loads(target.read_text()) != data:
            raise ValueError(f"immutable {name} differs")
        if not target.exists():
            atomic_json(target, data)
    return config, bindings, tasks


def complete_task(directory, task, binding_hash):
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    data = json.loads(marker.read_text())
    if data.get("task") != task or data.get("binding_sha256") != binding_hash or set(data.get("output_sha256", {})) != OUTPUTS:
        raise ValueError(f"invalid completion schema/task/binding: {directory}")
    for name, h in data["output_sha256"].items():
        if not (directory / name).is_file() or digest(directory / name) != h:
            raise ValueError(f"corrupt completed output: {directory / name}")
    return True


def partitions(task, inputs, assignments, subsets):
    mask = np.ones(len(subsets), dtype=bool)
    for k in ("split", "outer_fold", "draw", "n_train"):
        mask &= subsets[k].eq(task[k]).to_numpy()
    local = subsets.loc[mask]
    fit = local.loc[local.role.eq("fit"), "row_index"].to_numpy(dtype=int)
    cal = local.loc[local.role.eq("calibration"), "row_index"].to_numpy(dtype=int)
    policy = assignments.loc[assignments.split.eq(task["split"])]
    test = policy.loc[policy.outer_fold.eq(task["outer_fold"]), "row_index"].to_numpy(dtype=int)
    if not len(fit) or not len(cal) or not len(test) or set(fit) & set(cal) or set(fit) & set(test) or set(cal) & set(test):
        raise ValueError("invalid/overlapping task roles")
    if len(set(fit) | set(cal)) != len(local) or (task["n_train"] != -1 and len(local) != task["n_train"]):
        raise ValueError("label budget violation")
    if len(cal) != math.ceil(len(local) * .2):
        raise ValueError("calibration fraction violation")
    groups = policy.set_index("row_index").chemical_group.reindex(inputs.index).to_numpy()
    if task["split"] == "chemical_cluster" and set(groups[np.r_[fit, cal]]) & set(groups[test]):
        raise ValueError("outer group overlap")
    return fit, cal, test, groups


def task_features(task, features, fit, evaluate, groups, inputs):
    required = {"descriptors"} if task["method"] == B.DESCRIPTOR_METHOD else {"native_continuous"}
    train = {k: features[k][fit] for k in required}
    train["groups"] = groups[fit] if task["split"] == "chemical_cluster" else inputs.iloc[fit].record_id.to_numpy()
    return train, {k: features[k][evaluate] for k in required}


def run_task(task, config, inputs, features, assignments, subsets, run, binding_hash):
    directory = run / "tasks" / task["task_id"]
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "started.json", {**task, "started_at": now()})
    fit, cal, test, groups = partitions(task, inputs, assignments, subsets)
    evaluate = np.r_[cal, test]
    trainf, evalf = task_features(task, features, fit, evaluate, groups, inputs)
    result = B.fit_predict(task["method"], trainf, inputs.iloc[fit].pEC50.to_numpy(dtype=float),
                          inputs.iloc[fit].pEC50_standard_error.to_numpy(dtype=float), evalf,
                          config=config, checkpoint_path=None, seeds=tuple(config["seeds"]))
    payload = pickle.dumps(result["model_state"], protocol=pickle.HIGHEST_PROTOCOL)
    replay = B.predict_model(pickle.loads(payload), evalf)
    np.testing.assert_array_equal(replay["seed_predictions"], result["seed_predictions"])
    ycal, ytest = inputs.iloc[cal].pEC50.to_numpy(dtype=float), inputs.iloc[test].pEC50.to_numpy(dtype=float)
    pcal, ptest = result["pred"][:len(cal)], result["pred"][len(cal):]
    se = inputs.iloc[test].pEC50_standard_error.to_numpy(dtype=float)
    pred = pd.DataFrame({"record_id": inputs.iloc[test].record_id.to_numpy(), "row_index": test,
        **{k: task[k] for k in ("split", "outer_fold", "draw", "n_train", "method")},
        "y_true": ytest, "y_pred": ptest, "assay_se": se, "weight": assay_weights(se, .10),
        "ensemble_std": result["ensemble_std"][len(cal):], "chemical_group": groups[test],
        "nearest_similarity": nearest_similarity(features["morgan"][test], features["morgan"][fit])})
    metrics = {**task, "status": "complete", "completed_at": now(), "n_fit": len(fit),
        "n_calibration": len(cal), "n_test": len(test), **score_predictions(ytest, ptest, se, .10),
        "fit_seconds": result["fit_seconds"], "n_parameters": result["n_parameters"],
        "selection": result["selection"], "fit_audit": result["fit_audit"],
        "prediction_quantiles": np.quantile(ptest, [0, .01, .5, .99, 1]).tolist(),
        "maximum_absolute_error": float(np.max(np.abs(ptest-ytest))),
        "largest_error_record_id": inputs.iloc[test[np.argmax(np.abs(ptest-ytest))]].record_id,
        "potency_bias": {name: {"n": int(mask.sum()), "bias": float(np.mean((ptest-ytest)[mask])) if mask.any() else None}
             for name, mask in (("below_5", ytest < 5), ("5_to_6", (ytest >= 5) & (ytest < 6)), ("at_least_6", ytest >= 6))}}
    for coverage in config["interval_coverages"]:
        key = str(round(coverage * 100))
        radius = conformal_radius(np.abs(ycal-pcal), coverage)
        pred[f"lower_{key}"], pred[f"upper_{key}"] = ptest-radius, ptest+radius
        metrics[f"coverage_{key}"] = float(np.mean(np.abs(ytest-ptest) <= radius))
        metrics[f"width_{key}"], metrics[f"interval_{key}_finite"] = 2*radius, math.isfinite(radius)
    B._atomic(directory / "predictions.csv", pred.to_csv(index=False).encode())
    calframe = pd.DataFrame({"row_index": cal, "record_id": inputs.iloc[cal].record_id.to_numpy(), "y_true": ycal, "y_pred": pcal})
    B._atomic(directory / "calibration_predictions.csv", calframe.to_csv(index=False).encode())
    atomic_json(directory / "metrics.json", metrics)
    atomic_json(directory / "label_access.json", {"fit_record_ids": inputs.iloc[fit].record_id.tolist(),
        "calibration_record_ids": inputs.iloc[cal].record_id.tolist(), "test_record_ids": inputs.iloc[test].record_id.tolist(),
        "fit_row_indices": fit.tolist(), "calibration_row_indices": cal.tolist(), "test_row_indices": test.tolist(),
        "test_labels_accessible_to_fit": False, "calibration_labels_accessible_to_fit": False,
        "acquired_label_count": len(fit)+len(cal), "inner_indices_are_relative_to": "fit_row_indices"})
    buf = io.BytesIO()
    np.savez_compressed(buf, row_index=evaluate, predictions=result["seed_predictions"])
    B._atomic(directory / "seed_predictions.npz", buf.getvalue())
    B._atomic(directory / "model_state.pkl", payload)
    atomic_json(directory / "complete.json", {"task": task, "task_id": task["task_id"],
        "binding_sha256": binding_hash, "completed_at": now(),
        "output_sha256": {n: digest(directory / n) for n in sorted(OUTPUTS)}})
    if not complete_task(directory, task, binding_hash):
        raise ValueError("completion did not validate")
    return metrics


def verify(run, tasks, binding_hash, *, replay_count=0):
    valid, failed, pending = [], [], []
    for task in tasks:
        d = run / "tasks" / task["task_id"]
        if complete_task(d, task, binding_hash):
            valid.append(task)
        elif (d / "failure.json").exists():
            failed.append(task["task_id"])
        else:
            pending.append(task["task_id"])
    replays = []
    if replay_count and valid:
        inputs, features, assignments, subsets = load_prepared(PREPARED)
        features["descriptors"], _ = B.load_descriptors(DEFAULT_RUN / "features", inputs[["record_id", "canonical_smiles", "row_index"]])
        for task in valid[:replay_count]:
            d = run / "tasks" / task["task_id"]
            fit, cal, test, groups = partitions(task, inputs, assignments, subsets)
            audit = json.loads((d / "label_access.json").read_text())
            for key, indices in (("fit", fit), ("calibration", cal), ("test", test)):
                if audit[key + "_row_indices"] != indices.tolist() or audit[key + "_record_ids"] != inputs.iloc[indices].record_id.tolist():
                    raise ValueError("role identity replay mismatch")
            _, evalf = task_features(task, features, fit, np.r_[cal, test], groups, inputs)
            state = pickle.loads((d / "model_state.pkl").read_bytes())
            p = B.predict_model(state, evalf)
            with np.load(d / "seed_predictions.npz") as f:
                np.testing.assert_array_equal(f["row_index"], np.r_[cal, test])
                np.testing.assert_array_equal(f["predictions"], p["seed_predictions"])
            frame = pd.read_csv(d / "predictions.csv", float_precision="round_trip", dtype={"record_id": str})
            np.testing.assert_array_equal(frame.y_pred.to_numpy(), p["pred"][len(cal):])
            np.testing.assert_array_equal(frame.row_index.to_numpy(), test)
            metrics = json.loads((d / "metrics.json").read_text())
            for k, value in score_predictions(frame.y_true, frame.y_pred, frame.assay_se, .10).items():
                if value is not None and not np.isclose(value, metrics[k], rtol=0, atol=1e-12):
                    raise ValueError(f"metric replay mismatch {k}")
            replays.append({"task_id": task["task_id"], "max_abs_error": 0.0, "roles_and_metrics": "pass"})
    result = {"verified_at": now(), "expected": len(tasks), "valid": len(valid), "failed": len(failed),
              "pending": len(pending), "complete": len(valid) == len(tasks), "failed_task_ids": failed,
              "pending_task_ids": pending, "replays": replays}
    atomic_json(run / "verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("pending_task_ids", "failed_task_ids")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replay-count", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.prepare:
        inputs = pd.read_csv(PREPARED / "inputs.csv", usecols=["record_id", "canonical_smiles", "row_index"], dtype={"record_id": str})
        B.prepare_descriptors(inputs, DEFAULT_RUN / "features")
        return
    if args.freeze:
        freeze_protocol()
        return
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config, bindings, tasks = initialize(run)
    binding_hash = digest(run / "run_bindings.json")
    if args.verify:
        verify(run, tasks, binding_hash, replay_count=args.replay_count)
        return
    summary = verify(run, tasks, binding_hash)
    if args.initialize_only:
        return
    inputs, features, assignments, subsets = load_prepared(PREPARED)
    features["descriptors"], _ = B.load_descriptors(DEFAULT_RUN / "features", inputs[["record_id", "canonical_smiles", "row_index"]])
    signatures = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in config["input_sha256"]}
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    attempted, complete, failed = 0, summary["valid"], summary["failed"]
    def publish(status, current=None):
        atomic_json(run / "status.json", {"status": status, "updated_at": now(), "pid": os.getpid(),
            "expected": len(tasks), "complete": complete, "failed": failed,
            "pending": len(tasks)-complete-failed, "current_task": current, "attempted_this_invocation": attempted,
            "threads": 2, "runtime_cap_seconds": None})
    publish("running")
    with threadpool_limits(limits=2):
        for task in tasks:
            directory = run / "tasks" / task["task_id"]
            if complete_task(directory, task, binding_hash):
                continue
            was_failed = (directory / "failure.json").exists()
            if was_failed and not args.retry_failed:
                continue
            if STOP or (args.max_tasks is not None and attempted >= args.max_tasks):
                break
            assert_bindings(bindings, inputs=False)
            if any((Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != sig for p, sig in signatures.items()):
                raise ValueError("input stat drift during active run; no further fits permitted")
            publish("running", task["task_id"])
            print(f"START {task['task_id']}", flush=True)
            try:
                run_task(task, config, inputs, features, assignments, subsets, run, binding_hash)
                complete += 1
                if was_failed:
                    failed -= 1
                print(f"COMPLETE {task['task_id']}", flush=True)
            except Exception as e:
                failure = {**task, "failed_at": now(), "error": str(e), "traceback": traceback.format_exc(), "scientific_evidence": False}
                atomic_json(directory / f"failure_attempt_{time.time_ns()}.json", failure)
                atomic_json(directory / "failure.json", failure)
                if not was_failed:
                    failed += 1
                print(f"FAILED {task['task_id']}: {e}", flush=True)
            attempted += 1
            publish("running")
    assert_bindings(bindings)
    final = verify(run, tasks, binding_hash, replay_count=min(6, complete))
    publish("complete" if final["complete"] else "finished_with_failures" if not final["pending"] else "paused")
    if final["failed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
