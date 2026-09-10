#!/usr/bin/env python3
"""Source-bound, resumable six-readout queue; never uses a wall-clock cap.

Task partition/interval flow is deliberately derived from the immutable original
runner. Factory dispatch is local, never monkeypatched. A stronger completion
wrapper requires disk replay and exact reference identity/provenance checks.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import math
import os
import pickle
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_low_data import atomic_json, digest, now, conformal_radius, nearest_similarity, planned_tasks
from nesso_pxr.low_data_contract import load_prepared, score_predictions

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / 'experiments/20260906_low_data_followup/protocol_readouts.json'
readouts = importlib.import_module('experiments.20260906_low_data_followup.code.readouts')
STOP = False
REQUIRED_OUTPUTS = frozenset(('predictions.csv', 'metrics.json', 'label_access.json',
    'seed_predictions.npz', 'model_state.pt', 'replay.json', 'reference_bindings.json'))


def complete_task(directory, binding_sha=None, task=None):
    """Reject vacuous/partial/foreign completion markers and corrupted outputs."""
    directory = Path(directory)
    try:
        marker = json.loads((directory / 'complete.json').read_text())
        outputs = marker['output_sha256']
        if not REQUIRED_OUTPUTS.issubset(outputs):
            return False
        if marker.get('schema_version') != 2 or marker.get('task_id') != directory.name:
            return False
        if binding_sha is not None and marker.get('binding_sha256') != binding_sha:
            return False
        if task is not None and marker.get('task') != task:
            return False
        if not all(Path(n).name == n and (directory/n).is_file() and digest(directory/n) == h for n,h in outputs.items()):
            return False
        replay = json.loads((directory/'replay.json').read_text())
        return (all(replay.get(key) is True for key in (
            'exact_seed_replay', 'exact_csv_replay', 'exact_metric_replay',
            'reference_identity_roles_match'))
            and replay.get('model_sha256') == outputs['model_state.pt'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _fit_task(
    task: dict,
    config: dict,
    inputs: pd.DataFrame,
    features: dict,
    assignments: pd.DataFrame,
    subsets: pd.DataFrame,
    run_dir: Path,
) -> dict:
    from nesso_pxr.low_data_contract import assay_weights, score_predictions
    fit_predict = readouts.fit_predict

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



def bind(config, run_dir):
    """Verify original immutable bytes, bind new source/runtime and snapshot code."""
    original_root = ROOT/config['original_run']
    prior = json.loads((original_root/'run_bindings.json').read_text())
    sources = {ROOT/k: v for k,v in prior['sources'].items()}
    inputs = {Path(k): v for k,v in prior['inputs'].items()}
    for path, expected in {**sources, **inputs}.items():
        if digest(path) != expected:
            raise RuntimeError(f'original source/input drift: {path}')
    for path in (Path(__file__), Path(readouts.__file__), PROTOCOL):
        sources[path] = digest(path)
    inputs[original_root/'run_bindings.json'] = digest(original_root/'run_bindings.json')
    inputs[original_root/'task_manifest.json'] = digest(original_root/'task_manifest.json')
    inputs[original_root/'prepared/preparation_manifest.sha256'] = digest(original_root/'prepared/preparation_manifest.sha256')
    payload = dict(schema_version=1, sources={str(p):v for p,v in sorted(sources.items())},
                   inputs={str(p):v for p,v in sorted(inputs.items())}, config=config,
                   runtime=dict(python=sys.version, executable=sys.executable,
                     packages={p:version(p) for p in ('numpy','pandas','scikit-learn','torch','rdkit','lightgbm')},
                     numerical_threads=2))
    path = run_dir/'run_bindings.json'
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError('source/input/protocol/runtime drift: use a versioned attempt')
    if not path.exists():
        atomic_json(path, payload)
    for p, expected in sources.items():
        target = run_dir/'source_snapshot'/p.relative_to(ROOT)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        if digest(target) != expected:
            raise RuntimeError(f'bound snapshot drift: {target}')
    protocol_copy = run_dir/'protocol.json'
    if protocol_copy.exists() and json.loads(protocol_copy.read_text()) != config:
        raise RuntimeError('frozen protocol copy drift')
    if not protocol_copy.exists():
        atomic_json(protocol_copy, config)
    return digest(path)


def check_live_sources(run_dir):
    binding = json.loads((run_dir/'run_bindings.json').read_text())
    for p,h in binding['sources'].items():
        if digest(Path(p)) != h:
            raise RuntimeError(f'source drift during running queue: {p}')


def validate_reference(task, config, label_access, evaluate, test):
    references = {}
    prefix = task['task_id'].rsplit('__', 1)[0]
    for method in config['historical_references']:
        directory = ROOT/config['original_run']/'tasks'/f'{prefix}__{method}'
        marker = json.loads((directory/'complete.json').read_text())
        required = {'metrics.json','predictions.csv','label_access.json','seed_predictions.npz'}
        outputs = marker['output_sha256']
        if not required.issubset(outputs):
            raise RuntimeError(f'incomplete historical reference: {directory}')
        for name, h in outputs.items():
            if Path(name).name != name or digest(directory/name) != h:
                raise RuntimeError(f'corrupt historical reference: {directory/name}')
        old_access = json.loads((directory/'label_access.json').read_text())
        if old_access != label_access:
            raise RuntimeError(f'historical label roles differ: {directory}')
        with np.load(directory/'seed_predictions.npz', allow_pickle=False) as z:
            if not np.array_equal(z['row_index'], evaluate):
                raise RuntimeError(f'historical evaluation row order differs: {directory}')
        old = pd.read_csv(directory/'predictions.csv', float_precision='round_trip', dtype={'record_id':str}, keep_default_na=False)
        if not np.array_equal(old['row_index'].to_numpy(int), test):
            raise RuntimeError(f'historical test rows differ: {directory}')
        if old['record_id'].tolist() != label_access['test_record_ids']:
            raise RuntimeError(f'historical test identities differ: {directory}')
        references[method] = dict(directory=str(directory), complete_sha256=digest(directory/'complete.json'),
                                  output_sha256=outputs, identity_roles_match=True,
                                  metrics=json.loads((directory/'metrics.json').read_text()))
    return references


def run_task(task, config, inputs, features, assignments, subsets, run_dir, binding_sha):
    directory = run_dir/'tasks'/task['task_id']
    result = _fit_task(task, config, inputs, features, assignments, subsets, run_dir)
    # The original-style intermediate marker cannot pass complete_task. Publish
    # only after reading actual disk predictions/state and every reference.
    access = json.loads((directory/'label_access.json').read_text())
    with np.load(directory/'seed_predictions.npz', allow_pickle=False) as z:
        evaluate = z['row_index'].copy()
        stored = z['predictions'].copy()
    state = torch.load(directory/'model_state.pt', map_location='cpu', weights_only=False)
    replay = readouts.predict_model(state, {k:v[evaluate] for k,v in features.items()})
    if not np.array_equal(stored, replay['seed_predictions']):
        raise AssertionError('disk per-seed replay failed')
    test = evaluate[len(access['calibration_record_ids']):]
    pred = pd.read_csv(directory/'predictions.csv', float_precision='round_trip', dtype={'record_id':str})
    if not np.array_equal(pred['y_pred'].to_numpy(), replay['pred'][len(evaluate)-len(test):]):
        raise AssertionError('CSV prediction replay failed')
    recomputed = score_predictions(pred.y_true.to_numpy(), pred.y_pred.to_numpy(), pred.assay_se.to_numpy(), config['se_floor'])
    for key, value in recomputed.items():
        if value != result[key]:
            raise AssertionError(f'serialized metric mismatch: {key}')
    references = validate_reference(task, config, access, evaluate, test)
    atomic_json(directory/'reference_bindings.json', references)
    result['prediction_diagnostics'] = readouts.quantiles(pred.y_pred.to_numpy())
    result['potency_bias'] = {}
    for name, mask in [('pEC50_lt5', pred.y_true < 5), ('pEC50_5_to_lt6', (pred.y_true >= 5) & (pred.y_true < 6)), ('pEC50_ge6', pred.y_true >= 6)]:
        part = pred.loc[mask]
        result['potency_bias'][name] = dict(n=len(part), bias=float((part.y_pred-part.y_true).mean()) if len(part) else None)
    result['score_only_sensitivities'] = {str(floor):score_predictions(pred.y_true.to_numpy(), pred.y_pred.to_numpy(), pred.assay_se.to_numpy(), floor)
                                        for floor in config['score_sensitivity_floors']}
    result['paired_reference_effects'] = {method: {key: result[key]-ref['metrics'][key] for key in ('weighted_mae','mae')}
                                          for method,ref in references.items()}
    atomic_json(directory/'metrics.json', result)
    atomic_json(directory/'replay.json', dict(exact_seed_replay=True, exact_csv_replay=True,
        exact_metric_replay=True, reference_identity_roles_match=True, batch_size=1024,
        model_sha256=digest(directory/'model_state.pt')))
    outputs = {p.name:digest(p) for p in directory.iterdir() if p.is_file()
               and p.name not in {'complete.json','failure.json'} and '.tmp.' not in p.name}
    # fsync all payloads before the durable final marker.
    for name in outputs:
        with (directory/name).open('rb') as stream:
            os.fsync(stream.fileno())
    atomic_json(directory/'complete.json', dict(schema_version=2, task_id=task['task_id'], task=task,
        binding_sha256=binding_sha, completed_at=now(), output_sha256=outputs))
    if not complete_task(directory, binding_sha, task):
        raise AssertionError('final complete marker verification failed')
    return result


def reconcile(run_dir, tasks, binding_sha):
    states = {}
    failures = []
    for task in tasks:
        tid = task['task_id']
        d = run_dir/'tasks'/tid
        if complete_task(d, binding_sha, task):
            states[tid] = 'complete'
        elif (d/'failure.json').exists():
            states[tid] = 'failed'
            failures.append(dict(task_id=tid, state='failed', evidence=str(d/'failure.json')))
        elif (d/'complete.json').exists():
            states[tid] = 'invalid'
            failures.append(dict(task_id=tid, state='invalid', evidence=str(d/'complete.json')))
        else:
            states[tid] = 'pending'
    return states, failures


def atomic_csv(frame, path):
    temporary = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    frame.to_csv(temporary, index=False, float_format='%.17g')
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def report(run_dir, tasks, binding_sha):
    """Deterministic task/aggregate tables; no new fits, no pseudo-independent CI."""
    states, failures = reconcile(run_dir, tasks, binding_sha)
    rows, paired = [], []
    for task in sorted(tasks, key=lambda t:t['task_id']):
        if states[task['task_id']] != 'complete':
            continue
        m = json.loads((run_dir/'tasks'/task['task_id']/'metrics.json').read_text())
        rows.append({**task, **{k:m[k] for k in ('n_fit','n_calibration','n_test','weighted_mae','mae','spearman','fit_seconds')}})
        for ref,effect in sorted(m['paired_reference_effects'].items()):
            paired.append(dict(**task, reference=ref, weighted_mae_difference=effect['weighted_mae'], mae_difference=effect['mae']))
    frame = pd.DataFrame(rows, columns=['task_id','split','outer_fold','draw','n_train','method','n_fit','n_calibration','n_test','weighted_mae','mae','spearman','fit_seconds'])
    atomic_csv(frame, run_dir/'summary_tasks.csv')
    pf = pd.DataFrame(paired, columns=['task_id','split','outer_fold','draw','n_train','method','reference','weighted_mae_difference','mae_difference'])
    atomic_csv(pf, run_dir/'summary_paired.csv')
    aggregate = [] if frame.empty else frame.groupby(['split','n_train','method'], sort=True).agg(
        completed_tasks=('task_id','count'), mean_task_weighted_mae=('weighted_mae','mean'),
        mean_task_mae=('mae','mean')).reset_index().to_dict('records')
    atomic_csv(pd.DataFrame(aggregate, columns=['split','n_train','method','completed_tasks','mean_task_weighted_mae','mean_task_mae']), run_dir/'summary_aggregates.csv')
    result = dict(expected=len(tasks), valid_complete=sum(v=='complete' for v in states.values()),
                  counts={s:sum(v==s for v in states.values()) for s in ('complete','failed','invalid','pending')},
                  retrospective=True, metric_semantics='unweighted means of task-level weighted MAE, not pooled observations',
                  dependence='repeated draws/folds are not independent observations; no superiority margin or CI asserted',
                  aggregate=aggregate)
    atomic_json(run_dir/'summary.json', result)
    atomic_json(run_dir/'failure_ledger.json', failures)
    return result


def stop_requested(_signum, _frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['initialize','smoke','run','verify','report'], required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    config = json.loads(PROTOCOL.read_text())
    if tuple(config['methods']) != readouts.METHODS:
        raise RuntimeError('protocol/method mismatch')
    run_dir = (args.run_dir or ROOT/config['run_dir']).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    # Reporting and verification are read-only with respect to task artifacts.
    lock = None
    if args.mode not in ('verify','report'):
        lock = (run_dir/'runner.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding_sha = bind(config, run_dir)
    tasks = planned_tasks(config)
    manifest = run_dir/'task_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != tasks:
        raise RuntimeError('task manifest drift')
    if not manifest.exists():
        atomic_json(manifest, tasks)
    if args.mode == 'report':
        print(json.dumps(report(run_dir, tasks, binding_sha)), flush=True)
        return 0
    states, failures = reconcile(run_dir, tasks, binding_sha)
    if args.mode == 'verify':
        valid = all(s=='complete' for s in states.values())
        result = dict(status='complete' if valid else 'incomplete', expected=len(tasks),
            valid_complete=sum(s=='complete' for s in states.values()), binding_sha256=binding_sha,
            counts={s:sum(v==s for v in states.values()) for s in ('complete','failed','invalid','pending')},
            missing_or_failed=[t for t,s in states.items() if s!='complete'])
        atomic_json(run_dir/'verification.json', result)
        print(json.dumps({k:v for k,v in result.items() if k!='missing_or_failed'}), flush=True)
        return 0 if valid else 2
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    selected = [t for t in tasks if args.mode != 'smoke' or (t['n_train']==25 and t['outer_fold']==0 and t['draw']==0)]
    attempted = 0
    started = time.monotonic()
    def publish(state, current=None):
        atomic_json(run_dir/'status.json', dict(status=state, updated_at=now(), pid=os.getpid(),
            hostname=platform.node(), expected=len(tasks), selected=len(selected), attempted_this_invocation=attempted,
            elapsed_seconds=time.monotonic()-started, current_task=current,
            counts={s:sum(v==s for v in states.values()) for s in ('complete','failed','invalid','pending','running')},
            binding_sha256=binding_sha))
        atomic_json(run_dir/'failure_ledger.json', failures)
    publish('initialized')
    if args.mode == 'initialize':
        return 0
    if args.mode == 'run':
        gate = json.loads((run_dir/'pilot.json').read_text())
        if gate.get('status') != 'pass' or gate.get('binding_sha256') != binding_sha:
            raise RuntimeError('matching real N25 pilot gate required')
        for tid,h in gate['complete_markers'].items():
            if digest(run_dir/'tasks'/tid/'complete.json') != h or states[tid] != 'complete':
                raise RuntimeError(f'pilot artifact drift: {tid}')
    original_root = ROOT/config['original_run']
    inputs, features, assignments, subsets = load_prepared(original_root/'prepared')
    config_for_fit = {**config, 'checkpoint':json.loads((original_root/'protocol.json').read_text())['checkpoint']}
    for task in selected:
        tid = task['task_id']
        if states[tid] == 'complete':
            continue
        if states[tid] in ('failed','invalid') and not args.retry_failed:
            continue
        if STOP:
            break
        check_live_sources(run_dir)
        directory = run_dir/'tasks'/tid
        if args.retry_failed and directory.exists() and states[tid] in ('failed','invalid'):
            archive = run_dir/'failed_attempts'/f'{tid}__{time.time_ns()}'
            archive.parent.mkdir(parents=True, exist_ok=True)
            os.replace(directory, archive)
        states[tid] = 'running'
        publish('running', tid)
        print(f'START {tid}', flush=True)
        try:
            result = run_task(task, config_for_fit, inputs, features, assignments, subsets, run_dir, binding_sha)
            states[tid] = 'complete'
            print(f"COMPLETE {tid} weighted_mae={result['weighted_mae']:.9g}", flush=True)
        except Exception as error:
            states[tid] = 'failed'
            failure = dict(**task, failed_at=now(), error=str(error), traceback=traceback.format_exc(), scientific_evidence=False)
            atomic_json(directory/'failure.json', failure)
            failures.append(failure)
            print(f'FAILED {tid}: {error}', flush=True)
        attempted += 1
        publish('running')
    states, failures = reconcile(run_dir, tasks, binding_sha)
    selected_pass = all(states[t['task_id']]=='complete' for t in selected)
    complete = all(s=='complete' for s in states.values())
    if args.mode == 'smoke' and selected_pass:
        atomic_json(run_dir/'pilot.json', dict(status='pass', binding_sha256=binding_sha,
            n_tasks=len(selected), exact_disk_seed_csv_metric_replay=True,
            complete_markers={t['task_id']:digest(run_dir/'tasks'/t['task_id']/'complete.json') for t in selected},
            recipes_changed_after_smoke=False))
    publish('complete' if complete else 'paused' if STOP else 'smoke_pass' if args.mode=='smoke' and selected_pass else 'incomplete')
    # Full queue closeout is a separate process and remains resource bounded.
    if args.mode == 'run' and not STOP:
        verify = subprocess.run([sys.executable, str(Path(__file__)), '--mode','verify','--run-dir',str(run_dir)], check=False)
        reporting = subprocess.run([sys.executable, str(Path(__file__)), '--mode','report','--run-dir',str(run_dir)], check=False)
        if verify.returncode or reporting.returncode:
            publish('incomplete_or_reporting_failed')
            return 2
    return 0 if selected_pass or STOP else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print(traceback.format_exc(), flush=True)
        # Explicit run-level fatal evidence, without overwriting task failures.
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument('--run-dir', type=Path)
        a, _ = p.parse_known_args()
        d = a.run_dir or ROOT/'artifacts/experiments/low_data_followup_20260906/readouts'
        atomic_json(d/'runner_failure.json', dict(at=now(), pid=os.getpid(), traceback=traceback.format_exc(), scientific_evidence=False))
        raise
