"""Fixed nine-method strict panel: gated, source-bound, atomic, uncapped."""
from __future__ import annotations

import argparse
import fcntl
import importlib
import importlib.metadata
import io
import json
import math
import os
import pickle
import platform
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

split = importlib.import_module('experiments.20260906_low_data_followup.code.strict_split')
ROOT, EXPERIMENT, OUTPUT = split.ROOT, split.EXPERIMENT, split.OUTPUT
atomic_json, atomic_bytes, atomic_csv, digest = split.atomic_json, split.atomic_bytes, split.atomic_csv, split.digest
PROTOCOL = EXPERIMENT / 'protocol_strict.json'
MODULE_PREFIX = 'experiments.20260906_low_data_followup.code.'
METHODS = ('native_continuous', 'weighted_median', 'native_affine', 'head_pretrained',
           'head_random', 'morgan_ridge', 'descriptor_lightgbm_rdkit_mordred',
           'repr_ridge_unscaled', 'repr_ridge_stable')
READOUT_METHODS = {'repr_ridge_unscaled', 'repr_ridge_stable'}
BASELINE_METHODS = {'weighted_median', 'native_affine', 'descriptor_lightgbm_rdkit_mordred'}
REQUIRED_OUTPUTS = {'started.json', 'predictions.csv', 'metrics.json', 'label_access.json',
                    'seed_predictions.npz', 'model_state.pkl', 'diagnostics.json'}
STOP = False


def now():
    return datetime.now(UTC).isoformat()


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def read_protocol():
    c = json.loads(PROTOCOL.read_text())
    if tuple(c['methods']) != METHODS or c['threads'] != 2 or c['seeds'] != [42, 43, 44]:
        raise ValueError('fixed method/thread/seed contract changed')
    if c['chemical_similarity_cutoff'] != .35 or c['inner_folds'] != 3 or c['calibration_fraction'] != .2:
        raise ValueError('strict split/label contract changed')
    return c


def dependency_paths():
    paths = [EXPERIMENT / f'code/{name}.py' for name in ('readouts', 'baselines')]
    paths += [EXPERIMENT / f'protocol_{name}.json' for name in ('readouts', 'baselines')]
    paths += [EXPERIMENT / name for name in ('READOUTS.md', 'BASELINES.md')]
    paths += [ROOT / f'tests/test_followup_{name}.py' for name in ('readouts', 'baselines')]
    return paths


def load_dependencies(config):
    missing = [str(p.relative_to(ROOT)) for p in dependency_paths() if not p.is_file()]
    cache = ROOT / config['descriptor_cache']
    missing += [str((cache / n).relative_to(ROOT)) for n in ('descriptors.npz', 'descriptor_metadata.json') if not (cache / n).is_file()]
    if missing:
        raise RuntimeError('PENDING_DEPENDENCY: ' + ', '.join(missing))
    original = importlib.import_module('nesso_pxr.low_data_models')
    readouts = importlib.import_module(MODULE_PREFIX + 'readouts')
    baselines = importlib.import_module(MODULE_PREFIX + 'baselines')
    for module, methods in [(original, set(METHODS)-READOUT_METHODS-BASELINE_METHODS),
                            (readouts, READOUT_METHODS), (baselines, BASELINE_METHODS)]:
        if not methods <= set(module.METHODS) or not callable(module.fit_predict) or not callable(module.predict_model):
            raise ValueError(f'incompatible callable/replay method exports: {module.__name__}')
    if not callable(baselines.load_descriptors):
        raise ValueError('baseline descriptor loader unavailable')
    return original, readouts, baselines


def gate_bindings(config):
    # Docs are coordination inputs, not live source gates: reporting may progress.
    paths = [p for p in dependency_paths() if p.suffix != '.md']
    paths += [PROTOCOL, Path(__file__), EXPERIMENT / 'code/strict_split.py', ROOT / 'tests/test_followup_strict.py']
    paths += sorted((ROOT / 'src/nesso_pxr').glob('*.py'))
    paths += sorted((ROOT / config['descriptor_cache']).glob('*'))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}


def preflight(config):
    modules = load_dependencies(config)
    split.verify_preparation()
    identity = pd.read_csv(OUTPUT / 'identities.csv', dtype={'record_id': str}, keep_default_na=False)
    modules[2].load_descriptors(ROOT / config['descriptor_cache'], descriptor_identity(identity))
    before = gate_bindings(config)
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_followup_readouts.py',
               'tests/test_followup_baselines.py', 'tests/test_followup_strict.py']
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONPATH': 'src:.'},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    # No wall-clock timeout on real validation.
    atomic_bytes(OUTPUT / 'dependency_preflight.log', result.stdout.encode())
    if result.returncode:
        raise RuntimeError('dependency tests failed; see dependency_preflight.log')
    if gate_bindings(config) != before:
        raise ValueError('dependency sources changed during preflight')
    gate = dict(status='pass', verified_at=now(), command=command, bindings=before,
                test_log_sha256=digest(OUTPUT / 'dependency_preflight.log'))
    atomic_json(OUTPUT / 'dependency_gate.json', gate)
    print(result.stdout)
    return gate


def require_gate(config):
    modules = load_dependencies(config)
    path = OUTPUT / 'dependency_gate.json'
    if not path.is_file():
        raise RuntimeError('PENDING_DEPENDENCY: run preflight after A/B modules, cache and tests are ready')
    gate = json.loads(path.read_text())
    if (gate.get('status') != 'pass' or gate.get('bindings') != gate_bindings(config)
            or gate.get('test_log_sha256') != digest(OUTPUT / 'dependency_preflight.log')):
        raise ValueError('dependency gate stale/corrupt; revalidate before a NEW bound run')
    return modules


def task_manifest(audit):
    tasks = []
    for draw in range(10):
        for fold in range(5):
            for n in audit['supported_budgets'] + [-1]:
                for method in METHODS:
                    label = 'full' if n == -1 else str(n)
                    tasks.append(dict(task_id=f'strict_chemical__fold{fold}__draw{draw:02d}__n{label}__{method}',
                        split='strict_chemical', outer_fold=fold, draw=draw, n_train=n, method=method))
    return tasks


def build_bindings(config):
    files = {str(ROOT / p): h for p, h in gate_bindings(config).items()}
    manifest = json.loads((OUTPUT / 'preparation_manifest.json').read_text())
    for p in list(manifest['outputs']) + ['preparation_manifest.json', 'preparation_manifest.sha256',
                                         'dependency_gate.json', 'dependency_preflight.log']:
        files[str(OUTPUT / p)] = digest(OUTPUT / p)
    for p in manifest['original_inputs']:
        path = ROOT / config['original_preparation'] / p
        files[str(path)] = digest(path)
    checkpoint = ROOT / config['checkpoint']
    files[str(checkpoint)] = digest(checkpoint)
    # Compare released checkpoint to the immutable original preparation provenance.
    original_meta = json.loads((ROOT / config['original_preparation'] / 'preparation_manifest.json').read_text())
    if files[str(checkpoint)] != original_meta['sources']['checkpoint']['sha256']:
        raise ValueError('released checkpoint differs from original prepared provenance')
    return dict(files=files, config=config, runtime={'python': platform.python_version(),
        'executable': sys.executable, 'packages': {name: importlib.metadata.version(name) for name in
        ['numpy', 'pandas', 'rdkit', 'torch', 'scikit-learn', 'lightgbm']}})


def freeze_run(run_dir, bindings, tasks):
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [('run_bindings.json', bindings), ('task_manifest.json', tasks)]:
        p = run_dir / name
        if p.exists():
            if json.loads(p.read_text()) != obj:
                raise ValueError(f'source/input/protocol/task drift: {name}; use a versioned run')
        else:
            atomic_json(p, obj)
    for path, expected in bindings['files'].items():
        path = Path(path)
        if path.suffix not in {'.py', '.json'} or 'data/' in str(path):
            continue
        relative = path.relative_to(ROOT)
        target = run_dir / 'source_snapshot' / relative
        if not target.exists():
            atomic_bytes(target, path.read_bytes())
        if digest(target) != expected:
            raise ValueError('source snapshot mismatch')


def check_complete(directory, task, binding_hash):
    marker = directory / 'complete.json'
    if not marker.exists():
        return False
    try:
        c = json.loads(marker.read_text())
        if c['task'] != task or c['run_bindings_sha256'] != binding_hash or set(c['output_sha256']) != REQUIRED_OUTPUTS:
            raise ValueError('completion identity/binding/output-set mismatch')
        for name, h in c['output_sha256'].items():
            if digest(directory / name) != h:
                raise ValueError(f'output checksum differs: {name}')
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f'CORRUPT completion {directory}: {error}') from error
    return True


def provider(method, modules, config):
    if method in READOUT_METHODS:
        return modules[1], json.loads((EXPERIMENT / 'protocol_readouts.json').read_text())
    if method in BASELINE_METHODS:
        recipe = json.loads((EXPERIMENT / 'protocol_baselines.json').read_text())
        metadata = json.loads((ROOT / config['descriptor_cache'] / 'descriptor_metadata.json').read_text())
        recipe['descriptor_feature_identity'] = metadata['feature_identity_sha256']
        return modules[2], recipe
    return modules[0], config


def method_features(features, method, rows):
    # Original and A deliberately reject unknown keys; NaN descriptors stay in B.
    return {k: v[rows] for k, v in features.items()
            if k != 'descriptors' or method in BASELINE_METHODS}


def radius(residuals, coverage):
    rank = math.ceil((len(residuals) + 1) * coverage)
    return float(np.sort(residuals)[rank - 1]) if rank <= len(residuals) else math.inf


def run_task(task, config, modules, inputs, features, assignments, subsets, run_dir, binding_hash):
    from nesso_pxr.low_data_contract import assay_weights, score_predictions
    tid = task['task_id']
    directory = run_dir / 'tasks' / tid
    staging = run_dir / 'attempts' / f'{tid}__{time.time_ns()}'
    staging.mkdir(parents=True)
    atomic_json(staging / 'started.json', {**task, 'started_at': now(), 'run_bindings_sha256': binding_hash})
    try:
        local = subsets.loc[subsets.outer_fold.eq(task['outer_fold']) & subsets.draw.eq(task['draw']) & subsets.n_train.eq(task['n_train'])]
        fit = local.loc[local.role.eq('fit'), 'row_index'].to_numpy(dtype=int)
        cal = local.loc[local.role.eq('calibration'), 'row_index'].to_numpy(dtype=int)
        test = assignments.loc[assignments.outer_fold.eq(task['outer_fold']), 'row_index'].to_numpy(dtype=int)
        if not len(fit) or not len(cal) or not len(test) or set(fit)&set(cal) or set(np.r_[fit, cal])&set(test):
            raise ValueError('invalid label partitions')
        if task['n_train'] != -1 and len(local) != task['n_train']:
            raise ValueError('exact-N label budget differs')
        if len(cal) != math.ceil(.2 * len(local)):
            raise ValueError('calibration contract differs')
        groups = assignments.set_index('row_index').chemical_group.reindex(inputs.row_index).to_numpy()
        if set(groups[np.r_[fit, cal]]) & set(groups[test]):
            raise ValueError('chemical groups cross strict test boundary')
        evaluate = np.r_[cal, test]
        train_features = method_features(features, task['method'], fit)
        train_features['groups'] = groups[fit]
        eval_features = method_features(features, task['method'], evaluate)
        module, recipe = provider(task['method'], modules, config)
        recipe = {**recipe, 'threads': 2, 'inner_folds': 3, 'se_floor': .1}
        result = module.fit_predict(task['method'], train_features,
            inputs.iloc[fit].pEC50.to_numpy(dtype=float), inputs.iloc[fit].pEC50_standard_error.to_numpy(dtype=float),
            eval_features, config=recipe, checkpoint_path=ROOT / config['checkpoint'], seeds=tuple(config['seeds']))
        pred = np.asarray(result['pred'], dtype=float)
        spread = np.asarray(result['ensemble_std'], dtype=float)
        seed_pred = np.asarray(result['seed_predictions'], dtype=float)
        if pred.shape != (len(evaluate),) or spread.shape != pred.shape or seed_pred.shape != (3, len(evaluate)):
            raise ValueError('prediction/seed shape differs')
        if not all(np.isfinite(a).all() for a in (pred, spread, seed_pred)):
            raise ValueError('nonfinite point predictions; no dropping/clipping allowed')
        if not np.allclose(seed_pred.mean(axis=0), pred, rtol=0, atol=1e-12):
            raise ValueError('ensemble not mean of declared seed rows')
        payload = pickle.dumps(result['model_state'], protocol=pickle.HIGHEST_PROTOCOL)
        atomic_bytes(staging / 'model_state.pkl', payload)
        replay = module.predict_model(pickle.loads((staging / 'model_state.pkl').read_bytes()), eval_features)
        if not np.array_equal(np.asarray(replay['seed_predictions']), seed_pred):
            raise ValueError('saved-state exact same-batch replay differs')
        buffer = io.BytesIO()
        np.savez_compressed(buffer, row_index=evaluate, predictions=seed_pred, pred=pred, ensemble_std=spread)
        atomic_bytes(staging / 'seed_predictions.npz', buffer.getvalue())
        audit = result.get('fit_audit', {})
        if audit.get('test_labels_accessed') or audit.get('calibration_labels_accessed'):
            raise ValueError('factory reports forbidden label access')
        # Test/calibration outcomes enter only AFTER raw predictions and state are durable.
        ycal = inputs.iloc[cal].pEC50.to_numpy(dtype=float)
        ytest = inputs.iloc[test].pEC50.to_numpy(dtype=float)
        se = inputs.iloc[test].pEC50_standard_error.to_numpy(dtype=float)
        ptest = pred[len(cal):]
        frame = pd.DataFrame(dict(record_id=inputs.iloc[test].record_id.to_numpy(), row_index=test,
            y_true=ytest, y_pred=ptest, assay_se=se, weight=assay_weights(se, .1), ensemble_std=spread[len(cal):],
            **{k: task[k] for k in ['split', 'outer_fold', 'draw', 'n_train', 'method']}))
        metrics = dict(**task, status='complete', completed_at=now(), n_fit=len(fit), n_calibration=len(cal), n_test=len(test),
            **score_predictions(ytest, ptest, se, .1), fit_seconds=result['fit_seconds'],
            n_parameters=result['n_parameters'], selection=result['selection'], fit_audit=audit)
        intervals = {}
        for coverage in config['interval_coverages']:
            key = str(round(coverage * 100))
            r = radius(np.abs(ycal-pred[:len(cal)]), coverage)
            frame[f'lower_{key}'], frame[f'upper_{key}'] = ptest-r, ptest+r
            metrics[f'coverage_{key}'] = float(np.mean(np.abs(ytest-ptest) <= r))
            metrics[f'width_{key}'] = 2*r
            metrics[f'interval_{key}_finite'] = math.isfinite(r)
            intervals[key] = {'radius': r, 'finite': math.isfinite(r)}
        strata = []
        for label, mask in [('lt4', ytest < 4), ('4to5', (ytest >= 4)&(ytest < 5)),
                            ('5to6', (ytest >= 5)&(ytest < 6)), ('ge6', ytest >= 6)]:
            strata.append(dict(stratum=label, n=int(mask.sum()), bias=float((ptest-ytest)[mask].mean()) if mask.any() else None,
                weighted_mae=score_predictions(ytest[mask], ptest[mask], se[mask], .1)['weighted_mae'] if mask.any() else None))
        atomic_csv(staging / 'predictions.csv', frame)
        atomic_json(staging / 'metrics.json', json_safe(metrics))
        atomic_json(staging / 'diagnostics.json', json_safe(dict(predictions=split.summary(ptest),
            min_prediction_record_id=inputs.iloc[test[int(ptest.argmin())]].record_id,
            max_prediction_record_id=inputs.iloc[test[int(ptest.argmax())]].record_id,
            maximum_absolute_prediction=float(np.abs(ptest).max()), potency_strata=strata, intervals=intervals,
            prediction_clipping=False, exact_pickle_replay=True, method_recipe=recipe)))
        atomic_json(staging / 'label_access.json', dict(fit_record_ids=inputs.iloc[fit].record_id.tolist(),
            calibration_record_ids=inputs.iloc[cal].record_id.tolist(), test_record_ids=inputs.iloc[test].record_id.tolist(),
            fit_row_indices=fit.tolist(), calibration_row_indices=cal.tolist(), test_row_indices=test.tolist(),
            acquired_label_count=len(local), test_labels_accessible_to_fit=False, calibration_labels_accessible_to_fit=False,
            strict_pool_test_max=json.loads((OUTPUT / 'audit.json').read_text())['folds'][task['outer_fold']]['measured_pool_test_max'],
            raw_predictions_persisted_before_scoring=True))
        atomic_json(staging / 'complete.json', dict(task=task, run_bindings_sha256=binding_hash,
            completed_at=now(), output_sha256={n: digest(staging / n) for n in sorted(REQUIRED_OUTPUTS)}))
        check_complete(staging, task, binding_hash)
        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            raise ValueError('existing task directory must not be overwritten')
        os.replace(staging, directory)
        return metrics
    except Exception as error:
        atomic_json(staging / 'failure.json', dict(task=task, at=now(), error=str(error),
            traceback=traceback.format_exc(), scientific_evidence=False))
        raise


def verify_panel(run_dir, tasks, binding_hash, replay=False, data=None, modules=None, config=None):
    result = dict(expected=len(tasks), valid=0, failed=0, pending=0, invalid=[], replayed=0)
    for task in tasks:
        directory = run_dir / 'tasks' / task['task_id']
        try:
            valid = check_complete(directory, task, binding_hash)
            if valid:
                if replay:
                    inputs, features, assignments, subsets = data
                    access = json.loads((directory / 'label_access.json').read_text())
                    local = subsets.loc[subsets.outer_fold.eq(task['outer_fold']) & subsets.draw.eq(task['draw']) & subsets.n_train.eq(task['n_train'])]
                    fit = local.loc[local.role.eq('fit'), 'row_index'].to_numpy(dtype=int)
                    cal = local.loc[local.role.eq('calibration'), 'row_index'].to_numpy(dtype=int)
                    test = assignments.loc[assignments.outer_fold.eq(task['outer_fold']), 'row_index'].to_numpy(dtype=int)
                    for role, rows in [('fit', fit), ('calibration', cal), ('test', test)]:
                        if access[f'{role}_row_indices'] != rows.tolist() or access[f'{role}_record_ids'] != inputs.iloc[rows].record_id.tolist():
                            raise ValueError('saved label-role identity differs')
                    evaluate = np.r_[cal, test]
                    module, _ = provider(task['method'], modules, config)
                    state = pickle.loads((directory / 'model_state.pkl').read_bytes())
                    p = module.predict_model(state, method_features(features, task['method'], evaluate))
                    with np.load(directory / 'seed_predictions.npz') as seeds:
                        if not np.array_equal(seeds['row_index'], evaluate) or not np.array_equal(seeds['predictions'], p['seed_predictions']):
                            raise ValueError('saved seed state does not replay')
                    frame = pd.read_csv(directory / 'predictions.csv', dtype={'record_id': str}, keep_default_na=False, float_precision='round_trip')
                    if frame.record_id.tolist() != inputs.iloc[test].record_id.tolist() or not np.array_equal(frame.y_pred, p['pred'][len(cal):]):
                        raise ValueError('saved scored rows differ from raw replay')
                    from nesso_pxr.low_data_contract import score_predictions
                    metrics = json.loads((directory / 'metrics.json').read_text())
                    computed = score_predictions(inputs.iloc[test].pEC50.to_numpy(), p['pred'][len(cal):], inputs.iloc[test].pEC50_standard_error.to_numpy(), .1)
                    if any(metrics[k] != v for k, v in computed.items()):
                        raise ValueError('saved scores differ from recomputation')
                    result['replayed'] += 1
                result['valid'] += 1
            elif (run_dir / 'failures' / (task['task_id'] + '.json')).exists():
                result['failed'] += 1
            else:
                result['pending'] += 1
        except Exception as error:
            result['invalid'].append({'task_id': task['task_id'], 'error': str(error)})
    result['status'] = 'complete' if result['valid'] == result['expected'] else 'incomplete'
    return result


def descriptor_identity(inputs):
    """Pass exactly the structure-only columns accepted by the B cache API."""
    return inputs[['row_index', 'record_id', 'canonical_smiles']].copy()


def load_data(config, modules):
    # Preparation was independently replayed before these outcome columns are loaded.
    original = ROOT / config['original_preparation']
    inputs = pd.read_csv(original / 'inputs.csv', dtype={'record_id': str}, keep_default_na=False, float_precision='round_trip')
    with np.load(original / 'features.npz', allow_pickle=False) as arrays:
        features = {k: arrays[k] for k in arrays.files}
    loaded = modules[2].load_descriptors(ROOT / config['descriptor_cache'], descriptor_identity(inputs))
    descriptors = loaded[0] if isinstance(loaded, tuple) else loaded
    if np.asarray(descriptors).ndim != 2 or len(descriptors) != len(inputs):
        raise ValueError('descriptor loader returned wrong aligned matrix')
    features['descriptors'] = descriptors
    assignments = pd.read_csv(OUTPUT / 'assignments.csv')
    subsets = pd.read_csv(OUTPUT / 'subsets.csv')
    return inputs, features, assignments, subsets


def request_stop(_signum, _frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['dependencies', 'preflight', 'initialize', 'smoke', 'run', 'verify'])
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    config = read_protocol()
    if args.action == 'dependencies':
        try:
            load_dependencies(config)
            print('Dependencies available; run preflight to verify source-bound tests and cache.')
        except RuntimeError as error:
            atomic_json(OUTPUT / 'dependency_status.json', {'status': 'pending', 'at': now(), 'reason': str(error)})
            print(error)
            return 3
        return 0
    if args.action == 'preflight':
        preflight(config)
        return 0
    modules = require_gate(config)
    audit = split.verify_preparation()
    if audit['inner_group_infeasible_tasks']:
        raise ValueError('strict inner group contracts infeasible; inspect explicit preparation audit')
    tasks = task_manifest(audit)
    run_dir = (args.run_dir or (OUTPUT / 'smoke' if args.action == 'smoke' else ROOT / config['run_dir'])).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bindings = build_bindings(config)
        freeze_run(run_dir, bindings, tasks)
        binding_hash = digest(run_dir / 'run_bindings.json')
        data = load_data(config, modules) if args.action != 'initialize' else None
        verification = verify_panel(run_dir, tasks, binding_hash, replay=args.replay, data=data, modules=modules, config=config)
        if verification['invalid']:
            atomic_json(run_dir / 'verification.json', verification)
            raise ValueError('invalid completed artifacts; preserve and investigate, never overwrite')
        if args.action in {'initialize', 'verify'}:
            atomic_json(run_dir / 'verification.json', verification)
            print(json.dumps(verification, indent=2))
            return 0 if args.action == 'initialize' or verification['status'] == 'complete' else 2
        if args.action == 'smoke':
            selected = [t for t in tasks if t['outer_fold'] == 0 and t['draw'] == 0 and t['n_train'] == 25]
        else:
            selected = tasks
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        file_stats = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in bindings['files']}
        def live_guard():
            for p, expected in file_stats.items():
                current = (Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
                if current != expected:
                    raise ValueError(f'bound source/input changed during run: {p}')
            for p, h in bindings['files'].items():
                if Path(p).suffix == '.py' and digest(p) != h:
                    raise ValueError(f'source drift during run: {p}')
        states = {}
        for task in tasks:
            tid = task['task_id']
            states[tid] = 'valid' if check_complete(run_dir / 'tasks' / tid, task, binding_hash) else 'failed' if (run_dir / 'failures' / (tid + '.json')).exists() else 'pending'
        def publish(status, current=None):
            counts = {key: sum(v == key for v in states.values()) for key in ('valid', 'failed', 'pending')}
            atomic_json(run_dir / 'status.json', dict(**counts, expected=len(tasks), runner_status=status, current_task=current,
                        pid=os.getpid(), at=now(), threads=2, uncapped=True, selected_tasks=len(selected)))
        publish('running')
        for task in selected:
            if STOP:
                break
            live_guard()
            if check_complete(run_dir / 'tasks' / task['task_id'], task, binding_hash):
                continue
            failure = run_dir / 'failures' / (task['task_id'] + '.json')
            if failure.exists() and not args.retry_failed:
                continue
            publish('running', task['task_id'])
            print('START ' + task['task_id'], flush=True)
            try:
                run_task(task, config, modules, *data, run_dir, binding_hash)
                states[task['task_id']] = 'valid'
                if failure.exists():
                    atomic_bytes(run_dir / 'failure_history' / f'{task["task_id"]}.{time.time_ns()}.json', failure.read_bytes())
                    failure.unlink()
                print('COMPLETE ' + task['task_id'], flush=True)
            except Exception as error:
                states[task['task_id']] = 'failed'
                atomic_json(failure, dict(task=task, at=now(), error=str(error), traceback=traceback.format_exc(), scientific_evidence=False))
                print('FAILED ' + task['task_id'] + ': ' + str(error), flush=True)
            live_guard()
        if bindings != build_bindings(config):
            raise ValueError('source/input binding drift at final verification')
        final = verify_panel(run_dir, tasks, binding_hash, replay=True, data=data, modules=modules, config=config)
        atomic_json(run_dir / 'verification.json', final)
        publish('stopped' if STOP else 'complete' if final['status'] == 'complete' else 'smoke_finished' if args.action == 'smoke' else 'finished_with_failures')
        print(json.dumps(final, indent=2))
        if args.action == 'smoke':
            smoke_ok = all(check_complete(run_dir / 'tasks' / t['task_id'], t, binding_hash) for t in selected) and not final['invalid']
            atomic_json(run_dir / 'smoke_gate.json', dict(status='pass' if smoke_ok else 'fail', n_expected=len(selected),
                        task_ids=[t['task_id'] for t in selected], run_bindings_sha256=binding_hash, at=now()))
            return 0 if smoke_ok else 1
        return 0 if final['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
