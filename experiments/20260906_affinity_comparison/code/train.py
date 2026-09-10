#!/usr/bin/env python3
"""Matched both-member adaptation. prepare is CPU-only; run needs one GPU.

Cache packets are immutable trusted local data. Hash on first access in each
single task/verification lifetime, then guard inode/size/mtime and reload tensors
without hashing the large packet each epoch. A new process/task rehashes again.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
EXP = Path('experiments/20260906_affinity_comparison')
PROTOCOL = ROOT / EXP / 'train_protocol.json'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


pilot = module(ROOT / 'experiments/20260906_affinity_representation/code/pilot.py', 'affinity_pilot')
def isolated_apis():
    """Execute exact existing pure APIs without importing pandas/sklearn modules.

    AST-selected function bodies retain original implementations; their source
    files are snapshotted/bound. The pinned inference image has numpy/scipy but
    intentionally has no pandas/sklearn. No dependency installation is needed.
    """
    import ast
    import numpy as np
    from scipy.stats import spearmanr
    namespace = {'np': np, 'math': math, 'spearmanr': spearmanr}
    for path, names in [
        (ROOT / 'src/nesso_pxr/low_data_contract.py',
         {'_vector', 'assay_weights', 'weighted_mae', 'score_predictions'}),
        (ROOT / 'scripts/run_low_data.py', {'conformal_radius'})]:
        parsed = ast.parse(path.read_text())
        nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != names:
            raise ValueError('pure scoring API source contract changed')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{n: namespace[n] for n in
        ('assay_weights', 'weighted_mae', 'score_predictions', 'conformal_radius')})


low = isolated_apis()
digest, object_hash, read_json = pilot.digest, pilot.object_hash, pilot.read_json
atomic_json, rows = pilot.atomic_json, pilot.rows
configure, forward, parameter_hashes = pilot.configure, pilot.forward, pilot.parameter_hashes
MEMBERS = ('affinity_module', 'affinity_module2')


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        pilot.fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def event(output, kind, **data):
    atomic_json(Path(output) / 'events' / f'{time.time_ns()}-{kind}.json',
                {'kind': kind, 'time_ns': time.time_ns(), **data})


def tasks(protocol, smoke=False):
    result = []
    for fold in ([0] if smoke else protocol['folds']):
        for budget in ([6] if smoke else protocol['budgets']):
            for arm in protocol['training']['arms']:
                task = dict(split='pilot_engineering' if smoke else protocol['split'],
                            outer_fold=fold, draw=protocol['draw'], n_train=budget,
                            arm=arm, seed=protocol['training']['seed'], smoke=smoke)
                task['task_id'] = f"{'smoke__' if smoke else ''}fold{fold}__draw0__n{budget}__{arm}__seed42"
                result.append(task)
    if not smoke and len(result) != protocol['expected_fits']:
        raise ValueError('matrix count mismatch')
    return result


def select_roles(task, subsets, assignments, identities, pool_audit):
    """Identity/order-only selector; accepts no outcome table and never uses labels."""
    byrow = {str(r['row_index']): r for r in identities}
    if len(byrow) != len(identities) or len({r['record_id'] for r in identities}) != len(identities):
        raise ValueError('duplicate identity')
    local = [r for r in subsets if r['split'] == 'strict_chemical' and
             int(r['outer_fold']) == task['outer_fold'] and int(r['draw']) == task['draw'] and
             int(r['n_train']) == task['n_train']]
    n = task['n_train']
    ncal = math.ceil(.2 * n)
    if len(local) != n or [r['role'] for r in local] != ['fit']*(n-ncal)+['calibration']*ncal:
        raise ValueError('exact-N / inherited FIT-calibration order mismatch')
    if {r['split'] for r in assignments} != {'chemical_cluster'}:
        raise ValueError('inherited assignment policy mismatch')
    if len(assignments) != len(byrow) or {r['row_index'] for r in assignments} != set(byrow):
        raise ValueError('assignment identity coverage mismatch')
    groups = {r['row_index']: r['chemical_group'] for r in assignments}
    test = [r['row_index'] for r in assignments if int(r['outer_fold']) == task['outer_fold']]
    acquired = [r['row_index'] for r in local]
    if len(set(acquired)) != n or not test or set(acquired) & set(test):
        raise ValueError('duplicate / outer-test overlap')
    if {groups[i] for i in acquired} & {groups[i] for i in test}:
        raise ValueError('chemical group overlap')
    audit = {r['row_index']: r for r in pool_audit if int(r['outer_fold']) == task['outer_fold']}
    for i in acquired:
        a = audit[i]
        if a['record_id'] != byrow[i]['record_id'] or a['eligible'] != 'True' or not float(a['max_test_similarity']) < .35:
            raise ValueError('strict inherited measured purge failed')
    return {role: [byrow[i]['record_id'] for i in indices] for role, indices in
            [('fit', acquired[:-ncal]), ('calibration', acquired[-ncal:]), ('test', test)]}


def cache_api(smoke=False):
    if smoke:
        return SimpleNamespace(
            validate_stage=lambda output: pilot.validate_stage(SimpleNamespace(output=output)),
            load_record_packet=lambda output, lock, rid: pilot.validate_capture(
                SimpleNamespace(output=output), lock, next(r for r in lock['records'] if r['record_id'] == rid)),
            load_model=lambda lock, output: pilot.load_members(
                output / 'stage/snapshot' / lock['protocol']['inputs']['checkpoint'], lock['protocol']))
    return module(ROOT / EXP / 'code/cache.py', 'affinity_cache')


def sources(protocol, smoke):
    names = [str(EXP / 'code/train.py'), str(EXP / 'train_protocol.json'), str(EXP / 'TRAIN.md'),
             'tests/test_affinity_comparison_train.py',
             'experiments/20260906_affinity_representation/code/pilot.py',
             'experiments/20260906_affinity_representation/protocol_pilot.json',
             'experiments/20260906_affinity_representation/EXECUTION.md',
             'src/nesso_pxr/low_data_adapter.py', 'src/nesso_pxr/low_data_contract.py',
             'scripts/run_low_data.py', protocol['labels'], protocol['manifest']]
    if not smoke:
        names += [str(EXP / 'code/cache.py'), 'experiments/20260906_low_data_followup/code/strict_split.py']
        strict = ROOT / protocol['strict_root']
        meta = read_json(strict / 'preparation_manifest.json')
        if digest(strict / 'preparation_manifest.json') != (strict / 'preparation_manifest.sha256').read_text().strip():
            raise ValueError('strict manifest corrupt')
        for name, h in meta['outputs'].items():
            if digest(strict / name) != h:
                raise ValueError('strict output corrupt: ' + name)
        for name, h in meta['sources'].items():
            if digest(ROOT / name) != h:
                raise ValueError('strict source drift')
        original = (ROOT / protocol['labels']).parent
        for name, h in meta['original_inputs'].items():
            if digest(original / name) != h:
                raise ValueError('strict original input drift: ' + name)
        names += [str(Path(protocol['strict_root']) / n) for n in
                  [*meta['outputs'], 'preparation_manifest.json', 'preparation_manifest.sha256']]
    return sorted(set(names))


def prepare(args, api):
    if (args.output / 'stage').exists():
        return validate_run(args, api)
    p = read_json(PROTOCOL)
    cache_lock = api.validate_stage(args.cache)
    if cache_lock['protocol']['image_id'] != p['image_id']:
        raise ValueError('cache image mismatch')
    records = cache_lock['records']
    ids = [r['record_id'] for r in records]
    if len(set(ids)) != len(ids) or len(ids) != (8 if args.smoke else p['development_records']):
        raise ValueError('cache universe count mismatch')
    names = sources(p, args.smoke)
    identities = rows(ROOT / p['strict_root'] / 'identities.csv') if not args.smoke else []
    manifest = {r['record_id']: r for r in rows(ROOT / p['manifest'])}
    if any(manifest[rid]['modeling_role'] != 'development' for rid in ids):
        raise ValueError('non-development cache record')
    if not args.smoke and set(ids) != {r['record_id'] for r in identities}:
        raise ValueError('cache/strict universe differs')
    panel = tasks(p, args.smoke)
    roles = {}
    for task in panel:
        roles[task['task_id']] = ({'fit': ids[:4], 'calibration': ids[4:6], 'test': ids[6:]}
            if args.smoke else select_roles(task, *(rows(ROOT / p['strict_root'] / n) for n in
                ('subsets.csv', 'assignments.csv', 'identities.csv', 'pool_audit.csv'))))
    tmp = Path(tempfile.mkdtemp(prefix='.prepare-', dir=args.output))
    try:
        hashes = {}
        for name in names:
            source, target = ROOT / name, tmp / 'snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = digest(source)
            shutil.copyfile(source, target)
            if digest(target) != h or digest(source) != h:
                raise ValueError('source changed during snapshot')
            hashes[name] = h
        atomic_json(tmp / 'cache_binding.json', cache_lock)
        lock = {'schema': 1, 'protocol': p, 'smoke': args.smoke, 'files': hashes,
                'cache_binding': cache_lock['binding'], 'cache_lock_sha256': digest(tmp / 'cache_binding.json'),
                'tasks': panel, 'roles': roles, 'scientific_fits': 0 if args.smoke else len(panel)}
        lock['binding'] = object_hash(lock)
        atomic_json(tmp / 'manifest.json', lock)
        os.rename(tmp, args.output / 'stage')
        pilot.fsync_dir(args.output)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    event(args.output, 'prepared', binding=lock['binding'], smoke=args.smoke, expected_tasks=len(panel))
    return validate_run(args, api)


def validate_run(args, api):
    stage = args.output / 'stage'
    lock = read_json(stage / 'manifest.json')
    value = dict(lock)
    binding = value.pop('binding')
    if object_hash(value) != binding or lock['protocol'] != read_json(PROTOCOL) or lock['smoke'] != args.smoke:
        raise ValueError('run binding/protocol/smoke drift')
    for name, h in lock['files'].items():
        if digest(stage / 'snapshot' / name) != h or digest(ROOT / name) != h:
            raise ValueError('bound source/input drift: ' + name)
    cache_lock = api.validate_stage(args.cache)
    if cache_lock['binding'] != lock['cache_binding'] or digest(stage / 'cache_binding.json') != lock['cache_lock_sha256']:
        raise ValueError('foreign cache binding')
    if lock['tasks'] != tasks(lock['protocol'], args.smoke):
        raise ValueError('task matrix drift')
    if not args.smoke:
        strict = stage / 'snapshot' / lock['protocol']['strict_root']
        tables = [rows(strict / n) for n in ('subsets.csv', 'assignments.csv', 'identities.csv', 'pool_audit.csv')]
        for task in lock['tasks']:
            if select_roles(task, *tables) != lock['roles'][task['task_id']]:
                raise ValueError('role replay differs')
    return lock, cache_lock


class PacketReader:
    """Per-task validation lifetime; no unbounded tensor cache."""
    def __init__(self, output, lock, api):
        self.output, self.lock, self.api, self.seen = Path(output), lock, api, {}

    def signature(self, rid):
        root = self.output / 'capture' / rid
        return {p.name: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
                for p in root.iterdir() if p.is_file()}

    def __call__(self, rid):
        import torch
        if rid not in self.seen:
            before = self.signature(rid)
            packet = self.api.load_record_packet(self.output, self.lock, rid)
            if before != self.signature(rid):
                raise ValueError('packet mutated during first validation')
            self.seen[rid] = before
            return packet
        if self.signature(rid) != self.seen[rid]:
            raise ValueError('immutable packet changed within task')
        return {m: torch.load(self.output / 'capture' / rid / f'{m}.pt',
                             map_location='cpu', weights_only=True) for m in MEMBERS}


def point(output):
    a, b = [output[m]['affinity_pred_value'].reshape(()) for m in MEMBERS]
    return 6.0 - (a+b)/2.0


def weighted_loss(output, target, weight, weight_sum, protocol):
    return pilot.weighted_loss(output, target, weight, weight_sum, protocol)


def fit_labels(ids, byid):
    """Construct the only label-bearing payload accepted by optimize()."""
    assay_weights = low.assay_weights
    result = []
    for rid in ids:
        row = byid[rid]
        y, se = float(row['pEC50']), float(row['pEC50_standard_error'])
        if not math.isfinite(y):
            raise ValueError('invalid target')
        result.append({'record_id': rid, 'target': y, 'weight': float(assay_weights([se], .1)[0])})
    return result


def checkpoint(parent, epoch, model, params, optimizer, binding, task, audit):
    state = {'parameters': pilot.cpu(params), 'optimizer': pilot.cpu(optimizer.state_dict()),
             'epoch': epoch, 'binding': binding, 'task': task, 'audit': audit,
             'intended_hashes': parameter_hashes(model, True)}
    pilot.publish_packet(parent, f'epoch{epoch:03d}', {'state.pt': state},
                         {'binding': binding, 'task': task, 'epoch': epoch})


def latest_checkpoint(parent, binding, task, max_epochs):
    import torch
    found = sorted(p for p in Path(parent).glob('epoch*') if p.is_dir())
    if not found:
        return None
    expected = [f'epoch{i:03d}' for i in range(len(found))]
    if [p.name for p in found] != expected or len(found) > max_epochs+1:
        raise ValueError('noncontiguous/foreign epoch queue')
    for epoch, path in enumerate(found):
        meta = pilot.check_packet(path, binding)
        if meta['task'] != task or meta['epoch'] != epoch or set(meta['files']) != {'state.pt'}:
            raise ValueError('checkpoint task/epoch mismatch')
    state = torch.load(found[-1] / 'state.pt', map_location='cpu', weights_only=True)
    if state['binding'] != binding or state['task'] != task or state['epoch'] != len(found)-1:
        raise ValueError('checkpoint payload mismatch')
    return state


def restore(model, params, optimizer, state):
    import torch
    if set(params) != set(state['parameters']) or parameter_hashes(model, False) != state['audit']['frozen_before']:
        raise ValueError('restore parameter/frozen mismatch')
    with torch.no_grad():
        for name, p in params.items():
            q = state['parameters'][name]
            if p.shape != q.shape or p.dtype != q.dtype or not torch.isfinite(q).all():
                raise ValueError('invalid state tensor')
            p.copy_(q)
    if parameter_hashes(model, True) != state['intended_hashes']:
        raise ValueError('restored trainable hashes differ')
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])


def optimize(model, params, optimizer, labels, reader, protocol, parent, binding, task, audit,
             start_epoch=0, stop_after_epoch=None, forward_fn=None):
    """No calibration/test labels or evaluation callback cross this boundary."""
    import torch
    forward_fn = forward if forward_fn is None else forward_fn
    weight_sum = sum(r['weight'] for r in labels)
    epochs = protocol['training']['epochs']
    for epoch in range(start_epoch+1, epochs+1):
        if stop_after_epoch is not None and epoch > stop_after_epoch:
            return False
        optimizer.zero_grad(set_to_none=True)
        total, norms = 0.0, {}
        for label in labels:
            out = forward_fn(model, reader(label['record_id']))
            loss = weighted_loss(out, label['target'], label['weight'], weight_sum, protocol)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite weighted loss')
            loss.backward()
            total += float(loss.detach())
            del out, loss
        for name, p in params.items():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise ValueError('missing/nonfinite gradient: ' + name)
            if epoch <= 2:
                norms[name] = float(p.grad.norm())
                if norms[name] == 0 and not (epoch == 1 and name.endswith('lora_A')):
                    raise ValueError('zero intended gradient: ' + name)
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError('frozen gradient')
        optimizer.step()
        audit['history'].append({'epoch': epoch, 'weighted_huber': total, 'gradient_norms': norms})
        checkpoint(parent, epoch, model, params, optimizer, binding, task, audit)
        event(parent.parent.parent.parent, 'epoch', task_id=task['task_id'], epoch=epoch, weighted_huber=total)
    return True


def write_csv(path, records):
    if not records:
        raise ValueError('empty prediction table')
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    atomic_bytes(path, output.getvalue().encode())


def prediction_rows(model, reader, ids, baseline=False):
    import torch
    result = []
    with torch.no_grad():
        for rid in ids:
            packet = reader(rid)
            out = forward(model, packet)
            pilot.parity(out, forward(model, packet, kernels=True), 0.0)
            if baseline:
                pilot.parity(out, {m: packet[m]['native_output'] for m in MEMBERS}, 0.0)
            result.append({'record_id': rid, 'y_pred': float(point(out)),
                           **{m: float(out[m]['affinity_pred_value'].reshape(())) for m in MEMBERS}})
    return result


def score_tables(cal, test):
    import numpy as np
    score_predictions = low.score_predictions
    y, pred, se = [np.array([float(r[k]) for r in test]) for k in ('y_true', 'y_pred', 'assay_se')]
    residuals = np.abs(np.array([float(r['y_true'])-float(r['y_pred']) for r in cal]))
    metrics = {'n_test': len(test), 'n_calibration': len(cal), **score_predictions(y, pred, se, .1)}
    intervals = {}
    for level in (80, 90):
        radius = low.conformal_radius(residuals, level/100)
        intervals[str(level)] = {'radius': radius if math.isfinite(radius) else None,
            'finite': math.isfinite(radius), 'coverage': float(np.mean(np.abs(y-pred) <= radius)),
            'width': 2*radius if math.isfinite(radius) else None}
    metrics['intervals'] = intervals
    metrics['strata'] = []
    for name, mask in [('lt4', y < 4), ('4to5', (y >= 4)&(y < 5)),
                       ('5to6', (y >= 5)&(y < 6)), ('ge6', y >= 6)]:
        metrics['strata'].append({'stratum': name, 'n': int(mask.sum()),
            'scores': score_predictions(y[mask], pred[mask], se[mask], .1) if mask.any() else None})
    return metrics


def scored_rows(raw, byid):
    assay_weights = low.assay_weights
    result = []
    for r in raw:
        label = byid[r['record_id']]
        y, se = float(label['pEC50']), float(label['pEC50_standard_error'])
        result.append({**r, 'y_true': y, 'assay_se': se, 'weight': float(assay_weights([se], .1)[0])})
    return result


def add_intervals(test, metrics):
    for row in test:
        for level, interval in metrics['intervals'].items():
            radius = interval['radius'] if interval['finite'] else math.inf
            row['lower_'+level] = float(row['y_pred'])-radius
            row['upper_'+level] = float(row['y_pred'])+radius
    return test


OUTPUTS = {'baseline.csv', 'raw_predictions.csv', 'calibration.csv', 'test.csv', 'metrics.json',
           'audit.json', 'replay.json'}


def check_complete(directory, task, binding, epochs):
    marker = directory / 'complete.json'
    if not marker.exists():
        return False
    complete = read_json(marker)
    if complete['task'] != task or complete['binding'] != binding or set(complete['files']) != OUTPUTS:
        raise ValueError('foreign/incomplete completion')
    for name, h in complete['files'].items():
        if digest(directory / name) != h:
            raise ValueError('corrupt completion output: ' + name)
    state = latest_checkpoint(directory / 'epochs', binding, task, epochs)
    if state is None or state['epoch'] != epochs or complete['state_sha256'] != digest(directory / 'epochs' / f'epoch{epochs:03d}' / 'state.pt'):
        raise ValueError('incomplete final epoch')
    proof = read_json(directory / 'replay.json')
    if proof != {'binding': binding, 'task': task, 'all_rows': len(rows(directory / 'raw_predictions.csv')),
                 'point_max_abs': 0.0, 'native_parity_max_abs': 0.0, 'scoring_replayed': True}:
        raise ValueError('missing/foreign replay proof')
    if score_tables(rows(directory / 'calibration.csv'), rows(directory / 'test.csv')) != read_json(directory / 'metrics.json'):
        raise ValueError('saved CSV metric replay differs')
    return True


def optimizer_for(params, p):
    import torch
    t = p['training']
    return torch.optim.AdamW(params.values(), lr=t['learning_rate'], weight_decay=t['weight_decay'],
                             betas=(.9, .999), eps=1e-8, amsgrad=False, foreach=False, fused=False)


def run_task(args, lock, cache_lock, api, task):
    import torch
    directory = args.output / 'tasks' / task['task_id']
    p, binding = lock['protocol'], lock['binding']
    if check_complete(directory, task, binding, p['training']['epochs']):
        return 'skipped'
    directory.mkdir(parents=True, exist_ok=True)
    event(args.output, 'task_started', task=task, binding=binding)
    roles = lock['roles'][task['task_id']]
    eval_ids = roles['calibration'] + roles['test']
    reader = PacketReader(args.cache, cache_lock, api)
    state = latest_checkpoint(directory / 'epochs', binding, task, p['training']['epochs'])
    model = api.load_model(cache_lock, args.cache).eval()
    torch.manual_seed(task['seed'])
    if state is None:
        baseline = prediction_rows(model, reader, eval_ids, baseline=True)
        write_csv(directory / 'baseline.csv', baseline)
    params = configure(model, task['arm'], p)
    optimizer = optimizer_for(params, p)
    if state is None:
        zero = prediction_rows(model, reader, eval_ids, baseline=True)
        if zero != baseline:
            raise ValueError('zero-update prediction changed')
        audit = {'frozen_before': parameter_hashes(model, False), 'intended_before': parameter_hashes(model, True),
                 'history': [], 'roles': roles, 'fit_labels_only': True, 'n_acquired': task['n_train'],
                 'baseline_parity_rows': len(eval_ids), 'mode': 'eval with gradients', 'smoke': args.smoke}
        checkpoint(directory / 'epochs', 0, model, params, optimizer, binding, task, audit)
        start = 0
    else:
        restore(model, params, optimizer, state)
        audit, start = state['audit'], state['epoch']
    byid = {r['record_id']: r for r in rows(args.output / 'stage/snapshot' / p['labels'])}
    labels = fit_labels(roles['fit'], byid)
    finished = optimize(model, params, optimizer, labels, reader, p, directory / 'epochs', binding, task,
                        audit, start, args.stop_after_epoch)
    if not finished:
        event(args.output, 'task_paused', task=task, binding=binding)
        return 'paused'
    if parameter_hashes(model, False) != audit['frozen_before']:
        raise ValueError('frozen weights/buffers changed')
    after = parameter_hashes(model, True)
    if any(after[n] == h for n, h in audit['intended_before'].items()):
        raise ValueError('intended parameter unchanged')
    raw = prediction_rows(model, reader, eval_ids)
    write_csv(directory / 'raw_predictions.csv', raw)
    # Only now, with final state and label-free predictions durable, use CAL/TEST labels.
    cal = scored_rows(raw[:len(roles['calibration'])], byid)
    test = scored_rows(raw[len(roles['calibration']):], byid)
    metrics = score_tables(cal, test)
    write_csv(directory / 'calibration.csv', cal)
    write_csv(directory / 'test.csv', add_intervals(test, metrics))
    atomic_json(directory / 'metrics.json', metrics)
    atomic_json(directory / 'audit.json', {**audit, 'frozen_after': parameter_hashes(model, False),
                'intended_after': after, 'native_parity_all_eval_rows': True, 'no_tuning': True})
    del model, params, optimizer
    torch.cuda.empty_cache()
    replay_task(args, lock, cache_lock, api, task)
    atomic_json(directory / 'complete.json', {'binding': binding, 'task': task,
        'files': {name: digest(directory / name) for name in sorted(OUTPUTS)},
        'state_sha256': digest(directory / 'epochs' / f"epoch{p['training']['epochs']:03d}" / 'state.pt')})
    check_complete(directory, task, binding, p['training']['epochs'])
    event(args.output, 'task_completed', task=task, binding=binding)
    return 'completed'


def replay_task(args, lock, cache_lock, api, task, write_proof=True):
    import torch
    directory = args.output / 'tasks' / task['task_id']
    p = lock['protocol']
    state = latest_checkpoint(directory / 'epochs', lock['binding'], task, p['training']['epochs'])
    if state is None or state['epoch'] != p['training']['epochs']:
        raise ValueError('cannot replay incomplete task')
    model = api.load_model(cache_lock, args.cache).eval()
    torch.manual_seed(task['seed'])
    params = configure(model, task['arm'], p)
    restore(model, params, None, state)
    roles = lock['roles'][task['task_id']]
    ids = roles['calibration'] + roles['test']
    actual = prediction_rows(model, PacketReader(args.cache, cache_lock, api), ids)
    saved = rows(directory / 'raw_predictions.csv')
    expected = [{k: (r[k] if k == 'record_id' else float(r[k])) for k in r} for r in saved]
    if actual != expected:
        raise ValueError('fresh saved-state all-row exact replay failed')
    byid = {r['record_id']: r for r in rows(args.output / 'stage/snapshot' / p['labels'])}
    cal = scored_rows(actual[:len(roles['calibration'])], byid)
    test = scored_rows(actual[len(roles['calibration']):], byid)
    metric = score_tables(cal, test)
    test = add_intervals(test, metric)
    for name, records in [('calibration.csv', cal), ('test.csv', test)]:
        stored = rows(directory / name)
        numeric = [{k: (r[k] if k == 'record_id' else float(r[k])) for k in r} for r in stored]
        if numeric != records:
            raise ValueError('CSV identity/label/interval/float roundtrip differs: ' + name)
    if metric != read_json(directory / 'metrics.json') or score_tables(rows(directory / 'calibration.csv'), rows(directory / 'test.csv')) != metric:
        raise ValueError('independent saved-row scoring replay failed')
    if parameter_hashes(model, False) != state['audit']['frozen_before']:
        raise ValueError('replay mutated frozen state')
    proof = {'binding': lock['binding'], 'task': task, 'all_rows': len(ids), 'point_max_abs': 0.0,
             'native_parity_max_abs': 0.0, 'scoring_replayed': True}
    if write_proof:
        atomic_json(directory / 'replay.json', proof)
    elif read_json(directory / 'replay.json') != proof:
        raise ValueError('replay proof differs')
    del model, params
    torch.cuda.empty_cache()


def accounting(args, lock):
    result = {'binding': lock['binding'], 'smoke': args.smoke, 'expected': len(lock['tasks']),
              'completed': [], 'pending': [], 'scientific_fits': 0}
    expected = {t['task_id'] for t in lock['tasks']}
    found = {p.name for p in (args.output / 'tasks').glob('*') if p.is_dir()}
    if found - expected:
        raise ValueError('foreign task directories')
    for task in lock['tasks']:
        complete = check_complete(args.output / 'tasks' / task['task_id'], task, lock['binding'], lock['protocol']['training']['epochs'])
        result['completed' if complete else 'pending'].append(task['task_id'])
    result['scientific_fits'] = 0 if args.smoke else len(result['completed'])
    result['status'] = 'complete' if not result['pending'] else 'incomplete'
    result['failure_events'] = len(list((args.output / 'failures').glob('*.json')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'verify'])
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-gpu', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='distinct 8-record old-pilot engineering run, never scientific fits')
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--budget', type=int, choices=[100, 500])
    parser.add_argument('--task-limit', type=int)
    parser.add_argument('--stop-after-epoch', type=int, help='operational pause only; never declares a short fit complete')
    parser.add_argument('--replay', action='store_true', help='verify: fresh GPU replay of all completed scorer rows')
    args = parser.parse_args()
    args.cache, args.output = args.cache.resolve(), args.output.resolve()
    if args.cache == args.output or args.cache in args.output.parents:
        parser.error('run must be separate from immutable cache')
    if args.task_limit is not None and args.task_limit < 1:
        parser.error('--task-limit must be positive')
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= 40:
        parser.error('--stop-after-epoch must be 1..40')
    if args.smoke and (args.fold is not None or args.budget is not None):
        parser.error('scientific fold/budget selectors do not apply to smoke')
    if args.action == 'run' or args.replay:
        pilot.gpu(args)
    args.output.mkdir(parents=True, exist_ok=True)
    api = cache_api(args.smoke)
    with (args.output / '.train.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        task = None
        try:
            lock, cache_lock = prepare(args, api) if args.action == 'prepare' else validate_run(args, api)
            attempted = 0
            if args.action == 'run':
                for task in lock['tasks']:
                    if args.fold is not None and task['outer_fold'] != args.fold:
                        continue
                    if args.budget is not None and task['n_train'] != args.budget:
                        continue
                    if check_complete(args.output / 'tasks' / task['task_id'], task, lock['binding'], 40):
                        continue
                    if args.task_limit is not None and attempted >= args.task_limit:
                        break
                    attempted += 1
                    run_task(args, lock, cache_lock, api, task)
            result = accounting(args, lock)
            if args.action == 'verify' and args.replay:
                for task in lock['tasks']:
                    if task['task_id'] in result['completed']:
                        replay_task(args, lock, cache_lock, api, task, write_proof=False)
            result['attempted_this_invocation'] = attempted
            print(json.dumps(result, indent=2, allow_nan=False), flush=True)
            if args.action == 'verify' and result['pending']:
                return 2
        except Exception:
            atomic_json(args.output / 'failures' / f'{time.time_ns()}-{args.action}.json',
                {'action': args.action, 'task': task, 'traceback': traceback.format_exc(), 'scientific_evidence': False})
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
