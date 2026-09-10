"""CPU-only matched context controls. Trusted local artifacts; no Nesso import."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib
import io
import json
import math
import os
import pickle
import re
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'experiments/20260906_affinity_comparison'
OLD = ROOT / 'experiments/20260906_low_data_followup'
STRICT = ROOT / 'artifacts/experiments/low_data_followup_20260906/strict_split'
ORIGINAL = ROOT / 'artifacts/experiments/low_data_20260905_cut035_run2/prepared'
PROTOCOL = EXP / 'controls_protocol.json'
MEMBERS = ('affinity_module', 'affinity_module2')
DESCRIPTOR = 'descriptor_lightgbm_rdkit_mordred'
RIDGE = 'repr_ridge_stable'
NATIVE = 'native_continuous'
PREFIX = 'experiments.20260906_low_data_followup.code.'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def csv(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, float_precision='round_trip', **kwargs)


def atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def put_json(path, value):
    atomic(path, (json.dumps(safe(value), sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def put_array(path, **arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic(path, buf.getvalue())


def check_hashes(directory, hashes):
    for name, h in hashes.items():
        if digest(Path(directory) / name) != h:
            raise ValueError(f'checksum mismatch: {directory}/{name}')


def checked_file(path, info):
    if Path(path).stat().st_size != info['bytes'] or digest(path) != info['sha256']:
        raise ValueError(f'corrupt bound bytes: {path}')


def strict_identity(strict=STRICT):
    """Read only identity/role data; validate original immutable strict preparation."""
    strict = Path(strict)
    manifest = strict / 'preparation_manifest.json'
    if digest(manifest) != (strict / 'preparation_manifest.sha256').read_text().split()[0]:
        raise ValueError('strict manifest hash mismatch')
    meta = read_json(manifest)
    check_hashes(strict, meta['outputs'])
    check_hashes(ROOT, meta['sources'])
    identity = csv(strict / 'identities.csv', dtype={'record_id': str})
    assignments = csv(strict / 'assignments.csv')
    subsets = csv(strict / 'subsets.csv')
    if not np.array_equal(identity.row_index, np.arange(len(identity))) or identity.record_id.duplicated().any():
        raise ValueError('strict ordered identities invalid')
    if not np.array_equal(assignments.row_index, identity.row_index):
        raise ValueError('assignment row identity mismatch')
    cells = []
    for fold in range(5):
        for n in (100, 500):
            local = subsets.loc[(subsets.split == 'strict_chemical') & (subsets.outer_fold == fold) & (subsets.draw == 0) & (subsets.n_train == n)]
            roles = {r: local.loc[local.role == r, 'row_index'].to_numpy(int) for r in ('fit', 'calibration')}
            roles['test'] = assignments.loc[assignments.outer_fold == fold, 'row_index'].to_numpy(int)
            fit, cal, test = (roles[r] for r in ('fit', 'calibration', 'test'))
            if len(local) != n or len(cal) != math.ceil(.2*n) or len(fit) + len(cal) != n or len(set(np.r_[fit, cal, test])) != n + len(test):
                raise ValueError('exact-N/disjoint role membership invalid')
            groups = assignments.chemical_group.to_numpy()
            if set(groups[np.r_[fit, cal]]) & set(groups[test]) or len(set(groups[fit])) < 3:
                raise ValueError('strict chemical boundary/3-fold fit infeasible')
            cells.append(dict(outer_fold=fold, draw=0, n_train=n, roles=roles))
    return identity, assignments, cells


def validate_cache(cache, identity):
    """Hash-only host gate. All packets must be present; no Nesso/image import."""
    cache = Path(cache)
    stage = read_json(cache / 'stage/manifest.json')
    core = dict(stage)
    binding = core.pop('binding')
    if object_hash(core) != binding:
        raise ValueError('cache stage binding mismatch')
    required_stage = {'binding', 'binding_namespace', 'challenge_label_training', 'files', 'identity_manifest_sha256', 'identity_sources', 'kind', 'numeric_outcomes_copied', 'protocol', 'records', 'reused', 'schema', 'snapshot_bytes'}
    if set(stage) != required_stage or stage['schema'] != 1 or stage['binding_namespace'] != 'pxr-affinity-comparison-cache-v1':
        raise ValueError('cache stage schema mismatch')
    if set(identity.columns) != {'row_index', 'record_id', 'feature_row', 'canonical_smiles'}:
        raise ValueError('cache accepts strict identity columns only, not labels')
    records = stage['records']
    if len(records) != stage['protocol']['selection']['count'] or identity.record_id.duplicated().any():
        raise ValueError('cache cohort count/duplicate identity mismatch')
    for i, rec in enumerate(records):
        if set(rec) != {'row_index', 'record_id', 'feature_row', 'canonical_smiles', 'order', 'modeling_role', 'seed'}:
            raise ValueError('cache record schema includes missing/unknown/label fields')
        if not re.fullmatch(r'nesso_[0-9a-f]{20}', rec['record_id']):
            raise ValueError('unsafe record ID')
        rng = stage['protocol']['rng']
        seed = int.from_bytes(hashlib.sha256(f"{rng['namespace']}|{rng['base_seed']}|{rec['record_id']}".encode()).digest()[:4], 'big') % 2147483647
        if rec['order'] != i or rec['row_index'] != str(i) or rec['modeling_role'] != 'development' or rec['seed'] != seed:
            raise ValueError('cache record order/role/RNG mismatch')
    if [(str(r['row_index']), r['record_id']) for r in records] != list(zip(identity.row_index.astype(str), identity.record_id, strict=True)):
        raise ValueError('ordered cache/strict identities mismatch')
    if any(str(identity.iloc[i][k]) != rec[k] for i, rec in enumerate(records) for k in ('feature_row', 'canonical_smiles')):
        raise ValueError('cache structure/feature identity mismatch')
    if stage.get('kind') != 'full_development_native_cache' or stage.get('challenge_label_training') is not False or stage.get('numeric_outcomes_copied') is not False:
        raise ValueError('cache native/outcome-blind contract mismatch')
    if stage['protocol'] != read_json(EXP / 'cache_protocol.json'):
        raise ValueError('cache protocol mismatch')
    identity_path = cache / 'stage/identity_manifest.json'
    if digest(identity_path) != stage['identity_manifest_sha256'] or read_json(identity_path) != records:
        raise ValueError('cache identity manifest mismatch')
    for logical, h in stage['identity_sources'].items():
        if digest(ROOT / logical) != h:
            raise ValueError('cache identity source hash mismatch')
    source_hashes = {'stage/manifest.json': digest(cache / 'stage/manifest.json'), 'stage/identity_manifest.json': digest(identity_path)}
    for name, info in stage['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or set(info) != {'sha256', 'bytes'}:
            raise ValueError('cache source file schema/path mismatch')
        p = cache / 'stage/snapshot' / name
        checked_file(p, info)
        source_hashes[str(p.relative_to(cache))] = info['sha256']
    packets = []
    for rec in records:
        rid = rec['record_id']
        if Path(rid).name != rid or rid in ('.', '..'):
            raise ValueError('unsafe record ID')
        directory = cache / 'capture' / rid
        meta = read_json(directory / 'manifest.json')
        if meta['binding'] != binding or meta['record'] != rec or meta.get('challenge_label_training') is not False or meta.get('kind') != 'real_native_both_member_capture':
            raise ValueError('foreign/non-native packet identity or binding')
        if rid in stage.get('reused', {}):
            proof = stage['reused'][rid]
            if meta.get('reused_from') != {k: v for k, v in proof.items() if k != 'manifest'} or meta['files'] != proof['manifest']['files']:
                raise ValueError('reused pilot packet provenance mismatch')
        elif 'reused_from' in meta:
            raise ValueError('undeclared reused packet')
        required = {m + '.pt' for m in MEMBERS} | {'native_prediction.pt'}
        if set(meta['files']) != required or {p.name for p in directory.iterdir()} != required | {'manifest.json'}:
            raise ValueError('capture payload set mismatch')
        for name, info in meta['files'].items():
            checked_file(directory / name, info)
            source_hashes[str((directory / name).relative_to(cache))] = info['sha256']
        source_hashes[str((directory / 'manifest.json').relative_to(cache))] = digest(directory / 'manifest.json')
        packets.append(directory)
    return dict(binding=binding, files=source_hashes, records=records), packets


def build_features(cache, output, identity):
    """Once-through extraction; complete hash validation precedes resume skipping."""
    output = Path(output)
    source, packets = validate_cache(cache, identity)
    target = output / 'fresh_features'
    if target.exists():
        meta = read_json(target / 'manifest.json')
        if meta['cache'] != source or meta['extractor_sha256'] != digest(__file__):
            raise ValueError('feature cache binding/extractor drift')
        if meta.get('outcome_labels_read') is not False or set(meta['outputs']) != {'features.npz', 'native_points.npy', 'identities.csv'}:
            raise ValueError('feature cache output/schema mismatch')
        check_hashes(target, meta['outputs'])
        return target
    output.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.features-', dir=output))
    try:
        vectors = [[], []]
        scalars, points = [], []
        for directory in packets:
            outputs = []
            for i, member in enumerate(MEMBERS):
                packet = torch.load(directory / (member + '.pt'), map_location='cpu', weights_only=True)
                if packet.get('autocast_enabled') is not False or set(packet) != {'kwargs', 'cpu_rng', 'cuda_rng', 'native_output', 'autocast_enabled', 'removed_bookkeeping'}:
                    raise ValueError('native FP32/RNG capture schema mismatch')
                if set(packet['kwargs']) != {'s_inputs', 'z', 'pdistogram', 'feats', 'use_kernels'} or packet['kwargs']['use_kernels'] is not True:
                    raise ValueError('native kwargs schema mismatch')
                for key in ('cpu_rng', 'cuda_rng'):
                    v = packet[key]
                    if not isinstance(v, torch.Tensor) or v.dtype != torch.uint8 or v.ndim != 1 or not v.numel():
                        raise ValueError('native RNG schema mismatch')
                out = packet['native_output']
                if set(out) != {'affinity_repr', 'affinity_pred_value', 'affinity_logits_binary'}:
                    raise ValueError('native output schema mismatch')
                for key, width in [('affinity_repr', 384), ('affinity_pred_value', 1), ('affinity_logits_binary', 1)]:
                    v = out[key]
                    if v.dtype != torch.float32 or v.numel() != width or not torch.isfinite(v).all():
                        raise ValueError('native tensor dtype/384-width/scalar/finiteness mismatch')
                vectors[i].append(out['affinity_repr'].reshape(384).numpy().copy())
                outputs.append(out)
            a, b = (v['affinity_pred_value'].reshape(()) for v in outputs)
            # Match new trainer exactly: torch FP32 arithmetic BEFORE float64 conversion.
            native = torch.load(directory / 'native_prediction.pt', map_location='cpu', weights_only=True)
            if not torch.equal(((a+b)/2).reshape(-1), native['affinity_pred_value'].reshape(-1)):
                raise ValueError('native ensemble parity mismatch')
            points.append(float(6 - (a+b)/2))
            scalars.append([float(v[k].reshape(())) for v in outputs for k in ('affinity_pred_value', 'affinity_logits_binary')])
        put_array(temp / 'features.npz', member1=np.stack(vectors[0]), member2=np.stack(vectors[1]), scalar_features=np.asarray(scalars, dtype=np.float32))
        buf = io.BytesIO()
        np.save(buf, np.asarray(points, dtype=np.float64), allow_pickle=False)
        atomic(temp / 'native_points.npy', buf.getvalue())
        atomic(temp / 'identities.csv', identity[['row_index', 'record_id']].to_csv(index=False).encode())
        put_json(temp / 'manifest.json', dict(schema=1, cache=source, extractor_sha256=digest(__file__), outcome_labels_read=False,
            outputs={n: digest(temp / n) for n in ('features.npz', 'native_points.npy', 'identities.csv')}))
        fsync_dir(temp)
        os.rename(temp, target)
        fsync_dir(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return target


def load_labels(strict, original, identity):
    meta = read_json(Path(strict) / 'preparation_manifest.json')
    p = Path(original) / 'inputs.csv'
    if digest(p) != meta['original_inputs']['inputs.csv']:
        raise ValueError('outcome source hash mismatch')
    frame = csv(p, dtype={'record_id': str})
    if list(zip(frame.row_index, frame.record_id, strict=True)) != list(zip(identity.row_index, identity.record_id, strict=True)):
        raise ValueError('label identity mismatch')
    return frame


def task_id(cell, method):
    if cell['draw'] != 0 or cell['outer_fold'] not in range(5) or method not in (RIDGE, NATIVE, DESCRIPTOR):
        raise ValueError('unsupported control cell')
    return f"strict_chemical__fold{cell['outer_fold']}__draw00__n{cell['n_train']}__{method}"


def historical_sources(panel):
    """Verify every historical source snapshot; relevant artifacts separately gated."""
    panel = Path(panel)
    binding = read_json(panel / 'run_bindings.json')
    sources = {'run_bindings.json': panel / 'run_bindings.json'}
    # Bind the roles and outcome bytes to the HISTORICAL run, not just current preparation.
    relevant = [panel.parent / n for n in ('identities.csv', 'assignments.csv', 'subsets.csv', 'preparation_manifest.json')]
    original = ROOT / binding['config']['original_preparation']
    relevant.append(original / 'inputs.csv')
    used = {str(q.resolve()) for q in source_paths(panel.parent, original, ROOT)
            if q.suffix == '.py' or q.parent == OLD}
    for p in relevant:
        if digest(p) != binding['files'][str(p)]:
            raise ValueError('historical strict/outcome source hash mismatch')
    for absolute, expected in binding['files'].items():
        p = Path(absolute)
        relative = p.relative_to(ROOT)
        snapshot = panel / 'source_snapshot' / relative
        if snapshot.exists():
            if digest(snapshot) != expected:
                raise ValueError(f'historical source snapshot drift: {relative}')
            sources[str(Path('source_snapshot') / relative)] = snapshot
        if str(p) in used:
            if not snapshot.is_file() or digest(p) != expected:
                raise ValueError(f'used historical helper/protocol drift: {relative}')
    return binding, sources


def verify_descriptor(panel, cell, labels, binding=None, descriptors=None):
    """Read-only exact cell verification; NEVER fit. Replays saved descriptor state."""
    from nesso_pxr.low_data_contract import assay_weights, score_predictions
    panel = Path(panel)
    if binding is None:
        binding, _ = historical_sources(panel)
    tid = task_id(cell, DESCRIPTOR)
    directory = panel / 'tasks' / tid
    complete = read_json(directory / 'complete.json')
    task = dict(task_id=tid, split='strict_chemical', outer_fold=cell['outer_fold'], draw=0, n_train=cell['n_train'], method=DESCRIPTOR)
    required = {'started.json', 'predictions.csv', 'metrics.json', 'label_access.json', 'seed_predictions.npz', 'model_state.pkl', 'diagnostics.json'}
    if complete['task'] != task or complete['run_bindings_sha256'] != digest(panel / 'run_bindings.json') or set(complete['output_sha256']) != required:
        raise ValueError('historical descriptor completion/binding mismatch')
    check_hashes(directory, complete['output_sha256'])
    access = read_json(directory / 'label_access.json')
    for role, rows in cell['roles'].items():
        if access[f'{role}_row_indices'] != rows.tolist() or access[f'{role}_record_ids'] != labels.iloc[rows].record_id.tolist():
            raise ValueError('historical descriptor exact role membership mismatch')
    if access['calibration_labels_accessible_to_fit'] or access['test_labels_accessible_to_fit'] or access['acquired_label_count'] != cell['n_train']:
        raise ValueError('historical descriptor label budget/access mismatch')
    cal, test = cell['roles']['calibration'], cell['roles']['test']
    evaluate = np.r_[cal, test]
    frame = csv(directory / 'predictions.csv', dtype={'record_id': str})
    for key, expected in dict(row_index=test, record_id=labels.iloc[test].record_id.to_numpy(), y_true=labels.iloc[test].pEC50.to_numpy(), assay_se=labels.iloc[test].pEC50_standard_error.to_numpy(), weight=assay_weights(labels.iloc[test].pEC50_standard_error.to_numpy(), .1)).items():
        if not np.array_equal(frame[key], expected):
            raise ValueError(f'historical descriptor prediction identity/label/weight mismatch: {key}')
    for key in ('outer_fold', 'draw', 'n_train', 'method', 'split'):
        if not frame[key].eq(task[key]).all():
            raise ValueError('historical descriptor prediction cell mismatch')
    module = importlib.import_module(PREFIX + 'baselines')
    cache = ROOT / binding['config']['descriptor_cache']
    for name in ('descriptors.npz', 'descriptor_metadata.json'):
        if digest(cache / name) != binding['files'][str(cache / name)]:
            raise ValueError('descriptor feature source drift')
    if descriptors is None:
        descriptors, _ = module.load_descriptors(cache, labels[['row_index', 'record_id', 'canonical_smiles']])
    state = pickle.loads((directory / 'model_state.pkl').read_bytes())
    prediction = module.predict_model(state, {'descriptors': descriptors[evaluate]})
    with np.load(directory / 'seed_predictions.npz') as arr:
        if not np.array_equal(arr['row_index'], evaluate) or not np.array_equal(arr['predictions'], prediction['seed_predictions']) or not np.array_equal(arr['pred'], prediction['pred']):
            raise ValueError('historical descriptor saved-state replay mismatch')
    if not np.array_equal(frame.y_pred, prediction['pred'][len(cal):]) or not np.array_equal(frame.ensemble_std, prediction['ensemble_std'][len(cal):]):
        raise ValueError('historical descriptor CSV replay mismatch')
    scores = score_predictions(frame.y_true.to_numpy(), frame.y_pred.to_numpy(), frame.assay_se.to_numpy(), .1)
    saved = read_json(directory / 'metrics.json')
    if any(saved[k] != v for k, v in scores.items()):
        raise ValueError('historical descriptor metric replay mismatch')
    interval_frame, interval_metrics = scores_and_intervals(labels, cal, test, prediction['pred'])
    interval_test = interval_frame.loc[interval_frame.role == 'test']
    for key in ('80', '90'):
        for side in ('lower_', 'upper_'):
            if not np.array_equal(frame[side+key], interval_test[side+key]):
                raise ValueError('historical descriptor interval CSV replay mismatch')
        for metric in ('coverage_', 'width_'):
            if saved[metric+key] != interval_metrics['test'][metric+key]:
                raise ValueError('historical descriptor interval metric replay mismatch')
    return directory, prediction


def source_paths(strict, original, cache):
    paths = [Path(__file__), PROTOCOL, ROOT / 'tests/test_affinity_comparison_controls.py', OLD / 'protocol_readouts.json', OLD / 'protocol_baselines.json', OLD / 'protocol_strict.json']
    paths += [OLD / f'code/{n}.py' for n in ('readouts', 'baselines', 'run_strict', 'strict_split')]
    paths += [ROOT / f'src/nesso_pxr/{n}.py' for n in ('low_data_models', 'low_data_contract', 'low_data_adapter', 'chemistry', 'labels', 'protocol', 'splits', '__init__')]
    paths += [Path(strict) / n for n in ('preparation_manifest.json', 'preparation_manifest.sha256')]
    paths += [Path(strict) / n for n in read_json(Path(strict) / 'preparation_manifest.json')['outputs']]
    paths += [Path(original) / 'inputs.csv', Path(cache) / 'stage/manifest.json', EXP / 'code/cache.py', EXP / 'cache_protocol.json']
    return paths


def freeze_sources(output, strict, original, cache):
    output = Path(output)
    files = {str(p.resolve()): digest(p) for p in source_paths(strict, original, cache)}
    binding = dict(files=files, protocol=read_json(PROTOCOL), feature_manifest_sha256=digest(output / 'fresh_features/manifest.json'),
        historical_binding_sha256=digest(Path(strict) / 'panel/run_bindings.json'), runtime=dict(python=os.sys.version, numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__))
    p = output / 'controls_bindings.json'
    if p.exists() and read_json(p) != binding:
        raise ValueError('controls run source/protocol/input/runtime drift')
    if not p.exists():
        for i, (name, h) in enumerate(files.items()):
            target = output / 'source_snapshot' / f'{i:03d}_{Path(name).name}'
            atomic(target, Path(name).read_bytes())
            if digest(target) != h:
                raise ValueError('source changed during snapshot')
        put_json(p, binding)
    for i, (name, h) in enumerate(files.items()):
        if digest(output / 'source_snapshot' / f'{i:03d}_{Path(name).name}') != h:
            raise ValueError('controls source snapshot mismatch')
    return digest(p)


def scores_and_intervals(labels, cal, test, pred):
    from nesso_pxr.low_data_contract import assay_weights, score_predictions
    result, metrics = [], {}
    ycal = labels.iloc[cal].pEC50.to_numpy(float)
    radii = {}
    for coverage in (.8, .9):
        rank = math.ceil((len(cal)+1)*coverage)
        radii[str(round(coverage*100))] = float(np.sort(np.abs(ycal-pred[:len(cal)]))[rank-1]) if rank <= len(cal) else math.inf
    for role, rows, values in [('calibration', cal, pred[:len(cal)]), ('test', test, pred[len(cal):])]:
        y = labels.iloc[rows].pEC50.to_numpy(float)
        se = labels.iloc[rows].pEC50_standard_error.to_numpy(float)
        frame = pd.DataFrame(dict(row_index=rows, record_id=labels.iloc[rows].record_id.to_numpy(), role=role, y_true=y, y_pred=values, assay_se=se, weight=assay_weights(se, .1)))
        score = score_predictions(y, values, se, .1)
        score.update(raw_mae=score['mae'], rho=score['spearman'])
        score['strata'] = {}
        for name, mask in [('lt4', y < 4), ('4to5', (y >= 4)&(y < 5)), ('5to6', (y >= 5)&(y < 6)), ('ge6', y >= 6)]:
            score['strata'][name] = dict(n=int(mask.sum()), **(score_predictions(y[mask], values[mask], se[mask], .1) if mask.any() else {}))
        for key, radius in radii.items():
            frame['lower_'+key], frame['upper_'+key] = values-radius, values+radius
            score['coverage_'+key] = float(np.mean(np.abs(y-values) <= radius))
            score['width_'+key] = 2*radius
        result.append(frame)
        metrics[role] = score
    return pd.concat(result, ignore_index=True), dict(**metrics, radii=radii, interval_adaptation='within-N calibration labels, interval only; calibration coverage is in-sample descriptive')


def complete_cell(directory, binding_hash):
    marker = Path(directory) / 'complete.json'
    if not marker.exists():
        return False
    meta = read_json(marker)
    required = {'raw_predictions.npz', 'fit_audit.json', 'predictions.csv', 'calibration_predictions.csv', 'test_predictions.csv', 'metrics.json', 'roles.json'}
    if not required <= set(meta['outputs']):
        raise ValueError('incomplete controls output allowlist')
    if Path(directory).name.endswith('__' + RIDGE) and 'model_state.pkl' not in meta['outputs']:
        raise ValueError('ridge saved state missing from bound outputs')
    if meta['controls_bindings_sha256'] != binding_hash:
        raise ValueError('foreign controls cell binding')
    check_hashes(directory, meta['outputs'])
    return True


def run_cell(output, cell, method, features, native, labels, groups, binding_hash, historical=None, verify_only=False):
    if set(features) != {'member1', 'member2'}:
        raise ValueError('control features accept member vectors only, not labels')
    output = Path(output)
    target = output / 'cells' / task_id(cell, method)
    roles = cell['roles']
    fit, cal, test = (roles[r] for r in ('fit', 'calibration', 'test'))
    evaluate = np.r_[cal, test]
    module = importlib.import_module(PREFIX + 'readouts')
    eval_features = {k: features[k][evaluate] for k in ('member1', 'member2')}
    if complete_cell(target, binding_hash):
        arr = np.load(target / 'raw_predictions.npz')
        if not np.array_equal(arr['row_index'], evaluate):
            raise ValueError('control evaluation rows mismatch')
        if method == RIDGE:
            replay = module.predict_model(pickle.loads((target / 'model_state.pkl').read_bytes()), eval_features)['pred']
        elif method == NATIVE:
            replay = native[evaluate]
        else:
            replay = historical[1]['pred']
        if not np.array_equal(arr['pred'], replay):
            raise ValueError('control saved-state prediction replay mismatch')
        expected_frame, expected_metrics = scores_and_intervals(labels, cal, test, replay)
        actual = csv(target / 'predictions.csv', dtype={'record_id': str})
        if not actual.equals(expected_frame) or read_json(target / 'metrics.json') != safe(expected_metrics):
            raise ValueError('control CSV/metric replay mismatch')
        for role in ('calibration', 'test'):
            role_frame = csv(target / f'{role}_predictions.csv', dtype={'record_id': str})
            if not role_frame.equals(expected_frame.loc[expected_frame.role == role].reset_index(drop=True)):
                raise ValueError('role CSV replay mismatch')
        if read_json(target / 'roles.json') != {r: dict(row_index=rows.tolist(), record_id=labels.iloc[rows].record_id.tolist()) for r, rows in roles.items()}:
            raise ValueError('control role replay mismatch')
        return 'verified'
    if verify_only:
        raise ValueError(f'missing completed control cell: {target.name}')
    output.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.cell-', dir=output))
    try:
        if method == RIDGE:
            train = {k: features[k][fit] for k in ('member1', 'member2')}
            train['groups'] = groups[fit]
            recipe = read_json(OLD / 'protocol_readouts.json')
            if recipe['inner_folds'] != 3 or recipe['grids']['ridge_alpha'] != [1, 100, 1e4, 1e6]:
                raise ValueError('fixed ridge selection contract changed')
            fitted = module.fit_predict(RIDGE, train, labels.iloc[fit].pEC50.to_numpy(float), labels.iloc[fit].pEC50_standard_error.to_numpy(float), eval_features, config=recipe, checkpoint_path=None, seeds=(42,))
            atomic(temp / 'model_state.pkl', pickle.dumps(fitted['model_state'], protocol=pickle.HIGHEST_PROTOCOL))
            pred = module.predict_model(pickle.loads((temp / 'model_state.pkl').read_bytes()), eval_features)['pred']
            if not np.array_equal(pred, fitted['pred']):
                raise ValueError('fresh ridge disk replay differs')
            put_json(temp / 'fit_audit.json', dict(selection=fitted['selection'], audit=fitted['fit_audit'], new_final_ridge_fits=1))
        elif method == NATIVE:
            pred = native[evaluate]
            put_json(temp / 'fit_audit.json', dict(new_fits=0, point_labels_used=0, new_rng_native_point_anchor=True))
        else:
            source, prediction = historical
            shutil.copytree(source, temp / 'historical_payload')
            check_hashes(temp / 'historical_payload', read_json(source / 'complete.json')['output_sha256'])
            pred = prediction['pred']
            put_json(temp / 'fit_audit.json', dict(new_fits=0, reused_descriptor_cell=True, source=str(source), source_complete_sha256=digest(source / 'complete.json')))
        put_array(temp / 'raw_predictions.npz', row_index=evaluate, pred=pred)
        # No cal/test scoring until raw predictions (and fit state) are durable.
        frame, metrics = scores_and_intervals(labels, cal, test, pred)
        atomic(temp / 'predictions.csv', frame.to_csv(index=False).encode())
        for role in ('calibration', 'test'):
            atomic(temp / f'{role}_predictions.csv', frame.loc[frame.role == role].to_csv(index=False).encode())
        put_json(temp / 'metrics.json', metrics)
        put_json(temp / 'roles.json', {r: dict(row_index=rows.tolist(), record_id=labels.iloc[rows].record_id.tolist()) for r, rows in roles.items()})
        put_json(temp / 'complete.json', dict(controls_bindings_sha256=binding_hash, outputs={str(p.relative_to(temp)): digest(p) for p in sorted(temp.rglob('*')) if p.is_file()}))
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(temp, target)
        fsync_dir(target.parent)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return run_cell(output, cell, method, features, native, labels, groups, binding_hash, historical, verify_only=True)


def workflow(action, cache, output, strict=STRICT, original=ORIGINAL):
    identity, assignments, cells = strict_identity(strict)
    if action == 'verify-history':
        labels = load_labels(strict, original, identity)
        binding, _ = historical_sources(Path(strict) / 'panel')
        proof = []
        for cell in cells:
            directory, _ = verify_descriptor(Path(strict) / 'panel', cell, labels, binding)
            proof.append(dict(task_id=directory.name, complete_sha256=digest(directory / 'complete.json')))
        return dict(verified_descriptor_cells=len(cells), new_fits=0, read_only=True, verified_tasks=proof)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'controls.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = build_features(cache, output, identity)
        if action == 'build':
            return dict(features=str(target), count=len(identity), outcomes_read=False)
        binding_hash = freeze_sources(output, strict, original, cache)
        labels = load_labels(strict, original, identity)
        binding, sources = historical_sources(Path(strict) / 'panel')
        for name, source in sources.items():
            dest = output / 'historical_provenance' / name
            if not dest.exists():
                atomic(dest, source.read_bytes())
            if digest(dest) != digest(source):
                raise ValueError('copied historical provenance mismatch')
        with np.load(target / 'features.npz') as data:
            features = {k: data[k] for k in ('member1', 'member2')}
        native = np.load(target / 'native_points.npy', allow_pickle=False)
        historical_cells = {task_id(cell, DESCRIPTOR): verify_descriptor(Path(strict) / 'panel', cell, labels, binding) for cell in cells}
        count = 0
        for cell in cells:
            historical = historical_cells[task_id(cell, DESCRIPTOR)]
            for method in (RIDGE, NATIVE, DESCRIPTOR):
                run_cell(output, cell, method, features, native, labels, assignments.chemical_group.to_numpy(), binding_hash, historical, verify_only=action == 'verify')
                count += 1
        summary = dict(verified_control_evaluation_cells=count, new_final_ridge_fits=10, native_evaluation_cells=10, reused_descriptor_evaluation_cells=10, neural_new_fits_in_separate_trainer=20, heldout_fresh=False)
        if action != 'verify':
            put_json(output / 'controls_complete.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'run', 'verify', 'verify-history'])
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--strict', type=Path, default=STRICT)
    parser.add_argument('--original', type=Path, default=ORIGINAL)
    args = parser.parse_args()
    if args.action != 'verify-history' and (args.cache is None or args.output is None):
        parser.error('--cache and --output required')
    torch.set_num_threads(2)
    print(json.dumps(workflow(args.action, args.cache, args.output, args.strict, args.original), indent=2))


if __name__ == '__main__':
    main()
