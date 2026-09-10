"""Outcome-blind exact all-test-member Morgan purge and immutable preparation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs, rdBase
from nesso_pxr.chemistry import (
    bemis_murcko_scaffold, morgan_fingerprints, scaffold_group_key,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / 'experiments/20260906_low_data_followup'
ORIGINAL = ROOT / 'artifacts/experiments/low_data_20260905_cut035_run2/prepared'
OUTPUT = ROOT / 'artifacts/experiments/low_data_followup_20260906/strict_split'
THRESHOLD = 0.35
BUDGETS = [25, 50, 100, 250, 500]
IDENTITY_COLUMNS = ['row_index', 'record_id', 'feature_row', 'canonical_smiles']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def atomic_csv(path, frame):
    atomic_bytes(path, frame.to_csv(index=False).encode())


def checked_original(original):
    """Hash all original bytes; parse only structure/identity/split columns."""
    original = Path(original)
    manifest = original / 'preparation_manifest.json'
    if digest(manifest) != (original / 'preparation_manifest.sha256').read_text().split()[0]:
        raise ValueError('original manifest hash mismatch')
    meta = json.loads(manifest.read_text())
    for name, expected in meta['artifact_sha256'].items():
        if digest(original / name) != expected:
            raise ValueError(f'original artifact hash mismatch: {name}')
    identities = pd.read_csv(original / 'inputs.csv', usecols=IDENTITY_COLUMNS,
                             dtype={'record_id': str, 'canonical_smiles': str}, keep_default_na=False)
    if not np.array_equal(identities.row_index, np.arange(len(identities))):
        raise ValueError('identity order mismatch')
    for key in IDENTITY_COLUMNS[1:]:
        if identities[key].duplicated().any() or identities[key].eq('').any():
            raise ValueError(f'invalid identity mapping: {key}')
    assignments = pd.read_csv(original / 'assignments.csv')
    assignments = assignments.loc[assignments.split.eq('chemical_cluster')].reset_index(drop=True)
    if len(assignments) != len(identities) or not np.array_equal(assignments.row_index, identities.row_index):
        raise ValueError('original chemical assignment identity order mismatch')
    config = meta['config']
    if (config['morgan_radius'], config['fingerprint_bits'], config['n_folds'], config['draws']) != (2, 2048, 5, 10):
        raise ValueError('original contract differs')
    if set(assignments.outer_fold) != set(range(5)):
        raise ValueError('original outer fold coverage differs')
    if assignments.groupby('chemical_group').outer_fold.nunique().max() != 1:
        raise ValueError('original chemical groups cross test folds')
    original_subsets = pd.read_csv(original / 'subsets.csv')
    original_subsets = original_subsets.loc[original_subsets.split.eq('chemical_cluster')]
    return identities, assignments, original_subsets, meta


def summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'n': 0, 'minimum': None, 'maximum': None, 'mean': None, 'quantiles': None}
    return {'n': len(a), 'minimum': float(a.min()), 'maximum': float(a.max()),
            'mean': float(a.mean()), 'quantiles': dict(zip(
                ['0', '.05', '.25', '.5', '.75', '.95', '1'],
                np.quantile(a, [0, .05, .25, .5, .75, .95, 1]).tolist()))}


def exact_purge(fps, test, pool, threshold=THRESHOLD):
    """Every pair is measured with RDKit double precision; equality is removed."""
    if threshold != THRESHOLD:
        raise ValueError('strict threshold is frozen at .35')
    test = np.asarray(test, dtype=int)
    pool = np.asarray(pool, dtype=int)
    if not len(test) or not len(pool) or set(test) & set(pool):
        raise ValueError('empty or overlapping partition')
    sim = np.asarray([DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in test]) for i in pool])
    maxima = sim.max(axis=1)
    nearest = test[sim.argmax(axis=1)]
    mask = maxima < threshold
    eligible = pool[mask]
    # Independent direction/recalculation, not a claimed cutoff from grouping.
    reverse = np.asarray([max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in eligible]))
                          for i in test]) if len(eligible) else np.array([])
    if len(reverse) and (reverse.max() >= threshold or reverse.max() != maxima[mask].max()):
        raise ValueError('strict measured separation failed')
    return eligible, maxima, nearest, reverse


def filtered_subsets(assignments, original_subsets, eligible_by_fold, config, budgets):
    frames, orders = [], []
    for fold in range(config['n_folds']):
        pool = np.sort(assignments.loc[assignments.outer_fold.ne(fold), 'row_index'].to_numpy())
        eligible = set(eligible_by_fold[fold])
        if not eligible <= set(pool):
            raise ValueError('eligible rows outside original outer pool')
        for draw in range(config['draws']):
            old = original_subsets.loc[original_subsets.outer_fold.eq(fold) & original_subsets.draw.eq(draw)]
            full = old.loc[old.n_train.eq(-1), 'row_index'].to_numpy()
            expected = np.random.default_rng(np.random.SeedSequence([config['subset_seed'], 1, fold, draw])).permutation(pool)
            if not np.array_equal(full, expected):
                raise ValueError('original full draw order does not replay')
            for budget, block in old.groupby('n_train', sort=False):
                n = len(pool) if budget == -1 else budget
                ncal = math.ceil(config['calibration_fraction'] * n)
                if not np.array_equal(block.row_index, expected[:n]):
                    raise ValueError('original prefix order mismatch')
                if block.role.tolist() != ['fit'] * (n-ncal) + ['calibration'] * ncal:
                    raise ValueError('original calibration role mismatch')
            filtered = np.asarray([i for i in full if i in eligible], dtype=int)
            orders.extend(dict(outer_fold=fold, draw=draw, rank=rank, row_index=int(i),
                               original_rank=int(np.flatnonzero(full == i)[0]))
                          for rank, i in enumerate(filtered))
            for budget in budgets + [-1]:
                n = len(filtered) if budget == -1 else budget
                if n > len(filtered):
                    raise ValueError('requested unsupported budget')
                ncal = math.ceil(config['calibration_fraction'] * n)
                if ncal >= n or n - ncal < config['inner_folds']:
                    raise ValueError('eligible full pool cannot support fit/calibration contract')
                frames.append(pd.DataFrame(dict(split='strict_chemical', outer_fold=fold, draw=draw,
                    n_train=budget, row_index=filtered[:n],
                    role=['fit'] * (n-ncal) + ['calibration'] * ncal)))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(orders)


def build_tables(original):
    identities, assignments, original_subsets, meta = checked_original(original)
    fps = morgan_fingerprints(identities.canonical_smiles.tolist(), radius=2, n_bits=2048)
    with np.load(Path(original) / 'features.npz', allow_pickle=False) as data:
        cached = data['morgan']
    regenerated = np.asarray([np.asarray(fp) for fp in fps], dtype=cached.dtype)
    if not np.array_equal(cached, regenerated):
        raise ValueError('regenerated Morgan fingerprints differ from original feature rows')
    scaffolds = np.asarray([scaffold_group_key(s, bemis_murcko_scaffold(s)) for s in identities.canonical_smiles])
    groups = assignments.chemical_group.to_numpy()
    pools, records, test_records, folds = {}, [], [], []
    for fold in range(meta['config']['n_folds']):
        test = assignments.loc[assignments.outer_fold.eq(fold), 'row_index'].to_numpy()
        pool = assignments.loc[assignments.outer_fold.ne(fold), 'row_index'].to_numpy()
        eligible, maxima, nearest, reverse = exact_purge(fps, test, pool)
        pools[fold] = eligible
        ids = set(identities.iloc[eligible].record_id) & set(identities.iloc[test].record_id)
        smiles_overlap = set(identities.iloc[eligible].canonical_smiles) & set(identities.iloc[test].canonical_smiles)
        if ids or smiles_overlap:
            raise ValueError('strict pool/test identity overlap')
        for row, sim, nn in zip(pool, maxima, nearest, strict=True):
            records.append(dict(outer_fold=fold, row_index=int(row),
                record_id=identities.iloc[row].record_id, canonical_smiles=identities.iloc[row].canonical_smiles,
                eligible=bool(sim < THRESHOLD), max_test_similarity=float(sim), nearest_test_row_index=int(nn),
                nearest_test_record_id=identities.iloc[nn].record_id,
                reason='retained_max_lt_0.35' if sim < THRESHOLD else 'purged_similarity_ge_0.35_to_test'))
        for i, row in enumerate(test):
            test_records.append(dict(outer_fold=fold, row_index=int(row), record_id=identities.iloc[row].record_id,
                                     nearest_eligible_similarity=float(reverse[i]) if len(reverse) else None))
        folds.append(dict(outer_fold=fold, n_test=len(test), n_original_pool=len(pool), n_eligible=len(eligible),
            n_removed=len(pool)-len(eligible), identity_overlap=len(ids), canonical_overlap=len(smiles_overlap),
            test_unchanged=True, measured_pool_test_max=float(reverse.max()) if len(reverse) else None,
            original_pool_nearest_test=summary(maxima), retained_pool_nearest_test=summary(maxima[maxima < THRESHOLD]),
            removed_pool_nearest_test=summary(maxima[maxima >= THRESHOLD]), test_nearest_eligible=summary(reverse),
            chemical_groups_original=len(set(groups[pool])), chemical_groups_retained=len(set(groups[eligible])),
            scaffolds_original=len(set(scaffolds[pool])), scaffolds_retained=len(set(scaffolds[eligible])),
            test_scaffolds=len(set(scaffolds[test])),
            test_compounds_with_eligible_scaffold=int(np.isin(scaffolds[test], scaffolds[eligible]).sum()),
            pool_retention_fraction=float(len(eligible)/len(pool)),
            supported_budgets=[b for b in BUDGETS if b <= len(eligible)],
            excluded_budgets=[b for b in BUDGETS if b > len(eligible)]))
    supported = [b for b in BUDGETS if all(b <= len(p) for p in pools.values())]
    subsets, orders = filtered_subsets(assignments, original_subsets, pools, meta['config'], supported)
    group_feasibility = []
    for (fold, draw, budget), block in subsets.groupby(['outer_fold', 'draw', 'n_train']):
        fit = block.loc[block.role.eq('fit'), 'row_index'].to_numpy()
        group_feasibility.append(dict(outer_fold=int(fold), draw=int(draw), n_train=int(budget),
            n_fit=len(fit), n_calibration=int(block.role.eq('calibration').sum()),
            n_fit_chemical_groups=len(set(groups[fit])), inner_3fold_feasible=len(set(groups[fit])) >= 3))
    audit = dict(status='pass', outcome_blind=True, outcome_columns_read=[], outcome_scores_read=False,
        threshold=THRESHOLD, predicate='purge >= 0.35 to ANY original chemical TEST member',
        fingerprint={'radius': 2, 'bits': 2048, 'rdkit': rdBase.rdkitVersion, 'cached_bitwise_replay': True},
        n_identities=len(identities), n_folds=5, draws=10, supported_budgets=supported,
        excluded_budgets=[b for b in BUDGETS if b not in supported], include_eligible_full=True,
        all_folds_support_n500=500 in supported, folds=folds,
        inner_group_infeasible_tasks=[r for r in group_feasibility if not r['inner_3fold_feasible']],
        n_subset_tasks=len(group_feasibility), n_subset_rows=len(subsets),
        coverage_scope='Chemical support only; no outcome/potency-based coverage used for preparation.',
        interpretation='Retrospective same fixed development TEST rows; purged train support, not new untouched confirmation.')
    tables = {'identities.csv': identities, 'assignments.csv': assignments, 'pool_audit.csv': pd.DataFrame(records),
              'test_nearest_similarity.csv': pd.DataFrame(test_records), 'subsets.csv': subsets,
              'filtered_draw_orders.csv': orders, 'inner_feasibility.csv': pd.DataFrame(group_feasibility)}
    tables['removed.csv'] = tables['pool_audit.csv'].loc[~tables['pool_audit.csv'].eligible].reset_index(drop=True)
    return tables, audit, meta


def source_bindings():
    return {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT / 'src/nesso_pxr/chemistry.py']}


def verify_preparation(output=OUTPUT, original=ORIGINAL, recompute=True):
    output, original = Path(output), Path(original)
    manifest = json.loads((output / 'preparation_manifest.json').read_text())
    if digest(output / 'preparation_manifest.json') != (output / 'preparation_manifest.sha256').read_text().strip():
        raise ValueError('strict manifest hash mismatch')
    if manifest['sources'] != source_bindings():
        raise ValueError('strict preparation source drift')
    for name, h in manifest['outputs'].items():
        if digest(output / name) != h:
            raise ValueError(f'strict output hash mismatch: {name}')
    for name, h in manifest['original_inputs'].items():
        if digest(original / name) != h:
            raise ValueError(f'original input drift: {name}')
    audit = json.loads((output / 'audit.json').read_text())
    if recompute:
        tables, expected_audit, _ = build_tables(original)
        if audit != expected_audit:
            raise ValueError('structure-only audit does not replay')
        for name, frame in tables.items():
            if (output / name).read_bytes() != frame.to_csv(index=False).encode():
                raise ValueError(f'strict semantic/order replay failed: {name}')
    return audit


def prepare(output=OUTPUT, original=ORIGINAL):
    output, original = Path(output), Path(original)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'preparation_manifest.json').exists():
            return verify_preparation(output, original)
        if any(output.glob('*.csv')):
            raise ValueError('incomplete preparation exists; preserve it and use a versioned output')
        sources = source_bindings()
        tables, audit, meta = build_tables(original)
        for name, frame in tables.items():
            atomic_csv(output / name, frame)
        atomic_json(output / 'audit.json', audit)
        for path, h in sources.items():
            atomic_bytes(output / 'source_snapshot' / path, (ROOT / path).read_bytes())
            if digest(output / 'source_snapshot' / path) != h:
                raise ValueError('source changed while preparing')
        original_inputs = {name: digest(original / name) for name in
                           list(meta['artifact_sha256']) + ['preparation_manifest.json', 'preparation_manifest.sha256']}
        if any(original_inputs[n] != h for n, h in meta['artifact_sha256'].items()):
            raise ValueError('original changed while preparing')
        manifest = dict(schema_version=1, sources=sources, original_path=str(original.resolve()),
            original_inputs=original_inputs, outcome_blind=True,
            outputs={name: digest(output / name) for name in list(tables) + ['audit.json']})
        atomic_json(output / 'preparation_manifest.json', manifest)
        atomic_bytes(output / 'preparation_manifest.sha256', (digest(output / 'preparation_manifest.json') + '\n').encode())
        return verify_preparation(output, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'verify'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--original', type=Path, default=ORIGINAL)
    args = parser.parse_args()
    result = prepare(args.output, args.original) if args.action == 'prepare' else verify_preparation(args.output, args.original)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
