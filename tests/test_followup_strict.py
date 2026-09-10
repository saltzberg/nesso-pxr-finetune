"""Strict separation, frozen preparation and runner regression contracts."""
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rdkit import DataStructs

split = importlib.import_module('experiments.20260906_low_data_followup.code.strict_split')
runner = importlib.import_module('experiments.20260906_low_data_followup.code.run_strict')


def fp(bits):
    value = DataStructs.ExplicitBitVect(2048)
    for bit in bits:
        value.SetBit(bit)
    return value


def test_purge_any_member_and_inclusive_boundary():
    # 7/20 = .35 exactly; similarity to test member zero is irrelevant.
    fps = [fp(range(100, 120)), fp(range(20)), fp(range(7)), fp(range(6)), fp(range(100, 120))]
    eligible, maxima, nearest, reverse = split.exact_purge(fps, [0, 1], [2, 3, 4])
    assert eligible.tolist() == [3]
    assert maxima.tolist() == [.35, .3, 1.]
    assert nearest.tolist() == [1, 1, 0]
    assert reverse.max() == .3
    with pytest.raises(ValueError, match='frozen'):
        split.exact_purge(fps, [0], [2], threshold=.4)
    with pytest.raises(ValueError, match='overlapping'):
        split.exact_purge(fps, [0], [0])


def subset_fixture():
    config = dict(n_folds=2, draws=2, subset_seed=1701, calibration_fraction=.2, inner_folds=3)
    assignments = pd.DataFrame(dict(row_index=np.arange(40), outer_fold=np.repeat([0, 1], 20)))
    records = []
    eligible = {}
    for fold in range(2):
        pool = assignments.loc[assignments.outer_fold.ne(fold), 'row_index'].to_numpy()
        eligible[fold] = pool[::2]
        for draw in range(2):
            order = np.random.default_rng(np.random.SeedSequence([1701, 1, fold, draw])).permutation(pool)
            for budget in [5, 10, -1]:
                n = len(pool) if budget == -1 else budget
                nc = int(np.ceil(.2*n))
                records.append(pd.DataFrame(dict(outer_fold=fold, draw=draw, n_train=budget,
                    row_index=order[:n], role=['fit']*(n-nc)+['calibration']*nc)))
    return assignments, pd.concat(records, ignore_index=True), eligible, config


def test_filtered_orders_nested_exact_n_and_calibration():
    assignments, old, eligible, config = subset_fixture()
    subsets, orders = split.filtered_subsets(assignments, old, eligible, config, [5, 10])
    for (fold, draw), group in subsets.groupby(['outer_fold', 'draw']):
        full = group.loc[group.n_train.eq(-1), 'row_index'].tolist()
        original = old.loc[old.outer_fold.eq(fold)&old.draw.eq(draw)&old.n_train.eq(-1), 'row_index']
        assert full == [r for r in original if r in eligible[fold]]
        assert not set(full) & set(assignments.loc[assignments.outer_fold.eq(fold), 'row_index'])
        for n in [5, 10]:
            block = group.loc[group.n_train.eq(n)]
            assert block.row_index.tolist() == full[:n]
            nc = int(np.ceil(.2*n))
            assert block.role.tolist() == ['fit']*(n-nc)+['calibration']*nc
    assert len(orders) == 40


def test_original_draw_corruption_and_unsupported_budget_fail():
    assignments, old, eligible, config = subset_fixture()
    with pytest.raises(ValueError, match='unsupported'):
        split.filtered_subsets(assignments, old, eligible, config, [11])
    old.loc[0, 'row_index'] = 999
    with pytest.raises(ValueError, match='prefix'):
        split.filtered_subsets(assignments, old, eligible, config, [5])


def test_preparation_hash_and_source_corruption_fail(tmp_path, monkeypatch):
    original = tmp_path/'original'
    original.mkdir()
    split.atomic_json(tmp_path/'audit.json', {'status': 'pass'})
    manifest = dict(sources={'source': 'hash'}, outputs={'audit.json': split.digest(tmp_path/'audit.json')}, original_inputs={})
    split.atomic_json(tmp_path/'preparation_manifest.json', manifest)
    split.atomic_bytes(tmp_path/'preparation_manifest.sha256', split.digest(tmp_path/'preparation_manifest.json').encode())
    monkeypatch.setattr(split, 'source_bindings', lambda: {'source': 'hash'})
    assert split.verify_preparation(tmp_path, original, recompute=False)['status'] == 'pass'
    split.atomic_json(tmp_path/'audit.json', {'status': 'tampered'})
    with pytest.raises(ValueError, match='output hash'):
        split.verify_preparation(tmp_path, original, recompute=False)
    monkeypatch.setattr(split, 'source_bindings', lambda: {'source': 'changed'})
    with pytest.raises(ValueError, match='source drift'):
        split.verify_preparation(tmp_path, original, recompute=False)


def test_task_manifest_exact_unique_matrix():
    tasks = runner.task_manifest({'supported_budgets': [25, 50, 100, 250, 500]})
    assert len(tasks) == 2700
    assert len({t['task_id'] for t in tasks}) == len(tasks)
    assert {t['method'] for t in tasks} == set(runner.METHODS)
    reduced = runner.task_manifest({'supported_budgets': [25]})
    assert len(reduced) == 900
    assert {t['n_train'] for t in reduced} == {25, -1}


def test_completion_requires_exact_outputs_hashes_identity(tmp_path):
    task = {'task_id': 'test'}
    assert not runner.check_complete(tmp_path, task, 'binding')
    for name in runner.REQUIRED_OUTPUTS:
        split.atomic_bytes(tmp_path/name, b'fixture')
    marker = dict(task=task, run_bindings_sha256='binding',
                  output_sha256={n: split.digest(tmp_path/n) for n in runner.REQUIRED_OUTPUTS})
    split.atomic_json(tmp_path/'complete.json', marker)
    assert runner.check_complete(tmp_path, task, 'binding')
    with pytest.raises(ValueError, match='CORRUPT'):
        runner.check_complete(tmp_path, task, 'different')
    split.atomic_bytes(tmp_path/'predictions.csv', b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        runner.check_complete(tmp_path, task, 'binding')


def test_freeze_run_rejects_source_or_task_drift(tmp_path):
    binding = {'files': {}, 'config': {'threads': 2}}
    runner.freeze_run(tmp_path, binding, [{'task_id': 'a'}])
    runner.freeze_run(tmp_path, binding, [{'task_id': 'a'}])
    with pytest.raises(ValueError, match='drift'):
        runner.freeze_run(tmp_path, binding, [{'task_id': 'b'}])


def test_method_feature_routing_and_frozen_provider_grids():
    values = {'member1': np.ones((4, 384)), 'member2': np.ones((4, 384)),
              'descriptors': np.ones((4, 3)), 'morgan': np.ones((4, 2048))}
    for method in runner.METHODS:
        selected = runner.method_features(values, method, [1, 3])
        assert ('descriptors' in selected) == (method in runner.BASELINE_METHODS)
        assert selected['member1'].shape == (2, 384)
    modules = [SimpleNamespace(name=n) for n in ['original', 'readouts', 'baselines']]
    provider, recipe = runner.provider('repr_ridge_stable', modules, {})
    assert provider is modules[1]
    assert recipe['grids']['ridge_alpha'] == [1, 100, 10000, 1000000]
    assert runner.read_protocol()['threads'] == 2


def test_descriptor_identity_adapter_drops_feature_row_and_labels():
    baseline = importlib.import_module(runner.MODULE_PREFIX + 'baselines')
    frame = pd.DataFrame(dict(row_index=[0, 1], record_id=['001', 'NA'],
        canonical_smiles=['C', 'CC'], feature_row=[9, 7], pEC50=[4., 5.]))
    selected = runner.descriptor_identity(frame)
    assert selected.columns.tolist() == ['row_index', 'record_id', 'canonical_smiles']
    assert baseline._identity(selected)['record_ids'] == ['001', 'NA']
    assert 'feature_row' in frame and 'pEC50' in frame


def test_interval_finite_sample_and_infinite_radius():
    assert runner.radius(np.arange(5.), .8) == 4.
    assert np.isinf(runner.radius(np.arange(5.), .9))
