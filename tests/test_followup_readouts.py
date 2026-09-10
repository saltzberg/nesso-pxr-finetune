"""Stable readout regression, fold leakage, replay and runner integrity tests."""
import importlib
import json
import pickle

import numpy as np
import pytest

MODULE = 'experiments.20260906_low_data_followup.code.readouts'
r = importlib.import_module(MODULE)


def fixture():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(12, 768))
    x[:, 0] = 1
    x[:, 1] = 1 + np.arange(12) * 1e-12
    features = {'member1': x[:, :384], 'member2': x[:, 384:], 'groups': np.arange(12)}
    y = 4 + x[:, 2] * .3
    se = np.linspace(.1, .5, 12)
    config = dict(se_floor=.1, inner_folds=3, epochs=2, batch_size=64,
                  grids=dict(ridge_alpha=[1., 100.], neural_learning_rate=[.001]))
    return features, y, se, config


def test_stable_scale_adjacent_constants():
    x = np.array([[1., 1., 0.], [1., 1. + 1e-12, 2.]])
    w = np.array([1., 3.])
    s = r.fit_preprocessing(x, w, True)
    mean = np.average(x, axis=0, weights=w)
    var = np.average((x - mean) ** 2, axis=0, weights=w)
    np.testing.assert_allclose(s['mean'], mean)
    np.testing.assert_allclose(s['scale'], np.maximum.reduce([np.sqrt(var), np.full(3, .1*np.sqrt(var.mean())), np.full(3, 1e-8)]))
    assert s['scale'][0] == s['scale'][1]
    transformed = r.transform(x + np.array([1., 1., 0.]), s)
    assert np.isfinite(transformed).all() and np.max(np.abs(transformed)) < 30


def test_zero_spread_and_no_clipping():
    s = r.fit_preprocessing(np.ones((2, 3)), np.ones(2), True)
    np.testing.assert_array_equal(s['scale'], np.full(3, 1e-8))
    np.testing.assert_array_equal(r.transform(np.ones((2, 3)), s), np.zeros((2, 3)))
    assert r.transform(np.full((1, 3), 2.), s)[0, 0] == 1e8


@pytest.mark.parametrize('method', r.METHODS)
def test_every_method_replay_and_finite(method):
    f, y, se, c = fixture()
    result = r.fit_predict(method, f, y, se, f, config=c, checkpoint_path=None)
    replay = r.predict_model(pickle.loads(pickle.dumps(result['model_state'])), f)
    np.testing.assert_array_equal(result['seed_predictions'], replay['seed_predictions'])
    assert result['seed_predictions'].shape == (3, 12)
    assert np.isfinite(result['pred']).all()
    assert result['fit_audit']['serialization_verified']
    assert result['n_parameters'] == (769 if 'ridge' in method else 98561)


@pytest.mark.parametrize('method', ['repr_ridge_stable', 'repr_mlp_stable_centered'])
def test_instrument_inner_preprocessing_and_centering(monkeypatch, method):
    f, y, se, c = fixture()
    calls = []
    original = r.fit_preprocessing
    def watch(x, w, scaled):
        calls.append(x.copy())
        return original(x, w, scaled)
    monkeypatch.setattr(r, 'fit_preprocessing', watch)
    result = r.fit_predict(method, f, y, se, f, config=c, checkpoint_path=None, seeds=(42,))
    x = np.column_stack([f['member1'], f['member2']])
    fold_rows = [np.asarray(a['train_indices']) for a in result['selection']['folds']]
    for seen in calls:
        assert any(np.array_equal(seen, x[a]) for a in fold_rows) or np.array_equal(seen, x)
    assert sum(len(seen) < len(x) for seen in calls) >= 3
    audits = result['fit_audit']['preprocessing_fits']
    for a in audits:
        indices = np.asarray(a['train_indices'])
        if 'centered' in method:
            expected = np.average(y[indices], weights=1/np.maximum(se[indices], .1))
            assert a['target_offset'] == expected
        else:
            assert a['target_offset'] == 0


def test_evaluation_features_cannot_change_fit():
    f, y, se, c = fixture()
    altered = {k: v.copy() for k, v in f.items()}
    altered['member1'] += 1e4
    a = r.fit_predict('repr_mlp_stable_centered', f, y, se, f, config=c, checkpoint_path=None)
    b = r.fit_predict('repr_mlp_stable_centered', f, y, se, altered, config=c, checkpoint_path=None)
    assert a['selection'] == b['selection']
    np.testing.assert_array_equal(a['pred'], r.predict_model(b['model_state'], f)['pred'])
    assert np.max(np.abs(b['pred'])) > 10  # no post-hoc output clipping


def test_tiny_fold_and_labels_rejected():
    f, y, se, c = fixture()
    f['groups'][:] = 0
    with pytest.raises(ValueError):
        r.fit_predict(r.METHODS[0], f, y, se, f, config=c, checkpoint_path=None)
    f['pEC50'] = y
    with pytest.raises(ValueError, match='unknown'):
        r.fit_predict(r.METHODS[0], f, y, se, f, config=c, checkpoint_path=None)


def test_runner_complete_marker_not_vacuous(tmp_path):
    runner = importlib.import_module('experiments.20260906_low_data_followup.code.run_readouts')
    (tmp_path / 'complete.json').write_text(json.dumps({'output_sha256': {}}))
    assert not runner.complete_task(tmp_path)


def test_runner_task_manifest():
    runner = importlib.import_module('experiments.20260906_low_data_followup.code.run_readouts')
    config = json.loads(runner.PROTOCOL.read_text())
    tasks = runner.planned_tasks(config)
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 3600
    from collections import Counter
    assert Counter(t['method'] for t in tasks) == {m: 600 for m in r.METHODS}
    pilot = [t for t in tasks if t['n_train'] == 25 and t['outer_fold'] == t['draw'] == 0]
    assert len(pilot) == 12


@pytest.mark.parametrize('method', r.METHODS)
def test_torch_disk_replay_without_checkpoint(tmp_path, method):
    import torch
    f, y, se, c = fixture()
    result = r.fit_predict(method, f, y, se, f, config=c,
                           checkpoint_path=tmp_path/'nonexistent.pt')
    path = tmp_path/'state.pt'
    torch.save(result['model_state'], path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    np.testing.assert_array_equal(result['seed_predictions'], r.predict_model(state, f)['seed_predictions'])
    if method in r.RIDGE_METHODS:
        for seed_pred in result['seed_predictions']:
            np.testing.assert_array_equal(seed_pred, result['seed_predictions'][0])
        np.testing.assert_allclose(result['ensemble_std'], 0, rtol=0, atol=1e-14)
    assert result['fit_audit']['weight_decay'] == (.01 if method.endswith('_regularized') else 0.)


def valid_marker(directory):
    runner = importlib.import_module('experiments.20260906_low_data_followup.code.run_readouts')
    directory.mkdir(parents=True, exist_ok=True)
    for name in runner.REQUIRED_OUTPUTS:
        (directory/name).write_text('{}')
    replay = dict(exact_seed_replay=True, exact_csv_replay=True, exact_metric_replay=True,
                  reference_identity_roles_match=True, model_sha256=runner.digest(directory/'model_state.pt'))
    (directory/'replay.json').write_text(json.dumps(replay))
    task = {'task_id': directory.name}
    marker = dict(schema_version=2, task_id=directory.name, task=task, binding_sha256='bound',
                  output_sha256={n: runner.digest(directory/n) for n in runner.REQUIRED_OUTPUTS})
    (directory/'complete.json').write_text(json.dumps(marker))
    return runner, task, marker


@pytest.mark.parametrize('damage', ['payload', 'binding', 'task', 'missing', 'csv_replay', 'metric_replay', 'model_hash', 'path_traversal'])
def test_completion_rejects_corruption_and_foreign_artifacts(tmp_path, damage):
    runner, task, marker = valid_marker(tmp_path/'cell')
    d = tmp_path/'cell'
    assert runner.complete_task(d, 'bound', task)
    if damage == 'payload':
        (d/'predictions.csv').write_text('corrupt')
    elif damage == 'binding':
        marker['binding_sha256'] = 'foreign'
    elif damage == 'task':
        marker['task'] = {'task_id': 'foreign'}
    elif damage == 'missing':
        (d/'model_state.pt').unlink()
    elif damage == 'path_traversal':
        marker['output_sha256']['../other'] = 'bad'
    else:
        replay = json.loads((d/'replay.json').read_text())
        key = {'csv_replay': 'exact_csv_replay', 'metric_replay': 'exact_metric_replay', 'model_hash': 'model_sha256'}[damage]
        replay[key] = False if damage != 'model_hash' else 'foreign'
        (d/'replay.json').write_text(json.dumps(replay))
        marker['output_sha256']['replay.json'] = runner.digest(d/'replay.json')
    (d/'complete.json').write_text(json.dumps(marker))
    assert not runner.complete_task(d, 'bound', task)


def test_reconcile_resume_keeps_failures_and_incomplete_visible(tmp_path):
    runner, complete, _ = valid_marker(tmp_path/'tasks'/'complete')
    tasks = [complete] + [{'task_id': name} for name in ('failed', 'invalid', 'partial', 'absent')]
    for name in ('failed', 'invalid', 'partial'):
        (tmp_path/'tasks'/name).mkdir()
    (tmp_path/'tasks'/'failed'/'failure.json').write_text('{}')
    (tmp_path/'tasks'/'invalid'/'complete.json').write_text('{}')
    (tmp_path/'tasks'/'partial'/'started.json').write_text('{}')
    states, failures = runner.reconcile(tmp_path, tasks, 'bound')
    assert states == dict(complete='complete', failed='failed', invalid='invalid', partial='pending', absent='pending')
    assert {f['task_id'] for f in failures} == {'failed', 'invalid'}


def test_exact_inner_fit_call_order_and_weights(monkeypatch):
    f, y, se, c = fixture()
    calls = []
    original = r.fit_preprocessing
    def watch(x, w, scaled):
        calls.append((x.copy(), w.copy()))
        return original(x, w, scaled)
    monkeypatch.setattr(r, 'fit_preprocessing', watch)
    result = r.fit_predict('repr_ridge_stable', f, y, se, f, config=c, checkpoint_path=None)
    x = np.column_stack([f['member1'], f['member2']])
    folds = [np.asarray(a['train_indices']) for a in result['selection']['folds']]
    expected = folds * len(c['grids']['ridge_alpha']) + [np.arange(len(y))]
    assert len(calls) == len(expected)
    for (seen, w), ix in zip(calls, expected, strict=True):
        np.testing.assert_array_equal(seen, x[ix])
        np.testing.assert_array_equal(w, 1 / np.maximum(se[ix], .1))
