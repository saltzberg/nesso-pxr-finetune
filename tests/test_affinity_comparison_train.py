"""CPU-only synthetic fixtures, not Nesso scientific results."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison_train', ROOT / 'experiments/20260906_affinity_comparison/code/train.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
P = t.read_json(t.PROTOCOL)


def tables(n=100):
    identities = [{'row_index': str(i), 'record_id': f'r{i}'} for i in range(n+3)]
    subsets = [{'split': 'strict_chemical', 'outer_fold': '0', 'draw': '0', 'n_train': str(n),
                'row_index': str(i), 'role': 'fit' if i < int(.8*n) else 'calibration'} for i in range(n)]
    assignments = [{'split': 'chemical_cluster', 'outer_fold': '0' if i >= n else '1',
                    'row_index': str(i), 'chemical_group': str(i)} for i in range(n+3)]
    audit = [{'outer_fold': '0', 'row_index': str(i), 'record_id': f'r{i}', 'eligible': 'True',
              'max_test_similarity': '.3499999'} for i in range(n)]
    return subsets, assignments, identities, audit


@pytest.mark.parametrize('n', [100, 500])
def test_exact_n_disjoint_and_label_blind(n):
    task = {'outer_fold': 0, 'draw': 0, 'n_train': n}
    data = tables(n)
    roles = t.select_roles(task, *data)
    assert len(roles['fit']) == int(.8*n)
    assert len(roles['calibration']) == int(.2*n)
    assert not set(roles['fit']) & set(roles['calibration'])
    assert not (set(roles['fit']) | set(roles['calibration'])) & set(roles['test'])
    for row in data[2]:
        row.update(pEC50='POISON', pEC50_standard_error='POISON')
    assert roles == t.select_roles(task, *data)


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'role', 'overlap', 'group', 'purge', 'identity'])
def test_partition_rejects(defect):
    data = tables()
    if defect == 'missing': data[0].pop()
    if defect == 'duplicate': data[0][1]['row_index'] = '0'
    if defect == 'role': data[0][0]['role'] = 'calibration'
    if defect == 'overlap': data[1][0]['outer_fold'] = '0'
    if defect == 'group': data[1][100]['chemical_group'] = '0'
    if defect == 'purge': data[3][0]['max_test_similarity'] = '.35'
    if defect == 'identity': data[3][0]['record_id'] = 'foreign'
    with pytest.raises(ValueError):
        t.select_roles({'outer_fold': 0, 'draw': 0, 'n_train': 100}, *data)


def test_real_inherited_matrix_cpu_only():
    strict = ROOT / P['strict_root']
    if not strict.exists(): pytest.skip('canonical strict artifacts unavailable')
    data = [t.rows(strict / n) for n in ('subsets.csv', 'assignments.csv', 'identities.csv', 'pool_audit.csv')]
    panel = t.tasks(P)
    assert len(panel) == 20
    selected = [t.select_roles(task, *data) for task in panel]
    assert sum(len(r['fit'])*40 for r in selected) == 192000
    for i in range(0, 20, 2): assert selected[i] == selected[i+1]


def test_recipe_and_engineering_smoke_are_distinct():
    assert P['training']['epochs'] == 40
    assert (P['training']['learning_rate'], P['training']['weight_decay']) == (.0003, .0001)
    assert P['training']['rank'] == P['training']['alpha'] == 4
    assert {q['n_train'] for q in t.tasks(P)} == {100, 500}
    assert len(t.tasks(P, True)) == 2
    assert all(q['smoke'] and q['task_id'].startswith('smoke__') for q in t.tasks(P, True))


def test_weighted_ensemble_huber_and_both_member_gradients():
    a, b = torch.tensor(2., requires_grad=True), torch.tensor(4., requires_grad=True)
    out = {
        m: {'affinity_pred_value': v}
        for m, v in zip(t.MEMBERS, (a, b), strict=True)
    }
    actual = t.weighted_loss(out, 4., 10., 12., P)
    expected = torch.nn.functional.smooth_l1_loss(6-(a+b)/2, torch.tensor(4.), beta=.5)*10/12
    assert torch.equal(actual, expected)
    actual.backward()
    assert a.grad == b.grad != 0
    assert t.low.assay_weights([.01, .2]).tolist() == [10., 5.]


@pytest.mark.parametrize('se', [0, -1, float('nan'), float('inf')])
def test_invalid_se_fails(se):
    with pytest.raises(ValueError):
        t.fit_labels(['r'], {'r': {'pEC50': 4, 'pEC50_standard_error': se}})


def test_fit_labels_never_accesses_other_rows():
    class Guard(dict):
        def __getitem__(self, key):
            assert key == 'FIT'
            return super().__getitem__(key)
    assert len(t.fit_labels(['FIT'], Guard(FIT={'pEC50': 4, 'pEC50_standard_error': .2}))) == 1


def test_inherited_metric_and_conformal_api_parity():
    from nesso_pxr.low_data_contract import score_predictions
    y, pred, se = np.array([3., 4., 5., 6.]), np.array([3.1, 5., 4.8, 7.]), np.array([.1, .2, .3, .4])
    assert t.low.score_predictions(y, pred, se) == score_predictions(y, pred, se)
    assert t.low.conformal_radius(np.arange(20.), .8) == 16.
    assert t.low.conformal_radius(np.arange(20.), .9) == 18.
    assert np.isinf(t.low.conformal_radius(np.array([1., 2.]), .9))
    with pytest.raises(ValueError): t.low.conformal_radius([], .8)


class Tiny(nn.Module):
    """Synthetic two-member parameterized network; no native Nesso claims."""
    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.tensor(.7), requires_grad=False)
        self.register_buffer('constant', torch.tensor(.2))
        self.head = nn.Parameter(torch.tensor(.4))


def tiny_model():
    return nn.ModuleDict({m: Tiny() for m in t.MEMBERS}).eval()


def tiny_forward(model, packet, kernels=False):
    assert not any(m.training for m in model.modules())
    return {m: {'affinity_pred_value': q.head * packet + q.frozen + q.constant} for m, q in model.items()}


def tiny_setup():
    model = tiny_model()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    opt = t.optimizer_for(params, P)
    audit = {'frozen_before': t.parameter_hashes(model, False),
             'intended_before': t.parameter_hashes(model, True), 'history': []}
    return model, params, opt, audit


def run_tiny(parent, stop=None, state=None):
    model, params, opt, audit = tiny_setup()
    task, binding = t.tasks(P)[0], 'test-binding'
    start = 0
    if state is None:
        t.checkpoint(parent, 0, model, params, opt, binding, task, audit)
    else:
        t.restore(model, params, opt, state)
        audit, start = state['audit'], state['epoch']
    labels = [{'record_id': 'FIT1', 'target': 4.5, 'weight': 10.},
              {'record_id': 'FIT2', 'target': 4., 'weight': 5.}]
    def reader(rid):
        assert rid in ('FIT1', 'FIT2'), 'label isolation breached'
        return torch.tensor(1. if rid == 'FIT1' else 2.)
    done = t.optimize(model, params, opt, labels, reader, P, parent, binding, task, audit,
                       start, stop, tiny_forward)
    return model, opt, done


def test_atomic_resume_matches_uninterrupted_adamw(tmp_path):
    clean, clean_opt, done = run_tiny(tmp_path/'clean'/'epochs')
    assert done
    _, _, done = run_tiny(tmp_path/'resume'/'epochs', stop=2)
    assert not done
    state = t.latest_checkpoint(tmp_path/'resume'/'epochs', 'test-binding', t.tasks(P)[0], 40)
    assert state['epoch'] == 2
    resumed, resumed_opt, done = run_tiny(tmp_path/'resume'/'epochs', state=state)
    assert done
    assert t.parameter_hashes(clean, True) == t.parameter_hashes(resumed, True)
    assert t.parameter_hashes(clean, False) == t.parameter_hashes(resumed, False)
    for k, v in clean_opt.state_dict()['state'].items():
        for name, value in v.items():
            assert torch.equal(value, resumed_opt.state_dict()['state'][k][name])


@pytest.mark.parametrize('defect', ['bytes', 'binding', 'gap', 'frozen'])
def test_checkpoint_corruption_fails_closed(tmp_path, defect):
    run_tiny(tmp_path/'epochs', stop=2)
    task, binding = t.tasks(P)[0], 'test-binding'
    if defect == 'bytes':
        with (tmp_path/'epochs/epoch002/state.pt').open('ab') as f: f.write(b'corrupt')
    if defect == 'binding': binding = 'foreign'
    if defect == 'gap': (tmp_path/'epochs/epoch001').rename(tmp_path/'epochs/epoch007')
    if defect == 'frozen':
        state = t.latest_checkpoint(tmp_path/'epochs', binding, task, 40)
        model, params, opt, _ = tiny_setup()
        with torch.no_grad(): model[t.MEMBERS[0]].constant.add_(1)
        with pytest.raises(ValueError): t.restore(model, params, opt, state)
    else:
        with pytest.raises(ValueError): t.latest_checkpoint(tmp_path/'epochs', binding, task, 40)


def test_incomplete_temporary_epoch_not_committed(tmp_path):
    run_tiny(tmp_path/'epochs', stop=2)
    (tmp_path/'epochs/.epoch003-crash').mkdir()
    assert t.latest_checkpoint(tmp_path/'epochs', 'test-binding', t.tasks(P)[0], 40)['epoch'] == 2


def test_packet_reader_bounds_memory_and_checks_mutation(tmp_path):
    root = tmp_path/'capture/r'
    root.mkdir(parents=True)
    for m in t.MEMBERS: torch.save({'value': torch.tensor(1.)}, root/f'{m}.pt')
    (root/'manifest.json').write_text('{}')
    calls = []
    def load(output, lock, rid):
        calls.append(rid)
        return {m: torch.load(root/f'{m}.pt', weights_only=True) for m in t.MEMBERS}
    reader = t.PacketReader(tmp_path, {}, SimpleNamespace(load_record_packet=load))
    reader('r'); reader('r')
    assert calls == ['r']
    assert all(not isinstance(v, torch.Tensor) for v in reader.seen['r'].values())
    with (root/'affinity_module.pt').open('ab') as f: f.write(b'x')
    with pytest.raises(ValueError): reader('r')


def test_real_serialization_scorer_roundtrip(tmp_path):
    raw = [{'record_id': f'r{i}', 'y_pred': float(i)/7,
            'affinity_module': float(i), 'affinity_module2': float(i)} for i in range(24)]
    labels = {r['record_id']: {'pEC50': r['y_pred']+.123456789, 'pEC50_standard_error': .2} for r in raw}
    cal, test = t.scored_rows(raw[:20], labels), t.scored_rows(raw[20:], labels)
    metrics = t.score_tables(cal, test)
    t.write_csv(tmp_path/'cal.csv', cal)
    t.write_csv(tmp_path/'test.csv', t.add_intervals(test, metrics))
    assert t.score_tables(t.rows(tmp_path/'cal.csv'), t.rows(tmp_path/'test.csv')) == metrics
    assert len(metrics['strata']) == 4
    assert metrics['intervals']['90']['finite']


def test_no_pandas_sklearn_required_for_import():
    import subprocess, sys
    code = "import sys,runpy; sys.modules['pandas']=None; sys.modules['sklearn']=None; runpy.run_path(sys.argv[1],run_name='import_test')"
    out = subprocess.run([sys.executable, '-c', code, str(ROOT/'experiments/20260906_affinity_comparison/code/train.py')], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_cpu_cli_help():
    import subprocess, sys
    out = subprocess.run([sys.executable, str(ROOT/'experiments/20260906_affinity_comparison/code/train.py'), '--help'], capture_output=True, text=True)
    assert out.returncode == 0
    assert '--smoke' in out.stdout and '--stop-after-epoch' in out.stdout
