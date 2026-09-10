"""CPU-only engineering checks; fixtures never count as real Nesso execution."""
import copy
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('affinity_pilot', ROOT / 'experiments/20260906_affinity_representation/code/pilot.py')
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def protocol():
    return pilot.read_json(pilot.PROTOCOL)


def test_real_first_eight_identity_replay_without_outcomes():
    p = protocol()
    # Replace all labels with invalid values: identity selection must ignore them.
    tables = [pilot.rows(ROOT / p['inputs'][k]) for k in ('subsets', 'labels', 'manifest', 'assignments')]
    for row in tables[1]:
        row['pEC50'] = 'OUTCOME_NOT_OPENED'
        row['pEC50_standard_error'] = 'OUTCOME_NOT_OPENED'
    selected = pilot.select_records(*tables, p)
    assert [r['record_id'] for r in selected] == p['record_ids']
    assert all(r['role'] == 'fit' and r['modeling_role'] == 'development' for r in selected)
    assert not any('pEC50' in r for r in selected)
    assert len(set(r['seed'] for r in selected)) == 8
    assert [pilot.seed_for(r['record_id'], p) for r in reversed(selected)][::-1] == [r['seed'] for r in selected]
    tables[1][0]['row_index'] = tables[1][1]['row_index']
    with pytest.raises(ValueError, match='duplicate'):
        pilot.select_records(*tables, p)


def test_atomic_packet_resume_integrity_and_foreign_identity(tmp_path):
    record = {'record_id': 'literal_NA_0001', 'seed': 42}
    meta = pilot.publish_packet(tmp_path, 'one', {'data.pt': {'x': torch.arange(4)}},
                               {'binding': 'bound', 'record': record})
    before = {p.name: pilot.digest(p) for p in (tmp_path / 'one').iterdir()}
    assert pilot.check_packet(tmp_path / 'one', 'bound', record) == meta
    assert before == {p.name: pilot.digest(p) for p in (tmp_path / 'one').iterdir()}
    with pytest.raises(ValueError, match='foreign'):
        pilot.check_packet(tmp_path / 'one', 'other', record)
    with pytest.raises(ValueError, match='foreign'):
        pilot.check_packet(tmp_path / 'one', 'bound', {'record_id': 'wrong'})
    with pytest.raises(FileExistsError):
        pilot.publish_packet(tmp_path, 'one', {'data.pt': {}}, {})
    with (tmp_path / 'one/data.pt').open('ab') as f:
        f.write(b'corruption')
    with pytest.raises(ValueError, match='corrupt'):
        pilot.check_packet(tmp_path / 'one', 'bound', record)


def test_failed_atomic_packet_does_not_publish(tmp_path, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError('interrupted serialization')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(RuntimeError):
        pilot.publish_packet(tmp_path, 'one', {'data.pt': {}}, {})
    assert list(tmp_path.iterdir()) == []


def toy():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.esm_proj = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4))
            self.pairformer_stack = torch.nn.Module()
            self.pairformer_stack.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.pairformer_stack.layers[0].transition_z = torch.nn.Module()
            self.pairformer_stack.layers[0].transition_z.fc1 = torch.nn.Linear(4, 4)
            self.affinity_heads = torch.nn.Module()
            self.affinity_heads.to_affinity_pred_value = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
            self.affinity_heads.binary = torch.nn.Linear(4, 1)
            self.affinity_heads.to_affinity_pred_score = torch.nn.Linear(4, 1)
            self.affinity_heads.to_affinity_logits_binary = torch.nn.Linear(4, 1)
            self.affinity_out_mlp = torch.nn.Linear(4, 4)
            self.register_buffer('frozen_buffer', torch.ones(1))
            # Positive weights/input keep the synthetic ReLUs away from zero.
            for param in self.parameters():
                torch.nn.init.constant_(param, 0.15)

        def forward(self, x):
            z = self.pairformer_stack.layers[0].transition_z.fc1(self.esm_proj(x))
            z = self.affinity_out_mlp(z) + self.frozen_buffer
            return {'affinity_repr': z,
                    'affinity_pred_value': self.affinity_heads.to_affinity_pred_value(z),
                    'affinity_logits_binary': self.affinity_heads.to_affinity_logits_binary(z)}
    return torch.nn.ModuleDict({m: Toy() for m in protocol()['members']}).eval()


def test_both_member_matched_heads_freeze_lora_and_exact_state(tmp_path):
    import copy
    p = protocol()
    a = toy()
    b = copy.deepcopy(a)
    pa = pilot.configure(a, 'head_only', p)
    pb = pilot.configure(b, 'head_plus_lora', p)
    head = p['training']['continuous_head']
    assert pilot.tensor_hashes(pa) == pilot.tensor_hashes({k: v for k, v in pb.items() if head in k})
    assert sum(n.endswith('lora_A') for n in pb) == 6
    assert not any(x.training for x in b.modules())
    assert not any('binary' in n for n in pb)
    frozen = pilot.parameter_hashes(b, False)
    assert frozen == pilot.parameter_hashes(a, False)
    for member in p['members']:
        x = torch.randn(3, 4)
        assert torch.equal(a[member].esm_proj(x), b[member].esm_proj(x))
    loss = sum(q.square().sum() for q in pb.values())
    loss.backward()
    assert all(q.grad is not None for q in pb.values())
    assert all(q.grad is None for q in b.parameters() if not q.requires_grad)
    opt = torch.optim.SGD(pb.values(), lr=.01)
    opt.step()
    assert pilot.parameter_hashes(b, False) == frozen
    state = pilot.cpu(pb)
    torch.save(state, tmp_path / 'state.pt')
    reloaded = torch.load(tmp_path / 'state.pt', weights_only=True)
    assert pilot.tensor_hashes(state) == pilot.tensor_hashes(reloaded)


def test_synthetic_loss_conversion_and_weight_normalization():
    p = protocol()
    x, y = torch.tensor(1., requires_grad=True), torch.tensor(3., requires_grad=True)
    output = {
        m: {'affinity_pred_value': v}
        for m, v in zip(p['members'], (x, y), strict=True)
    }
    loss = pilot.weighted_loss(output, 5., 2., 4., p)
    expected = torch.nn.functional.smooth_l1_loss(torch.tensor(4.), torch.tensor(5.), beta=.5) / 2
    assert torch.equal(loss, expected)
    loss.backward()
    assert x.grad.item() == .25 and y.grad.item() == .25


def test_parity_rejects_schema_nonfinite_and_numerical_drift():
    a = {'member': {'value': torch.tensor([1.])}}
    assert pilot.parity(a, a) == 0
    for b in ({'member': {'value': torch.tensor([1.1])}},
              {'member': {'value': torch.tensor([float('nan')])}},
              {'member': {'value': torch.tensor(1.)}}):
        with pytest.raises(AssertionError):
            pilot.parity(a, b)


def test_gpu_gate_fails_before_any_cuda_work():
    with pytest.raises(RuntimeError, match='signoff'):
        pilot.gpu(SimpleNamespace(allow_gpu=False))


FROZEN_IDS = [
    'nesso_53247eafd9ba189c5e7b', 'nesso_0342a4fec815b477aaee',
    'nesso_7aba55ef8d2206f14577', 'nesso_b741e69fc6d4d980e17a',
    'nesso_4df39b1eae65834b2e0f', 'nesso_74c8f3fb680e85ab7be0',
    'nesso_0798c75e7293bd0c3666', 'nesso_e3d0254d85c904604b01',
]


def synthetic_tables():
    p = protocol()
    subsets, labels, manifest, assignments = [], [], [], []
    for index, rid in enumerate(FROZEN_IDS + ['extra_fit_not_selected']):
        role = {k: p['selection'][k] for k in ('split', 'outer_fold', 'draw', 'n_train', 'role')}
        role['row_index'] = str(index)
        subsets.append(role)
        labels.append({'row_index': str(index), 'record_id': rid, 'feature_row': str(index + 100),
                       'canonical_smiles': 'C', 'pEC50': '5.0', 'pEC50_standard_error': '0.2'})
        manifest.append({'record_id': rid, 'feature_row': str(index + 100), 'modeling_role': 'development'})
        assignments.append({'row_index': str(index), 'outer_fold': '1', 'split': 'chemical_cluster'})
    assignments.append({'row_index': '999', 'outer_fold': '0', 'split': 'chemical_cluster'})
    subsets.insert(0, {**subsets[0], 'role': 'calibration', 'row_index': '999'})
    return subsets, labels, manifest, assignments


def test_exact_real_file_prefix_and_literal_protocol_ids():
    p = protocol()
    assert p['record_ids'] == FROZEN_IDS
    tables = [pilot.rows(ROOT / p['inputs'][k]) for k in ('subsets', 'labels', 'manifest', 'assignments')]
    matching = [r for r in tables[0] if (r['split'], r['outer_fold'], r['draw'], r['n_train'], r['role'])
                == ('strict_chemical', '0', '0', '100', 'fit')]
    selected = pilot.select_records(*tables, p)
    assert [r['row_index'] for r in selected] == [r['row_index'] for r in matching[:8]]
    assert [r['record_id'] for r in selected] == FROZEN_IDS
    assert [r['order'] for r in selected] == list(range(8))
    test_rows = {r['row_index'] for r in tables[3] if r['outer_fold'] == '0'}
    assert not test_rows.intersection(r['row_index'] for r in selected)


@pytest.mark.parametrize('mutation, message', [
    ('order', 'first-eight'), ('calibration', 'first-eight'), ('fold', 'first-eight'),
    ('draw', 'first-eight'), ('budget', 'first-eight'), ('split', 'first-eight'),
    ('test', 'outer-test'), ('blind', 'non-development'), ('feature', 'feature-row'),
    ('duplicate_manifest', 'duplicate'), ('assignment_policy', 'assignment policy'),
    ('missing_test', 'missing outer-test'),
])
def test_selection_rejects_identity_and_role_drift(mutation, message):
    subsets, labels, manifest, assignments = synthetic_tables()
    if mutation == 'order':
        subsets[1], subsets[2] = subsets[2], subsets[1]
    elif mutation in ('calibration', 'fold', 'draw', 'budget', 'split'):
        key = {'calibration': 'role', 'fold': 'outer_fold', 'draw': 'draw', 'budget': 'n_train', 'split': 'split'}[mutation]
        subsets[1][key] = 'not_selected'
    elif mutation == 'test':
        assignments[0]['outer_fold'] = '0'
    elif mutation == 'blind':
        manifest[0]['modeling_role'] = 'blinded'
    elif mutation == 'feature':
        labels[0]['feature_row'] = '999'
    elif mutation == 'duplicate_manifest':
        manifest.append(dict(manifest[0]))
    elif mutation == 'assignment_policy':
        assignments[0]['split'] = 'random'
    elif mutation == 'missing_test':
        assignments.pop()
    with pytest.raises(ValueError, match=message):
        pilot.select_records(subsets, labels, manifest, assignments, protocol())


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


@pytest.mark.parametrize('se, target', [('0', '5'), ('-0.0', '5'), ('-1', '5'),
    ('nan', '5'), ('inf', '5'), ('-inf', '5'), ('0.2', 'nan'), ('0.2', 'inf')])
def test_label_policy_rejects_nonpositive_or_nonfinite(tmp_path, se, target):
    p = protocol()
    write_csv(tmp_path / 'stage/snapshot' / p['inputs']['labels'],
              [{'record_id': FROZEN_IDS[0], 'pEC50': target, 'pEC50_standard_error': se}])
    with pytest.raises(ValueError, match='invalid real assay'):
        pilot.label_panel(SimpleNamespace(output=tmp_path),
                          {'protocol': p, 'records': [{'record_id': FROZEN_IDS[0]}]})


def test_real_label_panel_and_weighted_loss_without_fitting(tmp_path):
    p = protocol()
    source = pilot.rows(ROOT / p['inputs']['labels'])
    wanted = {r['record_id']: r for r in source if r['record_id'] in FROZEN_IDS}
    write_csv(tmp_path / 'stage/snapshot' / p['inputs']['labels'], list(wanted.values()))
    labels = pilot.label_panel(SimpleNamespace(output=tmp_path),
                              {'protocol': p, 'records': [{'record_id': r} for r in FROZEN_IDS]})
    assert [r['record_id'] for r in labels] == FROZEN_IDS
    weights = [1 / max(float(wanted[r]['pEC50_standard_error']), 0.1) for r in FROZEN_IDS]
    assert [r['weight'] for r in labels] == weights
    # Synthetic predictions only: no Nesso forward pass or parameter fitting.
    output = {m: {'affinity_pred_value': torch.tensor(v, dtype=torch.float64)}
              for m, v in zip(p['members'], (1.0, 3.0), strict=True)}
    actual = sum(pilot.weighted_loss(output, r['pEC50'], r['weight'], sum(weights), p) for r in labels)
    errors = [abs(4.0 - float(wanted[r]['pEC50'])) for r in FROZEN_IDS]
    expected = sum(
        w * (e * e if e < 0.5 else e - 0.25)
        for w, e in zip(weights, errors, strict=True)
    ) / sum(weights)
    assert actual.item() == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('se, expected', [('0.001', 10.0), ('0.1', 10.0), ('0.2', 5.0)])
def test_label_inverse_se_floor(tmp_path, se, expected):
    p = protocol()
    write_csv(tmp_path / 'stage/snapshot' / p['inputs']['labels'],
              [{'record_id': '0001', 'pEC50': '5', 'pEC50_standard_error': se}])
    result = pilot.label_panel(SimpleNamespace(output=tmp_path),
                              {'protocol': p, 'records': [{'record_id': '0001'}]})
    assert result == [{'record_id': '0001', 'pEC50': 5.0, 'standard_error': float(se), 'weight': expected}]


def test_canonical_hashes_tree_copy_and_tensor_schema():
    assert pilot.canonical({'b': 2, 'a': [1, None]}) == b'{"a":[1,null],"b":2}'
    assert pilot.object_hash({'b': 2, 'a': 1}) == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    assert pilot.object_hash({'a': 1, 'b': 2}) == pilot.object_hash({'b': 2, 'a': 1})
    with pytest.raises(ValueError):
        pilot.canonical({'invalid': float('nan')})
    source = torch.arange(6.0, requires_grad=True).reshape(2, 3)
    nested = {'tuple': (source, [None, True, 3, 1.5, '0001']), 'scalar': torch.tensor(2, dtype=torch.int64)}
    cloned = pilot.cpu(nested)
    assert isinstance(cloned['tuple'], tuple) and isinstance(cloned['tuple'][1], list)
    assert cloned['tuple'][1] == nested['tuple'][1]
    assert not cloned['tuple'][0].requires_grad
    assert cloned['tuple'][0].data_ptr() != source.data_ptr()
    assert torch.equal(cloned['tuple'][0], source)
    with pytest.raises(TypeError, match='unsupported cache'):
        pilot.tree({'bad': {1, 2}}, lambda x: x)
    a = pilot.tensor_hashes({'x': source})
    assert a == pilot.tensor_hashes({'x': source.detach().clone()})
    assert a != pilot.tensor_hashes({'x': source.reshape(3, 2)})
    assert a != pilot.tensor_hashes({'x': source.double()})
    assert a != pilot.tensor_hashes({'x': source + 1})
    noncontiguous = source.T
    assert pilot.tensor_hashes({'x': noncontiguous}) == pilot.tensor_hashes({'x': noncontiguous.contiguous()})
    assert len(pilot.tensor_hashes({'scalar': nested['scalar'], 'bf16': source.bfloat16()})) == 2


@pytest.mark.parametrize('arm', ['head_only', 'head_plus_lora'])
def test_synthetic_two_step_loss_gradients_updates_and_disk_reload(tmp_path, arm):
    p = protocol()
    torch.manual_seed(42)
    base = toy().double()
    model = copy.deepcopy(base)
    x = torch.ones(1, 4, dtype=torch.float64)
    baseline = {m: module(x) for m, module in base.items()}
    params = pilot.configure(model, arm, p)
    assert all(p['training']['continuous_head'] in n or n.endswith(('lora_A', 'lora_B')) for n in params)
    assert not any(module.training for module in model.modules())
    pilot.parity({m: module(x) for m, module in model.items()}, baseline)
    frozen = pilot.parameter_hashes(model, False)
    before = pilot.cpu(params)
    opt = torch.optim.SGD(params.values(), lr=p['training']['learning_rate'])
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        output = {m: module(x) for m, module in model.items()}
        pilot.weighted_loss(output, 2.0, 1.0, 1.0, p).backward()
        for name, param in params.items():
            assert param.grad is not None and torch.isfinite(param.grad).all()
            if step == 0 and name.endswith('lora_A'):
                assert torch.count_nonzero(param.grad) == 0
            else:
                assert param.grad.norm() > 0, name
        assert all(q.grad is None for q in model.parameters() if not q.requires_grad)
        opt.step()
    assert pilot.parameter_hashes(model, False) == frozen
    assert all(not torch.equal(q, before[n]) for n, q in params.items())
    final = {m: module(x) for m, module in model.items()}
    state_path = tmp_path / 'state.pt'
    torch.save(pilot.cpu(params), state_path)
    loaded = torch.load(state_path, map_location='cpu', weights_only=True)
    reloaded = copy.deepcopy(base)
    restored_params = pilot.configure(reloaded, arm, p)
    with torch.no_grad():
        for name, q in restored_params.items():
            q.copy_(loaded[name])
    pilot.parity({m: module(x) for m, module in reloaded.items()}, final)
    assert pilot.parameter_hashes(reloaded, False) == frozen
    assert pilot.parameter_hashes(reloaded, True) == pilot.parameter_hashes(model, True)


def test_head_architecture_guard():
    model = toy()
    model['affinity_module'].affinity_heads.to_affinity_pred_value = torch.nn.Linear(4, 1)
    with pytest.raises(ValueError, match='continuous head architecture'):
        pilot.configure(model, 'head_only', protocol())


@pytest.mark.parametrize('other', [
    {}, {'member': {'wrong': torch.tensor([1.])}},
    {'member': {'value': torch.tensor([1.], dtype=torch.float64)}},
    {'member': {'value': torch.tensor([float('inf')])}},
])
def test_parity_rejects_keys_dtype_and_infinity(other):
    with pytest.raises(AssertionError):
        pilot.parity({'member': {'value': torch.tensor([1.])}}, other)


def test_parity_tolerance_is_explicit():
    left, right = {'v': torch.tensor([1.0])}, {'v': torch.tensor([1.125])}
    assert pilot.parity(left, right, atol=0.125) == 0.125
    with pytest.raises(AssertionError):
        pilot.parity(left, right)


@pytest.mark.parametrize('change', ['extra', 'missing', 'size', 'empty'])
def test_packet_file_contract_rejects_corruption(tmp_path, change):
    path = tmp_path / 'one'
    pilot.publish_packet(tmp_path, 'one', {'data.pt': torch.tensor([1])}, {'binding': 'bound'})
    if change == 'extra':
        (path / 'extra').write_text('not declared')
    elif change == 'missing':
        (path / 'data.pt').unlink()
    else:
        meta = pilot.read_json(path / 'manifest.json')
        if change == 'size':
            meta['files']['data.pt']['bytes'] += 1
        else:
            meta['files'] = {}
        pilot.atomic_json(path / 'manifest.json', meta)
    with pytest.raises(ValueError, match='packet|corrupt'):
        pilot.check_packet(path, 'bound')


@pytest.fixture
def staged_synthetic(tmp_path, monkeypatch):
    """Tiny fake files exercise real stage mechanics, never canonical artifacts."""
    root, output = tmp_path / 'repo', tmp_path / 'run'
    root.mkdir()
    output.mkdir()
    p = protocol()
    code = root / pilot.EXP / 'code/pilot.py'
    files = [code, root / pilot.EXP / 'README.md', root / pilot.EXP / 'IMPLEMENTATION_PLAN.md',
             root / 'tests/test_affinity_representation_pilot.py', root / 'src/nesso_pxr/low_data_adapter.py',
             root / 'scripts/audit_low_data_adapter.py', root / p['inputs']['ccd'],
             root / p['inputs']['checkpoint'] / 'model.safetensors']
    processed = root / p['inputs']['processed']
    for rid in FROZEN_IDS:
        files += [processed / 'records' / f'{rid}.json', processed / 'structures' / f'{rid}.npz',
                  processed / 'rdkit_conformers' / f'{rid}.pkl']
    files.append(processed / 'esm_embeddings/protein.safetensors')
    image = tmp_path / 'installed.py'
    files.append(image)
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic file for stage integrity testing only')
    tables = synthetic_tables()
    for key, records in zip(
        ('subsets', 'labels', 'manifest', 'assignments'), tables, strict=True
    ):
        write_csv(root / p['inputs'][key], records)
    proto = root / pilot.EXP / 'protocol_pilot.json'
    pilot.atomic_json(proto, p)
    monkeypatch.setattr(pilot, 'ROOT', root)
    monkeypatch.setattr(pilot, 'PROTOCOL', proto)
    monkeypatch.setattr(pilot, '__file__', str(code))
    monkeypatch.setattr(pilot, 'image_sources', lambda: {'image/nesso/installed.py': image})
    args = SimpleNamespace(output=output, image_id=p['image_id'])
    lock = pilot.stage(args)
    return args, lock, root, image


def test_stage_replay_is_immutable_and_copies_not_hardlinks(staged_synthetic):
    args, lock, root, _ = staged_synthetic
    before = {str(f.relative_to(args.output)): pilot.digest(f) for f in args.output.rglob('*') if f.is_file()}
    assert pilot.stage(args) == lock
    assert before == {str(f.relative_to(args.output)): pilot.digest(f) for f in args.output.rglob('*') if f.is_file()}
    assert [r['record_id'] for r in lock['records']] == FROZEN_IDS
    assert lock['challenge_label_training'] is False
    logical = str(pilot.EXP / 'README.md')
    assert (args.output / 'stage/snapshot' / logical).stat().st_ino != (root / logical).stat().st_ino
    assert lock['snapshot_bytes'] == sum(info['bytes'] for info in lock['files'].values())


@pytest.mark.parametrize('change', ['live_code', 'live_input', 'snapshot', 'image', 'protocol', 'binding', 'records'])
def test_stage_rejects_source_and_snapshot_drift(staged_synthetic, change):
    args, lock, root, image = staged_synthetic
    if change == 'live_code':
        target = root / pilot.EXP / 'code/pilot.py'
    elif change == 'live_input':
        target = root / lock['protocol']['inputs']['labels']
    elif change == 'snapshot':
        target = args.output / 'stage/snapshot' / pilot.EXP / 'README.md'
    elif change == 'image':
        target = image
    elif change == 'protocol':
        p = copy.deepcopy(lock['protocol'])
        p['training']['learning_rate'] *= 2
        pilot.atomic_json(pilot.PROTOCOL, p)
        target = None
    else:
        meta = copy.deepcopy(lock)
        if change == 'binding':
            meta['binding'] = 'foreign'
        else:
            meta['records'][0]['seed'] += 1
            meta.pop('binding')
            meta['binding'] = pilot.object_hash(meta)
        pilot.atomic_json(args.output / 'stage/manifest.json', meta)
        target = None
    if target is not None:
        target.write_text('changed bytes')
    with pytest.raises(ValueError, match='changed|replay failed'):
        pilot.validate_stage(args)


def test_cli_gpu_stages_fail_before_output_or_cuda(tmp_path):
    for stage in ('capture', 'train', 'verify'):
        args = [sys.executable, str(ROOT / pilot.EXP / 'code/pilot.py'), stage, '--output', str(tmp_path / 'absent')]
        if stage == 'verify':
            args.append('--replay')
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        assert result.returncode == 2
        assert 'requires explicit --allow-gpu' in result.stderr
    assert not (tmp_path / 'absent').exists()
