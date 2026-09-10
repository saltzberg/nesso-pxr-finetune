"""CPU checks use real frozen cohort/pilot; no GPU or canonical cache writes."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('affinity_cache', ROOT / 'experiments/20260906_affinity_comparison/code/cache.py')
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


def protocol():
    return cache.pilot.read_json(cache.PROTOCOL)


def test_real_3344_identity_order_and_no_outcomes():
    p = protocol()
    records = cache.live_records(p)
    assert len(records) == 3344
    assert [r['row_index'] for r in records] == list(map(str, range(3344)))
    assert [r['order'] for r in records] == list(range(3344))
    assert all(r['modeling_role'] == 'development' for r in records)
    assert all(set(r) == {*cache.IDENTITY, 'order'} for r in records)
    assert len({r['record_id'] for r in records}) == 3344
    assert all(r['seed'] == cache.pilot.seed_for(r['record_id'], p) for r in records)


@pytest.mark.parametrize('change', ['order', 'duplicate', 'role', 'feature', 'smiles', 'count'])
def test_real_identity_mutation_rejected(change):
    p = protocol()
    inputs = cache.identity_rows(ROOT / p['inputs']['cohort'], ('row_index', 'record_id', 'feature_row', 'canonical_smiles'))
    manifest = cache.identity_rows(ROOT / p['inputs']['manifest'], ('record_id', 'feature_row', 'canonical_smiles', 'modeling_role'))
    target = next(r for r in manifest if r['record_id'] == inputs[0]['record_id'])
    if change == 'order':
        inputs[0], inputs[1] = inputs[1], inputs[0]
    elif change == 'duplicate':
        inputs[1]['record_id'] = inputs[0]['record_id']
    elif change == 'role':
        target['modeling_role'] = 'lockbox'
    elif change == 'feature':
        target['feature_row'] = 'wrong'
    elif change == 'smiles':
        target['canonical_smiles'] = 'wrong'
    else:
        inputs.pop()
    with pytest.raises(ValueError):
        cache.select_records(inputs, manifest, p)


def real_reuse_lock():
    p = protocol()
    origin = ROOT / p['pilot_root']
    old = cache.checked_lock(origin / 'stage/manifest.json')
    records = cache.live_records(p)
    files = old['files']
    reused = cache.pilot_parity(p, records, files)
    return {'binding': 'distinct-full-cache-test-binding', 'protocol': p, 'records': records, 'reused': reused}


@pytest.fixture(scope='module')
def reuse_lock():
    return real_reuse_lock()


def test_real_pilot_source_checkpoint_semantics_parity(reuse_lock):
    assert len(reuse_lock['reused']) == 8
    assert reuse_lock['binding'] != protocol()['pilot_binding']
    old = cache.checked_lock(ROOT / protocol()['pilot_root'] / 'stage/manifest.json')
    for logical in (str(cache.OLD / 'code/pilot.py'), str(cache.OLD / 'protocol_pilot.json')):
        assert cache.pilot.digest(ROOT / logical) == old['files'][logical]['sha256']
    for field in cache.SEMANTICS:
        assert old['protocol'][field] == protocol()[field]


@pytest.mark.parametrize('change', ['native', 'rng', 'checkpoint', 'identity'])
def test_pilot_reuse_blocks_semantic_drift(change):
    p = protocol()
    records = cache.live_records(p)
    old = cache.checked_lock(ROOT / p['pilot_root'] / 'stage/manifest.json')
    files = copy.deepcopy(old['files'])
    if change == 'native':
        p['native']['recycling_steps'] = 4
    elif change == 'rng':
        p['rng']['base_seed'] += 1
    elif change == 'checkpoint':
        files[p['inputs']['checkpoint'] + '/model.safetensors']['sha256'] = 'wrong'
    else:
        selected = {r['record_id']: r for r in records}
        selected[old['records'][0]['record_id']]['seed'] += 1
    with pytest.raises(ValueError, match='BLOCKER'):
        cache.pilot_parity(p, records, files)


def test_actual_eight_import_bytes_schema_and_resume(tmp_path, reuse_lock):
    cache.import_pilot(tmp_path, reuse_lock)
    before = {}
    for rid, proof in reuse_lock['reused'].items():
        path = tmp_path / 'capture' / rid
        data = cache.load_record_packet(tmp_path, reuse_lock, rid)
        assert set(data) == {'affinity_module', 'affinity_module2'}
        for name, info in proof['manifest']['files'].items():
            assert cache.pilot.digest(path / name) == info['sha256']
        meta = cache.pilot.read_json(path / 'manifest.json')
        assert meta['reused_from']['manifest_sha256'] == proof['manifest_sha256']
        assert meta['record']['row_index'] == proof['record']['row_index']
        before[rid] = (path / 'manifest.json').stat().st_mtime_ns
    cache.import_pilot(tmp_path, reuse_lock)
    assert before == {rid: (tmp_path / 'capture' / rid / 'manifest.json').stat().st_mtime_ns for rid in before}
    report = cache.inventory(tmp_path, reuse_lock)
    assert (report['total'], report['valid'], report['missing'], report['invalid'], report['reused']) == (3344, 8, 3336, 0, 8)
    assert report['status'] == 'INCOMPLETE'
    rid = next(iter(before))
    with (tmp_path / 'capture' / rid / 'affinity_module.pt').open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        cache.import_pilot(tmp_path, reuse_lock)
    assert cache.inventory(tmp_path, reuse_lock)['status'] == 'FAIL'


def test_missing_failure_is_not_pass(tmp_path, reuse_lock):
    rid = reuse_lock['records'][0]['record_id']
    cache.pilot.atomic_json(tmp_path / 'failures' / f'{rid}.json', {'error': 'synthetic CPU failure'})
    report = cache.inventory(tmp_path, reuse_lock)
    assert report['status'] == 'FAIL'
    assert report['unresolved_failure_ids'] == [rid]
    assert report['valid'] + report['missing'] + report['invalid'] == 3344


def test_api_load_model_uses_members(tmp_path, reuse_lock, monkeypatch):
    sentinel = object()
    def load(checkpoint, p):
        assert checkpoint == tmp_path / 'stage/snapshot' / p['inputs']['checkpoint']
        return sentinel
    monkeypatch.setattr(cache.pilot, 'load_members', load)
    assert cache.load_model(reuse_lock, tmp_path) is sentinel


def test_training_and_docs_not_source_bound():
    assert set(cache.OWN_SOURCES) == {
        'experiments/20260906_affinity_comparison/code/cache.py',
        'experiments/20260906_affinity_comparison/cache_protocol.json',
        'tests/test_affinity_comparison_cache.py',
        'experiments/20260906_affinity_representation/code/pilot.py',
        'experiments/20260906_affinity_representation/protocol_pilot.json'}


@pytest.fixture
def staged_small(tmp_path, reuse_lock, monkeypatch):
    # Only source bytes are synthetic; identity selection is the real 3344 cohort.
    source = tmp_path / 'source.txt'
    source.write_text('CPU staging fixture, not model execution')
    monkeypatch.setattr(cache, 'source_map', lambda p, records: {'fixture.txt': source})
    monkeypatch.setattr(cache, 'pilot_parity', lambda p, records, files: reuse_lock['reused'])
    output = tmp_path / 'out'
    lock = cache.stage(output, protocol()['image_id'])
    return output, lock, source


def test_stage_snapshot_resume_without_training_binding(staged_small):
    output, lock, source = staged_small
    assert len(lock['records']) == 3344
    assert cache.validate_stage(output) == lock
    assert cache.stage(output, protocol()['image_id']) == lock
    frozen = output / 'stage/snapshot/fixture.txt'
    assert frozen.stat().st_ino != source.stat().st_ino
    assert not lock['numeric_outcomes_copied']
    assert not lock['challenge_label_training']


@pytest.mark.parametrize('change', ['source', 'snapshot', 'identity', 'binding'])
def test_stage_rejects_mutation(staged_small, change):
    output, lock, source = staged_small
    if change == 'source':
        source.write_text('changed')
    elif change == 'snapshot':
        (output / 'stage/snapshot/fixture.txt').write_text('changed')
    elif change == 'identity':
        cache.pilot.atomic_json(output / 'stage/identity_manifest.json', [])
    else:
        lock['binding'] = 'foreign'
        cache.pilot.atomic_json(output / 'stage/manifest.json', lock)
    with pytest.raises(ValueError):
        cache.validate_stage(output)


def test_partial_verify_and_complete_gate(staged_small):
    output, lock, source = staged_small
    report = cache.verify_cache(output, require_complete=False)
    assert report['status'] == 'INCOMPLETE' and report['missing'] == 3344
    with pytest.raises(ValueError, match='INCOMPLETE'):
        cache.verify_cache(output, require_complete=True)


def test_cli_help_and_gpu_gate(tmp_path):
    cmd = [sys.executable, str(ROOT / cache.EXP / 'code/cache.py')]
    assert subprocess.run(cmd + ['--help'], capture_output=True).returncode == 0
    result = subprocess.run(cmd + ['capture', '--output', str(tmp_path / 'no-write')], capture_output=True, text=True)
    assert result.returncode == 2 and '--allow-gpu' in result.stderr
    assert not (tmp_path / 'no-write').exists()
