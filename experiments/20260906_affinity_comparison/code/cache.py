#!/usr/bin/env python3
"""Full-development immutable native both-member cache; no assay values consumed."""
from __future__ import annotations
import argparse
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
EXP = Path('experiments/20260906_affinity_comparison')
OLD = Path('experiments/20260906_affinity_representation')
PROTOCOL = ROOT / EXP / 'cache_protocol.json'
_spec = importlib.util.spec_from_file_location('native_cache_pilot_helper', ROOT / OLD / 'code/pilot.py')
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)
IDENTITY = ('row_index', 'record_id', 'feature_row', 'canonical_smiles', 'modeling_role', 'seed')
SEMANTICS = ('image_id', 'members', 'native', 'precision', 'num_workers', 'matmul_precision', 'rng')
OWN_SOURCES = [str(EXP / 'code/cache.py'), str(EXP / 'cache_protocol.json'),
               'tests/test_affinity_comparison_cache.py', str(OLD / 'code/pilot.py'),
               str(OLD / 'protocol_pilot.json')]


def identity_rows(path, fields):
    """Only copy identity strings; never convert or return numeric assay fields."""
    with Path(path).open(newline='') as handle:
        return [{k: row[k] for k in fields} for row in csv.DictReader(handle)]


def select_records(inputs, manifest, protocol):
    count = protocol['selection']['count']
    if len(inputs) != count or [r['row_index'] for r in inputs] != [str(i) for i in range(count)]:
        raise ValueError('expected full ordered row_index cohort')
    by_id = {r['record_id']: r for r in manifest}
    if len(by_id) != len(manifest) or len({r['record_id'] for r in inputs}) != count:
        raise ValueError('duplicate record identity')
    development = {r['record_id'] for r in manifest if r['modeling_role'] == 'development'}
    if development != {r['record_id'] for r in inputs}:
        raise ValueError('exact development identity set differs')
    result = []
    for i, row in enumerate(inputs):
        rid = row['record_id']
        if not re.fullmatch(r'nesso_[0-9a-f]{20}', rid):
            raise ValueError('invalid literal record_id')
        match = by_id[rid]
        if row['feature_row'] != match['feature_row'] or row['canonical_smiles'] != match['canonical_smiles']:
            raise ValueError('feature-row/SMILES identity mismatch')
        result.append({**{k: row[k] for k in ('row_index', 'record_id', 'feature_row', 'canonical_smiles')},
                       'order': i, 'modeling_role': 'development', 'seed': pilot.seed_for(rid, protocol)})
    return result


def live_records(p):
    return select_records(identity_rows(ROOT / p['inputs']['cohort'],
                         ('row_index', 'record_id', 'feature_row', 'canonical_smiles')),
                         identity_rows(ROOT / p['inputs']['manifest'],
                         ('record_id', 'feature_row', 'canonical_smiles', 'modeling_role')), p)


def checked_lock(path):
    lock = pilot.read_json(path)
    unsigned = dict(lock)
    binding = unsigned.pop('binding')
    if pilot.object_hash(unsigned) != binding:
        raise ValueError('stage binding changed')
    return lock


def source_map(p, records):
    sources = {logical: ROOT / logical for logical in OWN_SOURCES}
    sources.update(pilot.image_sources())
    inp = p['inputs']
    sources[inp['ccd']] = ROOT / inp['ccd']
    for f in sorted((ROOT / inp['checkpoint']).rglob('*')):
        if f.is_file():
            sources[str(f.relative_to(ROOT))] = f
    for rec in records:
        rid = rec['record_id']
        base = ROOT / inp['processed']
        assets = [base / 'records' / f'{rid}.json', base / 'structures' / f'{rid}.npz']
        conformers = sorted((base / 'rdkit_conformers').glob(f'{rid}*.pkl'))
        if not conformers:
            raise ValueError(f'missing conformers: {rid}')
        for f in assets + conformers:
            sources[str(f.relative_to(ROOT))] = f
    esm = sorted((ROOT / inp['processed'] / 'esm_embeddings').glob('*.safetensors'))
    if not esm:
        raise ValueError('missing ESM assets')
    sources.update({str(f.relative_to(ROOT)): f for f in esm})
    return sources


def pilot_parity(p, records, files):
    """Fail closed: old native captures cannot be silently replaced by recapture."""
    origin = ROOT / p['pilot_root']
    old = checked_lock(origin / 'stage/manifest.json')
    if old['binding'] != p['pilot_binding']:
        raise ValueError('BLOCKER: unexpected pilot binding')
    for key in SEMANTICS:
        if old['protocol'][key] != p[key]:
            raise ValueError(f'BLOCKER: pilot native semantics differ: {key}')
    selected = {r['record_id']: r for r in records}
    relevant = {k for k in old['files'] if k.startswith(('image/', p['inputs']['checkpoint'] + '/',
                    p['inputs']['processed'] + '/')) or k in
                    (p['inputs']['ccd'], str(OLD / 'code/pilot.py'), str(OLD / 'protocol_pilot.json'))}
    if not relevant or not any(k.startswith('image/') for k in relevant):
        raise ValueError('BLOCKER: missing pilot model source proof')
    for logical in relevant:
        info = old['files'][logical]
        if logical not in files or files[logical]['sha256'] != info['sha256']:
            raise ValueError(f'BLOCKER: pilot source/input parity differs: {logical}')
        if pilot.digest(origin / 'stage/snapshot' / logical) != info['sha256']:
            raise ValueError(f'BLOCKER: corrupt pilot snapshot: {logical}')
    reused = {}
    if len(old['records']) != 8:
        raise ValueError('BLOCKER: expected eight pilot records')
    for rec in old['records']:
        rid = rec['record_id']
        if rid not in selected or any(rec[k] != selected[rid][k] for k in IDENTITY):
            raise ValueError(f'BLOCKER: pilot identity/RNG differs: {rid}')
        path = origin / 'capture' / rid
        meta = pilot.check_packet(path, old['binding'], rec)
        pilot.validate_capture(argparse.Namespace(output=origin), old, rec)
        if set(meta['files']) != {'affinity_module.pt', 'affinity_module2.pt', 'native_prediction.pt'}:
            raise ValueError('BLOCKER: pilot payload file schema differs')
        reused[rid] = {'binding': old['binding'], 'record': rec,
                       'manifest_sha256': pilot.digest(path / 'manifest.json'), 'manifest': meta,
                       'origin': str(Path(p['pilot_root']) / 'capture' / rid),
                       'source_native_filename': 'native_prediction.pt', 'target_native_filename': 'native_prediction.pt'}
    return reused


def stage(output, image_id):
    output = Path(output)
    if (output / 'stage').exists():
        return validate_stage(output)
    p = pilot.read_json(PROTOCOL)
    if image_id != p['image_id']:
        raise ValueError('inspected image digest must equal pinned protocol')
    records = live_records(p)
    sources = source_map(p, records)
    required = sum(f.stat().st_size for f in sources.values())
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < required + 2**30:
        raise ValueError('insufficient snapshot space')
    tmp = Path(tempfile.mkdtemp(prefix='.stage-', dir=output))
    try:
        files = {}
        for logical, source in sources.items():
            target = tmp / 'snapshot' / logical
            target.parent.mkdir(parents=True, exist_ok=True)
            h = pilot.digest(source)
            shutil.copyfile(source, target)
            if pilot.digest(target) != h or pilot.digest(source) != h:
                raise ValueError(f'source changed during snapshot: {logical}')
            with target.open('rb') as f:
                os.fsync(f.fileno())
            files[logical] = {'sha256': h, 'bytes': target.stat().st_size}
        reused = pilot_parity(p, records, files)
        pilot.atomic_json(tmp / 'identity_manifest.json', records)
        # Bind full source bytes without copying outcomes into the cache.
        identity_sources = {p['inputs'][k]: pilot.digest(ROOT / p['inputs'][k]) for k in ('cohort', 'manifest')}
        lock = {'schema': 1, 'kind': 'full_development_native_cache', 'protocol': p,
                'binding_namespace': 'pxr-affinity-comparison-cache-v1',
                'records': records, 'files': files, 'identity_sources': identity_sources,
                'identity_manifest_sha256': pilot.digest(tmp / 'identity_manifest.json'),
                'reused': reused, 'challenge_label_training': False, 'numeric_outcomes_copied': False,
                'snapshot_bytes': required}
        lock['binding'] = pilot.object_hash(lock)
        pilot.atomic_json(tmp / 'manifest.json', lock)
        os.rename(tmp, output / 'stage')
        pilot.fsync_dir(output)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return validate_stage(output)


def validate_stage(output):
    """CPU: verify source/snapshot/protocol/cohort; return binding/protocol/records lock."""
    output = Path(output)
    lock = checked_lock(output / 'stage/manifest.json')
    if lock['protocol'] != pilot.read_json(PROTOCOL) or lock['kind'] != 'full_development_native_cache':
        raise ValueError('cache protocol changed')
    p = lock['protocol']
    if lock['records'] != live_records(p):
        raise ValueError('record identity replay failed')
    identity = output / 'stage/identity_manifest.json'
    if pilot.digest(identity) != lock['identity_manifest_sha256'] or pilot.read_json(identity) != lock['records']:
        raise ValueError('identity manifest changed')
    for logical, h in lock['identity_sources'].items():
        if pilot.digest(ROOT / logical) != h:
            raise ValueError(f'identity source changed: {logical}')
    sources = source_map(p, lock['records'])
    if set(sources) != set(lock['files']):
        raise ValueError('source inventory changed')
    for logical, info in lock['files'].items():
        for f in (sources[logical], output / 'stage/snapshot' / logical):
            if f.stat().st_size != info['bytes'] or pilot.digest(f) != info['sha256']:
                raise ValueError(f'source/snapshot changed: {logical}')
    return lock


def load_model(lock, output):
    """GPU: load released affinity members (ModuleDict), eval; no adapters or fit."""
    return pilot.load_members(Path(output) / 'stage/snapshot' / lock['protocol']['inputs']['checkpoint'], lock['protocol'])


def _record(lock, rid):
    matches = [r for r in lock['records'] if r['record_id'] == rid]
    if len(matches) != 1:
        raise ValueError('foreign/duplicate record')
    return matches[0]


def load_record_packet(output, lock, rid):
    """CPU: hash/schema/identity validate; return {member_name: original payload}."""
    import torch
    path = Path(output) / 'capture' / rid
    meta = pilot.check_packet(path, lock['binding'], _record(lock, rid))
    members = lock['protocol']['members']
    if set(meta['files']) != {*(f'{m}.pt' for m in members), 'native_prediction.pt'} or meta['kind'] != 'real_native_both_member_capture':
        raise ValueError('native cache file/kind schema differs')
    if meta.get('challenge_label_training') is not False:
        raise ValueError('capture label-training policy differs')
    if rid in lock['reused']:
        proof = lock['reused'][rid]
        if meta.get('reused_from') != {k: v for k, v in proof.items() if k != 'manifest'}:
            raise ValueError('reused provenance differs')
        for name, info in meta['files'].items():
            oldname = 'native_prediction.pt' if name == 'native_prediction.pt' else name
            if info != proof['manifest']['files'][oldname]:
                raise ValueError('reused payload is not byte-identical')
    elif 'reused_from' in meta:
        raise ValueError('undeclared reused record')
    data = {m: torch.load(path / f'{m}.pt', map_location='cpu', weights_only=True) for m in members}
    for value in data.values():
        if set(value) != {'kwargs', 'cpu_rng', 'cuda_rng', 'autocast_enabled', 'removed_bookkeeping', 'native_output'}:
            raise ValueError('member payload schema differs')
        kw = value['kwargs']
        if set(kw) != {'s_inputs', 'z', 'pdistogram', 'feats', 'use_kernels'}:
            raise ValueError('native kwargs schema differs')
        if value['autocast_enabled'] is not False or kw['use_kernels'] is not True:
            raise ValueError('native precision/kernel policy differs')
        if set(value['native_output']) != {'affinity_pred_value', 'affinity_repr', 'affinity_logits_binary'}:
            raise ValueError('native output schema differs')
        for key in ('cpu_rng', 'cuda_rng'):
            t = value[key]
            if not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 or t.ndim != 1 or not t.numel():
                raise ValueError('RNG state schema differs')
        pilot.tree(kw, lambda t: _finite(t))
        pilot.tree(value['native_output'], lambda t: _finite(t))
    native = torch.load(path / 'native_prediction.pt', map_location='cpu', weights_only=True)
    pilot.tree(native, lambda t: _finite(t))
    a, b = (data[m]['native_output']['affinity_pred_value'] for m in members)
    if not torch.equal(((a + b) / 2).reshape(-1), native['affinity_pred_value'].reshape(-1)):
        raise ValueError('native ensemble/member mismatch')
    return data


def _finite(t):
    import torch
    if t.is_floating_point() and not torch.isfinite(t).all():
        raise ValueError('nonfinite native packet tensor')
    return t


def import_pilot(output, lock):
    """Copy original payload bytes, rename native filename, retain source manifest proof."""
    output = Path(output)
    parent = output / 'capture'
    parent.mkdir(parents=True, exist_ok=True)
    for rid, proof in lock['reused'].items():
        target = parent / rid
        if target.exists():
            load_record_packet(output, lock, rid)
            continue
        source = ROOT / proof['origin']
        if pilot.digest(source / 'manifest.json') != proof['manifest_sha256']:
            raise ValueError(f'BLOCKER: original pilot manifest changed: {rid}')
        pilot.check_packet(source, proof['binding'], proof['record'])
        tmp = Path(tempfile.mkdtemp(prefix='.' + rid + '-', dir=parent))
        try:
            meta = dict(proof['manifest'])
            files = {}
            for oldname, info in meta['files'].items():
                name = 'native_prediction.pt' if oldname == 'native_prediction.pt' else oldname
                shutil.copyfile(source / oldname, tmp / name)
                if pilot.digest(tmp / name) != info['sha256']:
                    raise ValueError('BLOCKER: pilot copy integrity failed')
                with (tmp / name).open('rb') as handle:
                    os.fsync(handle.fileno())
                files[name] = info
            meta.update(binding=lock['binding'], record=_record(lock, rid), files=files,
                        reused_from={k: v for k, v in proof.items() if k != 'manifest'})
            pilot.atomic_json(tmp / 'manifest.json', meta)
            os.rename(tmp, target)
            pilot.fsync_dir(parent)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        load_record_packet(output, lock, rid)
        event(output, 'reused', record_id=rid, original_manifest_sha256=proof['manifest_sha256'])


def event(output, kind, **fields):
    path = Path(output) / 'events.jsonl'
    with path.open('ab') as handle:
        handle.write(pilot.canonical({'event': kind, 'time_ns': time.time_ns(), **fields}) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def inventory(output, lock):
    output = Path(output)
    expected = {r['record_id'] for r in lock['records']}
    base = output / 'capture'
    found = {p.name for p in base.iterdir() if not p.name.startswith('.')} if base.exists() else set()
    valid, invalid, missing = [], {}, []
    for rid in sorted(expected):
        if rid not in found:
            missing.append(rid)
            continue
        try:
            load_record_packet(output, lock, rid)
            valid.append(rid)
        except Exception as exc:
            invalid[rid] = f'{type(exc).__name__}: {exc}'
    failures = output / 'failures'
    failed = {p.stem for p in failures.glob('*.json')} if failures.exists() else set()
    unresolved = sorted(failed - set(valid))
    extra = sorted(found - expected)
    result = {'binding': lock['binding'], 'total': len(expected), 'valid': len(valid),
              'missing': len(missing), 'invalid': len(invalid), 'unexpected': extra,
              'valid_ids': valid, 'missing_ids': missing, 'invalid_records': invalid,
              'unresolved_failure_ids': unresolved, 'reused': len(set(valid) & set(lock['reused']))}
    result['status'] = ('FAIL' if invalid or extra or unresolved else 'PASS' if not missing else 'INCOMPLETE')
    assert result['valid'] + result['missing'] + result['invalid'] == result['total']
    return result


def verify_cache(output, require_complete=True):
    """CPU full scan; partial is INCOMPLETE, never PASS; failures raise."""
    lock = validate_stage(output)
    report = inventory(output, lock)
    pilot.atomic_json(Path(output) / 'progress.json', report)
    if report['status'] == 'FAIL' or (require_complete and report['status'] != 'PASS'):
        raise ValueError(f"cache verification {report['status']}: valid={report['valid']} missing={report['missing']} invalid={report['invalid']}")
    return report


def capture(args, lock):
    torch = pilot.gpu(args)
    from lightning.pytorch import Trainer, seed_everything
    from nesso.data.inference import NessoInferenceDataModule
    from nesso.data.types import Manifest, Record
    from nesso.model.models.nesso1 import Nesso1
    output = args.output
    import_pilot(output, lock)
    report = inventory(output, lock)
    pilot.atomic_json(output / 'progress.json', report)
    if report['status'] == 'FAIL':
        raise ValueError('invalid existing cache; no silent recapture')
    pending = [r for r in lock['records'] if r['record_id'] not in set(report['valid_ids'])]
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        return report
    p = lock['protocol']
    snap = output / 'stage/snapshot'
    root = snap / p['inputs']['processed']
    model = Nesso1.from_pretrained(snap / p['inputs']['checkpoint']).eval()
    model.requires_grad_(False)
    model.predict_args.update(p['native'])
    trainer = Trainer(accelerator='gpu', devices=1, precision='bf16-mixed', logger=False,
                      enable_checkpointing=False, enable_progress_bar=False)
    event(output, 'capture_start', binding=lock['binding'], pending=len(pending), limit=args.limit)
    for rec in pending:
        rid = rec['record_id']
        captured, handles = {}, []
        def pre(member, record_capture=captured):
            def hook(module, positional, kwargs):
                if positional or member in record_capture:
                    raise RuntimeError('expected exactly one keyword-only member call')
                kw = {**kwargs, 'feats': {k: v for k, v in kwargs['feats'].items() if isinstance(v, torch.Tensor)}}
                record_capture[member] = {'kwargs': pilot.cpu(kw), 'cpu_rng': torch.get_rng_state().clone(),
                    'cuda_rng': torch.cuda.get_rng_state().clone(), 'autocast_enabled': torch.is_autocast_enabled('cuda'),
                    'removed_bookkeeping': [k for k, v in kwargs['feats'].items() if not isinstance(v, torch.Tensor)]}
            return hook
        def post(member, record_capture=captured):
            def hook(module, positional, hook_result):
                record_capture[member]['native_output'] = pilot.cpu(hook_result)
            return hook
        try:
            for member in p['members']:
                module = getattr(model, member)
                handles += [module.register_forward_pre_hook(pre(member), with_kwargs=True), module.register_forward_hook(post(member))]
            dm = NessoInferenceDataModule(manifest=Manifest(records=[Record.load(root / 'records' / f'{rid}.json')]),
                target_dir=root, esm_emb_dir=root / 'esm_embeddings', ligand_dir=root / 'rdkit_conformers',
                ccd_pkl=snap / p['inputs']['ccd'], num_workers=1, use_esm_all_layers=False, esm_emb_dim=1280, esm_num_layers=33)
            seed_everything(rec['seed'], workers=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = trainer.predict(model, datamodule=dm, return_predictions=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if len(result or []) != 1 or result[0].get('exception') or set(captured) != set(p['members']):
                raise RuntimeError('failed/incomplete native capture')
            if any('native_output' not in v or v['autocast_enabled'] for v in captured.values()):
                raise ValueError('native output/precision differs')
            outputs = {k: v for k, v in result[0].items() if isinstance(v, torch.Tensor)}
            a, b = (captured[m]['native_output']['affinity_pred_value'] for m in p['members'])
            if not torch.equal(((a + b) / 2).reshape(-1), outputs['affinity_pred_value'].cpu().reshape(-1)):
                raise ValueError('native ensemble/member mismatch')
            pilot.publish_packet(output / 'capture', rid,
                {**{f'{m}.pt': v for m, v in captured.items()}, 'native_prediction.pt': pilot.cpu(outputs)},
                {'binding': lock['binding'], 'record': rec, 'kind': 'real_native_both_member_capture',
                 'seconds': seconds, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'challenge_label_training': False})
            load_record_packet(output, lock, rid)
            report['valid_ids'].append(rid)
            report['missing_ids'].remove(rid)
            report['valid'] += 1
            report['missing'] -= 1
            report['status'] = 'PASS' if not report['missing'] else 'INCOMPLETE'
            pilot.atomic_json(output / 'progress.json', report)
            event(output, 'captured', record_id=rid, valid=report['valid'], total=report['total'], seconds=seconds)
        except Exception as exc:
            pilot.atomic_json(output / 'failures' / f'{rid}.json', {'binding': lock['binding'], 'record': rec,
                'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc(), 'time_ns': time.time_ns()})
            event(output, 'failed', record_id=rid, error=str(exc))
            pilot.atomic_json(output / 'progress.json', inventory(output, lock))
            raise
        finally:
            for handle in handles:
                handle.remove()
    return verify_cache(output, require_complete=args.limit is None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('stage', 'capture', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-id')
    parser.add_argument('--allow-gpu', action='store_true')
    parser.add_argument('--limit', type=int, help='maximum NEW native records this invocation; never changes full cohort')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and (args.limit < 1 or args.command != 'capture'):
        parser.error('--limit must be positive and applies only to capture')
    if args.command == 'capture' and not args.allow_gpu:
        parser.error('capture requires explicit --allow-gpu')
    args.output = args.output.resolve()
    if args.command == 'stage':
        args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.cache.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.command == 'stage':
                lock = stage(args.output, args.image_id)
                import_pilot(args.output, lock)
                result = verify_cache(args.output, require_complete=False)
            elif args.command == 'capture':
                result = capture(args, validate_stage(args.output))
            else:
                result = verify_cache(args.output, require_complete=args.require_complete)
            print(json.dumps(result, indent=2, allow_nan=False))
        except Exception as exc:
            event(args.output, 'command_failed', command=args.command, error=str(exc))
            raise


if __name__ == '__main__':
    main()
