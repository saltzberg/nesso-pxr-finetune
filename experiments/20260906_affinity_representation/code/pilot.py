#!/usr/bin/env python3
"""Immutable real-input, two-member PXR adaptation engineering pilot.

stage/verify are CPU-safe. capture/train require explicit --allow-gpu after
prior-analysis signoff. Run in the pinned image; never modify installed sources.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
ROOT = Path(__file__).resolve().parents[3]
EXP = Path('experiments/20260906_affinity_representation')
PROTOCOL = ROOT / EXP / 'protocol_pilot.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(canonical(value) + b'\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def seed_for(record_id, protocol):
    r = protocol['rng']
    text = f"{r['namespace']}|{r['base_seed']}|{record_id}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big') % 2147483647


def select_records(subsets, inputs, manifest, assignments, protocol):
    """Select using ONLY role/order/identity, never assay or prediction values."""
    s = protocol['selection']
    chosen = [r for r in subsets if all(r[k] == s[k] for k in
              ('split', 'outer_fold', 'draw', 'n_train', 'role'))][:s['count']]
    by_row = {r['row_index']: r for r in inputs}
    by_id = {r['record_id']: r for r in manifest}
    if len(by_row) != len(inputs) or len(by_id) != len(manifest):
        raise ValueError('duplicate identity in source tables')
    # Strict split preserves the original chemical_cluster test assignments.
    if {r['split'] for r in assignments} != {'chemical_cluster'}:
        raise ValueError('unexpected strict source assignment policy')
    test = {r['row_index'] for r in assignments
            if r['outer_fold'] == s['outer_fold']}
    if not test:
        raise ValueError('missing outer-test identities')
    result = []
    for order, role in enumerate(chosen):
        row = by_row[role['row_index']]
        rid = row['record_id']
        if by_id[rid]['modeling_role'] != 'development' or role['row_index'] in test:
            raise ValueError('non-development or outer-test record in FIT selection')
        if row['feature_row'] != by_id[rid]['feature_row']:
            raise ValueError('feature-row identity mismatch')
        result.append({**role, 'record_id': rid, 'feature_row': row['feature_row'],
                       'canonical_smiles': row['canonical_smiles'], 'order': order,
                       'modeling_role': 'development', 'seed': seed_for(rid, protocol)})
    ids = [r['record_id'] for r in result]
    if ids != protocol['record_ids'] or len(set(ids)) != s['count']:
        raise ValueError('frozen first-eight IDs do not replay')
    return result


def image_sources():
    spec = importlib.util.find_spec('nesso')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('stage requires the existing pinned Nesso image (no GPU needed)')
    package = Path(next(iter(spec.submodule_search_locations)))
    return {str(Path('image/nesso') / p.relative_to(package)): p
            for p in sorted(package.rglob('*.py'))}


def stage(args):
    if (args.output / 'stage').exists():
        return validate_stage(args)
    protocol = read_json(PROTOCOL)
    if args.image_id != protocol['image_id']:
        raise ValueError('externally inspected image ID must equal pinned protocol')
    inp = protocol['inputs']
    selected = select_records(*(rows(ROOT / inp[k]) for k in
                               ('subsets', 'labels', 'manifest', 'assignments')), protocol)
    sources = {str(EXP / 'code/pilot.py'): Path(__file__),
               str(EXP / 'protocol_pilot.json'): PROTOCOL,
               str(EXP / 'README.md'): ROOT / EXP / 'README.md',
               str(EXP / 'IMPLEMENTATION_PLAN.md'): ROOT / EXP / 'IMPLEMENTATION_PLAN.md',
               'tests/test_affinity_representation_pilot.py': ROOT / 'tests/test_affinity_representation_pilot.py',
               'src/nesso_pxr/low_data_adapter.py': ROOT / 'src/nesso_pxr/low_data_adapter.py',
               'scripts/audit_low_data_adapter.py': ROOT / 'scripts/audit_low_data_adapter.py'}
    sources.update(image_sources())
    for key in ('subsets', 'labels', 'manifest', 'assignments', 'ccd'):
        sources[inp[key]] = ROOT / inp[key]
    ck = ROOT / inp['checkpoint']
    for p in sorted(ck.rglob('*')):
        if p.is_file():
            sources[str(p.relative_to(ROOT))] = p
    processed = ROOT / inp['processed']
    for record in selected:
        rid = record['record_id']
        assets = [processed / 'records' / f'{rid}.json', processed / 'structures' / f'{rid}.npz']
        conformers = sorted((processed / 'rdkit_conformers').glob(f'{rid}*.pkl'))
        if not conformers:
            raise FileNotFoundError(f'no prepared ligand conformer: {rid}')
        for p in assets + conformers:
            sources[str(p.relative_to(ROOT))] = p
    esm = sorted((processed / 'esm_embeddings').glob('*.safetensors'))
    if not esm:
        raise FileNotFoundError('missing frozen ESM embedding')
    sources.update({str(p.relative_to(ROOT)): p for p in esm})
    required = sum(p.stat().st_size for p in sources.values())
    free = shutil.disk_usage(args.output).free
    if required + 2**30 > free:
        raise RuntimeError('insufficient space for immutable source/input snapshots')
    tmp = Path(tempfile.mkdtemp(prefix='.stage-', dir=args.output))
    try:
        bindings = {}
        for logical, source in sources.items():
            target = tmp / 'snapshot' / logical
            target.parent.mkdir(parents=True, exist_ok=True)
            before = digest(source)
            shutil.copyfile(source, target)
            if digest(target) != before or digest(source) != before:
                raise ValueError(f'source changed during snapshot: {source}')
            with target.open('rb') as f:
                os.fsync(f.fileno())
            bindings[logical] = {'sha256': before, 'bytes': target.stat().st_size,
                                 'original_path': str(source.resolve())}
        lock = {'schema': 1, 'protocol': protocol, 'records': selected,
                'image_id': args.image_id, 'files': bindings,
                'snapshot_bytes': required, 'free_bytes_before': free,
                'challenge_label_training': False,
                'note': 'labels snapshotted for future train stage, not used for selection/capture'}
        lock['binding'] = object_hash(lock)
        atomic_json(tmp / 'manifest.json', lock)
        os.rename(tmp, args.output / 'stage')
        fsync_dir(args.output)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return validate_stage(args)


def validate_stage(args):
    root = args.output / 'stage'
    lock = read_json(root / 'manifest.json')
    check = dict(lock)
    binding = check.pop('binding')
    if object_hash(check) != binding or lock['protocol'] != read_json(PROTOCOL):
        raise ValueError('stage binding/protocol changed')
    image = image_sources()
    for logical, info in lock['files'].items():
        frozen = root / 'snapshot' / logical
        live = image[logical] if logical.startswith('image/') else ROOT / logical
        if digest(frozen) != info['sha256'] or digest(live) != info['sha256']:
            raise ValueError(f'source/input/snapshot changed: {logical}')
    p = lock['protocol']
    replay = select_records(*(rows(root / 'snapshot' / p['inputs'][k]) for k in
                              ('subsets', 'labels', 'manifest', 'assignments')), p)
    if replay != lock['records']:
        raise ValueError('record identity/role replay failed')
    return lock


def tree(value, fn):
    import torch
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: tree(v, fn) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree(v, fn) for v in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f'unsupported cache value: {type(value)}')


def cpu(value):
    return tree(value, lambda t: t.detach().cpu().clone())


def tensor_hashes(state):
    import torch
    result = {}
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h = hashlib.sha256(f'{t.dtype}|{list(t.shape)}'.encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[name] = h.hexdigest()
    return result


def parameter_hashes(model, trainable):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {n.replace('.base.', '.'): t for n, t in model.state_dict().items()
             if (n in names) == trainable}
    return tensor_hashes(state)


def publish_packet(parent, name, payloads, metadata):
    """A single rename commits payloads AND their integrity/identity marker."""
    import torch
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / name
    if target.exists():
        raise FileExistsError(target)
    tmp = Path(tempfile.mkdtemp(prefix='.' + name + '-', dir=parent))
    try:
        files = {}
        for filename, payload in payloads.items():
            p = tmp / filename
            with p.open('wb') as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            files[filename] = {'sha256': digest(p), 'bytes': p.stat().st_size}
        atomic_json(tmp / 'manifest.json', {**metadata, 'files': files,
                    'payload_bytes': sum(v['bytes'] for v in files.values())})
        os.rename(tmp, target)
        fsync_dir(parent)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return read_json(target / 'manifest.json')


def check_packet(path, binding, record=None):
    meta = read_json(path / 'manifest.json')
    if meta['binding'] != binding or (record is not None and meta['record'] != record):
        raise ValueError('foreign packet binding or record identity')
    expected = {'manifest.json', *meta['files']}
    if {p.name for p in path.iterdir()} != expected or not meta['files']:
        raise ValueError('packet file set mismatch')
    for name, info in meta['files'].items():
        p = path / name
        if digest(p) != info['sha256'] or p.stat().st_size != info['bytes']:
            raise ValueError(f'corrupt packet: {p}')
    return meta


def gpu(args):
    import torch
    if not args.allow_gpu:
        raise RuntimeError('GPU stages require --allow-gpu after parent prior-analysis signoff')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('expose ONLY physical GPU0 using docker --gpus device=0')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    return torch


def capture(args, lock):
    torch = gpu(args)
    from lightning.pytorch import Trainer, seed_everything
    from nesso.data.inference import NessoInferenceDataModule
    from nesso.data.types import Manifest, Record
    from nesso.model.models.nesso1 import Nesso1
    p = lock['protocol']
    snap = args.output / 'stage/snapshot'
    ck = snap / p['inputs']['checkpoint']
    root = snap / p['inputs']['processed']
    model = Nesso1.from_pretrained(ck).eval()
    model.requires_grad_(False)
    model.predict_args.update(p['native'])
    # Persistent loaded model; each record has its own RNG stream and native call.
    trainer = Trainer(accelerator='gpu', devices=1, precision='bf16-mixed', logger=False,
                      enable_checkpointing=False, enable_progress_bar=False)
    for rec in lock['records']:
        rid = rec['record_id']
        target = args.output / 'capture' / rid
        if target.exists():
            validate_capture(args, lock, rec)
            continue
        captured = {}
        handles = []
        def pre(member):
            def hook(module, positional, kwargs):
                if positional or member in captured:
                    raise RuntimeError('expected exactly one keyword-only member call')
                kw = {**kwargs, 'feats': {k: v for k, v in kwargs['feats'].items()
                                         if isinstance(v, torch.Tensor)}}
                captured[member] = {'kwargs': cpu(kw), 'cpu_rng': torch.get_rng_state().clone(),
                                    'cuda_rng': torch.cuda.get_rng_state().clone(),
                                    'autocast_enabled': torch.is_autocast_enabled('cuda'),
                                    'removed_bookkeeping': [k for k, v in kwargs['feats'].items()
                                                            if not isinstance(v, torch.Tensor)]}
            return hook
        def post(member):
            def hook(module, positional, output):
                captured[member]['native_output'] = cpu(output)
            return hook
        for member in p['members']:
            module = getattr(model, member)
            handles += [module.register_forward_pre_hook(pre(member), with_kwargs=True),
                        module.register_forward_hook(post(member))]
        dm = NessoInferenceDataModule(manifest=Manifest(records=[Record.load(root / 'records' / f'{rid}.json')]),
             target_dir=root, esm_emb_dir=root / 'esm_embeddings', ligand_dir=root / 'rdkit_conformers',
             ccd_pkl=snap / p['inputs']['ccd'], num_workers=1, use_esm_all_layers=False,
             esm_emb_dim=1280, esm_num_layers=33)
        seed_everything(rec['seed'], workers=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            result = trainer.predict(model, datamodule=dm, return_predictions=True)
        finally:
            for h in handles:
                h.remove()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if len(result or []) != 1 or result[0].get('exception') or set(captured) != set(p['members']):
            raise RuntimeError(f'failed/incomplete native capture: {rid}')
        if any('native_output' not in x or x['autocast_enabled'] for x in captured.values()):
            raise RuntimeError('native output missing or affinity precision changed')
        outputs = {k: v for k, v in result[0].items() if isinstance(v, torch.Tensor)}
        a, b = (captured[m]['native_output']['affinity_pred_value'] for m in p['members'])
        if not torch.equal(((a + b) / 2).reshape(-1), outputs['affinity_pred_value'].cpu().reshape(-1)):
            raise AssertionError('native ensemble/member mismatch')
        payload = {f'{m}.pt': v for m, v in captured.items()}
        payload['native_prediction.pt'] = cpu(outputs)
        meta = publish_packet(args.output / 'capture', rid, payload,
                     {'binding': lock['binding'], 'record': rec, 'kind': 'real_native_both_member_capture',
                      'seconds': seconds, 'records_per_second': 1 / seconds,
                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                      'challenge_label_training': False})
        print(json.dumps({'captured': rid, 'bytes': meta['payload_bytes'], 'seconds': seconds}), flush=True)
    return verify(args, lock, require_train=False)


def validate_capture(args, lock, rec):
    import torch
    path = args.output / 'capture' / rec['record_id']
    meta = check_packet(path, lock['binding'], rec)
    expected = {f'{m}.pt' for m in lock['protocol']['members']} | {'native_prediction.pt'}
    if set(meta['files']) != expected or meta['kind'] != 'real_native_both_member_capture':
        raise ValueError('capture member/file contract changed')
    data = {m: torch.load(path / f'{m}.pt', map_location='cpu', weights_only=True)
            for m in lock['protocol']['members']}
    for value in data.values():
        if set(value['kwargs']) != {'s_inputs', 'z', 'pdistogram', 'feats', 'use_kernels'}:
            raise ValueError('exact affinity kwargs schema changed')
        if value['autocast_enabled'] or not value['kwargs']['use_kernels']:
            raise ValueError('native precision/kernel policy changed')
        if set(value['native_output']) != {'affinity_pred_value', 'affinity_repr', 'affinity_logits_binary'}:
            raise ValueError('native outputs incomplete')
    return data


def load_members(checkpoint, protocol):
    import torch
    from nesso.model.modules.affinity import AffinityModule
    from safetensors import safe_open
    hp = read_json(checkpoint / 'hparams.json')
    models = torch.nn.ModuleDict()
    with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as f:
        for member in protocol['members']:
            key = 'affinity_model_args2' if member == 'affinity_module2' else 'affinity_model_args'
            config = hp.get(key) or hp['affinity_model_args']
            model = AffinityModule(token_s=hp['token_s'], token_z=hp['token_z'], **config)
            prefix = member + '.'
            state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
            model.load_state_dict(state, strict=True)
            models[member] = model
    return models.cuda().eval()


def configure(models, arm, protocol):
    import torch
    from nesso_pxr.low_data_adapter import inject_lora
    t = protocol['training']
    models.requires_grad_(False)
    for model in models.values():
        if arm == 'head_plus_lora':
            inject_lora(model, t['lora_targets'], rank=t['rank'], alpha=t['alpha'])
        head = model.get_submodule(t['continuous_head'])
        if not isinstance(head, torch.nn.Sequential) or len(head) != 5:
            raise ValueError('unexpected pretrained continuous head architecture')
        head.requires_grad_(True)
    models.eval()
    return {n: p for n, p in models.named_parameters() if p.requires_grad}


def forward(models, packet, kernels=False):
    import torch
    if any(m.training for m in models.modules()):
        raise AssertionError('train mode changes native policy')
    out = {}
    for member, model in models.items():
        data = packet[member]
        kwargs = tree(data['kwargs'], lambda t: t.cuda())
        kwargs['use_kernels'] = data['kwargs']['use_kernels'] if kernels else False
        torch.set_rng_state(data['cpu_rng'])
        torch.cuda.set_rng_state(data['cuda_rng'])
        with torch.autocast('cuda', enabled=False):
            out[member] = model(**kwargs)
    return out


def parity(left, right, atol=0.0):
    import torch
    if set(left) != set(right):
        raise AssertionError('parity keys differ')
    worst = 0.0
    for key in left:
        if isinstance(left[key], dict):
            err = parity(left[key], right[key], atol)
        else:
            a, b = left[key].detach().cpu(), right[key].detach().cpu()
            if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise AssertionError('parity tensor schema/finite check failed')
            err = float((a - b).abs().max())
            if err > atol:
                raise AssertionError(f'native/replay parity failed at {key}: {err}')
        worst = max(worst, err)
    return worst


def weighted_loss(output, target, weight, weight_sum, protocol):
    import torch
    values = [output[m]['affinity_pred_value'].reshape(()) for m in protocol['members']]
    predicted = 6.0 - (values[0] + values[1]) / 2.0
    target_t = predicted.new_tensor(target)
    return torch.nn.functional.smooth_l1_loss(predicted, target_t,
             beta=protocol['training']['huber_beta']) * (weight / weight_sum)


def label_panel(args, lock):
    p = lock['protocol']
    source = rows(args.output / 'stage/snapshot' / p['inputs']['labels'])
    by_id = {r['record_id']: r for r in source}
    result = []
    for rec in lock['records']:
        row = by_id[rec['record_id']]
        y, se = float(row['pEC50']), float(row['pEC50_standard_error'])
        if not math.isfinite(y) or not math.isfinite(se) or se <= 0:
            raise ValueError('invalid real assay target/SE')
        result.append({'record_id': rec['record_id'], 'pEC50': y, 'standard_error': se,
                       'weight': 1 / max(se, p['training']['se_floor'])})
    return result


def replay_arm(args, lock, arm):
    torch = gpu(args)
    path = args.output / 'train' / arm
    meta = check_packet(path, lock['binding'])
    state = torch.load(path / 'state.pt', map_location='cpu', weights_only=True)
    saved = torch.load(path / 'outputs.pt', map_location='cpu', weights_only=True)
    p = lock['protocol']
    model = load_members(args.output / 'stage/snapshot' / p['inputs']['checkpoint'], p)
    torch.manual_seed(p['training']['seed'])
    params = configure(model, arm, p)
    if set(params) != set(state['parameters']):
        raise ValueError('trainable checkpoint keys differ')
    if parameter_hashes(model, False) != state['frozen_before']:
        raise ValueError('frozen pretrained reload hash differs')
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state['parameters'][name])
        for rec in lock['records']:
            packet = validate_capture(args, lock, rec)
            rid = rec['record_id']
            parity(forward(model, packet), saved['final'][rid], p['replay_atol'])
            parity(forward(model, packet, kernels=True), saved['final_native'][rid], p['replay_atol'])
    if parameter_hashes(model, True) != state['intended_after']:
        raise AssertionError('restored intended parameter hashes differ')
    del model
    torch.cuda.empty_cache()
    return meta


def train(args, lock):
    torch = gpu(args)
    p, t = lock['protocol'], lock['protocol']['training']
    for rec in lock['records']:
        validate_capture(args, lock, rec)
    labels = label_panel(args, lock)
    weight_sum = sum(r['weight'] for r in labels)
    shared_initial = None
    for arm in t['arms']:
        path = args.output / 'train' / arm
        if path.exists():
            replay_arm(args, lock, arm)
            # Recover interruption between atomic arm publish and replay proof.
            atomic_json(args.output / 'train' / f'{arm}.replay.json',
                        {'binding': lock['binding'], 'arm': arm, 'reload_max_abs': 0.0,
                         'manifest_sha256': digest(path / 'manifest.json')})
            old = read_json(path / 'manifest.json')['initial_continuous_hashes']
            if shared_initial is not None and old != shared_initial:
                raise AssertionError('matched head initialization differs')
            shared_initial = old
            continue
        model = load_members(args.output / 'stage/snapshot' / p['inputs']['checkpoint'], p)
        torch.manual_seed(t['seed'])
        outputs = {'baseline': {}, 'zero': {}, 'final': {}, 'final_native': {}}
        torch.cuda.reset_peak_memory_stats()
        prediction_start = time.perf_counter()
        with torch.no_grad():
            for rec in lock['records']:
                packet = validate_capture(args, lock, rec)
                baseline = forward(model, packet)
                parity(baseline, {m: v['native_output'] for m, v in packet.items()}, p['parity_atol'])
                parity(forward(model, packet, kernels=True), baseline, p['parity_atol'])
                outputs['baseline'][rec['record_id']] = cpu(baseline)
        torch.manual_seed(t['seed'])
        params = configure(model, arm, p)
        frozen = parameter_hashes(model, False)
        intended_before = parameter_hashes(model, True)
        initial_heads = {n: h for n, h in intended_before.items() if t['continuous_head'] in n}
        if shared_initial is not None and initial_heads != shared_initial:
            raise AssertionError('arms have different pretrained continuous heads')
        shared_initial = initial_heads
        initial = {n: q.detach().clone() for n, q in params.items()}
        with torch.no_grad():
            for rec in lock['records']:
                out = forward(model, validate_capture(args, lock, rec))
                parity(out, outputs['baseline'][rec['record_id']], 0.0)
                outputs['zero'][rec['record_id']] = cpu(out)
        torch.cuda.synchronize()
        baseline_seconds = time.perf_counter() - prediction_start
        optimizer = torch.optim.SGD(params.values(), lr=t['learning_rate'])
        history = []
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for step in range(t['steps']):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for rec, label in zip(lock['records'], labels):
                out = forward(model, validate_capture(args, lock, rec))
                loss = weighted_loss(out, label['pEC50'], label['weight'], weight_sum, p)
                if not torch.isfinite(loss):
                    raise AssertionError('nonfinite weighted Huber loss')
                loss.backward()
                total_loss += float(loss.detach())
                del out, loss
            norms = {}
            for name, param in params.items():
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise AssertionError(f'missing/nonfinite intended gradient: {name}')
                norms[name] = float(param.grad.norm())
                if norms[name] == 0 and not (step == 0 and name.endswith('lora_A')):
                    raise AssertionError(f'zero intended gradient: {name}')
            if any(q.grad is not None for q in model.parameters() if not q.requires_grad):
                raise AssertionError('frozen gradient detected')
            optimizer.step()
            history.append({'step': step, 'loss': total_loss, 'gradient_norms': norms})
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - start
        train_peak = torch.cuda.max_memory_allocated()
        reserved_peak = torch.cuda.max_memory_reserved()
        if parameter_hashes(model, False) != frozen:
            raise AssertionError('frozen weights/buffers changed')
        changes = {n: float((q - initial[n]).abs().max()) for n, q in params.items()}
        if any(v <= 0 for v in changes.values()):
            raise AssertionError('intended parameter did not update')
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for rec in lock['records']:
                packet = validate_capture(args, lock, rec)
                out = forward(model, packet)
                native = forward(model, packet, kernels=True)
                parity(out, native, p['parity_atol'])
                outputs['final'][rec['record_id']] = cpu(out)
                outputs['final_native'][rec['record_id']] = cpu(native)
        torch.cuda.synchronize()
        prediction_seconds = time.perf_counter() - start
        state = {'parameters': cpu(params), 'frozen_before': frozen,
                 'intended_before': intended_before, 'intended_after': parameter_hashes(model, True)}
        meta = {'binding': lock['binding'], 'arm': arm, 'record_ids': p['record_ids'],
                'challenge_label_training': True, 'heldout_efficacy_claim': False,
                'initial_continuous_hashes': initial_heads, 'labels': labels,
                'gradient_history': history, 'parameter_max_updates': changes,
                'frozen_unchanged': True, 'native_parity_before_after_max_abs': 0.0,
                'zero_update_max_abs': 0.0, 'baseline_and_zero_seconds': baseline_seconds,
                'training_seconds': training_seconds, 'training_record_steps': len(labels) * t['steps'],
                'training_records_per_second': len(labels) * t['steps'] / training_seconds,
                'prediction_seconds': prediction_seconds, 'prediction_record_passes': len(labels) * 2,
                'prediction_records_per_second': len(labels) * 2 / prediction_seconds,
                'training_peak_cuda_allocated_bytes': train_peak,
                'training_peak_cuda_reserved_bytes': reserved_peak,
                'prediction_peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'trainable_parameters': sum(q.numel() for q in params.values()),
                'torch_version': str(torch.__version__), 'mode': 'eval with gradients',
                'timing_scope': 'includes packet hash verification, disk load, host/device transfers; prediction includes differentiable + native kernel passes'}
        del model, optimizer, params, initial
        torch.cuda.empty_cache()
        publish_packet(args.output / 'train', arm, {'state.pt': state, 'outputs.pt': outputs}, meta)
        replay_arm(args, lock, arm)
        atomic_json(args.output / 'train' / f'{arm}.replay.json',
                    {'binding': lock['binding'], 'arm': arm, 'reload_max_abs': 0.0,
                     'manifest_sha256': digest(path / 'manifest.json')})
    return verify(args, lock, require_train=True)


def verify(args, lock, require_train=None):
    if require_train is None:
        require_train = args.require_train
    expected = set(lock['protocol']['record_ids'])
    captures = args.output / 'capture'
    found = {x.name for x in captures.iterdir() if x.is_dir() and not x.name.startswith('.')} if captures.exists() else set()
    if found != expected:
        raise ValueError(f'incomplete/foreign captures: missing={expected-found}; extra={found-expected}')
    metas = []
    for rec in lock['records']:
        validate_capture(args, lock, rec)
        metas.append(read_json(captures / rec['record_id'] / 'manifest.json'))
    arms = []
    heads = None
    if require_train:
        for arm in lock['protocol']['training']['arms']:
            path = args.output / 'train' / arm
            meta = check_packet(path, lock['binding'])
            if meta['arm'] != arm or meta['record_ids'] != lock['protocol']['record_ids']:
                raise ValueError('arm record contract differs')
            if heads is not None and meta['initial_continuous_hashes'] != heads:
                raise ValueError('matched heads differ')
            heads = meta['initial_continuous_hashes']
            marker = read_json(args.output / 'train' / f'{arm}.replay.json')
            if marker != {'binding': lock['binding'], 'arm': arm, 'reload_max_abs': 0.0,
                          'manifest_sha256': digest(path / 'manifest.json')}:
                raise ValueError('missing/foreign disk replay proof')
            if args.replay:
                replay_arm(args, lock, arm)
            arms.append(arm)
    result = {'status': 'verified', 'binding': lock['binding'], 'capture_records': len(metas),
              'capture_payload_bytes': sum(m['payload_bytes'] for m in metas),
              'per_record_payload_bytes': {m['record']['record_id']: m['payload_bytes'] for m in metas},
              'capture_seconds': sum(m['seconds'] for m in metas), 'verified_trained_arms': arms,
              'new_gpu_replay_performed': bool(args.replay and require_train),
              'heldout_efficacy_claim': False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['stage', 'capture', 'train', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-id', help='docker image inspect ID; mandatory for first stage')
    parser.add_argument('--allow-gpu', action='store_true', help='parent signoff required; expose GPU0 only')
    parser.add_argument('--require-train', action='store_true', help='verify both completed arms and replay proofs')
    parser.add_argument('--replay', action='store_true', help='verify saved models on GPU without fitting')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.stage in ('capture', 'train') or args.replay:
        if not args.allow_gpu:
            parser.error('GPU stage requires explicit --allow-gpu after prior analysis verification')
    if args.stage == 'stage':
        args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.pilot.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.stage == 'stage':
                lock = stage(args)
                result = {'status': 'staged_no_gpu_no_training', 'binding': lock['binding'],
                          'records': lock['records'], 'snapshot_bytes': lock['snapshot_bytes']}
            else:
                lock = validate_stage(args)
                result = {'capture': capture, 'train': train, 'verify': verify}[args.stage](args, lock)
            print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        except Exception:
            if args.stage != 'verify':
                failure = args.output / 'failures' / f'{time.time_ns()}-{args.stage}.json'
                atomic_json(failure, {'stage': args.stage, 'traceback': traceback.format_exc(),
                                     'scientific_evidence': False})
            raise


if __name__ == '__main__':
    main()
