"""Execute frozen argv stages durably; scientific completion uses stage verifiers."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic(path, obj):
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_spec(spec):
    if spec['physical_gpu'] != 0 or spec['wall_clock_cap'] is not None:
        raise ValueError('this user-authorized run is uncapped and GPU0-only')
    names = [s['name'] for s in spec['stages']]
    if not names or len(names) != len(set(names)):
        raise ValueError('stage names must be nonempty and unique')
    for p, expected in spec['source_sha256'].items():
        if sha(p) != expected:
            raise ValueError(f'frozen source drift: {p}')
    for stage in spec['stages']:
        if not stage['argv'] or not all(isinstance(a, str) for a in stage['argv']):
            raise ValueError('argv must be an explicit string list')
        argv = stage['argv']
        if '--gpus' in argv and argv[argv.index('--gpus') + 1] != 'device=0':
            raise ValueError('GPU command exposes an unauthorized device')
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    spec = validate_spec(json.loads(args.spec.read_text()))
    if args.check:
        print(json.dumps({'status': 'valid_spec', 'stages': len(spec['stages'])}))
        return
    root = Path(spec['runtime_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec_hash = sha(args.spec)
        old = root / 'status.json'
        if old.exists():
            state = json.loads(old.read_text())
            if state['spec_sha256'] != spec_hash:
                raise ValueError('cannot resume with a changed queue spec')
        else:
            state = {'spec_sha256': spec_hash, 'created_utc': now(), 'stages': {}, 'attempts': []}
        state['attempts'].append({'started_utc': now(), 'pid': os.getpid()})
        state.update(status='running', pid=os.getpid(), physical_gpu=0, wall_clock_cap=None)
        atomic(old, state)
        with (root / 'events.jsonl').open('a', buffering=1) as events:
            for stage in spec['stages']:
                name = stage['name']
                prior = state['stages'].get(name, {})
                if prior.get('status') == 'passed' and not stage.get('always_run', False):
                    continue
                validate_spec(spec)
                log_path = root / f'{name}.log'
                row = {'status': 'running', 'started_utc': now(), 'argv': stage['argv'], 'log': str(log_path)}
                state['stages'][name] = row
                state['current_stage'] = name
                atomic(old, state)
                events.write(json.dumps({'time': now(), 'event': 'start', 'stage': name}) + '\n')
                print('START', name, flush=True)
                started = time.monotonic()
                with log_path.open('ab') as log:
                    process = subprocess.Popen(stage['argv'], cwd=spec['repository'], env={**os.environ, **spec.get('environment', {})}, stdout=log, stderr=subprocess.STDOUT)
                    row['child_pid'] = process.pid
                    while True:
                        try:
                            rc = process.wait(timeout=10)
                            break
                        except subprocess.TimeoutExpired:
                            row.update(heartbeat_utc=now(), elapsed_seconds=time.monotonic()-started, log_bytes=log_path.stat().st_size)
                            atomic(old, state)
                row.update(returncode=rc, elapsed_seconds=time.monotonic()-started, finished_utc=now(), status='passed' if rc == 0 else 'failed')
                events.write(json.dumps({'time': now(), 'event': row['status'], 'stage': name, 'returncode': rc}) + '\n')
                if rc:
                    state['status'] = 'failed'
                    atomic(old, state)
                    raise RuntimeError(f'{name} failed; preserved log {log_path}; no later stage launched')
                atomic(old, state)
                print('PASS', name, flush=True)
            state.update(status='completed_worker_verifiers_passed_pending_parent_audit', finished_utc=now(), pid=0)
            atomic(old, state)

if __name__ == '__main__':
    main()
