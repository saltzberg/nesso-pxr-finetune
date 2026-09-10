"""Wait for the existing native writer without interrupting its packets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--binding', required=True)
    ap.add_argument('--retire-unit')
    ap.add_argument('--retire-pid', type=int)
    args = ap.parse_args()
    while True:
        report = json.loads((args.cache / 'progress.json').read_text())
        if report['binding'] != args.binding or report['total'] != 3344:
            raise RuntimeError('unexpected native cache identity')
        if report['invalid'] or report.get('unresolved_failure_ids') or report['status'] == 'FAIL':
            raise RuntimeError('native capture failed; no downstream training launched')
        worker = subprocess.run(['docker', 'inspect', '--format', '{{.State.Status}}', 'nesso-pxr-affinity-cache-v1'], capture_output=True, text=True)
        running = worker.returncode == 0 and worker.stdout.strip() in {'running', 'paused', 'created', 'restarting'}
        if not running:
            if report['status'] != 'PASS' or report['valid'] != 3344 or report['missing'] != 0:
                raise RuntimeError('native writer terminated before complete accounting')
            break
        print(json.dumps({'native_valid': report['valid'], 'native_expected': report['total'], 'writer': worker.stdout.strip()}), flush=True)
        time.sleep(30)
    if args.retire_unit:
        if args.retire_unit != 'nesso-pxr-affinity-comparison-v1.service' or args.retire_pid is None:
            raise ValueError('only the explicitly superseded own coordinator may be retired')
        pid = int(subprocess.check_output(['systemctl', '--user', 'show', args.retire_unit, '--property=MainPID', '--value'], text=True).strip())
        if pid not in {0, args.retire_pid}:
            raise RuntimeError('superseded coordinator PID changed; do not signal it')
        if pid:
            subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=main', '--signal=SIGKILL', args.retire_unit], check=True)
    print(json.dumps({'status': 'native_writer_complete_ready_for_independent_verification', 'valid': report['valid']}), flush=True)

if __name__ == '__main__':
    main()
