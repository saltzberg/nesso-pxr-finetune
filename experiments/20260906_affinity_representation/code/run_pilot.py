"""Durable sequential launcher; exact commands and logs remain run-owned."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path("experiments/20260906_affinity_representation")


def digest(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--external-assets", type=Path, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    previous = ROOT / "artifacts/experiments/low_data_followup_20260906/final_analysis"
    proof = json.loads((previous / "parent_verification.json").read_text())
    if (
        proof["status"] != "pass"
        or not proof["all_new_task_scores_match_previous_monitor"]
    ):
        raise RuntimeError("previous scientific analysis not verified")
    protocol = json.loads((ROOT / EXP / "protocol_pilot.json").read_text())
    image = protocol["image_id"]
    inspected = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
    ).strip()
    if inspected != image:
        raise RuntimeError("wrong image")
    gpu_state = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu),
            "--query-gpu=uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [x.strip() for x in gpu_state.split(",")]
    if int(parts[2]) > 100 or int(parts[3]) > 5:
        raise RuntimeError("selected GPU is not idle; no other job will be stopped")
    common = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=2g,mode=1777",
        "--cpus",
        "4",
        "--shm-size",
        "1g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-e",
        "XDG_CACHE_HOME=/tmp/cache",
        "-v",
        f"{ROOT}:/repo:ro",
        "-v",
        f"{output}:/run:rw",
        "-v",
        f"{args.external_assets.resolve()}:{args.external_assets.resolve()}:ro",
        "-w",
        "/repo",
        "-e",
        "PYTHONPATH=/repo/src:/repo",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "OMP_NUM_THREADS=4",
        "-e",
        "OPENBLAS_NUM_THREADS=4",
        "-e",
        "MKL_NUM_THREADS=4",
        "--entrypoint",
        "python",
    ]
    stages = [
        ("stage", False, ["--image-id", image]),
        ("capture", True, ["--allow-gpu"]),
        ("train", True, ["--allow-gpu"]),
        ("verify", True, ["--require-train", "--replay", "--allow-gpu"]),
    ]
    status = {
        "pid": os.getpid(),
        "started_utc": datetime.now(UTC).isoformat(),
        "image_id": image,
        "physical_gpu": args.gpu,
        "gpu_state_before": gpu_state,
        "launcher_sha256": digest(__file__),
        "previous_analysis_verification_sha256": digest(
            previous / "parent_verification.json"
        ),
        "previous_analysis_manifest_sha256": digest(
            previous / "analysis_manifest.json"
        ),
        "wall_clock_cap": None,
        "stages": [],
        "status": "running",
    }
    (runtime / "launcher.py").write_bytes(Path(__file__).read_bytes())
    save(runtime / "status.json", status)
    for stage, gpu, flags in stages:
        name = f"nesso-pxr-representation-{os.getpid()}-{stage}"
        cmd = (
            common
            + ["--name", name]
            + (["--gpus", f"device={args.gpu}"] if gpu else [])
        )
        cmd += [image, str(EXP / "code/pilot.py"), stage, "--output", "/run"] + flags
        row = {
            "stage": stage,
            "command": cmd,
            "started_utc": datetime.now(UTC).isoformat(),
            "status": "running",
        }
        status["stages"].append(row)
        save(runtime / "status.json", status)
        print("START", stage, flush=True)
        start = time.monotonic()
        with (runtime / f"{stage}.log").open("ab") as log:
            result = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        row.update(
            returncode=result.returncode,
            elapsed_seconds=time.monotonic() - start,
            status="passed" if result.returncode == 0 else "failed",
        )
        if result.returncode:
            status["status"] = "failed"
            save(runtime / "status.json", status)
            raise RuntimeError(
                f"{stage} failed: see run-owned log; no next stage started"
            )
        save(runtime / "status.json", status)
        print("PASS", stage, row["elapsed_seconds"], flush=True)
    status["status"] = "completed_pending_parent_audit"
    status["finished_utc"] = datetime.now(UTC).isoformat()
    save(runtime / "status.json", status)


if __name__ == "__main__":
    main()
