"""Synthetic command-runner fixtures, never scientific model results."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

queue = importlib.import_module(
    "experiments.20260906_affinity_comparison.code.run_queue"
)
assert queue.__file__ is not None
SCRIPT = str(queue.__file__)


def spec_for(tmp_path):
    code = Path(SCRIPT).resolve()
    return {
        "physical_gpu": 0,
        "wall_clock_cap": None,
        "repository": str(tmp_path),
        "runtime_root": str(tmp_path / "runtime"),
        "source_sha256": {str(code): queue.sha(code)},
        "stages": [{"name": "synthetic", "argv": [sys.executable, "-c", "pass"]}],
    }


def test_only_authorized_gpu_and_uncapped(tmp_path):
    spec = spec_for(tmp_path)
    assert queue.validate_spec(spec) is spec
    spec["physical_gpu"] = 1
    with pytest.raises(ValueError):
        queue.validate_spec(spec)
    spec["physical_gpu"] = 0
    spec["wall_clock_cap"] = 600
    with pytest.raises(ValueError):
        queue.validate_spec(spec)


def test_source_and_gpu_argv_drift(tmp_path):
    spec = spec_for(tmp_path)
    spec["stages"][0]["argv"] = ["docker", "run", "--gpus", "all", "image"]
    with pytest.raises(ValueError):
        queue.validate_spec(spec)
    spec = spec_for(tmp_path)
    spec["source_sha256"][str(Path(SCRIPT).resolve())] = "not-a-hash"
    with pytest.raises(ValueError):
        queue.validate_spec(spec)


def test_resume_skips_work_but_repeats_verification(tmp_path):
    spec = spec_for(tmp_path)

    def counter(name):
        return [
            sys.executable,
            "-c",
            f"from pathlib import Path;p=Path({name!r});p.write_text(str(int(p.read_text())+1 if p.exists() else 1))",
        ]

    spec["stages"] = [
        {"name": "fit_fixture", "argv": counter("fit_count")},
        {"name": "verify_fixture", "argv": counter("verify_count"), "always_run": True},
    ]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    command = [sys.executable, SCRIPT, "--spec", str(path)]
    for _ in range(2):
        subprocess.run(command, check=True, capture_output=True)
    assert (tmp_path / "fit_count").read_text() == "1"
    assert (tmp_path / "verify_count").read_text() == "2"
    result = json.loads((tmp_path / "runtime/status.json").read_text())
    assert result["pid"] == 0
    assert result["status"] == "completed_worker_verifiers_passed_pending_parent_audit"


def test_failure_stops_later_stages(tmp_path):
    spec = spec_for(tmp_path)
    spec["stages"] = [
        {"name": "failure", "argv": [sys.executable, "-c", "raise SystemExit(4)"]},
        {
            "name": "must_not_run",
            "argv": [sys.executable, "-c", "raise SystemExit(99)"],
        },
    ]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    result = subprocess.run(
        [sys.executable, SCRIPT, "--spec", str(path)], capture_output=True
    )
    assert result.returncode != 0
    state = json.loads((tmp_path / "runtime/status.json").read_text())
    assert state["status"] == "failed"
    assert state["stages"]["failure"]["returncode"] == 4
    assert "must_not_run" not in state["stages"]
