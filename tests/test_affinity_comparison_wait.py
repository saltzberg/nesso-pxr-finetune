"""Synthetic supervisor tests; never signal a real process."""

import importlib
import json
from types import SimpleNamespace

import pytest

wait = importlib.import_module(
    "experiments.20260906_affinity_comparison.code.wait_native"
)


def setup_case(tmp_path, monkeypatch, *, valid=3344, pid=12):
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "binding": "test-binding",
                "total": 3344,
                "valid": valid,
                "missing": 3344 - valid,
                "invalid": 0,
                "unresolved_failure_ids": [],
                "status": "PASS" if valid == 3344 else "INCOMPLETE",
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "wait_native",
            "--cache",
            str(tmp_path),
            "--binding",
            "test-binding",
            "--retire-unit",
            "nesso-pxr-affinity-comparison-v1.service",
            "--retire-pid",
            "12",
        ],
    )
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if argv[0] == "docker" else 0, stdout="")

    monkeypatch.setattr(wait.subprocess, "run", run)
    monkeypatch.setattr(wait.subprocess, "check_output", lambda *a, **k: str(pid))
    return calls


def test_completed_writer_retires_only_exact_old_coordinator(tmp_path, monkeypatch):
    calls = setup_case(tmp_path, monkeypatch)
    wait.main()
    assert calls[-1] == [
        "systemctl",
        "--user",
        "kill",
        "--kill-whom=main",
        "--signal=SIGKILL",
        "nesso-pxr-affinity-comparison-v1.service",
    ]


def test_pid_drift_never_signals(tmp_path, monkeypatch):
    calls = setup_case(tmp_path, monkeypatch, pid=999)
    with pytest.raises(RuntimeError, match="PID changed"):
        wait.main()
    assert all(c[0] != "systemctl" for c in calls)


def test_incomplete_dead_writer_fails_without_signalling(tmp_path, monkeypatch):
    calls = setup_case(tmp_path, monkeypatch, valid=10)
    with pytest.raises(RuntimeError, match="terminated before complete"):
        wait.main()
    assert all(c[0] != "systemctl" for c in calls)
