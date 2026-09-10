"""Synthetic fixtures for the read-only completion monitor (not model evidence)."""

import importlib
import json

import pytest

monitor = importlib.import_module(
    "experiments.20260906_low_data_followup.code.watch_completion"
)


@pytest.fixture
def packet(tmp_path):
    task = {"task_id": "one", "method": "test_method"}
    (tmp_path / "task_manifest.json").write_text(json.dumps([task]))
    (tmp_path / "run_bindings.json").write_text("{}")
    directory = tmp_path / "tasks/one"
    directory.mkdir(parents=True)
    outputs = {}
    for name in [
        "metrics.json",
        "predictions.csv",
        "label_access.json",
        "seed_predictions.npz",
        "model_state.pkl",
    ]:
        (directory / name).write_text(
            json.dumps(task) if name == "metrics.json" else "synthetic-test-fixture"
        )
        outputs[name] = monitor.digest(directory / name)
    marker = {
        "task": task,
        "binding_sha256": monitor.digest(tmp_path / "run_bindings.json"),
        "output_sha256": outputs,
    }
    (directory / "complete.json").write_text(json.dumps(marker))
    return tmp_path


def test_verified_and_live_counts_are_distinct(packet):
    live, _ = monitor.inspect_lane(packet)
    assert live["matrix_complete"] and not live["fully_verified"]
    final, rows = monitor.inspect_lane(packet, verify_outputs=True)
    assert final["fully_verified"] and final["expected"] == len(rows) == 1


def test_corruption_is_not_complete(packet):
    (packet / "tasks/one/model_state.pkl").write_text("corrupted")
    result, _ = monitor.inspect_lane(packet, verify_outputs=True)
    assert result["invalid"] == 1 and not result["fully_verified"]


def test_foreign_binding_rejected(packet):
    (packet / "run_bindings.json").write_text('{"changed":true}')
    result, _ = monitor.inspect_lane(packet)
    assert result["invalid"] == 1


def test_duplicate_manifest_rejected(packet):
    path = packet / "task_manifest.json"
    path.write_text(json.dumps(json.loads(path.read_text()) * 2))
    with pytest.raises(ValueError, match="duplicate"):
        monitor.inspect_lane(packet)


def test_failed_and_pending_partition(packet):
    (packet / "tasks/one/complete.json").unlink()
    result, _ = monitor.inspect_lane(packet)
    assert result["pending"] == 1
    (packet / "tasks/one/failure.json").write_text("{}")
    result, _ = monitor.inspect_lane(packet)
    assert result["failed"] == 1 and result["pending"] == 0


def test_exited_service_is_terminal():
    assert not monitor.service_is_running(
        {"ActiveState": "active", "SubState": "exited", "MainPID": "0"}
    )
    assert monitor.service_is_running(
        {"ActiveState": "active", "SubState": "running", "MainPID": "123"}
    )
    assert not monitor.service_is_running(
        {"ActiveState": "inactive", "SubState": "dead", "MainPID": "0"}
    )


def test_descriptive_means_are_per_task_not_pooled():
    common = dict(
        split="test",
        n_train=25,
        method="fixture",
        mae=None,
        spearman=None,
        fit_seconds=1,
    )
    rows = [
        {**common, "weighted_mae": 1, "n_test": 10},
        {**common, "weighted_mae": 3, "n_test": 1000},
    ]
    result = monitor.aggregates(rows)
    assert result[0]["mean_task_weighted_mae"] == 2
    assert result[0]["completed_tasks"] == 2
