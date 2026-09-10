"""Runner isolation and interval tests; all arrays here are synthetic fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_low_data import (
    atomic_json,
    complete_task,
    conformal_radius,
    digest,
    nearest_similarity,
    planned_tasks,
)


def test_conformal_small_calibration_keeps_unsupported_infinity():
    residuals = np.arange(1, 6, dtype=float)
    assert conformal_radius(residuals, 0.8) == 5
    assert np.isinf(conformal_radius(residuals, 0.9))
    with pytest.raises(ValueError):
        conformal_radius(np.array([np.nan]), 0.8)


def test_tanimoto_nearest_uses_fit_only():
    train = np.array([[1, 0, 0], [1, 1, 0]])
    test = np.array([[1, 0, 1], [1, 1, 0]])
    np.testing.assert_allclose(nearest_similarity(test, train), [0.5, 1.0])


def test_task_manifest_unique_and_all_conditions():
    config = {
        "budgets": [25, 50],
        "include_full_reference": True,
        "draws": 2,
        "n_folds": 3,
        "split_policies": ["random", "chemical_cluster"],
        "methods": ["repr_ridge", "head_pretrained"],
    }
    tasks = planned_tasks(config)
    assert len(tasks) == 72
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    assert {t["n_train"] for t in tasks} == {-1, 25, 50}


def test_completion_requires_readback_hashes(tmp_path: Path):
    path = tmp_path / "predictions.csv"
    path.write_text("fixture,value\nsynthetic,1\n")
    atomic_json(
        tmp_path / "complete.json", {"output_sha256": {path.name: digest(path)}}
    )
    assert complete_task(tmp_path)
    path.write_text("fixture,value\nsynthetic,2\n")
    assert not complete_task(tmp_path)


def test_atomic_json_handles_numpy_and_infinity(tmp_path: Path):
    path = tmp_path / "metric.json"
    atomic_json(path, {"width": np.inf, "n": np.int64(5), "vals": np.array([1, 2])})
    assert json.loads(path.read_text()) == {"width": None, "n": 5, "vals": [1, 2]}


def test_run_task_passes_only_fit_labels_to_factory(tmp_path, monkeypatch):
    """A synthetic canary checks the actual runner-to-fitter boundary."""
    import sys
    import types

    import pandas as pd

    from scripts.run_low_data import run_task

    model_module = types.ModuleType("nesso_pxr.low_data_models")
    metric_module = types.ModuleType("nesso_pxr.low_data_contract")

    def weights(se, floor):
        return 1 / np.maximum(se, floor)

    def scores(y, pred, se, floor):
        w = weights(se, floor)
        return {"weighted_mae": float(np.sum(w * abs(y - pred)) / w.sum())}

    calls = []

    def fitter(method, train_features, y, se, test_features, **kwargs):
        calls.append(method)
        np.testing.assert_array_equal(y, np.arange(4, dtype=float))
        np.testing.assert_array_equal(train_features["member1"][:, 0], np.arange(4))
        assert "pEC50" not in test_features
        n = len(test_features["member1"])
        return {
            "pred": np.ones(n),
            "ensemble_std": np.zeros(n),
            "seed_predictions": np.ones((1, n)),
            "n_parameters": 1,
            "fit_seconds": 0.0,
        }

    model_module.fit_predict = fitter
    metric_module.assay_weights = weights
    metric_module.score_predictions = scores
    monkeypatch.setitem(sys.modules, "nesso_pxr.low_data_models", model_module)
    monkeypatch.setitem(sys.modules, "nesso_pxr.low_data_contract", metric_module)
    inputs = pd.DataFrame(
        {
            "record_id": [f"synthetic_{i}" for i in range(12)],
            "row_index": np.arange(12),
            "pEC50": np.arange(12, dtype=float),
            "pEC50_standard_error": np.ones(12),
        }
    )
    feature = np.arange(12)[:, None]
    features = {"member1": feature, "morgan": np.eye(12, dtype=int)}
    assignments = pd.DataFrame(
        {
            "split": "random",
            "outer_fold": [1] * 5 + [0] * 7,
            "row_index": np.arange(12),
            "chemical_group": np.arange(12),
        }
    )
    subsets = pd.DataFrame(
        {
            "split": "random",
            "outer_fold": 0,
            "draw": 0,
            "n_train": 5,
            "row_index": np.arange(5),
            "role": ["fit"] * 4 + ["calibration"],
        }
    )
    task = {
        "task_id": "synthetic_fixture",
        "split": "random",
        "outer_fold": 0,
        "draw": 0,
        "n_train": 5,
        "method": "fixture_only",
    }
    config = {
        "se_floor": 0.1,
        "seeds": [42],
        "checkpoint": "unused-fixture",
        "interval_coverages": [0.8, 0.9],
    }
    result = run_task(task, config, inputs, features, assignments, subsets, tmp_path)
    assert calls == ["fixture_only"]
    assert result["n_fit"] == 4 and result["n_calibration"] == 1
    assert complete_task(tmp_path / "tasks" / "synthetic_fixture")
