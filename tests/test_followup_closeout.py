"""Closeout invariants: exact cells, paired identity, immutable evidence."""

import importlib

import pandas as pd
import pytest

closeout = importlib.import_module(
    "experiments.20260906_low_data_followup.code.closeout"
)


def fixture():
    methods = [
        "weighted_mean",
        "native_continuous",
        "scalar_ridge",
        "repr_ridge",
        "repr_mlp",
        "head_random",
        "head_pretrained",
        "dual_head",
        "head_lora",
        "morgan_ridge",
        "morgan_lightgbm",
        "morgan_knn",
    ]
    tasks = pd.DataFrame(
        [
            {
                "task_id": f"random__fold0__draw00__n25__{method}",
                "split": "random",
                "outer_fold": 0,
                "draw": 0,
                "n_train": 25,
                "method": method,
                "status": "complete",
                "verification": "verified",
                "weighted_mae": 0.5 + index / 100,
                "mae": 0.6 + index / 100,
                "bias": index / 1000,
                "weight_sum": 100.0,
                "n_fit": 20,
                "n_calibration": 5,
                "n_acquired": 25,
                "n_test": 100,
                "row_signature": "fixture-test-rows",
            }
            for index, method in enumerate(methods)
        ]
    )
    protocol = {
        "budgets": [25],
        "include_full_reference": False,
        "split_policies": ["random"],
        "n_folds": 1,
        "draws": 1,
        "methods": methods,
    }
    manifest = tasks[["task_id", *closeout.CELL, "method"]].to_dict("records")
    return tasks, manifest, protocol


def test_complete_paired_matrix():
    tasks, manifest, protocol = fixture()
    result = closeout.validate_matrix(tasks, manifest, protocol)
    assert result["expected_tasks"] == 12
    assert result["fully_paired_cells"] == 1


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "identity", "budget"])
def test_matrix_corruption_rejected(mutation):
    tasks, manifest, protocol = fixture()
    if mutation == "missing":
        tasks = tasks.iloc[:-1]
    elif mutation == "duplicate":
        tasks = pd.concat([tasks, tasks.iloc[:1]])
    elif mutation == "identity":
        tasks.loc[tasks.index[0], "row_signature"] = "wrong"
    else:
        tasks.loc[tasks.index[0], "n_fit"] += 1
    with pytest.raises(ValueError):
        closeout.validate_matrix(tasks, manifest, protocol)


def test_evidence_copy_cannot_replace_history(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.write_text("original evidence")
    closeout.immutable_copy(source, dest)
    closeout.immutable_copy(source, dest)
    source.write_text("changed evidence")
    with pytest.raises(ValueError, match="immutable"):
        closeout.immutable_copy(source, dest)
    assert dest.read_text() == "original evidence"
