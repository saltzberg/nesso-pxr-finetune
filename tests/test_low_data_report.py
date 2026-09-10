"""Synthetic fixtures only; never publish these as scientific evidence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nesso_pxr import low_data_report as report


def test_public_json_uses_repository_relative_paths(tmp_path):
    root = Path(report.__file__).resolve().parents[2]
    destination = tmp_path / "public.json"
    report._write_json(destination, {"path": str(root / "artifacts/input.json")})
    assert json.loads(destination.read_text())["path"] == "artifacts/input.json"


@pytest.mark.parametrize("split", ["random", "chemical_cluster"])
def test_real_knn_saved_spearman_round_trip(split):
    """Optional real-artifact regression: CSV parsing must preserve rank ties."""
    root = Path(report.__file__).resolve().parents[2]
    run = root / "artifacts/experiments/low_data_20260905_cut035"
    task = run / "tasks" / f"{split}__fold0__draw00__n25__morgan_knn"
    if not (task / "complete.json").is_file():
        pytest.skip("retained real integration pilot not available")
    config = json.loads((run / "protocol.json").read_text())
    prep = run / "prepared"
    meta = json.loads((task / "metrics.json").read_text())
    frame, verified = report._verify(
        task,
        meta,
        config,
        pd.read_csv(prep / "inputs.csv"),
        pd.read_csv(prep / "assignments.csv"),
        pd.read_csv(prep / "subsets.csv"),
    )
    assert len(frame) == meta["n_test"]
    assert verified["spearman"] == pytest.approx(meta["spearman"], abs=1e-14)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    report._write_json(path, value)


def seal(directory):
    write(
        directory / "complete.json",
        {
            "task_id": directory.name,
            "output_sha256": {
                p.name: report._hash(p)
                for p in directory.iterdir()
                if p.name != "complete.json"
            },
        },
    )


@pytest.fixture
def matrix(tmp_path):
    run = tmp_path / "run"
    prep = run / "prepared"
    prep.mkdir(parents=True)
    config = dict(
        budgets=[25],
        include_full_reference=False,
        split_policies=["random"],
        n_folds=2,
        draws=2,
        methods=["head_random", "head_pretrained"],
        calibration_fraction=0.2,
        se_floor=0.1,
    )
    write(prep / "protocol.json", config)
    inputs = pd.DataFrame(
        dict(
            row_index=range(50),
            record_id=[f"r{i}" for i in range(50)],
            pEC50=np.linspace(2, 6, 50),
            pEC50_standard_error=np.tile([0.05, 0.2], 25),
        )
    )
    inputs.to_csv(prep / "inputs.csv", index=False)
    assignments = pd.DataFrame(
        dict(
            split="random",
            row_index=range(50),
            outer_fold=np.repeat([0, 1], 25),
            chemical_group=range(50),
        )
    )
    assignments.to_csv(prep / "assignments.csv", index=False)
    subsets = []
    for fold in range(2):
        for draw in range(2):
            for j, index in enumerate(
                assignments.loc[assignments.outer_fold.ne(fold), "row_index"]
            ):
                subsets.append(
                    dict(
                        split="random",
                        outer_fold=fold,
                        draw=draw,
                        n_train=25,
                        row_index=index,
                        role="fit" if j < 20 else "calibration",
                    )
                )
    subsets = pd.DataFrame(subsets)
    subsets.to_csv(prep / "subsets.csv", index=False)
    write(prep / "split_audit.json", {"singleton_compound_fraction": 1.0})
    write(
        prep / "preparation_manifest.json",
        {
            "config": config,
            "artifact_sha256": {p.name: report._hash(p) for p in prep.iterdir()},
        },
    )
    write(
        run / "run_bindings.json",
        {
            "config": config,
            "sources": {
                str(Path(report.__file__).resolve()): report._hash(
                    Path(report.__file__)
                )
            },
            "inputs": {str(prep / "inputs.csv"): report._hash(prep / "inputs.csv")},
        },
    )
    write(
        run / "task_manifest.json",
        [
            dict(zip(report.KEY, key, strict=False), task_id=report._task_id(key))
            for key in report._expected(config)
        ],
    )
    for draw in range(2):
        for method in config["methods"]:
            key = ("random", 0, draw, 25, method)
            directory = run / "tasks" / report._task_id(key)
            directory.mkdir(parents=True)
            truth = inputs.iloc[:25]
            delta = 0.5 if method == "head_random" else 0.2 + 0.1 * draw
            frame = pd.DataFrame(
                dict(
                    record_id=truth.record_id,
                    row_index=truth.row_index,
                    **dict(zip(report.KEY, key, strict=False)),
                    y_true=truth.pEC50,
                    y_pred=truth.pEC50 + delta,
                    assay_se=truth.pEC50_standard_error,
                    weight=1 / np.maximum(truth.pEC50_standard_error, 0.1),
                    ensemble_std=0.0,
                    nearest_similarity=0.3,
                    chemical_group=range(25),
                    lower_80=truth.pEC50 + delta - 1,
                    upper_80=truth.pEC50 + delta + 1,
                    lower_90=-np.inf,
                    upper_90=np.inf,
                )
            )
            frame.to_csv(directory / "predictions.csv", index=False)
            meta = dict(
                zip(report.KEY, key, strict=False),
                task_id=directory.name,
                status="complete",
                n_fit=20,
                n_calibration=5,
                n_test=25,
                fit_seconds=1.0,
                n_parameters=10,
                selection={},
                fit_audit={},
                **report._score(frame, 0.1),
            )
            for level in [80, 90]:
                meta.update(report._interval(frame, level))
                meta[f"interval_{level}_finite"] = level == 80
            write(directory / "metrics.json", meta)
            write(
                directory / "label_access.json",
                dict(
                    fit_record_ids=inputs.record_id.iloc[25:45].tolist(),
                    calibration_record_ids=inputs.record_id.iloc[45:50].tolist(),
                    test_record_ids=truth.record_id.tolist(),
                    test_labels_accessible_to_fit=False,
                    calibration_labels_accessible_to_fit=False,
                    acquired_label_count=25,
                ),
            )
            values = np.r_[inputs.pEC50.iloc[45:50] + delta, frame.y_pred]
            np.savez(
                directory / "seed_predictions.npz",
                row_index=np.r_[np.arange(45, 50), np.arange(25)],
                predictions=np.tile(values, (3, 1)),
            )
            seal(directory)
    return run, tmp_path / "report"


def test_partial_matrix_and_artifacts(matrix):
    run, out = matrix
    result = report.build_report(run, out)
    assert (
        result["completed"],
        result["failed"],
        result["pending"],
        result["total_tasks"],
    ) == (4, 0, 4, 8)
    assert result["winner"] is None
    metrics = pd.read_csv(out / "metrics.csv")
    assert metrics.n_unique_compounds.tolist() == [25, 25]
    assert metrics.n_prediction_observations.tolist() == [50, 50]
    pairs = pd.read_csv(out / "paired_effects.csv")
    assert np.allclose(pairs.delta_weighted_mae, [-0.3, -0.2])
    tasks = pd.read_csv(out / "task_metrics.csv")
    assert tasks.width_90.eq(np.inf).all()
    assert tasks.interval_status_90.eq("unsupported_unbounded").all()
    for family in report.FAMILIES:
        assert (out / "figures" / f"{family}.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (out / "figures" / f"{family}.svg").read_text()
    hashes = json.loads((out / "artifact_manifest.json").read_text())["output_sha256"]
    assert all(report._hash(out / name) == digest for name, digest in hashes.items())
    assert "not confidence intervals" in (out / "index.html").read_text()


@pytest.mark.parametrize(
    "mutation", ["hash", "score", "identity", "empty_marker", "source", "seed"]
)
def test_reject_corruption(matrix, monkeypatch, mutation):
    run, out = matrix
    monkeypatch.setattr(report, "_draw_figures", lambda *args: None)
    directory = next((run / "tasks").iterdir())
    if mutation == "source":
        bindings = json.loads((run / "run_bindings.json").read_text())
        bindings["sources"] = {next(iter(bindings["sources"])): "0" * 64}
        write(run / "run_bindings.json", bindings)
    elif mutation == "empty_marker":
        write(
            directory / "complete.json",
            {"task_id": directory.name, "output_sha256": {}},
        )
    elif mutation == "seed":
        np.savez(
            directory / "seed_predictions.npz",
            row_index=np.arange(30),
            predictions=np.zeros((3, 30)),
        )
        seal(directory)
    elif mutation in ["hash", "score"]:
        meta = json.loads((directory / "metrics.json").read_text())
        meta["weighted_mae"] = 999
        write(directory / "metrics.json", meta)
        if mutation == "score":
            seal(directory)
    else:
        frame = pd.read_csv(directory / "predictions.csv")
        frame.loc[0, "record_id"] = "wrong"
        frame.to_csv(directory / "predictions.csv", index=False)
        seal(directory)
    result = report.build_report(run, out)
    assert result["failed"] >= 1
    assert result["completed"] < 4


def test_zero_results_and_failed_task(matrix):
    import shutil

    run, out = matrix
    shutil.rmtree(run / "tasks")
    result = report.build_report(run, out)
    assert result["completed"] == result["failed"] == 0
    assert result["pending"] == 8
    assert pd.read_csv(out / "metrics.csv").empty
    task = json.loads((run / "task_manifest.json").read_text())[0]
    write(
        run / "tasks" / task["task_id"] / "failure.json", {"error": "synthetic failure"}
    )
    result = report.build_report(run, out)
    assert result["failed"] == 1 and result["pending"] == 7
