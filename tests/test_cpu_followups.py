import numpy as np
import pandas as pd
import pytest
import torch

from nesso_pxr.cpu_followups import (
    _safe_tail_metrics,
    _stable_best,
    align_fold_labels_for_reporting,
    assign_standardized_folds,
    combined_head_objective,
    objective_label,
    standardize_smiles_record,
)


def test_stable_best_uses_mae_then_spearman_then_declared_order() -> None:
    frame = pd.DataFrame(
        {
            "candidate": ["first", "second", "third"],
            "candidate_order": [0, 1, 2],
            "mae": [0.5, 0.5, 0.6],
            "spearman": [0.7, 0.8, 0.9],
        }
    )
    assert _stable_best(frame, candidate_column="candidate")["candidate"] == "second"
    frame.loc[0, "spearman"] = 0.8
    assert _stable_best(frame, candidate_column="candidate")["candidate"] == "first"


def test_primary_fragment_parent_retains_charge_and_removes_counterion() -> None:
    record = standardize_smiles_record("CC[NH3+].[Cl-]")
    assert "." not in record["fragment_parent_smiles"]
    assert record["fragment_parent_formal_charge"] == 1
    assert record["charge_parent_formal_charge"] == 0


def test_standardized_connectivity_identity_is_never_split() -> None:
    smiles = [
        "C[C@H](O)F",
        "C[C@@H](O)F",
        "c1ccccc1",
        "c1ccncc1",
        "C1CCCCC1",
        "CCO",
        "CCN",
        "CC(=O)O",
        "CC#N",
        "COC",
    ]
    records = [standardize_smiles_record(value) for value in smiles]
    frame = pd.DataFrame(records)
    frame["feature_row"] = np.arange(len(frame))
    frame["original_id"] = [f"c{index}" for index in range(len(frame))]
    assigned = assign_standardized_folds(frame)
    stereo_pair = assigned.iloc[:2]
    assert stereo_pair["connectivity_inchikey"].nunique() == 1
    assert stereo_pair["standardized_fold"].nunique() == 1


def test_fold_label_alignment_removes_pure_label_permutation() -> None:
    historical = np.array([0, 0, 1, 1, 2, 2])
    permuted = np.array([2, 2, 0, 0, 1, 1])
    aligned, mapping = align_fold_labels_for_reporting(historical, permuted)
    assert np.array_equal(aligned, historical)
    assert mapping == {0: 1, 1: 2, 2: 0}


def test_combined_head_objectives_use_frozen_50_25_25_weights() -> None:
    prediction1 = torch.tensor([1.0])
    prediction2 = torch.tensor([3.0])
    ensemble = torch.tensor([2.0])
    target = torch.tensor([0.0])
    mse = combined_head_objective(
        prediction1,
        prediction2,
        ensemble,
        target,
        objective="mse",
        delta=None,
    )
    mae = combined_head_objective(
        prediction1,
        prediction2,
        ensemble,
        target,
        objective="mae",
        delta=None,
    )
    assert float(mse) == pytest.approx(4.5)
    assert float(mae) == pytest.approx(2.0)
    assert objective_label("huber", 0.5) == "huber_0.5"


def test_tail_metrics_predeclare_strictly_greater_than_six() -> None:
    metrics = _safe_tail_metrics(np.array([5.0, 6.0, 6.5]), np.array([6.1, 6.0, 6.2]))
    assert metrics["active_n"] == 1
    assert metrics["predicted_above_6"] == 2
    assert metrics["active_predicted_above_6"] == 1
