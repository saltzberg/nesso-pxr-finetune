"""Tests use explicit synthetic fixtures; no fixture is a scientific result."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nesso_pxr.chemistry import morgan_fingerprints
from nesso_pxr.low_data_contract import (
    _development_inputs,
    _make_assignments,
    _make_subsets,
    _validate_tables,
    assay_weights,
    load_prepared,
    score_predictions,
    weighted_mae,
)
from nesso_pxr.protocol import sha256_file


@pytest.mark.parametrize("bad", [[0.0], [-1.0], [np.nan], [np.inf], [], [[0.1]]])
def test_invalid_errors_fail(bad):
    with pytest.raises(ValueError):
        assay_weights(bad)


@pytest.mark.parametrize("floor", [0, -1, np.nan, np.inf])
def test_invalid_floor_fails(floor):
    with pytest.raises(ValueError):
        assay_weights([0.1], floor)


def test_weights_and_normalized_mae():
    np.testing.assert_allclose(assay_weights([0.01, 0.1, 0.2]), [10, 10, 5])
    assert weighted_mae([0, 0, 0], [1, 2, 4], [0.01, 0.1, 0.2]) == 2.0
    scores = score_predictions([0, 0, 0], [1, 2, 4], [0.01, 0.1, 0.2])
    assert scores["spearman"] is None
    assert scores["effective_n"] == pytest.approx(25**2 / 225)
    assert scores["bias"] == pytest.approx(7 / 3)
    assert scores["weighted_mae_floor_005"] == pytest.approx(60 / 35)
    assert scores["weighted_mae_floor_020"] == pytest.approx(7 / 3)


@pytest.mark.parametrize(
    "y,pred,se",
    [
        ([1], [1, 2], [0.1]),
        ([np.nan], [1], [0.1]),
        ([1], [np.inf], [0.1]),
        ([[1]], [1], [0.1]),
    ],
)
def test_metrics_reject_misalignment(y, pred, se):
    with pytest.raises(ValueError):
        score_predictions(y, pred, se)


def toy_inputs():
    return pd.DataFrame(
        {
            "row_index": np.arange(30),
            "record_id": [f"fixture_{i}" for i in range(30)],
            "feature_row": np.arange(30) * 2,
            "canonical_smiles": ["C" * (i + 1) for i in range(30)],
            "pEC50": np.arange(30, dtype=float),
            "pEC50_standard_error": np.full(30, 0.1),
        }
    )


def toy_config():
    return dict(
        n_folds=5,
        split_seed=123,
        split_policies=["random", "chemical_cluster"],
        subset_seed=1701,
        draws=2,
        budgets=[5, 10, 20],
        include_full_reference=True,
        calibration_fraction=0.2,
    )


def test_exact_nested_paired_budgets_and_disjointness():
    inputs, config = toy_inputs(), toy_config()
    groups = np.repeat(np.arange(10), 3)
    assignments = _make_assignments(inputs, groups, config)
    subsets = _make_subsets(assignments, config)
    _validate_tables(inputs, assignments, subsets, config)
    pd.testing.assert_frame_equal(
        assignments, _make_assignments(inputs, groups, config)
    )
    pd.testing.assert_frame_equal(subsets, _make_subsets(assignments, config))
    chem = assignments[assignments.split.eq("chemical_cluster")]
    assert chem.groupby("chemical_group").outer_fold.nunique().max() == 1
    for _, block in subsets.groupby(["split", "outer_fold", "draw"]):
        previous = set()
        for n in [5, 10, 20, -1]:
            acquired = block[block.n_train.eq(n)]
            assert previous <= set(acquired.row_index)
            previous = set(acquired.row_index)
            assert len(acquired) == (24 if n == -1 else n)
            assert sum(acquired.role.eq("calibration")) == int(
                np.ceil(0.2 * len(acquired))
            )


def test_budget_overflow_and_group_infeasibility_fail():
    inputs, config = toy_inputs(), toy_config()
    with pytest.raises(ValueError, match="groups"):
        _make_assignments(inputs, np.zeros(30, dtype=int), config)
    assignments = _make_assignments(inputs, np.arange(30), config)
    config["budgets"] = [25]
    with pytest.raises(ValueError, match="budget"):
        _make_subsets(assignments, config)


def test_table_validation_catches_leakage_and_duplicates():
    inputs, config = toy_inputs(), toy_config()
    assignments = _make_assignments(inputs, np.arange(30), config)
    subsets = _make_subsets(assignments, config)
    test_row = assignments.query(
        "split == 'random' and outer_fold == 0"
    ).row_index.iloc[0]
    corrupted = subsets.copy()
    selected = corrupted.query("split == 'random' and outer_fold == 0").index[0]
    corrupted.loc[selected, "row_index"] = test_row
    with pytest.raises(ValueError):
        _validate_tables(inputs, assignments, corrupted, config)
    with pytest.raises(ValueError):
        _validate_tables(
            inputs, assignments, pd.concat([subsets, subsets.iloc[:1]]), config
        )


def test_development_selection_uses_feature_indices_and_rejects_duplicate_identity():
    manifest = (
        toy_inputs()
        .drop(columns="row_index")
        .rename(columns={"canonical_smiles": "identity_canonical_smiles"})
    )
    manifest["modeling_role"] = "development"
    manifest.loc[0, "modeling_role"] = "lockbox"
    result = _development_inputs(
        manifest.sample(frac=1, random_state=3), expected_rows=29
    )
    np.testing.assert_array_equal(result.feature_row, np.arange(1, 30) * 2)
    np.testing.assert_array_equal(result.row_index, np.arange(29))
    manifest.loc[2, "identity_canonical_smiles"] = manifest.loc[
        1, "identity_canonical_smiles"
    ]
    with pytest.raises(ValueError, match="identity"):
        _development_inputs(manifest, expected_rows=29)


@pytest.fixture(scope="module")
def real_prepared():
    root = Path(__file__).resolve().parents[1]
    prepared = root / "artifacts/experiments/low_data_20260905/prepared"
    if not (prepared / "preparation_manifest.json").exists():
        pytest.skip("real low-data artifacts have not been prepared")
    return root, prepared, load_prepared(prepared)


def test_real_artifacts_reconcile_all_rows_hashes_and_fresh_morgan(real_prepared):
    from safetensors.numpy import load_file

    root, prepared, (inputs, features, assignments, subsets) = real_prepared
    manifest = json.loads((prepared / "preparation_manifest.json").read_text())
    assert len(inputs) == 3344
    assert len(assignments) == 6688
    assert len(subsets) == 360020
    assert subsets.groupby(["split", "outer_fold", "draw", "n_train"]).ngroups == 600
    for entry in manifest["sources"].values():
        assert sha256_file(root / entry["path"]) == entry["sha256"]
    for path, digest in manifest["code_sha256"].items():
        assert sha256_file(root / path) == digest
    original = pd.read_csv(root / manifest["sources"]["manifest"]["path"])
    dev = original[original.modeling_role.eq("development")].sort_values("feature_row")
    np.testing.assert_array_equal(inputs.record_id, dev.record_id)
    np.testing.assert_array_equal(inputs.feature_row, dev.feature_row)
    cached = load_file(str(root / manifest["sources"]["features"]["path"]))
    for member in (1, 2):
        np.testing.assert_array_equal(
            features[f"member{member}"],
            cached[f"member{member}_affinity_repr"][inputs.feature_row],
        )
    fresh = np.stack(
        [
            np.asarray(fp, dtype=np.uint8)
            for fp in morgan_fingerprints(inputs.canonical_smiles.tolist())
        ]
    )
    np.testing.assert_array_equal(features["morgan"], fresh)
    audit = json.loads((prepared / "split_audit.json").read_text())
    assert sum(row["n_test"] for row in audit["folds"]) == len(inputs) * 2
    assert all(row["identity_overlap"] == 0 for row in audit["folds"])
    chemical = [row for row in audit["folds"] if row["split"] == "chemical_cluster"]
    assert all(row["chemical_group_overlap"] == 0 for row in chemical)
    # Honest audit: Butina is not a strict cross-split similarity exclusion.
    assert sum(row["nearest_similarity_ge_cutoff_count"] for row in chemical) > 0


def test_real_scalar_branches_match_independent_reconstruction(real_prepared):
    import torch
    from safetensors import safe_open

    from scripts.reconstruct_nesso_binary_head import reconstruct_variant

    root, prepared, (inputs, features, _, _) = real_prepared
    manifest = json.loads((prepared / "preparation_manifest.json").read_text())
    state = {}
    with safe_open(
        str(root / manifest["sources"]["checkpoint"]["path"]),
        framework="pt",
        device="cpu",
    ) as checkpoint:
        for key in checkpoint.keys():
            if ".affinity_heads.to_affinity_" in key:
                state[key] = checkpoint.get_tensor(key)
    outputs = reconstruct_variant(
        torch.from_numpy(features["member1"]),
        torch.from_numpy(features["member2"]),
        state,
    )
    expected = np.column_stack(
        [
            6 - outputs["member1_affinity_pred_value"],
            6 - outputs["member2_affinity_pred_value"],
            6 - outputs["affinity_pred_value"],
            outputs["member1_binary_logit"],
            outputs["member2_binary_logit"],
            outputs["binary_logit"],
            outputs["member1_binder_probability"],
            outputs["member2_binder_probability"],
            outputs["binder_probability"],
        ]
    )
    np.testing.assert_allclose(features["scalar_features"], expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        features["native_continuous"], expected[:, 2], rtol=0, atol=1e-6
    )
    assert len(expected) == 3344
    assert manifest["model_fits"] == 0
    assert not manifest["native_replay"]["binary_reference_available"]


def test_load_prepared_rejects_artifact_tampering(real_prepared, tmp_path):
    _, prepared, _ = real_prepared
    copied = tmp_path / "prepared"
    shutil.copytree(prepared, copied)
    with (copied / "inputs.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_prepared(copied)


def test_load_prepared_rejects_manifest_tampering(real_prepared, tmp_path):
    _, prepared, _ = real_prepared
    copied = tmp_path / "prepared"
    shutil.copytree(prepared, copied)
    with (copied / "preparation_manifest.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_prepared(copied)
