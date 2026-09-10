"""Synthetic CPU tests are engineering checks, NOT actual assay performance."""

import importlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

c = importlib.import_module("experiments.20260906_affinity_comparison.code.controls")


def fake_cache(tmp_path, monkeypatch, n=3):
    cache = tmp_path / "cache"
    protocol = c.read_json(c.EXP / "cache_protocol.json")
    protocol["selection"]["count"] = n
    monkeypatch.setattr(c, "EXP", tmp_path / "experiment")
    c.put_json(c.EXP / "cache_protocol.json", protocol)
    identity = pd.DataFrame(
        {
            "row_index": range(n),
            "record_id": [f"nesso_{i:020x}" for i in range(n)],
            "feature_row": range(n),
            "canonical_smiles": ["C" * (i + 1) for i in range(n)],
        }
    )
    rng = protocol["rng"]
    records = [
        dict(
            row_index=str(i),
            record_id=r,
            feature_row=str(i),
            canonical_smiles=identity.iloc[i].canonical_smiles,
            order=i,
            modeling_role="development",
            seed=int.from_bytes(
                c.hashlib.sha256(
                    f"{rng['namespace']}|{rng['base_seed']}|{r}".encode()
                ).digest()[:4],
                "big",
            )
            % 2147483647,
        )
        for i, r in zip(identity.row_index, identity.record_id, strict=False)
    ]
    c.put_json(cache / "stage/identity_manifest.json", records)
    stage = dict(
        schema=1,
        binding_namespace="pxr-affinity-comparison-cache-v1",
        reused={},
        snapshot_bytes=0,
        records=records,
        files={},
        identity_sources={},
        protocol=protocol,
        kind="full_development_native_cache",
        challenge_label_training=False,
        numeric_outcomes_copied=False,
        identity_manifest_sha256=c.digest(cache / "stage/identity_manifest.json"),
    )
    stage["binding"] = c.object_hash(stage)
    c.put_json(cache / "stage/manifest.json", stage)
    for rec in records:
        directory = cache / "capture" / rec["record_id"]
        directory.mkdir(parents=True)
        outputs = []
        for i, member in enumerate(c.MEMBERS):
            out = dict(
                affinity_repr=torch.arange(384, dtype=torch.float32).reshape(1, 384)
                + i,
                affinity_pred_value=torch.tensor([0.12345678 + i], dtype=torch.float32),
                affinity_logits_binary=torch.tensor([0.35], dtype=torch.float32),
            )
            outputs.append(out)
            torch.save(
                dict(
                    kwargs=dict(
                        s_inputs=torch.zeros(1),
                        z=torch.zeros(1),
                        pdistogram=torch.zeros(1),
                        feats={},
                        use_kernels=True,
                    ),
                    removed_bookkeeping=[],
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.get_rng_state(),
                    native_output=out,
                    autocast_enabled=False,
                ),
                directory / (member + ".pt"),
            )
        torch.save(
            dict(
                affinity_pred_value=(
                    outputs[0]["affinity_pred_value"]
                    + outputs[1]["affinity_pred_value"]
                )
                / 2
            ),
            directory / "native_prediction.pt",
        )
        files = {
            p.name: dict(bytes=p.stat().st_size, sha256=c.digest(p))
            for p in directory.iterdir()
        }
        c.put_json(
            directory / "manifest.json",
            dict(
                binding=stage["binding"],
                record=rec,
                files=files,
                kind="real_native_both_member_capture",
                challenge_label_training=False,
            ),
        )
    return cache, identity


def test_fresh_native_fp32_resume_and_corruption(tmp_path, monkeypatch):
    cache, identity = fake_cache(tmp_path, monkeypatch)
    target = c.build_features(cache, tmp_path / "run", identity)
    assert c.read_json(target / "manifest.json")["outcome_labels_read"] is False
    with np.load(target / "features.npz") as f:
        assert f["member1"].shape == (3, 384)
        assert f["scalar_features"].shape == (3, 4)
        np.testing.assert_array_equal(
            f["member1"], np.tile(np.arange(384, dtype=np.float32), (3, 1))
        )
        np.testing.assert_array_equal(f["member2"], f["member1"] + 1)
    expected = float(6 - (torch.tensor(0.12345678) + torch.tensor(1.12345678)) / 2)
    assert np.array_equal(np.load(target / "native_points.npy"), np.repeat(expected, 3))
    hashes = {p.name: c.digest(p) for p in target.iterdir()}
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("resume deserialized large payload")
    )
    c.build_features(cache, tmp_path / "run", identity)
    assert hashes == {p.name: c.digest(p) for p in target.iterdir()}
    p = cache / "capture" / identity.record_id.iloc[0] / "affinity_module.pt"
    p.write_bytes(p.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupt bound"):
        c.build_features(cache, tmp_path / "run", identity)


def test_identity_and_incomplete_cache_fail_before_features(tmp_path, monkeypatch):
    cache, identity = fake_cache(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="ordered cache"):
        c.build_features(cache, tmp_path / "run", identity.iloc[::-1])
    (cache / "capture" / identity.record_id.iloc[0] / "native_prediction.pt").unlink()
    with pytest.raises(ValueError, match="payload set"):
        c.build_features(cache, tmp_path / "run", identity)
    assert not (tmp_path / "run/fresh_features").exists()


def synthetic_problem():
    rng = np.random.default_rng(891)
    features = {
        k: rng.normal(size=(36, 384)).astype(np.float32) for k in ("member1", "member2")
    }
    labels = pd.DataFrame(
        dict(
            row_index=range(36),
            record_id=[f"synthetic_{i}" for i in range(36)],
            pEC50=rng.normal(5, 1, 36),
            pEC50_standard_error=np.repeat(0.2, 36),
        )
    )
    cell = dict(
        outer_fold=0,
        draw=0,
        n_train=28,
        roles=dict(
            fit=np.arange(22), calibration=np.arange(22, 28), test=np.arange(28, 36)
        ),
    )
    return features, labels, cell


def test_fresh_ridge_real_fit_no_leakage_and_disk_replay(tmp_path, monkeypatch):
    features, labels, cell = synthetic_problem()
    readouts = importlib.import_module(c.PREFIX + "readouts")
    original_fit = readouts.fit_predict
    captured = []
    preprocessing = []
    original_preprocess = readouts.fit_preprocessing

    def spy_preprocess(x, w, scaled):
        preprocessing.append(x.copy())
        return original_preprocess(x, w, scaled)

    monkeypatch.setattr(readouts, "fit_preprocessing", spy_preprocess)

    def spy(method, train, y, se, evaluate, **kwargs):
        np.testing.assert_array_equal(y, labels.iloc[cell["roles"]["fit"]].pEC50)
        assert set(train) == {"member1", "member2", "groups"}
        assert kwargs["config"]["inner_folds"] == 3
        captured.append(len(y))
        return original_fit(method, train, y, se, evaluate, **kwargs)

    monkeypatch.setattr(readouts, "fit_predict", spy)
    native = np.linspace(4, 6, 36)
    for method in (c.RIDGE, c.NATIVE):
        c.run_cell(
            tmp_path,
            cell,
            method,
            features,
            native,
            labels,
            np.arange(36),
            "synthetic-binding",
        )
    assert captured == [22]
    payloads = {
        str(p.relative_to(tmp_path)): c.digest(p)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    audit = c.read_json(
        tmp_path / "cells" / c.task_id(cell, c.RIDGE) / "fit_audit.json"
    )["audit"]
    x = np.concatenate(
        [features[k][cell["roles"]["fit"]] for k in ("member1", "member2")], axis=1
    )
    assert len(preprocessing) == len(audit["preprocessing_fits"]) == 13
    for seen, entry in zip(preprocessing, audit["preprocessing_fits"], strict=False):
        np.testing.assert_array_equal(seen, x[entry["train_indices"]])
        if entry["scope"] == "inner_train":
            assert not set(entry["train_indices"]) & set(entry["validation_indices"])
    for method in (c.RIDGE, c.NATIVE):
        c.run_cell(
            tmp_path,
            cell,
            method,
            features,
            native,
            labels,
            np.arange(36),
            "synthetic-binding",
        )
    assert captured == [22]
    assert payloads == {
        str(p.relative_to(tmp_path)): c.digest(p)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    state = pickle.loads(
        (tmp_path / "cells" / c.task_id(cell, c.RIDGE) / "model_state.pkl").read_bytes()
    )
    assert state["seeds"] == (42,)
    with pytest.raises(ValueError, match="foreign controls"):
        c.run_cell(
            tmp_path, cell, c.RIDGE, features, native, labels, np.arange(36), "changed"
        )


def test_calibration_changes_intervals_not_native_points(tmp_path):
    features, labels, cell = synthetic_problem()
    pred = np.arange(14, dtype=float) / 10 + 4
    frame1, m1 = c.scores_and_intervals(
        labels, cell["roles"]["calibration"], cell["roles"]["test"], pred
    )
    altered = labels.copy()
    altered.loc[cell["roles"]["calibration"], "pEC50"] += 100
    frame2, m2 = c.scores_and_intervals(
        altered, cell["roles"]["calibration"], cell["roles"]["test"], pred
    )
    np.testing.assert_array_equal(frame1.y_pred, frame2.y_pred)
    assert m1["radii"] != m2["radii"]
    assert m1["test"]["weighted_mae"] == m2["test"]["weighted_mae"]


def test_real_strict_ten_exact_cells_and_descriptor_reuse_read_only(monkeypatch):
    if not (c.STRICT / "panel/run_bindings.json").exists():
        pytest.skip("real historical artifacts unavailable")
    identity, assignments, cells = c.strict_identity()
    assert len(identity) == 3344
    assert {(x["outer_fold"], x["draw"], x["n_train"]) for x in cells} == {
        (f, 0, n) for f in range(5) for n in (100, 500)
    }
    baseline = importlib.import_module(c.PREFIX + "baselines")
    monkeypatch.setattr(
        baseline,
        "fit_predict",
        lambda *a, **kw: pytest.fail("historical reuse attempted refit"),
    )
    result = c.workflow("verify-history", None, None)
    assert (
        result["verified_descriptor_cells"] == 10
        and result["new_fits"] == 0
        and result["read_only"] is True
    )
    assert {x["task_id"] for x in result["verified_tasks"]} == {
        c.task_id(cell, c.DESCRIPTOR) for cell in cells
    }
    assert all(len(x["complete_sha256"]) == 64 for x in result["verified_tasks"])


@pytest.mark.parametrize(
    "mutation",
    ["labels", "feature_row", "duplicate", "record_label", "stage_shape", "reused"],
)
def test_cache_identity_schema_and_reuse_rejected(tmp_path, monkeypatch, mutation):
    cache, identity = fake_cache(tmp_path, monkeypatch)
    stage = c.read_json(cache / "stage/manifest.json")
    if mutation == "labels":
        identity["pEC50"] = 5.0
    elif mutation == "feature_row":
        identity.loc[0, "feature_row"] = 99
    elif mutation == "duplicate":
        identity.loc[1, "record_id"] = identity.loc[0, "record_id"]
    elif mutation in ("record_label", "stage_shape"):
        if mutation == "record_label":
            stage["records"][0]["pEC50"] = 5.0
        else:
            stage["unexpected"] = True
        stage.pop("binding")
        stage["binding"] = c.object_hash(stage)
        c.put_json(cache / "stage/manifest.json", stage)
    else:
        path = cache / "capture" / identity.record_id.iloc[0] / "manifest.json"
        packet = c.read_json(path)
        packet["reused_from"] = {"binding": "undeclared"}
        c.put_json(path, packet)
    with pytest.raises(ValueError):
        c.build_features(cache, tmp_path / "run", identity)
    assert not (tmp_path / "run/fresh_features").exists()


def test_valid_pilot_reuse_and_feature_hash_corruption(tmp_path, monkeypatch):
    cache, identity = fake_cache(tmp_path, monkeypatch)
    stage = c.read_json(cache / "stage/manifest.json")
    rid = identity.record_id.iloc[0]
    source = cache / "capture" / rid / "manifest.json"
    old = c.read_json(source)
    proof = dict(
        binding=old["binding"],
        record=old["record"],
        manifest_sha256=c.digest(source),
        manifest=old,
        origin="fixture/capture/" + rid,
        source_native_filename="native_prediction.pt",
        target_native_filename="native_prediction.pt",
    )
    stage["reused"] = {rid: proof}
    stage.pop("binding")
    stage["binding"] = c.object_hash(stage)
    c.put_json(cache / "stage/manifest.json", stage)
    for record in stage["records"]:
        path = cache / "capture" / record["record_id"] / "manifest.json"
        meta = c.read_json(path)
        meta["binding"] = stage["binding"]
        if record["record_id"] == rid:
            meta["reused_from"] = {k: v for k, v in proof.items() if k != "manifest"}
        c.put_json(path, meta)
    target = c.build_features(cache, tmp_path / "run", identity)
    c.build_features(cache, tmp_path / "run", identity)
    (target / "features.npz").write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="checksum"):
        c.build_features(cache, tmp_path / "run", identity)


def test_source_freeze_resume_and_drift(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("# fixture source\n")
    strict, out = tmp_path / "strict", tmp_path / "run"
    c.put_json(strict / "panel/run_bindings.json", {"fixture": True})
    c.put_json(out / "fresh_features/manifest.json", {"fixture": True})
    monkeypatch.setattr(c, "source_paths", lambda *a: [source, c.PROTOCOL])
    first = c.freeze_sources(out, strict, tmp_path, tmp_path)
    assert first == c.freeze_sources(out, strict, tmp_path, tmp_path)
    source.write_text("# changed fixture source\n")
    with pytest.raises(ValueError, match="drift"):
        c.freeze_sources(out, strict, tmp_path, tmp_path)
    source.write_text("# fixture source\n")
    (out / "source_snapshot/000_source.py").write_text("# corrupt snapshot\n")
    with pytest.raises(ValueError, match="snapshot mismatch"):
        c.freeze_sources(out, strict, tmp_path, tmp_path)


def test_only_member_features_and_exact_draw_are_accepted(tmp_path):
    features, labels, cell = synthetic_problem()
    with pytest.raises(ValueError, match="not labels"):
        c.run_cell(
            tmp_path,
            cell,
            c.RIDGE,
            {**features, "pEC50": labels.pEC50.to_numpy()},
            np.zeros(36),
            labels,
            np.arange(36),
            "fixture",
        )
    with pytest.raises(ValueError, match="unsupported control cell"):
        c.task_id({**cell, "draw": 1}, c.NATIVE)
    paths = c.source_paths(c.STRICT, c.ORIGINAL, tmp_path)
    assert c.PROTOCOL in paths and Path(c.__file__) in paths
    assert not any(
        p.name in ("train.py", "report.py", "low_data_report.py") for p in paths
    )
