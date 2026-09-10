"""Baseline tiny fits, train-only transforms, replay and fail-closed resume."""
import importlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest

b = importlib.import_module("experiments.20260906_low_data_followup.code.baselines")
r = importlib.import_module("experiments.20260906_low_data_followup.code.run_baselines")


def fixture():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(18, 12))
    x[:, 0] = np.nan
    x[:, 1] = 1
    x[::4, 2] = np.nan
    f = {"descriptors": x, "native_continuous": np.linspace(2, 5, 18),
         "groups": np.repeat(np.arange(6), 3)}
    y = 4 + .3 * x[:, 3]
    return f, y, np.linspace(.05, .6, len(y))


@pytest.mark.parametrize("method", b.METHODS)
def test_tiny_fit_pickle_replay(method):
    f, y, se = fixture()
    result = b.fit_predict(method, f, y, se, f, config={}, checkpoint_path=None)
    replay = b.predict_model(pickle.loads(pickle.dumps(result["model_state"])), f)
    np.testing.assert_array_equal(result["seed_predictions"], replay["seed_predictions"])
    assert result["seed_predictions"].shape == (3, len(y))
    assert np.isfinite(result["pred"]).all()
    np.testing.assert_array_equal(result["ensemble_std"], np.zeros(len(y)))
    assert result["fit_audit"]["actual_final_fits"] == 1
    assert not result["fit_audit"]["test_labels_accessed"]
    assert not result["fit_audit"]["calibration_labels_accessed"]


def test_weighted_lower_median():
    assert b.weighted_median(np.array([9., 1., 3.]), np.array([1., 2., 1.])) == 1.
    f = {"native_continuous": np.arange(3.)}
    result = b.fit_predict("weighted_median", f, np.array([1., 3., 9.]),
                           np.array([.1, .1, 1.]), f, config={}, checkpoint_path=None)
    np.testing.assert_array_equal(result["pred"], np.full(3, 3.))


@pytest.mark.parametrize("slope,expected", [(4., 2.), (-1., 0.), (.5, .5)])
def test_affine_bounds_weighted_intercept_no_clipping(slope, expected):
    x = np.arange(6.)
    y = 7 + slope*x
    se = np.linspace(.1, .6, 6)
    result = b.fit_predict("native_affine", {"native_continuous": x}, y, se,
        {"native_continuous": np.array([1e4])}, config={}, checkpoint_path=None)
    model = result["model_state"]["model"]
    assert model["slope"] == pytest.approx(expected)
    assert model["intercept"] == pytest.approx(np.average(y-expected*x, weights=1/se))
    if expected:
        assert result["pred"][0] > 1000


def test_affine_constant_input():
    f = {"native_continuous": np.ones(4)}
    result = b.fit_predict("native_affine", f, np.arange(4.), np.ones(4), f,
                           config={}, checkpoint_path=None)
    assert result["model_state"]["model"]["slope"] == 0
    np.testing.assert_array_equal(result["pred"], np.full(4, 1.5))


def test_instrument_inner_training_only(monkeypatch):
    f, y, se = fixture()
    seen = []
    original = b._reduce
    def watch(x, labels, weights, seed):
        seen.append((x.copy(), labels.copy(), weights.copy()))
        return original(x, labels, weights, seed)
    monkeypatch.setattr(b, "_reduce", watch)
    result = b.fit_predict(b.DESCRIPTOR_METHOD, f, y, se, f, config={}, checkpoint_path=None)
    audit = result["fit_audit"]
    assert len(seen) == 4
    for (x, labels, weights), a in zip(seen[:3], audit["inner_transform_fits"], strict=True):
        tr, va = np.array(a["train_indices"]), np.array(a["validation_indices"])
        assert not set(tr) & set(va)
        assert not set(f["groups"][tr]) & set(f["groups"][va])
        np.testing.assert_array_equal(x, f["descriptors"][tr])
        np.testing.assert_array_equal(labels, y[tr])
        w = 1/np.maximum(se[tr], .1)
        np.testing.assert_allclose(weights, w/w.mean())
        assert a["gain_label_indices"] == tr.tolist()
        assert a["fit_array_sha256"] == b._hash_array(x)
    np.testing.assert_array_equal(seen[-1][0], f["descriptors"])
    altered = {**f, "descriptors": f["descriptors"] + 1e6}
    changed = b.fit_predict(b.DESCRIPTOR_METHOD, f, y, se, altered, config={}, checkpoint_path=None)
    assert result["selection"] == changed["selection"]
    np.testing.assert_array_equal(result["pred"], b.predict_model(changed["model_state"], f)["pred"])


def test_reducer_missing_constants_and_correlation():
    x = np.array([[0., 0., np.nan, 1.], [1., 2., np.nan, 1.], [2., 4., np.nan, 1.]])
    reducer = b.TrainOnlyReducer().fit(x)
    np.testing.assert_array_equal(reducer.selected_, [1])
    assert reducer.mean_[2] == 0
    np.testing.assert_array_equal(reducer.transform([[100., np.nan, 8., 4.]]), [[2.]])
    empty = b.TrainOnlyReducer().fit(np.ones((4, 3)))
    assert empty.transform(np.ones((2, 3))).shape == (2, 0)


def test_descriptor_constant_fallback_and_label_rejection():
    f = {"descriptors": np.ones((6, 3)), "groups": np.arange(6)}
    result = b.fit_predict(b.DESCRIPTOR_METHOD, f, np.arange(6.), np.ones(6), f,
                           config={}, checkpoint_path=None)
    np.testing.assert_array_equal(result["pred"], np.full(6, 2.5))
    with pytest.raises(ValueError, match="unknown"):
        b.fit_predict(b.METHODS[0], {**f, "pEC50": np.arange(6.)}, np.arange(6.), np.ones(6), f,
                      config={}, checkpoint_path=None)
    with pytest.raises(ValueError, match="not labels"):
        b._identity(pd.DataFrame({"record_id": ["001"], "canonical_smiles": ["C"], "pEC50": [4]}))


def test_descriptor_cache_identity_and_hash_gate(tmp_path):
    inputs = pd.DataFrame({"record_id": ["001", "NA"], "canonical_smiles": ["C", "CC"]})
    x = np.ones((2, 3))
    np.savez_compressed(tmp_path / "descriptors.npz", descriptors=x)
    meta = {"ordered_identity_sha256": b._hash_json(b._identity(inputs)),
            "feature_identity_sha256": b.descriptor_schema()["feature_identity_sha256"],
            "file_sha256": b._digest(tmp_path / "descriptors.npz"), "array_sha256": b._hash_array(x),
            "shape": list(x.shape)}
    (tmp_path / "descriptor_metadata.json").write_text(json.dumps(meta))
    np.testing.assert_array_equal(b.load_descriptors(tmp_path, inputs)[0], x)
    with pytest.raises(ValueError, match="ordered identity"):
        b.load_descriptors(tmp_path, inputs.iloc[::-1])
    with (tmp_path / "descriptors.npz").open("ab") as out:
        out.write(b"corrupt")
    with pytest.raises(ValueError, match="file hash"):
        b.load_descriptors(tmp_path, inputs)


def test_completed_task_resume_and_corruption(tmp_path):
    task = {"task_id": "tiny"}
    assert not r.complete_task(tmp_path, task, "binding")
    for name in r.OUTPUTS:
        (tmp_path / name).write_text("fixture payload")
    marker = {"task": task, "binding_sha256": "binding",
              "output_sha256": {name: r.digest(tmp_path / name) for name in r.OUTPUTS}}
    (tmp_path / "complete.json").write_text(json.dumps(marker))
    assert r.complete_task(tmp_path, task, "binding")
    assert r.complete_task(tmp_path, task, "binding")
    with pytest.raises(ValueError, match="schema/task/binding"):
        r.complete_task(tmp_path, task, "changed")
    (tmp_path / "predictions.csv").write_text("changed")
    with pytest.raises(ValueError, match="corrupt completed output"):
        r.complete_task(tmp_path, task, "binding")
    marker["output_sha256"] = {}
    (tmp_path / "complete.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="schema/task/binding"):
        r.complete_task(tmp_path, task, "binding")


def test_source_protocol_input_drift_gate(tmp_path, monkeypatch):
    protocol, source, data = [tmp_path / name for name in ("protocol", "source", "input")]
    for p in (protocol, source, data):
        p.write_text("original")
    monkeypatch.setattr(r, "PROTOCOL", protocol)
    monkeypatch.setattr(r, "ROOT", tmp_path)
    bindings = {"protocol_sha256": r.digest(protocol), "config": {
        "source_sha256": {"source": r.digest(source)}, "input_sha256": {str(data): r.digest(data)}}}
    r.assert_bindings(bindings)
    for p, message in ((protocol, "protocol drift"), (source, "source drift"), (data, "input drift")):
        p.write_text("changed")
        with pytest.raises(ValueError, match=message):
            r.assert_bindings(bindings)
        p.write_text("original")


def test_exact_original_paired_manifest():
    config = json.loads(r.PROTOCOL.read_text())
    tasks = r.planned_tasks(config)
    assert len(tasks) == len({t["task_id"] for t in tasks}) == 1800
    original = json.loads((r.ORIGINAL / "protocol.json").read_text())
    old_cells = {(t["split"], t["outer_fold"], t["draw"], t["n_train"])
                 for t in r.planned_tasks(original)}
    for method in b.METHODS:
        cells = {(t["split"], t["outer_fold"], t["draw"], t["n_train"])
                 for t in tasks if t["method"] == method}
        assert cells == old_cells
    assert config["runtime_cap_seconds"] is None
    assert config["threads"] == 2
    assert config["historical_reconstruction"] is False
