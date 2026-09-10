"""Deterministic, synthetic model-factory contracts (not scientific results)."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from nesso_pxr.low_data_models import (
    METHODS,
    InfeasibleInnerCV,
    fit_predict,
    load_model,
    make_neural_model,
    predict_model,
    save_model,
    weighted_smooth_l1,
)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    torch.set_num_threads(2)
    rng = np.random.default_rng(8)
    n = 16
    x1 = rng.normal(size=(n, 384)).astype("float32")
    x2 = rng.normal(size=(n, 384)).astype("float32")
    features = dict(
        member1=x1,
        member2=x2,
        morgan=rng.integers(0, 2, (n, 32)).astype("float32"),
        native_continuous=5 + x1[:, 0],
        scalar_features=x1[:, :6],
    )
    y = 5 + x1[:, 0] * 0.5
    se = np.linspace(0.05, 0.8, n)
    config = dict(
        se_floor=0.1,
        inner_folds=2,
        epochs=2,
        batch_size=64,
        threads=2,
        neural_huber_beta=0.5,
        head_lora_rank=4,
        grids=dict(
            ridge_alpha=[1.0, 100.0, 10000.0],
            neural_learning_rate=[0.0001, 0.001],
            lightgbm_n_estimators=[2, 3],
            knn_neighbors=[1, 5],
        ),
    )
    torch.manual_seed(4)
    state = {}
    for module in ("affinity_module", "affinity_module2"):
        for branch in ("to_affinity_pred_value", "to_affinity_pred_score"):
            net = torch.nn.Sequential(
                torch.nn.Linear(384, 384),
                torch.nn.ReLU(),
                torch.nn.Linear(384, 384),
                torch.nn.ReLU(),
                torch.nn.Linear(384, 1),
            )
            state.update(
                {
                    f"{module}.affinity_heads.{branch}.{k}": v
                    for k, v in net.state_dict().items()
                }
            )
        net = torch.nn.Linear(1, 1)
        state.update(
            {
                f"{module}.affinity_heads.to_affinity_logits_binary.{k}": v
                for k, v in net.state_dict().items()
            }
        )
    checkpoint = tmp_path_factory.mktemp("checkpoint") / "model.safetensors"
    save_file(state, str(checkpoint))
    return features, y, se, config, checkpoint


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_deterministic_finite_and_reload(method, fixture, tmp_path):
    features, y, se, config, checkpoint = fixture
    train = {k: v[:12] for k, v in features.items()}
    test = {k: v[12:] for k, v in features.items()}

    def run(test_features):
        return fit_predict(
            method,
            train,
            y[:12],
            se[:12],
            test_features,
            config=config,
            checkpoint_path=checkpoint,
            seeds=(42, 43),
        )

    result = run(test)
    repeat = run(test)
    assert result["pred"].shape == (4,)
    assert result["seed_predictions"].shape == (2, 4)
    assert np.isfinite(result["pred"]).all()
    np.testing.assert_array_equal(result["pred"], repeat["pred"])
    np.testing.assert_allclose(result["pred"], result["seed_predictions"].mean(axis=0))
    json.dumps(result["selection"], allow_nan=False)
    json.dumps(result["fit_audit"], allow_nan=False)
    assert result["fit_audit"]["serialization_verified"]
    path = tmp_path / "model.pkl"
    save_model(result["model_state"], path)
    restored = predict_model(load_model(path), test)
    for key in ("pred", "seed_predictions", "ensemble_std"):
        np.testing.assert_array_equal(restored[key], result[key])
    # Parent runner writes model_state directly with pickle, not save_model.
    direct = pickle.loads(pickle.dumps(result["model_state"]))
    bounded = predict_model(direct, test, batch_size=2)
    np.testing.assert_allclose(bounded["pred"], result["pred"], atol=2e-6, rtol=1e-6)
    assert predict_model(direct, {k: v[:0] for k, v in test.items()})[
        "seed_predictions"
    ].shape == (2, 0)
    assert result["peak_vram_bytes"] == 0
    # Transductive/test-feature changes must not alter selection or fitted state.
    changed = {k: v[::-1].copy() for k, v in test.items()}
    alternate = run(changed)
    assert alternate["selection"] == result["selection"]
    np.testing.assert_allclose(alternate["pred"][::-1], result["pred"], atol=1e-6)


def test_weighted_mean_and_invalid_errors(fixture):
    features, y, se, config, checkpoint = fixture
    result = fit_predict(
        "weighted_mean",
        features,
        y,
        se,
        features,
        config=config,
        checkpoint_path=checkpoint,
    )
    expected = np.average(y, weights=1 / np.maximum(se, 0.1))
    np.testing.assert_allclose(result["pred"], expected)
    for bad in (0, -1, np.nan, np.inf):
        invalid = se.copy()
        invalid[0] = bad
        with pytest.raises(ValueError):
            fit_predict(
                "weighted_mean",
                features,
                y,
                invalid,
                features,
                config=config,
                checkpoint_path=checkpoint,
            )


def test_group_cv_no_leakage_and_infeasible(fixture):
    features, y, se, config, checkpoint = fixture
    grouped = {**features, "groups": np.repeat(np.arange(4), 4)}
    result = fit_predict(
        "repr_ridge",
        grouped,
        y,
        se,
        features,
        config=config,
        checkpoint_path=checkpoint,
    )
    for fold in result["selection"]["folds"]:
        a = set(grouped["groups"][fold["train_indices"]])
        b = set(grouped["groups"][fold["validation_indices"]])
        assert not a & b
    with pytest.raises(InfeasibleInnerCV, match="distinct groups"):
        fit_predict(
            "repr_ridge",
            {**features, "groups": np.zeros(len(y))},
            y,
            se,
            features,
            config=config,
            checkpoint_path=checkpoint,
        )


def test_twins_architecture_parity_and_zero_update(fixture):
    features, _, _, config, checkpoint = fixture
    random = make_neural_model("head_random", config, checkpoint, seed=42)
    pretrained = make_neural_model("head_pretrained", config, checkpoint, seed=42)
    assert [(k, tuple(v.shape)) for k, v in random.state_dict().items()] == [
        (k, tuple(v.shape)) for k, v in pretrained.state_dict().items()
    ]
    assert not torch.equal(random.member1[0].weight, pretrained.member1[0].weight)
    x = [torch.from_numpy(features[k]) for k in ("member1", "member2")]
    with torch.no_grad():
        expected = pretrained(*x)[2]
        for method in ("dual_head", "head_lora"):
            model = make_neural_model(method, config, checkpoint, seed=42)
            torch.testing.assert_close(model(*x)[2], expected, rtol=0, atol=0)


def test_lora_gradients_frozen_invariance(fixture):
    features, y, _, config, checkpoint = fixture
    model = make_neural_model("head_lora", config, checkpoint, seed=42)
    before = {k: v.clone() for k, v in model.named_parameters()}
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001
    )
    x = [torch.from_numpy(features[k]) for k in ("member1", "member2")]
    loss = weighted_smooth_l1(
        model(*x), torch.from_numpy(y), torch.ones(len(y)), beta=0.5
    )
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    opt.step()
    assert any(
        not torch.equal(p, before[k])
        for k, p in model.named_parameters()
        if p.requires_grad
    )
    for k, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, before[k])


def test_weighted_smooth_l1_exact():
    predictions = tuple(torch.tensor([0.0, 2.0]) for _ in range(3))
    target = torch.zeros(2)
    weights = torch.tensor([3.0, 1.0])
    expected = torch.nn.functional.smooth_l1_loss(
        predictions[0], target, beta=0.5, reduction="none"
    )
    torch.testing.assert_close(
        weighted_smooth_l1(predictions, target, weights, beta=0.5),
        (expected * weights).sum() / weights.sum(),
    )


def test_protocol_coverage():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "experiments/20260905_low_data_adaptation/protocol.json").read_text()
    )
    assert set(METHODS) == set(protocol["methods"])


@pytest.mark.parametrize("method", ["scalar_ridge", "repr_ridge", "repr_mlp"])
def test_inner_scaler_fit_rows_only(method, fixture, monkeypatch):
    import nesso_pxr.low_data_models as module

    features, y, se, config, checkpoint = fixture
    observed = []
    original = module.StandardScaler.fit

    def track(self, x, *args, **kwargs):
        fitted = original(self, x, *args, **kwargs)
        observed.append((x.copy(), kwargs["sample_weight"].copy(), self.mean_.copy()))
        return fitted

    monkeypatch.setattr(module.StandardScaler, "fit", track)
    result = fit_predict(
        method,
        features,
        y,
        se,
        {k: v[::-1] * 100 for k, v in features.items()},
        config=config,
        checkpoint_path=checkpoint,
        seeds=(42,),
    )
    matrix = module._matrix(features, method)
    folds = result["selection"]["folds"]
    grid = config["grids"][
        "neural_learning_rate" if method == "repr_mlp" else "ridge_alpha"
    ]
    expected_indices = [f["train_indices"] for _ in grid for f in folds]
    expected_indices.append(list(range(len(y))))
    assert len(observed) == len(expected_indices)
    for (x, weights, mean), indices in zip(observed, expected_indices, strict=True):
        np.testing.assert_array_equal(x, matrix[indices])
        expected_w = 1 / np.maximum(se[indices], config["se_floor"])
        np.testing.assert_allclose(weights, expected_w / expected_w.mean())
        np.testing.assert_allclose(
            mean, np.average(matrix[indices], axis=0, weights=expected_w), atol=1e-12
        )


@pytest.mark.parametrize(
    "method,trainable,total",
    [
        ("repr_mlp", 98561, 98561),
        ("head_random", 592130, 592130),
        ("head_pretrained", 592130, 592130),
        ("dual_head", 1184268, 1184268),
        ("head_lora", 15368, 607498),
    ],
)
def test_exact_parameter_branches(method, trainable, total, fixture):
    _, _, _, config, checkpoint = fixture
    model = make_neural_model(method, config, checkpoint)
    assert sum(p.numel() for p in model.parameters()) == total
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == trainable
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (method != "head_lora" or "lora_" in name)


def test_dual_binary_branch_receives_gradients_after_gate_update(fixture):
    features, y, _, config, checkpoint = fixture
    model = make_neural_model("dual_head", config, checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    x = [torch.from_numpy(features[k]) for k in ("member1", "member2")]
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        weighted_smooth_l1(
            model(*x), torch.from_numpy(y), torch.ones(len(y))
        ).backward()
        optimizer.step()
    for member in ("member1", "member2"):
        for branch in (
            "continuous",
            "binary_score",
            "binary_logit",
            "binary_scale",
            "binary_offset",
        ):
            assert any(
                not torch.equal(p, before[k])
                for k, p in model.named_parameters()
                if k.startswith(f"{member}.{branch}")
            )


def test_replay_rejects_bad_artifact_and_batch_size(fixture):
    features, y, se, config, checkpoint = fixture
    result = fit_predict(
        "weighted_mean",
        features,
        y,
        se,
        features,
        config=config,
        checkpoint_path=checkpoint,
    )
    for size in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="batch_size"):
            predict_model(result["model_state"], features, batch_size=size)
    with pytest.raises(ValueError, match="artifact"):
        predict_model({"schema_version": 99}, features)


def test_native_ignores_target_values_and_replays_minimal_features(fixture):
    features, y, se, config, checkpoint = fixture
    outputs = [
        fit_predict(
            "native_continuous",
            features,
            labels,
            errors,
            {"native_continuous": features["native_continuous"]},
            config=config,
            checkpoint_path=checkpoint,
        )
        for labels, errors in ((y, se), (y + 100, se * 100))
    ]
    for output in outputs:
        np.testing.assert_array_equal(output["pred"], features["native_continuous"])
        assert not output["fit_audit"]["target_label_training"]
        assert output["n_parameters"] == 0
    with pytest.raises(ValueError, match="missing required"):
        predict_model(outputs[0]["model_state"], {"member1": features["member1"]})
    with pytest.raises(ValueError, match="possibly labeled"):
        predict_model(outputs[0]["model_state"], {**features, "pEC50": y})
