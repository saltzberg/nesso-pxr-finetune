from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from nesso_pxr.train_cached import (
    CachedHeadConfig,
    combined_huber_loss,
    fit_cached_fold,
    load_pretrained_heads,
    make_twin_heads,
    regression_metrics,
)


def _write_head_checkpoint(path: Path, model: torch.nn.Module) -> None:
    prefixes = {
        "member1": "affinity_module.affinity_heads.to_affinity_pred_value.",
        "member2": "affinity_module2.affinity_heads.to_affinity_pred_value.",
    }
    state = {}
    for member_name, prefix in prefixes.items():
        for key, value in getattr(model, member_name).state_dict().items():
            state[prefix + key] = value.detach().contiguous()
    save_file(state, path)


def test_cached_head_config_requires_normalized_loss_weights() -> None:
    with pytest.raises(ValueError):
        CachedHeadConfig(ensemble_loss_weight=0.5, member_loss_weight=0.2)


def test_combined_huber_loss_uses_frozen_50_25_25_contract() -> None:
    config = CachedHeadConfig()
    loss = combined_huber_loss(
        torch.tensor([1.0]),
        torch.tensor([3.0]),
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        config,
    )
    assert float(loss) == pytest.approx(0.875)


def test_regression_metrics_are_exact_for_perfect_predictions() -> None:
    values = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(values, values)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)


def test_pretrained_head_loader_round_trips_both_members(tmp_path: Path) -> None:
    torch.manual_seed(7)
    original = make_twin_heads()
    checkpoint = tmp_path / "model.safetensors"
    _write_head_checkpoint(checkpoint, original)
    restored = make_twin_heads()
    load_pretrained_heads(restored, checkpoint)
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_fit_cached_fold_learns_shared_offset_on_synthetic_features(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    count = 24
    member1 = torch.randn(count, 384)
    member2 = torch.randn(count, 384)
    initial = make_twin_heads()
    checkpoint = tmp_path / "model.safetensors"
    _write_head_checkpoint(checkpoint, initial)
    with torch.inference_mode():
        _, _, base = initial(member1, member2)

    frame = pd.DataFrame(
        {
            "feature_row": np.arange(count),
            "record_id": [f"record_{index}" for index in range(count)],
            "original_id": [f"original_{index}" for index in range(count)],
            "fold": [0] * 6 + [1] * 18,
            "pEC50": 6.0 - (base.numpy() + 0.25),
        }
    )
    config = CachedHeadConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=6,
        max_epochs=15,
        patience=5,
        min_delta=0.0,
        seed=3,
    )
    predictions, history, checkpoint_info = fit_cached_fold(
        member1,
        member2,
        frame,
        checkpoint,
        validation_fold=0,
        config=config,
        device="cpu",
    )
    assert len(predictions) == 6
    assert not history.empty
    assert 1 <= checkpoint_info["best_epoch"] <= config.max_epochs
    frozen_mae = float(np.mean(np.abs((6.0 - base[:6].numpy()) - frame.pEC50[:6])))
    fitted_mae = float(
        np.mean(np.abs(predictions["predicted_pEC50"] - predictions["pEC50"]))
    )
    assert fitted_mae < frozen_mae
