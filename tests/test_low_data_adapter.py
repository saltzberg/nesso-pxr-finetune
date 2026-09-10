"""Synthetic unit tests, not evidence of real Nesso feasibility."""

import copy

import pytest
import torch
from torch import nn

from nesso_pxr.low_data_adapter import (
    LoRALinear,
    adapter_checkpoint,
    inject_lora,
    load_adapter_checkpoint,
)


def fixture():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 2)).double()


def test_zero_parity_gradients_updates_frozen_and_reload(tmp_path):
    model = fixture()
    pristine = copy.deepcopy(model)
    x = torch.randn(4, 5, dtype=torch.float64)
    baseline = model(x).detach()
    inject_lora(model, ["0", "2"], rank=2, alpha=2)
    assert torch.equal(baseline, model(x))
    frozen = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not p.requires_grad
    }
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad), lr=0.1
    )
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all()
                # B=0 makes A's initial gradient zero by construction.
                if step or name.endswith("lora_B"):
                    assert p.grad.abs().sum() > 0
            else:
                assert p.grad is None
        optimizer.step()
    assert not torch.equal(baseline, model(x))
    for name, p in model.named_parameters():
        if name in frozen:
            assert torch.equal(p, frozen[name])
    checkpoint = adapter_checkpoint(model)
    path = tmp_path / "adapter.pt"
    torch.save(checkpoint, path)
    load_adapter_checkpoint(pristine, torch.load(path, weights_only=True))
    assert torch.equal(model(x), pristine(x))
    assert checkpoint["targets"] == ["0", "2"]
    assert all("lora_" in k for k in checkpoint["state"])


@pytest.mark.parametrize(
    "targets", [[], ["absent"], ["1"], ["0", "0"], ["0", "0.base"]]
)
def test_injection_fail_closed(targets):
    model = fixture()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises((ValueError, TypeError, AttributeError)):
        inject_lora(model, targets, rank=2)
    assert all(p.requires_grad for p in model.parameters())
    assert all(torch.equal(before[n], p) for n, p in model.state_dict().items())


@pytest.mark.parametrize(
    "rank,alpha", [(0, 1), (-1, 1), (20, 1), (1, float("nan")), (1, 0)]
)
def test_invalid_hyperparameters(rank, alpha):
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(2, 3), rank=rank, alpha=alpha)


def test_reload_rejects_wrong_base_and_missing_tensor():
    adapted = fixture()
    inject_lora(adapted, ["0"], rank=2)
    payload = adapter_checkpoint(adapted)
    wrong = fixture()
    with torch.no_grad():
        wrong[0].weight.add_(1)
    with pytest.raises(ValueError, match="base"):
        load_adapter_checkpoint(wrong, payload)
    broken = copy.deepcopy(payload)
    broken["state"].pop(next(iter(broken["state"])))
    with pytest.raises(ValueError, match="keys"):
        load_adapter_checkpoint(fixture(), broken)


def test_biasless_dtype_and_no_double_injection():
    model = nn.Sequential(nn.Linear(3, 4, bias=False)).double()
    inject_lora(model, ["0"], rank=2)
    assert model[0].lora_A.dtype == torch.float64
    assert model[0].base.bias is None
    with pytest.raises((ValueError, TypeError)):
        inject_lora(model, ["0"], rank=2)
