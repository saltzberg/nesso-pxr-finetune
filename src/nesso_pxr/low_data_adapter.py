"""Explicit low-rank interventions for frozen affinity modules (not head probes).

No Nesso installation is modified. Wrappers are injected into in-memory modules.
Keep the upstream model in eval mode: gradients do not require model.train().
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen Linear(x) + alpha/rank * B(A(x)); B starts exactly zero."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA requires an nn.Linear")
        if not isinstance(rank, int) or not 1 <= rank <= min(
            base.in_features, base.out_features
        ):
            raise ValueError(
                "rank must be positive and no larger than either Linear dimension"
            )
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.base = base
        self.rank, self.alpha = rank, float(alpha)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)
        self.train(base.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * (
            self.alpha / self.rank
        )


def inject_lora(
    model: nn.Module, targets: Iterable[str], *, rank: int = 4, alpha: float = 4.0
) -> list[str]:
    """Validate exact names before mutation; freeze everything except A and B."""
    targets = list(targets)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("targets must be a nonempty unique list")
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError("model already contains adapters")
    selected = []
    for name in targets:
        if not name:
            raise ValueError("root replacement is unsupported")
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{name} is not an nn.Linear")
        if not isinstance(rank, int) or not 1 <= rank <= min(
            layer.in_features, layer.out_features
        ):
            raise ValueError("invalid rank")
        selected.append((name, layer))
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid alpha")
    if len({id(layer) for _, layer in selected}) != len(selected):
        raise ValueError("shared target modules are unsupported")
    model.requires_grad_(False)
    for name, layer in selected:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(layer, rank=rank, alpha=alpha))
    return targets


def base_fingerprint(model: nn.Module) -> str:
    """Hash full frozen parameters/buffers using pre-injection names."""
    wrappers = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    tensors = {}
    for name, tensor in model.state_dict().items():
        if name.endswith((".lora_A", ".lora_B")):
            continue
        for prefix in wrappers:
            if name.startswith(prefix + ".base."):
                name = prefix + "." + name[len(prefix + ".base.") :]
                break
        tensors[name] = tensor
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{list(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_checkpoint(model: nn.Module) -> dict[str, Any]:
    modules = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if not modules:
        raise ValueError("no adapters")
    return {
        "schema_version": 1,
        "base_sha256": base_fingerprint(model),
        "targets": list(modules),
        "config": {n: {"rank": m.rank, "alpha": m.alpha} for n, m in modules.items()},
        "state": {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if n.endswith((".lora_A", ".lora_B"))
        },
    }


def load_adapter_checkpoint(model: nn.Module, payload: dict[str, Any]) -> None:
    """Restore adapter-only state into the exact original frozen model."""
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    if base_fingerprint(model) != payload["base_sha256"]:
        raise ValueError("base model fingerprint mismatch")
    targets = payload["targets"]
    expected = {f"{n}.{suffix}" for n in targets for suffix in ("lora_A", "lora_B")}
    if set(payload["state"]) != expected or set(payload["config"]) != set(targets):
        raise ValueError("adapter checkpoint keys mismatch")
    configs = [payload["config"][n] for n in targets]
    if not configs or any(c != configs[0] for c in configs):
        raise ValueError("only uniform rank/alpha checkpoints are supported")
    for name in targets:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            raise ValueError("reload requires an unwrapped base model")
        shapes = {
            "lora_A": (configs[0]["rank"], layer.in_features),
            "lora_B": (layer.out_features, configs[0]["rank"]),
        }
        for suffix, shape in shapes.items():
            value = payload["state"][f"{name}.{suffix}"]
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError("invalid adapter tensor")
    inject_lora(model, targets, **configs[0])
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in payload["state"].items():
            parameters[name].copy_(value)
