"""Controlled pooling ablations for Nesso affinity pair representations.

These helpers intervene only at the final pair-to-vector pooling operation.
They therefore test the *direct* contribution of receptor--ligand pair cells,
while leaving open indirect protein information already mixed into ``z``.
"""

from __future__ import annotations

from typing import Any

PAIR_POOL_VARIANTS = (
    "all_pairs",
    "ligand_ligand_only",
    "receptor_ligand_only",
)


def affinity_pair_masks(feats: dict[str, Any]) -> dict[str, Any]:
    """Return Nesso-compatible pair masks split by interaction type."""

    import torch

    pad = feats["token_pad_mask"].unsqueeze(-1)
    receptor = (feats["mol_type"] == 0).float().unsqueeze(-1) * pad
    ligand = feats["affinity_token_mask"].float().unsqueeze(-1) * pad
    off_diagonal = 1.0 - torch.eye(
        ligand.shape[1], device=ligand.device, dtype=ligand.dtype
    ).unsqueeze(-1).unsqueeze(0)
    receptor_ligand = (
        ligand[:, :, None] * receptor[:, None, :]
        + receptor[:, :, None] * ligand[:, None, :]
    ) * off_diagonal
    ligand_ligand = (ligand[:, :, None] * ligand[:, None, :]) * off_diagonal
    return {
        "all_pairs": receptor_ligand + ligand_ligand,
        "ligand_ligand_only": ligand_ligand,
        "receptor_ligand_only": receptor_ligand,
    }


def pool_affinity_representation(
    affinity_heads: Any,
    z: Any,
    feats: dict[str, Any],
    variant: str,
) -> Any:
    """Pool one pair subset and apply Nesso's frozen affinity output MLP."""

    if variant not in PAIR_POOL_VARIANTS:
        raise ValueError(f"unknown pair-pool variant: {variant}")
    mask = affinity_pair_masks(feats)[variant]
    pooled = (z * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-7)
    return affinity_heads.affinity_out_mlp(pooled)


def register_pair_pool_hooks(
    model: Any,
    captured: dict[str, dict[str, Any]],
) -> list[Any]:
    """Capture baseline and interaction-specific 384D representations."""

    def capture(member: str):
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: dict[str, Any],
        ) -> None:
            z = kwargs.get("z", args[0] if args else None)
            feats = kwargs.get("feats", args[1] if len(args) > 1 else None)
            if z is None or feats is None:
                raise RuntimeError("could not locate z and feats in affinity-head call")
            representations = {"all_pairs": output["affinity_repr"].detach().clone()}
            for variant in PAIR_POOL_VARIANTS[1:]:
                representations[variant] = (
                    pool_affinity_representation(module, z, feats, variant)
                    .detach()
                    .clone()
                )
            captured[member] = representations

        return hook

    return [
        model.affinity_module.affinity_heads.register_forward_hook(
            capture("member1"), with_kwargs=True
        ),
        model.affinity_module2.affinity_heads.register_forward_hook(
            capture("member2"), with_kwargs=True
        ),
    ]
