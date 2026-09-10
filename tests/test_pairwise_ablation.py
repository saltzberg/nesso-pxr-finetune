from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nesso_pxr.pairwise_ablation import (
    affinity_pair_masks,
    pool_affinity_representation,
)


def _features() -> dict[str, torch.Tensor]:
    return {
        "token_pad_mask": torch.ones((1, 4)),
        "mol_type": torch.tensor([[0, 0, 3, 3]]),
        "affinity_token_mask": torch.tensor([[0, 0, 1, 1]]),
    }


def test_affinity_pair_masks_partition_direct_interactions() -> None:
    masks = affinity_pair_masks(_features())
    assert float(masks["receptor_ligand_only"].sum()) == 8.0
    assert float(masks["ligand_ligand_only"].sum()) == 2.0
    assert float(masks["all_pairs"].sum()) == 10.0
    assert torch.equal(
        masks["all_pairs"],
        masks["receptor_ligand_only"] + masks["ligand_ligand_only"],
    )
    assert not torch.any(
        masks["receptor_ligand_only"].bool() & masks["ligand_ligand_only"].bool()
    )
    assert (
        torch.count_nonzero(
            torch.diagonal(masks["all_pairs"].squeeze(-1), dim1=1, dim2=2)
        )
        == 0
    )


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("receptor_ligand_only", 7.5),
        ("ligand_ligand_only", 12.5),
        ("all_pairs", 8.5),
    ],
)
def test_pool_affinity_representation_matches_masked_mean(
    variant: str, expected: float
) -> None:
    z = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)
    heads = SimpleNamespace(affinity_out_mlp=nn.Identity())
    representation = pool_affinity_representation(heads, z, _features(), variant)
    assert representation.shape == (1, 1)
    assert float(representation.item()) == pytest.approx(expected)


def test_pool_affinity_representation_rejects_unknown_variant() -> None:
    heads = SimpleNamespace(affinity_out_mlp=nn.Identity())
    with pytest.raises(ValueError, match="unknown pair-pool variant"):
        pool_affinity_representation(
            heads, torch.zeros((1, 4, 4, 1)), _features(), "bad"
        )
