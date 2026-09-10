"""Scaffold-aware Butina clustering and balanced five-fold assignment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from nesso_pxr.chemistry import (
    bemis_murcko_scaffold,
    canonicalize_smiles,
    morgan_fingerprints,
    scaffold_group_key,
)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class SplitConfig:
    n_folds: int = 5
    seed: int = 20260806
    radius: int = 2
    n_bits: int = 2048
    similarity_cutoff: float = 0.60

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if not 0 < self.similarity_cutoff < 1:
            raise ValueError("similarity_cutoff must be in (0, 1)")


def butina_cluster_ids(fingerprints: list, *, similarity_cutoff: float) -> np.ndarray:
    """Return deterministic Butina cluster IDs for RDKit fingerprints."""

    if not fingerprints:
        return np.empty(0, dtype=np.int64)
    distances: list[float] = []
    for index in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index], fingerprints[:index]
        )
        distances.extend(1.0 - value for value in similarities)
    clusters = Butina.ClusterData(
        distances,
        len(fingerprints),
        1.0 - similarity_cutoff,
        isDistData=True,
        reordering=True,
    )
    cluster_ids = np.full(len(fingerprints), -1, dtype=np.int64)
    for cluster_id, members in enumerate(clusters):
        cluster_ids[list(members)] = cluster_id
    if (cluster_ids < 0).any():
        raise RuntimeError("Butina did not assign every molecule")
    return cluster_ids


def assign_scaffold_aware_butina_folds(
    compounds: pd.DataFrame,
    *,
    id_column: str,
    smiles_column: str,
    config: SplitConfig | None = None,
) -> pd.DataFrame:
    """Assign compounds to balanced folds without splitting clusters/scaffolds.

    Butina clusters are first calculated from Morgan fingerprints. Clusters
    sharing a Bemis-Murcko scaffold are then unioned into indivisible
    components. Components are greedily assigned to the currently smallest
    fold, with seeded random tie-breaking.
    """

    if config is None:
        config = SplitConfig()
    required = {id_column, smiles_column}
    missing = required.difference(compounds.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    if compounds.empty:
        raise ValueError("compounds must not be empty")
    if compounds[id_column].isna().any() or compounds[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique and non-missing")

    result = compounds[[id_column, smiles_column]].copy().reset_index(drop=True)
    result["canonical_smiles"] = result[smiles_column].map(canonicalize_smiles)
    if result["canonical_smiles"].duplicated().any():
        raise ValueError(
            "canonical SMILES must be collapsed to biological compound groups first"
        )
    result["bemis_murcko_scaffold"] = result["canonical_smiles"].map(
        bemis_murcko_scaffold
    )
    result["scaffold_group"] = [
        scaffold_group_key(smiles, scaffold)
        for smiles, scaffold in zip(
            result["canonical_smiles"],
            result["bemis_murcko_scaffold"],
            strict=True,
        )
    ]

    fingerprints = morgan_fingerprints(
        result["canonical_smiles"].tolist(),
        radius=config.radius,
        n_bits=config.n_bits,
    )
    result["butina_cluster"] = butina_cluster_ids(
        fingerprints, similarity_cutoff=config.similarity_cutoff
    )

    disjoint = _DisjointSet(len(result))
    for column in ("butina_cluster", "scaffold_group"):
        for indices in result.groupby(column, sort=True).indices.values():
            anchor = int(indices[0])
            for index in indices[1:]:
                disjoint.union(anchor, int(index))

    roots = [disjoint.find(index) for index in range(len(result))]
    root_members: dict[int, list[int]] = defaultdict(list)
    for index, root in enumerate(roots):
        root_members[root].append(index)

    rng = np.random.default_rng(config.seed)
    randomized_ties = {root: float(rng.random()) for root in root_members}
    ordered_roots = sorted(
        root_members,
        key=lambda root: (-len(root_members[root]), randomized_ties[root], root),
    )
    fold_sizes = np.zeros(config.n_folds, dtype=np.int64)
    root_to_fold: dict[int, int] = {}
    for root in ordered_roots:
        smallest = np.flatnonzero(fold_sizes == fold_sizes.min())
        fold = int(rng.choice(smallest))
        root_to_fold[root] = fold
        fold_sizes[fold] += len(root_members[root])

    component_order = {
        root: component for component, root in enumerate(sorted(root_members))
    }
    result["cluster_component"] = [component_order[root] for root in roots]
    result["fold"] = [root_to_fold[root] for root in roots]
    return result


def assert_split_integrity(assignments: pd.DataFrame) -> None:
    """Raise if canonical, scaffold, or component groups cross fold boundaries."""

    required = {
        "canonical_smiles",
        "scaffold_group",
        "cluster_component",
        "fold",
    }
    missing = required.difference(assignments.columns)
    if missing:
        raise KeyError(f"missing audit columns: {sorted(missing)}")
    for column in ("canonical_smiles", "scaffold_group", "cluster_component"):
        maximum_roles = (
            assignments.groupby(column, dropna=False)["fold"].nunique().max()
        )
        if maximum_roles != 1:
            raise AssertionError(f"{column} is split across folds")
