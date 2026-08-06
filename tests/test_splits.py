import pandas as pd
import pytest

from nesso_pxr.splits import (
    SplitConfig,
    assert_split_integrity,
    assign_scaffold_aware_butina_folds,
)


def _example_compounds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound_id": [f"c{i}" for i in range(10)],
            "smiles": [
                "Cc1ccccc1",
                "CCc1ccccc1",
                "Oc1ccccc1",
                "c1ccncc1",
                "Cc1ccncc1",
                "C1CCCCC1",
                "CC1CCCCC1",
                "CCO",
                "CCCO",
                "CC(=O)O",
            ],
        }
    )


def test_assignment_is_deterministic_balanced_and_scaffold_safe() -> None:
    config = SplitConfig(n_folds=5, seed=17, similarity_cutoff=0.7)
    first = assign_scaffold_aware_butina_folds(
        _example_compounds(),
        id_column="compound_id",
        smiles_column="smiles",
        config=config,
    )
    second = assign_scaffold_aware_butina_folds(
        _example_compounds(),
        id_column="compound_id",
        smiles_column="smiles",
        config=config,
    )
    pd.testing.assert_frame_equal(first, second)
    assert_split_integrity(first)
    assert set(first["fold"]) == set(range(5))
    assert first.groupby("fold").size().max() - first.groupby("fold").size().min() <= 2


def test_same_ring_scaffold_is_never_split() -> None:
    assigned = assign_scaffold_aware_butina_folds(
        _example_compounds(), id_column="compound_id", smiles_column="smiles"
    )
    benzene_rows = assigned[assigned["compound_id"].isin(["c0", "c1", "c2"])]
    assert benzene_rows["scaffold_group"].nunique() == 1
    assert benzene_rows["fold"].nunique() == 1


def test_acyclic_compounds_do_not_form_one_giant_empty_scaffold() -> None:
    assigned = assign_scaffold_aware_butina_folds(
        _example_compounds(), id_column="compound_id", smiles_column="smiles"
    )
    acyclic = assigned[assigned["bemis_murcko_scaffold"] == ""]
    assert acyclic["scaffold_group"].nunique() == len(acyclic)


def test_duplicate_canonical_compounds_must_be_collapsed_first() -> None:
    duplicates = pd.DataFrame({"compound_id": ["a", "b"], "smiles": ["CCO", "OCC"]})
    with pytest.raises(ValueError, match="collapsed"):
        assign_scaffold_aware_butina_folds(
            duplicates, id_column="compound_id", smiles_column="smiles"
        )
