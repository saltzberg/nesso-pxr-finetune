import pandas as pd

from nesso_pxr.audit import _prepare_single_point


def test_empty_identity_collision_ledger_has_stable_schema() -> None:
    drc = pd.DataFrame(
        {
            "OCNT_ID": ["OCNT-1"],
            "identity_canonical_smiles": ["CCO"],
            "identity_scaffold": [""],
            "Molecule Name": ["D1"],
            "holdout_role": ["development"],
        }
    )
    single = pd.DataFrame({"OCNT_ID": ["OCNT-1"], "SMILES": ["CCO"]})

    _, collisions = _prepare_single_point(single, drc)

    assert collisions.empty
    assert collisions.columns.tolist() == [
        "OCNT_ID",
        "single_canonical_smiles",
        "id_match_molecule",
        "canonical_match_molecule",
        "reason",
    ]
