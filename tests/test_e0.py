from pathlib import Path

import pandas as pd

from nesso_pxr.e0 import ARTIFACT_PATTERNS, build_e0_manifest
from nesso_pxr.smoke import record_id_for_smiles


def _write_artifacts(root: Path, record_id: str) -> None:
    for directory, pattern in ARTIFACT_PATTERNS.items():
        path = root / directory / pattern.format(record_id=record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record_id)


def test_build_e0_manifest_orders_rows_and_uses_fallback_source(tmp_path: Path) -> None:
    training = pd.DataFrame(
        [
            {
                "source": "drc",
                "original_id": "second",
                "role": "lockbox_primary",
                "pEC50": 4.0,
            },
            {
                "source": "drc",
                "original_id": "first",
                "role": "development_primary",
                "pEC50": 5.0,
            },
            {
                "source": "single_point",
                "original_id": "ignored",
                "role": "development_auxiliary",
            },
        ]
    )
    prepared = pd.DataFrame(
        [
            {
                "original_ligand_id": "first",
                "prepared_ligand_id": "first",
                "prepared_smiles": "CC",
                "formal_charge": 0,
                "selection_reason": "test",
            },
            {
                "original_ligand_id": "second",
                "prepared_ligand_id": "second",
                "prepared_smiles": "CCC",
                "formal_charge": 0,
                "selection_reason": "test",
            },
        ]
    )
    training_path = tmp_path / "training.csv"
    prepared_path = tmp_path / "prepared.csv"
    training.to_csv(training_path, index=False)
    prepared.to_csv(prepared_path, index=False)

    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    first_id = record_id_for_smiles("CC")
    second_id = record_id_for_smiles("CCC")
    _write_artifacts(primary, first_id)
    _write_artifacts(fallback, second_id)
    esm = tmp_path / "pxr_esm.safetensors"
    esm.write_bytes(b"esm")

    output = tmp_path / "e0"
    audit = build_e0_manifest(
        training_path,
        prepared_path,
        [primary, fallback],
        esm,
        output,
        expected_rows=2,
    )

    selection = pd.read_csv(output / "selection.csv")
    assert selection["feature_row"].tolist() == [0, 1]
    assert selection["record_id"].tolist() == sorted([first_id, second_id])
    assert set(selection["artifact_source"]) == {
        "processed_source_0",
        "processed_source_1",
    }
    assert audit["rows"] == 2
    assert audit["role_counts"] == {
        "development_primary": 1,
        "lockbox_primary": 1,
    }
    for record_id in (first_id, second_id):
        assert (output / "processed" / "records" / f"{record_id}.json").is_symlink()
    assert (
        output
        / "processed"
        / "esm_embeddings"
        / "9ee8e7ba7e2644473a2ccd2a9ff22928.safetensors"
    ).is_symlink()
