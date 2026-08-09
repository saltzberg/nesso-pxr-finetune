"""Build the complete curated-DRC E0 feature manifest and artifact overlay."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nesso_pxr.protocol import sha256_file
from nesso_pxr.smoke import record_id_for_smiles

ARTIFACT_PATTERNS = {
    "records": "{record_id}.json",
    "structures": "{record_id}.npz",
    "rdkit_conformers": "{record_id}__B.pkl",
}


def _complete_source(root: Path, record_id: str) -> bool:
    return all(
        (root / directory / pattern.format(record_id=record_id)).is_file()
        for directory, pattern in ARTIFACT_PATTERNS.items()
    )


def _link_exact(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) == source:
            return
        raise FileExistsError(f"symlink target mismatch: {destination}")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing path: {destination}")
    destination.symlink_to(source)


def build_e0_manifest(
    training_manifest_path: Path,
    prepared_path: Path,
    processed_sources: list[Path],
    esm_path: Path,
    output_dir: Path,
    *,
    esm_id: str = "9ee8e7ba7e2644473a2ccd2a9ff22928",
    expected_rows: int = 4134,
) -> dict[str, Any]:
    """Create a stable all-curated-DRC row manifest and read-only artifact overlay."""

    if not processed_sources:
        raise ValueError("at least one processed source is required")

    rows = pd.read_csv(training_manifest_path)
    rows = rows.loc[rows["source"].eq("drc")].copy()
    prepared = pd.read_csv(prepared_path)
    prepared_columns = [
        "original_ligand_id",
        "prepared_ligand_id",
        "prepared_smiles",
        "formal_charge",
        "selection_reason",
    ]
    rows = rows.merge(
        prepared[prepared_columns],
        left_on="original_id",
        right_on="original_ligand_id",
        how="left",
        validate="one_to_one",
    )
    if rows["prepared_smiles"].isna().any():
        missing = rows.loc[rows["prepared_smiles"].isna(), "original_id"].tolist()
        raise ValueError(f"missing prepared states: {missing[:10]}")

    rows["record_id"] = rows["prepared_smiles"].map(record_id_for_smiles)
    if rows["record_id"].duplicated().any():
        duplicates = rows.loc[rows["record_id"].duplicated(False), "record_id"].tolist()
        raise ValueError(f"duplicate prepared record IDs: {duplicates[:10]}")
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} DRC rows, found {len(rows)}")

    source_labels = {
        root.resolve(): f"processed_source_{index}"
        for index, root in enumerate(processed_sources)
    }
    source_by_record: dict[str, Path] = {}
    missing_records: list[str] = []
    for record_id in rows["record_id"]:
        source = next(
            (
                root
                for root in processed_sources
                if _complete_source(root, str(record_id))
            ),
            None,
        )
        if source is None:
            missing_records.append(str(record_id))
        else:
            source_by_record[str(record_id)] = source.resolve()
    if missing_records:
        raise ValueError(f"incomplete Nesso artifacts: {missing_records[:10]}")

    rows["artifact_source"] = rows["record_id"].map(
        lambda record_id: source_labels[source_by_record[str(record_id)]]
    )
    rows = rows.sort_values("record_id", kind="stable").reset_index(drop=True)
    rows.insert(0, "feature_row", np.arange(len(rows), dtype=np.int64))

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.csv"
    rows.to_csv(selection_path, index=False)

    overlay = output_dir / "processed"
    for row in rows.itertuples(index=False):
        source_root = source_by_record[str(row.record_id)]
        for directory, pattern in ARTIFACT_PATTERNS.items():
            filename = pattern.format(record_id=row.record_id)
            _link_exact(
                source_root / directory / filename,
                overlay / directory / filename,
            )
    _link_exact(
        esm_path,
        overlay / "esm_embeddings" / f"{esm_id}.safetensors",
    )

    role_counts = {
        str(key): int(value)
        for key, value in rows["role"].value_counts().sort_index().items()
    }
    source_counts = Counter(rows["artifact_source"])
    audit = {
        "status": "pass",
        "rows": int(len(rows)),
        "unique_original_ids": int(rows["original_id"].nunique()),
        "unique_record_ids": int(rows["record_id"].nunique()),
        "role_counts": role_counts,
        "artifact_source_counts": dict(sorted(source_counts.items())),
        "selection_sha256": sha256_file(selection_path),
        "esm_sha256": sha256_file(esm_path),
        "esm_id": esm_id,
        "processed_overlay": str(overlay),
    }
    (output_dir / "manifest_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--processed-source",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--esm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=4134)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    audit = build_e0_manifest(
        args.training_manifest,
        args.prepared,
        args.processed_source,
        args.esm,
        args.output_dir,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
