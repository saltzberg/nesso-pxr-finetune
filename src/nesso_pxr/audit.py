"""Build the Milestone 0 identity, split, and leakage-audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nesso_pxr.chemistry import (
    bemis_murcko_scaffold,
    canonicalize_smiles,
)
from nesso_pxr.protocol import load_protocol, resolve_and_verify_sources, sha256_file
from nesso_pxr.splits import (
    SplitConfig,
    assert_split_integrity,
    assign_scaffold_aware_butina_folds,
)


@cache
def _canonical(smiles: str) -> str:
    return canonicalize_smiles(smiles)


@cache
def _scaffold(canonical_smiles: str) -> str:
    return bemis_murcko_scaffold(canonical_smiles)


def _compound_group_id(canonical_smiles: str) -> str:
    digest = hashlib.sha256(canonical_smiles.encode()).hexdigest()[:20]
    return f"pxr_{digest}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_drc(
    drc: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    emax_column: str,
    emax_minimum: float,
) -> pd.DataFrame:
    holdout_columns = [
        "Molecule Name",
        "holdout_role",
        "bemis_murcko_scaffold",
    ]
    merged = drc.merge(
        holdout[holdout_columns],
        on="Molecule Name",
        how="left",
        validate="one_to_one",
        suffixes=("", "_holdout"),
    )
    if merged["holdout_role"].isna().any():
        raise ValueError("DRC rows missing immutable holdout assignments")
    merged["identity_canonical_smiles"] = merged["canonical_smiles"].map(_canonical)
    merged["identity_scaffold"] = merged["identity_canonical_smiles"].map(_scaffold)
    merged["initial_emax_eligible"] = merged[emax_column].ge(emax_minimum)
    merged["initial_role"] = "excluded_initial_emax"
    merged.loc[
        merged["initial_emax_eligible"] & merged["holdout_role"].eq("development"),
        "initial_role",
    ] = "development_primary"
    merged.loc[
        merged["initial_emax_eligible"] & merged["holdout_role"].eq("holdout"),
        "initial_role",
    ] = "lockbox_primary"
    return merged


def _prepare_single_point(
    single: pd.DataFrame,
    drc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_id_smiles = single.groupby("OCNT_ID", sort=False)["SMILES"].nunique(
        dropna=False
    )
    if per_id_smiles.max() != 1:
        bad = per_id_smiles[per_id_smiles != 1].index.tolist()
        raise ValueError(f"single-point IDs with multiple structures: {bad[:10]}")

    compounds = single[["OCNT_ID", "SMILES"]].drop_duplicates("OCNT_ID").copy()
    compounds["single_canonical_smiles"] = compounds["SMILES"].map(_canonical)
    compounds["single_scaffold"] = compounds["single_canonical_smiles"].map(_scaffold)

    if drc["OCNT_ID"].duplicated().any():
        raise ValueError("DRC OCNT_ID values must be unique")
    if drc["identity_canonical_smiles"].duplicated().any():
        raise ValueError("DRC canonical structures must be unique")
    drc_by_id = drc.set_index("OCNT_ID")
    drc_by_canonical = drc.set_index("identity_canonical_smiles")

    records: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    for row in compounds.itertuples(index=False):
        ocnt_id = row.OCNT_ID
        canonical = row.single_canonical_smiles
        id_match = drc_by_id.loc[ocnt_id] if ocnt_id in drc_by_id.index else None
        canonical_match = (
            drc_by_canonical.loc[canonical]
            if canonical in drc_by_canonical.index
            else None
        )

        if (
            id_match is not None
            and canonical_match is not None
            and id_match["Molecule Name"] != canonical_match["Molecule Name"]
        ):
            collisions.append(
                {
                    "OCNT_ID": ocnt_id,
                    "single_canonical_smiles": canonical,
                    "id_match_molecule": id_match["Molecule Name"],
                    "canonical_match_molecule": canonical_match["Molecule Name"],
                    "reason": "id_and_canonical_map_to_different_drc_compounds",
                }
            )
            records.append(
                {
                    "OCNT_ID": ocnt_id,
                    "single_canonical_smiles": canonical,
                    "single_scaffold": row.single_scaffold,
                    "linked_drc_molecule": None,
                    "identity_canonical_smiles": canonical,
                    "identity_link_reason": "quarantined_identity_collision",
                    "auxiliary_role": "quarantined_identity_collision",
                }
            )
            continue

        match = id_match if id_match is not None else canonical_match
        if match is not None:
            canonical_variant = canonical != match["identity_canonical_smiles"]
            link_reason = (
                "ocnt_id_canonical_variant"
                if id_match is not None and canonical_variant
                else "ocnt_id"
                if id_match is not None
                else "canonical_smiles"
            )
            auxiliary_role = (
                "lockbox_auxiliary_excluded"
                if match["holdout_role"] == "holdout"
                else "development_auxiliary"
            )
            identity_canonical = match["identity_canonical_smiles"]
            linked_molecule = match["Molecule Name"]
        else:
            link_reason = "auxiliary_only"
            auxiliary_role = "development_auxiliary"
            identity_canonical = canonical
            linked_molecule = None

        records.append(
            {
                "OCNT_ID": ocnt_id,
                "single_canonical_smiles": canonical,
                "single_scaffold": row.single_scaffold,
                "linked_drc_molecule": linked_molecule,
                "identity_canonical_smiles": identity_canonical,
                "identity_link_reason": link_reason,
                "auxiliary_role": auxiliary_role,
            }
        )

    links = pd.DataFrame.from_records(records)
    holdout_scaffolds = set(
        drc.loc[drc["holdout_role"].eq("holdout"), "identity_scaffold"]
    )
    auxiliary_only = links["linked_drc_molecule"].isna()
    protected_scaffold = links["single_scaffold"].isin(holdout_scaffolds)
    links.loc[
        auxiliary_only
        & protected_scaffold
        & links["auxiliary_role"].eq("development_auxiliary"),
        "auxiliary_role",
    ] = "lockbox_scaffold_guard_excluded"
    collision_columns = [
        "OCNT_ID",
        "single_canonical_smiles",
        "id_match_molecule",
        "canonical_match_molecule",
        "reason",
    ]
    collision_ledger = pd.DataFrame.from_records(
        collisions,
        columns=collision_columns,
    )
    return links, collision_ledger


def _make_joint_folds(
    drc: pd.DataFrame,
    single_links: pd.DataFrame,
    split_config: SplitConfig,
) -> pd.DataFrame:
    primary = drc.loc[
        drc["initial_role"].eq("development_primary"),
        ["identity_canonical_smiles"],
    ].assign(source="drc_primary")
    auxiliary = single_links.loc[
        single_links["auxiliary_role"].eq("development_auxiliary"),
        ["identity_canonical_smiles"],
    ].assign(source="single_point")
    joint = pd.concat([primary, auxiliary], ignore_index=True)
    source_membership = (
        joint.groupby("identity_canonical_smiles", sort=True)["source"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .rename("sources")
        .reset_index()
    )
    source_membership["compound_group_id"] = source_membership[
        "identity_canonical_smiles"
    ].map(_compound_group_id)
    assignments = assign_scaffold_aware_butina_folds(
        source_membership,
        id_column="compound_group_id",
        smiles_column="identity_canonical_smiles",
        config=split_config,
    )
    assert_split_integrity(assignments)
    assignments = assignments.merge(
        source_membership[["compound_group_id", "sources"]],
        on="compound_group_id",
        how="left",
        validate="one_to_one",
    )
    assignments = assignments.drop(columns=["identity_canonical_smiles"])
    assignments = assignments.rename(
        columns={"canonical_smiles": "identity_canonical_smiles"}
    )
    return assignments


def _training_manifest(
    drc: pd.DataFrame,
    single: pd.DataFrame,
    single_links: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    fold_map = assignments.set_index("identity_canonical_smiles")["fold"]
    drc_rows = pd.DataFrame(
        {
            "row_id": drc["Molecule Name"],
            "source": "drc",
            "original_id": drc["Molecule Name"],
            "OCNT_ID": drc["OCNT_ID"],
            "raw_smiles": drc["SMILES"],
            "identity_canonical_smiles": drc["identity_canonical_smiles"],
            "role": drc["initial_role"],
            "pEC50": drc["pEC50"],
            "pEC50_standard_error": drc["pEC50_std.error (-log10(molarity))"],
            "curation_sample_weight": drc["curation_sample_weight"],
        }
    )
    drc_rows["fold"] = (
        drc_rows["identity_canonical_smiles"].map(fold_map).astype("Int64")
    )
    drc_rows["log2_fc_estimate"] = np.nan
    drc_rows["log2_fc_stderr"] = np.nan
    drc_rows["concentration_M"] = np.nan

    observations = single.merge(
        single_links,
        on="OCNT_ID",
        how="left",
        validate="many_to_one",
    )
    sp_rows = pd.DataFrame(
        {
            "row_id": [f"sp_{index}" for index in observations.index],
            "source": "single_point",
            "original_id": observations["OCNT_ID"],
            "OCNT_ID": observations["OCNT_ID"],
            "raw_smiles": observations["SMILES"],
            "identity_canonical_smiles": observations["identity_canonical_smiles"],
            "role": observations["auxiliary_role"],
            "pEC50": np.nan,
            "pEC50_standard_error": np.nan,
            "curation_sample_weight": np.nan,
            "log2_fc_estimate": observations["log2_fc_estimate"],
            "log2_fc_stderr": observations["log2_fc_stderr"],
            "concentration_M": observations["concentration_M"],
        }
    )
    sp_rows["fold"] = sp_rows["identity_canonical_smiles"].map(fold_map).astype("Int64")
    return pd.concat([drc_rows, sp_rows], ignore_index=True, sort=False)


def build_milestone0(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    paths = resolve_and_verify_sources(
        protocol, base_dir=protocol_path.resolve().parent
    )
    labels = protocol["labels"]
    split_values = protocol["splits"]
    split_config = SplitConfig(
        n_folds=int(split_values["n_folds"]),
        seed=int(split_values["seed"]),
        radius=int(split_values["morgan_radius"]),
        n_bits=int(split_values["fingerprint_bits"]),
        similarity_cutoff=float(split_values["tanimoto_similarity_cutoff"]),
    )

    drc_raw = pd.read_csv(paths["drc_curated"])
    single = pd.read_csv(paths["single_point"])
    prepared = pd.read_csv(paths["prepared_top_states"])
    holdout = pd.read_csv(paths["holdout_assignments"])
    blind = pd.read_csv(paths["blinded_activity"])

    forbidden_blind_columns = {
        "pEC50",
        labels["pEC50_standard_error_column"],
        labels["emax_column"],
    }
    observed_forbidden = forbidden_blind_columns.intersection(blind.columns)
    if observed_forbidden:
        raise ValueError(f"blinded activity file contains labels: {observed_forbidden}")

    drc = _prepare_drc(
        drc_raw,
        holdout,
        emax_column=labels["emax_column"],
        emax_minimum=float(labels["initial_emax_minimum"]),
    )
    single_links, collisions = _prepare_single_point(single, drc)
    assignments = _make_joint_folds(drc, single_links, split_config)
    manifest = _training_manifest(drc, single, single_links, assignments)

    fold_map = assignments.set_index("identity_canonical_smiles")["fold"]
    drc_id_role = drc.set_index("Molecule Name")["initial_role"].to_dict()
    drc_id_fold = (
        drc.set_index("Molecule Name")["identity_canonical_smiles"]
        .map(fold_map)
        .to_dict()
    )
    sp_id_role = single_links.set_index("OCNT_ID")["auxiliary_role"].to_dict()
    sp_id_fold = (
        single_links.set_index("OCNT_ID")["identity_canonical_smiles"]
        .map(fold_map)
        .to_dict()
    )
    blind_ids = set(blind["Molecule Name"])

    prepared_roles: list[str] = []
    prepared_folds: list[Any] = []
    for original_id in prepared["original_ligand_id"]:
        if original_id in drc_id_role:
            prepared_roles.append(drc_id_role[original_id])
            prepared_folds.append(drc_id_fold.get(original_id))
        elif original_id in sp_id_role:
            prepared_roles.append(sp_id_role[original_id])
            prepared_folds.append(sp_id_fold.get(original_id))
        elif original_id in blind_ids:
            prepared_roles.append("blind_label_free")
            prepared_folds.append(None)
        else:
            prepared_roles.append("not_in_initial_sources")
            prepared_folds.append(None)
    prepared_manifest = prepared.copy()
    prepared_manifest["role"] = prepared_roles
    prepared_manifest["fold"] = pd.array(prepared_folds, dtype="Int64")

    development = manifest[
        manifest["role"].isin(["development_primary", "development_auxiliary"])
    ]
    lockbox = manifest[manifest["role"].str.startswith("lockbox")]
    dev_ids = set(development["original_id"])
    lockbox_ids = set(lockbox["original_id"])
    dev_canonical = set(development["identity_canonical_smiles"])
    lockbox_canonical = set(lockbox["identity_canonical_smiles"])

    audit = {
        "status": "pass",
        "protocol_version": protocol["protocol_version"],
        "n_folds": split_config.n_folds,
        "counts": {
            "drc_rows": int(len(drc)),
            "drc_development_primary": int(
                drc["initial_role"].eq("development_primary").sum()
            ),
            "drc_lockbox_primary": int(drc["initial_role"].eq("lockbox_primary").sum()),
            "single_point_observations": int(len(single)),
            "single_point_compounds": int(single["OCNT_ID"].nunique()),
            "identity_collisions_quarantined": int(len(collisions)),
            "joint_development_compounds": int(len(assignments)),
            "prepared_top_states": int(len(prepared_manifest)),
            "blinded_activity_rows_label_free": int(len(blind)),
        },
        "fold_counts": {
            str(key): int(value)
            for key, value in assignments["fold"].value_counts().sort_index().items()
        },
        "checks": {
            "development_lockbox_original_id_overlap": len(dev_ids & lockbox_ids),
            "development_lockbox_canonical_overlap": len(
                dev_canonical & lockbox_canonical
            ),
            "scaffold_groups_crossing_folds": int(
                (assignments.groupby("scaffold_group")["fold"].nunique() > 1).sum()
            ),
            "butina_components_crossing_folds": int(
                (assignments.groupby("cluster_component")["fold"].nunique() > 1).sum()
            ),
            "blind_has_forbidden_label_columns": len(observed_forbidden),
        },
    }
    if any(audit["checks"].values()):
        audit["status"] = "fail"
        raise AssertionError(f"Milestone 0 leakage audit failed: {audit['checks']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "joint_butina_scaffold_folds.csv", index=False)
    single_links.to_csv(output_dir / "single_point_identity_links.csv", index=False)
    collisions.to_csv(output_dir / "identity_collision_ledger.csv", index=False)
    manifest.to_csv(output_dir / "training_row_manifest.csv", index=False)
    prepared_manifest.to_csv(
        output_dir / "prepared_state_role_manifest.csv", index=False
    )
    _write_json(output_dir / "leakage_audit.json", audit)
    source_manifest = {
        name: {
            "path": str(protocol["data"][name]["path"]),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    _write_json(output_dir / "source_manifest.json", source_manifest)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/protocol.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/manifests/milestone0")
    )
    args = parser.parse_args()
    audit = build_milestone0(args.config, args.output)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
