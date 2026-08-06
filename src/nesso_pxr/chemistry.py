"""Deterministic cheminformatics primitives used by split construction."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def canonicalize_smiles(smiles: str) -> str:
    """Return isomeric canonical SMILES or raise a reason-coded error."""

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("empty_smiles")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("rdkit_parse_failure")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def bemis_murcko_scaffold(canonical_smiles: str) -> str:
    """Return canonical Bemis-Murcko scaffold SMILES.

    RDKit returns an empty scaffold for acyclic compounds. Callers should use
    :func:`scaffold_group_key` to avoid treating all acyclic chemistry as one
    giant scaffold family.
    """

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError("rdkit_parse_failure")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def scaffold_group_key(canonical_smiles: str, scaffold: str) -> str:
    """Return a nonempty grouping key suitable for scaffold-aware splitting."""

    return scaffold if scaffold else f"ACYCLIC::{canonical_smiles}"


def morgan_fingerprints(
    canonical_smiles: list[str], *, radius: int = 2, n_bits: int = 2048
) -> list:
    """Generate count-free Morgan bit vectors with current RDKit APIs."""

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = []
    for smiles in canonical_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"rdkit_parse_failure:{smiles}")
        fingerprints.append(generator.GetFingerprint(mol))
    return fingerprints
