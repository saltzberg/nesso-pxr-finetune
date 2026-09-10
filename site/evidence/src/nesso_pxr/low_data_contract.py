"""Frozen label, feature, split and scoring contracts for low-data adaptation.

No fitting occurs here. All scalar features replay released heads on immutable
cached vectors without label arguments; native_continuous is 6 - log10(IC50/uM),
NOT a native cellular pEC50 prediction. Acquired budgets include calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs, rdBase
from scipy.stats import spearmanr

from nesso_pxr.chemistry import (
    bemis_murcko_scaffold,
    canonicalize_smiles,
    morgan_fingerprints,
    scaffold_group_key,
)
from nesso_pxr.protocol import sha256_file
from nesso_pxr.splits import butina_cluster_ids

FEATURE_KEYS = ("member1", "member2", "morgan", "native_continuous", "scalar_features")
SCALAR_COLUMNS = (
    "member1_pIC50_equivalent",
    "member2_pIC50_equivalent",
    "ensemble_pIC50_equivalent",
    "member1_binary_logit",
    "member2_binary_logit",
    "ensemble_binary_logit",
    "member1_binder_probability",
    "member2_binder_probability",
    "ensemble_binder_probability",
)
REPLAY_ATOL = 1e-6


def _vector(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite 1D vector")
    return values


def assay_weights(se, floor=0.10) -> np.ndarray:
    """Unnormalized inverse-SE weights, never inverse variance or curation weights."""
    if not np.isscalar(floor) or not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be finite and positive")
    se = _vector(se, "assay standard errors")
    if np.any(se <= 0):
        raise ValueError("assay standard errors must be strictly positive")
    return 1.0 / np.maximum(se, floor)


def weighted_mae(y, pred, se, floor=0.10) -> float:
    """Normalize inverse-SE weighted absolute errors by the sum of weights."""
    y, pred = _vector(y, "y"), _vector(pred, "pred")
    weights = assay_weights(se, floor)
    if y.shape != pred.shape or y.shape != weights.shape:
        raise ValueError("y, pred and se must have identical 1D shapes")
    return float(np.average(np.abs(pred - y), weights=weights))


def score_predictions(y, pred, se, floor=0.10) -> dict:
    main = weighted_mae(y, pred, se, floor)
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    weights = assay_weights(se, floor)
    rho = None
    if len(y) > 1 and np.ptp(y) > 0 and np.ptp(pred) > 0:
        rho = float(spearmanr(y, pred).statistic)
    return dict(
        weighted_mae=main,
        mae=float(np.mean(np.abs(pred - y))),
        spearman=rho,
        bias=float(np.mean(pred - y)),
        effective_n=float(weights.sum() ** 2 / np.square(weights).sum()),
        weighted_mae_floor_005=weighted_mae(y, pred, se, 0.05),
        weighted_mae_floor_020=weighted_mae(y, pred, se, 0.20),
    )


def _development_inputs(manifest, expected_rows=3344):
    required = {
        "record_id",
        "feature_row",
        "identity_canonical_smiles",
        "pEC50",
        "pEC50_standard_error",
        "modeling_role",
    }
    if not required <= set(manifest):
        raise ValueError(
            f"missing manifest columns: {sorted(required - set(manifest))}"
        )
    for col in ("record_id", "feature_row"):
        if manifest[col].isna().any() or manifest[col].duplicated().any():
            raise ValueError(f"manifest {col} must be unique and nonmissing")
    idx = manifest.feature_row.to_numpy(dtype=float)
    if not np.isfinite(idx).all() or np.any(idx < 0) or np.any(idx != np.floor(idx)):
        raise ValueError("feature_row must contain nonnegative integer indices")
    dev = manifest.loc[manifest.modeling_role.eq("development")].copy()
    if len(dev) != expected_rows:
        raise ValueError(f"expected {expected_rows} development rows, found {len(dev)}")
    dev = dev.sort_values("feature_row", kind="stable").reset_index(drop=True)
    canonical = dev.identity_canonical_smiles.map(canonicalize_smiles)
    if canonical.duplicated().any():
        raise ValueError(
            "duplicate canonical identity: exact-N unique budgets infeasible"
        )
    _vector(dev.pEC50, "development pEC50")
    assay_weights(dev.pEC50_standard_error)
    result = dev[["record_id", "feature_row", "pEC50", "pEC50_standard_error"]].copy()
    result.insert(0, "row_index", np.arange(len(dev), dtype=np.int64))
    result.insert(3, "canonical_smiles", canonical)
    result["feature_row"] = result.feature_row.astype(np.int64)
    return result


def _balanced_group_folds(groups, n_folds, seed):
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < n_folds:
        raise ValueError("too few independent groups for requested folds")
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(unique)), -counts))
    sizes = np.zeros(n_folds, dtype=int)
    mapping = {}
    for i in order:
        fold = int(rng.choice(np.flatnonzero(sizes == sizes.min())))
        mapping[unique[i]] = fold
        sizes[fold] += counts[i]
    return np.array([mapping[g] for g in groups], dtype=np.int64)


def _make_assignments(inputs, chemical_groups, config):
    if inputs.canonical_smiles.duplicated().any():
        raise ValueError("duplicate identity is incompatible with exact-N budgets")
    if len(chemical_groups) != len(inputs):
        raise ValueError("chemical groups are misaligned")
    n_folds = config["n_folds"]
    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")
    policies = config["split_policies"]
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("split policies must be nonempty and unique")
    frames = []
    for split in policies:
        if split not in ("random", "chemical_cluster"):
            raise ValueError(f"unknown split policy: {split}")
        groups = (
            inputs.canonical_smiles.to_numpy() if split == "random" else chemical_groups
        )
        fold = _balanced_group_folds(groups, n_folds, config["split_seed"])
        frames.append(
            pd.DataFrame(
                dict(
                    split=split,
                    outer_fold=fold,
                    row_index=inputs.row_index.to_numpy(),
                    chemical_group=chemical_groups,
                )
            )
        )
    return pd.concat(frames, ignore_index=True)


def _make_subsets(assignments, config):
    budgets = config["budgets"]
    if not budgets or any(not isinstance(n, int) or n < 2 for n in budgets):
        raise ValueError("budgets must contain positive integers >= 2")
    if len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be unique")
    fraction = config["calibration_fraction"]
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if not isinstance(config["draws"], int) or config["draws"] < 1:
        raise ValueError("draws must be a positive integer")
    frames = []
    for split, table in assignments.groupby("split", sort=True):
        # Explicit policy codes keep the RNG stable if policy-list order changes.
        policy_code = {"random": 0, "chemical_cluster": 1}[split]
        for fold in range(config["n_folds"]):
            pool = np.sort(table.loc[table.outer_fold.ne(fold), "row_index"].to_numpy())
            if max(budgets) > len(pool):
                raise ValueError(f"acquired budget exceeds {split} fold {fold} pool")
            for draw in range(config["draws"]):
                seed = np.random.SeedSequence(
                    [config["subset_seed"], policy_code, fold, draw]
                )
                order = np.random.default_rng(seed).permutation(pool)
                requested = sorted(budgets) + (
                    [-1] if config["include_full_reference"] else []
                )
                for budget in requested:
                    n = len(pool) if budget == -1 else budget
                    n_cal = math.ceil(fraction * n)
                    if n_cal >= n:
                        raise ValueError("calibration leaves no fit rows")
                    roles = np.full(n, "fit", dtype="<U11")
                    roles[-n_cal:] = "calibration"
                    frames.append(
                        pd.DataFrame(
                            dict(
                                split=split,
                                outer_fold=fold,
                                draw=draw,
                                n_train=budget,
                                row_index=order[:n],
                                role=roles,
                            )
                        )
                    )
    return pd.concat(frames, ignore_index=True)


def _validate_tables(inputs, assignments, subsets, config):
    """Independently verify row membership, roles, exact budgets and nested sets."""
    n = len(inputs)
    if not np.array_equal(inputs.row_index.to_numpy(), np.arange(n)):
        raise ValueError("inputs row_index must be contiguous in file order")
    for col in ("record_id", "feature_row", "canonical_smiles"):
        if inputs[col].isna().any() or inputs[col].duplicated().any():
            raise ValueError(f"inputs {col} must be unique and nonmissing")
    _vector(inputs.pEC50, "pEC50")
    assay_weights(inputs.pEC50_standard_error, config.get("se_floor", 0.1))
    keys = ["split", "outer_fold", "draw", "n_train"]
    if assignments.duplicated(["split", "row_index"]).any():
        raise ValueError("duplicate assignment")
    if subsets.duplicated(keys + ["row_index"]).any():
        raise ValueError("duplicate acquired row")
    if not set(subsets.role) <= {"fit", "calibration"}:
        raise ValueError("invalid subset role")
    if set(assignments.split) != set(config["split_policies"]):
        raise ValueError("assignment split coverage differs from protocol")
    if not set(subsets.row_index) <= set(range(n)):
        raise ValueError("subset row_index outside inputs")
    expected_keys = set()
    for split in config["split_policies"]:
        table = assignments[assignments.split.eq(split)]
        if set(table.row_index) != set(range(n)) or len(table) != n:
            raise ValueError("assignment rows differ from inputs")
        if set(table.outer_fold) != set(range(config["n_folds"])):
            raise ValueError("outer fold coverage differs from protocol")
        if table.chemical_group.isna().any():
            raise ValueError("missing chemical group")
        if (
            split == "chemical_cluster"
            and table.groupby("chemical_group").outer_fold.nunique().max() != 1
        ):
            raise ValueError("chemical group crosses test folds")
        for fold in range(config["n_folds"]):
            pool = set(table.loc[table.outer_fold.ne(fold), "row_index"])
            for draw in range(config["draws"]):
                requested = sorted(config["budgets"]) + (
                    [-1] if config["include_full_reference"] else []
                )
                for budget in requested:
                    expected_keys.add((split, fold, draw, budget))
    actual = {key: block for key, block in subsets.groupby(keys, sort=False)}
    if set(actual) != expected_keys:
        raise ValueError("subset task coverage differs from protocol")
    for (split, fold, draw), _block in subsets.groupby(keys[:3], sort=False):
        table = assignments[assignments.split.eq(split)]
        pool = set(table.loc[table.outer_fold.ne(fold), "row_index"])
        previous = set()
        requested = sorted(config["budgets"]) + (
            [-1] if config["include_full_reference"] else []
        )
        for budget in requested:
            acquired = actual[(split, fold, draw, budget)]
            rows = set(acquired.row_index)
            expected_n = len(pool) if budget == -1 else budget
            n_cal = math.ceil(config["calibration_fraction"] * expected_n)
            if len(rows) != expected_n or not rows <= pool:
                raise ValueError("acquired budget mismatch or outer-test leakage")
            if acquired.role.eq("calibration").sum() != n_cal:
                raise ValueError("calibration count mismatch")
            if not previous <= rows:
                raise ValueError("acquired budgets are not nested")
            # Persistence preserves acquisition order; its suffix is calibration.
            if (
                not acquired.role.iloc[:-n_cal].eq("fit").all()
                or not acquired.role.iloc[-n_cal:].eq("calibration").all()
            ):
                raise ValueError("calibration is not the acquisition suffix")
            previous = rows
    # Verify prefixes and seeded order, not merely weaker set nestedness.
    expected = _make_subsets(assignments, config)
    if not subsets.reset_index(drop=True).equals(expected):
        raise ValueError("subset order or membership does not replay declared RNG")


def _native_scalars(member1, member2, checkpoint_path):
    """CPU frozen functional readout, with no labels or model fitting interface."""
    import torch
    from safetensors import safe_open
    from torch.nn import functional as F

    torch.set_num_threads(min(2, torch.get_num_threads()))
    values, logits, probabilities = [], [], []
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        with torch.inference_mode():
            for i, array in enumerate((member1, member2), start=1):
                prefix = (
                    "affinity_module" if i == 1 else "affinity_module2"
                ) + ".affinity_heads."
                x = torch.as_tensor(array, dtype=torch.float32)

                def linear(z, key, prefix=prefix):
                    return F.linear(
                        z,
                        checkpoint.get_tensor(prefix + key + ".weight"),
                        checkpoint.get_tensor(prefix + key + ".bias"),
                    )

                def branch(name, x=x, linear=linear):
                    z = x
                    for layer in (0, 2, 4):
                        z = linear(z, f"{name}.{layer}")
                        if layer != 4:
                            z = F.relu(z)
                    return z

                values.append(branch("to_affinity_pred_value").squeeze(-1))
                score = branch("to_affinity_pred_score")
                logits.append(linear(score, "to_affinity_logits_binary").squeeze(-1))
                probabilities.append(torch.sigmoid(logits[-1]))
            ensemble_value = (values[0] + values[1]) / 2
            ensemble_probability = (probabilities[0] + probabilities[1]) / 2
            ensemble_logit = torch.logit(ensemble_probability.clamp(1e-6, 1 - 1e-6))
            scalar = torch.stack(
                [
                    6 - values[0],
                    6 - values[1],
                    6 - ensemble_value,
                    logits[0],
                    logits[1],
                    ensemble_logit,
                    probabilities[0],
                    probabilities[1],
                    ensemble_probability,
                ],
                dim=1,
            )
    replay = {
        "affinity_pred_value1": values[0].numpy(),
        "affinity_pred_value2": values[1].numpy(),
        "affinity_pred_value": ensemble_value.numpy(),
        "affinity_logits_binary": ensemble_logit.numpy(),
        "affinity_probability_binary": ensemble_probability.numpy(),
    }
    return scalar.numpy(), replay


def _validate_features(features, n, bits):
    if set(features) != set(FEATURE_KEYS):
        raise ValueError("feature keys differ from contract")
    shapes = {
        "member1": (n, 384),
        "member2": (n, 384),
        "morgan": (n, bits),
        "native_continuous": (n,),
        "scalar_features": (n, len(SCALAR_COLUMNS)),
    }
    for key, array in features.items():
        if array.shape != shapes[key] or not np.isfinite(array).all():
            raise ValueError(f"invalid or misaligned feature array: {key}")
    if not np.isin(features["morgan"], [0, 1]).all():
        raise ValueError("Morgan features must be binary")
    if not np.allclose(
        features["native_continuous"],
        features["scalar_features"][:, 2],
        atol=REPLAY_ATOL,
        rtol=0,
    ):
        raise ValueError("native continuous and scalar replay disagree")


def _summary(array):
    array = np.asarray(array, dtype=float)
    return dict(
        minimum=float(array.min()),
        median=float(np.median(array)),
        mean=float(array.mean()),
        maximum=float(array.max()),
        q05=float(np.quantile(array, 0.05)),
        q95=float(np.quantile(array, 0.95)),
    )


def _split_audit(inputs, assignments, fps, groups, config):
    counts = pd.Series(groups).value_counts()
    scaffolds = np.array(
        [
            scaffold_group_key(s, bemis_murcko_scaffold(s))
            for s in inputs.canonical_smiles
        ]
    )
    nearest_rows, folds = [], []
    for split in config["split_policies"]:
        table = assignments[assignments.split.eq(split)].sort_values("row_index")
        for fold in range(config["n_folds"]):
            test = table.loc[table.outer_fold.eq(fold), "row_index"].to_numpy()
            train = table.loc[table.outer_fold.ne(fold), "row_index"].to_numpy()
            train_fps = [fps[i] for i in train]
            nearest = np.array(
                [
                    max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps))
                    for i in test
                ]
            )
            nearest_rows.extend(
                dict(
                    split=split,
                    outer_fold=fold,
                    row_index=int(i),
                    nearest_outer_train_similarity=float(sim),
                )
                for i, sim in zip(test, nearest, strict=True)
            )
            test_inputs = inputs.iloc[test]
            weights = assay_weights(
                test_inputs.pEC50_standard_error, config["se_floor"]
            )
            folds.append(
                dict(
                    split=split,
                    outer_fold=fold,
                    n_test=len(test),
                    n_outer_train=len(train),
                    identity_overlap=0,
                    chemical_group_overlap=len(set(groups[test]) & set(groups[train])),
                    scaffold_group_overlap=len(
                        set(scaffolds[test]) & set(scaffolds[train])
                    ),
                    test_compounds_with_train_scaffold=int(
                        np.isin(scaffolds[test], scaffolds[train]).sum()
                    ),
                    nearest_outer_train_similarity=_summary(nearest),
                    nearest_similarity_ge_cutoff_count=int(
                        (nearest >= config["chemical_similarity_cutoff"]).sum()
                    ),
                    nearest_similarity_ge_080_count=int((nearest >= 0.8).sum()),
                    pEC50=_summary(test_inputs.pEC50),
                    assay_standard_error=_summary(test_inputs.pEC50_standard_error),
                    effective_n=float(weights.sum() ** 2 / np.square(weights).sum()),
                )
            )
    audit = dict(
        status="pass",
        n_compounds=len(inputs),
        n_groups=len(counts),
        singleton_groups=int((counts == 1).sum()),
        singleton_compounds=int(counts[counts == 1].sum()),
        singleton_compound_fraction=float(counts[counts == 1].sum() / len(inputs)),
        compounds_in_multicompound_groups=int(counts[counts > 1].sum()),
        largest_group=int(counts.max()),
        group_size_histogram={
            str(k): int(v) for k, v in counts.value_counts().sort_index().items()
        },
        largest_groups=[
            {"chemical_group": int(k), "n_compounds": int(v)}
            for k, v in counts.nlargest(20).items()
        ],
        chemical_method="RDKit Butina, reordering=True, no scaffold union",
        similarity_cutoff=config["chemical_similarity_cutoff"],
        caveat=(
            "Butina is centroid grouping, not connected components: cross-fold "
            "similarity can exceed the cutoff. Random splits are identity-held-out, "
            "not series-held-out. Nearest similarities here use the full outer "
            "training pool; task novelty must be recalculated from fit rows only."
        ),
        acquisition=(
            "Nested seeded exact-N prefixes; last "
            "ceil(calibration_fraction*N) rows calibrate within N. "
            "Fit-only sets need not nest. All methods share these manifests."
        ),
        folds=folds,
    )
    return audit, pd.DataFrame(nearest_rows)


def _json_write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _json_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def prepare_data(repo_root: Path, output_dir: Path, config: dict) -> dict:
    """Prepare the 3,344 real development rows; never access challenge features."""
    import torch
    from safetensors import safe_open

    repo_root, output_dir = Path(repo_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "preparation_manifest.json").exists():
        raise FileExistsError(
            "preparation is immutable; verify/load the existing artifact instead"
        )
    if (
        config.get("invalid_se_policy") != "fail"
        or config.get("curation_weight_in_primary") is not False
    ):
        raise ValueError("requires fail-closed SE policy and no curation multiplier")
    if config["chemical_cluster_method"] != "butina":
        raise ValueError("only preregistered Butina is implemented")
    if not 0 < config["chemical_similarity_cutoff"] < 1:
        raise ValueError("chemical similarity cutoff must be in (0, 1)")
    sources = {
        key: repo_root / config[key] for key in ("manifest", "features", "checkpoint")
    }
    source_hashes = {key: sha256_file(path) for key, path in sources.items()}
    manifest = pd.read_csv(sources["manifest"])
    if len(manifest) != 4134 or not np.array_equal(
        np.sort(manifest.feature_row.to_numpy()), np.arange(4134)
    ):
        raise ValueError(
            "source manifest must map all 4134 cached feature rows exactly once"
        )
    se = pd.to_numeric(
        manifest.loc[manifest.modeling_role.eq("development"), "pEC50_standard_error"],
        errors="coerce",
    ).to_numpy()
    valid = np.isfinite(se) & (se > 0)
    coverage = dict(
        source_rows=len(manifest),
        development_rows=len(se),
        valid_se_rows=int(valid.sum()),
        invalid_se_rows=int((~valid).sum()),
        invalid_se_policy="fail",
        excluded_roles={
            str(k): int(v)
            for k, v in manifest.loc[
                manifest.modeling_role.ne("development"), "modeling_role"
            ]
            .value_counts(dropna=False)
            .items()
        },
    )
    _json_write(output_dir / "coverage_audit.json", coverage)
    inputs = _development_inputs(manifest)
    idx = inputs.feature_row.to_numpy()
    with safe_open(str(sources["features"]), framework="np") as cache:
        cached = {key: cache.get_tensor(key) for key in cache.keys()}
    if any(len(array) != len(manifest) for array in cached.values()):
        raise ValueError("cached array row counts differ from manifest")
    member1 = np.asarray(cached["member1_affinity_repr"][idx], dtype=np.float32)
    member2 = np.asarray(cached["member2_affinity_repr"][idx], dtype=np.float32)
    scalar, reconstructed = _native_scalars(member1, member2, sources["checkpoint"])
    replay = {}
    for key, predicted in reconstructed.items():
        if key in cached:
            stored = cached[key][idx]
            error = float(np.max(np.abs(stored - predicted)))
            replay[key] = error
            if not np.isfinite(error) or error > REPLAY_ATOL:
                raise ValueError(f"native replay exceeds {REPLAY_ATOL}: {key}, {error}")
    for member in (1, 2):
        key = f"reconstructed_affinity_pred_value{member}"
        if key in cached:
            error = float(
                np.max(
                    np.abs(
                        cached[key][idx] - reconstructed[f"affinity_pred_value{member}"]
                    )
                )
            )
            replay[key] = error
            if not np.isfinite(error) or error > REPLAY_ATOL:
                raise ValueError(f"stored reconstructed replay failed: {key}")
    for required in (
        "affinity_pred_value",
        "affinity_pred_value1",
        "affinity_pred_value2",
    ):
        if required not in replay:
            raise ValueError(f"missing required native replay reference: {required}")
    fps = morgan_fingerprints(
        inputs.canonical_smiles.tolist(),
        radius=config["morgan_radius"],
        n_bits=config["fingerprint_bits"],
    )
    morgan = np.stack([np.asarray(fp, dtype=np.uint8) for fp in fps])
    features = dict(
        member1=member1,
        member2=member2,
        morgan=morgan,
        native_continuous=6.0 - cached["affinity_pred_value"][idx],
        scalar_features=scalar,
    )
    _validate_features(features, len(inputs), config["fingerprint_bits"])
    groups = butina_cluster_ids(
        fps, similarity_cutoff=config["chemical_similarity_cutoff"]
    )
    assignments = _make_assignments(inputs, groups, config)
    subsets = _make_subsets(assignments, config)
    _validate_tables(inputs, assignments, subsets, config)
    audit, nearest = _split_audit(inputs, assignments, fps, groups, config)
    weights = assay_weights(inputs.pEC50_standard_error, config["se_floor"])
    coverage.update(
        weight_definition="1 / max(assay_standard_error, se_floor)",
        se_floor=config["se_floor"],
        effective_n=float(weights.sum() ** 2 / np.square(weights).sum()),
        weight_sum=float(weights.sum()),
        weights=_summary(weights),
        floor_capped_rows=int((inputs.pEC50_standard_error < config["se_floor"]).sum()),
        curation_weight_used=False,
    )
    _json_write(output_dir / "coverage_audit.json", coverage)
    inputs.to_csv(output_dir / "inputs.csv", index=False)
    np.savez_compressed(output_dir / "features.npz", **features)
    assignments.to_csv(output_dir / "assignments.csv", index=False)
    subsets.to_csv(output_dir / "subsets.csv", index=False)
    nearest.to_csv(output_dir / "outer_nearest_similarity.csv", index=False)
    _json_write(output_dir / "split_audit.json", audit)
    _json_write(output_dir / "protocol.json", config)
    files = [
        "inputs.csv",
        "features.npz",
        "assignments.csv",
        "subsets.csv",
        "split_audit.json",
        "outer_nearest_similarity.csv",
        "coverage_audit.json",
        "protocol.json",
    ]
    # Rehash source artifacts after preparation to reject concurrent mutation.
    if source_hashes != {key: sha256_file(path) for key, path in sources.items()}:
        raise ValueError("source artifacts changed during preparation")
    code_paths = [
        "src/nesso_pxr/low_data_contract.py",
        "src/nesso_pxr/chemistry.py",
        "src/nesso_pxr/splits.py",
        "src/nesso_pxr/protocol.py",
        "scripts/prepare_low_data.py",
    ]
    result = dict(
        status="pass",
        schema_version="1.0.0",
        config=config,
        config_sha256=_json_hash(config),
        sources={k: {"path": config[k], "sha256": source_hashes[k]} for k in sources},
        code_sha256={p: sha256_file(repo_root / p) for p in code_paths},
        artifact_sha256={p: sha256_file(output_dir / p) for p in files},
        n_inputs=len(inputs),
        n_assignments=len(assignments),
        n_subset_rows=len(subsets),
        n_subset_tasks=int(
            subsets.groupby(["split", "outer_fold", "draw", "n_train"]).ngroups
        ),
        feature_shapes={k: list(v.shape) for k, v in features.items()},
        feature_dtypes={k: str(v.dtype) for k, v in features.items()},
        feature_array_sha256={
            k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            for k, v in features.items()
        },
        ordered_record_id_sha256=_json_hash(inputs.record_id.tolist()),
        scalar_columns=list(SCALAR_COLUMNS),
        native_replay=dict(
            atol=REPLAY_ATOL,
            rtol=0.0,
            maximum_absolute_errors=replay,
            binary_reference_available="affinity_probability_binary" in cached,
            binary_note=(
                "Binary branches reconstructed from the released checkpoint; "
                "current cache contains no binary native reference unless "
                "explicitly listed. Do not claim historical binary-output parity."
            ),
        ),
        prediction_flow=(
            "Frozen cached 384D member vectors -> released continuous "
            "and binary heads -> 9 raw scalar outputs. Native continuous "
            "= 6 - ensemble affinity; pIC50-equivalent, endpoint mismatch "
            "to cellular pEC50. No fitting, scaling, target-label calibration, "
            "clipping of potency, or outcome-dependent split selection."
        ),
        binary_ensemble_rule=(
            "mean member sigmoid probabilities; ensemble logit "
            "is logit(mean probability clamped to [1e-6, 1-1e-6])"
        ),
        challenge_label_training=False,
        model_fits=0,
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            rdkit=rdBase.rdkitVersion,
            torch=torch.__version__,
        ),
        cpu_threads=torch.get_num_threads(),
        coverage=coverage,
    )
    _json_write(output_dir / "preparation_manifest.json", result)
    manifest_hash = sha256_file(output_dir / "preparation_manifest.json")
    (output_dir / "preparation_manifest.sha256").write_text(
        manifest_hash + "  preparation_manifest.json\n"
    )
    load_prepared(output_dir)
    return result


def load_prepared(
    output_dir,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Read a complete preparation, verifying hashes and strict row alignment."""
    root = Path(output_dir)
    manifest_path = root / "preparation_manifest.json"
    expected_hash = (root / "preparation_manifest.sha256").read_text().split()[0]
    if sha256_file(manifest_path) != expected_hash:
        raise ValueError("preparation manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["status"] != "pass"
        or _json_hash(manifest["config"]) != manifest["config_sha256"]
    ):
        raise ValueError("invalid preparation status or config hash")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"prepared artifact hash mismatch: {name}")
    inputs = pd.read_csv(root / "inputs.csv")
    with np.load(root / "features.npz", allow_pickle=False) as arrays:
        features = {k: arrays[k] for k in arrays.files}
    assignments = pd.read_csv(root / "assignments.csv")
    subsets = pd.read_csv(root / "subsets.csv")
    _validate_features(features, len(inputs), manifest["config"]["fingerprint_bits"])
    _validate_tables(inputs, assignments, subsets, manifest["config"])
    if _json_hash(inputs.record_id.tolist()) != manifest["ordered_record_id_sha256"]:
        raise ValueError("record order hash mismatch")
    for key, array in features.items():
        if (
            hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            != manifest["feature_array_sha256"][key]
        ):
            raise ValueError(f"feature array digest mismatch: {key}")
    return inputs, features, assignments, subsets
