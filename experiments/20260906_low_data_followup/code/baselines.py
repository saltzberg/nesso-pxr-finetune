"""Retrospective, within-budget controls and explicitly nonhistorical 2D comparator.

Raw row-wise descriptors may be cached globally; no fitted transformation may be.
Pickle replay states are trusted-local artifacts only. No output clipping.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from nesso_pxr.low_data_contract import assay_weights
from nesso_pxr.low_data_models import FEATURE_KEYS, _folds

METHODS = ("weighted_median", "native_affine", "descriptor_lightgbm_rdkit_mordred")
DESCRIPTOR_METHOD = METHODS[-1]
RECIPE = {
    "identity": "canonical_rdkit2d_mordred2d_morgan_trainonly_reduction_lgbm_v1",
    "variance_threshold": 0.01,
    "correlation_threshold": 0.90,
    "top_k": 1000,
    "gain_selector_estimators": 150,
    "gain_selector_leaves": 63,
    "gain_selector_min_child_samples": 5,
    "grid": [
        {"num_leaves": leaves, "min_child_samples": child, "n_estimators": rounds}
        for leaves, child in ((7, 5), (63, 20))
        for rounds in (50, 150, 500)
    ],
    "learning_rate": 0.03,
    "reg_lambda": 5.0,
    "objective": "regression",
    "affine_slope_bounds": [0.0, 2.0],
}


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hash_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def descriptor_schema():
    from mordred import Calculator, descriptors
    from rdkit.Chem import Descriptors
    names = (["rdkit2d:" + n for n, _ in Descriptors._descList]
             + ["mordred2d:" + str(d) for d in Calculator(descriptors, ignore_3D=True).descriptors]
             + [f"morgan_r2_2048_nochirality:{i:04d}" for i in range(2048)])
    if len(set(names)) != len(names):
        raise ValueError("duplicate raw feature name")
    meta = {"feature_names": names, "versions": {p: version(p) for p in ("rdkit", "mordredcommunity", "numpy")},
            "structure_policy": "RDKit MolFromSmiles on supplied canonical_smiles; no prepared-top state selection",
            "morgan": {"radius": 2, "fpSize": 2048, "includeChirality": False},
            "mordred_ignore_3D": True, "learned_preprocessing": False,
            "identity": RECIPE["identity"]}
    meta["feature_identity_sha256"] = _hash_json(meta)
    return meta


def _identity(inputs):
    allowed = {"record_id", "canonical_smiles", "row_index"}
    if set(inputs.columns) - allowed:
        raise ValueError("descriptor preparation accepts structure/identity columns only, not labels")
    if not {"record_id", "canonical_smiles"} <= set(inputs.columns):
        raise ValueError("missing descriptor identity columns")
    if inputs[["record_id", "canonical_smiles"]].isna().any().any() or inputs.record_id.duplicated().any():
        raise ValueError("missing or duplicate record identity")
    return {"record_ids": inputs.record_id.astype(str).tolist(),
            "canonical_smiles": inputs.canonical_smiles.astype(str).tolist()}


def prepare_descriptors(inputs, output_dir=None):
    """Compute raw row-wise RDKit/Mordred/Morgan; return array, identity metadata.

    Any undefined or infinite descriptor becomes NaN (not a globally fitted value).
    The cache is immutable; existing outputs must match identity and schema exactly.
    """
    from mordred import Calculator, descriptors
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    identity = _identity(inputs)
    schema = descriptor_schema()
    if output_dir is not None and (Path(output_dir) / "descriptor_metadata.json").exists():
        return load_descriptors(output_dir, inputs)
    calc = Calculator(descriptors, ignore_3D=True)
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    x = np.empty((len(inputs), len(schema["feature_names"])), dtype=np.float64)
    with threadpool_limits(limits=2):
        for i, smi in enumerate(identity["canonical_smiles"]):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"invalid canonical SMILES at row {i}")
            values = []
            for _, fn in Descriptors._descList:
                try:
                    values.append(float(fn(mol)))
                except Exception:
                    values.append(np.nan)
            for v in calc(mol):
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    values.append(np.nan)
            x[i] = np.r_[values, np.asarray(fp.GetFingerprint(mol), dtype=np.float64)]
            if i and i % 100 == 0:
                print(f"DESCRIPTORS {i}/{len(inputs)}", flush=True)
    x[~np.isfinite(x)] = np.nan
    metadata = {**schema, **identity, "ordered_identity_sha256": _hash_json(identity),
                "shape": list(x.shape), "dtype": str(x.dtype), "array_sha256": _hash_array(x),
                "missing_per_feature": np.isnan(x).sum(axis=0).tolist()}
    if output_dir is not None:
        output_dir = Path(output_dir)
        buf = io.BytesIO()
        np.savez_compressed(buf, descriptors=x)
        path = output_dir / "descriptors.npz"
        if path.exists():
            raise ValueError("orphan descriptor file; preserve and use a new feature directory")
        _atomic(path, buf.getvalue())
        metadata["file_sha256"] = _digest(path)
        _atomic(output_dir / "descriptor_metadata.json", (json.dumps(metadata, indent=2) + "\n").encode())
    return x, metadata


def load_descriptors(directory, inputs):
    directory = Path(directory)
    meta = json.loads((directory / "descriptor_metadata.json").read_text())
    if meta["ordered_identity_sha256"] != _hash_json(_identity(inputs)):
        raise ValueError("descriptor ordered identity mismatch")
    if meta["feature_identity_sha256"] != descriptor_schema()["feature_identity_sha256"]:
        raise ValueError("descriptor implementation/feature identity mismatch")
    if meta["file_sha256"] != _digest(directory / "descriptors.npz"):
        raise ValueError("descriptor file hash mismatch")
    with np.load(directory / "descriptors.npz", allow_pickle=False) as f:
        x = f["descriptors"]
    if list(x.shape) != meta["shape"] or _hash_array(x) != meta["array_sha256"]:
        raise ValueError("descriptor array hash/shape mismatch")
    return x, meta


class TrainOnlyReducer:
    """Mean imputation, variance filter and correlation components fit on X only.

    Variance uses finite values, ddof=0; standardized mean-imputed values determine
    correlation with denominator n. Representatives: missing fraction ascending,
    variance descending, original feature index ascending. Different from missing
    historical source-priority tie-breaks, explicitly part of this comparator.
    """
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean_ = np.nanmean(x, axis=0)
            variance = np.nanvar(x, axis=0)
        self.mean_[~np.isfinite(self.mean_)] = 0.0
        self.variance_ = variance
        self.n_fit_ = len(x)
        self.fit_array_sha256_ = _hash_array(x)
        missing = np.isnan(x).mean(axis=0)
        keep = np.flatnonzero(np.isfinite(variance) & (variance > RECIPE["variance_threshold"]))
        if not len(keep):
            self.selected_ = np.empty(0, dtype=int)
            return self
        z = (np.where(np.isnan(x[:, keep]), self.mean_[keep], x[:, keep]) - self.mean_[keep]) / np.sqrt(variance[keep])
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        rows, cols = [], []
        for start in range(0, len(keep), 256):
            c = np.abs(z[:, start:start + 256].T @ z / len(x))
            a, b = np.where(c > RECIPE["correlation_threshold"])
            valid = b > a + start
            rows.extend((a[valid] + start).tolist())
            cols.extend(b[valid].tolist())
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(keep), len(keep))).tocsr()
        _, components = connected_components(graph, directed=False)
        rank = np.lexsort((keep, -variance[keep], missing[keep]))
        chosen = {}
        for i in rank:
            chosen.setdefault(int(components[i]), int(keep[i]))
        self.selected_ = np.asarray(sorted(chosen.values()), dtype=int)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)[:, self.selected_]
        return np.where(np.isnan(x), self.mean_[self.selected_], x)

    def audit(self):
        return {"n_fit": self.n_fit_, "fit_array_sha256": self.fit_array_sha256_,
                "n_raw": len(self.mean_), "n_reduced": len(self.selected_),
                "selected_indices": self.selected_.tolist(), "mean_sha256": _hash_array(self.mean_)}


def _lgbm(candidate, seed):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective="regression", learning_rate=0.03,
                         reg_lambda=5.0, reg_alpha=0.0, subsample=1.0,
                         colsample_bytree=1.0, max_depth=-1, min_child_weight=1e-3,
                         deterministic=True, force_col_wise=True, verbosity=-1,
                         n_jobs=2, random_state=int(seed), bagging_seed=int(seed),
                         feature_fraction_seed=int(seed), data_random_seed=int(seed),
                         **candidate)


def _reduce(x, y, w, seed):
    reducer = TrainOnlyReducer().fit(x)
    z = reducer.transform(x)
    if z.shape[1]:
        selector = _lgbm({"num_leaves": 63, "min_child_samples": 5, "n_estimators": 150}, seed)
        selector.fit(z, y, sample_weight=w)
        gain = selector.booster_.feature_importance(importance_type="gain")
        selected = np.argsort(-gain, kind="stable")[:RECIPE["top_k"]]
    else:
        gain = np.empty(0)
        selected = np.empty(0, dtype=int)
    return reducer, selected, gain


def _validate(features, expected=None):
    unknown = set(features) - (set(FEATURE_KEYS) | {"descriptors"})
    if unknown:
        raise ValueError(f"unknown/possibly labeled feature keys: {sorted(unknown)}")
    if not features:
        raise ValueError("empty feature dictionary")
    out = {k: np.asarray(v) for k, v in features.items()}
    n = len(next(iter(out.values()))) if expected is None else expected
    for k, v in out.items():
        if v.ndim not in (1, 2) or len(v) != n:
            raise ValueError(f"misaligned feature {k}")
        if k == "groups":
            if v.ndim != 1 or any(x is None or str(x) in ("", "nan", "<NA>") for x in v):
                raise ValueError("invalid groups")
        elif k == "descriptors":
            if v.ndim != 2 or not v.shape[1] or np.isinf(v).any():
                raise ValueError("raw descriptors must be nonempty-width 2D; missing is NaN, not infinity")
        elif not np.isfinite(v).all():
            raise ValueError(f"nonfinite feature {k}")
    return out, n


def weighted_median(y, weights):
    order = np.argsort(y, kind="stable")
    i = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side="left")
    return float(y[order[min(i, len(order) - 1)]])


def _predict(state, features):
    n = len(next(iter(features.values())))
    if state["kind"] == "constant":
        return np.full(n, state["value"], dtype=np.float64)
    if state["kind"] == "affine":
        return state["intercept"] + state["slope"] * features["native_continuous"]
    x = state["reducer"].transform(features["descriptors"])[:, state["gain_selected"]]
    return state["booster"].predict(x, num_threads=2)


def predict_model(model_state, features, *, batch_size=1024):
    del batch_size  # Trees have no batch-shape numerical dependency.
    if model_state.get("schema_version") != "followup_baselines_v1" or model_state["method"] not in METHODS:
        raise ValueError("unsupported baseline state")
    features, _ = _validate(features)
    for key, shape in model_state["feature_shapes"].items():
        if key not in features or list(features[key].shape[1:]) != shape:
            raise ValueError(f"replay feature shape mismatch: {key}")
    with threadpool_limits(limits=2):
        p = np.asarray(_predict(model_state["model"], features), dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise FloatingPointError("nonfinite/invalid predictions")
    return {"pred": p, "ensemble_std": np.zeros_like(p),
            "seed_predictions": np.repeat(p[None], len(model_state["seeds"]), axis=0)}


def fit_predict(method, train_features, y_train, assay_se_train, test_features, *,
                config, checkpoint_path, seeds=(42, 43, 44)):
    del checkpoint_path
    started = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f"unknown baseline {method}")
    y = np.asarray(y_train, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("fit labels must be finite nonempty 1D")
    if float(config.get("se_floor", .10)) != .10:
        raise ValueError("frozen fitting SE floor is .10")
    w = assay_weights(assay_se_train, .10)
    if w.shape != y.shape:
        raise ValueError("SE/label shape mismatch")
    w = w / w.mean()
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty unique")
    train, _ = _validate(train_features, len(y))
    test, ntest = _validate(test_features)
    required = {"descriptors"} if method == DESCRIPTOR_METHOD else {"native_continuous"} if method == "native_affine" else set()
    for key in required:
        if key not in train or key not in test or train[key].shape[1:] != test[key].shape[1:]:
            raise ValueError(f"missing/mismatched required feature: {key}")
    selection = {"metric": "pooled_inner_oof_assay_weighted_mae", "candidates": []}
    audit = {"n_fit": len(y), "n_prediction_rows": ntest, "threads": 2,
             "test_labels_accessed": False, "calibration_labels_accessed": False,
             "actual_final_fits": 1, "seed_rows_are_deterministic_copies": True,
             "fit_label_indices": list(range(len(y))), "prediction_clipping": False,
             "weight_definition": "1/max(SE,.10), normalized to training mean 1",
             "inner_transform_fits": []}
    with threadpool_limits(limits=2):
        if method == "weighted_median":
            model = {"kind": "constant", "value": weighted_median(y, w)}
            count = 1
        elif method == "native_affine":
            x = train["native_continuous"]
            if x.ndim != 1:
                raise ValueError("native_continuous must be one-dimensional")
            mx, my = np.average(x, weights=w), np.average(y, weights=w)
            variance = np.average((x - mx) ** 2, weights=w)
            slope = float(np.clip(np.average((x - mx) * (y - my), weights=w) / variance, 0, 2)) if variance > 0 else 0.0
            model = {"kind": "affine", "slope": slope, "intercept": float(my - slope * mx)}
            audit["affine_objective"] = "weighted squared error, slope constrained [0,2], unconstrained intercept; zero variance slope=0"
            count = 2
        else:
            folds, foldmeta = _folds(train, len(y), int(config.get("inner_folds", 3)), seeds[0])
            selection.update(foldmeta)
            oof = np.empty((len(RECIPE["grid"]), len(y)))
            for fold, (tr, va) in enumerate(folds):
                reducer, chosen, gain = _reduce(train["descriptors"][tr], y[tr], w[tr] / w[tr].mean(), seeds[0])
                a = reducer.transform(train["descriptors"][tr])[:, chosen]
                b = reducer.transform(train["descriptors"][va])[:, chosen]
                audit["inner_transform_fits"].append({"fold": fold, "train_indices": tr.tolist(),
                    "validation_indices": va.tolist(), **reducer.audit(), "gain_selected": chosen.tolist(),
                    "gain_sha256": _hash_array(gain), "gain_label_indices": tr.tolist()})
                for i, candidate in enumerate(RECIPE["grid"]):
                    if not a.shape[1]:
                        oof[i, va] = np.average(y[tr], weights=w[tr])
                    else:
                        fitted = _lgbm(candidate, seeds[0])
                        fitted.fit(a, y[tr], sample_weight=w[tr] / w[tr].mean())
                        oof[i, va] = fitted.booster_.predict(b, num_threads=2)
            scores = np.average(np.abs(oof - y), weights=w, axis=1)
            best = int(np.argmin(scores))
            selection["candidates"] = [{"parameters": c, "weighted_mae": float(s)} for c, s in zip(RECIPE["grid"], scores, strict=True)]
            selection["selected"] = RECIPE["grid"][best]
            reducer, chosen, gain = _reduce(train["descriptors"], y, w, seeds[0])
            audit["final_transform_fit"] = {**reducer.audit(), "train_indices": list(range(len(y))),
                                             "gain_selected": chosen.tolist(), "gain_sha256": _hash_array(gain)}
            x = reducer.transform(train["descriptors"])[:, chosen]
            if not x.shape[1]:
                model = {"kind": "constant", "value": float(np.average(y, weights=w))}
                audit["no_variable_features_fallback"] = "weighted mean (squared-loss optimum)"
                count = 1
            else:
                fitted = _lgbm(RECIPE["grid"][best], seeds[0])
                fitted.fit(x, y, sample_weight=w)
                model = {"kind": "descriptor_lgbm", "booster": fitted.booster_, "reducer": reducer,
                         "gain_selected": chosen, "gain": gain}
                count = sum(t["num_leaves"] for t in fitted.booster_.dump_model()["tree_info"])
            audit["descriptor_identity"] = config.get("descriptor_feature_identity", "caller_unspecified; raw width bound only")
        state = {"schema_version": "followup_baselines_v1", "method": method, "model": model,
                 "seeds": seeds, "threads": 2, "recipe": RECIPE,
                 "feature_shapes": {k: list(train[k].shape[1:]) for k in required},
                 "descriptor_feature_identity": config.get("descriptor_feature_identity")}
        result = predict_model(state, test)
        restored = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        replay = predict_model(restored, test)
        np.testing.assert_array_equal(result["seed_predictions"], replay["seed_predictions"])
    audit["serialization_replay_max_abs_error"] = 0.0
    audit["prediction_quantiles"] = np.quantile(result["pred"], [0, .01, .5, .99, 1]).tolist() if ntest else []
    return {**result, "n_parameters": int(count), "fit_seconds": time.perf_counter() - started,
            "selection": selection, "fit_audit": audit, "model_state": state}
