"""Leakage-isolated low-data PXR readouts over immutable cached features.

Only fit labels enter this module. ``test_features`` may concatenate calibration
and test rows, but must contain no labels. Native values are pIC50-equivalent,
not native cellular pEC50 predictions. Head LoRA adapts the final readout MLPs,
not the frozen upstream representation. Pickle artifacts are trusted-local only.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

METHODS = (
    "weighted_mean",
    "native_continuous",
    "scalar_ridge",
    "repr_ridge",
    "repr_mlp",
    "head_random",
    "head_pretrained",
    "dual_head",
    "head_lora",
    "morgan_ridge",
    "morgan_lightgbm",
    "morgan_knn",
)
NEURAL_METHODS = frozenset(
    ("repr_mlp", "head_random", "head_pretrained", "dual_head", "head_lora")
)
FEATURE_KEYS = frozenset(
    ("member1", "member2", "morgan", "native_continuous", "scalar_features", "groups")
)


class InfeasibleInnerCV(ValueError):
    """Fit-only groups cannot support the requested model-selection experiment."""


def _weights(se: np.ndarray, floor: float) -> np.ndarray:
    from nesso_pxr.low_data_contract import assay_weights

    return np.asarray(assay_weights(se, floor=floor), dtype=np.float64)


def _weighted_mae(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    if not np.isfinite(p).all():
        raise FloatingPointError("nonfinite model predictions")
    return float(np.average(np.abs(y - p), weights=w))


def _matrix(features: dict, method: str) -> np.ndarray:
    if method in ("scalar_ridge",):
        return np.asarray(features["scalar_features"], dtype=np.float64)
    if method.startswith("morgan_"):
        return np.asarray(features["morgan"], dtype=np.float64)
    return np.concatenate([features["member1"], features["member2"]], axis=1).astype(
        np.float64
    )


def _slice(features: dict, indices: np.ndarray) -> dict:
    return {key: value[indices] for key, value in features.items()}


def _required_features(method: str) -> set[str]:
    if method == "native_continuous":
        return {"native_continuous"}
    if method == "scalar_ridge":
        return {"scalar_features"}
    if method.startswith("morgan_"):
        return {"morgan"}
    if method == "weighted_mean":
        return set()
    return {"member1", "member2"}


def _validate_features(
    features: dict, *, expected: int | None = None
) -> tuple[dict, int]:
    unknown = set(features) - FEATURE_KEYS
    if unknown:
        raise ValueError(
            f"feature dictionary has unknown/possibly labeled keys: {sorted(unknown)}"
        )
    converted = {k: np.asarray(v) for k, v in features.items()}
    if not converted:
        raise ValueError("empty feature dictionary")
    for key, value in converted.items():
        if value.ndim not in (1, 2):
            raise ValueError(f"invalid feature shape for {key}")
        if expected is None:
            expected = len(value)
        if len(value) != expected:
            raise ValueError(f"unaligned feature rows for {key}")
        if key == "groups":
            if value.ndim != 1 or any(
                v is None or str(v).lower() in ("nan", "<na>", "") for v in value
            ):
                raise ValueError("groups must be nonmissing 1D identifiers")
        elif not np.isfinite(value).all():
            raise ValueError(f"nonfinite features: {key}")
        elif key in ("member1", "member2") and (
            value.ndim != 2 or value.shape[1] != 384
        ):
            raise ValueError("Nesso member features must have exactly 384 columns")
        elif key in ("morgan", "scalar_features") and (
            value.ndim != 2 or value.shape[1] < 1
        ):
            raise ValueError(f"{key} must be a nonempty-width matrix")
        elif key == "native_continuous" and value.ndim != 1:
            raise ValueError("native_continuous must be 1D pIC50-equivalent values")
    return converted, int(expected)


def _folds(features: dict, n: int, count: int, seed: int) -> tuple[list, dict]:
    if count < 2:
        raise InfeasibleInnerCV("inner_folds must be at least 2")
    if "groups" in features:
        groups = features["groups"].astype(str)
        unique = len(np.unique(groups))
        actual = min(count, unique)
        if actual < 2:
            raise InfeasibleInnerCV(
                f"inner group CV infeasible: {unique} distinct groups; need at least 2"
            )
        splits = list(GroupKFold(n_splits=actual).split(np.arange(n), groups=groups))
        policy = "GroupKFold; supplied chemical/identity groups never split"
    else:
        unique = n
        actual = min(count, n)
        if actual < 2:
            raise InfeasibleInnerCV("inner CV infeasible: fewer than 2 fit rows")
        splits = list(
            KFold(n_splits=actual, shuffle=True, random_state=seed).split(np.arange(n))
        )
        policy = "shuffled KFold; caller guarantees unique random-split identities"
    metadata = dict(
        policy=policy,
        requested_folds=count,
        actual_folds=actual,
        distinct_groups=unique,
        reduced_folds=actual != count,
        folds=[
            dict(train_indices=a.tolist(), validation_indices=b.tolist())
            for a, b in splits
        ],
    )
    return splits, metadata


def _branch() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(384, 384),
        nn.ReLU(),
        nn.Linear(384, 384),
        nn.ReLU(),
        nn.Linear(384, 1),
    )


class ReadoutLoRALinear(nn.Module):
    """Zero-update low-rank delta inside a frozen final-readout MLP."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        self.scale = 1.0  # explicit alpha=rank, therefore alpha/rank=1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        )


class DualMember(nn.Module):
    """Continuous + 1.5*tanh(scale*binary_logit+offset), zero-initial shift."""

    def __init__(self):
        super().__init__()
        self.continuous = _branch()
        self.binary_score = _branch()
        self.binary_logit = nn.Linear(1, 1)
        self.binary_scale = nn.Parameter(torch.zeros(()))
        self.binary_offset = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        continuous = self.continuous(x).squeeze(-1)
        logit = self.binary_logit(self.binary_score(x)).squeeze(-1)
        return continuous + 1.5 * torch.tanh(
            self.binary_scale * logit + self.binary_offset
        )


class TwinReadout(nn.Module):
    def __init__(self, dual: bool = False):
        super().__init__()
        self.member1 = DualMember() if dual else _branch()
        self.member2 = DualMember() if dual else _branch()

    def forward(self, x1, x2):
        # Released raw affinity is log10(micromolar); conversion gives a
        # pIC50-equivalent anchor, trained here against cellular pEC50.
        p1 = 6.0 - self.member1(x1).reshape(-1)
        p2 = 6.0 - self.member2(x2).reshape(-1)
        return p1, p2, (p1 + p2) / 2


class RepresentationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x1, x2):
        p = self.layers(torch.cat((x1, x2), dim=1)).squeeze(-1)
        return p, p, p


def _checkpoint_heads(path: Path | str) -> dict:
    from safetensors import safe_open

    # Avoid materializing the full multi-GB upstream checkpoint.
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = [
            k
            for k in handle.keys()
            if k.startswith(
                (
                    "affinity_module.affinity_heads.to_affinity_",
                    "affinity_module2.affinity_heads.to_affinity_",
                )
            )
        ]
        return {k: handle.get_tensor(k) for k in keys}


def _load_branch(module: nn.Module, state: dict, prefix: str) -> None:
    module.load_state_dict(
        {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)},
        strict=True,
    )


def make_neural_model(
    method: str,
    config: dict,
    checkpoint_path: Path | str | None,
    *,
    seed: int = 42,
    checkpoint_state: dict | None = None,
) -> nn.Module:
    """Build the exact model at zero updates, permitting parity/gradient audits.

    Random/pretrained twins differ ONLY in weight initialization. All architectures
    are fixed, not selected from historical larger-dataset settings. The dual-head
    bounded fusion follows the existing adapter architecture; no historical epoch,
    anchor regularization, or early-stopping choices are imported.
    """
    if method not in NEURAL_METHODS:
        raise ValueError(f"not a neural method: {method}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if method == "repr_mlp":
            return RepresentationMLP()
        model = TwinReadout(dual=method == "dual_head")
        if method != "head_random":
            state = (
                checkpoint_state
                if checkpoint_state is not None
                else _checkpoint_heads(checkpoint_path)
            )
            for member, prefix in (
                (model.member1, "affinity_module"),
                (model.member2, "affinity_module2"),
            ):
                prefix += ".affinity_heads."
                if method == "dual_head":
                    _load_branch(
                        member.continuous, state, prefix + "to_affinity_pred_value."
                    )
                    _load_branch(
                        member.binary_score, state, prefix + "to_affinity_pred_score."
                    )
                    _load_branch(
                        member.binary_logit,
                        state,
                        prefix + "to_affinity_logits_binary.",
                    )
                else:
                    _load_branch(member, state, prefix + "to_affinity_pred_value.")
        if method == "head_lora":
            rank = int(config.get("head_lora_rank", 4))
            if rank < 1:
                raise ValueError("head_lora_rank must be positive")
            model.requires_grad_(False)
            for member in (model.member1, model.member2):
                for index in (0, 2, 4):
                    member[index] = ReadoutLoRALinear(member[index], rank)
        return model


def weighted_smooth_l1(
    predictions: tuple,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    beta: float = 0.5,
) -> torch.Tensor:
    """50/25/25 ensemble/member weighted smooth-L1, explicitly not HuberLoss."""
    p1, p2, mean = predictions
    losses = (
        0.5 * F.smooth_l1_loss(mean, target, reduction="none", beta=beta)
        + 0.25 * F.smooth_l1_loss(p1, target, reduction="none", beta=beta)
        + 0.25 * F.smooth_l1_loss(p2, target, reduction="none", beta=beta)
    )
    return (losses * weights).sum() / weights.sum()


def _neural_inputs(features: dict, scaler: StandardScaler | None) -> tuple:
    if scaler is not None:
        x = scaler.transform(_matrix(features, "repr_mlp")).astype(np.float32)
        return torch.from_numpy(x[:, :384]), torch.from_numpy(x[:, 384:])
    return tuple(
        torch.as_tensor(np.array(features[k], dtype=np.float32, copy=True))
        for k in ("member1", "member2")
    )


def _predict_neural(model, features, scaler) -> np.ndarray:
    model.eval()
    if not len(features["member1"]):
        return np.empty(0, dtype=np.float64)
    with torch.inference_mode():
        return model(*_neural_inputs(features, scaler))[2].numpy().astype(np.float64)


def _train_neural(
    method,
    features,
    y,
    w,
    config,
    checkpoint_state,
    seed,
    learning_rate,
    epochs,
    validation=None,
):
    scaler = None
    if method == "repr_mlp":
        scaler = StandardScaler().fit(
            _matrix(features, method), sample_weight=w / w.mean()
        )
    model = make_neural_model(
        method, config, None, seed=seed, checkpoint_state=checkpoint_state
    )
    x1, x2 = _neural_inputs(features, scaler)
    target = torch.as_tensor(y, dtype=torch.float32)
    weight = torch.as_tensor(w / w.mean(), dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(int(seed))
    validation_predictions = []
    losses = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        total_loss = 0.0
        for ix in order.split(int(config.get("batch_size", 64))):
            optimizer.zero_grad(set_to_none=True)
            # Global weight normalization gives an unbiased minibatch objective;
            # batch-local renormalization would overweight noisy-only batches.
            loss = weighted_smooth_l1(
                model(x1[ix], x2[ix]),
                target[ix],
                weight[ix],
                beta=float(config.get("neural_huber_beta", 0.5)),
            )
            loss = loss * weight[ix].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite neural fitting loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ix)
        losses.append(total_loss / len(y))
        if validation is not None:
            validation_predictions.append(_predict_neural(model, validation, scaler))
    return dict(model=model.eval(), scaler=scaler), validation_predictions, losses


def _tanimoto_distance(test: np.ndarray, train: np.ndarray) -> np.ndarray:
    intersection = test @ train.T
    union = test.sum(axis=1)[:, None] + train.sum(axis=1)[None, :] - intersection
    similarity = np.divide(
        intersection, union, out=np.ones_like(intersection), where=union > 0
    )
    return 1.0 - similarity


def _fit_classical(method, features, y, w, candidate, seed, threads):
    if method == "weighted_mean":
        return dict(kind="constant", value=float(np.average(y, weights=w)))
    if method == "native_continuous":
        return dict(kind="native")
    x = _matrix(features, method)
    w = w / w.mean()
    if method == "morgan_knn":
        return dict(kind="knn", x=x, y=y.copy(), weights=w.copy(), k=int(candidate))
    if method == "morgan_lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=int(candidate),
            learning_rate=0.05,
            num_leaves=7,
            max_depth=-1,
            min_child_samples=5,
            min_child_weight=1e-3,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=int(seed),
            n_jobs=threads,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        model.fit(x, y, sample_weight=w)
        return dict(kind="lightgbm", model=model, scaler=None)
    # Scaling is fit-only and weighted. Binary Morgan bits deliberately remain
    # unscaled (no historical global PCA/descriptor table).
    scaler = None
    if method != "morgan_ridge":
        scaler = StandardScaler().fit(x, sample_weight=w)
        x = scaler.transform(x)
    model = Ridge(alpha=float(candidate), fit_intercept=True, solver="svd")
    model.fit(x, y, sample_weight=w)
    return dict(kind="ridge", model=model, scaler=scaler)


def _predict_one(state, method, features):
    kind = state.get("kind", "neural")
    n = len(next(iter(features.values())))
    if not n:
        return np.empty(0, dtype=np.float64)
    if kind == "constant":
        return np.full(n, state["value"], dtype=np.float64)
    if kind == "native":
        return np.asarray(features["native_continuous"], dtype=np.float64).copy()
    if kind == "neural":
        return _predict_neural(state["model"], features, state["scaler"])
    x = _matrix(features, method)
    if kind == "knn":
        distance = _tanimoto_distance(x, state["x"])
        # Stable ties: immutable caller row order, not test labels.
        neighbors = np.argsort(distance, axis=1, kind="stable")[:, : state["k"]]
        weights = state["weights"][neighbors]
        return (weights * state["y"][neighbors]).sum(axis=1) / weights.sum(axis=1)
    if state["scaler"] is not None:
        x = state["scaler"].transform(x)
    if kind == "lightgbm":
        return state["model"].booster_.predict(x)
    return state["model"].predict(x)


def predict_model(model_state: dict, features: dict, *, batch_size: int = 1024) -> dict:
    """Replay a trusted fitted state without labels or the source checkpoint.

    CPU work is thread-limited and prediction intermediates are row-bounded.
    All seed weights, fitted scalers, intercepts and dual fusion offsets are in
    the artifact. No calibration correction is applied here. Use the same batch
    size for bitwise replay; different GEMM batch shapes may round differently.
    Returned arrays still occupy O(seeds * prediction rows) memory.
    """
    if (
        model_state.get("schema_version") != 1
        or model_state.get("method") not in METHODS
    ):
        raise ValueError("unsupported low-data model artifact")
    if not model_state.get("models") or len(model_state["models"]) != len(
        model_state.get("seeds", ())
    ):
        raise ValueError("model artifact must contain one model per seed")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    features, n = _validate_features(features)
    missing = _required_features(model_state["method"]) - features.keys()
    if missing:
        raise ValueError(f"missing required replay features: {sorted(missing)}")
    if (
        model_state["method"].startswith("morgan_")
        and not np.isin(features["morgan"], [0, 1]).all()
    ):
        raise ValueError("Morgan comparators require binary fingerprint bits")
    for key, width in model_state.get("feature_shapes", {}).items():
        if key not in features or list(features[key].shape[1:]) != width:
            raise ValueError(f"replay feature missing or width mismatch: {key}")
    threads = min(2, max(1, int(model_state.get("threads", 2))))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    predictions = np.empty((len(model_state["models"]), n), dtype=np.float64)
    try:
        with threadpool_limits(limits=threads):
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                batch = {k: v[start:stop] for k, v in features.items()}
                for i, state in enumerate(model_state["models"]):
                    predictions[i, start:stop] = _predict_one(
                        state, model_state["method"], batch
                    )
    finally:
        torch.set_num_threads(previous_threads)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("nonfinite replay predictions")
    return dict(
        pred=predictions.mean(axis=0),
        ensemble_std=predictions.std(axis=0, ddof=0),
        seed_predictions=predictions,
    )


def save_model(model_state: dict, path: Path | str) -> None:
    """Serialize trusted-local sklearn/torch state; contains no checkpoint path.

    Preserve package versions with the run manifest. Pickle is not an untrusted
    exchange format. All fitted scalers, per-seed weights, and base heads are saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model_state, protocol=pickle.HIGHEST_PROTOCOL))


def load_model(path: Path | str) -> dict:
    """Load a trusted-local artifact written by :func:`save_model`."""
    with Path(path).open("rb") as stream:
        state = pickle.load(stream)
    if state.get("schema_version") != 1 or state.get("method") not in METHODS:
        raise ValueError("unsupported low-data model artifact")
    return state


def _parameter_count(state: dict) -> tuple[int, int]:
    if "model" in state and isinstance(state["model"], nn.Module):
        parameters = list(state["model"].parameters())
        return sum(p.numel() for p in parameters if p.requires_grad), sum(
            p.numel() for p in parameters
        )
    if state.get("kind") == "ridge":
        count = state["model"].coef_.size + 1
    elif state.get("kind") == "constant":
        count = 1
    elif state.get("kind") == "lightgbm":
        count = sum(
            tree["num_leaves"]
            for tree in state["model"].booster_.dump_model()["tree_info"]
        )
    else:
        count = 0
    return int(count), int(count)


def fit_predict(
    method,
    train_features,
    y_train,
    assay_se_train,
    test_features,
    *,
    config,
    checkpoint_path,
    seeds=(42, 43, 44),
) -> dict:
    """Select strictly within fit rows, refit, predict and verify serialization.

    Required feature keys depend on method. Native continuous is already converted
    to pIC50-equivalent coordinates. ``groups`` is fit-only chemical grouping; when
    omitted, the caller guarantees identity-unique random-split rows. Group-starved
    selection raises InfeasibleInnerCV rather than silently switching policy.

    Neural LR AND epoch are selected by pooled inner-OOF assay-weighted MAE using
    the first declared seed. Final models independently refit every declared seed.
    Nonneural models repeat identical predictions for each declared seed (not an
    uncertainty estimate). ``n_parameters`` is trainable count per seed (trees:
    fitted leaf values; KNN: zero optimized parameters, stored rows in fit_audit).
    """
    started = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    y = np.asarray(y_train, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("fit labels must be nonempty finite 1D")
    se = np.asarray(assay_se_train, dtype=np.float64)
    if se.shape != y.shape:
        raise ValueError("assay errors and fit labels must align")
    w = _weights(se, float(config["se_floor"]))
    train, _ = _validate_features(train_features, expected=len(y))
    test, n_test = _validate_features(test_features)
    required = _required_features(method)
    for features in (train, test):
        if required - features.keys():
            raise ValueError(
                f"missing required features: {sorted(required - features.keys())}"
            )
    for key in required:
        if train[key].shape[1:] != test[key].shape[1:]:
            raise ValueError(f"train/test feature widths disagree: {key}")
    if method.startswith("morgan_"):
        for features in (train, test):
            if not np.isin(features["morgan"], [0, 1]).all():
                raise ValueError("Morgan comparators require binary fingerprint bits")
    threads = min(2, max(1, int(config.get("threads", 2))))
    epochs = int(config.get("epochs", 40))
    if epochs < 1 or int(config.get("batch_size", 64)) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if float(config.get("neural_huber_beta", 0.5)) != 0.5:
        raise ValueError("protocol requires smooth-L1 beta=0.5")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with threadpool_limits(limits=threads):
            result = _fit_predict_impl(
                method,
                train,
                y,
                w,
                test,
                config,
                checkpoint_path,
                seeds,
                threads,
                epochs,
            )
    finally:
        torch.set_num_threads(previous_threads)
    result["fit_audit"].update(
        n_fit=len(y),
        n_prediction_rows=n_test,
        threads=threads,
        test_labels_accessed=False,
        calibration_labels_accessed=False,
        weight_definition=(
            "inverse max(assay SE, floor); normalized to fit mean 1 for optimization"
        ),
        se_floor=float(config["se_floor"]),
        effective_n=float(w.sum() ** 2 / np.square(w).sum()),
    )
    result["fit_seconds"] = time.perf_counter() - started
    return result


def _fit_predict_impl(
    method, train, y, w, test, config, checkpoint_path, seeds, threads, epochs
):
    checkpoint_state = None
    if method in ("head_pretrained", "dual_head", "head_lora"):
        checkpoint_state = _checkpoint_heads(checkpoint_path)
    selection = dict(
        metric="pooled_inner_oof_assay_weighted_mae",
        selection_seed=seeds[0],
        candidates=[],
    )
    candidate = None
    selected_epochs = epochs
    if method not in ("weighted_mean", "native_continuous"):
        folds, fold_audit = _folds(train, len(y), int(config["inner_folds"]), seeds[0])
        selection.update(fold_audit)
        grid_key = (
            "neural_learning_rate"
            if method in NEURAL_METHODS
            else "lightgbm_n_estimators"
            if method == "morgan_lightgbm"
            else "knn_neighbors"
            if method == "morgan_knn"
            else "ridge_alpha"
        )
        grid = config["grids"][grid_key]
        if not grid or any(not np.isfinite(v) or v <= 0 for v in grid):
            raise ValueError(f"invalid protocol grid: {grid_key}")
        best_score = np.inf
        for value in grid:
            record = dict(parameter=grid_key, value=float(value))
            if method == "morgan_knn" and any(int(value) > len(a) for a, _ in folds):
                record.update(
                    status="infeasible", reason="k exceeds inner training row count"
                )
                selection["candidates"].append(record)
                continue
            oof = np.full((epochs if method in NEURAL_METHODS else 1, len(y)), np.nan)
            for a, b in folds:
                if method in NEURAL_METHODS:
                    _, predictions, _ = _train_neural(
                        method,
                        _slice(train, a),
                        y[a],
                        w[a],
                        config,
                        checkpoint_state,
                        seeds[0],
                        float(value),
                        epochs,
                        validation=_slice(train, b),
                    )
                    oof[:, b] = np.asarray(predictions)
                else:
                    model = _fit_classical(
                        method, _slice(train, a), y[a], w[a], value, seeds[0], threads
                    )
                    oof[0, b] = _predict_one(model, method, _slice(train, b))
            scores = [_weighted_mae(y, pred, w) for pred in oof]
            epoch_index = int(np.argmin(scores))
            record.update(status="ok", weighted_mae=scores[epoch_index])
            if method in NEURAL_METHODS:
                record.update(epoch_weighted_mae=scores, best_epoch=epoch_index + 1)
            selection["candidates"].append(record)
            # Ties honor protocol grid ordering, then earliest epoch.
            if scores[epoch_index] < best_score:
                best_score = scores[epoch_index]
                candidate = value
                selected_epochs = epoch_index + 1
        if candidate is None:
            raise InfeasibleInnerCV(
                "all protocol grid candidates infeasible within fit rows"
            )
        selection["selected"] = {grid_key: float(candidate)}
        if method in NEURAL_METHODS:
            selection["selected"]["epochs"] = selected_epochs
    else:
        selection.update(policy="no model selection", selected={})
    states = []
    history = []
    if method in NEURAL_METHODS:
        for seed in seeds:
            state, _, losses = _train_neural(
                method,
                train,
                y,
                w,
                config,
                checkpoint_state,
                seed,
                float(candidate),
                selected_epochs,
            )
            states.append(state)
            history.append(dict(seed=seed, epoch_training_loss=losses))
    else:
        state = _fit_classical(method, train, y, w, candidate, seeds[0], threads)
        states = [state] * len(seeds)
    artifact = dict(
        schema_version=1,
        method=method,
        seeds=seeds,
        models=states,
        threads=threads,
        selection=selection,
        feature_shapes={
            k: list(train[k].shape[1:]) for k in sorted(_required_features(method))
        },
    )
    prediction = predict_model(artifact, test)
    # In-memory roundtrip covers every call, even when caller declines disk saves.
    payload = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    reloaded = predict_model(pickle.loads(payload), test)
    if not np.array_equal(prediction["seed_predictions"], reloaded["seed_predictions"]):
        raise AssertionError("serialization replay changed predictions")
    count, total = _parameter_count(states[0])
    audit = dict(
        serialization_verified=True,
        serialization_format="trusted-local Python pickle",
        serialized_bytes=len(payload),
        serialized_sha256=hashlib.sha256(payload).hexdigest(),
        trainable_parameters_per_seed=count,
        total_parameters_per_seed=total,
        seed_count=len(seeds),
        seed_ids=list(seeds),
        identical_non_neural_seed_replicas=method not in NEURAL_METHODS,
        final_fit_history=history,
        stored_training_rows=len(y) if method == "morgan_knn" else 0,
        upstream_representation_frozen=True,
        target_label_training=method != "native_continuous",
        feature_shapes=artifact["feature_shapes"],
        fitted_preprocessing_scope="inner train / final fit only",
        uncertainty_semantics=(
            "population SD across raw seed predictions; not a predictive interval"
        ),
        output_semantics=(
            "native pIC50-equivalent; not cellular pEC50"
            if method == "native_continuous"
            else "target-trained cellular pEC50 readout"
        ),
        adaptation_scope=(
            "parameter-efficient final-readout MLP LoRA; NOT representation learning"
            if method == "head_lora"
            else "cached-feature readout"
        ),
        optimizer=(
            "AdamW, weight_decay=0, gradient_clip_norm=1"
            if method in NEURAL_METHODS
            else None
        ),
        objective=(
            "weighted smooth-L1 beta=0.5, ensemble/member weights=0.5/0.25/0.25"
            if method in NEURAL_METHODS
            else "assay-weighted fit; inner weighted MAE selection"
        ),
    )
    return dict(
        **prediction,
        n_parameters=count,
        peak_vram_bytes=0,  # All factory tensors and estimators are CPU-only.
        selection=selection,
        fit_audit=audit,
        model_state=artifact,
    )
