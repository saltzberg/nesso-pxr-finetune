"""Leakage-safe CPU follow-up experiments for the Nesso-1 PXR audit.

The functions in this module deliberately keep all model selection inside
chemical-family training partitions.  The historical lockbox is only scored
after a development-only configuration has been selected and refit.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from nesso_pxr.chemistry import (
    bemis_murcko_scaffold,
    morgan_fingerprints,
    scaffold_group_key,
)
from nesso_pxr.comparison import (
    clustered_bootstrap,
    inverse_se_weights,
    paired_difference_rows,
    regression_metrics,
)
from nesso_pxr.protocol import sha256_file
from nesso_pxr.splits import butina_cluster_ids
from nesso_pxr.transfer_ablation import (
    fit_full_ridge_after_group_selection,
    nested_ridge_probe,
)

EXPECTED_CPUS = frozenset(range(6, 12))
SEEDS = (42, 43, 44)
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_REPLICATES = 2000
RIDGE_ALPHAS = (0.0, 1.0, 10.0, 100.0, 1000.0)
MLP_CANDIDATES = (
    (1e-4, 0.0),
    (3e-4, 0.0),
    (1e-3, 0.0),
    (3e-4, 1e-4),
    (1e-3, 1e-4),
)
HEAD_LEARNING_RATES = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
HEAD_OBJECTIVES = (
    ("mse", None),
    ("mae", None),
    ("huber", 0.25),
    ("huber", 0.5),
    ("huber", 1.0),
    ("huber", 2.0),
)


def objective_label(name: str, delta: float | None) -> str:
    """Return a stable label for a regression objective."""

    return name if delta is None else f"huber_{delta:g}"


def candidate_id(learning_rate: float, weight_decay: float) -> str:
    """Return a stable neural-probe candidate identifier."""

    return f"lr{learning_rate:g}_wd{weight_decay:g}"


def assert_resource_contract(max_workers: int) -> dict[str, Any]:
    """Fail closed if a heavy run escapes the agreed CPU-only envelope."""

    if not 1 <= max_workers <= 6:
        raise ValueError("max_workers must be between one and six")
    affinity = set(os.sched_getaffinity(0))
    if not affinity or not affinity.issubset(EXPECTED_CPUS):
        raise RuntimeError(
            f"CPU affinity {sorted(affinity)} is outside physical CPUs 6-11"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must disable CUDA")
    required_one = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    wrong = {
        name: os.environ.get(name)
        for name in required_one
        if os.environ.get(name) != "1"
    }
    if wrong:
        raise RuntimeError(f"thread environment is not pinned to one: {wrong}")
    if os.environ.get("MALLOC_ARENA_MAX") != "2":
        raise RuntimeError("MALLOC_ARENA_MAX must equal 2")
    return {
        "affinity": sorted(affinity),
        "max_workers": max_workers,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **{name.lower(): os.environ.get(name) for name in required_one},
        "malloc_arena_max": os.environ.get("MALLOC_ARENA_MAX"),
        "nice": os.nice(0),
    }


def package_versions() -> dict[str, str]:
    """Return versions used by the scientific run."""

    import lightgbm
    import matplotlib
    import rdkit
    import safetensors
    import scipy
    import sklearn
    import torch

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "rdkit": rdkit.__version__,
        "lightgbm": lightgbm.__version__,
        "safetensors": safetensors.__version__,
        "matplotlib": matplotlib.__version__,
    }


def portable_repo_path(path: Path) -> str:
    """Record a repository-relative path without leaking workstation layout."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def start_run_metadata(
    output_dir: Path,
    *,
    stage: str,
    max_workers: int,
    inputs: dict[str, Path],
    analysis_spec: Path,
) -> tuple[float, dict[str, Any]]:
    """Write immutable-at-start provenance before loading outcome labels."""

    output_dir.mkdir(parents=True, exist_ok=True)
    contract = assert_resource_contract(max_workers)
    metadata = {
        "stage": stage,
        "status": "running",
        "started_unix": time.time(),
        "command": ["python", *sys.argv],
        "working_directory": ".",
        "resource_contract": contract,
        "versions": package_versions(),
        "analysis_spec": portable_repo_path(analysis_spec),
        "analysis_spec_sha256": sha256_file(analysis_spec),
        "inputs": {
            name: {"path": portable_repo_path(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return time.perf_counter(), metadata


def finish_run_metadata(
    output_dir: Path,
    started: float,
    metadata: dict[str, Any],
    *,
    status: str = "complete",
) -> None:
    """Finalize runtime and peak-memory provenance."""

    metadata = dict(metadata)
    metadata.update(
        {
            "status": status,
            "wall_seconds": time.perf_counter() - started,
            "self_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "child_peak_rss_kib": resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_maxrss,
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def load_feature_bundle(
    manifest_path: Path, features_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the published row-aligned Nesso feature bundle."""

    from safetensors.torch import load_file

    manifest = (
        pd.read_csv(manifest_path)
        .sort_values("feature_row", kind="stable")
        .reset_index(drop=True)
    )
    tensors = load_file(str(features_path), device="cpu")
    if manifest["feature_row"].tolist() != list(range(len(manifest))):
        raise ValueError("manifest feature rows must be contiguous")
    if len(manifest) != len(tensors["member1_affinity_repr"]):
        raise ValueError("manifest/features row mismatch")
    return manifest, tensors


def split_feature_bundle(
    manifest: pd.DataFrame, tensors: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return development/lockbox frames and declared representation variants."""

    development = manifest.loc[manifest["modeling_role"].eq("development")].copy()
    lockbox = manifest.loc[manifest["modeling_role"].eq("lockbox")].copy()
    development = development.sort_values("feature_row", kind="stable").reset_index(
        drop=True
    )
    lockbox = lockbox.sort_values("feature_row", kind="stable").reset_index(drop=True)
    development["fold"] = development["fold"].astype(int)
    dev_positions = development["feature_row"].to_numpy(dtype=int)
    lock_positions = lockbox["feature_row"].to_numpy(dtype=int)

    def variants(positions: np.ndarray) -> dict[str, np.ndarray]:
        member1 = tensors["member1_affinity_repr"][positions].float().numpy()
        member2 = tensors["member2_affinity_repr"][positions].float().numpy()
        return {
            "member1_384d": member1,
            "member2_384d": member2,
            "mean_384d": ((member1 + member2) / 2.0).astype(np.float32),
            "concat_768d": np.concatenate([member1, member2], axis=1).astype(
                np.float32
            ),
        }

    return development, lockbox, variants(dev_positions), variants(lock_positions)


def _safe_tail_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    active = observed > 6.0
    return {
        **regression_metrics(observed, predicted),
        "active_threshold": 6.0,
        "active_n": int(active.sum()),
        "active_mae": (
            float(np.mean(np.abs(predicted[active] - observed[active])))
            if active.any()
            else float("nan")
        ),
        "predicted_above_6": int(np.sum(predicted > 6.0)),
        "active_predicted_above_6": int(np.sum(predicted[active] > 6.0)),
        "prediction_minimum": float(np.min(predicted)),
        "prediction_maximum": float(np.max(predicted)),
    }


def _stable_best(frame: pd.DataFrame, *, candidate_column: str) -> pd.Series:
    """Apply the frozen MAE/Spearman/grid-order selection rule."""

    required = {"mae", "spearman", "candidate_order", candidate_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing selection columns: {sorted(missing)}")
    return frame.sort_values(
        ["mae", "spearman", "candidate_order"],
        ascending=[True, False, True],
        kind="stable",
    ).iloc[0]


_WORKER_DATA: dict[str, Any] = {}


def set_worker_data(data: dict[str, Any]) -> None:
    """Set fork-inherited read-only arrays before creating a process pool."""

    global _WORKER_DATA
    _WORKER_DATA = data


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _parallel_map(function: Any, tasks: Sequence[Any], workers: int) -> list[Any]:
    if not tasks:
        return []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("fork"),
        initializer=_worker_init,
    ) as executor:
        return list(executor.map(function, tasks, chunksize=1))


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _partition_indices(
    mode: str, *, outer_fold: int | None, inner_fold: int | None
) -> tuple[np.ndarray, np.ndarray, bool]:
    folds = np.asarray(_WORKER_DATA["folds"], dtype=int)
    if mode == "inner":
        if outer_fold is None or inner_fold is None:
            raise ValueError("inner task requires outer and inner folds")
        train = np.flatnonzero((folds != outer_fold) & (folds != inner_fold))
        evaluation = np.flatnonzero((folds != outer_fold) & (folds == inner_fold))
        return train, evaluation, True
    if mode == "outer":
        if outer_fold is None:
            raise ValueError("outer task requires an outer fold")
        return (
            np.flatnonzero(folds != outer_fold),
            np.flatnonzero(folds == outer_fold),
            False,
        )
    if mode == "full_cv":
        if inner_fold is None:
            raise ValueError("full CV task requires an inner fold")
        return (
            np.flatnonzero(folds != inner_fold),
            np.flatnonzero(folds == inner_fold),
            True,
        )
    if mode == "lockbox":
        return (
            np.arange(len(folds), dtype=int),
            np.arange(len(_WORKER_DATA["y_lock"]), dtype=int),
            False,
        )
    raise ValueError(f"unknown task mode: {mode}")


@dataclass(frozen=True)
class MLPTask:
    representation: str
    hidden_layers: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    seed: int
    mode: Literal["inner", "outer", "full_cv", "lockbox"]
    outer_fold: int | None = None
    inner_fold: int | None = None
    fixed_epochs: int | None = None


def _fit_single_mlp_task(task: MLPTask) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional
    from torch.utils.data import DataLoader, TensorDataset

    _set_seed(task.seed)
    train_indices, evaluation_indices, use_evaluation_labels = _partition_indices(
        task.mode, outer_fold=task.outer_fold, inner_fold=task.inner_fold
    )
    x_dev = _WORKER_DATA["dev_representations"][task.representation]
    x_eval_source = (
        _WORKER_DATA["lock_representations"][task.representation]
        if task.mode == "lockbox"
        else x_dev
    )
    y_dev = np.asarray(_WORKER_DATA["y_dev"], dtype=np.float32)
    x_train = np.asarray(x_dev[train_indices], dtype=np.float32)
    x_evaluation = np.asarray(x_eval_source[evaluation_indices], dtype=np.float32)
    mean = np.mean(x_train, axis=0, keepdims=True)
    scale = np.std(x_train, axis=0, keepdims=True)
    scale[scale < 1e-7] = 1.0
    x_train = ((x_train - mean) / scale).astype(np.float32)
    x_evaluation = ((x_evaluation - mean) / scale).astype(np.float32)

    dimensions = (x_train.shape[1], *task.hidden_layers, 1)
    layers: list[nn.Module] = []
    for position, (input_dim, output_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        layers.append(nn.Linear(input_dim, output_dim))
        if position < len(dimensions) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=task.learning_rate, weight_decay=task.weight_decay
    )
    dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_dev[train_indices])
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(task.seed),
        num_workers=0,
    )
    evaluation_tensor = torch.from_numpy(x_evaluation)
    y_evaluation = (
        y_dev[evaluation_indices]
        if task.mode != "lockbox"
        else np.asarray(_WORKER_DATA["y_lock"], dtype=np.float32)[evaluation_indices]
    )
    max_epochs = task.fixed_epochs if task.fixed_epochs is not None else 40
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_mae = math.inf
    waiting = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x).squeeze(-1)
            loss = functional.huber_loss(prediction, batch_y, delta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if task.fixed_epochs is not None:
            continue
        model.eval()
        with torch.inference_mode():
            predicted = model(evaluation_tensor).squeeze(-1).numpy()
        validation_mae = float(np.mean(np.abs(predicted - y_evaluation)))
        if validation_mae < best_mae - 0.002:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += 1
            if waiting >= 6:
                break
    if task.fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("MLP task never produced an early-stopping checkpoint")
        model.load_state_dict(best_state, strict=True)
    else:
        best_epoch = task.fixed_epochs
    model.eval()
    with torch.inference_mode():
        prediction = model(evaluation_tensor).squeeze(-1).numpy().astype(float)
    result = {
        **asdict(task),
        "evaluation_indices": evaluation_indices,
        "prediction": prediction,
        "best_epoch": int(best_epoch),
    }
    if use_evaluation_labels:
        result.update(regression_metrics(y_evaluation, prediction))
    return result


def _capacity_metric_rows(
    predictions: pd.DataFrame, *, dataset: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (representation, model_class), subset in predictions.groupby(
        ["representation", "model_class"], sort=False
    ):
        rows.append(
            {
                "dataset": dataset,
                "representation": representation,
                "model_class": model_class,
                **_safe_tail_metrics(
                    subset["pEC50"].to_numpy(dtype=float),
                    subset["predicted_pEC50"].to_numpy(dtype=float),
                ),
            }
        )
    return rows


def _capacity_seed_rows(
    predictions: pd.DataFrame, *, dataset: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_columns = {
        42: "predicted_seed_42",
        43: "predicted_seed_43",
        44: "predicted_seed_44",
    }
    for (representation, model_class), subset in predictions.groupby(
        ["representation", "model_class"], sort=False
    ):
        for seed, column in seed_columns.items():
            if subset[column].isna().all():
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "representation": representation,
                    "model_class": model_class,
                    "seed": seed,
                    **regression_metrics(subset["pEC50"], subset[column]),
                }
            )
    return rows


def _bootstrap_capacity(
    development: pd.DataFrame,
    lockbox: pd.DataFrame,
    dev_predictions: pd.DataFrame,
    lock_predictions: pd.DataFrame,
    historical_dev_path: Path,
    historical_lock_path: Path,
    output_dir: Path,
) -> None:
    primary_models = {
        "ridge": "mean384_ridge",
        "one_hidden_mlp": "mean384_one_hidden_mlp",
        "two_hidden_mlp": "mean384_two_hidden_mlp",
    }
    dev_wide = development[["feature_row", "pEC50", "cluster_component"]].copy()
    for model_class, column in primary_models.items():
        subset = dev_predictions.loc[
            dev_predictions["representation"].eq("mean_384d")
            & dev_predictions["model_class"].eq(model_class),
            ["feature_row", "predicted_pEC50"],
        ].rename(columns={"predicted_pEC50": column})
        dev_wide = dev_wide.merge(
            subset, on="feature_row", how="left", validate="one_to_one"
        )
    historical_dev = pd.read_csv(historical_dev_path)[
        ["feature_row", "nesso_pEC50"]
    ].rename(columns={"nesso_pEC50": "published_ft_head"})
    dev_wide = dev_wide.merge(
        historical_dev, on="feature_row", how="left", validate="one_to_one"
    )
    model_columns = {
        "mean384_ridge": "mean384_ridge",
        "mean384_one_hidden_mlp": "mean384_one_hidden_mlp",
        "mean384_two_hidden_mlp": "mean384_two_hidden_mlp",
        "published_ft_head": "published_ft_head",
    }
    intervals, draws = clustered_bootstrap(
        dev_wide,
        model_columns,
        group_column="cluster_component",
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    intervals.insert(0, "dataset", "nested_development")
    point = {
        model: regression_metrics(dev_wide["pEC50"], dev_wide[column])
        for model, column in model_columns.items()
    }
    comparisons = (
        ("mean384_one_hidden_mlp", "mean384_ridge"),
        ("mean384_two_hidden_mlp", "mean384_ridge"),
        ("mean384_two_hidden_mlp", "mean384_one_hidden_mlp"),
        ("mean384_ridge", "published_ft_head"),
        ("mean384_one_hidden_mlp", "published_ft_head"),
        ("mean384_two_hidden_mlp", "published_ft_head"),
    )
    differences = paired_difference_rows(
        draws, point, comparisons, metrics=("mae", "spearman")
    )
    differences.insert(0, "dataset", "nested_development")

    lock_wide = lockbox[["original_id", "pEC50", "audit_scaffold_group"]].copy()
    for model_class, column in primary_models.items():
        subset = lock_predictions.loc[
            lock_predictions["representation"].eq("mean_384d")
            & lock_predictions["model_class"].eq(model_class),
            ["original_id", "predicted_pEC50"],
        ].rename(columns={"predicted_pEC50": column})
        lock_wide = lock_wide.merge(
            subset, on="original_id", how="left", validate="one_to_one"
        )
    historical_lock = pd.read_csv(historical_lock_path)[
        ["original_id", "predicted_pEC50_ensemble"]
    ].rename(columns={"predicted_pEC50_ensemble": "published_ft_head"})
    lock_wide = lock_wide.merge(
        historical_lock, on="original_id", how="left", validate="one_to_one"
    )
    lock_intervals, lock_draws = clustered_bootstrap(
        lock_wide,
        model_columns,
        group_column="audit_scaffold_group",
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    lock_intervals.insert(0, "dataset", "retrospective_lockbox_790")
    lock_point = {
        model: regression_metrics(lock_wide["pEC50"], lock_wide[column])
        for model, column in model_columns.items()
    }
    lock_differences = paired_difference_rows(
        lock_draws, lock_point, comparisons, metrics=("mae", "spearman")
    )
    lock_differences.insert(0, "dataset", "retrospective_lockbox_790")
    pd.concat([intervals, lock_intervals], ignore_index=True).to_csv(
        output_dir / "capacity_bootstrap_intervals.csv", index=False
    )
    pd.concat([differences, lock_differences], ignore_index=True).to_csv(
        output_dir / "capacity_paired_bootstrap_differences.csv", index=False
    )


def run_capacity_experiment(
    *,
    manifest_path: Path,
    features_path: Path,
    historical_dev_path: Path,
    historical_lock_path: Path,
    analysis_spec: Path,
    output_dir: Path,
    max_workers: int,
) -> None:
    """Run the fully nested frozen-representation capacity ladder."""

    started, run_metadata = start_run_metadata(
        output_dir,
        stage="experiment_a_capacity",
        max_workers=max_workers,
        inputs={
            "manifest": manifest_path,
            "features": features_path,
            "historical_development_predictions": historical_dev_path,
            "historical_lockbox_predictions": historical_lock_path,
        },
        analysis_spec=analysis_spec,
    )
    manifest, tensors = load_feature_bundle(manifest_path, features_path)
    development, lockbox, dev_representations, lock_representations = (
        split_feature_bundle(manifest, tensors)
    )
    y_dev = development["pEC50"].to_numpy(dtype=float)
    y_lock = lockbox["pEC50"].to_numpy(dtype=float)
    folds = development["fold"].to_numpy(dtype=int)
    set_worker_data(
        {
            "dev_representations": dev_representations,
            "lock_representations": lock_representations,
            "y_dev": y_dev,
            "y_lock": y_lock,
            "folds": folds,
        }
    )

    dev_prediction_frames: list[pd.DataFrame] = []
    lock_prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    full_selection_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []

    for representation, x_dev in dev_representations.items():
        ridge = nested_ridge_probe(x_dev, y_dev, folds, RIDGE_ALPHAS)
        ridge_prediction = ridge.predictions["predicted"].to_numpy(dtype=float)
        dev_prediction_frames.append(
            pd.DataFrame(
                {
                    "feature_row": development["feature_row"],
                    "original_id": development["original_id"],
                    "fold": folds,
                    "pEC50": y_dev,
                    "representation": representation,
                    "model_class": "ridge",
                    "predicted_seed_42": ridge_prediction,
                    "predicted_seed_43": ridge_prediction,
                    "predicted_seed_44": ridge_prediction,
                    "predicted_pEC50": ridge_prediction,
                    "prediction_seed_sd": 0.0,
                }
            )
        )
        for row in ridge.selections.to_dict("records"):
            selection_rows.append(
                {
                    "representation": representation,
                    "model_class": "ridge",
                    **row,
                    "selected_candidate": f"alpha{row['selected_alpha']:g}",
                    "fixed_refit_epochs": np.nan,
                }
            )
        lock_prediction, selected_alpha, full_screen = (
            fit_full_ridge_after_group_selection(
                x_dev,
                y_dev,
                folds,
                lock_representations[representation],
                RIDGE_ALPHAS,
            )
        )
        for order, row in enumerate(full_screen.to_dict("records")):
            screen_rows.append(
                {
                    "scope": "full_development",
                    "outer_fold": np.nan,
                    "representation": representation,
                    "model_class": "ridge",
                    "candidate": f"alpha{row['alpha']:g}",
                    "candidate_order": order,
                    **row,
                }
            )
        full_selection_rows.append(
            {
                "representation": representation,
                "model_class": "ridge",
                "selected_candidate": f"alpha{selected_alpha:g}",
                "selected_alpha": selected_alpha,
                "fixed_refit_epochs": np.nan,
            }
        )
        lock_prediction_frames.append(
            pd.DataFrame(
                {
                    "feature_row": lockbox["feature_row"],
                    "original_id": lockbox["original_id"],
                    "fold": np.nan,
                    "pEC50": y_lock,
                    "representation": representation,
                    "model_class": "ridge",
                    "predicted_seed_42": lock_prediction,
                    "predicted_seed_43": lock_prediction,
                    "predicted_seed_44": lock_prediction,
                    "predicted_pEC50": lock_prediction,
                    "prediction_seed_sd": 0.0,
                }
            )
        )

    architectures = {
        "one_hidden_mlp": (384,),
        "two_hidden_mlp": (384, 384),
    }
    inner_tasks: list[MLPTask] = []
    for representation in dev_representations:
        for hidden_layers in architectures.values():
            for outer_fold in sorted(np.unique(folds)):
                for learning_rate, weight_decay in MLP_CANDIDATES:
                    for inner_fold in sorted(
                        value for value in np.unique(folds) if value != outer_fold
                    ):
                        inner_tasks.append(
                            MLPTask(
                                representation=representation,
                                hidden_layers=hidden_layers,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                seed=42,
                                mode="inner",
                                outer_fold=int(outer_fold),
                                inner_fold=int(inner_fold),
                            )
                        )
    inner_results = _parallel_map(_fit_single_mlp_task, inner_tasks, max_workers)
    grouped_inner: dict[
        tuple[str, tuple[int, ...], int, float, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for result in inner_results:
        grouped_inner[
            (
                result["representation"],
                tuple(result["hidden_layers"]),
                int(result["outer_fold"]),
                float(result["learning_rate"]),
                float(result["weight_decay"]),
            )
        ].append(result)

    selected_outer: dict[tuple[str, str, int], dict[str, Any]] = {}
    for representation in dev_representations:
        for model_class, hidden_layers in architectures.items():
            for outer_fold in sorted(np.unique(folds)):
                candidates: list[dict[str, Any]] = []
                outer_training = folds != outer_fold
                for order, (learning_rate, weight_decay) in enumerate(MLP_CANDIDATES):
                    key = (
                        representation,
                        hidden_layers,
                        int(outer_fold),
                        learning_rate,
                        weight_decay,
                    )
                    results = grouped_inner[key]
                    inner_prediction = np.full(len(y_dev), np.nan, dtype=float)
                    epochs: list[int] = []
                    for result in results:
                        inner_prediction[result["evaluation_indices"]] = result[
                            "prediction"
                        ]
                        epochs.append(int(result["best_epoch"]))
                    metrics = regression_metrics(
                        y_dev[outer_training], inner_prediction[outer_training]
                    )
                    row = {
                        "scope": "outer_training",
                        "outer_fold": int(outer_fold),
                        "representation": representation,
                        "model_class": model_class,
                        "candidate": candidate_id(learning_rate, weight_decay),
                        "candidate_order": order,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "best_epochs": ";".join(str(value) for value in epochs),
                        **metrics,
                    }
                    screen_rows.append(row)
                    candidates.append(row)
                selected = _stable_best(
                    pd.DataFrame(candidates), candidate_column="candidate"
                ).to_dict()
                selected["fixed_refit_epochs"] = max(
                    1,
                    int(
                        np.rint(
                            np.median(
                                [
                                    int(value)
                                    for value in selected["best_epochs"].split(";")
                                ]
                            )
                        )
                    ),
                )
                selected_outer[(representation, model_class, int(outer_fold))] = (
                    selected
                )
                selection_rows.append(
                    {
                        "representation": representation,
                        "model_class": model_class,
                        "outer_fold": int(outer_fold),
                        "selected_candidate": selected["candidate"],
                        "learning_rate": selected["learning_rate"],
                        "weight_decay": selected["weight_decay"],
                        "fixed_refit_epochs": selected["fixed_refit_epochs"],
                        "inner_mae": selected["mae"],
                        "inner_spearman": selected["spearman"],
                    }
                )

    outer_tasks: list[MLPTask] = []
    for (representation, model_class, outer_fold), selected in selected_outer.items():
        for seed in SEEDS:
            outer_tasks.append(
                MLPTask(
                    representation=representation,
                    hidden_layers=architectures[model_class],
                    learning_rate=float(selected["learning_rate"]),
                    weight_decay=float(selected["weight_decay"]),
                    seed=seed,
                    mode="outer",
                    outer_fold=outer_fold,
                    fixed_epochs=int(selected["fixed_refit_epochs"]),
                )
            )
    outer_results = _parallel_map(_fit_single_mlp_task, outer_tasks, max_workers)
    grouped_outer: dict[tuple[str, str], dict[int, np.ndarray]] = defaultdict(
        lambda: {seed: np.full(len(y_dev), np.nan, dtype=float) for seed in SEEDS}
    )
    hidden_to_model = {value: key for key, value in architectures.items()}
    for result in outer_results:
        model_class = hidden_to_model[tuple(result["hidden_layers"])]
        grouped_outer[(result["representation"], model_class)][int(result["seed"])][
            result["evaluation_indices"]
        ] = result["prediction"]
    for (representation, model_class), seed_predictions in grouped_outer.items():
        stack = np.stack([seed_predictions[seed] for seed in SEEDS])
        if not np.isfinite(stack).all():
            raise AssertionError("nested neural capacity predictions are incomplete")
        dev_prediction_frames.append(
            pd.DataFrame(
                {
                    "feature_row": development["feature_row"],
                    "original_id": development["original_id"],
                    "fold": folds,
                    "pEC50": y_dev,
                    "representation": representation,
                    "model_class": model_class,
                    "predicted_seed_42": seed_predictions[42],
                    "predicted_seed_43": seed_predictions[43],
                    "predicted_seed_44": seed_predictions[44],
                    "predicted_pEC50": np.mean(stack, axis=0),
                    "prediction_seed_sd": np.std(stack, axis=0, ddof=1),
                }
            )
        )

    full_cv_tasks: list[MLPTask] = []
    for representation in dev_representations:
        for hidden_layers in architectures.values():
            for learning_rate, weight_decay in MLP_CANDIDATES:
                for inner_fold in sorted(np.unique(folds)):
                    full_cv_tasks.append(
                        MLPTask(
                            representation=representation,
                            hidden_layers=hidden_layers,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            seed=42,
                            mode="full_cv",
                            inner_fold=int(inner_fold),
                        )
                    )
    full_results = _parallel_map(_fit_single_mlp_task, full_cv_tasks, max_workers)
    grouped_full: dict[
        tuple[str, tuple[int, ...], float, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for result in full_results:
        grouped_full[
            (
                result["representation"],
                tuple(result["hidden_layers"]),
                float(result["learning_rate"]),
                float(result["weight_decay"]),
            )
        ].append(result)

    selected_full: dict[tuple[str, str], dict[str, Any]] = {}
    for representation in dev_representations:
        for model_class, hidden_layers in architectures.items():
            candidates = []
            for order, (learning_rate, weight_decay) in enumerate(MLP_CANDIDATES):
                results = grouped_full[
                    (representation, hidden_layers, learning_rate, weight_decay)
                ]
                prediction = np.full(len(y_dev), np.nan, dtype=float)
                epochs: list[int] = []
                for result in results:
                    prediction[result["evaluation_indices"]] = result["prediction"]
                    epochs.append(int(result["best_epoch"]))
                metrics = regression_metrics(y_dev, prediction)
                row = {
                    "scope": "full_development",
                    "outer_fold": np.nan,
                    "representation": representation,
                    "model_class": model_class,
                    "candidate": candidate_id(learning_rate, weight_decay),
                    "candidate_order": order,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "best_epochs": ";".join(str(value) for value in epochs),
                    **metrics,
                }
                screen_rows.append(row)
                candidates.append(row)
            selected = _stable_best(
                pd.DataFrame(candidates), candidate_column="candidate"
            ).to_dict()
            selected["fixed_refit_epochs"] = max(
                1,
                int(
                    np.rint(
                        np.median(
                            [int(value) for value in selected["best_epochs"].split(";")]
                        )
                    )
                ),
            )
            selected_full[(representation, model_class)] = selected
            full_selection_rows.append(
                {
                    "representation": representation,
                    "model_class": model_class,
                    "selected_candidate": selected["candidate"],
                    "learning_rate": selected["learning_rate"],
                    "weight_decay": selected["weight_decay"],
                    "fixed_refit_epochs": selected["fixed_refit_epochs"],
                    "development_cv_mae": selected["mae"],
                    "development_cv_spearman": selected["spearman"],
                }
            )

    lock_tasks: list[MLPTask] = []
    for (representation, model_class), selected in selected_full.items():
        for seed in SEEDS:
            lock_tasks.append(
                MLPTask(
                    representation=representation,
                    hidden_layers=architectures[model_class],
                    learning_rate=float(selected["learning_rate"]),
                    weight_decay=float(selected["weight_decay"]),
                    seed=seed,
                    mode="lockbox",
                    fixed_epochs=int(selected["fixed_refit_epochs"]),
                )
            )
    lock_results = _parallel_map(_fit_single_mlp_task, lock_tasks, max_workers)
    grouped_lock: dict[tuple[str, str], dict[int, np.ndarray]] = defaultdict(dict)
    for result in lock_results:
        model_class = hidden_to_model[tuple(result["hidden_layers"])]
        grouped_lock[(result["representation"], model_class)][int(result["seed"])] = (
            result["prediction"]
        )
    for (representation, model_class), seed_predictions in grouped_lock.items():
        stack = np.stack([seed_predictions[seed] for seed in SEEDS])
        lock_prediction_frames.append(
            pd.DataFrame(
                {
                    "feature_row": lockbox["feature_row"],
                    "original_id": lockbox["original_id"],
                    "fold": np.nan,
                    "pEC50": y_lock,
                    "representation": representation,
                    "model_class": model_class,
                    "predicted_seed_42": seed_predictions[42],
                    "predicted_seed_43": seed_predictions[43],
                    "predicted_seed_44": seed_predictions[44],
                    "predicted_pEC50": np.mean(stack, axis=0),
                    "prediction_seed_sd": np.std(stack, axis=0, ddof=1),
                }
            )
        )

    dev_predictions = pd.concat(dev_prediction_frames, ignore_index=True)
    lock_predictions = pd.concat(lock_prediction_frames, ignore_index=True)
    dev_predictions.to_csv(
        output_dir / "capacity_development_predictions.csv", index=False
    )
    lock_predictions.to_csv(
        output_dir / "capacity_lockbox_predictions.csv", index=False
    )
    pd.DataFrame(screen_rows).to_csv(
        output_dir / "capacity_inner_screens.csv", index=False
    )
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "capacity_outer_selections.csv", index=False
    )
    pd.DataFrame(full_selection_rows).to_csv(
        output_dir / "capacity_full_development_selections.csv", index=False
    )

    metric_rows = _capacity_metric_rows(
        dev_predictions, dataset="nested_development"
    ) + _capacity_metric_rows(lock_predictions, dataset="retrospective_lockbox_790")
    historical_dev = pd.read_csv(historical_dev_path)
    historical_lock = pd.read_csv(historical_lock_path)
    metric_rows.extend(
        [
            {
                "dataset": "nested_development",
                "representation": "published_twin_member_pair",
                "model_class": "published_pretrained_twin_head_finetune",
                **_safe_tail_metrics(
                    historical_dev["pEC50"], historical_dev["nesso_pEC50"]
                ),
            },
            {
                "dataset": "retrospective_lockbox_790",
                "representation": "published_twin_member_pair",
                "model_class": "published_pretrained_twin_head_finetune",
                **_safe_tail_metrics(
                    historical_lock["pEC50"],
                    historical_lock["predicted_pEC50_ensemble"],
                ),
            },
        ]
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "capacity_metrics.csv", index=False)
    seed_rows = _capacity_seed_rows(
        dev_predictions, dataset="nested_development"
    ) + _capacity_seed_rows(lock_predictions, dataset="retrospective_lockbox_790")
    pd.DataFrame(seed_rows).to_csv(
        output_dir / "capacity_seed_variability.csv", index=False
    )
    _bootstrap_capacity(
        development,
        lockbox,
        dev_predictions,
        lock_predictions,
        historical_dev_path,
        historical_lock_path,
        output_dir,
    )
    primary = metrics.loc[metrics["representation"].eq("mean_384d")]
    summary = {
        "status": "complete",
        "primary_representation": "mean_384d",
        "metrics": primary.to_dict("records"),
        "interpretation_guard": (
            "The random single-input probes differ from the published twin-head "
            "architecture and training initialization; their comparison isolates "
            "readout capacity, not a controlled initialization effect."
        ),
        "lockbox_status": "retrospective_validation",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    finish_run_metadata(output_dir, started, run_metadata)


def _canonical_smiles(molecule: Any) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def standardize_smiles_record(smiles: str) -> dict[str, Any]:
    """Apply the predeclared RDKit chemical-form sensitivity schemes."""

    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("rdkit_parse_failure")
    canonical = _canonical_smiles(molecule)
    cleaned_molecule = rdMolStandardize.Cleanup(Chem.Mol(molecule))
    cleaned = _canonical_smiles(cleaned_molecule)
    fragment_molecule = rdMolStandardize.FragmentParent(
        Chem.Mol(cleaned_molecule), skipStandardize=True
    )
    fragment = _canonical_smiles(fragment_molecule)
    charge_molecule = rdMolStandardize.ChargeParent(Chem.Mol(molecule))
    charge_parent = _canonical_smiles(charge_molecule)
    tautomer_molecule = rdMolStandardize.TautomerParent(Chem.Mol(molecule))
    tautomer_parent = _canonical_smiles(tautomer_molecule)
    inchi_key = Chem.MolToInchiKey(fragment_molecule)
    if not inchi_key:
        raise ValueError("inchi_key_failure")
    return {
        "rdkit_canonical_smiles": canonical,
        "cleaned_smiles": cleaned,
        "fragment_parent_smiles": fragment,
        "charge_parent_smiles": charge_parent,
        "tautomer_parent_smiles": tautomer_parent,
        "connectivity_inchikey": inchi_key.split("-")[0],
        "canonical_formal_charge": int(Chem.GetFormalCharge(molecule)),
        "fragment_parent_formal_charge": int(Chem.GetFormalCharge(fragment_molecule)),
        "charge_parent_formal_charge": int(Chem.GetFormalCharge(charge_molecule)),
        "raw_fragment_count": len(Chem.GetMolFrags(molecule)),
        "fragment_parent_fragment_count": len(Chem.GetMolFrags(fragment_molecule)),
    }


def standardize_manifest(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize every row and retain reason-coded failures."""

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        try:
            standardized = standardize_smiles_record(str(row.raw_smiles))
        except Exception as error:
            failures.append(
                {
                    "feature_row": int(row.feature_row),
                    "original_id": str(row.original_id),
                    "reason": f"{type(error).__name__}:{error}",
                }
            )
            standardized = {
                "rdkit_canonical_smiles": None,
                "cleaned_smiles": None,
                "fragment_parent_smiles": None,
                "charge_parent_smiles": None,
                "tautomer_parent_smiles": None,
                "connectivity_inchikey": None,
                "canonical_formal_charge": np.nan,
                "fragment_parent_formal_charge": np.nan,
                "charge_parent_formal_charge": np.nan,
                "raw_fragment_count": np.nan,
                "fragment_parent_fragment_count": np.nan,
            }
        records.append(
            {
                "feature_row": int(row.feature_row),
                "original_id": str(row.original_id),
                "raw_smiles": str(row.raw_smiles),
                **standardized,
            }
        )
    failure_frame = pd.DataFrame(
        failures,
        columns=["feature_row", "original_id", "reason"],
    )
    return pd.DataFrame(records), failure_frame


def identity_collision_audit(
    manifest: pd.DataFrame, standardized: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize identity collapse, label discordance, and fold/role crossings."""

    annotated = manifest.merge(
        standardized,
        on=["feature_row", "original_id", "raw_smiles"],
        how="left",
        validate="one_to_one",
    )
    schemes = {
        "raw_smiles": "raw_smiles",
        "rdkit_canonical": "rdkit_canonical_smiles",
        "cleaned": "cleaned_smiles",
        "fragment_parent_primary": "fragment_parent_smiles",
        "charge_parent": "charge_parent_smiles",
        "tautomer_parent": "tautomer_parent_smiles",
        "connectivity": "connectivity_inchikey",
    }
    summary: dict[str, Any] = {}
    collision_rows: list[dict[str, Any]] = []
    for scheme, column in schemes.items():
        valid = annotated.loc[annotated[column].notna()].copy()
        groups = valid.groupby(column, sort=True, dropna=False)
        sizes = groups.size()
        collided = set(sizes.loc[sizes > 1].index)
        dev = valid.loc[valid["modeling_role"].eq("development")]
        dev_folds = dev.groupby(column)["fold"].nunique()
        role_counts = valid.groupby(column)["modeling_role"].nunique()
        label_range = groups["pEC50"].agg(
            lambda values: float(values.max() - values.min())
        )
        summary[scheme] = {
            "column": column,
            "valid_rows": int(len(valid)),
            "unique_identities": int(valid[column].nunique()),
            "collision_identities": int((sizes > 1).sum()),
            "rows_in_collisions": int(sizes.loc[sizes > 1].sum()),
            "maximum_collision_size": int(sizes.max()) if len(sizes) else 0,
            "maximum_within_identity_pEC50_range": (
                float(label_range.max()) if len(label_range) else float("nan")
            ),
            "collision_identities_with_pEC50_range_ge_0_5": int(
                ((sizes > 1) & (label_range >= 0.5)).sum()
            ),
            "collision_identities_with_pEC50_range_ge_1_0": int(
                ((sizes > 1) & (label_range >= 1.0)).sum()
            ),
            "development_identities_crossing_historical_folds": int(
                (dev_folds > 1).sum()
            ),
            "identities_crossing_development_lockbox_roles": int(
                (role_counts > 1).sum()
            ),
        }
        for identity in sorted(collided):
            subset = valid.loc[valid[column].eq(identity)]
            development_folds = sorted(
                int(value)
                for value in subset.loc[
                    subset["modeling_role"].eq("development"), "fold"
                ].dropna()
            )
            collision_rows.append(
                {
                    "scheme": scheme,
                    "identity": identity,
                    "n": len(subset),
                    "pEC50_minimum": float(subset["pEC50"].min()),
                    "pEC50_maximum": float(subset["pEC50"].max()),
                    "pEC50_range": float(subset["pEC50"].max() - subset["pEC50"].min()),
                    "modeling_roles": ";".join(
                        sorted(subset["modeling_role"].astype(str).unique())
                    ),
                    "historical_development_folds": ";".join(
                        str(value) for value in development_folds
                    ),
                    "original_ids": ";".join(sorted(subset["original_id"].astype(str))),
                }
            )
    summary["change_counts"] = {
        "raw_multifragment_rows": int(
            (standardized["raw_fragment_count"].fillna(0) > 1).sum()
        ),
        "cleanup_changed_canonical_smiles": int(
            standardized["cleaned_smiles"]
            .ne(standardized["rdkit_canonical_smiles"])
            .sum()
        ),
        "fragment_parent_changed_cleaned_smiles": int(
            standardized["fragment_parent_smiles"]
            .ne(standardized["cleaned_smiles"])
            .sum()
        ),
        "charge_parent_changed_fragment_parent": int(
            standardized["charge_parent_smiles"]
            .ne(standardized["fragment_parent_smiles"])
            .sum()
        ),
        "tautomer_parent_changed_fragment_parent": int(
            standardized["tautomer_parent_smiles"]
            .ne(standardized["fragment_parent_smiles"])
            .sum()
        ),
        "fragment_parent_charge_changed_from_canonical": int(
            standardized["fragment_parent_formal_charge"]
            .ne(standardized["canonical_formal_charge"])
            .sum()
        ),
        "raw_canonical_nonzero_charge": int(
            standardized["canonical_formal_charge"].fillna(0).ne(0).sum()
        ),
        "fragment_parent_nonzero_charge": int(
            standardized["fragment_parent_formal_charge"].fillna(0).ne(0).sum()
        ),
        "charge_parent_nonzero_charge": int(
            standardized["charge_parent_formal_charge"].fillna(0).ne(0).sum()
        ),
        "nesso_prepared_nonzero_charge": int(
            pd.to_numeric(manifest["formal_charge"], errors="coerce")
            .fillna(0)
            .ne(0)
            .sum()
        ),
        "prepared_smiles_changed_from_identity_canonical": int(
            manifest["prepared_smiles"].ne(manifest["identity_canonical_smiles"]).sum()
        ),
    }
    return summary, pd.DataFrame(collision_rows)


class _StandardizedDisjointSet:
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


def assign_standardized_folds(
    development: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 20260806,
    similarity_cutoff: float = 0.60,
) -> pd.DataFrame:
    """Rebuild families from the conservative primary standardized structures."""

    required = {
        "feature_row",
        "original_id",
        "fragment_parent_smiles",
        "connectivity_inchikey",
    }
    missing = required.difference(development.columns)
    if missing:
        raise KeyError(f"missing standardized split columns: {sorted(missing)}")
    if development[list(required)].isna().any().any():
        raise ValueError("standardized development records must be complete")
    result = development[list(required)].copy().reset_index(drop=True)
    result["standardized_scaffold"] = result["fragment_parent_smiles"].map(
        bemis_murcko_scaffold
    )
    result["standardized_scaffold_group"] = [
        scaffold_group_key(smiles, scaffold)
        for smiles, scaffold in zip(
            result["fragment_parent_smiles"],
            result["standardized_scaffold"],
            strict=True,
        )
    ]
    fingerprints = morgan_fingerprints(
        result["fragment_parent_smiles"].tolist(), radius=2, n_bits=2048
    )
    result["standardized_butina_cluster"] = butina_cluster_ids(
        fingerprints, similarity_cutoff=similarity_cutoff
    )

    disjoint = _StandardizedDisjointSet(len(result))
    for column in (
        "fragment_parent_smiles",
        "connectivity_inchikey",
        "standardized_butina_cluster",
        "standardized_scaffold_group",
    ):
        for indices in result.groupby(column, sort=True).indices.values():
            indices = list(indices)
            for index in indices[1:]:
                disjoint.union(int(indices[0]), int(index))

    roots = [disjoint.find(index) for index in range(len(result))]
    root_members: dict[int, list[int]] = defaultdict(list)
    for index, root in enumerate(roots):
        root_members[root].append(index)
    rng = np.random.default_rng(seed)
    randomized_ties = {root: float(rng.random()) for root in root_members}
    ordered_roots = sorted(
        root_members,
        key=lambda root: (-len(root_members[root]), randomized_ties[root], root),
    )
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    root_to_fold: dict[int, int] = {}
    for root in ordered_roots:
        smallest = np.flatnonzero(fold_sizes == fold_sizes.min())
        fold = int(rng.choice(smallest))
        root_to_fold[root] = fold
        fold_sizes[fold] += len(root_members[root])
    component_order = {
        root: component for component, root in enumerate(sorted(root_members))
    }
    result["standardized_cluster_component"] = [component_order[root] for root in roots]
    result["standardized_fold"] = [root_to_fold[root] for root in roots]

    for column in (
        "fragment_parent_smiles",
        "connectivity_inchikey",
        "standardized_scaffold_group",
        "standardized_cluster_component",
    ):
        if result.groupby(column)["standardized_fold"].nunique().max() != 1:
            raise AssertionError(f"{column} crosses standardized folds")
    return result


def align_fold_labels_for_reporting(
    historical_folds: np.ndarray, standardized_folds: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """Hungarian-align arbitrary fold labels only for assignment-change reporting."""

    from scipy.optimize import linear_sum_assignment

    historical = np.asarray(historical_folds, dtype=int)
    standardized = np.asarray(standardized_folds, dtype=int)
    labels = sorted(np.unique(standardized))
    reference = sorted(np.unique(historical))
    contingency = np.zeros((len(labels), len(reference)), dtype=int)
    for row, standardized_label in enumerate(labels):
        for column, historical_label in enumerate(reference):
            contingency[row, column] = int(
                np.sum(
                    (standardized == standardized_label)
                    & (historical == historical_label)
                )
            )
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {
        labels[row]: reference[column]
        for row, column in zip(rows, columns, strict=True)
    }
    aligned = np.asarray([mapping[int(value)] for value in standardized], dtype=int)
    return aligned, mapping


def nearest_training_similarity(
    evaluation_fingerprints: Sequence[Any],
    training_fingerprints: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact nearest-neighbor Tanimoto and training positions."""

    from rdkit import DataStructs

    similarities = np.empty(len(evaluation_fingerprints), dtype=float)
    positions = np.empty(len(evaluation_fingerprints), dtype=int)
    for index, fingerprint in enumerate(evaluation_fingerprints):
        values = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, training_fingerprints),
            dtype=float,
        )
        nearest = int(np.argmax(values))
        similarities[index] = values[nearest]
        positions[index] = nearest
    return similarities, positions


def audit_standardized_fold_neighbors(
    development: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Directly audit every standardized validation-to-training nearest neighbor."""

    fingerprints = morgan_fingerprints(
        development["fragment_parent_smiles"].tolist(), radius=2, n_bits=2048
    )
    folds = development["standardized_fold"].to_numpy(dtype=int)
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        validation_indices = np.flatnonzero(folds == fold)
        training_indices = np.flatnonzero(folds != fold)
        similarities, nearest_local = nearest_training_similarity(
            [fingerprints[index] for index in validation_indices],
            [fingerprints[index] for index in training_indices],
        )
        nearest_indices = training_indices[nearest_local]
        subset = pd.DataFrame(
            {
                "feature_row": development.iloc[validation_indices][
                    "feature_row"
                ].to_numpy(),
                "original_id": development.iloc[validation_indices][
                    "original_id"
                ].to_numpy(),
                "standardized_fold": fold,
                "nearest_training_similarity": similarities,
                "nearest_training_feature_row": development.iloc[nearest_indices][
                    "feature_row"
                ].to_numpy(),
                "nearest_training_original_id": development.iloc[nearest_indices][
                    "original_id"
                ].to_numpy(),
            }
        )
        rows.append(subset)
        summaries.append(
            {
                "standardized_fold": int(fold),
                "n": len(similarities),
                "minimum": float(np.min(similarities)),
                "q05": float(np.quantile(similarities, 0.05)),
                "median": float(np.median(similarities)),
                "q95": float(np.quantile(similarities, 0.95)),
                "maximum": float(np.max(similarities)),
                "count_ge_0_5": int(np.sum(similarities >= 0.5)),
                "count_ge_0_6": int(np.sum(similarities >= 0.6)),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def standardized_lockbox_neighbors(
    development: pd.DataFrame, lockbox: pd.DataFrame
) -> pd.DataFrame:
    """Calculate lockbox-to-development similarity under the primary scheme."""

    development_fingerprints = morgan_fingerprints(
        development["fragment_parent_smiles"].tolist(), radius=2, n_bits=2048
    )
    lockbox_fingerprints = morgan_fingerprints(
        lockbox["fragment_parent_smiles"].tolist(), radius=2, n_bits=2048
    )
    similarities, nearest = nearest_training_similarity(
        lockbox_fingerprints, development_fingerprints
    )
    result = lockbox[["feature_row", "original_id"]].copy()
    result["nearest_development_similarity"] = similarities
    result["nearest_development_feature_row"] = development.iloc[nearest][
        "feature_row"
    ].to_numpy()
    result["nearest_development_original_id"] = development.iloc[nearest][
        "original_id"
    ].to_numpy()
    return result


def _lgbm_parameters(
    threads: int, *, seed: int, n_estimators: int = 5000
) -> dict[str, Any]:
    return {
        "objective": "regression",
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 5.0,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "n_jobs": threads,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
    }


def _two_d_weights(frame: pd.DataFrame) -> np.ndarray:
    return inverse_se_weights(
        frame["pEC50_standard_error"], frame["curation_sample_weight"]
    )


def run_standardized_lightgbm(
    development: pd.DataFrame,
    lockbox: pd.DataFrame,
    *,
    feature_path: Path,
    threads: int,
    top_k: int = 1000,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Run the historical nested 2D recipe under standardized chemical folds."""

    import lightgbm as lgb

    feature_table = pd.read_parquet(feature_path)
    if feature_table["Molecule Name"].duplicated().any():
        raise ValueError("2D feature identifiers must be unique")
    feature_names = [
        column
        for column in feature_table.columns
        if column not in {"Molecule Name", "SMILES"}
    ]
    feature_table = feature_table.set_index("Molecule Name")
    order = pd.concat(
        [development[["original_id"]], lockbox[["original_id"]]], ignore_index=True
    )
    aligned = feature_table.reindex(order["original_id"])
    if aligned["SMILES"].isna().any():
        raise ValueError("2D feature table does not cover the standardized cohort")
    x = aligned[feature_names].to_numpy(dtype=np.float32)
    x[~np.isfinite(x)] = 0.0
    dev_count = len(development)
    x_dev = x[:dev_count]
    x_lock = x[dev_count:]
    del aligned, feature_table, x
    gc.collect()

    y = development["pEC50"].to_numpy(dtype=float)
    folds = development["standardized_fold"].to_numpy(dtype=int)
    oof = np.full(len(development), np.nan, dtype=float)
    outer_rows: list[dict[str, Any]] = []
    selected_feature_frames: list[pd.DataFrame] = []

    for outer_fold in sorted(np.unique(folds)):
        outer_train = np.flatnonzero(folds != outer_fold)
        outer_validation = np.flatnonzero(folds == outer_fold)
        inner_folds = sorted(np.unique(folds[outer_train]))
        gain = np.zeros((len(inner_folds), len(feature_names)), dtype=np.float64)
        first_rounds: list[int] = []
        for position, inner_fold in enumerate(inner_folds):
            train_indices = np.flatnonzero(
                (folds != outer_fold) & (folds != inner_fold)
            )
            validation_indices = np.flatnonzero(
                (folds != outer_fold) & (folds == inner_fold)
            )
            model = lgb.LGBMRegressor(
                **_lgbm_parameters(
                    threads,
                    seed=20260809 + int(outer_fold) * 100 + int(inner_fold),
                )
            )
            model.fit(
                x_dev[train_indices],
                y[train_indices],
                sample_weight=_two_d_weights(development.iloc[train_indices]),
                eval_set=[(x_dev[validation_indices], y[validation_indices])],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            gain[position] = model.booster_.feature_importance(importance_type="gain")
            first_rounds.append(int(model.best_iteration_))
        mean_gain = gain.mean(axis=0)
        selected_indices = np.argsort(-mean_gain, kind="stable")[:top_k]
        selected_feature_frames.append(
            pd.DataFrame(
                {
                    "scope": "outer_training",
                    "outer_fold": int(outer_fold),
                    "rank": np.arange(1, len(selected_indices) + 1),
                    "feature_index": selected_indices,
                    "feature": [feature_names[index] for index in selected_indices],
                    "mean_inner_gain": mean_gain[selected_indices],
                }
            )
        )
        second_rounds: list[int] = []
        for inner_fold in inner_folds:
            train_indices = np.flatnonzero(
                (folds != outer_fold) & (folds != inner_fold)
            )
            validation_indices = np.flatnonzero(
                (folds != outer_fold) & (folds == inner_fold)
            )
            model = lgb.LGBMRegressor(
                **_lgbm_parameters(
                    threads,
                    seed=20261809 + int(outer_fold) * 100 + int(inner_fold),
                )
            )
            model.fit(
                x_dev[train_indices][:, selected_indices],
                y[train_indices],
                sample_weight=_two_d_weights(development.iloc[train_indices]),
                eval_set=[
                    (
                        x_dev[validation_indices][:, selected_indices],
                        y[validation_indices],
                    )
                ],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            second_rounds.append(int(model.best_iteration_))
        fixed_rounds = max(1, int(np.rint(np.median(second_rounds))))
        final_model = lgb.LGBMRegressor(
            **_lgbm_parameters(
                threads,
                seed=20262809 + int(outer_fold),
                n_estimators=fixed_rounds,
            )
        )
        final_model.fit(
            x_dev[outer_train][:, selected_indices],
            y[outer_train],
            sample_weight=_two_d_weights(development.iloc[outer_train]),
        )
        oof[outer_validation] = final_model.predict(
            x_dev[outer_validation][:, selected_indices]
        )
        outer_rows.append(
            {
                "outer_fold": int(outer_fold),
                "outer_train_rows": len(outer_train),
                "outer_validation_rows": len(outer_validation),
                "input_feature_count": len(feature_names),
                "selected_feature_count": len(selected_indices),
                "first_pass_best_iterations": ";".join(
                    str(value) for value in first_rounds
                ),
                "second_pass_best_iterations": ";".join(
                    str(value) for value in second_rounds
                ),
                "fixed_refit_iterations": fixed_rounds,
                **regression_metrics(y[outer_validation], oof[outer_validation]),
            }
        )

    if not np.isfinite(oof).all():
        raise AssertionError("standardized nested LightGBM predictions are incomplete")

    full_gain = np.zeros((len(np.unique(folds)), len(feature_names)), dtype=float)
    full_first_rounds: list[int] = []
    for position, held_out_fold in enumerate(sorted(np.unique(folds))):
        training = folds != held_out_fold
        validation = folds == held_out_fold
        model = lgb.LGBMRegressor(
            **_lgbm_parameters(threads, seed=20263809 + int(held_out_fold))
        )
        model.fit(
            x_dev[training],
            y[training],
            sample_weight=_two_d_weights(development.loc[training]),
            eval_set=[(x_dev[validation], y[validation])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        full_gain[position] = model.booster_.feature_importance(importance_type="gain")
        full_first_rounds.append(int(model.best_iteration_))
    full_mean_gain = full_gain.mean(axis=0)
    full_selected = np.argsort(-full_mean_gain, kind="stable")[:top_k]
    selected_feature_frames.append(
        pd.DataFrame(
            {
                "scope": "full_development",
                "outer_fold": np.nan,
                "rank": np.arange(1, len(full_selected) + 1),
                "feature_index": full_selected,
                "feature": [feature_names[index] for index in full_selected],
                "mean_inner_gain": full_mean_gain[full_selected],
            }
        )
    )
    full_second_rounds: list[int] = []
    for held_out_fold in sorted(np.unique(folds)):
        training = folds != held_out_fold
        validation = folds == held_out_fold
        model = lgb.LGBMRegressor(
            **_lgbm_parameters(threads, seed=20264809 + int(held_out_fold))
        )
        model.fit(
            x_dev[training][:, full_selected],
            y[training],
            sample_weight=_two_d_weights(development.loc[training]),
            eval_set=[(x_dev[validation][:, full_selected], y[validation])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        full_second_rounds.append(int(model.best_iteration_))
    full_fixed_rounds = max(1, int(np.rint(np.median(full_second_rounds))))
    full_model = lgb.LGBMRegressor(
        **_lgbm_parameters(threads, seed=20265809, n_estimators=full_fixed_rounds)
    )
    full_model.fit(
        x_dev[:, full_selected],
        y,
        sample_weight=_two_d_weights(development),
    )
    lock_prediction = full_model.predict(x_lock[:, full_selected])
    method = {
        "model": "LightGBM regression",
        "descriptor_input": (
            "published reduced 2D features generated from original prepared states"
        ),
        "standardization_role": (
            "primary RDKit fragment parents define family folds and similarity only"
        ),
        "outer_split": "standardized Butina/Murcko/connectivity families",
        "inner_split": "remaining standardized outer family folds",
        "feature_selection": "mean inner LightGBM gain, top 1000",
        "stopping": "median inner best iteration then fixed-round refit",
        "sample_weight": "inverse pEC50 standard error times curation weight",
        "input_feature_count": len(feature_names),
        "top_k": top_k,
        "threads": threads,
        "full_development_first_pass_iterations": full_first_rounds,
        "full_development_second_pass_iterations": full_second_rounds,
        "full_development_fixed_refit_iterations": full_fixed_rounds,
        "parameters": _lgbm_parameters(threads, seed=20260809),
    }
    return (
        oof,
        np.asarray(lock_prediction, dtype=float),
        pd.DataFrame(outer_rows),
        pd.concat(selected_feature_frames, ignore_index=True),
        pd.DataFrame(
            {
                "feature": [feature_names[index] for index in full_selected],
                "gain": full_mean_gain[full_selected],
            }
        ),
        method,
    )


def _standardized_similarity_bins(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        values,
        bins=[-np.inf, 0.3, 0.4, 0.5, 0.6, np.inf],
        labels=["<=0.3", "(0.3,0.4]", "(0.4,0.5]", "(0.5,0.6]", ">0.6"],
    )


def run_standardization_experiment(
    *,
    manifest_path: Path,
    features_path: Path,
    reduced_2d_features_path: Path,
    historical_ridge_path: Path,
    historical_lgbm_path: Path,
    analysis_spec: Path,
    output_dir: Path,
    max_workers: int,
) -> None:
    """Run chemical-form, standardized-family, ridge, and LightGBM stress tests."""

    started, run_metadata = start_run_metadata(
        output_dir,
        stage="experiment_b_standardization",
        max_workers=max_workers,
        inputs={
            "manifest": manifest_path,
            "nesso_features": features_path,
            "reduced_2d_features": reduced_2d_features_path,
            "historical_ridge_predictions": historical_ridge_path,
            "historical_lgbm_predictions": historical_lgbm_path,
        },
        analysis_spec=analysis_spec,
    )
    manifest, tensors = load_feature_bundle(manifest_path, features_path)
    standardized, failures = standardize_manifest(manifest)
    standardized.to_csv(output_dir / "standardized_structures.csv", index=False)
    failures.to_csv(output_dir / "standardization_failures.csv", index=False)
    identity_summary, collision_groups = identity_collision_audit(
        manifest, standardized
    )
    collision_groups.to_csv(output_dir / "identity_collision_groups.csv", index=False)
    if len(failures):
        (output_dir / "standardization_audit.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failures": len(failures),
                    "identity_schemes": identity_summary,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        finish_run_metadata(
            output_dir, started, run_metadata, status="failed_standardization"
        )
        raise RuntimeError("standardization failures prevent a complete split rebuild")

    annotated = manifest.merge(
        standardized,
        on=["feature_row", "original_id", "raw_smiles"],
        how="left",
        validate="one_to_one",
    )
    development = annotated.loc[annotated["modeling_role"].eq("development")].copy()
    lockbox = annotated.loc[annotated["modeling_role"].eq("lockbox")].copy()
    development = development.sort_values("feature_row", kind="stable").reset_index(
        drop=True
    )
    lockbox = lockbox.sort_values("feature_row", kind="stable").reset_index(drop=True)
    assignments = assign_standardized_folds(development)
    assignment_columns = [
        "feature_row",
        "standardized_scaffold",
        "standardized_scaffold_group",
        "standardized_butina_cluster",
        "standardized_cluster_component",
        "standardized_fold",
    ]
    development = development.merge(
        assignments[assignment_columns],
        on="feature_row",
        how="left",
        validate="one_to_one",
    )
    aligned_folds, fold_mapping = align_fold_labels_for_reporting(
        development["fold"].to_numpy(dtype=int),
        development["standardized_fold"].to_numpy(dtype=int),
    )
    development["aligned_standardized_fold"] = aligned_folds
    development["assignment_changed_after_alignment"] = development[
        "aligned_standardized_fold"
    ] != development["fold"].to_numpy(dtype=int)
    development.to_csv(output_dir / "standardized_development_folds.csv", index=False)

    neighbor_rows, neighbor_summary = audit_standardized_fold_neighbors(development)
    neighbor_rows.to_csv(
        output_dir / "standardized_fold_nearest_neighbors.csv", index=False
    )
    neighbor_summary.to_csv(
        output_dir / "standardized_fold_neighbor_summary.csv", index=False
    )
    lock_neighbors = standardized_lockbox_neighbors(development, lockbox)
    lock_neighbors.to_csv(
        output_dir / "standardized_lockbox_nearest_neighbors.csv", index=False
    )
    lockbox = lockbox.merge(
        lock_neighbors[
            [
                "feature_row",
                "nearest_development_similarity",
                "nearest_development_feature_row",
                "nearest_development_original_id",
            ]
        ],
        on="feature_row",
        how="left",
        validate="one_to_one",
    )
    lockbox["standardized_scaffold"] = lockbox["fragment_parent_smiles"].map(
        bemis_murcko_scaffold
    )
    lockbox["standardized_scaffold_group"] = [
        scaffold_group_key(smiles, scaffold)
        for smiles, scaffold in zip(
            lockbox["fragment_parent_smiles"],
            lockbox["standardized_scaffold"],
            strict=True,
        )
    ]

    family_sizes = (
        development.groupby("standardized_cluster_component", sort=True)
        .agg(
            n=("feature_row", "size"),
            standardized_fold=("standardized_fold", "first"),
            pEC50_minimum=("pEC50", "min"),
            pEC50_median=("pEC50", "median"),
            pEC50_maximum=("pEC50", "max"),
        )
        .reset_index()
    )
    family_sizes.to_csv(output_dir / "standardized_family_sizes.csv", index=False)
    fold_labels = (
        development.groupby("standardized_fold", sort=True)
        .agg(
            n=("feature_row", "size"),
            families=("standardized_cluster_component", "nunique"),
            pEC50_minimum=("pEC50", "min"),
            pEC50_q25=("pEC50", lambda values: float(np.quantile(values, 0.25))),
            pEC50_median=("pEC50", "median"),
            pEC50_q75=("pEC50", lambda values: float(np.quantile(values, 0.75))),
            pEC50_maximum=("pEC50", "max"),
        )
        .reset_index()
    )
    fold_labels.to_csv(
        output_dir / "standardized_fold_label_distributions.csv", index=False
    )

    _, _, dev_representations, lock_representations = split_feature_bundle(
        manifest, tensors
    )
    if development["original_id"].tolist() != (
        manifest.loc[manifest["modeling_role"].eq("development")]
        .sort_values("feature_row", kind="stable")["original_id"]
        .tolist()
    ):
        raise AssertionError(
            "standardized development order differs from Nesso features"
        )
    y_dev = development["pEC50"].to_numpy(dtype=float)
    standardized_folds = development["standardized_fold"].to_numpy(dtype=int)
    ridge = nested_ridge_probe(
        dev_representations["mean_384d"],
        y_dev,
        standardized_folds,
        RIDGE_ALPHAS,
    )
    ridge_oof = ridge.predictions["predicted"].to_numpy(dtype=float)
    ridge_lock, ridge_alpha, ridge_full_screen = fit_full_ridge_after_group_selection(
        dev_representations["mean_384d"],
        y_dev,
        standardized_folds,
        lock_representations["mean_384d"],
        RIDGE_ALPHAS,
    )
    ridge.selections.to_csv(
        output_dir / "standardized_ridge_outer_selections.csv", index=False
    )
    ridge_full_screen.assign(selected_alpha=ridge_alpha).to_csv(
        output_dir / "standardized_ridge_full_development_screen.csv", index=False
    )

    (
        lgbm_oof,
        lgbm_lock,
        lgbm_outer,
        lgbm_features,
        lgbm_full_features,
        lgbm_method,
    ) = run_standardized_lightgbm(
        development,
        lockbox,
        feature_path=reduced_2d_features_path,
        threads=max_workers,
    )
    lgbm_outer.to_csv(output_dir / "standardized_lgbm_outer_summary.csv", index=False)
    lgbm_features.to_csv(
        output_dir / "standardized_lgbm_selected_features.csv", index=False
    )
    lgbm_full_features.to_csv(
        output_dir / "standardized_lgbm_full_development_features.csv", index=False
    )
    (output_dir / "standardized_lgbm_method.json").write_text(
        json.dumps(lgbm_method, indent=2, sort_keys=True) + "\n"
    )

    dev_predictions = development[
        [
            "feature_row",
            "original_id",
            "pEC50",
            "fold",
            "standardized_fold",
            "standardized_cluster_component",
        ]
    ].copy()
    dev_predictions["standardized_mean384_ridge"] = ridge_oof
    dev_predictions["standardized_split_2d_lightgbm"] = lgbm_oof
    dev_predictions.to_csv(
        output_dir / "standardized_split_development_predictions.csv", index=False
    )
    lock_predictions = lockbox[
        [
            "feature_row",
            "original_id",
            "pEC50",
            "nearest_development_similarity",
            "standardized_scaffold_group",
        ]
    ].copy()
    lock_predictions["standardized_mean384_ridge"] = ridge_lock
    lock_predictions["standardized_split_2d_lightgbm"] = lgbm_lock
    lock_predictions.to_csv(
        output_dir / "standardized_split_lockbox_predictions.csv", index=False
    )

    metric_rows: list[dict[str, Any]] = []
    for dataset, frame in (
        ("standardized_nested_development", dev_predictions),
        ("retrospective_standardized_lockbox_790", lock_predictions),
    ):
        for model, column in {
            "mean384_ridge": "standardized_mean384_ridge",
            "2d_lightgbm": "standardized_split_2d_lightgbm",
        }.items():
            metric_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    **_safe_tail_metrics(frame["pEC50"], frame[column]),
                }
            )
    historical_ridge = pd.read_csv(historical_ridge_path)
    historical_ridge = historical_ridge.loc[
        historical_ridge["model"].eq("ridge_mean_384d")
        & historical_ridge["dataset"].eq("nested_development")
    ]
    historical_lgbm = pd.read_csv(historical_lgbm_path)
    metric_rows.extend(
        [
            {
                "dataset": "historical_nested_development",
                "model": "mean384_ridge",
                **_safe_tail_metrics(
                    historical_ridge["pEC50"],
                    historical_ridge["predicted_pEC50"],
                ),
            },
            {
                "dataset": "historical_nested_development",
                "model": "2d_lightgbm",
                **_safe_tail_metrics(
                    historical_lgbm["pEC50"],
                    historical_lgbm["two_d_pEC50"],
                ),
            },
        ]
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "standardized_split_metrics.csv", index=False)

    lock_predictions["similarity_bin"] = _standardized_similarity_bins(
        lock_predictions["nearest_development_similarity"]
    )
    stratified_rows: list[dict[str, Any]] = []
    for similarity_bin, subset in lock_predictions.groupby(
        "similarity_bin", observed=True, sort=True
    ):
        for model, column in {
            "mean384_ridge": "standardized_mean384_ridge",
            "2d_lightgbm": "standardized_split_2d_lightgbm",
        }.items():
            stratified_rows.append(
                {
                    "dataset": "retrospective_standardized_lockbox_790",
                    "similarity_bin": str(similarity_bin),
                    "model": model,
                    **regression_metrics(subset["pEC50"], subset[column]),
                }
            )
    pd.DataFrame(stratified_rows).to_csv(
        output_dir / "standardized_lockbox_similarity_metrics.csv", index=False
    )

    model_columns = {
        "mean384_ridge": "standardized_mean384_ridge",
        "2d_lightgbm": "standardized_split_2d_lightgbm",
    }
    dev_intervals, dev_draws = clustered_bootstrap(
        dev_predictions,
        model_columns,
        group_column="standardized_cluster_component",
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    dev_intervals.insert(0, "dataset", "standardized_nested_development")
    dev_points = {
        model: regression_metrics(dev_predictions["pEC50"], dev_predictions[column])
        for model, column in model_columns.items()
    }
    dev_differences = paired_difference_rows(
        dev_draws,
        dev_points,
        [("2d_lightgbm", "mean384_ridge")],
        metrics=("mae", "spearman"),
    )
    dev_differences.insert(0, "dataset", "standardized_nested_development")
    lock_intervals, lock_draws = clustered_bootstrap(
        lock_predictions,
        model_columns,
        group_column="standardized_scaffold_group",
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    lock_intervals.insert(0, "dataset", "retrospective_standardized_lockbox_790")
    lock_points = {
        model: regression_metrics(lock_predictions["pEC50"], lock_predictions[column])
        for model, column in model_columns.items()
    }
    lock_differences = paired_difference_rows(
        lock_draws,
        lock_points,
        [("2d_lightgbm", "mean384_ridge")],
        metrics=("mae", "spearman"),
    )
    lock_differences.insert(0, "dataset", "retrospective_standardized_lockbox_790")
    pd.concat([dev_intervals, lock_intervals], ignore_index=True).to_csv(
        output_dir / "standardized_split_bootstrap_intervals.csv", index=False
    )
    pd.concat([dev_differences, lock_differences], ignore_index=True).to_csv(
        output_dir / "standardized_split_paired_differences.csv", index=False
    )

    split_audit = {
        "status": "pass",
        "primary_standardization": (
            "RDKit Cleanup then FragmentParent(skipStandardize=True); charge retained"
        ),
        "standardization_failures": len(failures),
        "identity_schemes": identity_summary,
        "historical_alignment_mapping": {
            str(key): int(value) for key, value in fold_mapping.items()
        },
        "rows_changing_fold_after_optimal_label_alignment": int(
            development["assignment_changed_after_alignment"].sum()
        ),
        "fraction_changing_fold_after_optimal_label_alignment": float(
            development["assignment_changed_after_alignment"].mean()
        ),
        "standardized_families": int(
            development["standardized_cluster_component"].nunique()
        ),
        "largest_standardized_family": int(family_sizes["n"].max()),
        "validation_neighbors_ge_0_6": int(
            (neighbor_rows["nearest_training_similarity"] >= 0.6).sum()
        ),
        "lockbox_neighbors_ge_0_6": int(
            (lockbox["nearest_development_similarity"] >= 0.6).sum()
        ),
        "representation_inputs": {
            "nesso": (
                "cached representations from the actual original prepared Nesso inputs"
            ),
            "lightgbm": (
                "published reduced descriptors from original prepared structures"
            ),
            "standardized_chemistry": (
                "used for family folds, identity audit, and similarity only"
            ),
        },
        "comparison_guard": (
            "Metrics under historical and standardized folds are not ordinary "
            "per-row paired deltas because each row is predicted by a differently "
            "composed training set."
        ),
        "lockbox_status": "retrospective_validation",
    }
    (output_dir / "standardization_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n"
    )
    finish_run_metadata(output_dir, started, run_metadata)


def load_pretrained_twin_state(checkpoint_path: Path) -> dict[str, Any]:
    """Extract only the two released regression heads into in-memory state."""

    from safetensors.torch import load_file

    checkpoint = load_file(str(checkpoint_path), device="cpu")
    prefixes = {
        "member1": "affinity_module.affinity_heads.to_affinity_pred_value.",
        "member2": "affinity_module2.affinity_heads.to_affinity_pred_value.",
    }
    state: dict[str, Any] = {}
    for member, prefix in prefixes.items():
        selected = {
            f"{member}.{key.removeprefix(prefix)}": value.detach().clone()
            for key, value in checkpoint.items()
            if key.startswith(prefix)
        }
        state.update(selected)
    expected_suffixes = {
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    }
    for member in prefixes:
        observed = {
            key.removeprefix(f"{member}.")
            for key in state
            if key.startswith(f"{member}.")
        }
        if observed != expected_suffixes:
            raise ValueError(f"incomplete pretrained state for {member}: {observed}")
    return state


def _elementwise_objective(
    prediction: Any,
    target: Any,
    *,
    objective: str,
    delta: float | None,
) -> Any:
    from torch.nn import functional

    if objective == "mse":
        return functional.mse_loss(prediction, target, reduction="none")
    if objective == "mae":
        return functional.l1_loss(prediction, target, reduction="none")
    if objective == "huber":
        if delta is None or delta <= 0:
            raise ValueError("Huber objective requires positive delta")
        return functional.huber_loss(prediction, target, delta=delta, reduction="none")
    raise ValueError(f"unknown objective: {objective}")


def combined_head_objective(
    prediction1: Any,
    prediction2: Any,
    ensemble: Any,
    target: Any,
    *,
    objective: str,
    delta: float | None,
) -> Any:
    """Apply the historical 50/25/25 twin-head loss contract."""

    losses = (
        0.50
        * _elementwise_objective(ensemble, target, objective=objective, delta=delta)
        + 0.25
        * _elementwise_objective(prediction1, target, objective=objective, delta=delta)
        + 0.25
        * _elementwise_objective(prediction2, target, objective=objective, delta=delta)
    )
    return losses.mean()


@dataclass(frozen=True)
class HeadTask:
    objective: str
    delta: float | None
    learning_rate: float
    seed: int
    mode: Literal["inner", "outer", "full_cv", "lockbox"]
    outer_fold: int | None = None
    inner_fold: int | None = None
    fixed_epochs: int | None = None


def _fit_head_task(task: HeadTask) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from nesso_pxr.train_cached import make_twin_heads

    _set_seed(task.seed)
    train_indices, evaluation_indices, use_evaluation_labels = _partition_indices(
        task.mode, outer_fold=task.outer_fold, inner_fold=task.inner_fold
    )
    member1_dev = np.asarray(_WORKER_DATA["member1_dev"], dtype=np.float32)
    member2_dev = np.asarray(_WORKER_DATA["member2_dev"], dtype=np.float32)
    if task.mode == "lockbox":
        member1_evaluation = np.asarray(
            _WORKER_DATA["member1_lock"][evaluation_indices], dtype=np.float32
        )
        member2_evaluation = np.asarray(
            _WORKER_DATA["member2_lock"][evaluation_indices], dtype=np.float32
        )
        y_evaluation = np.asarray(_WORKER_DATA["y_lock"], dtype=np.float32)[
            evaluation_indices
        ]
    else:
        member1_evaluation = member1_dev[evaluation_indices]
        member2_evaluation = member2_dev[evaluation_indices]
        y_evaluation = np.asarray(_WORKER_DATA["y_dev"], dtype=np.float32)[
            evaluation_indices
        ]
    y_dev = np.asarray(_WORKER_DATA["y_dev"], dtype=np.float32)
    target = 6.0 - y_dev
    dataset = TensorDataset(
        torch.from_numpy(member1_dev[train_indices]),
        torch.from_numpy(member2_dev[train_indices]),
        torch.from_numpy(target[train_indices]),
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(task.seed),
        num_workers=0,
    )
    model = make_twin_heads()
    model.load_state_dict(_WORKER_DATA["head_state"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=task.learning_rate, weight_decay=0.0
    )
    evaluation_tensor1 = torch.from_numpy(member1_evaluation)
    evaluation_tensor2 = torch.from_numpy(member2_evaluation)
    evaluation_target = torch.from_numpy(6.0 - y_evaluation)
    max_epochs = task.fixed_epochs if task.fixed_epochs is not None else 30
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_mae = math.inf
    waiting = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch1, batch2, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction1, prediction2, ensemble = model(batch1, batch2)
            loss = combined_head_objective(
                prediction1,
                prediction2,
                ensemble,
                batch_target,
                objective=task.objective,
                delta=task.delta,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if task.fixed_epochs is not None:
            continue
        model.eval()
        with torch.inference_mode():
            _, _, evaluation_ensemble = model(evaluation_tensor1, evaluation_tensor2)
        predicted_p_ec50 = 6.0 - evaluation_ensemble.numpy()
        validation_mae = float(np.mean(np.abs(predicted_p_ec50 - y_evaluation)))
        if validation_mae < best_mae - 0.005:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            waiting = 0
        else:
            waiting += 1
            if waiting >= 5:
                break
    if task.fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("head task never produced a checkpoint")
        model.load_state_dict(best_state, strict=True)
    else:
        best_epoch = task.fixed_epochs
    model.eval()
    with torch.inference_mode():
        prediction1, prediction2, ensemble = model(
            evaluation_tensor1, evaluation_tensor2
        )
        native_loss = combined_head_objective(
            prediction1,
            prediction2,
            ensemble,
            evaluation_target,
            objective=task.objective,
            delta=task.delta,
        )
    prediction = (6.0 - ensemble.numpy()).astype(float)
    result = {
        **asdict(task),
        "objective_label": objective_label(task.objective, task.delta),
        "evaluation_indices": evaluation_indices,
        "prediction": prediction,
        "best_epoch": int(best_epoch),
        "native_loss": float(native_loss),
        "evaluation_n": len(evaluation_indices),
    }
    if use_evaluation_labels:
        result.update(regression_metrics(y_evaluation, prediction))
    return result


def _head_metric_rows(
    predictions: pd.DataFrame, *, dataset: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, subset in predictions.groupby("model", sort=False):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                **_safe_tail_metrics(subset["pEC50"], subset["predicted_pEC50"]),
            }
        )
    return rows


def _plot_head_lr_screen(screen: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    objectives = [objective_label(name, delta) for name, delta in HEAD_OBJECTIVES]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, label in zip(axes.flat, objectives, strict=True):
        subset = screen.loc[
            screen["scope"].eq("outer_training") & screen["objective_label"].eq(label)
        ]
        for outer_fold, fold_frame in subset.groupby("outer_fold", sort=True):
            fold_frame = fold_frame.sort_values("learning_rate")
            axis.plot(
                fold_frame["learning_rate"],
                fold_frame["native_loss"],
                marker="o",
                linewidth=0.9,
                alpha=0.55,
                label=f"outer {int(outer_fold)}",
            )
        aggregate = (
            subset.groupby("learning_rate")["native_loss"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("learning_rate")
        )
        sem = aggregate["std"] / np.sqrt(aggregate["count"])
        axis.errorbar(
            aggregate["learning_rate"],
            aggregate["mean"],
            yerr=sem,
            color="black",
            marker="s",
            linewidth=1.8,
            capsize=3,
            label="mean +/- SE",
        )
        axis.set_xscale("log")
        axis.set_title(label)
        axis.set_xlabel("AdamW learning rate")
        axis.set_ylabel("inner validation native loss")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        "Nested inner validation loss versus learning rate "
        "(loss scales are objective-specific)"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _bootstrap_head_models(
    development: pd.DataFrame,
    lockbox: pd.DataFrame,
    dev_predictions: pd.DataFrame,
    lock_predictions: pd.DataFrame,
    historical_dev_path: Path,
    historical_lock_path: Path,
    output_dir: Path,
) -> None:
    labels = [objective_label(name, delta) for name, delta in HEAD_OBJECTIVES] + [
        "global_selector"
    ]
    dev_wide = development[["feature_row", "pEC50", "cluster_component"]].copy()
    lock_wide = lockbox[["original_id", "pEC50", "audit_scaffold_group"]].copy()
    for label in labels:
        dev_subset = dev_predictions.loc[
            dev_predictions["model"].eq(label),
            ["feature_row", "predicted_pEC50"],
        ].rename(columns={"predicted_pEC50": label})
        lock_subset = lock_predictions.loc[
            lock_predictions["model"].eq(label),
            ["original_id", "predicted_pEC50"],
        ].rename(columns={"predicted_pEC50": label})
        dev_wide = dev_wide.merge(
            dev_subset, on="feature_row", how="left", validate="one_to_one"
        )
        lock_wide = lock_wide.merge(
            lock_subset, on="original_id", how="left", validate="one_to_one"
        )
    historical_dev = pd.read_csv(historical_dev_path)[
        ["feature_row", "nesso_pEC50"]
    ].rename(columns={"nesso_pEC50": "published_historical_ft"})
    historical_lock = pd.read_csv(historical_lock_path)[
        ["original_id", "predicted_pEC50_ensemble"]
    ].rename(columns={"predicted_pEC50_ensemble": "published_historical_ft"})
    dev_wide = dev_wide.merge(
        historical_dev, on="feature_row", how="left", validate="one_to_one"
    )
    lock_wide = lock_wide.merge(
        historical_lock, on="original_id", how="left", validate="one_to_one"
    )
    model_columns = {label: label for label in [*labels, "published_historical_ft"]}
    comparisons = [(label, "huber_0.5") for label in labels if label != "huber_0.5"] + [
        ("global_selector", "published_historical_ft"),
        ("huber_0.5", "published_historical_ft"),
    ]
    output_intervals: list[pd.DataFrame] = []
    output_differences: list[pd.DataFrame] = []
    for dataset, frame, group_column in (
        (
            "nested_development",
            dev_wide,
            "cluster_component",
        ),
        (
            "retrospective_lockbox_790",
            lock_wide,
            "audit_scaffold_group",
        ),
    ):
        intervals, draws = clustered_bootstrap(
            frame,
            model_columns,
            group_column=group_column,
            replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        points = {
            model: regression_metrics(frame["pEC50"], frame[column])
            for model, column in model_columns.items()
        }
        differences = paired_difference_rows(
            draws, points, comparisons, metrics=("mae", "spearman")
        )
        intervals.insert(0, "dataset", dataset)
        differences.insert(0, "dataset", dataset)
        output_intervals.append(intervals)
        output_differences.append(differences)
    pd.concat(output_intervals, ignore_index=True).to_csv(
        output_dir / "loss_lr_bootstrap_intervals.csv", index=False
    )
    pd.concat(output_differences, ignore_index=True).to_csv(
        output_dir / "loss_lr_paired_bootstrap_differences.csv", index=False
    )


def run_loss_lr_experiment(
    *,
    manifest_path: Path,
    features_path: Path,
    checkpoint_path: Path,
    historical_dev_path: Path,
    historical_lock_path: Path,
    analysis_spec: Path,
    output_dir: Path,
    max_workers: int,
) -> None:
    """Run the fully nested objective-by-learning-rate head study."""

    started, run_metadata = start_run_metadata(
        output_dir,
        stage="experiment_c_loss_lr",
        max_workers=max_workers,
        inputs={
            "manifest": manifest_path,
            "features": features_path,
            "checkpoint": checkpoint_path,
            "historical_development_predictions": historical_dev_path,
            "historical_lockbox_predictions": historical_lock_path,
        },
        analysis_spec=analysis_spec,
    )
    manifest, tensors = load_feature_bundle(manifest_path, features_path)
    development, lockbox, dev_representations, lock_representations = (
        split_feature_bundle(manifest, tensors)
    )
    y_dev = development["pEC50"].to_numpy(dtype=float)
    y_lock = lockbox["pEC50"].to_numpy(dtype=float)
    folds = development["fold"].to_numpy(dtype=int)
    set_worker_data(
        {
            "member1_dev": dev_representations["member1_384d"],
            "member2_dev": dev_representations["member2_384d"],
            "member1_lock": lock_representations["member1_384d"],
            "member2_lock": lock_representations["member2_384d"],
            "y_dev": y_dev,
            "y_lock": y_lock,
            "folds": folds,
            "head_state": load_pretrained_twin_state(checkpoint_path),
        }
    )

    inner_tasks: list[HeadTask] = []
    for outer_fold in sorted(np.unique(folds)):
        for objective, delta in HEAD_OBJECTIVES:
            for learning_rate in HEAD_LEARNING_RATES:
                for inner_fold in sorted(
                    value for value in np.unique(folds) if value != outer_fold
                ):
                    inner_tasks.append(
                        HeadTask(
                            objective=objective,
                            delta=delta,
                            learning_rate=learning_rate,
                            seed=42,
                            mode="inner",
                            outer_fold=int(outer_fold),
                            inner_fold=int(inner_fold),
                        )
                    )
    inner_results = _parallel_map(_fit_head_task, inner_tasks, max_workers)
    grouped_inner: dict[tuple[int, str, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for result in inner_results:
        grouped_inner[
            (
                int(result["outer_fold"]),
                result["objective_label"],
                float(result["learning_rate"]),
            )
        ].append(result)

    screen_rows: list[dict[str, Any]] = []
    outer_selections: list[dict[str, Any]] = []
    selected_objective: dict[tuple[int, str], dict[str, Any]] = {}
    selected_global: dict[int, dict[str, Any]] = {}
    labels = [objective_label(name, delta) for name, delta in HEAD_OBJECTIVES]
    objective_details = {
        objective_label(name, delta): (name, delta) for name, delta in HEAD_OBJECTIVES
    }
    for outer_fold in sorted(np.unique(folds)):
        all_candidates: list[dict[str, Any]] = []
        outer_training = folds != outer_fold
        for objective_order, (objective, delta) in enumerate(HEAD_OBJECTIVES):
            label = objective_label(objective, delta)
            objective_candidates: list[dict[str, Any]] = []
            for lr_order, learning_rate in enumerate(HEAD_LEARNING_RATES):
                results = grouped_inner[(int(outer_fold), label, learning_rate)]
                prediction = np.full(len(y_dev), np.nan, dtype=float)
                epochs: list[int] = []
                weighted_native_loss = 0.0
                validation_count = 0
                for result in results:
                    prediction[result["evaluation_indices"]] = result["prediction"]
                    epochs.append(int(result["best_epoch"]))
                    weighted_native_loss += float(result["native_loss"]) * int(
                        result["evaluation_n"]
                    )
                    validation_count += int(result["evaluation_n"])
                metrics = regression_metrics(
                    y_dev[outer_training], prediction[outer_training]
                )
                row = {
                    "scope": "outer_training",
                    "outer_fold": int(outer_fold),
                    "objective": objective,
                    "delta": delta,
                    "objective_label": label,
                    "learning_rate": learning_rate,
                    "candidate": f"{label}_lr{learning_rate:g}",
                    "candidate_order": lr_order,
                    "global_candidate_order": (
                        objective_order * len(HEAD_LEARNING_RATES) + lr_order
                    ),
                    "native_loss": weighted_native_loss / validation_count,
                    "best_epochs": ";".join(str(value) for value in epochs),
                    **metrics,
                }
                screen_rows.append(row)
                objective_candidates.append(row)
                all_candidates.append(row)
            selected = _stable_best(
                pd.DataFrame(objective_candidates), candidate_column="candidate"
            ).to_dict()
            selected["fixed_refit_epochs"] = max(
                1,
                int(
                    np.rint(
                        np.median(
                            [int(value) for value in selected["best_epochs"].split(";")]
                        )
                    )
                ),
            )
            selected_objective[(int(outer_fold), label)] = selected
            outer_selections.append(
                {
                    "selection_scope": "within_objective",
                    "outer_fold": int(outer_fold),
                    "objective_label": label,
                    "selected_objective": objective,
                    "selected_delta": delta,
                    "selected_learning_rate": selected["learning_rate"],
                    "fixed_refit_epochs": selected["fixed_refit_epochs"],
                    "inner_native_loss": selected["native_loss"],
                    "inner_mae": selected["mae"],
                    "inner_spearman": selected["spearman"],
                }
            )
        global_frame = pd.DataFrame(all_candidates).copy()
        global_frame["candidate_order"] = global_frame["global_candidate_order"]
        global_best = _stable_best(global_frame, candidate_column="candidate").to_dict()
        global_best["fixed_refit_epochs"] = max(
            1,
            int(
                np.rint(
                    np.median(
                        [int(value) for value in global_best["best_epochs"].split(";")]
                    )
                )
            ),
        )
        selected_global[int(outer_fold)] = global_best
        outer_selections.append(
            {
                "selection_scope": "global_all_objectives",
                "outer_fold": int(outer_fold),
                "objective_label": "global_selector",
                "selected_objective": global_best["objective"],
                "selected_delta": (
                    None
                    if pd.isna(global_best["delta"])
                    else float(global_best["delta"])
                ),
                "selected_learning_rate": global_best["learning_rate"],
                "fixed_refit_epochs": global_best["fixed_refit_epochs"],
                "inner_native_loss": global_best["native_loss"],
                "inner_mae": global_best["mae"],
                "inner_spearman": global_best["spearman"],
            }
        )

    outer_tasks: list[HeadTask] = []
    for (outer_fold, label), selected in selected_objective.items():
        objective, delta = objective_details[label]
        for seed in SEEDS:
            outer_tasks.append(
                HeadTask(
                    objective=objective,
                    delta=delta,
                    learning_rate=float(selected["learning_rate"]),
                    seed=seed,
                    mode="outer",
                    outer_fold=outer_fold,
                    fixed_epochs=int(selected["fixed_refit_epochs"]),
                )
            )
    outer_results = _parallel_map(_fit_head_task, outer_tasks, max_workers)
    objective_seed_predictions: dict[str, dict[int, np.ndarray]] = defaultdict(
        lambda: {seed: np.full(len(development), np.nan, dtype=float) for seed in SEEDS}
    )
    for result in outer_results:
        objective_seed_predictions[result["objective_label"]][int(result["seed"])][
            result["evaluation_indices"]
        ] = result["prediction"]
    dev_frames: list[pd.DataFrame] = []
    for label in labels:
        stack = np.stack([objective_seed_predictions[label][seed] for seed in SEEDS])
        if not np.isfinite(stack).all():
            raise AssertionError(f"incomplete nested predictions for {label}")
        dev_frames.append(
            pd.DataFrame(
                {
                    "feature_row": development["feature_row"],
                    "original_id": development["original_id"],
                    "fold": folds,
                    "pEC50": y_dev,
                    "model": label,
                    "predicted_seed_42": stack[0],
                    "predicted_seed_43": stack[1],
                    "predicted_seed_44": stack[2],
                    "predicted_pEC50": np.mean(stack, axis=0),
                    "prediction_seed_sd": np.std(stack, axis=0, ddof=1),
                }
            )
        )

    global_seed_predictions = {
        seed: np.full(len(development), np.nan, dtype=float) for seed in SEEDS
    }
    for outer_fold, selected in selected_global.items():
        label = str(selected["objective_label"])
        objective_selected = selected_objective[(outer_fold, label)]
        if not math.isclose(
            float(selected["learning_rate"]),
            float(objective_selected["learning_rate"]),
        ):
            raise AssertionError("global selector disagrees with within-objective LR")
        validation = folds == outer_fold
        for seed in SEEDS:
            global_seed_predictions[seed][validation] = objective_seed_predictions[
                label
            ][seed][validation]
    global_stack = np.stack([global_seed_predictions[seed] for seed in SEEDS])
    dev_frames.append(
        pd.DataFrame(
            {
                "feature_row": development["feature_row"],
                "original_id": development["original_id"],
                "fold": folds,
                "pEC50": y_dev,
                "model": "global_selector",
                "predicted_seed_42": global_stack[0],
                "predicted_seed_43": global_stack[1],
                "predicted_seed_44": global_stack[2],
                "predicted_pEC50": np.mean(global_stack, axis=0),
                "prediction_seed_sd": np.std(global_stack, axis=0, ddof=1),
            }
        )
    )
    dev_predictions = pd.concat(dev_frames, ignore_index=True)

    full_tasks: list[HeadTask] = []
    for objective, delta in HEAD_OBJECTIVES:
        for learning_rate in HEAD_LEARNING_RATES:
            for held_out_fold in sorted(np.unique(folds)):
                full_tasks.append(
                    HeadTask(
                        objective=objective,
                        delta=delta,
                        learning_rate=learning_rate,
                        seed=42,
                        mode="full_cv",
                        inner_fold=int(held_out_fold),
                    )
                )
    full_results = _parallel_map(_fit_head_task, full_tasks, max_workers)
    grouped_full: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for result in full_results:
        grouped_full[
            (result["objective_label"], float(result["learning_rate"]))
        ].append(result)
    full_selections: list[dict[str, Any]] = []
    selected_full_objective: dict[str, dict[str, Any]] = {}
    all_full_candidates: list[dict[str, Any]] = []
    for objective_order, (objective, delta) in enumerate(HEAD_OBJECTIVES):
        label = objective_label(objective, delta)
        objective_candidates: list[dict[str, Any]] = []
        for lr_order, learning_rate in enumerate(HEAD_LEARNING_RATES):
            results = grouped_full[(label, learning_rate)]
            prediction = np.full(len(y_dev), np.nan, dtype=float)
            epochs: list[int] = []
            weighted_native_loss = 0.0
            validation_count = 0
            for result in results:
                prediction[result["evaluation_indices"]] = result["prediction"]
                epochs.append(int(result["best_epoch"]))
                weighted_native_loss += float(result["native_loss"]) * int(
                    result["evaluation_n"]
                )
                validation_count += int(result["evaluation_n"])
            metrics = regression_metrics(y_dev, prediction)
            row = {
                "scope": "full_development",
                "outer_fold": np.nan,
                "objective": objective,
                "delta": delta,
                "objective_label": label,
                "learning_rate": learning_rate,
                "candidate": f"{label}_lr{learning_rate:g}",
                "candidate_order": lr_order,
                "global_candidate_order": (
                    objective_order * len(HEAD_LEARNING_RATES) + lr_order
                ),
                "native_loss": weighted_native_loss / validation_count,
                "best_epochs": ";".join(str(value) for value in epochs),
                **metrics,
            }
            screen_rows.append(row)
            objective_candidates.append(row)
            all_full_candidates.append(row)
        selected = _stable_best(
            pd.DataFrame(objective_candidates), candidate_column="candidate"
        ).to_dict()
        selected["fixed_refit_epochs"] = max(
            1,
            int(
                np.rint(
                    np.median(
                        [int(value) for value in selected["best_epochs"].split(";")]
                    )
                )
            ),
        )
        selected_full_objective[label] = selected
        full_selections.append(
            {
                "selection_scope": "within_objective",
                "objective_label": label,
                "selected_objective": objective,
                "selected_delta": delta,
                "selected_learning_rate": selected["learning_rate"],
                "fixed_refit_epochs": selected["fixed_refit_epochs"],
                "development_cv_native_loss": selected["native_loss"],
                "development_cv_mae": selected["mae"],
                "development_cv_spearman": selected["spearman"],
            }
        )
    full_global_frame = pd.DataFrame(all_full_candidates).copy()
    full_global_frame["candidate_order"] = full_global_frame["global_candidate_order"]
    selected_full_global = _stable_best(
        full_global_frame, candidate_column="candidate"
    ).to_dict()
    selected_full_global["fixed_refit_epochs"] = max(
        1,
        int(
            np.rint(
                np.median(
                    [
                        int(value)
                        for value in selected_full_global["best_epochs"].split(";")
                    ]
                )
            )
        ),
    )
    full_selections.append(
        {
            "selection_scope": "global_all_objectives",
            "objective_label": "global_selector",
            "selected_objective": selected_full_global["objective"],
            "selected_delta": (
                None
                if pd.isna(selected_full_global["delta"])
                else float(selected_full_global["delta"])
            ),
            "selected_learning_rate": selected_full_global["learning_rate"],
            "fixed_refit_epochs": selected_full_global["fixed_refit_epochs"],
            "development_cv_native_loss": selected_full_global["native_loss"],
            "development_cv_mae": selected_full_global["mae"],
            "development_cv_spearman": selected_full_global["spearman"],
        }
    )

    lock_tasks: list[HeadTask] = []
    for label, selected in selected_full_objective.items():
        objective, delta = objective_details[label]
        for seed in SEEDS:
            lock_tasks.append(
                HeadTask(
                    objective=objective,
                    delta=delta,
                    learning_rate=float(selected["learning_rate"]),
                    seed=seed,
                    mode="lockbox",
                    fixed_epochs=int(selected["fixed_refit_epochs"]),
                )
            )
    lock_results = _parallel_map(_fit_head_task, lock_tasks, max_workers)
    lock_seed_predictions: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for result in lock_results:
        lock_seed_predictions[result["objective_label"]][int(result["seed"])] = result[
            "prediction"
        ]
    lock_frames: list[pd.DataFrame] = []
    for label in labels:
        stack = np.stack([lock_seed_predictions[label][seed] for seed in SEEDS])
        lock_frames.append(
            pd.DataFrame(
                {
                    "feature_row": lockbox["feature_row"],
                    "original_id": lockbox["original_id"],
                    "pEC50": y_lock,
                    "model": label,
                    "predicted_seed_42": stack[0],
                    "predicted_seed_43": stack[1],
                    "predicted_seed_44": stack[2],
                    "predicted_pEC50": np.mean(stack, axis=0),
                    "prediction_seed_sd": np.std(stack, axis=0, ddof=1),
                }
            )
        )
    full_global_label = str(selected_full_global["objective_label"])
    within_global_objective = selected_full_objective[full_global_label]
    if not math.isclose(
        float(selected_full_global["learning_rate"]),
        float(within_global_objective["learning_rate"]),
    ):
        raise AssertionError("full global selector disagrees with objective LR")
    global_lock_stack = np.stack(
        [lock_seed_predictions[full_global_label][seed] for seed in SEEDS]
    )
    lock_frames.append(
        pd.DataFrame(
            {
                "feature_row": lockbox["feature_row"],
                "original_id": lockbox["original_id"],
                "pEC50": y_lock,
                "model": "global_selector",
                "predicted_seed_42": global_lock_stack[0],
                "predicted_seed_43": global_lock_stack[1],
                "predicted_seed_44": global_lock_stack[2],
                "predicted_pEC50": np.mean(global_lock_stack, axis=0),
                "prediction_seed_sd": np.std(global_lock_stack, axis=0, ddof=1),
            }
        )
    )
    lock_predictions = pd.concat(lock_frames, ignore_index=True)

    screen = pd.DataFrame(screen_rows)
    screen.to_csv(output_dir / "loss_lr_inner_screen.csv", index=False)
    pd.DataFrame(outer_selections).to_csv(
        output_dir / "loss_lr_outer_selections.csv", index=False
    )
    pd.DataFrame(full_selections).to_csv(
        output_dir / "loss_lr_full_development_selections.csv", index=False
    )
    dev_predictions.to_csv(
        output_dir / "loss_lr_development_predictions.csv", index=False
    )
    lock_predictions.to_csv(output_dir / "loss_lr_lockbox_predictions.csv", index=False)
    _plot_head_lr_screen(screen, output_dir / "validation_loss_vs_learning_rate.png")

    metric_rows = _head_metric_rows(
        dev_predictions, dataset="nested_development"
    ) + _head_metric_rows(lock_predictions, dataset="retrospective_lockbox_790")
    historical_dev = pd.read_csv(historical_dev_path)
    historical_lock = pd.read_csv(historical_lock_path)
    metric_rows.extend(
        [
            {
                "dataset": "nested_development",
                "model": "published_historical_ft",
                **_safe_tail_metrics(
                    historical_dev["pEC50"], historical_dev["nesso_pEC50"]
                ),
            },
            {
                "dataset": "retrospective_lockbox_790",
                "model": "published_historical_ft",
                **_safe_tail_metrics(
                    historical_lock["pEC50"],
                    historical_lock["predicted_pEC50_ensemble"],
                ),
            },
        ]
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "loss_lr_metrics.csv", index=False)

    seed_rows: list[dict[str, Any]] = []
    for dataset, frame in (
        ("nested_development", dev_predictions),
        ("retrospective_lockbox_790", lock_predictions),
    ):
        for model, subset in frame.groupby("model", sort=False):
            for seed in SEEDS:
                column = f"predicted_seed_{seed}"
                seed_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        **regression_metrics(subset["pEC50"], subset[column]),
                    }
                )
    pd.DataFrame(seed_rows).to_csv(
        output_dir / "loss_lr_seed_variability.csv", index=False
    )
    _bootstrap_head_models(
        development,
        lockbox,
        dev_predictions,
        lock_predictions,
        historical_dev_path,
        historical_lock_path,
        output_dir,
    )

    labels_array = development["pEC50"].to_numpy(dtype=float)
    standard_errors = development["pEC50_standard_error"].to_numpy(dtype=float)
    context = {
        "pEC50_minimum": float(np.min(labels_array)),
        "pEC50_q25": float(np.quantile(labels_array, 0.25)),
        "pEC50_median": float(np.median(labels_array)),
        "pEC50_q75": float(np.quantile(labels_array, 0.75)),
        "pEC50_maximum": float(np.max(labels_array)),
        "pEC50_iqr": float(
            np.quantile(labels_array, 0.75) - np.quantile(labels_array, 0.25)
        ),
        "active_threshold": 6.0,
        "active_n": int(np.sum(labels_array > 6.0)),
        "assay_standard_error_median": float(np.median(standard_errors)),
        "assay_standard_error_q75": float(np.quantile(standard_errors, 0.75)),
        "huber_delta_0_5_concentration_fold_error": float(10**0.5),
        "huber_delta_0_5_over_median_assay_standard_error": float(
            0.5 / np.median(standard_errors)
        ),
        "huber_delta_0_5_fraction_of_pEC50_iqr": float(
            0.5 / (np.quantile(labels_array, 0.75) - np.quantile(labels_array, 0.25))
        ),
        "target_coordinate": (
            "6-pEC50; exactly equivalent to direct pEC50 after final-layer "
            "sign/offset transformation"
        ),
    }
    (output_dir / "loss_scale_context.json").write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n"
    )
    development_metrics = metrics.loc[metrics["dataset"].eq("nested_development")]
    summary = {
        "status": "complete",
        "development_metrics": development_metrics.to_dict("records"),
        "selected_learning_rates_by_outer_fold": outer_selections,
        "compression_interpretation_rule": (
            "A loss materially relieves compression only if active-tail MAE, "
            "prediction maximum, and count above pEC50 6 improve together without "
            "a clear overall-MAE penalty."
        ),
        "huber_delta_context": context,
        "lockbox_status": "retrospective_validation",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    finish_run_metadata(output_dir, started, run_metadata)
