"""Read-only first-study analysis and a separately owned, evidence-linked site.

No model factory, checkpoint deserialization, extraction, or fitting is invoked.
Original evidence is copied byte-for-byte; changed existing copies fail closed.
"""

# ruff: noqa: E501 -- embedded report prose and CSS retain readable source paragraphs.
from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import ProxyHandler, build_opener

import markdown
import numpy as np
import pandas as pd

EXPERIMENT = Path("experiments/20260906_low_data_followup")
RUN = Path("artifacts/experiments/low_data_20260905_cut035_run2")
OLD_SITE = Path("site/low-data-verified")
SITE = Path("site/low-data-assessment")
KEYS = ["split", "n_train", "method"]
CELL = ["split", "outer_fold", "draw", "n_train"]
CONTRASTS = [
    ("head_pretrained", "head_random"),
    ("head_pretrained", "morgan_lightgbm"),
    ("head_lora", "head_pretrained"),
    ("dual_head", "head_pretrained"),
    ("head_pretrained", "native_continuous"),
    ("repr_ridge", "morgan_lightgbm"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        payload = payload.encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: dict) -> None:
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, float_precision="round_trip", keep_default_na=False, na_values=[""]
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def immutable_copy(source: Path, target: Path) -> dict:
    digest = sha256(source)
    if target.exists():
        require(sha256(target) == digest, f"changed immutable evidence: {target}")
    else:
        atomic_write(target, source.read_bytes())
    require(sha256(target) == digest, f"copy mismatch: {target}")
    return {"sha256": digest, "bytes": source.stat().st_size}


def validate_matrix(tasks: pd.DataFrame, manifest: list, protocol: dict) -> dict:
    columns = ["task_id", *CELL, "method"]
    require(not tasks.task_id.duplicated().any(), "duplicate task IDs")
    require(not tasks.duplicated(CELL + ["method"]).any(), "duplicate matrix cells")
    budgets = protocol["budgets"] + ([-1] if protocol["include_full_reference"] else [])
    expected = set(
        itertools.product(
            protocol["split_policies"],
            range(protocol["n_folds"]),
            range(protocol["draws"]),
            budgets,
            protocol["methods"],
        )
    )
    observed = set(tasks[CELL + ["method"]].itertuples(index=False, name=None))
    require(expected == observed, "incomplete or unexpected matrix cells")
    manifest_frame = pd.DataFrame(manifest)
    require(len(manifest_frame) == len(tasks), "manifest count mismatch")
    require(
        set(map(tuple, manifest_frame[columns].to_numpy()))
        == set(map(tuple, tasks[columns].to_numpy())),
        "manifest task mismatch",
    )
    require(
        tasks.status.eq("complete").all() and tasks.verification.eq("verified").all(),
        "unverified or incomplete tasks",
    )
    require(
        np.isfinite(tasks[["weighted_mae", "mae", "bias", "weight_sum"]]).all().all(),
        "nonfinite metrics",
    )
    require(tasks.weight_sum.gt(0).all(), "nonpositive weights")
    require(
        (tasks.n_fit + tasks.n_calibration == tasks.n_acquired).all(),
        "label budget accounting mismatch",
    )
    finite = tasks.n_train.ne(-1)
    require(
        tasks.loc[finite, "n_train"].eq(tasks.loc[finite, "n_acquired"]).all(),
        "acquired N mismatch",
    )
    require(
        tasks.n_calibration.eq(np.ceil(tasks.n_acquired * 0.2)).all(),
        "calibration budget mismatch",
    )
    for column in ["row_signature", "n_test", "n_fit", "n_calibration", "weight_sum"]:
        require(
            tasks.groupby(CELL)[column].nunique(dropna=False).eq(1).all(),
            f"unpaired cell {column}",
        )
    return {
        "expected_tasks": len(expected),
        "validated_task_rows": len(tasks),
        "fully_paired_cells": tasks.groupby(CELL).ngroups,
        "methods_per_cell": len(protocol["methods"]),
        "prediction_observations": int(tasks.n_test.sum()),
    }


def aggregate(tasks: pd.DataFrame) -> pd.DataFrame:
    result = (
        tasks.groupby(KEYS)
        .agg(
            n_cells=("task_id", "size"),
            wmae_cell_mean=("weighted_mae", "mean"),
            wmae_cell_median=("weighted_mae", "median"),
            wmae_cell_min=("weighted_mae", "min"),
            wmae_cell_max=("weighted_mae", "max"),
            weighted_error_sum=("weighted_error_sum", "sum"),
            weight_sum=("weight_sum", "sum"),
            raw_mae=("mae", "mean"),
            bias=("bias", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_median=("fit_seconds", "median"),
            fit_seconds_sum=("fit_seconds", "sum"),
            coverage80=("coverage_80", "mean"),
            coverage90=("coverage_90", "mean"),
            width90=("width_90", "mean"),
            wmae_floor005=("weighted_mae_floor_005", "mean"),
            wmae_floor020=("weighted_mae_floor_020", "mean"),
        )
        .reset_index()
    )
    result["wmae_pooled"] = result.weighted_error_sum / result.weight_sum
    return result


def paired(tasks: pd.DataFrame) -> pd.DataFrame:
    wide = tasks.pivot(index=CELL, columns="method", values="weighted_mae")
    rows = []
    for a, b in CONTRASTS:
        delta = (wide[a] - wide[b]).rename("delta").reset_index()
        for (split, budget), group in delta.groupby(["split", "n_train"]):
            fold = group.groupby("outer_fold").delta.mean()
            x = group.delta
            rows.append(
                dict(
                    split=split,
                    n_train=int(budget),
                    method_a=a,
                    method_b=b,
                    n_cells=len(x),
                    mean_delta=float(x.mean()),
                    median_delta=float(x.median()),
                    min_delta=float(x.min()),
                    max_delta=float(x.max()),
                    fraction_negative=float((x < 0).mean()),
                    fold_mean_min=float(fold.min()),
                    fold_mean_max=float(fold.max()),
                    n_negative_fold_means=int((fold < 0).sum()),
                )
            )
    return pd.DataFrame(rows)


def compare_tables(actual: pd.DataFrame, reference: pd.DataFrame, keys: list) -> float:
    a = actual.set_index(keys).sort_index()
    b = reference.set_index(keys).sort_index()
    require(a.index.equals(b.index), "aggregate identity mismatch")
    maximum = 0.0
    for column in b.columns:
        require(column in a, f"missing aggregate column {column}")
        av, bv = a[column].to_numpy(float), b[column].to_numpy(float)
        require(
            np.allclose(av, bv, atol=2e-12, rtol=2e-12, equal_nan=True),
            f"original metric mismatch: {column}",
        )
        finite = np.isfinite(av) & np.isfinite(bv)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(av[finite] - bv[finite]))))
    return maximum


def prediction_diagnostics(
    root: Path, tasks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Re-score every original prediction, keep all extremes, verify all output hashes."""
    tails, potency = [], []
    checked = 0
    discrepancy = 0.0
    ids = set()
    prepared = root / RUN / "prepared"
    inputs = read_csv(prepared / "inputs.csv").set_index("row_index")
    subsets = read_csv(prepared / "subsets.csv")
    assignments = read_csv(prepared / "assignments.csv")
    roles = {}
    for key, group in subsets.groupby(CELL + ["role"], sort=False):
        roles[key] = inputs.loc[group.row_index, "record_id"].tolist()
    test_ids = {}
    for key, group in assignments.groupby(["split", "outer_fold"], sort=False):
        test_ids[key] = inputs.loc[group.row_index, "record_id"].tolist()
    required_outputs = {
        "metrics.json",
        "seed_predictions.npz",
        "model_state.pt",
        "predictions.csv",
        "started.json",
        "label_access.json",
    }
    for task in tasks.itertuples(index=False):
        directory = root / RUN / "tasks" / task.task_id
        completion = json.loads((directory / "complete.json").read_text())
        require(completion["task_id"] == task.task_id, "completion identity mismatch")
        require(
            set(completion["output_sha256"]) == required_outputs,
            "task output inventory mismatch",
        )
        for name, expected in completion["output_sha256"].items():
            require(
                sha256(directory / name) == expected,
                f"task hash mismatch: {task.task_id}/{name}",
            )
            checked += 1
        require(
            completion["output_sha256"]["predictions.csv"] == task.prediction_sha256,
            "task table prediction binding mismatch",
        )
        require(
            completion["output_sha256"]["metrics.json"] == task.metrics_sha256,
            "task table metric binding mismatch",
        )
        data = read_csv(directory / "predictions.csv")
        access = json.loads((directory / "label_access.json").read_text())
        cell = (task.split, task.outer_fold, task.draw, task.n_train)
        for role in ["fit", "calibration"]:
            require(
                access[f"{role}_record_ids"] == roles[(*cell, role)],
                f"prepared label-role mismatch: {task.task_id}/{role}",
            )
        expected_test = test_ids[(task.split, task.outer_fold)]
        require(
            access["test_record_ids"] == expected_test == data.record_id.tolist(),
            f"prepared test-role mismatch: {task.task_id}",
        )
        fit_ids, cal_ids = (
            set(access["fit_record_ids"]),
            set(access["calibration_record_ids"]),
        )
        require(
            not fit_ids.intersection(cal_ids)
            and not (fit_ids | cal_ids).intersection(expected_test),
            "label role overlap",
        )
        require(
            len(data) == task.n_test and not data.record_id.duplicated().any(),
            "prediction identity/count mismatch",
        )
        for col in ["split", "outer_fold", "draw", "n_train", "method"]:
            require(
                data[col].eq(getattr(task, col)).all(),
                f"prediction task identity: {col}",
            )
        ids.update(data.record_id)
        y, p, se = (data[k].to_numpy(float) for k in ["y_true", "y_pred", "assay_se"])
        require(
            np.isfinite(np.column_stack([y, p, se])).all() and (se > 0).all(),
            "invalid original predictions or assay errors",
        )
        w = 1 / np.maximum(se, 0.1)
        require(
            np.array_equal(w, data.weight.to_numpy(float)), "evaluation weight mismatch"
        )
        error = np.abs(p - y)
        for name, value in [
            ("weighted_mae", np.average(error, weights=w)),
            ("mae", error.mean()),
            ("bias", (p - y).mean()),
            (
                "weighted_mae_floor_005",
                np.average(error, weights=1 / np.maximum(se, 0.05)),
            ),
            (
                "weighted_mae_floor_020",
                np.average(error, weights=1 / np.maximum(se, 0.2)),
            ),
        ]:
            diff = abs(float(value) - getattr(task, name))
            discrepancy = max(discrepancy, diff)
            require(diff < 2e-12, f"prediction metric mismatch: {task.task_id}/{name}")
        tails.append(
            dict(
                task_id=task.task_id,
                split=task.split,
                n_train=task.n_train,
                method=task.method,
                pred_min=float(p.min()),
                pred_q01=float(np.quantile(p, 0.01)),
                pred_median=float(np.median(p)),
                pred_q99=float(np.quantile(p, 0.99)),
                pred_max=float(p.max()),
                pred_sd=float(p.std()),
                observed_sd=float(y.std()),
                max_absolute_error=float(error.max()),
            )
        )
        bins = np.digitize(y, [3.5, 4.5, 5.5])
        for i, label in enumerate(["<3.5", "[3.5,4.5)", "[4.5,5.5)", ">=5.5"]):
            mask = bins == i
            potency.append(
                dict(
                    split=task.split,
                    n_train=task.n_train,
                    method=task.method,
                    potency_bin=label,
                    n_observations=int(mask.sum()),
                    error_sum=float(error[mask].sum()),
                    bias_sum=float((p - y)[mask].sum()),
                    weight_sum=float(w[mask].sum()),
                    weighted_error_sum=float((w * error)[mask].sum()),
                )
            )
    bins = (
        pd.DataFrame(potency)
        .groupby(KEYS + ["potency_bin"], sort=True)
        .sum()
        .reset_index()
    )
    bins["raw_mae"] = bins.error_sum / bins.n_observations
    bins["bias"] = bins.bias_sum / bins.n_observations
    bins["weighted_mae"] = bins.weighted_error_sum / bins.weight_sum
    return (
        pd.DataFrame(tails),
        bins,
        {
            "checked_task_output_hashes": checked,
            "prediction_tasks_recomputed": len(tasks),
            "label_access_files_matched_prepared": len(tasks),
            "unique_development_compounds": len(ids),
            "max_prediction_metric_discrepancy": discrepancy,
            "model_replay": "Prior independent assessment replayed four systems exactly; this closeout hashes model bytes but does not deserialize or replay models.",
        },
    )


def md_table(headers: list, rows: list) -> str:
    return "\n".join(
        [
            "| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def report_text(
    agg: pd.DataFrame,
    differences: pd.DataFrame,
    facts: dict,
    sources: dict,
    created: str,
) -> str:
    indexed = agg.set_index(KEYS)

    def value(split, n, method, metric="wmae_cell_mean"):
        return float(indexed.loc[(split, n, method), metric])

    rows = []
    for method in facts["methods"]:
        rows.append(
            [method]
            + [
                f"{value(split, n, method):.4f}"
                for split in ["random", "chemical_cluster"]
                for n in [25, 100, 500, -1]
            ]
        )
    main = md_table(
        [
            "Method",
            "Random 25",
            "100",
            "500",
            "Full",
            "Cluster 25",
            "100",
            "500",
            "Full",
        ],
        rows,
    )
    p = differences.set_index(["split", "n_train", "method_a", "method_b"])
    contrast_rows = []
    for split in ["random", "chemical_cluster"]:
        for n in [25, 100, 500, -1]:
            q = p.loc[(split, n, "head_pretrained", "head_random")]
            contrast_rows.append(
                [
                    split,
                    "full" if n == -1 else n,
                    f"{q.mean_delta:+.4f}",
                    f"{q.fold_mean_min:+.4f} to {q.fold_mean_max:+.4f}",
                ]
            )
    contrasts = md_table(
        ["Split", "Labels", "Pretrained − random", "Draw-averaged fold range"],
        contrast_rows,
    )
    diagnostics = md_table(
        [
            "Split / full budget",
            "Weighted MAE",
            "Raw MAE",
            "Bias (pred − true)",
            "90% coverage",
            "90% width",
        ],
        [
            [split]
            + [
                f"{value(split, -1, 'head_pretrained', metric):.4f}"
                for metric in [
                    "wmae_cell_mean",
                    "raw_mae",
                    "bias",
                    "coverage90",
                    "width90",
                ]
            ]
            for split in ["random", "chemical_cluster"]
        ],
    )
    evidence = "\n".join(
        f"- [{name}]({item['copy']})"
        for name, item in sources.items()
        if not name.startswith("bound/")
    )
    return f"""# Nesso PXR: completed low-data study

This study tests how well target-fitted readouts on frozen Nesso-1 vectors predict cellular PXR pEC50 from small, counted label budgets.

Created: {created}. Last edited: {facts["generated_at"]}. Retrospective development-set assessment; no model fits were run for this closeout.

## What was learned

**The first study is complete: {facts["validated_task_rows"]:,}/{facts["expected_tasks"]:,} task systems, {facts["fully_paired_cells"]:,} fully paired cells. Scientific completion does not establish a deployment-ready winner.** Twin continuous heads are the provisional Nesso route within the tested recipes. More labels help; released head initialization does not show a consistent advantage over the matched random head. Extra binary branches add complexity without a compelling observed gain, and rank-4 final-head LoRA trades accuracy for a modest fitting-cost reduction.

At N500, pretrained-head mean weighted MAE is {value("random", 500, "head_pretrained"):.4f} / {value("chemical_cluster", 500, "head_pretrained"):.4f} (random / cluster), versus native {value("random", 500, "native_continuous"):.4f} / {value("chemical_cluster", 500, "native_continuous"):.4f}. At N25, adaptation can worsen the cluster score. The scaled generic MLP and ridge have demonstrated extrapolation vulnerabilities; these are not evidence that all MLPs, ridge models, or frozen representations fail.

The comparator is **Morgan-only LightGBM**, not the historical descriptor-rich system. Full-budget Morgan ridge is close to the continuous heads, and random-full raw MAE favors Morgan ridge ({value("random", -1, "morgan_ridge", "raw_mae"):.4f} versus {value("random", -1, "head_pretrained", "raw_mae"):.4f}). This study cannot establish superiority over the absent richer comparator. Weighted median and native linear calibration were absent too.

## What was tested and where labels entered

```text
Canonical identity + prepared ligand state
→ released Nesso-1 v1.0.0 cached affinity vectors (2 × 384D, frozen)
  OR deterministic radius-2 / 2,048-bit Morgan fingerprints
→ method-specific preprocessing and readout [fit labels only]
→ arithmetic mean of final neural seeds 42 / 43 / 44
→ residual split-conformal intervals [reserved calibration labels inside N]
→ fixed outer-test cellular pEC50 evaluation [never fitting or selection]
```

The matrix is two policies × five folds × ten paired draws × six budgets × twelve methods. N=25/50/100/250/500 means 20/40/80/200/400 fit-and-selection labels plus 5/10/20/50/100 calibration labels. Full uses 2,675–2,676 acquired labels, 2,140 for fitting and 535–536 for calibration—not all 3,344 development labels. Acquired prefixes nest, but fit-only sets need not nest as calibration suffixes move. Repeated native point predictions are an unchanged anchor. Nonneural models fit once per task and repeat identical predictions across seed slots, not three independent fits.

| Method | Exact fitted flow and boundary |
| --- | --- |
| weighted_mean | Inverse-SE-weighted fit-label mean; not the absolute-loss-optimal weighted median. |
| native_continuous | Unchanged released two-member mean after `6 − raw_affinity`; pIC50-equivalent affinity, not native cellular pEC50. No point-prediction fitting; intervals still use calibration labels. |
| scalar_ridge | Nine reconstructed member/ensemble continuous values, binary logits and probabilities → weighted fit-only StandardScaler → weighted ridge. |
| repr_ridge | Concatenated 768D vectors → weighted fit-only StandardScaler → weighted ridge. |
| repr_mlp | Same scaled 768D inputs → 768–128–1 ReLU MLP, direct pEC50 output, no target centering; 98,561 fitted parameters per seed. |
| head_random / head_pretrained | Unscaled vectors → matched twin 384–384–384–1 ReLU MLPs → `6 − raw` per member → mean; 592,130 fitted parameters per seed. Only head initialization differs; upstream encoder is pretrained and frozen in both. |
| dual_head | Both continuous and binary-score MLPs, binary-logit maps and fusion gates train; raw member output = `continuous + 1.5*tanh(scale*binary_logit + offset)` before conversion. Gates start at zero; no extra binary assay labels; 1,184,268 fitted parameters per seed. |
| head_lora | Rank-4 A/B updates in all three Linear layers of both released continuous MLPs; alpha/rank=1, zero-B initialization. 15,368 fitted and 607,498 stored neural parameters per seed; base heads and vectors frozen. Not upstream adaptation. |
| morgan_ridge | Unscaled binary Morgan bits → weighted ridge; no descriptors or PCA. |
| morgan_lightgbm | Morgan-only absolute-loss trees; seven leaves, learning rate .05, min_child_samples 5, L2=1, 50/150 trees selected inside fit labels. |
| morgan_knn | Weighted label mean among 1/5 nearest Tanimoto neighbors; retains fit bits and labels. |

Selection uses three-fold **GroupKFold for both policies in the actual runner**: chemical IDs for cluster tasks, unique record IDs for random tasks. Every inner scaler is fitted anew inside its inner-training partition; final scalers use only final fit rows. Ridge alpha is 1/100/10,000. Neural LR .0001/.001 and epochs 1–40 are chosen by pooled inner-OOF weighted MAE with seed 42; all final seeds refit. AdamW has zero weight decay, batch size 64, gradient norm clipping 1, weighted smooth-L1 beta .5 and ensemble/member loss weights .50/.25/.25. Training weights are globally normalized. Ridge fits weighted squared loss; trees fit weighted absolute loss. Label curves also change optimizer update counts, not just label count at fixed computation. Generic MLP versus twin head changes scaling, architecture, initialization and output parameterization together: it is not a controlled one-factor comparison.

## Results: task means, not independent replicates

Headline weighted MAE is `sum(w * abs(pred − true)) / sum(w)`, with **w = 1 / max(assay SE, 0.10)**, in pEC50 units. It is inverse SE, not inverse variance or model-uncertainty weighting; curation weights are excluded. The floor is a fixed pre-outcome engineering cap, not independently measured assay noise. Weights are fixed across methods and budgets on the same test identities.

{main}

Each entry averages 50 paired fold/draw task scores; all original extremes are retained. N50 and N250, medians/ranges, raw MAE, bias, interval diagnostics, cost and floor .05/.20 score-only sensitivities are in [recomputed aggregates](aggregates.csv). [Original pooled metrics](sources/metrics.csv) instead divide total weighted error by total weight. Small differences from task means are expected and verified; neither aggregation grants independence to repeated rows. Score sensitivities do not retrain or select a new method.

### Released head initialization: mixed descriptive effects

{contrasts}

Negative means lower error for pretrained heads. Fold ranges are observed variation, **not confidence intervals**. [All six paired contrasts](paired.csv) retain task median/range and fold-average signs. There is no accepted practical improvement margin and no dependence-aware inferential claim of superiority or equivalence. A random head here is not a random encoder.

### Bias, potency and uncertainty remain limitations

{diagnostics}

Full-budget pretrained-head task means are above. [Potency-stratified error and bias](potency.csv) report all saved observations in explicitly left-closed bins [3.5,4.5), [4.5,5.5), with tails &lt;3.5 and ≥5.5. [Prediction-tail diagnostics](prediction_tails.csv) preserve every task's range and quantiles. These pooled diagnostic observation counts include repeated compounds; they are not independent sample sizes. Low potency or a numeric SE alone does not establish noise, inactivity or censoring. pEC50 is potency, not Emax/efficacy or clinical liability.

Ordinary residual split-conformal intervals are not SE-weighted coverage guarantees. N25 has only five calibration labels: all nominal 90% intervals are unbounded, so 100% observed coverage is not useful precision. Larger-budget marginal coverage does not imply conditional or chemical-shift coverage. Seed SD is a separate uncalibrated diagnostic. The original revised reporter preserves the runner's residual predicate; 25 task/coverage-level floating-boundary differences were independently reconciled without widening intervals ([receipt](sources/interval_replay.json)).

## What failed: reproducible does not mean safe

In saved-state probes from the [independent assessment](sources/independent_assessment.md), random fold2/draw09 N50 representation ridge has a fit-only scale 5.77×10⁻⁶, a held-out standardized value around 232,721 and a single-feature contribution of magnitude 1,706.817. Prediction −1,704.6863 for measured pEC50 2.69 replays exactly; task weighted MAE is 5.3879. This is a fitted-system scaling/extrapolation vulnerability, not CSV corruption or evidence of valid deployment behavior. Removing even its worst row descriptively does not repair the task.

The worst N25 MLP probe has a standardized held-out value around 7,966 and prediction −354.51 despite decreasing training loss. Its selected epoch is 25, so a universally too-short 40-epoch schedule is not demonstrated. N25 MLP task-median errors ({value("random", 25, "repr_mlp", "wmae_cell_median"):.4f} / {value("chemical_cluster", 25, "repr_mlp", "wmae_cell_median"):.4f}) show the weakness is broader than one row. These probes establish poor out-of-sample behavior, not that scaling alone explains every failure. Stable scaling, regularization and target centering are **new controlled hypotheses**. No clipping, extreme-row removal or post-hoc replacement changes the official scores here.

## Generalization: what remains unproven

Only 3,344 development identities enter this matrix. The historically opened 790- and 513-compound challenge sets are excluded from fitting, tuning and evaluation here; that exclusion does not make this a prospective blinded study. Neither upstream pretraining identity/endpoint overlap nor every historical development choice is ruled out by these artifact checks. Follow-up recipes informed by these outcomes remain retrospective and hypothesis-generating, including any repartition of these same compounds.

The chemical split holds Morgan Butina .35 **centroid group IDs** apart: {facts["n_groups"]:,} groups, {facts["singleton_compounds"]:,} singleton compounds ({facts["singleton_percent"]:.2f}%), largest group {facts["largest_group"]}. It does not enforce strict pairwise chemical separation. The actual complementary-pool nearest-neighbor median is {facts["cluster_nn_median"]:.3f}, q95 {facts["cluster_nn_q95"]:.3f}, maximum {facts["cluster_nn_max"]:.3f}; {facts["cluster_nn_ge035_percent"]:.2f}% have an outer-pool neighbor at or above .35. Shared scaffolds remain. These similarities use the full outer pool, not the smaller fitted subset. The structure-only .50→.35 amendment and both histories are preserved. Similar random/cluster scores do not prove robustness to chemical shift; no temporal, prospective, cross-target or strict-novelty validation is established.

There are {facts["prediction_observations"]:,} prediction observations but {facts["unique_development_compounds"]:,} unique compounds. Test identities repeat across draws/budgets, outer training pools overlap, and seeds share labels. Fifty task scores are not fifty independent experiments. Unique-compound effective weighted N is 2,782.34; all assay SEs are finite and positive and 1,107 are at/below .10, as independently verified. No independent-row bootstrap or invented uncertainty/margin is used.

Final-head LoRA is not the separate upstream feasibility pilot. That pilot updated 12,288 rank-4 parameters in first-member `esm_proj.1`, `esm_proj.3` and `pairformer_stack.layers.0.transition_z.fc1` on one real input using two SGD steps and a synthetic detached-native-plus-one target, **without an assay label**. Gradient/frozen-weight/reload checks establish an engineering path, not PXR learning or both-member adaptation. No upstream fits or new models were run for this closeout.

## Completion and verification

This closeout freshly reconciles the final task table against the original protocol Cartesian product and task manifest, re-scores all {facts["prediction_observations"]:,} original prediction rows, and checks all {facts["checked_task_output_hashes"]:,} task-output SHA-256 values. It also rechecks {facts["bound_sources_checked"]} bound source files and {facts["bound_inputs_checked"]} bound inputs. Maximum prediction metric discrepancy: {facts["max_prediction_metric_discrepancy"]:.3g}; original aggregate and paired tables match within declared floating-point tolerances ([verification](verification.json)). The independent reviewer previously replayed four representative fitted systems exactly; this reporting pass hashes those model artifacts but does not deserialize or replay them again.

Summed factory fitting time is {facts["factory_fit_hours"]:.3f} hours. This includes inner selection, final fitting, prediction and in-memory replay, but excludes extraction, preparation, queue interruptions and reporting; it is not elapsed wall time or deployment cost. Actual factory numerical threads are capped at two. Historical reports/models/code and full-matrix plots are untouched. Historical partial-status wording is superseded by the completed artifacts, not silently edited. The assessment copies below are byte-identical snapshots, not rewritten independent conclusions.

## Evidence and reproduction

- [Machine-readable verification](verification.json), [source-copy hashes](source_manifest.json), [generated-file hashes](artifact_manifest.json)
- [Recomputed aggregates](aggregates.csv), [paired effects](paired.csv), [potency diagnostics](potency.csv), [all task prediction tails](prediction_tails.csv)
- [Report Markdown](report.md); [report generator](closeout.py)

{evidence}

Run from the PXR repository root using the existing runtime (reporting only):

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src:. \\
  python \\
  experiments/20260906_low_data_followup/code/closeout.py build
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src:. \\
  python -m pytest \\
  tests/test_followup_closeout.py -q -o addopts=''
python \\
  experiments/20260906_low_data_followup/code/closeout.py verify \\
  --url http://192.168.6.154:8890/low-data-assessment/
```

The LAN page is served by the existing site service on port 8890. No follow-up status links are published until actual report targets exist; completion here refers only to the first study, not the new lanes.
"""


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value and not value.startswith("#"):
                self.links.add(value)


def verify_site(output: Path, url: str | None = None) -> dict:
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        require(sha256(output / name) == digest, f"generated artifact mismatch: {name}")
    source_manifest = json.loads((output / "source_manifest.json").read_text())
    for entry in source_manifest["files"].values():
        require(
            sha256(output / entry["copy"]) == entry["sha256"],
            "source copy hash mismatch",
        )
    parser = Links()
    parser.feed((output / "index.html").read_text())
    targets = {
        "index.html",
        "artifact_manifest.json",
        "source_manifest.json",
        *manifest["files"],
    }
    for link in parser.links:
        parsed = urlparse(link)
        require(
            not parsed.scheme and not parsed.netloc,
            f"unexpected external resource: {link}",
        )
        relative = unquote(parsed.path)
        target = (output / relative).resolve()
        require(
            target.is_relative_to(output.resolve()) and target.is_file(),
            f"broken link: {link}",
        )
        targets.add(relative)
    # Include every immutable source, not only directly visible links.
    targets.update(entry["copy"] for entry in source_manifest["files"].values())
    fetched = []
    if url:
        require(url.endswith("/"), "verification URL must end in /")
        opener = build_opener(ProxyHandler({}))
        for name in sorted(targets):
            target_url = urljoin(url, "" if name == "index.html" else name)
            with opener.open(target_url, timeout=60) as response:
                body = response.read()
                require(response.status == 200, f"HTTP failure: {target_url}")
                require(
                    hashlib.sha256(body).hexdigest() == sha256(output / name),
                    f"served bytes mismatch: {target_url}",
                )
                fetched.append(
                    dict(
                        url=target_url,
                        status=response.status,
                        bytes=len(body),
                        sha256=hashlib.sha256(body).hexdigest(),
                    )
                )
    return {
        "verified_at": datetime.now(UTC).isoformat(),
        "url": url,
        "html_resource_links": len(parser.links),
        "verified_files": len(targets),
        "http_resources_verified": len(fetched),
        "resources": fetched,
    }


CSS = """body{margin:0;background:#f7f7f3;color:#182529;font:17px/1.65 system-ui,sans-serif}
main{max-width:1080px;margin:auto;padding:40px 28px 80px}h1{font-size:2.4rem;line-height:1.15;max-width:800px}
h2{margin-top:2.7rem;border-top:1px solid #c9d1d1;padding-top:1.2rem;line-height:1.25}
h3{margin-top:2rem}a{color:#075868;text-underline-offset:.18em}a:focus{outline:3px solid #e8ae41}
table{border-collapse:collapse;width:100%;font-size:.87rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;border-bottom:1px solid #cbd4d4;padding:9px;vertical-align:top}th{background:#e5eeec}
.table-scroll{overflow-x:auto;margin:24px 0}pre{overflow:auto;padding:18px;background:#e7eeec;font-size:.85rem}
code{font-size:.87em}p,li{max-width:960px}nav{font-size:.85rem}nav a{margin-right:18px}
@media(max-width:640px){main{padding:24px 16px}h1{font-size:1.9rem}body{font-size:16px}}
"""


def build(root: Path, output: Path, closeout_path: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _build(root, output, closeout_path)


def _build(root: Path, output: Path, closeout_path: Path | None) -> dict:
    run = root / RUN
    receipt = json.loads((run / "independent_assessment_verification.json").read_text())
    for relative, expected in receipt["source_hashes"].items():
        require(
            sha256(root / relative) == expected, f"changed assessed source: {relative}"
        )
    bindings = json.loads((run / "run_bindings.json").read_text())
    for group in ["sources", "inputs"]:
        for path, expected in bindings[group].items():
            require(sha256(root / path) == expected, f"changed bound {group}: {path}")
    protocol = json.loads((run / "protocol.json").read_text())
    tasks = read_csv(root / OLD_SITE / "task_metrics.csv")
    facts = validate_matrix(
        tasks, json.loads((run / "task_manifest.json").read_text()), protocol
    )
    require(
        set(p.name for p in (run / "tasks").iterdir() if p.is_dir())
        == set(tasks.task_id),
        "unexpected or missing task directories",
    )
    agg, differences = aggregate(tasks), paired(tasks)
    facts["max_assessment_aggregate_discrepancy"] = compare_tables(
        agg, read_csv(run / "independent_assessment_aggregates.csv"), KEYS
    )
    facts["max_assessment_paired_discrepancy"] = compare_tables(
        differences,
        read_csv(run / "independent_assessment_paired.csv"),
        ["split", "n_train", "method_a", "method_b"],
    )
    pooled = (
        tasks.groupby(KEYS)
        .agg(
            weighted_error_sum=("weighted_error_sum", "sum"),
            weight_sum=("weight_sum", "sum"),
            error_sum=("error_sum", "sum"),
            bias_sum=("bias_sum", "sum"),
            n=("n_test", "sum"),
        )
        .reset_index()
    )
    pooled["weighted_mae"] = pooled.weighted_error_sum / pooled.weight_sum
    pooled["mae"] = pooled.error_sum / pooled.n
    pooled["bias"] = pooled.bias_sum / pooled.n
    facts["max_original_pooled_discrepancy"] = compare_tables(
        pooled,
        read_csv(root / OLD_SITE / "metrics.csv")[
            KEYS + ["weighted_mae", "mae", "bias"]
        ],
        KEYS,
    )
    tails, potency, checks = prediction_diagnostics(root, tasks)
    facts.update(checks)
    require(facts["unique_development_compounds"] == 3344, "wrong development universe")
    audit = json.loads((run / "prepared/split_audit.json").read_text())
    nn = read_csv(run / "prepared/outer_nearest_similarity.csv")
    nn = nn.loc[nn.split.eq("chemical_cluster"), "nearest_outer_train_similarity"]
    facts.update(
        n_groups=audit["n_groups"],
        singleton_compounds=audit["singleton_compounds"],
        largest_group=audit["largest_group"],
        singleton_percent=100 * audit["singleton_compound_fraction"],
        cluster_nn_median=float(nn.median()),
        cluster_nn_q95=float(nn.quantile(0.95)),
        cluster_nn_max=float(nn.max()),
        cluster_nn_ge035_percent=float(nn.ge(0.35).mean() * 100),
        bound_sources_checked=len(bindings["sources"]),
        bound_inputs_checked=len(bindings["inputs"]),
        factory_fit_hours=float(tasks.fit_seconds.sum() / 3600),
        methods=protocol["methods"],
        generated_at=datetime.now(UTC).isoformat(),
        no_model_fits=True,
        aggregation="Arithmetic mean of 50 paired task scores; pooled weighted loss retained separately; no independent-row inference.",
    )
    sources = {}
    inventory = {
        p.name: p for p in sorted(run.glob("independent_assessment*")) if p.is_file()
    }
    for name in [
        "task_metrics.csv",
        "metrics.csv",
        "summary.json",
        "interval_replay.json",
    ]:
        inventory[name] = root / OLD_SITE / name
    for name in ["protocol.json", "run_bindings.json", "task_manifest.json"]:
        inventory[name] = run / name
    for name in [
        "split_audit.json",
        "outer_nearest_similarity.csv",
        "preparation_manifest.json",
    ]:
        inventory[name] = run / "prepared" / name
    for name in [
        "README.md",
        "MODEL_NOTES.md",
        "SPLIT_SENSITIVITY.md",
        "ADAPTER_FEASIBILITY.md",
        "REPORTING_NOTES.md",
        "IMPLEMENTATION_PLAN.md",
    ]:
        inventory[name] = root / "experiments/20260905_low_data_adaptation" / name
    for name in bindings["sources"]:
        inventory["bound/" + name] = root / name
    for name, path in inventory.items():
        copy = "sources/" + name
        sources[name] = dict(
            source=str(path.relative_to(root)),
            copy=copy,
            **immutable_copy(path, output / copy),
        )
    write_json(
        output / "source_manifest.json",
        {
            "files": sources,
            "large_original_inputs": {
                str((root / name).relative_to(root)): value
                for name, value in bindings["inputs"].items()
            },
            "note": "Evidence and bound source code copied byte-for-byte. Large input/model/task trees stay at canonical original paths; hashes freshly checked, no duplicate model fits.",
        },
    )
    for name, frame in [
        ("aggregates.csv", agg),
        ("paired.csv", differences),
        ("prediction_tails.csv", tails),
        ("potency.csv", potency),
    ]:
        atomic_write(output / name, frame.to_csv(index=False))
    created = facts["generated_at"]
    if (output / "verification.json").exists():
        created = json.loads((output / "verification.json").read_text())["created_at"]
    facts["created_at"] = created
    write_json(output / "verification.json", facts)
    text = report_text(agg, differences, facts, sources, created)
    atomic_write(output / "report.md", text)
    if closeout_path:
        # Keep canonical experiment-document links valid from its own directory.
        document = re.sub(
            r"\]\(([^)]+)\)", r"](../../site/low-data-assessment/\1)", text
        )
        atomic_write(closeout_path, document)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    # A compact landing view, with full technical detail available without JS.
    for heading in [
        "What was tested and where labels entered",
        "Evidence and reproduction",
    ]:
        pattern = rf"(<h2[^>]*>{re.escape(heading)}</h2>)(.*?)(?=<h2|\Z)"
        body = re.sub(
            pattern,
            r"<details><summary>\1</summary>\2</details>",
            body,
            flags=re.DOTALL,
        )
    body = body.replace("<table>", '<div class="table-scroll"><table>').replace(
        "</table>", "</table></div>"
    )
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nesso PXR | Completed low-data assessment</title><link rel="stylesheet" href="style.css"></head><body><main><nav aria-label="Report resources"><a href="report.md">Report</a><a href="aggregates.csv">Data</a><a href="sources/independent_assessment.md">Independent assessment</a><a href="verification.json">Verification</a></nav>'
        + body
        + "</main></body></html>\n"
    )
    atomic_write(output / "index.html", html)
    atomic_write(output / "style.css", CSS)
    atomic_write(output / "closeout.py", Path(__file__).read_bytes())
    names = [
        "index.html",
        "style.css",
        "closeout.py",
        "report.md",
        "verification.json",
        "source_manifest.json",
        "aggregates.csv",
        "paired.csv",
        "potency.csv",
        "prediction_tails.csv",
    ]
    write_json(
        output / "artifact_manifest.json",
        {"files": {name: sha256(output / name) for name in names}},
    )
    verify_site(output)
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--url", help="Exact LAN page URL; verify bytes of page and every resource"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "build":
        result = build(root, root / SITE, root / EXPERIMENT / "CLOSEOUT.md")
    else:
        result = verify_site(root / SITE, args.url)
        if args.url:
            write_json(root / SITE / "http_verification.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
