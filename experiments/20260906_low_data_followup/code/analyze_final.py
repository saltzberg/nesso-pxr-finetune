"""Retrospective, read-only saved-prediction analysis. No model fitting/imports.

Run with OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHONPATH=src:. python
experiments/20260906_low_data_followup/code/analyze_final.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "2"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/experiments/low_data_followup_20260906"
OLD = ROOT / "artifacts/experiments/low_data_20260905_cut035_run2"
OUT = RUN / "final_analysis"
LANES = {
    "readouts": RUN / "readouts",
    "baselines": RUN / "baselines",
    "strict": RUN / "strict_split/panel",
    "original": OLD,
}
EXPECTED = {"readouts": 3600, "baselines": 1800, "strict": 2700}
KEY = ["split", "n_train", "method"]
CELL = ["split", "n_train", "outer_fold", "draw"]
BINS = [0.0, 0.2, 0.35, 0.5, 0.7, 1.00000001]
BIN_LABELS = ["[0,0.2)", "[0.2,0.35)", "[0.35,0.5)", "[0.5,0.7)", "[0.7,1]"]
VERSION = 1


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_csv(path):
    return pd.read_csv(
        path,
        float_precision="round_trip",
        keep_default_na=False,
        na_values=[""],
        dtype={"record_id": str},
    )


def atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def save_csv(path, frame):
    atomic(path, frame.to_csv(index=False))
    restored = read_csv(path)
    assert list(restored.columns) == list(frame.columns) and len(restored) == len(frame)
    for c in frame.select_dtypes(include="number").columns:
        np.testing.assert_allclose(
            restored[c].to_numpy(float),
            frame[c].to_numpy(float),
            rtol=0,
            atol=0,
            equal_nan=True,
        )


def kish(w):
    w = np.asarray(w, float)
    return (
        float(w.sum() ** 2 / np.square(w).sum())
        if len(w) and np.square(w).sum()
        else 0.0
    )


def regression(y, p):
    """Unweighted descriptive moments, not an applied calibration model."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    yc, pc = y - y.mean(), p - p.mean()
    vy, vp, cov = np.mean(yc * yc), np.mean(pc * pc), np.mean(yc * pc)
    sy = cov / vy if vy > 0 else np.nan
    sp = cov / vp if vp > 0 else np.nan
    yiqr, piqr = (
        np.diff(np.quantile(y, [0.25, 0.75]))[0],
        np.diff(np.quantile(p, [0.25, 0.75]))[0],
    )
    return dict(
        pred_on_observed_slope=sy,
        pred_on_observed_intercept=p.mean() - sy * y.mean(),
        observed_on_pred_slope=sp,
        observed_on_pred_intercept=y.mean() - sp * p.mean(),
        sd_ratio=np.sqrt(vp / vy) if vy > 0 else np.nan,
        iqr_ratio=piqr / yiqr if yiqr else np.nan,
        observed_mean=y.mean(),
        predicted_mean=p.mean(),
        observed_sd=np.sqrt(vy),
        predicted_sd=np.sqrt(vp),
        predicted_min=p.min(),
        predicted_max=p.max(),
    )


def interval_stats(frame, metrics, level):
    y, p = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    lower, upper = (
        frame[f"lower_{level}"].to_numpy(),
        frame[f"upper_{level}"].to_numpy(),
    )
    # Runner's predicate is abs(y-p)<=radius, NOT rounded endpoints.
    width = metrics.get(f"width_{level}")
    if width is None:
        assert not metrics[f"interval_{level}_finite"]
        width = np.inf
    radius = width / 2
    np.testing.assert_array_equal(lower, p - radius)
    np.testing.assert_array_equal(upper, p + radius)
    covered = np.abs(y - p) <= radius
    endpoint = (y >= lower) & (y <= upper)
    np.testing.assert_allclose(
        covered.mean(), metrics[f"coverage_{level}"], atol=1e-14, rtol=0
    )
    return {
        f"coverage_{level}": covered.mean(),
        f"width_{level}": width,
        f"unbounded_rows_{level}": int(np.isinf(upper - lower).sum()),
        f"endpoint_boundary_disagreements_{level}": int(
            np.count_nonzero(covered != endpoint)
        ),
    }


def summarize_predictions(frame):
    """Pooled draw-observations: repeated identities explicitly collapse for ESS."""
    if not len(frame):
        return dict(
            prediction_rows=0,
            unique_compounds=0,
            chemical_groups=0,
            effective_n_identity=0.0,
            effective_n_cluster=0.0,
            sparse=True,
            weighted_mae=np.nan,
            mae=np.nan,
            bias=np.nan,
            weighted_bias=np.nan,
            weight_sum=0.0,
        )
    w = frame.weight.to_numpy()
    err = frame.y_pred.to_numpy() - frame.y_true.to_numpy()
    iw = frame.groupby("record_id").weight.sum()
    gw = frame.groupby("chemical_group").weight.sum()
    return dict(
        prediction_rows=len(frame),
        unique_compounds=len(iw),
        chemical_groups=len(gw),
        effective_n_identity=kish(iw),
        effective_n_cluster=kish(gw),
        sparse=bool(len(iw) < 30 or kish(iw) < 20),
        weighted_mae=np.dot(w, np.abs(err)) / w.sum(),
        mae=np.abs(err).mean(),
        bias=err.mean(),
        weighted_bias=np.dot(w, err) / w.sum(),
        weight_sum=w.sum(),
    )


def conditional_bootstrap(a, b, repeats=1000, seed=20260906):
    """Poisson cluster multiplier bootstrap; fixed fits/draws, equal-fold WMAE.

    One multiplier per chemical group shared across identities/draws/methods.
    Recompute within-fold reliability denominators then average folds. No
    fold/draw independence assumption; no training-population uncertainty.
    """
    cols = ["record_id", "outer_fold", "chemical_group"]
    x = a.merge(b, on=cols, suffixes=("_a", "_b"), validate="one_to_one")
    assert len(x) == len(a) == len(b)
    np.testing.assert_array_equal(x.weight_a, x.weight_b)
    np.testing.assert_array_equal(x.draw_count_a, x.draw_count_b)
    groups, gi = np.unique(x.chemical_group.to_numpy(), return_inverse=True)
    folds = sorted(x.outer_fold.unique())
    num = np.zeros((len(groups), len(folds)))
    den = num.copy()
    delta = x.abs_error_a.to_numpy() - x.abs_error_b.to_numpy()
    if "y_true_a" in x:
        np.testing.assert_array_equal(x.y_true_a, x.y_true_b)
    for j, fold in enumerate(folds):
        sel = x.outer_fold.to_numpy() == fold
        np.add.at(num[:, j], gi[sel], (delta * x.weight_a.to_numpy())[sel])
        np.add.at(den[:, j], gi[sel], x.weight_a.to_numpy()[sel])
    point = np.mean(num.sum(axis=0) / den.sum(axis=0))
    rng = np.random.default_rng(seed)
    sampled = []
    for start in range(0, repeats, 100):
        mult = rng.poisson(1, size=(min(100, repeats - start), len(groups)))
        denominators = mult @ den
        assert (denominators > 0).all()
        sampled.extend(((mult @ num) / denominators).mean(axis=1).tolist())
    lo, hi = np.quantile(sampled, [0.025, 0.975])
    return dict(
        conditional_delta=point,
        conditional_low95=lo,
        conditional_high95=hi,
        bootstrap_replicates=repeats,
        resampling_groups=len(groups),
        unique_compounds=len(x),
        effective_n_identity=kish(x.weight_a),
        effective_n_cluster=kish(den.sum(axis=1)),
    )


def inventory():
    status_path = RUN / "completion_monitor/status.json"
    status = json.loads(status_path.read_text())
    assert status["all_output_sets_verified"]
    manifests, report = {}, {}
    for lane, base in LANES.items():
        tasks = json.loads((base / "task_manifest.json").read_text())
        assert len({t["task_id"] for t in tasks}) == len(tasks)
        if lane == "original":
            tasks = [
                t
                for t in tasks
                if t["split"] == "chemical_cluster"
                and t["method"]
                in ["repr_ridge", "repr_mlp", "head_pretrained", "head_random"]
            ]
        else:
            assert len(tasks) == EXPECTED[lane] == status["lanes"][lane]["complete"]
            assert (
                status["lanes"][lane]["failed"] == status["lanes"][lane]["invalid"] == 0
            )
            actual = {p.parent.name for p in (base / "tasks").glob("*/complete.json")}
            assert actual == {t["task_id"] for t in tasks}
        manifests[lane] = tasks
        report[lane] = dict(
            expected=len(tasks),
            manifest_sha256=digest(base / "task_manifest.json"),
            run_bindings_sha256=digest(base / "run_bindings.json"),
            retained_failure_files=[
                str(p.relative_to(ROOT))
                for p in (base / "tasks").glob("*/failure.json")
            ],
        )
    atomic(OUT / "inventory.json", json.dumps(report, indent=2))
    return manifests, report


def strict_novelty():
    """Use frozen Morgan bits, actual fit rows; no labels/features fitted."""
    from rdkit import DataStructs

    cache = OUT / "cache/strict_novelty.csv"
    inputs = [
        OLD / "prepared/features.npz",
        RUN / "strict_split/subsets.csv",
        RUN / "strict_split/assignments.csv",
    ]
    signature = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    sigfile = cache.with_suffix(".json")
    if (
        cache.exists()
        and sigfile.exists()
        and json.loads(sigfile.read_text()) == signature
    ):
        return read_csv(cache)
    with np.load(inputs[0]) as z:
        bits = z["morgan"]
    fps = []
    for row in bits:
        fp = DataStructs.ExplicitBitVect(bits.shape[1])
        fp.SetBitsFromList(np.flatnonzero(row).tolist())
        fps.append(fp)
    sim = np.asarray(
        [DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps], dtype=np.float64
    )
    subsets, assignments = read_csv(inputs[1]), read_csv(inputs[2])
    rows = []
    for (fold, draw, budget), s in subsets.groupby(["outer_fold", "draw", "n_train"]):
        test = assignments.loc[assignments.outer_fold.eq(fold), "row_index"].to_numpy(
            int
        )
        fit = s.loc[s.role.eq("fit"), "row_index"].to_numpy(int)
        nearest = sim[np.ix_(test, fit)].max(axis=1)
        rows.extend(
            dict(
                outer_fold=fold,
                draw=draw,
                n_train=budget,
                row_index=int(i),
                nearest_similarity=float(v),
            )
            for i, v in zip(test, nearest, strict=False)
        )
    f = pd.DataFrame(rows)
    save_csv(cache, f)
    atomic(sigfile, json.dumps(signature))
    return f


def extract_chunk(lane, key, tasks, inventory_row, novelty, groups):
    split, budget, method = key
    name = f"{lane}__{split}__n{budget}__{method}"
    directory = OUT / "cache" / name
    marker = directory / "done.json"
    # Each chunk has a complete source binding and independently round-tripped CSVs.
    sources = {}
    for task in tasks:
        d = LANES[lane] / "tasks" / task["task_id"]
        complete = json.loads((d / "complete.json").read_text())
        if lane == "original":
            assert complete["task_id"] == task["task_id"]
            original_metrics = json.loads((d / "metrics.json").read_text())
            assert all(original_metrics[k] == v for k, v in task.items())
        else:
            assert complete["task"] == task
            bind = complete.get("binding_sha256", complete.get("run_bindings_sha256"))
            assert bind == inventory_row["run_bindings_sha256"]
        for file in ("predictions.csv", "metrics.json"):
            h = digest(d / file)
            assert h == complete["output_sha256"][file], (d, file)
            sources[str((d / file).relative_to(ROOT))] = h
        sources[str((d / "complete.json").relative_to(ROOT))] = digest(
            d / "complete.json"
        )
    signature = dict(version=VERSION, analysis_sha256=digest(__file__), sources=sources)
    if marker.exists():
        saved = json.loads(marker.read_text())
        output_hashes = saved.pop("output_hashes", {})
        if saved == signature and output_hashes:
            assert all(digest(directory / p) == h for p, h in output_hashes.items()), (
                "corrupt cached analysis chunk"
            )
            return directory
    frames, metrics_rows = [], []
    for task in tasks:
        d = LANES[lane] / "tasks" / task["task_id"]
        f = read_csv(d / "predictions.csv")
        m = json.loads((d / "metrics.json").read_text())
        assert not f.record_id.duplicated().any() and len(f) == m["n_test"]
        for c in CELL + ["method"]:
            assert f[c].eq(task[c]).all()
        assert (
            m["n_fit"] + m["n_calibration"] == task["n_train"] or task["n_train"] == -1
        )
        if "chemical_group" not in f:
            f["chemical_group"] = f.row_index.map(groups)
        assert f.chemical_group.notna().all()
        if lane == "strict":
            ns = novelty[
                (novelty.outer_fold == task["outer_fold"])
                & (novelty.draw == task["draw"])
                & (novelty.n_train == budget)
            ]
            f = f.merge(
                ns[["row_index", "nearest_similarity"]],
                on="row_index",
                validate="one_to_one",
            )
        assert np.isfinite(
            f[["y_true", "y_pred", "weight", "nearest_similarity"]].to_numpy()
        ).all()
        np.testing.assert_allclose(
            f.weight, 1 / np.maximum(f.assay_se, 0.1), rtol=1e-14, atol=0
        )
        y, p, w = f.y_true.to_numpy(), f.y_pred.to_numpy(), f.weight.to_numpy()
        err = p - y
        values = dict(
            weighted_mae=np.dot(w, abs(err)) / w.sum(),
            mae=abs(err).mean(),
            bias=err.mean(),
            spearman=spearmanr(y, p).statistic
            if np.ptp(p) > 0 and np.ptp(y) > 0
            else np.nan,
        )
        for c in values:
            expected = np.nan if m.get(c) is None else m[c]
            np.testing.assert_allclose(
                values[c],
                expected,
                rtol=2e-12,
                atol=2e-12,
                equal_nan=True,
                err_msg=f"{d}: {c}",
            )
        row = dict(
            lane=lane,
            **task,
            **values,
            **regression(y, p),
            n_test=len(f),
            n_fit=m["n_fit"],
            n_calibration=m["n_calibration"],
            fit_seconds=m["fit_seconds"],
            n_parameters=m.get("n_parameters"),
            original_weighted_mae=m["weighted_mae"],
            original_mae=m["mae"],
            original_spearman=m.get("spearman"),
            max_absolute_error=abs(err).max(),
            ensemble_spread_error_spearman=spearmanr(f.ensemble_std, abs(err)).statistic
            if np.ptp(f.ensemble_std) > 0
            else np.nan,
        )
        for level in (80, 90):
            row.update(interval_stats(f, m, level))
        metrics_rows.append(row)
        frames.append(f)
    allf = pd.concat(frames, ignore_index=True)
    summary = []
    masks = [
        ("all", np.ones(len(allf), bool)),
        ("potency_lt4", allf.y_true < 4),
        ("potency_ge4", allf.y_true >= 4),
        ("potency_ge6_descriptive_tail", allf.y_true >= 6),
    ]
    binned = pd.cut(allf.nearest_similarity, BINS, right=False, labels=BIN_LABELS)
    masks += [(f"novelty_{label}", binned == label) for label in BIN_LABELS]
    total_weight = allf.weight.sum()
    for label, mask in masks:
        stats = summarize_predictions(allf.loc[mask])
        stats["weight_share"] = stats["weight_sum"] / total_weight
        summary.append(
            dict(
                lane=lane,
                split=split,
                n_train=budget,
                method=method,
                stratum=label,
                **stats,
            )
        )
    allf["abs_error"] = abs(allf.y_pred - allf.y_true)
    assert (
        allf.groupby("record_id")[["weight", "y_true", "chemical_group", "outer_fold"]]
        .nunique()
        .eq(1)
        .all()
        .all()
    )
    molecule = (
        allf.groupby(["record_id", "outer_fold", "chemical_group"])
        .agg(
            weight=("weight", "first"),
            y_true=("y_true", "first"),
            abs_error=("abs_error", "mean"),
            draw_count=("draw", "nunique"),
        )
        .reset_index()
    )
    assert molecule.draw_count.eq(10).all()
    save_csv(directory / "tasks.csv", pd.DataFrame(metrics_rows))
    save_csv(directory / "strata.csv", pd.DataFrame(summary))
    save_csv(directory / "molecules.csv", molecule)
    signature["output_hashes"] = {
        p: digest(directory / p) for p in ["tasks.csv", "strata.csv", "molecules.csv"]
    }
    atomic(marker, json.dumps(signature, sort_keys=True))
    return directory


def make_comparisons(task_frame, chunks, repeats):
    definitions = []
    for a, b in [
        ("descriptor_lightgbm_rdkit_mordred", "head_pretrained"),
        ("repr_ridge_stable", "head_pretrained"),
        ("head_pretrained", "head_random"),
    ]:
        definitions.append(("strict", a, "strict", b, "strict_chemical", "primary"))
    for la, a, lb, b in [
        ("readouts", "repr_ridge_stable", "original", "repr_ridge"),
        ("readouts", "repr_ridge_unscaled", "original", "repr_ridge"),
        ("readouts", "repr_ridge_stable", "readouts", "repr_ridge_unscaled"),
        ("readouts", "repr_mlp_stable", "readouts", "repr_mlp_unscaled"),
        ("readouts", "repr_mlp_stable", "original", "repr_mlp"),
        ("readouts", "repr_mlp_unscaled", "original", "repr_mlp"),
        ("readouts", "repr_mlp_stable_centered", "readouts", "repr_mlp_stable"),
        (
            "readouts",
            "repr_mlp_stable_centered_regularized",
            "readouts",
            "repr_mlp_stable_centered",
        ),
        ("readouts", "repr_ridge_stable", "original", "head_pretrained"),
    ]:
        definitions.append((la, a, lb, b, "chemical_cluster", "controlled_readout"))
    paired_rows, fold_rows, summary = [], [], []
    for la, a, lb, b, split, kind in definitions:
        aa = task_frame[
            (task_frame.lane == la)
            & (task_frame.method == a)
            & (task_frame.split == split)
        ]
        bb = task_frame[
            (task_frame.lane == lb)
            & (task_frame.method == b)
            & (task_frame.split == split)
        ]
        joined = aa.merge(bb, on=CELL, suffixes=("_a", "_b"), validate="one_to_one")
        assert len(joined) == len(aa) == len(bb) == 300
        for budget, sub in joined.groupby("n_train"):
            meta = dict(
                split=split,
                n_train=int(budget),
                lane_a=la,
                method_a=a,
                lane_b=lb,
                method_b=b,
                comparison_type=kind,
            )
            for metric in ["weighted_mae", "mae", "spearman"]:
                sub[f"delta_{metric}"] = sub[f"{metric}_a"] - sub[f"{metric}_b"]
            for _, r in sub.iterrows():
                paired_rows.append(
                    dict(
                        **meta,
                        outer_fold=int(r.outer_fold),
                        draw=int(r.draw),
                        **{
                            f"delta_{m}": r[f"delta_{m}"]
                            for m in ["weighted_mae", "mae", "spearman"]
                        },
                    )
                )
            fold = sub.groupby("outer_fold")[
                [f"delta_{m}" for m in ["weighted_mae", "mae", "spearman"]]
            ].mean()
            for f, r in fold.iterrows():
                fold_rows.append(dict(**meta, outer_fold=int(f), **r.to_dict()))
            ma = read_csv(chunks[(la, split, budget, a)] / "molecules.csv")
            mb = read_csv(chunks[(lb, split, budget, b)] / "molecules.csv")
            ci = conditional_bootstrap(ma, mb, repeats=repeats)
            np.testing.assert_allclose(
                ci["conditional_delta"],
                sub.delta_weighted_mae.mean(),
                atol=2e-12,
                rtol=2e-12,
            )
            summary.append(
                dict(
                    **meta,
                    paired_tasks=len(sub),
                    folds=len(fold),
                    mean_delta_weighted_mae=sub.delta_weighted_mae.mean(),
                    mean_delta_mae=sub.delta_mae.mean(),
                    mean_delta_spearman=sub.delta_spearman.mean(),
                    negative_task_count=int((sub.delta_weighted_mae < 0).sum()),
                    negative_fold_count=int((fold.delta_weighted_mae < 0).sum()),
                    fold_delta_min=fold.delta_weighted_mae.min(),
                    fold_delta_max=fold.delta_weighted_mae.max(),
                    **ci,
                )
            )
    save_csv(OUT / "paired_tasks.csv", pd.DataFrame(paired_rows))
    save_csv(OUT / "paired_folds.csv", pd.DataFrame(fold_rows))
    result = pd.DataFrame(summary)
    save_csv(OUT / "paired_effects.csv", result)
    return result


def findings(inv, tasks, means, effects, strata):
    lines = [
        "# Retrospective final saved-prediction analysis",
        "",
        "All choices and intervals here are retrospective. No new model fits; no formal promotion, significance test, multiplicity-adjusted claim, or untouched-validation claim.",
        "",
        "## Accounting and estimand",
        f"- Reconciled readouts {inv['readouts']['expected']}, baselines {inv['baselines']['expected']}, strict {inv['strict']['expected']} task identities/bindings. Original chemical-cluster controls: {inv['original']['expected']} tasks.",
        "- Prediction and metric bytes were rehashed and all MAE, weighted MAE, bias, Spearman and interval scores recomputed. Completion-monitor verification binds all model/output sets; this analysis does not re-deserialize/refit models or rehash their large state files.",
        "- Primary is equal-weight mean over 5 folds × 10 paired draw tasks, NOT 50 independent replications. N includes fit, tuning and calibration; -1 is eligible-full, not a fixed N or optimal ceiling.",
        "- Conditional 95% intervals: 1,000 seeded Poisson(1) chemical-cluster multipliers, shared across both methods and every repeated identity/draw; averaged within-compound absolute errors over the fixed ten fits, recomputed weighted denominator per fold, equal-fold average. These are empirical test-cohort reweighting sensitivities conditional on all trained models, selected subsets, labels and this inspected cohort. They exclude training-set/selection/initialization population uncertainty and do not establish prospective coverage. Fold effects are descriptive; overlapping training folds are not independent.",
        "",
        "## Primary strict comparisons",
        "Delta = method A minus B weighted MAE (negative favors A); intervals are conditional as above.",
        "| A vs B | N | delta | conditional 95% | negative folds / 5 | raw MAE delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in effects[effects.comparison_type == "primary"].iterrows():
        lines.append(
            f"| {r.method_a} vs {r.method_b} | {int(r.n_train)} | {r.mean_delta_weighted_mae:.6f} | [{r.conditional_low95:.6f}, {r.conditional_high95:.6f}] | {int(r.negative_fold_count)} | {r.mean_delta_mae:.6f} |"
        )
    lines += [
        "",
        "On strict folds, the descriptor comparator has lower primary weighted MAE at N250/N500/full, while the pretrained head leads at N25/N50/N100. N100 reverses under raw MAE. Stable ridge versus the head at N500 spans zero under the conditional reweighting interval; there is no uniform pretrained-head initialization advantage.",
        "",
        "## Controlled original chemical-cluster readouts",
        "Original pathological scores remain unchanged. Stable-versus-unscaled within-follow-up comparisons control the candidate grid. Original-versus-follow-up ridge also expands alpha from [1,100,10000] to [1,100,10000,1000000], so it does not isolate scaling alone. Centering and added regularization are separately paired, not post-hoc prediction repairs.",
        "| A vs B | N | weighted MAE delta |",
        "|---|---:|---:|",
    ]
    for _, r in effects[
        (effects.comparison_type == "controlled_readout")
        & effects.n_train.isin([25, 100, 500])
    ].iterrows():
        lines.append(
            f"| {r.method_a} vs {r.method_b} | {int(r.n_train)} | {r.mean_delta_weighted_mae:.6f} |"
        )
    lines += [
        "",
        "## Potency, calibration, novelty, intervals and cost",
        "The >=6 tail is an overlapping, explicitly descriptive sensitivity; <4 and >=4 partition the cohort. Low potency is not itself evidence of noise/censoring. Pooled stratum scores average errors of fixed fits, not errors of an ensemble averaged over draws.",
        "| strict method | N | <4 WMAE / bias | >=4 WMAE / bias | >=6 count / WMAE / bias | predicted-on-observed slope / SD ratio | 90% coverage / width | mean task seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    chosen = [
        "descriptor_lightgbm_rdkit_mordred",
        "head_pretrained",
        "repr_ridge_stable",
    ]
    for _, r in means[
        (means.lane == "strict")
        & means.method.isin(chosen)
        & means.n_train.isin([25, 100, 500])
    ].iterrows():
        s = strata[
            (strata.lane == "strict")
            & (strata.n_train == r.n_train)
            & (strata.method == r.method)
        ].set_index("stratum")
        low, high, tail = [
            s.loc[k]
            for k in ["potency_lt4", "potency_ge4", "potency_ge6_descriptive_tail"]
        ]
        lines.append(
            f"| {r.method} | {int(r.n_train)} | {low.weighted_mae:.6f} / {low.bias:.6f} | {high.weighted_mae:.6f} / {high.bias:.6f} | {int(tail.unique_compounds)} / {tail.weighted_mae:.6f} / {tail.bias:.6f} | {r.pred_on_observed_slope:.6f} / {r.sd_ratio:.6f} | {r.coverage_90:.6f} / {r.width_90:.6f} | {r.fit_seconds:.6f} |"
        )
    counts = tasks.groupby("lane").agg(
        tasks=("task_id", "size"), seconds=("fit_seconds", "sum")
    )
    lines += [
        "",
        "- Better global error does not imply improved high-potency sensitivity: at strict N500 the descriptor model has lower overall weighted MAE but larger >=6 underprediction than the pretrained head (pooled tail biases -1.270879 versus -1.230562). The tail comprises 56 unique compounds / 560 dependent predictions per method and budget; pooled stratum weighting differs from an equal-task tail mean.",
        "- Fixed nearest-fit Morgan Tanimoto bins: [0,.2), [.2,.35), [.35,.5), [.5,.7), [.7,1]. Strict nearest similarities were recomputed from frozen Morgan bits and actual fit-only rows; calibration compounds are not fitted neighbors. Empty bins remain present. Sparse means <30 distinct compounds or identity-weight ESS <20. ESS collapses repeated weights per identity (and separately per chemical cluster), never counts repeats as new compounds.",
        "- Calibration slopes are unweighted descriptive covariances: predicted-on-observed diagnoses compression, observed-on-predicted is the inverse-direction calibration diagnostic, neither is applied to predictions. SD and IQR ratios and both intercepts are in task_metrics.csv.",
        f"- Infinite intervals retained: {int(tasks.unbounded_rows_90.sum())} dependent prediction rows have infinite 90% width; these count as covered, not missing. Rounded-endpoint predicates differ from the saved radius predicate on {int(tasks.endpoint_boundary_disagreements_80.sum())} (80%) and {int(tasks.endpoint_boundary_disagreements_90.sum())} (90%) rows. Original radius semantics exactly replayed.",
        "- Costs are saved runner fit_seconds (selection/refit pipeline timing as implemented), not new stopwatch measurements; no total-wall-time, extraction cost, or peak-VRAM inference is made. Native/reference repeats and -1 draws are not independent compute/scientific replicates. costs.csv retains per-cell parameters and summed/mean task timings. One-time cache extraction cost and peak VRAM unavailable in analyzed task metrics.",
        "- Recorded task seconds by lane: "
        + "; ".join(
            f"{lane} {r.seconds:.6f}s ({int(r.tasks)} tasks)"
            for lane, r in counts.iterrows()
        )
        + ".",
        "- Head pretrained/random isolates released-head initialization on the same pretrained frozen representation, not upstream pretraining value. Descriptor comparator is RDKit2D + Mordred2D + Morgan, not the unavailable complete historical feature universe. Strict and original full-pool Ns differ; strict purging is not proof of out-of-target generalization.",
        "- No upstream representation adaptation or second-target binding evidence is supplied by this retrospective matrix. Deployment chemistry and a practical promotion margin remain unspecified.",
        "",
        "Machine-readable task_metrics.csv, task_means.csv, paired_tasks.csv, paired_folds.csv, paired_effects.csv, strata.csv, costs.csv and analysis_manifest.json are authoritative. Cache chunks retain per-identity errors for replay.",
    ]
    atomic(OUT / "findings.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    manifests, inv = inventory()
    novelty = strict_novelty()
    assignments = read_csv(RUN / "strict_split/assignments.csv")
    groups = assignments.set_index("row_index").chemical_group
    assert groups.index.is_unique
    chunks = {}
    for lane, tasks in manifests.items():
        frame = pd.DataFrame(tasks)
        for key, group in frame.groupby(KEY, sort=True):
            d = extract_chunk(
                lane, key, group.to_dict("records"), inv[lane], novelty, groups
            )
            chunks[(lane, *key)] = d
            print(f"chunk {len(chunks)} {d.name}", flush=True)
    tasks = pd.concat(
        [read_csv(p / "tasks.csv") for p in chunks.values()], ignore_index=True
    )
    strata = pd.concat(
        [read_csv(p / "strata.csv") for p in chunks.values()], ignore_index=True
    )
    assert tasks.groupby("lane").size().to_dict() == {
        k: v["expected"] for k, v in inv.items()
    }
    save_csv(OUT / "task_metrics.csv", tasks)
    save_csv(OUT / "strata.csv", strata)
    numeric = [
        c
        for c in tasks.select_dtypes(include="number").columns
        if c not in ["n_train", "outer_fold", "draw"]
    ]
    # numpy means intentionally retain infinity (pandas grouped infinity sums can yield NaN).
    means = (
        tasks.groupby(["lane", *KEY])[numeric]
        .agg(lambda x: np.mean(x.to_numpy()))
        .reset_index()
    )
    save_csv(OUT / "task_means.csv", means)
    cost = (
        tasks.groupby(["lane", *KEY])
        .agg(
            task_count=("task_id", "size"),
            fit_seconds_sum=("fit_seconds", "sum"),
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_median=("fit_seconds", "median"),
            fit_seconds_min=("fit_seconds", "min"),
            fit_seconds_max=("fit_seconds", "max"),
            n_parameters_min=("n_parameters", "min"),
            n_parameters_max=("n_parameters", "max"),
        )
        .reset_index()
    )
    save_csv(OUT / "costs.csv", cost)
    effects = make_comparisons(tasks, chunks, args.bootstrap_replicates)
    # Reconcile original comparator rows to the independent original report, not just task JSON.
    ref = read_csv(ROOT / "site/low-data-verified/task_metrics.csv").set_index(
        "task_id"
    )
    original = tasks[tasks.lane == "original"].set_index("task_id")
    for c in ["weighted_mae", "mae", "spearman"]:
        np.testing.assert_allclose(
            original[c],
            ref.loc[original.index, c],
            rtol=2e-12,
            atol=2e-12,
            equal_nan=True,
        )
    findings(inv, tasks, means, effects, strata)
    source_paths = [
        Path(__file__),
        ROOT / "tests/test_followup_final_analysis.py",
        RUN / "completion_monitor/status.json",
        ROOT / ".hermes/plans/20260905-1954-nesso-low-data-adaptation.md",
        ROOT / ".hermes/plans/20260906-1652-nesso-representation-adaptation-next.md",
        ROOT / "site/low-data-verified/task_metrics.csv",
        OLD / "prepared/inputs.csv",
        OLD / "prepared/features.npz",
        RUN / "strict_split/subsets.csv",
        RUN / "strict_split/assignments.csv",
    ]
    source_paths += [
        b / f
        for b in LANES.values()
        for f in ["task_manifest.json", "run_bindings.json"]
    ]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    for p in chunks.values():
        sources.update(json.loads((p / "done.json").read_text())["sources"])
    output_names = [
        "inventory.json",
        "findings.md",
        "paired_effects.csv",
        "task_means.csv",
        "costs.csv",
        "task_metrics.csv",
        "strata.csv",
        "paired_tasks.csv",
        "paired_folds.csv",
    ]
    outputs = {name: digest(OUT / name) for name in output_names}
    manifest = dict(
        schema_version=VERSION,
        created_at=datetime.now(UTC).isoformat(),
        retrospective=True,
        new_model_fits=0,
        threads=2,
        complete=True,
        lane_counts={k: v["expected"] for k, v in inv.items()},
        analyzed_prediction_rows=int(tasks.n_test.sum()),
        chunks=len(chunks),
        paired_comparisons=len(effects),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=20260906,
        uncertainty="Conditional fixed-model/test-cohort Poisson chemical-cluster reweighting; no independent repeats or training population uncertainty",
        source_hashes=sources,
        output_hashes=outputs,
        elapsed_seconds=time.monotonic() - start,
        limitations=[
            "Retrospective outcome-informed analyses; no promotion margin/multiplicity correction",
            "Large model hashes inherited from hashed completion status; prediction/metric/marker hashes directly recomputed",
            "One-time feature extraction and peak VRAM unavailable in analyzed metrics",
        ],
    )
    atomic(
        OUT / "analysis_manifest.json",
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in manifest.items()
                if k not in ["source_hashes", "output_hashes"]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
