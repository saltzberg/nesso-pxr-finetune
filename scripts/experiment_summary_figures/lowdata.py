#!/usr/bin/env python3
"""Reporting-only figures for pooling and the two low-data matrices.

Run with the review virtualenv; reads saved aggregate tables, never fits models.
Each exported row retains its source path, zero-based source row, and source
metric. Verification reloads the exported CSV and checks every mark against the
source table. Full-pool references are intentionally outside budget curves.
"""
from common import ROOT, OUT, plt, pd, INK, ACCENT, GRAY, publish
import json
import math
from pathlib import Path

OLD = 'artifacts/experiments/low_data_20260905_cut035_run2/'
FOLLOW = 'artifacts/experiments/low_data_followup_20260906/final_analysis/'
POOL = 'reports/transfer_ablation/pair_pool_full/'
DOC = 'experiments/20260906_low_data_followup/'
HEAD = 'head_pretrained'
DESC = 'descriptor_lightgbm_rdkit_mordred'
RIDGE = 'repr_ridge_stable'
SLUGS = ['pair-pooling', 'low-data', 'stable-readouts', 'descriptor-followup', 'strict-followup']
BUDGETS = [25, 50, 100, 250, 500]
WEIGHT = ('Weighted MAE uses 1/max(assay SE, 0.10); lower is better. '
          'Budgets include 80% FIT/inner selection and 20% CAL labels. ')
COHORT = ('Retrospective development cohort of 3,344 compounds; each mean-task point equally averages '
          'five outer folds × ten paired draws (50 tasks), not a pooled weighted score. '
          'Neural predictions average three seeds before scoring; repeated draws are dependent. '
          'No inferential bands are drawn. ')


def load(path):
    d = pd.read_csv(ROOT/path)
    d['source_row'] = d.index
    d['source_file'] = path
    return d


def marks(d, metric, panel, aggregation):
    d = d.copy()
    d['source_metric'] = metric
    d['value'] = d[metric]
    d['panel'] = panel
    d['aggregation'] = aggregation
    return d


def budget_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks(BUDGETS, [str(n) for n in BUDGETS])
    ax.minorticks_off()
    ax.set_xlabel('Acquired PXR labels')
    ax.set_ylabel('Mean-task weighted MAE')
    ax.grid(axis='y', color='#e5e5e5', linewidth=.55)
    ax.set_axisbelow(True)


def curve(ax, d, method, label, color, marker='o', linestyle='-', **kwargs):
    q = d[d.method == method].sort_values('n_train')
    assert q.n_train.tolist() == BUDGETS, (method, q.n_train.tolist())
    ax.plot(q.n_train, q.value, marker=marker, markersize=4.5, color=color,
            linewidth=1.6, linestyle=linestyle, label=label, **kwargs)


def legend(fig, ax, columns=2):
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(.5, -.015),
               ncol=columns, frameon=False, fontsize=9, handlelength=2.2)


def export(fig, slug, d, sources, message, caption, design):
    # Source documents are recorded alongside numeric tables, not treated as data.
    sources = list(dict.fromkeys(list(d.source_file) + sources))
    publish(fig, slug, d, sources, message, caption, design)


def pair_pooling():
    metrics = load(POOL+'ridge_probe_metrics.csv')
    ci = load(POOL+'paired_family_bootstrap_differences.csv')
    a = marks(metrics[metrics.representation == 'ridge_mean_384d'], 'mae', 'raw_mae', 'compound-level unweighted MAE')
    b = marks(ci[(ci.representation == 'ridge_mean_384d') & (ci.metric == 'mae') &
                 (ci.comparison == 'ligand_ligand_only_minus_all_pairs')],
              'estimate', 'paired_difference', 'paired compound-level MAE difference')
    assert len(a) == 6 and len(b) == 2
    cohorts = ['nested_development', 'lockbox_all_790']
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.7), gridspec_kw={'width_ratios': [1, 1.08]})
    for method, label, color, marker in [('all_pairs', 'All pairs', GRAY, 'o'),
                                         ('ligand_ligand_only', 'Ligand–ligand', INK, 'o'),
                                         ('receptor_ligand_only', 'Receptor–ligand', ACCENT, '|')]:
        q = a[a.pool_variant == method].set_index('dataset').loc[cohorts]
        axes[0].scatter(q.value, [1, 0], c=color, marker=marker, s=58, label=label, zorder=3)
    for y, cohort in zip([1, 0], cohorts):
        q = a[a.dataset == cohort]
        axes[0].plot([q.value.min(), q.value.max()], [y, y], color=GRAY, lw=1, zorder=1)
        r = b[b.dataset == cohort].iloc[0]
        axes[1].errorbar(r.value, y, xerr=[[r.value-r.ci_lower], [r.ci_upper-r.value]],
                         fmt='o', color=INK, capsize=3, markersize=5)
    axes[0].set_yticks([1, 0], ['Development\nN = 3,344', 'Opened validation\nN = 790'])
    axes[0].set_xlabel('Raw MAE')
    axes[0].set_xlim(.58, .65)
    axes[0].set_xticks([.58, .60, .62, .64])
    axes[1].set_yticks([1, 0], [])
    axes[1].axvline(0, color=GRAY, lw=.8)
    axes[1].set_xlabel('Δ raw MAE vs all pairs')
    axes[1].set_title('Ligand–ligand final pool', fontsize=11)
    axes[1].set_xlim(-.05, .025)
    axes[1].set_xticks([-.04, -.02, 0, .02])
    for ax in axes:
        ax.set_ylim(-.6, 1.6)
    fig.subplots_adjust(left=.23, right=.98, bottom=.25, top=.88, wspace=.22)
    legend(fig, axes[0], 3)
    export(fig, 'pair-pooling', pd.concat([a, b], ignore_index=True), [POOL+'REPORT.md'],
           'Ligand–ligand final pooling improves development MAE, but its validation advantage is unresolved.',
           'Nested mean-384D ridge, 3,344 development and 790 previously opened validation compounds. '
           'Left: unweighted compound-level pEC50 MAE for three final pools (lower is better). '
           'Right: ligand–ligand minus all-pairs MAE; negative favors ligand–ligand. '
           'Saved paired 95% chemical-family bootstrap intervals use 2,000 replicates '
           '(2,931 development and 735 validation clusters). Upstream features were frozen and already '
           'protein-conditioned; this is not a protein-removal ablation.',
           'Absolute paired marks anchor effect size; saved difference intervals directly expose the unresolved validation effect. '
           'Receptor–ligand-only is a diagnostic tick, not the principal contrast; no value annotations.')


def low_data():
    a = load(OLD+'independent_assessment_aggregates.csv')
    a = a[(a.split == 'chemical_cluster') & a.n_train.isin(BUDGETS)]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 4))
    left = marks(a[a.method.isin(['repr_mlp', 'repr_ridge'])], 'wmae_cell_mean', 'original_vector_probes', 'mean-task weighted MAE')
    right = marks(a[a.method.isin([HEAD, 'head_random', 'morgan_ridge', 'native_continuous'])], 'wmae_cell_mean', 'practical_readouts', 'mean-task weighted MAE')
    for method, lab, col, marker in [('repr_mlp', 'Vector MLP', ACCENT, 's'), ('repr_ridge', 'Vector ridge', GRAY, 'o')]:
        curve(axes[0], left, method, lab, col, marker)
    for method, lab, col, marker, style in [(HEAD, 'Pretrained head', INK, 'o', '-'),
            ('head_random', 'Random head', ACCENT, 's', '--'), ('morgan_ridge', 'Morgan ridge', GRAY, '^', '-'),
            ('native_continuous', 'Native output', '#444444', '', ':')]:
        curve(axes[1], right, method, lab, col, marker, style)
    for ax in axes:
        budget_axis(ax)
    axes[0].set_title('Original vector probes', fontsize=11)
    axes[1].set_title('Native output and fitted readouts', fontsize=11)
    axes[0].set_ylim(.5, 2.05)
    axes[1].set_ylim(.50, .73)
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.28), frameon=False, fontsize=9)
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.28), frameon=False, fontsize=9, ncol=2, columnspacing=.8)
    fig.subplots_adjust(left=.10, right=.98, top=.88, bottom=.35, wspace=.36)
    export(fig, 'low-data', pd.concat([left, right], ignore_index=True),
           [OLD+'independent_assessment.md', 'experiments/20260905_low_data_adaptation/README.md'],
           'Continuous heads improve with more labels; the original vector probes show large, irregular errors.',
           COHORT+WEIGHT+'Original chemical-cluster split only, N25–500; full outer-pool reference omitted. '
           'Panels have different y ranges to keep the practical head comparison legible. '
           'Saved-model inspection in the independent assessment found scaling/extrapolation failures; '
           'the curves alone do not isolate a scaling cause. Random versus pretrained refers only to head initialization, '
           'not upstream pretraining. Native output is an unchanged binding-affinity anchor, not native cellular pEC50.',
           'Separate probe failures from the practical comparison rather than compress all methods under an outlier-dominated y scale. '
           'Two explicit mean-task axes and external legends; no confidence interpretation of repeated draws.')


def stable_readouts():
    t = load(FOLLOW+'task_means.csv')
    methods = ['repr_mlp', 'repr_mlp_stable', 'repr_mlp_stable_centered', 'repr_mlp_stable_centered_regularized']
    d = marks(t[(t.split == 'chemical_cluster') & t.n_train.isin(BUDGETS) & t.method.isin(methods)],
              'weighted_mae', 'mlp_learning_curves', 'mean-task weighted MAE')
    fig, ax = plt.subplots(figsize=(7.1, 3.9))
    for method, lab, col, marker, style in [
            ('repr_mlp', 'Original scaling', GRAY, 's', '-'),
            ('repr_mlp_stable', 'Stable scaling', ACCENT, '^', '-'),
            ('repr_mlp_stable_centered', '+ target centering', INK, 'o', '-'),
            ('repr_mlp_stable_centered_regularized', '+ weight decay', '#222222', 'x', ':')]:
        curve(ax, d, method, lab, col, marker, style)
    budget_axis(ax)
    ax.set_ylim(.48, 2.02)
    fig.subplots_adjust(left=.12, right=.97, bottom=.27, top=.96)
    legend(fig, ax, 2)
    export(fig, 'stable-readouts', d, [DOC+'READOUTS.md', DOC+'RESULTS.md'],
           'Stable scaling and target centering reduce MLP error; adding weight decay barely changes the curve.',
           COHORT+WEIGHT+'Chemical-cluster split, original and follow-up MLP recipes, N25–500. '
           'All use the same frozen concatenated 768D vectors and 768–128–1 MLP architecture. '
           'Stable per-feature scaling has a training-only variance floor; centered variants subtract and restore '
           'the training weighted target mean. Only the final variant adds AdamW decay 0.01. '
           'The centered and regularized curves nearly coincide; no prediction clipping or outlier exclusion.',
           'A sequence of four matched recipe curves makes the large preprocessing changes and negligible incremental decay effect visible. '
           'Cross markers preserve the regularized curve where it overlaps the centered circles.')


def descriptor_followup():
    t, s = load(FOLLOW+'task_means.csv'), load(FOLLOW+'strata.csv')
    a = marks(t[(t.lane == 'strict') & t.n_train.isin(BUDGETS) & t.method.isin([HEAD, DESC])],
              'weighted_mae', 'budget_curve', 'mean-task weighted MAE')
    strata = ['potency_lt4', 'potency_ge4']
    b = marks(s[(s.lane == 'strict') & (s.n_train == 500) & s.method.isin([HEAD, DESC]) & s.stratum.isin(strata)],
              'weighted_mae', 'potency_strata_N500', 'pooled within-stratum weighted MAE across repeated draws')
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.8), gridspec_kw={'width_ratios': [1.15, 1]})
    for method, label, color, marker in [(HEAD, 'Pretrained head', INK, 'o'), (DESC, 'Descriptor LightGBM', ACCENT, 's')]:
        curve(axes[0], a, method, label, color, marker)
        q = b[b.method == method].set_index('stratum').loc[strata]
        axes[1].plot([0, 1], q.value, marker=marker, color=color, label=label, lw=1.6, markersize=5)
    budget_axis(axes[0])
    axes[0].set_ylim(.46, .75)
    axes[1].set_xticks([0, 1], ['< 4', '≥ 4'])
    axes[1].set_xlim(-.2, 1.2)
    axes[1].set_ylim(.30, 1.38)
    axes[1].set_xlabel('Observed pEC50 stratum')
    axes[1].set_ylabel('Pooled weighted MAE')
    axes[1].set_title('N500', fontsize=11)
    fig.subplots_adjust(left=.10, right=.98, bottom=.25, top=.91, wspace=.4)
    legend(fig, axes[0], 2)
    export(fig, 'descriptor-followup', pd.concat([a, b], ignore_index=True), [DOC+'BASELINES.md', DOC+'RESULTS.md'],
           'Descriptors overtake the head from N250 overall, while the head retains lower N500 error at pEC50 ≥4.',
           COHORT+WEIGHT+'Strict chemical separation (retained FIT/CAL-to-test Morgan similarity <0.35). '
           'Left: mean-task scores. Right: N500 pooled within-stratum weighted errors across ten dependent draws, '
           'not means of task errors; 1,036 unique compounds / 10,360 predictions below pEC50 4 and '
           '2,308 / 23,080 at or above 4 (16.35% and 83.65% of evaluation weight). '
           'Descriptor features are canonical-SMILES RDKit2D, Mordred2D and Morgan with training-only reduction; '
           'this is not the historical richer descriptor pipeline. Potency strata are retrospective, not a routing rule.',
           'Learning-curve crossing communicates the budget-dependent ordering; a separate potency panel shows the reversal '
           'that the overall score conceals. Explicit axis names keep mean-task and pooled metrics distinct.')


def strict_followup():
    t, s = load(FOLLOW+'task_means.csv'), load(FOLLOW+'strata.csv')
    methods = [HEAD, DESC, RIDGE]
    a = marks(t[(t.n_train == 500) & t.method.isin(methods) & t.split.isin(['chemical_cluster', 'strict_chemical'])],
              'weighted_mae', 'split_comparison_N500', 'mean-task weighted MAE')
    strata = ['novelty_[0,0.2)', 'novelty_[0.2,0.35)']
    b = marks(s[(s.lane == 'strict') & (s.n_train == 500) & s.method.isin(methods) & s.stratum.isin(strata)],
              'weighted_mae', 'novelty_strata_N500', 'pooled within-stratum weighted MAE across repeated draws')
    assert len(a) == 6 and len(b) == 6
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.8), sharey=True)
    for method, label, color, marker in [(HEAD, 'Pretrained head', INK, 'o'), (DESC, 'Descriptor LightGBM', ACCENT, 's'),
                                         (RIDGE, 'Stable vector ridge', GRAY, '^')]:
        q = a[a.method == method].set_index('split').loc[['chemical_cluster', 'strict_chemical']]
        axes[0].plot([0, 1], q.value, marker=marker, color=color, label=label, lw=1.6, markersize=5)
        q = b[b.method == method].set_index('stratum').loc[strata]
        axes[1].plot([0, 1], q.value, marker=marker, color=color, lw=1.6, markersize=5)
    axes[0].set_xticks([0, 1], ['Cluster', 'Strict'])
    axes[0].set_xlabel('Training-pool separation')
    axes[0].set_ylabel('Mean-task weighted MAE')
    axes[1].set_xticks([0, 1], ['< 0.20', '0.20–< 0.35'])
    axes[1].set_xlabel('Nearest-FIT Morgan similarity')
    axes[1].set_ylabel('Pooled weighted MAE')
    axes[1].tick_params(labelleft=True)
    axes[0].set_title('N500 · same test compounds', fontsize=11)
    axes[1].set_title('N500 · strict pool', fontsize=11)
    for ax in axes:
        ax.set_xlim(-.2, 1.2)
        ax.set_ylim(.45, .95)
    fig.subplots_adjust(left=.10, right=.98, bottom=.25, top=.90, wspace=.4)
    legend(fig, axes[0], 3)
    export(fig, 'strict-followup', pd.concat([a, b], ignore_index=True), [DOC+'STRICT_SPLIT.md', DOC+'RESULTS.md'],
           'Neighbor purging barely moves overall error, but the most novel strict-test predictions remain harder.',
           COHORT+WEIGHT+'N500 = 400 FIT + 100 CAL. Strict pools exclude every candidate with Morgan similarity '
           '≥0.35 to any test compound; test identities are unchanged, training draws can change. '
           'Left: mean-task scores. Right: strict pooled within-stratum scores across ten dependent draws. '
           'Similarity <0.20 contains 906 prediction rows, 257 unique compounds and 1.93% of weight; '
           '0.20–<0.35 contains 32,534 rows, 3,319 unique compounds and 98.07% of weight. '
           'Stratum membership varies by FIT draw, so unique-compound counts overlap. '
           'The shared numerical y range supports comparison but the aggregation differs explicitly between panels; '
           'novelty results are descriptive, not prospective generalization guarantees.',
           'A flat paired split comparison contrasts with the steep novelty-stratum comparison on the same numerical scale. '
           'Three models show that the pattern is not peculiar to one readout; all text is axis, legend or panel identification.')


def verify():
    """Independently reread exported data and compare marks/CI to indexed sources."""
    counts = {}
    for slug in SLUGS:
        d = pd.read_csv(OUT/f'{slug}.csv')
        provenance = json.loads((OUT/f'{slug}.json').read_text())
        assert len(d) == provenance['plotted_rows']
        cache = {p: pd.read_csv(ROOT/p) for p in d.source_file.unique()}
        for _, row in d.iterrows():
            original = cache[row.source_file].iloc[int(row.source_row)]
            assert math.isclose(row.value, original[row.source_metric], rel_tol=1e-13, abs_tol=1e-14)
            for field in ['ci_lower', 'ci_upper', 'bootstrap_replicates', 'bootstrap_clusters',
                          'n_cells', 'n_test', 'n_fit', 'n_calibration', 'prediction_rows',
                          'unique_compounds', 'weight_share', 'n_train', 'method', 'split',
                          'lane', 'dataset', 'pool_variant', 'representation', 'stratum']:
                if field not in original.index or field not in row.index or pd.isna(row[field]):
                    continue
                actual, expected = row[field], original[field]
                if isinstance(expected, str):
                    assert actual == expected, (slug, field)
                else:
                    assert math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-14), (slug, field)
        for suffix in ['svg', 'png', 'csv', 'json']:
            assert (OUT/f'{slug}.{suffix}').stat().st_size > 0
        counts[slug] = len(d)
    print('VERIFIED exported marks and retained source identifiers:', json.dumps(counts))


def contact_sheet():
    from PIL import Image, ImageDraw
    thumbs = []
    for slug in SLUGS:
        im = Image.open(OUT/f'{slug}.png').convert('RGB')
        im.thumbnail((760, 450))
        tile = Image.new('RGB', (790, 490), 'white')
        ImageDraw.Draw(tile).text((15, 8), slug, fill='black')
        tile.paste(im, ((790-im.width)//2, 35))
        thumbs.append(tile)
    sheet = Image.new('RGB', (1580, 1470), '#eeeeee')
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i % 2)*790, (i//2)*490))
    path = OUT/'lowdata-contact-sheet.png'
    sheet.save(path)
    print(path)


if __name__ == '__main__':
    pair_pooling()
    low_data()
    stable_readouts()
    descriptor_followup()
    strict_followup()
    verify()
    contact_sheet()
