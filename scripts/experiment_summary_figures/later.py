#!/usr/bin/env python3
"""Seven reporting-only summaries: saved results, no fitting or inference."""
from pathlib import Path
import numpy as np
from common import ROOT, OUT, plt, pd, INK, ACCENT, GRAY, publish

RESULTS = ROOT/'reports/architecture/results'
EFFICACY = ROOT/'experiments/20260908_PXR-MW-efficacy-diagnostic'
CYP = Path('/home/dan/projects/ADMET-CYP/experiments/20260908_CYP-Nesso-MW-case-study')
WEIGHTS = 'Errors use weights 1/max(reported assay SE, 0.1); lower weighted MAE is better.'
N500 = 'N500 = 400 FIT + 100 CAL per fold, one draw and seed; five reused strict-chemical development folds, not independent replicates or a fresh holdout.'


def read(p):
    return pd.read_csv(p, float_precision='round_trip')


def finish(fig, slug, data, sources, message, caption, design):
    import textwrap
    fig.suptitle(textwrap.fill(message, width=69), fontsize=12, x=.02, ha='left')
    publish(fig, slug, data, sources + [Path(__file__)], message, caption, design)


def upstream():
    source=RESULTS/'pooled_metrics.csv'
    methods=['head_only','head_plus_lora','repr_ridge_stable','descriptor_lightgbm_rdkit_mordred']
    labels=['Head-only','Head + LoRA','Frozen-vector ridge','Descriptors']
    d=read(source);d=d[d.method.isin(methods)].copy()
    assert len(d)==8 and (d.n==3344).all()
    d['plot_y']=d.method.map(dict(zip(methods,range(4))))
    fig,axs=plt.subplots(1,2,figsize=(7,3.6),sharex=True,sharey=True,layout='constrained')
    for ax,b in zip(axs,[100,500]):
        z=d[d.budget==b]
        ax.scatter(z.weighted_mae,z.plot_y,c=[ACCENT if m=='head_plus_lora' else INK for m in z.method],s=44)
        ax.set(title=f'{b} acquired labels',xlabel='Weighted MAE (pEC50)',xlim=(.48,.64),yticks=range(4),yticklabels=labels)
        ax.invert_yaxis() if b==100 else None
    finish(fig,'upstream-comparison',d,[source,ROOT/'experiments/20260906_affinity_comparison/RESULTS.md'],
           'LoRA adds a small gain; descriptors lead at 500 labels',
           'Pooled held-out scores, 3,344 compounds once per method and budget. N100 = 80 FIT + 20 CAL; '+N500+' Both neural arms used 40 full-FIT updates with matched optimizer; controls had different tuning procedures. '+WEIGHTS,
           'Aligned method dot plots on a shared error scale compare the matched neural gain with practical controls; no bars on a truncated baseline.')


def geometry():
    source=EFFICACY/'artifacts/merged_compounds.csv'
    d=read(source)[['record_id','fold','MW','native','head_plus_lora_500','pEC50']].copy()
    assert len(d)==3344 and d.record_id.is_unique
    d['absolute_prediction_change']=(d.head_plus_lora_500-d.native).abs()
    d['budget']=500;d['method']='head_plus_lora'
    fig,ax=plt.subplots(figsize=(6.5,4),layout='constrained')
    ax.scatter(d.MW,d.absolute_prediction_change,s=8,alpha=.25,c=INK,edgecolors='none',rasterized=True)
    ax.set(xlabel='Molecular weight (Da)',ylabel='Absolute change from native (log units)',ylim=(0,None))
    finish(fig,'geometry',d,[source,ROOT/'experiments/20260906_affinity_comparison/RESULTS.md'],
           'Smaller compounds tend to move more under adaptation',
           'All 3,344 development compounds, one saved N500 head-plus-LoRA prediction each. '+N500+' Unweighted absolute movement |LoRA − native| is not error reduction. Native is the released IC50-equivalent scalar conversion, not measured cellular potency. Full MW and movement ranges retained; size association is descriptive, not a causal mechanism.',
           'A full-cohort continuous MW-versus-movement scatter exposes the size association without thresholds, regression lines, or selecting only successful corrections.')


def mw_controls():
    source=RESULTS/'mw-controls/metrics.csv'
    methods=['native_continuous','head_plus_lora','native_mw','native_mw_interaction']
    labels=['Native unchanged','Head + LoRA','Score + MW','Score + MW + interaction']
    d=read(source);d=d[(d.budget==500)&d.method.isin(methods)&d.subset.isin(['lt215','gt215'])].copy()
    assert len(d)==8
    d['plot_y']=d.method.map(dict(zip(methods,range(4))))
    fig,axs=plt.subplots(1,2,figsize=(7,3.6),sharex=True,sharey=True,layout='constrained')
    for ax,subset,title in zip(axs,['lt215','gt215'],['Below 215 Da · n=237','Above 215 Da · n=3,107']):
        z=d[d.subset==subset]
        ax.scatter(z.weighted_mae,z.plot_y,c=[ACCENT if m=='native_mw_interaction' else INK for m in z.method],s=42)
        ax.set(title=title,xlabel='Weighted MAE (pEC50)',yticks=range(4),yticklabels=labels,xlim=(.45,1.95))
    axs[0].invert_yaxis()
    finish(fig,'mw-controls',d,[source,ROOT/'experiments/20260908_mw_controls/README.md'],
           'A scalar interaction beats LoRA on small compounds—not larger ones',
           'Pooled N500 held-out scores: 237 compounds below 215 Da and 3,107 above; none exactly 215 Da. '+N500+' Score+MW has three fitted coefficients; interaction adds a fourth. Controls used FIT-only weighted least squares, not the neural objective; the native reference is unchanged. '+WEIGHTS+' Subgroup correction does not imply a biological mechanism.',
           'Two molecular-size strata on identical error scales show both the simple-control success and its limit; direct method labels avoid a legend.')


def aggregate(frame,fold=None):
    rows=[]
    for stratum,mask in [('All',np.ones(len(frame),dtype=bool)),('pEC50 ≥4',frame.y_true>=4),('pEC50 <4',frame.y_true<4)]:
        z=frame.loc[mask]
        for method in ['descriptor','ridge','blend']:
            rows.append(dict(fold=fold,stratum=stratum,method=method,n=len(z),weight_sum=z.weight.sum(),weighted_mae=np.average(abs(z[method]-z.y_true),weights=z.weight),budget=500))
    return pd.DataFrame(rows)


def hybrid_pilot():
    base=ROOT/'artifacts/experiments/descriptor_nesso_blend_20260908'
    source=base/'test_predictions.csv';raw=read(source)
    assert len(raw)==669 and raw.record_id.is_unique
    d=aggregate(raw,0);d=d[d.method.isin(['descriptor','blend'])].copy()
    strata=['All','pEC50 ≥4','pEC50 <4'];d['plot_y']=d.stratum.map(dict(zip(strata,range(3))))
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    for i,s in enumerate(strata):
        z=d[d.stratum==s].set_index('method');ax.plot(z.loc[['descriptor','blend'],'weighted_mae'],[i,i],color=GRAY,lw=1.2)
    for method,label,color in [('descriptor','Descriptors',GRAY),('blend','Blend',INK)]:
        z=d[d.method==method];ax.scatter(z.weighted_mae,z.plot_y,s=48,color=color,label=label,zorder=3)
    ax.set(yticks=range(3),yticklabels=strata,xlabel='Weighted MAE (pEC50)',xlim=(.39,.80));ax.invert_yaxis()
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.21),ncol=2,frameon=False)
    finish(fig,'hybrid-pilot',d,[source,base/'metrics.json',ROOT/'experiments/20260908_descriptor_nesso_blend/README.md'],
           'The pilot blend helps higher potency, but hurts lower potency',
           'One preselected reused outer fold (fold 0), 669 compounds: 476 at pEC50 ≥4 and 193 below 4. N500 = 400 FIT + 100 CAL; CAL outcomes unused. Frozen-vector ridge receives a FIT-inner-OOF-selected mixture weight of approximately 0.3983; this is not LoRA. '+WEIGHTS+' Group rows are within-stratum scores, not contributions to the total; the higher-potency group carries 83.42% of evaluation weight. Exploratory pilot, not independent confirmation.',
           'Paired dots show descriptor-to-blend direction separately for pooled and potency-stratum errors without confusing within-stratum error with weighted contribution.')


def hybrid_allfolds():
    base=ROOT/'artifacts/experiments/descriptor_nesso_blend_allfolds_20260908'
    source=base/'all_test_predictions.csv';raw=read(source)
    assert len(raw)==3344 and raw.record_id.is_unique
    stats=pd.concat([aggregate(z,int(f)) for f,z in raw.groupby('outer_fold')],ignore_index=True)
    rows=[]
    for (fold,stratum),z in stats.groupby(['fold','stratum']):
        z=z.set_index('method');a=z.loc['descriptor'];b=z.loc['blend']
        rows.append(dict(fold=fold,stratum=stratum,n=a.n,weight_sum=a.weight_sum,descriptor_weighted_mae=a.weighted_mae,blend_weighted_mae=b.weighted_mae,delta_weighted_mae=b.weighted_mae-a.weighted_mae,budget=500,provenance='reused_pilot' if fold==0 else 'new_fit'))
    d=pd.DataFrame(rows);d=d[d.stratum.isin(['All','pEC50 <4'])].copy()
    assert len(d)==10
    assert (d.loc[d.stratum=='All','delta_weighted_mae']<0).all()
    assert (d.loc[d.stratum=='pEC50 <4','delta_weighted_mae']>0).all()
    fig,axs=plt.subplots(1,2,figsize=(7,3.7),sharex=True,sharey=True,layout='constrained')
    for ax,s,title in zip(axs,['All','pEC50 <4'],['All held-out compounds','Lower potency: pEC50 <4']):
        z=d[d.stratum==s];ax.axvline(0,color=GRAY,lw=.8)
        ax.scatter(z.delta_weighted_mae,z.fold,s=45,color=INK if s=='All' else ACCENT)
        ax.set(title=title,yticks=range(5),yticklabels=['Fold 0 · pilot','Fold 1','Fold 2','Fold 3','Fold 4'],xlim=(-.03,.07),xticks=[-.02,0,.02,.04,.06])
    axs[0].invert_yaxis();fig.supxlabel('Blend − descriptor weighted MAE (pEC50); negative = better',fontsize=10)
    finish(fig,'hybrid-allfolds',d,[source,base/'metrics.json',ROOT/'experiments/20260908_descriptor_nesso_blend_allfolds/README.md'],
           'Every fold gains overall—and pays a lower-potency penalty',
           'Paired within-fold weighted-MAE differences for the same held-out compounds: fold totals 669/669/669/668/669; 3,344 unique compounds overall, 1,036 below pEC50 4. '+N500+' Fold 0 reuses the inspected pilot byte-identically; folds 1–4 apply the locked procedure, selecting their own FIT-only blend weights. '+WEIGHTS+' Below-4 compounds carry 16.4% of pooled weight; this is not uniform improvement or retraining-variability evidence.',
           'Aligned fold-level difference dots share a zero and scale, making consistent aggregate gains and consistent lower-potency losses directly visible without treating folds as independent confidence intervals.')


def cyp_transfer():
    source=CYP/'artifacts/contrasts.csv';d=read(source)
    comparisons=['additive_minus_nesso','interaction_minus_additive']
    d=d[(d.split=='butina')&(d.subgroup=='all')&d.comparison.isin(comparisons)].copy()
    names=['CYP1A2','CYP2C9','CYP2D6','CYP3A4'];d['plot_y']=d.isoform.map(dict(zip(names,range(4))))
    assert len(d)==8
    fig,axs=plt.subplots(1,2,figsize=(7,3.7),sharex=True,sharey=True,layout='constrained')
    for ax,c,title in zip(axs,comparisons,['Add MW to fitted score','Add score–MW interaction']):
        z=d[d.comparison==c];ax.axvline(0,color=GRAY,lw=.8)
        ax.errorbar(z.delta_mae,z.plot_y,xerr=np.vstack([z.delta_mae-z.lo95,z.hi95-z.delta_mae]),fmt='o',color=INK,ms=5,capsize=0,lw=1.3)
        ax.set(title=title,yticks=range(4),yticklabels=names,xlim=(-.03,.007),xticks=[-.02,-.01,0])
    axs[0].invert_yaxis();fig.supxlabel('Change in unweighted MAE (pIC50); negative = better',fontsize=10)
    finish(fig,'cyp-transfer',d,[source,CYP/'README.md'],
           'MW helps two CYP assays; the interaction is not a general gain',
           'Primary five-fold Butina pooled same-row direct-inhibition evaluation, not TDI: CYP1A2/2C9/2D6/3A4 n=1,412/1,285/1,493/2,335 (6,525 compound–endpoint observations, 4,905 unique compounds). Equal-weight OLS and MAE: supplied std/CI are not justified sampling SE, so no PXR uncertainty floor. Left: additive minus fitted-score-only; right: interaction minus additive. Whiskers are saved 95% paired whole-group bootstrap intervals (2,000 draws, seed 42), conditional on saved fits, no retraining uncertainty or multiplicity adjustment. Reused development splits, no fixed PXR N500 budget.',
           'Two aligned contrast panels retain all four endpoint identities and interval uncertainty; the zero line distinguishes additional information from general interaction benefit without overlaying seven methods.')


def efficacy_uncertainty():
    source=EFFICACY/'artifacts/merged_compounds.csv';raw=read(source)
    assert len(raw)==3344 and raw.record_id.is_unique
    rows=[];fig,axs=plt.subplots(1,2,figsize=(7,3.8),sharey=True,layout='constrained')
    for ax,variable,label in zip(axs,['Emax_ratio','raw_pEC50_SE'],['Emax / positive control','Reported pEC50 standard error']):
        for small,color in [(False,GRAY),(True,INK)]:
            z=raw.loc[(raw.MW<215)==small,['record_id','MW',variable]].sort_values([variable,'record_id']).copy()
            n=len(z);z['ecdf']=np.arange(1,n+1)/n;z['value']=z[variable];z['variable']=variable;z['stratum']='MW <215 Da' if small else 'MW ≥215 Da';z['denominator']=n
            rows.append(z.drop(columns=variable));ax.step(z.value,z.ecdf,where='post',color=color,lw=1.7,label=z.stratum.iloc[0]+f' (n={n:,})')
        ax.set(xlabel=label,ylim=(0,1),yticks=[0,.5,1])
    axs[0].set_xscale('log');axs[0].set_xticks([.25,.5,1,2,4,8]);axs[0].set_xticklabels(['0.25','0.5','1','2','4','8'])
    axs[0].set_ylabel('Fraction of compounds ≤ x');axs[1].set_xlim(0,.75)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.12),ncol=2,frameon=False)
    d=pd.concat(rows,ignore_index=True);assert len(d)==6688
    finish(fig,'efficacy-uncertainty',d,[source,EFFICACY/'README.md'],
           'Small compounds have higher uncertainty, not lower normalized efficacy',
           'Unweighted empirical cumulative distributions for all 3,344 saved development IDs: 237 below 215 Da and 3,107 at least 215 Da; no missing values or Emax refiltering. Each compound appears once in each panel; each stratum has its own denominator. Normalized Emax uses a logarithmic x-axis and retains values above one and the full tail. Raw baseline-relative Emax is a different scale and is lower in small compounds. Higher reported SE describes less precise fitted potency, not a demonstrated biological cause; dose-response curves and the normalization denominator are unavailable. No model fit or inference.',
           'Full empirical distributions show the uncertainty shift alongside normalized efficacy without smoothing, box plots, median overlays, or truncating the long efficacy tail.')


if __name__=='__main__':
    for function in [upstream,geometry,mw_controls,hybrid_pilot,hybrid_allfolds,cyp_transfer,efficacy_uncertainty]:
        function()
