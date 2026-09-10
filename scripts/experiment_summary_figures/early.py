#!/usr/bin/env python3
"""Reporting-only summaries of seven early PXR experiments; no model execution."""
import json
import numpy as np
from common import ROOT, OUT, plt, pd, INK, ACCENT, GRAY, publish

MC='reports/model_comparison/'
TA='reports/transfer_ablation/'
CPU=TA+'cpu_followups/'
SLUGS=['conception','head-only','transfer-probes','capacity','loss-lr','binary-dual']

def read(path):
    d=pd.read_csv(ROOT/path)
    d['source_path']=path
    d['source_row']=np.arange(len(d))
    return d

def long(d,metrics,ids):
    return d[ids+['source_path','source_row']+metrics].melt(id_vars=ids+['source_path','source_row'],value_vars=metrics,var_name='metric',value_name='value')

def save(fig,slug,d,sources,message,caption,design):
    publish(fig,slug,d,sources,message,caption,design)
    saved=pd.read_csv(OUT/f'{slug}.csv')
    assert len(saved)==len(d)
    np.testing.assert_allclose(saved.value,d.value,rtol=1e-14,atol=1e-14)
    for p,g in saved.groupby('source_path'):
        src=pd.read_csv(ROOT/p)
        for row in g.itertuples():
            col=getattr(row,'source_metric',row.metric)
            np.testing.assert_allclose(row.value,src.iloc[row.source_row][col],rtol=1e-14,atol=1e-14)

def dots(ax,d,metric,models,labels,color=INK):
    q=d[d.metric.eq(metric)].set_index('model').loc[models]
    ax.scatter(q.value,range(len(models)),s=44,color=color,zorder=3)
    ax.set_yticks(range(len(models)),labels)
    ax.set_ylim(len(models)-.5,-.5)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.grid(axis='x',color='#eeeeee',lw=.6)


def main():
    metrics=read(MC+'metrics.csv')
    # Original matched comparison, not a mix of later weighted budgets.
    models=['frozen_nesso','affine_frozen_nesso','fine_tuned_nesso','nested_2d_lightgbm']
    q=metrics.query("dataset == 'nested_development' and subset == 'all' and metric == 'mae'").copy()
    q=q[q.model.isin(models)].rename(columns={'estimate':'value'})
    q['source_metric']='estimate'
    fig,ax=plt.subplots(figsize=(6.7,3.1))
    dots(ax,q,'mae',models,['Native continuous score','Affine-calibrated score','Adapted continuous heads','Ligand-only LightGBM'])
    ax.set(xlabel='Raw MAE (pEC50)',xlim=(0,1.0))
    fig.tight_layout()
    save(fig,'conception',q,[MC+'metrics.csv',MC+'SCIENTIFIC_REPORT.md','nesso1_pxr_finetuning_poc_spec.md'],
         'Head adaptation improves on the native continuous score, but ligand-only LightGBM remains stronger.',
         'Same 3,344 development compounds and five chemical-family outer folds; pooled unweighted MAE, lower is better. Native continuous affinity is a pIC50-equivalent score, not a measured cellular pEC50. Learned readouts use training-only selection; fixed legacy 2D unsupervised prefilter was not rebuilt within folds. Retrospective evidence.',
         'Aligned aggregate dots on a common zero-based error axis expose both the adaptation gain and remaining descriptor gap; no uncertainty inferred.')

    p=read(MC+'nested_aligned_predictions.csv')
    d=long(p,['nesso_pEC50','two_d_pEC50'],['record_id','original_id','fold','pEC50'])
    d['model']=d.metric.map({'nesso_pEC50':'fine_tuned_nesso','two_d_pEC50':'nested_2d_lightgbm'})
    assert p.record_id.nunique()==3344
    for col,model in [('nesso_pEC50','fine_tuned_nesso'),('two_d_pEC50','nested_2d_lightgbm')]:
        expected=metrics.query("dataset == 'nested_development' and subset == 'all' and metric == 'mae' and model == @model").estimate.item()
        np.testing.assert_allclose((p[col]-p.pEC50).abs().mean(),expected,atol=1e-12)
    lo=np.floor(min(p.pEC50.min(),p.nesso_pEC50.min(),p.two_d_pEC50.min()))
    hi=np.ceil(max(p.pEC50.max(),p.nesso_pEC50.max(),p.two_d_pEC50.max()))
    fig,axes=plt.subplots(1,2,figsize=(7,3.7),sharex=True,sharey=True)
    for ax,col,title in zip(axes,['nesso_pEC50','two_d_pEC50'],['Adapted continuous heads','Ligand-only LightGBM']):
        ax.plot([lo,hi],[lo,hi],color=GRAY,lw=.8,zorder=0)
        ax.scatter(p.pEC50,p[col],s=7,alpha=.27,color=INK,edgecolors='none',rasterized=True)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed pEC50',title=title)
        ax.set_aspect('equal')
    axes[0].set_ylabel('Predicted pEC50')
    fig.tight_layout()
    save(fig,'head-only',d,[MC+'nested_aligned_predictions.csv',MC+'metrics.csv',MC+'SCIENTIFIC_REPORT.md'],
         'Both fitted models compress the potent tail despite lower overall error for LightGBM.',
         'Each panel shows all 3,344 identical development compounds, using five-fold nested out-of-fold predictions; diagonal is exact prediction. Heads average three seeds. The 55 observed pEC50 >6 compounds include only one head prediction >6 and no LightGBM prediction >6. Closer to the diagonal is better; raw errors, not uncertainty-weighted scores. No lockbox or challenge rows.',
         'Matched observed-versus-predicted point clouds use identical square axes and transparency, revealing range compression without conclusion annotations.')

    r=read(TA+'representation_probe_metrics.csv').query("dataset == 'nested_development'")
    r=r[r.model.str.startswith('ridge_')]
    d=long(r,['mae','spearman'],['dataset','model','n'])
    refs=metrics.query("dataset == 'nested_development' and subset == 'all' and model in ['fine_tuned_nesso','nested_2d_lightgbm'] and metric in ['mae','spearman']").rename(columns={'estimate':'value'})
    refs['source_metric']='estimate'
    d['source_metric']=d.metric
    d=pd.concat([d,refs],ignore_index=True)
    models=['ridge_member1_384d','ridge_member2_384d','ridge_mean_384d','ridge_concat_768d','fine_tuned_nesso','nested_2d_lightgbm']
    labels=['Ridge · member 1 (384D)','Ridge · member 2 (384D)','Ridge · mean (384D)','Ridge · concat (768D)','Adapted continuous heads','Ligand-only LightGBM']
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.8),sharey=True)
    for ax,metric,xlabel,lim in zip(axes,['mae','spearman'],['Raw MAE (pEC50)','Spearman correlation'],[(.48,.72),(.53,.77)]):
        dots(ax,d,metric,models,labels); ax.set(xlabel=xlabel,xlim=lim)
    axes[1].tick_params(labelleft=False)
    fig.tight_layout()
    save(fig,'transfer-probes',d,[TA+'representation_probe_metrics.csv',TA+'README.md',MC+'metrics.csv'],
         'Frozen-vector ridge recovers substantial PXR signal without the pretrained nonlinear heads.',
         'Pooled outer-fold development metrics for 3,344 compounds. Ridge penalties selected within training folds; mean and member probes use 384D, concatenation 768D. Raw MAE lower and Spearman higher are better. These estimates do not prove equivalence or protein-specific causality; exploratory nonnested MLP screens are not shown.',
         'Two aligned dot panels preserve the error-versus-ranking distinction and show every ridge representation alongside the historical heads and descriptor reference.')

    path=CPU+'experiment_a_capacity/capacity_metrics.csv'
    r=read(path); r=r[~r.representation.eq('published_twin_member_pair')]
    d=long(r,['mae'],['dataset','representation','model_class','n'])
    d['capacity']=d.model_class.map({'ridge':0,'one_hidden_mlp':1,'two_hidden_mlp':2})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.9),sharey=True)
    styles=[('member1_384d','Member 1 · 384D',GRAY,'o'),('member2_384d','Member 2 · 384D','#333333','s'),('mean_384d','Mean · 384D',INK,'o'),('concat_768d','Concat · 768D',ACCENT,'^')]
    for ax,cohort,title in zip(axes,['nested_development','retrospective_lockbox_790'],['Development · 3,344','Retrospective validation · 790']):
        for rep,label,color,mark in styles:
            g=d[d.dataset.eq(cohort)&d.representation.eq(rep)].sort_values('capacity')
            assert len(g)==3
            ax.plot(g.capacity,g.value,marker=mark,color=color,lw=1,label=label,ms=5)
        ax.set(xticks=[0,1,2],xticklabels=['Ridge','1 hidden','2 hidden'],title=title,ylim=(.54,.70),xlim=(-.15,2.15))
    axes[0].set_ylabel('Raw MAE (pEC50)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,.16,1,1))
    save(fig,'capacity',d,[path,CPU+'README.md'],
         'Extra readout capacity changes error modestly and inconsistently across frozen representations.',
         'Complete 4-representation × 3-readout matrix: 3,344 nested development and 790 retrospective validation compounds. Scaling and five tuning candidates per readout are training-only; neural predictions average three seeds. Raw MAE lower is better. Lines link architecture choices, not parameter counts. Primary mean-384D development gains over ridge have unresolved paired intervals; upstream pretraining is unchanged.',
         'Capacity ladders on a common error scale show all representation choices rather than selecting a favorable architecture or seed; validation is a separate panel.')

    # Chemical splits is a methods page: its rejected identity chart is archived,
    # never regenerated or included in the visible summary inventory.

    path=CPU+'experiment_c_loss_lr/loss_lr_metrics.csv'
    r=read(path).query("dataset == 'nested_development'")
    d=long(r,['mae','prediction_maximum'],['dataset','model','n','active_n','predicted_above_6'])
    models=['mse','mae','huber_0.25','huber_0.5','huber_1','huber_2','global_selector','published_historical_ft']
    labels=['MSE','MAE','Huber 0.25','Huber 0.5','Huber 1','Huber 2','Joint loss × LR selector','Historical heads']
    fig,axes=plt.subplots(1,2,figsize=(7.1,4),sharey=True)
    dots(axes[0],d,'mae',models,labels); axes[0].set(xlabel='Raw MAE (pEC50)',xlim=(.615,.665))
    dots(axes[1],d,'prediction_maximum',models,labels); axes[1].set(xlabel='Maximum predicted pEC50',xlim=(5.4,6.15),xticks=[5.5,5.75,6])
    axes[1].axvline(6,color=GRAY,lw=.8,ls='--'); axes[1].tick_params(labelleft=False)
    fig.tight_layout()
    save(fig,'loss-lr',d,[path,CPU+'README.md'],
         'Joint loss and learning-rate selection reduces average error without restoring predictions above pEC50 6.',
         'All 3,344 nested development predictions per model; 55 observations exceed pEC50 6. Each objective selects LR within outer training folds, with weight decay zero; joint selector also selects objective. Raw MAE lower is better; maximum prediction is a range diagnostic, not an accuracy metric. Dashed line marks pEC50 6. No uncertainty inferred from these maxima.',
         'Aligned objective rows pair average error with the actual prediction ceiling, keeping distinct units on separate axes and avoiding a misleading aggregate-only success claim.')

    path=TA+'binary_head_reanalysis/regression_metrics.csv'
    models=['affinity_only_linear','continuous_plus_binary_polynomial2','fine_tuned_continuous_head','fine_tuned_dual_branch_activity_adapter','lightgbm']
    labels=['Continuous · affine','Both scalars · quadratic','Continuous heads · fitted','Dual heads · fitted','Ligand-only LightGBM']
    r=read(path); r=r[r.cohort.eq('all')&r.dataset.isin(['development','challenge'])&r.model.isin(models)]
    d=long(r,['mae'],['dataset','cohort','model','n'])
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.6),sharey=True)
    for ax,cohort,title in zip(axes,['development','challenge'],['Development · 3,344','Public challenge · 513']):
        dots(ax,d[d.dataset.eq(cohort)],'mae',models,labels)
        ax.set(xlabel='Raw MAE (pEC50)',title=title,xlim=(.48,.90))
    axes[1].tick_params(labelleft=False)
    fig.tight_layout()
    save(fig,'binary-dual',d,[path,TA+'binary_head_reanalysis/REPORT.md'],
         'Using both branches helps scalar calibration, but dual-head adaptation does not improve consistently across cohorts.',
         'Raw unweighted MAE, lower is better. Development uses 3,344 chemical-family outer-fold predictions; public challenge uses 513 retrospective public-label compounds. Calibration and adapters fitted on development only. Binary outputs predict binding, not cellular activation; calibrations here target functional pEC50. The challenge LightGBM is the established augmented/offset comparator, not the exact nested-development estimator.',
         'Identical method rows and shared error limits make the small development dual-head advantage and reversed challenge ordering visible, while retaining the scalar-calibration comparison.')

    from PIL import Image, ImageOps, ImageDraw
    sheet=Image.new('RGB',(1520,4*470),'white')
    for i,slug in enumerate(SLUGS):
        im=Image.open(OUT/f'{slug}.png').convert('RGB')
        im.thumbnail((740,430))
        x=(i%2)*760; y=(i//2)*470
        sheet.paste(im,(x+(760-im.width)//2,y+30))
        ImageDraw.Draw(sheet).text((x+15,y+8),slug,fill='black')
    sheet.save(OUT/'early-contact-sheet.png')
    print('Verified all seven CSV exports against exact source rows; contact sheet:',OUT/'early-contact-sheet.png')

if __name__=='__main__':
    main()
