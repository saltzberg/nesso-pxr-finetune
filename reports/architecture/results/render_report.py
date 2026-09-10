"""Render verified saved results; canonical prose is RESULTS.md. No fitting."""
from pathlib import Path
import json,html,re,hashlib,shutil,datetime
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parents[2];OUT=R/'reports/architecture/results';OUT.mkdir(parents=True,exist_ok=True)
README=R/'experiments/20260906_affinity_comparison/RESULTS.md'
assert json.loads((H/'verification.json').read_text())['status']=='PASS'
read=lambda name:pd.read_csv(H/(name+'.csv'),float_precision='round_trip')
p=read('pooled_metrics');f=read('fold_metrics');pairs=read('paired_differences');d=read('heldout_predictions');s=read('stratum_metrics');hist=read('training_history')
names={'native_continuous':'Native unchanged','head_only':'Head-only','head_plus_lora':'Head + LoRA','repr_ridge_stable':'Ridge on Nesso vectors','descriptor_lightgbm_rdkit_mordred':'Ligand-descriptor LightGBM'}
colors={'native_continuous':'#77818b','head_only':'#2166ac','head_plus_lora':'#b85c00','repr_ridge_stable':'#556b5e','descriptor_lightgbm_rdkit_mordred':'#414a54'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'axes.edgecolor':'#8a9299','xtick.color':'#46505a','ytick.color':'#46505a','axes.labelcolor':'#26323b','text.color':'#26323b','svg.fonttype':'none','savefig.facecolor':'white'})
from mw_markers import scatter_mw,audit_figure,AUDIT,D
figures=[]
def save(fig,name):
 audit_figure(fig,name)
 for ext in ['svg','png']:fig.savefig(OUT/(name+'.'+ext),dpi=190,bbox_inches='tight',metadata={'Creator':'Nesso saved-prediction analysis'} if ext=='svg' else None)
 plt.close(fig);figures.append(name)
fig,axes=plt.subplots(1,2,figsize=(10.6,4.6),sharex=True,sharey=True,layout='constrained')
lims=[]
for ax,n in zip(axes,[100,500]):
 a=f[(f.budget==n)&(f.method=='head_plus_lora')].set_index('fold');b=f[(f.budget==n)&(f.method=='head_only')].set_index('fold');delta=(a.weighted_mae-b.weighted_mae).reindex(range(5))
 v=pairs[(pairs.budget.astype(str)==str(n))&(pairs.model_a=='head_plus_lora')&(pairs.model_b=='head_only')].iloc[0]
 ax.axvline(0,color='#aab1b6',lw=.8);ax.plot(delta.to_numpy(),np.arange(5),'o',ms=5,color='#626c75')
 ax.errorbar(v.weighted_mae_delta,6,xerr=[[v.weighted_mae_delta-v.weighted_ci_low],[v.weighted_ci_high-v.weighted_mae_delta]],fmt='D',ms=6,color=colors['head_plus_lora'],elinewidth=1.1,capsize=3)
 ax.plot(float(delta.mean()),7,'D',ms=5,markerfacecolor='white',markeredgecolor='#626c75')
 ax.set_title(f'N{n}',loc='left',fontweight='bold');ax.set_yticks([0,1,2,3,4,6,7],['Fold 0','Fold 1','Fold 2','Fold 3','Fold 4','Pooled','Fold mean']);ax.set_ylim(7.65,-.6);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.set_xlabel('Δ weighted MAE (LoRA − head-only)')
 lims.extend(delta);lims.extend([v.weighted_ci_low,v.weighted_ci_high])
span=max(abs(x) for x in lims)*1.16
for ax in axes:ax.set_xlim(-span,span)
save(fig,'paired-adaptation-effect')
# Each panel includes every compound. Controls use a separate explicitly broader axis range.
def scatters(methods,name):
 fig,axes=plt.subplots(2,len(methods),figsize=(3.45*len(methods),6.8),sharex=True,sharey=True,layout='constrained',squeeze=False)
 sub=d[d.method.isin(methods)];lo=float(min(sub.y_true.min(),sub.y_pred.min()));hi=float(max(sub.y_true.max(),sub.y_pred.max()));pad=.035*(hi-lo);lo-=pad;hi+=pad
 for row,n in enumerate([100,500]):
  for col,m in enumerate(methods):
   ax=axes[row,col];v=d[(d.budget==n)&(d.method==m)];assert len(v)==3344
   ax.plot([lo,hi],[lo,hi],color='#bdc3c7',lw=.8,zorder=1);scatter_mw(ax,v.y_true,v.y_pred,v.record_id,s=3,alpha=.30,color=colors[m],linewidths=0,zorder=2)
   ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect='equal');ax.set_title(f'{names[m]} · N{n}',fontsize=10,loc='left')
   if row==1:ax.set_xlabel('Observed pEC50')
   if col==0:ax.set_ylabel('Predicted value')
 save(fig,name)
scatters(['native_continuous','head_only','head_plus_lora'],'prediction-compression')
scatters(['repr_ridge_stable','descriptor_lightgbm_rdkit_mordred'],'control-predictions-full-range')
# Post-hoc descriptive influence audit, not a changed official score.
influence=[];contrib=[]
for n in [100,500]:
 a=d[(d.budget==n)&(d.method=='head_plus_lora')].sort_values('record_id').reset_index(drop=True);b=d[(d.budget==n)&(d.method=='head_only')].sort_values('record_id').reset_index(drop=True);assert a.record_id.tolist()==b.record_id.tolist()
 dl=np.abs(a.y_pred-a.y_true)-np.abs(b.y_pred-b.y_true);part=pd.DataFrame({'g':a.chemical_group,'num':dl*a.weight,'den':a.weight}).groupby('g')[['num','den']].sum();num=part.num.sum();den=part.den.sum();loo=(num-part.num)/(den-part.den)
 influence.append({'budget':n,'comparison':'LoRA minus head-only','official_weighted_delta':num/den,'leave_one_group_out_min':loo.min(),'leave_one_group_out_max':loo.max(),'descriptive_posthoc':True})
 for st,mask in [('lt4',a.y_true<4),('4to5',(a.y_true>=4)&(a.y_true<5)),('5to6',(a.y_true>=5)&(a.y_true<6)),('ge6',a.y_true>=6)]:
  wn=a.loc[mask,'weight'].sum();local=np.dot(dl[mask],a.loc[mask,'weight'])/wn;contrib.append({'budget':n,'stratum':st,'n':int(mask.sum()),'weight_share':wn/den,'delta_weighted_mae':local,'contribution_to_overall_delta':local*wn/den})
pd.DataFrame(influence).to_csv(H/'influence_diagnostics.csv',index=False);pd.DataFrame(contrib).to_csv(H/'potency_contributions.csv',index=False)
# Verify nested acquisition and explicitly changed calibration roles.
lock=json.loads((R/'artifacts/cloud/50146714/remote_results/stage/manifest.json').read_text());rolecheck=[]
for fold in range(5):
 small=lock['roles'][f'fold{fold}__draw0__n100__head_only__seed42'];large=lock['roles'][f'fold{fold}__draw0__n500__head_only__seed42'];assert set(small['fit']+small['calibration'])<=set(large['fit']+large['calibration']);assert set(small['calibration'])<=set(large['fit']);rolecheck.append({'fold':fold,'small_calibration_becomes_large_fit':len(small['calibration'])})
(H/'role_nesting.json').write_text(json.dumps(rolecheck,indent=2))
# A small Markdown renderer supports exactly the canonical document's plain sections and table.
def slug(s):return re.sub(r'[^a-z0-9]+','-',s.lower()).strip('-')
def md(text):
 lines=text.splitlines();out=[];para=[];i=0
 def flush():
  if para:out.append('<p>'+html.escape(' '.join(para))+'</p>');para.clear()
 while i<len(lines):
  line=lines[i]
  if line.startswith('#'):
   flush();level=len(line)-len(line.lstrip('#'));title=line[level:].strip();out.append(f'<h{level} id="{slug(title)}">{html.escape(title)}</h{level}>')
  elif line.startswith('|'):
   flush();rows=[]
   while i<len(lines) and lines[i].startswith('|'):
    cells=[c.strip() for c in lines[i].strip('|').split('|')]
    if not all(re.fullmatch(r'[-: ]+',c) for c in cells):rows.append(cells)
    i+=1
   out.append('<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows[1:])+'</tbody></table></div>');continue
  elif not line.strip():flush()
  else:para.append(line)
  i+=1
 flush();return '\n'.join(out)
from proximity_figure import build as build_proximity_figure
build_proximity_figure(H,save)
from size_charge_figures import build as build_size_charge_figures
build_size_charge_figures(H,save)
build_size_charge_figures(H,save,movement=True)
pd.DataFrame(AUDIT).to_csv(H/'mw_marker_audit.csv',index=False)
assert len(AUDIT)==38 and sum(a['n'] for a in AUDIT)==3344*26
body=md(README.read_text())
def figure(name,caption):
 if name != 'paired-adaptation-effect':caption += ' Medium-blue open circles: molecular weight <215 Da; other compounds retain their original marks.'
 return f'<figure><img src="{name}.png" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption><p class="exports"><a href="{name}.svg" download>SVG</a> · <a href="{name}.png" download>PNG</a></p></figure>'
body=body.replace('<h2 id="absolute-performance">',figure('paired-adaptation-effect','Negative differences favor LoRA. Gray dots: individual folds. Orange diamond: pooled effect with a retrospective 95% paired chemical-group bootstrap interval. Open diamond: equal-fold mean, not an independent estimate.')+'<h2 id="absolute-performance">')
body=body.replace('<h2 id="are-the-biggest-changes-near-the-fitting-chemistry">',figure('prediction-compression','All 3,344 held-out compounds in each panel, no clipping. Shared square axes and a reference diagonal; no fitted correction. Native scores are the unchanged IC50-equivalent conversion. Overlapping dots are not omitted data.')+'<details><summary>Control predictions, including every extreme value</summary>'+figure('control-predictions-full-range','Controls use a broader shared axis range than the neural panels so extreme predictions remain visible. All compounds remain in the official metrics.')+'</details><h2 id="are-the-biggest-changes-near-the-fitting-chemistry">')
body=body.replace('<h3 id="large-changes-can-overshoot">',figure('change-versus-training-similarity','Gray dots: all 3,344 held-out compounds per panel. Orange rings: the 20 largest absolute native-relative changes in that panel. Both rows and columns share linear axes; every point is within view. Similarity is nearest actual FIT Morgan radius-2/2,048-bit Tanimoto, without chirality; calibration compounds are excluded. The split excludes similarity ≥0.35. Magnitude does not show direction or whether error improved; examples below retain both. These are paired compounds and reused development folds, not independent refits.')+'<p class="exports"><a href="proximity_top20.csv" download>Top 20 compounds and nearest neighbors (CSV)</a> · <a href="proximity_all.csv" download>All compounds (CSV)</a></p><h3 id="large-changes-can-overshoot">')
body=body.replace('<img src="change-versus-training-similarity.png"', '<div class="proximity-plot"><img src="change-versus-training-similarity.png"')
start=body.index('<div class="proximity-plot">');end=body.index('<figcaption>',start);body=body[:end]+'</div>'+body[end:]
body=body.replace('<h3 id="does-the-size-association-survive-adjustment">',figure('error-change-versus-size','Each panel contains all 3,344 held-out compounds. X = |fitted − truth| − |native − truth|; left of zero improves absolute error, right of zero worsens it. The vertical line marks no change. Y is molecular weight calculated from the identity SMILES, not a distance. Shared linear axes retain all values. Scroll horizontally on small screens; SVG and PNG exports preserve full resolution.')+'<details><summary>Separate prepared charge and observed potency: N100 and N500</summary>'+''.join(figure(f'error-change-stratified-n{n}',f'N{n}: columns separate net-neutral/charged prepared states and observed pEC50 below/at least 4; rows compare head-only and LoRA. Each method partitions the same 3,344 compounds. All panels share the main figure axes. Counts describe compounds, not independent training runs.') for n in [100,500])+'</details><p class="exports"><a href="size_charge_points.csv" download>All error changes and chemistry (CSV)</a> · <a href="size_charge_strata.csv" download>Crossed-group summaries (CSV)</a></p><h3 id="does-the-size-association-survive-adjustment">')
body=body.replace('<h3 id="does-the-size-association-survive-adjustment">',figure('error-change-versus-movement','X = |fitted − truth| − |native − truth|: negative improves, positive worsens absolute error. Y = |fitted − native|, the absolute change from native. Both axes use log10 prediction units. All 3,344 compounds appear in each panel. The molecular-weight plots above are retained.')+'<h3 id="does-the-size-association-survive-adjustment">')
for name in ['error-change-versus-movement','error-change-versus-size','error-change-stratified-n100','error-change-stratified-n500']:
 marker=f'<img src="{name}.png"';body=body.replace(marker,'<div class="error-plot">'+marker);start=body.index('<div class="error-plot">',body.index(marker)-24);end=body.index('<figcaption>',start);body=body[:end]+'</div>'+body[end:]
exports=['pooled_metrics.csv','fold_metrics.csv','paired_differences.csv','stratum_metrics.csv','novelty_metrics.csv','training_history.csv','heldout_predictions.csv','potency_contributions.csv','influence_diagnostics.csv','sources.csv','cohort.json','role_nesting.json','analysis_protocol.json','verification.json','bootstrap_draws.npz','analyze.py','render_report.py','design_review.md']
exports += ['verify_report.py','independent_audit/README.md','independent_audit/audit.py','independent_audit/results.json','independent_audit/pooled_metrics.csv','independent_audit/input_hashes.json']
exports += ['proximity.py','proximity_protocol.json','proximity_verification.json','proximity_all.csv','proximity_top20.csv','proximity_summary.csv','proximity_correlations.csv','proximity_bins.csv','PROXIMITY_RESULTS.txt','proximity_figure.py','proximity_figure_points.csv','proximity_figure_verification.json','proximity_design_review.md']
exports += ['size_charge_analysis.py','size_charge_protocol.json','size_charge_points.csv','size_charge_correlations.csv','size_charge_strata.csv','size_charge_verification.json','size_charge_figures.py','size_charge_figure_verification.json','mover_chemistry.py','mover_chemistry_summary.csv','mover_chemistry_all.csv','mover_chemistry_examples.png']
exports += ['mw_markers.py','mw_marker_audit.csv']
exports += ['verify_size_charge.py','size_charge_acceptance.json','size_charge_independent_review.json','size_charge_independent_review.txt']
exports += ['internal_feature_diagnostic/'+n for n in ['analyze.py','verify.py','summary.json','verification.json','input_provenance.json','output_sha256.json','native_feature_scores.csv','associations.csv','adaptation_feature_rows.csv','pca_variance.csv','pca_loadings.csv','pca_center.csv','stratum_medians.csv']]
for name in exports:
 (OUT/name).parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(H/name,OUT/name)
shutil.copy2(README,OUT/'RESULTS.md')
co=json.loads((H/'cohort.json').read_text());body+='<details><summary>Full numerical downloads and provenance</summary><ul>'+''.join(f'<li><a href="{html.escape(n)}" download>{html.escape(n)}</a></li>' for n in ['RESULTS.md',*exports])+'</ul></details>'
style='''*{box-sizing:border-box}body{margin:0;background:white;color:#26323b;font:15px/1.65 system-ui,-apple-system,Segoe UI,sans-serif;overflow-wrap:anywhere}main{max-width:1120px;margin:auto;padding:32px 28px 70px}h1{font-size:32px;line-height:1.2;font-weight:600;letter-spacing:-.6px}h2{font-size:22px;line-height:1.3;margin-top:36px;font-weight:600}p{max-width:960px}nav{font-size:13px;border-bottom:1px solid #dce1e5;padding-bottom:14px;margin-bottom:24px;display:flex;gap:24px;flex-wrap:wrap}a{color:#2166ac;text-decoration:none}a:hover{text-decoration:underline}figure{margin:26px 0 36px}img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#55646f}.exports{font-size:13px;margin:6px 0}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th{text-align:left;font-weight:600;border-bottom:1px solid #aab5bd;padding:9px 10px}td{padding:7px 10px;border-bottom:1px solid #e5e9ec}td:nth-child(n+3){font-variant-numeric:tabular-nums;text-align:right}tbody tr:nth-child(6) td{border-top:1px solid #8a969f}details{margin:24px 0;border-top:1px solid #dce1e5;padding-top:14px}summary{cursor:pointer;font-weight:600}footer{font-size:12px;color:#657480;border-top:1px solid #dce1e5;padding-top:20px;margin-top:40px}@media(max-width:600px){main{padding:20px 14px}h1{font-size:27px}h2{font-size:20px}body{font-size:14px}}@media print{nav,.exports,details{display:none}main{padding:0;max-width:none}h2{break-after:avoid}figure{break-inside:avoid}}'''
style+=' .proximity-plot{max-width:760px;overflow:auto}.proximity-plot img{min-width:640px} h3{font-size:17px;margin-top:26px}.error-plot{overflow:auto;max-width:900px}.error-plot img{min-width:760px}'
text=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nesso adaptation — results</title><style>{style}</style></head><body><main><nav><a href="../">Architecture</a><a href="#answer">Answer</a><a href="#absolute-performance">Scores</a><a href="#the-improvement-is-not-uniform">Potency trade-offs</a><a href="#are-the-biggest-changes-near-the-fitting-chemistry">Training similarity</a><a href="#size-charge-and-distance-from-truth">Error changes</a><a href="#decision">Decision</a><a href="RESULTS.md" download>Editable source</a></nav>{body}<footer>Read-only analysis of saved development-fold predictions. No new models fitted. Report generated {datetime.datetime.now(datetime.timezone.utc).isoformat()}. Canonical RESULTS.md SHA256: {hashlib.sha256(README.read_bytes()).hexdigest()}.</footer></main></body></html>'
(OUT/'index.html').write_text(text)
allowed=['index.html','RESULTS.md',*exports,*[f'{x}.{ext}' for x in figures for ext in ['svg','png']]]
manifest={'status':'rendered_pending_browser_check','figures':figures,'canonical_source':str(README.relative_to(R)),'canonical_sha256':hashlib.sha256(README.read_bytes()).hexdigest(),'files':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in allowed},'new_predictive_fits':0}
(OUT/'report_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'output':str(OUT/'index.html'),'files':len(allowed),'figures':len(figures),'influence':influence,'role_checks':rolecheck},indent=2))
