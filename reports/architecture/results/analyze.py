"""Read-only, independently scored Nesso comparison. No torch or model fitting."""
from pathlib import Path
import json,hashlib,math,re,datetime,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
R=Path('/home/dan/projects/nesso-finetune/PXR');H=Path(__file__).resolve().parent
T=R/'artifacts/cloud/50146714/remote_results'
C=R/'artifacts/experiments/affinity_comparison_20260906/controls_v1'
S=R/'artifacts/experiments/low_data_followup_20260906/strict_split'
INPUTS={};checks=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p);INPUTS[str(p)]=digest(p);return p

def load(p):return json.loads(bind(p).read_text())
def csv(p):
 d=pd.read_csv(bind(p),dtype=str,keep_default_na=False)
 for c in ['y_true','y_pred','assay_se','weight','lower_80','upper_80','lower_90','upper_90','affinity_module','affinity_module2','nearest_eligible_similarity']:
  if c in d:d[c]=d[c].astype(float)
 return d

def score(d):
 y=d.y_true.to_numpy(float);p=d.y_pred.to_numpy(float);se=d.assay_se.to_numpy(float)
 assert len(d)>0 and np.isfinite([y,p,se]).all() and (se>=0).all()
 err=p-y;a=np.abs(err);w=1/np.maximum(se,.1)
 q=lambda x:float(np.quantile(x,.75)-np.quantile(x,.25))
 out={'n':len(d),'weight_sum':float(w.sum()),'effective_n':float(w.sum()**2/(w*w).sum()),'weighted_mae':float(np.dot(w,a)/w.sum()),'mae':float(a.mean()),'spearman':float(spearmanr(y,p).statistic) if np.ptp(y)>0 and np.ptp(p)>0 else None,'bias':float(err.mean()),'weighted_bias':float(np.dot(w,err)/w.sum()),'rmse':float(np.sqrt(np.mean(err**2))),'observed_mean':float(y.mean()),'predicted_mean':float(p.mean()),'observed_sd':float(y.std()),'predicted_sd':float(p.std()),'sd_ratio':float(p.std()/y.std()) if y.std()>0 else None,'observed_iqr':q(y),'predicted_iqr':q(p),'iqr_ratio':q(p)/q(y) if q(y)>0 else None,'prediction_min':float(p.min()),'prediction_max':float(p.max()),'observed_min':float(y.min()),'observed_max':float(y.max()),'calibration_slope_descriptive':float(np.dot(p-p.mean(),y-y.mean())/np.dot(p-p.mean(),p-p.mean())) if np.ptp(p)>0 else None}
 for floor in [.05,.2]:
  wf=1/np.maximum(se,floor);out['weighted_mae_floor_'+('005' if floor==.05 else '020')]=float(np.dot(wf,a)/wf.sum())
 for lev in [80,90]:
  if 'radius_'+str(lev) in d:
   rad=d['radius_'+str(lev)].to_numpy(float)
   out['coverage_'+str(lev)]=float(np.mean(a<=rad));out['width_'+str(lev)]=float(np.mean(2*rad))
 return out

def check_scores(actual,stored,name):
 for key in ['mae','weighted_mae','weighted_mae_floor_005','weighted_mae_floor_020','spearman','bias','effective_n']:
  if key in stored:
   assert actual[key] is not None and abs(actual[key]-stored[key])<1e-12,(name,key,actual[key],stored[key])
 checks.append(name)

protocol=load(H/'analysis_protocol.json');lock=load(T/'stage/manifest.json')
assert len(lock['tasks'])==20 and not lock['smoke']
ids=csv(S/'identities.csv');assign=csv(S/'assignments.csv')
identity=ids.merge(assign,on='row_index',validate='one_to_one').set_index('record_id')
assert len(identity)==3344 and identity.index.is_unique
assert identity.groupby('chemical_group').outer_fold.nunique().max()==1
near=csv(S/'test_nearest_similarity.csv').set_index('record_id')
assert near.index.is_unique and (near.nearest_eligible_similarity<.35).all()
LABELS=csv(R/lock['protocol']['labels']).set_index('record_id')
frames=[];folds=[];histories=[];baseline_diffs=[];sources=[]
def accept(d,n,fold,method,radii):
 d=d.copy();assert d.record_id.is_unique
 assert set(d.record_id)==set(identity.index[identity.outer_fold==str(fold)])
 assert np.array_equal(d.y_true.to_numpy(),LABELS.loc[d.record_id,'pEC50'].astype(float).to_numpy())
 assert np.array_equal(d.assay_se.to_numpy(),LABELS.loc[d.record_id,'pEC50_standard_error'].astype(float).to_numpy())
 assert np.array_equal(d.weight.to_numpy(),1/np.maximum(d.assay_se.to_numpy(),.1))
 d['budget']=n;d['fold']=fold;d['method']=method
 d['chemical_group']=identity.loc[d.record_id,'chemical_group'].to_numpy()
 d['nearest_eligible_similarity']=near.loc[d.record_id,'nearest_eligible_similarity'].to_numpy()
 for lev in [80,90]:d['radius_'+str(lev)]=radii[str(lev)]
 a=score(d);folds.append(dict(budget=n,fold=fold,method=method,**a));frames.append(d);return a

for task in lock['tasks']:
 tid=task['task_id'];p=T/'tasks'/tid;n=task['n_train'];fold=task['outer_fold'];method=task['arm'];marker=load(p/'complete.json')
 assert marker['task']==task and marker['binding']==lock['binding']
 for name,h in marker['files'].items():assert digest(bind(p/name))==h,(tid,name)
 assert digest(bind(p/'epochs/epoch040/state.pt'))==marker['state_sha256']
 d=csv(p/'test.csv');cal=csv(p/'calibration.csv');roles=lock['roles'][tid]
 assert list(d.record_id)==roles['test'] and list(cal.record_id)==roles['calibration']
 assert len(roles['fit'])==int(.8*n) and len(roles['calibration'])==n-int(.8*n)
 assert not set(roles['fit'])&set(roles['test']) and not set(roles['calibration'])&set(roles['test']) and not set(roles['fit'])&set(roles['calibration'])
 audit=load(p/'audit.json');assert audit['frozen_before']==audit['frozen_after'] and len(audit['history'])==40
 assert all(audit['intended_before'][k]!=v for k,v in audit['intended_after'].items())
 for row in audit['history']:histories.append({'task':tid,'budget':n,'fold':fold,'method':method,'epoch':row['epoch'],'weighted_huber':row['weighted_huber']})
 m=load(p/'metrics.json');radii={k:v['radius'] if v['finite'] else math.inf for k,v in m['intervals'].items()}
 a=accept(d,n,fold,method,radii);check_scores(a,m,tid)
 for lev in [80,90]:
  rank=math.ceil((len(cal)+1)*lev/100);errors=np.sort(np.abs(cal.y_pred.to_numpy()-cal.y_true.to_numpy()));rr=errors[rank-1] if rank<=len(cal) else math.inf
  assert rr==radii[str(lev)]
  assert a['coverage_'+str(lev)]==m['intervals'][str(lev)]['coverage']
 sources.append({'task_id':tid,'method':method,'budget':n,'fold':fold,'path':str(p),'origin':'preserved neural fit; 3 historical local fits, 17 remote continuation fits across matrix'})

for p in sorted((C/'cells').iterdir()):
 if not p.is_dir():continue
 match=re.fullmatch(r'strict_chemical__fold(\d)__draw00__n(100|500)__(.+)',p.name);assert match,p
 fold,n,method=int(match[1]),int(match[2]),match[3]
 marker=load(p/'complete.json')
 for name,h in marker['outputs'].items():assert digest(bind(p/name))==h,(p,name)
 roles=load(p/'roles.json');troles=lock['roles'][f'fold{fold}__draw0__n{n}__head_only__seed42']
 for role in ['fit','calibration','test']:assert roles[role]['record_id']==troles[role],(p,role)
 d=csv(p/'test_predictions.csv');m=load(p/'metrics.json');a=accept(d,n,fold,method,m['radii']);check_scores(a,m['test'],p.name)
 for lev in [80,90]:assert a['coverage_'+str(lev)]==m['test']['coverage_'+str(lev)]
 if method=='native_continuous':
  native=d.set_index('record_id').y_pred
  for arm in ['head_only','head_plus_lora']:
   b=csv(T/'tasks'/f'fold{fold}__draw0__n{n}__{arm}__seed42'/'baseline.csv').set_index('record_id').loc[d.record_id]
   diff=float(np.max(np.abs(b.y_pred.to_numpy()-native.to_numpy())));assert diff<=1e-4
   baseline_diffs.append({'budget':n,'fold':fold,'arm':arm,'max_abs':diff})
 sources.append({'task_id':p.name,'method':method,'budget':n,'fold':fold,'path':str(p),'origin':'reused historical fit' if method.startswith('descriptor') else 'fresh native capture' if method=='native_continuous' else 'fresh fitted ridge'})
assert len(frames)==50 and len(folds)==50
allrows=pd.concat(frames,ignore_index=True);foldmetrics=pd.DataFrame(folds)
assert not allrows.duplicated(['budget','method','record_id']).any()
assert allrows.groupby(['budget','method']).size().eq(3344).all()
methods=['native_continuous','head_only','head_plus_lora','repr_ridge_stable','descriptor_lightgbm_rdkit_mordred']
for method in methods:
 for n in [100,500]:assert (foldmetrics[(foldmetrics.method==method)&(foldmetrics.budget==n)].fold.nunique()==5)
na=allrows[(allrows.method=='native_continuous')&(allrows.budget==100)].sort_values('record_id');nb=allrows[(allrows.method=='native_continuous')&(allrows.budget==500)].sort_values('record_id');assert np.array_equal(na.y_pred.to_numpy(),nb.y_pred.to_numpy())
pooled=[];strata=[];novelty=[]
for (n,method),d in allrows.groupby(['budget','method']):
 fm=foldmetrics[(foldmetrics.budget==n)&(foldmetrics.method==method)]
 pooled.append({'budget':n,'method':method,**score(d),'macro_weighted_mae':float(fm.weighted_mae.mean()),'macro_mae':float(fm.mae.mean()),'macro_spearman':float(fm.spearman.mean())})
 for name,mask in [('lt4',d.y_true<4),('4to5',(d.y_true>=4)&(d.y_true<5)),('5to6',(d.y_true>=5)&(d.y_true<6)),('ge6',d.y_true>=6),('ge4',d.y_true>=4)]:
  if mask.any():strata.append({'budget':n,'method':method,'stratum':name,**score(d[mask])})
 for lo,hi in [(0,.2),(.2,.3),(.3,.35)]:
  sub=d[(d.nearest_eligible_similarity>=lo)&(d.nearest_eligible_similarity<hi)]
  if len(sub):novelty.append({'budget':n,'method':method,'similarity_lower':lo,'similarity_upper':hi,**score(sub)})
# Paired cluster bootstrap: one set of group draws reused for every arm and budget.
base=na.sort_values('record_id').reset_index(drop=True);groups=sorted(base.chemical_group.unique());gmap={g:i for i,g in enumerate(groups)};gi=base.chemical_group.map(gmap).to_numpy();G=len(groups);B=protocol['bootstrap']['replicates'];rng=np.random.default_rng(protocol['bootstrap']['seed']);draws=np.zeros((B,G),dtype=np.int16)
for fold in range(5):
 gidx=np.array([gmap[g] for g in sorted(base[base.fold==fold].chemical_group.unique())]);draws[:,gidx]=rng.multinomial(len(gidx),np.full(len(gidx),1/len(gidx)),size=B)
np.savez_compressed(H/'bootstrap_draws.npz',counts=draws,chemical_groups=np.array(groups))
weights=base.weight.to_numpy();gw=np.bincount(gi,weights=weights,minlength=G);gn=np.bincount(gi,minlength=G);dw=draws@gw;dn=draws@gn
comparisons=[('head_plus_lora','head_only'),('head_only','native_continuous'),('head_plus_lora','native_continuous'),('head_only','repr_ridge_stable'),('head_plus_lora','repr_ridge_stable'),('head_only','descriptor_lightgbm_rdkit_mordred'),('head_plus_lora','descriptor_lightgbm_rdkit_mordred')]
paired=[]
for n in [100,500]:
 by={m:allrows[(allrows.budget==n)&(allrows.method==m)].sort_values('record_id').reset_index(drop=True) for m in methods}
 for a,b in comparisons:
  assert by[a].record_id.tolist()==by[b].record_id.tolist()==base.record_id.tolist()
  delta=np.abs(by[a].y_pred.to_numpy()-base.y_true.to_numpy())-np.abs(by[b].y_pred.to_numpy()-base.y_true.to_numpy())
  wboot=(draws@np.bincount(gi,weights=weights*delta,minlength=G))/dw;boot=(draws@np.bincount(gi,weights=delta,minlength=G))/dn
  fda=foldmetrics[(foldmetrics.budget==n)&(foldmetrics.method==a)].set_index('fold');fdb=foldmetrics[(foldmetrics.budget==n)&(foldmetrics.method==b)].set_index('fold');fd=fda.weighted_mae-fdb.weighted_mae
  paired.append({'budget':n,'model_a':a,'model_b':b,'weighted_mae_delta':float(np.dot(weights,delta)/weights.sum()),'weighted_ci_low':float(np.quantile(wboot,.025)),'weighted_ci_high':float(np.quantile(wboot,.975)),'mae_delta':float(delta.mean()),'mae_ci_low':float(np.quantile(boot,.025)),'mae_ci_high':float(np.quantile(boot,.975)),'fold_wins_for_a':int((fd<0).sum()),'fold_deltas':[float(fd.loc[f]) for f in range(5)]})
# Budget changes are paired on the same held-out compounds, not extra observations.
for method in methods[1:]:
 a=allrows[(allrows.budget==500)&(allrows.method==method)].sort_values('record_id');b=allrows[(allrows.budget==100)&(allrows.method==method)].sort_values('record_id');delta=np.abs(a.y_pred.to_numpy()-base.y_true.to_numpy())-np.abs(b.y_pred.to_numpy()-base.y_true.to_numpy());boot=(draws@np.bincount(gi,weights=weights*delta,minlength=G))/dw
 paired.append({'budget':'500-minus-100','model_a':method,'model_b':method,'weighted_mae_delta':float(np.dot(weights,delta)/weights.sum()),'weighted_ci_low':float(np.quantile(boot,.025)),'weighted_ci_high':float(np.quantile(boot,.975))})
# Fold/group composition, not an assertion that all clusters are substantial series.
gs=base.groupby('chemical_group').size();composition={'compounds':len(base),'chemical_groups':len(gs),'singleton_groups':int((gs==1).sum()),'singleton_compounds':int(gs[gs==1].sum()),'largest_group':int(gs.max()),'fold_sizes':{str(k):int(v) for k,v in base.groupby('fold').size().items()},'effective_weighted_n':score(base)['effective_n'],'native_baseline_max_abs_vs_capture':max(x['max_abs'] for x in baseline_diffs),'neurals':20,'control_cells':30,'metric_cells_recomputed':len(checks)}
for name,data in [('pooled_metrics',pooled),('fold_metrics',folds),('stratum_metrics',strata),('novelty_metrics',novelty),('training_history',histories),('paired_differences',paired),('sources',sources)]:
 pd.DataFrame(data).to_csv(H/(name+'.csv'),index=False)
 (H/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
allrows.to_csv(H/'heldout_predictions.csv',index=False)
(H/'cohort.json').write_text(json.dumps(composition,indent=2))
(H/'verification.json').write_text(json.dumps({'status':'PASS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fit_count':20,'control_count':30,'metric_cells_recomputed':len(checks),'method_budget_groups':10,'rows':len(allrows),'unique_test_compounds':len(base),'primary_metric_tolerance':1e-12,'native_baseline_diffs':baseline_diffs,'input_sha256':INPUTS,'analysis_code_sha256':digest(Path(__file__))},indent=2))
print(pd.DataFrame(pooled)[['budget','method','weighted_mae','mae','spearman','sd_ratio','coverage_80','coverage_90']].to_string(index=False));print('\nPAIRED');print(pd.DataFrame(paired).drop(columns=['fold_deltas'],errors='ignore').to_string(index=False));print('\nCOHORT',json.dumps(composition))
