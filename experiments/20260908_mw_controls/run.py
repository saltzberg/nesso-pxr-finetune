"""Fixed CPU WLS controls. Original FIT/CAL/test roles; no tuning."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib,math
import numpy as np,pandas as pd
from scipy.stats import spearmanr
R=Path('/home/dan/projects/nesso-finetune/PXR');H=R/'artifacts/analysis/mw_controls_20260908';H.mkdir(parents=True,exist_ok=True)
P=R/'experiments/20260908_mw_controls/protocol.json';A=R/'artifacts/analysis/affinity_comparison_20260907';F=R/'artifacts/experiments/affinity_comparison_20260906/controls_v1/fresh_features';T=R/'artifacts/cloud/50146714/remote_results'
inputs={}
def bind(p):inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return pd.read_csv(bind(p),float_precision='round_trip',dtype={'record_id':str})
protocol=json.loads(bind(P).read_text());lock=json.loads(bind(T/'stage/manifest.json').read_text());meta=json.loads(bind(F/'manifest.json').read_text())
for name,h in meta['outputs'].items():assert hashlib.sha256(bind(F/name).read_bytes()).hexdigest()==h
ids=read(F/'identities.csv');assert ids.record_id.is_unique
native=np.load(bind(F/'native_points.npy'));assert native.shape==(3344,) and np.isfinite(native).all()
d=read(R/lock['protocol']['labels']).set_index('record_id').loc[ids.record_id].copy();d['native']=native
chem=read(A/'mover_chemistry_all.csv').set_index('record_id');d['MW']=chem.loc[d.index,'MW'];d['w']=1/np.maximum(d.pEC50_standard_error,.1)
assert len(d)==3344 and d.index.is_unique and np.isfinite(d[['pEC50','native','MW','w']]).all().all()
models=['mw','native','native_mw','native_mw_interaction'];rows=[];states=[];checks=[]
def design(z,mean,scale,model):
 v=(z[['native','MW']].to_numpy()-mean)/scale
 if model=='mw':return np.column_stack([np.ones(len(z)),v[:,1]])
 if model=='native':return np.column_stack([np.ones(len(z)),v[:,0]])
 if model=='native_mw':return np.column_stack([np.ones(len(z)),v])
 return np.column_stack([np.ones(len(z)),v,v[:,0]*v[:,1]])
for n in [100,500]:
 for fold in range(5):
  roles=lock['roles'][f'fold{fold}__draw0__n{n}__head_only__seed42'];assert roles==lock['roles'][f'fold{fold}__draw0__n{n}__head_plus_lora__seed42']
  fi,ca,te=[d.loc[roles[k]] for k in ['fit','calibration','test']]
  assert len(fi)==int(.8*n) and len(ca)==n-len(fi) and not set(fi.index)&set(ca.index) and not set(te.index)&set(fi.index)|set(te.index)&set(ca.index)
  mean=fi[['native','MW']].mean().to_numpy();scale=fi[['native','MW']].std(ddof=0).to_numpy();assert (scale>0).all()
  for model in models:
   cell=H/'cells'/f'n{n}_fold{fold}_{model}';cell.mkdir(parents=True,exist_ok=True)
   X=design(fi,mean,scale,model);sqrtw=np.sqrt(fi.w.to_numpy());wx=X*sqrtw[:,None];wy=fi.pEC50.to_numpy()*sqrtw
   beta,res,rank,singular=np.linalg.lstsq(wx,wy,rcond=None);assert rank==X.shape[1]
   b2=np.linalg.solve(wx.T@wx,wx.T@wy)
   q=design(te,mean,scale,model);pred=q@beta;assert np.isfinite(pred).all();err=float(np.max(abs(q@b2-pred)));assert err<1e-10
   cp=design(ca,mean,scale,model)@beta;resid=np.sort(abs(cp-ca.pEC50.to_numpy()));radii={str(c):float(resid[math.ceil((len(ca)+1)*c/100)-1]) for c in [80,90]}
   state=dict(model=model,budget=n,fold=fold,mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),roles=roles,radii=radii,protocol_sha256=inputs[str(P)],columns=['native','MW'])
   (cell/'state.json').write_text(json.dumps(state,indent=2));loaded=json.loads((cell/'state.json').read_text());replay=design(te,np.array(loaded['mean']),np.array(loaded['scale']),loaded['model'])@np.array(loaded['beta']);assert np.array_equal(pred,replay)
   for role,z,pp in [('test',te,pred),('calibration',ca,cp)]:
    out=pd.DataFrame(dict(record_id=z.index,y_true=z.pEC50.to_numpy(),y_pred=pp,assay_se=z.pEC50_standard_error.to_numpy(),weight=z.w.to_numpy(),MW=z.MW.to_numpy(),native=z.native.to_numpy(),budget=n,fold=fold,method=model))
    for c in [80,90]:out[f'radius_{c}']=radii[str(c)]
    out.to_csv(cell/(role+'.csv'),index=False)
    if role=='test':rows.append(out)
   checks.append(dict(cell=cell.name,rank=int(rank),condition_number=float(singular.max()/singular.min()),normal_equation_max_error=err,serialized_replay_exact=True))
   states.append(state)
assert len(checks)==40
new=pd.concat(rows);assert new.groupby(['budget','method']).size().eq(3344).all() and not new.duplicated(['budget','method','record_id']).any()
refs=read(A/'heldout_predictions.csv');refs['MW']=chem.loc[refs.record_id,'MW'].to_numpy();allrows=pd.concat([new,refs],ignore_index=True)
metrics=[];foldmetrics=[]
def score(z):
 e=(z.y_pred-z.y_true).to_numpy();w=z.weight.to_numpy();out=dict(n=len(z),mae=float(np.mean(abs(e))),weighted_mae=float(np.average(abs(e),weights=w)),spearman=float(spearmanr(z.y_pred,z.y_true).statistic),bias=float(np.mean(e)),weight_sum=float(w.sum()))
 for c in [80,90]:out[f'coverage_{c}']=float(np.mean(abs(e)<=z[f'radius_{c}'].to_numpy()))
 return out
for (n,m),z in allrows.groupby(['budget','method']):
 for sub,mask in [('full',z.MW.notna()),('lt215',z.MW<215),('gt215',z.MW>215)]:
  a=z[mask];metrics.append(dict(budget=n,method=m,subset=sub,**score(a)))
  for fold,b in a.groupby('fold'):foldmetrics.append(dict(budget=n,method=m,subset=sub,fold=fold,**score(b)))
m=pd.DataFrame(metrics);fm=pd.DataFrame(foldmetrics);assert len(m)==54 and len(fm)==270
old=read(A/'pooled_metrics.csv');join=m[m.subset=='full'].merge(old,on=['budget','method'],suffixes=('_new','_old'));assert len(join)==10
for key in ['mae','weighted_mae','spearman']:assert np.max(abs(join[key+'_new']-join[key+'_old']))<1e-12
new.to_csv(H/'predictions.csv',index=False);m.to_csv(H/'metrics.csv',index=False);fm.to_csv(H/'fold_metrics.csv',index=False)
contrasts=[]
for n in [100,500]:
 for sub in ['full','lt215','gt215']:
  z=m[(m.budget==n)&(m.subset==sub)].set_index('method')
  for a,b in [('native_mw','mw'),('native_mw','native'),('native_mw_interaction','native_mw'),('head_only','native_mw'),('head_plus_lora','native_mw'),('head_plus_lora','native_mw_interaction')]:
   ff=fm[(fm.budget==n)&(fm.subset==sub)].pivot(index='fold',columns='method',values='weighted_mae');delta=ff[a]-ff[b]
   contrasts.append(dict(budget=n,subset=sub,model_a=a,model_b=b,weighted_mae_delta=float(z.loc[a,'weighted_mae']-z.loc[b,'weighted_mae']),mae_delta=float(z.loc[a,'mae']-z.loc[b,'mae']),fold_wins_for_a=int((delta<0).sum())))
pd.DataFrame(contrasts).to_csv(H/'contrasts.csv',index=False)
(H/'verification.json').write_text(json.dumps(dict(status='PASS',completed_fits=40,new_test_rows=len(new),metrics_rows=len(m),equal215=int((d.MW==215).sum()),source_hashes=inputs,checks=checks,reference_reconciliation='PASS',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print(m[['budget','method','subset','weighted_mae','mae','spearman']].to_string(index=False));print('\nCONTRASTS');print(pd.DataFrame(contrasts).to_string(index=False))
