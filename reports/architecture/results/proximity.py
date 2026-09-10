from pathlib import Path
import json,hashlib,math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem,DataStructs,rdBase
from rdkit.Chem import rdFingerprintGenerator
H=Path(__file__).resolve().parent;R=H.parents[2];T=R/'artifacts/cloud/50146714/remote_results';S=R/'artifacts/experiments/low_data_followup_20260906/strict_split'
inputs={}
def load(p,csv=False):
 inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return pd.read_csv(p) if csv else json.loads(p.read_text())
lock=load(T/'stage/manifest.json');ids=load(S/'identities.csv',True).set_index('record_id');labels=load(R/lock['protocol']['labels'],True).set_index('record_id');near=load(S/'test_nearest_similarity.csv',True).set_index('record_id')
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
fps={i:gen.GetFingerprint(Chem.MolFromSmiles(s)) for i,s in ids.canonical_smiles.items()}
rows=[]
for n in [100,500]:
 for f in range(5):
  tid=f'fold{f}__draw0__n{n}__head_only__seed42';roles=lock['roles'][tid];fit=roles['fit'];cal=roles['calibration'];test=roles['test']
  assert len(fit)==int(.8*n) and not set(test)&set(fit+cal)
  frames={}
  for arm in ['head_only','head_plus_lora']:
   key=f'fold{f}__draw0__n{n}__{arm}__seed42';assert lock['roles'][key]==roles
   frames[arm]=load(T/'tasks'/key/'test.csv',True).set_index('record_id').loc[test]
  base=load(T/'tasks'/tid/'baseline.csv',True).set_index('record_id').loc[test]
  for i in test:
   sim=np.array(DataStructs.BulkTanimotoSimilarity(fps[i],[fps[j] for j in fit]));cs=np.array(DataStructs.BulkTanimotoSimilarity(fps[i],[fps[j] for j in cal]));j=fit[int(sim.argmax())]
   assert max(sim.max(),cs.max())<.35 and max(sim.max(),cs.max())<=near.loc[i,'nearest_eligible_similarity']+1e-12
   y=float(frames['head_only'].loc[i,'y_true']);b=float(base.loc[i,'y_pred']);p=float(frames['head_only'].loc[i,'y_pred']);q=float(frames['head_plus_lora'].loc[i,'y_pred'])
   rows.append(dict(budget=n,fold=f,record_id=i,smiles=ids.loc[i,'canonical_smiles'],y_true=y,native=b,head_only=p,head_plus_lora=q,weight=float(frames['head_only'].loc[i,'weight']),fit_similarity=float(sim.max()),budget_similarity=float(max(sim.max(),cs.max())),nearest_fit_id=j,nearest_fit_smiles=ids.loc[j,'canonical_smiles'],nearest_fit_y=float(labels.loc[j,'pEC50'])))
d=pd.DataFrame(rows);assert len(d)==6688 and not d.duplicated(['budget','record_id']).any();d.to_csv(H/'proximity_all.csv',index=False)
summ=[];tops=[];bins=[];folds=[]
for n,x in d.groupby('budget'):
 for arm,ref in [('head_only','native'),('head_plus_lora','native'),('head_plus_lora','head_only')]:
  z=x.copy();z['comparison']=arm+'-minus-'+ref;z['change']=z[arm]-z[ref];z['abs_change']=z.change.abs();z['error_improvement']=(z[ref]-z.y_true).abs()-(z[arm]-z.y_true).abs();z['closer_to_nearest_label']=(z[arm]-z.nearest_fit_y).abs()<(z[ref]-z.nearest_fit_y).abs();z=z.sort_values(['abs_change','record_id'],ascending=[False,True]);tops.append(z.head(20))
  def stat(a):
   return dict(n=len(a),fit_similarity_median=float(a.fit_similarity.median()),fit_similarity_min=float(a.fit_similarity.min()),fit_similarity_max=float(a.fit_similarity.max()),budget_similarity_median=float(a.budget_similarity.median()),mean_abs_change=float(a.abs_change.mean()),median_change=float(a.change.median()),improved_fraction=float((a.error_improvement>0).mean()),weighted_error_improvement=float(np.average(a.error_improvement,weights=a.weight)),toward_nearest_label_fraction=float(a.closer_to_nearest_label.mean()))
  for name,a in [('all',z),('top20',z.head(20)),('top_decile',z.head(math.ceil(len(z)*.1)))]:summ.append(dict(budget=int(n),comparison=arm+'-minus-'+ref,selection=name,**stat(a)))
  for f,a in [('pooled',z),*list(z.groupby('fold'))]:folds.append(dict(budget=int(n),comparison=arm+'-minus-'+ref,fold=str(f),rho_abs_change_similarity=float(spearmanr(a.abs_change,a.fit_similarity).statistic)))
  for lo,hi in [(0,.2),(.2,.25),(.25,.3),(.3,.35)]:
   a=z[(z.fit_similarity>=lo)&(z.fit_similarity<hi)]
   if len(a):bins.append(dict(budget=int(n),comparison=arm+'-minus-'+ref,lo=lo,hi=hi,**stat(a)))
for name,a in [('proximity_summary',summ),('proximity_correlations',folds),('proximity_bins',bins)]:
 pd.DataFrame(a).to_csv(H/(name+'.csv'),index=False)
 (H/(name+'.json')).write_text(json.dumps(a,indent=2))
pd.concat(tops).to_csv(H/'proximity_top20.csv',index=False)
(H/'proximity_verification.json').write_text(json.dumps(dict(status='PASS',rows=len(d),unique_test_compounds=int(d.record_id.nunique()),rdkit=rdBase.rdkitVersion,maximum_fit_similarity=float(d.fit_similarity.max()),maximum_budget_similarity=float(d.budget_similarity.max()),inputs=inputs,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print(pd.DataFrame(summ).to_string(index=False));print(pd.DataFrame(folds).to_string(index=False));print(pd.DataFrame(bins).to_string(index=False));print(pd.concat(tops).groupby(['budget','comparison']).head(3)[['budget','comparison','record_id','y_true','native','head_only','head_plus_lora','change','fit_similarity','nearest_fit_id','nearest_fit_y']].to_string(index=False))
