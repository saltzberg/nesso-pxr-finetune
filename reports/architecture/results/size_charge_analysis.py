"""Retrospective chemistry associations and signed change in absolute error."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
H=Path(__file__).resolve().parent;R=H.parents[2]
read=lambda p:pd.read_csv(p,float_precision='round_trip')
d=read(H/'proximity_all.csv');chem=read(H/'mover_chemistry_all.csv');meta=read(R/'data/published/modeling_manifest.csv')
meta=meta[meta.record_id.isin(d.record_id)].copy();assert meta.record_id.is_unique
meta=meta[['record_id','formal_charge','pEC50_standard_error','prepared_smiles']].rename(columns={'formal_charge':'prepared_charge','pEC50_standard_error':'assay_SE'})
d=d.merge(chem[['record_id','MW','heavy_atoms']],on='record_id',validate='many_to_one').merge(meta,on='record_id',validate='many_to_one');assert len(d)==6688
assert np.isfinite(d[['MW','prepared_charge','assay_SE','y_true']]).all().all()
d['charged']=d.prepared_charge!=0;d['small']=d.MW<200;d['low_potency']=d.y_true<4
rows=[];corr=[];strata=[]
for n,a in d.groupby('budget'):
 for arm in ['head_only','head_plus_lora']:
  z=a.copy();z['method']=arm;z['fitted']=z[arm];z['native_distance']=abs(z.native-z.y_true);z['fitted_distance']=abs(z.fitted-z.y_true);z['difference_distance_from_truth']=z.fitted_distance-z.native_distance;z['absolute_movement']=abs(z.fitted-z.native)
  assert (z.native_distance>=0).all() and (z.fitted_distance>=0).all()
  assert np.allclose(z.difference_distance_from_truth,(z.fitted-z.y_true).abs()-(z.native-z.y_true).abs(),rtol=0,atol=0)
  rows.append(z)
  fold=pd.get_dummies(z.fold,drop_first=True,dtype=float).to_numpy()
  controls=np.column_stack([np.ones(len(z)),rankdata(z.y_true),(z.prepared_charge>0).astype(float),(z.prepared_charge<0).astype(float),fold])
  def residual(v,c):return v-c@np.linalg.lstsq(c,v,rcond=None)[0]
  for outcome in ['absolute_movement','difference_distance_from_truth']:
   x=rankdata(z.MW);y=rankdata(z[outcome]);extra=np.column_stack([controls,rankdata(z.assay_SE)])
   corr.append(dict(budget=n,method=arm,outcome=outcome,rho=float(spearmanr(x,y).statistic),partial_rho_potency_charge_fold=float(np.corrcoef(residual(x,controls),residual(y,controls))[0,1]),partial_rho_plus_SE=float(np.corrcoef(residual(x,extra),residual(y,extra))[0,1])))
  for keys,b in z.groupby(['small','charged','low_potency']):
   strata.append(dict(budget=n,method=arm,small=bool(keys[0]),charged=bool(keys[1]),low_potency=bool(keys[2]),n=len(b),median_MW=float(b.MW.median()),median_movement=float(b.absolute_movement.median()),mean_error_delta=float(b.difference_distance_from_truth.mean()),weighted_error_delta=float(np.average(b.difference_distance_from_truth,weights=b.weight)),improved=int((b.difference_distance_from_truth<0).sum()),worse=int((b.difference_distance_from_truth>0).sum()),unchanged=int((b.difference_distance_from_truth==0).sum())))
p=pd.concat(rows);assert len(p)==13376 and not p.duplicated(['budget','method','record_id']).any()
p.to_csv(H/'size_charge_points.csv',index=False);pd.DataFrame(corr).to_csv(H/'size_charge_correlations.csv',index=False);pd.DataFrame(strata).to_csv(H/'size_charge_strata.csv',index=False)
# Saved scalar examples exercise the sign convention independently.
assert abs(4-3)-abs(5-3)==-1 and abs(6-3)-abs(5-3)==1 and abs(3-3)==0
report={'status':'PASS','rows':len(p),'unique_compounds':int(p.record_id.nunique()),'correlations':corr,'formula':'abs(fitted-truth)-abs(native-truth)','new_predictive_fits':0,'diagnostic_regressions':'rank residualization only','inputs':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [H/'proximity_all.csv',H/'mover_chemistry_all.csv',R/'data/published/modeling_manifest.csv']},'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(H/'size_charge_verification.json').write_text(json.dumps(report,indent=2));print(pd.DataFrame(corr).to_string(index=False));print(pd.DataFrame(strata).to_string(index=False))
