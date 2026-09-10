"""Fresh-process verification of all40 saved CPU control models."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.linear_model import LinearRegression
R=Path('/home/dan/projects/nesso-finetune/PXR');H=R/'artifacts/analysis/mw_controls_20260908';F=R/'artifacts/experiments/affinity_comparison_20260906/controls_v1/fresh_features'
read=lambda p:pd.read_csv(p,float_precision='round_trip')
v=json.loads((H/'verification.json').read_text());assert v['completed_fits']==40
for p,h in v['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
lock=json.loads((R/'artifacts/cloud/50146714/remote_results/stage/manifest.json').read_text());d=read(R/lock['protocol']['labels']).set_index('record_id');ids=read(F/'identities.csv');native=np.load(F/'native_points.npy');d['native']=pd.Series(native,index=ids.record_id);d['MW']=read(R/'artifacts/analysis/affinity_comparison_20260907/mover_chemistry_all.csv').set_index('record_id').MW
checks=[]
for p in sorted((H/'cells').glob('*/state.json')):
 s=json.loads(p.read_text());roles=lock['roles'][f"fold{s['fold']}__draw0__n{s['budget']}__head_only__seed42"];assert s['roles']==roles
 fi=d.loc[roles['fit']];te=d.loc[roles['test']];cal=d.loc[roles['calibration']];mean=np.array(s['mean']);scale=np.array(s['scale'])
 assert np.array_equal(mean,fi[['native','MW']].mean().to_numpy());assert np.array_equal(scale,fi[['native','MW']].std(ddof=0).to_numpy())
 def x(z):
  a=(z[['native','MW']].to_numpy()-mean)/scale
  if s['model']=='mw':return a[:,1:2]
  if s['model']=='native':return a[:,:1]
  if s['model']=='native_mw':return a
  return np.column_stack([a,a[:,0]*a[:,1]])
 model=LinearRegression().fit(x(fi),fi.pEC50,sample_weight=1/np.maximum(fi.pEC50_standard_error,.1))
 pp=read(p.parent/'test.csv');assert pp.record_id.tolist()==roles['test'];err=float(np.max(abs(model.predict(x(te))-pp.y_pred)));assert err<1e-10
 b=np.array(s['beta']);replay=np.column_stack([np.ones(len(te)),x(te)])@b;assert np.array_equal(replay,pp.y_pred.to_numpy())
 cp=read(p.parent/'calibration.csv');assert cp.record_id.tolist()==roles['calibration'];assert np.allclose(cp.y_pred,model.predict(x(cal)),rtol=0,atol=1e-10)
 checks.append(dict(cell=p.parent.name,max_sklearn_difference=err,serialized_replay_exact=True))
assert len(checks)==40
out=dict(status='PASS',models_checked=len(checks),source_hashes_checked=len(v['source_hashes']),checks=checks)
(H/'independent_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps({'status':'PASS','models_checked':40,'max_sklearn_difference':max(c['max_sklearn_difference'] for c in checks)}))
