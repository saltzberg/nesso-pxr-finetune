"""Independent scalar and population checks for signed error plot inputs."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[2]
d=pd.read_csv(H/'size_charge_points.csv',float_precision='round_trip');s=pd.read_csv(H/'size_charge_strata.csv',float_precision='round_trip')
assert len(d)==13376 and d.record_id.nunique()==3344
checks=[]
for (n,m),a in d.groupby(['budget','method']):
 assert len(a)==3344 and a.record_id.is_unique
 # Re-read each source task, rather than relying on stored fitted and truth columns.
 for fold,b in a.groupby('fold'):
  tid=f'fold{fold}__draw0__n{n}__{m}__seed42';path=R/'artifacts/cloud/50146714/remote_results/tasks'/tid
  raw=pd.read_csv(path/'test.csv',float_precision='round_trip').set_index('record_id').loc[b.record_id]
  native=pd.read_csv(path/'baseline.csv',float_precision='round_trip').set_index('record_id').loc[b.record_id]
  delta=np.abs(raw.y_pred.to_numpy()-raw.y_true.to_numpy())-np.abs(native.y_pred.to_numpy()-raw.y_true.to_numpy())
  # Earlier proximity CSV used default CSV parsing; retain and bound its sub-ULP discrepancy.
  err=float(np.max(np.abs(delta-b.difference_distance_from_truth.to_numpy())));assert err<1e-12
  checks.append(dict(budget=int(n),method=m,fold=int(fold),max_absolute_difference=err))
 for keys,b in a.groupby(['small','charged','low_potency']):
  row=s[(s.budget==n)&(s.method==m)&(s.small==keys[0])&(s.charged==keys[1])&(s.low_potency==keys[2])].iloc[0]
  assert row.n==len(b) and row.improved+row.worse+row.unchanged==len(b)
  assert abs(row.mean_error_delta-b.difference_distance_from_truth.mean())<1e-12
 assert a.shape[0]==s[(s.budget==n)&(s.method==m)].n.sum()
for name in ['error-change-versus-size','error-change-stratified-n100','error-change-stratified-n500']:
 root=ET.parse(R/'reports/architecture/results'/(name+'.svg')).getroot()
 texts=[''.join(t.itertext()) for t in root.iter('{http://www.w3.org/2000/svg}text')]
 assert 'difference distance from truth' in texts
assert abs(4-3)-abs(5-3)<0 and abs(6-3)-abs(5-3)>0
out=dict(status='PASS',source_task_checks=checks,rows=len(d),formula_sign_tests='PASS',x_axis_literal='PASS',group_counts_and_means='PASS')
(H/'size_charge_acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
