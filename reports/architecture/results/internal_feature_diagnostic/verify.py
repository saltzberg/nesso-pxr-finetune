#!/usr/bin/env python3
"""Independent saved-output checks plus byte-identical analyzer replay; CPU only."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
P=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n): return pd.read_csv(P/n,float_precision='round_trip')
def main():
    saved=json.loads((P/'output_sha256.json').read_text())
    for name,h in saved.items(): assert sha(P/name)==h
    f=read('native_feature_scores.csv'); a=read('associations.csv'); l=read('adaptation_feature_rows.csv')
    s=json.loads((P/'summary.json').read_text())
    assert len(f)==s['n_unique']==f.record_id.nunique()==3344
    assert len(l)==s['n_ledger_rows']==6688 and len(a)==s['n_association_rows']
    assert l.groupby('budget').record_id.nunique().to_dict()=={100:3344,500:3344}
    assert not f[['MW','y_true','prepared_formal_charge','norm_l2','PC1']].isna().any().any()
    for budget,g in l.groupby('budget'):
        for label,adapt,ref in [('head','head_only','native'),('lora_total','head_plus_lora','native'),('lora_increment','head_plus_lora','head_only')]:
            np.testing.assert_array_equal(g[label+'_magnitude'],abs(g[adapt]-g[ref]))
            np.testing.assert_array_equal(g[label+'_error_delta'],abs(g[adapt]-g.y_true)-abs(g[ref]-g.y_true))
    raw=float(spearmanr(f.norm_l2,f.MW).statistic)
    expected=a.query("scope=='native_unique' and feature=='norm_l2' and target=='MW' and method=='spearman' and adjustment=='none'").correlation.iloc[0]
    assert abs(raw-expected)<1e-14
    # Independent QR nuisance projection, rather than analyzer's least-squares solver.
    z=np.column_stack([np.ones(len(f)),pd.get_dummies(f['fold'],drop_first=True),pd.get_dummies(f.prepared_formal_charge,drop_first=True),rankdata(f.y_true)])
    q,_=np.linalg.qr(z)
    xy=np.column_stack([rankdata(f.norm_l2),rankdata(f.MW)])
    res=xy-q@(q.T@xy)
    partial=float(np.corrcoef(res.T)[0,1])
    expected=a.query("scope=='native_unique' and feature=='norm_l2' and target=='MW' and method=='spearman' and adjustment=='fold_potency_charge'").correlation.iloc[0]
    assert abs(partial-expected)<1e-12
    loading=read('pca_loadings.csv').drop(columns='feature_coordinate').to_numpy()
    np.testing.assert_allclose(loading.T@loading,np.eye(5),atol=1e-12)
    variance=read('pca_variance.csv')
    np.testing.assert_allclose(f[[f'PC{i}' for i in range(1,6)]].var().to_numpy(),variance.eigenvalue.iloc[:5],rtol=1e-12)
    subprocess.run([sys.executable,str(P/'analyze.py')],check=True,stdout=subprocess.DEVNULL)
    after=json.loads((P/'output_sha256.json').read_text())
    assert saved==after,'Analyzer output bytes changed on replay'
    result=dict(status='PASS',output_hashes_verified=len(saved),unique_compounds=len(f),adaptation_rows=len(l),association_rows=len(a),independent_spearman=raw,independent_qr_partial_spearman=partial,pc_orthogonality_and_variance='PASS',adaptation_arithmetic='exact',analyzer_byte_identical_replay=True,verifier_sha256=sha(Path(__file__)))
    (P/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
