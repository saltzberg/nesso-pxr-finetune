#!/usr/bin/env python3
"""Exploratory CPU-only saved-native-feature associations; no inference/predictive fits.
Run: python artifacts/analysis/affinity_comparison_20260907/internal_feature_diagnostic/analyze.py
PCA is descriptive, centered/unscaled on unique development compounds, not OOF prediction.
OLS projections below only define partial correlations; no predictive model is trained.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits, threadpool_info

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
ANALYSIS = OUT.parent
FEATURES = ROOT/'artifacts/experiments/affinity_comparison_20260906/controls_v1/fresh_features'
CACHE = Path('/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1')
inputs = {}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def bind(path, expected=None):
    path = Path(path)
    digest = sha(path)
    if expected is not None:
        assert digest == expected, f'Hash mismatch: {path}'
    inputs[str(path)] = digest
    return path

def read(path):
    return pd.read_csv(bind(path), dtype={'record_id':str}, keep_default_na=False, float_precision='round_trip')

def put(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')

def correlation(a,b):
    if len(a)<4 or np.ptp(a)<1e-12 or np.ptp(b)<1e-12:
        return np.nan
    return float(np.corrcoef(a,b)[0,1])

def associate(frame, xs, ys, scope, budget='unique', fold='pooled'):
    rows=[]
    for x in xs:
        for y in ys:
            for method in ('pearson','spearman'):
                for adjustment in ('none','fold_potency_charge','fold_potency_charge_MW'):
                    nuisance = [] if adjustment=='none' else ['fold','y_true','prepared_formal_charge']
                    if adjustment.endswith('_MW'): nuisance += ['MW']
                    nuisance = [v for v in nuisance if v not in (x,y)]
                    cols=list(dict.fromkeys([x,y]+nuisance))
                    f=frame[cols].replace([np.inf,-np.inf],np.nan).dropna()
                    a=f[x].to_numpy(float); b=f[y].to_numpy(float)
                    if method=='spearman': a=rankdata(a); b=rankdata(b)
                    design=[np.ones((len(f),1))]
                    for c in nuisance:
                        if c in ('fold','prepared_formal_charge'):
                            design.append(pd.get_dummies(f[c].astype(str),drop_first=True).to_numpy(float))
                        else:
                            v=f[c].to_numpy(float)
                            if method=='spearman': v=rankdata(v)
                            design.append(v[:,None])
                    z=np.column_stack(design)
                    if nuisance:
                        ab=np.column_stack([a,b]); residual=ab-z@np.linalg.lstsq(z,ab,rcond=None)[0]
                        a,b=residual.T
                    rows.append(dict(scope=scope,budget=budget,fold=fold,feature=x,target=y,method=method,
                        adjustment=adjustment,actual_covariates=';'.join(nuisance),n=len(f),
                        nuisance_rank=int(np.linalg.matrix_rank(z)),correlation=correlation(a,b)))
    return rows

def main():
    meta=json.loads(bind(FEATURES/'manifest.json').read_text())
    bindings=json.loads(bind(FEATURES.parent/'controls_bindings.json').read_text())
    assert sha(FEATURES/'manifest.json')==bindings['feature_manifest_sha256']
    for name,digest in meta['outputs'].items(): bind(FEATURES/name,digest)
    extractor=ROOT/'experiments/20260906_affinity_comparison/code/controls.py'
    bind(extractor,meta['extractor_sha256'])
    bind(ROOT/'experiments/20260906_affinity_comparison/code/train.py')
    stage_path=CACHE/'stage/manifest.json'
    bind(stage_path,bindings['files'][str(stage_path)])
    stage=json.loads(stage_path.read_text())
    assert stage['binding']==meta['cache']['binding']
    assert stage['records']==meta['cache']['records']
    assert stage['challenge_label_training'] is False and meta['outcome_labels_read'] is False
    assert stage['numeric_outcomes_copied'] is False
    for path,digest in stage['identity_sources'].items(): bind(ROOT/path,digest)
    identities=read(FEATURES/'identities.csv')
    records=pd.DataFrame(meta['cache']['records'])
    assert identities.record_id.is_unique and records.record_id.is_unique
    assert identities.record_id.tolist()==records.record_id.tolist()
    assert np.array_equal(identities.row_index.to_numpy(),np.arange(len(records)))
    assert np.array_equal(records.row_index.astype(int),identities.row_index)
    strict_path=ROOT/'artifacts/experiments/low_data_followup_20260906/strict_split/identities.csv'
    bind(strict_path,bindings['files'][str(strict_path)])
    strict=read(strict_path)
    for col in ('record_id','canonical_smiles','feature_row'):
        assert records[col].astype(str).tolist()==strict[col].astype(str).tolist()
    source=read(ROOT/'data/published/modeling_manifest.csv')
    assert source.record_id.is_unique and source.feature_row.is_unique
    source=source.set_index('record_id').loc[identities.record_id].reset_index()
    assert np.array_equal(source.feature_row.astype(int),records.feature_row.astype(int))
    assert (source.modeling_role=='development').all()
    assert source.canonical_smiles.tolist()==records.canonical_smiles.tolist()
    proximity=read(ANALYSIS/'proximity_all.csv')
    chemistry=read(ANALYSIS/'mover_chemistry_all.csv')
    assert not proximity.duplicated(['budget','fold','record_id']).any()
    assert chemistry.record_id.is_unique
    assert set(chemistry.record_id)==set(identities.record_id)
    for budget,g in proximity.groupby('budget'):
        assert g.record_id.is_unique and set(g.record_id)==set(identities.record_id)
    common=proximity.merge(chemistry,on=['budget','fold','record_id'],suffixes=('_p','_c'),validate='one_to_one')
    assert len(common)==len(chemistry)
    diagnostics_parity = {}
    for col in ('smiles','y_true','native','head_only','head_plus_lora'):
        if col=='smiles':
            assert (common[col+'_p']==common[col+'_c']).all(),col
        else:
            difference=abs(common[col+'_p']-common[col+'_c'])
            diagnostics_parity[col]=dict(different_exact=int((difference!=0).sum()),max_absolute_difference=float(difference.max()))
            assert difference.max() <= 1e-12, col
    # Only chemistry descriptors are used from chemistry; predictions always from proximity.
    chemistry=chemistry.set_index('record_id').loc[identities.record_id].reset_index()
    assert chemistry.smiles.tolist()==records.canonical_smiles.tolist()
    assert np.array_equal(chemistry.y_true,source.pEC50.astype(float))
    z=np.load(FEATURES/'features.npz',allow_pickle=False)
    assert set(z.files)=={'member1','member2','scalar_features'}
    assert z['member1'].shape==z['member2'].shape==(len(identities),384)
    assert z['member1'].dtype==z['member2'].dtype==np.float32
    x=np.concatenate([z['member1'],z['member2']],axis=1).astype(np.float64)
    assert np.isfinite(x).all()
    native=np.load(FEATURES/'native_points.npy',allow_pickle=False)
    scalar=z['scalar_features']
    reconstructed=(np.float32(6)-(scalar[:,0]+scalar[:,2])/np.float32(2)).astype(float)
    assert np.array_equal(native,reconstructed)
    # The caller's prediction ledger can use a slightly different remote FP32 native baseline.
    baseline_diff=[]
    for budget,g in proximity.groupby('budget'):
        g=g.set_index('record_id').loc[identities.record_id]
        delta=g.native.to_numpy()-native
        baseline_diff.append(dict(budget=int(budget),n=len(g),different_exact=int(np.count_nonzero(delta)),max_absolute_difference=float(np.max(abs(delta)))))
        assert np.array_equal(g.y_true.to_numpy(),chemistry.y_true.to_numpy())
        assert np.array_equal(g['fold'].to_numpy(),chemistry['fold'].to_numpy())
    mean=x.mean(axis=0); centered=x-mean
    eigenvalues,loadings=np.linalg.eigh(centered.T@centered/(len(x)-1))
    order=np.argsort(eigenvalues)[::-1]; eigenvalues=eigenvalues[order]; loadings=loadings[:,order[:5]]
    for j in range(5):
        if loadings[np.argmax(abs(loadings[:,j])),j]<0: loadings[:,j]*=-1
    scores=centered@loadings
    f=chemistry[['record_id','fold','smiles','y_true','MW','heavy_atoms','formal_charge']].copy()
    f=f.rename(columns={'formal_charge':'identity_formal_charge'})
    f['feature_array_row']=identities.row_index
    f['source_feature_row']=source.feature_row
    f['row_id']=source.row_id
    f['prepared_smiles']=source.prepared_smiles
    f['prepared_formal_charge']=source.formal_charge.astype(int)
    f['prepared_is_charged']=(f.prepared_formal_charge!=0).astype(int)
    f['native_cached']=native
    f['norm_l2']=np.linalg.norm(x,axis=1)
    f['centered_norm_l2']=np.linalg.norm(centered,axis=1)
    f['member1_norm_l2']=np.linalg.norm(x[:,:384],axis=1)
    f['member2_norm_l2']=np.linalg.norm(x[:,384:],axis=1)
    for j in range(5): f[f'PC{j+1}']=scores[:,j]
    feature_names=['norm_l2','centered_norm_l2','member1_norm_l2','member2_norm_l2']+[f'PC{j+1}' for j in range(5)]
    targets=['MW','heavy_atoms','prepared_formal_charge','prepared_is_charged','y_true','native_cached']
    associations=associate(f,feature_names,targets,'native_unique')
    for fold,g in f.groupby('fold'):
        associations+=associate(g,feature_names,targets,'native_fold',fold=int(fold))
    ledger=proximity.merge(f.drop(columns=['fold','smiles','y_true']),on='record_id',validate='many_to_one')
    # Adaptation measured against the exact native value supplied by each prediction row.
    adaptation_targets=[]
    for label,adapt,reference in [('head','head_only','native'),('lora_total','head_plus_lora','native'),('lora_increment','head_plus_lora','head_only')]:
        ledger[label+'_signed_change']=ledger[adapt]-ledger[reference]
        ledger[label+'_magnitude']=abs(ledger[label+'_signed_change'])
        ledger[label+'_error_delta']=abs(ledger[adapt]-ledger.y_true)-abs(ledger[reference]-ledger.y_true)
        adaptation_targets += [label+'_signed_change',label+'_magnitude',label+'_error_delta']
    for budget,g in ledger.groupby('budget'):
        associations+=associate(g,feature_names,adaptation_targets,'adaptation',budget=int(budget))
        associations+=associate(g,['MW'],adaptation_targets,'size_adaptation',budget=int(budget))
        for fold,gf in g.groupby('fold'):
            associations+=associate(gf,['norm_l2','PC1','PC2'],adaptation_targets,'adaptation_fold',budget=int(budget),fold=int(fold))
    a=pd.DataFrame(associations)
    a.to_csv(OUT/'associations.csv',index=False)
    f.to_csv(OUT/'native_feature_scores.csv',index=False)
    ledger.to_csv(OUT/'adaptation_feature_rows.csv',index=False)
    pd.DataFrame({'PC':np.arange(1,len(eigenvalues)+1),'eigenvalue':eigenvalues,'explained_variance_fraction':eigenvalues/eigenvalues.sum()}).to_csv(OUT/'pca_variance.csv',index=False)
    pd.DataFrame(loadings,columns=[f'PC{j+1}' for j in range(5)]).rename_axis('feature_coordinate').to_csv(OUT/'pca_loadings.csv')
    pd.DataFrame({'feature_coordinate':np.arange(768),'mean':mean}).to_csv(OUT/'pca_center.csv',index=False)
    strata=[]
    groups={'MW_lt200':np.where(f.MW<200,'<200','>=200'),'potency_lt4':np.where(f.y_true<4,'<4','>=4'),'prepared_charge':f.prepared_formal_charge.astype(str)}
    for dimension,values in groups.items():
        for value in sorted(set(values)):
            sub=f.loc[np.asarray(values)==value]
            r=dict(dimension=dimension,value=value,n=len(sub))
            for col in ['MW','y_true','prepared_formal_charge']+feature_names:
                r[col+'_median']=float(sub[col].median())
            strata.append(r)
    pd.DataFrame(strata).to_csv(OUT/'stratum_medians.csv',index=False)
    summary=dict(status='complete',analysis='retrospective descriptive associations, not predictive evaluation',
        n_unique=len(f),n_ledger_rows=len(ledger),budgets=sorted(proximity.budget.unique().tolist()),
        shape=list(x.shape),prepared_charge_counts={str(k):int(v) for k,v in f.prepared_formal_charge.value_counts().sort_index().items()},
        charge_different_from_identity=int((f.prepared_formal_charge!=f.identity_formal_charge).sum()),
        top5_explained_variance=(eigenvalues[:5]/eigenvalues.sum()).tolist(),native_baseline_discrepancy=baseline_diff,
        diagnostics_prediction_parity=diagnostics_parity,
        n_association_rows=len(a),missing_native_features=0,missing_prepared_charge=0,
        pca='Centered, unscaled FP64 covariance eigendecomposition over all unique development rows; largest absolute loading positive; descriptive only, not a learned predictive transformation.',
        partial='Pearson residual correlation or rank-first residual correlation (partial Spearman). Intercept, categorical fold and prepared charge, continuous potency; remove target from covariates; optional MW. Nuisance projection only, not predictive model fitting.',
        error_delta='absolute adapted error minus absolute reference error; negative improves. LoRA increment compares head_plus_lora with head_only, not matched representation deltas.',
        provenance_validation='All feature-cache output hashes, extractor hash, controls-bound feature manifest, stage manifest hash, stage identity sources, strict identities verified. Exact ordered records/feature_row/SMILES, development roles, labels and FP32 cached scalar parity verified. Full upstream multi-GB tensor/checkpoint inventory not rehashed; its historical hashes remain in source manifests.',
        limitations=['Global PCA is exploratory/transductive descriptive geometry; no held-out predictive gain or causal claim.',
          'No adapted feature tensors are used. Native-feature association with LoRA output changes is not evidence of LoRA-induced internal-feature change.',
          'Prepared charge is the selected solution-state formal charge, not identity-SMILES charge; MW is the existing identity-SMILES descriptor.',
          'No uncertainty intervals or multiplicity-corrected significance tests; repeated budgets analyzed separately, each compound once per budget.',
          'Fold and potency/charge adjustment is descriptive linear/rank residualization; nonlinear chemistry and structural confounding remain.'],
        runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,threads=2,gpu_used=False,new_inference=False,predictive_fits=0,threadpools=threadpool_info()))
    put('summary.json',summary)
    put('input_provenance.json',dict(input_sha256=inputs,script_sha256=sha(__file__),feature_definition='Concatenation of affinity_module.native_output.affinity_repr (384) then affinity_module2 (384); not scalar_features and not mean-pooled across members.',cache_binding=meta['cache']['binding'],cache_path=str(CACHE)))
    output_names=['associations.csv','native_feature_scores.csv','adaptation_feature_rows.csv','pca_variance.csv','pca_loadings.csv','pca_center.csv','stratum_medians.csv','summary.json','input_provenance.json']
    put('output_sha256.json',{n:sha(OUT/n) for n in output_names})
    print(json.dumps({k:summary[k] for k in ['status','n_unique','n_ledger_rows','budgets','shape','prepared_charge_counts','charge_different_from_identity','top5_explained_variance','native_baseline_discrepancy','n_association_rows']},indent=2))
    print(a.query("scope=='native_unique' and feature in ['norm_l2','PC1','PC2'] and target in ['MW','prepared_formal_charge','y_true'] and method=='spearman' and adjustment!='fold_potency_charge_MW'")[['feature','target','adjustment','correlation']].to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
