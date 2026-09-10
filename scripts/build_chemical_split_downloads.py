#!/usr/bin/env python3
"""Reporting-only exact ID joins of SAVED split records. Never create splits or fit.
Run: review/.venv/bin/python PXR/scripts/build_chemical_split_downloads.py
Native tables stay immutable. Outputs retain literal source columns/order plus
original_id (OADMET), OCNT_ID, raw_smiles, and saved chemical_group where applicable.
"""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/chemical-splits'
LOW = ROOT / 'artifacts/experiments/low_data_20260905_cut035_run2/prepared'
STRICT = ROOT / 'artifacts/experiments/low_data_followup_20260906/strict_split'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def tables():
    sources = {}
    def read(p):
        sources[str(p.relative_to(ROOT))] = digest(p)
        return pd.read_csv(p, dtype=str, keep_default_na=False)
    pub = read(ROOT/'data/published/modeling_manifest.csv')
    identities = read(LOW/'inputs.csv')[['row_index','record_id','feature_row','canonical_smiles']]
    assert len(pub) == 4134 and len(identities) == 3344
    for key in ['record_id','feature_row','original_id']:
        assert pub[key].is_unique and pub[key].ne('').all()
    for key in identities:
        assert identities[key].is_unique and identities[key].ne('').all()
    assert identities.row_index.tolist() == list(map(str,range(3344)))
    identity = identities.merge(pub[['record_id','feature_row','identity_canonical_smiles','original_id','OCNT_ID','raw_smiles','modeling_role']], on='record_id', how='left', validate='one_to_one', suffixes=('','_published'), indicator=True)
    assert identity._merge.eq('both').all()
    assert identity.feature_row.eq(identity.feature_row_published).all()
    assert identity.canonical_smiles.eq(identity.identity_canonical_smiles).all()
    assert identity.modeling_role.eq('development').all()
    identity = identity[list(identities)+['original_id','OCNT_ID','raw_smiles']]
    low_a = read(LOW/'assignments.csv')
    strict_a = read(STRICT/'assignments.csv')
    assert len(low_a)==6688 and not low_a.duplicated(['split','row_index']).any()
    assert set(low_a.split)=={'random','chemical_cluster'}
    pd.testing.assert_frame_equal(low_a[low_a.split.eq('chemical_cluster')].reset_index(drop=True),strict_a)
    strict_ids = read(STRICT/'identities.csv')
    pd.testing.assert_frame_equal(identities,strict_ids)
    for policy,a in low_a.groupby('split'):
        assert set(a.row_index)==set(identities.row_index) and set(a.outer_fold)==set('01234')
        if policy=='chemical_cluster':
            assert a.groupby('chemical_group').outer_fold.nunique().max()==1
    def join(a):
        result=a.merge(identity,on='row_index',how='left',validate='many_to_one',sort=False,indicator=True)
        assert len(result)==len(a) and result._merge.eq('both').all()
        result=result.drop(columns='_merge')
        pd.testing.assert_frame_equal(result[list(a)],a.reset_index(drop=True))
        return result
    outputs={'low-data-outer-assignments.csv':join(low_a), 'strict-outer-assignments.csv':join(strict_a)}
    counts={}
    for label,root,a in [('low-data',LOW,low_a),('strict',STRICT,strict_a)]:
        subsets=read(root/'subsets.csv')
        keys=['split','outer_fold','draw','n_train']
        assert not subsets.duplicated(keys+['row_index']).any()
        assert set(subsets.role)=={'fit','calibration'}
        assert set(subsets.draw)==set(map(str,range(10)))
        clusters=a.drop_duplicates('row_index')[['row_index','chemical_group']]
        result=join(subsets).merge(clusters,on='row_index',validate='many_to_one',sort=False)
        for key,b in subsets.groupby(keys,sort=False):
            policy,fold,draw,budget=key
            outer=a if label=='strict' else a[a.split.eq(policy)]
            assert not set(b.row_index)&set(outer.loc[outer.outer_fold.eq(fold),'row_index'])
            n=len(b);ncal=(n+4)//5
            assert b.role.tolist()==['fit']*(n-ncal)+['calibration']*ncal
            if budget!='-1': assert n==int(budget)
        outputs[label+'-acquired-roles.csv']=result
        counts[label]={'assignment_rows':len(a),'acquired_rows':len(subsets),'tasks':subsets.groupby(keys).ngroups,'role_counts':subsets.role.value_counts().to_dict(),'fold_counts':a.groupby(['split','outer_fold']).size().rename('n_test').reset_index().to_dict('records')}
    pool=read(STRICT/'pool_audit.csv')
    # Check native identities before supplementing the already labeled audit.
    checked=pool.merge(identity,on='row_index',validate='many_to_one',suffixes=('','_identity'))
    for col in ['record_id','canonical_smiles']:
        assert checked[col].eq(checked[col+'_identity']).all()
    pool=pool.merge(identity[['row_index','original_id','OCNT_ID','raw_smiles']],on='row_index',validate='many_to_one')
    pool=pool.merge(strict_a[['row_index','chemical_group']],on='row_index',validate='many_to_one')
    assert pool.eligible.eq('True').equals(pool.max_test_similarity.astype(float).lt(.35))
    eligible=set(zip(pool.loc[pool.eligible.eq('True'),'outer_fold'],pool.loc[pool.eligible.eq('True'),'row_index']))
    acquired=outputs['strict-acquired-roles.csv']
    assert set(zip(acquired.outer_fold,acquired.row_index))<=eligible
    outputs['strict-pool-audit.csv']=pool
    counts['strict']['pool_counts']=pool.groupby(['outer_fold','eligible']).size().rename('n').reset_index().to_dict('records')
    # Validate consumed source bytes against native frozen manifests, not live defaults.
    for root,field in [(LOW,'artifact_sha256'),(STRICT,'outputs')]:
        p=root/'preparation_manifest.json';sources[str(p.relative_to(ROOT))]=digest(p)
        m=json.loads(p.read_text())
        for key,h in list(sources.items()):
            path=ROOT/key
            if path.parent==root and path.name in m[field]: assert h==m[field][path.name]
    status=ROOT/'artifacts/experiments/low_data_20260905_cut035_run2/status.json'
    s=json.loads(status.read_text());sources[str(status.relative_to(ROOT))]=digest(status)
    assert s['complete']==s['total_tasks']==7200 and set(s['tasks'].values())=={'complete'}
    counts['completed_low_data_models']={p:sum(k.startswith(p+'__') for k in s['tasks']) for p in ['random','chemical_cluster']}
    sources[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    return outputs,sources,counts

def build(output=OUT):
    outputs,sources,counts=tables();output.mkdir(parents=True,exist_ok=True)
    exports={}
    for name,frame in outputs.items():
        p=output/name;frame.to_csv(p,index=False)
        exports[name]={'rows':len(frame),'sha256':digest(p),'columns':list(frame)}
    copies={'initial-modeling-manifest.csv':ROOT/'data/published/modeling_manifest.csv','initial-scaffold-holdout.csv':ROOT/'data/derived/holdout_assignments.csv'}
    for name,p in copies.items():
        sources[str(p.relative_to(ROOT))]=digest(p);shutil.copyfile(p,output/name)
        df=pd.read_csv(p,dtype=str,keep_default_na=False)
        exports[name]={'rows':len(df),'sha256':digest(output/name),'columns':list(df),'byte_equal_source':str(p.relative_to(ROOT))}
    result={'derivation':'Exact many-to-one row_index -> saved identities, verified record_id AND feature_row AND canonical_smiles against published manifest; original IDs attached by record_id. No new splits, labels, fitting, chemistry transformations or prediction execution. Outer assignment fold is the saved TEST fold; other folds form the outer pool, not necessarily acquired FIT. Native fit/calibration role strings are unchanged. Strict outer file retains native chemical_cluster label because TEST assignments are identical; strict acquired roles use native strict_chemical.', 'sources':sources,'exports':exports,'counts':counts}
    (output/'manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(counts,sort_keys=True))
    return result

if __name__=='__main__':
    build()
