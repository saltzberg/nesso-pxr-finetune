#!/usr/bin/env python3
"""Independent disk replay: manual transforms, new Booster, rankdata metrics, LP weight."""
import run as P
import numpy as np
import pandas as pd
import pickle,json
from scipy.optimize import linprog
from scipy.stats import rankdata
from lightgbm import Booster

def predict(state,desc,features,rows):
    if state['method']==P.C.DESCRIPTOR:
        m=state['model']; assert m['kind']=='descriptor_lgbm'
        reducer=m['reducer']; selected=reducer.selected_
        z=desc[rows][:,selected]
        z=np.where(np.isnan(z),reducer.mean_[selected],z)[:,m['gain_selected']]
        return Booster(model_str=m['booster'].model_to_string()).predict(z,num_threads=2)
    s=state['models'][0]; pre=s['preprocessing']; model=s['model']
    x=np.concatenate([features['member1'][rows],features['member2'][rows]],axis=1).astype(float)
    return ((x-pre['mean'])/pre['scale'])@model.coef_+model.intercept_

def independent_metrics(y,p,se):
    def values(mask):
        a,b,e=y[mask],p[mask],se[mask]; w=1/np.maximum(e,.1)
        rho=float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
        return {'n':int(mask.sum()),'weighted_mae':float(np.sum(w*abs(a-b))/sum(w)),'raw_mae':float(sum(abs(a-b))/len(a)),'spearman':rho}
    m=values(np.ones(len(y),bool)); m['ge4']=values(y>=4); return m

def lp_alpha(y,d,r,w):
    n=len(y); delta=r-d; residual=y-d
    # |residual-alpha*delta| <= slack; objective independent LP formulation.
    objective=np.r_[0.,w/w.sum()]
    matrix=np.vstack([np.column_stack([-delta,-np.eye(n)]),np.column_stack([delta,-np.eye(n)])])
    b=np.r_[-residual,residual]
    fit=linprog(objective,A_ub=matrix,b_ub=b,bounds=[(0,1)]+[(0,None)]*n,method='highs',options={'threads':2})
    assert fit.success
    return float(fit.x[0]),float(fit.fun)

def verify():
    O=P.OUT
    marker=P.C.read_json(O/'complete.json'); P.C.check_hashes(O,marker['outputs'])
    bindings=P.C.read_json(O/'bindings.json')
    for p,h in bindings['inputs'].items(): assert P.sha(p)==h,p
    ident,assign,cell,labels,desc,dm,features,_=P.sources()
    fit,cal,test=(cell['roles'][r] for r in ('fit','calibration','test'))
    roles=P.C.csv(O/'roles.csv',dtype={'record_id':str,'chemical_group':str})
    for name,rows in cell['roles'].items():
        assert set(roles.loc[roles.role==name,'row_index'])==set(rows)
    groups=assign.chemical_group.astype(str).to_numpy()
    assert not set(groups[np.r_[fit,cal]])&set(groups[test])
    expected_models={f'meta{k}_{m}' for k in range(3) for m in ('descriptor','ridge')}|{'final_descriptor','final_ridge'}
    assert {p.name for p in (O/'models').iterdir()}==expected_models
    oof=P.C.csv(O/'inner_oof.csv',dtype={'record_id':str})
    assert np.array_equal(oof.row_index,fit)
    seen={'descriptor':[], 'ridge':[]}; maxerr=0.; transform_calls=0
    for name in sorted(expected_models):
        directory=O/'models'/name
        audit=P.C.read_json(directory/'audit.json'); tr=np.array(audit['training_rows']); va=np.array(audit['prediction_rows'])
        assert set(tr)<=set(fit) and not set(tr)&set(va)
        state=pickle.loads((directory/'model.pkl').read_bytes())
        arr=np.load(directory/'predictions.npz'); assert np.array_equal(arr['row_index'],va)
        new=predict(state,desc,features,va)
        maxerr=max(maxerr,float(np.max(abs(new-arr['pred']))))
        np.testing.assert_allclose(new,arr['pred'],rtol=0,atol=1e-12)
        kind=name.split('_')[-1]
        if name.startswith('meta'):
            assert not set(groups[tr])&set(groups[va])
            seen[kind].extend(va.tolist())
            mapped=oof.set_index('row_index').loc[va,kind].to_numpy()
            np.testing.assert_array_equal(mapped,arr['pred'])
        for f in audit['selection']['folds']:
            a,b=tr[f['train_indices']],tr[f['validation_indices']]
            assert not set(groups[a])&set(groups[b])
            assert set(np.r_[a,b])==set(tr)
        valid_sets=[tr[f['train_indices']] for f in audit['selection']['folds']]+[tr]
        for entry in audit['instrumented_transform_calls']:
            rr=np.asarray(entry['row_index'])
            assert any(np.array_equal(rr,v) for v in valid_sets)
            x=desc[rr] if kind=='descriptor' else np.concatenate([features['member1'][rr],features['member2'][rr]],axis=1).astype(float)
            assert P.B._hash_array(x)==entry['input_sha256']
            transform_calls+=1
    for rr in seen.values(): assert sorted(rr)==sorted(fit.tolist()) and len(rr)==400
    alpha,loss=lp_alpha(oof.y_true.to_numpy(),oof.descriptor.to_numpy(),oof.ridge.to_numpy(),oof.weight.to_numpy())
    selection=P.C.read_json(O/'blend_selection.json')
    np.testing.assert_allclose(alpha,selection['alpha_ridge'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(loss,selection['selected_weighted_mae'],atol=1e-12,rtol=0)
    ev=P.C.csv(O/'predictions.csv',dtype={'record_id':str})
    np.testing.assert_array_equal(ev.row_index,np.r_[cal,test])
    assert ev.record_id.tolist()==ident.iloc[np.r_[cal,test]].record_id.tolist()
    for kind in ('descriptor','ridge'):
        raw=np.load(O/'models'/('final_'+kind)/'predictions.npz')
        np.testing.assert_array_equal(raw['pred'],ev[kind])
    np.testing.assert_array_equal(ev.blend,(1-selection['alpha_ridge'])*ev.descriptor+selection['alpha_ridge']*ev.ridge)
    tf=P.C.csv(O/'test_predictions.csv',dtype={'record_id':str})
    assert np.array_equal(tf.row_index,test)
    np.testing.assert_array_equal(tf.y_true,labels.iloc[test].pEC50)
    np.testing.assert_array_equal(tf.assay_se,labels.iloc[test].pEC50_standard_error)
    np.testing.assert_array_equal(tf.weight,1/np.maximum(tf.assay_se,.1))
    saved=P.C.read_json(O/'metrics.json'); replay={}
    for kind in ('descriptor','ridge','blend'):
        np.testing.assert_array_equal(tf[kind],ev.loc[ev.role=='test',kind])
        replay[kind]=independent_metrics(tf.y_true.to_numpy(),tf[kind].to_numpy(),tf.assay_se.to_numpy())
        for key in ('weighted_mae','raw_mae','spearman','n'):
            np.testing.assert_allclose(replay[kind][key],saved[kind][key],atol=1e-12,rtol=0)
            np.testing.assert_allclose(replay[kind]['ge4'][key],saved[kind]['ge4'][key],atol=1e-12,rtol=0)
    counts=roles.loc[roles.role=='test','chemical_group'].value_counts()
    result={'status':'PASS','models_replayed':len(expected_models),'transform_calls_audited':transform_calls,'prediction_max_abs_error':maxerr,'metric_tolerance':1e-12,'independent_lp_alpha':alpha,'lp_weighted_mae':loss,'fit':len(fit),'calibration':len(cal),'test':len(test),'test_groups':len(counts),'test_singleton_groups':int((counts==1).sum()),'test_singleton_compounds':int((counts==1).sum()),'test_largest_group':int(counts.max()),'metrics':replay,'weighted_mae_blend_minus_descriptor':saved['blend']['weighted_mae']-saved['descriptor']['weighted_mae'],'verification_code_sha256':P.sha(__file__)}
    P.put(O/'verification.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    with P.threadpool_limits(limits=2): verify()
