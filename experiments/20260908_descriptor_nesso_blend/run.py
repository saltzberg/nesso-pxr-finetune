#!/usr/bin/env python3
"""Bounded nested CPU blend pilot. Only --run fits; --verify never fits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, json, pickle, time, hashlib, importlib, traceback, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EXP=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/experiments/descriptor_nesso_blend_20260908'
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits, threadpool_info
from scipy.stats import spearmanr
C=importlib.import_module('experiments.20260906_affinity_comparison.code.controls')
B=importlib.import_module(C.PREFIX+'baselines')
R=importlib.import_module(C.PREFIX+'readouts')
from nesso_pxr.low_data_models import _folds
from nesso_pxr.low_data_contract import assay_weights, score_predictions
FRESH=ROOT/'artifacts/experiments/affinity_comparison_20260906/controls_v1/fresh_features'
CONFIG={'se_floor':.1,'inner_folds':3,'grids':{'ridge_alpha':[1,100,10000,1000000]}}
torch.set_num_threads(2)
sha=C.digest
put=C.put_json

def blend_weight(y,d,r,w):
    y,d,r,w=map(lambda v:np.asarray(v,dtype=float),(y,d,r,w))
    if not (y.shape==d.shape==r.shape==w.shape) or not all(np.isfinite(v).all() for v in (y,d,r,w)) or np.any(w<=0):
        raise ValueError('invalid blend inputs')
    delta=r-d
    candidates=np.unique(np.r_[0.,1.,np.clip((y[delta!=0]-d[delta!=0])/delta[delta!=0],0,1)])
    losses=np.array([np.average(abs(y-((1-a)*d+a*r)),weights=w) for a in candidates])
    best=int(np.argmin(losses))
    return float(candidates[best]), candidates, losses

def sources():
    ident,assign,cells=C.strict_identity()
    cell=next(c for c in cells if c['n_train']==500 and c['outer_fold']==0)
    binding=C.read_json(FRESH.parent/'controls_bindings.json')
    assert sha(FRESH/'manifest.json')==binding['feature_manifest_sha256']
    meta=C.read_json(FRESH/'manifest.json')
    C.check_hashes(FRESH,meta['outputs'])
    historical_snapshots={sha(p):p for p in (FRESH.parent/'source_snapshot').glob('*') if p.is_file()}
    used={str(Path(m.__file__)) for m in (C,B,R)}
    for p,h in binding['files'].items():
        if sha(p)!=h:
            assert h in historical_snapshots, f'Missing immutable historical snapshot: {p}'
            if p==str(Path(C.__file__)):
                assert Path(p).read_text().replace(', strict=True)', ')')==historical_snapshots[h].read_text().replace(', strict=True)', ')'), 'Only reviewed zip strictness change allowed'
            else:
                assert p not in used and '/src/nesso_pxr/' not in p, p
            print('Historical source drift; immutable snapshot verified (controls: zip strictness only):',p,flush=True)
    fi=C.csv(FRESH/'identities.csv',dtype={'record_id':str})
    assert fi.columns.tolist()==['row_index','record_id']
    pd.testing.assert_frame_equal(fi,ident[['row_index','record_id']],check_dtype=False)
    assert ident.feature_row.is_unique and ident.record_id.is_unique
    assert meta['outcome_labels_read'] is False
    assert meta['extractor_sha256']==binding['files'][str(Path(C.__file__))]
    # Preserve full producer packet/source hash inventory; verify stage plus extracted arrays.
    stage=Path('/nvme1/dan/nesso-finetune/affinity_comparison_20260906/cache_v1/stage/manifest.json')
    assert meta['cache']['files']['stage/manifest.json']==sha(stage)
    assert meta['cache']['binding']==C.read_json(stage)['binding']
    assert [(str(r['row_index']),r['record_id'],str(r['feature_row']),r['canonical_smiles']) for r in meta['cache']['records']]==[tuple(map(str,row)) for row in ident[['row_index','record_id','feature_row','canonical_smiles']].itertuples(index=False,name=None)]
    labels=C.load_labels(C.STRICT,C.ORIGINAL,ident)
    hist=C.read_json(C.STRICT/'panel/run_bindings.json')
    cache=ROOT/hist['config']['descriptor_cache']
    for n in ('descriptors.npz','descriptor_metadata.json'):
        assert sha(cache/n)==hist['files'][str(cache/n)]
    desc,dm=B.load_descriptors(cache,ident[['row_index','record_id','canonical_smiles']])
    arr=np.load(FRESH/'features.npz')
    features={k:arr[k] for k in ('member1','member2')}
    assert all(x.shape==(len(ident),384) and np.isfinite(x).all() for x in features.values())
    inputs={(Path(p) if sha(p)==h else historical_snapshots[h]):h for p,h in binding['files'].items()}
    for p in [Path(C.__file__),FRESH/'manifest.json',FRESH/'identities.csv',FRESH/'features.npz',FRESH.parent/'controls_bindings.json',cache/'descriptors.npz',cache/'descriptor_metadata.json',C.STRICT/'panel/run_bindings.json']+list(EXP.glob('*.py'))+[EXP/'protocol.json']:
        inputs[p]=sha(p)
    return ident,assign,cell,labels,desc,dm,features,inputs

def feature_slice(kind,rows,desc,features,groups,training=False):
    f={'descriptors':desc[rows]} if kind=='descriptor' else {k:v[rows] for k,v in features.items()}
    if training: f['groups']=groups[rows]
    return f

def fit_one(kind,name,tr,va,desc,features,groups,y,se,dm):
    module,method=(B,C.DESCRIPTOR) if kind=='descriptor' else (R,C.RIDGE)
    cfg=dict(CONFIG,descriptor_feature_identity=dm['feature_identity_sha256'])
    # Instrument actual transform calls, not only declared audit indices.
    local=feature_slice(kind,tr,desc,features,groups,True)
    subfolds,_=_folds(local,len(tr),3,42)
    expected=[tr[a] for a,b in subfolds]+[tr]
    x=desc if kind=='descriptor' else np.concatenate([features['member1'],features['member2']],axis=1).astype(float)
    hashes={B._hash_array(x[rr]):rr for rr in expected}
    observed=[]
    def check(xx):
        h=B._hash_array(xx)
        assert h in hashes, 'preprocessing touched rows outside declared tuning train'
        observed.append({'input_sha256':h,'row_index':hashes[h].tolist()})
    if kind=='descriptor':
        original=B.TrainOnlyReducer.fit
        def instrument(self,xx):
            check(xx); return original(self,xx)
        B.TrainOnlyReducer.fit=instrument
    else:
        original=R.fit_preprocessing
        def instrument(xx,ww,scaled):
            check(xx); return original(xx,ww,scaled)
        R.fit_preprocessing=instrument
    try:
        result=module.fit_predict(method,local,y[tr],se[tr],feature_slice(kind,va,desc,features,groups),config=cfg,checkpoint_path=None,seeds=(42,))
    finally:
        if kind=='descriptor': B.TrainOnlyReducer.fit=original
        else: R.fit_preprocessing=original
    assert len(observed)==(4 if kind=='descriptor' else 13)
    for fold in result['selection']['folds']:
        a,b=tr[fold['train_indices']],tr[fold['validation_indices']]
        assert not set(groups[a])&set(groups[b])
        assert not set(np.r_[a,b])&set(va)
    directory=OUT/'models'/name
    C.atomic(directory/'model.pkl',pickle.dumps(result['model_state'],protocol=pickle.HIGHEST_PROTOCOL))
    C.put_array(directory/'predictions.npz',row_index=va,pred=result['pred'])
    put(directory/'audit.json',dict(training_rows=tr,prediction_rows=va,selection=result['selection'],fit_audit=result['fit_audit'],instrumented_transform_calls=observed,fit_seconds=result['fit_seconds']))
    print(name, 'seconds',round(result['fit_seconds'],3),'selected',result['selection']['selected'],flush=True)
    return result['pred']

def metrics(y,p,se):
    primary=score_predictions(y,p,se,.1)
    mask=y>=4
    return {'n':len(y),'weighted_mae':primary['weighted_mae'],'raw_mae':primary['mae'],'spearman':primary['spearman'],'ge4':{'n':int(mask.sum()),'weighted_mae':float(np.average(abs(y[mask]-p[mask]),weights=assay_weights(se[mask],.1))),'raw_mae':float(np.mean(abs(y[mask]-p[mask]))),'spearman':float(spearmanr(y[mask],p[mask]).statistic)}}

def run():
    if OUT.exists(): raise ValueError('Output already exists; refuse accidental refit. Use --verify.')
    t=time.perf_counter()
    ident,assign,cell,labels,desc,dm,features,inputs=sources()
    OUT.mkdir(parents=True)
    put(OUT/'attempt.json',{'start_utc':pd.Timestamp.now(tz='UTC').isoformat(),'protocol_sha256':sha(EXP/'protocol.json'),'scope':'one fold/one budget','pid':os.getpid()})
    try:
        fit,cal,test=(cell['roles'][r] for r in ('fit','calibration','test'))
        assert (len(fit),len(cal))==(400,100)
        groups=assign.chemical_group.astype(str).to_numpy()
        roles=ident.copy(); roles['chemical_group']=groups; roles['role']='unused'
        for role,ix in [('fit',fit),('calibration',cal),('test',test)]: roles.loc[ix,'role']=role
        C.atomic(OUT/'roles.csv',roles.to_csv(index=False).encode())
        folds,meta=_folds({'groups':groups[fit]},len(fit),3,42)
        put(OUT/'nested_roles.json',{'fit_order':fit,'meta_folds':meta})
        C.atomic(OUT/'protocol.json',(EXP/'protocol.json').read_bytes())
        for p,h in inputs.items():
            if p.suffix in ('.py','.json') and p.is_relative_to(ROOT) and 'fresh_features' not in str(p):
                C.atomic(OUT/'source_snapshot'/p.relative_to(ROOT),p.read_bytes())
        C.atomic(OUT/'native_feature_manifest.json',(FRESH/'manifest.json').read_bytes())
        C.atomic(OUT/'native_feature_map.csv',(FRESH/'identities.csv').read_bytes())
        put(OUT/'bindings.json',{'inputs':{str(p):h for p,h in inputs.items()},'descriptor_identity':dm['feature_identity_sha256'],'runtime':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'threadpools':threadpool_info(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']},'native_validation':'Producer native feature/source hash manifest retained; extracted arrays, exact source-feature map, stage and bound source bytes verified. Bulk upstream tensor payloads not reread; immutable producer hash inventory preserved.'})
        # CAL/test label arrays intentionally unavailable to all fitting calls.
        y=np.full(len(labels),np.nan); se=np.full(len(labels),np.nan)
        y[fit]=labels.iloc[fit].pEC50.to_numpy(float); se[fit]=labels.iloc[fit].pEC50_standard_error.to_numpy(float)
        oof=np.full((len(fit),2),np.nan); meta_id=np.full(len(fit),-1)
        for k,(a,b) in enumerate(folds):
            tr,va=fit[a],fit[b]
            assert not set(groups[tr])&set(groups[va])
            for j,kind in enumerate(('descriptor','ridge')):
                oof[b,j]=fit_one(kind,f'meta{k}_{kind}',tr,va,desc,features,groups,y,se,dm)
            meta_id[b]=k
        assert np.isfinite(oof).all() and np.all(meta_id>=0)
        w=assay_weights(se[fit],.1)
        alpha,candidates,losses=blend_weight(y[fit],oof[:,0],oof[:,1],w)
        frame=pd.DataFrame({'row_index':fit,'record_id':ident.iloc[fit].record_id.to_numpy(),'chemical_group':groups[fit],'meta_fold':meta_id,'y_true':y[fit],'assay_se':se[fit],'weight':w,'descriptor':oof[:,0],'ridge':oof[:,1],'blend':(1-alpha)*oof[:,0]+alpha*oof[:,1]})
        C.atomic(OUT/'inner_oof.csv',frame.to_csv(index=False).encode())
        put(OUT/'blend_selection.json',{'alpha_ridge':alpha,'alpha_descriptor':1-alpha,'candidate_alpha':candidates,'candidate_weighted_mae':losses,'selected_weighted_mae':float(losses.min()),'descriptor_inner_weighted_mae':float(np.average(abs(y[fit]-oof[:,0]),weights=w)),'ridge_inner_weighted_mae':float(np.average(abs(y[fit]-oof[:,1]),weights=w)),'selection_rows':fit,'calibration_or_test_labels_used':False})
        evaluate=np.r_[cal,test]
        pred={kind:fit_one(kind,'final_'+kind,fit,evaluate,desc,features,groups,y,se,dm) for kind in ('descriptor','ridge')}
        pred['blend']=(1-alpha)*pred['descriptor']+alpha*pred['ridge']
        ev=ident.iloc[evaluate][['row_index','record_id']].copy(); ev['role']=['calibration']*len(cal)+['test']*len(test)
        for kind,values in pred.items(): ev[kind]=values
        C.atomic(OUT/'predictions.csv',ev.to_csv(index=False).encode())
        # Only now score outer labels. CAL outcomes stay unused.
        testframe=ev.loc[ev.role=='test'].copy()
        yt=labels.iloc[test].pEC50.to_numpy(float); st=labels.iloc[test].pEC50_standard_error.to_numpy(float)
        testframe['y_true']=yt; testframe['assay_se']=st; testframe['weight']=assay_weights(st,.1)
        C.atomic(OUT/'test_predictions.csv',testframe.to_csv(index=False).encode())
        scores={kind:metrics(yt,testframe[kind].to_numpy(),st) for kind in pred}
        put(OUT/'metrics.json',scores)
        # Historical artifacts are opened only after all fits/weight/predictions are frozen.
        hd,hpred=C.verify_descriptor(C.STRICT/'panel',cell,labels,descriptors=desc)
        np.testing.assert_allclose(pred['descriptor'],hpred['pred'],rtol=0,atol=0)
        hr=FRESH.parent/'cells'/C.task_id(cell,C.RIDGE)
        complete=C.read_json(hr/'complete.json'); C.check_hashes(hr,complete['outputs'])
        rp=np.load(hr/'raw_predictions.npz')
        assert np.array_equal(rp['row_index'],evaluate)
        np.testing.assert_array_equal(pred['ridge'],rp['pred'])
        put(OUT/'baseline_parity.json',{'fresh_descriptor_vs_historical_max_abs':0.,'fresh_ridge_vs_historical_max_abs':0.,'descriptor_complete_path':str(hd/'complete.json'),'descriptor_complete_sha256':sha(hd/'complete.json'),'ridge_complete_path':str(hr/'complete.json'),'ridge_complete_sha256':sha(hr/'complete.json'),'historical_models_used_in_training':False})
        put(OUT/'failures.json',[])
        put(OUT/'runtime.json',{'elapsed_seconds':time.perf_counter()-t,'finish_utc':pd.Timestamp.now(tz='UTC').isoformat(),'actual_final_recipe_fits':8,'meta_fits':6,'full_fit_models':2,'threads':2,'gpu_used':False})
        allow=[p for p in OUT.rglob('*') if p.is_file() and 'source_snapshot' not in p.parts]
        put(OUT/'complete.json',{'outputs':{str(p.relative_to(OUT)):sha(p) for p in allow},'status':'fits_and_historical_parity_complete; independent verify required'})
        print(json.dumps(scores,indent=2),flush=True)
    except BaseException:
        put(OUT/'failures.json',[{'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-t}]); raise

if __name__=='__main__':
    with threadpool_limits(limits=2):
        if sys.argv[1:]==['--run']: run()
        else: raise SystemExit('Usage: run.py --run; independent replay: verify.py')
