#!/usr/bin/env python3
"""Locked continuation: import the immutable pilot functions, change cell/output only."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import importlib, importlib.util, json, contextlib, io, unittest
P=importlib.import_module('experiments.20260908_descriptor_nesso_blend.run')
EXP=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/experiments/descriptor_nesso_blend_allfolds_20260908'
PILOT=P.OUT
PILOT_EXP=P.EXP
ORIGINAL_SOURCES=P.sources
ORIGINAL_PUT=P.put
sys.modules['run']=P
spec=importlib.util.spec_from_file_location('pilot_verify',PILOT_EXP/'verify.py')
V=importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
np=P.np; pd=P.pd
METHODS=('descriptor','ridge','blend')

def inventory(path):
    return {str(p.relative_to(ROOT)):P.sha(p) for p in sorted(path.rglob('*')) if p.is_file()}

def immutable():
    protocol=P.C.read_json(EXP/'protocol.json')
    for p,h in protocol['pilot_immutable_files'].items():
        assert P.sha(ROOT/p)==h,p
    assert inventory(PILOT)=={p:h for p,h in protocol['pilot_immutable_files'].items() if p.startswith(str(PILOT.relative_to(ROOT))+'/')}
    return protocol

def prepare():
    assert not OUT.exists() and not (EXP/'protocol.json').exists()
    protocol=P.C.read_json(PILOT_EXP/'protocol.json')
    protocol.update(experiment=EXP.name,status='approved exploratory locked-recipe continuation; persisted before fits',
      approval={'user':'redo with locked recipe to see if it holds','scope':'remaining outer folds 1-4, N500 draw0; aggregate immutable pilot fold0','learned_parameters':'rerun identical nested selection independently per fold; do not freeze pilot weight or hyperparameters'},
      scope={'split':'strict_chemical','outer_folds':[0,1,2,3,4],'new_fit_outer_folds':[1,2,3,4],'draw':0,'n_train':500,'fit':400,'calibration':100},
      boundaries='Only new experiment and matching artifacts. No GPU/cloud, new model or tuning changes, source changes, root README/site edits. Reused development folds; not fresh holdout or prospective confirmation.',
      pilot_reuse_map={'0':{'path':str(PILOT.relative_to(ROOT)),'action':'read-only original eight model systems and predictions; no refit/copy rebinding','complete_sha256':P.sha(PILOT/'complete.json')}},
      pilot_immutable_files={**inventory(PILOT),**inventory(PILOT_EXP)},
      aggregation='Each of 3344 unique compound IDs once per method; pooled and unweighted macro-fold weighted/raw MAE and Spearman, per-fold values, >=4/<4 strata and paired blend-minus-descriptor changes.')
    ORIGINAL_PUT(EXP/'protocol.json',protocol)
    OUT.mkdir(parents=True)
    ORIGINAL_PUT(OUT/'protocol.json',protocol)
    print('PREFIT PROTOCOL PERSISTED',EXP/'protocol.json')

def configure(fold):
    P.OUT=PILOT if fold==0 else OUT/f'fold{fold}'
    P.EXP=PILOT_EXP if fold==0 else EXP
    def sources():
        ident,assign,_,labels,desc,dm,features,inputs=ORIGINAL_SOURCES()
        _,_,cells=P.C.strict_identity()
        matching=[c for c in cells if c['n_train']==500 and c['draw']==0 and c['outer_fold']==fold]
        assert len(matching)==1
        for path in list(PILOT_EXP.glob('*.py'))+[PILOT_EXP/'protocol.json']+list(EXP.glob('*.py'))+[EXP/'protocol.json']:
            inputs[path]=P.sha(path)
        return ident,assign,matching[0],labels,desc,dm,features,inputs
    P.sources=sources
    P.put=ORIGINAL_PUT

def run():
    immutable()
    for fold in range(1,5):
        configure(fold)
        if P.OUT.exists():
            assert (P.OUT/'complete.json').exists(), 'Incomplete attempt; manual review required, no automatic refit'
            P.C.check_hashes(P.OUT,P.C.read_json(P.OUT/'complete.json')['outputs'])
            print('REUSED COMPLETED NEW FOLD',fold)
            continue
        P.run()
        immutable()
    print('FOUR NEW FOLDS COMPLETE')

def summarize(frame):
    result={}
    for method in METHODS:
        y=frame.y_true.to_numpy(); p=frame[method].to_numpy(); se=frame.assay_se.to_numpy()
        m=V.independent_metrics(y,p,se)
        low=y<4; w=1/np.maximum(se[low],.1)
        from scipy.stats import rankdata
        m['lt4']={'n':int(low.sum()),'weighted_mae':float(np.average(abs(y[low]-p[low]),weights=w)), 'raw_mae':float(np.mean(abs(y[low]-p[low]))),'spearman':float(np.corrcoef(rankdata(y[low]),rankdata(p[low]))[0,1])}
        result[method]=m
    return result

def aggregate(write=True):
    frames=[]; per={}; weights={}
    for fold in range(5):
        directory=PILOT if fold==0 else OUT/f'fold{fold}'
        f=P.C.csv(directory/'test_predictions.csv',dtype={'record_id':str})
        f['outer_fold']=fold; f['provenance']='immutable_pilot_reuse' if fold==0 else 'new_fit'
        frames.append(f); per[str(fold)]=summarize(f)
        weights[str(fold)]=P.C.read_json(directory/'blend_selection.json')['alpha_ridge']
    frame=pd.concat(frames,ignore_index=True).sort_values('row_index').reset_index(drop=True)
    assert len(frame)==3344 and frame.record_id.nunique()==3344 and frame.row_index.is_unique
    ident,assign,_=P.C.strict_identity()
    assert set(frame.record_id)==set(ident.record_id)
    assert np.isfinite(frame[list(METHODS)].to_numpy()).all()
    macro={m:{k:float(np.mean([per[str(f)][m][k] for f in range(5)])) for k in ('weighted_mae','raw_mae','spearman')} for m in METHODS}
    for m in METHODS:
        for stratum in ('ge4','lt4'):
            macro[m][stratum]={k:float(np.mean([per[str(f)][m][stratum][k] for f in range(5)])) for k in ('weighted_mae','raw_mae','spearman')}
    pooled=summarize(frame)
    delta={str(f):per[str(f)]['blend']['weighted_mae']-per[str(f)]['descriptor']['weighted_mae'] for f in range(5)}
    result={'unique_test_ids':3344,'methods':list(METHODS),'pooled':pooled,'macro_fold':macro,'per_fold':per,'alpha_ridge_by_fold':weights,'weighted_mae_blend_minus_descriptor_by_fold':delta,'improving_folds':sum(d<0 for d in delta.values()),'equal_folds':sum(d==0 for d in delta.values()),'worsening_folds':sum(d>0 for d in delta.values()),'reused_model_systems':8,'new_model_systems':32,'new_folds':4,'fresh_holdout':False}
    if write:
        P.C.atomic(OUT/'all_test_predictions.csv',frame.to_csv(index=False).encode())
        ORIGINAL_PUT(OUT/'metrics.json',result)
    else:
        old=P.C.csv(OUT/'all_test_predictions.csv',dtype={'record_id':str})
        pd.testing.assert_frame_equal(old,frame,check_dtype=False)
        assert P.C.read_json(OUT/'metrics.json')==result
    return result

def verify():
    immutable()
    replay=[]
    original_fit=P.fit_one
    def forbidden(*args,**kwargs): raise AssertionError('Verifier attempted fit')
    P.fit_one=forbidden
    try:
        for fold in range(5):
            configure(fold)
            def capture(path,data):
                assert path==P.OUT/'verification.json'
                ORIGINAL_PUT(OUT/f'verification_fold{fold}.json',data)
                replay.append(data)
            P.put=capture
            with contextlib.redirect_stdout(io.StringIO()): V.verify()
            before=inventory(P.OUT)
            try: P.run()
            except ValueError as e: assert 'Output already exists' in str(e)
            else: raise AssertionError('refit guard failed')
            assert before==inventory(P.OUT)
            print('VERIFIED fold',fold,'models',replay[-1]['models_replayed'],'maxerr',replay[-1]['prediction_max_abs_error'])
    finally:
        P.fit_one=original_fit; P.put=ORIGINAL_PUT
    result=aggregate(write=not (OUT/'metrics.json').exists())
    immutable()
    report={'status':'PASS','folds':5,'models_replayed':sum(r['models_replayed'] for r in replay),'new_model_systems':32,'reused_model_systems':8,'transform_calls_audited':sum(r['transform_calls_audited'] for r in replay),'max_prediction_abs_error':max(r['prediction_max_abs_error'] for r in replay),'unique_test_ids_per_method':3344,'no_refit_guards_passed':5,'pilot_bytes_unchanged':True,'all_source_and_output_hashes_verified':True,'failures':[]}
    ORIGINAL_PUT(OUT/'verification.json',report)
    ORIGINAL_PUT(OUT/'failures.json',[])
    outputs=[OUT/'metrics.json',OUT/'all_test_predictions.csv',OUT/'verification.json',OUT/'protocol.json',OUT/'failures.json']+list(OUT.glob('verification_fold*.json'))
    ORIGINAL_PUT(OUT/'complete.json',{'outputs':{str(p.relative_to(OUT)):P.sha(p) for p in outputs},'status':'PASS','fold0_reuse':str(PILOT.relative_to(ROOT)),'new_folds':[1,2,3,4]})
    print(json.dumps({'verification':report,'pooled':result['pooled'],'macro_fold':result['macro_fold'],'weights':result['alpha_ridge_by_fold'],'deltas':result['weighted_mae_blend_minus_descriptor_by_fold']},indent=2))

if __name__=='__main__':
    with P.threadpool_limits(limits=2):
        {'--prepare':prepare,'--run':run,'--verify':verify}[sys.argv[1]]()
