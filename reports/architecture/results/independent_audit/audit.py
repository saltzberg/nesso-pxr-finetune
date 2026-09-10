"""Independent saved-prediction audit. CPU only; no model imports or fitting."""
from pathlib import Path
import hashlib, json, math, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path('/home/dan/projects/nesso-finetune/PXR')
OUT=Path(__file__).resolve().parent
hashes={}; checks=[]; discrepancies=[]
def sha(p):
    p=Path(p); h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    value=h.hexdigest(); hashes[str(p.relative_to(ROOT))]=value; return value
def load(p): sha(p); return json.loads(Path(p).read_text())
def csv(p): sha(p); return pd.read_csv(p, dtype={'record_id':str},keep_default_na=False,float_precision='round_trip')
def check(name,ok,detail=None):
    checks.append({'check':name,'pass':bool(ok),'detail':detail})
    if not ok: discrepancies.append(checks[-1])
def metrics(d):
    y=d.y_true.to_numpy(float); p=d.y_pred.to_numpy(float); se=d.assay_se.to_numpy(float); w=1/np.maximum(se,.1); e=p-y
    result={'n':len(d),'unique_ids':d.record_id.nunique(),'weighted_mae':np.average(abs(e),weights=w),'mae':abs(e).mean(),'spearman':spearmanr(y,p).statistic if np.ptp(p)>0 else None,'bias':e.mean(),'effective_n':w.sum()**2/(w*w).sum(),'invalid_se':int((~np.isfinite(se)|(se<0)).sum())}
    for name,v in [('prediction',p),('observed',y)]:
        result[name]={'mean':v.mean(),'sd':v.std(),'iqr':np.subtract(*np.percentile(v,[75,25])),'min':v.min(),'max':v.max(),'unique':len(np.unique(v))}
    result['sd_ratio']=p.std()/y.std(); result['iqr_ratio']=result['prediction']['iqr']/result['observed']['iqr']
    if np.ptp(p)>0:
        slope=np.mean((p-p.mean())*(y-y.mean()))/np.var(p)
        result['descriptive_calibration_slope']=slope; result['descriptive_calibration_intercept']=y.mean()-slope*p.mean()
    result['strata']={}
    for label,mask in [('lt4',y<4),('ge4',y>=4),('ge6',y>=6)]:
        result['strata'][label]={'n':int(mask.sum()),'mae':abs(e[mask]).mean(),'weighted_mae':np.average(abs(e[mask]),weights=w[mask]),'bias':e[mask].mean()}
    for c in [80,90]:
        if f'covered_{c}' in d:
            result[f'coverage_{c}']=d[f'covered_{c}'].mean(); result[f'width_{c}']=d[f'width_{c}'].mean()
    return result
protocol=load(ROOT/'experiments/20260906_affinity_comparison/train_protocol.json')
load(ROOT/'experiments/20260906_affinity_comparison/controls_protocol.json')
for p in ['README.md','CONTROLS.md']: sha(ROOT/'experiments/20260906_affinity_comparison'/p)
labels=csv(ROOT/protocol['labels']); labelmap=labels.set_index('record_id'); byrow=labels.set_index('row_index')
subsets=csv(ROOT/protocol['strict_root']/'subsets.csv'); assignments=csv(ROOT/protocol['strict_root']/'assignments.csv')
frames={}; folds=[]; role_rows=[]; native_differences=[]
def validate_rows(d,cal,fold,n,name):
    roles=subsets.query('outer_fold==@fold and draw==0 and n_train==@n')
    sets={role:set(byrow.loc[rows.row_index,'record_id']) for role,rows in roles.groupby('role')}
    testids=set(byrow.loc[assignments.loc[assignments.outer_fold==fold,'row_index'],'record_id'])
    check(name+':test identities',set(d.record_id)==testids and d.record_id.is_unique)
    check(name+':calibration identities',set(cal.record_id)==sets['calibration'] and cal.record_id.is_unique)
    check(name+':label budget',len(sets['fit'])==int(.8*n) and len(sets['calibration'])==int(.2*n) and len(sets['fit']|sets['calibration'])==n)
    check(name+':role disjointness',not(sets['fit']&sets['calibration'] or testids&(sets['fit']|sets['calibration'])))
    for role,x in [('test',d),('calibration',cal)]:
        ref=labelmap.loc[x.record_id]
        check(name+':'+role+' labels',np.array_equal(x.y_true.to_numpy(float),ref.pEC50.to_numpy(float)))
        check(name+':'+role+' SE',np.array_equal(x.assay_se.to_numpy(float),ref.pEC50_standard_error.to_numpy(float)))
        check(name+':'+role+' weights',np.array_equal(x.weight.to_numpy(float),1/np.maximum(x.assay_se.to_numpy(float),.1)))
        check(name+':'+role+' finite',np.isfinite(x[['y_true','y_pred','assay_se','weight']].to_numpy(float)).all())
    role_rows.append({'task':name,'fold':fold,'N':n,'fit':len(sets['fit']),'calibration':len(sets['calibration']),'distinct_acquired':len(sets['fit']|sets['calibration']),'test':len(d)})
def intervals(d,cal,stored,name):
    residual=np.sort(abs(cal.y_true.to_numpy(float)-cal.y_pred.to_numpy(float)))
    for c in [80,90]:
        k=math.ceil((len(cal)+1)*c/100); radius=residual[k-1] if k<=len(cal) else float('inf')
        target=stored['intervals'][str(c)]['radius'] if 'intervals' in stored else stored['radii'][str(c)]
        check(name+f':radius {c}',radius==target,{'computed':radius,'saved':target})
        pred=d.y_pred.to_numpy(float); y=d.y_true.to_numpy(float)
        check(name+f':endpoints {c}',np.array_equal(d[f'lower_{c}'],pred-radius) and np.array_equal(d[f'upper_{c}'],pred+radius))
        d[f'covered_{c}']=abs(y-pred)<=radius; d[f'width_{c}']=2*radius
        endpoint=(y>=d[f'lower_{c}'])&(y<=d[f'upper_{c}'])
        check(name+f':coverage metric {c}',float(d[f'covered_{c}'].mean())==(stored['intervals'][str(c)]['coverage'] if 'intervals' in stored else stored['test'][f'coverage_{c}']))
        if np.any(endpoint!=d[f'covered_{c}']): discrepancies.append({'check':name+':endpoint predicate differs','coverage':c,'rows':d.loc[endpoint!=d[f'covered_{c}'],'record_id'].tolist()})
def add(d,method,n,fold,stored=None):
    d=d.copy(); d['fold']=fold; d['method']=method; d['budget']=n
    frames.setdefault((method,n),[]).append(d)
    m=metrics(d); folds.append({'method':method,'budget':n,'fold':fold,**m})
    if stored is not None:
        for key in ['weighted_mae','mae','spearman','bias','effective_n']:
            check(f'{method}/{n}/{fold}:metric {key}',np.isclose(m[key],stored[key],rtol=0,atol=1e-12),{'computed':m[key],'saved':stored[key]})
    return d
neural=sorted((ROOT/'artifacts/cloud/50146714/remote_results/tasks').iterdir())
check('complete neural matrix',len(neural)==20)
for p in neural:
    complete=load(p/'complete.json'); t=complete['task']; n=t['n_train']; fold=t['outer_fold']; method=t['arm']
    for rel,h in complete['files'].items(): check(str(p.name)+':hash '+rel,sha(p/rel)==h)
    state=p/'epochs/epoch040/state.pt'; check(p.name+':final state hash',sha(state)==complete['state_sha256'])
    audit=load(p/'audit.json'); replay=load(p/'replay.json')
    check(p.name+':saved frozen invariance',audit.get('frozen_before')==audit.get('frozen_after') and audit.get('frozen_before') is not None)
    d=csv(p/'test.csv'); cal=csv(p/'calibration.csv'); saved=load(p/'metrics.json')
    validate_rows(d,cal,fold,n,p.name); intervals(d,cal,saved,p.name); add(d,method,n,fold,saved)
    raw=csv(p/'raw_predictions.csv').set_index('record_id')
    for x in [d,cal]:check(p.name+':raw prediction parity',np.array_equal(raw.loc[x.record_id,'y_pred'],x.y_pred))
    base=csv(p/'baseline.csv').set_index('record_id')
    b=d[['record_id','y_true','assay_se','weight']].copy(); b['y_pred']=base.loc[d.record_id,'y_pred'].to_numpy()
    if method=='head_only':add(b,'native_remote',n,fold)
    else:
        ref=next(x for x in frames[('native_remote',n)] if x.fold.iloc[0]==fold)
        check(p.name+':native across arms',np.array_equal(b.y_pred,ref.y_pred))
controlroot=ROOT/'artifacts/experiments/affinity_comparison_20260906/controls_v1'
controls=sorted((controlroot/'cells').iterdir()); check('complete control matrix',len(controls)==30)
load(controlroot/'controls_complete.json')
for p in controls:
    parts=p.name.split('__'); fold=int(parts[1][4:]); n=int(parts[3][1:]); method=parts[4]
    complete=load(p/'complete.json')
    for rel,h in complete['outputs'].items():check(p.name+':hash '+rel,sha(p/rel)==h)
    d=csv(p/'test_predictions.csv'); cal=csv(p/'calibration_predictions.csv'); saved=load(p/'metrics.json')
    validate_rows(d,cal,fold,n,p.name); intervals(d,cal,saved,p.name); add(d,method,n,fold,saved['test'])
    if method=='native_continuous':
        ref=next(x for x in frames[('native_remote',n)] if x.fold.iloc[0]==fold).set_index('record_id')
        diff=d.y_pred.to_numpy()-ref.loc[d.record_id,'y_pred'].to_numpy()
        native_differences.append({'fold':fold,'budget':n,'max_abs':abs(diff).max(),'nonzero':int(np.count_nonzero(diff))})
    if method=='descriptor_lightgbm_rdkit_mordred':
        historical=ROOT/protocol['strict_root']/'panel/tasks'/p.name
        check(p.name+':historical original present',historical.exists())
        if historical.exists():
            for q in (p/'historical_payload').iterdir():
                if q.is_file():check(p.name+':historical byte identity '+q.name,sha(q)==sha(historical/q.name))
pooled=[]
for (method,n),pieces in frames.items():
    d=pd.concat(pieces,ignore_index=True); check(f'{method}/{n}:pooled unique cohort',len(d)==3344 and d.record_id.nunique()==3344)
    pooled.append({'method':method,'budget':n,**metrics(d)})
paired=[]
for n in [100,500]:
    for comparator in ['head_only','native_remote','repr_ridge_stable','descriptor_lightgbm_rdkit_mordred']:
        diffs=[]
        for fold in range(5):
            a=next(x for x in folds if x['method']=='head_plus_lora' and x['budget']==n and x['fold']==fold)
            b=next(x for x in folds if x['method']==comparator and x['budget']==n and x['fold']==fold)
            diffs.append({k:a[k]-b[k] for k in ['weighted_mae','mae','spearman']})
        paired.append({'budget':n,'comparison':'head_plus_lora minus '+comparator,'fold_differences':diffs,'weighted_mae_improved_folds':sum(x['weighted_mae']<0 for x in diffs)})
group_sizes=assignments.groupby('chemical_group').size()
role_summary=[]
for n in [100,500]:
    selected=subsets.query('draw==0 and n_train==@n')
    role_summary.append({'budget':n,'unique_acquired_across_folds':selected.row_index.nunique(),'unique_fit_across_folds':selected.loc[selected.role=='fit','row_index'].nunique(),'unique_calibration_across_folds':selected.loc[selected.role=='calibration','row_index'].nunique()})
role_transitions=[]
for f in range(5):
    small=subsets.query('draw==0 and outer_fold==@f and n_train==100')
    large=subsets.query('draw==0 and outer_fold==@f and n_train==500')
    role_transitions.append({'fold':f,'N100_calibration_becomes_N500_fit':len(set(small.loc[small.role=='calibration','row_index']) & set(large.loc[large.role=='fit','row_index']))})
expected={f'fold{f}__draw0__n{n}__{arm}__seed42' for f in range(5) for n in [100,500] for arm in ['head_only','head_plus_lora']}
check('exact neural task identifiers',{p.name for p in neural}==expected)
result={'scope':'Saved artifacts only; no fitting, model inference, GPU, or external API. Conditional on reused development folds, one draw and initialization; not a fresh holdout or independent confirmation. No refit uncertainty inferred.','target':'Cellular PXR pEC50 from prepared labels. Native 6-mean(log10(IC50/uM)) is pIC50-equivalent, not native cellular pEC50 support.','label_roles':role_rows,'pooled':pooled,'folds':folds,'paired':paired,'native_remote_vs_cpu':native_differences,'checks':checks,'discrepancies':discrepancies,'group_structure':{'groups':len(group_sizes),'singleton_groups':int((group_sizes==1).sum()),'singleton_compounds_fraction':float((group_sizes==1).sum()/3344),'largest':int(group_sizes.max())},'runtime':{'python':sys.executable,'numpy':np.__version__,'pandas':pd.__version__},'verification_limit':'Checks hashes including final neural state; does not execute saved models or independently reproduce training gradients/frozen tensor digests. Stored replay reports are evidence, not new model replay.'}
result['unique_label_role_counts']=role_summary
result['cross_budget_role_transitions']=role_transitions
result['macro_fold_metrics']=[{'method':method,'budget':n,**{k:float(np.mean([x[k] for x in folds if x['method']==method and x['budget']==n])) for k in ['weighted_mae','mae','spearman']}} for method,n in frames]
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'results.json').write_text(json.dumps(clean(result),indent=2)+'\n')
sha(Path(__file__))
(OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in pooled]).to_csv(OUT/'pooled_metrics.csv',index=False)
pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in folds]).to_csv(OUT/'fold_metrics.csv',index=False)
print(pd.read_csv(OUT/'pooled_metrics.csv')[['method','budget','weighted_mae','mae','spearman','sd_ratio','coverage_80','coverage_90']].to_string(index=False))
print('CHECKS',len(checks),'FAILURES',sum(not x['pass'] for x in checks),'DISCREPANCIES',len(discrepancies)); print(json.dumps(clean(discrepancies),indent=2)[:10000]); print('PAIRED',json.dumps(clean(paired))); print('NATIVE',json.dumps(clean(native_differences)))
