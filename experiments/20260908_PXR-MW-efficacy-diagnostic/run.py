"""No inference or predictor fitting: source-bound retrospective diagnostics."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import json, hashlib, argparse, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=Path(__file__).resolve().parent; R=E.parents[1]
S={}
def bind(p):
 p=Path(p); S[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest(); return p
def read(p): return pd.read_csv(bind(p),float_precision='round_trip',keep_default_na=False)
def numeric(d,cols):
 for c in cols: d[c]=pd.to_numeric(d[c],errors='coerce')
 return d
def partial(d,x,y,controls):
 z=d[[x,y]+controls].dropna(); a=np.column_stack([np.ones(len(z))]+[rankdata(z[c]) for c in controls]); xx=rankdata(z[x]); yy=rankdata(z[y]); return float(np.corrcoef(xx-a@np.linalg.lstsq(a,xx,rcond=None)[0],yy-a@np.linalg.lstsq(a,yy,rcond=None)[0])[0,1])
def main(out):
 out.mkdir(exist_ok=True,parents=True); bind(E/'protocol.json'); bind(Path(__file__))
 F=R/'artifacts/experiments/affinity_comparison_20260906/controls_v1/fresh_features'; A=R/'artifacts/analysis/affinity_comparison_20260907'
 ids=read(F/'identities.csv'); fm=json.loads(bind(F/'manifest.json').read_text())
 for n in ['identities.csv','native_points.npy']: assert hashlib.sha256(bind(F/n).read_bytes()).hexdigest()==fm['outputs'][n]
 native=np.load(F/'native_points.npy'); assert ids.record_id.is_unique and len(ids)==3344 and native.shape==(3344,)
 m=read(R/'data/published/modeling_manifest.csv'); assert m.record_id.is_unique
 d=ids.merge(m,on='record_id',validate='one_to_one'); assert d.record_id.tolist()==ids.record_id.tolist() and set(d.modeling_role)=={'development'}
 inp=read(R/'artifacts/experiments/low_data_20260905_cut035_run2/prepared/inputs.csv').set_index('record_id').loc[d.record_id]
 assert inp.feature_row.tolist()==d.feature_row.tolist() and inp.canonical_smiles.tolist()==d.identity_canonical_smiles.tolist()
 assert np.array_equal(inp.pEC50,d.pEC50) and np.array_equal(inp.pEC50_standard_error,d.pEC50_standard_error)
 d['native']=native; d['native_affinity_log10_uM']=6-d.native
 chem=read(A/'mover_chemistry_all.csv'); assert chem.record_id.is_unique
 c=chem.set_index('record_id').loc[d.record_id]; d['MW']=c.MW.to_numpy(); assert c.smiles.tolist()==d.identity_canonical_smiles.tolist()
 raw=read(R/'data/openadmet/pxr-challenge_TRAIN.csv'); curated=read(R/'data/derived/pxr_train_curated_weighted.csv'); assert raw['Molecule Name'].is_unique and raw.OCNT_ID.is_unique
 rr=raw.set_index('Molecule Name').loc[d.original_id]; cc=curated.set_index('Molecule Name').loc[d.original_id]
 assert rr.OCNT_ID.tolist()==d.OCNT_ID.tolist() and rr.SMILES.tolist()==d.raw_smiles.tolist(); assert np.array_equal(rr.pEC50,d.pEC50)
 cols={'Emax_estimate (log2FC vs. baseline)':'Emax_log2FC','Emax.vs.pos.ctrl_estimate (dimensionless)':'Emax_ratio','Emax_std.error (log2FC vs. baseline)':'Emax_log2FC_SE','Emax.vs.pos.ctrl_std.error (dimensionless)':'Emax_ratio_SE','pEC50_std.error (-log10(molarity))':'raw_pEC50_SE'}
 for col,n in cols.items():
  assert np.array_equal(rr[col],cc[col]); d[n]=pd.to_numeric(rr[col],errors='coerce').to_numpy()
 for col in rr.columns:
  if 'ci.' in col: d[col]=rr[col].to_numpy()
 d['assay_batch']=rr['OCNT Batch'].to_numpy(); d['small']=d.MW<215; d['emax_qualified']=d.Emax_ratio>=.6
 pred=read(A/'heldout_predictions.csv'); ref=pred[(pred.budget==100)&(pred.method=='native_continuous')].set_index('record_id').loc[d.record_id]
 d['chemical_group']=ref.chemical_group.to_numpy(); assert np.isfinite(d.chemical_group).all()
 discrepancies={'chemistry_native_max_abs':float(np.max(abs(c.native.to_numpy()-native))),'saved_reference_native_max_abs':float(np.max(abs(ref.y_pred.to_numpy()-native))),'prepared_vs_raw_SE_max_abs':float(np.max(abs(d.pEC50_standard_error-d.raw_pEC50_SE)))}
 for b in [100,500]:
  for method in ['head_only','head_plus_lora']:
   z=pred[(pred.budget==b)&(pred.method==method)].set_index('record_id').loc[d.record_id]; assert np.array_equal(z.chemical_group,d.chemical_group)
   d[f'{method}_{b}']=z.y_pred.to_numpy(); d[f'downward_{method}_{b}']=d.native-z.y_pred.to_numpy()
 binary=read(R/'reports/transfer_ablation/binary_head_reanalysis/development_predictions.csv').set_index('record_id').loc[d.record_id]
 d['native_binary_probability']=binary.binder_probability.to_numpy(); d['native_binary_logit']=binary.binary_logit.to_numpy(); d['historical_binary_packet_scalar']=binary.pIC50_equivalent.to_numpy(); discrepancies['historical_binary_packet_scalar_max_abs']=float(np.max(abs(binary.pIC50_equivalent.to_numpy()-native))); discrepancies['historical_binary_packet_scalar_differing_gt1e_5']=int((abs(binary.pIC50_equivalent.to_numpy()-native)>1e-5).sum())
 # Same-packet binary logits are already cached: use these instead of the older packet.
 d['historical_binary_probability']=d.native_binary_probability
 with np.load(bind(F/'features.npz')) as f: scal=f['scalar_features']
 assert hashlib.sha256((F/'features.npz').read_bytes()).hexdigest()==fm['outputs']['features.npz']
 assert scal.shape==(3344,4) and np.array_equal((6-(scal[:,0]+scal[:,2])/np.float32(2)).astype(float),native)
 from scipy.special import expit, logit
 d['native_binary_member1_logit']=scal[:,1]; d['native_binary_member2_logit']=scal[:,3]
 d['native_binary_probability']=(expit(scal[:,1])+expit(scal[:,3]))/np.float32(2)
 d['native_binary_logit']=logit(d.native_binary_probability)
 # Source qualification is audited, not used as a new selection filter.
 assert ((d.role=='development_primary')==d.emax_qualified).all()
 d.to_csv(out/'merged_compounds.csv',index=False)
 outcomes=['Emax_ratio','pEC50_standard_error']; variables=['MW','native','pEC50','Emax_ratio','Emax_log2FC','pEC50_standard_error','Emax_ratio_SE','Emax_log2FC_SE','native_binary_probability']
 assert np.isfinite(d[variables]).all().all()
 lower=max(d.loc[d.small,'native'].min(),d.loc[~d.small,'native'].min()); upper=min(d.loc[d.small,'native'].max(),d.loc[~d.small,'native'].max()); support=d.native.between(lower,upper)
 d['score_bin']=np.floor(d.native/.5)*.5
 strata=[]
 for scope,mask in [('full',np.ones(len(d),bool)),('common_support',support)]:
  for small,z in d[mask].groupby('small'):
   row=dict(scope=scope,small=bool(small),n=len(z),groups=z.chemical_group.nunique(),qualified=int(z.emax_qualified.sum()))
   for col in variables:
    for stat,v in [('median',z[col].median()),('q25',z[col].quantile(.25)),('q75',z[col].quantile(.75)),('min',z[col].min()),('max',z[col].max())]: row[col+'_'+stat]=float(v)
   strata.append(row)
 pd.DataFrame(strata).to_csv(out/'stratum_summaries.csv',index=False)
 bins=d.groupby(['score_bin','small']).agg(n=('record_id','size'),groups=('chemical_group','nunique'),Emax_median=('Emax_ratio','median'),SE_median=('pEC50_standard_error','median'),MW_median=('MW','median')).reset_index(); bins.to_csv(out/'score_bins.csv',index=False)
 ass=[]
 for scope,mask in [('full',np.ones(len(d),bool)),('small',d.small),('large',~d.small),('common_support',support)]:
  z=d[mask]
  for x,y in itertools.combinations(variables,2): ass.append(dict(scope=scope,x=x,y=y,n=len(z),rho=float(spearmanr(z[x],z[y]).statistic)))
 pd.DataFrame(ass).to_csv(out/'associations.csv',index=False)
 adj=[]
 for y in outcomes:
  for controls in [['native'],['native','pEC50']]: adj.append(dict(x='MW',y=y,controls=','.join(controls),n=len(d),rho=partial(d,'MW',y,controls)))
 for x in [c for c in d if c.startswith('downward_')]:
  for y in outcomes:
   for controls in [[],['MW','native','pEC50']]: adj.append(dict(x=x,y=y,controls=','.join(controls),n=len(d),rho=partial(d,x,y,controls)))
 pd.DataFrame(adj).to_csv(out/'adjusted_and_correction_associations.csv',index=False)
 # Cluster bootstrap samples groups once per draw, preserves every member and multiplicity.
 groups=[np.asarray(v) for v in d.groupby('chemical_group').indices.values()]; rng=np.random.default_rng(20260908); boot={}
 for scope,mask in [('full',np.ones(len(d),bool)),('common_support',support.to_numpy())]:
  for y in outcomes: boot[(scope,y)]=[]
 for _ in range(1000):
  ii=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]); z=d.iloc[ii]
  for scope,mask in [('full',np.ones(len(d),bool)),('common_support',support.to_numpy())]:
   q=z[mask[ii]]
   for y in outcomes: boot[(scope,y)].append(float(q.loc[q.small,y].median()-q.loc[~q.small,y].median()))
 contrasts=[]
 for (scope,y),vals in boot.items():
  z=d if scope=='full' else d[support]; small=z[z.small]; large=z[~z.small]
  contrasts.append(dict(scope=scope,outcome=y,n_small=len(small),n_large=len(large),groups_small=small.chemical_group.nunique(),groups_large=large.chemical_group.nunique(),delta_median=float(small[y].median()-large[y].median()),ci_low=float(np.quantile(vals,.025)),ci_high=float(np.quantile(vals,.975)),bootstrap_draws=1000))
 pd.DataFrame(contrasts).to_csv(out/'primary_contrasts.csv',index=False)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey='row',layout='constrained')
 for j,small in enumerate([True,False]):
  z=d[d.small==small]
  for i,y in enumerate(outcomes):
   ax=axs[i,j]; ax.scatter(z.native,z[y],s=10,alpha=.45,color='#176b83' if small else '#b66435',edgecolors='none'); ax.set_yscale('log'); ax.set_title(f'{"<215" if small else "≥215"} Da · n={len(z)} · ρ={spearmanr(z.native,z[y]).statistic:.2f}'); ax.set_xlabel('Native continuous IC50-equivalent score → favorable'); ax.set_ylabel('Emax / positive control (dimensionless)' if i==0 else 'Reported pEC50 SE (log10 molarity)'); ax.grid(alpha=.15)
   if i==0: ax.axhline(.6,color='gray',ls=':',lw=1)
 fig.suptitle('All development compounds: native score versus activation and fit uncertainty\nDotted line: historical Emax qualification only; log y axes, no omitted tails',fontsize=12); fig.savefig(out/'native_score_outcomes.png',dpi=180); plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 for ax,x,y in zip(axs,['MW','MW','Emax_ratio'],['Emax_ratio','pEC50_standard_error','pEC50_standard_error']):
  for small,color in [(False,'#b66435'),(True,'#176b83')]:
   z=d[d.small==small];ax.scatter(z[x],z[y],s=9,alpha=.35,color=color,edgecolors='none',label=f'{"<215" if small else "≥215"} Da (n={len(z)})')
  ax.set_xlabel('MW (Da)' if x=='MW' else 'Emax / positive control');ax.set_ylabel('Emax / positive control' if y=='Emax_ratio' else 'Reported pEC50 SE');ax.set_yscale('log');ax.set_title(f'Full-cohort Spearman ρ={spearmanr(d[x],d[y]).statistic:.2f}');ax.grid(alpha=.15)
  if x!='MW':ax.set_xscale('log')
 axs[0].legend(frameon=False,fontsize=8);fig.savefig(out/'mw_emax_uncertainty.png',dpi=180);plt.close(fig)
 audit=dict(n=len(d),raw_n=len(raw),curated_n=len(curated),development_primary=int(d.emax_qualified.sum()),lower_emax=int((~d.emax_qualified).sum()),small=int(d.small.sum()),exactly215=int((d.MW==215).sum()),groups=len(groups),singleton_groups=sum(len(g)==1 for g in groups),largest_group=max(map(len,groups)),common_support=[float(lower),float(upper)],missingness=d[variables].isna().sum().to_dict(),identity_checks='PASS: record/order/feature row/identity SMILES/OCNT/raw SMILES/labels/curated efficacy/qualification/group parity',discrepancies=discrepancies,raw_hard_exclusions=sorted(set(raw['Molecule Name'])-set(curated['Molecule Name'])),runtime=dict(python=os.sys.executable,numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,threads=1))
 (out/'audit.json').write_text(json.dumps(audit,indent=2)); (out/'source_hashes.json').write_text(json.dumps(S,indent=2,sort_keys=True)); print(json.dumps(audit,indent=2)); print(pd.DataFrame(contrasts).to_string(index=False)); print(pd.DataFrame(adj).to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=E/'artifacts');main(p.parse_args().out)
