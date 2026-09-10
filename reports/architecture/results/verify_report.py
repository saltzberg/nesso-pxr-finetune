"""Independent report acceptance: data, paired resamples, canonical table and HTTP bytes."""
from pathlib import Path
import json,hashlib,re,math
from urllib.request import build_opener,ProxyHandler
from urllib.parse import urljoin,urlparse
from html.parser import HTMLParser
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[2];OUT=R/'reports/architecture/results'
read=lambda p:pd.read_csv(p,float_precision='round_trip')
a=read(H/'pooled_metrics.csv');b=read(H/'independent_audit/pooled_metrics.csv');j=a.merge(b,on=['budget','method'],validate='one_to_one',suffixes=('_parent','_independent'));assert len(j)==10
cols=['weighted_mae','mae','spearman','bias','effective_n','sd_ratio','iqr_ratio','coverage_80','coverage_90','width_80','width_90'];maxdiff={c:float(np.max(np.abs(j[c+'_parent']-j[c+'_independent']))) for c in cols};assert max(maxdiff.values())<1e-11
fa=read(H/'fold_metrics.csv');fb=read(H/'independent_audit/fold_metrics.csv')
if 'outer_fold' in fb:fb=fb.rename(columns={'outer_fold':'fold'})
fj=fa.merge(fb,on=['budget','method','fold'],validate='one_to_one',suffixes=('_parent','_independent'));assert len(fj)==50
for c in ['weighted_mae','mae','spearman']:assert np.max(np.abs(fj[c+'_parent']-fj[c+'_independent']))<1e-12
# Reconstruct a saved bootstrap draw through compound multiplicities, not group dot-product code.
d=read(H/'heldout_predictions.csv');boot=np.load(H/'bootstrap_draws.npz');groups=boot['chemical_groups'];counts=boot['counts'];assert counts.shape[0]==4000
base=d[(d.budget==100)&(d.method=='head_only')].sort_values('record_id').reset_index(drop=True);gmap={str(g):i for i,g in enumerate(groups)};indices=base.chemical_group.astype(str).map(gmap).to_numpy();assert not pd.isna(indices).any()
for fold in range(5):
 gs=np.unique(indices[base.fold==fold]);assert np.all(counts[:,gs].sum(axis=1)==len(gs))
for n in [100,500]:
 x=d[(d.budget==n)&(d.method=='head_plus_lora')].sort_values('record_id').reset_index(drop=True);y=d[(d.budget==n)&(d.method=='head_only')].sort_values('record_id').reset_index(drop=True);loss=np.abs(x.y_pred-x.y_true)-np.abs(y.y_pred-y.y_true);w=y.weight.to_numpy();numerators=np.bincount(indices,weights=w*loss,minlength=len(groups));denominators=np.bincount(indices,weights=w,minlength=len(groups))
 for k in [0,100,3999]:
  multiplicity=counts[k,indices];expanded=np.average(loss,weights=multiplicity*w);grouped=(counts[k]@numerators)/(counts[k]@denominators);assert abs(expanded-grouped)<1e-14
# Check every rounded canonical score-table entry against computed values.
text=(R/'experiments/20260906_affinity_comparison/RESULTS.md').read_text();mapping={'Native unchanged':'native_continuous','Head-only':'head_only','Head + LoRA':'head_plus_lora','Ridge on frozen Nesso vectors':'repr_ridge_stable','Ligand-descriptor LightGBM':'descriptor_lightgbm_rdkit_mordred'};table_rows=0
for line in text.splitlines():
 if re.match(r'\| (100|500) \|',line):
  n,name,*values=[s.strip() for s in line.strip('|').split('|')];v=a[(a.budget==int(n))&(a.method==mapping[name])].iloc[0]
  assert values==[f'{v[c]:.4f}' for c in ['weighted_mae','mae','spearman']];table_rows+=1
assert table_rows==10
manifest=json.loads((OUT/'report_manifest.json').read_text())
for name,h in manifest['files'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h,name
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  at=dict(attrs)
  for key in ['href','src']:
   if key in at and not at[key].startswith('#'):self.links.append(at[key])
parser=Links();parser.feed((OUT/'index.html').read_text());urls=sorted(set(urljoin('http://192.168.6.154:8776/results/',x) for x in parser.links));opener=build_opener(ProxyHandler({}));verified=[]
for url in urls:
 response=opener.open(url,timeout=20);data=response.read();assert response.status==200
 rel=urlparse(url).path
 expected=(R/'reports/architecture/index.html') if rel=='/' else (R/'reports/architecture'/rel.removeprefix('/'))
 if rel=='/results/':expected=OUT/'index.html'
 if rel=='/':expected=R/'reports/architecture/index.html'
 if expected.is_dir():expected=expected/'index.html'
 assert expected.is_file() and data==expected.read_bytes(),url
 verified.append(url)
assert opener.open('http://192.168.6.154:8776/results/',timeout=20).read()==(OUT/'index.html').read_bytes()
report={'status':'PASS','independent_pooled_rows':len(j),'independent_fold_rows':len(fj),'max_metric_discrepancies':maxdiff,'canonical_table_rows':table_rows,'bootstrap_reconstruction':'PASS','http_exact_byte_links':len(verified),'verified_urls':verified,'report_manifest_sha256':hashlib.sha256((OUT/'report_manifest.json').read_bytes()).hexdigest()}
(H/'acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
