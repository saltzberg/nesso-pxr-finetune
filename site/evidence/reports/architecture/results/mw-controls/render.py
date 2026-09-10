from pathlib import Path
import json,html,hashlib,shutil
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/dan/projects/nesso-finetune/PXR');E=R/'experiments/20260908_mw_controls';H=R/'artifacts/analysis/mw_controls_20260908';O=R/'reports/architecture/results/mw-controls';O.mkdir(parents=True,exist_ok=True)
assert json.loads((H/'independent_verification.json').read_text())['status']=='PASS'
m=pd.read_csv(H/'metrics.csv');f=pd.read_csv(H/'fold_metrics.csv');names={'mw':'MW only','native':'Native score, fitted mapping','native_mw':'Native score + MW','native_mw_interaction':'Native score + MW + interaction','head_only':'Head-only','head_plus_lora':'Head + LoRA','native_continuous':'Native unchanged','repr_ridge_stable':'Frozen-feature ridge','descriptor_lightgbm_rdkit_mordred':'Descriptor LightGBM'}
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','font.size':9})
fig,axes=plt.subplots(3,2,figsize=(10,8),sharex=True,sharey=True,layout='constrained');shown=['mw','native','native_mw','native_mw_interaction','head_plus_lora'];vals=[]
for row,sub in enumerate(['full','lt215','gt215']):
 for col,n in enumerate([100,500]):
  ax=axes[row,col];a=f[(f.budget==n)&(f.subset==sub)].pivot(index='fold',columns='method',values='weighted_mae');ax.axvline(0,color='#adb5bd',lw=.7)
  for j,name in enumerate(shown):
   delta=(a[name]-a.head_only).to_numpy();ax.scatter(delta,np.full(len(delta),j),s=18,alpha=.65,color='#526572');vals.extend(delta)
  ax.set_title(f'N{n} · '+{'full':'Full set','lt215':'<215 Da','gt215':'>215 Da'}[sub],loc='left');ax.set_yticks(range(5),[names[k] for k in shown]);ax.set_ylim(4.5,-.5)
  if row==2:ax.set_xlabel('Weighted MAE difference from head-only')
span=max(abs(v) for v in vals)*1.08
for ax in axes.flat:ax.set_xlim(-span,span)
for ext in ['png','svg']:fig.savefig(O/('paired-folds.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig);assert len(vals)==150
source=(E/'README.md').read_text();body=''
for para in source.split('\n\n'):
 if para.startswith('#'):lev=len(para)-len(para.lstrip('#'));body+=f'<h{lev}>'+html.escape(para[lev:].strip())+f'</h{lev}>'
 else:body+='<p>'+html.escape(para)+'</p>'
body+='<h2>Matched held-out performance</h2><p>Entries are weighted MAE / MAE / Spearman. Errors lower is better; ranking higher is better.</p>'
for n in [100,500]:
 body+=f'<h3>N{n}</h3><div class="scroll"><table><tr><th>Model</th><th>Full (3,344)</th><th>&lt;215 Da (237)</th><th>&gt;215 Da (3,107)</th></tr>'
 for name in names:
  body+='<tr><td>'+names[name]+'</td>'
  for sub in ['full','lt215','gt215']:
   a=m[(m.budget==n)&(m.method==name)&(m.subset==sub)].iloc[0];body+='<td>'+ ' / '.join(f'{a[k]:.3f}' for k in ['weighted_mae','mae','spearman'])+'</td>'
  body+='</tr>'
 body+='</table></div>'
body+='<figure><img src="paired-folds.png" alt="Individual fold differences in weighted MAE from head-only"><figcaption>Each dot is one saved fold. Negative favors the indicated model over head-only. Shared scales retain all differences; dots can overlap. These are reused development folds, not independent prospective experiments.</figcaption><a href="paired-folds.svg">SVG</a> · <a href="paired-folds.png">PNG</a></figure>'
files=[]
for root,items in [(E,['README.md','protocol.json','run.py','verify.py','render.py']),(H,['metrics.csv','fold_metrics.csv','contrasts.csv','predictions.csv','verification.json','independent_verification.json','design.md'])]:
 for name in items:shutil.copy2(root/name,O/name);files.append(name)
body+='<h2>Downloads</h2>'+''.join(f'<p><a href="{n}">{n}</a></p>' for n in files)
(O/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nesso MW controls</title><style>body{font:15px/1.6 system-ui;color:#26323b;max-width:1100px;margin:auto;padding:24px}h1{font-size:28px}h2{margin-top:32px}table{border-collapse:collapse;white-space:nowrap}td,th{padding:8px 14px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}img{width:100%}figure{margin:24px 0}a{color:#2166ac}figcaption{font-size:13px}</style></head><body><a href="../">Back to adaptation report</a>'+body+'</body></html>')
files+=['index.html','paired-folds.png','paired-folds.svg'];(O/'manifest.json').write_text(json.dumps({n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in files},indent=2));print(str(O/'index.html'))
