"""Proximity figure from saved diagnostic rows; no inference or fitting."""
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mw_markers import scatter_mw

def build(H,save):
 d=pd.read_csv(H/'proximity_all.csv',float_precision='round_trip')
 top=pd.read_csv(H/'proximity_top20.csv',float_precision='round_trip')
 assert len(d)==6688 and not d.duplicated(['budget','record_id']).any()
 verification=json.loads((H/'proximity_verification.json').read_text());assert verification['status']=='PASS'
 fig,axes=plt.subplots(2,2,figsize=(9.6,8.4),sharex=True,sharey=True,layout='constrained')
 ymax=max(float((d[m]-d.native).abs().max()) for m in ['head_only','head_plus_lora'])*1.06
 plotted=[]
 for row,n in enumerate([100,500]):
  for col,(m,label) in enumerate([('head_only','Head-only'),('head_plus_lora','Head + LoRA')]):
   ax=axes[row,col];v=d[d.budget==n].copy();assert len(v)==3344
   v['abs_change']=(v[m]-v.native).abs();v=v.sort_values(['abs_change','record_id'],ascending=[False,True]);t=v.head(20)
   saved=top[(top.budget==n)&(top.comparison==m+'-minus-native')]
   assert t.record_id.tolist()==saved.record_id.tolist()
   assert np.allclose(t.abs_change,saved.abs_change,rtol=0,atol=1e-14)
   assert (v.fit_similarity>=0).all() and (v.fit_similarity<.35).all() and (v.abs_change<ymax).all()
   scatter_mw(ax,v.fit_similarity,v.abs_change,v.record_id,s=7,c='#75808a',alpha=.24,linewidths=0)
   ax.scatter(t.fit_similarity,t.abs_change,s=27,facecolors='none',edgecolors='#b85c00',linewidths=1,zorder=3)
   ax.set_title(f'{label} · N{n}',loc='left',fontsize=11)
   ax.set_xlim(-.006,.36);ax.set_ylim(-.07,ymax);ax.set_xticks([0,.1,.2,.3,.35]);ax.set_box_aspect(1)
   if row==1:ax.set_xlabel('Nearest fitting-compound similarity')
   if col==0:ax.set_ylabel('Absolute change from native (log10 units)')
   v['method']=m;v['top20']=v.record_id.isin(t.record_id);plotted.append(v[['budget','method','record_id','fit_similarity','abs_change','top20']])
 save(fig,'change-versus-training-similarity')
 p=pd.concat(plotted);assert len(p)==13376 and int(p.top20.sum())==80;p.to_csv(H/'proximity_figure_points.csv',index=False)
 (H/'proximity_figure_verification.json').write_text(json.dumps({'status':'PASS','panels':4,'rows_per_panel':3344,'highlighted_per_panel':20,'total_marks_before_highlight_overlay':len(p),'xlim':[-.006,.36],'ylim':[-.07,ymax],'no_clipping':True,'input_sha256':{name:hashlib.sha256((H/name).read_bytes()).hexdigest() for name in ['proximity_all.csv','proximity_top20.csv']},'code_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()},indent=2))
