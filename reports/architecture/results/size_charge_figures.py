"""Figures use the user's signed difference of nonnegative absolute errors."""
import numpy as np,pandas as pd,json
import matplotlib.pyplot as plt
from mw_markers import scatter_mw

def build(H,save,movement=False):
 d=pd.read_csv(H/'size_charge_points.csv',float_precision='round_trip');assert len(d)==13376
 assert np.allclose(d.difference_distance_from_truth,np.abs(d.fitted-d.y_true)-np.abs(d.native-d.y_true),rtol=0,atol=1e-14)
 ycol='absolute_movement' if movement else 'MW'
 ylabel='Absolute change from native' if movement else 'Molecular weight (Da)'
 original_save=save
 if movement:
  save=lambda fig,name: original_save(fig,'error-change-versus-movement')
 x=d.difference_distance_from_truth;pad=.04*(x.max()-x.min());xl=(float(x.min()-pad),float(x.max()+pad));yl=(-.07,float(d[ycol].max()*1.04))
 def panel(ax,v,title):
  ax.axvline(0,color='#919ba3',lw=.7,zorder=1)
  scatter_mw(ax,v.difference_distance_from_truth,v[ycol],v.record_id,s=5,alpha=.30,c='#536b7c',linewidths=0,zorder=2)
  ax.set(xlim=xl,ylim=yl);ax.set_box_aspect(1);ax.set_title(title,loc='left',fontsize=9)
  assert v.difference_distance_from_truth.between(*xl).all() and v[ycol].between(*yl).all()
 fig,axes=plt.subplots(2,2,figsize=(9,8),sharex=True,sharey=True,layout='constrained')
 names={'head_only':'Head-only','head_plus_lora':'Head + LoRA'}
 for row,n in enumerate([100,500]):
  for col,m in enumerate(names):
   v=d[(d.budget==n)&(d.method==m)];assert len(v)==3344;panel(axes[row,col],v,f'{names[m]} · N{n}')
   if row==1:axes[row,col].set_xlabel('difference distance from truth')
   if col==0:axes[row,col].set_ylabel(ylabel)
 save(fig,'error-change-versus-size')
 if movement:return
 counts=[]
 for n in [100,500]:
  fig,axes=plt.subplots(2,4,figsize=(13.6,7),sharex=True,sharey=True,layout='constrained')
  for row,m in enumerate(names):
   for col,(charged,low) in enumerate([(False,True),(True,True),(False,False),(True,False)]):
    v=d[(d.budget==n)&(d.method==m)&(d.charged==charged)&(d.low_potency==low)]
    title=('Charged' if charged else 'Net-neutral')+' · '+('pEC50 <4' if low else 'pEC50 ≥4')+f' · n={len(v):,}'
    panel(axes[row,col],v,title)
    if col==0:axes[row,col].set_ylabel(names[m]+'\n'+ylabel)
    if row==1:axes[row,col].set_xlabel('difference distance from truth',fontsize=9)
    counts.append(dict(budget=n,method=m,charged=charged,low_potency=low,n=len(v)))
  save(fig,f'error-change-stratified-n{n}')
 assert sum(c['n'] for c in counts)==len(d)
 (H/'size_charge_figure_verification.json').write_text(json.dumps(dict(status='PASS',formula='abs(fitted-truth)-abs(native-truth)',x_axis='difference distance from truth',positive_means='worse',xlim=xl,ylim=yl,complete_rows=len(d),stratified_panels=counts),indent=2))
