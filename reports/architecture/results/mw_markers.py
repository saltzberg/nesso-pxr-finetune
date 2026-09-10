"""Shared identity-based <215 Da marker encoding for compound plots."""
from pathlib import Path
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent
D=pd.read_csv(H/'mover_chemistry_all.csv',float_precision='round_trip').set_index('record_id')
assert D.index.is_unique and len(D)==3344
COLOR='#397db8'
AUDIT=[]
def scatter_mw(ax,x,y,ids,**kwargs):
 ids=list(ids);mw=D.loc[ids,'MW'].to_numpy();small=mw<215;x=np.asarray(x);y=np.asarray(y)
 assert len(x)==len(y)==len(ids) and np.isfinite(mw).all()
 ax.scatter(x[~small],y[~small],**kwargs)
 mark=ax.scatter(x[small],y[small],s=17,marker='o',facecolors='none',edgecolors=COLOR,linewidths=.9,alpha=1,zorder=4)
 assert len(mark.get_facecolors())==0 and len(mark.get_offsets())==int(small.sum())
 ax._mw_counts={'n':len(ids),'below_215':int(small.sum()),'at_least_215':int((~small).sum())}
 return mark

def audit_figure(fig,name):
 for i,ax in enumerate(fig.axes):
  if hasattr(ax,'_mw_counts'):AUDIT.append(dict(figure=name,panel=i,**ax._mw_counts))
