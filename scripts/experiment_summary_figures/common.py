"""Shared export contract for reporting-only experiment summary figures."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'site/assets/experiment-summaries'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,
    'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white',
    'axes.labelcolor':'#333333','text.color':'#222222','axes.edgecolor':'#777777',
    'savefig.dpi':170})
INK='#356c78'
ACCENT='#ad501b'
GRAY='#929292'

def publish(fig, slug, data, sources, message, caption, design):
    """Export exact plotted rows and source hashes; no scientific fitting."""
    if not isinstance(data,pd.DataFrame): data=pd.DataFrame(data)
    assert len(data)>0, slug
    data.to_csv(OUT/f'{slug}.csv', index=False)
    for suffix in ['svg','png']:
        fig.savefig(OUT/f'{slug}.{suffix}',bbox_inches='tight')
    plt.close(fig)
    provenance=[]
    for source in sources:
        p=Path(source)
        if not p.is_absolute(): p=ROOT/p
        assert p.is_file(), p
        provenance.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    record={'slug':slug,'message':message,'caption':caption,'design':design,
            'sources':provenance,'plotted_rows':len(data),
            'svg':f'assets/experiment-summaries/{slug}.svg',
            'png':f'assets/experiment-summaries/{slug}.png',
            'data':f'assets/experiment-summaries/{slug}.csv'}
    (OUT/f'{slug}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(slug, len(data), message)
    return record
