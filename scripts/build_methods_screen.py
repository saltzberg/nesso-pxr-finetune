#!/usr/bin/env python3
"""Reporting-only native figure and fine-tuning synthesis publication."""
from pathlib import Path
import csv, hashlib, json, shutil, math
ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/'site'
OUT = ROOT/'reports/methods-screen'
PRED = 'reports/architecture/results/heldout_predictions.csv'
METRICS = 'reports/architecture/results/pooled_metrics.csv'
EVIDENCE = [PRED, METRICS, 'reports/narrative/verification.json', 'scripts/build_narrative_figures.py', 'scripts/build_methods_screen.py', 'scripts/build_experimental_site.py', 'experiments/20260906_affinity_comparison/train_protocol.json', 'experiments/20260906_affinity_comparison/RESULTS.md', 'experiments/20260906_affinity_comparison/EXECUTION.md', 'reports/architecture/results/RESULTS.md', 'reports/architecture/results/sources.csv', 'artifacts/cloud/50146714/continuation.json']
PAGE = 'experiments/fine-tuning-screen/index.html'

def build_assets():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in csv.DictReader((ROOT/PRED).open()) if r['budget']=='500' and r['method']=='native_continuous']
    assert len(rows)==len({r['record_id'] for r in rows})==3344
    x = [float(r['y_true']) for r in rows]; y = [float(r['y_pred']) for r in rows]; w = [float(r['weight']) for r in rows]
    score = math.fsum(wi*abs(yi-xi) for wi,xi,yi in zip(w,x,y))/math.fsum(w)
    raw = math.fsum(abs(yi-xi) for xi,yi in zip(x,y))/len(x)
    limits = json.loads((ROOT/'reports/narrative/verification.json').read_text())['axis_limits']
    assert min(x+y)>=limits[0] and max(x+y)<=limits[1]
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'native-points.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.size':11,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(5.4,5.6),layout='constrained')
    fig.patch.set_facecolor('#fffffb'); ax.set_facecolor('#fffffb')
    ax.plot(limits,limits,color='#999999',linewidth=.8,linestyle='--',zorder=0)
    ax.scatter(x,y,s=5,c='#356c78',alpha=.28,edgecolors='none')
    ax.set(xlim=limits,ylim=limits,aspect='equal',xlabel='Observed assay pEC50',ylabel='Native affinity-equivalent score')
    ax.set_title(f'Nesso-1 native\nWeighted MAE: {score:.4f}',loc='left',pad=12)
    ax.spines[['top','right']].set_visible(False)
    for ext in ['svg','png']:
        fig.savefig(OUT/f'native.{ext}',dpi=180,metadata={'Creator':'Saved Nesso predictions; no fitting'} if ext=='svg' else None)
    plt.close(fig)
    audit={'n':len(rows),'method':'native_continuous','budget':500,'weighted_mae':score,'raw_mae':raw,'weight_sum':math.fsum(w),'weight_definition':'1/max(reported pEC50 SE, 0.1)','axis_limits':limits,'observed_range':[min(x),max(x)],'prediction_range':[min(y),max(y)],'sources':{k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in [PRED,METRICS,'reports/narrative/verification.json']}}
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit

def source_keys():
    return set(EVIDENCE) | {str(p.relative_to(ROOT)) for p in OUT.glob('*')}

def render_screen():
    from build_experimental_site import MD, shell
    body=MD.render((SITE/'experiment-content/fine-tuning-screen.md').read_text())
    target=SITE/PAGE; target.parent.mkdir(parents=True,exist_ok=True)
    output=shell('Fine-tuning screen',body,'../../')
    output=output.replace('</head>', '<style>main{overflow-wrap:anywhere}main code{white-space:normal}main table{max-width:100%;overflow-wrap:normal}main td:nth-child(n+3){white-space:nowrap}main th{min-width:70px}</style></head>')
    target.write_text(output)

def publish():
    import urllib.request
    edits=json.load(urllib.request.urlopen('http://100.102.92.33:8895/__review/text-edits?path=/index.html'))
    assert not edits['edits'], 'Pending review edits: do not publish'
    from build_experimental_site import render_overview
    audit=build_assets()
    sources=json.loads((SITE/'source-map.json').read_text())
    mapped={r['source_key']:r for r in sources}
    for key in sorted(source_keys()):
        p=ROOT/key; dest=SITE/'evidence'/key; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
        mapped[key]={'source':str(p),'source_key':key,'site_path':str(dest.relative_to(SITE)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    render_screen(); render_overview()
    manifest=json.loads((SITE/'experiment-manifest.json').read_text())
    if PAGE not in manifest['generated_pages']: manifest['generated_pages'].append(PAGE)
    (SITE/'experiment-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (SITE/'source-map.json').write_text(json.dumps(sorted(mapped.values(),key=lambda r:r['source_key']),indent=2,sort_keys=True)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': publish()
