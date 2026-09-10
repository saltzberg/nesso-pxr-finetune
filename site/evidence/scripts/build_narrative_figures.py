#!/usr/bin/env python3
"""Render saved N500 predictions only; no fitting, inference or clipping."""
from pathlib import Path
import csv, json, math, hashlib, html
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/narrative'
INPUTS = ['reports/architecture/results/heldout_predictions.csv','reports/architecture/results/mw-controls/predictions.csv','artifacts/experiments/descriptor_nesso_blend_allfolds_20260908/all_test_predictions.csv','experiments/20260908_PXR-MW-efficacy-diagnostic/artifacts/stratum_summaries.csv']

def read(key):
    return list(csv.DictReader((ROOT/key).open()))

def text(x,y,s,size=13,anchor='middle'):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{html.escape(str(s))}</text>'

def build():
    OUT.mkdir(exist_ok=True,parents=True)
    held=[r for r in read(INPUTS[0]) if r['budget']=='500']
    controls=[r for r in read(INPUTS[1]) if r['budget']=='500']
    mw={r['record_id']:float(r['MW']) for r in controls}
    panels={}
    for r in held+controls:
        panels.setdefault(r['method'],[]).append(r)
    blend=read(INPUTS[2])
    for key in ['descriptor','blend']:
        panels[key]=[dict(r,y_pred=r[key],fold=r['outer_fold']) for r in blend]
    ids=set(mw)
    assert len(ids)==3344
    reference={r['record_id']:r for r in panels['descriptor_lightgbm_rdkit_mordred']}
    for key,rows in panels.items():
        assert len(rows)==len({r['record_id'] for r in rows})==3344,key
        assert {r['record_id'] for r in rows}==ids
        for r in rows:
            ref=reference[r['record_id']]
            assert int(r['fold'])==int(ref['fold'])
            assert abs(float(r['y_true'])-float(ref['y_true']))<1e-12
            assert abs(float(r['weight'])-float(ref['weight']))<1e-10
    for r in panels['descriptor']:
        assert abs(float(r['y_pred'])-float(reference[r['record_id']]['y_pred']))<1e-10
    names={'native_continuous':'Native Nesso','head_only':'Fitted heads','head_plus_lora':'Heads + upstream LoRA','descriptor_lightgbm_rdkit_mordred':'Ligand descriptors','mw':'MW alone','native_mw':'Native score + MW','native_mw_interaction':'Score + MW + interaction','descriptor':'Ligand descriptors','blend':'Descriptors + frozen Nesso'}
    groups={'transfer':['native_continuous','head_only','head_plus_lora','descriptor_lightgbm_rdkit_mordred'],'mw':['native_continuous','mw','native_mw','native_mw_interaction'],'hybrid':['descriptor','blend']}
    groups['mw-overview'] = ['native_continuous', 'head_plus_lora', 'native_mw']
    used=set(sum(groups.values(),[]))
    vals=[float(r[c]) for k in used for r in panels[k] for c in ['y_true','y_pred']]
    lo,hi=math.floor(min(vals)),math.ceil(max(vals))
    audit={'axis_limits':[lo,hi],'n_per_panel':3344,'low_mw_n':sum(v<215 for v in mw.values()),'mw_threshold_da':215,'sources':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in INPUTS},'panels':{}}
    for group,keys in groups.items():
        for columns, suffix in [(2, ""), (1, "-mobile")]:
            width,height=440*columns,440*math.ceil(len(keys)/columns)
            svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height+132}" role="img"><title>Observed versus predicted pEC50: {group}</title><rect width="100%" height="100%" fill="#fffffb"/><g font-family="Arial,sans-serif" fill="#333">']
            for i,key in enumerate(keys):
                ox,oy=(i%columns)*440,(i//columns)*440
                if group == 'mw-overview' and columns == 2:
                    ox, oy = [(220, 0), (0, 440), (440, 440)][i]
                x0,y0,size=ox+66,oy+48,330
                panel_name = ({'native_continuous': 'Nesso-1 Native', 'head_plus_lora': 'Nesso-1 LoRA', 'native_mw': 'Nesso-1 Native + MW'}[key] if group == 'mw-overview' else names[key])
                def px(v): return x0+(v-lo)/(hi-lo)*size
                def py(v): return y0+size-(v-lo)/(hi-lo)*size
                svg += [text(ox+230,oy+24,panel_name,17),f'<path d="M{x0},{y0} V{y0+size} H{x0+size}" stroke="#999" fill="none"/>',f'<path d="M{px(lo)},{py(lo)} L{px(hi)},{py(hi)}" stroke="#999" stroke-dasharray="4 4" fill="none"/>']
                for tick in range(math.ceil(lo/2)*2,hi+1,2):
                    svg += [text(px(tick),y0+size+20,tick,12),text(x0-10,py(tick)+4,tick,12,'end')]
                ylabel = 'Native affinity-equivalent score' if key == 'native_continuous' else 'Predicted pEC50'
                svg += [text(ox+230,oy+428,'Observed assay pEC50',14),f'<text transform="translate({ox+19},{oy+215}) rotate(-90)" text-anchor="middle" font-size="14">{ylabel}</text>']
                rows=sorted(panels[key],key=lambda r:mw[r['record_id']]<215)
                for r in rows:
                    x,y=float(r['y_true']),float(r['y_pred']); small=mw[r['record_id']]<215
                    assert lo<=x<=hi and lo<=y<=hi
                    svg.append(f'<circle cx="{px(x):.4f}" cy="{py(y):.4f}" r="{2 if small else 1.65}" fill="{"#ad501b" if small else "#356c78"}" fill-opacity="{0.65 if small else 0.25}"/>')
                wmae=sum(float(r['weight'])*abs(float(r['y_pred'])-float(r['y_true'])) for r in rows)/sum(float(r['weight']) for r in rows)
                svg.append(text(x0+9,y0+22,f'Weighted MAE: {wmae:.3f}',15,'start'))
                subset_mae = None
                if group in ('mw', 'mw-overview', 'hybrid'):
                    subset = [r for r in rows if (mw[r['record_id']]<215 if group in ('mw', 'mw-overview') else float(r['y_true'])<4)]
                    subset_mae = sum(float(r['weight'])*abs(float(r['y_pred'])-float(r['y_true'])) for r in subset)/sum(float(r['weight']) for r in subset)
                    label = 'MW <215 Da' if group in ('mw', 'mw-overview') else 'Observed pEC50 <4'
                    svg.append(text(x0+9,y0+43,f'{label}: {subset_mae:.3f}',14,'start'))
                audit['panels'][key]={'n':len(rows),'weighted_mae':wmae,'subset_weighted_mae':subset_mae,'raw_mae':sum(abs(float(r['y_pred'])-float(r['y_true'])) for r in rows)/len(rows)}
            legend = [
                'PXR · 3,344 compounds · 5 reused folds · 400 fit/100 cal',
                'Weighted MAE: lower is better; weights = 1/max(pEC50 SE, 0.1)',
                'Orange: MW <215 Da (237); blue: MW ≥215 Da (3,107)',
                'Dashed diagonal: numerical agreement with observed pEC50',
            ]
            if group=='hybrid':
                legend.append('Second score: weighted MAE for observed pEC50 <4 (1,036).')
            elif group in ('mw', 'mw-overview'):
                legend.append('Second score: weighted MAE for the 237 orange compounds.')
            if group in ('transfer','mw','mw-overview'):
                legend.append('Native: affinity-equivalent score, not activation pEC50.')
            for j,line in enumerate(legend):
                svg.append(text(18,height+18+j*19,line,12,'start'))
            svg.append('</g></svg>');(OUT/f'{group}{suffix}.svg').write_text(''.join(svg)+'\n')
    # A deliberately small, directly labelled median comparison, not a causal diagram.
    strata={r['small']:r for r in read(INPUTS[3]) if r['scope']=='full'}
    for columns, suffix in [(3, ""), (1, "-mobile")]:
        svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {293*columns} {300*math.ceil(3/columns)}" role="img"><title>Efficacy and uncertainty medians by molecular weight</title><rect width="100%" height="100%" fill="#fffffb"/><g font-family="Arial,sans-serif" fill="#333">']
        for i,(key,label,limit) in enumerate([('Emax_ratio_median','Normalized Emax',1.5),('Emax_log2FC_median','Raw log2 fold-change Emax',3),('pEC50_standard_error_median','pEC50 standard error',0.7)]):
            ox=(i%columns)*293; oy=(i//columns)*300
            svg.append(f'<g transform="translate(0,{oy})">')
            svg.append(text(ox+146,30,label,15))
            for j,small in enumerate(['True','False']):
                v=float(strata[small][key]); x=ox+45+j*130; y=240-v/limit*160
                svg += [f'<line x1="{x}" y1="240" x2="{x}" y2="{y}" stroke="#ddd"/>',f'<circle cx="{x}" cy="{y}" r="5" fill="{"#ad501b" if small=="True" else "#356c78"}"/>',text(x,y-12,f'{v:.3f}',14),text(x,265,'<215 Da' if small=='True' else '≥215 Da',12)]
            svg += [f'<line x1="{ox+22}" y1="240" x2="{ox+250}" y2="240" stroke="#aaa"/>',text(ox+18,245,'0',11,'end')]
            svg.append('</g>')
        svg.append('</g></svg>');(OUT/f'efficacy{suffix}.svg').write_text(''.join(svg)+'\n')
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print(json.dumps(audit,sort_keys=True))
if __name__=='__main__': build()
