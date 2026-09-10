#!/home/dan/projects/nesso-finetune/review/.venv/bin/python
"""Redraw the reference organization; render and audit using Chromium, no model execution.
Source: PXR/artifacts/architecture/architecture_audit.md (pinned source/AST audit).
Entropy is a diagnostic branch, not an affinity input. ESM conditions the trunk,
not z_init directly. Refinement is optional; both affinity members are retained.
"""
from pathlib import Path
import html
import json
import hashlib
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/architecture'
W, H = 2640, 1630
parts = []
boxes = {}
wires = []

def text(id, x, y, value, size=23, weight=400, color='#52616f', owner=None, anchor='middle'):
    parts.append(f'<text id="text-{id}" data-box="{owner or ""}" x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(value)}</text>')

def box(id, x, y, w, h, fill='#ffffff', stroke='#9daab6', dash=False):
    boxes[id] = (x,y,w,h)
    parts.append(f'<rect id="box-{id}" data-layout-box="true" x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="{fill}" stroke="{stroke}" stroke-width="2"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')

def lines(id, values, start=39, step=30, size=23, weight=400, color='#52616f'):
    x,y,w,h=boxes[id]
    for i,v in enumerate(values):
        text(f'{id}-{start}-{i}', x+w/2, y+start+i*step,v,size,weight,color,id)

def wire(id, points, dash=False, arrow=True):
    wires.append({'id':id,'points':points,'dashed':dash})
    d='M '+' L '.join(f'{x},{y}' for x,y in points)
    parts.append(f'<path id="wire-{id}" d="{d}" fill="none" stroke="#5e6b76" stroke-width="2.2" stroke-linejoin="round"'+(' stroke-dasharray="7 6"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')

def build():
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="figure-title figure-desc"><title id="figure-title">Nesso-1 architecture: coarse-grained cofolding for affinity prediction</title><desc id="figure-desc">Five columns preserve the detailed reference layout. Dashed conditioning routes occupy empty gutters. Entropy is a diagnostic branch and does not feed the affinity model.</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#5e6b76"/></marker></defs><style>text{{font-family:Arial,Helvetica,sans-serif}}</style><rect width="100%" height="100%" fill="#fafbfd"/>')
    text('title',60,60,'Nesso-1 architecture: coarse-grained cofolding for affinity prediction',38,700,'#18232d',anchor='start')
    text('subtitle',60,105,'Pairwise token geometry is the central latent representation; no MSA stack and no all-atom diffusion decoder.',24,anchor='start')
    for x,label in [(240,'1  Inputs'),(690,'2  Token encoding'),(1180,'3  Initial pair state'),(1700,'4  Structural latent'),(2280,'5  Affinity')]:
        text('stage-'+str(x),x,161,label,25,700,'#354450')
    box('protein',60,260,360,180,'#edf4fc','#87a4c6')
    lines('protein',['Protein sequence','residue tokens'],42,34,28,700,'#18232d')
    lines('protein',['amino-acid sequence'],132)
    box('ligand',60,530,360,185,'#f5effb','#ac91c7')
    lines('ligand',['Ligand SMILES / graph','heavy-atom tokens'],43,34,27,700,'#18232d')
    lines('ligand',['atoms + bonds'],136)
    box('esm',510,260,360,180,'#eef7f1','#80ab97')
    lines('esm',['Frozen ESM-2','protein embeddings'],43,34,28,700,'#18232d')
    lines('esm',['pretrained protein LM features','token-aligned · 1,280 channels'],124,29,21)
    box('encoder',510,530,360,235)
    lines('encoder',['Atom / token encoder'],43,30,28,700,'#18232d')
    lines('encoder',['atom features → token features','local atom attention → pooling','+ residue-type encoding','residue / ligand token features','s_inputs · 384 channels'],88,29,21)
    box('initial',980,300,400,275,'#fff8e9','#be9b53')
    lines('initial',['Initial pair representation','z⁽⁰⁾ᵢⱼ'],42,36,28,700,'#18232d')
    lines('initial',['protein–protein · protein–ligand','ligand–protein · ligand–ligand','token projections + relative position','+ bond flags / bond types','pair features · 128 channels'],123,29,22)
    box('matrix',1000,615,360,265)
    lines('matrix',['z pair matrix'],38,30,26,700,'#18232d')
    text('matrix-col-p',1150,698,'protein',20,owner='matrix'); text('matrix-col-l',1270,698,'ligand',20,owner='matrix')
    text('matrix-row-p',1017,752,'protein',18,owner='matrix',anchor='start');text('matrix-row-l',1017,819,'ligand',18,owner='matrix',anchor='start')
    for id,x,y,label,fill in [('pp',1100,715,'PP','#edf2f6'),('pl',1220,715,'PL','#ddebfc'),('lp',1100,782,'LP','#ddebfc'),('ll',1220,782,'LL','#f0e5f9')]:
        box(id,x,y,108,57,fill);lines(id,[label],37,30,24,700,'#18232d')
    box('trunk',1480,260,440,470,'#f2f6fa','#7c909f')
    lines('trunk',['Recycled Nesso-1 trunk'],43,32,29,700,'#18232d')
    lines('trunk',['pair-only Pairformer · 48 blocks'],82,30,24,700)
    box('updates',1515,378,370,190)
    lines('updates',['Pair update blocks'],36,30,25,700,'#18232d')
    lines('updates',['triangle multiplication (out / in)','triangle attention (start / end)','pair transition'],80,32,22,600,'#253440')
    lines('trunk',['updates z only'],352,30,28,700,'#354a5d')
    lines('trunk',['ESM-conditioned · no single track','same stack reused at each pass','no MSA stack'],394,29,23)
    box('recycle',1480,810,440,205,'#fffdf6','#b6aa85',True)
    lines('recycle',['Adaptive recycling context'],40,30,27,700,'#18232d')
    lines('recycle',['Optional inference refinement','Pass 1: full protein','Later passes: protein tokens ≤ 22 Å','from ligand, subject to token budget'],82,29,22)
    box('distogram',2060,260,460,170,'#edf4fc','#87a4c6')
    lines('distogram',['Distogram head','zᵢⱼ + zⱼᵢ → P(dᵢⱼ)'],40,35,28,700,'#18232d')
    lines('distogram',['64 distance-bin logits → probabilities','for token pairs'],120,28,22)
    box('distances',2060,490,460,195,'#eff8f2','#80ab97')
    lines('distances',['Expected distances','E[dᵢⱼ]'],40,35,28,700,'#18232d')
    lines('distances',['coarse-grained geometry','protein Cβ / Gly Cα','ligand heavy atoms'],119,28,23)
    box('entropy',2060,770,460,175,'#fff2f2','#be8989')
    lines('entropy',['Protein–ligand entropy · H_PL'],38,30,26,700,'#18232d')
    lines('entropy',['normalized Shannon entropy of','PL distance distributions','structure-uncertainty diagnostic'],81,29,23)
    box('crop',2060,995,460,60,'#fff8e9','#be9b53')
    lines('crop',['15 Å expected-distance affinity crop'],39,30,23,700,'#354450')
    box('affinity',2060,1090,460,250,'#f5effb','#a087c2')
    lines('affinity',['Affinity Pairformer'],41,30,29,700,'#18232d')
    lines('affinity',['2 independent members × 8 blocks','conditions on:','token / atom features + ESM embeddings','cropped z + distance-bin probabilities','masked pair pooling → output MLPs'],88,30,22)
    box('pbind',2060,1400,215,160,'#eef7f1','#80ab97')
    lines('pbind',['P(bind)'],37,30,27,700,'#18232d')
    lines('pbind',['classification','mean member','probabilities'],76,28,22)
    box('value',2305,1400,215,160,'#edf4fc','#87a4c6')
    lines('value',['Affinity'],37,30,27,700,'#18232d')
    lines('value',['regression','mean member','values'],76,28,22)
    box('structural-note',510,1380,1410,180)
    lines('structural-note',['What Nesso-1 structurally represents'],40,30,29,700,'#18232d')
    lines('structural-note',['A probability distribution over coarse pairwise distances,','not an explicit all-atom structure.','No side-chain coordinate decoder · No heavy-atom diffusion module'],87,32,23)
    # Main flow: all paths terminate at box boundaries.
    wire('protein-esm',[(420,350),(510,350)])
    wire('protein-encoder',[(240,440),(240,480),(690,480),(690,530)])
    wire('ligand-encoder',[(420,620),(510,620)])
    wire('encoder-initial',[(870,570),(980,490)])
    wire('initial-trunk',[(1380,480),(1480,480)])
    wire('trunk-distogram',[(1920,370),(2060,345)])
    wire('distogram-expectation',[(2290,430),(2290,490)])
    wire('distogram-entropy',[(2520,380),(2570,380),(2570,850),(2520,850)])
    wire('expected-crop',[(2060,590),(2020,590),(2020,1025),(2060,1025)])
    wire('crop-affinity',[(2290,1055),(2290,1090)])
    wire('trunk-refinement',[(1920,655),(1960,655),(1960,910),(1920,910)],True)
    wire('refinement-return',[(1480,910),(1430,910),(1430,550),(1480,550)],True)
    wire('esm-trunk',[(870,305),(925,305),(925,215),(1700,215),(1700,260)],True)
    text('esm-trunk-label',1230,205,'ESM conditioning',20)
    wire('pair-affinity',[(1920,580),(1985,580),(1985,1115),(2060,1115)],True)
    wire('esm-affinity',[(1700,215),(2600,215),(2600,1170),(2520,1170)],True)
    wire('token-affinity',[(870,690),(905,690),(905,1240),(2060,1240)],True)
    text('esm-route-label',2270,205,'ESM embeddings → affinity',21)
    text('token-route-label',1420,1227,'Token / atom features',21)
    wire('binding-output',[(2190,1340),(2167,1400)])
    wire('affinity-output',[(2390,1340),(2412,1400)])
    text('legend',60,1510,'Solid: data flow',22,anchor='start')
    text('legend2',60,1545,'Dashed: conditioning / recycling',22,anchor='start')
    parts.append('</svg>')
    return ''.join(parts)

AUDIT_JS = """() => {
 const texts=[...document.querySelectorAll('text')];
 const failures=[];let checked=0;
 for(const t of texts){ const id=t.dataset.box;if(!id)continue;checked++;
  const b=document.getElementById('box-'+id).getBBox(),r=t.getBBox();
  if(r.x<b.x+8||r.y<b.y+8||r.x+r.width>b.x+b.width-8||r.y+r.height>b.y+b.height-8)failures.push({text:t.id,box:id,b:{x:b.x,y:b.y,w:b.width,h:b.height},r:{x:r.x,y:r.y,w:r.width,h:r.height}});
 }
 const overlaps=[];for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
 const a=texts[i].getBBox(),b=texts[j].getBBox();if(a.x<b.x+b.width&&a.x+a.width>b.x&&a.y<b.y+b.height&&a.y+a.height>b.y)overlaps.push([texts[i].id,texts[j].id]);}
 return {boxes:document.querySelectorAll('[data-layout-box]').length,texts:texts.length,box_texts_checked:checked,containment_failures:failures,text_overlap_failures:overlaps};
}"""

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    svg=OUT/'nesso1-architecture-redrawn.svg';png=svg.with_suffix('.png')
    ref=ROOT.parent/'assets/nesso1_architecture_precise.png'
    before=hashlib.sha256(ref.read_bytes()).hexdigest()
    svg.write_text(build())
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':W,'height':H},device_scale_factor=1)
        page.set_content('<html><body style="margin:0">'+svg.read_text()+'</body></html>');page.evaluate('document.fonts.ready')
        audit=page.evaluate(AUDIT_JS)
        # Sample every wire segment against rectangle interiors (nested blocks included).
        collisions=[]
        for wire_data in wires:
            for (x1,y1),(x2,y2) in zip(wire_data['points'],wire_data['points'][1:]):
                n=max(int(abs(x2-x1)+abs(y2-y1)),1)
                for id,(x,y,w,h) in boxes.items():
                    if any(x+2<x1+(x2-x1)*i/n<x+w-2 and y+2<y1+(y2-y1)*i/n<y+h-2 for i in range(n+1)):
                        collisions.append([wire_data['id'],id])
        crossings=[]
        def cross(a,b): return a[0]*b[1]-a[1]*b[0]
        for i,wa in enumerate(wires):
            for wb in wires[i+1:]:
                for a,b in zip(wa['points'],wa['points'][1:]):
                    for c,d in zip(wb['points'],wb['points'][1:]):
                        r=(b[0]-a[0],b[1]-a[1]);s=(d[0]-c[0],d[1]-c[1]);den=cross(r,s)
                        if not den: continue
                        ca=(c[0]-a[0],c[1]-a[1]);t=cross(ca,s)/den;u=cross(ca,r)/den
                        if 0<t<1 and 0<u<1: crossings.append([wa['id'],wb['id']])
        audit.update(wires_checked=len(wires),wire_box_collisions=collisions,wire_wire_crossings=crossings,canvas=[W,H],reference_sha256=before)
        assert not crossings,audit
        assert not audit['containment_failures'],audit
        assert not audit['text_overlap_failures'],audit
        assert not collisions,audit
        page.screenshot(path=str(png),full_page=True)
        browser.close()
    assert hashlib.sha256(ref.read_bytes()).hexdigest()==before
    audit['reference_unchanged']=True
    (OUT/'nesso1-architecture-redrawn-verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    print(svg);print(png)

if __name__=='__main__':
    main()
