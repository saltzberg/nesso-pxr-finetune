#!/usr/bin/env python3
"""Reporting-only subtle architecture highlights and completed-fit scope audit. Never imports Nesso/torch.
Run with review/.venv/bin/python; Chromium exports PNG from the exact SVG.
"""
from pathlib import Path
import csv
import hashlib
import html
import json
import shutil
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/architecture'
STEM = 'nesso1-finetuning'
PROTOCOL = 'experiments/20260906_affinity_comparison/train_protocol.json'
AUDIT = 'artifacts/architecture/architecture_audit.md'
MEMBERS = ['affinity_module', 'affinity_module2']
TARGETS = ['esm_proj.1', 'esm_proj.3', 'pairformer_stack.layers.0.transition_z.fc1']

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def inventory():
    rows = []
    for m in MEMBERS:
        for i, din, dout in [(0,384,384),(2,384,384),(4,384,1)]:
            rows.append(dict(module=f'{m}.affinity_heads.to_affinity_pred_value.{i}',kind='head',input=din,output=dout,weight_shape=[dout,din],bias_shape=[dout],rank=0,alpha=0,trainable=dout*din+dout))
        for t, din, dout in zip(TARGETS,[1280,384,128],[384,384,512]):
            rows.append(dict(module=f'{m}.{t}',kind='lora',input=din,output=dout,weight_shape=[dout,din],A_shape=[4,din],B_shape=[dout,4],rank=4,alpha=4,trainable=4*(din+dout)))
    return rows

def expected_names(arm):
    return {r['module']+'.'+suffix for r in inventory() if r['kind']=='head' or arm=='head_plus_lora' for suffix in (['weight','bias'] if r['kind']=='head' else ['lora_A','lora_B'])}

def collect_evidence():
    protocol=json.loads((ROOT/PROTOCOL).read_text())
    assert protocol['members']==MEMBERS
    assert protocol['training']['lora_targets']==TARGETS
    assert protocol['training']['rank']==protocol['training']['alpha']==4
    source_keys={PROTOCOL,AUDIT,'experiments/20260906_affinity_comparison/RESULTS.md','reports/architecture/results/sources.csv','reports/architecture/results/analysis_protocol.json','reports/architecture/results/pooled_metrics.csv'}
    rows=list(csv.DictReader((ROOT/'reports/architecture/results/sources.csv').open()))
    fits=[]
    for row in rows:
        if row['method'] not in ['head_only','head_plus_lora']:
            continue
        folder=Path(row['path'])
        complete=json.loads((folder/'complete.json').read_text())
        audit=json.loads((folder/'audit.json').read_text())
        replay=json.loads((folder/'replay.json').read_text())
        assert complete['task']['task_id']==row['task_id'] and not complete['task']['smoke']
        assert complete['task']['arm']==row['method']
        for name in ['audit.json','replay.json']:
            assert digest(folder/name)==complete['files'][name]
        assert set(audit['intended_before'])==set(audit['intended_after'])==expected_names(row['method'])
        assert all(audit['intended_before'][k]!=audit['intended_after'][k] for k in audit['intended_before'])
        assert audit['frozen_before']==audit['frozen_after']
        assert len(audit['history'])==40 and audit['history'][-1]['epoch']==40
        assert replay['scoring_replayed'] and replay['point_max_abs']==replay['native_parity_max_abs']==0
        files=[str((folder/name).relative_to(ROOT)) for name in ['complete.json','audit.json','replay.json']]
        source_keys.update(files)
        fits.append(dict(task_id=row['task_id'],arm=row['method'],budget=int(row['budget']),fold=int(row['fold']),epochs=40,changed_tensors=len(audit['intended_after']),frozen_tensors=len(audit['frozen_after']),frozen_unchanged=True,replay_max_abs=replay['point_max_abs'],files=files))
    assert len(fits)==20
    assert {(r['fold'],r['budget'],r['arm']) for r in fits}=={(f,n,a) for f in range(5) for n in [100,500] for a in ['head_only','head_plus_lora']}
    counts={k:sum(r['trainable'] for r in inventory() if r['kind']==k) for k in ['head','lora']}
    assert counts=={'head':592130,'lora':24576}
    return dict(scope='Completed matched 20-fit comparison; saved JSON/hash audit only, no new checkpoint replay or inference',members=MEMBERS,counts={**counts,'head_plus_lora':sum(counts.values())},layers=inventory(),fits=fits,sources=[dict(source_key=k,sha256=digest(ROOT/k),bytes=(ROOT/k).stat().st_size) for k in sorted(source_keys)])

SELECTED = 'nesso1-architecture-selected'
APPROVED = 'nesso1-architecture-redrawn'
HIGHLIGHTS = {
    'box-affinity': {'fill': '#eef4f7', 'stroke': '#7293a5'},
    'box-value': {'fill': '#f8f1e9', 'stroke': '#b39a7c'},
}

def svg():
    """Change only two existing rectangle fills/borders; preserve all geometry/text."""
    import re
    source = (OUT/f'{APPROVED}.svg').read_text()
    for node, styles in HIGHLIGHTS.items():
        pattern = rf'<rect id="{node}"[^>]*>'
        matches = re.findall(pattern, source)
        assert len(matches) == 1, node
        selected = matches[0]
        for key, value in styles.items():
            selected = re.sub(rf'{key}="[^"]*"', f'{key}="{value}"', selected)
        source = source.replace(matches[0], selected, 1)
    return source

def source_keys():
    evidence=json.loads((OUT/f'{STEM}-evidence.json').read_text())
    return {s['source_key'] for s in evidence['sources']} | {f'reports/architecture/{STEM}{suffix}' for suffix in ['-evidence.json','-layers.csv']} | {f'reports/architecture/{SELECTED}{suffix}' for suffix in ['.svg','.png','-layout.json']} | {'scripts/build_architecture_finetuning.py','scripts/build_experimental_site.py','scripts/build_architecture_redraw.py','site/experiment-content/architecture-finetuning.md'}

def publish():
    site=ROOT/'site'
    mapped={s['source_key']:s for s in json.loads((site/'source-map.json').read_text())}
    for key in sorted(source_keys()):
        source=ROOT/key; target=site/'evidence'/key
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        assert digest(source)==digest(target)
        mapped[key]=dict(source=str(source),source_key=key,site_path=str(target.relative_to(site)),bytes=source.stat().st_size,sha256=digest(source))
    (site/'source-map.json').write_text(json.dumps(list(sorted(mapped.values(),key=lambda s:s['source_key'])),indent=2,sort_keys=True)+'\n')

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=collect_evidence()
    (OUT/f'{STEM}-evidence.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    with (OUT/f'{STEM}-layers.csv').open('w') as f:
        keys=['module','kind','input','output','weight_shape','bias_shape','A_shape','B_shape','rank','alpha','trainable']
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(inventory())
    path=OUT/f'{SELECTED}.svg'; path.write_text(svg())
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':2640,'height':1630},device_scale_factor=1)
        geometry = """() => [...document.querySelectorAll('svg *')].filter(e=>typeof e.getBBox==='function').map(e=>({tag:e.tagName,id:e.id,text:e.textContent,...Object.fromEntries(['x','y','width','height'].map(k=>[k,e.getBBox()[k]]))}))"""
        page.goto((OUT/f'{APPROVED}.svg').as_uri())
        before = page.evaluate(geometry)
        page.goto(path.as_uri())
        after = page.evaluate(geometry)
        assert before == after, 'Approved SVG layout or labels changed'
        page.screenshot(path=str(OUT/f'{SELECTED}.png'))
        browser.close()
    layout={'approved_sha256':digest(OUT/f'{APPROVED}.svg'), 'selected_sha256':digest(path), 'geometry_unchanged':True, 'highlighted_nodes':HIGHLIGHTS, 'elements':after}
    (OUT/f'{SELECTED}-layout.json').write_text(json.dumps(layout,indent=2)+'\n')
    print(json.dumps({'fits':len(evidence['fits']),'counts':evidence['counts'],'geometry_unchanged':layout['geometry_unchanged']}))

if __name__=='__main__':
    build()
