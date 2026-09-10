"""Focused reporting-only tests: completed scope, bytes, layout and integration."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_architecture_finetuning as zoom

def test_completed_scope_and_inventory():
    evidence=zoom.collect_evidence()
    saved=json.loads((zoom.OUT/f'{zoom.STEM}-evidence.json').read_text())
    assert evidence==saved
    assert len(evidence['fits'])==20
    assert len(evidence['layers'])==12
    assert evidence['counts']=={'head':592130,'lora':24576,'head_plus_lora':616706}
    assert all(r['changed_tensors']==(12 if r['arm']=='head_only' else 24) for r in evidence['fits'])
    assert len({r['module'] for r in evidence['layers']})==12

def test_selected_layout_and_styles_only():
    path=zoom.OUT/f'{zoom.SELECTED}.svg'
    assert path.read_text()==zoom.svg()
    original=ET.parse(zoom.OUT/f'{zoom.APPROVED}.svg').getroot()
    selected=ET.parse(path).getroot()
    assert len(list(original.iter()))==len(list(selected.iter()))
    changes=[]
    for a,b in zip(original.iter(),selected.iter()):
        assert a.tag==b.tag and a.text==b.text and a.tail==b.tail
        if a.attrib!=b.attrib:
            changes.append(a.get('id'))
            assert {k:v for k,v in a.attrib.items() if k not in ['fill','stroke']}=={k:v for k,v in b.attrib.items() if k not in ['fill','stroke']}
            assert {k:b.get(k) for k in ['fill','stroke']}==zoom.HIGHLIGHTS[a.get('id')]
    assert set(changes)=={'box-affinity','box-value'}
    assert len(selected.findall('.//*[@data-layout-box]'))==21
    layout=json.loads((zoom.OUT/f'{zoom.SELECTED}-layout.json').read_text())
    assert layout['geometry_unchanged']
    assert layout['approved_sha256']==zoom.digest(zoom.OUT/f'{zoom.APPROVED}.svg')
    assert layout['selected_sha256']==zoom.digest(path)

def test_page_section_order_and_canonical_content():
    page=(ROOT/'site/architecture/index.html').read_text()
    assert page.index('nesso1-architecture-selected.png') < page.index('id="fine-tuning"')
    for legacy in ['nesso1-finetuning.svg','nesso1-finetuning.png','nesso-native-pipeline.svg','nesso-adaptation-comparison.svg','nesso-repeated-operators.svg']:
        assert legacy not in page
    assert page.count('<img ')==1
    assert page.count('id="fine-tuning"')==1
    import build_experimental_site as builder
    assert builder.MD.render((builder.CONTENT/'architecture-finetuning.md').read_text()) in page
    for word in ['592,130','24,576','616,706','external ESM model','binary-dual','affinity_module2']:
        assert word in page
    assert 'finetuning_source_keys()' in (ROOT/'scripts/build_experimental_site.py').read_text()

def test_evidence_publication():
    source_map={r['source_key']:r for r in json.loads((ROOT/'site/source-map.json').read_text())}
    for key in zoom.source_keys():
        row=source_map[key]
        original=(ROOT/key).read_bytes()
        assert (ROOT/'site'/row['site_path']).read_bytes()==original
        assert hashlib.sha256(original).hexdigest()==row['sha256']
        assert len(original)==row['bytes']

def test_approved_primary_unchanged():
    before=ROOT.parent/'review/architecture-finetuning/before-hashes.json'
    if before.exists():
        for key,value in json.loads(before.read_text()).items():
            if 'nesso1-architecture-redrawn.' in key:
                assert zoom.digest(ROOT/key)==value
