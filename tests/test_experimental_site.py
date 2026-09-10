"""Reporting-only completeness, portable link, byte and numerical audits."""
import csv
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.links.extend(attrs[k] for k in ('href', 'src', 'srcset') if k in attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])

def test_inventory_and_all_local_links():
    m = json.loads((SITE/'experiment-manifest.json').read_text())
    assert len(m['experiments']) == m['expected_count'] == 21
    assert len({e['slug'] for e in m['experiments']}) == 21
    assert set(m['directory_coverage']) == {p.name for p in (ROOT/'experiments').iterdir() if p.is_dir()}
    assert all(m['directory_coverage'].values())
    for e in m['experiments']:
        assert e['status'] == 'completed'
        text = (SITE/e['page']).read_text()
        for heading in ['Question','Design and fitted/frozen flow','Results','Implication and limits','Source evidence']:
            assert heading in text
    for relative in m['generated_pages']:
        page = SITE/relative
        parser = Links()
        parser.feed(page.read_text())
        for link in parser.links:
            u = urlsplit(link)
            if u.scheme or u.netloc:
                assert link in {
                    'https://huggingface.co/blog/recursionpharma/nesso1',
                    'https://huggingface.co/spaces/openadmet/pxr-challenge',
                    'https://github.com/saltzberg/nesso-pxr-finetune',
                }, (page, link)
                continue
            assert not u.path.startswith('/'), (page, link)
            target = (page.parent/unquote(u.path)).resolve() if u.path else page
            assert target.is_relative_to(SITE.resolve()), (page, link)
            if target.is_dir():
                target = target / 'index.html'
            assert target.is_file(), (page, link)
            if u.fragment:
                destination = Links()
                destination.feed(target.read_text())
                assert u.fragment in destination.ids, (page, link)

def test_all_curated_bytes_and_hashes():
    sources = json.loads((SITE/'source-map.json').read_text())
    assert sources
    for s in sources:
        copied = SITE / s['site_path']
        assert len(copied.read_bytes()) == s['bytes']
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == s['sha256']

def test_headline_values_from_saved_tables():
    overview = (SITE/'index.html').read_text()
    transfer = (SITE/'chapters/transfer/index.html').read_text()
    hybrid = (SITE/'chapters/hybrid/index.html').read_text()
    interpretation = (SITE/'chapters/interpretation/index.html').read_text()
    assert 'narrative/efficacy.svg' not in interpretation + overview
    efficacy = interpretation
    assert '<table>' in interpretation and '<table>' in overview
    rows = list(csv.DictReader((ROOT/'reports/architecture/results/pooled_metrics.csv').open()))
    expected = {'head_only','head_plus_lora','repr_ridge_stable','descriptor_lightgbm_rdkit_mordred'}
    selected = [r for r in rows if r['budget'] == '500' and r['method'] in expected]
    assert len(selected) == 4 and {r['method'] for r in selected} == expected
    for r in selected:
        assert f"{float(r['weighted_mae']):.3f}" in transfer
    m = json.loads((ROOT/'artifacts/experiments/descriptor_nesso_blend_allfolds_20260908/metrics.json').read_text())
    for method in ['descriptor','blend']:
        for key in ['weighted_mae','raw_mae','spearman']:
            assert f"{m['pooled'][method][key]:.3f}" in hybrid
        for group in ['ge4','lt4']:
            assert f"{m['pooled'][method][group]['weighted_mae']:.3f}" in hybrid
    assert m['improving_folds'] == 5 and not m['fresh_holdout']
    rows = list(csv.DictReader((ROOT/'experiments/20260908_PXR-MW-efficacy-diagnostic/artifacts/stratum_summaries.csv').open()))
    for r in rows:
        if r['scope'] == 'full':
            assert f"{float(r['Emax_ratio_median']):.3f}" in efficacy
            assert f"{float(r['pEC50_standard_error_median']):.3f}" in efficacy
            if r['small'] == 'True':
                assert r['n'] == r['qualified'] == '237'
    mw = list(csv.DictReader((ROOT/'reports/architecture/results/mw-controls/metrics.csv').open()))
    values = {f"{float(r['weighted_mae']):.3f}" for r in mw if r['budget']=='500' and r['subset']=='lt215'}
    assert {'1.834','1.179','1.163','0.863'} <= values
    for value in ['1.834','1.179','1.163','0.863']:
        assert value in (SITE/'chapters/mw/index.html').read_text()


def test_narrative_structure_and_saved_prediction_figures():
    import math
    import xml.etree.ElementTree as ET
    manifest = json.loads((SITE/'experiment-manifest.json').read_text())
    assert len(manifest['chapters']) == 4
    overview = (SITE/'index.html').read_text()
    assert '<details id="experiments">' in overview
    assert overview.count('class="narrative"') == 3
    for c in manifest['chapters']:
        body = (SITE/c['page']).read_text()
        assert len(body.split()) > 300
        assert ('<table>' in body if c['slug']=='interpretation' else 'class="narrative"' in body)
    audit = json.loads((ROOT/'reports/narrative/verification.json').read_text())
    assert audit['n_per_panel'] == 3344 and audit['low_mw_n'] == 237
    rows = list(csv.DictReader((ROOT/'reports/architecture/results/heldout_predictions.csv').open()))
    rows += list(csv.DictReader((ROOT/'reports/architecture/results/mw-controls/predictions.csv').open()))
    for method, metrics in audit['panels'].items():
        if method in ['descriptor','blend']:
            selected = list(csv.DictReader((ROOT/'artifacts/experiments/descriptor_nesso_blend_allfolds_20260908/all_test_predictions.csv').open()))
            selected = [dict(r,y_pred=r[method]) for r in selected]
        else:
            selected = [r for r in rows if r['method']==method and r['budget']=='500']
        assert len(selected)==3344
        score = sum(float(r['weight'])*abs(float(r['y_pred'])-float(r['y_true'])) for r in selected)/sum(float(r['weight']) for r in selected)
        assert math.isclose(score,metrics['weighted_mae'],abs_tol=1e-12)
        for r in selected:
            assert audit['axis_limits'][0] <= float(r['y_pred']) <= audit['axis_limits'][1]
    for name,n in [('transfer',4),('mw',4),('hybrid',2)]:
        svg = ET.parse(ROOT/f'reports/narrative/{name}.svg')
        circles = svg.findall('.//{http://www.w3.org/2000/svg}circle')
        assert len(circles) == n*3344
        assert sum(c.attrib['fill']=='#ad501b' for c in circles) == n*237
    for key,expected in audit['sources'].items():
        assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest()==expected


def test_mobile_panels_preserve_points_and_native_axis_semantics():
    import math
    import xml.etree.ElementTree as ET
    ns = '{http://www.w3.org/2000/svg}'
    for name, count in [('transfer', 4), ('mw', 4), ('hybrid', 2)]:
        desktop = ET.parse(ROOT/f'reports/narrative/{name}.svg')
        mobile = ET.parse(ROOT/f'reports/narrative/{name}-mobile.svg')
        a = desktop.findall(f'.//{ns}circle')
        b = mobile.findall(f'.//{ns}circle')
        assert len(a) == len(b) == count*3344
        for j, (d, m) in enumerate(zip(a, b, strict=True)):
            panel = j // 3344
            assert d.attrib['fill'] == m.attrib['fill']
            assert math.isclose(float(d.attrib['cx'])-(panel%2)*440, float(m.attrib['cx']), abs_tol=0.00011)
            assert math.isclose(float(d.attrib['cy'])+(panel-panel//2)*440, float(m.attrib['cy']), abs_tol=0.00011)
        if name in ('transfer', 'mw'):
            for tree in (desktop, mobile):
                labels = [e.text or '' for e in tree.findall(f'.//{ns}text')]
                assert labels.count('Native affinity-equivalent score') == 1
                assert labels.count('Predicted pEC50') == count-1
    overview = (SITE/'index.html').read_text()
    assert overview.count('media="(max-width: 600px)"') == 3
    assert overview.count('class="plot-key"') == 0


def test_figures_carry_verified_scores_and_legends():
    import xml.etree.ElementTree as ET
    audit = json.loads((ROOT/'reports/narrative/verification.json').read_text())
    groups = {'transfer':['native_continuous','head_only','head_plus_lora','descriptor_lightgbm_rdkit_mordred'], 'mw':['native_continuous','mw','native_mw','native_mw_interaction'], 'hybrid':['descriptor','blend']}
    for name, methods in groups.items():
        for suffix in ('','-mobile'):
            tree = ET.parse(ROOT/f'reports/narrative/{name}{suffix}.svg')
            labels = [e.text or '' for e in tree.findall('.//{http://www.w3.org/2000/svg}text')]
            assert any('lower is better' in s for s in labels)
            assert any('Dashed diagonal' in s for s in labels)
            assert any('Orange:' in s and 'blue:' in s for s in labels)
            scores = [s for s in labels if s.startswith('Weighted MAE:') and 'lower' not in s]
            assert scores == [f"Weighted MAE: {audit['panels'][m]['weighted_mae']:.3f}" for m in methods]
    for r in csv.DictReader((ROOT/'experiments/20260908_PXR-MW-efficacy-diagnostic/artifacts/stratum_summaries.csv').open()):
        if r['scope']=='full':
            for page in [SITE/'index.html', SITE/'chapters/interpretation/index.html']:
                for key in ['Emax_ratio_median','Emax_log2FC_median','pEC50_standard_error_median']:
                    assert f"{float(r[key]):.3f}" in page.read_text()
