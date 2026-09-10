import csv, json, math, sys
from pathlib import Path
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'site'

def test_exact_methods_and_placement():
    soup=BeautifulSoup((SITE/'index.html').read_text(),'html.parser')
    label=next(p for p in soup.select('p') if p.get_text().strip()=='Methods and experiments:')
    assert [(a.text,a['href']) for a in label.find_next_sibling('ul').select('a')]==[
        ('Architecture analysis and layer selection','architecture/index.html'),
        ('Head only tuning','experiments/head-only/index.html'),
        ('Fine-tuning screen','experiments/fine-tuning-screen/index.html'),
        ('Chemical splits','experiments/identity-standardization/index.html')]
    assert soup.select_one('#methodology').find_previous('figure')['class']==['native-baseline']
    assert len(soup.select('.native-baseline img'))==1

def test_native_exact_saved_rows_metrics_and_range():
    rows=[r for r in csv.DictReader((ROOT/'reports/architecture/results/heldout_predictions.csv').open()) if r['budget']=='500' and r['method']=='native_continuous']
    exported=list(csv.DictReader((ROOT/'reports/methods-screen/native-points.csv').open()))
    assert rows==exported and len(rows)==len({r['record_id'] for r in rows})==3344
    a=json.loads((ROOT/'reports/methods-screen/verification.json').read_text())
    expected=sum(float(r['weight'])*abs(float(r['y_pred'])-float(r['y_true'])) for r in rows)/sum(float(r['weight']) for r in rows)
    assert math.isclose(a['weighted_mae'],expected,abs_tol=1e-12)
    assert a['axis_limits']==json.loads((ROOT/'reports/narrative/verification.json').read_text())['axis_limits']
    assert all(a['axis_limits'][0]<=float(r[c])<=a['axis_limits'][1] for r in rows for c in ['y_pred','y_true'])
    for suffix in ['svg','png']: assert (SITE/f'evidence/reports/methods-screen/native.{suffix}').stat().st_size>1000

def test_screen_metrics_and_scientific_scope():
    soup=BeautifulSoup((SITE/'experiments/fine-tuning-screen/index.html').read_text(),'html.parser')
    table=soup.select('tbody tr'); assert len(table)==8
    names={'Native unchanged':'native_continuous','Head-only':'head_only','Head + LoRA':'head_plus_lora','Ligand-descriptor LightGBM':'descriptor_lightgbm_rdkit_mordred'}
    metrics={(r['budget'],r['method']):r for r in csv.DictReader((ROOT/'reports/architecture/results/pooled_metrics.csv').open())}
    for row in table:
        cells=[c.get_text() for c in row.select('td')]; budget=cells[0].split()[0][1:]; ref=metrics[budget,names[cells[1]]]
        assert cells[2:]==[f'{float(ref[k]):.4f}' for k in ['weighted_mae','mae','spearman']]
    text=soup.get_text()
    for phrase in ['20 scientific fits','3 historical local fits and 17 remote continuation fits','eight-compound','0.0001','retrospective','esm_proj.1','esm_proj.3','pairformer_stack.layers.0.transition_z.fc1']: assert phrase in text

def test_synthesis_inventory_and_renderer_persistence():
    m=json.loads((SITE/'experiment-manifest.json').read_text())
    assert len([e for e in m['experiments'] if e.get('show_in_navigation',True)])==19
    assert not any(e['slug']=='fine-tuning-screen' for e in m['experiments'])
    assert m['generated_pages'].count('experiments/fine-tuning-screen/index.html')==1
    sys.path.insert(0,str(ROOT/'scripts'))
    from build_methods_screen import render_screen
    before=(SITE/'experiments/fine-tuning-screen/index.html').read_bytes()
    render_screen()
    assert before==(SITE/'experiments/fine-tuning-screen/index.html').read_bytes()
    build=(ROOT/'scripts/build_experimental_site.py').read_text()
    assert 'generated.append(screen_page)' in build and 'screen_source_keys()' in build
