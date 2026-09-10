"""Completeness and provenance contracts for the experiment figure refresh."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'site'
ASSETS=SITE/'assets/experiment-summaries'


def test_all_visible_experiments_have_summary_figures():
    m=json.loads((SITE/'experiment-manifest.json').read_text())
    visible=[e for e in m['experiments'] if e.get('show_in_navigation',True)]
    excluded={e['slug'] for e in m['experiments'] if not e.get('show_in_navigation',True)}
    assert excluded=={'upstream-feasibility','both-member-pilot'}
    summaries=json.loads((ASSETS/'manifest.json').read_text())
    assert len(visible)==m['visible_count']==summaries['visible_experiment_count']==19
    figure_entries=[e for e in visible if e.get('summary_figure_enabled',True)]
    assert {e['slug'] for e in visible if not e.get('summary_figure_enabled',True)}=={'identity-standardization'}
    assert {e['slug'] for e in figure_entries}=={e['slug'] for e in summaries['figures']}
    assert len(summaries['figures'])==summaries['expected_count']==18
    index=(SITE/'index.html').read_text()
    for slug in excluded:
        assert f'experiments/{slug}/index.html' not in index
        assert (SITE/f'experiments/{slug}/index.html').is_file()
    for e in visible:
        page=(SITE/e['page']).read_text()
        md=(SITE/f'experiment-content/{e["slug"]}.md').read_text()
        if not e.get('summary_figure_enabled',True):
            assert '<figure' not in page and '<img' not in page
            assert '<!-- experiment-summary:start -->' not in md
            assert 'Butina' in page and 'Random' in page
            continue
        assert page.count('<figure class="experiment-summary">')==1
        assert md.count('<!-- experiment-summary:start -->')==1
        assert f'assets/experiment-summaries/{e["slug"]}.png' in page
        assert f'assets/experiment-summaries/{e["slug"]}.csv' in page


def test_summary_render_is_reproducible(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import build_experimental_site as builder
    entries=json.loads((SITE/'experiment-manifest.json').read_text())['experiments']
    monkeypatch.setattr(builder,'SITE',tmp_path)
    for e in entries:
        if not e.get('show_in_navigation',True):
            continue
        builder.render_experiment(e)
        assert (tmp_path/e['page']).read_bytes()==(SITE/e['page']).read_bytes()
    builder.render_overview(entries)
    assert (tmp_path/'index.html').read_bytes()==(SITE/'index.html').read_bytes()


def test_summary_exports_and_exact_source_hashes():
    manifest=json.loads((ASSETS/'manifest.json').read_text())
    for item in manifest['figures']:
        assert item['message'] and item['caption'] and item['design']
        ET.parse(SITE/item['svg'])
        with Image.open(SITE/item['png']) as image:
            assert image.width>=650 and image.height>=300
            image.verify()
        with (SITE/item['data']).open() as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==item['plotted_rows']>0
        assert item['sources']
        for source in item['sources']:
            path=Path(source['path'])
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
