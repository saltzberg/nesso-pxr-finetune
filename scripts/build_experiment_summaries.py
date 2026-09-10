#!/usr/bin/env python3
"""Generate and publish evidence-first summary figures; no fitting/inference."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from build_experimental_site import ROOT, SITE, CONTENT, render_experiment, render_overview

ASSETS=SITE/'assets/experiment-summaries'
START='<!-- experiment-summary:start -->'
END='<!-- experiment-summary:end -->'

def build(reuse=False):
    if not reuse:
        for group in ('early','lowdata','later'):
            subprocess.run([sys.executable,str(ROOT/f'scripts/experiment_summary_figures/{group}.py')],check=True)
    manifest=json.loads((CONTENT/'manifest.json').read_text())
    visible=[e for e in manifest['experiments'] if e.get('show_in_navigation',True)]
    assert len(visible)==manifest['visible_count']
    summaries=[]
    for e in visible:
        slug=e['slug']
        if not e.get('summary_figure_enabled', True):
            assert slug == 'identity-standardization', 'Only the user-excluded methods page may omit a summary'
            text=(CONTENT/f'{slug}.md').read_text()
            assert '<figure' not in text and START not in text
            e.pop('summary_figure', None)
            e.pop('summary_message', None)
            continue
        meta=json.loads((ASSETS/f'{slug}.json').read_text())
        assert meta['slug']==slug and meta['message'] and meta['caption']
        for key in ('svg','png','data'):
            assert (SITE/meta[key]).is_file(),(slug,key)
        ET.parse(SITE/meta['svg'])
        rows=list(csv.DictReader((SITE/meta['data']).open()))
        assert len(rows)==meta['plotted_rows']>0
        for source in meta['sources']:
            assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256'],source
        block=f'''{START}
<p class="summary-message">{html.escape(meta['message'])}</p>
<figure class="experiment-summary"><a href="../../{meta['svg']}" title="Open full-size figure"><img src="../../{meta['png']}" alt="{html.escape(meta['message'],quote=True)}"></a><figcaption>{html.escape(meta['caption'])}</figcaption></figure>
<p class="summary-downloads"><a href="../../{meta['png']}" download>PNG</a> · <a href="../../{meta['svg']}" download>SVG</a> · <a href="../../{meta['data']}" download>Plotted data</a> · <a href="../../assets/experiment-summaries/{slug}.json" download>Figure provenance</a></p>
{END}'''
        path=CONTENT/f'{slug}.md'
        text=path.read_text()
        if START in text:
            assert text.count(START)==text.count(END)==1
            text=re.sub(re.escape(START)+r'.*?'+re.escape(END),lambda _:block,text,flags=re.S)
        else:
            anchor='## Design and fitted/frozen flow'
            assert text.count(anchor)==1
            text=text.replace(anchor,block+'\n\n'+anchor)
        path.write_text(text)
        e['summary_figure']=meta['svg']
        e['summary_message']=meta['message']
        e['page']=f'experiments/{slug}/index.html'
        render_experiment(e)
        summaries.append(meta)
    # Preserve complete source-directory coverage and the two archived pages.
    published=json.loads((SITE/'experiment-manifest.json').read_text())
    existing={e['slug']:e for e in published['experiments']}
    for e in manifest['experiments']:
        if not e.get('summary_figure_enabled', True):
            existing[e['slug']].pop('summary_figure', None)
            existing[e['slug']].pop('summary_message', None)
        existing[e['slug']].update(e)
    published['visible_count']=len(visible)
    published['summary_figure_count']=len(summaries)
    (SITE/'experiment-manifest.json').write_text(json.dumps(published,indent=2,sort_keys=True)+'\n')
    (CONTENT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ASSETS/'manifest.json').write_text(json.dumps({'expected_count':len(summaries),'visible_experiment_count':len(visible),'excluded_summary_slugs':['identity-standardization'],'figures':summaries},indent=2)+'\n')
    render_overview(published['experiments'])
    print(json.dumps({'visible_experiments':len(visible),'summary_figures':len(summaries),'archived_engineering_pages':len(manifest['experiments'])-len(visible)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse',action='store_true',help='Validate and publish already generated figures')
    build(parser.parse_args().reuse)
