#!/usr/bin/env python3
"""Deterministic, reporting-only site build. No inference or model fitting."""
from pathlib import Path
import csv
import hashlib
import html
import json
import shutil
import re
from markdown_it import MarkdownIt
from build_narrative_figures import build as build_figures

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'
CONTENT = SITE / 'experiment-content'
CYP = ROOT.parent.parent / 'ADMET-CYP' / 'experiments' / '20260908_CYP-Nesso-MW-case-study'
MD = MarkdownIt('commonmark', {'html': True}).enable('table')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def source_path(s):
    return CYP / s[4:] if s.startswith('CYP/') else ROOT / s

def responsive_figures(body):
    def replace_image(match):
        source = match.group(1)
        mobile = source[:-4] + '-mobile.svg'
        return (f'<a class="figure-link" href="{source}" title="Open full-size figure">'
                f'<picture><source media="(max-width: 600px)" srcset="{mobile}">'
                f'{match.group(0)}</picture></a>')
    return re.sub(r'<img\b[^>]*\bsrc="([^"]*/reports/narrative/(?:transfer|mw|mw-overview|hybrid|efficacy)\.svg)"[^>]*>', replace_image, body)

def shell(title, body, prefix=''):
    body = responsive_figures(body)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>{html.escape(title)} · Nesso PXR</title><link rel="stylesheet" href="{prefix}assets/style.css"><style>main img{{max-width:100%;height:auto}} .evidence{{overflow-wrap:anywhere}} table{{display:block;overflow-x:auto}} .study-status{{color:#276749}} .study-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}} .study-grid a{{border:1px solid #d8d4cc;padding:16px;text-decoration:none}} .study-grid small{{display:block;color:#66635f;margin-top:8px}} figure.narrative{{max-width:880px;display:flex;flex-direction:column;gap:8px}} figure.narrative .plot-key{{margin:0}} @media(max-width:600px){{figure.narrative .plot-key{{order:-1}}}} .figure-link{{cursor:zoom-in;display:block}} .figure-link picture{{display:block}} header nav{{display:flex;flex-wrap:wrap;gap:8px 19px}} header nav a{{margin:0;white-space:nowrap}} figure.narrative img{{border:0}} .plot-key{{font:13px/1.5 system-ui,sans-serif;margin-top:12px}} .small-dot{{color:#ad501b}} .other-dot{{color:#356c78}} .reading-path{{font:14px/1.8 system-ui,sans-serif;border-block:1px solid #d8d4cc;padding:16px 0;max-width:100%}} details{{margin:28px 0}} summary{{cursor:pointer;font:600 15px/1.5 system-ui,sans-serif}} .chapter-nav{{margin:28px 0}}</style></head><body><header><div class="masthead"><div class="brand">Nesso-1 · PXR experimental case study</div><p class="updated-at">Updated 2026-09-08</p></div><nav><a href="{prefix}index.html">Overview</a><a href="{prefix}chapters/transfer/index.html">Read the story</a><a href="{prefix}index.html#experiments">Supporting archive</a><a href="{prefix}architecture/index.html">Architecture</a></nav></header><main>{body}</main><footer><p class="fineprint">Nesso → PXR · <a href="{prefix}source-map.json">Evidence provenance</a> · <a href="{prefix}experiment-content/index.md" download>Editable overview</a></p></footer></body></html>\n'''

def render_overview(entries=None):
    """Render only the editable overview; do not rebuild scientific figures."""
    if entries is None:
        entries = json.loads((SITE/'experiment-manifest.json').read_text())['experiments']
    entries = [e for e in entries if e.get('show_in_navigation', True)]
    outline = [('introduction', 'Introduction'), ('methodology', '1. General methodology'),
               ('results', '2. Results'), ('transfer', '2.1. Transfer and tuning'),
               ('mw', '2.2. Molecular-weight controls'), ('hybrid', '2.3. Frozen features'),
               ('interpretation', '2.4. Interpretation'), ('experiments', 'Supporting experiments')]
    links = ''.join(f'<a href="#{key}">{label}</a>' for key, label in outline)
    main = MD.render((CONTENT/'index.md').read_text()).replace('<h1>', '<h1 id="introduction">', 1)
    main = '<nav class="page-outline-top" aria-label="Important subpages"><a href="architecture/index.html">Nesso-1 Architecture</a><a href="https://github.com/saltzberg/nesso-pxr-finetune">GitHub repository</a></nav>' + main
    main += f'<details id="experiments"><summary>Supporting experiments · {len(entries)} studies</summary><p>Complete study records and original evidence, including diagnostic side branches.</p><div class="study-grid">'
    for e in entries:
        main += f'<a href="{e["page"]}"><strong>{html.escape(e["title"])}</strong><small>Completed · {html.escape(e["question"])}</small></a>'
    main += '</div></details><p><a href="experiment-content/index.md" download>Edit/read the canonical overview Markdown</a> · <a href="source-map.json" download>Source map and SHA-256 hashes</a></p>'
    output = shell('Finetuning Nesso-1 to predict PXR induction', main)
    output = re.sub(r'<nav>.*?</nav>', '', output, count=1)
    sidebar = '<aside class="page-outline"><nav aria-label="Page outline"><b>On this page</b>' + links + '</nav></aside>'
    output = output.replace('<body>', '<body class="overview-page">' + sidebar, 1)
    css = """.page-outline{position:fixed;left:24px;top:30px;width:200px;font:13px/1.5 system-ui,sans-serif}
.page-outline nav{display:flex;flex-direction:column;gap:12px}.page-outline nav a{margin:0}
.overview-page header,.overview-page main,.overview-page footer{width:calc(100% - 300px);max-width:1120px;margin-left:260px;margin-right:40px}
.page-outline-top{display:flex;flex-wrap:wrap;gap:10px 18px;border-bottom:1px solid var(--rule);padding-bottom:18px;margin:0 0 28px}
.page-outline-top a{margin:0}.overview-page [id]{scroll-margin-top:24px}
@media(max-width:950px){.page-outline{display:none}.overview-page header,.overview-page main,.overview-page footer{width:calc(100% - 40px);margin-left:20px;margin-right:20px}}
"""
    output = output.replace('</head>', '<style>' + css + '</style></head>')
    (SITE/'index.html').write_text(output)

def render_experiment(entry):
    """Render an experiment from its editable Markdown and existing assets."""
    slug = entry['slug']
    body = MD.render((CONTENT/f'{slug}.md').read_text())
    body += '<details><summary>Source evidence</summary><p>Original reports and exact tables (historical report links may refer to unbundled files).</p><ul class="evidence">'
    for source in entry['sources']:
        body += f'<li><a download href="../../evidence/{html.escape(source)}">{html.escape(source)}</a></li>'
    body += '</ul></details>'
    target = SITE/f'experiments/{slug}/index.html'
    target.parent.mkdir(parents=True,exist_ok=True)
    output = shell(entry['title'], body, '../../')
    output = output.replace('</head>', '<style>.experiment-summary{max-width:760px;margin:24px 0 32px}.experiment-summary img{border:0;width:100%;height:auto}.summary-message{font-weight:600;max-width:760px}.summary-downloads{font:12px/1.5 system-ui,sans-serif}</style></head>')
    target.write_text(output)

def render_architecture():
    """Publish the detailed architecture redraw without rebuilding experiments."""
    diagram = '../evidence/reports/architecture/nesso1-architecture-selected'
    architecture = ('<h1>Nesso-1 architecture</h1>'
        '<p>Nesso-1 predicts affinity from a coarse-grained pairwise representation, not an all-atom structure. '
        'The diagram follows the inputs, token encoding, pair-only trunk, structural latent representation and affinity outputs.</p>'
        f'<figure class="architecture-figure"><div class="architecture-canvas"><a href="{diagram}.svg" title="Open full-size architecture">'
        f'<img src="{diagram}.svg" alt="Detailed Nesso-1 architecture: protein and ligand inputs, token encoders, initial pair matrix, recycled pair-only trunk, distogram and affinity branch"></a></div>'
        '<figcaption>Solid arrows show the main computational flow. Dashed connections show conditioning and recycling. '
        '<strong style="color:#527185">Affinity Pairformer (pale blue)</strong>: selected LoRA adapters inside each affinity branch, including local <code>esm_proj</code> projections—not the external ESM model. '
        '<strong style="color:#8c7153">Affinity output (pale sand)</strong>: trained continuous readout in both members. '
        'Only the selected layers below train; all base weights outside the readout stay frozen. '
        'Open the vector figure to zoom; on small screens, scroll horizontally.</figcaption></figure>'
        f'<p class="fineprint"><a href="{diagram}.svg">Open full-size SVG</a> · <a download href="{diagram}.png">Download PNG</a></p>'
        + '<section id="fine-tuning">' + MD.render((CONTENT/'architecture-finetuning.md').read_text()) + '</section>')
    output = shell('Architecture',architecture,'../')
    output = output.replace('</head>', '<style>header,main,footer{width:min(1700px,calc(100% - 40px))}.architecture-figure{max-width:none}.architecture-canvas{overflow-x:auto}.architecture-figure img{border:0;min-width:1100px}.architecture-figure figcaption{max-width:1000px}#fine-tuning{scroll-margin-top:20px;overflow-wrap:anywhere}#fine-tuning p,#fine-tuning table{max-width:1100px}</style></head>')
    (SITE/'architecture').mkdir(exist_ok=True)
    (SITE/'architecture/index.html').write_text(output)


def build():
    build_figures()
    from build_chemical_split_downloads import build as build_split_downloads, OUT as split_output
    build_split_downloads()
    from build_methods_screen import build_assets, source_keys as screen_source_keys, render_screen, PAGE as screen_page
    build_assets()
    manifest = json.loads((CONTENT/'manifest.json').read_text())
    entries = manifest['experiments']
    assert len(entries) == manifest['expected_count']
    old = SITE/'index.pre-experimental-20260908.html'
    if not old.exists():
        shutil.copyfile(SITE/'index.html', old)
    sources = []
    generated = ['index.html', 'architecture/index.html']
    all_sources = sorted({s for e in entries for s in e['sources']} | {'reports/architecture/nesso-native-pipeline.svg','reports/architecture/nesso-adaptation-comparison.svg','reports/architecture/nesso-repeated-operators.svg'})
    all_sources = sorted(set(all_sources) | {str(p.relative_to(ROOT)) for p in (ROOT/'reports/narrative').glob('*')} | {'reports/architecture/results/prediction-compression.png','reports/architecture/results/control-predictions-full-range.png','reports/architecture/results/mw-controls/index.html','reports/architecture/results/mw-controls/paired-folds.png','reports/architecture/results/mw-controls/paired-folds.svg'})
    all_sources = sorted(set(all_sources) | {'reports/architecture/nesso1-architecture-redrawn.svg', 'reports/architecture/nesso1-architecture-redrawn.png'})
    # Include the subtly highlighted approved architecture and exact layer evidence on full rebuilds.
    from build_architecture_finetuning import source_keys as finetuning_source_keys
    all_sources = sorted(set(all_sources) | finetuning_source_keys())
    all_sources = sorted(set(all_sources) | screen_source_keys())
    all_sources = sorted(set(all_sources) | {str(p.relative_to(ROOT)) for p in split_output.iterdir() if p.is_file()} | {'scripts/build_chemical_split_downloads.py'})
    # The original MW report is navigable, so retain every sibling download.
    all_sources = sorted(set(all_sources) | {
        str(p.relative_to(ROOT))
        for p in (ROOT/'reports/architecture/results/mw-controls').iterdir()
        if p.is_file()
    })
    for s in all_sources:
        src = source_path(s)
        assert src.is_file(), s
        target = SITE/'evidence'/s
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        assert digest(src) == digest(target)
        sources.append({'source':str(src),'source_key':s,'site_path':str(target.relative_to(SITE)),'bytes':src.stat().st_size,'sha256':digest(src)})
    # Preserve the original report's parent navigation without a directory listing.
    archive_rel = 'evidence/reports/architecture/results/index.html'
    archive_body = ('<h1>Original result records</h1>'
                    '<p><a href="../../../../chapters/transfer/index.html">Read the tuning comparison and scatterplots</a></p>'
                    '<p><a href="../../../../chapters/mw/index.html">Read the molecular-weight chapter</a> · '
                    '<a href="mw-controls/index.html">Original molecular-weight report</a></p>')
    (SITE/archive_rel).write_text(shell('Original result records', archive_body, '../../../../'))
    generated.extend([archive_rel, 'evidence/reports/architecture/results/mw-controls/index.html'])
    for e in entries:
        rel = f"experiments/{e['slug']}/index.html"
        render_experiment(e)
        generated.append(rel)
        e['page'] = rel
        e['canonical_markdown'] = f"experiment-content/{e['slug']}.md"
    render_architecture()
    render_screen()
    generated.append(screen_page)
    chapters = [('transfer','Transfer and tuning'),('mw','The molecular-weight control'),('hybrid','Complementary frozen features'),('interpretation','Interpretation and conclusion')]
    for slug, title in chapters:
        rel = f'chapters/{slug}/index.html'
        body = MD.render((CONTENT/'chapters'/f'{slug}.md').read_text())
        body += '<nav class="chapter-nav">' + ' · '.join(f'<a href="../{s}/index.html">{html.escape(t)}</a>' for s,t in chapters if s != slug) + '</nav>'
        body += f'<p class="fineprint"><a download href="../../experiment-content/chapters/{slug}.md">Editable chapter Markdown</a></p>'
        target = SITE/rel
        target.parent.mkdir(exist_ok=True,parents=True)
        target.write_text(shell(title,body,'../../'))
        generated.append(rel)
    manifest['chapters'] = [{'slug':s,'title':t,'page':f'chapters/{s}/index.html'} for s,t in chapters]
    render_overview(entries)
    # Every experiment directory is represented explicitly; fail on new unreviewed studies.
    directories = sorted(x.name for x in (ROOT/'experiments').iterdir() if x.is_dir())
    coverage = {d:[e['slug'] for e in entries if any(s.startswith('experiments/'+d+'/') for s in e['sources'])] for d in directories}
    assert all(coverage.values()), coverage
    manifest['directory_coverage'] = coverage
    manifest['generated_pages'] = generated
    (SITE/'experiment-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (SITE/'source-map.json').write_text(json.dumps(sources,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'experiments':len(entries),'experiment_directories':len(coverage),'generated_pages':len(generated),'curated_sources':len(sources),'curated_bytes':sum(s['bytes'] for s in sources)},sort_keys=True))

if __name__ == '__main__':
    build()
