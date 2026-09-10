"""Render the canonical case study and verify displayed scores. No fitting."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'reports/architecture/results'
OUT = ROOT / 'reports/architecture/case-study'
NAMES = {
    'Native Nesso, unchanged': 'native_continuous',
    'Native score, fitted linear mapping': 'native',
    'Native score + MW': 'native_mw',
    'Native score + MW + interaction': 'native_mw_interaction',
    'Fine-tuned continuous heads': 'head_only',
    'Fine-tuned heads + upstream LoRA': 'head_plus_lora',
    'Ridge on frozen Nesso vectors': 'repr_ridge_stable',
    'Ligand-descriptor LightGBM': 'descriptor_lightgbm_rdkit_mordred',
}


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    inputs = [RESULTS / 'heldout_predictions.csv', RESULTS / 'mw-controls/predictions.csv']
    neural, controls = map(read_csv, inputs)
    mw = {}
    for row in controls:
        rid, value = row['record_id'], float(row['MW'])
        if rid in mw:
            assert mw[rid] == value, 'MW identity mismatch'
        mw[rid] = value
    assert len(mw) == 3344
    assert sum(v < 215 for v in mw.values()) == 237
    assert sum(v > 215 for v in mw.values()) == 3107
    grouped = defaultdict(list)
    for row in neural + controls:
        w = float(row['weight'])
        assert math.isclose(w, 1 / max(float(row['assay_se']), .1), abs_tol=1e-12)
        grouped[(int(row['budget']), row['method'])].append(row)
    scores = {}
    for (budget, method), rows in grouped.items():
        assert len(rows) == len({r['record_id'] for r in rows}) == 3344
        for subset in ['full', 'lt215']:
            selected = [r for r in rows if subset == 'full' or mw[r['record_id']] < 215]
            weighted_error = math.fsum(float(r['weight']) * abs(float(r['y_pred']) - float(r['y_true'])) for r in selected)
            weight = math.fsum(float(r['weight']) for r in selected)
            scores[(budget, method, subset)] = weighted_error / weight
    source = OUT / 'README.md'
    text = source.read_text()
    small = text.split('### Error among compounds below 215 Da')[1].split('### What this')[0]
    full = text.split('### Full-cohort weighted error')[1].split('## How strong')[0]
    checked = []
    for section, subset, precision in [(small, 'lt215', 3), (full, 'full', 4)]:
        for line in section.splitlines():
            fields = [f.strip() for f in line.strip('|').split('|')]
            if len(fields) != 3 or fields[0] not in NAMES:
                continue
            for budget, displayed in zip([100, 500], fields[1:]):
                actual = scores[(budget, NAMES[fields[0]], subset)]
                assert displayed == f'{actual:.{precision}f}', (fields[0], budget, displayed, actual)
                checked.append({'budget': budget, 'method': NAMES[fields[0]], 'subset': subset, 'weighted_mae': actual, 'displayed': displayed})
    assert len(checked) == 24
    body = MarkdownIt('commonmark').enable('table').render(text)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    style = '''body{font:16px/1.65 system-ui,sans-serif;color:#26323b;max-width:940px;margin:auto;padding:28px}h1{font-size:34px;line-height:1.2}h2{margin-top:2.2em;line-height:1.3}h3{margin-top:1.6em}a{color:#2166ac}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:9px;border-bottom:1px solid #dbe1e5;text-align:left}th:not(:first-child),td:not(:first-child){text-align:right}pre{background:#f5f6f7;padding:16px;overflow:auto}code{font-size:.9em}.scroll{overflow:auto}nav{font-size:14px;border-bottom:1px solid #ddd;padding-bottom:12px}@media(max-width:600px){body{padding:18px;font-size:15px}h1{font-size:28px}table{min-width:570px}}'''
    body = body.replace('<table>', '<div class="scroll"><table>').replace('</table>', '</table></div>')
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><meta name="source-sha256" content="{sha}"><title>Nesso-1 PXR case study</title><style>{style}</style></head><body><nav><a href="../index.html">Architecture</a> · <a href="../results/index.html">Adaptation results</a> · <a href="../results/mw-controls/index.html">Molecular-weight controls</a></nav><main>{body}</main></body></html>'
    (OUT / 'index.html').write_text(document)
    verification = {'status': 'PASS', 'scope': 'saved-prediction arithmetic, displayed tables, identity/weight checks and local links; no training or model-state replay', 'displayed_values_checked': len(checked), 'source_sha256': sha, 'inputs': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'scores': checked}
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key not in ('href', 'src') or not value:
                    continue
                url = urlsplit(value)
                if not url.scheme and url.path:
                    assert (OUT / unquote(url.path)).resolve().exists(), value
    Links().feed(document)
    assert not re.search(r'TODO|TBD|FIXME|example\.com', document)
    print(json.dumps({'status': 'PASS', 'displayed_values_checked': len(checked), 'html': str(OUT / 'index.html'), 'source_sha256': sha}))


if __name__ == '__main__':
    main()
