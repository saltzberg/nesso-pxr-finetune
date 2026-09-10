"""The detailed redraw is canonical and keeps supporting architecture records."""
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]

def test_architecture_redraw_published():
    page=(ROOT/'site/architecture/index.html').read_text()
    assert page.count('class="architecture-figure"')==1
    assert 'nesso1-architecture-selected.svg' in page
    assert 'nesso1-architecture-selected.png' in page
    assert 'Adaptation details and earlier diagrams' not in page
    assert page.count('<img ')==1
    for name in ['nesso-native-pipeline.svg','nesso-adaptation-comparison.svg','nesso-repeated-operators.svg']:
        assert name not in page
        assert (ROOT/'reports/architecture'/name).is_file()
    for ext in ['svg','png']:
        name=f'nesso1-architecture-redrawn.{ext}'
        assert (ROOT/'reports/architecture'/name).read_bytes()==(ROOT/'site/evidence/reports/architecture'/name).read_bytes()
    tree=ET.parse(ROOT/'reports/architecture/nesso1-architecture-redrawn.svg')
    assert tree.getroot().get('viewBox')

def test_architecture_renderer_reproducible(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import build_experimental_site as builder
    monkeypatch.setattr(builder,'SITE',tmp_path)
    builder.render_architecture()
    assert (tmp_path/'architecture/index.html').read_bytes()==(ROOT/'site/architecture/index.html').read_bytes()
