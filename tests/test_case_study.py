"""Real saved-artifact checks for the consolidated case-study renderer."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('case_study_renderer', ROOT / 'scripts/render_case_study.py')
assert spec is not None and spec.loader is not None
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def test_real_render_is_byte_identical():
    renderer.main()
    before = {name: (renderer.OUT / name).read_bytes() for name in ['index.html', 'verification.json']}
    renderer.main()
    assert before == {name: (renderer.OUT / name).read_bytes() for name in before}


def test_wrong_displayed_score_is_rejected(tmp_path, monkeypatch):
    text = (renderer.OUT / 'README.md').read_text()
    assert '| Fine-tuned continuous heads | 1.165 | 1.320 |' in text
    (tmp_path / 'README.md').write_text(text.replace('| Fine-tuned continuous heads | 1.165 | 1.320 |', '| Fine-tuned continuous heads | 1.165 | 1.249 |'))
    monkeypatch.setattr(renderer, 'OUT', tmp_path)
    with pytest.raises(AssertionError):
        renderer.main()
    assert not (tmp_path / 'index.html').exists()
