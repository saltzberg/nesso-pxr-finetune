"""Reporting-only split exports must retain saved identities and assignments."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/chemical-splits'


def test_downloads_preserve_saved_assignments_and_acquisitions():
    spec = importlib.util.spec_from_file_location('split_downloads', ROOT / 'scripts/build_chemical_split_downloads.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected, _, counts = module.tables()
    assert counts['completed_low_data_models'] == {'random': 3600, 'chemical_cluster': 3600}
    for name, frame in expected.items():
        saved = pd.read_csv(OUT / name, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(saved, frame.reset_index(drop=True))
        assert saved.record_id.ne('').all()
        assert saved.canonical_smiles.ne('').all()
        assert saved.original_id.ne('').all()


def test_download_source_and_published_hashes():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for path, expected in manifest['sources'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert len(manifest['exports']) == 7
    for name, item in manifest['exports'].items():
        data = (OUT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        assert (ROOT / 'site/evidence/reports/chemical-splits' / name).read_bytes() == data
