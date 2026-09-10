"""Temporary synthetic report fixtures; never scientific result evidence."""

import importlib
import json

import pytest

publisher = importlib.import_module(
    "experiments.20260906_low_data_followup.code.publish_final_analysis"
)


def test_build_preserves_csv_bytes_and_escapes_raw_html(tmp_path, monkeypatch):
    source = tmp_path / "analysis"
    source.mkdir()
    (source / "findings.md").write_text(
        "# Synthetic unit test\n<script>bad()</script>\n"
    )
    data = b"method,value\nfixture,0\n"
    (source / "fixture.csv").write_bytes(data)
    destination = tmp_path / "site"
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    monkeypatch.setattr(publisher, "ANALYSIS", source)
    monkeypatch.setattr(publisher, "DEST", destination)
    result = publisher.build()
    assert result["tables"] == 1
    assert (destination / "tables/fixture.csv").read_bytes() == data
    assert "<script>" not in (destination / "index.html").read_text()
    manifest = json.loads((destination / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        assert publisher.sha((destination / name).read_bytes()) == entry["sha256"]


def test_no_result_tables_is_not_publishable(tmp_path, monkeypatch):
    (tmp_path / "findings.md").write_text("# Synthetic no-data fixture\n")
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    monkeypatch.setattr(publisher, "ANALYSIS", tmp_path)
    monkeypatch.setattr(publisher, "DEST", tmp_path / "site")
    with pytest.raises(ValueError, match="no real analysis tables"):
        publisher.build()
