from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from nesso_pxr.protocol import sha256_file
from scripts.download_nesso_checkpoint import NESSO_SHA256, download_checkpoint

REPO_ROOT = Path(__file__).parents[1]
TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN = (
    "/" + "home/",
    "/" + "Users/",
    "/" + "data1/",
    "ADMET" + "-PXR",
    "local/" + "nesso",
)


def test_source_text_has_no_machine_specific_paths() -> None:
    # The staged site is an immutable publication artifact and its source map
    # preserves historical provenance. Enforce portability on executable and
    # repository-maintained source text instead.
    immutable_sources = set()
    for record in json.loads((REPO_ROOT / "site/source-map.json").read_text()):
        source = REPO_ROOT / record["source_key"]
        copy = REPO_ROOT / "site" / record["site_path"]
        if source.is_file() and sha256_file(source) == sha256_file(copy):
            immutable_sources.add(source.resolve())
    summary_manifest = json.loads(
        (REPO_ROOT / "site/assets/experiment-summaries/manifest.json").read_text()
    )
    for figure in summary_manifest["figures"]:
        for record in figure["sources"]:
            source = Path(record["path"])
            if source.is_file() and sha256_file(source) == record["sha256"]:
                immutable_sources.add(source.resolve())

    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode().split("\0")
    candidates = {
        REPO_ROOT / relative
        for relative in tracked
        if relative
        and not relative.startswith("site/")
        and not relative.startswith("experiments/")
        and Path(relative).suffix in TEXT_SUFFIXES
    }
    # This checkout intentionally contains untracked active implementation files.
    # Include executable/test surfaces without sweeping immutable result trees.
    for root in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "tests"):
        candidates.update(root.rglob("*.py"))
    candidates.update((REPO_ROOT / "experiments").glob("*/code/*.py"))

    for path in sorted(candidates):
        if path.resolve() in immutable_sources:
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN:
            assert forbidden not in text, f"{path} contains {forbidden!r}"


def test_published_input_manifest_matches_files() -> None:
    manifest_path = REPO_ROOT / "data" / "published" / "SHA256SUMS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest
    for filename, expected in manifest.items():
        path = manifest_path.parent / filename
        assert path.is_file(), filename
        assert sha256_file(path) == expected


def test_checkpoint_downloader_accepts_verified_existing_file(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"checkpoint fixture"
    destination = tmp_path / "model.safetensors"
    destination.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        "scripts.download_nesso_checkpoint.NESSO_SHA256",
        expected,
    )
    assert download_checkpoint(destination) == destination
    assert NESSO_SHA256 != expected
