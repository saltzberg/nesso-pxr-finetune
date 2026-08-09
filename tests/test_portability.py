from __future__ import annotations

import hashlib
import json
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


def test_tracked_text_has_no_machine_specific_paths() -> None:
    candidates = [
        REPO_ROOT / "Dockerfile",
        REPO_ROOT / ".dockerignore",
        REPO_ROOT / ".gitignore",
    ]
    candidates.extend(
        path
        for path in REPO_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in TEXT_SUFFIXES
        and ".git" not in path.parts
        and ".pytest_cache" not in path.parts
        and ".ruff_cache" not in path.parts
        and "artifacts" not in path.parts
    )
    for path in candidates:
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
