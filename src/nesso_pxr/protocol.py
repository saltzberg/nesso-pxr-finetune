"""Protocol loading and immutable-source verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping")
    return protocol


def resolve_and_verify_sources(
    protocol: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Path]:
    """Resolve configured inputs relative to the protocol file or current directory."""

    root = Path(protocol.get("data_root", ".")).expanduser()
    if not root.is_absolute():
        anchor = Path.cwd() if base_dir is None else Path(base_dir)
        root = anchor / root
    root = root.resolve()
    resolved: dict[str, Path] = {}
    for name, source in protocol["data"].items():
        path = root / source["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        expected = source["sha256"]
        if observed != expected:
            raise ValueError(
                f"checksum mismatch for {name}: expected {expected}, "
                f"observed {observed}"
            )
        resolved[name] = path
    return resolved
