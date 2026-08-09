from pathlib import Path

import pytest

from nesso_pxr.protocol import load_protocol, resolve_and_verify_sources

REPO_ROOT = Path(__file__).parents[1]


def test_source_verification_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n")
    protocol = {
        "data_root": str(tmp_path),
        "data": {"example": {"path": "source.csv", "sha256": "not-the-hash"}},
    }
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_and_verify_sources(protocol)


def test_repository_protocol_resolves_only_repository_local_inputs() -> None:
    protocol_path = REPO_ROOT / "configs" / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    resolved = resolve_and_verify_sources(
        protocol,
        base_dir=protocol_path.parent,
    )
    assert resolved
    for path in resolved.values():
        assert path.is_relative_to(REPO_ROOT)
