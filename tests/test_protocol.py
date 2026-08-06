from pathlib import Path

import pytest

from nesso_pxr.protocol import resolve_and_verify_sources


def test_source_verification_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n")
    protocol = {
        "data_root": str(tmp_path),
        "data": {"example": {"path": "source.csv", "sha256": "not-the-hash"}},
    }
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_and_verify_sources(protocol)
