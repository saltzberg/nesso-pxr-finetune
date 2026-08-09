#!/usr/bin/env python3
"""Download and verify the released Nesso-1 checkpoint used in this study."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NESSO_REVISION = "1896c84c7186c506c7efd79051480809d51098bf"
NESSO_FILENAME = "v1.0.0/model.safetensors"
NESSO_SHA256 = "9928a8a824d147d665e76656804af1cd91c86731516d0b561f9fd1c91ee45622"
NESSO_URL = (
    "https://huggingface.co/recursionpharma/nesso/resolve/"
    f"{NESSO_REVISION}/{NESSO_FILENAME}"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(destination: Path, *, force: bool = False) -> Path:
    """Download atomically and return a checksum-verified checkpoint path."""

    if destination.is_file() and not force:
        observed = sha256_file(destination)
        if observed == NESSO_SHA256:
            return destination
        raise ValueError(
            f"existing checkpoint checksum mismatch at {destination}: {observed}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        with urllib.request.urlopen(NESSO_URL) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        observed = sha256_file(temporary)
        if observed != NESSO_SHA256:
            raise ValueError(
                f"downloaded checkpoint checksum mismatch: expected {NESSO_SHA256}, "
                f"observed {observed}"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "nesso-1" / NESSO_FILENAME,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    checkpoint = download_checkpoint(args.output, force=args.force)
    print(f"verified {checkpoint}")
    print(f"sha256 {NESSO_SHA256}")


if __name__ == "__main__":
    main()
