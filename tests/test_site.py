from __future__ import annotations

import hashlib
import json
from pathlib import Path

SITE_ROOT = Path(__file__).parents[1] / "site"


def test_site_matches_published_release_manifest() -> None:
    """The repository site tree is the exact staged public release."""
    manifest_path = SITE_ROOT / "release-manifest.json"
    release = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert release["route"] == "/projects/finetuning-nesso-1/"
    assert release["github"] == "https://github.com/saltzberg/nesso-pxr-finetune"

    expected = set(release["files"]) | {"release-manifest.json"}
    actual = {
        path.relative_to(SITE_ROOT).as_posix()
        for path in SITE_ROOT.rglob("*")
        if path.is_file()
    }
    assert actual == expected

    for relative, record in release["files"].items():
        path = SITE_ROOT / relative
        payload = path.read_bytes()
        assert len(payload) == record["bytes"], relative
        assert hashlib.sha256(payload).hexdigest() == record["sha256"], relative


def test_release_contains_only_the_current_experimental_site() -> None:
    current = {
        "index.html",
        "architecture/index.html",
        "experiment-manifest.json",
        "source-map.json",
    }
    assert all((SITE_ROOT / relative).is_file() for relative in current)

    superseded = {
        "index.pre-experimental-20260908.html",
        "introduction.html",
        "data.html",
        "modeling.html",
        "training-results.html",
        "low-data",
        "low-data-verified",
        "low-data-assessment",
        "low-data-followup",
    }
    assert all(not (SITE_ROOT / relative).exists() for relative in superseded)


def test_release_assets_are_complete() -> None:
    figure_root = SITE_ROOT / "assets" / "figures"
    expected = {
        "01_predicted_vs_observed.png",
        "02_performance_vs_distance.png",
        "03_residual_vs_activity.png",
        "04_learning_curves.png",
        "05_hyperparameter_landscape.png",
        "06_head_behavior.png",
        "07_binder_proxy_roc_pr.png",
    }
    assert {path.name for path in figure_root.glob("*.png")} == expected
    assert all(
        (figure_root / filename).stat().st_size > 10_000 for filename in expected
    )
