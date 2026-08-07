from __future__ import annotations

import json
import math
from html.parser import HTMLParser
from pathlib import Path

SITE_ROOT = Path(__file__).parents[1] / "site"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.meta_robots: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag in {"a", "link", "script"}:
            target = values.get("href") or values.get("src")
            if target:
                self.links.append(target)
        if tag == "meta" and values.get("name") == "robots" and values.get("content"):
            self.meta_robots.append(values["content"])


def test_site_pages_are_noindex_and_links_resolve() -> None:
    for page in SITE_ROOT.glob("*.html"):
        parser = LinkParser()
        parser.feed(page.read_text(encoding="utf-8"))
        assert "noindex,nofollow" in parser.meta_robots
        for link in parser.links:
            if link.startswith(("http://", "https://", "#")):
                continue
            assert (page.parent / link).resolve().exists(), (
                f"broken link in {page}: {link}"
            )


def test_training_history_contains_completed_measured_points() -> None:
    payload = json.loads((SITE_ROOT / "assets/training-history.json").read_text())
    assert payload["status"] == "complete"
    assert payload["runs"]
    for run in payload["runs"]:
        assert run["epochs"]
        for point in run["epochs"]:
            assert point["epoch"] >= 1
            assert math.isfinite(point["train_loss"])
            assert math.isfinite(point["validation_loss"])
