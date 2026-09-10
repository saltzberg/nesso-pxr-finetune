"""Publish a compact LAN report from completed retrospective analysis artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "artifacts/experiments/low_data_followup_20260906/final_analysis"
DEST = ROOT / "site/low-data-followup"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def build():
    findings = (ANALYSIS / "findings.md").read_text()
    # Presentation copy uses repo-relative paths; artifact original stays intact.
    findings = findings.replace(str(ROOT) + "/", "")
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    summary_path = ROOT / "experiments/20260906_low_data_followup/RESULTS.md"
    summary = summary_path.read_text() if summary_path.exists() else ""
    content = renderer.render(summary) if summary else ""
    content += (
        "<details><summary>Detailed analysis and conditional comparisons</summary>"
        + renderer.render(findings)
        + "</details>"
    )
    combined_report = summary + "\n\n" + findings
    manifest = {
        "scope": "Retrospective saved-prediction analysis; no new model fitting",
        "files": {},
    }
    links = []
    for path in sorted(ANALYSIS.glob("*.csv")):
        data = path.read_bytes()
        target = DEST / "tables" / path.name
        atomic(target, data)
        relative = str(target.relative_to(DEST))
        manifest["files"][relative] = {
            "source": str(path.relative_to(ROOT)),
            "sha256": sha(data),
        }
        links.append(f'<li><a href="{escape(relative)}">{escape(path.name)}</a></li>')
    if not links:
        raise ValueError("no real analysis tables")
    for name in ["analysis_manifest.json", "parent_verification.json"]:
        path = ANALYSIS / name
        if path.exists():
            data = path.read_bytes()
            atomic(DEST / name, data)
            manifest["files"][name] = {
                "source": str(path.relative_to(ROOT)),
                "sha256": sha(data),
            }
            links.append(f'<li><a href="{name}">{name}</a></li>')
    atomic(DEST / "report.md", combined_report.encode())
    manifest["files"]["report.md"] = {
        "sources": [str((ANALYSIS / "findings.md").relative_to(ROOT))]
        + ([str(summary_path.relative_to(ROOT))] if summary else []),
        "sha256": sha(combined_report.encode()),
        "presentation_paths": "repository_relative",
    }
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Nesso PXR: final low-data follow-up analysis</title>"
        "<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1060px;"
        "margin:36px auto;padding:0 24px;color:#222;background:#fff}"
        "h1{font-size:1.65rem}h2{font-size:1.2rem;margin-top:2rem}"
        "table{border-collapse:collapse;display:block;overflow-x:auto;"
        "font-size:.9rem}"
        "td,th{padding:6px 12px;text-align:right;border-bottom:1px solid #ddd}"
        "td:first-child,th:first-child{text-align:left}"
        "a{color:#215b78}code{font-size:.85em}pre{overflow:auto}nav{font-size:.9rem;color:#666}</style>"
        '<nav><a href="../">Project</a> · '
        '<a href="../low-data-assessment/">First-study assessment</a></nav>'
        "<main>" + content + "<h2>Source tables</h2><ul>" + "".join(links) + "</ul>"
        '<p><a href="report.md">Editable report copy</a> · '
        '<a href="manifest.json">Published-file provenance</a></p></main></html>'
    )
    atomic(DEST / "index.html", page.encode())
    manifest["files"]["index.html"] = {"sha256": sha(page.encode())}
    atomic(DEST / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return {
        "files": len(manifest["files"]),
        "tables": sum(name.startswith("tables/") for name in manifest["files"]),
        "report": str(DEST / "index.html"),
    }


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.urls.extend(value for key, value in attrs if key == "href" and value)


def verify(url):
    manifest = json.loads((DEST / "manifest.json").read_text())
    checked = []
    for name, item in manifest["files"].items():
        with urlopen(urljoin(url, name), timeout=30) as response:
            data = response.read()
        if (
            sha(data) != item["sha256"]
            or sha((DEST / name).read_bytes()) != item["sha256"]
        ):
            raise ValueError(f"published hash mismatch: {name}")
        checked.append(name)
    parser = Links()
    parser.feed((DEST / "index.html").read_text())
    broken = []
    for href in sorted(set(parser.urls)):
        if href.startswith(("http:", "https:", "mailto:", "#")):
            continue
        try:
            with urlopen(urljoin(url, href), timeout=30) as response:
                if response.status != 200:
                    broken.append(href)
        except Exception:
            broken.append(href)
    if broken:
        raise ValueError(f"broken report links: {broken}")
    evidence = {
        "url": url,
        "verified_file_count": len(checked),
        "verified_files": checked,
        "link_count": len(set(parser.urls)),
        "broken_links": broken,
    }
    atomic(
        ANALYSIS / "http_verification.json",
        (json.dumps(evidence, indent=2) + "\n").encode(),
    )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "verify"])
    parser.add_argument("--url", default="http://192.168.6.154:8890/low-data-followup/")
    args = parser.parse_args()
    print(json.dumps(build() if args.mode == "build" else verify(args.url), indent=2))


if __name__ == "__main__":
    main()
