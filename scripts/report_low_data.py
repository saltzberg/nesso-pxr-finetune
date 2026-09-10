#!/usr/bin/env python3
"""Generate an auditable low-data report without fitting models."""

import argparse
import json
from pathlib import Path

from nesso_pxr.low_data_report import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_report(args.run_dir, args.report_dir)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "status",
                    "total_tasks",
                    "completed",
                    "failed",
                    "pending",
                    "provenance_errors",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
