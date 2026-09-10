"""Wait for supervised execution, then require complete verified artifacts."""

import json
import subprocess
import time
from pathlib import Path

SERVICES = [
    "nesso-pxr-low-data-finish.service",
    "nesso-pxr-low-data-report-finish.service",
]
while True:
    states = [
        subprocess.run(
            ["systemctl", "--user", "is-active", name],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        for name in SERVICES
    ]
    if not any(state in {"active", "activating", "deactivating"} for state in states):
        break
    time.sleep(30)
run = Path("artifacts/experiments/low_data_20260905_cut035_run2")
status = json.loads((run / "status.json").read_text())
report = json.loads(Path("site/low-data-verified/summary.json").read_text())
print(
    json.dumps(
        {
            "runner_status": status["status"],
            "total_tasks": report["total_tasks"],
            "verified_complete": report["completed"],
            "failed": report["failed"],
            "pending": report["pending"],
            "provenance_errors": report["provenance_errors"],
        },
        indent=2,
    )
)
assert status["status"] == "complete", "Runner did not complete: investigate/resume"
assert report["completed"] == report["total_tasks"], "Not all tasks verified"
assert report["failed"] == 0 and not report["provenance_errors"]
print("FULL MATRIX COMPLETE AND VERIFIED")
