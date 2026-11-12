"""Verify that committed review files match their published SHA manifest."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(r"C:\Users\35884\Documents\Spacecraft\过程文件\远端交付\审查工作树")
RELATIVE = "docs/reviews/v3e_60k_value_r2_20260930"
MANIFEST = ROOT / RELATIVE / "REVIEW_FILES_SHA256.txt"


def main():
    safe = f"safe.directory={ROOT.as_posix()}"
    head = subprocess.check_output(["git", "-c", safe, "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = f"{RELATIVE}/{name}"
        blob = subprocess.check_output(["git", "-c", safe, "-C", str(ROOT), "show", f"HEAD:{path}"])
        assert hashlib.sha256(blob).hexdigest() == expected, path
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
        checked += 1
    print(json.dumps({"status": "verified", "commit": head, "files": checked}))


if __name__ == "__main__":
    main()
