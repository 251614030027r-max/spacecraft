"""Independently verify the published R2 delivery and external large artifacts."""

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r"C:\Users\35884\Documents\Spacecraft")
LATEST = BASE / "上层交付/最新"
NAME = "V3E_60K_VALUE_R2_MAIN_DELIVERY_20260929"
RECORD = BASE / "过程文件/V3e价值第二轮/记录/V3E_VALUE_R2_FINAL_VERIFIED_20260930.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in rows or len(digest) != 64:
            raise ValueError(f"Invalid manifest entry: {line}")
        rows[relative] = digest
    return rows


def main():
    receipt_path = LATEST / "V3E_VALUE_R2_DELIVERY_RECEIPT_20260929.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bundle = LATEST / NAME
    archive = LATEST / f"{NAME}.zip"
    assert receipt["status"] == "evaluations_completed"
    assert receipt["role"] == "second-round main result"
    assert Path(receipt["archive"]["path"]) == archive
    assert Path(receipt["report"]) == bundle / "REPORT.md"
    assert len(list(LATEST.glob("*.zip"))) == 1
    assert archive.stat().st_size == receipt["archive"]["bytes"]
    assert sha(archive) == receipt["archive"]["sha256"]
    all_files = manifest(bundle / "ALL_FILES_SHA256.txt")
    json_files = manifest(bundle / "JSON_SHA256.txt")
    assert len(json_files) == receipt["json_count"] == 118
    assert all(all_files[name] == digest for name, digest in json_files.items())
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    assert actual == set(all_files) | {"ALL_FILES_SHA256.txt"}
    for relative, digest in all_files.items():
        assert sha(bundle / relative) == digest, relative
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == actual
        for relative, digest in all_files.items():
            assert hashlib.sha256(z.read(relative)).hexdigest() == digest, relative
        assert hashlib.sha256(z.read("ALL_FILES_SHA256.txt")).hexdigest() == sha(bundle / "ALL_FILES_SHA256.txt")
    combined = json.loads((bundle / "combined_results.json").read_text(encoding="utf-8"))
    assert combined["execution_status"] == "evaluations_completed"
    assert combined["official_readout_r2"]["verdict"] == receipt["verdict"] == "does not hold"
    assert receipt["m6_gates"] == {"262420": "PASS", "262421": "STOP", "262422": "STOP"}
    large = json.loads((bundle / "large_artifact_locations.json").read_text(encoding="utf-8"))
    assert len(large) == 33
    for item in large:
        path = Path(item["path"])
        assert path.stat().st_size == item["bytes"], str(path)
        assert sha(path) == item["sha256"], str(path)
    record = {
        "verified_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "verified",
        "archive": str(archive),
        "archive_sha256": sha(archive),
        "zip_entries": len(actual),
        "manifest_files": len(all_files),
        "json_files": len(json_files),
        "external_large_artifacts": len(large),
        "external_large_artifact_bytes": sum(item["bytes"] for item in large),
        "verdict": receipt["verdict"],
        "m6_gates": receipt["m6_gates"],
        "errors": [],
    }
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
