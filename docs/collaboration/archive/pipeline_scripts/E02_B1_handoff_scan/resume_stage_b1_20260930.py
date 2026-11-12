"""Resume Stage B1 after the recorded external process interruption.

Run only after confirming no scan or supervisor process remains and archiving
their stale locks. Completed seed JSON files are reused unchanged.
"""

import json
import os
import traceback

import run_stage_b1_20260930 as stage


def main():
    if not stage.STATE.is_file():
        raise RuntimeError("Existing Stage B1 state is missing")
    data = json.loads(stage.STATE.read_text(encoding="utf-8"))
    if data.get("status") != "b1_running":
        raise RuntimeError("Expected interrupted b1_running state")
    if not (stage.RECORDS / "B1_INTERRUPTION_20260930.json").is_file():
        raise RuntimeError("Interruption evidence is missing")
    if not (stage.ROOT / "eval/v3e/stage_b/262420/seed_262006.json").is_file():
        raise RuntimeError("Previous smoke result is missing")
    verifier = stage.RECORDS / "smoke_verification_262006.stdout.log"
    if not verifier.is_file() or "ALL_PASS" not in verifier.read_text(encoding="utf-8"):
        raise RuntimeError("Previous smoke gate is not verified")
    descriptor = os.open(stage.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    try:
        data.setdefault("resumptions", []).append(dict(at=stage.stamp(), interruption_record=str(stage.RECORDS / "B1_INTERRUPTION_20260930.json"), reason="external process interruption; no RuntimeError logged"))
        stage.save(data, "resuming", supervisor_pid=os.getpid())
        stage.preflight(data)
        stage.scan(data, suffix="_resume1")
        stage.readout(data)
    except Exception as exc:
        stage.save(data, "stopped", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        os.close(descriptor)
        stage.LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
