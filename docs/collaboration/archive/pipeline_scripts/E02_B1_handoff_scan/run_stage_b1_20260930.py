"""Run the frozen Stage B1 smoke gate, then six owned scan workers and readout.

This is orchestration only. Scientific commands and decision thresholds are
those in docs/STAGE_B_RUN_ORDER_LOWER_20260930.md at a714c61.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"D:\py\DRL2")
PROCESS = Path(r"C:\Users\35884\Documents\Spacecraft\过程文件\阶段B1")
RECORDS = PROCESS / "记录"
SCRIPTS = PROCESS / "脚本"
STATE = RECORDS / "STAGE_B1_EXECUTION_20260930.json"
LOCK = RECORDS / "STAGE_B1_SUPERVISOR_20260930.lock"
PY = ROOT / ".venv/Scripts/python.exe"
COMMIT = "a714c61cc3327525ef6c103ecfcb9dc46e552a07"
SEEDS = "262000-262047,270000-270047"
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
ENV = os.environ.copy()
ENV.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def stamp():
    return datetime.now(timezone.utc).astimezone().isoformat()


def save(data, status, **extra):
    data.update(status=status, updated_at=stamp(), **extra)
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, STATE)


def git(*args):
    return subprocess.run(
        ["git", "-c", "safe.directory=D:/py/DRL2", "-C", str(ROOT), *args],
        text=True, capture_output=True, check=True, creationflags=FLAGS,
    ).stdout.strip()


def run_logged(args, name):
    out = RECORDS / f"{name}.stdout.log"
    err = RECORDS / f"{name}.stderr.log"
    with out.open("w", encoding="utf-8") as stdout, err.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            args, cwd=ROOT, env=ENV, stdout=stdout, stderr=stderr,
            creationflags=FLAGS,
        )
    return result.returncode, str(out), str(err)


def preflight(data):
    if git("rev-parse", "HEAD") != COMMIT:
        raise RuntimeError("Stage B1 code commit changed")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked worktree is dirty")
    required = [
        ROOT / "eval/v3e/pure_mpc.json",
        *[ROOT / f"logs/v3e_{seed}/final_model.zip" for seed in (262420, 262421, 262422)],
        *[ROOT / f"eval/v3e/{seed}/learned_only.json" for seed in (262420, 262421, 262422)],
        *[ROOT / f"eval/v3e/{seed}/m2_{part}.json" for seed in (262420, 262421, 262422) for part in "abcd"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing fixed inputs: " + ", ".join(missing))
    save(data, "preflight_passed", input_count=len(required))


def smoke(data):
    output = ROOT / "eval/v3e/stage_b/262420/seed_262006.json"
    if not output.exists():
        save(data, "smoke_running", smoke_started_at=stamp())
        command = [str(PY), "-u", "-B", "-m", "experiments.v3_handoff_scan", "--run-dir", "logs/v3e_262420", "--seeds", "262006", "--scan", "failures", "--output-dir", "eval/v3e/stage_b/262420"]
        code, out, err = run_logged(command, "smoke_262006")
        if code:
            raise RuntimeError(f"262006 smoke failed with exit {code}; logs: {out}, {err}")
    command = [str(PY), "-B", str(SCRIPTS / "verify_seed_262006.py")]
    code, out, err = run_logged(command, "smoke_verification_262006")
    if code:
        raise RuntimeError(f"262006 smoke fidelity STOP_AND_REPORT; logs: {out}, {err}")
    if "ALL_PASS" not in Path(out).read_text(encoding="utf-8"):
        raise RuntimeError("262006 verifier did not print ALL_PASS")
    save(data, "smoke_passed", smoke_completed_at=stamp(), smoke_result=str(output), smoke_verification_log=out)


def stop_owned(workers):
    for worker in workers:
        process = worker["process"]
        if process.poll() is None:
            process.terminate()
    for worker in workers:
        process = worker["process"]
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        worker["stdout_handle"].close()
        worker["stderr_handle"].close()


def scan(data, suffix=""):
    plan = [(262420, 1), (262421, 3), (262422, 2)]
    workers = []
    try:
        save(data, "b1_starting", b1_started_at=stamp())
        for seed, count in plan:
            for index in range(1, count + 1):
                name = f"b1_{seed}_{index}{suffix}"
                out = RECORDS / f"{name}.stdout.log"
                err = RECORDS / f"{name}.stderr.log"
                if out.exists() or err.exists():
                    raise RuntimeError(f"Worker log already exists: {name}")
                stdout = out.open("w", encoding="utf-8")
                stderr = err.open("w", encoding="utf-8")
                command = [str(PY), "-u", "-B", "-m", "experiments.v3_handoff_scan", "--run-dir", f"logs/v3e_{seed}", "--seeds", SEEDS, "--scan", "failures", "--output-dir", f"eval/v3e/stage_b/{seed}"]
                process = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=stdout, stderr=stderr, creationflags=FLAGS)
                workers.append(dict(name=name, seed=seed, process=process, stdout=str(out), stderr=str(err), stdout_handle=stdout, stderr_handle=stderr))
                save(data, "b1_running", workers=[dict(name=w["name"], seed=w["seed"], pid=w["process"].pid, stdout=w["stdout"], stderr=w["stderr"]) for w in workers])
                time.sleep(20)
        while True:
            failed = [(w, w["process"].poll()) for w in workers if w["process"].poll() not in (None, 0)]
            if failed:
                name, code = failed[0][0]["name"], failed[0][1]
                raise RuntimeError(f"Worker {name} exited {code}; inspect stderr before any continuation")
            if all(w["process"].poll() == 0 for w in workers):
                break
            time.sleep(30)
        for worker in workers:
            worker["stdout_handle"].close()
            worker["stderr_handle"].close()
        counts = {str(seed): len(list((ROOT / f"eval/v3e/stage_b/{seed}").glob("seed_*.json"))) for seed, _ in plan}
        locks = {str(seed): len(list((ROOT / f"eval/v3e/stage_b/{seed}").glob("seed_*.lock"))) for seed, _ in plan}
        if any(count != 96 for count in counts.values()) or any(locks.values()):
            raise RuntimeError(f"B1 outputs incomplete: counts={counts}, locks={locks}")
        save(data, "b1_scans_completed", b1_scan_counts=counts, b1_completed_at=stamp())
    except Exception:
        stop_owned(workers)
        raise


def readout(data):
    base = "eval/v3e"
    command = [str(PY), "-B", "-m", "experiments.v3_handoff_readout"]
    for seed in (262420, 262421, 262422):
        command.extend(["--scan", f"{seed}={base}/stage_b/{seed}"])
    for seed in (262420, 262421, 262422):
        command.extend(["--formal-learned", f"{seed}={base}/{seed}/learned_only.json"])
    command.extend(["--formal-pure", f"{base}/pure_mpc.json"])
    for seed in (262420, 262421, 262422):
        files = ",".join(f"{base}/{seed}/m2_{part}.json" for part in "abcd")
        command.extend(["--m2", f"{seed}={files}"])
    command.extend(["--seeds", SEEDS, "--output", f"{base}/stage_b/readout_b1.json"])
    code, out, err = run_logged(command, "readout_b1")
    if code:
        raise RuntimeError(f"B1 readout exited {code}; logs: {out}, {err}")
    result = json.loads((ROOT / "eval/v3e/stage_b/readout_b1.json").read_text(encoding="utf-8"))
    save(data, "b1_readout_completed", verdict=result["verdict"], readout="D:/py/DRL2/eval/v3e/stage_b/readout_b1.json", readout_log=out)


def main():
    RECORDS.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    data = dict(schema="stage_b1_supervisor_v1", started_at=stamp(), supervisor_pid=os.getpid(), code_commit=COMMIT, repo=str(ROOT), python=str(PY))
    try:
        if STATE.exists():
            raise RuntimeError("Existing Stage B1 state found; refusing duplicate launch")
        save(data, "starting")
        preflight(data)
        smoke(data)
        scan(data)
        readout(data)
    except Exception as exc:
        save(data, "stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        os.close(descriptor)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
