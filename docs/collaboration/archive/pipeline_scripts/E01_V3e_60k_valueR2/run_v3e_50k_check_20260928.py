"""Local orchestration of the user-authorized V3e 50k check; no training mutation."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from datetime import datetime

ROOT = Path(r"D:\py\DRL2_v3e")
DELIVERY = Path(r"C:\Users\35884\Documents\Spacecraft")
OUT = ROOT / "eval" / "v3e"
AUDIT = DELIVERY / "V3E_50K_EXECUTION_20260928.json"
PYTHON = Path(r"D:\py\DRL2\.venv\Scripts\python.exe")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from experiments.check_v3b_training_health import check

def now():
    return datetime.now().astimezone().isoformat()

state = {"started_at": now(), "training_left_running": True, "cwd": str(ROOT),
         "evaluation_commit": subprocess.check_output(["git", "-c", "safe.directory=D:/py/DRL2_v3e", "rev-parse", "HEAD"], text=True).strip(),
         "timeout_statistic": "proxy: not completed and l >= ceil(max_time_s / decision_period_s); monitor has no time_failure column",
         "integrity": [], "jobs": {}}

def save():
    AUDIT.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

for seed in (262420, 262421, 262422):
    run = ROOT / "logs" / f"v3e_{seed}"
    manifest = json.loads((run / "manifest.json").read_text())
    health = check(run)
    if health["status"] != "OK":
        raise RuntimeError(f"health STOP: {health}")
    assert manifest["code_dirty"] is False and manifest["hyperparameters"]["timeout_is_terminal"] is True
    assert manifest["observation_dimension"] == 42 and manifest["action_dimension"] == 2
    model = run / "checkpoints" / "sac_mpc_50000_steps.zip"
    with zipfile.ZipFile(model) as z:
        assert z.testzip() is None
        model_data = json.loads(z.read("data"))
        assert model_data["num_timesteps"] == 50000
    with (run / "train.monitor.csv").open(newline="") as f:
        f.readline()
        rows = list(csv.DictReader(f))
    cumulative = 0
    groups = {}
    timeout_limit = math.ceil(manifest["training_environment"]["max_time_s"] / manifest["decision_period_s"])
    for row in rows:
        group = cumulative // 10000 * 10000
        groups.setdefault(group, []).append(row)
        cumulative += int(row["l"])
    stats = []
    for start, rr in groups.items():
        stats.append({"start_decisions": start, "end_decisions": start + 10000, "episodes": len(rr),
                      "completed": sum(r["completed"] == "True" for r in rr),
                      "completion_rate": sum(r["completed"] == "True" for r in rr) / len(rr),
                      "mean_return": sum(float(r["r"]) for r in rr) / len(rr),
                      "timeout_proxy_fraction": sum(r["completed"] != "True" and int(r["l"]) >= timeout_limit for r in rr) / len(rr),
                      "mean_episode_decisions": sum(int(r["l"]) for r in rr) / len(rr)})
    state["integrity"].append({"seed": seed, "manifest": {k: manifest.get(k) for k in ("code_commit", "code_dirty", "observation_dimension", "action_dimension", "status", "seed", "training_branch", "waypoint_parametrization")},
                                "timeout_is_terminal": manifest["hyperparameters"]["timeout_is_terminal"], "health": health,
                                "checkpoint": str(model), "checkpoint_steps": model_data["num_timesteps"],
                                "checkpoint_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "bins_by_episode_start": stats})

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
OUT.mkdir(parents=True, exist_ok=True)
active = {}
COMMON = ["--episodes", "48", "--seed", "262000", "--horizon", "35"]
V3 = ["--parametrization", "task_state_v3", "--phase-time-observation", "--execution-feedback", "--adaptive-task"]

def launch(name, module, args, expected):
    if any(Path(p).exists() for p in expected):
        raise FileExistsError(f"refusing duplicate/overwrite: {name} {expected}")
    logdir = OUT / "check_50k_logs"
    logdir.mkdir(exist_ok=True)
    stdout_path, stderr_path = logdir / f"{name}.stdout.log", logdir / f"{name}.stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError(f"existing logs: {name}")
    stdout, stderr = stdout_path.open("w"), stderr_path.open("w")
    cmd = [str(PYTHON), "-u", "-B", "-m", module, *args]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    active[name] = (proc, stdout, stderr)
    state["jobs"][name] = {"pid": proc.pid, "started_at": now(), "command": cmd,
                           "stdout": str(stdout_path), "stderr": str(stderr_path), "status": "running"}
    save()
    print(json.dumps({"event": "started", "job": name, "pid": proc.pid}), flush=True)

launch("pure_mpc", "experiments.evaluate_hybrid_policy", COMMON + ["--parametrization", "arrival_condition", "--adaptive-task", "--control", "desired_pose", "--output", "eval/v3e/pure_mpc.json"], [OUT / "pure_mpc.json"])
launch("nominal", "experiments.evaluate_hybrid_policy", COMMON + V3 + ["--control", "v3_nominal", "--output", "eval/v3e/nominal.json"], [OUT / "nominal.json"])
for seed in (262420, 262421, 262422):
    launch(str(seed), "experiments.evaluate_hybrid_policy", COMMON + V3 + ["--model", f"logs/v3e_{seed}/checkpoints/sac_mpc_50000_steps.zip", "--output", f"eval/v3e/{seed}/learned_only_50k.json"], [OUT / str(seed) / "learned_only_50k.json"])
launch("dryrun_m2", "experiments.v3_collect_value_data", ["--run-dir", "logs/v3e_262420", "--model-name", "checkpoints/sac_mpc_50000_steps.zip", "--seeds", "270040-270044", "--probes", "1", "--output", "eval/v3e/dryrun_50k/m2", "--allow-incomplete-run"], [OUT / "dryrun_50k/m2.json", OUT / "dryrun_50k/m2.npz"])
save()
while active:
    for name, (proc, stdout, stderr) in list(active.items()):
        code = proc.poll()
        if code is None:
            continue
        stdout.close(); stderr.close(); del active[name]
        state["jobs"][name].update(status="completed" if code == 0 else "failed", exit_code=code, completed_at=now())
        save()
        print(json.dumps({"event": "finished", "job": name, "exit_code": code}), flush=True)
        if code != 0:
            state.setdefault("errors", []).append(name)
            continue
        if name == "dryrun_m2":
            launch("dryrun_m3", "experiments.v3_fit_values", ["--data", "eval/v3e/dryrun_50k/m2.npz", "--holdout", "270044", "--output-dir", "eval/v3e/dryrun_50k/values"], [OUT / "dryrun_50k/values/m3_report.json"])
        elif name == "dryrun_m3":
            launch("dryrun_m6", "experiments.v3_calibrate_values", ["--run-dir", "logs/v3e_262420", "--model-name", "checkpoints/sac_mpc_50000_steps.zip", "--values", "eval/v3e/dryrun_50k/values", "--seeds", "262100-262101", "--checkpoints", "0,10", "--output", "eval/v3e/dryrun_50k/m6.json", "--allow-incomplete-run"], [OUT / "dryrun_50k/m6.json"])
        elif name == "dryrun_m6":
            launch("dryrun_m5", "experiments.evaluate_hybrid_policy", ["--episodes", "2", "--seed", "262100", "--horizon", "35", *V3, "--model", "logs/v3e_262420/checkpoints/sac_mpc_50000_steps.zip", "--arbiter", "one_way", "--values", "eval/v3e/dryrun_50k/values", "--output", "eval/v3e/dryrun_50k/m5.json"], [OUT / "dryrun_50k/m5.json"])
    if active:
        time.sleep(20)

if not any(state["jobs"][name]["status"] != "completed" for name in ("pure_mpc", "nominal", "262420", "262421", "262422")):
    launch("readout", "experiments.v3e_early_readout", ["--pure-mpc", "eval/v3e/pure_mpc.json", "--nominal", "eval/v3e/nominal.json", "--v3d", *[f"../DRL2/eval/v3d/learned_only_50k_{s}.json" for s in (262420,262421,262422)], "--v3e", *[f"eval/v3e/{s}/learned_only_50k.json" for s in (262420,262421,262422)], "--output", "eval/v3e/early_50k_readout.json"], [OUT / "early_50k_readout.json"])
    proc, stdout, stderr = active.pop("readout")
    code = proc.wait(); stdout.close(); stderr.close()
    state["jobs"]["readout"].update(status="completed" if code == 0 else "failed", exit_code=code, completed_at=now())
    if code:
        state.setdefault("errors", []).append("readout")
state["completed_at"] = now()
save()
print(json.dumps({"event": "all_finished", "errors": state.get("errors", []), "audit": str(AUDIT)}), flush=True)
