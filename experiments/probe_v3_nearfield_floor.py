"""Is the near-field floor still needed once the gate and reference path are fixed?
Scripted learned-branch policy that advances at full rate while holding commitment
at c (the v3b failure mode), with the floor ON (default) or OFF (floor radius = pose
radius). Counts QP zero fallbacks; runs on the v3e worktree code."""
import sys, json, dataclasses, numpy as np
# usage: python -B experiments/probe_v3_nearfield_floor.py SEED COMMIT on|off OUT.json REPO_ROOT
sys.path.insert(0, sys.argv[5])
from env.hybrid_env import PrecaptureHybridEnv
from train.train_hybrid import accelerated_training_configs
seed, commit, floor, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
ec, hc = accelerated_training_configs(horizon_steps=35, waypoint_parametrization="task_state_v3", execution_feedback=True, adaptive_task=True)
if floor == "off":
    hc = dataclasses.replace(hc, v3_nearfield_radius_m=3.0)
env = PrecaptureHybridEnv(ec, hc)
fb = [0]; orig = env.controller.command
def rec(*a, **k):
    w, d = orig(*a, **k); fb[0] += int(d.used_zero_fallback); return w, d
env.controller.command = rec
env.reset(seed=seed); done = False; n = 0
while not done:
    a1 = 1.0 if env._task_commitment < commit - 1e-9 else 0.0
    _, _, t, tr, info = env.step_with_branch(np.array([1.0, a1]), branch="learned"); n += 1; done = t or tr
json.dump({"seed": seed, "commit": commit, "floor": floor, "decisions": n, "fallbacks": fb[0], "completed": bool(info.get("completed")), "min_range": None}, open(out, "w"))
