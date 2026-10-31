"""Resumable real closed-loop mission diagnostics; no training or formal openings."""
from __future__ import annotations
import argparse, json, os, time, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from dataclasses import asdict

def policy_action(env, row, obs, model=None):
    from env.t1_n_review import action_for_tau
    if row == "zero":
        return np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)
    if env.generator.remaining_s(env.env) > 0:
        return np.zeros(1)
    if row == "earliest":
        tau = env.earliest_feasible_summary()
        tau = 20. if tau is None else tau
    elif row.startswith("tau"):
        tau = float(row[3:])
    elif row == "mission_rule":
        # Public analytic time rule, no future scores or per-opening lookup.
        tau = min(110., env.mission["deadline_s"] - env.env.time_s - 60.)
        if env.mission["budget_m_s"] < 1.45:
            tau = min(135., env.mission["deadline_s"] - env.env.time_s - 60.)
        early = env.earliest_feasible_summary()
        if early is not None:
            tau = max(tau, early)
    else:
        raise ValueError(row)
    return np.array([action_for_tau(tau, env.env.time_s, env.env.config.max_time_s,
                                    env.generator.deadline_margin_s)], np.float32)

def run(job):
    opening, row, out, profile, capture_snapshots = job
    path = Path(out) / f"{row}_{opening}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from env.t1_b_task import T1BReviewEnv, MissionDistribution
    from env.t1_n_review import T1NReviewEnv
    start = time.perf_counter()
    mission = MissionDistribution(**profile) if profile else MissionDistribution()
    env = T1BReviewEnv(openings=[opening], mission=mission)
    obs, reset = env.reset()
    model = None
    if row == "zero":
        import torch
        torch.set_num_threads(1)
        from train.t1_n_sac import T1NSAC
        model = T1NSAC("MlpPolicy", env, seed=0, device="cpu", buffer_size=100,
                       learning_starts=2000, policy_kwargs={"net_arch":[128,128]},
                       mc_lambda=0., eq_lambda=0., delta_lambda=0., verbose=0)
    initial_obs = obs.tolist()
    zero_hash = None
    if model is not None:
        zero_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                          for p in model.policy.parameters())).hexdigest()
    steps, ended, info = [], False, {}
    while not ended:
        a = policy_action(env, row, obs, model) if env._due else env.continue_action()
        obs, reward, term, trunc, info = env.step(a)
        e, g = env.env, env.generator
        q = e.chaser_state.position-e.target_state.position
        v = e.chaser_state.rotation@e.chaser_state.velocity-e.target_state.rotation@e.target_state.velocity
        steps.append({"t":e.time_s,"r":reward,"dv":info["delta_v_step_m_s"],
                      "budget_used":e._delta_v,"a":a.tolist(),"review":info["review"],
                      "execution":info["execution"],"q":q.tolist(),"v":v.tolist(),
                      "observation":obs.tolist(),"control_trace":info["resource_control_trace"]})
        ended = term or trunc
    gamma = env.env.config.reward.gamma
    result = {"opening":opening,"row":row,"profile":asdict(mission),"mission":reset["mission"],
              "initial_observation":initial_obs,"zero_policy_sha256":zero_hash,
              "event":info["event"],"return":sum(x["r"] for x in steps),
              "discounted_return":sum(gamma**k*x["r"] for k,x in enumerate(steps)),
              "summary":env.env.summary(),"end_range_m":env.env.metrics().range_m,
              "violations":info.get("violations",{}),"steps":steps,"wall_s":time.perf_counter()-start}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,default=float),encoding="utf-8")
    tmp.replace(path)
    return {k:v for k,v in result.items() if k not in ("steps","initial_observation")}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--openings",default="292000-292011")
    p.add_argument("--rows",default="earliest,tau110,tau150")
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--profile")
    a=p.parse_args()
    from experiments.t1n_eval import parse_openings
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    profile=json.loads(Path(a.profile).read_text(encoding="utf-8-sig")) if a.profile else None
    jobs=[(o,r,str(out),profile,False) for o in parse_openings(a.openings) for r in a.rows.split(",")]
    (out/"manifest.json").write_text(json.dumps({"jobs":jobs,"started_utc":time.time(),"argv":vars(a)},indent=2),encoding="utf-8")
    results=[]
    with ProcessPoolExecutor(a.workers) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        for f in as_completed(futures):
            r=f.result();results.append(r)
            print(json.dumps({k:r[k] for k in ("opening","row","event","return","wall_s")}),flush=True)
    (out/"finished.json").write_text(json.dumps({"completed":len(results),"time":time.time()}),encoding="utf-8")
if __name__=="__main__":main()
