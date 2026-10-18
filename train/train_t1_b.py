"""T1-B same-task SAC comparison; DEV smoke is never a formal training run."""
import argparse,json,time,os,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
from functools import partial
from dataclasses import asdict,replace
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from env.t1_gym import T1GymConfig
from env.t1_b_task import T1BReviewEnv,MissionDistribution,OBSERVATION_NAMES
from env.t1_b_cf import T1BFeedbackEnv,ModelFeedbackConfig
from train.t1_b_sac import T1BSAC
from train.t1_tcpf import CosineLearningRate

def json_default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return str(x)

def make_env(arm,seed,rank,mission):
    cls=T1BFeedbackEnv if arm=="tcpf" else T1BReviewEnv
    return cls(config=T1GymConfig(seed_base=1_000_000*(seed+1)+rank*100_000),mission=mission)

class Record(BaseCallback):
    def __init__(self,out,marks):
        super().__init__();self.out=out;self.marks=set(marks);self.saved=set();self.start=time.perf_counter()
    def _on_training_start(self):
        self.handle=(self.out/"decisions.jsonl").open("a",encoding="utf-8")
        self.model.save(self.out/"checkpoints/step_0")
        self.saved.add(0)
    def _on_step(self):
        n=int(self.num_timesteps)
        for e,info in enumerate(self.locals["infos"]):
            line={"step":n,"env":e,"reward":float(self.locals["rewards"][e]),
                  "action_proposed":self.locals["actions"][e],
                  "action_executed":info.get("executed_action"),
                  "event":info.get("event"),"review":info.get("review"),"execution":info.get("execution"),
                  "mission_deadline_s":info.get("mission_deadline_s"),"mission_budget_m_s":info.get("mission_budget_m_s"),
                  "budget_remaining_m_s":info.get("budget_remaining_m_s"),
                  "delta_v_step_m_s":info.get("delta_v_step_m_s"),"counterfactual":info.get("counterfactual")}
            self.handle.write(json.dumps(line,default=json_default)+"\n")
            if "episode_summary" in info:
                with (self.out/"episodes.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps({"step":n,"env":e,**info["episode_summary"],
                        "mission_deadline_s":info.get("mission_deadline_s"),"mission_budget_m_s":info.get("mission_budget_m_s")},default=json_default)+"\n")
        if n%20<len(self.locals["infos"]):
            finite=all(bool(torch.isfinite(p).all()) for p in self.model.policy.parameters())
            heartbeat={"steps":n,"wall_s":time.perf_counter()-self.start,"utc":datetime.now(timezone.utc).isoformat(),
                       "finite":finite,"pid":os.getpid(),"updates":self.model._n_updates,
                       "actor_updates":self.model.actor_updates}
            tmp=self.out/"heartbeat.tmp";tmp.write_text(json.dumps(heartbeat));tmp.replace(self.out/"heartbeat.json")
            self.handle.flush()
            if not finite:
                (self.out/"ABORTED.json").write_text(json.dumps(heartbeat|{"reason":"nonfinite"}))
                return False
        for mark in sorted(self.marks-self.saved):
            if n>=mark:
                self.model.save(self.out/f"checkpoints/step_{mark}")
                for key in ("delta_stats","delta_calibration","tcpf_stats"):
                    (self.out/f"checkpoints/step_{mark}_{key}.json").write_text(json.dumps(getattr(self.model,key),default=json_default))
                self.saved.add(mark)
        return True
    def _on_training_end(self):self.handle.close()

def main():
    p=argparse.ArgumentParser();p.add_argument("--arm",choices=["r3","tcpf"],required=True)
    p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=0)
    p.add_argument("--steps",type=int,default=10000);p.add_argument("--n-envs",type=int,default=4)
    p.add_argument("--profile");p.add_argument("--dev-smoke",action="store_true");p.add_argument("--ready-receipt")
    a=p.parse_args();torch.set_num_threads(1)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    mission=MissionDistribution(**json.loads(Path(a.profile).read_text(encoding="utf-8-sig"))) if a.profile else MissionDistribution()
    if not a.dev_smoke:
        if not a.ready_receipt:raise ValueError("Evidence delivery required before formal training")
        receipt=json.loads(Path(a.ready_receipt).read_text(encoding="utf-8-sig"))
        if not receipt.get("engineering_ready") or not receipt.get("delivered"):
            raise ValueError("No valid tested engineering delivery")
        dirty=subprocess.check_output(["git","status","--porcelain"],text=True)
        if dirty.strip():raise ValueError("Formal training requires committed source")
    (out/"checkpoints").mkdir(parents=True)
    factories=[partial(make_env,a.arm,a.seed,k,mission) for k in range(a.n_envs)]
    env=DummyVecEnv(factories) if a.n_envs==1 else SubprocVecEnv(factories,start_method="spawn")
    model=T1BSAC("MlpPolicy",env,seed=a.seed,device="cpu",verbose=0,
        policy_kwargs={"net_arch":[128,128]},gamma=.999,buffer_size=100000 if not a.dev_smoke else 3000,
        batch_size=256 if not a.dev_smoke else 64,learning_starts=2000 if not a.dev_smoke else 32,
        learning_rate=CosineLearningRate(1e-4,1e-5),ent_coef=.005,tau=.005,
        train_freq=(1,"step"),gradient_steps=a.n_envs,interval_bellman=True,actor_every_transitions=3,
        mc_lambda=0.,eq_lambda=.1,audit_every=0,delta_lambda=0.,
        delta_target_share=.05 if a.arm=="tcpf" else None,delta_cap_share=.1,
        delta_warmup_decisions=2000 if not a.dev_smoke else 32,
        delta_calibration_steps=500 if not a.dev_smoke else 32,
        delta_probe_every=5 if not a.dev_smoke else 1,delta_min_probes=30 if not a.dev_smoke else 5,
        delta_diagnostics_every=20 if not a.dev_smoke else 1)
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    manifest={"kind":"DEV_SMOKE_NOT_FORMAL" if a.dev_smoke else "PAIRED_FORMAL_DEVELOPMENT_TRAINING",
              "args":vars(a),"mission":asdict(mission),"feedback":asdict(ModelFeedbackConfig()),
              "observation":list(OBSERVATION_NAMES),"git_head":head,"started":datetime.now(timezone.utc).isoformat(),
              "network":[128,128],"gamma":.999,"alpha":.005,"critic_updates_per_real_transition":1,
              "actor_every_real_transitions":3,"learning_starts":model.learning_starts,
              "auxiliary":"short MPC model Bellman transition into the same critic" if a.arm=="tcpf" else None}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    marks=[min(a.steps,200),500,2000,5000,10000,a.steps] if a.dev_smoke else [500,2000,5000,10000,a.steps]
    model.learn(total_timesteps=a.steps,callback=Record(out,[m for m in marks if m<=a.steps]))
    model.save(out/"final")
    for key in ("delta_stats","delta_calibration","tcpf_stats","fix_stats"):
        (out/f"{key}.json").write_text(json.dumps(getattr(model,key),default=json_default))
    (out/"interval_audit.json").write_text(json.dumps({"counts":model.replay_buffer.interval_counts,
        "lengths":model.replay_buffer.interval_lengths,"actor_updates":model.actor_updates,"updates":model._n_updates}))
    (out/"finished.json").write_text(json.dumps({"steps":int(model.num_timesteps),"aborted":(out/"ABORTED.json").exists(),
        "finished":datetime.now(timezone.utc).isoformat()}))
    env.close()
if __name__=="__main__":main()
