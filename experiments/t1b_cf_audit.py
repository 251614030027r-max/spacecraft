"""Independent short-window truth validation of MPC model counterfactuals.

Truth forks exist only in this diagnostic, never in the training wrapper.
States are scheduled by review index, before reading their prediction error.
"""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from copy import deepcopy
import numpy as np

def truth_rollout(predictor,real,action,h):
    predictor.shadow.mission=deepcopy(real.mission)
    predictor.shadow.env.config=real.env.config
    predictor.sync(real,fresh_solvers=True)
    predictor.shadow.mission=deepcopy(real.mission)
    sh=predictor.shadow;sh.generator.last_outcome=real.generator.last_outcome
    rewards=[];steps=[];event=None
    for k in range(h):
        if k and sh._due:break
        obs,r,t,x,i=sh.step(action if k==0 else sh.continue_action())
        rewards.append(r);event=i["event"]
        steps.append({"observation":obs.tolist(),"r":r,"event":event,"t":sh.env.time_s,
                      "dv":i["delta_v_step_m_s"],"execution":i["execution"]})
        if t or x:break
    e=sh.env;q=e.chaser_state.position-e.target_state.position
    v=e.chaser_state.rotation@e.chaser_state.velocity-e.target_state.rotation@e.target_state.velocity
    return {"rewards":rewards,"return":sum(e.config.reward.gamma**k*r for k,r in enumerate(rewards)),
            "event":event,"observation":obs,"q":q,"v":v,"steps":steps}

def clean(r):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()}

def job(args):
    opening,out,profile=args;path=Path(out)/f"audit_{opening}.json"
    if path.exists():return json.loads(path.read_text(encoding="utf-8"))
    from env.t1_b_task import T1BReviewEnv,MissionDistribution
    from env.t1_b_cf import MissionPredictor,alternate_action,ModelFeedbackConfig
    from env.t1_counterfactual import CounterfactualPredictor
    from dynamics.spacecraft import pack_state
    from experiments.t1b_diagnostic import policy_action
    start=time.perf_counter();mission=MissionDistribution(**profile) if profile else MissionDistribution()
    e=T1BReviewEnv(openings=[opening],mission=mission)
    obs,reset=e.reset();p=MissionPredictor(e)
    truth=CounterfactualPredictor(shadow_factory=lambda:T1BReviewEnv(openings=[opening],mission=mission))
    cfg=ModelFeedbackConfig();records=[];ended=False
    while not ended:
        a=policy_action(e,"tau110",obs) if e._due else e.continue_action()
        if e._due and e._review_index in (0,3,6,9,12,15):
            alt=alternate_action(e,a)
            if alt is not None:
                snapshot={"target":pack_state(e.env.target_state).tolist(),
                          "chaser":pack_state(e.env.chaser_state).tolist(),"observation":obs.tolist(),
                          "t":e.env.time_s,"dv":e.env._delta_v,"review":e.review_state(),
                          "a":a.tolist(),"alt":alt.tolist()}
                p.sync(e);f=p.rollout(a,2)
                p.sync(e);c=p.rollout(alt,2)
                tf=truth_rollout(truth,e,a,2);tc=truth_rollout(truth,e,alt,2)
                errors={}
                for label,pr,tr in (("factual",f,tf),("alternative",c,tc)):
                    errors[label]={"reward":abs(pr["return"]-tr["return"]),
                        "position_m":float(np.linalg.norm(pr["q"]-tr["q"])),
                        "velocity_m_s":float(np.linalg.norm(pr["v"]-tr["v"])),
                        "observation":float(np.linalg.norm(pr["observation"]-tr["observation"])),
                        "terminal_match":pr["event"]==tr["event"]}
                records.append({"snapshot":snapshot,"model_factual":clean(f),"model_alternative":clean(c),
                    "truth_factual":clean(tf),"truth_alternative":clean(tc),"errors":errors,
                    "true_local_difference":tc["return"]-tf["return"],
                    "model_local_difference":c["return"]-f["return"],
                    "true_physical_difference_m":float(np.linalg.norm(tc["q"]-tf["q"]))})
                assert np.array_equal(np.asarray(snapshot["chaser"]),pack_state(e.env.chaser_state))
        obs,r,t,x,info=e.step(a);ended=t or x
    result={"opening":opening,"mission":reset["mission"],"config":asdict(cfg),"records":records,
            "event":info["event"],"summary":e.env.summary(),"wall_s":time.perf_counter()-start}
    path.write_text(json.dumps(result,default=float),encoding="utf-8")
    return {"opening":opening,"pairs":len(records),"event":info["event"],"wall_s":result["wall_s"]}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--profile")
    p.add_argument("--openings",default="292040-292043");p.add_argument("--workers",type=int,default=3)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    from experiments.t1n_eval import parse_openings
    profile=json.loads(Path(a.profile).read_text(encoding="utf-8-sig")) if a.profile else None
    (out/"manifest.json").write_text(json.dumps(vars(a)|{"scheduled_reviews":[0,3,6,9,12,15]},indent=2))
    with ProcessPoolExecutor(a.workers) as pool:
        fs=[pool.submit(job,(o,str(out),profile)) for o in parse_openings(a.openings)]
        for f in as_completed(fs):print(json.dumps(f.result()),flush=True)
    (out/"finished.json").write_text(json.dumps({"completed":True,"time":time.time()}))
if __name__=="__main__":main()
