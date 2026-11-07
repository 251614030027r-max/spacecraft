import json, pickle, numpy as np
from dataclasses import replace
from env.t1_b_task import MissionDistribution
from env.t1_b_feasible_v1 import bin_center
from env.t1_b_feasible_v5 import FeasibleArrivalEnvV5
from env.t1_b_feasible_v7 import snapshot, proxy, make_predictor
prof = json.load(open('evidence/T1B_V1/profile_mission_v3.json', encoding='utf-8-sig'))
env = FeasibleArrivalEnvV5(openings=[292140], mission=MissionDistribution(**prof), feedback=True); env.reset()
# advance a few steps so the controller/generator carry state
for _ in range(4): env.step(np.array([bin_center(2, 3)], np.float32))
pred = env.predictor; pred.interval_config = replace(pred.interval_config, qp_stride_controls=1)
pred.sync(env); a = pred.rollout(np.zeros(1, np.float32), h=8)
snap = pickle.loads(pickle.dumps(snapshot(env)))
spec = {'config': env.config, 'env_config': env.env.config, 'mission_distribution': env.mission_distribution, 'mission_override': env.mission_override, 'earliest_summary': env.earliest_summary}
p2 = make_predictor(pickle.loads(pickle.dumps(spec))); p2.sync(proxy(snap)); b = p2.rollout(np.zeros(1, np.float32), h=8)
print('direct', a['steps'], a['return'], a['event'], 'snapshot', b['steps'], b['return'], b['event'])
print('max |q diff|', float(np.max(np.abs(np.asarray(a['q']) - np.asarray(b['q'])))), 'reward diffs', np.max(np.abs(np.array(a['rewards']) - np.array(b['rewards']))))
