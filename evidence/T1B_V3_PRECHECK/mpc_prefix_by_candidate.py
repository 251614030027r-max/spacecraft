import json, sys, numpy as np, glob
from env.t1_b_task import MissionDistribution
from env.t1_b_feasible_v1 import bin_center
from env.t1_b_feasible_v3 import FeasibleArrivalEnvV3
prof = json.load(open('evidence/T1B_V1/profile_mission_v3.json', encoding='utf-8-sig'))
op = int(sys.argv[1]); rows = []
env = FeasibleArrivalEnvV3(openings=[op], mission=MissionDistribution(**prof), feedback=True); obs, _ = env.reset()
choices = env.support(); pred = env.predictor
for j in range(len(choices)):
    pred.sync(env); r = pred.rollout(np.array([bin_center(j, len(choices))], np.float32))
    o = np.asarray(r['observation'])
    rows.append({'j': j, 'tau': choices[j], 'prefix_return': r['return'], 'steps': r['steps'], 'budget_remaining_end': float(o[44]*2), 'plan_remaining_end': float(o[28]*200)})
print(json.dumps({'opening': op, 'rows': rows}))
