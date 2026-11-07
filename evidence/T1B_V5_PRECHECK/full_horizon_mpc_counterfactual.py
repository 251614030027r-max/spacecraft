"""Full-horizon MPC-model counterfactual of each first-review candidate (shadow, nominal RK4 model,
MPC QP every 5 controls), continuation = V1 teacher; compared offline with the truth scan."""
import json, sys, time, numpy as np
from env.t1_b_task import MissionDistribution
from env.t1_b_feasible_v1 import bin_center
from env.t1_b_feasible_v3 import FeasibleArrivalEnvV3
prof = json.load(open('evidence/T1B_V1/profile_mission_v3.json', encoding='utf-8-sig'))
op = int(sys.argv[1]); stride = int(sys.argv[2]) if len(sys.argv) > 2 else 5
env = FeasibleArrivalEnvV3(openings=[op], mission=MissionDistribution(**prof), feedback=True); obs, _ = env.reset()
choices = env.support(); pred = env.predictor; from dataclasses import replace; pred.interval_config = replace(pred.interval_config, qp_stride_controls=stride)
res = []
for j in range(len(choices)):
    t0 = time.perf_counter(); pred.sync(env); a = bin_center(j, len(choices)); total, disc, steps, ev = 0., 1., 0, None
    while steps < 200:
        r = pred.rollout(np.array([a], np.float32), h=200)
        total += disc * r['return']; disc *= r['discount']; steps += r['steps']; ev = r['event']
        if r['done'] or r['steps'] == 0: break
        z = pred.shadow._observation(); n = int((z[49:52] > .5).sum())
        a = bin_center(0, n) if n else 0.
    res.append({'j': j, 'tau': choices[j], 'pred_return': total, 'pred_event': ev, 'pred_steps': steps,
                'pred_dv': pred.shadow.env._delta_v, 'wall_s': time.perf_counter() - t0})
print(json.dumps({'opening': op, 'stride': stride, 'rows': res}, default=float))
