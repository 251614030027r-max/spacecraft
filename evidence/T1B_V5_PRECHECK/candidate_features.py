import json, sys, numpy as np
from env.t1_b_task import MissionDistribution
from env.t1_b_feasible_v3 import FeasibleArrivalEnvV3
from env.t1_gstar import free_body_rotations, _rotation_at
prof = json.load(open('evidence/T1B_V1/profile_mission_v3.json', encoding='utf-8-sig'))
out = []
for op in [int(x) for x in sys.argv[1:]]:
    env = FeasibleArrivalEnvV3(openings=[op], mission=MissionDistribution(**prof)); obs, info = env.reset()
    ch = env.support(); audit = env._support_cache[2]; e = env.env; t = e.target_state
    q = e.chaser_state.position - t.position; los = q / np.linalg.norm(q)
    hist = free_body_rotations(t.rotation, t.omega, e.target_parameters.inertia, max(ch) + 61.)
    rows = []
    for j, tau in enumerate(ch):
        R, w = _rotation_at(hist, tau); port = R @ e.config.task.approach_axis
        R2, _ = _rotation_at(hist, tau + 30.); port2 = R2 @ e.config.task.approach_axis
        a = [x for x in audit if abs(x['tau'] - tau) < 1e-6][0]
        rows.append({'j': j, 'tau': tau, 'transfer_dv_lb': a.get('necessary_dv_lower_bound'),
                     'port_los_angle_deg': float(np.degrees(np.arccos(np.clip(port @ los, -1, 1)))),
                     'port_los_angle_30s_later_deg': float(np.degrees(np.arccos(np.clip(port2 @ los, -1, 1))))})
    out.append({'opening': op, 'mission': env.mission, 'candidates': rows})
print(json.dumps(out, default=float))
