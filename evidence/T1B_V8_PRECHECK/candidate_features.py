import json, sys, numpy as np
from multiprocessing import Pool
def feat(op):
    """G* first-decision candidate features from the deterministic reset (no truth, no rollout)."""
    from env.t1_b_task import MissionDistribution
    from env.t1_b_feasible_v3 import FeasibleArrivalEnvV3
    from env.t1_gstar import free_body_rotations, _rotation_at
    prof = json.load(open('evidence/T1B_V1/profile_mission_v3.json', encoding='utf-8-sig'))
    env = FeasibleArrivalEnvV3(openings=[op], mission=MissionDistribution(**prof)); obs, info = env.reset()
    ch = env.support(); audit = env._support_cache[2]; e = env.env; t = e.target_state
    q = e.chaser_state.position - t.position; los = q / np.linalg.norm(q)
    hist = free_body_rotations(t.rotation, t.omega, e.target_parameters.inertia, max(ch) + 61.)
    rows = []
    for j, tau in enumerate(ch):
        a = [x for x in audit if abs(x['tau'] - tau) < 1e-6][0]
        ang = []
        for dt in (0., 10., 20., 30., 45., 60.):
            R, w = _rotation_at(hist, tau + dt); port = R @ e.config.task.approach_axis
            ang.append(float(np.degrees(np.arccos(np.clip(port @ los, -1, 1)))))
        rows.append({'tau': float(tau), 'dv_lb': float(a.get('necessary_dv_lower_bound') or 0.), 'ang': ang, 'audit': {k: v for k, v in a.items() if isinstance(v, (int, float, bool))}})
    return {'opening': op, 'obs': [float(x) for x in obs], 'mission': {k: v for k, v in env.mission.items() if k != 'distribution'}, 'cands': rows}
if __name__ == '__main__':
    ops = [int(x) for x in open(sys.argv[1]).read().split()]
    with Pool(4) as p:
        with open(sys.argv[2], 'w') as f:
            for r in p.imap(feat, ops): f.write(json.dumps(r) + '\n'); f.flush()
