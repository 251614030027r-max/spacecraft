import sys, time
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '.')
import numpy as np
from env.t2_stage import T2StageEnv
for op, a in ((295300, (1.0, 0.0)), (295301, (0.0, 0.0)), (295302, (-0.5, 0.0)), (295303, (1.0, 0.8))):
    env = T2StageEnv(openings=[op], potential='opportunity'); env.reset(options={'opening': op})
    rows = []
    while True:
        t0 = time.perf_counter(); p_new = env._phi(); dt = time.perf_counter() - t0
        env.potential_kind = 'lead24'; p_old = env._phi(); env.potential_kind = 'opportunity'
        rows.append((round(env.env.time_s), round(float(np.linalg.norm(env.env.chaser_state.position - env.env.target_state.position)), 1), round(p_new, 2), round(p_old, 2), round(dt * 1e3, 1)))
        _, r, term, trunc, i = env.step(np.array(a))
        if term or trunc: break
    print(op, a, 'deadline', env.mission['deadline_s'], 'event', i['event'], 't', round(env.env.time_s, 1))
    print('   (t, range, phi_opp, phi_lead24, ms):', rows)
