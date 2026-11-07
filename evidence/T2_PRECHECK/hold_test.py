import sys, numpy as np
from dataclasses import replace
from env.t2_stage import T2StageEnv
from env.t1_p_stage import relative_inertial
from env.t1_task import T1InitialDistribution
rng_m = float(sys.argv[1]); op = int(sys.argv[2])
env = T2StageEnv(openings=[op], shaping=False)
env.base_config = replace(env.base_config, initial=replace(env.base_config.initial, range_m=(rng_m, rng_m + 1e-6), inertial_speed_max_m_s=0.0))
env.reset()
g = env.generator; q0, v0 = relative_inertial(env.env)
g.mode, g.plan, g.hold_position = 'hold', None, q0.copy()
print('start r %.2f |v| %.3f omega %.2f deg/s' % (np.linalg.norm(q0), np.linalg.norm(v0), np.degrees(np.linalg.norm(env.env.target_state.omega))))
for k in range(10):
    r, ev, info = env._advance()
    q, v = relative_inertial(env.env)
    print('t %4.0f r %.2f dev %.2f |v| %.3f v_rad %+.3f dv %.3f mode %s' % (env.env.time_s, np.linalg.norm(q), np.linalg.norm(q - q0), np.linalg.norm(v), q @ v / np.linalg.norm(q), env.env._delta_v, g.mode), flush=True)
