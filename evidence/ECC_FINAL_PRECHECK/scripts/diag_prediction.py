"""Section 6B diagnosis: prediction against truth for one decision. Fly a fixed intent to time T, take the governor's
probe of the intent (holding it), then fly on and compare every 2 s node: predicted vs true relative position / velocity,
range, braking-envelope margin, the entry pre-screen and whether the real executor accepted an entry.
Usage: PYTHONPATH=. python -B diag_prediction.py OPENING T P Z"""
import sys
import numpy as np
from env.t2_probe import ExecutorProbe, executor_gate
from env.t2_stage import (BRIDGE_ANGLE_DEG, BRIDGE_TAUS_S, T2StageEnv, angle_deg, port_forecast, relative_inertial, so3_exp)
from env.t1_p_stage import plan_state
from experiments.t2 import mission_profile

op, T, a = int(sys.argv[1]), float(sys.argv[2]), np.array([float(sys.argv[3]), float(sys.argv[4])])
env = T2StageEnv(openings=[op], shaping=False, mission=mission_profile('urgent'), obs_profile='cut_priv')
env.reset(options={'opening': op})
while env.env.time_s < T - 1e-6:
    env.step(a)
pr = ExecutorProbe(env); q, v = relative_inertial(env.env); tg = env.env.target_state
rot, om, t0 = tg.rotation.copy(), tg.omega.copy(), float(env.env.time_s)
d = pr.probe_intent(a, q, v, rot, om, t0, float(env.env._delta_v), env.mission, True, gate=True)
print('probe:', {k: d[k] for k in ('event', 'kind', 't', 'admissible', 'tau', 'end')})
# predicted node states (first stage) from the probe's own stage plan
from env.t2_stage import frame_now, spin_frame, stage_goal
frame = spin_frame(rot, om, q, pr.task.approach_axis)
goal = stage_goal(q, v, frame, a, pr.mass, geometry=True, task=pr.task)
plan = pr.stage_planner.solve(q, v, goal['q_goal'], goal['v_goal'])
rows = []
step = env.env.step
def logged(*x, **k):
    out = step(*x, **k)
    t = env.env.time_s - t0
    if abs(t / 2 - round(t / 2)) < 1e-6 and t <= 16 + 1e-6:
        qt, vt = relative_inertial(env.env)
        qp, vp = plan_state(plan, t, pr.mass) if plan['feasible'] else (q, v)
        gt = executor_gate(qt, vt, env.env.target_state.rotation, pr.task, pr.descent_radius_m)
        gp = executor_gate(qp, vp, rot @ so3_exp(t * om), pr.task, pr.descent_radius_m)
        ang_t = min(angle_deg(port_forecast(env.env, tau), qt) for tau in BRIDGE_TAUS_S)
        ang_p = min(angle_deg(rot @ so3_exp((t + tau) * om) @ pr.task.approach_axis, qp) for tau in BRIDGE_TAUS_S)
        rows.append((round(t, 1), env.generator.mode, round(float(np.linalg.norm(qt - qp)), 3), round(float(np.linalg.norm(vt - vp)), 4),
                     round(gt['range_m'], 2), round(gp['range_m'], 2), round(gt['radial_margin_m_s'], 3), round(gp['radial_margin_m_s'], 3),
                     round(ang_t, 1), round(ang_p, 1), env.stats['bridge_attempts']))
    return out
env.env.step = logged
env.step(a)
print('t mode |dq| |dv| range_true range_pred margin_true margin_pred minangle_true minangle_pred bridge_attempts')
for r in rows: print(*r)
