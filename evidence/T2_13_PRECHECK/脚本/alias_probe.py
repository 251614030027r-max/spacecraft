"""EACC iteration pre-check (sandbox, exploratory): (1) how many of the K counterfactual intents of a decision lead the
executor to (nearly) the same stage goal as the executed intent or as another candidate -- the executor's projection
(rate limit, range floor, braking envelope) maps different intents to the same executed stage; (2) step durations
(SMDP consistency: stages shorter than 16 s, macro entries).

    python -B evidence/T2_13_PRECHECK/脚本/alias_probe.py OUT.json [N_OPENINGS]

Behaviour: uniform random intents (early training) on urgent training-range openings 1,000,000+ (U(140, 200) s).
A stage goal fixes the executor's stage plan exactly (the planner is deterministic in (state, goal)); the executed
stage of two intents with equal goals is identical. Signature distance: ||v_goal_i - v_goal_j|| (m/s); the goal position
follows from it (q_goal = q + h (v + v_goal) / 2: 0.01 m/s = 0.08 m at the stage end).
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import numpy as np, torch; torch.set_num_threads(1)
from env.t2_stage import T2StageEnv, T2Mission, stage_goal, frame_now
from env.t1_p_stage import relative_inertial

out, n_open = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8
mission = T2Mission(urgent_deadline_s=(140.0, 200.0))
rows, steps = [], []
op, used = 1_000_000, 0
grid = np.array([(p, z) for p in np.linspace(-1, 1, 21) for z in np.linspace(-1, 1, 21)])
while used < n_open:
    env = T2StageEnv(openings=[op], candidates=8, shaping=True, seed=op, mission=mission, entry_stages=1)
    env.reset(options={'opening': op}); op += 1
    if not env.mission['urgent']:
        continue
    used += 1; rng = np.random.default_rng(op); done = False
    while not done:
        if env.generator.mode in ('bridge', 'descent'):
            _, _, term, trunc, info = env.step(np.zeros(2)); done = term or trunc
            steps.append({'kind': 'macro', 'stage_s': info['stage_s'], 'event': info['event']}); continue
        e = env.env; q, v = relative_inertial(e); fr = frame_now(e, q); m = e.chaser_parameters.mass
        a = rng.uniform(-1, 1, 2)
        goal = lambda x: stage_goal(q, v, fr, np.asarray(x, float), m, geometry=True, task=e.config.task)
        g_f = goal(a)
        cands = env.candidate_actions(a)
        G = [goal(c) for c in cands]
        dv_f = [float(np.linalg.norm(g['v_goal'] - g_f['v_goal'])) for g in G]
        pair = [float(np.linalg.norm(G[i]['v_goal'] - G[j]['v_goal'])) for i in range(len(G)) for j in range(i + 1, len(G))]
        GG = np.array([goal(x)['v_goal'] for x in grid])
        d_grid = np.linalg.norm(GG[:, None, :] - GG[None, :, :], axis=-1)
        a_dist = np.linalg.norm(grid[:, None, :] - grid[None, :, :], axis=-1)
        far = a_dist >= 0.5
        rows.append({'opening': op - 1, 't': e.time_s, 'range_m': float(np.linalg.norm(q)), 'rate_limited': g_f['rate_limited'],
                     'projected': g_f['safety_projected'], 'cand_to_fact': dv_f, 'cand_pairs': pair,
                     'cand_projected': [g['safety_projected'] for g in G], 'cand_rate_limited': [g['rate_limited'] for g in G],
                     'grid_projected_share': float(np.mean([goal(x)['safety_projected'] for x in grid[::7]])),
                     'grid_far_alias_share': float(((d_grid < 0.005) & far).sum() / max(far.sum(), 1)),
                     'grid_spread_m_s': float(np.ptp(GG, axis=0).max())})
        _, _, term, trunc, info = env.step(a); done = term or trunc
        steps.append({'kind': 'stage', 'stage_s': info['stage_s'], 'held': info['held'], 'event': info['event']})
    print(json.dumps({'opening': op - 1, 'decisions': len(rows), 'event': info['event']}), flush=True)
cf = np.array([x for r in rows for x in r['cand_to_fact']]); cp = np.array([x for r in rows for x in r['cand_pairs']])
st = [s for s in steps if s['kind'] == 'stage' and s['event'] is None]
summary = {'openings': used, 'decisions': len(rows),
           'cand_to_fact_below': {str(t): float(np.mean(cf < t)) for t in (0.001, 0.005, 0.01, 0.02)},
           'cand_pairs_below': {str(t): float(np.mean(cp < t)) for t in (0.001, 0.005, 0.01, 0.02)},
           'decisions_with_a_candidate_aliasing_fact_0.005': float(np.mean([min(r['cand_to_fact']) < 0.005 for r in rows])),
           'cand_projected_share': float(np.mean([x for r in rows for x in r['cand_projected']])),
           'cand_rate_limited_share': float(np.mean([x for r in rows for x in r['cand_rate_limited']])),
           'grid_far_alias_share_mean': float(np.mean([r['grid_far_alias_share'] for r in rows])),
           'grid_far_alias_share_by_range': {f'{lo}-{hi}': float(np.mean([r['grid_far_alias_share'] for r in rows if lo <= r['range_m'] < hi] or [np.nan]))
                                             for lo, hi in ((6, 8), (8, 10), (10, 12), (12, 20))},
           'nonterminal_stage_steps': len(st), 'short_stage_share': float(np.mean([s['stage_s'] < 15.999 for s in st])) if st else None,
           'held_share': float(np.mean([s['held'] for s in st])) if st else None,
           'macro_steps': sum(s['kind'] == 'macro' for s in steps),
           'stage_s_short_values': sorted(round(s['stage_s'], 1) for s in st if s['stage_s'] < 15.999)[:40]}
out.write_text(json.dumps({'summary': summary, 'rows': rows, 'steps': steps}, default=float), encoding='utf-8')
print(json.dumps(summary, indent=1))
