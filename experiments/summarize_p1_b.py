import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'logs/p1_b_20260910'

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    assert m['status']=='runs_complete_pending_analysis'
    rows=[(name,json.loads((OUT/(name+'.json')).read_text())['records'][0]) for name in m['commands']]
    rescued=sorted({r['seed'] for _,r in rows if r['completed']})
    remaining=sorted(set(m['seeds'])-set(rescued))
    lines=['# P1-B: approach-rate scan', '',
           'Only the seed(s) not rescued by P1-A are evaluated. The fixed grid is radial steps {0.1,0.2,0.4,0.8,1.6} m per 2 s decision × starts {0,60,120} s × horizons {20,35}. Full-state, no training, no MPC/task/reward changes.', '',
           'Before the start, hold the existing frozen inertial point. At the start, initialize reference radius from the current range and decrease it each decision along the desired-position ray, stopping at the desired position. Nominal radial speed is 0.05–0.8 m/s; the initial direction change and existing waypoint clipping are not bounded by that nominal radial speed. Moving waypoints are a diagnostic device, **not a final learned-interface decision**.', '',
           'Engineering acceleration uses existing runtime_diagnostics=False and cache_target_trajectory=False switches. A full 262006/h20/commit_at(0) replay matches P1-A completion, end time and fallback count, with impulse difference recorded in the manifest. On-demand propagation retains identical RK45 parameters and timestamps. Parallel runtimes have no real-time interpretation.', '',
           '| Seed | h | Start s | Step m | Completed | End s | Force impulse N s | Worst normalized truth margin | Reason | Artifact |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for name,r in rows:
        lines.append(f"| {r['seed']} | {r['horizon']} | {r['commit_time_s_setting']:g} | {r['radius_step_m']:g} | {r['completed']} | {r['final_time_s']:.1f} | {r['force_impulse_n_s']:.3f} | {r['minimum_truth_normalized_margin']:.6g} | {r['termination_reason']} | `{name}.json` |")
    lines += ['',f'Rescued by this stage: {rescued}. Still unrescued: {remaining}.',
              'Gate: skip T4 and continue T5.' if not remaining else 'Gate: continue T4 on the remaining seeds.',
              'This is a post-hoc action-space ceiling on the declared finite grid. It does not establish learnability, global infeasibility, or benefits from any trained coupling.', '']
    (ROOT/'docs/P1_B_APPROACH_RATE_SCAN.md').write_text('\n'.join(lines),encoding='utf-8')
    m.update(status='completed',rescued=rescued,unrescued=remaining,next_stage='T5' if not remaining else 'T4',
             validation='3 selected tests passed: ramp scheduling, bitwise target propagation, diagnostics command equivalence')
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    print(json.dumps({k:m[k] for k in ['rescued','unrescued','next_stage']},indent=2))

if __name__=='__main__': main()
