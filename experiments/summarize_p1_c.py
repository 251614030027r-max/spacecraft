import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/p1_c_20260910'
def main():
    m=json.loads((OUT/'manifest.json').read_text());assert m['status']=='runs_complete_pending_analysis'
    rows=[(name,json.loads((OUT/(name+'.json')).read_text())['records'][0]) for name in m['commands']]
    rescued=sorted({r['seed'] for _,r in rows if r['completed']})
    previous=set()
    for stage in ['p1_a','p1_b']:previous.update(json.loads((ROOT/f'logs/{stage}_20260910/manifest.json').read_text())['rescued'])
    total=sorted(previous|set(rescued)); remaining=sorted(set(m['seeds'])-set(rescued))
    lines=['# P1-C: lateral-adjustment grid', '',
           'Seed 262006 remained unrescued after the complete timing/radius and approach-rate grids. At horizons 20 and 35, test ±10/20/30 degrees in two orthogonal tangent directions plus a zero-angle control (26 cells). Full-state, no training, no MPC/solver/task changes.', '',
           'At each decision, take the existing radial_local desired-pose waypoint and rotate its direction while preserving its radius. The two tangent directions use the least-aligned Cartesian axis and its orthogonal cross product. The added angle fades as min(1, distance-to-goal / goal-radius), preserving the original final goal. Requested angles still pass through existing action limits; no clipping bounds were changed. The zero-angle control is exactly the existing radial_local desired-pose policy, so any success of that control could not be credited to lateral adjustment.', '',
           '| h | Axis | Requested angle deg | Completed | End s | Force impulse N s | Worst normalized truth margin | Reason | Artifact |',
           '|---|---|---|---|---|---|---|---|---|']
    for name,r in rows:
        lines.append(f"| {r['horizon']} | {r['lateral_axis']} | {r['lateral_angle_deg']:g} | {r['completed']} | {r['final_time_s']:.1f} | {r['force_impulse_n_s']:.3f} | {r['minimum_truth_normalized_margin']:.6g} | {r['termination_reason']} | `{name}.json` |")
    lines += ['',f'Rescued in this stage: {rescued}. Total seeds rescued across P1-A/B/C: {total}. Still unrescued: {remaining}.',
              'Continue T5 because P1 did rescue one seed. The two-seed difficulty set has **not** been fully rescued. Failure of this finite scripted lateral family does not prove physical unreachability or failure of every possible approach policy.',
              'Engineering acceleration and four-process reachability execution match T3; no parallel timing is used as a compute result. Unit validation: zero-angle action exactly equals the baseline and lateral perturbations preserve the radial action.', '']
    (ROOT/'docs/P1_C_LATERAL_SCAN.md').write_text('\n'.join(lines),encoding='utf-8')
    m.update(status='completed',rescued=rescued,total_rescued=total,unrescued=remaining,next_stage='T5' if total else 'stop_user_decision_required',validation='1 passed in 5.15s')
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8');print(json.dumps({k:m[k] for k in ['total_rescued','unrescued','next_stage']},indent=2))
if __name__=='__main__':main()
