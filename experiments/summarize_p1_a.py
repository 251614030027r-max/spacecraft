import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/p1_a_20260910'

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    cells=m['cells']
    for c in cells:
        c['result']=json.loads((OUT/c['artifact']).read_text())['records'][0]
    controls={c['seed'] for c in cells if c['layer']=='timing_only' and c['result']['completed']}
    rescued={c['seed'] for c in cells if c['result']['completed']}
    lines=['# P1-A: timing control and incremental hold-radius scan', '',
           'Full-state / perception=None; unchanged MPC, dynamics, constraints, reward, 300 s budget and absolute waypoint parameterization. The only added policy freedom is the radius of the existing frozen inertial hold. No training or learned interface changes.', '',
           f'Timing-only control rescues {sorted(controls)}; timing plus radius rescues {sorted(rescued)}. Adding hold radius beyond the fixed commit-time scan rescues **{len(rescued-controls)} additional seed(s)**: {sorted(rescued-controls)}.',
           'These are post-hoc per-seed action-space ceilings, **not proof of a learnable policy**. Results from different horizons are shown separately; rescue at some horizon does not imply rescue at every horizon.', '',
           'Each cell below is `C/F; end time s; force impulse N s; worst normalized RK45 truth margin`. F means failure, so its time/fuel must not be read as a successful-mission performance figure. All predeclared cells are reported. Six final cells also reuse an earlier identical hold episode that terminated before its commit time (source references and reasons are in the manifest). Initial-radius cells reuse the identical timing-only control; at t=0 no hold command executes, so all radii reuse that same control trajectory.', '']
    def value(c):
        r=c['result']
        return f"{'C' if r['completed'] else 'F'}; {r['final_time_s']:.1f}; {r['force_impulse_n_s']:.2f}; {r['minimum_truth_normalized_margin']:.5g}"
    for seed in m['seeds']:
        for h in m['horizons']:
            subset=[c for c in cells if c['seed']==seed and c['horizon']==h]
            lines += [f'## Seed {seed}, h{h}', '', '### Timing-only control', '', '| Commit s | Result | Artifact |', '|---|---|---|']
            for c in sorted([c for c in subset if c['layer']=='timing_only'],key=lambda c:c['time_s']):
                lines.append(f"| {c['time_s']} | {value(c)} | `{c['artifact']}` |")
            lines += ['', '### Timing × radius increment', '', '| Commit s | Initial radius | 12 m | 14 m | 16 m | 18 m |', '|---|---|---|---|---|---|']
            for t in m['times_s']:
                cs=[next(c for c in subset if c['layer']=='timing_radius' and c['time_s']==t and c['radius_m']==r) for r in m['radii_m']]
                lines.append('| '+str(t)+' | '+' | '.join(value(c) for c in cs)+' |')
            success=sum(c['result']['completed'] for c in subset if c['layer']=='timing_radius')
            lines += ['',f'Radius-layer successful cells: {success}/30. '+('Every cell succeeds: this points to a baseline-setting issue, not a selective decision problem.' if success==30 else 'No cell succeeds.' if success==0 else 'Only part of the fixed grid succeeds; the table exposes parameter sensitivity.'),'']
    remaining=sorted(set(m['seeds'])-rescued)
    lines += ['## Gate and artifacts', '',
              f'Unrescued seeds: {remaining}. '+('Skip T3/T4; continue T5.' if not remaining else 'Continue T3 on only these unrescued seeds.'),
              'Per-cell JSON and the exact grid/commands are in `logs/p1_a_20260910/`. Each JSON records actual impulse and worst truth margin over every micro step, not merely the final macro decision. The legacy qp_zero_fallbacks field still describes the final macro decision; qp_fallbacks_total is the episode total. No parallel time is used for compute claims.', '']
    (ROOT/'docs/P1_A_COMMIT_TIME_AND_HOLD_RADIUS.md').write_text('\n'.join(lines),encoding='utf-8')
    # Keep manifest compact: original cells point to raw results rather than duplicating them.
    for c in cells: del c['result']
    m.update(status='completed',timing_only_rescued=sorted(controls),rescued=sorted(rescued),unrescued=remaining,
             additional_radius_rescues=sorted(rescued-controls),next_stage='T5' if not remaining else 'T3')
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    print(json.dumps({k:m[k] for k in ['timing_only_rescued','rescued','unrescued','next_stage']},indent=2))

if __name__=='__main__': main()
