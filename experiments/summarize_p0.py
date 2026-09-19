"""Summarize the predeclared horizon grid without selecting outcomes."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/p0_horizons_20260910'

def main():
    rows={}
    for h in [20,30,35,50]:
        for seed in range(262000,262012):
            path=(ROOT/f'logs/upper_na_f1/na_f1_h{h}_{seed}.json' if h in [20,50]
                  else OUT/f'fixed_h{h}_{seed}.json')
            rows[h,seed]=json.loads(path.read_text())
    failures={h:[s for s in range(262000,262012) if not rows[h,s]['completed']] for h in [20,30,35,50]}
    persistent=sorted(set.intersection(*(set(v) for v in failures.values())))
    lines=['# P0: tested-horizon persistent failures (2026-09-10)', '',
           'Baseline `3e5694c`; T0 parent `ca42c3a`. Full-state / perception=None / precapture_planning, fixed reference, analytic_local, frozen attitude reference, one episode per seed and horizon. No MPC, task, reward or solver changes.', '',
           '## Completion grid', '', '| Seed | h20 | h30 | h35 | h50 |', '|---|---|---|---|---|']
    for seed in range(262000,262012):
        lines.append('| '+str(seed)+' | '+' | '.join('completed' if rows[h,seed]['completed'] else '**failed**' for h in [20,30,35,50])+' |')
    lines += ['',f'Four-horizon intersection F = {persistent}; |F| = {len(persistent)}.',
              '**Gate: continue T2.**' if len(persistent)>=2 else '**Gate: STOP and report to the user; T2-T5 not executed.**', '',
              'This only establishes behavior in the tested horizon range; it does not establish failure at every possible horizon or physical unreachability.', '',
              '## Pure MPC descriptive rows', '',
              'Time and impulse means below are conditional on completion. Completion rate uses all 12 episodes. For h20/h50 the original pre-transition final_time_s is converted to actual elapsed time as steps × 0.1 s; impulse is sum(force_norm_n × 0.1 s). Their original traces do not record all truth margins, so that entry is unavailable rather than inferred.', '',
              '| Horizon | Completed | Mean completion time s | Mean force impulse N s | Worst normalized truth margin, successful episodes |', '|---|---|---|---|---|']
    for h in [20,30,35,50]:
        success=[rows[h,s] for s in range(262000,262012) if rows[h,s]['completed']]
        times=[r.get('terminal_time_s',r['steps']*.1) for r in success]
        impulses=[r.get('force_impulse_n_s',sum(t['force_norm_n']*.1 for t in r['trace'])) for r in success]
        margins=[r['minimum_truth_normalized_margin'] for r in success if 'minimum_truth_normalized_margin' in r]
        lines.append(f'| h{h} | {len(success)}/12 | {statistics.mean(times):.3f} | {statistics.mean(impulses):.3f} | '+(f'{min(margins):.6g}' if len(margins)==len(success) else 'unavailable in historical traces')+' |')
    lines += ['', '## Difficulty metrics', '',
              'Supplemental h20/h50 replays fill missing telemetry; their completion outcomes must match the preserved originals. New terminal values include the final RK45 transition. Utilization is max-axis absolute command divided by the per-axis limit; force norm must not be divided by a single-axis limit.', '',
              '| Seed | h | End s | Range m | Reason | FOV peak deg | Force mean/peak utilization | Torque mean/peak utilization | Illegal crossings | QP infeasible fraction | Fallback fraction | Truth margin |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for seed in [262005,262006]:
        for h in [20,30,35,50]:
            r=json.loads((OUT/f'fixed_h{h}_{seed}.json').read_text())
            assert r['completed']==rows[h,seed]['completed'], (h,seed,'replay outcome mismatch')
            lines.append(f"| {seed} | {h} | {r['terminal_time_s']:.1f} | {r['terminal_range_m']:.3f} | {r['termination_reason']} | {r['post_fov_angle_max_deg']:.3f} | {r['force_axis_utilization_mean']:.3f}/{r['force_axis_utilization_peak']:.3f} | {r['torque_axis_utilization_mean']:.3f}/{r['torque_axis_utilization_peak']:.3f} | {r['illegal_terminal_entry_count']:.0f} | {r['qp_infeasible_fraction']:.3f} | {r['qp_fallback_fraction']:.3f} | {r['minimum_truth_normalized_margin']:.6g} |")
    lines += ['', '## Deployment-equivalent serial compute', '',
              'The following is the new measured compute reference. One process, no concurrent reachability workers, seed 262000, 300 steps (30 s prefix), diagnostics disabled, controller.command only; includes cold first solve. It is a sampled prefix, not proof of an entire-episode hard real-time bound. Environment RK45 propagation is recorded separately in each JSON. Repeated short runs vary with machine load.', '',
              '| h | Mean ms | p95 ms | Max ms | p95 / 100 ms | Over-budget fraction |', '|---|---|---|---|---|---|']
    for h in [20,30,35,50]:
        r=json.loads((OUT/f'compute_h{h}.json').read_text())
        t=r['controller_ms']
        lines.append(f"| {h} | {t['mean']:.3f} | {t['p95']:.3f} | {t['max']:.3f} | {r['controller_over_budget']['p95']:.3f} | {r['fraction_over_budget']:.4f} |")
    lines += ['', '## Artifacts and boundaries', '',
              '- Original h20/h50 episodes: `logs/upper_na_f1/na_f1_h{20,50}_*.json` (preserved).',
              '- New h30/h35 episodes, four supplemental difficulty replays, and compute JSON: `logs/p0_horizons_20260910/`.',
              '- Manifest contains the fixed seed grid, settings, commands, timing/parallel separation and gate. Reproduce with `python -B -m experiments.run_p0_horizons` in a fresh output directory; the runner refuses to overwrite this evidence.',
              '- No training, no model loading, no T6/T7 interface change. The previous unsubstantiated h30/h35 and diagnostics-disabled compute numbers remain excluded.', '']
    (ROOT/'docs/P0_HORIZON_PERSISTENT_FAILURES.md').write_text('\n'.join(lines),encoding='utf-8')
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest.update(status='completed',failure_sets=failures,persistent_failure_seeds=persistent,
                    gate='continue_T2' if len(persistent)>=2 else 'stop_user_decision_required')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'failures':failures,'F':persistent,'gate':manifest['gate']},indent=2))

if __name__=='__main__':
    main()
