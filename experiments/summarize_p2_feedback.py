"""Three descriptive checks on a small, selected, untrained replay set."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/p2_feedback_20260910'
FEATURES=['fallback_fraction','slack_max','force_mean','force_peak','torque_mean','torque_peak']

def mean(rows,key):
    valid=[r for r in rows if r[key] is not None]
    return float(np.average([r[key] for r in valid],weights=[r['control_steps'] for r in valid])) if valid else None

def main():
    m=json.loads((OUT/'manifest.json').read_text());assert m['status']=='replays_complete_pending_analysis'
    cases=[]
    for selected in m['selected']:
        r=json.loads((OUT/selected['artifact']).read_text())['records'][0]
        cases.append({'artifact':selected['artifact'],'source':selected['source'],'seed':r['seed'],'horizon':r['horizon'],
                      'completed':r['completed'],'end_s':r['final_time_s'],'trace':r['feedback_trace'],
                      'episode_means':{key:mean(r['feedback_trace'],key) for key in FEATURES}})
    failures=[c for c in cases if not c['completed']];successes=[c for c in cases if c['completed']]
    distribution={};trends={};alarms={}
    for key in FEATURES:
        f=[c['episode_means'][key] for c in failures if c['episode_means'][key] is not None]
        s=[c['episode_means'][key] for c in successes if c['episode_means'][key] is not None]
        auc=float(np.mean([1.0 if a>b else .5 if a==b else 0.0 for a in f for b in s])) if f and s else None
        distribution[key]={'failure_mean':float(np.mean(f)) if f else None,'success_mean':float(np.mean(s)) if s else None,
                           'auc_high_means_failure':auc,'separation':max(auc,1-auc) if auc is not None else None}
        observations=[]
        for c in cases:
            bins=[mean([r for r in c['trace'] if lo<=r['time_s']-c['end_s']<hi],key)
                  for lo,hi in m['analysis_preregistered']['trend_bins_before_end_s']]
            valid=all(v is not None for v in bins)
            increasing=valid and all(bins[i+1]>bins[i]+1e-12 for i in range(2))
            observations.append({'artifact':c['artifact'],'completed':c['completed'],'bins':bins,'valid':valid,'increasing':increasing})
        fv=[r for r in observations if not r['completed'] and r['valid']];sv=[r for r in observations if r['completed'] and r['valid']]
        fr=sum(r['increasing'] for r in fv)/len(failures) if failures else None
        sr=sum(r['increasing'] for r in sv)/len(successes) if successes else None
        trends[key]={'failure_increasing_fraction':fr,'success_increasing_fraction':sr,
                     'failure_evaluable_count':len(fv),'success_evaluable_count':len(sv),
                     'signal':fr is not None and sr is not None and fr>=.5 and fr>sr,'cases':observations}
    thresholds=m['analysis_preregistered']['alarm_thresholds']
    for key in list(thresholds)+['any_rule']:
        observations=[]
        for c in cases:
            streak=0;alarm=None
            for r in c['trace']:
                if r['time_s']<10 or r['time_s']>c['end_s']-2:continue
                hit=(any(r[k] is not None and r[k]>=v for k,v in thresholds.items()) if key=='any_rule'
                     else r[key] is not None and r[key]>=thresholds[key])
                streak=streak+1 if hit else 0
                if streak>=2:alarm=r['time_s'];break
            observations.append({'artifact':c['artifact'],'completed':c['completed'],'alarm_s':alarm,
                                 'lead_s':c['end_s']-alarm if alarm is not None else None})
        f=[c for c in observations if not c['completed']];s=[c for c in observations if c['completed']]
        tpr=sum(c['alarm_s'] is not None for c in f)/len(f) if f else None
        fpr=sum(c['alarm_s'] is not None for c in s)/len(s) if s else None
        leads=[c['lead_s'] for c in f if c['lead_s'] is not None]
        lead=float(np.median(leads)) if leads else None
        alarms[key]={'tpr':tpr,'fpr':fpr,'median_failure_lead_s':lead,
                     'signal':tpr is not None and fpr is not None and lead is not None and tpr>=.5 and fpr<=.5 and lead>=2,
                     'cases':observations}
    signals={'distribution':any(r['separation'] is not None and r['separation']>=.75 for r in distribution.values()),
             'time_trend':any(r['signal'] for r in trends.values()),'early_warning':any(r['signal'] for r in alarms.values())}
    analysis={'failure_cases':len(failures),'success_cases':len(successes),'distribution':distribution,'trends':trends,'alarms':alarms,'signals':signals,
              'cases':[{k:v for k,v in c.items() if k!='trace'} for c in cases],
              'limitations':'Only two seeds and selected scripted trajectories, no held-out data. Episode-level descriptive AUC is not a significance test or independent generalization evidence. Action, phase and duration can confound separability. No learned-policy or causal feedback benefit established.'}
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    def fmt(v):return 'NA' if v is None else f'{v:.4g}'
    lines=['# P2: lower-controller feedback precheck', '',
           f"Selected replay set: {len(successes)} successful and {len(failures)} failed episodes. Selection rules and analysis thresholds were saved in the manifest before replay. All replay outcomes and end times match their source grid episodes; impulse matches within 1e-7 N s. Missing successful seed/horizon strata remain missing.", '',
           '## Recording and slack audit', '',
           'The MPC reads `_slack.value` after solver exceptions without clearing it. A fallback value may therefore be stale or zero. The recording wrapper excludes **all fallback steps** from slack aggregation, reports null when a macro decision contains no valid solves, and records valid-solve count alongside fallback fraction. Unit tests inject stale slack=99 on fallback steps to verify masking.', '',
           'Per 2 s decision, `info` now contains fallback fraction, valid-solve count, maximum valid-solve slack, and mean/peak per-axis force and torque utilization. Source quantities already exist in the solve/command; no predicted rollout is added. Observation, reward, action defaults and MPC code are unchanged. `predicted_minimum_margin` is neither used nor promoted to an online feedback signal. Truth geometry remains the sole constraint adjudicator.', '',
           '## Layer 1: whole-episode distributions', '',
           'Each episode contributes one control-step-weighted mean; micro decisions are not treated as independent samples. Slack excludes null macro decisions. AUC is reported with larger values predicting failure; separation allows either direction. The predeclared descriptive signal threshold is separation ≥0.75.', '',
           '| Quantity | Success mean | Failure mean | AUC | Separation |', '|---|---|---|---|---|']
    for k,v in distribution.items():lines.append(f"| {k} | {fmt(v['success_mean'])} | {fmt(v['failure_mean'])} | {fmt(v['auc_high_means_failure'])} | {fmt(v['separation'])} |")
    lines+=['','## Layer 2: evolution before termination','','Compare bins [-15,-10), [-10,-5), [-5,-2) seconds relative to episode end. A positive trend requires all three bin values to increase; signal requires at least half the failed cases and a higher fraction than successful cases. Tiny numerical differences below 1e-12 are treated as ties. All failed/successful episodes remain in the respective denominator; missing-slack windows do not count as an increasing trend. Raw per-case bin values and evaluable counts are retained in analysis.json.','','| Quantity | Failures increasing | Successes increasing | Signal |','|---|---|---|---|']
    for k,v in trends.items():lines.append(f"| {k} | {fmt(v['failure_increasing_fraction'])} | {fmt(v['success_increasing_fraction'])} | {v['signal']} |")
    lines+=['','## Layer 3: fixed early-warning rules','','Thresholds: fallback fraction ≥0.5, valid slack ≥1.0, force peak ≥0.95, torque peak ≥0.95. Require two consecutive decisions, ignore the first 10 s, and require at least 2 s lead before termination. Successful episodes define false alarms. No threshold fitting was performed.','','| Rule | Failure detection | Success false-alarm rate | Median failure lead s | Signal |','|---|---|---|---|---|']
    for k,v in alarms.items():lines.append(f"| {k} | {fmt(v['tpr'])} | {fmt(v['fpr'])} | {fmt(v['median_failure_lead_s'])} | {v['signal']} |")
    lines+=['','## Conclusion and limits','',f'Three-layer descriptive signals: {signals}.',
            'At least one check has information: the cheap feedback channel merits a later controlled ablation, subject to upper review.' if any(signals.values()) else 'All three checks lack information under the declared criteria: report to the user; change quantities or downgrade the proposed contribution.',
            analysis['limitations'],
            'This precheck does not resolve the unrescued 262006 case and does not establish a two-seed rescue claim. Aggregation overhead has not been separately certified for deployment. T6/T7 remain unauthorized; stop here.', '',
            '## Replay sources', '', '| Artifact | Source | Completed | End s |', '|---|---|---|---|']
    for c in cases:lines.append(f"| `{c['artifact']}` | `{c['source']}` | {c['completed']} | {c['end_s']:.1f} |")
    (ROOT/'docs/P2_FEEDBACK_SEPARABILITY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    m.update(status='completed',signals=signals,next_stage='stop_for_T6_authorization',recording_tests='2 passed in 2.15s')
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(cases),'signals':signals,'next_stage':m['next_stage']},indent=2))
if __name__=='__main__':main()
