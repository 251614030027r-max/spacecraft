"""Read saved JSON only; no model loads, simulation or training."""
import json, statistics, collections
from pathlib import Path
r = Path(__file__).resolve().parents[1]
out = {}
for arm in ('e_sac', 'e_tcpf'):
    d = r / ('TRAIN_' + arm + '_s0')
    decisions = []
    for line in (d / 'decisions.jsonl').open(encoding='utf-8'):
        try: x = json.loads(line)
        except json.JSONDecodeError: continue
        if x.get('support', {}).get('learning_decision'): decisions.append(x)
    counts = collections.Counter(x['support']['candidate_index'] for x in decisions)
    valid = []
    p = d / 'model_prefix_releases.jsonl'
    if p.exists():
        for line in p.open(encoding='utf-8'):
            try: x = json.loads(line)
            except json.JSONDecodeError: continue
            if x['gate']['weight'] > 0:
                f, a = x['payload']['factual'], x['payload']['alternative']
                valid.append({'later': a['tau'] > f['tau'], 'delta': a['return'] - f['return'],
                              'both_terminal': bool(a['done'] and f['done']),
                              'fact_error': abs(f['return']-x['actual_return']), 'weight': x['gate']['weight']})
    audit = json.loads((d / 'learning_audit.json').read_text(encoding='utf-8'))
    shares = [x['applied_gradient_share'] for x in audit if x['phase']=='on' and x['applied_gradient_share']>0]
    out[arm] = {'heartbeat': json.loads((d/'heartbeat.json').read_text()),
        'real_decisions':len(decisions), 'candidate_counts':dict(counts),
        'valid_model_pairs':len(valid),
        'valid_later_better':sum(x['later'] and x['delta']>0 for x in valid),
        'valid_earlier_better':sum(not x['later'] and x['delta']>0 for x in valid),
        'later_vs_earlier_delta_median': statistics.median([x['delta'] if x['later'] else -x['delta'] for x in valid]) if valid else None,
        'valid_both_terminal':sum(x['both_terminal'] for x in valid),
        'accepted_factual_error_max':max([x['fact_error'] for x in valid],default=None),
        'median_nonzero_logged_gradient_share':statistics.median(shares) if shares else None,
        'notes':['Counts are saved real decisions, not environment steps. CF pairs share factual branches; not independent samples.',
                 'Gradient statistics are periodic logged probes, not every historical update.',
                 'A factual gate does not validate the alternative branch against real dynamics.']}
p = r/'记录'/'EXISTING_SIGNAL_DIAGNOSIS.json'
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))
