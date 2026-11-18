"""Evaluation-only probe: an existing screened policy with the long-arrival entry candidate on (``T2StageEnv.long_entry``).

    python -B -m experiments.ecc_le_probe --policy RUN_DIR|FIXED --label L --openings A-B --output DIR [--workers W] [--off]

Urgent mission, deterministic actor, certifier on (as the policy was trained or, for a fixed intent, on). ``--off`` runs
the same episodes without the long-arrival entry (pairing check). One JSON per episode and SUMMARY.json
(outcomes, captures, long entries used). No training, no change to the frozen paths (the flag defaults to off).
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def job(j):
    import torch; torch.set_num_threads(1)
    from experiments.ecc2_gate0 import _actor, _env
    from experiments.t2 import dump, run_episode
    f = Path(j['out']) / f"{j['label']}_{j['opening']}.json"
    if f.exists():
        return json.loads(f.read_text(encoding='utf-8'))
    try:
        cfg, act = _actor(j['policy'])
        env = _env(j['opening'], cfg)
        env.audit_logging = False
        env.long_entry = not j['off']
        res = run_episode(env, j['opening'], act)
        res.update(label=j['label'], long_entries=env.stats.get('long_entries', 0), cert_interventions=env.stats.get('cert_interventions', 0),
                   entries_lost=env.stats.get('entries_lost', 0), long_entry_on=not j['off'])
    except Exception as error:  # noqa: BLE001 -- recorded
        import traceback
        res = {'opening': j['opening'], 'label': j['label'], 'event': 'error', 'error': repr(error), 'traceback': traceback.format_exc()}
    dump(f, res)
    return res


def main():
    from experiments.final import block
    ap = argparse.ArgumentParser(); ap.add_argument('--policy', required=True); ap.add_argument('--label', required=True)
    ap.add_argument('--openings', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--off', action='store_true')
    a = ap.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(policy=a.policy, label=a.label, opening=o, out=str(out), off=a.off) for o in block(a.openings)]
    res = []
    with ProcessPoolExecutor(a.workers) as pool:
        for f in as_completed([pool.submit(job, x) for x in jobs]):
            r = f.result(); res.append(r)
            print(json.dumps({k: r.get(k) for k in ('opening', 'event', 'return', 'long_entries', 'error')}), flush=True)
    s = {'label': a.label, 'policy': a.policy, 'long_entry_on': not a.off, 'n': len(res), 'events': dict(Counter(r['event'] for r in res)),
         'captures': sum(r['event'] == 'success' for r in res), 'mean_return': float(np.mean([r['return'] for r in res if 'return' in r])),
         'episodes_with_long_entry': sum(bool(r.get('long_entries')) for r in res),
         'captures_with_long_entry': sum(bool(r.get('long_entries')) and r['event'] == 'success' for r in res)}
    (out / f'SUMMARY_{a.label}.json').write_text(json.dumps(s, indent=1), encoding='utf-8'); print(json.dumps(s))


if __name__ == '__main__':
    main()
