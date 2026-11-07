"""T2.12 self-check: the e_sac arm (reduced_priv) trained at the T2.11 code and at the T2.12 code, same seed, one worker
(deterministic order) -- policy parameters, episodes (without T2.12's added return fields), model readings, update order.

    (old checkout)  python -B -m experiments.t2 train --arm e_sac --obs reduced_priv --seed 0 --steps 30 --workers 1 --checkpoints 30 --smoke --urgent-deadline 140,200 --output OLD
    (new checkout)  python -B -m experiments.t2 train ... --output NEW
    python -B evidence/T2_12_PRECHECK/脚本/sac_identity.py OLD NEW OUT.json
"""
import io, json, sys, zipfile
import torch
old, new, out = sys.argv[1], sys.argv[2], sys.argv[3]
res = {}
a = json.load(open(f'{old}/episodes.json')); b = json.load(open(f'{new}/episodes.json'))
res['episodes_equal'] = a == [{k: v for k, v in e.items() if k not in ('return_true', 'return_shaped')} for e in b]
res['episodes'] = len(a)
res['model_stats_equal'] = json.load(open(f'{old}/model_stats.json')) == json.load(open(f'{new}/model_stats.json'))
za, zb = zipfile.ZipFile(f'{old}/final.zip'), zipfile.ZipFile(f'{new}/final.zip')
pa = torch.load(io.BytesIO(za.read('policy.pth')), weights_only=False); pb = torch.load(io.BytesIO(zb.read('policy.pth')), weights_only=False)
res['policy_tensors'] = len(pa); res['policy_equal'] = all(torch.equal(pa[k], pb[k]) for k in pa)
fa, fb = json.load(open(f'{old}/finished.json')), json.load(open(f'{new}/finished.json'))
res['worker_order_equal'] = fa['worker_order'] == fb['worker_order']; res['updates'] = [fa['updates'], fb['updates']]
res['heads'] = [json.load(open(f'{old}/manifest.json'))['git_head'], json.load(open(f'{new}/manifest.json'))['git_head']]
res['versions'] = [json.load(open(f'{d}/manifest.json'))['learner_version'] for d in (old, new)]
res['return_true_rebuilt'] = [(e['return_true'], e['mission']['time_cost_per_s'], e['time_s'], e['mission']['fuel_cost_per_m_s'], e['delta_v'], e['event']) for e in b]
json.dump(res, open(out, 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k != 'return_true_rebuilt'}, indent=1))
