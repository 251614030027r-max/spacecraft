import sys, json, hashlib, zipfile, io
import torch as th
def params(z):
    with zipfile.ZipFile(z) as f:
        sd = th.load(io.BytesIO(f.read('policy.pth')), map_location='cpu')
    return {k: v for k, v in sd.items()}
for name in ('checkpoints/step_20.zip', 'final.zip'):
    a, b = params(f'{sys.argv[1]}/{name}'), params(f'{sys.argv[2]}/{name}')
    same = a.keys() == b.keys() and all(th.equal(a[k], b[k]) for k in a)
    print(name, 'policy parameters bitwise identical' if same else 'DIFFERENT', len(a))
ea, eb = (json.load(open(f'{d}/episodes.json')) for d in sys.argv[1:3])
strip = lambda e: [{k: x.get(k) for k in ('steps', 'event', 'time_s', 'delta_v', 'return_true')} for x in e]
print('episodes identical' if strip(ea) == strip(eb) else 'episodes DIFFERENT', len(ea))
sa, sb = (json.load(open(f'{d}/model_stats.json')) for d in sys.argv[1:3])
keys = ('steps', 'update', 'real_td', 'sigma', 'actor_loss', 'q_pi_mean')
print('training statistics identical' if [{k: x.get(k) for k in keys} for x in sa] == [{k: x.get(k) for k in keys} for x in sb] else 'stats DIFFERENT', len(sa))
xa = [x.get('exec_stats') for x in ea]; xb = [x.get('exec_stats') for x in eb]
print('exec_stats identical' if xa == xb else 'exec_stats DIFFERENT')
ma, mb = (json.load(open(f'{d}/manifest.json')) for d in sys.argv[1:3])
print('manifest diffs', {k: (ma.get(k), mb.get(k)) for k in set(ma) | set(mb) if ma.get(k) != mb.get(k)})
