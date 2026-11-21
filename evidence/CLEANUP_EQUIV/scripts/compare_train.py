import json, sys, zipfile, io
from pathlib import Path
import torch
a, b = Path(sys.argv[1]), Path(sys.argv[2])
def tensors(z):
    with zipfile.ZipFile(z) as f:
        out = {}
        for n in f.namelist():
            if n.endswith('.pth'):
                sd = torch.load(io.BytesIO(f.read(n)), map_location='cpu', weights_only=False)
                for k, v in (sd.items() if isinstance(sd, dict) else []):
                    if torch.is_tensor(v): out[f'{n}:{k}'] = v
                    elif isinstance(v, dict):
                        for k2, v2 in v.items():
                            if torch.is_tensor(v2): out[f'{n}:{k}.{k2}'] = v2
        return out
res = {}
for ck in ['checkpoints/step_0.zip', 'checkpoints/step_20.zip', 'checkpoints/step_40.zip', 'final.zip']:
    ta, tb = tensors(a / ck), tensors(b / ck)
    res[ck] = {'tensors': len(ta), 'same_keys': set(ta) == set(tb),
               'bitwise_identical': set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)}
ea, eb = json.loads((a / 'episodes.json').read_text()), json.loads((b / 'episodes.json').read_text())
common = lambda e, f: [{k: x[k] for k in x if k in f} for x in e]
keys = set(eb[0]) if eb else set()
res['episodes'] = {'n': len(ea), 'identical_on_kept_fields': common(ea, keys) == eb, 'dropped_fields': sorted(set(ea[0]) - keys) if ea else []}
ma, mb = json.loads((a / 'model_stats.json').read_text()), json.loads((b / 'model_stats.json').read_text())
mk = set(mb[0]) if mb else set()
res['model_stats'] = {'n': len(ma), 'identical_on_kept_fields': common(ma, mk) == mb and len(ma) == len(mb),
                      'dropped_fields': sorted(set(ma[0]) - mk) if ma else []}
fa, fb = json.loads((a / 'finished.json').read_text()), json.loads((b / 'finished.json').read_text())
res['finished'] = {k: fa[k] == fb[k] for k in ('steps', 'updates', 'worker_order')}
print(json.dumps(res))
