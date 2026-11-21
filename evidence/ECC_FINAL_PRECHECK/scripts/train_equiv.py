"""Training-level equivalence: two smoke runs (same config, seed, steps, 1 worker) from a reference checkout and the
working tree -- policy parameters of every checkpoint, episodes, training statistics, manifest differences.
Usage: python -B train_equiv.py OLD_RUN NEW_RUN"""
import json, sys, zipfile, io
from pathlib import Path
import torch

def params(z):
    with zipfile.ZipFile(z) as f:
        return torch.load(io.BytesIO(f.read('policy.pth')), map_location='cpu', weights_only=True)

old, new = Path(sys.argv[1]), Path(sys.argv[2])
rep = {}
for ck in sorted((old / 'checkpoints').glob('*.zip')) + [old / 'final.zip']:
    rel = ck.relative_to(old); a, b = params(ck), params(new / rel)
    rep[str(rel)] = {'tensors': len(a), 'bitwise_identical': a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)}
ea, eb = json.loads((old / 'episodes.json').read_text()), json.loads((new / 'episodes.json').read_text())
rep['episodes'] = {'n': len(ea), 'identical': ea == eb}
sa, sb = json.loads((old / 'model_stats.json').read_text()), json.loads((new / 'model_stats.json').read_text())
rep['model_stats_identical'] = sa == sb
ma, mb = json.loads((old / 'manifest.json').read_text()), json.loads((new / 'manifest.json').read_text())
rep['manifest_diffs'] = {k: (ma.get(k), mb.get(k)) for k in sorted(set(ma) | set(mb)) if ma.get(k) != mb.get(k) and k not in ('output',)}
print(json.dumps(rep, indent=1))
