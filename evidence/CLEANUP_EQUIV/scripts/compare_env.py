"""python -B compare_env.py REF_DIR NEW_DIR: compare env_<side>_<case>.json files produced by env_equiv.py."""
import json, sys
from pathlib import Path

CASES = ('ecc_urgent', 'each_urgent', 'nogeo_economy', 'rigid_urgent', 'ecc_natural', 'ecc_resource', 'ecc_lowfuel')
ref, new = Path(sys.argv[1]), Path(sys.argv[2])
out = {}
for c in CASES:
    a, b = json.loads((ref / f'env_ref_{c}.json').read_text()), json.loads((new / f'env_new_{c}.json').read_text())
    first = None
    if a != b:
        for k in a:
            for i, (x, y) in enumerate(zip(a[k], b.get(k, []))):
                if x != y:
                    first = {'episode': k, 'row': i, 'ref': x, 'new': y}; break
            if first: break
    out[c] = {'identical': a == b, 'rows': sum(len(v) for v in a.values()),
              'final_events': [v[-3][3] if isinstance(v[-3], list) else None for v in a.values()],
              'cert_interventions': [json.loads(v[-2]).get('cert_interventions') for v in a.values()], 'first_difference': first}
print(json.dumps(out, indent=1))
