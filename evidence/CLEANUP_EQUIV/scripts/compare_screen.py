import json, sys
from pathlib import Path
DROP = {'wall_s', 'cert_wall_s_total', 'model'}
ref, new = Path(sys.argv[1]), Path(sys.argv[2]); out = {}
for f in sorted(ref.rglob('*.json')):
    a, b = json.loads(f.read_text()), json.loads((new / f.relative_to(ref)).read_text())
    s = lambda x: {k: ([{kk: vv for kk, vv in d.items() if kk not in DROP} for d in v] if k == 'decisions' else v) for k, v in x.items() if k not in DROP}
    out[str(f.relative_to(ref))] = {'identical_on_kept_fields': s(a) == s(b), 'event': a['event'], 'only_ref': sorted(set(a) - set(b)), 'only_new': sorted(set(b) - set(a))}
print(json.dumps(out, indent=1))
