"""python -B compare_eval.py REF_OUT NEW_OUT: per-decision comparison of eval_equiv.py outputs on the kept fields."""
import json, sys
from pathlib import Path

DROPPED = {'model', 'govern', 'gov_replacements', 'gov_would_replace', 'gov_decisions', 'gov_wall_s', 'wall_s', 'cert_wall_s_total'}
ref, new = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
for f in sorted(ref.glob('E_*.json')):
    a, b = json.loads(f.read_text()), json.loads((new / f.name).read_text())
    def strip(x):
        x = {k: v for k, v in x.items() if k not in DROPPED}
        x['decisions'] = [{k: ({kk: vv for kk, vv in v.items() if kk != 'wall_s'} if k == 'cert' else v)
                           for k, v in d.items() if k not in DROPPED} for d in x['decisions']]
        return x
    sa, sb = strip(a), strip(b)
    res[f.name] = {'identical_on_kept_fields': sa == sb, 'event': a['event'], 'decisions': len(a['decisions']),
                   'keys_only_in_ref': sorted(set(a) - set(b)), 'keys_only_in_new': sorted(set(b) - set(a))}
print(json.dumps(res, indent=1))
