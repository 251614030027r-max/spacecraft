"""python -B compare.py REF.json [REF.json ...] -- NEW.json [NEW.json ...]: row-by-row comparison of env_equiv.py outputs."""
import json, sys
i = sys.argv.index('--')
ref, new = {}, {}
for f in sys.argv[1:i]: ref.update(json.load(open(f)))
for f in sys.argv[i + 1:]: new.update(json.load(open(f)))
out = {'episodes': {}, 'all_identical': True}
kinds = {'keep': 0, 'correct': 0, 'launch': 0, 'none': 0}
for k in sorted(ref):
    a, b = ref[k], new.get(k)
    first = None
    if a != b:
        for j, (x, y) in enumerate(zip(a, b or [])):
            if x != y:
                first = {'row': j, 'ref': x, 'new': y}; break
        if first is None:
            first = {'rows_ref': len(a), 'rows_new': len(b or [])}
    for row in a[2:-2]:
        c = json.loads(row[10])
        if c: kinds[c['kind']] += 1
    out['episodes'][k] = {'identical': a == b, 'decisions': len(a) - 4, 'final_event': a[-3][3], 'first_difference': first}
    out['all_identical'] &= a == b
out['coordination_decisions_by_kind'] = kinds
print(json.dumps(out, indent=1))
