import json, glob, sys, collections
R = sys.argv[1]
rec = {}
for f in glob.glob(R + '/*.json'):
    r = json.load(open(f)); rec[(r['policy'], r['certify'], r['opening'])] = r
for pol in ('c_10_0', 'rally'):
    pairs = [(rec[(pol, False, o)], rec[(pol, True, o)]) for (p, c, o) in rec if p == pol and c and (pol, False, o) in rec]
    if not pairs: continue
    b = sum(x['event'] == 'success' for x, _ in pairs); c = sum(y['event'] == 'success' for _, y in pairs)
    d = [y['return'] - x['return'] for x, y in pairs]
    unsafe = sum(y['event'] == 'unsafe' for _, y in pairs); iv = sum(y['cert_interventions'] for _, y in pairs)
    flips = [(x['opening'], x['event'], y['event']) for x, y in pairs if x['event'] != y['event']]
    print(pol, 'urgent pairs', len(pairs), 'captures base/cert', b, c, 'mean diff %.1f' % (sum(d) / len(d)), 'unsafe', unsafe, 'interventions', iv, 'flips', flips)
eco = [r for (p, c, o), r in rec.items() if p == 'c_10_0' and c and not r['mission']['urgent']]
print('economy cert runs', len(eco), 'interventions', sum(r['cert_interventions'] for r in eco), 'events', collections.Counter(r['event'] for r in eco))
