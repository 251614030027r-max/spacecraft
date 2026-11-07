import sys,json,collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
by=collections.defaultdict(dict)
for r in rows: by[r['opening']][r['policy']]=r
pols=sys.argv[2].split(',')
for o in sorted(by):
    m=next(iter(by[o].values()))['mission']
    print(o, 'U' if m['urgent'] else 'E', m['deadline_s'], '  '.join('%-7s %6.1f dv%4.2f t%3.0f' % (by[o][p]['event'][:7], by[o][p]['return'], by[o][p]['delta_v'], by[o][p]['time_s']) if p in by[o] else '%-26s'%'-' for p in pols))
done=[o for o in by if all(p in by[o] for p in pols)]
for p in pols:
    rs=[by[o][p] for o in done]
    if rs: print(p, 'n',len(rs),'mean %.1f'%(sum(r['return'] for r in rs)/len(rs)), 'capture', sum(r['event']=='success' for r in rs))
if done:
    import numpy as np
    uni=max(np.mean([by[o][p]['return'] for o in done]) for p in pols); best=np.mean([max(by[o][p]['return'] for p in pols) for o in done])
    print('complete %d: best uniform %.1f, per-opening best %.1f, gap %.1f; best policy per opening:'%(len(done),uni,best,best-uni), collections.Counter(max(pols,key=lambda p:by[o][p]['return']) for o in done))
