import csv,io,json
from pathlib import Path
for seed in [262460,262461,262462]:
    raw=(Path('D:/py/DRL2/logs')/f'final2_{seed}'/'train.monitor.csv').read_bytes()
    raw=raw[:raw.rfind(b'\n')+1].decode('utf-8-sig')
    rows=list(csv.DictReader(io.StringIO('\n'.join(x for x in raw.splitlines() if not x.startswith('#')))))
    step=0
    for r in rows:
        step+=int(r['l']);r['step']=step;r['ok']=r['completed'].lower()=='true';r['return']=float(r['r'])
    def stats(a):
        return {'episodes':len(a),'completed':sum(r['ok'] for r in a),'rate_percent':round(100*sum(r['ok'] for r in a)/len(a),2),'mean_return':round(sum(r['return'] for r in a)/len(a),3)} if a else None
    blocks={f'{lo//1000}k-{(lo+5000)//1000}k':stats([r for r in rows if lo<r['step']<=lo+5000]) for lo in range(25000,55000,5000)}
    print(json.dumps({'seed':seed,'episode_end_step':step,'last50':stats(rows[-50:]),'previous50':stats(rows[-100:-50]),'last100':stats(rows[-100:]),'blocks':blocks},ensure_ascii=False))
