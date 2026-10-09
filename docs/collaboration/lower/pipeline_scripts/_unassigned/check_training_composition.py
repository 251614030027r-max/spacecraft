import csv,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'reports'/'ppt_20261009'/'plot_data';out={}
for seed in (262460,262461,262462):
    with (root/f'train_{seed}_snapshot.csv').open(encoding='utf-8-sig',newline='') as f:f.readline();rows=list(csv.DictReader(f))
    step=0
    for r in rows:step+=int(r['l']);r['step']=step
    stats=[]
    for lo,hi in ((0,10000),(20000,30000),(40000,50000)):
        batch=[r for r in rows if lo<r['step']<=hi];handoff=[r for r in batch if r['handoff']=='True'];solo=[r for r in batch if r['handoff']!='True']
        rate=lambda group:round(100*sum(r['completed']=='True' for r in group)/len(group),1) if group else None
        stats.append({'steps':[lo,hi],'episodes':len(batch),'completion_percent':rate(batch),'handoff_episode_percent':round(100*len(handoff)/len(batch),1),'completion_if_handoff_percent':rate(handoff),'completion_without_handoff_percent':rate(solo),'mean_return':round(sum(float(r['r']) for r in batch)/len(batch),3)})
    out[seed]=stats
print(json.dumps(out,ensure_ascii=False,indent=2))
