from pathlib import Path
import csv,json,numpy as np
root=Path(__file__).resolve().parents[1]/'reports'/'ppt_20261009';data=root/'plot_data';out={}
for seed in (262460,262461,262462):
    with (data/f'train_{seed}_snapshot.csv').open(encoding='utf-8-sig',newline='') as f:f.readline();rows=list(csv.DictReader(f))
    x=np.cumsum([int(r['l']) for r in rows]);bins=[]
    assert all(int(r['l'])==int(r['learned_decisions'])+int(r['handoff']=='True') for r in rows)
    with (data/f'train_{seed}_plotted.csv').open(encoding='utf-8-sig',newline='') as f:plotted=list(csv.DictReader(f))
    assert len(plotted)==len(rows)
    for i,r in enumerate(plotted):
        assert int(r['outer_decisions'])==x[i] and float(r['return'])==float(rows[i]['r'])
        if i>=49:
            assert np.isclose(float(r['return_mean50']),np.mean([float(z['r']) for z in rows[i-49:i+1]]))
            assert np.isclose(float(r['completed_mean50_percent']),100*np.mean([z['completed']=='True' for z in rows[i-49:i+1]]))
    for lo,hi in ((0,10000),(10000,20000),(20000,30000),(30000,40000),(40000,60000)):
        selected=[r for r,k in zip(rows,x) if lo<k<=hi]
        group={}
        for label,rs in [('all',selected),('handoff',[r for r in selected if r['handoff']=='True']),('no_handoff',[r for r in selected if r['handoff']!='True'])]:
            group[label]={'episodes':len(rs),'completed':sum(r['completed']=='True' for r in rs),'completed_percent':100*np.mean([r['completed']=='True' for r in rs]) if rs else None,'mean_return':float(np.mean([float(r['r']) for r in rs])) if rs else None,'mean_outer_length':float(np.mean([int(r['l']) for r in rs])) if rs else None}
        group['range']=[lo,min(hi,int(x[-1]))];bins.append(group)
    out[str(seed)]=bins
(data/'TRAINING_INTERPRETATION_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for seed,bins in out.items():
    print(seed)
    print('TOTALS',json.dumps({k:{field:sum(r[k][field] for r in bins) for field in ('episodes','completed')} for k in ('all','handoff','no_handoff')}))
    for r in bins:print(r['range'],'n=',r['all']['episodes'],'completed=',round(r['all']['completed_percent'],1),'handoff=',r['handoff']['episodes'],'handoff_success=',round(r['handoff']['completed_percent'],1) if r['handoff']['episodes'] else None,'no_handoff_success=',round(r['no_handoff']['completed_percent'],1) if r['no_handoff']['episodes'] else None,'return=',round(r['all']['mean_return'],2))
