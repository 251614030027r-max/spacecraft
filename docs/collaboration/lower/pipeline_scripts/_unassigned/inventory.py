from pathlib import Path
import json,hashlib,csv,numpy as np
top=Path(__file__).resolve().parents[1];(top/'记录').mkdir(exist_ok=True);(top/'reports').mkdir(exist_ok=True)
main=Path('D:/py/DRL2');read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));out={}
audit=read(Path('D:/py/DRL2_v3e/eval/v3e/final_training_audit_60k.json'));out['training_audit_bins']={r['seed']:[b['completion_rate'] for b in r['bins_by_episode_start']] for r in audit['runs']}
formal={}
for row,seed in [('pure',None)]+[(r,s) for r in ('learned','stopping') for s in (262430,262431,262432)]:
    path=main/'eval/stopping/formal'/row
    if seed:path=path/str(seed)
    rs=[read(path/f'seed_{k}.json') for k in range(267000,267048)]
    formal[row+('_'+str(seed) if seed else '')]={'count':len(rs),'clean':sum(r['clean_completion'] for r in rs),'violation':sum(not r['zero_violation'] for r in rs),'model_sha':list(set(r['model_sha256'] for r in rs))}
out['formal']=formal
screen={}
for row in ('pure','nominal'):
    rs=[read(main/f'eval/regime_screen/w2.36_r15/{row}/seed_{k}.json') for k in range(269000,269048)]
    screen[row]={'count':len(rs),'clean':sum(r['clean_completion'] for r in rs),'median_time':float(np.median([r['survival_s'] for r in rs if r['clean_completion']])),'failure_kinds':sorted(set(tuple(r['failure']) for r in rs if not r['clean_completion']))}
out['screen']=screen
b=read(main/'eval/v3e/stage_b/readout_b1b2.json');out['b2']={s:r['summary'] for s,r in b['b2'].items() if isinstance(r,dict) and 'summary' in r};out['b_keys']=list(b)
pure=read(main/'eval/v3e/pure_mpc.json');out['v3e_pure_type']=str(type(pure));out['v3e_pure_keys']=list(pure)[:10] if isinstance(pure,dict) else len(pure)
out['formal_new_exists']=(main/'eval/final2/formal').exists()
(top/'记录/INVENTORY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
