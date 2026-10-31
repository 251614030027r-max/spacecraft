from pathlib import Path
import json,statistics as st,collections,hashlib,datetime
R=Path(__file__).resolve().parents[1];O=R/'诊断';O.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def mean(xs):
 xs=[x for x in xs if isinstance(x,(int,float))];return st.mean(xs) if xs else None
def summary(xs):
 xs=[x for x in xs if isinstance(x,(int,float))];return {'n':len(xs),'mean':mean(xs),'median':st.median(xs) if xs else None,'min':min(xs) if xs else None,'max':max(xs) if xs else None}
def readcases(arm,s,k):return {d['opening']:d for f in (R/f'C_eval/{arm}_s{s}/step_{k}').glob('*.json') if not f.name.startswith('ERROR_') for d in [load(f)]}
def trajectory(d):
 ds=d['decisions'];stage=[x for x in ds if x['event'] is None]
 return {'event':d['event'],'return':d['return'],'time_s':d['time_s'],'delta_v':d['delta_v'],'deadline':d['mission']['deadline_s'],'stages':d['stages'],'holds':d['holds'],'rejected':d['rejected'],'entries_lost':d.get('entries_lost'),'macro':d['macro'],'stage_points':[{'t':x['t'],'range':round(x['range_m'],3),'a':[round(y,3) for y in x['a']],'mode':x['mode'],'event':x['event']} for x in ds]}
result={'jst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'scope':'Read-only frozen JSON/source analysis; no new simulation or model forward','pairs':{},'model_phases':{},'training_phases':{},'recurring_openings':{}}
rec=collections.defaultdict(list);allloss=[]
for seed in (0,1):
 for k in (1000,2000,3000,4000):
  a,b=readcases('e_mve',seed,k),readcases('e_sac',seed,k)
  if len(a)!=48 or len(b)!=48:continue
  assert set(a)==set(b)
  rows=[]
  for op in sorted(a):
   x,y=a[op],b[op];assert x['mission']==y['mission'];delta=x['return']-y['return'];cat=('S' if x['event']=='success' else 'F')+('S' if y['event']=='success' else 'F')
   row={'opening':op,'delta':delta,'urgent':x['mission']['urgent'],'deadline':x['mission']['deadline_s'],'category':cat,'mve_event':x['event'],'sac_event':y['event'],'time_diff':x['time_s']-y['time_s'],'dv_diff':x['delta_v']-y['delta_v']}
   rows.append(row);rec[op].append(dict(row,seed=seed,k=k))
   if cat=='FS':allloss.append({'seed':seed,'k':k,'opening':op,'delta':delta,'MVE':trajectory(x),'SAC':trajectory(y)})
  d={'mean_delta':mean([x['delta'] for x in rows]),'events':{'mve':dict(collections.Counter(x['event'] for x in a.values())),'sac':dict(collections.Counter(x['event'] for x in b.values()))},'decomposition':{},'splits':{},'loss_openings':[x for x in rows if x['category']=='FS'],'gain_openings':[x for x in rows if x['category']=='SF']}
  for cat in ('SS','FS','SF','FF'):
   xs=[x for x in rows if x['category']==cat];d['decomposition'][cat]={'n':len(xs),'mean_delta':mean([x['delta'] for x in xs]),'contribution_to_48mean':sum(x['delta'] for x in xs)/48,'time_diff_mean':mean([x['time_diff'] for x in xs]),'dv_diff_mean':mean([x['dv_diff'] for x in xs])}
  for name,pred in {'urgent':lambda x:x['urgent'],'economy':lambda x:not x['urgent'],'high':lambda x:x['urgent'] and x['deadline']<160,'mid':lambda x:x['urgent'] and 160<=x['deadline']<180,'low':lambda x:x['urgent'] and x['deadline']>=180,'old24':lambda x:x['opening']<294024,'new24':lambda x:x['opening']>=294024}.items():
   xs=[x for x in rows if pred(x)];d['splits'][name]={'n':len(xs),'mean_delta':mean([x['delta'] for x in xs]),'losses':sum(x['category']=='FS' for x in xs),'gains':sum(x['category']=='SF' for x in xs)}
  result['pairs'][f's{seed}_{k}']=d
for op,xs in rec.items():result['recurring_openings'][str(op)]={'sum_delta':sum(x['delta'] for x in xs),'loss_count':sum(x['category']=='FS' for x in xs),'gain_count':sum(x['category']=='SF' for x in xs),'observations':xs}
for arm in ('e_mve','e_sac'):
 for seed in (0,1):
  tag=f'{arm}_s{seed}';ms=load(R/f'B/TRAIN_{tag}/model_stats.json');eps=load(R/f'B/TRAIN_{tag}/episodes.json')
  result['model_phases'][tag]={};result['training_phases'][tag]={}
  for lo,hi in ((0,500),(500,1000),(1000,2000),(2000,3000),(3000,4001)):
   rs=[x for x in ms if lo<=x['steps']<hi];ep=[x for x in eps if lo<=x['steps']<hi];m={}
   for field in ('real_td','sigma','model_td','loss_share','grad_cos','entry_bias_now','entry_unanchored_share','abs_err_median','label_age_mean','age_capped_share','q_pi_mean','w2_mean','pi_gap_sigma_median','two_stage_share','decisive_share','stage2_entry_share'):
    m[field]=summary([x.get(field) for x in rs])
   gs=[x['grad_cos'] for x in rs if x.get('grad_cos') is not None];m['negative_grad_share']=sum(x<0 for x in gs)/len(gs) if gs else None
   result['model_phases'][tag][f'{lo}_{hi}']=m
   result['training_phases'][tag][f'{lo}_{hi}']={}
   for name,ss in [('all',ep),('urgent',[x for x in ep if x['mission']['urgent']]),('economy',[x for x in ep if not x['mission']['urgent']])]:
    result['training_phases'][tag][f'{lo}_{hi}'][name]={'n':len(ss),'capture_rate':mean([x['event']=='success' for x in ss]),'events':dict(collections.Counter(x['event'] for x in ss)),'entry_bias':summary([x.get('entry_bias') for x in ss])}
result['lost_trajectories']=allloss
result['source_hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path('D:/py/DRL2/env/t2_stage.py'),Path('D:/py/DRL2/train/t2_mve_sac.py')]}
(O/'DIAGNOSIS_DATA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE PAIRS')
for name,d in result['pairs'].items():print(name,json.dumps({x:d[x] for x in ('mean_delta','events','decomposition','splits')},ensure_ascii=False))
print('TOP RECURRING LOSSES')
for op,d in sorted(result['recurring_openings'].items(),key=lambda x:x[1]['sum_delta'])[:10]:print(op,json.dumps({k:v for k,v in d.items() if k!='observations'}))
print('MODEL PHASES COMPACT')
for tag,phases in result['model_phases'].items():
 for phase,d in phases.items():print(tag,phase,json.dumps({k:(v if not isinstance(v,dict) else {'n':v['n'],'mean':v['mean']}) for k,v in d.items() if k in ('sigma','loss_share','grad_cos','negative_grad_share','entry_bias_now','label_age_mean','age_capped_share','q_pi_mean','entry_unanchored_share')},ensure_ascii=False))
print('TRAIN CAPTURE')
for tag,phases in result['training_phases'].items():
 print(tag,json.dumps({phase:{name:v['capture_rate'] for name,v in d.items()} for phase,d in phases.items()}))
print('LOST TRACE EXAMPLES')
for x in sorted(allloss,key=lambda x:x['delta'])[:4]:print(json.dumps(x,ensure_ascii=False))
