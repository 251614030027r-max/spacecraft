"""User-requested resource-only boost; no science or observer changes."""
from pathlib import Path
import psutil,json,time
R=Path(__file__).resolve().parents[1]
def main():
 state=json.loads((R/'RUN_STATE.json').read_text(encoding='utf-8'));assert state['phase']=='training'
 owner=json.loads((R/'记录/OS_ANCESTRY.json').read_text(encoding='utf-8'));root=psutil.Process(owner['pid']);assert root.create_time()==owner['created']
 changes=[];workers=[];before={}
 for name,v in state['processes'].items():
  if not name.startswith('train_'):continue
  p=psutil.Process(v['pid']);assert p.ppid()==root.pid
  for q in [p]+p.children(recursive=True):
   try:
    old={'priority':int(q.nice()),'affinity':q.cpu_affinity()};q.cpu_affinity(list(range(12)));q.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    changes.append({'pid':q.pid,'created':q.create_time(),'training':name,'before':old,'after':{'priority':'AboveNormal','affinity':q.cpu_affinity()},'rss_MiB':q.memory_info().rss/2**20,'os_threads':q.num_threads()});before[q.pid]=sum(q.cpu_times()[:2]);workers.append(q)
   except psutil.NoSuchProcess:pass
 time.sleep(3)
 activity=[]
 for q in workers:
  try:activity.append({'pid':q.pid,'cpu_seconds_over3s':sum(q.cpu_times()[:2])-before[q.pid]})
  except psutil.NoSuchProcess:pass
 out={'timestamp':time.time(),'user_request':'Accelerate, use all CPU','scientific_config_unchanged':True,'workers_per_training':2,'models_restarted':False,'observer_unchanged':True,'changes':changes,'activity':activity,'available_memory_GiB':psutil.virtual_memory().available/2**30}
 (R/'记录/RESOURCE_BOOST_20261101.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=True))
if __name__=='__main__':main()
