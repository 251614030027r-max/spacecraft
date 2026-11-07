"""User 10-30: machine idle; normal process priority and all available CPUs. No scientific settings changed."""
import os, json, time, traceback
from pathlib import Path
import psutil
R=Path(__file__).resolve().parents[1]
def read(p):
 try:return json.loads(p.read_text(encoding='utf-8'))
 except (FileNotFoundError,json.JSONDecodeError):return {}
def main():
 owner=read(R/'记录/OS_ANCESTRY.json');root=psutil.Process(owner['pid']);assert root.create_time()==owner['created']
 me=psutil.Process();assert not any('codex' in p.name().lower() or 'chatgpt' in p.name().lower() for p in me.parents())
 with (R/'记录/RESOURCE_CHANGES.jsonl').open('a',encoding='utf-8',buffering=1) as out:
  while True:
   if read(R/'RUN_STATE.json').get('phase') in ('failed_stopped','complete_delivered'):break
   try:
    root=psutil.Process(owner['pid'])
    if root.create_time()!=owner['created']:break
    for p in [root]+root.children(recursive=True):
     try:
      if p.nice()!=psutil.NORMAL_PRIORITY_CLASS:
       before=int(p.nice());p.nice(psutil.NORMAL_PRIORITY_CLASS)
       out.write(json.dumps({'time':time.time(),'pid':p.pid,'created':p.create_time(),'from':before,'to':'NORMAL','affinity':p.cpu_affinity(),'science_workers_unchanged':True})+'\n')
     except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   except psutil.NoSuchProcess:break
   time.sleep(60)
 (R/'记录/RESOURCE_EXIT.json').write_text(json.dumps({'returncode':0,'time':time.time()}),encoding='utf-8')
if __name__=='__main__':
 try:main()
 except Exception:(R/'记录/RESOURCE_ERROR.json').write_text(traceback.format_exc(),encoding='utf-8');raise
