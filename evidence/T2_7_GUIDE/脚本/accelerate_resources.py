"""Resource priority only, by verified live PID identity; no scientific/observer changes."""
import json,time
import psutil
import runtime as a
R=a.R;state=a.read(R/'RUN_STATE.json');assert state['phase']=='training'
changes=[]
for name,info in state['processes'].items():
 if not name.startswith('train_') or not info['running']:continue
 receipt=a.read(R/f'记录/{name}_LAUNCH.json');root=psutil.Process(info['pid']);assert root.create_time()==receipt['created']
 for p in [root]+root.children(recursive=True):
  try:
   if p.name().lower() not in ('python.exe','pythonw.exe'):continue
   assert p.cpu_affinity()==list(range(12))
   before=int(p.nice());p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
   changes.append({'pid':p.pid,'created':p.create_time(),'before':before,'after':int(p.nice()),'affinity':p.cpu_affinity()})
  except psutil.NoSuchProcess:pass
record={'jst':a.now(),'authorization':'user asked accelerate and ETA; machine idle','resource_only':True,'changes':changes,'training_restarted':False,'worker_count_changed':False,'science_changed':False,'observer_changed':False,'priority':'AboveNormal, existing training only','note':'No CPU reservation; all12 allowed. Higher priority does not guarantee speedup when CPU is already uncontended.'}
a.dump(R/'输入/RESOURCE_ACCELERATION_20261101.json',record)
cur=a.read(a.CURRENT);cur['current_scope']['priority']='AboveNormal for existing training; future eval Normal';cur['resource_acceleration']=str(R/'输入/RESOURCE_ACCELERATION_20261101.json');cur['updated_at']=a.now();a.dump(a.CURRENT,cur)
print(json.dumps({'status':'applied','processes_changed':len(changes),'priority':'AboveNormal'}))
