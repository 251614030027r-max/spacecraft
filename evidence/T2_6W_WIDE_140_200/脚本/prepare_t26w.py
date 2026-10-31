"""Frozen SHA, actual pytest exit and deadline smoke admission; never formal training."""
from pathlib import Path
import subprocess,os,json,time,datetime,re,traceback,psutil
R=Path(__file__).resolve().parents[1];REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe');SHA='e1166eb5daee069e7f14b6f48d24dcf43c36d66f'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def dump(name,data):(R/'记录'/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def run(name,args):
 with (R/f'记录/{name}.out.log').open('ab') as fo,(R/f'记录/{name}.err.log').open('ab') as fe:
  p=subprocess.Popen([PY,'-B',*map(str,args)],cwd=REPO,stdout=fo,stderr=fe,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
  dump(name+'_LAUNCH.json',{'jst':now(),'pid':p.pid,'created':psutil.Process(p.pid).create_time(),'command':[PY,'-B',*map(str,args)]});rc=p.wait();dump(name+'_EXIT.json',{'jst':now(),'returncode':rc});assert rc==0,name+' failed; original stderr preserved'
def main():
 p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);anc=[{'pid':q.pid,'created':q.create_time(),'name':q.name()} for q in p.parents()];assert not any('codex' in q['name'].lower() or 'chatgpt' in q['name'].lower() for q in anc)
 with (R/'记录/PREPARE.lock').open('x') as f:f.write(str(p.pid))
 dump('PREPARE_LAUNCH.json',{'jst':now(),'pid':p.pid,'created':p.create_time(),'ancestors':anc})
 git=['git','-c','safe.directory='+str(REPO),'-C',str(REPO)];assert subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()==SHA and not subprocess.check_output([*git,'status','--porcelain'],text=True).strip()
 run('pytest',['-m','pytest','-q','tests/test_t2_stage.py','-o','cache_dir='+str(R/'记录/pytest_cache'),'--basetemp',R/'记录/pytest_tmp'])
 raw=(R/'记录/pytest.out.log').read_bytes();(R/'pytest.log').write_bytes(raw);m=re.search(rb'(\d+) passed',raw);assert m and int(m[1])>=44
 dump('PYTEST_VERIFIED.json',{'jst':now(),'status':'passed','expected':44,'actual_passed':int(m[1]),'science_SHA':SHA,'native_returncode':0})
 run('deadline_smoke',['-m','experiments.t2','screen','--openings','294017','--policies','c_10_0','--urgent-deadline','140,200','--workers','1','--output',R/'smoke'])
 smoke=json.loads((R/'smoke/c_10_0_294017.json').read_text(encoding='utf-8'));assert smoke['mission']['urgent'] is True and abs(smoke['mission']['deadline_s']-164.)<1e-9
 dump('PREPARE_VERIFIED.json',{'jst':now(),'status':'passed','science_SHA':SHA,'pytest_passed':int(m[1]),'opening':294017,'mission':smoke['mission'],'not_a_training_or_performance_gate':True})
if __name__=='__main__':
 try:main()
 except Exception:dump('PREPARE_ERROR.json',{'jst':now(),'traceback':traceback.format_exc()});raise
