"""Fixed 30k structural evaluation only; never restart or terminate training."""
import hashlib,json,os,shutil,subprocess,sys,time,zipfile
from datetime import datetime
from pathlib import Path
MAIN=Path('D:/py/DRL2');PY=MAIN/'.venv/Scripts/python.exe'
REC=Path('C:/Users/35884/Documents/Spacecraft/过程文件/最终主线重训/记录')
DATA=MAIN/'eval/final2/devcheck';STATE=REC/'DEV30_EXECUTION.json'
COMMIT='f2f8169acd580ea96a578d45022dd8952447eca5';SEEDS=(262460,262461,262462)
ENV=os.environ.copy();ENV.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(MAIN),PYTHONIOENCODING='utf-8')
S={'phase':'waiting_for_checkpoints','workers':{},'checkpoint_sha256':{},'counts':{},'training_untouched':True}
def now():return datetime.now().astimezone().isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save():
    S.update(updated_at=now(),coordinator_pid=os.getpid());temp=STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(S,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,STATE)
def frozen():
    def git(*args):return subprocess.check_output(['git','-c',f'safe.directory={MAIN}','-C',str(MAIN),*args],encoding='utf-8').strip()
    assert git('rev-parse','HEAD')==COMMIT,'Science commit changed'
    assert not git('status','--porcelain','--untracked-files=no'),'Tracked code dirty'
def spawn(seed,args,label):
    out=REC/f'dev30_{seed}_{label}.stdout.log';err=REC/f'dev30_{seed}_{label}.stderr.log'
    with out.open('xb') as o,err.open('xb') as e:
        p=subprocess.Popen([str(PY),'-u','-B','-m','experiments.v3_stopping',*map(str,args)],cwd=MAIN,env=ENV,stdout=o,stderr=e,creationflags=0x08004000)
    return p
def run():
    if STATE.exists():raise RuntimeError('Existing DEV30 state; do not duplicate')
    if DATA.exists() and any(DATA.rglob('seed_*.json')):raise RuntimeError('Existing evaluation results; inspect before continuing')
    frozen();save();jobs={};stable={};checks={}
    try:
        while len(checks)<3:
            frozen()
            for s in SEEDS:
                run=MAIN/f'logs/final2_{s}';cp=run/'checkpoints/stopping_30000_outer_decisions.zip'
                if s not in jobs and cp.exists():
                    stat=(cp.stat().st_size,cp.stat().st_mtime_ns)
                    if stable.get(s)==stat:
                        with zipfile.ZipFile(cp) as z:assert z.testzip() is None,'Checkpoint CRC failed'
                        m=read(run/'manifest.json')
                        assert m['code_commit']==COMMIT and not m['code_dirty'] and m['stopping']['stop_rule']=='value' and m['stopping']['bellman_stop_value']=='soft'
                        S['checkpoint_sha256'][str(s)]=sha(cp)
                        p=spawn(s,['evaluate','--run-dir',f'logs/final2_{s}','--model-name','checkpoints/stopping_30000_outer_decisions.zip','--allow-incomplete-run','--row','stopping','--seeds','266000-266047','--output-dir',DATA/str(s)],'evaluate')
                        jobs[s]=p;S['workers'][str(s)]={'pid':p.pid,'started_at':now(),'exit_code':None};S['phase']='structural_evaluation_running'
                    stable[s]=stat
                if s in jobs and s not in checks:
                    code=jobs[s].poll();S['counts'][str(s)]=len(list((DATA/str(s)).glob('seed_*.json')))
                    if code is not None:
                        S['workers'][str(s)]['exit_code']=code
                        if code:raise RuntimeError(f'Evaluation failed: {s}; preserve locks and logs')
                        d=DATA/str(s)
                        assert {p.name for p in d.glob('seed_*.json')}=={f'seed_{i}.json' for i in range(266000,266048)} and not list(d.glob('*.lock')) and not list(d.glob('*.tmp'))
                        assert sha(cp)==S['checkpoint_sha256'][str(s)]
                        shutil.copy2(run/'train.monitor.csv',REC/f'dev30_{s}_monitor_before.csv')
                        p=spawn(s,['devcheck','--run-dir',f'logs/final2_{s}','--eval-dir',d,'--output',DATA/f'devcheck_{s}.json'],'official')
                        if p.wait()!=0:raise RuntimeError(f'Official devcheck failed: {s}')
                        shutil.copy2(run/'train.monitor.csv',REC/f'dev30_{s}_monitor_after.csv')
                        checks[s]=read(DATA/f'devcheck_{s}.json');S.setdefault('devchecks',{})[str(s)]=checks[s]
            save()
            if len(checks)<3:time.sleep(300 if jobs else 10)
        keys=('D1_not_collapsed','D2_learned_reaches_mid_late','D3_state_dependent_stopping_value')
        d4=all(r['D4_stopping_matches_value_rule']['pass'] for r in checks.values())
        n=sum(all(r[k]['pass'] for k in keys) for r in checks.values())
        S['gate']={'all_D4':d4,'D1_D2_D3_passing_models':n,'passed':d4 and n>=2}
        S['phase']='structure_pass_training_continues' if d4 and n>=2 else 'structure_fail_requires_verified_pid_stop'
        save();print(S['phase'],flush=True)
    except Exception as e:S['phase']='evaluation_fault';S['error']=repr(e);save();raise
if __name__=='__main__':run()
