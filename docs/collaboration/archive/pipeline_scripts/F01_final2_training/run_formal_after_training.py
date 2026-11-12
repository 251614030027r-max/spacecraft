"""Frozen run-order orchestration. Never start, restart, or stop training."""
import ctypes,hashlib,json,os,subprocess,time,zipfile
from datetime import datetime
from pathlib import Path
MAIN=Path('D:/py/DRL2');PY=MAIN/'.venv/Scripts/python.exe'
REC=Path(__file__).resolve().parents[1]/'记录'
STATE=REC/'FORMAL_EXECUTION.json';E=MAIN/'eval/final2/formal';R=MAIN/'eval/final2/replay'
COMMIT='f2f8169acd580ea96a578d45022dd8952447eca5'
SEEDS=(262460,262461,262462);TRAIN_PIDS={262460:20820,262461:16908,262462:8364}
ENV=os.environ.copy();ENV.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(MAIN),PYTHONIOENCODING='utf-8')
S={'phase':'waiting_for_all_training','workers':{},'training_untouched':True,'science_commit':COMMIT}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def now():return datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save():
    S.update(updated_at=now(),supervisor_pid=os.getpid());tmp=STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(S,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,STATE)
def phase(p):S['phase']=p;save();print(now(),p,flush=True)
def alive(pid):
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.restype=ctypes.c_void_p;api.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    api.CloseHandle.argtypes=[ctypes.c_void_p]
    api.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    h=api.OpenProcess(0x1000,False,pid)
    if not h:return False
    code=ctypes.c_ulong()
    try:
        assert api.GetExitCodeProcess(h,ctypes.byref(code))
        return code.value==259
    finally:api.CloseHandle(h)
def frozen():
    def git(*a):return subprocess.check_output(['git','-c',f'safe.directory={MAIN}','-C',str(MAIN),*a],encoding='utf-8').strip()
    assert git('rev-parse','HEAD')==COMMIT,'Scientific HEAD changed'
    assert not git('status','--porcelain','--untracked-files=no'),'Tracked worktree dirty'
    assert read(REC/'DEV30_EXECUTION.json')['gate']['passed'],'30k gate not passed'
    for seed in SEEDS:
        m=read(MAIN/f'logs/final2_{seed}/manifest.json')
        assert m['code_commit']==COMMIT and not m['code_dirty'] and m['method']=='learned_stopping_option'
        assert m['regime']['name']=='w2.36_r15' and m['stopping']['stop_rule']=='value' and m['stopping']['bellman_stop_value']=='soft'
    for seed,digest in S.get('final_model_sha256',{}).items():assert sha(MAIN/f'logs/final2_{seed}/final_model.zip')==digest
def spawn(name,module,args):
    frozen()
    with (REC/f'formal_{name}.stdout.log').open('xb') as out,(REC/f'formal_{name}.stderr.log').open('xb') as err:
        p=subprocess.Popen([str(PY),'-u','-B','-m',module,*map(str,args)],cwd=MAIN,env=ENV,stdout=out,stderr=err,creationflags=0x08004000)
    S['workers'][name]={'pid':p.pid,'started_at':now(),'exit_code':None};save();return p
def batch(jobs,replay=False):
    active={};queue=list(jobs)
    while queue or active:
        frozen()
        while queue and len(active)<6:
            name,module,args,dest,expected=queue.pop(0)
            active[name]=(spawn(name,module,args),dest,expected)
        for name,(p,dest,expected) in list(active.items()):
            S['workers'][name]['count']=len(list(dest.glob('seed_*.json')))
            code=p.poll()
            if code is None:continue
            S['workers'][name].update(exit_code=code,finished_at=now());save()
            assert code==0,f'Worker failed: {name}; preserve logs and locks'
            actual={int(x.stem.split('_')[-1]) for x in dest.glob('seed_*.json')}
            # Same directory has both disjoint replay blocks running concurrently.
            assert set(expected)<=actual,f'Missing results: {name}'
            if replay:
                assert all(read(dest/f'seed_{i}.json').get('fidelity_ok') is True for i in expected),f'Replay fidelity failed: {name}'
            del active[name]
        save()
        if active:time.sleep(300)
    for _,_,_,dest,_ in jobs:
        assert not list(dest.glob('*.lock')) and not list(dest.glob('*.tmp')),f'Unreleased evidence locks: {dest}'
def official(name,module,args):
    p=spawn(name,module,args);code=p.wait();S['workers'][name].update(exit_code=code,finished_at=now());save()
    assert code==0,f'Official tool failed: {name}'
def mappings(flag,folder):
    return [v for s in SEEDS for v in (flag,f'{s}={E/folder/str(s)}')]
def run():
    if STATE.exists():raise RuntimeError('Existing formal state: do not duplicate or blindly restart')
    if E.exists() or R.exists():raise RuntimeError('Existing formal/replay directory: inspect before starting')
    frozen();phase('waiting_for_all_training')
    try:
        while True:
            frozen();ready=True;S['training']={}
            for seed in SEEDS:
                m=read(MAIN/f'logs/final2_{seed}/manifest.json');live=alive(TRAIN_PIDS[seed])
                S['training'][str(seed)]={'status':m['status'],'compute_pid':TRAIN_PIDS[seed],'alive':live,'outer_decisions':m.get('actual_outer_decisions')}
                if m['status']!='completed':
                    assert live,f'Training process missing: {seed}; no restart'
                    ready=False
                else:
                    assert m['actual_outer_decisions']==60000,f'Incomplete budget: {seed}'
                    if live:ready=False
            save()
            if ready:break
            time.sleep(300)
        S['final_model_sha256']={}
        for seed in SEEDS:
            p=MAIN/f'logs/final2_{seed}/final_model.zip'
            with zipfile.ZipFile(p) as z:assert z.testzip() is None
            S['final_model_sha256'][str(seed)]=sha(p)
        phase('formal_evaluation_blinded')
        block='271000-271047,272000-272047';expected=list(range(271000,271048))+list(range(272000,272048));jobs=[]
        for row in ('pure','nominal'):
            dest=E/row
            args=['evaluate','--row',row,'--seeds',block,'--output-dir',dest]
            if row=='pure':module='experiments.v3_stopping';args+=['--run-dir','logs/final2_262460']
            else:module='experiments.regime_screen';args+=['--regime','w2.36_r15']
            jobs.append((row,module,args,dest,expected))
        for seed in SEEDS:
            for row in ('stopping','learned'):
                dest=E/row/str(seed)
                jobs.append((f'{row}_{seed}','experiments.v3_stopping',['evaluate','--run-dir',f'logs/final2_{seed}','--row',row,'--seeds',block,'--output-dir',dest],dest,expected))
        batch(jobs)
        for _,_,_,dest,_ in jobs:assert {int(p.stem.split('_')[-1]) for p in dest.glob('seed_*.json')}==set(expected)
        phase('formal_official_readout')
        official('readout','experiments.v3_stopping',['readout','--pure',E/'pure',*mappings('--stopping','stopping'),*mappings('--learned','learned'),'--seeds','271000-271047','--output',E/'readout_271000.json'])
        result=read(E/'readout_271000.json');S['verdict']=result.get('verdict');save()
        assert S['verdict']!='FIDELITY_FAIL' and not result.get('problems',[]),'Official fidelity failure'
        official('tables','experiments.final_tables',['--pure',E/'pure','--nominal',E/'nominal',*mappings('--learned','learned'),*mappings('--stopping','stopping'),'--output',E/'final_tables.json'])
        assert not read(E/'final_tables.json').get('problems',[]),'Paired table fidelity failure'
        phase('descriptive_replay')
        jobs=[]
        for seed in SEEDS:
            for first in (271000,272000):
                args=['replay','--run-dir',f'logs/final2_{seed}','--seeds',f'{first}-{first+47}','--formal-dir',E/'stopping'/str(seed),'--output-dir',R/str(seed)]
                if first==271000:args+=['--counterfactual']
                jobs.append((f'replay_{first}_{seed}','experiments.stopping_replay',args,R/str(seed),list(range(first,first+48))))
        batch(jobs,replay=True)
        official('replay_summary','experiments.stopping_replay',['summarize',*[v for seed in SEEDS for v in ('--replay',f'{seed}={R/str(seed)}')],'--output',R/'replay_summary.json'])
        phase('evidence_completed_ready_for_handoff')
    except Exception as err:
        S['error']=repr(err);phase('fault_preserve_evidence');raise
if __name__=='__main__':run()
