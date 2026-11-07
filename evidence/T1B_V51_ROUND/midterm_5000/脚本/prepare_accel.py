"""Prepare a scheduler-only adoption; existing scientific PIDs are never restarted."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'脚本/run_round.py').read_text(encoding='utf-8')
external='''
class AdoptedProcess:
    def __init__(self,pid,key):
        self.pid=pid;self.key=key;self.returncode=None
        api=ctypes.windll.kernel32;api.OpenProcess.restype=ctypes.c_void_p
        self.handle=api.OpenProcess(0x1000|0x100000,False,pid)
        if not self.handle:raise RuntimeError('Cannot adopt original training PID '+str(pid))
    def poll(self):
        code=ctypes.c_ulong();assert ctypes.windll.kernel32.GetExitCodeProcess(ctypes.c_void_p(self.handle),ctypes.byref(code))
        self.returncode=None if code.value==259 else int(code.value);return self.returncode
'''
s=s.replace('def main():',external+'\ndef main():',1)
s=s.replace("ROOT/'记录/PIPELINE.lock'","ROOT/'记录/PIPELINE_ACCEL.lock'",1)
old="    S.update(topic='t1b_v51_round'"
i=s.index(old);j=s.index('    try:\n        while True:',i)
s=s[:i]+'''    S.update(read(ROOT/'STATE.json'));S.update(pid=os.getpid(),scheduler='run_round_accel.py',scheduler_only_adoption=True)
    for key,record in S['training'].items():children[key]=AdoptedProcess(record['pid'],key)
    initial=read(ROOT/'记录/RESOURCE_ADMISSION.json')['initial'];last_admission=time.monotonic();measured_cost=read(ROOT/'记录/RESOURCE_ADMISSION.json')['estimated_additional_pair']
    scheduled=set();audited=set();gated={f'{a}_s0' for a in ARMS if read(ROOT/f'M_{a}_s0.json').get('pass')}
    shared=False;pending=[];inflight={};pool=None;failure=None
    S['completed_evaluations']=sum(1 for p in ROOT.glob('eval_e_*/*.json') if 'opening' in read(p))
    S['completed_shared_0k']=sum(1 for p in (ROOT/'eval_shared_0k').glob('*.json') if 'opening' in read(p))
    S['resource_adjustment']='User requested modest acceleration; preserve existing training PIDs, raise only eval concurrency2->3 with runtime memory admission.';save()
''' +s[j:]
old="pending.extend((op,str(checkpoint),f'{key}_{mark}',str(out),'v5') for op in OPS)"
assert old in s;s=s.replace(old,"pending.extend((op,str(checkpoint),f'{key}_{mark}',str(out),'v5') for op in OPS if not (out/f'{key}_{mark}_{op}.json').exists())")
old="pending[0:0]=[(op,str(zero),'shared_0k',str(out),'v5') for op in OPS]"
assert old in s;s=s.replace(old,"pending[0:0]=[(op,str(zero),'shared_0k',str(out),'v5') for op in OPS if not (out/f'shared_0k_{op}.json').exists()]")
old="            budget=0 if mem['free_commit_gb']<4. or mem['free_physical_gb']<2.0 else (1 if len(running)>=3 else 2 if running else 4)"
assert old in s
s=s.replace(old,'''            runtime=read(ROOT/'记录/RUNTIME_RESOURCE_POLICY.json')
            target=int(runtime.get('eval_during_training',3)) if running else int(runtime.get('eval_after_training',4))
            if pool is not None and not inflight and (mem['free_commit_gb']<2.8 or mem['free_physical_gb']<1.2):
                pool.shutdown(wait=True);pool=None;mem=memory()
            if pool is None:
                capacity=min(int((mem['free_commit_gb']-2.8)/1.25),int((mem['free_physical_gb']-1.2)/.3))
                budget=max(0,min(target,capacity))
            else:budget=0 if mem['free_commit_gb']<2.8 or mem['free_physical_gb']<1.2 else min(target,pool._max_workers)''')
compile(s,str(ROOT/'脚本/run_round_accel.py'),'exec')
(ROOT/'脚本/run_round_accel.py').write_text(s,encoding='utf-8')
