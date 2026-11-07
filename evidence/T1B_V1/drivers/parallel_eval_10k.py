"""Scheduling-only acceleration. Frozen evaluator/weights/tasks are unchanged.

Replaces the waiting queue controller, never the running training processes.
At most8 evaluation jobs while4 training workers run; then up to12 evaluation
workers. No extra missions or repeat training are added for CPU utilization.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import sys,json,time,hashlib,ctypes
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
sys.path.insert(0,'D:/py/DRL2')
from experiments.t1b_repair_v1 import ROOT,dump,eval_job
ARMS=['e_sac','e_tcpf']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def alive(pid):
    # Only query the exact already verified process identity. No process mutation.
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False
    code=ctypes.c_ulong()
    try:
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        return code.value==259
    finally:kernel.CloseHandle(handle)
def main():
    record=ROOT/'记录';lock=record/'PARALLEL_EVAL.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    state=json.loads((ROOT/'STATE.json').read_text(encoding='utf-8-sig'))
    assert state['phase']=='matched_seed0_continuation_10k'
    assert not alive(19996),'Old queue still alive; never compete with its evaluation scheduling'
    state.update(pid=os.getpid(),phase='parallel_10k_eval_with_remaining_training',next_action='Frozen10k original8 + expanded8 per arm; evaluation pool8 alongside4 training workers, then12; no further training.',scheduler='parallel_eval_10k.py')
    source_pids={'e_sac':16068,'e_tcpf':13124};pending=[];inflight={};scheduled=set();completed=[];errors=[];last_write=0
    jobs={}
    for arm in ARMS:
        checkpoint=str(ROOT/f'结果/CONTINUE_{arm}_seed0_10k/checkpoints/step_10000.zip')
        for suffix,openings in [('',range(292140,292148)),('_expanded',range(292120,292128))]:
            output=ROOT/f'结果/eval_{arm}_10000{suffix}';output.mkdir(exist_ok=True)
            label=f'{arm}_rl_10000{suffix}'
            group=[(op,checkpoint,label,str(output)) for op in openings]
            jobs[(arm,suffix)]=group
            dump(output/'MANIFEST.json',{'jobs':group,'fixed_openings':list(openings),'training':False,'policy_readout':'Unchanged deterministic tanh(mean) mapped to verified support bin.','scheduler':'parallel_eval_10k.py','maximum_evaluation_workers':12,'science_commit':'636b65517bf73985b4e544c67decf857b4fffc19'})
    dump(record/'CPU_TEST_POLICY.json',{'user_instruction':'进度如何，加速测完，后续测试务必吃满CPU','logical_cpus':os.cpu_count(),'maximum_evaluation_workers':12,'while_training_eval_budget':8,'per_worker_BLAS_OMP_threads':1,'science_unchanged':True,'fixed_total_missions':32,'controller_replacement_only':'Idle waiting controller19996 stopped by exact identity; training13124 and its workers retained.','scheduler_sha256':sha(__file__),'original_driver_sha256':sha(ROOT/'脚本/continue_v1_10k.py')})
    with ProcessPoolExecutor(max_workers=12) as pool:
        while True:
            training_live=False
            for arm in ARMS:
                live=alive(source_pids[arm]);training_live|=live
                end=ROOT/f'结果/CONTINUE_{arm}_seed0_10k/finished.json'
                if arm not in scheduled and end.exists() and not live:
                    finish=json.loads(end.read_text(encoding='utf-8'));assert finish['steps']==10000 and finish['finite']
                    scheduled.add(arm)
                    for suffix in ['', '_expanded']:pending.extend(jobs[(arm,suffix)])
                elif arm not in scheduled and not live and not end.exists():
                    raise RuntimeError(f'{arm} training stopped without finished receipt; preserve recovery and inspect stderr')
            budget=8 if training_live else 12
            while pending and len(inflight)<budget:
                job=pending.pop(0);inflight[pool.submit(eval_job,job)]=job
            now=time.monotonic()
            if now-last_write>=30:
                last_write=now
                state.update(phase='parallel_10k_eval_with_remaining_training' if training_live else 'evaluate_predeclared_10k_and_expanded_dev',training_active=training_live,active=[{'kind':'existing_training','name':arm,'pid':source_pids[arm]} for arm in ARMS if alive(source_pids[arm])]+[{'kind':'frozen_eval','opening':j[0],'label':j[2]} for j in inflight.values()],parallel_eval_completed=len(completed),parallel_eval_total=32,eval_worker_budget=budget,last_native_check_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
                dump(ROOT/'STATE.json',state)
                dump(record/'PARALLEL_EVAL_STATUS.json',{'pid':os.getpid(),'completed':len(completed),'total':32,'inflight':len(inflight),'budget':budget,'training_live':training_live,'errors':errors})
            ready,_=wait(inflight,timeout=30,return_when=FIRST_COMPLETED) if inflight else (set(),set())
            if not inflight and training_live:time.sleep(30)
            for future in ready:
                job=inflight.pop(future)
                try:
                    result=future.result();completed.append({'opening':job[0],'label':job[2],'result':result});print(json.dumps({'opening':job[0],'label':job[2],'event':result.get('event'),'discounted_return':result.get('discounted_return'),'completed':len(completed)},ensure_ascii=False),flush=True)
                except Exception as ex:errors.append({'opening':job[0],'label':job[2],'error':repr(ex)})
            for (arm,suffix),group in jobs.items():
                if all((Path(j[3])/f'{j[2]}_{j[0]}.json').exists() for j in group):
                    dump(Path(group[0][3])/'finished.json',{'completed':True,'count':8,'scheduler':'parallel_eval_10k.py'})
            if len(scheduled)==2 and not pending and not inflight:break
    state.update(active=[],training_active=False,phase='matched_10k_complete_readout_required' if not errors else 'parallel_evaluation_failure_read_stderr',parallel_eval_completed=len(completed),next_action='Publish complete10k and expandeddev paired evidence; no automatic extra training.')
    dump(ROOT/'STATE.json',state);dump(record/'PARALLEL_EVAL_EXIT.json',{'pid':os.getpid(),'exit_code':0 if not errors else 1,'completed':len(completed),'errors':errors})
    lock.unlink()
    if errors:raise RuntimeError(str(errors))
if __name__=='__main__':
    try:main()
    except Exception as ex:
        dump(ROOT/'记录/PARALLEL_EVAL_ERROR.json',{'pid':os.getpid(),'error':repr(ex)})
        raise
