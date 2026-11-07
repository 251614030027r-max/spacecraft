"""Finish the prescribed two-block 5k comparison; never trains or changes models."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import sys,json,hashlib,datetime,traceback,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'脚本'))
from readonly_eval import checked_eval
OPS=[*range(292140,292148),*range(292120,292128)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,x):(ROOT/'记录'/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    with (ROOT/'记录/FINISH_5K.lock').open('x') as f:f.write(str(os.getpid()))
    models={};jobs=[];reused=[]
    for arm in ('e_sac','e_tcpf'):
        checkpoint=ROOT/f'TRAIN_{arm}_s0/checkpoints/step_5000.zip'
        with zipfile.ZipFile(checkpoint) as z:assert z.testzip() is None
        models[arm]={'path':str(checkpoint),'sha256':sha(checkpoint)}
        label=f'{arm}_s0_5000';out=ROOT/f'eval_{label}';out.mkdir(exist_ok=True)
        for op in OPS:
            p=out/f'{label}_{op}.json'
            if p.exists():
                r=json.loads(p.read_text(encoding='utf-8-sig'))
                assert r['opening']==op and Path(r['checkpoint']).resolve()==checkpoint.resolve()
                assert r['interface']=='v4' and (out/f'{label}_{op}.trace.jsonl').is_file()
                reused.append({'path':str(p),'sha256':sha(p)})
            else:jobs.append((op,str(checkpoint),label,str(out),'v4'))
    dump('FINISH_5K_STATUS.json',{'phase':'evaluating','pid':os.getpid(),'jobs':len(jobs),'reused':len(reused),'workers':4,'models':models})
    with ProcessPoolExecutor(max_workers=4) as pool:
        for n,f in enumerate(as_completed([pool.submit(checked_eval,j) for j in jobs]),1):
            r=f.result();print(json.dumps({'completed_missing':n,'opening':r['opening'],'label':r['label'],'event':r['event'],'return':r['discounted_return']}),flush=True)
            dump('FINISH_5K_STATUS.json',{'phase':'evaluating','pid':os.getpid(),'completed_missing':n,'jobs':len(jobs),'reused':len(reused)})
    assert all(sha(Path(x['path']))==x['sha256'] for x in models.values())
    assert all(sha(Path(x['path']))==x['sha256'] for x in reused)
    dump('FINISH_5K_EXIT.json',{'phase':'matched_5k_evaluation_complete','pid':os.getpid(),'completed_missing':len(jobs),'reused':len(reused),'total_5k_episodes':32,'models':models,'reused_result_hashes':reused,'no_training_or_updates':True,'completed_jst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()})
if __name__=='__main__':
    try:main()
    except Exception as ex:dump('FINISH_5K_ERROR.json',{'error':repr(ex),'traceback':traceback.format_exc()});raise
