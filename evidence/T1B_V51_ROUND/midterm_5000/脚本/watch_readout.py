"""Low-frequency phase readouts and exactly the two requested midterm pushes."""
import os,json,time,subprocess,ctypes,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY='D:/py/DRL2/.venv/Scripts/python.exe'
def read(p):
    try:return json.loads(p.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}
def write(name,x):(ROOT/'记录'/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def run(script,*args):
    name=script+('_'+args[0] if args else '')
    with (ROOT/f'记录/{name}.out.log').open('a') as out,(ROOT/f'记录/{name}.err.log').open('a') as err:
        rc=subprocess.run([PY,'-B',str(ROOT/f'脚本/{script}.py'),*args],cwd='D:/py/DRL2',stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    write(name.upper()+'_EXIT.json',{'exit_code':rc,'watch_pid':os.getpid()});return rc
def complete(mark):return all((ROOT/f'eval_{a}_s0_{mark}/{a}_s0_{mark}_{o}.json').exists() for a in ('e_sac','e_tcpf') for o in [*range(292140,292148),*range(292120,292128)])
def main():
    with (ROOT/'记录/WATCH.lock').open('x') as f:f.write(str(os.getpid()))
    last=None
    while True:
        s=read(ROOT/'STATE.json');terminal=(ROOT/'记录/PIPELINE_EXIT.json').exists();error=(ROOT/'记录/PIPELINE_ERROR.json').exists();sig=(s.get('phase'),tuple(s.get('started_seeds',[])),s.get('completed_evaluations',0)//16,s.get('completed_shared_0k',0)//8,len(list(ROOT.glob('audit_*.json'))),terminal,error)
        if s and sig!=last:
            if run('publish_readout')==0:last=sig
        for mark in (5000,10000):
            label=f'midterm_{mark}'
            if complete(mark) and not (ROOT/f'记录/PUSH_{label}.json').exists() and not (ROOT/f'记录/PUSH_{label}_FAILED.json').exists():
                if run('push_snapshot',label)!=0:write(f'PUSH_{label}_FAILED.json',{'inspect_original_stderr':True,'do_not_restart_training':True})
        if terminal:
            if not (ROOT/'记录/PUSH_final.json').exists():run('push_snapshot','final')
            break
        if error:break
        time.sleep(300)
if __name__=='__main__':
    try:main()
    except Exception as ex:write('WATCH_ERROR.json',{'error':repr(ex),'traceback':traceback.format_exc()});raise
