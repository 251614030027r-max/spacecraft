"""Low frequency checkpoint/phase publication, then final ordinary push; never starts science."""
import os,time,json,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY='D:/py/DRL2/.venv/Scripts/python.exe'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
def read(p):
    try:return json.loads(p.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def run(name):
    with (ROOT/f'记录/{name}.out.log').open('a',encoding='utf-8') as out,(ROOT/f'记录/{name}.err.log').open('a',encoding='utf-8') as err:
        code=subprocess.run([PY,'-B',str(ROOT/f'脚本/{name}.py')],cwd='D:/py/DRL2',stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    write(ROOT/f'记录/{name.upper()}_EXIT.json',{'exit_code':code,'watcher_pid':os.getpid()});return code
def main():
    with (ROOT/'记录/WATCH.lock').open('x') as f:f.write(str(os.getpid()))
    last=None
    while True:
        state=read(ROOT/'STATE.json');terminal=(ROOT/'记录/PIPELINE_EXIT.json').exists();error=(ROOT/'记录/PIPELINE_ERROR.json').exists();marks=sorted(p.name for p in ROOT.glob('audit_*.json'));sig=(state.get('phase'),state.get('seed'),tuple(marks),state.get('completed_evaluations',0)//16,state.get('completed_shared_0k',0)//8,terminal,error)
        if state and sig!=last:
            code=run('publish_readout')
            if code==0:last=sig
        if terminal:
            code=run('finalize_round');write(ROOT/'记录/FINALIZER_EXIT.json',{'exit_code':code,'watcher_pid':os.getpid()});break
        if error:break
        time.sleep(120)
if __name__=='__main__':
    try:main()
    except Exception as ex:write(ROOT/'记录/WATCH_ERROR.json',{'error':repr(ex),'traceback':traceback.format_exc()});raise
