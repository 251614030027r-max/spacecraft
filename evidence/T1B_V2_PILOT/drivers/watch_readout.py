"""Low-frequency actual V2 delivery, then final offline readout/publication. Never trains."""
import json,time,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY='D:/py/DRL2/.venv/Scripts/python.exe'
def read(p):
    try:return json.loads(p.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,json.JSONDecodeError):return {}
def run(script,stem):
    with (ROOT/f'记录/{stem}.out.log').open('a',encoding='utf-8') as out,(ROOT/f'记录/{stem}.err.log').open('a',encoding='utf-8') as err:
        return subprocess.run([PY,'-B',str(ROOT/'脚本'/script)],cwd='D:/py/DRL2',stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW).returncode
last=None
while True:
    s=read(ROOT/'STATE.json');phase=s.get('phase');terminal=(ROOT/'记录/PIPELINE_EXIT.json').exists();marks=tuple(read(ROOT/f'TRAIN_{a}_s0/heartbeat.json').get('steps',0)//1000 for a in ['e_sac','e_tcpf']);key=(phase,marks,terminal)
    if phase and key!=last:
        code=run('publish_readout.py','publication')
        (ROOT/'记录/PUBLICATION_STATUS.json').write_text(json.dumps({'pid':os.getpid(),'phase':phase,'step_buckets':marks,'terminal':terminal,'exit_code':code}),encoding='utf-8')
        if code==0:last=key
    if terminal and last==key:
        script=ROOT/'脚本/finalize_pilot.py'
        if script.exists():
            code=run(script.name,'finalizer');(ROOT/'记录/FINALIZER_EXIT.json').write_text(json.dumps({'pid':os.getpid(),'exit_code':code}),encoding='utf-8')
            break
    time.sleep(120)
