"""Read-only scientific preflight; records written only under this new topic."""
import csv, hashlib, json, re, subprocess
from datetime import datetime
from pathlib import Path
ROOT=Path('C:/Users/35884/Documents/Spacecraft'); MAIN=Path('D:/py/DRL2')
TOPIC=ROOT/'过程文件/最终主线重训'; REC=TOPIC/'记录'
COLLAB=ROOT/'过程文件/协作/Git工作树'
COMMIT='f2f8169acd580ea96a578d45022dd8952447eca5'
def git(*args):
    return subprocess.check_output(['git','-c',f'safe.directory={MAIN}','-C',str(MAIN),*args],encoding='utf-8').strip()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert git('rev-parse','HEAD')==COMMIT
assert not git('status','--porcelain','--untracked-files=no')
assert not git('diff','2e5c236','HEAD','--stat','--','env','controllers','dynamics')
test=REC/'preflight_tests.log';text=test.read_text(encoding='utf-8-sig',errors='replace')
assert re.search(r'\d+ passed',text) and not any(w in text for w in ('FAILED','ERROR',' interrupted')),text
screen=MAIN/'eval/regime_screen/readout_final.json';r=json.loads(screen.read_text(encoding='utf-8'))
assert r['selected']=='w2.36_r15' and not r['problems']
assert not (MAIN/'eval/final2').exists()
for s in (262460,262461,262462):assert not (MAIN/f'logs/final2_{s}').exists()
old={}
for s in (262450,262451,262452):
    run=MAIN/f'logs/final_{s}';monitor=run/'train.monitor.csv'
    with monitor.open(newline='') as f:
        f.readline();rows=list(csv.DictReader(f))
    checkpoints=sorted((run/'checkpoints').glob('stopping_*_outer_decisions.zip'),key=lambda p:int(p.stem.split('_')[1]))
    last=checkpoints[-1]
    old[str(s)]={'classification':'INVALID_INTERRUPTED_DO_NOT_USE','directory':str(run),'manifest_original_status':'running',
        'last_checkpoint':str(last),'checkpoint_sha256':sha(last),'checkpoint_modified_at':datetime.fromtimestamp(last.stat().st_mtime).astimezone().isoformat(),
        'monitor_rows':len(rows),'monitor_sha256':sha(monitor),'manifest_sha256':sha(run/'manifest.json'),
        'interruption_observed':'2026-10-07T18:52+09:00','last_supervisor_snapshot':'2026-10-07T16:35+09:00'}
write(REC/'INVALID_INTERRUPTED_RUNS.json',old)
items=[screen,test,COLLAB/'docs/collaboration/upper/run_orders/FINAL_RERUN_20261007.md',
       MAIN/'train/stopping.py',MAIN/'train/train_stopping.py',MAIN/'experiments/v3_stopping.py',
       MAIN/'experiments/stopping_replay.py',MAIN/'experiments/final_tables.py',
       TOPIC/'脚本/Start-FinalRerun.ps1',TOPIC/'脚本/Show-FinalRerun.ps1']
receipt={'prepared_at':datetime.now().astimezone().isoformat(),'status':'ready_for_manual_training','training_started':False,
    'commit':COMMIT,'collaboration_order_commit':'19aa467','tests_passed':True,'tests_summary':text.strip().splitlines()[-1],
    'tracked_clean':True,'physics_control_diff_vs_2e5c236_empty':True,'selected_regime':'w2.36_r15','screen_reused':True,
    'seeds':[262460,262461,262462],'outer_decisions_each':60000,'stop_rule':'value','bellman_stop_value':'soft',
    'manual_launch_only':True,'monitor_read_only':True,'files':[{'path':str(p),'sha256':sha(p)} for p in items]}
write(REC/'PREPARATION.json',receipt);print(receipt['tests_summary']);print('READY_FOR_MANUAL_TRAINING; NO_TRAINING_STARTED')
