"""Frozen 268000 value-rule check; no training, no early result inspection."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT=Path('C:/Users/35884/Documents/Spacecraft')
TOPIC=ROOT/'过程文件/价值规则检验'
REC=TOPIC/'记录'
MAIN=Path('D:/py/DRL2')
PY=MAIN/'.venv/Scripts/python.exe'
DATA=MAIN/'eval/stopping/value_check'
STATE=REC/'VALUE_CHECK_EXECUTION.json'
LOCK=REC/'supervisor.lock'
COMMIT='758c9a89b3121ad4e3f73d2c716b1471dc94fe86'
MODELS={'262430':'a3f36bde8589853efefc7abb24f13f3b068974badd3e9ad2b104742d272b0b3b',
        '262431':'4b5b53ee4903e1af0d7534898360a3c0db003eb88b9582d55dbfefed33721b82',
        '262432':'8b56adbf09d4a9753a040955324f3f75cda487bfcc77695644f16f6fdcccd74b'}
ENV=os.environ.copy()
ENV.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(MAIN),PYTHONIOENCODING='utf-8')
COLLAB=ROOT/'过程文件/协作/Git工作树'
REMOTE='https://github.com/251614030027r-max/spacecraft.git'
BRANCH='review/stopping-value-check-20261006'
S={'schema':'value_check_v1','phase':'initializing','commit':COMMIT,'started_at':None,'workers':[],
   'events':[],'block':'268000-268047','development_only':True,'training_authorized':False}

def now(): return datetime.now().astimezone().isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def save(phase=None):
    if phase and S['phase']!=phase:
        S['phase']=phase;S['events'].append({'at':now(),'phase':phase});print(now(),phase,flush=True)
    S['updated_at']=now();S['supervisor_pid']=os.getpid();write(STATE,S)
def git(repo,*args):
    r=subprocess.run(['git','-c',f'safe.directory={repo}','-C',str(repo),*args],capture_output=True,text=True,encoding='utf-8',errors='replace')
    if r.returncode: raise RuntimeError(r.stderr.strip())
    return r.stdout.strip()
def frozen():
    if git(MAIN,'rev-parse','HEAD')!=COMMIT: raise RuntimeError('Fixed science commit changed')
    if git(MAIN,'status','--porcelain','--untracked-files=no'): raise RuntimeError('Dirty tracked checkout')
    if git(MAIN,'diff','2e5c236','HEAD','--stat','--','env','controllers','dynamics','train'): raise RuntimeError('Frozen physical/control/train code changed')
    for m,h in MODELS.items():
        p=MAIN/f'logs/stop_{m}/final_model.zip'
        if sha(p)!=h: raise RuntimeError('Frozen model hash mismatch: '+m)
        manifest=read(p.parent/'manifest.json')
        if manifest['status']!='completed' or manifest['actual_outer_decisions']!=60000 or manifest['code_dirty'] or manifest['code_commit']!='2e5c236f7412caea3b203519619ef7a48bb16df9':
            raise RuntimeError('Invalid frozen model run: '+m)
def precheck():
    frozen()
    test=(REC/'pytest_preflight.log').read_text(encoding='utf-8-sig',errors='replace')
    if '13 passed' not in test or 'FAILED' in test or 'ERROR' in test: raise RuntimeError('Required tests did not pass')
    if DATA.exists() and any(DATA.iterdir()): raise RuntimeError('Value-check directory is not empty; never overwrite')
    write(REC/'PRECHECK.json',{'at':now(),'commit':COMMIT,'tracked_clean':True,'physics_control_training_diff_empty':True,
         'tests':test.strip().splitlines()[-1],'models':MODELS,'test_log_sha256':sha(REC/'pytest_preflight.log'),
         'preregistration_sha256':sha(MAIN/'docs/STOPPING_VALUE_RULE_CHECK_20261006.md')})
def spawn(name,args):
    out=REC/(name+'.stdout.log');err=REC/(name+'.stderr.log')
    if out.exists() or err.exists(): raise RuntimeError('Existing worker logs: '+name)
    command=[str(PY),'-u','-B','-m','experiments.v3_stopping',*args]
    with out.open('wb') as o,err.open('wb') as e:
        p=subprocess.Popen(command,cwd=MAIN,env=ENV,stdout=o,stderr=e,creationflags=0x08000000)
    w={'name':name,'pid':p.pid,'command':command,'started_at':now(),'stdout':str(out),'stderr':str(err),'exit_code':None}
    S['workers'].append(w);save();return p,w
def wait_all(batch,phase):
    save(phase)
    while batch:
        for p,w in list(batch):
            code=p.poll()
            if code is not None:
                w.update(exit_code=code,exit_observed_at=now());batch.remove((p,w));save()
                if code: raise RuntimeError(f"Worker failed: {w['name']} code={code}; preserve workers/locks and inspect")
        if batch: time.sleep(60)
def groups():
    return {'pure':DATA/'pure',**{f'{row}_{m}':DATA/row/m for row in ('value','stopping') for m in MODELS}}
def evaluate():
    batch=[]
    for m in MODELS:
        for row in ('value','stopping'):
            batch.append(spawn(f'vc_{row}_{m}',['evaluate','--run-dir',f'logs/stop_{m}','--row',row,'--seeds','268000-268047','--output-dir',str(DATA/row/m)]))
    # This 12-logical-core host has capacity for the six rows plus two Pure workers.
    for i in (1,2):
        batch.append(spawn(f'vc_pure_{i}',['evaluate','--run-dir','logs/stop_262430','--row','pure','--seeds','268000-268047','--output-dir',str(DATA/'pure')]))
    save('evaluation_running_blinded')
    while batch:
        for p,w in list(batch):
            code=p.poll()
            if code is not None:
                w.update(exit_code=code,exit_observed_at=now());batch.remove((p,w));save()
                if code: raise RuntimeError(f"Worker failed: {w['name']} code={code}; no automatic retries or lock deletion")
        # Names only. No JSON read before all 336 files AND all worker exits.
        S['counts']={k:len(list(d.glob('seed_*.json'))) for k,d in groups().items()};save()
        if batch:time.sleep(60)
    expected={f'seed_{s}.json' for s in range(268000,268048)}
    for k,d in groups().items():
        if {p.name for p in d.glob('seed_*.json')}!=expected or list(d.glob('*.lock')) or list(d.glob('*.tmp')):
            raise RuntimeError('Incomplete seeds or remaining locks/temp: '+k)
    frozen()
    # Post-completion provenance check supplements official readout without changing criteria.
    for k,d in groups().items():
        for p in d.glob('seed_*.json'):
            r=read(p)
            if r['code_commit']!=COMMIT or r['code_dirty'] or r['max_decisions'] is not None:raise RuntimeError('Result provenance mismatch: '+str(p))
    args=['readout-value','--seeds','268000-268047','--pure',str(DATA/'pure')]
    for m in MODELS:args+=['--value',f'{m}={DATA}/value/{m}','--stopping',f'{m}={DATA}/stopping/{m}']
    args+=['--output',str(DATA/'readout_value.json')]
    wait_all([spawn('vc_official_readout',args)],'official_readout')
    result=read(DATA/'readout_value.json');S['verdict']=result['verdict'];S['problems']=result['problems'];save('readout_completed')
    return result
def build_delivery(result):
    build=TOPIC/'审查材料_20261006'
    build.mkdir(exist_ok=False)
    def copy(src,rel):
        dst=build/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if sha(src)!=sha(dst):raise RuntimeError('Copied evidence SHA differs')
    for p in DATA.rglob('*.json'):copy(p,'results/'+p.relative_to(DATA).as_posix())
    for p in REC.iterdir():
        if p.is_file() and (p.name.startswith('vc_') or p.name in {'PRECHECK.json','VALUE_CHECK_EXECUTION.json','pytest_preflight.log'}):copy(p,'records/'+p.name)
    copy(MAIN/'docs/STOPPING_VALUE_RULE_CHECK_20261006.md','STOPPING_VALUE_RULE_CHECK_20261006.md')
    copy(Path(__file__),'run_value_check.py')
    (build/'REPORT.md').write_text('# Value-rule check: independent developer delivery\n\n'+
        f"Commit: {COMMIT}; block 268000–268047; 336 episodes, all eight workers exited. Verdict: {result['verdict']}.\n\n"+
        'Frozen models unchanged, no training. Physics/control/train diff vs2e5c236 empty; specified tests13 passed; model SHA in PRECHECK. Value rule: first Q_H >= Q_C at deterministic action, ignoring entropy. Stopping-head controls reported only. Gate: strictly more clean completions than same-block Pure, no more violations, destroyed Pure successes <=2; at least two models.\n\n'+
        'Historical267000: Pure41, stopping37/41/36, learned12/31/17; METHOD_DOES_NOT_HOLD. Different block, background only. New result is development validation, excluded from paper performance numbers. No automatic method changes, new rules or training after verdict.\n\n'+
        'Current raw results and all vc logs in this package. Models excluded; frozen hashes: '+json.dumps(MODELS)+'\n\n'+
        'Supervisor events record start/end and errors; no early result inspection. Parallel runtime is not a real-time performance claim.\n\n'+
        'Official readout:\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    listing={p.relative_to(build).as_posix():sha(p) for p in sorted(build.rglob('*')) if p.is_file()}
    write(build/'FILES_SHA256.json',listing)
    (build/'ALL_FILES_SHA256.txt').write_text(''.join(f'{h}  {n}\n' for n,h in listing.items()),encoding='utf-8')
    archive=ROOT/'上层交付/待发布/VALUE_RULE_CHECK_20261006.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(build.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(build).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('ZIP CRC failed')
        if set(z.namelist())!=set(listing)|{'FILES_SHA256.json','ALL_FILES_SHA256.txt'}:raise RuntimeError('ZIP member set mismatch')
        for n,h in listing.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('ZIP SHA failed: '+n)
    receipt={'verdict':result['verdict'],'commit':COMMIT,'zip':str(archive),'zip_sha256':sha(archive),'bytes':archive.stat().st_size,'crc_ok':True,'members_sha_ok':True}
    write(REC/'DELIVERY_RECEIPT.json',receipt)
    latest=ROOT/'上层交付/最新';history=ROOT/'上层交付/历史/STOPPING_FORMAL_20261006'
    if history.exists():raise RuntimeError('Archive destination exists; verified new ZIP left in staging')
    if sha(latest/'STOPPING_FORMAL_20261006.zip')!='94ab2368ea2d50d35aecc67230e0a6e37606dd5047d4ada202e316e33583f2ae':raise RuntimeError('Latest delivery has changed')
    history.mkdir()
    for p in list(latest.iterdir()):
        if not p.is_file():raise RuntimeError('Unexpected directory in latest')
        h=sha(p);target=history/p.name;p.rename(target)
        if sha(target)!=h:raise RuntimeError('Historical move SHA changed')
    shutil.copy2(build/'REPORT.md',latest/'REPORT.md');shutil.copy2(DATA/'readout_value.json',latest/'readout_value.json')
    final_archive=latest/archive.name;archive.rename(final_archive);receipt['zip']=str(final_archive)
    write(latest/'VALUE_CHECK_DELIVERY_RECEIPT.json',receipt);write(REC/'DELIVERY_RECEIPT.json',receipt)
    (latest/'README.md').write_text(f"# 当前交付：价值规则开发检验\n\n官方判定：{result['verdict']}。读REPORT.md与readout_value.json；本轮完整336个结果及日志在{archive.name}，SHA256 {receipt['zip_sha256']}。仅开发验证；停止等待用户/上层决定，不自动训练或改方法。固定协作入口collab/spacecraft的docs/collaboration。\n",encoding='utf-8')
    S['delivery']=receipt;save('delivery_verified')
    return build
def publish_review(build):
    repo=TOPIC/'Git审查工作树'
    if repo.exists():raise RuntimeError('Review checkout exists; no overwrite')
    r=subprocess.run(['git','-c',f'safe.directory={MAIN}','clone','--shared','--no-checkout',str(MAIN),str(repo)],capture_output=True,text=True)
    if r.returncode:raise RuntimeError(r.stderr)
    git(repo,'switch','-c',BRANCH,COMMIT);git(repo,'remote','set-url','origin',REMOTE);git(repo,'config','core.autocrlf','false')
    dest=repo/'docs/reviews/stopping_value_check_20261006';shutil.copytree(build,dest)
    git(repo,'add','--','docs/reviews/stopping_value_check_20261006');git(repo,'commit','-m','Deliver preregistered frozen-model value-rule check and all episode evidence')
    git(repo,'push','-u','origin',BRANCH)
    commit=git(repo,'rev-parse','HEAD')
    if git(repo,'ls-remote','origin','refs/heads/'+BRANCH).split()[0]!=commit:raise RuntimeError('Remote review SHA mismatch')
    S['review_commit']=commit;S['review_url']=f'https://github.com/251614030027r-max/spacecraft/tree/{commit}/docs/reviews/stopping_value_check_20261006';save('review_published')
    # Update the collaboration checkout only when clean and still owned by this topic.
    if git(COLLAB,'status','--porcelain'):raise RuntimeError('Collaboration checkout dirty; review already published, update index manually')
    git(COLLAB,'pull','--ff-only','origin','collab/spacecraft')
    hub=COLLAB/'docs/collaboration'
    handoff=hub/'lower/handoffs/value_check_20261006.md'
    handoff.write_text(f"# 价值规则开发检验交接\n\n科学提交{COMMIT}，冻结三模型，268000–268047七行336回合完成；官方{S['verdict']}。无新训练。\n\n[完整审查证据]({S['review_url']})：REPORT、官方读数、全部结果与vc日志、SHA。结论只用于开发检验，停下等待用户/上层决定。\n",encoding='utf-8')
    current=read(hub/'CURRENT.json')
    if current.get('topic')=='stopping_value_rule_check':
        current.update(phase='await_user_decision' if S['verdict']=='VALUE_RULE_HOLDS' else 'closed_negative' if S['verdict']=='VALUE_RULE_DOES_NOT_HOLD' else 'stopped_fidelity',official_verdict=S['verdict'],review_commit=commit,review_url=S['review_url'])
        write(hub/'CURRENT.json',current)
    with (hub/'lower/README.md').open('a',encoding='utf-8') as f:f.write('\n最新价值检验：[交接](handoffs/value_check_20261006.md)。\n')
    with (hub/'reviews/README.md').open('a',encoding='utf-8') as f:f.write(f"\n[价值规则检验完整证据]({S['review_url']})，官方{S['verdict']}，仅开发验证。\n")
    git(COLLAB,'add','--','docs/collaboration');git(COLLAB,'commit','-m','Record completed value-rule developer check and evidence link');git(COLLAB,'push','origin','collab/spacecraft')
def run():
    fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'at':now()},f)
    S['started_at']=now()
    try:
        precheck();result=evaluate();build=build_delivery(result);publish_review(build)
        save('await_user_decision' if result['verdict']=='VALUE_RULE_HOLDS' else 'closed_negative' if result['verdict']=='VALUE_RULE_DOES_NOT_HOLD' else 'stopped_fidelity')
    except Exception as e:
        S['error']=str(e);save('execution_or_delivery_fault');print(str(e),file=sys.stderr,flush=True)
    finally:
        LOCK.unlink(missing_ok=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('check','start','run','status'));args=parser.parse_args()
    if args.mode=='status':
        if not STATE.exists():print('STATUS: NOT_STARTED');return
        s=read(STATE);print('STATUS:',s['phase'],'UPDATED:',s['updated_at']);print('COUNTS:',s.get('counts',{}));print('VERDICT:',s.get('verdict'));print('ERROR:',s.get('error'));print('REVIEW:',s.get('review_url'));return
    if args.mode=='check':precheck();print('PRECHECK_OK');return
    if args.mode=='start':
        precheck()
        if LOCK.exists() or STATE.exists():raise RuntimeError('Supervisor state/lock exists; no duplicate start')
        out=REC/'supervisor.stdout.log';err=REC/'supervisor.stderr.log'
        if out.exists() or err.exists():raise RuntimeError('Supervisor logs exist; no overwrite')
        with out.open('wb') as o,err.open('wb') as e:
            p=subprocess.Popen([str(PY),'-u','-B',str(Path(__file__).resolve()),'run'],cwd=MAIN,env=ENV,stdout=o,stderr=e,creationflags=0x08000000)
        write(REC/'SUPERVISOR_LAUNCH.json',{'wrapper_pid':p.pid,'at':now()});print('VALUE_CHECK_STARTED PID=',p.pid);return
    run()
if __name__=='__main__':main()
