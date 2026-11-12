"""Authorized frozen mainline. Support assets stay outside the science checkout."""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from datetime import datetime
from pathlib import Path

ROOT=Path('C:/Users/35884/Documents/Spacecraft')
TOPIC=ROOT/'过程文件/最终主线'; REC=TOPIC/'记录'
MAIN=Path('D:/py/DRL2'); PY=MAIN/'.venv/Scripts/python.exe'
COLLAB=ROOT/'过程文件/协作/Git工作树'
STATE=REC/'FINAL_MAINLINE_EXECUTION.json'; LOCK=REC/'supervisor.lock'
COMMIT='867a3a13e4ad1125a3420bd7ca2bc65d67bb2ba9'
BASE='e855bb7c901682268a8c18c1e71c7d49b2f9b541'
REMOTE='https://github.com/251614030027r-max/spacecraft.git'
GRID=[f'w{w}_r{r}' for w in ('2.36','3.00','3.50') for r in (15,18,20)]
SEEDS=(262450,262451,262452)
SCREEN=MAIN/'eval/regime_screen'; DEV=MAIN/'eval/final/devcheck'; FORMAL=MAIN/'eval/final/formal'
ENV=os.environ.copy(); ENV.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(MAIN),PYTHONIOENCODING='utf-8')
S={'schema':'final_mainline_v1','phase':'initializing','science_commit':COMMIT,'authority':'User 2026-10-06: read collaboration FINAL_MAINLINE_20261006 and execute',
   'run_order_commit':'39ecee2b08cc63d95d0ab1e0e7ed98253dbe6aeb','started_at':None,'workers':[],'events':[],
   'selected_regime':None,'training_seeds':list(SEEDS),'budget_each':60000,'counts':{},'readout_blinded_until_all_exits':True}

def now(): return datetime.now().astimezone().isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(phase=None):
    if phase and phase!=S['phase']:
        S['phase']=phase;S['events'].append({'at':now(),'phase':phase});print(now(),phase,flush=True)
    S.update(updated_at=now(),supervisor_pid=os.getpid());write(STATE,S)
def git(repo,*args):
    r=subprocess.run(['git','-c',f'safe.directory={repo}','-C',str(repo),*args],capture_output=True,encoding='utf-8',errors='replace')
    if r.returncode:raise RuntimeError(r.stderr)
    return r.stdout.strip()
def frozen():
    if git(MAIN,'rev-parse','HEAD')!=COMMIT or git(MAIN,'status','--porcelain','--untracked-files=no'):raise RuntimeError('Science HEAD changed or tracked worktree dirty')
    for base,paths in [(BASE,['env','controllers','dynamics','train','experiments','tests']),('2e5c236',['env','controllers','dynamics'])]:
        if git(MAIN,'diff',base,'HEAD','--stat','--',*paths):raise RuntimeError('Forbidden scientific diff against '+base)
def precheck():
    frozen();test=(REC/'preflight_tests.log').read_text(encoding='utf-8-sig',errors='replace')
    if ' passed' not in test or any(x in test for x in ('FAILED','ERROR',' interrupted')):raise RuntimeError('Prescribed tests not confirmed passed')
    for p in (SCREEN,MAIN/'eval/final',*(MAIN/f'logs/final_{s}' for s in SEEDS)):
        if p.exists():raise RuntimeError('Existing experiment path; no overwrite: '+str(p))
    docs={}
    for name,prefix,suffix in [('FINAL_MAINLINE_RUN_ORDER_20261006.md','62e869e7','6aa0'),('REGIME_SCREEN_20261006.md','12f4ce24','d296')]:
        content=subprocess.run(['git','-c',f'safe.directory={MAIN}','-C',str(MAIN),'show',f'{BASE}:docs/{name}'],capture_output=True,check=True).stdout
        h=hashlib.sha256(content).hexdigest()
        if not h.startswith(prefix) or not h.endswith(suffix):raise RuntimeError('Preregistration hash differs: '+name)
        live=(MAIN/'docs'/name).read_bytes().replace(b'\r\n',b'\n')
        if live!=content:raise RuntimeError('Current preregistration differs from frozen blob')
        docs[name]={'git_lf_sha256':h,'working_file_sha256':sha(MAIN/'docs'/name)}
    write(REC/'PRECHECK.json',{'at':now(),'commit':COMMIT,'scientific_diff_vs_e855bb7_empty':True,'physics_control_diff_vs_2e5c236_empty':True,
        'tracked_clean':True,'tests':test.strip().splitlines()[-1],'test_log_sha256':sha(REC/'preflight_tests.log'),'preregistrations':docs,'fresh_paths':True})

def spawn(name,module,args,group=None,directory=None,attempt=0):
    frozen(); stem=f'{name}_attempt{attempt}'
    out=REC/(stem+'.stdout.log');err=REC/(stem+'.stderr.log')
    if out.exists() or err.exists():raise RuntimeError('Existing worker log: '+stem)
    cmd=[str(PY),'-u','-B','-m',module,*map(str,args)]
    with out.open('wb') as o,err.open('wb') as e:p=subprocess.Popen(cmd,cwd=MAIN,env=ENV,stdout=o,stderr=e,creationflags=0x08000000)
    w={'name':name,'module':module,'args':list(map(str,args)),'pid':p.pid,'started_at':now(),'exit_code':None,'stdout':str(out),'stderr':str(err),
       'group':group,'directory':str(directory) if directory else None,'attempt':attempt}
    S['workers'].append(w);save();return p,w
def observed(job):
    p,w=job;code=p.poll()
    if code is not None and w['exit_code'] is None:w.update(exit_code=code,exit_observed_at=now());save()
    return code
def command(name,module,args):
    job=spawn(name,module,args)
    while observed(job) is None:time.sleep(5)
    if job[1]['exit_code']:raise RuntimeError('Official command failed: '+name)
def verify_complete(d,block):
    a,b=map(int,block.split('-'));expected={f'seed_{s}.json' for s in range(a,b+1)}
    if {p.name for p in d.glob('seed_*.json')}!=expected or list(d.glob('*.lock')) or list(d.glob('*.tmp')):raise RuntimeError('Incomplete group: '+str(d))
def evaluate_jobs(specs,phase,block):
    jobs=[spawn(name,module,args,group,d) for name,module,args,group,d in specs];save(phase)
    groups={g:d for _,_,_,g,d in specs}; retries={g:0 for g in groups}
    while True:
        for job in jobs:observed(job)
        S['counts']={g:len(list(d.glob('seed_*.json'))) for g,d in groups.items()};save()
        # Retry only a failed group after every original process in that group has exited.
        for g,d in groups.items():
            current=[j for j in jobs if j[1]['group']==g and j[1]['attempt']==retries[g]]
            if current and all(j[0].poll() is not None for j in current) and any(j[1]['exit_code'] for j in current):
                if retries[g]>=1:raise RuntimeError('Repeated evaluation error; retained evidence: '+g)
                backup=REC/'异常残余'/g;backup.mkdir(parents=True,exist_ok=False)
                for p in [*d.glob('*.lock'),*d.glob('*.tmp')]:shutil.copy2(p,backup/p.name);p.unlink()
                retries[g]+=1;S['events'].append({'at':now(),'group':g,'action':'retry same command once after all group workers exited; lock/tmp evidence preserved'})
                for _,w in current:jobs.append(spawn(w['name'],w['module'],w['args'],g,d,retries[g]))
        if all(j[0].poll() is not None for j in jobs):break
        time.sleep(300)
    for d in groups.values():verify_complete(d,block)
    frozen()

def screen_specs(names,row):
    return [(f'screen_{row}_{g}','experiments.regime_screen',['evaluate','--regime',g,'--row',row,'--output-dir',SCREEN/g/row],f'{g}_{row}',SCREEN/g/row) for g in names]
def screen_readout(label):
    p=SCREEN/f'readout_{label}.json';command('screen_readout_'+label,'experiments.regime_screen',['readout','--root',SCREEN,'--output',p])
    r=read(p)
    if r['problems']:raise RuntimeError('Screen fidelity problems: '+str(r['problems']))
    return r
def screening():
    evaluate_jobs(screen_specs(GRID,'pure'),'screen_stage1_pure_running','269000-269047')
    r=screen_readout('stage1');passing=r['stage1_passing']
    if not passing and all(r['cells'][g]['pure']['clean']>36 for g in GRID):
        evaluate_jobs(screen_specs(['w3.00_r18_v0.15','w3.50_r18_v0.15'],'pure'),'screen_fallback_pure_running','269000-269047')
        r=screen_readout('fallback');passing=r['stage1_passing']
    if passing:evaluate_jobs(screen_specs(passing,'nominal'),'screen_stage2_nominal_running','269000-269047')
    r=screen_readout('final');S['selected_regime']=r['selected'];save('screen_completed');sync_phase()
    return r['selected']

def manifest(s,completed=False):
    p=MAIN/f'logs/final_{s}/manifest.json'
    if not p.exists():return None
    m=read(p)
    if m['code_commit']!=COMMIT or m['code_dirty'] or m['method']!='learned_stopping_option' or m['regime']['name']!=S['selected_regime'] or m['stopping']['stop_rule']!='value' or not m['fresh_initialization']:
        raise RuntimeError('Training manifest fidelity failed: '+str(s))
    if completed and (m['status']!='completed' or m['actual_outer_decisions']!=60000):raise RuntimeError('Training did not complete full budget: '+str(s))
    return m
def progress():
    result={}
    for s in SEEDS:
        d=MAIN/f'logs/final_{s}'; checkpoints=sorted(int(p.stem.split('_')[1]) for p in (d/'checkpoints').glob('stopping_*_outer_decisions.zip'))
        result[str(s)]={'checkpoint_outer_decisions':max(checkpoints,default=0),'manifest_status':None}
        p=d/'manifest.json'
        if p.exists():
            try:
                m=read(p);result[str(s)].update(manifest_status=m.get('status'),actual_outer_decisions=m.get('actual_outer_decisions'))
            except (ValueError,OSError):pass
    return result
def stop_training(jobs):
    for p,w in jobs:
        if p.poll() is None:
            r=subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,encoding='utf-8',errors='replace')
            S['events'].append({'at':now(),'action':'stop all training at preregistered structural gate','pid':p.pid,'returncode':r.returncode,'output':r.stdout+r.stderr})
            p.wait(timeout=30);observed((p,w))
    save()
def training():
    jobs=[spawn(f'train_{s}','train.train_stopping',['--steps',60000,'--seed',s,'--run-name',f'final_{s}','--regime',S['selected_regime'],'--device','cpu']) for s in SEEDS]
    save('formal_training_running');sync_phase();devjobs={};checks={};stable={}; gate=None
    while True:
        frozen()
        for s,job in zip(SEEDS,jobs):
            m=manifest(s);code=observed(job)
            if code is not None:
                if code:raise RuntimeError(f'Training interrupted for {s}; no resume; peer processes preserved')
                manifest(s,True)
            if m is None and time.time()-datetime.fromisoformat(job[1]['started_at']).timestamp()>180:raise RuntimeError('Training manifest missing: '+str(s))
            cp=MAIN/f'logs/final_{s}/checkpoints/stopping_30000_outer_decisions.zip'
            if cp.exists() and s not in devjobs:
                stat=(cp.stat().st_size,cp.stat().st_mtime_ns)
                if stable.get(s)==stat:
                    with zipfile.ZipFile(cp) as z:
                        if z.testzip():raise RuntimeError('Checkpoint ZIP CRC failed')
                    S.setdefault('checkpoint_sha256',{})[str(s)]=sha(cp)
                    devjobs[s]=spawn(f'dev_{s}','experiments.v3_stopping',['evaluate','--run-dir',f'logs/final_{s}','--model-name','checkpoints/stopping_30000_outer_decisions.zip','--allow-incomplete-run','--row','stopping','--seeds','266000-266047','--output-dir',DEV/str(s)])
                    save('training_with_30k_structural_evaluation')
                stable[s]=stat
            if s in devjobs and s not in checks and observed(devjobs[s]) is not None:
                if devjobs[s][1]['exit_code']:
                    w=devjobs[s][1]
                    if w['attempt']>=1:raise RuntimeError('Repeated structural evaluation failure: '+str(s))
                    backup=REC/'异常残余'/f'dev_{s}';backup.mkdir(parents=True,exist_ok=False)
                    for p in [*(DEV/str(s)).glob('*.lock'),*(DEV/str(s)).glob('*.tmp')]:shutil.copy2(p,backup/p.name);p.unlink()
                    devjobs[s]=spawn(w['name'],w['module'],w['args'],attempt=1)
                    S['events'].append({'at':now(),'action':'Retry structural evaluation after sole group worker exited; retained residues','seed':s});save();continue
                verify_complete(DEV/str(s),'266000-266047')
                monitor=MAIN/f'logs/final_{s}/train.monitor.csv'
                shutil.copy2(monitor,REC/f'devcheck_{s}_monitor_before.csv')
                command(f'devcheck_{s}','experiments.v3_stopping',['devcheck','--run-dir',f'logs/final_{s}','--eval-dir',DEV/str(s),'--output',DEV/f'devcheck_{s}.json'])
                shutil.copy2(monitor,REC/f'devcheck_{s}_monitor_after.csv')
                checks[s]=read(DEV/f'devcheck_{s}.json')
                if sha(cp)!=S['checkpoint_sha256'][str(s)]:raise RuntimeError('Frozen 30k checkpoint changed')
        S['training_progress']=progress();S['counts']={f'dev_{s}':len(list((DEV/str(s)).glob('seed_*.json'))) for s in devjobs};save()
        if len(checks)==3 and gate is None:
            d4=all(r['D4_stopping_matches_value_rule']['pass'] for r in checks.values())
            pass3=sum(all(r[k]['pass'] for k in ('D1_not_collapsed','D2_learned_reaches_mid_late','D3_state_dependent_stopping_value')) for r in checks.values())
            gate=d4 and pass3>=2;S['structural_gate']={'passed':gate,'all_D4':d4,'D1_D2_D3_passing_models':pass3};save('structural_gate_passed' if gate else 'structural_gate_failed');sync_phase()
            if not gate:stop_training(jobs);return False
        if gate and all(p.poll() is not None for p,w in jobs):
            for s in SEEDS:manifest(s,True)
            S['final_model_sha256']={str(s):sha(MAIN/f'logs/final_{s}/final_model.zip') for s in SEEDS};save('formal_training_completed');return True
        time.sleep(300)

def formal():
    save('formal_evaluation_preparing');sync_phase()
    specs=[]
    for s in SEEDS:
        for row in ('stopping','learned'):
            d=FORMAL/row/str(s);specs.append((f'formal_{row}_{s}','experiments.v3_stopping',['evaluate','--run-dir',f'logs/final_{s}','--row',row,'--seeds','271000-271047','--output-dir',d],f'{row}_{s}',d))
    for i in (1,2):specs.append((f'formal_pure_{i}','experiments.v3_stopping',['evaluate','--run-dir','logs/final_262450','--row','pure','--seeds','271000-271047','--output-dir',FORMAL/'pure'],'pure',FORMAL/'pure'))
    evaluate_jobs(specs,'formal_evaluation_blinded','271000-271047')
    for s in SEEDS:
        if sha(MAIN/f'logs/final_{s}/final_model.zip')!=S['final_model_sha256'][str(s)]:raise RuntimeError('Formal model changed')
    args=['readout','--pure',FORMAL/'pure','--seeds','271000-271047','--output',FORMAL/'readout.json']
    for s in SEEDS:args+=['--stopping',f'{s}={FORMAL}/stopping/{s}','--learned',f'{s}={FORMAL}/learned/{s}']
    command('formal_official_readout','experiments.v3_stopping',args)
    r=read(FORMAL/'readout.json');S['verdict']=r['verdict'];save('formal_readout_completed');return r

def delivery(verdict):
    save('delivery_building');build=TOPIC/'审查材料_20261006';build.mkdir(exist_ok=False)
    def copy(p,rel):
        dst=build/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        if sha(p)!=sha(dst):raise RuntimeError('Evidence copy hash mismatch')
    for data,label in ((SCREEN,'screen'),(DEV,'devcheck'),(FORMAL,'formal')):
        if data.exists():
            for p in data.rglob('*.json'):copy(p,label+'/'+p.relative_to(data).as_posix())
    for s in SEEDS:
        for name in ('manifest.json','train.monitor.csv'):
            p=MAIN/f'logs/final_{s}'/name
            if p.exists():copy(p,f'training/{s}/{name}')
    for p in REC.rglob('*'):
        if p.is_file() and p.name not in ('supervisor.lock','supervisor.stdout.log','supervisor.stderr.log') and 'pytest_tmp' not in p.parts:copy(p,'records/'+p.relative_to(REC).as_posix())
    for p in REC.glob('supervisor.*.log'):copy(p,'records/'+p.name)
    for name in ('FINAL_MAINLINE_RUN_ORDER_20261006.md','REGIME_SCREEN_20261006.md'):copy(MAIN/'docs'/name,'preregistration/'+name)
    copy(Path(__file__),'run_final_mainline.py')
    copy(COLLAB/'docs/collaboration/upper/run_orders/FINAL_MAINLINE_20261006.md','preregistration/COLLAB_RUN_ORDER.md')
    models={}
    for s in SEEDS:
        for p in (MAIN/f'logs/final_{s}').glob('*.zip'):models[str(p)]=sha(p)
        p=MAIN/f'logs/final_{s}/checkpoints/stopping_30000_outer_decisions.zip'
        if p.exists():models[str(p)]=sha(p)
    write(build/'MODEL_PATHS_SHA256.json',models)
    results={str(p.relative_to(MAIN)):read(p) for p in [*SCREEN.glob('readout_*.json'),*DEV.glob('devcheck_*.json'),FORMAL/'readout.json'] if p.exists()}
    report=('# 最终主线独立审查交付\n\n'+f'状态：{verdict}；科学提交 `{COMMIT}`；授权协作提交 `39ecee2`；选定工况 `{S["selected_regime"]}`。全部从零训练，种子262450/262451/262452，各60000外层决策；30k只作结构检查，不选择检查点。模型仅列路径与SHA，不入包。\n\n'+
        '固定方法：β=σ((Q_H−V_C)/0.005)，部署首次Q_H≥V_C单向交接；训练概率限制[0.002,0.01]；γ=0.999。物理/控制/奖励/接口未变，筛选只按预注册工况。\n\n'+
        '筛选块269000–269047：Pure干净完成29–36，失败超时占比≥2/3，违规≤2；nominal救回≥3，取最近工况。结构块266000–266047：全模型D4过、至少2模型D1–D3全过，否则停全部训练。正式块271000–271047共7行336回合：至少2模型stopping完成严格多于Pure、违规不多于Pure、毁掉Pure成功≤2。协调增益为同模型stopping相对learned；并行耗时不能作为实时性证据。\n\n'+
        '历史对照（不同块，只作背景）：267000块Pure41，learned12/31/17，stopping37/41/36，毁掉5/3/5，METHOD_DOES_NOT_HOLD；268000块Pure39、违规1，价值规则36/38/38、违规0、毁掉5/2/2，VALUE_RULE_DOES_NOT_HOLD。历史证据固定提交6c5cb97、2d8a581；不复制旧包。\n\n'+
        'PRECHECK记录工作树/代码差异、测试结果、预注册Git-LF及工作文件SHA；状态事件、命令、PID、退出码、全部日志在records。结构判读使用官方命令读取当时训练监视表，前后快照随包保留。全部本轮原始逐开局结果随包；只在组齐且进程退出后运行官方判读。无后续方法修改或新训练。\n\n'+
        '阶段事件与异常：\n```json\n'+json.dumps(S,ensure_ascii=False,indent=2)+'\n```\n\n官方读数（不另造判据）：\n```json\n'+json.dumps(results,ensure_ascii=False,indent=2)+'\n```\n')
    (build/'REPORT.md').write_text(report,encoding='utf-8')
    listing={p.relative_to(build).as_posix():sha(p) for p in sorted(build.rglob('*')) if p.is_file()};write(build/'FILES_SHA256.json',listing)
    archive=ROOT/'上层交付/待发布/FINAL_MAINLINE_20261006.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in build.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(build).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('Delivery CRC failed')
        if set(z.namelist())!=set(listing)|{'FILES_SHA256.json'}:raise RuntimeError('Delivery ZIP members mismatch')
        for n,h in listing.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('Delivery member SHA failed')
    latest=ROOT/'上层交付/最新';history=ROOT/'上层交付/历史/VALUE_RULE_CHECK_20261006'
    if history.exists():raise RuntimeError('Historical destination already exists; package retained in staging')
    if sha(latest/'VALUE_RULE_CHECK_20261006.zip')!='b5c09766ced10247b09c497ca78a68f2afab63f3a5d65542245d81ae44547882':raise RuntimeError('Latest delivery changed; do not replace')
    history.mkdir()
    for p in list(latest.iterdir()):
        dest=history/p.name
        if not p.is_file() or not dest.resolve().is_relative_to(ROOT.resolve()) or dest.exists():raise RuntimeError('Unsafe archive move')
        h=sha(p);p.rename(dest)
        if sha(dest)!=h:raise RuntimeError('Archive move SHA mismatch')
    dst=latest/archive.name;archive.rename(dst);shutil.copy2(build/'REPORT.md',latest/'REPORT.md')
    receipt={'verdict':verdict,'science_commit':COMMIT,'zip':str(dst),'sha256':sha(dst),'bytes':dst.stat().st_size,'crc_ok':True,'all_member_sha_ok':True}
    write(latest/'FINAL_MAINLINE_DELIVERY_RECEIPT.json',receipt);write(REC/'DELIVERY_RECEIPT.json',receipt)
    (latest/'README.md').write_text(f'# 最新交付：最终主线\n\n状态{verdict}；报告和独立ZIP包含本轮原始证据、读数与SHA，模型仅列路径及SHA。等待上层审查。\n',encoding='utf-8')
    S['delivery']=receipt;save('delivery_verified');return build

def publish_current(phase,review=None):
    if git(COLLAB,'status','--porcelain'):raise RuntimeError('Collaboration checkout dirty; do not overwrite')
    git(COLLAB,'pull','--ff-only','origin','collab/spacecraft');hub=COLLAB/'docs/collaboration'
    current={'schema':'spacecraft_collaboration_v1','verified_date_jst':now()[:10],'topic':'final_mainline','phase':phase,'scientific_commit':COMMIT,
        'run_order':'upper/run_orders/FINAL_MAINLINE_20261006.md','run_order_commit':S['run_order_commit'],'authority':S['authority'],
        'supervisor_state':str(STATE),'selected_regime':S['selected_regime'],'training_seeds':list(SEEDS),'outer_decisions_each':60000,
        'screen_block':'269000-269047','structural_block':'266000-266047','formal_block':'271000-271047',
        'official_verdict':S.get('verdict'),'review_url':review,'local_delivery':str(ROOT/'上层交付/最新'),
        'next_action':'自动按冻结单推进；预注册停止点停止，正式交付后等上层。'}
    write(hub/'CURRENT.json',current)
    handoff=hub/'lower/handoffs/final_mainline_20261006.md'
    handoff.write_text('# 最终主线下层交接\n\n'+f'阶段：{phase}。科学提交 `{COMMIT}` 相对 `e855bb7` 的env/controllers/dynamics/train/experiments/tests差异为空。用户已授权完整执行。状态入口 `{STATE}`；按阶段低频监督。\n\n'+(f'[独立完整审查证据]({review})；官方结论 `{S.get("verdict")}`；等待上层审查。\n' if review else '正在执行冻结工况筛选；筛选通过后自动从零训练3×60k，并于30k并行做结构门控；不读取未完成评测内容。\n'),encoding='utf-8')
    lower=hub/'lower/README.md';txt=lower.read_text(encoding='utf-8')
    line='\n最新最终主线：[下层交接](handoffs/final_mainline_20261006.md)。\n'
    if line not in txt:lower.write_text(txt+line,encoding='utf-8')
    if review:
        index=hub/'reviews/README.md'
        with index.open('a',encoding='utf-8') as f:f.write(f'\n[最终主线完整审查证据]({review})，{S.get("verdict")}。\n')
    git(COLLAB,'add','--','docs/collaboration/CURRENT.json','docs/collaboration/lower')
    if review:git(COLLAB,'add','--','docs/collaboration/reviews/README.md')
    git(COLLAB,'commit','-m','Record authorized final mainline execution and verified handoff');git(COLLAB,'push','origin','collab/spacecraft')
    if git(COLLAB,'ls-remote','origin','refs/heads/collab/spacecraft').split()[0]!=git(COLLAB,'rev-parse','HEAD'):raise RuntimeError('Collaboration remote commit mismatch')
def publish_review(build):
    repo=TOPIC/'Git审查工作树';branch='review/final-mainline-20261006'
    r=subprocess.run(['git','-c',f'safe.directory={MAIN}','clone','--shared','--no-checkout',str(MAIN),str(repo)],capture_output=True,encoding='utf-8',errors='replace')
    if r.returncode:raise RuntimeError(r.stderr)
    git(repo,'switch','-c',branch,COMMIT);git(repo,'remote','set-url','origin',REMOTE);git(repo,'config','core.autocrlf','false')
    rel='docs/reviews/final_mainline_20261006';shutil.copytree(build,repo/rel)
    git(repo,'add','--',rel);git(repo,'commit','-m','Deliver frozen final mainline raw evidence official readouts and hashes');git(repo,'push','-u','origin',branch)
    commit=git(repo,'rev-parse','HEAD')
    if git(repo,'ls-remote','origin','refs/heads/'+branch).split()[0]!=commit:raise RuntimeError('Review remote SHA mismatch')
    S.update(review_commit=commit,review_url=f'https://github.com/251614030027r-max/spacecraft/tree/{commit}/{rel}');save('review_published')
    publish_current('await_upper_review',S['review_url'])
def sync_phase():
    try:publish_current(S['phase'],S.get('review_url'))
    except Exception as e:
        S['events'].append({'at':now(),'action':'Collaboration update failed; scientific work continues unchanged','error':repr(e)});save()
def run():
    fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'at':now()},f)
    S['started_at']=now()
    try:
        precheck();save('precheck_passed')
        if not screening():S['verdict']='SCREEN_STOP'
        elif not training():S['verdict']='STRUCTURE_STOP'
        else:formal()
        build=delivery(S['verdict']);publish_review(build);save('await_upper_review')
    except Exception as e:
        S['error']=repr(e);save('execution_or_delivery_fault');print(repr(e),file=sys.stderr,flush=True)
    finally:LOCK.unlink(missing_ok=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('check','start','run','status','publish-status'));args=parser.parse_args()
    if args.mode=='status':
        if not STATE.exists():print('NOT_STARTED');return
        s=read(STATE);print('阶段:',s['phase'],'更新:',s['updated_at'],'工况:',s.get('selected_regime'));print('评测:',s.get('counts'));print('训练:',json.dumps(progress(),ensure_ascii=False));print('结论:',s.get('verdict'),'异常:',s.get('error'));return
    if args.mode=='check':precheck();print('PRECHECK_OK');return
    if args.mode=='publish-status':
        if STATE.exists():S.update(read(STATE))
        publish_current(S['phase'],S.get('review_url'));print('COLLAB_UPDATED');return
    if args.mode=='start':
        precheck()
        if STATE.exists() or LOCK.exists():raise RuntimeError('Existing supervisor state/lock; no duplicate launch')
        out=REC/'supervisor.stdout.log';err=REC/'supervisor.stderr.log'
        if out.exists() or err.exists():raise RuntimeError('Existing supervisor logs; no overwrite')
        with out.open('wb') as o,err.open('wb') as e:p=subprocess.Popen([str(PY),'-u','-B',str(Path(__file__).resolve()),'run'],cwd=MAIN,env=ENV,stdout=o,stderr=e,creationflags=0x08000000)
        write(REC/'SUPERVISOR_LAUNCH.json',{'pid':p.pid,'at':now()});print('FINAL_MAINLINE_STARTED',p.pid);return
    run()
if __name__=='__main__':main()
