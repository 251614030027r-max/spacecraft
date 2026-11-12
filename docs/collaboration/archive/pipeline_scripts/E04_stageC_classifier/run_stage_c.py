"""Frozen Stage C queue; never trains SAC or inspects C3 before all rows finish."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from datetime import datetime
from pathlib import Path

ROOT=Path('C:/Users/35884/Documents/Spacecraft'); TOPIC=ROOT/'过程文件/阶段C'
REC=TOPIC/'记录'; CODE=TOPIC/'脚本'; STATE=REC/'STAGE_C_EXECUTION.json'
MAIN=Path('D:/py/DRL2'); PY=MAIN/'.venv/Scripts/python.exe'; DATA=MAIN/'eval/v3e/stage_c'
COMMIT='e6694268634be63d5d66a4da637128ac2c5123a1'; MODELS=['262420','262421','262422']
LOCK=REC/'supervisor.lock'; ACTIVE=[]
ENV=os.environ.copy(); ENV.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(MAIN),PYTHONIOENCODING='utf-8')
S={'schema':'stage_c_supervisor_v1','code_commit':COMMIT,'repo':str(MAIN),'python':str(PY),'supervisor_pid':os.getpid(),'started_at':datetime.now().astimezone().isoformat(),'phase':'initializing','workers':[],'events':[]}
def now(): return datetime.now().astimezone().isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8'); os.replace(t,p)
def state(phase=None):
    if phase and phase!=S['phase']:
        S['phase']=phase; S['events'].append({'at':now(),'phase':phase}); print(now(),phase,flush=True)
    S['updated_at']=now(); write(STATE,S)
def git(*args):
    r=subprocess.run(['git','-c',f'safe.directory={MAIN}','-C',str(MAIN),*args],capture_output=True,text=True,encoding='utf-8')
    if r.returncode: raise RuntimeError(r.stderr)
    return r.stdout.strip()
def clean():
    if git('rev-parse','HEAD')!=COMMIT or git('status','--porcelain','--untracked-files=no'): raise RuntimeError('固定提交/干净工作树核对失败，禁止继续')
def preparation():
    clean()
    diff=git('diff','a714c61',COMMIT,'--','env','controllers','dynamics','train','experiments/v3_common.py','experiments/v3_handoff_scan.py','experiments/evaluate_hybrid_policy.py')
    if diff: raise RuntimeError('科学系统相对B阶段发生改动')
    latest=ROOT/'上层交付/最新'; receipt=read(latest/'STAGE_B2_DELIVERY_RECEIPT.json'); archive=latest/'STAGE_B2_LITE_262000_BLOCK.zip'
    if sha(archive)!=receipt['zip_sha256']: raise RuntimeError('B2包SHA错误')
    audit=[]
    with zipfile.ZipFile(archive) as z:
        if z.testzip(): raise RuntimeError('B2包CRC错误')
        hashes=json.loads(z.read('FILES_SHA256.json'))
        for m in MODELS:
            for phase, member in [('stage_b','comparison_b1'),('stage_b2','scans_b2')]:
                for seed in range(262000,262048):
                    rel=f'{member}/{m}/seed_{seed}.json'; p=MAIN/f'eval/v3e/{phase}/{m}/seed_{seed}.json'
                    if sha(p)!=hashes[rel] or hashlib.sha256(z.read(rel)).hexdigest()!=hashes[rel]: raise RuntimeError('B1/B2原始证据不符: '+str(p))
                    audit.append({'path':str(p),'sha256':hashes[rel]})
            p=MAIN/f'logs/v3e_{m}/final_model.zip'; expected=read(REC.parent.parent/'阶段B2/记录/B1_FROZEN_RECEIPT.json')
            ref=next(x for x in expected['models'] if x['model']==m)
            if sha(p)!=ref['sha256']: raise RuntimeError('模型SHA不符')
            audit.append({'path':str(p),'sha256':ref['sha256']})
    if DATA.exists() and any(DATA.iterdir()): raise RuntimeError('Stage C结果目录非空，禁止覆盖或重复启动')
    DATA.mkdir(parents=True,exist_ok=True)
    testlog=(REC/'pytest_preflight.log').read_text(encoding='utf-8',errors='replace')
    if 'passed' not in testlog or 'FAILED' in testlog or 'ERROR' in testlog: raise RuntimeError('指定测试未全部通过')
    write(REC/'PRECHECK.json',{'at':now(),'commit':COMMIT,'scientific_diff_empty':True,'tests':testlog.strip().splitlines()[-1],'b2_zip_sha256':receipt['zip_sha256'],'b2_already_in_main':True,'copy_not_needed':True,'verified_inputs':audit,'prereg_sha256':sha(MAIN/'docs/STAGE_C_PREREGISTRATION_20261002.md'),'run_order_sha256':sha(MAIN/'docs/STAGE_C_RUN_ORDER_LOWER_20261002.md')})
def spawn(name,args,group=None):
    out=REC/(name+'.stdout.log'); err=REC/(name+'.stderr.log')
    if out.exists() or err.exists(): raise RuntimeError('日志已存在: '+name)
    command=[str(PY),'-u','-B','-m','experiments.v3_stage_c',*args]
    with out.open('wb') as o,err.open('wb') as e:
        p=subprocess.Popen(command,cwd=MAIN,env=ENV,stdout=o,stderr=e,creationflags=0x08000000)
    w={'name':name,'pid':p.pid,'at':now(),'command':command,'stdout':str(out),'stderr':str(err),'group':group,'exit_code':None}
    S['workers'].append(w); ACTIVE.append((p,w)); state(); return p,w
def finish(p,w):
    w['exit_code']=p.returncode; w['exit_observed_at']=now(); state()
    if p.returncode: raise RuntimeError(f"{w['name']}退出码{p.returncode}；保留现场，不盲续跑")
def monitor(batch,phase):
    state(phase)
    while batch:
        for p,w in list(batch):
            if p.poll() is not None: finish(p,w); batch.remove((p,w))
        state()
        if batch: time.sleep(30)
def stop_children():
    # Windows venv launcher has a real Python child; terminate only our worker trees.
    for p,w in ACTIVE:
        if p.poll() is None:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
            w['stopped_on_error']=now()
def fit():
    args=['fit']
    for m in MODELS:
        args+=['--record',f'{m}={DATA}/c1_{m}.npz','--b1',f'{m}={MAIN}/eval/v3e/stage_b/{m}','--b2',f'{m}={MAIN}/eval/v3e/stage_b2/{m}']
    args+=['--output-dir',str(DATA/'c2')]
    monitor([spawn('c2_fit',args)],'c2_running')
    result=read(DATA/'c2/c2_report.json'); S['c2_verdict']=result['c2_verdict']; S['tau']=result['tau']; state()
    return result
def evaluate():
    clean(); classifier=DATA/'c2/handoff_classifier.pt'; S['classifier_sha256']=sha(classifier)
    batch=[]
    for m in MODELS:
        for row in ['hybrid','learned']:
            args=['evaluate','--run-dir',str(MAIN/f'logs/v3e_{m}'),'--row',row,'--output-dir',str(DATA/f'c3/{row}/{m}')]
            if row=='hybrid': args+=['--classifier',str(classifier)]
            batch.append(spawn(f'c3_{row}_{m}',args,{'row':row,'model':m}))
    pending=['pure_1','pure_2']; state('c3_running_blinded')
    lastcount=0
    while batch or pending:
        for p,w in list(batch):
            if p.poll() is not None: finish(p,w); batch.remove((p,w))
        while pending and len(batch)<6:
            name=pending.pop(0)
            batch.append(spawn('c3_'+name,['evaluate','--run-dir',str(MAIN/'logs/v3e_262420'),'--row','pure','--output-dir',str(DATA/'c3/pure')],{'row':'pure','model':None}))
        if time.monotonic()-lastcount>=300:
            # Counts only: never open C3 result content before all seven groups finish.
            dirs={'pure':DATA/'c3/pure'}
            dirs.update({f'{row}_{m}':DATA/f'c3/{row}/{m}' for row in ['hybrid','learned'] for m in MODELS})
            S['c3_counts']={k:len(list(d.glob('seed_*.json'))) for k,d in dirs.items()}; lastcount=time.monotonic()
        state()
        if batch: time.sleep(30)
    clean()
    if sha(classifier)!=S['classifier_sha256']: raise RuntimeError('C3期间判断器发生变化')
    # Additional provenance checks supplement, never replace, the official readout.
    for row in ['pure','hybrid','learned']:
        for m in ([None] if row=='pure' else MODELS):
            d=DATA/'c3/pure' if m is None else DATA/f'c3/{row}/{m}'
            expected={f'seed_{s}.json' for s in range(266000,266048)}
            if {p.name for p in d.glob('seed_*.json')}!=expected or list(d.glob('*.lock')) or list(d.glob('*.tmp')): raise RuntimeError('C3开局集合/锁不符: '+str(d))
            for p in d.glob('seed_*.json'):
                r=read(p)
                if r['code_commit']!=COMMIT or r['code_dirty']: raise RuntimeError('C3版本不一致')
                if row=='hybrid' and r['classifier_sha256']!=S['classifier_sha256']: raise RuntimeError('C3判断器不一致')
                if m and r['model_sha256']!=sha(MAIN/f'logs/v3e_{m}/final_model.zip'): raise RuntimeError('C3模型不一致')
    args=['readout','--pure',str(DATA/'c3/pure')]
    for m in MODELS: args+=['--learned',f'{m}={DATA}/c3/learned/{m}','--hybrid',f'{m}={DATA}/c3/hybrid/{m}']
    args+=['--output',str(DATA/'c3_readout.json')]
    monitor([spawn('c3_official_readout',args)],'official_readout')
    result=read(DATA/'c3_readout.json'); S['verdict']=result['verdict']; state(); return result
def path_safe(p):
    p=Path(p).resolve()
    if not p.is_relative_to(ROOT.resolve()): raise RuntimeError('路径超出Spacecraft: '+str(p))
    return p
def copy(src,dst):
    dst=path_safe(dst); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    if sha(src)!=sha(dst): raise RuntimeError('复制哈希不符')
def delivery(c2,result=None):
    verdict=S.get('verdict',S.get('c2_verdict','EXECUTION_ERROR')); stamp=datetime.now().strftime('%Y%m%d')
    build=TOPIC/'交付展开'; build.mkdir(exist_ok=False)
    report=['# Stage C 独立交付', '', f'生成时间：{now()}。官方判定：{verdict}；执行提交：{COMMIT}。', '', '本轮固定31维core/target_attitude/remaining_time，三模型池化、每轨迹等权，按开局6折，5个31→64→64→1网络、400轮；τ由预注册最小网格值且折外加权精度≥95%确定。只此一次拟合，不调参、不加数据。C3固定266000–266047，7行共336回合。效率要求消除相对Pure额外时间与Δv各至少一半，违规回合不多于Pure，毁掉Pure成功≤2；至少2模型通过。', '', '必要历史对照：B块Pure 37/48、learned-only 42/33/38；B1非退化6/14/10，PROCEED。B2有效，毁掉交接轨迹18/42、10/33、18/38；节时中位59.7/67/57.5秒，对应Δv变化中位-0.384/-0.510/-0.307 m/s。原始状态相关，事后交接最优点不是在线收益。V3e价值第二轮does not hold不变。', '', 'B2已在主工程，以已核验ZIP的FILES_SHA256逐项比对144个B2与144个B1原始JSON，无需重复复制；PRECHECK中列出哈希、模型SHA、指定测试结果和预注册SHA。', '', '## C2官方报告', '```json',json.dumps(c2,ensure_ascii=False,indent=2),'```']
    if result:
        report+=['','## C3官方判读','```json',json.dumps(result,ensure_ascii=False,indent=2),'```']
    report+=['','## 边界与执行记录','events及全部进程起止、日志见记录；C3完成前仅数文件与核对进程退出，未读中间结果。阶段并行墙钟不作正式实时性数据。没有自动启动E训练：用户最新指令及预注册要求用户启动；A仅准备训练指令后待用户，B等待上层改接口，C_STOP/FIDELITY_FAIL停止。']
    (build/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    for p in DATA.rglob('*'):
        if p.is_file(): copy(p,build/'results'/p.relative_to(DATA))
    for p in REC.iterdir():
        if p.is_file() and p.name!='supervisor.lock': copy(p,build/'records'/p.name)
    for p in CODE.iterdir():
        if p.is_file(): copy(p,build/'tools'/p.name)
    for name in ['docs/STAGE_C_PREREGISTRATION_20261002.md','docs/STAGE_C_RUN_ORDER_LOWER_20261002.md','experiments/v3_stage_c.py','tests/test_v3_stage_c.py']: copy(MAIN/name,build/'frozen_code'/name)
    # Raw labels / observations required to independently review this fitting round.
    for m in MODELS:
        for phase in ['stage_b','stage_b2']:
            for seed in range(262000,262048): copy(MAIN/f'eval/v3e/{phase}/{m}/seed_{seed}.json',build/f'inputs/{phase}/{m}/seed_{seed}.json')
        copy(MAIN/f'logs/v3e_{m}/manifest.json',build/f'inputs/manifests/{m}.json')
    listing={p.relative_to(build).as_posix():sha(p) for p in sorted(build.rglob('*')) if p.is_file()}
    for filename, subset in [('ALL_FILES_SHA256.txt',listing),('JSON_SHA256.txt',{k:v for k,v in listing.items() if k.endswith('.json')})]:
        (build/filename).write_text(''.join(f'{h}  {k}\n' for k,h in subset.items()),encoding='utf-8')
    listing={p.relative_to(build).as_posix():sha(p) for p in sorted(build.rglob('*')) if p.is_file()}; write(build/'FILES_SHA256.json',listing)
    target=ROOT/f'上层交付/待发布/STAGE_C_{stamp}.zip'; target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(build.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(build).as_posix())
    with zipfile.ZipFile(target) as z:
        if z.testzip(): raise RuntimeError('C包CRC不符')
        if set(z.namelist())!=set(listing)|{'FILES_SHA256.json'}: raise RuntimeError('C包成员不符')
        for k,h in listing.items():
            if hashlib.sha256(z.read(k)).hexdigest()!=h: raise RuntimeError('C包内SHA不符')
    receipt={'at':now(),'status':'verified','verdict':verdict,'code_commit':COMMIT,'zip_sha256':sha(target),'zip_bytes':target.stat().st_size,'crc_ok':True,'all_members_sha256_ok':True}
    write(REC/'STAGE_C_DELIVERY_RECEIPT.json',receipt)
    latest=path_safe(ROOT/'上层交付/最新'); old=read(latest/'STAGE_B2_DELIVERY_RECEIPT.json')
    if old['zip_sha256']!=read(REC/'PRECHECK.json')['b2_zip_sha256']: raise RuntimeError('最新交付已换轮，保留C待发布包，不自动归档')
    if any(p.is_dir() for p in latest.iterdir()): raise RuntimeError('最新目录包含目录，停止自动发布')
    history=path_safe(ROOT/'上层交付/历史/STAGE_B2_LITE_20261002')
    moves=[(path_safe(p),path_safe(history/p.name),sha(p)) for p in latest.iterdir()]
    if history.exists(): raise RuntimeError('历史目标已存在，禁止覆盖')
    history.mkdir(parents=True)
    for src,dst,h in moves:
        src.rename(dst)
        if sha(dst)!=h: raise RuntimeError('历史移动SHA变化')
    copy(build/'REPORT.md',latest/'REPORT.md'); copy(DATA/'c2/c2_report.json',latest/'c2_report.json')
    if result: copy(DATA/'c3_readout.json',latest/'c3_readout.json')
    copy(REC/'STAGE_C_DELIVERY_RECEIPT.json',latest/'STAGE_C_DELIVERY_RECEIPT.json')
    target.rename(path_safe(latest/target.name))
    (latest/'README.md').write_text(f'# 当前交付：Stage C（{stamp}）\n\n官方判定：{verdict}。先读REPORT.md、c2_report.json及存在时的c3_readout.json；独立原始证据包{target.name}，ZIP SHA256 {receipt["zip_sha256"]}。核验回执STAGE_C_DELIVERY_RECEIPT.json。旧B2位于历史/STAGE_B2_LITE_20261002。A待用户启动新训练；B等上层改接口；其他判定停止。\n',encoding='utf-8')
    with (ROOT/'上层交付/历史/README.md').open('a',encoding='utf-8') as f: f.write('\n- Stage B2-lite，2026-10-02，有效，归档STAGE_B2_LITE_20261002，SHA256 '+old['zip_sha256']+'。\n')
    S['delivery']=str(latest); S['zip_sha256']=receipt['zip_sha256']; state()
def run():
    preparation(); state('c1_running')
    batch=[spawn('c1_'+m,['record','--run-dir',str(MAIN/f'logs/v3e_{m}'),'--b1-dir',str(MAIN/f'eval/v3e/stage_b/{m}'),'--output',str(DATA/f'c1_{m}.npz')]) for m in MODELS]
    monitor(batch,'c1_running'); clean()
    for m in MODELS:
        meta=read(DATA/f'c1_{m}.json')
        if meta['code_commit']!=COMMIT or meta['code_dirty'] or sha(DATA/f'c1_{m}.npz')!=meta['npz_sha256']: raise RuntimeError('C1来源/哈希不符')
    c2=fit(); result=None
    if c2['c2_verdict']=='TAU_FOUND': result=evaluate()
    else: S['verdict']='C_STOP'; state()
    delivery(c2,result)
    verdict=S['verdict']; stage={'A_GO_TO_E':'await_user_training','B_GO_TO_D':'await_upper_interface','C_STOP':'stopped_c_stop','FIDELITY_FAIL':'stopped_fidelity'}[verdict]
    if verdict=='A_GO_TO_E':
        (TOPIC/'接续/E训练启动指令.md').write_text('A_GO_TO_E：按用户最新指令由用户启动新训练。冻结提交e669426，种子262430/262431/262432，各60k；先读取本轮报告，训练期间不拉代码。具体手动脚本将在交付后由本窗口核验提供，当前尚未启动训练。\n',encoding='utf-8')
    state(stage)
    (TOPIC/'接续/C阶段最终状态.md').write_text(f'# Stage C阶段收口\n\n{now()}，官方判定{verdict}，阶段{stage}。独立交付{S["delivery"]}，ZIP SHA256 {S["zip_sha256"]}。未启动E训练，未修改接口，等待预注册规定的用户/上层下一动作。\n',encoding='utf-8')
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare-only',action='store_true'); args=parser.parse_args()
    REC.mkdir(parents=True,exist_ok=True)
    if args.prepare_only: preparation(); print('PREPARE_OK: 科学系统/测试/288原始JSON/模型SHA通过；未启动C1'); return
    if STATE.exists(): raise RuntimeError('状态已存在，禁止重复启动；先核对现场')
    fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try: run()
    except Exception as e:
        stop_children(); S['error']=str(e); state('execution_error'); raise
    finally:
        if LOCK.exists(): LOCK.unlink()
if __name__=='__main__': main()
