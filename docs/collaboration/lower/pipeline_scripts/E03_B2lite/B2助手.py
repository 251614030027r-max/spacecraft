"""Manual status / official readout / auditable package; never starts or resumes scans."""
from __future__ import annotations
import argparse, base64, hashlib, json, os, shutil, subprocess, sys, zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path('C:/Users/35884/Documents/Spacecraft')
TOPIC = ROOT / '过程文件/阶段B2'
REC = TOPIC / '记录'
MAIN = Path('D:/py/DRL2')
SCAN = TOPIC / '扫描工作树_a714c61'
PY = MAIN / '.venv/Scripts/python.exe'
OLD = 'a714c61cc3327525ef6c103ecfcb9dc46e552a07'
NEW = 'd24b00a86d2320a3ff31ebe3841f536efdb566c4'
MODELS = ('262420', '262421', '262422')
SEEDS = set(range(262000, 262048))
OUT = REC / 'readout_b1b2.json'
BUILD = TOPIC / '交付展开'
PACKAGE = ROOT / '上层交付/待发布/STAGE_B2_LITE_262000_BLOCK.zip'
LAUNCH = REC / 'B2_FIXED_WORKTREE_LAUNCH_20261001.json'
ANCHOR = REC / 'B1_FIXED_DELIVERY_ANCHOR.json'

def now(): return datetime.now().astimezone().isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
def safe(p):
    p = Path(p).resolve()
    if not p.is_relative_to(ROOT.resolve()): raise RuntimeError(f'路径超出Spacecraft: {p}')
    return p
def ps(code):
    enc = base64.b64encode(code.encode('utf-16-le')).decode()
    r = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',enc], capture_output=True)
    if r.returncode: raise RuntimeError('进程查询/操作失败，不能推断进程已退出。'+r.stderr.decode(errors='replace')[-1200:])
    return r.stdout.decode('utf-8-sig').strip()
def processes():
    code = """$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $rows=@(Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | Where-Object { $_.CommandLine -like '*experiments.v3_handoff_scan*' } | ForEach-Object { [pscustomobject]@{pid=$_.ProcessId;parent_pid=$_.ParentProcessId;command=$_.CommandLine;cpu_s=([double]$_.KernelModeTime+[double]$_.UserModeTime)/10000000;created=[string]$_.CreationDate} }); ConvertTo-Json -InputObject $rows -Depth 4 -Compress"""
    return json.loads(ps(code) or '[]')
def is_b2(p):
    c = p.get('command') or ''
    owned = {w['pid'] for w in read(LAUNCH)['workers']}
    return (p['pid'] in owned or p.get('parent_pid') in owned) and '--scan' in c and 'successes' in c and str(MAIN/'eval/v3e/stage_b2').lower() in c.lower()
def git(root, *args):
    r = subprocess.run(['git', '-c', f'safe.directory={root}', '-C', str(root), *args], capture_output=True, text=True, encoding='utf-8')
    if r.returncode: raise RuntimeError(r.stderr)
    return r.stdout.strip()
def check_versions():
    for root, commit in ((MAIN, NEW), (SCAN, OLD)):
        if git(root,'rev-parse','HEAD') != commit: raise RuntimeError(f'版本发生变化: {root}；交上层核对，禁止自动拉取')
        if git(root,'status','--porcelain','--untracked-files=no'): raise RuntimeError(f'已跟踪工作树不干净: {root}')
    if git(MAIN, 'diff', OLD, NEW, '--', 'env','controllers','dynamics','train','experiments/v3_common.py','experiments/v3_handoff_scan.py','experiments/evaluate_hybrid_policy.py'):
        raise RuntimeError('扫描科学代码有变化')
def errors():
    fatal, notes = [], []
    for w in read(LAUNCH)['workers']:
        for key in ('stdout','stderr'):
            p = Path(w[key])
            if not p.exists(): continue
            t = p.read_text(encoding='utf-8', errors='replace')
            if any(s in t for s in ('Traceback (most recent call last)', 'RuntimeError', 'MemoryError', 'snapshot handoff differs', 'two learned passes differ')):
                fatal.append(str(p))
            elif key == 'stderr' and t.strip(): notes.append(str(p))
    return sorted(set(fatal)), notes
def status():
    if not LAUNCH.exists(): raise RuntimeError('找不到本次B2启动记录')
    proc = processes()
    fatal, notes = errors()
    snap = {'at':now(), 'processes':proc, 'fatal_logs':fatal, 'stderr_notes':notes, 'models':{}}
    previous = read(REC/'B2_STATUS_LATEST.json') if (REC/'B2_STATUS_LATEST.json').exists() else {}
    complete = True
    anomaly = bool(fatal)
    for model in MODELS:
        d = MAIN/f'eval/v3e/stage_b2/{model}'
        files = list(d.glob('seed_*.json'))
        actual = set()
        for p in files:
            try: actual.add(int(p.stem.split('_')[1]))
            except ValueError: anomaly = True
        missing, extra = sorted(SEEDS-actual), sorted(actual-SEEDS)
        locks, tmp = list(d.glob('*.lock')), list(d.glob('*.tmp'))
        workers = [p for p in proc if is_b2(p) and f'v3e_{model}' in p['command']]
        # Windows venv python.exe is a wrapper; show CPU for its real children.
        launch_ids={w['pid'] for w in read(LAUNCH)['workers']}
        compute=[p for p in workers if p.get('parent_pid') in launch_ids]
        display=compute or workers
        old = {p['pid']:p for p in previous.get('processes',[])}
        deltas = {p['pid']:round(p['cpu_s']-old[p['pid']]['cpu_s'],1) for p in display if p['pid'] in old and p['created']==old[p['pid']]['created']}
        snap['models'][model] = {'done':len(actual & SEEDS), 'missing':missing, 'extra':extra, 'locks':[p.name for p in locks], 'temporary':[p.name for p in tmp], 'pids':[p['pid'] for p in workers], 'compute_pids':[p['pid'] for p in compute], 'cpu_delta_s':deltas}
        complete &= not missing and not extra and not locks and not tmp
        anomaly |= bool(extra) or (bool(missing or locks or tmp) and not workers)
        print(f"{model}: {len(actual & SEEDS)}/48，计算PID={[p['pid'] for p in display]}，锁={len(locks)}，临时文件={len(tmp)}，CPU累计增量={deltas}")
    if read(LAUNCH).get('status') != 'started': anomaly = True
    snap['state'] = 'ATTENTION' if anomaly else ('READY' if complete and not proc else 'RUNNING')
    write(REC/'B2_STATUS_LATEST.json',snap)
    print('状态:',snap['state'])
    if fatal: print('错误日志:', '\n'.join(fatal))
    if notes: print('stderr有内容（不直接认定故障，请看日志）:', '\n'.join(notes))
    if snap['state']=='READY': print('三模型完整、无锁/临时文件、扫描进程全部退出；可运行 Finalize。')
    elif snap['state']=='ATTENTION': print('保留现场，禁止重复启动/清锁/补跑；运行 Fault 收集故障包并提交上层。')
    else: print('继续等待；失败交接重放可能数小时无新JSON。CPU增量是活跃线索，不是科学判定。')
    return snap
def anchor():
    if not ANCHOR.exists(): raise RuntimeError('B1固定入口缺失')
    a=read(ANCHOR)
    for entry in a['files']:
        if sha(entry['path'])!=entry['sha256']: raise RuntimeError('B1固定交付哈希变化: '+entry['path'])
    return a
def inputs():
    a=anchor()
    rows={}
    for e in a['files']: rows[e['path']]=e['sha256']
    receipt=read(a['receipt'])
    for m in receipt['models']:
        if sha(m['path'])!=m['sha256']: raise RuntimeError('模型哈希不符: '+m['model'])
        rows[m['path']]=m['sha256']
    for model in MODELS:
        for phase in ('stage_b','stage_b2'):
            for seed in sorted(SEEDS):
                p=MAIN/f'eval/v3e/{phase}/{model}/seed_{seed}.json'
                s=read(p)
                if s.get('code_commit')!=OLD or s.get('code_dirty') is not False: raise RuntimeError('扫描版本/工作树不符: '+str(p))
                rows[str(p)]=sha(p)
        for name in ('learned_only','m2_a','m2_b','m2_c','m2_d'):
            p=MAIN/f'eval/v3e/{model}/{name}.json'; rows[str(p)]=sha(p)
        p=MAIN/f'logs/v3e_{model}/manifest.json'; rows[str(p)]=sha(p)
    p=MAIN/'eval/v3e/pure_mpc.json'; rows[str(p)]=sha(p)
    for name in ('experiments/v3_handoff_readout.py','docs/STAGE_B_HANDOFF_WINDOW_PREREGISTRATION_20260930.md','docs/STAGE_B_RUN_ORDER_LOWER_20260930.md'):
        p=MAIN/name; rows[str(p)]=sha(p)
    return rows
def official():
    current=inputs()
    ledger=REC/'READOUT_INPUTS_SHA256.json'
    if OUT.exists():
        if not ledger.exists() or read(ledger)['inputs']!=current or read(ledger).get('readout_sha256')!=sha(OUT):
            raise RuntimeError('已有判读与输入/输出哈希不符，禁止覆盖；提交上层核对')
        print('复用已核验的官方判读，不重复计算')
        return read(OUT)
    args=[str(PY),'-B','-m','experiments.v3_handoff_readout']
    for m in MODELS:
        args+=['--scan',f'{m}={MAIN}/eval/v3e/stage_b/{m}', '--formal-learned',f'{m}={MAIN}/eval/v3e/{m}/learned_only.json', '--m2',m+'='+','.join(f'{MAIN}/eval/v3e/{m}/m2_{x}.json' for x in 'abcd'), '--b2',f'{m}={MAIN}/eval/v3e/stage_b2/{m}']
    args+=['--formal-pure',str(MAIN/'eval/v3e/pure_mpc.json'),'--seeds','262000-262047','--output',str(OUT)]
    env=os.environ.copy(); env['PYTHONPATH']=str(MAIN)
    r=subprocess.run(args,cwd=MAIN,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (REC/'official_readout.stdout.log').write_text(r.stdout,encoding='utf-8')
    (REC/'official_readout.stderr.log').write_text(r.stderr,encoding='utf-8')
    if r.returncode: raise RuntimeError('官方判读失败，查看记录/official_readout.stderr.log；禁止覆盖重跑')
    if inputs()!=current: raise RuntimeError('判读期间输入改变，停止并提交上层')
    write(ledger,{'at':now(),'args':args,'inputs':current,'readout_sha256':sha(OUT)})
    print(r.stdout)
    return read(OUT)
def report(result):
    valid=result.get('b2',{}).get('valid') is True and result['verdict']=='PROCEED' and not result['fidelity_problems']
    lines=['# Stage B2-lite 独立交付', '', f'生成时间：{now()}。B1 verdict={result["verdict"]}；B2 valid={result.get("b2",{}).get("valid")}。本轮状态：'+('核验通过，可作C设计的机制证据。' if valid else '无效/异常，只作故障证据；停止，不得补跑或用于C。'), '', 'B1固定262000–262047，三模型48×3；失败轨迹6/15/10，非退化6/14/10，仅单点0/1/0，连续窗口21/45/34。Pure MPC完成37/48；learned-only完成42/33/38。旧V3e价值第二轮does not hold不变。', '', f'扫描提交：`{OLD}`；判读提交：`{NEW}`。三个原60k模型，单块48×3，successes、stride=10、verify-prefix=middle，无截断。进程启动/日志见records和logs；终点仅记录本次确认全部退出时间，未观测到的实际退出时间不填造。并行墙钟不作为正式实时性数据。', '', '流程偏差：用户要求B1判定后才手动启动B2，主工程已更新，故建立独立固定a714c61工作树；未在d24扫描，科学代码和模型不变。启动记录包含开始时间与PID。', '', '| 模型 | 成功扫描轨迹 | 有毁掉交接的轨迹 | 毁掉状态原始数 | 毁掉连续段 | 有更快干净交接的轨迹 | 最佳节时中位数/s | 对应Δv变化中位数/(m/s) |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    keys=['successes','with_destroying_handoff','destroy_states_raw','independent_destroy_runs','with_faster_clean_handoff','median_best_time_saving_s','median_delta_v_change_at_best_m_s']
    for m in MODELS:
        summary=result.get('b2',{}).get(m,{}).get('summary',{})
        lines.append('| '+m+' | '+' | '.join('未定义' if summary.get(k) is None else str(summary[k]) for k in keys)+' |')
    lines+=['', '注意：效率行只表示更快且干净；Δv可能升或降，不能自动称“更省”。连续段按stride 10的采样点定义，不能解释成逐决策完整交接窗口，更不能视为统计独立轨迹。原始状态高度相关。', '', '## 精确性问题（全文）', '```json',json.dumps({'b1':result['fidelity_problems'],'b2':result.get('b2',{}).get('fidelity_problems',[])},ensure_ascii=False,indent=2),'```', '', '## 证据与下一步', 'readout_b1b2.json为官方读数；scans_b2为本轮144个原始JSON；comparison_b1是核对逐位学习轨迹所必需的144个固定B1 JSON；inputs为精确性对照、模型manifest与SHA（不装模型ZIP）。B1历史交付固定路径/哈希见B1_FIXED_DELIVERY_ANCHOR.json，不嵌旧ZIP。', '', '本轮只补反方向样本及效率线索，不改变B1判定，不进论文性能表。b2.valid=false时全部统计不可用于C，先向上层报告故障；有效时交上层讨论C目标、独立样本单位、校准和新块验证，C拟合/D重训/E正式评估均须新预注册与用户授权。']
    return '\n'.join(lines)+'\n', valid
def copyfile(src, dst):
    dst=safe(dst); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    if sha(src)!=sha(dst): raise RuntimeError('复制哈希不符: '+str(src))
def verify_package():
    package=PACKAGE if PACKAGE.exists() else ROOT/'上层交付/最新'/PACKAGE.name
    rp=PACKAGE.with_suffix('.receipt.json') if PACKAGE.exists() else ROOT/'上层交付/最新/STAGE_B2_DELIVERY_RECEIPT.json'
    receipt=read(rp)
    if sha(package)!=receipt['zip_sha256']: raise RuntimeError('ZIP哈希不符')
    with zipfile.ZipFile(package) as z:
        if z.testzip(): raise RuntimeError('ZIP CRC不符')
        listing=json.loads(z.read('FILES_SHA256.json'))
        expected=set(listing)|{'FILES_SHA256.json'}
        if len(z.namelist())!=len(expected) or set(z.namelist())!=expected: raise RuntimeError('ZIP成员集合不符')
        for name,h in listing.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=h: raise RuntimeError('包内SHA不符: '+name)
    print('ZIP_CRC_AND_ALL_SHA_OK:',package)
    return receipt
def finalize():
    if status()['state']!='READY': raise RuntimeError('未达到READY，不允许判读或打包')
    check_versions()
    result=official()
    if (ROOT/'上层交付/最新/STAGE_B2_DELIVERY_RECEIPT.json').exists():
        receipt=verify_package()
        if receipt['readout_sha256']!=sha(OUT): raise RuntimeError('已发布ZIP与读数不符')
        print('本轮已判读、打包并发布，无需重复执行。'); return
    if PACKAGE.exists():
        receipt=verify_package()
        if receipt['readout_sha256']!=sha(OUT): raise RuntimeError('已有ZIP与读数不一致')
        print('交付包已存在且核验通过；运行 Publish 或手动提交上层。'); return
    if BUILD.exists(): raise RuntimeError('交付展开已存在但ZIP未完成；保留现场，使用Fault交上层核对，不自动覆盖')
    safe(BUILD).mkdir(parents=True)
    text,valid=report(result)
    (BUILD/'REPORT.md').write_text(text,encoding='utf-8')
    copyfile(OUT,BUILD/'readout_b1b2.json')
    copyfile(ANCHOR,BUILD/'B1_FIXED_DELIVERY_ANCHOR.json')
    copyfile(read(ANCHOR)['readout'],BUILD/'readout_b1.json')
    copyfile(read(ANCHOR)['receipt'],BUILD/'B1_DELIVERY_RECEIPT.json')
    for m in MODELS:
        for seed in sorted(SEEDS):
            copyfile(MAIN/f'eval/v3e/stage_b2/{m}/seed_{seed}.json', BUILD/f'scans_b2/{m}/seed_{seed}.json')
            copyfile(MAIN/f'eval/v3e/stage_b/{m}/seed_{seed}.json', BUILD/f'comparison_b1/{m}/seed_{seed}.json')
        for name in ('learned_only','m2_a','m2_b','m2_c','m2_d'):
            copyfile(MAIN/f'eval/v3e/{m}/{name}.json',BUILD/f'inputs/{m}/{name}.json')
        copyfile(MAIN/f'logs/v3e_{m}/manifest.json',BUILD/f'inputs/{m}/manifest.json')
    copyfile(MAIN/'eval/v3e/pure_mpc.json',BUILD/'inputs/pure_mpc.json')
    for p in REC.glob('*'):
        if p.is_file() and p.suffix in ('.json','.log'): copyfile(p,BUILD/('logs' if p.suffix=='.log' else 'records')/p.name)
    for p in (TOPIC/'脚本').glob('*'):
        if p.is_file(): copyfile(p,BUILD/'tools'/p.name)
    for name in ('docs/STAGE_B_HANDOFF_WINDOW_PREREGISTRATION_20260930.md','docs/STAGE_B_RUN_ORDER_LOWER_20260930.md','experiments/v3_handoff_readout.py'):
        copyfile(MAIN/name,BUILD/'frozen_readout'/name)
    raw_listing={p.relative_to(BUILD).as_posix():sha(p) for p in sorted(BUILD.rglob('*')) if p.is_file()}
    (BUILD/'JSON_SHA256.txt').write_text(''.join(f'{h}  {name}\n' for name,h in raw_listing.items() if name.endswith('.json')),encoding='utf-8')
    (BUILD/'ALL_FILES_SHA256.txt').write_text(''.join(f'{h}  {name}\n' for name,h in raw_listing.items()),encoding='utf-8')
    listing={p.relative_to(BUILD).as_posix():sha(p) for p in sorted(BUILD.rglob('*')) if p.is_file()}
    write(BUILD/'FILES_SHA256.json',listing)
    safe(PACKAGE).parent.mkdir(parents=True,exist_ok=True)
    temporary=PACKAGE.with_suffix('.tmp.zip')
    if temporary.exists(): raise RuntimeError('临时ZIP已存在，保留现场供核对')
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUILD.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(BUILD).as_posix())
    temporary.rename(PACKAGE)
    write(PACKAGE.with_suffix('.receipt.json'),{'at':now(),'status':'verified' if valid else 'invalid_evidence_only','b1_verdict':result['verdict'],'b2_valid':valid,'zip_sha256':sha(PACKAGE),'zip_bytes':PACKAGE.stat().st_size,'readout_sha256':sha(OUT),'scan_commit':OLD,'readout_commit':NEW,'anchor_sha256':sha(ANCHOR)})
    verify_package()
    print('包已生成:',PACKAGE, '\nB2_VALID=',valid)
    if not valid: print('B2核验未过：提交上层故障证据，禁止补跑或用于C。')
def publish():
    receipt=verify_package()
    latest=safe(ROOT/'上层交付/最新')
    marker=latest/'STAGE_B2_DELIVERY_RECEIPT.json'
    if marker.exists() and read(marker).get('zip_sha256')==receipt['zip_sha256']:
        if sha(latest/PACKAGE.name)!=receipt['zip_sha256']: raise RuntimeError('已发布ZIP哈希不符')
        print('已发布，无需重复轮换'); return
    a=anchor()
    original=read(a['receipt'])
    current=latest/'STAGE_B1_DELIVERY_RECEIPT_20261001.json'
    if not current.exists() or read(current)['zip_sha256']!=original['zip_sha256']:
        raise RuntimeError('最新目录已换轮或不符B1固定入口，停止发布；待发布ZIP仍可直接交上层')
    if any(p.is_dir() for p in latest.iterdir()): raise RuntimeError('最新目录含目录，拒绝自动搬移')
    history=safe(ROOT/'上层交付/历史/STAGE_B1_262000_BLOCK_20261001')
    moves=[(safe(p),safe(history/p.name)) for p in latest.iterdir()]
    if history.exists() or any(dst.exists() for _,dst in moves): raise RuntimeError('历史目标已存在，禁止覆盖')
    # Check all final paths before moving any closed delivery assets.
    hashes={src.name:sha(src) for src,_ in moves}
    history.mkdir(parents=True)
    for src,dst in moves:
        src.rename(dst)
        if sha(dst)!=hashes[src.name]: raise RuntimeError('历史归档哈希变化')
    copyfile(BUILD/'REPORT.md',latest/'REPORT.md')
    copyfile(OUT,latest/'readout_b1b2.json')
    safe(PACKAGE).rename(safe(latest/PACKAGE.name))
    copyfile(PACKAGE.with_suffix('.receipt.json'),marker)
    (latest/'README.md').write_text('# 当前交付：Stage B2-lite\n\n状态：'+receipt['status']+'；B1 PROCEED，B2 valid='+str(receipt['b2_valid'])+'。先读REPORT.md和readout_b1b2.json，完整原始证据见'+PACKAGE.name+'，ZIP哈希见STAGE_B2_DELIVERY_RECEIPT.json。旧B1归档于'+str(history)+'。此前V3e价值仲裁does not hold不变。\n',encoding='utf-8')
    idx=ROOT/'上层交付/历史/README.md'
    with idx.open('a',encoding='utf-8') as f: f.write('\n- Stage B1，PROCEED，原262000块交付，归档：'+history.name+'，ZIP SHA256 '+original['zip_sha256']+'。\n')
    if sha(latest/PACKAGE.name)!=receipt['zip_sha256']: raise RuntimeError('发布后ZIP哈希不符')
    print('PUBLISHED:',latest, '\n提交上层：REPORT.md、readout_b1b2.json、STAGE_B2_DELIVERY_RECEIPT.json及ZIP。')
def fault():
    try: status()
    except Exception as e: write(REC/'FAULT_STATUS_QUERY_ERROR.json',{'at':now(),'error':str(e)})
    target=safe(TOPIC/'故障包'/('B2_FAULT_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip'))
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in REC.iterdir():
            if p.is_file(): z.write(p,'records/'+p.name)
        for p in (TOPIC/'接续').glob('*.md'): z.write(p,'context/'+p.name)
    print('故障包（不含未完成科学结果）:',target,'SHA256=',sha(target))
def stop_on_error():
    fatal,_=errors()
    if not fatal: raise RuntimeError('未发现明确错误日志；本脚本不因进度慢停止任务')
    owned=','.join(str(int(w['pid'])) for w in read(LAUNCH)['workers'])
    ps("$ErrorActionPreference='Stop'; $owned=@("+owned+"); $matched=@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { ($_.ProcessId -in $owned -or $_.ParentProcessId -in $owned) -and $_.CommandLine -like '*experiments.v3_handoff_scan*' -and $_.CommandLine -like '*--scan*successes*' -and $_.CommandLine -like '*stage_b2*' }); $matched | Sort-Object @{Expression={if ($_.ParentProcessId -in $owned) {0} else {1}}} | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }")
    print('仅停止本工作树B2错误运行；锁与原始结果保留，禁止清锁/补跑。')
    fault()
def selftest():
    # Exercise archive CRC/SHA detection in an isolated temporary test folder.
    import tempfile
    global PACKAGE
    original=PACKAGE
    try:
        with tempfile.TemporaryDirectory(dir=REC,prefix='package_qa_') as d:
            PACKAGE=Path(d)/'test.zip'
            data=b'known evidence\n'; h=hashlib.sha256(data).hexdigest()
            with zipfile.ZipFile(PACKAGE,'x',zipfile.ZIP_DEFLATED) as z:
                z.writestr('raw.json',data)
                z.writestr('FILES_SHA256.json',json.dumps({'raw.json':h}))
            rp=PACKAGE.with_suffix('.receipt.json')
            write(rp,{'zip_sha256':sha(PACKAGE)})
            verify_package()
            write(rp,{'zip_sha256':'0'*64})
            try: verify_package()
            except RuntimeError: pass
            else: raise AssertionError('未拒绝错误ZIP哈希')
            with zipfile.ZipFile(PACKAGE,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('raw.json',data+b'changed')
                z.writestr('FILES_SHA256.json',json.dumps({'raw.json':h}))
            write(rp,{'zip_sha256':sha(PACKAGE)})
            try: verify_package()
            except RuntimeError: pass
            else: raise AssertionError('未拒绝包内原始文件变化')
    finally: PACKAGE=original
    saved_status=globals()['status']; saved_official=globals()['official']
    try:
        globals()['status']=lambda: {'state':'RUNNING'}
        def forbidden_readout(): raise AssertionError('尚在运行却尝试判读')
        globals()['official']=forbidden_readout
        try: finalize()
        except RuntimeError as e: assert 'READY' in str(e)
        else: raise AssertionError('未拒绝未完成实验的判读')
    finally:
        globals()['status']=saved_status; globals()['official']=saved_official
    check_versions(); anchor()
    print('SELFTEST_OK: 版本/B1锚点/ZIP CRC/SHA/篡改拒绝/未完成判读保护通过；未正式判读、未发布、未停止进程。')
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['Status','Finalize','Verify','Publish','Fault','StopOnError','SelfTest']); args=parser.parse_args()
    REC.mkdir(parents=True,exist_ok=True)
    try:
        {'Status':status,'Finalize':finalize,'Verify':verify_package,'Publish':publish,'Fault':fault,'StopOnError':stop_on_error,'SelfTest':selftest}[args.mode]()
    except Exception as e:
        write(REC/'B2_HELPER_LAST_ERROR.json',{'at':now(),'mode':args.mode,'error':str(e)})
        print('STOP:',e,'\n现场保留。不要自动重跑/清锁；把本错误及Fault包交上层。',file=sys.stderr); return 1
    return 0
if __name__=='__main__': sys.exit(main())
