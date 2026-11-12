"""A-only inventory. Metadata for active paths, never mutate scientific checkouts."""
import os, csv, json, hashlib, subprocess, ctypes, re
os.environ['GIT_OPTIONAL_LOCKS']='0'
from pathlib import Path
from datetime import datetime, timezone
BASE=Path(r'C:\Users\35884\Documents\Spacecraft');TOP=BASE/'过程文件/整理_20261010';OUT=TOP/'交付'
OUT.mkdir(parents=True,exist_ok=True)
COLLAB=BASE/'过程文件/协作/Git工作树';MAIN=Path('D:/py/DRL2')
api=ctypes.WinDLL('kernel32');api.GetCurrentProcess.restype=ctypes.c_void_p;api.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32];assert api.SetPriorityClass(api.GetCurrentProcess(),0x4000)
def emit(n,rs,fields=None):
    rs=list(rs)
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rs[0]) if rs else fields or ['empty']);w.writeheader();w.writerows(rs)
def git(root,*args):
    return subprocess.check_output(['git','-c',f'safe.directory={root.as_posix()}','-C',str(root),*args],stderr=subprocess.STDOUT,encoding='utf-8',errors='replace').strip()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def is_active(p):
    s=str(p).replace('\\','/').lower()
    return '/eval/final2/' in s or s.endswith('/eval/final2') or '/logs/final2_262461' in s or ('/最终主线重训/记录/' in s and any(x in p.name for x in ['formal_','priority_','PRIORITY_EXECUTION','FORMAL_TWO_EXECUTION','train_262461']))
SKIP={'.git','.venv','venv','node_modules','__pycache__','.pytest_cache'}
errors=[];excluded=[]
def walk(root):
    if not root.exists():return []
    out=[]
    for d,dirs,files in os.walk(root,onerror=lambda e:errors.append(str(e))):
        for name in list(dirs):
            if name in SKIP:excluded.append(str(Path(d)/name));dirs.remove(name)
        for name in files:
            p=Path(d)/name
            try:
                st=p.stat();out.append((p,st.st_size,st.st_mtime))
            except OSError as e:errors.append(str(e))
    return out

# Discover all Git checkouts in project-owned local roots, pruning dependencies.
repos=set()
for root in (Path('D:/py'),BASE):
    for d,dirs,files in os.walk(root,onerror=lambda e:errors.append(str(e))):
        p=Path(d)
        if '.git' in dirs or '.git' in files:repos.add(p)
        dirs[:]=[x for x in dirs if x not in SKIP and x not in ('logs','eval','记录','reports','plot_data','inputs')]
for root in list(repos):
    try:
        for line in git(root,'worktree','list','--porcelain').splitlines():
            if line.startswith('worktree '):repos.add(Path(line[9:]))
    except Exception as e:errors.append(str(e))
snapshots=[];tracked={};indexes={}
for i,root in enumerate(sorted(repos)):
    commands={'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],
              'status':['status','--porcelain','--untracked-files=all'],'stash':['stash','list'],
              'worktrees':['worktree','list','--porcelain'],'remotes':['remote','-v'],'branches':['branch','-vv']}
    snap={'path':str(root)}
    for k,a in commands.items():
        try:snap[k]=git(root,*a)
        except Exception as e:snap[k]='ERROR '+str(e)
    snapshots.append(snap)
    try:tracked[root]=set(git(root,'ls-files','-z').split('\0'))
    except Exception:tracked[root]=set()
    indexes[root]=walk(root)
    print('INDEXED',root,len(indexes[root]),flush=True)
(OUT/'git_snapshots.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2),encoding='utf-8')
registry=COLLAB/'docs/collaboration/registry/EXPERIMENT_REGISTRY_20261007.csv'
with registry.open(encoding='utf-8-sig',newline='') as f:ledger=list(csv.DictReader(f));ledgerfields=list(ledger[0])
ids={r['experiment_id'].split('_')[0]:r['experiment_id'] for r in ledger}
def eid(p):
    s=str(p).replace('\\','/').lower()
    pairs=[('final2/devcheck','F02'),('final2/replay','F05'),('final2/formal','F03'),('final2_','F01'),('最终主线重训','F01'),('final_26245','F00'),('adp_rf_','I01'),('/eval/adp','I01'),('v2_26241','I02'),('stage_c','E04'),('阶段c','E04'),('stage_b2','E03'),('阶段b2','E03'),('stage_b','E02'),('阶段b1','E02'),('value_rule','E09'),('价值规则检验','E09'),('regime_screen','E10'),('最终主线','E10'),('label_check','E08'),('formal_review','E07'),('stop_dev','E05'),('stop_26243','E06'),('/stopping/formal','E06'),('停止头','E06'),('v3e','E01'),('t12','H04'),('noncoop','H05'),('v3b','H06'),('v3c','H06'),('v3d','H06'),('v3_bidirectional','H03'),('v2_arrival','H03'),('t6','H03'),('t7','H03'),('p0_horizon','H02'),('compute_h35','H02'),('phase2','H01'),('gatefree','H01'),('fullmission','H01')]
    for x,i in pairs:
        if x in s:return ids[i]
    return 'ORPHAN'
def status(p):
    if is_active(p):return 'ACTIVE'
    i=eid(p)
    if i.startswith(('F00','I01','I02')) or 'invalid' in str(p).lower():return 'INVALID'
    if any(x in str(p).lower() for x in ('__pycache__','tensorboard','.tmp','.lock','.pytest_cache')):return 'CACHE'
    if i.startswith(('E01','E02','E03','E04','E06','E07','E08','E10','H04')):return 'OFFICIAL'
    if i!='ORPHAN':return 'EXPLORATORY'
    return 'UNKNOWN'
def order(p):
    s=str(p).lower()
    if 'final2' in s or '最终主线重训' in s:return 'upper/run_orders/FINAL_RERUN_20261007.md + user 2026-10-09 priority amendment'
    if '阶段基准图' in s:return 'upper/run_orders/FIGURES_INTERIM_20261009.md + user presentation revision'
    if '整理_20261010' in s:return 'upper/run_orders/PROJECT_CLEANUP_20261010.md A only'
    if 'value' in s or '价值规则' in s:return 'docs/STOPPING_VALUE_RULE_CHECK_20261006.md (historical)'
    if 'stopping' in s or '停止头' in s:return 'docs/STOPPING_RUN_ORDER_LOWER_20261002.md (historical)'
    return 'UNRESOLVED: see evidence references; no authority inferred'

process_files=walk(BASE/'过程文件')
documents=[]
for root,files in indexes.items():
    for p,b,t in files:
        if p.suffix.lower()=='.md' and b<700000 and not is_active(p) and ('docs' in p.parts or p.name in ('CLAUDE.md','HISTORY.md')):
            try:documents.append((p,p.read_text(encoding='utf-8-sig',errors='replace')))
            except OSError:pass
for p,b,t in process_files:
    if p.suffix.lower()=='.md' and b<300000 and not is_active(p) and 'Git工作树' not in p.parts:
        try:documents.append((p,p.read_text(encoding='utf-8-sig',errors='replace')))
        except OSError:pass
refindex={}
for doc,text in documents:
    tokens=set(re.findall(r'[\w.-]+\.(?:py|ps1|bat|ipynb|json|md)',text,re.I))
    tokens.update(re.findall(r'(?:phase2|final2|final|stop|v3e|adp_rf|probe|t12|t7)[\w.-]+',text,re.I))
    for token in tokens:refindex.setdefault(token,[]).append(str(doc))
def refs(p):return (refindex.get(p.name,[])+refindex.get(p.stem,[]))[:12]
untracked=[]
sourcegroups=[(root,files) for root,files in indexes.items()]
sourcegroups.append((BASE,process_files))
seen=set()
for root,files in sourcegroups:
    for p,b,t in files:
        if str(p) in seen:continue
        seen.add(str(p))
        if p.suffix.lower() not in ('.py','.ps1','.bat','.ipynb','.md','.json'):continue
        if p.suffix.lower()=='.json' and (p.name.startswith('seed_') or p.name in ('FILES_SHA256.json','INPUTS_SHA256.json') or '/inputs/' in p.as_posix()):continue
        owner=next((r for r in sorted(repos,key=lambda x:len(str(x)),reverse=True) if p.is_relative_to(r)),None)
        rel=p.relative_to(owner).as_posix() if owner else ''
        if owner and rel in tracked[owner]:continue
        r=[] if is_active(p) else refs(p)
        used=bool(r) and p.suffix.lower() in ('.py','.ps1','.bat','.ipynb')
        untracked.append({'path':str(p),'repository':str(owner or 'outside Git'),'bytes':b,'experiment_id':eid(p),'active':is_active(p),
                          'produced_decision_numbers':'DOCUMENTED_REFERENCE_NEEDS_CODE_RESULT_RECONCILIATION' if used else 'NOT_ESTABLISHED',
                          'evidence_refs':' | '.join(r),'recommendation':'COMMIT_TO_GIT' if used else ('ARCHIVE_ONLY' if r or p.suffix=='.md' else 'SCRATCH'),
                          'note':'Reference is traceability evidence, not proof every proposed script was executed; no active content read'})
emit('untracked_code.csv',untracked)

assets=[];hashes=[];variants=[];seenvariants=set()
scanroots=[MAIN/'logs',MAIN/'eval',Path('D:/py/DRL2_v3e/logs'),BASE/'过程文件',BASE/'上层交付',BASE/'历史记录']
for root in scanroots:
    files=next(([(p,b,t) for p,b,t in xs if p.is_relative_to(root)] for repo,xs in indexes.items() if root.is_relative_to(repo)),None)
    if files is None:files=process_files if root==BASE/'过程文件' else walk(root)
    dirs=[root]+[p for p in root.glob('*') if p.is_dir()]+[q for p in root.glob('*') if p.is_dir() for q in p.glob('*') if q.is_dir()]
    for d in dirs:
        subset=[(p,b,t) for p,b,t in files if p.is_relative_to(d)]
        manifest={};mp=d/'manifest.json'
        if mp.exists() and not is_active(mp):
            try:manifest=read(mp)
            except Exception as e:errors.append(str(e))
        summary={k:manifest.get(k) for k in ('code_commit','code_dirty','seed','method','regime','status','actual_decision_steps','actual_outer_decisions','actual_model_timesteps') if k in manifest}
        stat=status(d);id=eid(d)
        assets.append({'path':str(d),'bytes':sum(b for p,b,t in subset),'files':len(subset),'experiment_id':id,'status':stat,
                       'manifest_summary':json.dumps(summary,ensure_ascii=False),'earliest_mtime':min((t for p,b,t in subset),default=''),
                       'latest_mtime':max((t for p,b,t in subset),default=''),'hash_file':'hash_inventory.csv',
                       'recommendation':{'ACTIVE':'KEEP_ACTIVE','INVALID':'INVALID_KEEP','CACHE':'CACHE_DELETABLE','UNKNOWN':'ASK_USER'}.get(stat,'KEEP_ARCHIVE')})
        # A directory with an actual manifest or direct logged run is an evidenced attempt, not a proposed method.
        if (manifest or (root.name=='logs' and d.parent==root and subset)) and str(d) not in seenvariants:
            seenvariants.add(str(d));reference=refs(d)
            variants.append({'name':manifest.get('run_name',d.name),'date':manifest.get('created_at_utc',manifest.get('started_at_utc','')),
                             'run_dir':str(d),'experiment_id':id,'order':order(d),'method':manifest.get('method',manifest.get('experiment_mode',manifest.get('algorithm','UNKNOWN'))),
                             'configuration':json.dumps({k:manifest[k] for k in ('seed','regime','stopping','requested_timesteps','requested_outer_decisions','mainline') if k in manifest},ensure_ascii=False),
                             'result':f"manifest.status={manifest.get('status','NO_MANIFEST')}; recorded budget={manifest.get('actual_outer_decisions',manifest.get('actual_decision_steps',manifest.get('actual_model_timesteps','UNKNOWN')))}",
                             'evidence':str(mp) if manifest else str(d)+' (existing run files) 引用: '+' | '.join(reference),
                             'stop_reason':'OPERATOR_INTERRUPTION_INVALID_NO_RESUME' if id.startswith('F00') else ('EVALUATOR_BUG_INVALID: registry' if id.startswith(('I01','I02')) else manifest.get('stop_reason',manifest.get('reason','NOT_DOCUMENTED; do not infer from filename'))),
                             'execution_evidence':'manifest or existing logged run; not a new experiment'})
    if root.name in ('logs','eval'):
        for p,b,t in files:
            if is_active(p):continue
            key=p.name in ('manifest.json','train.monitor.csv','final_model.zip') or p.name.startswith('seed_') and p.suffix=='.json' or 'readout' in p.name and p.suffix=='.json'
            if not key:continue
            # Hash only completed immutable assets. Unknown/running manifests are excluded conservatively.
            parent=p.parent
            if root.name=='logs' and (parent/'manifest.json').exists():
                try:
                    mm=read(parent/'manifest.json')
                    if mm.get('status') in ('running','training','started'):continue
                except Exception:continue
            try:
                h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
                hashes.append({'path':str(p),'bytes':b,'sha256':h,'verification':'LIVE_READ_ENDED_ASSET','experiment_id':eid(p)})
            except Exception as e:errors.append(str(e))
    print('ASSETS',root,'directories',len(dirs),flush=True)
emit('local_assets.csv',assets);emit('hash_inventory.csv',hashes)

# Add probes/parameter attempts from local numerical reports, retaining literal source lines.
for doc,text in documents:
    if any(x in doc.as_posix().lower() for x in ('/upper/run_orders/','/upper/cleanup/')):continue
    for i,line in enumerate(text.splitlines(),1):
        if len(line)>1000 or not re.search(r'probe|diagnos|消融|探针|参数|D0b|D0|D1|T[6-9]|T1[0-2]|V3[bcd]|alpha|gamma|horizon|fallback',line,re.I):continue
        if not re.search(r'\d|停止|失败|完成|pass|fail|result|实测|判定',line,re.I):continue
        key=(str(doc),line.strip())
        if key in seenvariants:continue
        seenvariants.add(key)
        variants.append({'name':'REPORT_LITERAL: '+line.strip()[:140],'date':'see source document','run_dir':'UNRESOLVED','experiment_id':eid(doc),'order':order(doc),'method':'historical report mention','configuration':'',
                         'result':line.strip(),'evidence':f'{doc}:{i}','stop_reason':'see literal evidence; no root cause inferred','execution_evidence':'historical statement; proposed/tested ambiguity retained for upper review'})
emit('tried_variants.csv',variants)
processrows=[]
for p,b,t in process_files:
    category='临时' if any(x in str(p).lower() for x in ('backup','备份','pytest','matplotlib','临时')) else ('交接' if p.suffix=='.md' or '接续' in p.parts else ('执行回执' if p.suffix=='.json' or 'receipt' in p.name.lower() else '运行记录'))
    processrows.append({'path':str(p),'bytes':b,'category':category,'order':order(p),'experiment_id':eid(p),'active_metadata_only':is_active(p)})
emit('process_files.csv',processrows)
emit('excluded_dependencies.csv',[{'path':x,'reason':'dependency/cache/Git internals excluded from substantive asset counts'} for x in sorted(set(excluded))])
(OUT/'READ_ERRORS.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
sha_index={r['path']:r['sha256'] for r in hashes}
for r in ledger:
    paths=[a['path'] for a in assets if a['experiment_id']==r['experiment_id']]
    if paths:r['local_result_path']='; '.join(paths[:20])
    count=sum(x['experiment_id']==r['experiment_id'] for x in hashes)
    r['local_sha_verified']=f'{count} ended files live hashed; active files metadata only' if count else 'n/a: no new active hashes; see inventory'
    r['lower_note']=r.get('lower_note','')+' | A-only 2026-10-09 inventory, run order 960a852; no move/delete/scientific Git mutation; see cleanup_inventory_20261010'
emit('EXPERIMENT_REGISTRY_LOCAL_COLUMNS.csv',ledger,ledgerfields)
counts={'repositories':len(repos),'asset_directories':len(assets),'untracked_candidates':len(untracked),'variant_entries':len(variants),'ended_key_hashes':len(hashes),'process_files':len(processrows),'errors':len(errors)}
report='# 本地只读资产盘点（A段）\n\n执行单PROJECT_CLEANUP_20261010，固定协作提交960a852；盘点实际时间2026-10-09（JST），20261010是执行单/主题名称。不执行B、C：不改科学分支、不打标签、不移动删除、不重写历史、不运行实验。\n\n'
report+='## 范围与限制\n\n扫描D:/py和Spacecraft中的Git检出，并纳入git worktree list；仓库外的其他磁盘路径不在本次发现范围。活动eval/final2与logs/final2_262461仅名称/大小/修改时间，不读取内容、不算SHA。后台日志和监督状态也保留。依赖、Git内部和缓存目录排除实质统计，另列excluded_dependencies.csv；统计不是磁盘占用总量。工作树与协作工程均保留原位。\n\n'
report+='## 产物\n\n'+''.join(f'- {k}: {v}\n' for k,v in counts.items())
report+='\n3.1 git_snapshots.json含每检出完整status/stash/worktree/remote；3.2 untracked_code.csv按证据引用标COMMIT_TO_GIT/ARCHIVE_ONLY/SCRATCH，不把被引用等同已执行，临时JSON非配置候选另由process_files保留；3.3–3.4 local_assets.csv每目录映射实验ID或ORPHAN，UNKNOWN保持待上层确认，INVALID保留；3.5 hash_inventory.csv仅结束资产；3.6 process_files.csv含逐文件类别/执行单，无法确定单据标UNRESOLVED；3.8 tried_variants.csv包含实际运行及历史报告逐行摘录，未核定是否执行的文字明确标注，不能用它生成确定的负结果。\n\n扫描与映射是保守盘点，不对实验结果做新解读；根因与是否重开由上层核定。旧台账缺少路径的探针不编造对应关系。原始文件原位保留。\n\n## 目录总表\n\n|目录|实验|状态|文件数|字节|\n|---|---|---|---:|---:|\n'
report+=''.join(f"|{a['path']}|{a['experiment_id']}|{a['status']}|{a['files']}|{a['bytes']}|\n" for a in assets)
(OUT/'LOCAL_ASSET_MAP.md').write_text(report,encoding='utf-8')
(OUT/'COUNTS.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',json.dumps(counts),flush=True)
