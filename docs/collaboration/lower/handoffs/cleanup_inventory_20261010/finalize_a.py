from pathlib import Path
import json,csv,hashlib,shutil,subprocess,os,re
os.environ['GIT_OPTIONAL_LOCKS']='0'
BASE=Path(r'C:\Users\35884\Documents\Spacecraft');TOP=BASE/'过程文件/整理_20261010';OUT=TOP/'交付';REPO=BASE/'过程文件/协作/Git工作树';C=REPO/'docs/collaboration'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rs):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assets=rows(OUT/'local_assets.csv');untracked=rows(OUT/'untracked_code.csv');variants=rows(OUT/'tried_variants.csv');hashes=rows(OUT/'hash_inventory.csv')
literal=[r for r in variants if r['name'].startswith('REPORT_LITERAL:')]
write(OUT/'variant_evidence_mentions.csv',literal)
attempts=[r for r in variants if not r['name'].startswith('REPORT_LITERAL:')]
# Collapse duplicate clone runs and repeated report mentions into a usable trial index.
dedup={}
for r in attempts:
    key=(r['name'],r['configuration'],r['result'])
    if key in dedup:
        dedup[key]['evidence']+=' | duplicate location: '+r['evidence']
    else:dedup[key]=r
families={}
for r in literal:
    if re.search(r'建议|待执行|计划|未执行|proposed|will ',r['result'],re.I):continue
    names=re.findall(r'\b(?:D\d[a-z]?|T\d{1,2}|V3[b-e]|P\d|probe\d|horizon|gamma|alpha|fallback)\b',r['result'],re.I)
    for name in names:
        key=(r['experiment_id'],name.lower())
        if key not in families:
            rr=dict(r);rr['name']='HISTORICAL_EVIDENCE_FAMILY: '+name;rr['method']='report family index, not a single verified trial';families[key]=rr
        elif r['evidence'] not in families[key]['evidence'] and len(families[key]['evidence'])<2500:
            families[key]['evidence']+=' | '+r['evidence']
variants=list(dedup.values())+list(families.values());write(OUT/'tried_variants.csv',variants)
counts=read(OUT/'COUNTS.json');counts.update(variant_entries=len(variants),run_entries=len(dedup),historical_evidence_families=len(families),literal_evidence_mentions=len(literal))
(OUT/'COUNTS.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding='utf-8')
active_dirs=[a['path'].rstrip('\\/') for a in assets if a['status']=='ACTIVE']
for a in assets:
    a['contains_active_asset']=any(d.startswith(a['path'].rstrip('\\/')+os.sep) for d in active_dirs)
    if a['contains_active_asset']:a['recommendation']='KEEP_ACTIVE'
write(OUT/'local_assets.csv',assets)
assert all(a['experiment_id'] and a['status'] for a in assets)
assert not any('/eval/final2/' in h['path'].replace('\\','/').lower() or '/logs/final2_262461' in h['path'].replace('\\','/').lower() for h in hashes)
# Keep unknown source candidates, rather than calling unverifiable scientific code disposable scratch.
for r in untracked:
    rp=Path(r['path']);r['file_role']='CONFIG_OR_RECEIPT_JSON_CANDIDATE' if rp.suffix.lower()=='.json' else ('SOURCE' if rp.suffix.lower() in ('.py','.ps1','.bat','.ipynb') else 'DOCUMENT')
    if rp.suffix.lower()=='.json' and '/eval/' in rp.as_posix().lower():r['file_role']='RESULT_JSON_POINTER_NOT_CONFIG'
    if r['recommendation']=='SCRATCH' and not any(x in r['path'].lower() for x in ('pytest','fixture','cache','临时','scratch')):
        r['recommendation']='ARCHIVE_ONLY';r['note']+='; retained until numerical provenance is resolved'
# Numerical provenance for local orchestration/analysis: preserve candidates alongside matching immutable outputs.
for r in untracked:
    p=Path(r['path'])
    if p.suffix.lower() not in ('.py','.ps1','.bat','.ipynb'):continue
    if '脚本' not in p.parts:continue
    topic=p.parent.parent
    ended=[a for a in assets if a['path'].startswith(str(topic)) and a['status']!='ACTIVE']
    if ended and any((topic/name).exists() for name in ('交付','reports','审查材料_20261006','记录')):
        r['recommendation']='COMMIT_TO_GIT'
        r['produced_decision_numbers']='TOPIC_NUMERICAL_OUTPUT_PRESENT_NEEDS_PER_SCRIPT_RECONCILIATION'
        r['evidence_refs']=r['evidence_refs'] or '; '.join(a['path'] for a in ended[:3])
        r['note']+='; conservative provenance-preservation recommendation, not a claim of execution'
write(OUT/'untracked_code.csv',untracked)
snaps=read(OUT/'git_snapshots.json');good=[s for s in snaps if not s['head'].startswith('ERROR')]
for snap in snaps:
    snap['remotes']=re.sub(r'(https?://)[^/\s]+@',r'\1[REDACTED]@',snap['remotes'])
(OUT/'git_snapshots.json').write_text(json.dumps(snaps,ensure_ascii=False,indent=2),encoding='utf-8')
main=next(s for s in good if s['path'].replace('\\','/')=='D:/py/DRL2')
assert main['head']=='f2f8169acd580ea96a578d45022dd8952447eca5'
assert not any(line[:2] not in ('??','!!') for line in main['status'].splitlines()), 'Tracked science changes detected'
known=[(r['experiment_id'],r['primary_readout'],r['primary_readout_sha256']) for r in rows(C/'registry/EXPERIMENT_REGISTRY_20261007.csv') if len(r['primary_readout_sha256'])==64]
comparisons=[]
for eid,p,h in known:
    candidates=[x for x in hashes if x['path'].replace('\\','/').endswith(p.replace('\\','/'))]
    comparisons.append({'experiment_id':eid,'reference_path':p,'expected_sha256':h,'matched_paths':'; '.join(x['path'] for x in candidates),
                        'result':'MATCH' if candidates and all(x['sha256']==h for x in candidates) else ('UNRESOLVED_REFERENCE_LOCATION' if not candidates else 'MISMATCH')})
write(OUT/'known_hash_comparison.csv',comparisons)
# Fixture repos and an uninitialized Spacecraft .git are discovery facts, not working scientific clones.
extra='\n\n## 核验补充\n\n- Git发现包含迁移测试fixture与Spacecraft未初始化的.git；无有效HEAD的条目保留错误，不冒充实验工程。真实Git状态完整附在git_snapshots.json。\n- 未证实逐脚本执行的数字来源只标候选COMMIT_TO_GIT，优先保留；需上层复核源码—结果关系后再入科学分支。未知代码默认ARCHIVE_ONLY，不能据此删除。\n- 哈希核验使用known_hash_comparison.csv；引用未定位标UNRESOLVED，不能写成通过。活动目录没有内容读取或哈希。\n- tried_variants.csv中真实运行由manifest/现存日志证明；REPORT_LITERAL行是出处索引，提案与实做可能混杂，不能直接作为DEAD_ENDS定论。\n'
extra+=f'\n试验清单已去重：{len(dedup)}条有运行资产的记录，{len(families)}条历史证据族索引；{len(literal)}条原文出处单独保留在variant_evidence_mentions.csv，提案与实做歧义保留，不直接写成负结果。前文variant_entries是初扫条数，最终以COUNTS.json为准。\n两处pytest临时目录访问受限，已列READ_ERRORS.json；本次不改变权限、不绕过限制，不据此宣称全盘零遗漏。\n'
with (OUT/'LOCAL_ASSET_MAP.md').open('a',encoding='utf-8') as f:f.write(extra)
shutil.copy2(TOP/'脚本/inventory_a.py',OUT/'inventory_a.py');shutil.copy2(Path(__file__),OUT/'finalize_a.py')
(OUT/'RUN_ORDER.md').write_bytes((C/'upper/run_orders/PROJECT_CLEANUP_20261010.md').read_bytes())
(OUT/'.gitattributes').write_text('* -text\n',encoding='utf-8')
manifest={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='FILES_SHA256.json'}
(OUT/'FILES_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
assert all(sha(OUT/n)==v for n,v in manifest.items())
dest=C/'lower/handoffs/cleanup_inventory_20261010'
assert not dest.exists(),'Delivery already exists: do not overwrite'
shutil.copytree(OUT,dest)
# Only lower-owned local columns are updated; scientific conclusions and upper columns unchanged.
source=rows(OUT/'EXPERIMENT_REGISTRY_LOCAL_COLUMNS.csv');original=rows(C/'registry/EXPERIMENT_REGISTRY_20261007.csv');byid={r['experiment_id']:r for r in source}
for r in original:
    for field in ('local_result_path','local_sha_verified','lower_note'):r[field]=byid[r['experiment_id']][field]
write(C/'registry/EXPERIMENT_REGISTRY_20261007.csv',original)
current=read(C/'CURRENT.json');current['asset_inventory']={'phase':'A_READONLY_COMPLETED_B_C_PENDING_USER_CONFIRMATION','run_order':'upper/run_orders/PROJECT_CLEANUP_20261010.md','run_order_commit':'960a8520d8ab92e357196e5ebe997741ae863da3','handoff':'lower/handoffs/cleanup_inventory_20261010/LOCAL_ASSET_MAP.md','verified_date_jst':'2026-10-09','scientific_files_moved_or_deleted':False}
current['phase']='formal_evaluation_active_271000_priority';current['next_action']='Keep current evaluations running; 271000 priority delivery, retain seed262461; B/C cleanup only after experiment delivery and user confirmation'
current['priority_supervisor_state']=str(BASE/'过程文件/最终主线重训/记录/PRIORITY_EXECUTION.json')
current.pop('training_progress_snapshot',None)
(C/'CURRENT.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
lower=C/'lower/README.md'
with lower.open('a',encoding='utf-8') as f:f.write('\n\n只读盘点A段（实际2026-10-09，主题20261010）：[LOCAL_ASSET_MAP](handoffs/cleanup_inventory_20261010/LOCAL_ASSET_MAP.md)。含未入Git代码候选、历史尝试/探针、原位目录与哈希。B/C未执行，活动实验不动。\n')
print(json.dumps({'valid_git_checkouts':len(good),'asset_directories':len(assets),'variant_entries':len(variants),'untracked_candidates':len(untracked),'source_preservation_candidates':sum(r['recommendation']=='COMMIT_TO_GIT' for r in untracked),'hashed_ended_files':len(hashes),'package_files':len(manifest),'hash_comparison':comparisons},ensure_ascii=False))
