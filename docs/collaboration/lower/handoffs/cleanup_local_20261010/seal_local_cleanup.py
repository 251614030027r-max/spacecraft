from pathlib import Path
import csv,json,hashlib,shutil
BASE=Path(r'C:\Users\35884\Documents\Spacecraft');TOP=BASE/'过程文件/整理_20261010';REC=TOP/'记录/C_LOCAL'
C=BASE/'过程文件/协作/Git工作树/docs/collaboration';D=C/'lower/handoffs/cleanup_local_20261010';D.mkdir(parents=True,exist_ok=True);assert not list(D.iterdir())
manifest=REC/'MANIFEST_SHA256.csv'
rows=list(csv.DictReader(manifest.open(encoding='utf-8-sig',newline='')))
unique={}
for r in rows:
    key=r['archive'].lower()
    if key in unique:assert unique[key]==r
    else:unique[key]=r
raw=REC/'MANIFEST_SHA256_inventory_rows.csv';shutil.copy2(manifest,raw)
with manifest.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(unique.values())
receipt=json.loads((REC/'ARCHIVE_RECEIPT.json').read_text(encoding='utf-8'))
sources={r['source'].lower() for r in rows}
mapping={}
for r in rows:mapping.setdefault(r['source'],set()).add((r['experiment_id'],r['status']))
conflicts={k:sorted(v) for k,v in mapping.items() if len(v)>1}
(REC/'MAPPING_AMBIGUITIES.json').write_text(json.dumps(conflicts,ensure_ascii=False,indent=2),encoding='utf-8')
receipt.update(files=len(unique),unique_source_files=len(sources),bytes=sum(int(r['bytes']) for r in unique.values()),source_roots=len({str(Path(r['source']).parents[0]) for r in unique.values()}),inventory_processed_rows=len(rows),duplicate_inventory_rows=len(rows)-len(unique),mapping_ambiguous_source_files=len(conflicts),copy_hash_verification='Every source/destination SHA verified; exact duplicate archive rows deduplicated; source aliases retained and mapping ambiguities disclosed')
# Keep original root count as inventory cardinality, not unique directory count.
receipt['inventory_root_rows']=172;receipt.pop('source_roots')
(REC/'ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
for name in ('MANIFEST_SHA256.csv','ARCHIVE_RECEIPT.json','DELIVERY_PATH_MAP.json','MAPPING_AMBIGUITIES.json'):shutil.copy2(REC/name,D/name)
for name in ('archive_closed_assets.py','organize_local_entries.py','seal_local_cleanup.py'):shutil.copy2(TOP/'脚本'/name,D/name)
text=f'''# 本地整理阶段交付

用户最新授权：上层已判F01失败；剩余评估仅补齐记录；允许整理DRL2与Spacecraft。未停止进程，未改方法或训练。用户决定和上层两模型审查不替代仍待齐全的三模型冻结官方判读。

## 已完成

- 新干净副本D:/py/spacecraft，科学分支046878c9803136f67fb31540a13213fd8343c6b4，工作树干净。该版本较原核验目标60d768a仅改CLAUDE.md、docs/DEAD_ENDS.md、docs/HISTORY.md。只做Git状态验证；不是已经通过测试与行为一致性验收。
- 依只读盘点映射，已关闭DRL2资产复制到D:/spacecraft_archive：{receipt['files']}个归档文件、{receipt['bytes']}字节，对应{receipt['unique_source_files']}个唯一源文件；源/副本逐文件SHA一致。归档在experiments/<实验ID>/DRL2/，作废运行在_invalid/<实验ID>/DRL2/。盘点重复目标条目已去重，逐源路径见MANIFEST_SHA256.csv。发现{len(conflicts)}个源文件被盘点重复归入不同实验/状态，原分类均保留在MAPPING_AMBIGUITIES.json，不能将E01等宽泛映射视作已核验科学归属；未删除别名副本，后续审查统一归属后再去重。
- DRL2、DRL2_v3e和历史工作树原件保留；不移动、不删除、不改只读属性。eval/final2、logs/final2_*、最终主线重训的状态/监督/日志/锁全不整理，活动代码工作树未切分支。
- Spacecraft上层交付/最新现在直接提供F01阶段报告、上层署名审查、192个完整逐开局证据和哈希；旧价值规则轮已移入上层交付/历史/VALUE_RULE_CHECK_20261006，迁移前后SHA一致，见DELIVERY_PATH_MAP.json。不添加ZIP展开冗余。项目说明已重写为当前事实，旧入口完整快照保存在过程文件/整理_20261010/接续/。
- Git的7个归档标签、6个旧分支清理及106候选脚本去重已在cleanup_b123_20261010交付；本轮不重复执行、不合并main。

## 仍待完成的验收

评估/既有重放全部退出并交付后：整理目标60d768a的完整pytest与271000前3开局Pure/262460 stopping共6回合行为逐字段核对；原工程转只读及活动产物归档。此时不额外运行验收仿真，不占用现有8个评估槽位。046878c的文档增量与60d768a区分记录。

外置盘/网盘离线备份未提供目标，尚未建立；同机D盘复制不是离线备份。未知/ORPHAN产物、旧工作树、DRL2_v3e均保留，不以分类不全为由删除；缓存删除按执行单由用户办理。正在运行的评估继续原协议，阶段结束后不新增研究实验，下一步等待用户与上层决定。
'''
(D/'README.md').write_text(text,encoding='utf-8')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file()}
(D/'FILES_SHA256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
(D/'.gitattributes').write_text('* -text\n',encoding='utf-8')
state=json.loads((C/'CURRENT.json').read_text(encoding='utf-8'))
state['local_cleanup'].update(archive_files=receipt['files'],archive_bytes=receipt['bytes'],archive_sha_verified=True)
(C/'CURRENT.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
with (C/'lower/README.md').open('a',encoding='utf-8') as f:f.write('\n- [本地整理阶段交付](handoffs/cleanup_local_20261010/README.md)：新干净副本、历史复制与SHA、最新交付入口；活动评估保留，测试/行为验收与离线备份尚待完成。\n')
print(json.dumps(receipt,ensure_ascii=False))
