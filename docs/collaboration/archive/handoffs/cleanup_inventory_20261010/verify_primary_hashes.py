from pathlib import Path
import csv,json,hashlib,shutil
BASE=Path(r'C:\Users\35884\Documents\Spacecraft');OUT=BASE/'过程文件/整理_20261010/交付';C=BASE/'过程文件/协作/Git工作树/docs/collaboration';PUB=C/'lower/handoffs/cleanup_inventory_20261010'
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rs):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
locations={'E04_stageC_classifier':Path('D:/py/DRL2/eval/v3e/stage_c/c2_report.json'),
           'E07_upper_formal_review':Path('D:/py/DRL2/eval/stopping/formal_review_20261006.json'),
           'E09_value_rule_check':BASE/'过程文件/价值规则检验/审查材料_20261006/results/readout_value.json'}
comparisons=readcsv(OUT/'known_hash_comparison.csv');hashes=readcsv(OUT/'hash_inventory.csv')
for r in comparisons:r['actual_byte_sha256']='';r['lf_normalized_sha256']=''
for r in comparisons:
    if r['experiment_id'] not in locations:continue
    p=locations[r['experiment_id']];assert '/eval/final2/' not in p.as_posix().lower()
    digest=sha(p);lf_digest=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    r.update(matched_paths=str(p),actual_byte_sha256=digest,lf_normalized_sha256=lf_digest,
             result='MATCH' if digest==r['expected_sha256'] else ('BYTE_DIFF_CRLF_LF_MATCH' if lf_digest==r['expected_sha256'] else 'MISMATCH_REQUIRES_REVIEW'))
    hashes.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest,'verification':'LIVE_READ_ENDED_PRIMARY_REFERENCE','experiment_id':r['experiment_id']})
writecsv(OUT/'hash_inventory.csv',hashes);writecsv(OUT/'known_hash_comparison.csv',comparisons)
counts=json.loads((OUT/'COUNTS.json').read_text(encoding='utf-8'));counts['ended_key_hashes']=len(hashes)
(OUT/'COUNTS.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'LOCAL_ASSET_MAP.md').open('a',encoding='utf-8') as f:f.write('\n补充定位E04、E07、E09主读数，按真实字节SHA与LF规范化SHA分开核对；不能把行尾等价写成原文件字节一致。逐项状态见known_hash_comparison.csv，哈希总数以COUNTS.json为准。\n')
for path in (C/'registry/EXPERIMENT_REGISTRY_20261007.csv',OUT/'EXPERIMENT_REGISTRY_LOCAL_COLUMNS.csv'):
    ledger=readcsv(path)
    for r in ledger:
        if r['experiment_id'] in locations:
            comparison=next(x for x in comparisons if x['experiment_id']==r['experiment_id'])
            r['local_sha_verified']+='; primary_readout '+comparison['result'];r['local_result_path']+='; '+str(locations[r['experiment_id']])
    writecsv(path,ledger)
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
for p in OUT.iterdir():
    if p.is_file() and p.name!='FILES_SHA256.json':shutil.copy2(p,PUB/p.name)
for root in (OUT,PUB):
    files={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='FILES_SHA256.json'}
    (root/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8');assert all(sha(root/n)==h for n,h in files.items())
print('KNOWN_PRIMARY_HASHES:',[(r['experiment_id'],r['result']) for r in comparisons],'; ended files:',len(hashes))
