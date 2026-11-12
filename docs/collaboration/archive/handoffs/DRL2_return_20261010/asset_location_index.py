from pathlib import Path
import csv,json,hashlib,shutil,subprocess
B=Path(r'C:\Users\35884\Documents\Spacecraft');R=B/'过程文件/整理_20261010/记录/RETURN_DRL2'
C=B/'过程文件/协作/Git工作树/docs/collaboration';D=C/'lower/handoffs/DRL2_return_20261010';D.mkdir(parents=True,exist_ok=False)
rows=list(csv.DictReader((R/'PRE_CLEAN_MANIFEST_SHA256.csv').open(encoding='utf-8-sig',newline='')))
with (D/'ASSET_LOCATION_INDEX.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['original_path','current_primary_path','verified_backup_path','sha256','bytes','original_present'])
    w.writeheader()
    for r in rows:
        exists=Path(r['source']).is_file()
        w.writerow({'original_path':r['source'],'current_primary_path':r['source'] if exists else r['archive'],'verified_backup_path':r['archive'],'sha256':r['sha256'],'bytes':r['bytes'],'original_present':exists})
for n in ('PREPARED.json','RETURN_RECEIPT.json','PRE_CLEAN_MANIFEST_SHA256.csv','DELIVERY_PATH_MAP.json'):
    shutil.copy2(R/n,D/n)
for n in ('return_drl2.py','asset_location_index.py','retest_setup_errors.py'):
    shutil.copy2(B/'过程文件/整理_20261010/脚本'/n,D/n)
paths=[('current_code_and_specs','D:/py/DRL2','4795910102cf9fbf1dfec74b77e9449623a7b51e','CLAUDE.md; docs/current; docs/HISTORY.md; docs/DEAD_ENDS.md'),('tracked_historical_evidence','D:/py/DRL2/evidence','4795910102cf9fbf1dfec74b77e9449623a7b51e','Recovered by checking out science branch'),('historical_artifact_lookup','D:/py/DRL2/docs/archive/ARTIFACT_INDEX.csv','archive/pre-cleanup-20261010','1618 Git paths/blob identifiers; existing records only'),('historical_code_lookup','D:/py/DRL2/docs/archive/CODE_INDEX.csv','archive/pre-cleanup-20261010','Removed scripts and their experiments/judgments; not new execution authority'),('historical_doc_lookup','D:/py/DRL2/docs/archive/PATH_MAP.csv','4795910102cf9fbf1dfec74b77e9449623a7b51e','Old to new documentation paths'),('E01_review','docs/reviews','archive/review/v3e-60k-value-r2-20260930','Use original Git tag/tree paths'),('E04_review','docs/reviews','archive/review/v3e-stage-c-20261002','Use original Git tag/tree paths'),('E06_review','docs/reviews/stopping_20261006','archive/review/stopping-20261006-audit','Full original stopping evidence at fixed tag'),('E09_review','docs/reviews','archive/review/stopping-value-check-20261006','Use original Git tag/tree paths'),('F01_closed_records','docs/collaboration/lower/handoffs/F01_closed_records_20261010','collab/spacecraft','650 actual records; 118 never-produced records remain missing')]
with (D/'GIT_ASSET_ENTRIES.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['asset','path_or_root','version','note']);w.writerows(paths)
(D/'.gitattributes').write_text('* -text\n',encoding='utf-8')
print('ASSET_LOCATIONS_FILLED',len(rows),flush=True)
