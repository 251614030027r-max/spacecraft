from pathlib import Path
import json,hashlib,shutil,subprocess
TOP=Path(__file__).resolve().parents[1];REPO=Path('C:/Users/35884/Documents/Spacecraft/过程文件/协作/Git工作树');LOCAL=TOP/'reports/figures_interim_20261009';PUBLIC=REPO/'docs/collaboration/lower/handoffs/figures_interim_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for root in (LOCAL,PUBLIC):
    shutil.copy2(REPO/'docs/collaboration/upper/run_orders/FIGURES_INTERIM_20261009.md',root/'RUN_ORDER.md')
    p=root/'README.md';s=p.read_text(encoding='utf-8');s+='\n上层已在c4fecf2同步更正图2真值违规：Pure0、stopping0/0/0、learned1/0/1，与本包及用户明确确认一致；RUN_ORDER.md保存该更正版。\n';p.write_text(s,encoding='utf-8')
    files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'};(root/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8');assert all(sha(root/n)==v for n,v in files.items())
    print('BYTE_SHA_VERIFIED',root.name,len(files))
