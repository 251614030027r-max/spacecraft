"""Copy closed scientific assets only; never mutate the active experiment tree."""
from pathlib import Path
import csv, hashlib, json, shutil, time

BASE=Path(r'C:\Users\35884\Documents\Spacecraft')
TOP=BASE/'过程文件/整理_20261010'
REC=TOP/'记录/C_LOCAL'; REC.mkdir(parents=True,exist_ok=True)
SRC=Path(r'D:\py\DRL2').resolve()
DEST=Path(r'D:\spacecraft_archive').resolve()
inventory=BASE/'过程文件/协作/Git工作树/docs/collaboration/lower/handoffs/cleanup_inventory_20261010/local_assets.csv'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
rows=list(csv.DictReader(inventory.open(encoding='utf-8-sig',newline='')))
candidates=[]
for r in rows:
    p=Path(r['path']).resolve()
    if p.parent not in (SRC/'logs',SRC/'eval'):continue
    if p.name.startswith('final2') or p.name.startswith('__'):continue
    if r['recommendation'] not in ('KEEP_ARCHIVE','INVALID_KEEP'):continue
    if r['contains_active_asset'].lower()=='true':continue
    candidates.append((p,r))
manifest=REC/'MANIFEST_SHA256.csv'
assert not manifest.exists(),'Existing receipt: preserve it; no duplicate archive'
count=total=0
with manifest.open('w',encoding='utf-8-sig',newline='') as out:
    writer=csv.DictWriter(out,fieldnames=['source','archive','sha256','bytes','experiment_id','status'])
    writer.writeheader()
    for source,r in candidates:
        relative=source.relative_to(SRC)
        root=DEST/('_invalid' if r['recommendation']=='INVALID_KEEP' else 'experiments')/r['experiment_id']/'DRL2'/relative
        assert root.resolve().is_relative_to(DEST)
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            assert not p.is_symlink(),str(p)
            target=root/p.relative_to(source)
            assert target.resolve().is_relative_to(DEST)
            target.parent.mkdir(parents=True,exist_ok=True)
            before=p.stat(); digest=sha(p)
            if target.exists():assert sha(target)==digest, str(target)
            else:shutil.copy2(p,target)
            after=p.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(p)
            assert sha(target)==digest,str(target)
            writer.writerow(dict(source=str(p),archive=str(target),sha256=digest,bytes=before.st_size,experiment_id=r['experiment_id'],status=r['status']))
            count+=1; total+=before.st_size
        out.flush()
        print(source.name,count,total,flush=True)
result={'status':'CLOSED_ASSETS_COPIED_HASH_VERIFIED','source_roots':len(candidates),'files':count,'bytes':total,'archive_root':str(DEST),'active_exclusions':['D:/py/DRL2/eval/final2','D:/py/DRL2/logs/final2_*','Spacecraft/过程文件/最终主线重训'],'source_files_moved_deleted_modified':False,'DRL2_v3e':'preserved unchanged; not duplicated','offline_backup':'PENDING: no external destination supplied','readonly_transition':'DEFERRED until all workers exit','tests_behavior_verification':'DEFERRED until all evidence delivered and workers exit'}
(REC/'ARCHIVE_RECEIPT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
