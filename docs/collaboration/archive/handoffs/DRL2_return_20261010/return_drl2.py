"""Archive stopped outputs, then return the cleaned science branch to DRL2.
No simulation, training, raw-data edits, Git history rewriting or force-push.
"""
from pathlib import Path
import hashlib,json,csv,shutil,subprocess,sys
BASE=Path(r'C:\Users\35884\Documents\Spacecraft');TOP=BASE/'过程文件/整理_20261010'
REC=TOP/'记录/RETURN_DRL2';REC.mkdir(parents=True,exist_ok=True)
MAIN=Path(r'D:\py\DRL2').resolve(); ARCH=Path(r'D:\spacecraft_archive\return_drl2_20261010').resolve()
COLLAB=BASE/'过程文件/协作/Git工作树'
TARGET='4795910102cf9fbf1dfec74b77e9449623a7b51e';BRANCH='claude/sac-mpc-coupling-design-ns7g6i'
FROZEN='f2f8169acd580ea96a578d45022dd8952447eca5'
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.STDOUT)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for v in iter(lambda:f.read(1024*1024),b''):h.update(v)
    return h.hexdigest()
def save(name,data):(REC/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def models():return {str(s):sha(MAIN/f'logs/final2_{s}/final_model.zip') for s in (262460,262461,262462)}
def prepare():
    assert git(MAIN,'rev-parse','HEAD').decode().strip()==FROZEN
    assert not git(MAIN,'diff','--name-only') and not git(MAIN,'diff','--cached','--name-only'),'Tracked user changes; stop'
    assert not (REC/'PREPARED.json').exists(),'Existing receipt; no repeat'
    (REC/'git_status_before.txt').write_bytes(git(MAIN,'status','--porcelain','--untracked-files=all'))
    (REC/'git_worktrees_before.txt').write_bytes(git(MAIN,'worktree','list','--porcelain'))
    # Git objects verify the retained scientific code is byte-identical.
    def tree(ref):
        return {p.decode():h.split()[2].decode() for item in git(COLLAB,'ls-tree','-r','-z',ref).split(b'\0') if item for h,p in [item.split(b'\t',1)]}
    old,new=tree(FROZEN),tree(TARGET)
    roots=('env/','controllers/','dynamics/','estimation/','train/','experiments/','eval/')
    kept=[p for p in new if p.startswith(roots) and p.endswith('.py')]
    assert all(new[p]==old.get(p) for p in kept),'Scientific behavior changed'
    config={str(p.relative_to(MAIN)):sha(p) for p in (MAIN/'.idea').rglob('*') if p.is_file()}
    config['.venv/Scripts/python.exe']=sha(MAIN/'.venv/Scripts/python.exe')
    baseline={'frozen_commit':FROZEN,'target_commit':TARGET,'model_sha256':models(),'environment_sha256':config,'retained_scientific_py_files':len(kept)}
    count=total=0
    with (REC/'PRE_CLEAN_MANIFEST_SHA256.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['source','archive','sha256','bytes']);w.writeheader()
        for name in ('logs','eval','local_artifacts'):
            root=MAIN/name
            for p in sorted(root.rglob('*')):
                if not p.is_file():continue
                assert not p.is_symlink()
                dest=ARCH/'pre_clean_source'/p.relative_to(MAIN)
                assert dest.resolve().is_relative_to(ARCH)
                dest.parent.mkdir(parents=True,exist_ok=True)
                st=p.stat();h=sha(p)
                if dest.exists():assert sha(dest)==h
                else:shutil.copy2(p,dest)
                assert sha(dest)==h and p.stat().st_size==st.st_size and p.stat().st_mtime_ns==st.st_mtime_ns
                w.writerow(dict(source=str(p),archive=str(dest),sha256=h,bytes=st.st_size));count+=1;total+=st.st_size
            f.flush();print('BACKED_UP',name,count,total,flush=True)
    baseline.update(backup_files=count,backup_bytes=total,archive_root=str(ARCH))
    save('PREPARED.json',baseline);print('PREPARED_ALL_SHA_OK',flush=True)
def migrate():
    before=json.loads((REC/'PREPARED.json').read_text(encoding='utf-8'))
    assert not (REC/'RETURN_RECEIPT.json').exists(),'Already migrated'
    assert not git(MAIN,'diff','--name-only') and not git(MAIN,'diff','--cached','--name-only')
    assert models()==before['model_sha256']
    print(git(MAIN,'fetch','origin',BRANCH).decode(),flush=True)
    assert git(MAIN,'rev-parse','origin/'+BRANCH).decode().strip()==TARGET,'Remote advanced; inspect new version first'
    present=subprocess.run(['git','-C',str(MAIN),'show-ref','--verify','--quiet','refs/heads/'+BRANCH]).returncode==0
    if present:
        git(MAIN,'switch',BRANCH);git(MAIN,'merge','--ff-only','origin/'+BRANCH)
    else:git(MAIN,'switch','--track','-c',BRANCH,'origin/'+BRANCH)
    moved=[]
    # Git already removes backed-up tracked historical outputs. Only residual
    # untracked/ignored closed directories move; final2 model paths stay intact.
    paths=[p for p in (MAIN/'logs').iterdir() if p.name not in ('final2_262460','final2_262461','final2_262462')]
    paths += [p for p in (MAIN/'eval').iterdir() if p.is_dir()]
    if (MAIN/'local_artifacts').exists():paths.append(MAIN/'local_artifacts')
    for p in paths:
        source=p.resolve();dest=ARCH/'residual_outputs'/source.relative_to(MAIN)
        assert source.is_relative_to(MAIN) and dest.resolve().is_relative_to(ARCH) and not dest.exists()
        # Verify pre-switch evidence exists before moving a directory.
        if source.is_dir():
            for f in source.rglob('*'):
                if f.is_file():
                    saved=ARCH/'pre_clean_source'/f.relative_to(MAIN)
                    assert saved.is_file() and sha(f)==sha(saved),str(f)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(source),str(dest))
        moved.append({'old':str(source),'new':str(dest)})
    assert models()==before['model_sha256']
    assert all(sha(MAIN/p)==h for p,h in before['environment_sha256'].items())
    assert git(MAIN,'rev-parse','HEAD').decode().strip()==TARGET
    status=git(MAIN,'status','--porcelain','--untracked-files=all').decode()
    assert not status,status
    save('RETURN_RECEIPT.json',{**before,'status':'LATEST_SCIENCE_RETURNED_TO_DRL2','git_status':'clean','branch':BRANCH,'paths_moved':moved,'models_and_environment_unchanged':True,'simulations_rerun':False,'missing_evaluations':'Preserved as missing; no fabricated or substituted results'})
    print('DRL2_RETURNED_CLEAN_ENV_MODELS_SHA_OK',flush=True)
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='migrate':migrate()
