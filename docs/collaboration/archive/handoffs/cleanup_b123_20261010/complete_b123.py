"""B1-3 only in collaboration checkout; no active scientific checkout Git."""
from pathlib import Path
import subprocess,json,csv,io,hashlib,shutil,shlex
BASE=Path(r'C:\Users\35884\Documents\Spacecraft');TOP=BASE/'过程文件/整理_20261010';REC=TOP/'记录/B123';REC.mkdir(parents=True,exist_ok=True)
REPO=BASE/'过程文件/协作/Git工作树';C=REPO/'docs/collaboration'
FREEZE='813e27f982ee24076d7ef9ae2cbb6061e0881c42';CLEAN='60d768ab47a3236ebe4cf3fcd05553ed4652c792';FROZEN='f2f8169acd580ea96a578d45022dd8952447eca5'
BRANCHES={'v2-evaluation-20260924':'736edbca691336efe8e5296bc2f10d8a39e35190',
          'v3-t0b-20260925':'f50eb78350604431fb042eca6b3f655cd5c26bab',
          'review/v3e-60k-value-r2-20260930':'dc6682d4ec11e5858bedc9e619ac6c4142fc7e66',
          'review/v3e-stage-c-20261002':'b383cd285de78561f116b1dc6cdab10fe3694088',
          'review/stopping-20261006-audit':'6c5cb97a319e425bd616d4fe323ed1a9c43902a4',
          'review/stopping-value-check-20261006':'2d8a5818037b6d753f4866969fae372da6cbef84'}
TAGS={'archive/pre-cleanup-20261010':FREEZE,**{'archive/'+b:s for b,s in BRANCHES.items()}}
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],stderr=subprocess.STDOUT)
def refs(*a):return {line.split()[1]:line.split()[0] for line in git('ls-remote',*a,'origin').decode().splitlines()}
def save(name,data):(REC/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def tree(ref):
    out={}
    for x in git('ls-tree','-r','-z',ref).split(b'\0'):
        if x:
            h,p=x.split(b'\t',1);out[p.decode()]=h.split()[2].decode()
    return out
def sha(b):return hashlib.sha256(b).hexdigest()
def verify_cleanup():
    old,new,frozen=tree(FREEZE),tree(CLEAN),tree(FROZEN)
    roots=('env/','controllers/','dynamics/','estimation/','train/','experiments/','eval/')
    kept=[p for p in new if p.startswith(roots) and p.endswith(('.py','.ps1'))]
    assert all(new[p]==frozen[p] for p in kept)
    art=list(csv.DictReader(io.StringIO(git('show',CLEAN+':docs/archive/ARTIFACT_INDEX.csv').decode('utf-8-sig'))))
    chinese=[]
    for r in art:
        assert r['path'] in old and old[r['path']]==r['blob_sha1'],r['path']
        if any(ord(x)>127 for x in r['path']):
            content=git('show',FREEZE+':'+r['path'])
            assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==r['blob_sha1']
            chinese.append(r['path'])
    dead=git('show',CLEAN+':docs/DEAD_ENDS.md').decode()
    assert '无违规完成 20/10/0' not in dead and '零违规回合 20/10/0' in dead
    assert '02_...' not in dead and '05_...' not in dead
    save('CLEANUP_FIX_VERIFICATION.json',{'commit':CLEAN,'retained_code_files':len(kept),'unchanged':True,'artifact_rows':len(art),'retrieved_chinese_paths':len(chinese),'chinese_paths':chinese,'dead_ends_wording_checked':True,'tests_not_run':True})
    print('CLEANUP_VERIFIED',len(kept),len(art),len(chinese),flush=True)
def create_tags():
    before=refs('--heads','--tags');save('REMOTE_BEFORE.json',before)
    for branch,commit in BRANCHES.items():assert before.get('refs/heads/'+branch)==commit,branch+' changed'
    for name,commit in TAGS.items():
        local=subprocess.run(['git','-C',str(REPO),'rev-parse','--verify','refs/tags/'+name],capture_output=True)
        if local.returncode==0:
            assert git('rev-parse','refs/tags/'+name+'^{}').decode().strip()==commit
            assert git('cat-file','-t','refs/tags/'+name).decode().strip()=='tag'
        else:git('tag','-a',name,commit,'-m','Immutable archive before cleanup; retain original paths, evidence and branch tip.')
    save('TAG_PLAN.json',TAGS);print('ANNOTATED_TAGS_READY',len(TAGS),flush=True)
def verify_remote_tags():
    remote=refs('--heads','--tags')
    for name,commit in TAGS.items():assert remote.get('refs/tags/'+name+'^{}')==commit,name
    for branch,commit in BRANCHES.items():assert remote.get('refs/heads/'+branch)==commit,branch+' changed'
    save('REMOTE_TAGS_VERIFIED.json',remote);print('REMOTE_TAGS_VERIFIED_7',flush=True)
def verify_deleted():
    remote=refs('--heads','--tags')
    assert set(x for x in remote if x.startswith('refs/heads/'))=={'refs/heads/main','refs/heads/claude/sac-mpc-coupling-design-ns7g6i','refs/heads/collab/spacecraft'}
    for name,commit in TAGS.items():assert remote.get('refs/tags/'+name+'^{}')==commit
    before=json.loads((REC/'REMOTE_BEFORE.json').read_text(encoding='utf-8'))
    for branch in ('main','claude/sac-mpc-coupling-design-ns7g6i','collab/spacecraft'):assert remote['refs/heads/'+branch]==before['refs/heads/'+branch]
    save('REMOTE_BRANCHES_VERIFIED.json',remote);print('REMOTE_3_BRANCHES_ARCHIVES_PRESERVED',flush=True)
def scripts():
    assert git('branch','--show-current').decode().strip()=='collab/spacecraft'
    dest=C/'lower/pipeline_scripts';assert not dest.exists(),'Existing pipeline package; inspect before overwrite'
    old=tree(FREEZE);oldhash={}
    for path,blob in old.items():oldhash.setdefault(blob,[]).append(path)
    source=C/'lower/handoffs/cleanup_inventory_20261010/untracked_code.csv'
    with source.open(encoding='utf-8-sig',newline='') as f:candidates=[r for r in csv.DictReader(f) if r['recommendation']=='COMMIT_TO_GIT']
    assert len(candidates)==106,len(candidates)
    seen={};index=[];stats={'candidates':len(candidates),'copied':0,'identical_to_git':0,'duplicate':0}
    for r in candidates:
        src=Path(r['path']);assert src.suffix.lower() in ('.py','.ps1','.bat','.ipynb')
        assert '/eval/final2/' not in src.as_posix().lower() and '/logs/final2_262461/' not in src.as_posix().lower()
        content=src.read_bytes();digest=sha(content);blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        experiment=r['experiment_id'];target=dest/('_unassigned' if experiment=='ORPHAN' else experiment)/src.name
        if blob in oldhash:
            known=sorted(oldhash[blob])[0];action=f'identical_to_git:{known}@{FREEZE}';stats['identical_to_git']+=1
        elif digest in seen:action='duplicate_of:'+seen[digest];stats['duplicate']+=1
        else:
            if target.exists():target=target.with_name(target.stem+'__'+digest[:12]+target.suffix)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content);assert sha(target.read_bytes())==digest
            seen[digest]=target.relative_to(C).as_posix();action='copied';stats['copied']+=1
        index.append({'original_local_path':str(src),'sha256':digest,'experiment_id':experiment,'handling':action,
                      'stored_path':seen.get(digest,''),'run_order_or_handoff':r['evidence_refs'],
                      'provenance_limit':r['produced_decision_numbers']})
    dest.mkdir(exist_ok=True)
    with (dest/'INDEX.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
    (dest/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    (dest/'README.md').write_text('# 历史流水线代码归档\n\n证据链归档，不是新的执行授权。先读当前执行单，不直接运行历史启动/训练/清理脚本。保留原字节与SHA；修改后需新版本，不能覆盖历史证据。106个入库建议是保守候选，不证明逐脚本均已执行；provenance_limit保留盘点的不确定性。\n\n'+json.dumps(stats,ensure_ascii=False)+'\n',encoding='utf-8')
    save('PIPELINE_STATS.json',stats);print('PIPELINE',json.dumps(stats),flush=True)
if __name__=='__main__':
    import sys
    {'verify-cleanup':verify_cleanup,'create-tags':create_tags,'verify-tags':verify_remote_tags,'verify-deleted':verify_deleted,'scripts':scripts}[sys.argv[1]]()
