"""Ordinary immutable evidence pushes through a temporary index; live HEAD stays frozen."""
import os,sys,json,hashlib,subprocess,datetime,zipfile,gzip,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SPACE=ROOT.parents[1];REPO=Path('D:/py/DRL2');COLLAB=SPACE/'过程文件/协作/Git工作树';PY=str(REPO/'.venv/Scripts/python.exe')
FROZEN='b6f5c8a22728b6ca3970d6201525cd4047cba8d8';BRANCH='claude/sac-mpc-coupling-design-ns7g6i'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def git(repo,*args,env=None,data=None,raw=False):
    b=subprocess.check_output(['git','-c','safe.directory='+str(repo).replace('\\','/'),'-C',str(repo),*args],env=env,input=data)
    return b if raw else b.decode('utf-8').strip()
def main(label):
    assert label in ('midterm_5000','midterm_10000','final','user_review_20261020_1525')
    build=ROOT/'发布快照'/label;snapshot=build/'assets';prefix='evidence/T1B_V51_ROUND/'+label
    assert git(REPO,'rev-parse','HEAD')==FROZEN and not git(REPO,'status','--porcelain')
    head=git(REPO,'rev-parse','HEAD');index_hash=sha(REPO/'.git/index')
    if not (build/'SNAPSHOT_READY.json').exists():
        assert not build.exists(),'Partial local snapshot exists; inspect it before retry, never overwrite'
        subprocess.run([PY,'-B',str(ROOT/'脚本/publish_readout.py')],cwd=REPO,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        snapshot.mkdir(parents=True);cuts={}
        def copy(p,n,cut=False):
            if not p.is_file():return
            b=p.read_bytes()
            if cut:b=b[:b.rfind(b'\n')+1] if b'\n' in b else b'';cuts[n]={'source':str(p),'complete_line_prefix_bytes':len(b)}
            target=snapshot/n;target.parent.mkdir(parents=True,exist_ok=True)
            if len(b)>50_000_000:
                assert n.endswith('.jsonl');target=target.with_suffix('.jsonl.gz')
                with gzip.open(target,'wb',compresslevel=3) as f:f.write(b)
                cuts[n]={**cuts.get(n,{}),'original_sha256':hashlib.sha256(b).hexdigest(),'lossless_gzip':target.relative_to(snapshot).as_posix()}
            else:target.write_bytes(b)
        for p in (ROOT/'交付构建').iterdir():
            if p.is_file() and p.suffix!='.zip':copy(p,p.name)
        # Select original raw sources directly rather than nesting a delivery ZIP.
        for folder in ('输入','脚本','记录','contract'):
            for p in (ROOT/folder).glob('*'):
                if p.is_file() and p.suffix in ('.json','.md','.py','.csv'):copy(p,folder+'/'+p.name)
        copy(ROOT/'记录/pytest.log','pytest.log');copy(ROOT/'STATE.json','STATE_SNAPSHOT.json')
        copy(ROOT.parent/'T1B_V3_ROUND_20261019/expert/EXPERT_RECEIPT.json','expert/EXPERT_RECEIPT.json')
        copy(ROOT.parent/'T1B_V3_ROUND_20261019/expert/expert_replay.pkl','expert/expert_replay.pkl')
        for p in ROOT.glob('audit_*.json'):copy(p,p.name)
        for p in ROOT.glob('M_*.json'):copy(p,p.name)
        for p in ROOT.glob('CF_LABEL_*'):copy(p,p.name)
        for p in ROOT.glob('eval_*/*'):
            if p.is_file():copy(p,p.relative_to(ROOT).as_posix(),p.suffix=='.jsonl')
        model_hashes={}
        for s in range(3):
            for arm in ('e_sac','e_tcpf'):
                key=f'{arm}_s{s}';base=ROOT/f'TRAIN_{key}'
                for p in base.glob('*'):
                    if p.is_file() and p.suffix in ('.json','.jsonl'):copy(p,'training/'+key+'/'+p.name,p.suffix=='.jsonl')
                for p in (base/'checkpoints').glob('step_*.zip'):
                    with zipfile.ZipFile(p) as z:
                        if z.testzip() is not None:continue
                    copy(p,'models/'+key+'/'+p.name);model_hashes[key+'/'+p.name]=sha(p)
        dump(snapshot/'LOG_CUTS.json',cuts);dump(snapshot/'MODEL_SHA256.json',model_hashes)
        for folder,pattern in [('T1B_V1_AUDIT/candidate_truth_scan','r_*.json'),('T1B_V2_PILOT/scan_292120','scan_*.json')]:
            for p in (REPO/'evidence'/folder).glob(pattern):copy(p,'truth/'+folder.split('/')[-1]+'/'+p.name)
        (snapshot/'.gitattributes').write_text('* -text\n',encoding='ascii')
        manifest={p.relative_to(snapshot).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in snapshot.rglob('*') if p.is_file()}
        dump(snapshot/'MANIFEST.json',{'scientific_commit':FROZEN,'stage':label,'files':manifest,'snapshot_jst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'complete_line_log_cuts':cuts,'all_three_research_goals_claimed':False})
        dump(build/'SNAPSHOT_READY.json',{'files':{p.relative_to(snapshot).as_posix():sha(p) for p in snapshot.rglob('*') if p.is_file()}})
    ready=read(build/'SNAPSHOT_READY.json')
    for n,h in ready['files'].items():assert sha(snapshot/n)==h
    if (build/'SCIENCE_PUSH_RECEIPT.json').exists():science=read(build/'SCIENCE_PUSH_RECEIPT.json')['commit']
    else:
        git(REPO,'fetch','origin',BRANCH);parent=git(REPO,'rev-parse','origin/'+BRANCH);env=os.environ.copy();env['GIT_INDEX_FILE']=str(build/'temporary_git_index');git(REPO,'read-tree',parent,env=env)
        for n in sorted(ready['files']):
            p=snapshot/n;oid=git(REPO,'hash-object','-w','--stdin',data=p.read_bytes());path=prefix+'/'+n;git(REPO,'update-index','--add','--cacheinfo','100644',oid,path,env=env)
            assert hashlib.sha256(git(REPO,'cat-file','blob',':'+path,env=env,raw=True)).hexdigest()==sha(p)
        tree=git(REPO,'write-tree',env=env);science=git(REPO,'commit-tree',tree,'-p',parent,data=('evidence(T1B_V51): immutable '+label+' actual matched results; running freeze unchanged\n').encode())
        dump(build/'COMMIT_CREATED.json',{'commit':science,'parent':parent,'tree':tree});git(REPO,'push','origin',science+':refs/heads/'+BRANCH)
        assert git(REPO,'ls-remote','origin','refs/heads/'+BRANCH).split()[0]==science;dump(build/'SCIENCE_PUSH_RECEIPT.json',{'commit':science})
    assert git(REPO,'rev-parse','HEAD')==head and sha(REPO/'.git/index')==index_hash and not git(REPO,'status','--porcelain')
    assert not git(COLLAB,'status','--porcelain');git(COLLAB,'fetch','origin','collab/spacecraft');git(COLLAB,'merge','--no-edit','origin/collab/spacecraft')
    hand=COLLAB/'docs/collaboration/lower/handoffs/t1b_v51_round_20261020'/label;hand.mkdir(parents=True,exist_ok=True)
    for name in ('RESULT.md','ADMISSION.md','SAME_TASK_EVAL.csv','POLICY_AND_GATE_AUDIT.json','MPC_FULL_HORIZON_AUDIT.json','THREE_SEED_COMPARISON.json','MODEL_SHA256.json','记录/INTERIM_DIAGNOSIS_1515.md','记录/EXISTING_SIGNAL_DIAGNOSIS.json'):
        if not (snapshot/name).is_file():continue
        (hand/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(snapshot/name,hand/name);git(COLLAB,'add','--', (hand/name).relative_to(COLLAB).as_posix())
    state=read(ROOT/'STATE.json');dump(COLLAB/'docs/collaboration/CURRENT.json',{'topic':'t1b_v51_round','scientific_commit':FROZEN,'run_order':'docs/collaboration/upper/run_orders/T1_B_V51_ROUND_20261020.md','evidence_commit':science,'execution_state':state['phase'],'training_active':state['training_active'],'native_state':str(ROOT/'STATE.json'),'handoff':hand.relative_to(COLLAB).as_posix()+'/RESULT.md','all_three_research_goals_claimed':False})
    git(COLLAB,'add','--','docs/collaboration/CURRENT.json');git(COLLAB,'commit','-q','-m','lower: V5.1 '+label+' factual review, preserve active freeze');git(COLLAB,'push','origin','HEAD:refs/heads/collab/spacecraft');collab=git(COLLAB,'rev-parse','HEAD');assert git(COLLAB,'ls-remote','origin','refs/heads/collab/spacecraft').split()[0]==collab
    receipt={'stage':label,'scientific_commit':FROZEN,'science_evidence_commit':science,'collab_commit':collab,'files':len(ready['files']),'all_blob_sha256_verified':True,'real_scientific_head_and_index_unchanged':True,'ordinary_push_only':True,'entry_url':'https://github.com/251614030027r-max/spacecraft/tree/'+science+'/'+prefix};dump(build/'PUSH_RECEIPT.json',receipt);dump(ROOT/f'记录/PUSH_{label}.json',receipt);dump(SPACE/'上层交付/最新'/('PUSH_'+label+'.json'),receipt);print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main(sys.argv[1])
