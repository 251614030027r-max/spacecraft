"""Publish bounded V4 scientific assets only after all scientific workers close."""
import os
os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0='D:/py/DRL2')
import json,hashlib,subprocess,shutil,gzip,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SPACE=ROOT.parents[1];REPO=Path('D:/py/DRL2');COLLAB=SPACE/'过程文件/协作/Git工作树';PY=str(REPO/'.venv/Scripts/python.exe');TARGET=REPO/'evidence/T1B_V4_RESOURCE_RESTART1_20261020';FROZEN='8684cca3d3c3a2997f3ed3f2c106c247b94e20e4';BRANCH='claude/sac-mpc-coupling-design-ns7g6i'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def git(repo,*a):return subprocess.check_output(['git','-c','safe.directory='+str(repo).replace('\\','/'),'-C',str(repo),*a],encoding='utf-8').strip()
def run(repo,*a):subprocess.run(['git','-c','safe.directory='+str(repo).replace('\\','/'),'-C',str(repo),*a],check=True)
def publish(repo,branch):
    run(repo,'fetch','origin',branch);code=subprocess.run(['git','-c','safe.directory='+str(repo).replace('\\','/'),'-C',str(repo),'merge-base','--is-ancestor','origin/'+branch,'HEAD']).returncode
    if code==1:run(repo,'merge','--no-edit','origin/'+branch)
    elif code!=0:raise RuntimeError('Ancestor check failed')
    run(repo,'push','origin','HEAD:refs/heads/'+branch);head=git(repo,'rev-parse','HEAD');assert git(repo,'ls-remote','origin','refs/heads/'+branch).split()[0]==head;return head
def main():
    exit=read(ROOT/'记录/PIPELINE_EXIT.json');assert exit['all_science_workers_closed'];assert git(REPO,'rev-parse','HEAD')==FROZEN and not git(REPO,'status','--porcelain')
    subprocess.run([PY,'-B',str(ROOT/'脚本/publish_readout.py')],cwd=REPO,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    assert not TARGET.exists(),'Never overwrite published or partly staged assets: inspect previous finalizer journal instead'
    TARGET.mkdir(parents=True);manifest={};missing=[];compressed=[]
    def copy(src,rel):
        src=Path(src)
        if not src.is_file():missing.append({'requested':rel,'status':'NOT_GENERATED; never substituted'});return
        dest=TARGET/rel;assert dest.resolve().is_relative_to(TARGET.resolve()) and not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);h=sha(src)
        if src.stat().st_size>50_000_000:
            if src.suffix!='.jsonl':missing.append({'requested':rel,'status':'over50MB','sha256':h});return
            dest=dest.with_suffix(dest.suffix+'.gz')
            with src.open('rb') as a,gzip.open(dest,'wb',compresslevel=4) as b:shutil.copyfileobj(a,b)
            with gzip.open(dest,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==h
            compressed.append({'source_name':rel,'compressed':dest.relative_to(TARGET).as_posix(),'original_sha256':h,'original_bytes':src.stat().st_size,'lossless_gzip':True})
        else:shutil.copy2(src,dest);assert sha(dest)==h
        manifest[dest.relative_to(TARGET).as_posix()]={'source':str(src),'sha256':sha(dest),'bytes':dest.stat().st_size,'original_sha256':h}
    for n in ('RESULT.md','ADMISSION.md','SAME_TASK_EVAL.csv','ASSESSMENT.json','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','MODEL_SHA256.json','TRAINING_STATUS.md','THREE_SEED_COMPARISON.json','SHARED_0K.json'):copy(ROOT/'交付构建'/n,n)
    for folder in ('输入','脚本','记录','contract'):
        for p in (ROOT/folder).glob('*'):
            if p.is_file() and (p.suffix in ('.json','.md','.py','.ps1','.csv') or p.name in ('pytest.log','contract.log','expert.log','pipeline.err.log')):copy(p,folder+'/'+p.name)
    copy(ROOT.parent/'T1B_V3_ROUND_20261019/expert/EXPERT_RECEIPT.json','expert/EXPERT_RECEIPT.json');copy(ROOT.parent/'T1B_V3_ROUND_20261019/expert/expert_replay.pkl','expert/expert_replay.pkl')
    for pat in ('M_*.json','audit_*.json','CF_LABEL_*.json','CF_LABEL_*.csv','L_JUDGMENT.json'):
        for p in ROOT.glob(pat):copy(p,p.name)
    for seed in range(3):
        for arm in ('e_sac','e_tcpf'):
            key=f'{arm}_s{seed}';path=ROOT/f'TRAIN_{key}'
            if not path.exists():missing.append({'run':key,'status':'AUTHORIZED_BUT_NOT_STARTED_BEFORE_HOST_REBOOT'});continue
            for n in ('manifest.json','finished.json','heartbeat.json','learning_audit.json','feedback_audit.json','delta_calibration.json','decisions.jsonl'):
                copy(path/n,f'training/{key}/{n}')
            if arm=='e_tcpf':copy(path/'model_prefix_releases.jsonl',f'training/{key}/model_prefix_releases.jsonl')
            for mark in (0,500,2000,5000,10000,20000):copy(path/f'checkpoints/step_{mark}.zip',f'models/{key}/step_{mark}.zip')
            for mark in (2000,5000,10000,20000):
                for p in (ROOT/f'eval_{key}_{mark}').glob('*'):
                    if p.is_file():copy(p,f'eval_{key}_{mark}/{p.name}')
    for p in (ROOT/'eval_shared_0k').glob('*'):
        if p.is_file():copy(p,'eval_shared_0k/'+p.name)
    (TARGET/'.gitattributes').write_text('* -text\n',encoding='ascii');dump(TARGET/'MISSING.json',missing);dump(TARGET/'COMPRESSED_LOGS.json',compressed);dump(TARGET/'MANIFEST.json',{'scientific_commit':FROZEN,'files':manifest,'exit':exit,'source_truth':'Original main truth at evidence/T1B_V1_AUDIT/candidate_truth_scan; expanded at evidence/T1B_V2_PILOT/scan_292120, fixed e33a0dd; original task/continuation caveat retained.'})
    files=sorted(p for p in TARGET.rglob('*') if p.is_file());(TARGET/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(TARGET).as_posix()}\n' for p in files),encoding='utf-8');files=sorted(p for p in TARGET.rglob('*') if p.is_file())
    for p in files:
        rel=p.relative_to(REPO).as_posix();run(REPO,'add','-f','--',rel);blob=subprocess.check_output(['git','-c','safe.directory=D:/py/DRL2','-C',str(REPO),'cat-file','blob',':'+rel]);assert hashlib.sha256(blob).hexdigest()==sha(p)
    assert all(not n or n.startswith('evidence/T1B_V4_RESOURCE_RESTART1_20261020/') for n in git(REPO,'diff','--cached','--name-only','-z').split('\0'));run(REPO,'commit','-q','-m','evidence(T1B_V4_ROUND): frozen actual mechanism gates and matched development results');science=publish(REPO,BRANCH)
    assert not git(COLLAB,'status','--porcelain');run(COLLAB,'fetch','origin','collab/spacecraft');run(COLLAB,'merge','--no-edit','origin/collab/spacecraft')
    hand=COLLAB/'docs/collaboration/lower/handoffs/t1b_v4_resource_restart1_20261020';hand.mkdir(parents=True,exist_ok=True)
    for n in ('RESULT.md','ADMISSION.md','SAME_TASK_EVAL.csv','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','MODEL_SHA256.json','TRAINING_STATUS.md','THREE_SEED_COMPARISON.json','SHARED_0K.json'):shutil.copy2(ROOT/'交付构建'/n,hand/n)
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();dump(COLLAB/'docs/collaboration/CURRENT.json',{'topic':'t1b_v4_round','scientific_commit':FROZEN,'evidence_commit':science,'run_order':'docs/collaboration/upper/run_orders/T1_B_V4_ROUND_20261019.md','execution_state':exit['phase'],'training_active':False,'updated_jst':stamp,'handoff':'docs/collaboration/lower/handoffs/t1b_v4_resource_restart1_20261020/RESULT.md','native_state':str(ROOT/'STATE.json'),'all_three_research_goals_claimed':False})
    for p in sorted(hand.iterdir()):run(COLLAB,'add','--',p.relative_to(COLLAB).as_posix())
    run(COLLAB,'add','--','docs/collaboration/CURRENT.json');run(COLLAB,'commit','-q','-m','lower: deliver T1B V4 frozen conditional round actual outcomes');collab=publish(COLLAB,'collab/spacecraft')
    receipt={'scientific_commit':FROZEN,'science_evidence_head':science,'collab_head':collab,'phase':exit['phase'],'evidence_files':len(files),'all_original_blobs_sha256_verified':True,'ordinary_push_only':True,'models_unmodified':True,'jst':stamp,'entry_url':'https://github.com/251614030027r-max/spacecraft/tree/'+science+'/evidence/T1B_V4_RESOURCE_RESTART1_20261020'};dump(ROOT/'记录/FINAL_PUSH_RECEIPT.json',receipt);dump(SPACE/'上层交付/最新/GIT_V4_PUSH_RECEIPT.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
