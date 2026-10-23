"""Bounded final offline readout and ordinary publication authorized by V2 run order5.
No scientific execution or training is started here.
"""
import json,hashlib,subprocess,shutil,zipfile,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SPACE=ROOT.parents[1];REPO=Path('D:/py/DRL2');PY=str(REPO/'.venv/Scripts/python.exe')
SCIENCE='a1698acefd0f3f4cf7298438bd842ef63ae0aee0';BRANCH='claude/sac-mpc-coupling-design-ns7g6i'
COLLAB=SPACE/'过程文件/协作/Git工作树';TARGET=REPO/'evidence/T1B_V2_PILOT'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(repo,*args):return subprocess.check_output(['git','-c','safe.directory='+str(repo).replace('\\','/'),'-C',str(repo),*args],encoding='utf-8').strip()
def run(repo,*args):subprocess.run(['git','-c','safe.directory='+str(repo).replace('\\','/'),'-C',str(repo),*args],check=True)
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def publish(repo,branch):
    run(repo,'fetch','origin',branch)
    code=subprocess.run(['git','-c','safe.directory='+str(repo).replace('\\','/'),'-C',str(repo),'merge-base','--is-ancestor','origin/'+branch,'HEAD']).returncode
    if code==1:run(repo,'merge','--no-edit','origin/'+branch)
    elif code!=0:raise RuntimeError('Ancestor check failed')
    run(repo,'push','origin','HEAD:refs/heads/'+branch)
    head=git(repo,'rev-parse','HEAD');remote=git(repo,'ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==head
    return head
def main():
    exit=read(ROOT/'记录/PIPELINE_EXIT.json');assert exit['all_science_workers_closed']
    assert git(REPO,'rev-parse','HEAD')==SCIENCE and not git(REPO,'status','--porcelain')
    subprocess.run([PY,'-B',str(ROOT/'脚本/audit_trajectories.py')],cwd=REPO,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    subprocess.run([PY,'-B',str(ROOT/'脚本/publish_readout.py')],cwd=REPO,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    assert not TARGET.exists(),'Never overwrite a prior V2 evidence publication'
    TARGET.mkdir(parents=True);manifest={};missing=[]
    def copy(src,relative):
        src=Path(src)
        if not src.is_file():missing.append({'requested':relative,'status':'NOT_GENERATED / no substitute'});return
        if src.stat().st_size>50_000_000:missing.append({'requested':relative,'status':'Over50MB; use exact coverage JSON instead','source_sha256':sha(src)});return
        dest=TARGET/relative;assert dest.resolve().is_relative_to(TARGET.resolve()) and not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        assert sha(dest)==sha(src);manifest[relative]={'source':str(src),'sha256':sha(dest),'bytes':dest.stat().st_size}
    for name in ['RESULT.md','SAME_TASK_EVAL.csv','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','TRAINING_STATUS.md','ASSESSMENT.json','MODEL_SHA256.json']:
        copy(ROOT/'交付构建'/name,name)
    for path in (ROOT/'记录').glob('*.json'):copy(path,'records/'+path.name)
    for name in ['pytest.log','contract.log','pipeline.err.log']:
        copy(ROOT/'记录'/name,'records/'+name)
    copy(ROOT/'contract/CONTRACT.json','CONTRACT.json')
    for path in (ROOT/'输入').glob('*.md'):copy(path,'protocol/'+path.name)
    for path in (ROOT/'脚本').glob('*.py'):copy(path,'drivers/'+path.name)
    for path in ROOT.glob('audit_*.json'):copy(path,path.name)
    for path in ROOT.glob('coverage_*.json'):copy(path,path.name)
    for name in ['B2_TRAJECTORY_AUDIT.json','B2_TRAJECTORY_COMPARISON.csv']:copy(ROOT/name,name)
    for arm in ['e_sac','e_tcpf']:
        path=ROOT/f'TRAIN_{arm}_s0'
        for name in ['manifest.json','finished.json','learning_audit.json','feedback_audit.json','delta_calibration.json','decisions.jsonl','heartbeat.json']:
            copy(path/name,f'training/{arm}/{name}')
        if arm=='e_tcpf':copy(path/'model_prefix_releases.jsonl',f'training/{arm}/model_prefix_releases.jsonl')
        for mark in [2000,5000,10000]:copy(path/f'checkpoints/step_{mark}.zip',f'models/{arm}/step_{mark}.zip')
        # Failure evidence may end before2k; never fabricate a requested model.
        if exit['failure']:
            for mark in [0,500]:
                if (path/f'checkpoints/step_{mark}.zip').exists():copy(path/f'checkpoints/step_{mark}.zip',f'models/{arm}/step_{mark}.zip')
        for mark in [2000,5000,10000]:
            for p in (ROOT/f'eval_{arm}_{mark}').glob('*'):
                if p.is_file():copy(p,f'eval_{arm}_{mark}/{p.name}')
    for p in (ROOT/'scan_292120').glob('*.json'):copy(p,'scan_292120/'+p.name)
    (TARGET/'.gitattributes').write_text('* -text\n',encoding='ascii')
    dump(TARGET/'MISSING.json',missing);dump(TARGET/'MANIFEST.json',{'scientific_commit':SCIENCE,'files':manifest,'exit':exit,'original_truth_source':'evidence/T1B_V1_AUDIT/candidate_truth_scan at a1698ac, unchanged; fixedV1 teacher continuation','no_extra_training':True})
    files=sorted(p for p in TARGET.rglob('*') if p.is_file());(TARGET/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(TARGET).as_posix()}\n' for p in files),encoding='utf-8')
    files=sorted(p for p in TARGET.rglob('*') if p.is_file())
    for p in files:
        relative=p.relative_to(REPO).as_posix();run(REPO,'add','-f','--',relative)
        blob=subprocess.check_output(['git','-c','safe.directory=D:/py/DRL2','-C',str(REPO),'cat-file','blob',':'+relative]);assert hashlib.sha256(blob).hexdigest()==sha(p)
    staged=git(REPO,'diff','--cached','--name-only','-z').split('\0');assert all(not n or n.startswith('evidence/T1B_V2_PILOT/') for n in staged)
    run(REPO,'commit','-q','-m','evidence(T1B_V2_PILOT): seed0 fixed pilot actual outcomes and preregistered gates')
    science_head=publish(REPO,BRANCH)
    handoff=COLLAB/'docs/collaboration/lower/handoffs/t1b_v2_pilot_20261019';handoff.mkdir(parents=True,exist_ok=True)
    for name in ['RESULT.md','SAME_TASK_EVAL.csv','ASSESSMENT.json','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','MODEL_SHA256.json','TRAINING_STATUS.md']:
        shutil.copy2(ROOT/'交付构建'/name,handoff/name)
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    current={'topic':'t1b_v2_pilot','scientific_commit':SCIENCE,'evidence_commit':science_head,'run_order':'docs/collaboration/upper/run_orders/T1_B_V2_PILOT_20261019.md','updated_jst':stamp,'execution_state':exit['phase'],'training_active':False,'native_state':str(ROOT/'STATE.json'),'handoff':'docs/collaboration/lower/handoffs/t1b_v2_pilot_20261019/RESULT.md','scope':'seed0 pilot only; A1/A2 failure no retune; no multi-seed GO','all_three_research_goals_claimed':False}
    dump(COLLAB/'docs/collaboration/CURRENT.json',current)
    for p in sorted(handoff.iterdir()):run(COLLAB,'add','--',p.relative_to(COLLAB).as_posix())
    run(COLLAB,'add','--','docs/collaboration/CURRENT.json');run(COLLAB,'commit','-q','-m','lower: deliver frozen T1B V2 seed0 pilot actual gate and paired outcomes')
    collab_head=publish(COLLAB,'collab/spacecraft')
    receipt={'scientific_commit':SCIENCE,'science_evidence_head':science_head,'collab_head':collab_head,'phase':exit['phase'],'evidence_files':len(files),'individual_force_add':True,'all_original_evidence_blobs_sha256_verified':True,'ordinary_push_only':True,'models_unmodified':True,'new_training_started':False,'timestamp_jst':stamp,'entry_url':'https://github.com/251614030027r-max/spacecraft/tree/'+science_head+'/evidence/T1B_V2_PILOT'}
    dump(ROOT/'记录/FINAL_PUSH_RECEIPT.json',receipt);dump(SPACE/'上层交付/最新/GIT_V2_PUSH_RECEIPT.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
