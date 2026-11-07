"""Immutable midterm snapshots + fast-forward publication via a temporary Git index.
Keep the actively running scientific checkout HEAD, real index and modules untouched.
"""
import os,json,csv,hashlib,subprocess,shutil,datetime,collections,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=Path('D:/py/DRL2');COLLAB=ROOT.parents[1]/'过程文件/协作/Git工作树';FROZEN='e33a0dd7b0bd0d73b7d3ad16561f1dd494d242fa';BRANCH='claude/sac-mpc-coupling-design-ns7g6i';NAME='T1B_V3_MIDTERM_20261019';PREFIX='evidence/'+NAME;BUILD=ROOT/'中期审查_10k_v2';PY=str(REPO/'.venv/Scripts/python.exe')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def git(repo,*a,env=None,raw=False,data=None):
    r=subprocess.check_output(['git','-c','safe.directory='+str(repo).replace('\\','/'),'-C',str(repo),*a],env=env,input=data)
    return r if raw else r.decode('utf-8').strip()
def csvwrite(p,data):
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
    assert git(REPO,'rev-parse','HEAD')==FROZEN and not git(REPO,'status','--porcelain');before_index=sha(REPO/'.git/index');before_head=git(REPO,'rev-parse','HEAD')
    assert not BUILD.exists(),'Do not overwrite an existing midterm snapshot or partial publication; inspect receipt first'
    # Refresh independent local readout; publisher serializes with its native watcher.
    with (ROOT/'记录/midterm_readout.log').open('w',encoding='utf-8') as log:subprocess.run([PY,'-B',str(ROOT/'脚本/publish_readout.py')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    BUILD.mkdir();snapshot=BUILD/'assets';snapshot.mkdir();manifest={};cuts={}
    def copy(src,rel,log=False):
        src=Path(src)
        if not src.is_file():return
        data=src.read_bytes();original_size=len(data)
        if log:
            end=data.rfind(b'\n');data=data[:end+1] if end>=0 else b''
        dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        if log:cuts[rel]={'source':str(src),'read_bytes':original_size,'complete_prefix_bytes':len(data),'sha256':sha(dest),'snapshot_jst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}
        manifest[rel]={'source':str(src),'bytes':len(data),'sha256':sha(dest)}
    for n in ('RESULT.md','ADMISSION.md','SAME_TASK_EVAL.csv','ASSESSMENT.json','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','MODEL_SHA256.json','TRAINING_STATUS.md'):copy(ROOT/'交付构建'/n,n)
    copy(ROOT/'STATE.json','STATE_SNAPSHOT.json');copy(ROOT/'expert/EXPERT_RECEIPT.json','expert/EXPERT_RECEIPT.json');copy(ROOT/'expert/expert_replay.pkl','expert/expert_replay.pkl');copy(ROOT/'contract/CONTRACT.json','CONTRACT.json');copy(ROOT/'记录/pytest.log','pytest.log');copy(ROOT/'输入/T1_B_V3_ROUND_20261019.md','RUN_ORDER.md')
    for p in ROOT.glob('M_*.json'):copy(p,p.name)
    for p in ROOT.glob('audit_*.json'):copy(p,p.name)
    for p in ROOT.glob('CF_LABEL_*'):copy(p,p.name)
    coverage=[];intervals=[];counts={};modelhash={}
    for arm in ('e_sac','e_tcpf'):
        base=ROOT/f'TRAIN_{arm}_s0'
        for n in ('manifest.json','heartbeat.json','learning_audit.json','feedback_audit.json','delta_calibration.json'):copy(base/n,f'training/{arm}/{n}')
        for n in ('decisions.jsonl','model_prefix_releases.jsonl'):copy(base/n,f'training/{arm}/{n}',True)
        for mark in (0,2000,5000,10000):
            p=base/f'checkpoints/step_{mark}.zip';assert p.exists();copy(p,f'models/{arm}/step_{mark}.zip');modelhash[f'{arm}_{mark}']=sha(p)
        data=[json.loads(x) for x in (snapshot/f'training/{arm}/decisions.jsonl').read_bytes().splitlines()];buckets={};opened={};episodes=[];episodecounts=collections.Counter()
        for r in data:
            e=r['env'];learn=r['support']['learning_decision'];step=r['step']
            if learn:
                if e in opened:
                    z=opened.pop(e);intervals.append({**z,'end_global_step':step,'terminal':False,'end_event':'next_learning_decision','bootstrap_discount':.999**z['length'],'audit_only':'Reconstructed from persisted real rewards, not original saved sampled TD targets'})
                opened[e]={'arm':arm,'env':e,'start_global_step':step,'candidate_index':r['support']['candidate_index'],'has_plan_at_start':r['observation_before'][27],'length':0,'discounted_real_interval_return':0.};episodecounts[e]+=1
                k=((step-1)//1000,int(r['support']['candidate_index']));buckets[k]=buckets.get(k,0)+1
            if e in opened:
                z=opened[e];z['discounted_real_interval_return']+=.999**z['length']*float(r['reward']);z['length']+=1
            if r.get('event'):
                episodes.append(episodecounts.pop(e,0))
                if e in opened:
                    z=opened.pop(e);intervals.append({**z,'end_global_step':step,'terminal':True,'end_event':r['event'],'bootstrap_discount':0.,'audit_only':'Reconstructed from persisted real rewards, not original saved sampled TD targets'})
        for (bucket,j),n in sorted(buckets.items()):coverage.append({'arm':arm,'global_step_begin':bucket*1000+1,'global_step_end':(bucket+1)*1000,'candidate_index':j,'decision_count':n,'is_expert':False,'partial_tail':(bucket+1)*1000>max(r['step'] for r in data)})
        counts[arm]={'last_complete_logged_step':max(r['step'] for r in data),'real_transition_rows':len(data),'learning_decisions':sum(r['support']['learning_decision'] for r in data),'learning_with_plan':sum(r['support']['learning_decision'] and r['observation_before'][27]>.5 for r in data),'completed_episode_decision_counts':episodes,'unclosed_intervals':list(opened.values()),'critic_updates_and_probe_details':'training/'+arm+'/learning_audit.json','coverage_source':'Raw complete-line prefix, online only. Expert is supplied separately as original1000-transition pickle.'}
        for mark in (2000,5000,10000):
            for p in (ROOT/f'eval_{arm}_s0_{mark}').glob('*'):
                if p.is_file():copy(p,f'eval_{arm}_{mark}/{p.name}')
        for n in (f'{arm}_s0.err.log',f'{arm}_s0.out.log'):copy(ROOT/'记录'/n,'logs/'+n,True)
    csvwrite(snapshot/'REPLAY_COVERAGE_BY_1K.csv',coverage);csvwrite(snapshot/'REAL_DECISION_INTERVALS.csv',intervals);dump(snapshot/'REPLAY_SNAPSHOT_AUDIT.json',counts);dump(snapshot/'MODEL_SHA256.json',modelhash);dump(snapshot/'LOG_CUTS.json',cuts)
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();a=read(snapshot/'ASSESSMENT.json');rows=list(csv.DictReader((snapshot/'SAME_TASK_EVAL.csv').open(encoding='utf-8-sig')));table=[]
    for mark in (2000,5000,10000):
        for arm in ('e_sac','e_tcpf'):
            for block in ('292140–147','292120–127'):
                rr=[r for r in rows if r['arm']==arm and int(r['checkpoint'])==mark and r['block']==block];assert len(rr)==8
                table.append(f"|{mark}|{arm}|{block}|{sum(int(r['capture']) for r in rr)}/8|{sum(float(r['discounted_return']) for r in rr)/8:.9f}|{sum(r['first_index'] not in ('','0') for r in rr)}|")
    text=f'''# V3 10k 中期审查：训练尚未完成，未改方案

冻结科学 {FROZEN}；seed0，快照时间 {stamp}。运行中实际时钟逐臂见STATE_SNAPSHOT/heartbeat与LOG_CUTS，快照并非暂停后同时取样；日志只保留完整行前缀，所有文件SHA可核验。两臂均完成0/2k/5k/10k模型，2k/5k/10k两块共96条完整评估。本报告不是20k L最终判定，不提前执行种子1/2。

## 中期事实

13项测试/合同通过；共享V3专家1000步，SHA 9c91cb76f10999ca36aecf4172bb092e4e52de9fcfbf62402a76306e701e1d0f。M在2k两臂均通过：无已有计划学习决策、平均1.63/1.54次决策，原8熵均8/8>.05，TCPF有效配对39对（更早12/更晚27）。M仅机制准入，不代表持续学习成功。

|步数|臂|开发块|捕获|平均原折扣回报|首选非最早开局数|
|---|---|---|---:|---:|---:|
'''+ '\n'.join(table)+'''

10k两臂主开发块均退回全部最早，所有8状态Q最高均为最早；原块均7/8、36.538164，扩展均6/8、16.220946，两臂逐例一致。TCPF在5k的主块36.787153、扩展17.489189小幅差异没有保持到10k。两臂10k原8 entropy>.05均0/8；E-SAC的p=[1,0,0]（float32），E-TCPF最早概率约0.99597–0.99993。不能继续用“梯度有值/机制通过”声称学习有效或MPC独立收益成立。

例292140（经济型）：E-SAC10k Qmin最早/中/晚约−10.13/−34.31/−59.36，E-TCPF约−8.35/−11.17/−13.86；两者偏好均最早。固定V1教师后续真值早100s约−129.83、晚140s约+56，是需要上层重点复核的排序矛盾；V3承诺后计划丢失可能重决策，不能把固定教师后续回报直接当SAC当前策略严格Q。

## 审查材料及边界

8个原始模型ZIP（两臂0/2k/5k/10k）与SHA；6份逐状态概率、均值/标准差、Q1/Q2审计；96条逐开局结果与逐步观察/动作轨迹；各臂manifest、heartbeat、learning_audit，TCPF有效配对原始payload与门控、校准；在线完整审议/真实步日志、按1k分桶候选覆盖、真实决策区间重构表；V3原始专家pickle及回执；CF_LABEL逐对即时差、自举差和Y。区间重构使用实际持久化奖励，不冒充原保存Replay内某次采样TD目标；CF_LABEL用冻结检查点target critic和当时留存配对复算，不冒充历史每次更新标签。当前学习探针日志含真实TD、辅助损失、gTD/gCF、实际权重、梯度份额及Actor梯度，请据此联合判断，不预设根因。

历史必要数值：主块总是最早7/8、约36.54，真值最佳首选8/8、约60.87；扩展总是最早6/8、约16.38，最佳首选6/8、约18.70。原始真值已在同一冻结代码evidence/T1B_V1_AUDIT/candidate_truth_scan和evidence/T1B_V2_PILOT/scan_292120，本轮必要副本与SHA列于MANIFEST；不复跑、不挑开局。V1两臂主块2k→10k始终7/8、35.582057，扩展10k6/8、15.505363；V2两臂2k主块7/8，E-SAC36.754445/E-TCPF35.582057，5k E-SAC原8熵仅2/8按当时A1停止。接口与后续规则不同，跨轮回报不是严格同条件消融。

## 仍在执行的权限

继续原V3冻结两臂20k；M已过，不将V2的5k A1套回V3；20k L要求至少一臂主块平均回报>36.54且有首选非最早开局，通过才自动seed1/2各两臂20k。当前无新训练器、奖励、温度、覆盖、任务或安全修改。中期推送采用独立临时Git索引+普通快进push，本机运行中的科学HEAD、真实index及代码不变，最终交付会在训练结束后普通合并远端证据；不强推、不覆盖模型。
'''
    (snapshot/'MIDTERM_REVIEW.md').write_text(text,encoding='utf-8')
    # Required truth is small; copy immutable originals with exact provenance.
    for folder,pattern in [('T1B_V1_AUDIT/candidate_truth_scan','r_*.json'),('T1B_V2_PILOT/scan_292120','scan_*.json')]:
        for p in (REPO/'evidence'/folder).glob(pattern):copy(p,'truth/'+folder.split('/')[-1]+'/'+p.name)
    copy(Path(__file__),'drivers/midterm_push.py');copy(ROOT/'脚本/run_round.py','drivers/run_round.py');copy(ROOT/'脚本/publish_readout.py','drivers/publish_readout.py')
    # Small exact-byte snapshots, no old archives or scientific worktree writes.
    for p in sorted(snapshot.rglob('*')):
        if not p.is_file():continue
        if p.stat().st_size>50_000_000:
            assert p.suffix=='.jsonl'
            dest=p.with_suffix('.jsonl.gz')
            with p.open('rb') as src,gzip.open(dest,'wb',compresslevel=4) as dst:shutil.copyfileobj(src,dst)
            with gzip.open(dest,'rb') as src:assert hashlib.sha256(src.read()).hexdigest()==sha(p)
            # Keep local original; Git packaging selects only compressed representation.
            manifest[p.relative_to(snapshot).as_posix()]={'source':str(p),'original_sha256':sha(p),'git_lossless_gzip':dest.relative_to(snapshot).as_posix()}
    (snapshot/'.gitattributes').write_text('* -text\n',encoding='ascii')
    files=[p for p in snapshot.rglob('*') if p.is_file() and not (p.suffix=='.jsonl' and p.stat().st_size>50_000_000)]
    payload_manifest={p.relative_to(snapshot).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size,'provenance':manifest.get(p.relative_to(snapshot).as_posix(),{})} for p in files};dump(snapshot/'MANIFEST.json',{'scientific_commit':FROZEN,'snapshot_jst':stamp,'files':payload_manifest,'log_cuts':cuts,'scope':'Immutable midterm only; training ongoing. Raw prefix capture, no fabricated terminal/final weights.'})
    (snapshot/'SHA256SUMS').write_text(''.join(f"{sha(p)}  {p.relative_to(snapshot).as_posix()}\n" for p in sorted(files)),encoding='utf-8');files.extend([snapshot/'MANIFEST.json',snapshot/'SHA256SUMS'])
    git(REPO,'fetch','origin',BRANCH);parent=git(REPO,'rev-parse','origin/'+BRANCH);index=BUILD/'temporary_git_index';env=os.environ.copy();env['GIT_INDEX_FILE']=str(index);git(REPO,'read-tree',parent,env=env)
    for p in sorted(files):
        path=PREFIX+'/'+p.relative_to(snapshot).as_posix();oid=git(REPO,'hash-object','-w','--stdin',data=p.read_bytes());git(REPO,'update-index','--add','--cacheinfo','100644',oid,path,env=env);assert hashlib.sha256(git(REPO,'cat-file','blob',':'+path,env=env,raw=True)).hexdigest()==sha(p)
    tree=git(REPO,'write-tree',env=env);commit=git(REPO,'commit-tree',tree,'-p',parent,data=b'evidence(T1B_V3): immutable seed0 midterm 0/2k/5k/10k review without changing running HEAD\n');dump(BUILD/'COMMIT_CREATED.json',{'commit':commit,'parent':parent,'tree':tree})
    git(REPO,'push','origin',commit+':refs/heads/'+BRANCH);assert git(REPO,'ls-remote','origin','refs/heads/'+BRANCH).split()[0]==commit
    assert git(REPO,'rev-parse','HEAD')==before_head and sha(REPO/'.git/index')==before_index and not git(REPO,'status','--porcelain')
    assert not git(COLLAB,'status','--porcelain');git(COLLAB,'fetch','origin','collab/spacecraft');git(COLLAB,'merge','--no-edit','origin/collab/spacecraft')
    hand=COLLAB/'docs/collaboration/lower/handoffs/t1b_v3_midterm_20261019';hand.mkdir(parents=True,exist_ok=True)
    for n in ('MIDTERM_REVIEW.md','SAME_TASK_EVAL.csv','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json','REPLAY_SNAPSHOT_AUDIT.json','MODEL_SHA256.json','MANIFEST.json'):shutil.copy2(snapshot/n,hand/n)
    current=read(COLLAB/'docs/collaboration/CURRENT.json');current.update(topic='t1b_v3_round',scientific_commit=FROZEN,execution_state='seed0_training_and_readout',training_active=True,midterm_evidence_commit=commit,midterm_handoff='docs/collaboration/lower/handoffs/t1b_v3_midterm_20261019/MIDTERM_REVIEW.md',updated_jst=stamp,native_state=str(ROOT/'STATE.json'));dump(COLLAB/'docs/collaboration/CURRENT.json',current)
    for p in sorted(hand.iterdir()):git(COLLAB,'add','--',p.relative_to(COLLAB).as_posix())
    git(COLLAB,'add','--','docs/collaboration/CURRENT.json');git(COLLAB,'commit','-q','-m','lower: V3 seed0 immutable 10k midterm review; no change to active training');git(COLLAB,'push','origin','HEAD:refs/heads/collab/spacecraft');collab=git(COLLAB,'rev-parse','HEAD');assert git(COLLAB,'ls-remote','origin','refs/heads/collab/spacecraft').split()[0]==collab
    receipt={'scientific_commit':FROZEN,'science_evidence_commit':commit,'collab_commit':collab,'midterm_files':len(files),'model_zips':8,'complete_eval_rows':len(rows),'all_staged_blob_sha256_verified':True,'running_head_unchanged':True,'real_index_sha256_unchanged':True,'ordinary_push_only':True,'training_not_restarted_or_changed':True,'snapshot_jst':stamp,'entry_url':'https://github.com/251614030027r-max/spacecraft/tree/'+commit+'/'+PREFIX};dump(ROOT/'记录/MIDTERM_PUSH_RECEIPT.json',receipt)
    latest=ROOT.parents[1]/'上层交付/最新';shutil.copy2(snapshot/'MIDTERM_REVIEW.md',latest/'MIDTERM_REVIEW.md');dump(latest/'MIDTERM_PUSH_RECEIPT.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
