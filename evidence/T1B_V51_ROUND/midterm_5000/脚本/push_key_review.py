"""Publish only an explicit small allowlist of research reports; no models, Replay or raw logs."""
import os, json, hashlib, datetime
from pathlib import Path
from push_snapshot import git, sha, dump, ROOT, REPO, COLLAB, FROZEN, BRANCH, SPACE

label='key_review_20261020_1530'
build=ROOT/'发布快照'/label
assets=build/'assets'
assert not build.exists(), 'Immutable review already exists: inspect receipt'
assets.mkdir(parents=True)
names=['RESULT.md','SAME_TASK_EVAL.csv','POLICY_AND_GATE_AUDIT.json','MPC_COUPLING_AUDIT.json',
       'MPC_FULL_HORIZON_AUDIT.json','MODEL_SHA256.json','TRAINING_STATUS.md','THREE_SEED_COMPARISON.json']
for n in names:
    (assets/n).write_bytes((SPACE/'上层交付/最新'/n).read_bytes())
for n in ['INTERIM_DIAGNOSIS_1515.md','EXISTING_SIGNAL_DIAGNOSIS.json']:
    (assets/n).write_bytes((ROOT/'记录'/n).read_bytes())
state=json.loads((ROOT/'STATE.json').read_text(encoding='utf-8'))
dump(assets/'LIVE_PROGRESS.json',{k:state.get(k) for k in ['scientific_commit','updated_jst','phase','last_steps','completed_evaluations','active_runs','run_scope']})
(assets/'README.md').write_text('# V5.1 用户要求的即时中期审查\n\n先读INTERIM_DIAGNOSIS_1515.md。冻结b6f5c8a，seed0双臂仍训练；RESULT.md等阶段报告的生成时间在各文件内，LIVE_PROGRESS.json为本次现场进度。TCPF5k评估尚未齐，不将其列为完整配对。已有E-SAC2/5/10k与TCPF2k均无超越V4的收益。诊断是有限证据的工作判断，不是确定bug归因。\n\n本包仅含精简报告、逐开局指标、审计统计、模型哈希和诊断；不含模型、Replay、专家文件或原始轨迹日志。后续5k完整配对另按既定阶段交付。\n',encoding='utf-8')
dump(assets/'MANIFEST.json',{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in assets.iterdir()})
assert sum(p.stat().st_size for p in assets.iterdir())<500_000
receipt={}
for repo, branch, prefix, key in [(REPO,BRANCH,'evidence/T1B_V51_ROUND/'+label,'science'),
        (COLLAB,'collab/spacecraft','docs/collaboration/lower/handoffs/t1b_v51_round_20261020/'+label,'collab')]:
    assert git(repo,'remote','get-url','origin')=='https://github.com/251614030027r-max/spacecraft.git'
    assert not git(repo,'status','--porcelain')
    head=git(repo,'rev-parse','HEAD'); index=Path(git(repo,'rev-parse','--absolute-git-dir'))/'index'; before=sha(index)
    if repo==REPO: assert head==FROZEN
    git(repo,'fetch','origin',branch); parent=git(repo,'rev-parse','origin/'+branch)
    env=os.environ.copy(); env['GIT_INDEX_FILE']=str(build/(key+'_index'))
    git(repo,'read-tree',parent,env=env)
    for p in sorted(assets.iterdir()):
        oid=git(repo,'hash-object','-w','--stdin',data=p.read_bytes())
        git(repo,'update-index','--add','--cacheinfo','100644',oid,prefix+'/'+p.name,env=env)
    tree=git(repo,'write-tree',env=env)
    commit=git(repo,'commit-tree',tree,'-p',parent,data=b'evidence: V5.1 requested interim key reports; incomplete 5k clearly marked\n')
    dump(build/(key+'_COMMIT.json'),{'commit':commit,'parent':parent})
    git(repo,'push','origin',commit+':refs/heads/'+branch)
    assert git(repo,'ls-remote','origin','refs/heads/'+branch).split()[0]==commit
    for p in assets.iterdir():
        assert hashlib.sha256(git(repo,'cat-file','blob',commit+':'+prefix+'/'+p.name,raw=True)).hexdigest()==sha(p)
    assert git(repo,'rev-parse','HEAD')==head and sha(index)==before
    receipt[key]={'commit':commit,'url':'https://github.com/251614030027r-max/spacecraft/tree/'+commit+'/'+prefix,'files':len(list(assets.iterdir())),'blob_sha256_verified':True,'head_index_unchanged':True}
    dump(build/(key+'_PUSH_RECEIPT.json'),receipt[key])
dump(build/'PUSH_RECEIPT.json',receipt)
dump(ROOT/'记录/PUSH_KEY_REVIEW.json',receipt)
dump(SPACE/'上层交付/最新/PUSH_KEY_REVIEW.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
