"""Append completed TD3 evidence using separate worktrees; never touches CURRENT or models in Git."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/workspace/td3_cloud_runs/TD3_C_7e55a65d_20261118')
SCIENCE = Path('/workspace/spacecraft-science-evidence-td3')
COLLAB = Path('/workspace/spacecraft-collab-td3')
SCIENCE_BRANCH = 'claude/sac-mpc-coupling-design-ns7g6i'
COLLAB_BRANCH = 'collab/spacecraft'
REL = Path('evidence/FORMAL/TD3_C_CLOUD_20261118_7e55a65d')
HANDOFF = Path('docs/collaboration/lower/handoffs/formal/TD3_C_CLOUD_20261118_7e55a65d')
ENV = dict(os.environ, GIT_TERMINAL_PROMPT='0')
STATE = Path('/workspace/td3_cloud_publication.json')


def git(repo, *args):
    return subprocess.check_output(['git', *args],cwd=repo,env=ENV,stderr=subprocess.STDOUT,text=True)


def publish(repo, branch, directory, title):
    # A full fetch retains ancestry needed for normal fast-forward/rebase. No
    # source checkout used by the learner is ever changed.
    git(repo,'fetch','--filter=blob:none','origin',branch)
    if git(repo,'rev-parse','HEAD').strip() != git(repo,'rev-parse',f'origin/{branch}').strip():
        # At the start of a stage the worktree has only new untracked paths.
        git(repo,'merge','--ff-only',f'origin/{branch}')
    paths=[p for p in sorted(directory.rglob('*')) if p.is_file()]
    checksum=directory/'SHA256SUMS'
    checksum.write_text(''.join(hashlib.file_digest(p.open('rb'),'sha256').hexdigest()+'  '+p.relative_to(directory).as_posix()+'\n'
                                for p in paths if p != checksum))
    for p in sorted(directory.rglob('*')):
        if p.is_file():
            git(repo,'add','-f','--',str(p.relative_to(repo)))
    staged=git(repo,'diff','--cached','--name-only').splitlines()
    prefix=str(directory.relative_to(repo))+'/'
    assert staged and all(x.startswith(prefix) and 'CURRENT' not in x for x in staged),staged
    git(repo,'commit','-m',title,'-m','Collaboration: user-authorized cloud TD3 execution, 2026-11-18')
    for attempt in range(3):
        try:
            result=git(repo,'push','origin',f'HEAD:refs/heads/{branch}')
            sha=git(repo,'rev-parse','HEAD').strip()
            print(json.dumps({'event':'published','branch':branch,'commit':sha,'title':title}),flush=True)
            STATE.write_text(json.dumps({'last_published_branch':branch,'commit':sha,'title':title},indent=2))
            return
        except subprocess.CalledProcessError:
            if attempt==2:
                raise
            git(repo,'fetch','--filter=blob:none','origin',branch)
            git(repo,'rebase',f'origin/{branch}')


def copies(destination, pairs):
    destination.mkdir(parents=True,exist_ok=False)
    for src, name in pairs:
        target=destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target)


def main():
    initial=SCIENCE/REL/'preflight'
    copies(initial,[(ROOT/x,x) for x in ('PREFLIGHT.json','pytest.log','requirements-resolved.txt')]
           +[(COLLAB/HANDOFF/'STARTED.md','STARTED.md')]
           +[(Path('/workspace/td3-cloud-ops')/x, 'ops/'+x) for x in ('sitecustomize.py','supervisor.py','verify.py','publisher.py')])
    publish(SCIENCE,SCIENCE_BRANCH,initial,'evidence: append frozen cloud TD3_C preflight and actual start')
    seen=set()
    while True:
        for seed in (20,21,22):
            run=ROOT/'B'/f'TRAIN_TD3_C_s{seed}'
            tag=f'train_s{seed}'
            if (ROOT/f'VERIFIED_TRAIN_s{seed}.json').exists() and tag not in seen:
                files=['manifest.json','episodes.json','model_stats.json','finished.json','heartbeat.json','progress.json','progress.jsonl','episodes.live.jsonl','SHA256SUMS']
                pairs=[(run/x,x) for x in files]
                pairs += [(ROOT/f'VERIFIED_TRAIN_s{seed}.json','VERIFIED.json'),(ROOT/'logs'/f'TRAIN_TD3_C_s{seed}.log','train.log')]
                dest=SCIENCE/REL/tag;copies(dest,pairs)
                publish(SCIENCE,SCIENCE_BRANCH,dest,f'evidence: verified cloud TD3_C seed {seed} at 5000 decisions')
                seen.add(tag)
            tag=f'eval_s{seed}'
            if (ROOT/f'VERIFIED_s{seed}.json').exists() and tag not in seen:
                # Both source files are closed before the verifier publishes its marker.
                dest=SCIENCE/REL/tag
                copies(dest,[(ROOT/f'VERIFIED_s{seed}.json','VERIFIED.json'),(ROOT/f'TD3_C_s{seed}_EPISODES.csv','EPISODES.csv')])
                publish(SCIENCE,SCIENCE_BRANCH,dest,f'evidence: verified TD3_C seed {seed} urgent and five validation checkpoints')
                handoff=COLLAB/HANDOFF/tag
                copies(handoff,[(ROOT/f'VERIFIED_s{seed}.json','VERIFIED.json')])
                (handoff/'RESULT.md').write_text(f'# TD3_C 云端种子 {seed} 阶段交付\n\n5000 决策训练与注册的 2 个采集 worker 已核验；urgent 300000–300095 共 96 回合，验证 295200–295223 的 1000/2000/3000/4000/5000 检查点各 24 回合。无 error；numerical_failure 按正式未捕获记录保留。科学证据路径：{REL/tag}。本阶段不宣称完整多方法矩阵 complete。\n')
                publish(COLLAB,COLLAB_BRANCH,handoff,f'docs: deliver verified cloud TD3_C seed {seed}')
                seen.add(tag)
        status=json.loads((ROOT/'STATUS.json').read_text())
        if status['status']=='blocked':
            dest=COLLAB/HANDOFF/'blocked'
            copies(dest,[(ROOT/'STATUS.json','STATUS.json'),(ROOT/'SCHEDULE.jsonl','SCHEDULE.jsonl')])
            publish(COLLAB,COLLAB_BRANCH,dest,'docs: record cloud TD3 interruption with preserved evidence')
            return
        if status['status']=='complete_authorized_TD3_scope':
            # The supervisor finalizes hashes and the archive after writing STATUS.
            archive=Path(str(ROOT)+'_DELIVERY.tar.gz.sha256')
            if archive.exists() and len(seen)==6:
                dest=SCIENCE/REL/'complete'
                copies(dest,[(ROOT/'STATUS.json','STATUS.json'),(ROOT/'SHA256SUMS','RAW_SHA256SUMS'),(archive,'DELIVERY.tar.gz.sha256')])
                publish(SCIENCE,SCIENCE_BRANCH,dest,'evidence: complete authorized cloud TD3_C three-seed delivery')
                dest=COLLAB/HANDOFF/'complete'
                copies(dest,[(ROOT/'STATUS.json','STATUS.json'),(archive,'DELIVERY.tar.gz.sha256')])
                (dest/'RESULT.md').write_text('# TD3_C 云端授权范围全部完成\n\n种子 20、21、22 各 5000 决策，288 个 urgent 回合与 360 个检查点验证回合均已通过阶段核验。各阶段 manifest、日志、回合表和 SHA256 位于科学分支 evidence/FORMAL/TD3_C_CLOUD_20261118_7e55a65d。模型和原始回合留在独立云端目录及 DELIVERY.tar.gz；CURRENT 未改。此次仅完成 TD3 授权范围，不是全量 RUN_FORMAL。\n')
                publish(COLLAB,COLLAB_BRANCH,dest,'docs: deliver completed authorized cloud TD3_C experiment')
                return
        time.sleep(30)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        STATE.write_text(json.dumps({'status':'publication_blocked','error':repr(error),'traceback':traceback.format_exc()},indent=2))
        raise
