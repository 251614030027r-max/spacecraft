"""Documentation-only correction: native metadata delivered; external live logs stay local."""
from pathlib import Path
import os,shutil
import runtime as a
R=a.R;r=a.read(R/'记录/INTERIM_EVAL_PUSH_RECEIPT.json');local=Path(r['local_report']).parent
snapshot=max((R/'发布快照').glob('evaluation_interim_*'),key=lambda p:p.name)
collab=a.COLLAB/'docs/collaboration/lower/handoffs/t2_7_guide_20261031/interim_eval_20261101'
old=(local/'RESULT_INTERIM_EVAL.md').read_text(encoding='utf-8');new=old.replace('与完整外部回合/滚动日志','；外部逐回合/30秒滚动日志和原始TensorBoard事件留在本机，不在本即时推送载荷中')
assert new!=old
(R/'记录/INTERIM_REPORT_BEFORE_SCOPE_CLARIFICATION.md').write_text(old,encoding='utf-8')
for root in (snapshot,local,collab):
 (root/'RESULT_INTERIM_EVAL.md').write_text(new,encoding='utf-8')
 lines=(root/'SHA256SUMS').read_text(encoding='utf-8').splitlines();lines=[a.sha(root/'RESULT_INTERIM_EVAL.md')+'  RESULT_INTERIM_EVAL.md' if line.endswith('  RESULT_INTERIM_EVAL.md') else line for line in lines];(root/'SHA256SUMS').write_text('\n'.join(lines)+'\n',encoding='utf-8')
parent=a.git('ls-remote','origin','refs/heads/'+a.SCIENCE).split()[0]
# Existing publication parent is available locally; a new document-only child preserves it.
env=os.environ.copy();env['GIT_INDEX_FILE']=str(R/'记录/interim_doc_scope.index');a.git('read-tree',parent,env=env)
for n in ('RESULT_INTERIM_EVAL.md','SHA256SUMS'):
 blob=a.git('hash-object','-w',snapshot/n);a.git('update-index','--add','--cacheinfo','100644,'+blob+',evidence/T2_7_GUIDE/interim_20261101/'+n,env=env)
tree=a.git('write-tree',env=env);science=a.git('commit-tree',tree,'-p',parent,'-m','docs: clarify native metadata versus local external telemetry in interim delivery');a.git('push','origin',science+':refs/heads/'+a.SCIENCE)
cur=a.read(a.CURRENT);cur['interim_science_evidence_commit']=science;a.dump(a.CURRENT,cur)
a.dump(collab/'SCIENCE_POINTER.json',{'commit':science,'path':'evidence/T2_7_GUIDE/interim_20261101/','raw_cases_in_science_branch':True})
a.git('add','--',str(collab.relative_to(a.COLLAB)),'docs/collaboration/CURRENT.json',repo=a.COLLAB);a.git('commit','-m','docs: clarify interim delivered logs scope; evaluation continues',repo=a.COLLAB);commit=a.git('rev-parse','HEAD',repo=a.COLLAB);a.git('push','origin','HEAD:refs/heads/collab/spacecraft',repo=a.COLLAB)
assert a.git('ls-remote','origin','refs/heads/'+a.SCIENCE).split()[0]==science and a.git('ls-remote','origin','refs/heads/collab/spacecraft',repo=a.COLLAB).split()[0]==commit
r.update(science_commit=science,collab_commit=commit,report_SHA256=a.sha(local/'RESULT_INTERIM_EVAL.md'),documentation_scope_clarified=True);a.dump(R/'记录/INTERIM_EVAL_PUSH_RECEIPT.json',r);shutil.copy2(R/'记录/INTERIM_EVAL_PUSH_RECEIPT.json',local/'PUSH_RECEIPT.json');print(r)
