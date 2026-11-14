"""Evidence transport/reporting only. No simulation or scientific code changes."""
from pathlib import Path
import json, os, time, subprocess, shutil, math
import runtime as a
R=a.R; HANDOFF=a.COLLAB/'docs/collaboration/lower/handoffs/t2_7_guide_20261031'
def collab_commit(files,message,receipt):
 try:
  assert not a.git('diff','--cached','--name-only',repo=a.COLLAB),'unrelated staged files'
  a.git('fetch','origin','collab/spacecraft',repo=a.COLLAB)
  a.git('merge','--ff-only','origin/collab/spacecraft',repo=a.COLLAB)
  rel=[str(Path(p).relative_to(a.COLLAB)).replace('\\','/') for p in files]
  a.git('add','--',*rel,repo=a.COLLAB)
  if a.git('diff','--cached','--name-only',repo=a.COLLAB):a.git('commit','-m',message,repo=a.COLLAB)
  head=a.git('rev-parse','HEAD',repo=a.COLLAB)
  a.git('push','origin','HEAD:refs/heads/collab/spacecraft',repo=a.COLLAB)
  remote=a.git('ls-remote','origin','refs/heads/collab/spacecraft',repo=a.COLLAB).split()[0];assert remote==head
  out={'jst':a.now(),'status':'pushed_verified','commit':head}
 except Exception as e:out={'jst':a.now(),'status':'publication_pending','error':str(e)}
 a.dump(R/'记录'/receipt,out);return out
def science_commit(assets):
 try:
  a.git('fetch','origin',a.SCIENCE);parent=a.git('rev-parse','origin/'+a.SCIENCE)
  assert a.git('merge-base',a.AUDIT,parent)==a.AUDIT,'remote science history diverged from audit source'
  index=R/'记录/science_publication.index';assert not index.exists();env=os.environ.copy();env['GIT_INDEX_FILE']=str(index)
  a.git('read-tree',parent,env=env)
  paths=[p for p in sorted(assets.rglob('*')) if p.is_file()]
  assert all('\n' not in str(p) and '"' not in str(p) for p in paths)
  cmd=['git','-c','safe.directory='+str(a.REPO),'-C',str(a.REPO)]
  hashed=subprocess.run([*cmd,'hash-object','-w','--stdin-paths'],input=('\n'.join(p.as_posix() for p in paths)+'\n').encode('utf-8'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
  assert hashed.returncode==0,hashed.stderr.decode('utf-8','replace')
  hashes=hashed.stdout.decode().splitlines();assert len(hashes)==len(paths)
  records=b''.join(('100644 '+h+'\tevidence/T2_7_GUIDE/'+p.relative_to(assets).as_posix()+'\0').encode('utf-8') for p,h in zip(paths,hashes))
  staged=subprocess.run([*cmd,'update-index','-z','--index-info'],input=records,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
  assert staged.returncode==0,staged.stderr.decode('utf-8','replace')
  tree=a.git('write-tree',env=env)
  commit=a.git('commit-tree',tree,'-p',parent,'-m','evidence: T2.7 guided 2x2 training and registered verdict')
  a.git('push','origin',commit+':refs/heads/'+a.SCIENCE)
  assert a.git('ls-remote','origin','refs/heads/'+a.SCIENCE).split()[0]==commit
  out={'status':'pushed_verified','commit':commit,'parent':parent,'jst':a.now(),'science_checkout_unchanged':a.git('rev-parse','HEAD')}
 except Exception as e:out={'status':'publication_pending','jst':a.now(),'error':str(e)}
 a.dump(R/'记录/SCIENCE_PUSH_RECEIPT.json',out);return out
