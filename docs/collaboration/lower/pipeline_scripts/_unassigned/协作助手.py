"""Synchronize only the collaboration checkout; never touches experiment Git."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path('C:/Users/35884/Documents/Spacecraft')
CHECKOUT = ROOT / '过程文件/协作/Git工作树'
BRANCH = 'collab/spacecraft'
REMOTE = 'https://github.com/251614030027r-max/spacecraft.git'
URL = 'https://github.com/251614030027r-max/spacecraft/tree/collab/spacecraft/docs/collaboration'

def git(*args):
    r = subprocess.run(['git','-c',f'safe.directory={CHECKOUT}','-C',str(CHECKOUT),*args],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    return r.stdout.strip()

def validate():
    if git('branch','--show-current') != BRANCH:
        raise RuntimeError('Unexpected checkout branch; do not switch experiment Git.')
    if git('remote','get-url','origin') != REMOTE:
        raise RuntimeError('Unexpected GitHub destination.')

def status():
    validate()
    print('COLLAB_BRANCH:', BRANCH)
    print('LOCAL_HEAD:', git('rev-parse','HEAD'))
    print('WORKTREE:', 'dirty' if git('status','--porcelain') else 'clean')
    print('URL:', URL)
    path=CHECKOUT/'docs/collaboration/CURRENT.json'
    if path.exists():
        state=json.loads(path.read_text(encoding='utf-8-sig'))
        for key in ('phase','official_verdict','scientific_commit','upper_review_commit'):
            print(key + ':', state.get(key))

def sync():
    validate()
    if git('status','--porcelain'):
        raise RuntimeError('Local changes exist; publish or resolve them first. No reset or overwrite.')
    print(git('pull','--ff-only','origin',BRANCH))
    status()

def publish(message):
    validate()
    changed=git('diff','--name-only','HEAD').splitlines()
    untracked=git('ls-files','--others','--exclude-standard').splitlines()
    paths=changed+untracked
    if not paths:
        print('NOTHING_TO_PUBLISH')
        return
    for path in paths:
        if not path.startswith('docs/collaboration/'):
            raise RuntimeError('Out-of-scope change: '+path)
        p=CHECKOUT/path
        if p.exists() and (p.suffix.lower() not in {'.md','.json','.txt','.yaml','.yml'} or p.stat().st_size>5_000_000):
            raise RuntimeError('Non-document or oversized collaboration file: '+path)
    git('fetch','origin',BRANCH)
    if git('rev-parse','HEAD') != git('rev-parse',f'origin/{BRANCH}'):
        raise RuntimeError('Remote changed; synchronize/reconcile before publishing. Never force-push.')
    git('add','--','docs/collaboration')
    print(git('commit','-m',message))
    print(git('push','origin',f'HEAD:{BRANCH}'))
    print('REMOTE_HEAD:', git('ls-remote','origin',f'refs/heads/{BRANCH}'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('status','sync','publish'))
    parser.add_argument('--message',default='Update Spacecraft collaboration status and handoff')
    args=parser.parse_args()
    {'status':status,'sync':sync,'publish':lambda:publish(args.message)}[args.mode]()

if __name__ == '__main__':
    main()
