"""Preserve original bytes through Git's Windows EOL filters and verify staged blobs."""
import json,hashlib,shutil,subprocess
from pathlib import Path
WORK=Path(__file__).resolve().parents[1];REPO=Path('D:/py/DRL2');DEST=REPO/'evidence/T1B_V1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-c','safe.directory=D:/py/DRL2','-C',str(REPO),*args])
assert git('rev-parse','HEAD').decode().strip()=='636b65517bf73985b4e544c67decf857b4fffc19'
manifest=json.loads((DEST/'MANIFEST.json').read_text(encoding='utf-8'))
for a in manifest['copied_original_files']:
    if a['path']=='drivers/seal_review_assets.py':continue
    assert sha(DEST/a['path'])==a['sha256']==sha(a['source']),a['path']
(DEST/'.gitattributes').write_text('* -text\n',encoding='ascii')
src=Path(__file__);dst=DEST/'drivers/seal_review_assets.py';shutil.copy2(src,dst)
manifest['copied_original_files']=[a for a in manifest['copied_original_files'] if a['path']!='drivers/seal_review_assets.py']
manifest['copied_original_files'].append({'path':dst.relative_to(DEST).as_posix(),'source':str(src),'bytes':dst.stat().st_size,'sha256':sha(dst),'role':'Second assembly step: package-local EOL preservation and staged-byte verification'})
if '.gitattributes' not in manifest['derived_files']:manifest['derived_files'].append('.gitattributes')
manifest['git_byte_preservation']='Package-local * -text; no core/global settings changed. All staged blobSHA256 verified against original copied bytes.'
(DEST/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
if 'Git byte integrity:' not in (DEST/'README.md').read_text(encoding='utf-8'):
    with (DEST/'README.md').open('a',encoding='utf-8') as f:f.write('\nGit byte integrity: package-local `.gitattributes` disables EOL conversion for these review assets only. Each staged Git blob is verified with SHA256 against the supplied original-byte manifest; no global Git settings changed.\n')
files=sorted(p for p in DEST.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
(DEST/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(DEST).as_posix()}\n' for p in files),encoding='utf-8')
files=sorted(p for p in DEST.rglob('*') if p.is_file())
for p in files:
    relative=p.relative_to(REPO).as_posix()
    subprocess.run(['git','-c','safe.directory=D:/py/DRL2','-C',str(REPO),'add','-f','--',relative],check=True)
    subprocess.run(['git','-c','safe.directory=D:/py/DRL2','-C',str(REPO),'add','-f','--renormalize','--',relative],check=True)
    blob=git('cat-file','blob',':'+relative)
    assert hashlib.sha256(blob).hexdigest()==sha(p),relative
staged=[p for p in git('diff','--cached','--name-only','-z').decode('utf-8').split('\0') if p]
assert len(staged)==len(files) and all(p.startswith('evidence/T1B_V1/') for p in staged)
models=json.loads((DEST/'models/MODEL_SHA256.json').read_text(encoding='utf-8'))['models'];assert len(models)==8
receipt={'files':len(files),'total_bytes':sum(p.stat().st_size for p in files),'models':models,'all_staged_blobs_match_original_bytes':True,'individual_git_add_f':True,'no_science_code_change':True,'file_hashes':{p.relative_to(DEST).as_posix():sha(p) for p in files}}
(WORK/'记录/SEALED_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'files':len(files),'bytes':receipt['total_bytes'],'models':8,'all_staged_SHA256_verified':True}))
