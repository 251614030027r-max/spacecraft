from pathlib import Path
import json,subprocess,hashlib
repo=Path('C:/Users/35884/Documents/Spacecraft/过程文件/协作/Git工作树');prefix='docs/collaboration/lower/handoffs/figures_interim_20261009';manifest=json.loads((repo/prefix/'FILES_SHA256.json').read_text(encoding='utf-8'))
for name,digest in manifest.items():
    blob=subprocess.check_output(['git','-C',str(repo),'show',':'+prefix+'/'+name]);assert hashlib.sha256(blob).hexdigest()==digest,name
names=subprocess.check_output(['git','-C',str(repo),'diff','--cached','--name-only','-z']).decode().split('\0');assert all(not n or n.startswith('docs/collaboration/') for n in names)
print('GIT_BYTES_VERIFIED',len(manifest),'ALL_CHANGES_COLLAB_ONLY')
