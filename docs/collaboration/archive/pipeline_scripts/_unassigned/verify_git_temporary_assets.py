from pathlib import Path
import subprocess,json,hashlib
repo=Path('C:/Users/35884/Documents/Spacecraft/过程文件/协作/Git工作树');prefix='docs/collaboration/lower/handoffs/ppt_temporary_20261009'
manifest=json.loads((repo/prefix/'FILES_SHA256.json').read_text(encoding='utf-8'))
for name,digest in manifest.items():
    blob=subprocess.check_output(['git','-C',str(repo),'show',':'+prefix+'/'+name]);assert hashlib.sha256(blob).hexdigest()==digest,name
print('GIT_INDEX_BYTE_HASHES_VERIFIED',len(manifest))
