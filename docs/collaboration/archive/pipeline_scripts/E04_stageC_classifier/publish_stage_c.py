"""Publish the authorized small review + verified ZIP; credentials never logged."""
import base64, hashlib, json, os, subprocess, urllib.error, urllib.request, zipfile
from pathlib import Path

ROOT=Path('C:/Users/35884/Documents/Spacecraft'); REC=ROOT/'过程文件/阶段C/记录'
LATEST=ROOT/'上层交付/最新'; receipt=json.loads((LATEST/'STAGE_C_DELIVERY_RECEIPT.json').read_text())
ZIP=LATEST/'STAGE_C_20261002.zip'; sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(ZIP)!=receipt['zip_sha256']: raise RuntimeError('ZIP SHA mismatch')
with zipfile.ZipFile(ZIP) as z:
    if z.testzip(): raise RuntimeError('ZIP CRC mismatch')
    listing=json.loads(z.read('FILES_SHA256.json'))
    if set(z.namelist())!=set(listing)|{'FILES_SHA256.json'}: raise RuntimeError('ZIP members mismatch')
    for name,h in listing.items():
        if hashlib.sha256(z.read(name)).hexdigest()!=h: raise RuntimeError('Member SHA mismatch: '+name)
print('INDEPENDENT_ZIP_VERIFICATION_OK',flush=True)
env=os.environ.copy(); env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
r=subprocess.run(['git','-C','D:/py/DRL2','credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,env=env,timeout=30)
credentials=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
token=credentials.get('password')
if r.returncode or not token: raise RuntimeError('No usable existing GitHub credential; verified local delivery retained')
BASE='https://api.github.com/repos/251614030027r-max/spacecraft'; COMMIT=receipt['code_commit']
def api(method,path,payload=None,raw=None,ctype='application/json'):
    data=raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    request=urllib.request.Request(path if path.startswith('https://') else BASE+path,data=data,method=method,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'Spacecraft-Stage-C-Delivery','Content-Type':ctype})
    try:
        with urllib.request.urlopen(request,timeout=120) as response: return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code==404 and method=='GET': return None
        raise RuntimeError(f'GitHub {method} failed HTTP {e.code}; local evidence preserved') from None
branch='review/v3e-stage-c-20261002'; ref=api('GET','/git/ref/heads/'+branch)
if ref is None: api('POST','/git/refs',{'ref':'refs/heads/'+branch,'sha':COMMIT})
prefix='docs/reviews/v3e_stage_c_20261002/'
files={name:(LATEST/name).read_bytes() for name in ['REPORT.md','c2_report.json','STAGE_C_DELIVERY_RECEIPT.json']}
files['ALL_FILES_SHA256.txt']=(ROOT/'过程文件/阶段C/交付展开/ALL_FILES_SHA256.txt').read_bytes()
for name,data in files.items():
    path=prefix+name; existing=api('GET','/contents/'+path+'?ref='+branch)
    if existing:
        if existing.get('encoding')!='base64' or base64.b64decode(existing['content'])!=data: raise RuntimeError('Existing remote review differs; not overwriting '+name)
    else: api('PUT','/contents/'+path,{'message':'docs: Stage C frozen C2 stop result and audit evidence','content':base64.b64encode(data).decode(),'branch':branch})
tag='v3e-stage-c-20261002'; release=api('GET','/releases/tags/'+tag)
if release is None:
    release=api('POST','/releases',{'tag_name':tag,'target_commitish':COMMIT,'name':'Stage C: C2_STOP (2026-10-02)','body':'Frozen Stage C stopped at C2: no threshold reached 95% out-of-fold trajectory-weighted precision. C3 and SAC retraining were not run. Full independent evidence ZIP; SHA256 '+receipt['zip_sha256']+'. Small review: '+branch+'.','draft':False,'prerelease':False})
assets=api('GET',f'/releases/{release["id"]}/assets')
asset=next((x for x in assets if x['name']==ZIP.name),None)
if asset:
    if asset['size']!=ZIP.stat().st_size: raise RuntimeError('Existing release asset differs; not replacing')
else:
    asset=api('POST',release['upload_url'].split('{')[0]+'?name='+ZIP.name,raw=ZIP.read_bytes(),ctype='application/zip')
if asset['size']!=ZIP.stat().st_size: raise RuntimeError('Remote asset size differs')
digest=asset.get('digest')
if digest and digest!='sha256:'+receipt['zip_sha256']: raise RuntimeError('Remote asset digest differs')
result={'status':'published','review_url':'https://github.com/251614030027r-max/spacecraft/tree/'+branch+'/'+prefix.rstrip('/'),'release_url':release['html_url'],'download_url':asset['browser_download_url'],'zip_sha256':receipt['zip_sha256'],'github_asset_digest':digest,'asset_bytes':asset['size']}
(REC/'REMOTE_DELIVERY_RECEIPT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
