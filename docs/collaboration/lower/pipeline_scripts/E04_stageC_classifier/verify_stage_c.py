import hashlib,json,zipfile
from pathlib import Path
base=Path('C:/Users/35884/Documents/Spacecraft')
latest=base/'上层交付/最新'
r=json.loads((latest/'STAGE_C_DELIVERY_RECEIPT.json').read_text())
p=latest/'STAGE_C_20261002.zip'
assert hashlib.sha256(p.read_bytes()).hexdigest()==r['zip_sha256']
with zipfile.ZipFile(p) as z:
    assert z.testzip() is None
    hashes=json.loads(z.read('FILES_SHA256.json'))
    assert set(z.namelist())==set(hashes)|{'FILES_SHA256.json'}
    for name,h in hashes.items(): assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    for name in ('REPORT.md','results/c2/c2_report.json'):
        current=latest/('c2_report.json' if name.endswith('c2_report.json') else name)
        assert hashlib.sha256(current.read_bytes()).hexdigest()==hashes[name]
data=Path('D:/py/DRL2/eval/v3e/stage_c')
for m in ('262420','262421','262422'):
    meta=json.loads((data/f'c1_{m}.json').read_text())
    assert hashlib.sha256((data/f'c1_{m}.npz').read_bytes()).hexdigest()==meta['npz_sha256']
c2=json.loads((latest/'c2_report.json').read_text())
assert c2['c2_verdict']=='C_STOP' and c2['tau'] is None
assert not (data/'c3').exists() and not (data/'c2/handoff_classifier.pt').exists()
result={'status':'verified','zip_sha256':r['zip_sha256'],'crc':True,'all_members_sha':True,'c1_npz_sha':True,'c2_verdict':'C_STOP','c3_not_started':True,'classifier_not_created':True,'member_count':len(hashes)+1}
(base/'过程文件/阶段C/记录/FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
