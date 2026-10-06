import hashlib
import json
import zipfile
from pathlib import Path

root = Path('C:/Users/35884/Documents/Spacecraft')
latest = root / '上层交付/最新'
record = root / '过程文件/停止头/记录'
receipt = json.loads((latest / 'STOPPING_DELIVERY_RECEIPT.json').read_text(encoding='utf-8-sig'))
archive = latest / 'STOPPING_FORMAL_20261006.zip'
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

assert digest(archive) == receipt['zip_sha256']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names = z.namelist()
    assert len(names) == len(set(names))
    checked = 0
    for line in z.read('ALL_FILES_SHA256.txt').decode('utf-8-sig').splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
        checked += 1
    for name, path in [('formal/evaluation/readout.json', latest / 'readout.json'),
                       ('development/evaluation/devcheck.json', latest / 'devcheck.json')]:
        assert z.read(name) == path.read_bytes(), name
    formal_json = [n for n in names if n.startswith('formal/evaluation/') and '/seed_' in n and n.endswith('.json')]
    assert len(formal_json) == 336
    checkpoint_models = [n for n in names if '/checkpoints/' in n and n.endswith('.zip')]
    final_models = [n for n in names if n.endswith('/final_model.zip')]
    result = dict(status='verified', zip_sha256=receipt['zip_sha256'], crc_ok=True,
                  sha_members_checked=checked, members=len(names), formal_episodes=len(formal_json),
                  bundled_checkpoint_model_count=len(checkpoint_models), final_models_excluded=not final_models,
                  bundled_final_model_paths=final_models,
                  checkpoint_examples=checkpoint_models[:4])
record.joinpath('FORMAL_INDEPENDENT_VERIFICATION_20261006.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
