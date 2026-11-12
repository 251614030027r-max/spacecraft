"""Read-only launch verification; no policy rollout."""
import ast
import json
from pathlib import Path
import sys

PROCESS = Path(r'C:\Users\35884\Documents\Spacecraft\过程文件\V3e价值第二轮')
for path in (PROCESS / '脚本').glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'))
sys.path.insert(0, str(PROCESS / '脚本'))
import run_v3e_value_r2_20260929 as runner
sys.path.insert(0, str(runner.ROOT))
from experiments.v3_common import parse_seed_range

prior = runner.first_state()
assert prior['jobs']['m5_262420']['status'] in ('running','completed')
assert not runner.STATE.exists() and not runner.LOCK.exists()
commit = runner.git(runner.ROOT, 'rev-parse', 'HEAD')
assert commit == 'c84ed7435f790198630ee7011fc7fd74b7307b48'
assert not runner.git(runner.ROOT, 'status','--porcelain','--untracked-files=no')
doc = runner.ROOT / 'docs/V3E_60K_CALIBRATION_RULING_20260929.md'
supplied = PROCESS / '接续' / doc.name
assert doc.read_text(encoding='utf-8') == supplied.read_text(encoding='utf-8-sig')
holdout = parse_seed_range(runner.HOLDOUT)
assert len(holdout) == len(set(holdout)) == 40
learned_seed_pool = set(range(270000,270048)) | set(range(271000,271192))
assert len(learned_seed_pool - set(holdout)) == 200
assert not learned_seed_pool.intersection(range(262000,262048))
assert not learned_seed_pool.intersection(range(262100,262124))
for seed in runner.SEEDS:
    assert runner.sha(runner.ROOT / f'logs/v3e_{seed}/final_model.zip') == prior['models'][str(seed)]['model_sha256']
    for chunk in 'abcd':
        for output in map(Path, prior['jobs'][f'm2_{seed}_{chunk}']['outputs']):
            assert runner.sha(output) == prior['jobs'][f'm2_{seed}_{chunk}']['checks'][str(output)]['sha256']
        assert not (runner.OUT / str(seed) / f'm2x_{chunk}.npz').exists()
        assert not (runner.OUT / str(seed) / f'm2x_{chunk}.json').exists()
    for name in ('m6_r2.json','arbitrated_r2.json'):
        assert not (runner.OUT / str(seed) / name).exists()
result = {'commit':commit,'policy_and_original_data_hashes_verified':True,
          'supplement_per_model':192,'independent_V_L_train_seeds':200,'independent_V_L_holdout_seeds':40,
          'calibration_seeds':'262112-262123','first_round_M5_preserved':True,
          'total_compute_limit_including_first_round':6,'targeted_tests':'10 passed',
          'diagnostic_hypothesis':'Additional independent outcomes may improve V_L; not a proven unique cause.'}
(PROCESS / '记录/PREFLIGHT_R2_20260929.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
