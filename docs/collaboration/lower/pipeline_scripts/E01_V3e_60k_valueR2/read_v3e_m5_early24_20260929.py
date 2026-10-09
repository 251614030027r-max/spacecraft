"""Read existing M5 records once per model; never simulate or change the formal queue."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
from spacecraft_asset_layout import RECORDS, STATE

BASE = Path(r'C:\Users\35884\Documents\Spacecraft')
ROOT = Path(r'D:\py\DRL2_v3e')
OUT = ROOT / 'eval' / 'v3e'
FIRST24 = list(range(262000, 262024))
sys.path.insert(0, str(ROOT))
from experiments.v3e_early_readout import _row

def load(path):
    raw = path.read_bytes()
    return json.loads(raw), {'path': str(path), 'sha256_at_read': hashlib.sha256(raw).hexdigest()}

def subset(payload):
    indexed = {int(r['seed']): r for r in payload['records']}
    if len(indexed) != len(payload['records']):
        raise ValueError('Duplicate episode seeds')
    if not all(s in indexed for s in FIRST24):
        return None
    return {'records': [indexed[s] for s in FIRST24]}

def fmt(value):
    return '无共同完成' if value is None else f'{value:.3f}'

def main():
    state, _ = load(STATE)
    created = []
    for seed in (262420, 262421, 262422):
        dest = RECORDS / f'V3E_M5_EARLY24_20260929_{seed}.json'
        md = dest.with_suffix('.md')
        job = state['jobs'].get(f'm5_{seed}', {})
        if dest.exists() or job.get('status') not in ('running', 'completed'):
            continue
        final = OUT / str(seed) / 'arbitrated.json'
        source = final if final.exists() else Path(str(final) + '.partial')
        if not source.exists():
            continue
        try:
            payload, arb_artifact = load(source)
        except (json.JSONDecodeError, FileNotFoundError):
            # The evaluator rewrites the partial at each episode boundary.
            continue
        arb = subset(payload)
        if arb is None:
            continue
        pure_full, pure_artifact = load(OUT / 'pure_mpc.json')
        learned_full, learned_artifact = load(OUT / str(seed) / 'learned_only.json')
        pure, learned = subset(pure_full), subset(learned_full)
        assert pure is not None and learned is not None
        rows = [_row(pure, pure, 'Pure MPC'), _row(pure, learned, f'{seed} 60k learned-only'),
                _row(pure, arb, f'{seed} 60k arbitrated')]
        records = arb['records']
        result = {'generated_at': datetime.now().astimezone().isoformat(), 'model_seed': seed,
                  'episode_seeds': FIRST24, 'rows': rows,
                  'mean_learned_branch_share': mean(float(r['learned_branch_share']) for r in records),
                  'handback_episodes': sum(r['handback_decision'] is not None for r in records),
                  'm6_gate': state['models'][str(seed)].get('m6_gate'),
                  'arbitrated_records_sha256': hashlib.sha256(json.dumps(records, sort_keys=True).encode('utf-8')).hexdigest(),
                  'input_artifacts_at_read': [pure_artifact, learned_artifact, arb_artifact],
                  'boundary': 'Fixed first 24 episodes observed from the existing 48-episode M5 run. Not a formal two-layer verdict; no selection, stopping or tuning. Full evaluation continues. Partial source hash is a snapshot hash.'}
        report = ['# V3e 60k 仲裁前24开局中途观察', '',
                  f"生成时间：{result['generated_at']}；训练种子 {seed}；M6 {result['m6_gate']}。", '',
                  '固定开局262000–262023，读取既有评估结果，不额外仿真。此表仅供中途观察，完整48开局继续；不作正式两层判定，不用于停实验、挑种子或改参数。', '',
                  '|行|完成/24|真值违规回合|QP回退步|救回Pure失败|丢失Pure成功|共同成功Δt(s)|共同成功Δv(m/s)|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            report.append(f"|{r['label']}|{r['completed']}|{r['episodes_with_truth_violation']}|{r['qp_zero_fallback_steps']}|{r['rescued']}|{r['destroyed']}|{fmt(r['common_success_time_delta_s_mean'])}|{fmt(r['common_success_dv_delta_mean'])}|")
        report += ['', f"仲裁平均learned分支占比 {result['mean_learned_branch_share']:.3%}；交还Pure MPC的回合数 {result['handback_episodes']}/24。Δt、Δv相对Pure，正值为更慢或更耗燃料。输入路径、读取时哈希及固定24回合记录哈希见同名JSON。"]
        if result['m6_gate'] == 'INCONCLUSIVE':
            report += ['', 'M6不确定，不得声称价值已校准。']
        md.write_text('\n'.join(report) + '\n', encoding='utf-8')
        with dest.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        created.append({'seed': seed, 'report': str(md), 'rows': rows})
    print(json.dumps({'created': created}, ensure_ascii=False))

if __name__ == '__main__':
    main()
