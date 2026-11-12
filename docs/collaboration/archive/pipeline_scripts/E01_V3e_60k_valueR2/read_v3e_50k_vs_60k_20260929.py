"""Read completed paired 50k/60k results while the formal value pipeline continues."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from spacecraft_asset_layout import RECORDS, STATE

BASE = Path(r'C:\Users\35884\Documents\Spacecraft')
ROOT = Path(r'D:\py\DRL2_v3e')
OUT = ROOT / 'eval' / 'v3e'
sys.path.insert(0, str(ROOT))
from eval.adaptive_pairing import compare_payloads

state = json.loads(STATE.read_text(encoding='utf-8'))
assert all(state['jobs'][f'learned_{s}']['status'] == 'completed' for s in (262420, 262421, 262422)), '60k evaluation is unfinished'
rows = []
for seed in (262420, 262421, 262422):
    paths = [OUT / str(seed) / 'learned_only_50k.json', OUT / str(seed) / 'learned_only.json']
    old, new = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    pair = compare_payloads(old, new)
    deltas = pair['common_success_candidate_minus_baseline']
    row = {'seed': seed, 'completed_50k': sum(r['completed'] for r in old['records']),
           'completed_60k': sum(r['completed'] for r in new['records']),
           'truth_violation_episodes_50k': sum(r['constraint_violated'] for r in old['records']),
           'truth_violation_episodes_60k': sum(r['constraint_violated'] for r in new['records']),
           'qp_zero_fallback_steps_50k': sum(r['zero_fallback_steps_total'] for r in old['records']),
           'qp_zero_fallback_steps_60k': sum(r['zero_fallback_steps_total'] for r in new['records']),
           'rescued_old_failure': pair['quadrants']['rescued']['count'],
           'lost_old_success': pair['quadrants']['destroyed']['count'],
           'common_success_60k_minus_50k_time_s': (deltas['survival_s'] or {}).get('mean'),
           'common_success_60k_minus_50k_dv_m_s': (deltas['equivalent_delta_v_m_s'] or {}).get('mean'),
           'paired_detail': pair,
           'input_artifacts': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    rows.append(row)
result = {'generated_at': datetime.now().astimezone().isoformat(), 'rows': rows,
          'boundary': 'Same 48-seed deterministic learned-only comparisons; value and arbitration phases continue. This is not an arbitration efficacy verdict or a statistical convergence claim.'}
json_path = RECORDS / 'V3E_50K_VS_60K_LEARNED_READOUT_20260929.json'
md_path = RECORDS / 'V3E_50K_VS_60K_LEARNED_READOUT_20260929.md'
json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
report = ['# V3e 50k→60k 固定开局比较', '', f"生成时间：{result['generated_at']}。", '',
          '三个种子全部采用262000–262047的48个相同开局，确定性动作。下表来自实际策略评估；正式价值/仲裁流水线继续运行，尚不据此判断完整方法有效性。', '',
          '|种子|50k完成|60k完成|救回旧失败|丢失旧成功|违规回合50k→60k|共同成功Δt(s)|共同成功Δv(m/s)|',
          '|---|---:|---:|---:|---:|---|---:|---:|']
def fmt(v):
    return '无共同完成' if v is None else f'{v:.3f}'
for r in rows:
    report.append(f"|{r['seed']}|{r['completed_50k']}/48|{r['completed_60k']}/48|{r['rescued_old_failure']}|{r['lost_old_success']}|{r['truth_violation_episodes_50k']}→{r['truth_violation_episodes_60k']}|{fmt(r['common_success_60k_minus_50k_time_s'])}|{fmt(r['common_success_60k_minus_50k_dv_m_s'])}|")
report += ['', 'Δt和Δv仅在该种子50k与60k均成功的开局计算，正值表示60k更慢或更耗燃料。完整配对分布、QP回退步数与输入SHA-256见同名JSON。完成数相同不意味着策略未变；不以单个模型或单个指标声称统计收敛，也不据此挑种子、停实验或改参数。']
md_path.write_text('\n'.join(report) + '\n', encoding='utf-8')
print(json.dumps({'report': str(md_path), 'json': str(json_path), 'rows': [{k:v for k,v in r.items() if k not in ('paired_detail', 'input_artifacts')} for r in rows]}, ensure_ascii=False, indent=2))
