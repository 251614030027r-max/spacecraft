"""Summarise the preregistered 40k early read; not a formal method verdict."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(r'D:\py\DRL2')
OUTPUT = Path(r'C:\Users\35884\Documents\Spacecraft\V3D_40K_EARLY_READOUT_20260927.json')
EXPECTED = set(range(264100, 264112))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    path = ROOT / 'eval/v3d/early' / name
    payload = json.loads(path.read_text())
    rows = payload['records']
    assert len(rows) == 12 and {int(r['seed']) for r in rows} == EXPECTED
    assert payload['horizon'] == 35
    return path, payload, {int(r['seed']): r for r in rows}


def summary(rows):
    rr = list(rows.values())
    success = [r for r in rr if r['completed']]
    return {
        'episodes': len(rr),
        'completed': len(success),
        'zero_violation_completed': sum(bool(r['truth_geometry_zero_violation_completed']) for r in rr),
        'episodes_with_truth_violation': sum(bool(r['constraint_violated']) for r in rr),
        'completed_seeds': sorted(r['seed'] for r in success),
        'failed_seeds': sorted(r['seed'] for r in rr if not r['completed']),
        'timeout_seeds': sorted(r['seed'] for r in rr if r.get('time_failure')),
        'qp_infeasible_steps': sum(r.get('qp_infeasible_steps_total', 0) for r in rr),
        'zero_fallback_steps': sum(r.get('zero_fallback_steps_total', 0) for r in rr),
        'max_consecutive_zero_wrench_steps': max(r.get('max_consecutive_zero_wrench_steps', 0) for r in rr),
        'mean_success_time_s': mean(r['survival_s'] for r in success) if success else None,
        'mean_success_delta_v_m_s': mean(r['equivalent_delta_v_m_s'] for r in success) if success else None,
        'mean_all_episode_delta_v_m_s': mean(r['equivalent_delta_v_m_s'] for r in rr),
    }


base_path, _, base = load('pure_mpc_264100.json')
result = {
    'created_at_local': datetime.now().isoformat(timespec='seconds'),
    'purpose': '40k deterministic early read only; no stopping, checkpoint selection, tuning or paper claim',
    'seed_block': [264100, 264111],
    'checkpoint_steps': 40000,
    'evaluation_execution': 'one evaluator at a time, while three training runs continue; timing is not a formal realtime measurement',
    'training_commit': 'b05e389483cafab15c94c4852cecf08579f29111',
    'pure_mpc': {'path': str(base_path), 'sha256': sha(base_path), **summary(base)},
    'models': {},
}
for seed in (262420, 262421, 262422):
    path, payload, candidate = load(f'{seed}_40k.json')
    assert payload.get('arbiter', 'off') == 'off'
    quadrants = {k: [] for k in ('retained', 'rescued', 'destroyed', 'both_failed')}
    for s in sorted(EXPECTED):
        b, c = bool(base[s]['completed']), bool(candidate[s]['completed'])
        quadrant = 'retained' if b and c else 'rescued' if c else 'destroyed' if b else 'both_failed'
        quadrants[quadrant].append(s)
    common = quadrants['retained']
    model = ROOT / f'logs/v3d_{seed}/checkpoints/sac_mpc_40000_steps.zip'
    result['models'][str(seed)] = {
        'path': str(path), 'sha256': sha(path),
        'checkpoint': str(model), 'checkpoint_sha256': sha(model),
        **summary(candidate),
        'quadrants': {k: {'count': len(v), 'seeds': v} for k, v in quadrants.items()},
        'common_success_time_delta_s': mean(candidate[s]['survival_s'] - base[s]['survival_s'] for s in common) if common else None,
        'common_success_delta_v_delta_m_s': mean(candidate[s]['equivalent_delta_v_m_s'] - base[s]['equivalent_delta_v_m_s'] for s in common) if common else None,
    }
OUTPUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
