"""Independent read-only recalculation; no simulation or model selection."""
import csv
import hashlib
import json
import shutil
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path('C:/Users/35884/Documents/Spacecraft')
TOPIC = ROOT / '过程文件/停止头'
REPO = Path('D:/py/DRL2')
BUILD = TOPIC / '审查材料_20261006'
COMMIT = '2e5c236f7412caea3b203519619ef7a48bb16df9'
SEEDS = list(range(267000, 267048))
MODELS = ('262430', '262431', '262432')

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def copy(src, rel):
    dst = BUILD / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    assert sha(src) == sha(dst)
def load(d, row, model=None):
    assert {p.name for p in d.glob('seed_*.json')} == {f'seed_{s}.json' for s in SEEDS}
    assert not list(d.glob('*.lock')) and not list(d.glob('*.tmp'))
    result = {}
    for s in SEEDS:
        p = d / f'seed_{s}.json'
        r = read(p)
        assert r['seed'] == s and r['row'] == row
        assert r['code_commit'] == COMMIT and r['code_dirty'] is False
        assert r['max_decisions'] is None
        assert r['clean_completion'] == (r['completed'] and r['zero_violation'])
        if model:
            assert r['model_sha256'] == sha(REPO / f'logs/stop_{model}/final_model.zip')
        copy(p, 'raw/formal/' + p.relative_to(REPO / 'eval/stopping/formal').as_posix())
        result[s] = r
    return result

def equivalent(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a, float):
        return abs(a-b) <= 1e-12
    return a == b

BUILD.mkdir(parents=True, exist_ok=True)
pure = load(REPO / 'eval/stopping/formal/pure', 'pure')
official = read(REPO / 'eval/stopping/formal/readout.json')
assert official['problems'] == []
summary = {}
cases = []
for m in MODELS:
    stop = load(REPO / f'eval/stopping/formal/stopping/{m}', 'stopping', m)
    learned = load(REPO / f'eval/stopping/formal/learned/{m}', 'learned', m)
    clean = lambda r: [s for s in SEEDS if r[s]['clean_completion']]
    pset, lset, sset = set(clean(pure)), set(clean(learned)), set(clean(stop))
    rescued, destroyed = sorted(sset-pset), sorted(pset-sset)
    rescued_l, destroyed_l = sorted(sset-lset), sorted(lset-sset)
    violations = {name: sum(not r[s]['zero_violation'] for s in SEEDS)
                  for name, r in [('pure', pure), ('learned', learned), ('hybrid', stop)]}
    k0 = [s for s in SEEDS if stop[s]['handoff_k'] == 0]
    for s in k0:
        assert stop[s]['clean_completion'] == pure[s]['clean_completion']
        assert abs(stop[s]['survival_s'] - pure[s]['survival_s']) < 1e-8
    for s in SEEDS:
        r = stop[s]
        trace = r['trace']
        k = r['handoff_k']
        beta = trace['beta']
        assert len(beta) == len(trace['q_handoff']) == len(trace['q_continue'])
        if k is None:
            assert all(v < .5 for v in beta)
            assert len(beta) == r['decisions'] == r['learned_decisions']
        else:
            assert len(beta) == k+1 and r['learned_decisions'] == k
            assert all(v < .5 for v in beta[:-1]) and beta[-1] >= .5
    both = sorted(lset & sset)
    triple = sorted(pset & lset & sset)
    median = lambda values: st.median(values) if values else None
    excess = {row: {'time_s': median([r[s]['survival_s']-pure[s]['survival_s'] for s in triple]),
                    'delta_v_m_s': median([r[s]['equivalent_delta_v_m_s']-pure[s]['equivalent_delta_v_m_s'] for s in triple])}
              for row, r in [('learned', learned), ('hybrid', stop)]}
    metrics = dict(clean_completions=dict(pure=len(pset), learned=len(lset), hybrid=len(sset)),
                   violation_episodes=violations, rescued=rescued, destroyed=destroyed,
                   shared_clean_seeds=len(triple), median_excess_over_pure_on_shared=excess,
                   efficiency_acceptable=all(excess['hybrid'][key] <= max(0, .5*excess['learned'][key])
                                            for key in ('time_s', 'delta_v_m_s')),
                   handoff_k=sorted(stop[s]['handoff_k'] for s in SEEDS if stop[s]['handoff_k'] is not None),
                   never_handed_off=sum(stop[s]['handoff_k'] is None for s in SEEDS),
                   learned_share=st.mean(stop[s]['learned_decisions']/max(stop[s]['decisions'],1) for s in SEEDS),
                   passes=len(sset)>len(pset) and violations['hybrid']<=violations['pure'] and len(destroyed)<=2)
    assert equivalent(metrics, official['models'][m]), (m, metrics)
    coordination = dict(clean_completions=dict(learned_only=len(lset), stopping=len(sset)),
                        rescued_from_learned_only=rescued_l, destroyed_from_learned_only=destroyed_l,
                        shared_clean=len(both), median_time_change_s=median([stop[s]['survival_s']-learned[s]['survival_s'] for s in both]),
                        median_delta_v_change_m_s=median([stop[s]['equivalent_delta_v_m_s']-learned[s]['equivalent_delta_v_m_s'] for s in both]),
                        handoff_k=[stop[s]['handoff_k'] for s in SEEDS])
    # Floating-point mean implementation can differ at insignificant precision.
    assert equivalent(coordination, official['coordination_gain'][m])
    for s in destroyed:
        r = stop[s]
        cases.append(dict(model=m, seed=s, pure= {k:pure[s][k] for k in ('clean_completion','survival_s','failure')},
                          stopping={k:r[k] for k in ('clean_completion','survival_s','failure','qp_zero_fallbacks','handoff_k','learned_decisions')},
                          learned={k:learned[s][k] for k in ('clean_completion','survival_s','failure')},
                          trigger_beta=None if r['handoff_k'] is None else r['trace']['beta'][-1],
                          predicted_q_handoff=None if r['handoff_k'] is None else r['trace']['q_handoff'][-1],
                          predicted_q_continue=None if r['handoff_k'] is None else r['trace']['q_continue'][-1]))
    manifest_path = REPO / f'logs/stop_{m}/manifest.json'
    manifest = read(manifest_path)
    assert manifest['code_commit']==COMMIT and manifest['code_dirty'] is False
    assert manifest['status']=='completed' and manifest['actual_outer_decisions']==60000
    assert manifest['method']=='learned_stopping_option' and manifest['gradient_updates']==58000
    monitor = REPO / f'logs/stop_{m}/train.monitor.csv'
    with monitor.open(encoding='utf-8-sig', newline='') as f:
        f.readline()
        episodes=list(csv.DictReader(f))
    summary[m] = dict(metrics=metrics, coordination=coordination,
                      k0_handoffs=len(k0), failure_reasons=dict(Counter(v for s in SEEDS for v in stop[s]['failure'])),
                      qp_zero_fallbacks=sum(stop[s]['qp_zero_fallbacks'] for s in SEEDS),
                      model_sha256=sha(REPO / f'logs/stop_{m}/final_model.zip'),
                      training_episode_count=len(episodes), training_monitor_fields=list(episodes[0]) if episodes else [],
                      training_budget={key:manifest.get(key) for key in ('actual_outer_decisions','continue_transitions','handoff_episodes','suffix_simulated_decisions','gradient_updates')})
    copy(manifest_path, f'training/stop_{m}/manifest.json')
    copy(monitor, f'training/stop_{m}/train.monitor.csv')
assert not any(v['metrics']['passes'] for v in summary.values())
assert official['verdict']=='METHOD_DOES_NOT_HOLD'
write(BUILD / 'INDEPENDENT_AUDIT.json', dict(commit=COMMIT, verdict=official['verdict'],
      official_fields_recomputed_match=True, episodes=336, deployment_first_crossing_all_match=True,
      k0_pure_outcomes_and_times_match=True, models=summary))
write(BUILD / 'DESTROYED_PURE_CASES.json', cases)
for name in ('readout.json','devcheck.json','当前情况整理_20261006.md','STOPPING_DELIVERY_RECEIPT.json'):
    copy(ROOT / '上层交付/最新' / name, name)
copy(TOPIC / '记录/FORMAL_INDEPENDENT_VERIFICATION_20261006.json', 'ORIGINAL_ZIP_AUDIT.json')
for name in ('独立正式审查_20261006.py','核验正式交付_20261006.py'):
    copy(TOPIC / '脚本' / name, 'audit_tools/' + name)
for p in (REPO / 'eval/stopping/dev').rglob('*.json'):
    copy(p, 'raw/development/' + p.relative_to(REPO / 'eval/stopping/dev').as_posix())
hashes={p.relative_to(BUILD).as_posix():sha(p) for p in sorted(BUILD.rglob('*')) if p.is_file()}
write(BUILD / 'FILES_SHA256.json', hashes)
print(json.dumps(dict(verdict=official['verdict'], official_recalculation='match', destroyed_cases=len(cases),
                     files=len(hashes), cases=cases), ensure_ascii=False))
