"""Package formal final V3e results or a program-fault handoff, without large binaries."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
from spacecraft_asset_layout import PUBLIC, SCRIPTS, RECORDS, STATE, public_bundle, publish

BASE = Path(r'C:\Users\35884\Documents\Spacecraft')
ROOT = Path(r'D:\py\DRL2_v3e')
OUT = ROOT / 'eval' / 'v3e'
SEEDS = (262420, 262421, 262422)
sys.path.insert(0, str(ROOT))
from eval.adaptive_pairing import compare_payloads
from experiments.v3e_early_readout import _row

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def artifact(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def fmt(value):
    return '无共同完成' if value is None else f'{value:.3f}'

state = read(STATE)
assert state['status'] in ('evaluations_completed', 'program_fault_stop'), 'Pipeline is still running'
bundle = PUBLIC / '待发布' / 'V3E_60K_VALUE_R1_SUPPLEMENT_20260929'
bundle.mkdir(parents=True, exist_ok=True)
shutil.copytree(public_bundle('V3E_OVERNIGHT_DELIVERY_20260929'), bundle / 'historical_50k_and_training', dirs_exist_ok=True)
shutil.copytree(public_bundle('V3E_60K_CALIBRATION_UPPER_FEEDBACK_20260929'), bundle / 'calibration_upper_feedback_snapshot', dirs_exist_ok=True)
shutil.copy2(STATE, bundle / 'execution_snapshot.json')
shutil.copy2(RECORDS / 'V3E_FINAL_TRAINING_AUDIT_20260929.json', bundle / 'final_training_audit.json')
shutil.copy2(ROOT / 'docs/V3E_50K_RULING_AND_60K_ORDER_20260929.md', bundle / 'UPPER_RULING_AND_ORDER.md')
reproduction = bundle / 'reproduction'
reproduction.mkdir(exist_ok=True)
for name in ('run_v3e_60k_pipeline_20260929.py', 'package_v3e_60k_pipeline_20260929.py', 'read_v3e_50k_vs_60k_20260929.py', 'read_v3e_m5_early24_20260929.py'):
    shutil.copy2(SCRIPTS / name, reproduction / name)
shutil.copy2(SCRIPTS / 'spacecraft_asset_layout.py', reproduction / 'spacecraft_asset_layout.py')
for name in ('V3E_50K_VS_60K_LEARNED_READOUT_20260929.json', 'V3E_50K_VS_60K_LEARNED_READOUT_20260929.md'):
    if (RECORDS / name).exists():
        shutil.copy2(RECORDS / name, bundle / name)
for path in RECORDS.glob('V3E_M5_EARLY24_20260929_*.*'):
    if path.suffix in ('.json', '.md'):
        dest = bundle / 'interim_24_episode_observations'
        dest.mkdir(exist_ok=True)
        shutil.copy2(path, dest / path.name)
logs = bundle / 'command_logs'
logs.mkdir(exist_ok=True)
for job in state['jobs'].values():
    for key in ('stdout', 'stderr'):
        path = Path(job[key])
        if path.exists():
            shutil.copy2(path, logs / path.name)
    for path in map(Path, job['outputs']):
        if path.exists() and path.suffix == '.json':
            target = bundle / 'formal_60k' / path.relative_to(OUT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        partial = Path(str(path) + '.partial')
        if partial.exists() and path.suffix == '.json':
            target = bundle / 'unfinished_program_fault' / partial.relative_to(OUT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(partial, target)
large = []
for seed in SEEDS:
    run = ROOT / 'logs' / f'v3e_{seed}'
    large.append(artifact(run / 'final_model.zip'))
    dest = bundle / 'final_training' / str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'train.monitor.csv'):
        shutil.copy2(run / name, dest / name)
    for path in sorted((OUT / str(seed)).glob('m2_*.npz')):
        large.append(artifact(path))
    for path in sorted((OUT / str(seed) / 'values').glob('*.pt')):
        large.append(artifact(path))
(bundle / 'large_artifact_locations.json').write_text(json.dumps(large, ensure_ascii=False, indent=2), encoding='utf-8')

pure = read(OUT / 'pure_mpc.json')
nominal = read(OUT / 'nominal.json')
rows = [_row(pure, pure, 'Pure MPC（复用）'), _row(pure, nominal, 'v3_nominal（复用）')]
budget_pairs = []
for seed in SEEDS:
    old = read(OUT / str(seed) / 'learned_only_50k.json')
    rows.append(_row(pure, old, f'V3e {seed} @50k learned-only'))
    new_path = OUT / str(seed) / 'learned_only.json'
    if state['jobs'].get(f'learned_{seed}', {}).get('status') == 'completed' and new_path.exists():
        new = read(new_path)
        rows.append(_row(pure, new, f'V3e {seed} @60k learned-only'))
        pair = compare_payloads(old, new)
        budget_pairs.append({'seed': seed, 'completed_50k': sum(r['completed'] for r in old['records']),
            'completed_60k': pair['candidate_completed'], 'rescued_old_failure': pair['quadrants']['rescued']['count'],
            'lost_old_success': pair['quadrants']['destroyed']['count'],
            'common_success_60k_minus_50k': pair['common_success_candidate_minus_baseline']})
    arb_path = OUT / str(seed) / 'arbitrated.json'
    if state['jobs'].get(f'm5_{seed}', {}).get('status') == 'completed' and arb_path.exists():
        rows.append(_row(pure, read(arb_path), f'V3e {seed} @60k arbitrated'))
readout_path = OUT / 'readout.json'
readout = read(readout_path) if state['jobs'].get('readout', {}).get('status') == 'completed' and readout_path.exists() else None
summaries = {str(seed): {'gate': state['models'][str(seed)].get('m6_gate'),
                         'summary': state['models'][str(seed)].get('m6_summary'),
                         'm5_status': state['jobs'].get(f'm5_{seed}', {}).get('status', 'not_started')} for seed in SEEDS}
combined = {'generated_at': datetime.now().astimezone().isoformat(), 'execution_status': state['status'],
            'evaluation_commit': state['evaluation_commit'], 'rows': rows, 'pairs_50k_to_60k': budget_pairs,
            'calibration_by_model': summaries, 'official_two_layer_readout': readout,
            'errors': state['errors'], 'interim_truth_violations_are_not_a_stop_gate': True,
            'official_label_note': 'v3e_early_readout prints @50k for every --v3e input; original JSON/logs retained and display labels here use actual input models',
            'timing_boundary': 'Parallel compute times are not formal serial real-time evidence'}
(bundle / 'combined_results.json').write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding='utf-8')
with (bundle / 'comparison_table.csv').open('w', newline='', encoding='utf-8-sig') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

report = ['# V3e 60k 第一轮价值补充结果', '', f"生成时间：{combined['generated_at']}。", '',
          '2026-09-29最新上层裁定要求统一补采V_L并用第二轮价值作为主结果。本文保留第一轮M5完整结果及第一轮判读，仅作补充；第二轮队列继续，不能把本包当第二轮最终交付。', '', '## 执行结论', '']
if state['status'] == 'program_fault_stop':
    report += ['本轮出现程序故障，已停止新任务并终止本队列仍在执行的子进程，保留已完成结果与未完成片段。未完成阶段不作为正式结果。', '',
               *[f"- {e['at']}: {e['error']}" for e in state['errors']]]
else:
    report += ['依据2026-09-29上层裁定完成本轮允许的流程。learned-only真值违规作为结果报告；安全标准仍在仲裁行按预注册规则判读，没有放宽。', '']
    if readout:
        report.append(f"官方两层总体判读：**{readout['verdict']}**。两层均通过 {readout['models_passing_both_layers']} 个，只有第一层通过 {readout['models_passing_layer1_only']} 个；分别保留三个训练种子的结果，未挑种子。")
    else:
        report.append('无获准完成的仲裁行，因此没有正式两层总体判定；不得补造仲裁结果。')
report += ['', 'Pure MPC与平缓推进复用此前48开局结果，未重跑。50k历史报告中的安全待裁定状态已被本次上层放行取代；历史材料保留在historical_50k_and_training供追溯，不代表本轮当前状态。', '',
           '## 配对结果', '',
           '所有正式评估使用262000–262047；共同成功Δt和Δv均相对Pure MPC，正值为更慢或更耗燃料。', '',
           '|行|完成/48|零违规完成|真值违规回合|QP回退步|救回Pure失败|丢失Pure成功|共同成功Δt(s)|共同成功Δv(m/s)|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    report.append(f"|{row['label']}|{row['completed']}|{row['zero_violation_completed']}|{row['episodes_with_truth_violation']}|{row['qp_zero_fallback_steps']}|{row['rescued']}|{row['destroyed']}|{fmt(row['common_success_time_delta_s_mean'])}|{fmt(row['common_success_dv_delta_mean'])}|")
report += ['', '## 50k→60k 同开局比较', '',
           '|种子|50k完成|60k完成|救回50k失败|丢失50k成功|共同成功Δt(s)|共同成功Δv(m/s)|',
           '|---|---:|---:|---:|---:|---:|---:|']
for pair in budget_pairs:
    delta = pair['common_success_60k_minus_50k']
    dt = (delta['survival_s'] or {}).get('mean')
    dv = (delta['equivalent_delta_v_m_s'] or {}).get('mean')
    report.append(f"|{pair['seed']}|{pair['completed_50k']}|{pair['completed_60k']}|{pair['rescued_old_failure']}|{pair['lost_old_success']}|{fmt(dt)}|{fmt(dv)}|")
report += ['', '此处比较来自同开局确定性评估，不能由训练回合统计替代；三种子分别判读，不能用选择最好种子的方式替代预注册总体条件。', '',
           '## M6校准与M5入口', '', '|种子|M6 gate|决定性点|一致率|M5状态|', '|---|---|---:|---:|---|']
for seed in SEEDS:
    record = summaries[str(seed)]
    summary = record['summary'] or {}
    report.append(f"|{seed}|{record['gate'] or '未完成'}|{summary.get('decisive_checkpoints', '未完成')}|{summary.get('sign_agreement_decisive', '未完成')}|{record['m5_status']}|")
report += ['', 'M6 PASS和INCONCLUSIVE均按裁定进入M5；INCONCLUSIVE不得称价值已经校准。M6 STOP只阻止对应模型M5，其他模型继续，依据上层新执行单C节。各模型完整summary保留在combined_results.json和原始m6.json，未用试运行价值替代正式拟合。', '',
           '## 预注册两层判定', '',
           '第一层：仲裁完成数不低于Pure MPC，丢失Pure成功不超过2，真值违规回合不超过Pure MPC的0，平均learned分支占比至少5%。第二层：救回数大于丢失数。至少两个模型均通过两层才判方法成立；否则按官方脚本报告保护基线但无净收益或不成立。learned-only不设通过条件。', '']
if readout:
    report += ['|种子|L1保护基线|L2净提升|平均learned占比|逐项L1条件|', '|---|---|---|---:|---|']
    for seed, model in readout['models'].items():
        judgement = model['judgement']
        checks = ', '.join(f'{k}={v}' for k, v in judgement['layer1_checks'].items())
        report.append(f"|{seed}|{judgement['layer1_baseline_protection']}|{judgement['layer2_net_improvement']}|{model['arbitrated']['mean_learned_branch_share']:.3f}|{checks}|")
report += ['', '## 两个官方判读脚本输出', '']
for name in ('readout', 'table_60k'):
    job = state['jobs'].get(name)
    report += [f'### {name}', '']
    if job and job['status'] == 'completed':
        report += ['```text', Path(job['stdout']).read_text(encoding='utf-8'), '```', '']
    else:
        report += [state.get('readout_not_run_reason', '尚未完成'), '']
report += ['## 复现与证据范围', '',
           f"训练提交a03632a，正式验证提交{state['evaluation_commit']}。本次拉取只增加裁定文档与证据JSON，训练/MPC/价值算法未改变；原V3d目录仍为b05e389。已有模型哈希与最终审计一致，三个训练运行均自然完成60000步且健康门通过。每个进程OMP/MKL/OPENBLAS线程数为1，最多6个计算进程按内存排队，没有更改科学参数。", '',
           '正式结果JSON、两个判读脚本原始输出、全部命令日志与执行快照保留在包内。M2 NPZ、价值PT、最终模型ZIP只列绝对路径、大小与SHA-256，未打入包或提交Git。historical_50k_and_training保留旧50k对照、试运行、训练终态、哈希和版本信息，试运行质量数字不用于方法判定。JSON_SHA256.txt与ALL_FILES_SHA256.txt用于核验本包；并行测量的计算耗时不用于正式实时性结论。']
(bundle / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
json_files = sorted(bundle.rglob('*.json'))
(bundle / 'JSON_SHA256.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(bundle).as_posix()}' for p in json_files) + '\n', encoding='utf-8')
files = sorted(p for p in bundle.rglob('*') if p.is_file() and p != bundle / 'ALL_FILES_SHA256.txt')
(bundle / 'ALL_FILES_SHA256.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(bundle).as_posix()}' for p in files) + '\n', encoding='utf-8')
archive_path = bundle.with_suffix('.zip')
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(bundle.rglob('*')):
        if path.is_file():
            archive.write(path, path.relative_to(bundle).as_posix())
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    for path in bundle.rglob('*'):
        if path.is_file():
            assert hashlib.sha256(archive.read(path.relative_to(bundle).as_posix())).hexdigest() == sha(path)
final_bundle = PUBLIC / '最新' / bundle.name
final_archive = PUBLIC / '最新' / archive_path.name
archive_info = artifact(archive_path)
archive_info['path'] = str(final_archive)
receipt = {'generated_at': datetime.now().astimezone().isoformat(), 'status': state['status'],
           'role': 'first-round value supplementary result; second-round main evaluation continues',
           'report': str(final_bundle / 'REPORT.md'), 'archive': archive_info,
           'json_count': len(json_files), 'zip_crc_and_contents_verified': True,
           'm6_gates': {str(seed): summaries[str(seed)]['gate'] for seed in SEEDS},
           'verdict': readout['verdict'] if readout else None}
receipt_path = PUBLIC / '待发布' / 'V3E_60K_DELIVERY_RECEIPT_20260929.json'
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
publish(bundle, archive_path, receipt_path)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
