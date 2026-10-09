"""Create the requested upper-review snapshot after all three formal M6 jobs finish."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

BASE = Path(r'C:\Users\35884\Documents\Spacecraft')
ROOT = Path(r'D:\py\DRL2_v3e')
OUT = ROOT / 'eval' / 'v3e'
SEEDS = (262420, 262421, 262422)
DEST = BASE / 'V3E_60K_CALIBRATION_UPPER_FEEDBACK_20260929'
sys.path.insert(0, str(ROOT))
from experiments.v3e_early_readout import _row

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def fmt(value):
    return '无共同完成' if value is None else f'{value:.3f}'

def main():
    state = read(BASE / 'V3E_60K_EXECUTION_20260929.json')
    assert all(state['jobs'][f'm6_{s}']['status'] == 'completed' for s in SEEDS)
    DEST.mkdir(exist_ok=False)
    raw_dir = DEST / 'raw_json'
    raw_dir.mkdir()
    shutil.copy2(BASE / 'V3E_60K_EXECUTION_20260929.json', DEST / 'execution_snapshot.json')
    shutil.copy2(BASE / 'V3E_FINAL_TRAINING_AUDIT_20260929.json', DEST / 'final_training_audit_historical_snapshot.json')
    shutil.copy2(ROOT / 'docs/V3E_50K_RULING_AND_60K_ORDER_20260929.md', DEST / 'UPPER_RULING_AND_ORDER.md')
    shutil.copy2(BASE / Path(__file__).name, DEST / Path(__file__).name)
    pairs = read(BASE / 'V3E_50K_VS_60K_LEARNED_READOUT_20260929.json')['rows']
    shutil.copy2(BASE / 'V3E_50K_VS_60K_LEARNED_READOUT_20260929.json', DEST / '50k_vs_60k_paired.json')
    pure = read(OUT / 'pure_mpc.json')
    rows = [_row(pure, pure, 'Pure MPC（复用）'), _row(pure, read(OUT / 'nominal.json'), '平缓推进（复用）')]
    for name in ('pure_mpc.json', 'nominal.json'):
        shutil.copy2(OUT / name, raw_dir / name)
    model_summaries, locations = {}, []
    for seed in SEEDS:
        run = ROOT / 'logs' / f'v3e_{seed}'
        model_summaries[str(seed)] = {'m6': read(OUT / str(seed) / 'm6.json'),
                                    'm3': read(OUT / str(seed) / 'values/m3_report.json'),
                                    'training_health': state['models'][str(seed)]['health'],
                                    'm5_status': state['jobs'][f'm5_{seed}']['status']}
        for budget, name in ((50, 'learned_only_50k.json'), (60, 'learned_only.json')):
            path = OUT / str(seed) / name
            rows.append(_row(pure, read(path), f'{seed}@{budget}k learned-only'))
            shutil.copy2(path, raw_dir / f'{seed}_{name}')
        for stage in ('m2', 'm3', 'm6'):
            names = [f'm2_{seed}_{chunk}' for chunk in 'abcd'] if stage == 'm2' else [f'{stage}_{seed}']
            for job_name in names:
                job = state['jobs'][job_name]
                assert job['status'] == 'completed'
                for output in map(Path, job['outputs']):
                    checked = job['checks'][str(output)]
                    if output.suffix == '.json':
                        assert sha(output) == checked['sha256'], output
                        shutil.copy2(output, raw_dir / f'{seed}_{output.name}')
                    else:
                        locations.append({'path': str(output), 'bytes': output.stat().st_size,
                                          'sha256': checked['sha256'], 'hash_source': 'completed_job_validation'})
                for key in ('stdout', 'stderr'):
                    source = Path(job[key])
                    log_dir = DEST / 'command_logs'
                    log_dir.mkdir(exist_ok=True)
                    shutil.copy2(source, log_dir / source.name)
        locations.append({'path': str(run / 'final_model.zip'), 'bytes': (run / 'final_model.zip').stat().st_size,
                          'sha256': state['models'][str(seed)]['model_sha256'], 'hash_source': 'formal_pipeline_preflight'})
        shutil.copy2(run / 'manifest.json', raw_dir / f'{seed}_manifest.json')
    progress = {}
    for seed in SEEDS:
        job = state['jobs'][f'm5_{seed}']
        progress[str(seed)] = {'status': job['status'], 'episodes_finished': None}
        source = OUT / str(seed) / 'arbitrated.json'
        if not source.exists():
            source = Path(str(source) + '.partial')
        if source.exists():
            raw = source.read_bytes()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            progress[str(seed)]['episodes_finished'] = len(payload['records'])
            snapshot = DEST / f'{seed}_arbitration_progress_snapshot.json'
            snapshot.write_bytes(raw)
            progress[str(seed)]['snapshot_sha256'] = sha(snapshot)
    generated = datetime.now().astimezone().isoformat()
    result = {'generated_at': generated, 'execution_status': state['status'], 'errors': state['errors'],
              'evaluation_commit': state['evaluation_commit'], 'rows_vs_pure': rows,
              'pairs_50k_to_60k': pairs, 'models': model_summaries, 'arbitration_progress': progress,
              'boundary': 'All three formal M6 complete. M5 incomplete or skipped; no final arbitration efficacy verdict. This is the requested upper-review snapshot, not the final delivery.'}
    (DEST / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (DEST / 'large_artifact_locations.json').write_text(json.dumps(locations, ensure_ascii=False, indent=2), encoding='utf-8')
    report = ['# V3e 60k校准完成：上层阶段反馈', '', f'快照时间：{generated}（Asia/Tokyo）。', '',
              '## 1. 当前结论与执行状态', '',
              '**三个60k模型均已完成learned-only正式测试、完整M2数据采集、M3拟合及M6校准；M6为PASS / STOP / STOP。262420获准进入M5，262421与262422按上层执行单C节跳过M5。队列无程序故障，262420仲裁继续。** 本文是校准完成后的阶段交付，尚非最终仲裁结果。', '',
              '相较50k，三种子的固定48开局完成数由39/32/30变为42/33/38；完成数均提升，但收益、代价及价值排序质量有明显种子差异。不能把训练末段完成率、理论oracle上限或单个最好种子当成实际仲裁效果。', '',
              '## 2. 训练、版本与科学规则', '',
              f"三次训练均自然完成60000决策步，final_model ZIP内部num_timesteps=60000，训练提交a03632a且训练时tracked clean；末100回合完成率73%/81%/77%，参考跳变违规均0，训练QP门均通过。正式评估使用{state['evaluation_commit']}；本次更新只新增上层裁定与小型证据，未改变算法。原D:\\py\\DRL2仍保留V3d b05e389，V3e在D:\\py\\DRL2_v3e独立执行。", '',
              '正式策略测试固定262000–262047、确定性动作、horizon=35、task_state_v3、phase-time-observation、execution-feedback及adaptive-task；Pure MPC与nominal直接复用50k时的同48开局结果，无重跑。learned-only违规是要报告的结果，不设中途安全STOP；最终仲裁真值违规仍必须不超过Pure MPC的0。没有降低安全门、挑checkpoint、挑训练种子、改参数或复用dryrun价值。', '',
              '## 3. 同48开局：50k→60k实际策略变化', '',
              '|训练种子|完成50k→60k|救回50k失败/丢失50k成功|违规回合50k→60k|QP回退步50k→60k|共同成功Δt(s)|共同成功Δv(m/s)|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for p in pairs:
        report.append(f"|{p['seed']}|{p['completed_50k']}→{p['completed_60k']}|{p['rescued_old_failure']}/{p['lost_old_success']}|{p['truth_violation_episodes_50k']}→{p['truth_violation_episodes_60k']}|{p['qp_zero_fallback_steps_50k']}→{p['qp_zero_fallback_steps_60k']}|{fmt(p['common_success_60k_minus_50k_time_s'])}|{fmt(p['common_success_60k_minus_50k_dv_m_s'])}|")
    report += ['', '此表Δt/Δv为60k减50k，仅比较该训练种子两个预算均完成的开局；正值更慢或更耗燃料。262420完成、时间和燃料均改善，但QP回退增多；262421仅小幅增加完成数，燃料改善而时间增加；262422完成数明显增加且未丢失50k成功，但共同成功回合更慢、更耗燃料。整体不是50k后的统一平台期，也不能据一次48开局测试断言统计收敛。', '',
               '## 4. 与Pure MPC配对的完整策略表', '',
               '|行|完成/48|零违规完成|违规回合|QP回退步|救回Pure失败|丢失Pure成功|共同成功Δt(s)|共同成功Δv(m/s)|',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        report.append(f"|{r['label']}|{r['completed']}|{r['zero_violation_completed']}|{r['episodes_with_truth_violation']}|{r['qp_zero_fallback_steps']}|{r['rescued']}|{r['destroyed']}|{fmt(r['common_success_time_delta_s_mean'])}|{fmt(r['common_success_dv_delta_mean'])}|")
    report += ['', '本表Δt/Δv相对Pure MPC，与上一表的50k→60k比较对象不同。历史V3d→V3e参考点修复的50k消融仍按上层已审定结果保留：共同完成多耗燃料由约2.9–3.2降为0.8–1.4 m/s；它不等于60k仲裁已经有效。', '',
               '## 5. M2与M3：采集与价值拟合', '',
               '每模型完整48个价值采集开局270000–270047，分a/b/c/d四块，默认每开局2个probe。每开局Pure全程、learned全程及两个learned前缀→Pure续跑，共192条轨迹/模型，三个模型合计576条；不是144条普通策略评估。所有块已完成且NPZ数值有限、JSON数值有限。没有减半；270040–270047的8个预定留出开局完整保留，其余40个用于训练。', '',
               '拟合目标为未折扣task utility-to-go，而非shaped SAC回报；V_L、V_B分别5个bootstrap头，V_B按原规则屏蔽task_state和applied_direction。下表是独立留出集误差，不是M6通过门。', '',
               '|种子|M2观测总数|V_L留出状态数|V_L留出MAE|V_L留出R²|V_B留出状态数|V_B留出MAE|V_B留出R²|',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        m = model_summaries[str(seed)]['m3']
        l, b = m['V_L']['holdout'], m['V_B']['holdout']
        assert m['episodes'] == 192 and m['target'] == 'task' and m['holdout_seeds'] == '270040-270047'
        assert len(m['V_L']['heads']) == len(m['V_B']['heads']) == 5
        report.append(f"|{seed}|{m['states']}|{l['states']}|{l['mae']:.3f}|{l['r2']:.3f}|{b['states']}|{b['mae']:.3f}|{b['r2']:.3f}|")
    report += ['', '留出误差用于观察泛化质量，不能以训练误差替代。即使M6排序通过，也不表示绝对价值或不确定性估计已经全面准确；本轮不据误差临时换网络、改采样或重拟合。', '',
               '## 6. M6：正式校准完整判定', '',
               '独立校准开局262100–262111，检查点0/10/20、z=1；在相同决策状态对继续learned与切Pure分别实跑到终止，将真实任务效用差与价值预测差比较。门：决定性点少于5为INCONCLUSIVE；不少于5且符号一致率≥80%为PASS，否则STOP。决定性点与完成结果不同的关键点是两个不同集合。', '',
               '|种子|总点|决定性点|符号正确/决定性点|一致率|M6门|完成结果不同的关键点|关键点正确率|误选点|MAE_L / MAE_B|',
               '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for seed in SEEDS:
        s = model_summaries[str(seed)]['m6']['summary']
        n = s['decisive_checkpoints']
        correct = round(n * s['sign_agreement_decisive'])
        report.append(f"|{seed}|{s['checkpoints']}|{n}|{correct}/{n}|{s['sign_agreement_decisive']:.1%}|{s['gate']}|{s['critical_checkpoints']}|{s['critical_sign_accuracy']:.1%}|{s['wrong_picks']}|{s['mae_L']:.3f} / {s['mae_B']:.3f}|")
    report += ['', '**判读：**262420的排序一致率达到原门槛，允许继续仲裁。262421、262422均有足量有效检查点，仍明显低于80%；这是STOP，不是样本不足导致的INCONCLUSIVE。误选2/6是检查点数，不是正式评估违规回合数。现有证据表明这两个种子的价值排序未达到校准标准；尚不足以锁定单一原因，也不能把learned-only完成提升等同于仲裁价值可靠。逐检查点原始mu、sd、G与失败原因均在各m6.json。', '',
               '## 7. 仲裁进度、总体判定边界与后续', '',
               '|种子|M5状态|已写出回合数（快照）|', '|---|---|---:|']
    for seed in SEEDS:
        p = progress[str(seed)]
        report.append(f"|{seed}|{p['status']}|{p['episodes_finished'] if p['episodes_finished'] is not None else '未运行'}|")
    report += ['', '262420继续one_way、z=1、同48正式开局；262421/422按上层执行单C节不跑M5，未擅自绕过M6。没有程序错误或NaN导致全局停机。已写出的M5片段仅证明进度，不作完整安全、净收益或实时性结论；满固定前24开局才另交中途观察，完整48继续。', '',
               '**预注册总体条件要求至少两个模型均通过两层。当前只有一个模型获准进入M5，因此本轮按现行门规则无法满足“至少两个模型通过”的方法成立条件；这不预判262420单模型的仲裁效果。**其实际完成数、丢失Pure成功、真值违规、learned占比及救回净收益，仍等待48回合结束后的官方判读。最终L1保持完成数≥37、丢失≤2、违规≤0、平均learned占比≥5%；L2救回>丢失。', '',
               '执行端继续现有队列，最终两个官方判读与总包由后台接续。任何新规则、重拟合、追加样本或让STOP模型做探索性M5均需上层另行明确，不在本轮自行扩展。', '',
               '## 8. 证据、复现与交付范围', '',
               '本包包含：本报告、summary.json、原始Pure/nominal与50k/60k learned-only JSON、12块M2元数据、3份M3拟合报告、3份M6完整结果、对应M2/M3/M6命令stdout/stderr、训练manifest、执行状态快照、50k→60k配对明细、上层执行单及当前M5进度快照（若存在）。JSON_SHA256.txt与ALL_FILES_SHA256.txt覆盖包内证据。模型ZIP/M2 NPZ/价值PT只登记路径、大小和完成时SHA-256，不入包、不提交Git、不推送。', '',
               'final_training_audit_historical_snapshot.json是清晨训练终态审计：其中旧“evaluation_60k_started=false/安全待裁定”仅代表当时状态，已由上层本次放行和execution_snapshot.json取代；不再作为停止依据。所有并行耗时仅用于执行进度估计，不作正式串行实时性结论。完整最终总包仍待262420仲裁与官方判读完成，本文不冒充最终交付。']
    (DEST / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (DEST / 'JSON_SHA256.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(DEST).as_posix()}' for p in sorted(DEST.rglob('*.json'))) + '\n', encoding='utf-8')
    (DEST / 'ALL_FILES_SHA256.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(DEST).as_posix()}' for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'ALL_FILES_SHA256.txt') + '\n', encoding='utf-8')
    archive_path = DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for line in (DEST / 'ALL_FILES_SHA256.txt').read_text(encoding='utf-8').splitlines():
            digest, name = line.split('  ', 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    receipt = {'generated_at': generated, 'report': str(DEST / 'REPORT.md'), 'archive': str(archive_path),
               'archive_sha256': sha(archive_path), 'archive_bytes': archive_path.stat().st_size,
               'zip_crc_and_manifest_hashes_verified': True, 'stage': 'all_M6_complete_M5_pending',
               'models_m6': {str(s): model_summaries[str(s)]['m6']['summary'] for s in SEEDS},
               'arbitration_progress': progress}
    (BASE / 'V3E_60K_CALIBRATION_UPPER_FEEDBACK_RECEIPT_20260929.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == '__main__':
    main()
