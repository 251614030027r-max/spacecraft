"""Publish the main round-two result or a program-fault evidence handoff."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

BASE = Path(r'C:\Users\35884\Documents\Spacecraft')
ROOT = Path(r'D:\py\DRL2_v3e')
PROCESS = BASE / '过程文件/V3e价值第二轮'
RECORDS = PROCESS / '记录'
OUT = ROOT / 'eval/v3e'
SEEDS = (262420,262421,262422)
sys.path.insert(0,str(BASE / '过程文件/V3e60k/脚本'))
from spacecraft_asset_layout import PUBLIC, public_bundle, publish
sys.path.insert(0,str(ROOT))
from experiments.v3e_early_readout import _row

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def fmt(value):
    return '未完成/无共同成功' if value is None else f'{value:.3f}'

def main():
    state = read(RECORDS/'V3E_VALUE_R2_EXECUTION_20260929.json')
    assert state['status'] in ('evaluations_completed','program_fault_stop')
    name = 'V3E_60K_VALUE_R2_MAIN_DELIVERY_20260929'
    if state['status']=='program_fault_stop':
        name = 'V3E_VALUE_R2_PROGRAM_FAULT_20260929'
    bundle = PUBLIC / '待发布' / name
    bundle.mkdir(parents=True,exist_ok=True)
    first_path = BASE/'V3E_60K_EXECUTION_20260929.json'
    if not first_path.exists():
        first_path = BASE/'过程文件/V3e60k/运行归档'/first_path.name
    first = read(first_path)
    shutil.copy2(first_path,bundle/'first_round_execution_snapshot.json')
    shutil.copy2(RECORDS/'V3E_VALUE_R2_EXECUTION_20260929.json',bundle/'round_two_execution_snapshot.json')
    source = PROCESS/'接续/V3E_60K_CALIBRATION_RULING_20260929.md'
    shutil.copy2(source,bundle/'UPPER_ROUND_TWO_RULING.md')
    try:
        prior_bundle = public_bundle('V3E_60K_VALUE_R1_SUPPLEMENT_20260929')
    except FileNotFoundError:
        prior_bundle = public_bundle('V3E_60K_CALIBRATION_UPPER_FEEDBACK_20260929')
    shutil.copytree(prior_bundle,bundle/'first_round_supplement',dirs_exist_ok=True)
    raw = bundle/'round_two_raw_json'; raw.mkdir(exist_ok=True)
    logs = bundle/'round_two_logs'; logs.mkdir(exist_ok=True)
    large = []
    for artifact in state.get('reused_original_inputs',[]):
        if Path(artifact['path']).suffix=='.npz':
            large.append({**artifact,'bytes':Path(artifact['path']).stat().st_size,'role':'reused_original_M2'})
    for job_name,job in state['jobs'].items():
        for key in ('stdout','stderr'):
            path = Path(job[key])
            if path.exists():
                shutil.copy2(path,logs/path.name)
        for path in map(Path,job['outputs']):
            if job['status']=='completed' and path.exists():
                digest = sha(path)
                assert digest==job['checks'][str(path)]['sha256'],path
                if path.suffix=='.json':
                    target = raw/path.relative_to(OUT); target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,target)
                else:
                    large.append({'path':str(path),'bytes':path.stat().st_size,'sha256':digest,'role':job['stage']})
            partial = Path(str(path)+'.partial')
            if partial.exists():
                target = bundle/'unfinished_snapshots'/partial.relative_to(OUT)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(partial,target)
    for seed in SEEDS:
        model = ROOT/f'logs/v3e_{seed}/final_model.zip'
        digest = sha(model)
        assert digest==first['models'][str(seed)]['model_sha256']
        large.append({'path':str(model),'bytes':model.stat().st_size,'sha256':digest,'role':'frozen_60k_policy'})
    (bundle/'large_artifact_locations.json').write_text(json.dumps(large,ensure_ascii=False,indent=2),encoding='utf-8')
    pure = read(OUT/'pure_mpc.json')
    rows = [_row(pure,pure,'Pure MPC（复用）'),_row(pure,read(OUT/'nominal.json'),'nominal（复用）')]
    calibration,fit_comparisons = {},{}
    for seed in SEEDS:
        rows.append(_row(pure,read(OUT/str(seed)/'learned_only.json'),f'{seed} 60k learned-only（复用）'))
        if first['jobs'][f'm5_{seed}']['status']=='completed':
            rows.append(_row(pure,read(OUT/str(seed)/'arbitrated.json'),f'{seed} 第一轮价值仲裁（补充）'))
        if state['jobs'].get(f'm5r2_{seed}',{}).get('status')=='completed':
            rows.append(_row(pure,read(OUT/str(seed)/'arbitrated_r2.json'),f'{seed} 第二轮价值仲裁（主结果）'))
        old_m3 = read(OUT/str(seed)/'values/m3_report.json')
        new_path = OUT/str(seed)/'values_r2/m3_report.json'
        new_m3 = read(new_path) if state['jobs'].get(f'm3r2_{seed}',{}).get('status')=='completed' else None
        fit_comparisons[str(seed)] = {'first_round':old_m3,'second_round':new_m3}
        calibration[str(seed)] = {'first_round':first['models'][str(seed)].get('m6_summary'),
                                 'second_round':state['models'][str(seed)].get('m6_summary'),
                                 'm5_r2_status':state['jobs'].get(f'm5r2_{seed}',{}).get('status','not_started')}
    readout_path = OUT/'readout_r2.json'
    readout = read(readout_path) if state['jobs'].get('readout_r2',{}).get('status')=='completed' else None
    generated = datetime.now().astimezone().isoformat()
    combined = {'generated_at':generated,'execution_status':state['status'],'role':'second-round main result',
        'evaluation_commit':state.get('evaluation_commit'),'rows_vs_pure':rows,'fit_comparisons':fit_comparisons,
        'calibration_by_seed':calibration,'official_readout_r2':readout,'errors':state['errors'],
        'diagnostic_boundary':'Data insufficiency was the preregistered hypothesis, not a proven unique cause. R1/R2 calibration blocks differ; agreement-rate differences are not same-start paired gains.'}
    (bundle/'combined_results.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    report = ['# V3e 60k 价值第二轮主结果交付','',f'生成时间：{generated}。','','## 执行结论','']
    if state['status']=='program_fault_stop':
        report += ['第二轮出现程序故障，已保存完成结果和未完成片段；未完成阶段不作正式结果。没有自行改科学参数或重跑昂贵任务。',
                   *[f"- {e['at']}: {e['error']}" for e in state['errors']],'']
    elif readout:
        report += [f"官方第二轮总体判读：**{readout['verdict']}**；两层均通过{readout['models_passing_both_layers']}个，只有第一层通过{readout['models_passing_layer1_only']}个。三个训练种子分别保留，STOP模型按上层规则注明，未挑种子。",'']
    else:
        report += [state.get('readout_not_run_reason','无完整获准的第二轮仲裁结果'),'不能声称第二轮方法成立。','']
    report += ['主结果统一使用values_r2；262420第一轮M5按裁定完整跑完并作为补充保留，不能拿第一轮通过者替换第二轮主结果。Pure MPC、nominal与60k learned-only全部复用，没有重训SAC或重复基线。','','## 冻结设计与输入边界','',
        f"训练提交a03632a，第二轮执行提交{state.get('evaluation_commit')}；Git更新仅增加--learned-only采集选项、测试和裁定文档，训练/MPC/拟合/仲裁算法不变。三个模型最终ZIP的SHA与第一轮一致。预检10项通过，证据在reproduction中。",'',
        '每模型新增271000–271191共192个独立learned_full回合；旧270000–270047全量M2保留。V_L总240个独立开局，训练200、留出40（270040–270047和271160–271191）。原数据192条加新数据192条，共384条轨迹/模型。V_B数据选择的训练与留出状态数核对保持原值。每值函数仍5个bootstrap头、同网络、同fit seed与超参数、task utility目标及V_B掩码，未设置备用方法。','',
        '第二轮独立校准块262112–262123、检查点0/10/20、z=1；旧校准块仅保留追溯。决定性点少于5为INCONCLUSIVE，不少于5且一致率≥80%为PASS，否则STOP；PASS/INCONCLUSIVE进入M5，STOP只跳过对应模型。正式M5固定262000–262047、48回合、one_way、z=1，安全标准仍为真值违规≤Pure的0。','','## V_L/V_B留出误差：第一轮与第二轮','',
        '|种子|轮次|V_L留出状态|V_L MAE|V_L R²|V_B留出状态|V_B MAE|V_B R²|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for label,key in (('R1','first_round'),('R2','second_round')):
            item=fit_comparisons[str(seed)][key]
            if item is None:
                report.append(f'|{seed}|{label}|未完成||||||'); continue
            l,b=item['V_L']['holdout'],item['V_B']['holdout']
            report.append(f"|{seed}|{label}|{l['states']}|{fmt(l['mae'])}|{fmt(l['r2'])}|{b['states']}|{fmt(b['mae'])}|{fmt(b['r2'])}|")
    report += ['','留出集扩大，R1与R2汇总误差不是同一组样本的配对比较；原8开局保留，新32开局没有参与拟合。留出MAE/R²用于报告，不临时替代校准门。','','## 校准与仲裁入口','',
        '|种子|轮次|决定性点|一致率|误选检查点|M6门|第二轮M5状态|','|---|---|---:|---:|---:|---|---|']
    for seed in SEEDS:
        item=calibration[str(seed)]
        for label,key in (('R1','first_round'),('R2','second_round')):
            s=item[key]
            if s is None:
                report.append(f"|{seed}|{label}|未完成||||{item['m5_r2_status']}|"); continue
            report.append(f"|{seed}|{label}|{s['decisive_checkpoints']}|{s['sign_agreement_decisive']:.1%}|{s['wrong_picks']}|{s['gate']}|{item['m5_r2_status'] if label=='R2' else '补充'}|")
    report += ['','INCONCLUSIVE不称已校准，STOP不称样本不足。“误选”是检查点数，不是正式违规回合数。两轮校准开局不同，不能把一致率差直接当同开局提升。数据不足是本轮预先登记的诊断假设；结果是否支持及后续价值方法是否需要改变，应由上层据证据裁定，本轮未预设替代方法。','','## 相同48开局的完整策略表','',
        '|行|完成/48|零违规完成|违规回合|QP回退步|救回Pure失败|丢失Pure成功|共同成功Δt(s)|共同成功Δv(m/s)|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        report.append(f"|{r['label']}|{r['completed']}|{r['zero_violation_completed']}|{r['episodes_with_truth_violation']}|{r['qp_zero_fallback_steps']}|{r['rescued']}|{r['destroyed']}|{fmt(r['common_success_time_delta_s_mean'])}|{fmt(r['common_success_dv_delta_mean'])}|")
    report += ['','Δt/Δv只在该行与Pure共同完成的开局计算，正值更慢/更耗燃料。第一轮与第二轮仲裁明确分开，未池化或用最好的轮次替代统一R2主结果。','','## 第二轮预注册两层判定','',
        'L1：完成≥Pure37/48、丢失Pure成功≤2、真值违规≤0、平均learned占比≥5%。L2：救回>丢失。至少两个模型均通过两层才判方法成立；STOP导致缺少仲裁结果的种子不能补造通过。']
    if readout:
        report += ['','|种子|L1|L2|平均learned占比|逐项L1|','|---|---|---|---:|---|']
        for seed,model in readout['models'].items():
            j=model['judgement']; checks=', '.join(f'{k}={v}' for k,v in j['layer1_checks'].items())
            report.append(f"|{seed}|{j['layer1_baseline_protection']}|{j['layer2_net_improvement']}|{model['arbitrated']['mean_learned_branch_share']:.3f}|{checks}|")
        report += ['','官方判读原始输出：','','```text',Path(state['jobs']['readout_r2']['stdout']).read_text(encoding='utf-8'),'```']
    report += ['','## 最终工程使用入口','',
        f"后续PyCharm工程和运行工作目录统一使用 {ROOT}（独立Git worktree）；本次执行版本为 {state.get('evaluation_commit')}，SAC训练版本为 {state.get('training_commit')}。解释器为 D:/py/DRL2/.venv/Scripts/python.exe；原DRL2保存V3d历史对照。固定本次提交复现，不自动切回旧工程或替换为未验证的远端版本。",
        '三个策略模型使用 logs/v3e_<seed>/final_model.zip；第二轮主价值文件使用 eval/v3e/<seed>/values_r2/values_L.pt 和 values_B.pt，第一轮values/只作补充。逐种子校准门和两层判定见上表；STOP或未完成模型不称已验证通过。模型和价值文件的绝对路径、大小、SHA见large_artifact_locations.json。',
        '','## 证据与复现','',
        'first_round_supplement完整保留第一轮交付与判读；round_two_raw_json保留补采元数据、拟合、校准、仲裁及官方判读原始JSON，round_two_logs保留命令日志，两个execution_snapshot保留命令/PID/退出码/依赖/数值检查及哈希。模型ZIP、M2/M2x NPZ、价值PT仅在large_artifact_locations.json列路径、大小和SHA，未入包、不提交Git、不推送。',
        '每进程OMP/MKL/OPENBLAS=1，总计算槽≤6且包含第一轮队列；可用内存低时只推迟启动。并行计算耗时不作正式串行实时性结论。JSON_SHA256.txt及ALL_FILES_SHA256.txt核验所有包内证据；发布在上层交付/最新，过程脚本与日志在过程文件/V3e价值第二轮。']
    (bundle/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    reproduction=bundle/'reproduction'; reproduction.mkdir(exist_ok=True)
    for path in (PROCESS/'脚本').glob('*.py'):
        shutil.copy2(path,reproduction/path.name)
    for filename in ('PREFLIGHT_TESTS_20260929.log','PREFLIGHT_R2_20260929.json','PIPELINE_R2.stdout.log','PIPELINE_R2.stderr.log'):
        path=RECORDS/filename
        if path.exists(): shutil.copy2(path,reproduction/filename)
    shutil.copy2(BASE/'过程文件/V3e60k/脚本/spacecraft_asset_layout.py',reproduction/'spacecraft_asset_layout.py')
    engineering=BASE/'过程文件/工程整理'
    if (engineering/'接续/工程与Git编排_20260930.md').exists():
        shutil.copy2(engineering/'接续/工程与Git编排_20260930.md',bundle/'工程与Git编排_20260930.md')
        shutil.copy2(engineering/'记录/REPOSITORY_LAYOUT_PREPARED_20260930.json',reproduction/'REPOSITORY_LAYOUT_PREPARED_20260930.json')
        for path in (engineering/'脚本').glob('*.ps1'):
            shutil.copy2(path,reproduction/path.name)
    json_files=sorted(bundle.rglob('*.json'))
    (bundle/'JSON_SHA256.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(bundle).as_posix()}' for p in json_files)+'\n',encoding='utf-8')
    files=sorted(p for p in bundle.rglob('*') if p.is_file() and p!=bundle/'ALL_FILES_SHA256.txt')
    (bundle/'ALL_FILES_SHA256.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(bundle).as_posix()}' for p in files)+'\n',encoding='utf-8')
    archive_path=bundle.with_suffix('.zip')
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.write(path,path.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for line in (bundle/'ALL_FILES_SHA256.txt').read_text(encoding='utf-8').splitlines():
            digest,path=line.split('  ',1)
            assert hashlib.sha256(archive.read(path)).hexdigest()==digest,path
    receipt={'generated_at':generated,'status':state['status'],'role':'second-round main result',
        'project_root':str(ROOT),'python_executable':r'D:\py\DRL2\.venv\Scripts\python.exe',
        'evaluation_commit':state.get('evaluation_commit'),'training_commit':state.get('training_commit'),
        'report':str(PUBLIC/'最新'/name/'REPORT.md'),'archive':{'path':str(PUBLIC/'最新'/archive_path.name),
        'bytes':archive_path.stat().st_size,'sha256':sha(archive_path)},'json_count':len(json_files),
        'zip_crc_and_manifest_hashes_verified':True,'verdict':readout['verdict'] if readout else None,
        'm6_gates':{str(s):state['models'][str(s)].get('m6_gate') for s in SEEDS}}
    receipt_path=PUBLIC/'待发布'/'V3E_VALUE_R2_DELIVERY_RECEIPT_20260929.json'
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    publish(bundle,archive_path,receipt_path)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    main()
