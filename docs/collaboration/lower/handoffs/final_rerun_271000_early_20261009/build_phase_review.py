"""Interim review from COMPLETE 271000 rows only. No partial learned/261 data."""
from pathlib import Path
import csv,json,hashlib,shutil,statistics
from collections import Counter
from datetime import datetime
BASE=Path(r'C:\Users\35884\Documents\Spacecraft');TOP=BASE/'过程文件/最终主线重训';REC=TOP/'记录';COLLAB=BASE/'过程文件/协作/Git工作树';PUB=COLLAB/'docs/collaboration/lower/handoffs/final_rerun_271000_early_20261009';LOCAL=TOP/'交付/271000_early_20261009'
E=Path('D:/py/DRL2/eval/final2/formal');COMMIT='f2f8169acd580ea96a578d45022dd8952447eca5';seeds=list(range(271000,271048))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csvwrite(p,rs):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
gates=read(LOCAL/'readout_gates_271000.json');phase={'science_commit':COMMIT,'block':'271000-271047','scope':'PARTIAL_TWO_MODEL_REVIEW_NOT_FINAL_THREE_MODEL_VERDICT','sample_rows':{},'model_sha256':{},'three_model_verdict':'PENDING','gate_function':'experiments.v3_stopping.gate_verdict'}
rows={};episode_rows=[];paired=[];failed=[]
for row,model in [('pure',None),('nominal',None),('stopping',262460),('stopping',262462)]:
    source=E/row/(str(model) if model else '')
    assert all((source/f'seed_{s}.json').exists() for s in seeds)
    records={s:read(source/f'seed_{s}.json') for s in seeds};name=row if model is None else str(model)
    for s,r in records.items():
        assert r['seed']==s and r['row']==row and r['code_commit']==COMMIT and not r['code_dirty'] and r.get('max_decisions') is None
        if row=='nominal':assert r['regime']=='w2.36_r15'
    if model:
        digest=sha(Path(f'D:/py/DRL2/logs/final2_{model}/final_model.zip'))
        assert {r['model_sha256'] for r in records.values()}=={digest};phase['model_sha256'][name]=digest
    rows[name]=records;success=[r for r in records.values() if r['clean_completion']]
    ks=[r.get('handoff_k') for r in records.values()]
    phase['sample_rows'][name]={'clean':len(success),'violations':sum(not r['zero_violation'] for r in records.values()),
                              'failure_counts':dict(Counter(reason for r in records.values() for reason in r['failure'])),
                              'median_own_success_time_s':statistics.median(r['survival_s'] for r in success) if success else None,
                              'median_own_success_delta_v_m_s':statistics.median(r['equivalent_delta_v_m_s'] for r in success) if success else None}
    if model:
        phase['sample_rows'][name]['handoff']={'immediate':sum(k==0 for k in ks),'never':sum(k is None for k in ks),'mid':sum(k not in (0,None) for k in ks)}
        assert phase['sample_rows'][name]['clean']==gates['models'][name]['clean_completions']['method']
    for s,r in records.items():
        item={'row':row,'model':model or '', 'seed':s,'clean':r['clean_completion'],'zero_violation':r['zero_violation'],'failure':'|'.join(r['failure']),
              'time_s':r['survival_s'],'delta_v_m_s':r['equivalent_delta_v_m_s'],'handoff_k':r.get('handoff_k'),'decisions':r['decisions']}
        episode_rows.append(item)
        if not r['clean_completion']:failed.append({**item,'pure_clean_same_seed':rows['pure'][s]['clean_completion']})
    dest=LOCAL/'raw'/row/(str(model) if model else '')
    dest.mkdir(parents=True,exist_ok=True)
    for s in seeds:
        target=dest/f'seed_{s}.json'
        if target.exists():assert sha(target)==sha(source/f'seed_{s}.json')
        else:shutil.copy2(source/f'seed_{s}.json',target)
for name in ('nominal','262460','262462'):
    shared=[s for s in seeds if rows['pure'][s]['clean_completion'] and rows[name][s]['clean_completion']]
    for s in shared:
        p,r=rows['pure'][s],rows[name][s]
        paired.append({'candidate':name,'seed':s,'pure_time_s':p['survival_s'],'candidate_time_s':r['survival_s'],'time_change_s':r['survival_s']-p['survival_s'],
                       'pure_delta_v_m_s':p['equivalent_delta_v_m_s'],'candidate_delta_v_m_s':r['equivalent_delta_v_m_s'],'delta_v_change_m_s':r['equivalent_delta_v_m_s']-p['equivalent_delta_v_m_s']})
    phase['sample_rows'][name]['shared_success_vs_pure']={'n':len(shared),'median_time_change_s':statistics.median(rows[name][s]['survival_s']-rows['pure'][s]['survival_s'] for s in shared) if shared else None,
                                                       'median_delta_v_change_m_s':statistics.median(rows[name][s]['equivalent_delta_v_m_s']-rows['pure'][s]['equivalent_delta_v_m_s'] for s in shared) if shared else None}
progress=[]
for directory in sorted(p for p in E.rglob('*') if p.is_dir()):
    files=list(directory.glob('seed_*.json'))
    if files:progress.append({'row_directory':directory.relative_to(E).as_posix(),'271000_files':sum(p.stem.startswith('seed_2710') for p in files),'272000_files':sum(p.stem.startswith('seed_2720') for p in files)})
snapshot={'timestamp':datetime.now().astimezone().isoformat(),'content_read_scope':'Only complete 271000 Pure/nominal/stopping262460/stopping262462; other rows filename counts only','progress':progress}
dump(LOCAL/'PHASE_SUMMARY.json',phase);dump(LOCAL/'evaluation_progress.json',snapshot)
csvwrite(LOCAL/'episodes_271000_complete_rows.csv',episode_rows);csvwrite(LOCAL/'failed_openings_271000.csv',failed);csvwrite(LOCAL/'paired_common_success_271000.csv',paired)
report='''# 本次实验阶段状况：供上层立即审查

**这是阶段证据，不是三模型最终判读。** 用户要求不等262461，立即交付已有完整信息。仍按冻结规则继续该种子评估；“它肯定更差”是用户预期，不作为测得事实。未完成learned、262461及272000行的结果内容未读取。

科学版本f2f8169，方法value stopping、soft Bellman、γ=0.999、原接口与奖励不变，工况w2.36_r15。262460/262462均完成60k外层预算；部署首次Q_H≥Q_C(s,μ(s))单向交给Pure。模型SHA见PHASE_SUMMARY.json，训练manifest/monitor在training/。整理后的60d768a未用于本次仿真，不混同科学代码版本。

## 已测得事实：271000–271047

|行|干净完成/48|违规回合|较Pure救回|毁掉Pure成功|官方单模型门槛|
|---|---:|---:|---:|---:|---|
'''
for name in ('pure','nominal','262460','262462'):
    item=phase['sample_rows'][name];g=gates['models'].get(name)
    report+=f"|{name}|{item['clean']}|{item['violations']}|{len(g['rescued']) if g else '—'}|{len(g['destroyed']) if g else '—'}|{('通过' if g['passes'] else '不通过') if g else '基线，不替代冻结Pure'}|\n"
report+='''
门槛固定为完成数高于Pure、违规不多于Pure、毁掉Pure成功≤2；三模型至少两个通过。当前仅262462通过，262460不通过，262461待齐全；不能宣布最终METHOD_HOLDS或METHOD_DOES_NOT_HOLD。

## 已有信息的审查价值与边界

1. 同一协议两个种子结局不同：262462满足门槛，但净增仅2个完成；262460完成比Pure少5、且产生3次违规、毁掉9个Pure成功开局。现有结果支持“跨种子表现不一致”，不证明整个方法必然无效或第三种子必然失败。
2. 共同成功样本的效率变化另见paired_common_success_271000.csv，避免各行成功样本构成不同。262460相对Pure中位时间+23.2s、Δv约+0.1822m/s；262462中位时间+2.2s、Δv约+0.2049m/s。单模型过完成/安全门不等于效率优势。
3. learned行尚不齐，交接相对同一策略的协调增益仍未知；不能把优于Pure的两次净完成直接归因于交接。
4. 停止时机、未交接失败、即时交接与中途交接的分布是描述事实。失败是否源于Q_H偏差、Q_C偏差或任务策略，需4.8确定性/反事实数据，当前不写成实测根因。未新增模块、门槛或补实验。

## 逐开局审查定位

'''
for name in ('262460','262462'):
    g=gates['models'][name];report+=f"- {name}：救回 {g['rescued']}；毁掉 {g['destroyed']}。\n"
    violation=[(s,r['failure'],r.get('handoff_k')) for s,r in rows[name].items() if not r['zero_violation']]
    report+=f'- 违规开局（种子、失败类型、handoff_k）原样摘要：{violation}。k为空表示全程未交接，不把这类失败解释为MPC交接后失败。\n'
report+='''
## 包与后续

raw/包含同块完整192个结果（Pure48、nominal48、两模型stopping各48），本包直接提供逐开局证据；CSV为独立重算摘要，FILES_SHA256逐文件核验。readout_gates_271000.json是冻结官方gate_verdict阶段输出，完整readout还需learned与第三种子；进度快照只列数量。

当前继续全部已授权评估；两模型learned齐后启动271000重放/反事实，262461保留并正常评估。272000是描述性第二块，不改变271000门槛；完成后停止扩展实验，交付并进入项目整体审查。此次发布不停止进程、不更新科学代码、不修改训练或控制。
'''
(LOCAL/'PHASE_REVIEW.md').write_text(report,encoding='utf-8')
(LOCAL/'README.md').write_text('# 271000阶段审查入口\n\n[立即供上层审查的实验状况](PHASE_REVIEW.md) · [指标与模型SHA](PHASE_SUMMARY.json) · [官方门槛阶段读数](readout_gates_271000.json) · [失败开局](failed_openings_271000.csv) · [共同成功配对](paired_common_success_271000.csv) · [进度快照](evaluation_progress.json)。\n\n完整192份原始结果与两模型训练证据随包提供；这是阶段报告，不是三模型最终结论，不剔除262461。\n',encoding='utf-8')
shutil.copy2(Path(__file__),LOCAL/'build_phase_review.py')
for p in LOCAL.rglob('*'):
    if p.is_file() and p.name!='FILES_SHA256.json':
        dst=PUB/p.relative_to(LOCAL);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for folder in (LOCAL,PUB):
    hashes={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='FILES_SHA256.json'}
    dump(folder/'FILES_SHA256.json',hashes);assert all(sha(folder/n)==h for n,h in hashes.items())
C=COLLAB/'docs/collaboration';state=read(C/'CURRENT.json');state['interim_experiment_review']={'scope':'271000 two-model complete rows; third pending','handoff':'lower/handoffs/final_rerun_271000_early_20261009/PHASE_REVIEW.md','science_commit':COMMIT,'review_does_not_stop_experiments':True,'final_verdict':'PENDING'};dump(C/'CURRENT.json',state)
with (C/'lower/README.md').open('a',encoding='utf-8') as f:f.write('\n\n当前实验阶段审查（不等262461）：[PHASE_REVIEW](handoffs/final_rerun_271000_early_20261009/PHASE_REVIEW.md)。完整192回合，最终三模型判读待齐；现有评估继续。\n')
print(json.dumps({'published_complete_episodes':len(episode_rows),'summary':phase['sample_rows']},ensure_ascii=False))
