"""Replot frozen exports only; curate five main PNGs and two backup PNGs."""
from pathlib import Path
import csv, json, hashlib, shutil
import plot_interim as p

def csvrows(name):
    with (p.DATA/name).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('clean','violation'):r[k]=r[k]=='True'
        for k in ('time_s','delta_v_m_s'):r[k]=float(r[k])
        r['seed']=int(r['seed']);r['model']=int(r['model']) if r['model'] else ''
    return rows

before={f.relative_to(p.DATA).as_posix():p.sha(f) for f in p.DATA.rglob('*') if f.is_file()}
captions=p.read(p.OUT/'FIGURE_CAPTIONS.json')
formal=csvrows('fig2_formal_rows.csv');baseline=csvrows('fig3_baselines.csv')
assert [sum(r['clean'] for r in formal if r['row']==row and r['model']==s)
        for row in ('learned','stopping') for s in (262430,262431,262432)]==[12,31,17,37,41,36]
assert [sum(r['clean'] for r in baseline if r['row']==row) for row in ('pure','nominal')]==[34,45]
p.comparison(formal,[262430,262431,262432],'fig2_formal_comparison',captions['fig2_formal_comparison'])
p.baseline_scatter(baseline,'fig3_baseline_time_energy',captions['fig3_baseline_time_energy'])
p.cases()
for name in ('fig_trajectory_3d','fig_state_response'):
    old=p.OUT/(name+'.png')
    if old.exists():old.unlink()  # newly rendered version lives in backup; no duplicate.
assert before=={f.relative_to(p.DATA).as_posix():p.sha(f) for f in p.DATA.rglob('*') if f.is_file()}, 'Scientific export changed'
readme='''# 汇报图表：正文五张，备份两张

仅高清PNG（400 dpi）。只读取已有资产、调整标注；不新增仿真，不修改训练或控制方法。临时汇报资产，不作为本轮方法论文性能。

## 正文顺序

1. [事后交接上界](fig4_hindsight_timing.png)：动机；48/48是事后最优机会，非可部署方法性能。
2. [V3e学习层训练](fig1_v3e_training.png)：历史学习层训练，非本轮价值停止方法。
3. [原工况正式轮](fig2_formal_comparison.png)：负结果保留；交接完成增益+25/+10/+19，未超过Pure。箱线图每箱注明自身干净完成样本数n，共同成功配对数据另存。
4. [新工况两条基线](fig3_baseline_time_energy.png)：图例注明Pure完成34/48、nominal完成45/48；不是学习方法性能。
5. [交接价值差](fig_handoff_value.png)：已有50k案例的阈值触发机制，非总体收益证明。

## 备份页

- [三维轨迹](备份页/fig_trajectory_3d.png)
- [状态响应](备份页/fig_state_response.png)：交接文字顶部横排，避免遮挡曲线。

两张备份是开发开局266019、50k中间模型：协调133.9/161.1s，Pure106.8s。本例协调更慢，仅说明执行与捕获过程，不能当优势案例。正式案例待正式评估后按冻结规则选取。

[数字核对](CHECKS.md) · [支撑与限制](REVIEW.md) · [绘图数据](plot_data/) · [源SHA](INPUTS_SHA256.json) · [复用](REUSE.md) · [本包SHA](FILES_SHA256.json)。

已剔除当前方法的无支撑正文训练图；旧版本仅历史保留。归档执行单的PDF/SVG要求由用户PNG-only指令取代。训练进度及结构门控为独立动态状态，不在本次标注更新中刷新或推断。
'''
(p.OUT/'README.md').write_text(readme,encoding='utf-8')
(p.OUT/'PRESENTATION_REVISION.md').write_text('# 本次修订\n\n用户授权三处标注调整：图2(b)(c)每箱n；图3图例完成数；状态响应交接顶部横排。正文五张，备份两张。全部绘图CSV SHA逐文件与修订前一致；既有案例导出核对通过；不开展新实验。\n',encoding='utf-8')
shutil.copy2(p.TOP/'脚本/plot_interim.py',p.OUT/'plot_interim.py')
shutil.copy2(Path(__file__),p.OUT/'revise_presentation.py')
pub=p.COLLAB/'docs/collaboration/lower/handoffs/figures_interim_20261009'
for name in ('fig2_formal_comparison.png','fig3_baseline_time_energy.png','fig_handoff_value.png','README.md','PRESENTATION_REVISION.md','plot_interim.py','revise_presentation.py'):
    shutil.copy2(p.OUT/name,pub/name)
(pub/'备份页').mkdir(exist_ok=True)
for name in ('fig_trajectory_3d.png','fig_state_response.png'):
    shutil.copy2(p.OUT/'备份页'/name,pub/'备份页'/name)
    if (pub/name).exists():(pub/name).unlink()
for root in (p.OUT,pub):
    # Update only links to relocated backup figures, preserving evidence reports.
    for f in root.glob('*.md'):
        if f.name in ('README.md','RUN_ORDER.md'):continue
        s=f.read_text(encoding='utf-8-sig')
        for name in ('fig_trajectory_3d.png','fig_state_response.png'):
            s=s.replace(']('+name+')','](备份页/'+name+')')
        f.write_text(s,encoding='utf-8')
    files={f.relative_to(root).as_posix():p.sha(f) for f in sorted(root.rglob('*')) if f.is_file() and f.name!='FILES_SHA256.json'}
    (root/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(p.sha(root/k)==v for k,v in files.items())
    assert len(list(root.glob('*.png')))==5 and len(list((root/'备份页').glob('*.png')))==2
    assert not list(root.rglob('*.svg')) and not list(root.rglob('*.pdf'))
handoff=pub.parent/'figures_interim_20261009.md'
with handoff.open('a',encoding='utf-8') as f:f.write('\n\n2026-10-09汇报版修订：正文按事后上界→V3e训练→历史正式轮→新工况基线→价值触发排列，共五张；轨迹和状态响应移至备份页。每箱样本数与基线完成数已标明，交接标注顶部横排。数据与科学结论不变，仅PNG。\n')
(p.REC/'PRESENTATION_REVISION.json').write_text(json.dumps({'checks':p.checks,'plot_data_unchanged':True,'main_png':5,'backup_png':2,'no_new_simulation':True},ensure_ascii=False,indent=2),encoding='utf-8')
print('VERIFIED: 5 main PNG, 2 backup PNG, all scientific plot data unchanged')
